# setiastro/saspro/calibration_io.py
"""
Lean I/O + master-frame cache for the calibration hot loops.

Why this module exists
----------------------
`calibrate_lights` (GPU path) and `_calibrate_lights_cpu` currently route every
frame through the general-purpose `legacy.image_manager.load_image` /
`save_image`. Those are correct but heavy: format sniffing, WCS/SIP handling,
range normalization, mono/color heuristics — all paid once *per frame*, on top
of the actual pixels. On a 100-frame set that overhead is multiplied 100×, and
it is the same on the GPU and CPU paths, which is exactly why toggling hardware
acceleration barely moved the wall-clock.

Everything here is a thin, eager, FITS-first path modelled on primitives that
already live in stacking_suite.py and are already trusted in the register /
integrate code:
  * reads  ~ `_fits_read_any_hdu_noscale` (line 3723) / `load_image_fast_norm` (797)
  * writes ~ the bare `fits.PrimaryHDU(...).writeto(...)` used at 19900 / 22205 / 22270

Three wins, each independently benchmarkable:
  1. read_light_fast()  — eager float32 FITS read, no WCS, no rescale round-trip.
  2. write_calibrated_fast() — direct float32 writeto with the scaling keywords
     scrubbed so a 16-bit source header can't corrupt a 32-bit-float output.
  3. MasterCache — decode each master dark/flat ONCE, not once per light. This
     removes the 100× redundant master decode in the CPU path (17950 / 18001).

Non-FITS inputs (XISF/TIFF/RAW) still delegate to the app's generic loader —
this module deliberately does not re-implement those; it only fast-paths FITS,
which is the overwhelming majority of light frames.

Nothing here imports Qt, torch, or stacking_suite, so it is safe to import from
anywhere and cheap to unit-test in isolation.
"""
from __future__ import annotations

import os
import time
import threading
from contextlib import contextmanager
from typing import Callable, Optional, Tuple

import numpy as np
from astropy.io import fits


_FITS_EXTS = (".fits", ".fit", ".fts", ".fits.gz", ".fit.gz", ".fts.gz", ".fz")

# Keywords that describe an integer encoding. If they survive onto a float32
# PrimaryHDU, astropy may re-apply them on write and silently rescale the data.
# We strip them; astropy recomputes SIMPLE/BITPIX/NAXIS* from the array itself.
_SCALING_KEYS = ("BSCALE", "BZERO", "BLANK", "BITPIX")


# ─────────────────────────────────────────────────────────────────────────────
# small helpers
# ─────────────────────────────────────────────────────────────────────────────
def is_fits_path(path: str) -> bool:
    lp = path.lower()
    return any(lp.endswith(ext) for ext in _FITS_EXTS)


def _to_native_endian(a: np.ndarray) -> np.ndarray:
    """Return an array in the machine's native byte order without a needless copy."""
    if a.dtype.byteorder not in ("=", "|"):
        return a.astype(a.dtype.newbyteorder("="), copy=False)
    return a


def _apply_blank_bscale_bzero(raw: np.ndarray, hdr: fits.Header) -> np.ndarray:
    """
    Mirror of the manual scaling used by _fits_read_any_hdu_noscale: apply
    BLANK -> NaN (integer data only), then BZERO + BSCALE*x, as float32.
    Reading with do_not_scale_image_data=True hands us the raw stored ints, so
    we do the affine transform once, ourselves, and land straight in float32.
    """
    bscale = hdr.get("BSCALE", 1)
    bzero = hdr.get("BZERO", 0)
    blank = hdr.get("BLANK", None)

    out = raw
    if blank is not None and np.issubdtype(raw.dtype, np.integer):
        # promote to float first so NaN is representable
        out = raw.astype(np.float32, copy=True)
        out[raw == blank] = np.nan
    else:
        out = raw.astype(np.float32, copy=False)

    if bscale != 1:
        out = out * np.float32(bscale)
    if bzero != 0:
        out = out + np.float32(bzero)

    return np.ascontiguousarray(out, dtype=np.float32)


def classify_layout(img: np.ndarray) -> Tuple[bool, bool]:
    """
    Return (is_mono, is_color_hwc). Matches the shape heuristics used throughout
    stacking_suite: color == 3D with a size-3 axis; everything else is mono
    (including single-plane CFA, which stays mosaiced through calibration).
    """
    if img.ndim == 2:
        return True, False
    if img.ndim == 3 and img.shape[-1] == 3:
        return False, True          # HWC color
    if img.ndim == 3 and img.shape[0] == 3:
        return False, False         # CHW color (still color, not HWC)
    return True, False


def bayer_from_header(hdr: fits.Header) -> Optional[str]:
    pat = str(hdr.get("BAYERPAT") or "").strip().upper()
    return pat if pat in ("RGGB", "BGGR", "GRBG", "GBRG") else None


# ─────────────────────────────────────────────────────────────────────────────
# READ
# ─────────────────────────────────────────────────────────────────────────────
def read_fits_fast(path: str, memmap: bool = False) -> Tuple[Optional[np.ndarray],
                                                              Optional[fits.Header]]:
    """
    Eager FITS read → (float32 image, header). No WCS build, no normalization to
    [0,1], no format sniffing. Applies BLANK/BSCALE/BZERO manually. Returns the
    first HDU carrying 2D/3D numeric image data. (None, None) on miss/error.
    """
    try:
        with fits.open(path, memmap=memmap, do_not_scale_image_data=True,
                       ignore_missing_end=True, uint=False) as hdul:
            for h in hdul:
                try:
                    d = h.data
                except Exception:
                    continue
                if not isinstance(d, np.ndarray) or d.ndim not in (2, 3) or d.size == 0:
                    continue
                d = _to_native_endian(d)
                d = np.squeeze(d)
                d = _apply_blank_bscale_bzero(d, h.header)
                if d.ndim == 3 and d.shape[-1] == 1:
                    d = np.squeeze(d, axis=-1)
                d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
                return np.ascontiguousarray(d, dtype=np.float32), h.header
    except Exception:
        return None, None
    return None, None


# ⚠️ DELIBERATELY NO read_light_fast() HERE.
#
# It is tempting to swap the producer's `load_image(fi["light_file"])` for a bare
# read_fits_fast(). DO NOT. load_image does two things beyond decoding that the
# calibration math depends on:
#   1. range normalization — it returns img / 65535.0 for 16-bit sources
#      (BITPIX-conditional; see stacking_suite.py:5132 / 4542). Masters are also
#      brought to that same [0,1] scale (_maybe_normalize_16bit_float). A raw
#      read leaves the light in [0,65535], so `light - dark` mixes scales and the
#      output is garbage.
#   2. ROWORDER orientation — it flips bottom-up frames to SASpro's top-down
#      convention so lights and masters register (see the master-build note near
#      stacking_suite.py:15504). A raw read skips the flip → dark subtraction runs
#      one vertical flip out of register.
#
# So the light-read is NOT a safe naive fast-path. Correct order of operations:
#   (a) profile it with StageTimer below — confirm the read is actually a hot spot
#       before touching it (on this workload the WRITE + cosmetic are the likelier
#       costs);
#   (b) if load_image really dominates, optimize load_image itself, or build a
#       reader that reproduces BOTH the /65535 normalization AND the ROWORDER flip
#       exactly. read_fits_fast() above is a raw-ADU building block for that, not a
#       replacement on its own.
#
# The safe, semantics-preserving wins live below: write_calibrated_fast (the
# consumer never needs load_image's read behaviour) and MasterCache (decode each
# master once instead of once per light — pure memoization of the app's own loader).


# ─────────────────────────────────────────────────────────────────────────────
# WRITE
# ─────────────────────────────────────────────────────────────────────────────
def scrub_scaling_keywords(hdr: fits.Header) -> fits.Header:
    """
    Return a copy of `hdr` with integer-encoding keywords removed so a float32
    write is clean. astropy re-derives SIMPLE/BITPIX/NAXIS* from the data array.
    """
    h = fits.Header(hdr) if hdr is not None else fits.Header()
    for k in _SCALING_KEYS:
        try:
            if k in h:
                del h[k]
        except Exception:
            pass
    return h


def write_calibrated_fast(
    path: str,
    img: np.ndarray,
    header: Optional[fits.Header],
    is_mono: bool,
    history: Optional[str] = None,
) -> None:
    """
    Drop-in consumer write, replacing:
        save_image(img_array=..., filename=..., original_format="fit",
                   bit_depth="32-bit floating point",
                   original_header=hdr, is_mono=is_mono)

    Expects `img` already in the on-disk layout the consumer produces (2D mono,
    or HWC for 3-plane color — the consumer's transpose at ~19015 stays). Writes
    a single float32 PrimaryHDU with scaling keywords scrubbed. Only FITS is
    fast-pathed here; callers that need XISF/TIFF output should keep save_image.
    """
    # Preserve NaN as no-data (satellite trails) — write float32 straight
    # through; scrub only +/-inf. FITS float32 stores NaN natively.
    arr = np.ascontiguousarray(
        np.nan_to_num(np.asarray(img, dtype=np.float32),
                      nan=np.nan, posinf=0.0, neginf=0.0)
    )

    hdr = scrub_scaling_keywords(header)
    is_color_hwc = (arr.ndim == 3 and arr.shape[-1] == 3)
    try:
        hdr["DEBAYERED"] = (bool(is_color_hwc),
                            "Color debayered" if is_color_hwc else "Mono / CFA")
    except Exception:
        pass
    if history:
        try:
            hdr.add_history(history)
        except Exception:
            pass

    tmp = path + ".tmp"          # atomic-ish: write then replace, so a crash
    fits.PrimaryHDU(data=arr, header=hdr).writeto(   # never leaves a half file
        tmp, overwrite=True, output_verify="silentfix"
    )
    os.replace(tmp, path)


# ─────────────────────────────────────────────────────────────────────────────
# MASTER CACHE  — decode each master once, not once per light
# ─────────────────────────────────────────────────────────────────────────────
class MasterCache:
    """
    Memoize decoded + normalized master frames by path. Thread-safe so the
    producer / GPU / consumer threads can share one instance.

    The CPU path currently does, inside the per-light loop:
        dark, _, _, _ = load_image(master_dark_path)
        dark = _maybe_normalize_16bit_float(dark, ...)
    for EVERY light — i.e. the same master is decoded N times. Wrap the app's own
    loader + normalizer once and call `cache.get(path)` instead; behaviour is
    identical, the decode happens once.

        cache = MasterCache(loader_fn=load_image,
                            normalize_fn=_maybe_normalize_16bit_float)
        dark, dark_is_mono = cache.get(master_dark_path)   # (np.float32|None, bool)

    get() returns a (array, is_mono) pair: the array is None on failure, and
    is_mono carries load_image's authoritative mono/color flag so callers keep
    their existing layout decisions (the CPU path's HWC->CHW transpose depends
    on it — deriving it from shape alone would misclassify some frames).
    """

    def __init__(
        self,
        loader_fn: Callable[[str], tuple],
        normalize_fn: Optional[Callable[..., np.ndarray]] = None,
    ):
        self._loader = loader_fn
        self._normalize = normalize_fn
        self._store: dict[str, Tuple[Optional[np.ndarray], bool]] = {}
        self._lock = threading.Lock()

    def get(self, path: Optional[str]) -> Tuple[Optional[np.ndarray], bool]:
        """Return (normalized float32 array | None, is_mono)."""
        if not path:
            return None, True
        with self._lock:
            if path in self._store:
                return self._store[path]
        # decode outside the lock so slow disk reads don't serialize threads
        entry = self._decode(path)
        with self._lock:
            self._store[path] = entry
        return entry

    def _decode(self, path: str) -> Tuple[Optional[np.ndarray], bool]:
        # Always go through the app's real loader (+ normalizer). Masters need the
        # SAME /65535 normalization and ROWORDER flip that load_image applies to
        # lights, or dark/flat land at the wrong scale/orientation. The win here is
        # NOT a leaner decode — it's decoding ONCE instead of once per light, so
        # using the full, correct loader costs nothing on a handful of masters.
        try:
            res = self._loader(path)
            arr = res[0] if res else None
            is_mono = bool(res[3]) if (res is not None and len(res) > 3) \
                else (arr is None or arr.ndim < 3)
            if arr is None:
                return None, is_mono
            if self._normalize is not None:
                arr = self._normalize(arr, name=os.path.basename(path))
            if arr is None:
                return None, is_mono
            return np.ascontiguousarray(arr, dtype=np.float32), is_mono
        except Exception:
            return None, True

    def clear(self) -> None:
        with self._lock:
            self._store.clear()


# ─────────────────────────────────────────────────────────────────────────────
# OPTIONAL: torch-free elementwise calibration
# ─────────────────────────────────────────────────────────────────────────────
# Dark subtraction and flat division are memory-bandwidth-bound elementwise ops.
# On unified-memory Apple silicon, shipping each light to MPS and back buys
# nothing and adds a full host<->device sync per frame. These vectorized numpy
# versions keep the data in RAM and are competitive with (often faster than) the
# MPS round-trip on an M-series Air. Reserve the GPU for genuinely heavy work
# (the cosmetic sort/median, the satellite CNN), not for a subtract.
def subtract_dark_np(light: np.ndarray, dark: Optional[np.ndarray],
                     pedestal: float = 0.0) -> np.ndarray:
    out = np.asarray(light, dtype=np.float32)
    if dark is not None:
        out = out - np.asarray(dark, dtype=np.float32)   # broadcast handles CHW vs HW
    if pedestal:
        out = out + np.float32(pedestal)
    np.clip(out, 0.0, None, out=out)
    return out


def divide_flat_np(light: np.ndarray, flat: Optional[np.ndarray]) -> np.ndarray:
    if flat is None:
        return np.asarray(light, dtype=np.float32)
    f = np.asarray(flat, dtype=np.float32)
    f = np.where(np.isfinite(f) & (f > 0), f, np.float32(1.0))
    return (np.asarray(light, dtype=np.float32) / f).astype(np.float32, copy=False)


# ─────────────────────────────────────────────────────────────────────────────
# instrumentation — settle "where does the time go" in one run
# ─────────────────────────────────────────────────────────────────────────────
class StageTimer:
    """
    Cheap per-stage accumulator. Wrap the three shared hot points and print at
    the end; you'll see load vs compute vs save in wall-clock, per frame.

        T = StageTimer()
        with T("load"):  light_data, hdr, bit_depth, is_mono = load_image(path)
        with T("gpu"):   light_data = calibration_pipeline_gpu(...)
        with T("save"):  write_calibrated_fast(out, light_data, hdr, is_mono)
        ...
        print(T.report())
    """
    def __init__(self):
        self._t: dict[str, float] = {}
        self._n: dict[str, int] = {}
        self._lock = threading.Lock()

    @contextmanager
    def __call__(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            with self._lock:
                self._t[name] = self._t.get(name, 0.0) + dt
                self._n[name] = self._n.get(name, 0) + 1

    def report(self) -> str:
        with self._lock:
            rows = []
            for k in sorted(self._t, key=lambda x: -self._t[x]):
                tot, n = self._t[k], self._n[k]
                rows.append(f"  {k:<8} {tot:8.2f}s total  {tot/max(n,1)*1000:8.1f} ms/call  (n={n})")
        return "StageTimer:\n" + "\n".join(rows)