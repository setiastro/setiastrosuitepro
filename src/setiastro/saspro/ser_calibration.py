# src/setiastro/saspro/ser_calibration.py
# ---------------------------------------------------------------------------
# SetiAstro Suite Pro — flat-field calibration for the SER stacker
# Copyright (C) Franklin Marek / SetiAstro. Distributed under GPLv3.
# ---------------------------------------------------------------------------
"""
Flat-field calibration for the SER stacker (planetary / lunar / solar).

Loads a flat from:
  • .ser video (median- or mean-stacked across the video)
  • .tif / .tiff single image
  • .fit / .fits / .fits.gz / .fz single image

Prepares it in the SAME channel layout as the run's frames (matches
debayer / force_rgb / bayer_pattern) and normalises so each channel's
mean is 1.0.  Frames are then corrected by pixel-wise division:

    corrected = raw_frame / flat

The flat is loaded ONCE per run at full sensor resolution.  ROI cropping
happens at frame-access time inside ``_CalibratedSource`` in
``ser_stacker.py`` (which just view-slices the flat — no copies).

Public API
----------
load_flat_for_run(flat_path, *, frame_shape, debayer, force_rgb,
                  bayer_pattern, ser_stack_method='median',
                  progress_cb=None, log_cb=None) -> np.ndarray
apply_flat(frame, flat, *, roi=None) -> np.ndarray
"""
from __future__ import annotations
import os
import numpy as np
from typing import Optional, Callable

# ---------------------------------------------------------------------------
# Extensions
# ---------------------------------------------------------------------------
_FITS_EXT = (".fit", ".fits", ".fz")     # ".fits.gz" handled separately (double-ext)
_TIF_EXT  = (".tif", ".tiff")
_SER_EXT  = (".ser",)


def _is_fits_path(path: str) -> bool:
    lower = path.lower()
    return lower.endswith(_FITS_EXT) or lower.endswith(".fits.gz")


# ---------------------------------------------------------------------------
# File loaders (raw, unnormalised)
# ---------------------------------------------------------------------------
def _load_fits_array(path: str) -> np.ndarray:
    """First image HDU of a FITS file, as a numpy array (dtype preserved)."""
    from astropy.io import fits
    with fits.open(path, memmap=False) as hdul:
        for h in hdul:
            if h.data is not None:
                return np.asarray(h.data)
    raise ValueError(f"No image HDU found in FITS: {path}")


def _load_tiff_array(path: str) -> np.ndarray:
    """Read TIFF via tifffile if available (safe for 16/32-bit + float),
    else cv2 fallback (BGR -> RGB fix applied)."""
    try:
        import tifffile
        return np.asarray(tifffile.imread(path))
    except Exception:
        import cv2
        arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise IOError(f"Failed to read TIFF: {path}")
        if arr.ndim == 3 and arr.shape[2] >= 3:
            # cv2 returns BGR; swap so channel order matches other loaders
            arr = arr[..., ::-1].copy()
        return np.asarray(arr)


def _to_float01(arr: np.ndarray) -> np.ndarray:
    """
    Convert to float32 in the [0, 1] range using dtype maxval for integers.
    Floats are kept as-is (normalisation happens per-channel later).
    """
    a = np.asarray(arr)
    if np.issubdtype(a.dtype, np.floating):
        return a.astype(np.float32, copy=False)
    if np.issubdtype(a.dtype, np.integer):
        info = np.iinfo(a.dtype)
        return (a.astype(np.float32) / float(info.max))
    return a.astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# SER flat: median- or mean-stack all (or a subsample of) frames
# ---------------------------------------------------------------------------
def _combine_ser(
    path: str,
    *,
    debayer: bool,
    force_rgb: bool,
    bayer_pattern: Optional[str],
    method: str,
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
    log_cb: Optional[Callable[[str], None]] = None,
) -> np.ndarray:
    """
    Stack all (or up to MAX_STACK by uniform subsampling) frames of a SER
    flat via median (default) or mean.  Returns a float32 array in
    whatever channel layout ``debayer`` / ``force_rgb`` produce.

    We reuse the same ``open_planetary_source`` code path as the frames so
    the flat undergoes IDENTICAL debayer / RGB handling — no per-pixel
    surprises from a divergent load path.
    """
    from setiastro.saspro.imageops.serloader import open_planetary_source

    method = str(method).lower().strip() or "median"

    src = open_planetary_source(path, cache_items=4)
    try:
        n = int(src.meta.frames)
        if n <= 0:
            raise ValueError(f"SER flat contains no frames: {path}")

        # Cap the stack.  Flat SERs can be very long; a uniform 200-frame
        # subsample is more than enough for a stable median.
        MAX_STACK = 200
        if n > MAX_STACK:
            step = int(np.ceil(n / MAX_STACK))
            frame_indices = list(range(0, n, step))[:MAX_STACK]
        else:
            frame_indices = list(range(n))

        if log_cb:
            log_cb(f"Flat: {method}-stacking {len(frame_indices)}/{n} frames "
                   f"from {os.path.basename(path)}")

        # Load one frame to know the target shape, then accumulate.
        # For memory safety with large sensors we accumulate incrementally
        # for mean; for median we need the full stack (unavoidable).
        first = src.get_frame(
            int(frame_indices[0]), roi=None, debayer=debayer,
            to_float01=True, force_rgb=force_rgb,
            bayer_pattern=bayer_pattern,
        )
        first = np.asarray(first, dtype=np.float32)

        total = len(frame_indices)

        if method == "mean":
            acc = np.zeros_like(first, dtype=np.float64)  # f64 to avoid drift
            acc += first
            for k, i in enumerate(frame_indices[1:], start=1):
                fr = src.get_frame(
                    int(i), roi=None, debayer=debayer, to_float01=True,
                    force_rgb=force_rgb, bayer_pattern=bayer_pattern,
                )
                acc += np.asarray(fr, dtype=np.float32)
                if progress_cb and (k % max(1, total // 20) == 0):
                    progress_cb(k + 1, total, "Flat")
            if progress_cb:
                progress_cb(total, total, "Flat")
            return (acc / float(total)).astype(np.float32, copy=False)

        # median
        stack = np.empty((total,) + first.shape, dtype=np.float32)
        stack[0] = first
        for k, i in enumerate(frame_indices[1:], start=1):
            fr = src.get_frame(
                int(i), roi=None, debayer=debayer, to_float01=True,
                force_rgb=force_rgb, bayer_pattern=bayer_pattern,
            )
            stack[k] = np.asarray(fr, dtype=np.float32)
            if progress_cb and (k % max(1, total // 20) == 0):
                progress_cb(k + 1, total, "Flat")
        if progress_cb:
            progress_cb(total, total, "Flat")
        return np.median(stack, axis=0).astype(np.float32, copy=False)

    finally:
        try:
            src.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Bayer helpers (used only when the source flat is a raw mono file but the
# run has debayer=True)
# ---------------------------------------------------------------------------
def _bayer_cv2_code(pattern: Optional[str]):
    """Map a Bayer pattern name to a cv2 debayer code (RGB output)."""
    if not pattern:
        return None
    try:
        import cv2
    except Exception:
        return None
    p = str(pattern).upper().replace("BAYER_", "").strip()
    return {
        "RGGB": cv2.COLOR_BayerBG2RGB,
        "BGGR": cv2.COLOR_BayerRG2RGB,
        "GRBG": cv2.COLOR_BayerGB2RGB,
        "GBRG": cv2.COLOR_BayerGR2RGB,
    }.get(p)


def _match_frame_shape(
    flat: np.ndarray,
    *,
    frame_ndim: int,
    frame_channels: int,
    debayer: bool,
    bayer_pattern: Optional[str],
    log_cb: Optional[Callable[[str], None]] = None,
) -> np.ndarray:
    """
    Coerce a loaded flat to the ndim/channels the run's frames will use.

    - (H, W, 1) collapses to (H, W).
    - RGB flat needed but got mono: debayer if pattern given, else replicate.
    - Mono flat needed but got RGB: Rec.709 luminance.
    """
    f = np.asarray(flat, dtype=np.float32)

    if f.ndim == 3 and f.shape[2] == 1:
        f = f[..., 0]

    want_rgb = (frame_ndim == 3 and frame_channels >= 3)

    if want_rgb and f.ndim == 2:
        # Mono flat, need RGB
        if debayer and bayer_pattern:
            code = _bayer_cv2_code(bayer_pattern)
            if code is not None:
                try:
                    import cv2
                    f16 = np.clip(f * 65535.0, 0, 65535.0).astype(np.uint16)
                    rgb = cv2.cvtColor(f16, code)
                    f = rgb.astype(np.float32) / 65535.0
                    if log_cb:
                        log_cb(f"Flat: debayered mono flat with pattern {bayer_pattern}")
                except Exception as e:
                    if log_cb:
                        log_cb(f"Flat: debayer failed ({e}); replicating mono into RGB")
                    f = np.dstack([f, f, f])
            else:
                if log_cb:
                    log_cb(f"Flat: bayer pattern {bayer_pattern!r} unrecognised; "
                           "replicating mono into RGB")
                f = np.dstack([f, f, f])
        else:
            if log_cb:
                log_cb("Flat: mono flat + no bayer pattern; replicating into RGB")
            f = np.dstack([f, f, f])

    elif not want_rgb and f.ndim == 3 and f.shape[2] >= 3:
        # RGB flat, need mono → Rec.709 luminance
        f = (0.2126 * f[..., 0] + 0.7152 * f[..., 1]
             + 0.0722 * f[..., 2]).astype(np.float32)
        if log_cb:
            log_cb("Flat: RGB flat folded to luminance for mono pipeline")

    return f


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
def _normalize_flat(flat: np.ndarray, *, min_frac: float = 0.10) -> np.ndarray:
    """
    In-place per-channel normalisation to mean = 1.0, followed by a floor
    of ``min_frac`` to prevent runaway values in deep vignette corners
    where ``frame / flat`` would otherwise blow up.

    - RGB (H, W, 3+): each channel divided by its own mean.  This lets the
      color cast of the raw flat cancel against the color cast of the raw
      light on divide (standard flat-field convention).
    - Mono (H, W): global mean.
    """
    f = np.asarray(flat, dtype=np.float32)
    # Copy if it's a memory-mapped view — we're going to mutate.
    if not f.flags.writeable:
        f = f.copy()

    if f.ndim == 3 and f.shape[2] >= 3:
        for c in range(f.shape[2]):
            m = float(f[..., c].mean())
            if m > 1e-8:
                f[..., c] /= m
    else:
        m = float(f.mean())
        if m > 1e-8:
            f /= m

    np.maximum(f, float(min_frac), out=f)
    return f


# ---------------------------------------------------------------------------
# Public: load + prepare
# ---------------------------------------------------------------------------
def load_flat_for_run(
    flat_path: str,
    *,
    frame_shape: tuple,
    debayer: bool,
    force_rgb: bool,
    bayer_pattern: Optional[str],
    ser_stack_method: str = "median",
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
    log_cb: Optional[Callable[[str], None]] = None,
) -> np.ndarray:
    """
    Load a flat calibration file and return a normalised flat matched to
    the frame layout.

    Parameters
    ----------
    flat_path : str
        Path to a .ser, .tif/.tiff, or .fit/.fits[.gz]/.fz file.
    frame_shape : tuple
        Shape that ``src.get_frame(..., roi=None)`` returns for this run
        (i.e. after debayer / force_rgb are applied).  Used to validate
        the flat's sensor size and coerce channel layout.
    debayer, force_rgb, bayer_pattern
        Must mirror the run's config so the flat lands in the same layout.
    ser_stack_method : "median" | "mean"
        Combination method for SER video flats.  Median is default
        (rejects outlier cosmic rays / passing airplane pixels).

    Returns
    -------
    np.ndarray (float32)
        Normalised flat with per-channel mean == 1.0 and values floored
        away from zero.  Same shape as ``frame_shape``.
    """
    if not flat_path or not os.path.isfile(flat_path):
        raise IOError(f"Flat file not found: {flat_path}")

    ext = os.path.splitext(flat_path)[1].lower()

    if ext in _SER_EXT:
        raw = _combine_ser(
            flat_path,
            debayer=debayer,
            force_rgb=force_rgb,
            bayer_pattern=bayer_pattern,
            method=ser_stack_method,
            progress_cb=progress_cb,
            log_cb=log_cb,
        )
    elif ext in _TIF_EXT:
        raw = _to_float01(_load_tiff_array(flat_path))
        if log_cb:
            log_cb(f"Flat: loaded TIFF {os.path.basename(flat_path)} "
                   f"shape={raw.shape} dtype={raw.dtype}")
    elif _is_fits_path(flat_path):
        raw = _to_float01(_load_fits_array(flat_path))
        if log_cb:
            log_cb(f"Flat: loaded FITS {os.path.basename(flat_path)} "
                   f"shape={raw.shape} dtype={raw.dtype}")
    else:
        raise ValueError(
            f"Unsupported flat file type: {ext!r}. "
            "Use .ser, .tif/.tiff, or .fit/.fits[.gz]/.fz"
        )

    # Match run's channel layout
    frame_ndim = 3 if (len(frame_shape) == 3 and frame_shape[-1] >= 3) else 2
    frame_channels = int(frame_shape[-1]) if frame_ndim == 3 else 1

    flat = _match_frame_shape(
        raw,
        frame_ndim=frame_ndim,
        frame_channels=frame_channels,
        debayer=debayer,
        bayer_pattern=bayer_pattern,
        log_cb=log_cb,
    )

    # Validate sensor dimensions BEFORE we start normalising & handing
    # this to the stacker — cheaper failure surface.
    tgt_hw = (int(frame_shape[0]), int(frame_shape[1]))
    if flat.shape[:2] != tgt_hw:
        raise ValueError(
            f"Flat dimensions {flat.shape[:2]} do not match sensor "
            f"{tgt_hw}. Flats must be captured at the same resolution as "
            "your lights (ROI + binning included)."
        )

    flat = _normalize_flat(flat, min_frac=0.10)

    if log_cb:
        c = flat.shape[2] if flat.ndim == 3 else 1
        log_cb(f"Flat: ready ({flat.shape[0]}×{flat.shape[1]}, "
               f"{c} channel{'s' if c > 1 else ''}, normalised to mean=1.0)")

    return flat


# ---------------------------------------------------------------------------
# Public: per-frame application
# ---------------------------------------------------------------------------
def apply_flat(frame: np.ndarray, flat: Optional[np.ndarray],
               *, roi=None) -> np.ndarray:
    """
    Divide ``frame`` by the (optionally ROI-cropped) ``flat``.

    - ``flat`` is None → returns frame unchanged.
    - Result is clipped to [0, 1] (matches the float01 convention of the
      rest of the SER stacker).
    - Broadcasting: mono flat vs colour frame promotes to per-channel via
      trailing None axis; RGB flat vs mono frame folds to luminance
      (defensive fallback — normal prep already handles this).
    """
    if flat is None:
        return frame

    f = np.asarray(frame, dtype=np.float32)
    if roi is not None:
        x, y, w, h = [int(v) for v in roi]
        fl = flat[y:y + h, x:x + w]
    else:
        fl = flat

    if f.ndim == 3 and fl.ndim == 2:
        fl = fl[..., None]
    elif f.ndim == 2 and fl.ndim == 3 and fl.shape[2] >= 3:
        fl = (0.2126 * fl[..., 0] + 0.7152 * fl[..., 1]
              + 0.0722 * fl[..., 2]).astype(np.float32)

    if fl.shape[:2] != f.shape[:2]:
        raise ValueError(
            f"Flat crop {fl.shape[:2]} != frame shape {f.shape[:2]}. "
            "Flat was not prepared at the sensor's full resolution."
        )

    out = f / fl
    np.clip(out, 0.0, 1.0, out=out)
    return out