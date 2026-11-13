# src/setiastro/saspro/stacking_measure_worker.py
"""
Standalone, GUI-free worker for the stacking-suite measurement phase.

This module exists so the per-frame measurement (preview load + OpenCV star /
FWHM + NumPy stats) can run in ``ProcessPoolExecutor`` children and use every
core without the GIL. It is deliberately import-light: it must NOT import PyQt
or ``stacking_suite`` (which pulls in the whole GUI), otherwise spawning workers
on Windows becomes slow/fragile.

The load + measure helpers below are copied VERBATIM from the corresponding
methods/functions in ``stacking_suite.py`` (``_quick_preview_any``,
``_bin_from_header_fast_any``, ``_get_image_size``, ``_measure_fwhm_halfres``,
``_mad_noise``, ``_star_count_ecc_size``, ``_resize_to_scale``) so a frame
measured here is byte-identical to one measured in-process. The measurement
path is pure OpenCV / NumPy / astropy — no numba call — so worker processes pay
no JIT cost.

IMPORTANT: if you change the measurement logic in ``stacking_suite.py``, mirror
the change here (and vice-versa), or the thread fallback and the process path
will disagree.
"""
from __future__ import annotations

import io
import json
import os
import math
import time
import numpy as np
import cv2
from astropy.io import fits

# cv2-based star counter (NOT numba — importing numba_utils defines dispatchers
# but no JIT runs unless a numba function is actually called, which this path
# never does).
from setiastro.saspro.legacy.numba_utils import compute_star_count_fast_preview


# ── copied verbatim from stacking_suite._resize_to_scale ──────────────────────
def _resize_to_scale(img: np.ndarray, scale_x: float, scale_y: float) -> np.ndarray:
    """
    Resample image to apply per-axis scale. Works for 2D or HxWx3.
    Uses OpenCV if present (fast, high quality). Fallbacks to SciPy; last resort = NumPy kron for integer upscales.
    """
    if scale_x == 1.0 and scale_y == 1.0:
        return img

    h, w = img.shape[:2]
    new_w = max(1, int(round(w * (scale_x))))
    new_h = max(1, int(round(h * (scale_y))))

    # Prefer OpenCV
    try:
        import cv2
        interp = cv2.INTER_CUBIC if (scale_x > 1.0 or scale_y > 1.0) else cv2.INTER_AREA
        if img.ndim == 2:
            return cv2.resize(img.astype(np.float32, copy=False), (new_w, new_h), interpolation=interp)
        else:
            return cv2.resize(img.astype(np.float32, copy=False), (new_w, new_h), interpolation=interp)
    except Exception:
        pass

    # SciPy fallback
    try:
        from scipy.ndimage import zoom
        if img.ndim == 2:
            return zoom(img.astype(np.float32, copy=False), (scale_y, scale_x), order=3)
        else:
            # zoom each channel; zoom expects zoom per axis
            return zoom(img.astype(np.float32, copy=False), (scale_y, scale_x, 1.0), order=3)
    except Exception:
        pass

    # Last resort (integer upscale only): kron
    sx_i = int(round(scale_x)); sy_i = int(round(scale_y))
    if abs(scale_x - sx_i) < 1e-6 and abs(scale_y - sy_i) < 1e-6 and sx_i >= 1 and sy_i >= 1:
        if img.ndim == 2:
            return np.kron(img, np.ones((sy_i, sx_i), dtype=np.float32))
        else:
            up2d = np.kron(img[..., 0], np.ones((sy_i, sx_i), dtype=np.float32))
            out = np.empty((up2d.shape[0], up2d.shape[1], img.shape[2]), dtype=np.float32)
            for c in range(img.shape[2]):
                out[..., c] = np.kron(img[..., c], np.ones((sy_i, sx_i), dtype=np.float32))
            return out

    # If all else fails, return original
    return img


# ── copied verbatim from stacking_suite._bin_from_header_fast_any ─────────────
def _bin_from_header_fast_any(fp: str) -> tuple[int, int]:
    ext = os.path.splitext(fp)[1].lower()
    # FITS quick path
    if ext in (".fits", ".fit", ".fz"):
        try:
            h = fits.getheader(fp, ext=0)
            # common variants
            xb = int(h.get("XBINNING", h.get("XBIN", 1)))
            yb = int(h.get("YBINNING", h.get("YBIN", 1)))
            return max(1, xb), max(1, yb)
        except Exception:
            return (1, 1)
    # XISF path
    if ext == ".xisf":
        try:
            from setiastro.saspro.legacy.xisf import XISF
            x = XISF(fp)
            ims = x.get_images_metadata()
            if not ims:
                return (1, 1)
            m0 = ims[0]
            fkw = m0.get("FITSKeywords", {})
            def _kw(name, default=None):
                vals = fkw.get(name)
                return vals[0]["value"] if vals and "value" in vals[0] else default
            xb = int(_kw("XBINNING", _kw("XBIN", 1)) or 1)
            yb = int(_kw("YBINNING", _kw("YBIN", 1)) or 1)
            return max(1, xb), max(1, yb)
        except Exception:
            return (1, 1)
    return (1, 1)


# ── copied verbatim from stacking_suite._get_image_size (PIL/tifffile lazy) ───
def _get_image_size(fp):
    ext = os.path.splitext(fp)[1].lower()

    if ext in (".fits", ".fit", ".fz"):
        hdr0 = fits.getheader(fp, ext=0)
        w = int(hdr0.get("NAXIS1", 0) or 0)
        h = int(hdr0.get("NAXIS2", 0) or 0)
        if not (w and h):
            raise IOError(f"Cannot read FITS size for {fp}")
        return w, h
    else:
        # try Pillow
        from PIL import Image
        try:
            with Image.open(fp) as img:
                w, h = img.size
        except Exception:
            # Pillow failed on TIFF or exotic format → try tifffile
            try:
                import tifffile as tiff
                arr = tiff.imread(fp)
                h, w = arr.shape[:2]
            except Exception:
                # last resort: OpenCV
                arr = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
                if arr is None:
                    raise IOError(f"Cannot read image size for {fp}")
                h, w = arr.shape[:2]
    return w, h


# ── copied verbatim from stacking_suite._quick_preview_any ────────────────────
def _quick_preview_any(fp: str, target_xbin: int, target_ybin: int):
    ext = os.path.splitext(fp)[1].lower()
    # 1) Load a small-ish *image* block without full decode work
    img = None; hdr = None

    def _superpixel2x2(x: np.ndarray) -> np.ndarray:
        h, w = x.shape[:2]
        h2, w2 = h - (h % 2), w - (w % 2)
        if h2 <= 0 or w2 <= 0:
            return x.astype(np.float32, copy=False)
        x = x[:h2, :w2].astype(np.float32, copy=False)
        if x.ndim == 2:
            return (x[0:h2:2, 0:w2:2] + x[0:h2:2, 1:w2:2] +
                    x[1:h2:2, 0:w2:2] + x[1:h2:2, 1:w2:2]) * 0.25
        else:
            r = x[..., 0]; g = x[..., 1]; b = x[..., 2]
            L = 0.2126*r + 0.7152*g + 0.0722*b
            return (L[0:h2:2, 0:w2:2] + L[0:h2:2, 1:w2:2] +
                    L[1:h2:2, 0:w2:2] + L[1:h2:2, 1:w2:2]) * 0.25

    try:
        if ext in (".fits", ".fit", ".fz"):
            # primary data only for speed
            data = fits.getdata(fp, ext=0)
            hdr  = fits.getheader(fp, ext=0)
            img = np.asanyarray(data)
        elif ext == ".xisf":
            from setiastro.saspro.legacy.xisf import XISF
            x = XISF(fp)
            ims = x.get_images_metadata()
            if not ims:
                return None
            hdr = ims[0]  # carry XISF image metadata dict as "header"
            img = x.read_image(0)  # channels-last
        else:
            # generic formats: use your existing thumb path
            w, h = _get_image_size(fp)
            # cheap preview load (PIL) — grayscale
            from PIL import Image
            im = Image.open(fp).convert("L")
            img = np.asarray(im, dtype=np.float32) / 255.0
            hdr = {}

        a = np.asarray(img)
        if a.ndim == 3 and a.shape[0] == 3 and a.shape[-1] != 3:
            a = np.transpose(a, (1, 2, 0))   # FITS colour is (3, H, W) -> HWC
        if a.ndim == 3 and a.shape[-1] == 1:
            a = a[...,0]
        # 2D CFA frames (BAYERPAT in the header) build the preview from the
        # MAX of each 2×2 Bayer block instead of its mean. The old flat
        # ``(R + Gr + Gb + B) / 4`` silently diluted narrowband-OSC star
        # peaks by ~4× — a bright Ha star lives in the single R pixel of
        # its 2×2 block, so averaging it with its three dark Bayer siblings
        # dropped it below sep's 5-σ detector and under-represented the
        # frame to the quality scorer. OSC measurement looked "hella
        # quicker" than mono because many narrowband stars never made it
        # past detection, so the capped-at-100 FWHM loop ran on a thinner
        # candidate set (SAME code, just a thinner preview). Taking the
        # per-block MAX preserves the peak regardless of which Bayer colour
        # carries the signal, so OSC and mono see comparable star density,
        # run the full candidate budget, and produce comparable weights.
        _cfa_bp = None
        if a.ndim == 2 and hdr is not None:
            _bp_raw = None
            try:
                _bp_raw = hdr.get("BAYERPAT") if hasattr(hdr, "get") else None
                if not _bp_raw and hasattr(hdr, "get"):
                    _fkw = hdr.get("FITSKeywords", None)
                    if isinstance(_fkw, dict):
                        _vals = _fkw.get("BAYERPAT", None)
                        if _vals:
                            _bp_raw = _vals[0].get("value")
            except Exception:
                _bp_raw = None
            if _bp_raw:
                _bpn = str(_bp_raw).strip().upper()
                if _bpn in ("RGGB", "BGGR", "GRBG", "GBRG"):
                    _cfa_bp = _bpn
        if _cfa_bp is not None:
            _h, _w = a.shape[:2]
            _h2, _w2 = _h - (_h % 2), _w - (_w % 2)
            if _h2 > 0 and _w2 > 0:
                _ca = a[:_h2, :_w2].astype(np.float32, copy=False)
                _tl = _ca[0:_h2:2, 0:_w2:2]; _tr = _ca[0:_h2:2, 1:_w2:2]
                _bl = _ca[1:_h2:2, 0:_w2:2]; _br = _ca[1:_h2:2, 1:_w2:2]
                prev2d = np.maximum(np.maximum(_tl, _tr), np.maximum(_bl, _br))
            else:
                prev2d = a.astype(np.float32, copy=False)
        elif a.ndim == 3 and a.shape[-1] == 3:
            prev2d = _superpixel2x2(a)
        else:
            prev2d = _superpixel2x2(a)

        # resample preview to the target bin scale
        xb, yb = _bin_from_header_fast_any(fp)
        sx = float(xb) / float(target_xbin)
        sy = float(yb) / float(target_ybin)
        if (abs(sx - 1.0) > 1e-6) or (abs(sy - 1.0) > 1e-6):
            prev2d = _resize_to_scale(prev2d, sx, sy)

        return np.ascontiguousarray(prev2d.astype(np.float32, copy=False))
    except Exception:
        return None


# ── copied verbatim from stacking_suite._star_count_ecc_size ──────────────────
def _star_count_ecc_size(preview, **kwargs):
    kwargs.setdefault("return_size", True)
    try:
        res = compute_star_count_fast_preview(preview, **kwargs)
    except Exception:
        return 0, 0.0, 0.0

    if isinstance(res, (tuple, list)):
        n = len(res)
        c    = res[0] if n >= 1 else 0
        ecc  = res[1] if n >= 2 else 0.0
        size = res[2] if n >= 3 else 0.0
    else:
        # bare scalar count
        c, ecc, size = res, 0.0, 0.0

    try:
        c = int(c)
    except Exception:
        c = 0
    try:
        ecc = float(ecc)
    except Exception:
        ecc = 0.0
    try:
        size = float(size)
    except Exception:
        size = 0.0
    return c, ecc, size


# ── copied verbatim from stacking_suite._measure_fwhm_halfres ─────────────────
def _measure_fwhm_halfres(img2d, *, max_stars=100, stamp=11, detect_sigma=5.0):
    a = np.asarray(img2d, dtype=np.float32)
    if a.ndim != 2 or a.size == 0:
        return 0.0
    if not np.isfinite(a).all():                 # fill NaN with background first
        fin = a[np.isfinite(a)]
        a = np.where(np.isfinite(a), a, float(np.median(fin)) if fin.size else 0.0)
    med = float(np.median(a))
    mad = float(np.median(np.abs(a - med))) * 1.4826
    if not (mad > 0):
        mad = float(a.std()) or 1e-6
    thr = med + detect_sigma * mad

    bw = (a > thr).astype(np.uint8)
    if int(bw.sum()) == 0:
        return 0.0
    try:
        n, _lbl, stats, cent = cv2.connectedComponentsWithStats(bw, connectivity=8)
    except Exception:
        return 0.0
    if n <= 1:
        return 0.0

    H, W = a.shape
    cands = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < 2 or area > 400:        # reject hot pixels & big blobs/galaxies
            continue
        cx, cy = cent[i]
        iy, ix = int(round(cy)), int(round(cx))
        if 0 <= iy < H and 0 <= ix < W:
            cands.append((float(a[iy, ix]), cx, cy))
    if not cands:
        return 0.0
    cands.sort(reverse=True)               # brightest first
    cands = cands[:max_stars]

    half = stamp // 2
    yy, xx = np.mgrid[0:stamp, 0:stamp]
    fwhms = []
    for _peak, cx, cy in cands:
        ix, iy = int(round(cx)), int(round(cy))
        if ix - half < 0 or iy - half < 0 or ix + half + 1 > W or iy + half + 1 > H:
            continue
        win = a[iy - half:iy + half + 1, ix - half:ix + half + 1].astype(np.float32) - med
        win[win < 0.0] = 0.0
        s = float(win.sum())
        if s <= 0.0:
            continue
        xbar = float((win * xx).sum() / s)
        ybar = float((win * yy).sum() / s)
        sxx = float((win * (xx - xbar) ** 2).sum() / s)
        syy = float((win * (yy - ybar) ** 2).sum() / s)
        sigma = math.sqrt(max(0.0, 0.5 * (sxx + syy)))   # round-equivalent σ
        if sigma > 0.0 and math.isfinite(sigma):
            fwhms.append(2.3548 * sigma)
    if not fwhms:
        return 0.0
    return float(np.median(fwhms))


# ── copied verbatim from stacking_suite._mad_noise ────────────────────────────
def _mad_noise(arr) -> float:
    a = np.asarray(arr, dtype=np.float32).ravel()
    a = a[np.isfinite(a)]                         # drop NaN
    if a.size == 0:
        return 0.0
    med = float(np.median(a))
    mad = float(np.median(np.abs(a - med)))
    return mad * 1.4826


# ── measurement cache (sidecar index) ────────────────────────────────────────
# A frame's measurement is a pure function of (its pixels, the target binning,
# the measurement code), so it can be cached. The first implementation put it
# in the frame's own FITS header; that was wrong, and the way it was wrong is
# worth recording.
#
# A FITS header is a PREFIX of the file. Adding a card past the 2880-byte
# blocks already allocated forces astropy to re-lay-out the file, which moves
# the data with it. Measured on 899 real 187 MB calibrated frames:
#
#     unstamped header: 67 cards, 5 free slots -- the stamp needed 14
#     first stamp:      372 ms, the whole 187 MB file rewritten
#     re-stamp:         3-6 ms, in place
#     measure:          1.65 s
#
# So nearly every frame paid a full rewrite, from 36 worker processes at once,
# on top of the 187 MB/frame reads. The drive saturated, the CPU went idle and
# the measurement phase went from 0.25 s/frame to 0.5-1.0 s/frame. 226 of the
# 899 had too little header room to EVER be cached. Packing the payload into
# one long CONTINUE card is not a way out either -- astropy rejects it.
#
# Hence a sidecar. One JSON index for the whole stacking directory, living in
# Measurements/ beside Calibrated/, Normalized_Images/, Aligned_Images/ and
# Master_Light/, keyed by each frame's absolute path -- so measuring the same
# frames from the calibrated folder or the aligned folder hits the same index.
#
# Validity is (size, mtime_ns). That needs one os.stat per frame and opens
# nothing, where the header scan it replaces cost ~1.6 ms/frame. It is also a
# STRONGER check than the old one: any edit to the file at all invalidates it,
# where the header stamp could only notice a changed geometry or Bayer pattern.
#
# The parent loads the index once, filters, runs the pool on the misses, then
# writes the index once. Workers never touch it -- that is what keeps the write
# contention that caused the regression from coming back.
#
# !! BUMP MEASURE_SCHEMA_VERSION whenever the measurement changes !!
# Any edit to _quick_preview_any, _star_count_ecc_size, _measure_fwhm_halfres,
# _mad_noise or measure_file's statistics changes what these numbers mean, and
# a stale cache would silently feed the old numbers to the weight scorer.
MEASURE_SCHEMA_VERSION = 3      # 3: dense fields star-counted in strips (sep_tiled)
                                # 2: sidecar index (1 was the SASM* header stamp)

MEASUREMENT_DIR_NAME = "Measurements"
MEASUREMENT_INDEX_NAME = "measurements.json"
_INDEX_MAX_ENTRIES = 100_000    # prune before this becomes a slow JSON load


def measurement_index_path(stacking_dir) -> str | None:
    """Where the index lives for a given stacking directory, or None."""
    if not stacking_dir:
        return None
    try:
        return os.path.join(str(stacking_dir), MEASUREMENT_DIR_NAME,
                            MEASUREMENT_INDEX_NAME)
    except Exception:
        return None


def _frame_key(fp: str) -> str:
    return os.path.normcase(os.path.abspath(str(fp)))


def _frame_stat(fp: str):
    """(size, mtime_ns) or None. One stat; the file is never opened."""
    try:
        st = os.stat(fp)
        return int(st.st_size), int(st.st_mtime_ns)
    except OSError:
        return None


def load_measurement_index(stacking_dir) -> dict:
    """Read the index. Any problem yields an empty index, never an exception --
    a cache that cannot be read just means everything gets measured."""
    path = measurement_index_path(stacking_dir)
    if not path or not os.path.exists(path):
        return {}
    try:
        with io.open(path, "r", encoding="utf-8") as fh:
            blob = json.load(fh)
        if not isinstance(blob, dict):
            return {}
        if int(blob.get("schema", -1)) != MEASURE_SCHEMA_VERSION:
            return {}               # measurement changed; the whole file is stale
        entries = blob.get("frames")
        return entries if isinstance(entries, dict) else {}
    except Exception:
        return {}


def save_measurement_index(stacking_dir, entries: dict) -> bool:
    """Write the index atomically. Drops entries whose file is gone, so the
    index cannot grow without bound as frames are deleted or re-stacked."""
    path = measurement_index_path(stacking_dir)
    if not path:
        return False
    try:
        live = {k: v for k, v in entries.items() if os.path.exists(k)}
        if len(live) > _INDEX_MAX_ENTRIES:
            # keep the most recently measured
            live = dict(sorted(live.items(),
                               key=lambda kv: kv[1].get("t", 0),
                               reverse=True)[:_INDEX_MAX_ENTRIES])
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with io.open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"schema": MEASURE_SCHEMA_VERSION, "frames": live}, fh)
        os.replace(tmp, path)       # atomic; a crash cannot leave a torn index
        return True
    except Exception:
        try:
            if os.path.exists(path + ".tmp"):
                os.remove(path + ".tmp")
        except Exception:
            pass
        return False


def index_lookup(entries: dict, fp, target_xbin, target_ybin):
    """Cached (mean, med, count, ecc, fwhm, noise) for this frame, or None.

    None for every reason: not indexed, the file changed, or it was measured
    at a different target binning. JSON round-trips Python floats exactly
    (json uses repr), so a cache hit reproduces a fresh measurement bit for
    bit -- which matters because these feed the weight scorer.
    """
    try:
        e = entries.get(_frame_key(fp))
        if not e:
            return None
        if int(e.get("xb", -1)) != int(target_xbin) or \
                int(e.get("yb", -1)) != int(target_ybin):
            return None
        stat = _frame_stat(fp)
        if stat is None or int(e.get("size", -1)) != stat[0] or \
                int(e.get("mtime", -1)) != stat[1]:
            return None
        payload = (float(e["mean"]), float(e["med"]), int(e["cnt"]),
                   float(e["ecc"]), float(e["fwhm"]), float(e["noise"]))
    except Exception:
        return None
    if not all(math.isfinite(v) for v in payload):
        return None                  # a NaN here would silently floor a weight
    return payload


def index_store(entries: dict, fp, target_xbin, target_ybin, payload) -> bool:
    """Record one measurement. Returns False if it could not be recorded."""
    try:
        mean, med, c, ecc, size, noise = payload
        vals = (float(mean), float(med), float(ecc), float(size), float(noise))
        if not all(math.isfinite(v) for v in vals):
            return False             # never cache a NaN
        stat = _frame_stat(fp)
        if stat is None:
            return False
        entries[_frame_key(fp)] = {
            "size": stat[0], "mtime": stat[1],
            "xb": int(target_xbin), "yb": int(target_ybin),
            "mean": vals[0], "med": vals[1], "cnt": int(c),
            "ecc": vals[2], "fwhm": vals[3], "noise": vals[4],
            "t": int(time.time()),
        }
        return True
    except Exception:
        return False


# ── the process-pool entry point ──────────────────────────────────────────────
def measure_file(fp, target_xbin, target_ybin, use_cache=False):
    """Load + measure one frame. Runs in a worker process.

    Returns (status, fp, payload), mirroring StackingSuite._measure_one exactly:
      ("ok",     fp, (mean, med, count, ecc, fwhm, noise))
      ("err",    fp, "<error>")    preview loaded but measurement failed
      ("noprev", fp, <error|None>) no usable image data — caller skips the frame

    `use_cache` is accepted and IGNORED. It is kept so older callers and any
    pickled job still work. Caching is now the parent's job via the sidecar
    index above: workers that wrote the cache themselves were the cause of the
    36-way file-write contention that made the measurement phase 4x slower.
    """
    try:
        preview = _quick_preview_any(fp, target_xbin, target_ybin)
    except Exception as e:
        return ("noprev", fp, f"{type(e).__name__}: {e}")
    if preview is None:
        return ("noprev", fp, None)
    try:
        preview = np.asarray(preview, dtype=np.float32)

        # NaN = satellite-trail no-data. The star detector (cv2) and plain
        # np.mean/np.median all return garbage/NaN on it, which floors the
        # frame's weight. Measure on the FINITE, NONZERO pixels only; give the
        # star detector a NaN-free copy (fill non-finite with the background).
        finite = np.isfinite(preview)
        vals   = preview[finite & (preview != 0.0)]
        if vals.size == 0:
            return ("err", fp, "no finite data in preview")

        pmin  = float(np.min(vals))
        med   = float(np.median(vals - pmin))     # background — the old Bg=nan
        mean  = float(np.mean(vals))

        bg_fill = float(np.median(vals))
        preview_filled = np.where(finite, preview, bg_fill).astype(np.float32, copy=False)

        c, ecc, _blind = _star_count_ecc_size(preview_filled)
        size  = _measure_fwhm_halfres(preview_filled)
        noise = _mad_noise(vals)                  # noise on finite pixels only
        return ("ok", fp, (mean, med, c, ecc, size, noise))
    except Exception as e:
        return ("err", fp, f"{type(e).__name__}: {e}")
