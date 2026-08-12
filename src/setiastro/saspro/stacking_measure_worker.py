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

import os
import math
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
        if a.ndim == 3 and a.shape[-1] == 1:
            a = a[...,0]
        # if it's color, make a luma-like 2×2 superpixel preview; if mono/CFA, same superpixel trick
        if a.ndim == 3 and a.shape[-1] == 3:
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
def _measure_fwhm_halfres(img2d, *, max_stars: int = 100, stamp: int = 11,
                          detect_sigma: float = 5.0) -> float:
    a = np.asarray(img2d, dtype=np.float32)
    if a.ndim != 2 or a.size == 0:
        return 0.0

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
    """Robust MAD-based noise estimate, star-insensitive."""
    a = np.asarray(arr, dtype=np.float32).ravel()
    if a.size == 0:
        return 0.0
    med = float(np.median(a))
    mad = float(np.median(np.abs(a - med)))
    return mad * 1.4826  # scale to Gaussian-sigma equivalent


# ── the process-pool entry point ──────────────────────────────────────────────
def measure_file(fp, target_xbin, target_ybin):
    """Load + measure one frame. Runs in a worker process.

    Returns (status, fp, payload), mirroring StackingSuite._measure_one exactly:
      ("ok",     fp, (mean, med, count, ecc, fwhm, noise))
      ("err",    fp, "<error>")    preview loaded but measurement failed
      ("noprev", fp, <error|None>) no usable image data — caller skips the frame
    """
    try:
        preview = _quick_preview_any(fp, target_xbin, target_ybin)
    except Exception as e:
        return ("noprev", fp, f"{type(e).__name__}: {e}")
    if preview is None:
        return ("noprev", fp, None)
    try:
        mean  = float(np.mean(preview))
        pmin  = float(np.nanmin(preview))
        med   = float(np.median(preview - pmin))
        c, ecc, _blind = _star_count_ecc_size(preview)
        size  = _measure_fwhm_halfres(preview)
        noise = _mad_noise(preview)
        return ("ok", fp, (mean, med, c, ecc, size, noise))
    except Exception as e:
        return ("err", fp, f"{type(e).__name__}: {e}")
