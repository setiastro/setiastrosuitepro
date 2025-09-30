# pro/psf_utils.py
from __future__ import annotations
import numpy as np
import sep

EPS = 1e-6

def _to_luma(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2: return img.astype(np.float32, copy=False)
    r, g, b = img[...,0], img[...,1], img[...,2]
    return (0.2126*r + 0.7152*g + 0.0722*b).astype(np.float32)

def _cutout(arr, cy, cx, k):
    H, W = arr.shape
    y0 = int(round(cy)) - k//2
    x0 = int(round(cx)) - k//2
    y1, x1 = y0 + k, x0 + k
    if y0 < 0 or x0 < 0 or y1 > H or x1 > W:
        return None
    return arr[y0:y1, x0:x1].astype(np.float32, copy=False)

def _subpixel_shift_to_center(patch: np.ndarray) -> np.ndarray:
    """Shift brightest-peak to exact center using Fourier shift (sub-pixel)."""
    from scipy.ndimage import fourier_shift
    import numpy.fft as fft
    k = patch.shape[0]
    # peak location
    yy, xx = np.unravel_index(np.argmax(patch), patch.shape)
    cy = (k-1)/2
    shift = (cy-yy, cy-xx)
    F = fft.fftn(patch)
    Fs = fourier_shift(F, shift)
    out = fft.ifftn(Fs).real.astype(np.float32)
    return out

def compute_psf_kernel_for_image(
    image: np.ndarray,
    *,
    ksize: int = 21,
    det_sigma: float = 6.0,
    max_stars: int = 60,
    max_ecc: float = 0.5,        # 0 = round, ~0.5 ok
    min_flux: float = 0.0,
    max_frac_saturation: float = 0.8
) -> np.ndarray | None:
    """
    Returns a normalized (ksize×ksize) PSF kernel or None on failure.
    - Uses SEP to detect stars.
    - Rejects saturated, very elongated, and low-flux sources.
    - Median-combines sub-pixel centered cutouts.
    """
    if image is None:
        return None
    img = _to_luma(image)
    # Robust background
    bkg = sep.Background(img)
    data = img - bkg.back()
    try:
        err = bkg.globalrms
    except Exception:
        err = float(np.median(bkg.rms()))
    # Detect
    sources = sep.extract(data, det_sigma, err=err)
    if sources is None or len(sources) == 0:
        return None

    # Compute eccentricity proxy and filter
    a = np.array(sources["a"], dtype=np.float32)
    b = np.array(sources["b"], dtype=np.float32)
    ecc = np.sqrt(1.0 - (b / np.maximum(a, 1e-9))**2)  # 0..1
    flux = np.array(sources["flux"], dtype=np.float32)
    # Dynamic saturation guess: near-maximum pixel inside cutout
    # We'll do a quick pre-check using local max later.

    idx = np.where(
        (np.isfinite(a)) & (np.isfinite(b)) &
        (a > 0.5) & (b > 0.5) &
        (ecc <= max_ecc) &
        (flux > min_flux)
    )[0]

    if idx.size == 0:
        return None

    # Sort by flux descending and cap
    idx = idx[np.argsort(-flux[idx])]  # bright first
    idx = idx[:max_stars]

    patches = []
    for i in idx:
        cy, cx = float(sources["y"][i]), float(sources["x"][i])
        patch = _cutout(data, cy, cx, ksize)
        if patch is None:
            continue
        # Saturation test: center fraction must be below max_frac_saturation of its local peak
        peak = float(np.max(patch))
        center = patch[ksize//2, ksize//2]
        if peak <= EPS or (center / (peak + EPS)) > max_frac_saturation:
            # if center is already too close to peak value, it's likely saturated/flat-topped
            continue
        # Sub-pixel center and normalize
        try:
            patch = _subpixel_shift_to_center(patch)
        except Exception:
            pass
        s = float(np.sum(patch))
        if s <= 0:
            continue
        patches.append(patch / (s + EPS))

    if not patches:
        return None

    psf = np.median(np.stack(patches, axis=0), axis=0)
    s = float(np.sum(psf))
    if s <= 0:
        return None
    return (psf / (s + EPS)).astype(np.float32)
