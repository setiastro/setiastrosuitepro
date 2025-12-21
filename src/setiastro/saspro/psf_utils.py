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
    ksize: int | None = 21,
    det_sigma: float = 6.0,
    max_stars: int = 60,
    max_ecc: float = 0.5,
    min_flux: float = 0.0,
    max_frac_saturation: float = 0.98,   # was 0.8 → far too strict
    return_info: bool = True             # new: return (psf, info)
) -> np.ndarray | tuple[np.ndarray, dict] | None:
    """
    Returns a normalized (ksize×ksize) PSF (or (psf, info) if return_info=True).
    - SEP detects stars; rejects saturated/elongated/low-flux sources.
    - Subpixel centers each cutout and median-combines.
    - Auto-selects ksize if None, or downsizes when stars are very small.
    """
    if image is None:
        return None
    img = _to_luma(image)
    info = {}

    # Robust background & detection
    bkg = sep.Background(img)
    data = img - bkg.back()
    try: err = bkg.globalrms
    except Exception: err = float(np.median(bkg.rms()))
    sources = sep.extract(data, det_sigma, err=err)
    if sources is None or len(sources) == 0:
        return None

    # Star size & shape
    a = np.array(sources["a"], dtype=np.float32)   # SEP Gaussian sigma along major axis (≈ σ)
    b = np.array(sources["b"], dtype=np.float32)
    ecc = np.sqrt(1.0 - (b / np.maximum(a, 1e-9))**2)
    flux = np.array(sources["flux"], dtype=np.float32)

    # Estimate typical FWHM in px from 'a' (use median of central bulk)
    good_a = a[np.isfinite(a) & (a > 0.5)]
    if good_a.size:
        sigma_med = float(np.median(good_a))
        fwhm_med  = 2.3548 * sigma_med
    else:
        sigma_med, fwhm_med = 1.2, 2.8  # fallback

    # Auto kernel size if None or wildly big vs star size
    if (ksize is None) or (ksize > int(6.0 * sigma_med) + 1):
        ksize = int(2 * np.ceil(3.0 * sigma_med) + 1)
        ksize = int(np.clip(ksize, 9, 25))  # clamp to practical window
    k = int(ksize) | 1  # enforce odd
    info.update({"ksize": k, "fwhm_med_px": fwhm_med})

    # Filtering
    idx = np.where(
        (np.isfinite(a)) & (np.isfinite(b)) &
        (a > 0.5) & (b > 0.5) &
        (ecc <= max_ecc) &
        (flux > min_flux)
    )[0]
    info["detected"] = int(len(sources))

    if idx.size == 0:
        return None

    # Bright-ish first, cap
    idx = idx[np.argsort(-flux[idx])]
    idx = idx[:max_stars]

    patches, rejected = [], 0
    for i in idx:
        cy, cx = float(sources["y"][i]), float(sources["x"][i])
        patch = _cutout(data, cy, cx, k)
        if patch is None:
            rejected += 1; continue
        peak = float(np.max(patch))
        center = float(patch[k//2, k//2])
        # reject *obvious* clipped cores only
        if peak > 0 and (center / (peak + EPS)) >= max_frac_saturation:
            rejected += 1; continue
        try:
            patch = _subpixel_shift_to_center(patch)
        except Exception:
            pass
        s = float(np.sum(patch))
        if s <= 0:
            rejected += 1; continue
        patches.append(patch / (s + EPS))

    info["rejected"] = int(rejected)
    info["used_stars"] = int(len(patches))

    if not patches:
        return None

    psf = np.median(np.stack(patches, axis=0), axis=0).astype(np.float32, copy=False)
    s = float(psf.sum())
    if s <= 0:
        return None
    psf = psf / (s + EPS)
    return (psf, info) if return_info else psf

