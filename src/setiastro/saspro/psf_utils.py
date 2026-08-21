# saspro/psf_utils.py
from __future__ import annotations
import numpy as np
import sep
sep.set_extract_pixstack(20000000)

from typing import Optional, Tuple

EPS = 1e-6


# =============================================================================
#  Shared descending-threshold star detection
#
#  Rich star fields at low sigma balloon sep.extract time (Frank's example:
#  53,105 sources at sigma=15 on one image), and most of those extra
#  detections are noise-level shape measurements that then pollute any
#  downstream median PSF / FWHM statistics. The waterfall runs sep.Background
#  ONCE (the expensive fixed cost) and re-extracts with progressively lower
#  sigma thresholds until we have enough well-formed stars for a stable
#  statistic, or we hit the user-specified floor.
#
#  This lives in psf_utils.py so both the PSF viewer and other star-based
#  analyses can share it. image_peeker_pro.py has an equivalent private copy
#  that should be refactored to import from here.
# =============================================================================
def detect_stars_waterfall(gray: np.ndarray,
                           sigma_ladder: Tuple[float, ...] = (100.0, 50.0, 25.0, 12.0, 6.0, 3.0),
                           target_count: int = 1000,
                           min_area: int = 5,
                           quality_filter: bool = True) -> Optional[dict]:
    """
    Descending-threshold star detection. Runs sep.Background once and
    re-extracts at each rung of `sigma_ladder` until the (optionally
    quality-filtered) catalogue has >= target_count entries, then stops.

    quality_filter (default True) drops SEP detections with:
      - flag != 0                   (saturation, blending, truncation)
      - npix < min_area             (single-pixel spikes, cosmic rays)
      - malformed ellipse (a > 0, b > 0, b <= a)
    Set False to preserve pre-filter counts (matches older callers that
    computed statistics over the raw extract).

    Returns None if even the last rung produced no usable stars, otherwise
    a dict of float64 arrays plus:
      n              : number of stars in the returned catalogue
      sigma_stopped  : the sigma level of the last extract we accepted
      total_at_stop  : raw sep.extract count at that sigma (pre-filter),
                       so callers can report "1024/17800 stars kept".
    """
    data = np.ascontiguousarray(gray, dtype=np.float32)
    try:
        bkg = sep.Background(data)
    except Exception:
        return None
    sub = data - bkg.back()
    try:
        rms = bkg.globalrms
    except Exception:
        rms = float(np.median(bkg.rms()))

    best: Optional[dict] = None
    for sig in sigma_ladder:
        try:
            stars = sep.extract(sub, thresh=float(sig), err=rms)
        except Exception:
            # Pixstack overflow at very low sigma on a huge field lands here;
            # keep whatever we already had and stop descending.
            break
        if stars is None or len(stars) == 0:
            continue

        names = stars.dtype.names
        n_raw = int(len(stars))
        if quality_filter:
            flag = stars['flag'] if 'flag' in names else np.zeros(n_raw, dtype=int)
            npix = stars['npix'] if 'npix' in names else np.full(n_raw, min_area + 1, dtype=int)
            a = stars['a']; b = stars['b']
            good = (flag == 0) & (npix >= min_area) & (a > 0) & (b > 0) & (b <= a)
        else:
            good = np.ones(n_raw, dtype=bool)

        n_good = int(good.sum())
        if n_good == 0:
            continue

        best = {
            'x':     stars['x'][good].astype(np.float64, copy=False),
            'y':     stars['y'][good].astype(np.float64, copy=False),
            'a':     stars['a'][good].astype(np.float64, copy=False),
            'b':     stars['b'][good].astype(np.float64, copy=False),
            'theta': stars['theta'][good].astype(np.float64, copy=False),
            'flux':  (stars['flux'] if 'flux' in names
                      else np.zeros(n_good))[good].astype(np.float64, copy=False),
            'flag':  (stars['flag'] if 'flag' in names
                      else np.zeros(n_good, dtype=int))[good].astype(np.int32, copy=False),
            'npix':  (stars['npix'] if 'npix' in names
                      else np.zeros(n_good, dtype=int))[good].astype(np.int32, copy=False),
            'n':             n_good,
            'sigma_stopped': float(sig),
            'total_at_stop': n_raw,
        }
        if n_good >= target_count:
            break

    return best


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
    return_info: bool = True,            # new: return (psf, info)
    target_stars: int = 1000,            # waterfall target (see detect_stars_waterfall)
) -> np.ndarray | tuple[np.ndarray, dict] | None:
    """
    Returns a normalized (ksize×ksize) PSF (or (psf, info) if return_info=True).
    - Detects stars via descending-sigma waterfall (fast on rich fields).
      det_sigma acts as the FLOOR of the waterfall; the ladder terminates
      early once target_stars well-formed detections are in hand, so a
      50k-star image doesn't burn seconds on shape fits we'll throw away.
    - Rejects saturated/elongated/low-flux sources.
    - Subpixel centers each cutout and median-combines.
    - Auto-selects ksize if None, or downsizes when stars are very small.
    """
    if image is None:
        return None
    img = _to_luma(image)
    info: dict = {}

    # Waterfall detection. Slot det_sigma in as the floor of the ladder so
    # that callers who genuinely need low-sigma get it, and higher det_sigma
    # values just skip cheaper rungs above them (no cost).
    ladder = tuple(s for s in (100.0, 50.0, 25.0, 12.0, 6.0, 3.0)
                   if s >= float(det_sigma))
    if not ladder:
        ladder = (float(det_sigma),)

    cat = detect_stars_waterfall(img, sigma_ladder=ladder,
                                 target_count=target_stars,
                                 quality_filter=True)
    if cat is None:
        return None

    a    = cat['a'].astype(np.float32, copy=False)
    b    = cat['b'].astype(np.float32, copy=False)
    flux = cat['flux'].astype(np.float32, copy=False)
    x    = cat['x']
    y    = cat['y']

    ecc = np.sqrt(1.0 - (b / np.maximum(a, 1e-9)) ** 2)

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
    info.update({
        "ksize":         k,
        "fwhm_med_px":   fwhm_med,
        "sigma_stopped": cat.get('sigma_stopped'),
        "total_at_stop": cat.get('total_at_stop'),
    })

    # Shape / flux filter on top of the waterfall's quality filter.
    idx = np.where(
        np.isfinite(a) & np.isfinite(b) &
        (a > 0.5) & (b > 0.5) &
        (ecc <= max_ecc) &
        (flux > min_flux)
    )[0]
    info["detected"] = int(cat['n'])

    if idx.size == 0:
        return None

    # Bright-ish first, cap
    idx = idx[np.argsort(-flux[idx])]
    idx = idx[:max_stars]

    # SEP-subtracted image for cutouts. The waterfall already subtracted
    # background inside detect_stars_waterfall, but only for detection --
    # rebuild it here so callers get a clean sky-subtracted cutout too.
    bkg = sep.Background(img)
    data = img - bkg.back()

    patches, rejected = [], 0
    for i in idx:
        cy, cx = float(y[i]), float(x[i])
        patch = _cutout(data, cy, cx, k)
        if patch is None:
            rejected += 1; continue
        peak = float(np.max(patch))
        center = float(patch[k // 2, k // 2])
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