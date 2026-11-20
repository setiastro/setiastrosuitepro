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
                           quality_filter: bool = True,
                           strips: bool = False) -> Optional[dict]:
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

    strips=True extracts a dense rung over horizontal strips
    (sep_tiled.extract_strips): the same catalogue, but sep.extract holds the
    GIL for the whole call -- one full-frame call on a rich 26 MP master is
    ~6 s at sigma 100 and ~3 min at sigma 15 with the GUI frozen -- and strips
    hold it ~0.5 s at a time (and are faster there). For callers on a QThread
    in the GUI process. Only dense star-field rungs are stripped (_strips_pay);
    elsewhere one call is faster and holds the GIL about as long.

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
            if strips and _strips_pay(sub, float(sig) * float(rms), int(min_area)):
                from setiastro.saspro.sep_tiled import extract_strips
                stars = extract_strips(sep, sub, float(sig), err=rms, minarea=int(min_area))
            else:
                stars = sep.extract(sub, thresh=float(sig), err=rms, minarea=int(min_area))
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


# Strips keep each GIL hold short, but every strip call also sets up SEP's
# process-global pixel stack (20M in the GUI process, ~0.25 s a call), so they
# only pay on a dense star field, where one call's cost grows faster than its
# object count. On nebulosity they lose: an object taller than two strip
# margins makes every strip it crosses re-extract with wider margins, so the
# strips cost several times one call and their GIL holds grow toward one
# call's anyway. Connected components of the smoothed, thresholded image
# predict both (object count and tall objects), and cv2 counts them without
# holding the GIL. Measured on real masters, one call vs 256-row strips:
#   star fields  9.2k objects, 0 tall   6.4 s vs 5.4 s  (longest hold 0.44 s)
#                41.7k objects, 0 tall  ~3 min vs 33 s (4.2 s)
#                8.2k objects, 3 tall   13.1 s vs 5.9 s (0.64 s)
#                13.1k objects, 5 tall  45 s vs 25 s    (3.3 s)
#   nebulosity   5.8k objects, 11 tall  7.4 s vs 26 s   (4.2 s)  -> one call
#                2.9k objects           2.9 s vs 12.6 s          -> one call
_STRIP_MIN_OBJECTS = 4000
_STRIP_TALL_ROWS = 128          # 2 x sep_tiled.extract_strips' default margin
_SEP_SMOOTH = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], np.float32) / 16.0  # SEP's default filter


def _strips_pay(sub: np.ndarray, level: float, min_area: int) -> bool:
    """Is extracting at absolute `level` a dense star field, where strips beat one call?"""
    min_area = max(1, int(min_area))
    if int(np.count_nonzero(sub > level)) <= _STRIP_MIN_OBJECTS * min_area:
        return False                       # too few pixels for that many objects
    try:
        import cv2
        sm = cv2.filter2D(sub, -1, _SEP_SMOOTH, borderType=cv2.BORDER_REPLICATE)
        _, _, stats, _ = cv2.connectedComponentsWithStats((sm > level).astype(np.uint8), connectivity=8)
    except Exception:
        return False                       # one call, as without strips
    st = stats[1:]
    st = st[st[:, cv2.CC_STAT_AREA] >= min_area]
    n_obj = int(len(st))
    if n_obj <= _STRIP_MIN_OBJECTS:
        return False
    n_tall = int(np.count_nonzero(st[:, cv2.CC_STAT_HEIGHT] > _STRIP_TALL_ROWS))
    return n_tall * 1000 <= n_obj          # at most ~1 extended object per 1000


def median_star_stamp(gray: np.ndarray, xs, ys, half: int,
                      max_stars: int = 500) -> Optional[np.ndarray]:
    """
    Median of real-image cutouts around detected stars: what an actual star
    in this image looks like at native pixel resolution. The PSF Viewer's
    "true pixels" star and Convolution/Deconvolution's stellar PSF both come
    from here.

    Each (2*half+1)^2 cutout is centred on the rounded centroid, has the
    median of its 1-pixel border subtracted as background and is divided by
    its peak, so bright stars don't dominate the median and a neighbour in one
    cutout doesn't survive it. Cutouts that leave the image are skipped. Over
    max_stars detections, an even spread through the catalogue is used.
    Returns peak-normalised float32 (peak ~1), or None if no cutout fit.
    """
    img = np.ascontiguousarray(gray, dtype=np.float32)
    H, W = img.shape
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if xs.size == 0:
        return None

    # Cap for speed -- 500 stamps is plenty to nail a median.
    if xs.size > max_stars:
        idx = np.linspace(0, xs.size - 1, max_stars).astype(int)
        xs = xs[idx]; ys = ys[idx]

    stamps = []
    for xi, yi in zip(xs, ys):
        cx = int(round(xi)); cy = int(round(yi))
        x0 = cx - half; x1 = cx + half + 1
        y0 = cy - half; y1 = cy + half + 1
        if x0 < 0 or y0 < 0 or x1 > W or y1 > H:
            continue
        s = img[y0:y1, x0:x1]
        # Background from the 1-pixel edge of the stamp
        border = np.concatenate([s[0], s[-1], s[1:-1, 0], s[1:-1, -1]])
        bg = float(np.median(border))
        s  = s - bg
        pk = float(s.max())
        if pk > 0.0:
            stamps.append(s / pk)

    if not stamps:
        return None
    return np.median(np.stack(stamps, axis=0), axis=0).astype(np.float32)


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