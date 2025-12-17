# pro/star_metrics.py
from __future__ import annotations
import numpy as np
import sep

def measure_stars_sep(
    img_norm_mono: np.ndarray,
    *,
    thresh_sigma: float = 7.0,
    minarea: int = 16,
    deblend_nthresh: int = 32,
    aggregate: str = "median",
) -> tuple[int, float, float]:
    """
    img_norm_mono: float32 [0..1] 2D array.
    Returns (star_count, fwhm, ecc) using SEP:
      - FWHM = 2.3548 * sqrt(a*b)  (a,b are RMS ellipse axes)
      - ecc  = sqrt(1 - (b/a)^2)
    """
    try:
        # background & noise (consistent for both callers)
        bkg = sep.Background(img_norm_mono)
        back = bkg.back()
        grms = bkg.globalrms

        cat = sep.extract(
            img_norm_mono - back,
            thresh_sigma,
            err=grms,
            minarea=minarea,
            clean=True,
            deblend_nthresh=deblend_nthresh,
        )
        if cat is None or len(cat) == 0:
            return 0, 0.0, 0.0

        a = np.clip(cat["a"].astype(np.float32), 1e-3, None)
        b = np.clip(cat["b"].astype(np.float32), 1e-3, None)

        fwhm_vals = 2.3548 * np.sqrt(a * b)
        ratios = np.clip(b / a, 0.0, 1.0)
        ecc_vals = np.sqrt(1.0 - ratios * ratios)

        agg = np.nanmedian if aggregate == "median" else np.nanmean
        return len(cat), float(agg(fwhm_vals)), float(agg(ecc_vals))

    except Exception:
        # identical failure shape for both callers
        return 0, 0.0, 0.0
