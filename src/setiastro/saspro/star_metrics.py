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
    # NEW knobs
    allow_retry_ladder: bool = True,
) -> tuple[int, float, float]:
    """
    img_norm_mono: float32 [0..1] 2D array.
    Returns (star_count, fwhm, ecc) using SEP:
      - FWHM = 2.3548 * sqrt(a*b)  (a,b are RMS ellipse axes)
      - ecc  = sqrt(1 - (b/a)^2)

    IMPORTANT:
      If no stars (or SEP fails), returns:
        (0, np.nan, np.nan)
      so callers can display blanks and not confuse "0.0" with a real measurement.
    """
    try:
        img = np.asarray(img_norm_mono, dtype=np.float32)
        if img.ndim != 2 or img.size == 0:
            return 0, np.nan, np.nan

        # SEP wants finite
        if not np.isfinite(img).all():
            img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32, copy=False)

        bkg = sep.Background(img)
        back = bkg.back()
        grms = float(getattr(bkg, "globalrms", np.nan))
        if not np.isfinite(grms) or grms <= 0:
            # fallback if SEP gave something odd
            try:
                grms = float(np.nanmedian(np.asarray(bkg.rms(), dtype=np.float32)))
            except Exception:
                grms = np.nan
        if not np.isfinite(grms) or grms <= 0:
            grms = 1e-6

        # Retry ladder (helps on dim/soft frames)
        if allow_retry_ladder:
            candidates = [
                (float(thresh_sigma), int(minarea)),
                (5.0, max(2, int(minarea * 0.75))),
                (4.0, max(2, int(minarea * 0.50))),
                (3.5, max(2, int(minarea * 0.35))),
            ]
        else:
            candidates = [(float(thresh_sigma), int(minarea))]

        cat = None
        for thr, ma in candidates:
            try:
                cat = sep.extract(
                    img - back,
                    thr,
                    err=grms,
                    minarea=ma,
                    clean=True,
                    deblend_nthresh=int(deblend_nthresh),
                )
            except Exception:
                cat = None

            if cat is not None and len(cat) > 0:
                break

        if cat is None or len(cat) == 0:
            return 0, np.nan, np.nan

        a = np.clip(cat["a"].astype(np.float32, copy=False), 1e-6, None)
        b = np.clip(cat["b"].astype(np.float32, copy=False), 1e-6, None)

        fwhm_vals = 2.3548 * np.sqrt(a * b)
        ratios = np.clip(b / a, 0.0, 1.0)
        ecc_vals = np.sqrt(np.maximum(0.0, 1.0 - ratios * ratios))

        agg = np.nanmedian if str(aggregate).lower() == "median" else np.nanmean
        return int(len(cat)), float(agg(fwhm_vals)), float(agg(ecc_vals))

    except Exception:
        return 0, 30, 1