#src/setiastro/saspro/ensemble_transform.py
# ─────────────────────────────────────────────────────────────────────────
# Sequence-specific photometric transform via ensemble linear regression
#
# Shared engine used by both the Magnitude tool and the Exoplanet Detector
# (re-exported from photometry_aavso as aav.fit_sequence_transform etc.).
# Depends only on numpy.
#
# Model (Gary, "CCD Transformation Equations for Use with Single-Image
# Photometry"; the same regression vsopy runs via scipy.stats.linregress):
#
#       (M_cat - m_inst) = T * CI + Z
#
#   m_inst = -2.5 log10(F); CI is the catalog colour index of each ensemble
#   star (e.g. B-V); T is the sequence colour term (slope); Z the zero point
#   (intercept). The classic path assumes T == 0 and Z = median(M_cat - m_inst)
#   — exactly the special case returned when the slope is not trustworthy, so
#   adopting this is behaviour-preserving on sparse fields and only *adds* a
#   colour term where the data support one.
#
# Two ways to apply the result:
#   • transform_target()          absolute form  M = m_inst + Z + T*CI_target
#                                 (single-epoch use; needs the fitted intercept)
#   • differential_correction()   delta  = T*(CI_target - color_ref)
#                                 (time-series use; keep your per-frame ZP and
#                                  add ONLY the colour term relative to the
#                                  ensemble's median colour)
# ─────────────────────────────────────────────────────────────────────────
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass
class SequenceTransform:
    """Result of an ensemble regression for one band.

    zp / color_term describe  (M_cat - m_inst) = color_term * CI + zp.
    When `use_color` is False the fit was not trustworthy and only `zp`
    (a robust median) should be used — that is the classic slope-1 ZP.
    `color_ref` is the median colour index of the comps that survived
    clipping; it is the reference point for differential_correction().
    """
    zp: float                       # intercept Z (mag)
    zp_err: float                   # 1σ on Z
    color_term: float               # slope T (mag / mag of colour)
    color_term_err: float           # 1σ on T
    rvalue: float                   # Pearson r of the fit (sequence quality)
    resid_rms: float                # RMS of residuals (mag) after clipping
    color_ref: float                # median CI of kept comps (nan if none)
    n: int                          # comp stars used in the final fit
    use_color: bool                 # True → apply color_term; False → ZP only
    keep_mask: np.ndarray           # bool mask into the input arrays

    def note(self, band_key: str) -> str:
        """One-line provenance string for the AAVSO NOTES field."""
        if self.use_color:
            return (f"ensemble regression vs {band_key}: "
                    f"Z={self.zp:.3f}±{self.zp_err:.3f}, "
                    f"T={self.color_term:+.3f}±{self.color_term_err:.3f}, "
                    f"r={self.rvalue:.3f}, rms={self.resid_rms:.3f}, N={self.n}")
        return (f"ensemble ZP vs {band_key}: Z={self.zp:.3f} "
                f"(median; colour term not significant), "
                f"rms={self.resid_rms:.3f}, N={self.n}")


def _ols(x: np.ndarray, y: np.ndarray):
    """Plain OLS y = slope*x + intercept. Returns (slope, intercept,
    slope_err, intercept_err, rvalue). Mirrors scipy.stats.linregress but
    keeps the module scipy-free. Assumes len>=3 and non-degenerate x."""
    n = x.size
    xbar = x.mean(); ybar = y.mean()
    dx = x - xbar; dy = y - ybar
    Sxx = float((dx * dx).sum())
    Syy = float((dy * dy).sum())
    Sxy = float((dx * dy).sum())
    slope = Sxy / Sxx
    intercept = ybar - slope * xbar
    resid = y - (intercept + slope * x)
    dof = max(n - 2, 1)
    s2 = float((resid * resid).sum()) / dof
    slope_err = float(np.sqrt(s2 / Sxx))
    intercept_err = float(np.sqrt(s2 * (1.0 / n + xbar * xbar / Sxx)))
    rval = 0.0 if Syy <= 0 else float(Sxy / np.sqrt(Sxx * Syy))
    return slope, intercept, slope_err, intercept_err, rval


def fit_sequence_transform(
    cat_mag,                       # M_cat: catalog mag in the target band
    inst_mag,                      # m_inst = -2.5 log10(F)  (NOT flux)
    color_index=None,             # CI per comp star (e.g. B-V); None → ZP only
    *,
    min_n_for_color: int = 5,      # below this, don't trust a fitted slope
    slope_sig: float = 2.0,        # require |T| >= slope_sig * T_err to apply
    clip_sigma: float = 3.0,       # MAD sigma-clip on residuals
    clip_iters: int = 3,
    sigma_floor: float = 0.05,     # don't over-clip a tight sequence
) -> SequenceTransform:
    """Fit the sequence transform for one band from an ensemble of comp stars.

    Robustly rejects bad comp stars (MAD clip on the fit residuals — a wrong
    catalog mag gets clipped instead of masquerading as scatter), then decides
    whether the colour term is statistically supported. If it is not (too few
    stars, flat/degenerate colour range, or slope within `slope_sig`σ of zero),
    it returns the classic robust-median zero point with color_term = 0 and
    use_color = False, so callers keep their current behaviour on sparse fields.
    """
    M = np.asarray(cat_mag, dtype=np.float64)
    mi = np.asarray(inst_mag, dtype=np.float64)
    y = M - mi                                  # == m_cat + 2.5 log10 F

    finite = np.isfinite(y)
    have_color = color_index is not None
    CI = np.asarray(color_index, dtype=np.float64) if have_color else None
    if have_color:
        finite &= np.isfinite(CI)

    keep = finite.copy()

    def _cref(mask) -> float:
        if not have_color:
            return float("nan")
        cc = CI[mask]
        return float(np.median(cc)) if cc.size else float("nan")

    def _median_only(mask) -> SequenceTransform:
        yy = y[mask]
        if yy.size == 0:
            return SequenceTransform(np.nan, np.nan, 0.0, np.nan, 0.0,
                                     np.nan, _cref(mask), 0, False, mask)
        z = float(np.median(yy))
        rms = float(np.sqrt(np.mean((yy - z) ** 2))) if yy.size else np.nan
        zerr = (float(1.4826 * np.median(np.abs(yy - z)) / np.sqrt(yy.size))
                if yy.size > 1 else np.nan)
        return SequenceTransform(z, zerr, 0.0, np.nan, 0.0, rms,
                                 _cref(mask), int(yy.size), False, mask)

    # Not enough to regress, or no colour info → robust median ZP.
    if not have_color or int(keep.sum()) < 3:
        for _ in range(clip_iters):
            yy = y[keep]
            if yy.size < 3:
                break
            med = np.median(yy)
            sd = max(1.4826 * np.median(np.abs(yy - med)), sigma_floor)
            newkeep = keep.copy()
            newkeep[keep] = np.abs(yy - med) <= clip_sigma * sd
            if np.array_equal(newkeep, keep):
                break
            keep = newkeep
        return _median_only(keep)

    # Degenerate colour range (all comps ~same colour) → slope meaningless.
    if float(np.ptp(CI[keep])) < 1e-3:
        return _median_only(keep)

    # Robust regression: fit, MAD-clip residuals, refit.
    slope = intercept = slope_err = intercept_err = rval = np.nan
    for _ in range(clip_iters):
        xk = CI[keep]; yk = y[keep]
        if xk.size < 3 or float(np.ptp(xk)) < 1e-3:
            break
        slope, intercept, slope_err, intercept_err, rval = _ols(xk, yk)
        resid = yk - (intercept + slope * xk)
        sd = max(1.4826 * np.median(np.abs(resid - np.median(resid))), sigma_floor)
        newkeep = keep.copy()
        newkeep[keep] = np.abs(resid - np.median(resid)) <= clip_sigma * sd
        if np.array_equal(newkeep, keep) or int(newkeep.sum()) < 3:
            break
        keep = newkeep

    n = int(keep.sum())
    xk = CI[keep]; yk = y[keep]
    if n < 3 or float(np.ptp(xk)) < 1e-3 or not np.isfinite(slope):
        return _median_only(keep)

    resid = yk - (intercept + slope * xk)
    resid_rms = float(np.sqrt(np.mean(resid ** 2)))
    cref = float(np.median(xk))

    use_color = (
        n >= min_n_for_color
        and np.isfinite(slope_err) and slope_err > 0
        and abs(slope) >= slope_sig * slope_err
    )
    if not use_color:
        z = float(np.median(yk))
        zerr = float(1.4826 * np.median(np.abs(yk - z)) / np.sqrt(n)) if n > 1 else np.nan
        return SequenceTransform(z, zerr, 0.0, slope_err, rval,
                                 resid_rms, cref, n, False, keep)

    return SequenceTransform(float(intercept), float(intercept_err),
                             float(slope), float(slope_err), float(rval),
                             resid_rms, cref, n, True, keep)


def transform_target(xfm: SequenceTransform, inst_mag_target: float,
                     color_index_target: Optional[float] = None
                     ) -> Tuple[float, bool]:
    """Absolute form: standard magnitude from the fitted intercept + colour term.

    Returns (mag, transformed). Use this for single-epoch work where the fitted
    intercept IS the zero point. For a per-frame ZP time series use
    differential_correction() instead. Falls back to ZP-only when the fit did
    not support a colour term or the target colour is unknown.
    """
    if xfm.use_color and color_index_target is not None and np.isfinite(color_index_target):
        return float(inst_mag_target + xfm.zp + xfm.color_term * color_index_target), True
    return float(inst_mag_target + xfm.zp), False


def differential_correction(xfm: SequenceTransform, color_index_target,
                            color_index_target_err=0.0):
    """Differential colour term relative to the ensemble median colour.

    Returns (delta_mag, delta_err) to ADD to a magnitude that already carries a
    per-frame ensemble zero point:  m_corrected = m + delta.  Because the
    per-frame ZP equals (approximately) Z + T*color_ref, adding
    T*(CI_target - color_ref) recovers the fully colour-corrected magnitude
    without touching the frame-to-frame differential ZP. Scalars or arrays.

    When the fit did not support a colour term (use_color False) or color_ref
    is undefined, returns zeros so the caller's magnitude is unchanged.
    """
    ct = np.asarray(color_index_target, dtype=np.float64)
    if not xfm.use_color or not np.isfinite(xfm.color_ref):
        return np.zeros_like(ct), np.zeros_like(ct)
    dcol = ct - float(xfm.color_ref)
    delta = xfm.color_term * dcol
    derr = np.sqrt((xfm.color_term_err * dcol) ** 2
                   + (xfm.color_term * np.asarray(color_index_target_err,
                                                   dtype=np.float64)) ** 2)
    return delta, derr


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    N = 30
    CI = rng.uniform(0.2, 1.6, N)
    M = rng.uniform(11, 16, N)
    m_inst = M - (21.0 + 0.08 * CI) + rng.normal(0, 0.02, N)
    M[3] += 1.5; M[17] -= 1.2                    # inject two bad catalog mags
    xfm = fit_sequence_transform(M, m_inst, CI)
    print(xfm.note("V"))
    assert xfm.use_color and xfm.n >= 25
    assert abs(xfm.color_term - 0.08) < 0.03 and abs(xfm.zp - 21.0) < 0.03
    assert np.isfinite(xfm.color_ref)
    assert not xfm.keep_mask[3] and not xfm.keep_mask[17]

    # differential correction is zero at the reference colour, T-scaled away from it
    d0, _ = differential_correction(xfm, xfm.color_ref)
    d1, _ = differential_correction(xfm, xfm.color_ref + 1.0)
    assert abs(float(d0)) < 1e-9 and abs(float(d1) - xfm.color_term) < 1e-9

    # Sparse / no-colour field → behaviour-preserving median ZP, no colour term
    xfm2 = fit_sequence_transform(M[:3], m_inst[:3], None)
    print(xfm2.note("V"))
    assert not xfm2.use_color and xfm2.color_term == 0.0
    dd, ee = differential_correction(xfm2, 0.7)
    assert float(dd) == 0.0 and float(ee) == 0.0
    print("OK")