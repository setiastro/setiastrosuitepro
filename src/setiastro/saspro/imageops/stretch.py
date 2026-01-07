# imageops/stretch.py
from __future__ import annotations
import numpy as np

# ---- Try Numba kernels from legacy ----
try:
    from setiastro.saspro.legacy.numba_utils import (
        numba_mono_from_img,
        numba_color_linked_from_img,
        numba_color_unlinked_from_img,

        # keep these too if other callers still use them
        numba_mono_final_formula,
        numba_color_final_formula_linked,
        numba_color_final_formula_unlinked,
    )
    _HAS_NUMBA = True
except Exception:
    _HAS_NUMBA = False

    def numba_mono_from_img(img, bp, denom, median_rescaled, target_median):
        r = (img - bp) / denom
        med = float(median_rescaled)
        num = (med - 1.0) * target_median * r
        den = med * (target_median + r - 1.0) - target_median * r
        den = np.where(np.abs(den) < 1e-12, 1e-12, den)
        return num / den

    def numba_color_linked_from_img(img, bp, denom, median_rescaled, target_median):
        r = (img - bp) / denom
        med = float(median_rescaled)
        num = (med - 1.0) * target_median * r
        den = med * (target_median + r - 1.0) - target_median * r
        den = np.where(np.abs(den) < 1e-12, 1e-12, den)
        return num / den

    def numba_color_unlinked_from_img(img, bp3, denom3, meds_rescaled3, target_median):
        bp3 = np.asarray(bp3, dtype=np.float32).reshape((1, 1, 3))
        denom3 = np.asarray(denom3, dtype=np.float32).reshape((1, 1, 3))
        meds = np.asarray(meds_rescaled3, dtype=np.float32).reshape((1, 1, 3))
        r = (img - bp3) / denom3
        num = (meds - 1.0) * target_median * r
        den = meds * (target_median + r - 1.0) - target_median * r
        den = np.where(np.abs(den) < 1e-12, 1e-12, den)
        return num / den


from setiastro.saspro.luminancerecombine import (
    LUMA_PROFILES,
    resolve_luma_profile_weights,
    compute_luminance,
    recombine_luminance_linear_scale,
    _estimate_noise_sigma_per_channel,   # <-- add this
)

def _sample_flat(x: np.ndarray, max_n: int = 400_000) -> np.ndarray:
    flat = np.asarray(x, np.float32).reshape(-1)
    n = flat.size
    if n <= max_n:
        return flat
    stride = max(1, n // max_n)
    return flat[::stride]

def _robust_sigma_lower_half_fast(x: np.ndarray, max_n: int = 400_000) -> float:
    s = _sample_flat(x, max_n=max_n)
    med = float(np.median(s))
    lo = s[s <= med]
    if lo.size < 16:
        mad = float(np.median(np.abs(s - med)))
    else:
        med_lo = float(np.median(lo))
        mad = float(np.median(np.abs(lo - med_lo)))
    return 1.4826 * mad

def _compute_blackpoint_sigma(img: np.ndarray, sigma: float) -> float:
    """
    Compute blackpoint using robust sigma so the slider actually works.
    Returns bp clamped to [min..0.99].
    """
    img = np.asarray(img, dtype=np.float32)
    med = float(np.median(img))
    sig = float(sigma)

    noise = _robust_sigma_lower_half_fast(img)
    bp = med - sig * noise

    # Clamp to valid range
    mn = float(img.min())
    bp = max(mn, bp)
    bp = min(bp, 0.99)
    return float(bp), med


def _compute_blackpoint_sigma_per_channel(img: np.ndarray, sigma: float) -> np.ndarray:
    """
    Per-channel version for unlinked color.
    """
    sig = float(sigma)
    bp = np.zeros(3, dtype=np.float32)
    for c in range(3):
        ch = img[..., c].astype(np.float32, copy=False)
        med = float(np.median(ch))
        noise = _robust_sigma_lower_half_fast(ch)
        b = med - sig * noise
        b = max(float(ch.min()), b)
        b = min(b, 0.99)
        bp[c] = b
    return bp

def hdr_compress_highlights(x: np.ndarray, amount: float, knee: float = 0.75) -> np.ndarray:
    """
    Smooth soft-knee highlight compression with C1 continuity at the knee.

    IMPORTANT:
    - We want highlights to get *dimmer* as amount increases.
    - For the Hermite curve on t in [0..1], keeping m0=1 and making m1>1
      puts the curve BELOW f(t)=t (compression), while still ending at 1.

    amount: 0..1 (0=off)
    knee:   0..1 where compression starts
    """
    a = float(np.clip(amount, 0.0, 1.0))
    if a <= 0.0:
        return x.astype(np.float32, copy=False)

    k = float(np.clip(knee, 0.0, 0.99))
    y = x.astype(np.float32, copy=False)

    hi = y > k
    if not np.any(hi):
        return np.clip(y, 0.0, 1.0).astype(np.float32, copy=False)

    # Normalize region above knee to t in [0..1]
    t = (y[hi] - k) / (1.0 - k)
    t = np.clip(t, 0.0, 1.0)

    # End slope at t=1:
    # a=0 -> m1=1 (identity)
    # a=1 -> m1=5 (stronger compression but still stable; avoid too-large slopes)
    m1 = 1.0 + 4.0 * a
    m1 = float(np.clip(m1, 1.0, 5.0))

    # Cubic Hermite: p0=0, p1=1, m0=1 (match slope at knee), m1=m1 (>1 compresses)
    t2 = t * t
    t3 = t2 * t

    h10 = (t3 - 2.0 * t2 + t)         # m0
    h01 = (-2.0 * t3 + 3.0 * t2)      # p1
    h11 = (t3 - t2)                   # m1

    f = h10 * 1.0 + h01 * 1.0 + h11 * m1

    y2 = y.copy()
    y2[hi] = k + (1.0 - k) * np.clip(f, 0.0, 1.0)

    return np.clip(y2, 0.0, 1.0).astype(np.float32, copy=False)


def hdr_compress_highlights_L(L: np.ndarray, amount: float, knee: float = 0.75) -> np.ndarray:
    """
    Same as hdr_compress_highlights(), but for luminance arrays.
    """
    a = float(np.clip(amount, 0.0, 1.0))
    if a <= 0.0:
        return L.astype(np.float32, copy=False)

    k = float(np.clip(knee, 0.0, 0.99))
    y = L.astype(np.float32, copy=False)

    hi = y > k
    if not np.any(hi):
        return np.clip(y, 0.0, 1.0).astype(np.float32, copy=False)

    t = (y[hi] - k) / (1.0 - k)
    t = np.clip(t, 0.0, 1.0)

    m1 = 1.0 + 4.0 * a
    m1 = float(np.clip(m1, 1.0, 5.0))

    t2 = t * t
    t3 = t2 * t

    h10 = (t3 - 2.0 * t2 + t)
    h01 = (-2.0 * t3 + 3.0 * t2)
    h11 = (t3 - t2)

    f = h10 * 1.0 + h01 * 1.0 + h11 * m1

    y2 = y.copy()
    y2[hi] = k + (1.0 - k) * np.clip(f, 0.0, 1.0)

    return np.clip(y2, 0.0, 1.0).astype(np.float32, copy=False)


def _resolve_rgb_weights_for_luma(method: str, w) -> np.ndarray:
    """
    Returns normalized RGB weights for recombine_luminance_linear_scale.
    method: rec709/rec601/rec2020 or anything else -> defaults to rec709.
    w: optional weights from resolve_luma_profile_weights
    """
    if w is not None and np.asarray(w).size == 3:
        rw = np.asarray(w, dtype=np.float32).copy()
        s = float(rw.sum())
        if s > 0:
            rw /= s
        else:
            rw = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
        return rw

    m = str(method).lower()
    if m == "rec601":
        return np.array([0.2990, 0.5870, 0.1140], dtype=np.float32)
    if m == "rec2020":
        return np.array([0.2627, 0.6780, 0.0593], dtype=np.float32)
    return np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def hdr_compress_color_luminance(
    rgb: np.ndarray,
    amount: float,
    knee: float,
    luma_mode: str = "rec709",
) -> np.ndarray:
    """
    WaveScaleHDR-style: compress highlights in luminance, then recombine by linear scaling.
    rgb: (H,W,3) float32 in [0..1] (or close).
    """
    a = float(np.clip(amount, 0.0, 1.0))
    if a <= 0.0:
        return rgb.astype(np.float32, copy=False)

    resolved_method, w, _ = resolve_luma_profile_weights(luma_mode)
    rw = _resolve_rgb_weights_for_luma(resolved_method, w)

    # Compute luminance from CURRENT rgb, compress luminance, recombine by scale
    if resolved_method == "snr":
        ns = _estimate_noise_sigma_per_channel(rgb)
        Y = compute_luminance(rgb, method="snr", weights=None, noise_sigma=ns)
    else:
        Y = compute_luminance(rgb, method=resolved_method, weights=rw)
    Yc = hdr_compress_highlights(Y, a, knee=float(knee))

    return recombine_luminance_linear_scale(
        rgb,
        Yc,
        weights=rw,
        blend=1.0,
        highlight_soft_knee=0.25,
    )

def _apply_mtf(data: np.ndarray, m: float) -> np.ndarray:
    """
    Midtones Transfer Function (PixInsight-style).
    Moves current median toward target without hard clipping.
    """
    m = float(m)
    x = data.astype(np.float32, copy=False)
    term1 = (m - 1.0) * x
    term2 = (2.0 * m - 1.0) * x - m
    with np.errstate(divide="ignore", invalid="ignore"):
        y = term1 / term2
    return np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32, copy=False)


def _compute_mtf_m_from_median(current_bg: float, target_bg: float) -> float:
    """
    Solve for 'm' such that MTF moves current median to target median.
    """
    cb = float(current_bg)
    tb = float(target_bg)
    cb = float(np.clip(cb, 1e-6, 1.0 - 1e-6))
    tb = float(np.clip(tb, 1e-6, 1.0 - 1e-6))

    den = cb * (2.0 * tb - 1.0) - tb
    if abs(den) < 1e-12:
        den = 1e-12
    m = (cb * (tb - 1.0)) / den
    return float(np.clip(m, 1e-6, 1.0 - 1e-6))


def _high_range_rescale_and_softclip(
    img: np.ndarray,
    target_bg: float,
    pedestal: float = 0.001,
    soft_ceil_pct: float = 99.0,
    hard_ceil_pct: float = 99.99,
    floor_sigma: float = 2.7,
    softclip_threshold: float = 0.98,
    softclip_rolloff: float = 2.0,
) -> np.ndarray:
    """
    VeraLux-like "ready-to-use" high range manager:
      - robust floor (median - k*sigma)
      - soft/hard ceilings (percentiles)
      - rescale with safety to avoid clipping
      - MTF median -> target_bg
      - soft clip rolloff near 1.0

    Expects HWC float32-ish, can be out of [0..1] (we fix it safely).
    """
    x = img.astype(np.float32, copy=False)

    # Compute luminance proxy for stats (works for mono too)
    if x.ndim == 2 or (x.ndim == 3 and x.shape[2] == 1):
        L = x.squeeze()
        is_rgb = False
    else:
        is_rgb = True
        # Rec709 luma proxy; we only use it for stats
        L = 0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]

    # Robust floor (use your existing robust sigma estimator)
    med = float(np.median(L))
    sig = float(_robust_sigma_lower_half_fast(L))
    global_floor = max(float(np.min(L)), med - float(floor_sigma) * sig)

    # Percentile ceilings (stride sample for speed)
    flat = L.reshape(-1)
    stride = max(1, flat.size // 500000)
    sample = flat[::stride]

    soft_ceil = float(np.percentile(sample, float(soft_ceil_pct)))
    hard_ceil = float(np.percentile(sample, float(hard_ceil_pct)))

    if soft_ceil <= global_floor:
        soft_ceil = global_floor + 1e-6
    if hard_ceil <= soft_ceil:
        hard_ceil = soft_ceil + 1e-6

    ped = float(np.clip(pedestal, 0.0, 0.05))

    # Contrast scale aims for 0.98, safety scale aims for 1.0
    scale_contrast = (0.98 - ped) / (soft_ceil - global_floor + 1e-12)
    scale_safety = (1.0 - ped) / (hard_ceil - global_floor + 1e-12)
    s = float(min(scale_contrast, scale_safety))

    y = (x - global_floor) * s + ped

    # Clamp to [0..1] before MTF + softclip
    y = np.clip(y, 0.0, 1.0).astype(np.float32, copy=False)

    # Recompute bg and apply MTF to land median near target
    if target_bg is not None:
        tb = float(target_bg)
        if 0.0 < tb < 1.0:
            if not is_rgb:
                cur = float(np.median(y.squeeze()))
            else:
                Ly = 0.2126 * y[..., 0] + 0.7152 * y[..., 1] + 0.0722 * y[..., 2]
                cur = float(np.median(Ly))

            if 0.0 < cur < 1.0 and abs(cur - tb) > 1e-3:
                m = _compute_mtf_m_from_median(cur, tb)
                y = _apply_mtf(y, m)
                y = np.clip(y, 0.0, 1.0).astype(np.float32, copy=False)

    # Final soft clip rolloff near highlights
    if softclip_threshold is not None and softclip_rolloff is not None:
        y = hdr_compress_highlights(y, amount=1.0, knee=float(softclip_threshold))
        # NOTE: hdr_compress_highlights() already does a hermite rolloff;
        # we map rolloff to that by using knee as threshold.

    return np.clip(y, 0.0, 1.0).astype(np.float32, copy=False)


# ---- Optional curves boost (gentle S-curve) ----
from functools import lru_cache

@lru_cache(maxsize=128)
def _calculate_curve_points(target_median: float, curves_boost: float):
    """Calculate curve control points with caching."""
    tm = float(target_median)
    cb = float(curves_boost)

    # These match your original formula
    p3x = 0.25 * (1.0 - tm) + tm
    p4x = 0.75 * (1.0 - tm) + tm
    p3y = p3x ** (1.0 - cb)
    p4y = (p4x ** (1.0 - cb)) ** (1.0 - cb)

    # Original 6-point curve
    xvals = np.array([
        0.0,
        0.5 * tm,
        tm,
        p3x,
        p4x,
        1.0
    ], dtype=np.float32)

    yvals = np.array([
        0.0,
        0.5 * tm,
        tm,
        p3y,
        p4y,
        1.0
    ], dtype=np.float32)
    
    return xvals, yvals

def apply_curves_adjustment(image: np.ndarray,
                            target_median: float,
                            curves_boost: float) -> np.ndarray:
    """
    curves_boost ∈ [0,1]. 0 = no change, 1 = strong S-curve.

    This reproduces the original Statistical Stretch curves behavior:
    we build a 1D curve from 6 control points and apply it as a
    piecewise-linear LUT over [0,1].
    """
    # No curve? Just return as-is (but float32 / clipped)
    if curves_boost <= 0.0:
        return np.clip(image, 0.0, 1.0).astype(np.float32, copy=False)

    img = np.clip(image.astype(np.float32, copy=False), 0.0, 1.0)

    # Get cached curve points
    xvals, yvals = _calculate_curve_points(target_median, curves_boost)

    # Apply the 1D LUT per channel using np.interp (piecewise linear)
    # Apply the 1D LUT per channel using np.interp (piecewise linear)
    # Optimization: np.interp can handle N-D 'x' array directly.
    # No need to loop over channels or flatten/reshape if we pass the whole array.
    
    out = np.interp(img, xvals, yvals).astype(np.float32, copy=False)

    return np.clip(out, 0.0, 1.0)

# ---- Public API used by Pro ----
def stretch_mono_image(image: np.ndarray,
                       target_median: float,
                       normalize: bool = False,
                       apply_curves: bool = False,
                       curves_boost: float = 0.0,
                       blackpoint_sigma: float = 5.0,
                       no_black_clip: bool = False,
                       hdr_compress: bool = False,
                       hdr_amount: float = 0.0,
                       hdr_knee: float = 0.75,
                       high_range: bool = False,
                       highrange_pedestal: float = 0.001,
                       highrange_soft_ceil_pct: float = 99.0,
                       highrange_hard_ceil_pct: float = 99.99,
                       highrange_softclip_threshold: float = 0.98,
                       highrange_softclip_rolloff: float = 2.0) -> np.ndarray:
    img = image.astype(np.float32, copy=False)

    sig = float(blackpoint_sigma)

    if no_black_clip:
        bp = float(img.min())
        med_img = float(np.median(img))  # only if you still need it
    else:
        bp, med_img = _compute_blackpoint_sigma(img, sig)

    denom = max(1.0 - bp, 1e-12)
    med_rescaled = (med_img - bp) / denom

    # NO rescaled array needed anymore
    out = numba_mono_from_img(img, bp, denom, float(med_rescaled), float(target_median))

    if apply_curves:
        out = apply_curves_adjustment(out, float(target_median), float(curves_boost))

    if hdr_compress and hdr_amount > 0.0:
        out = hdr_compress_highlights(out, float(hdr_amount), knee=float(hdr_knee))

    if normalize:
        mx = float(out.max())
        if mx > 0:
            out = out / mx

    if high_range:
        out = _high_range_rescale_and_softclip(
            out,
            target_bg=float(target_median),
            pedestal=float(highrange_pedestal),
            soft_ceil_pct=float(highrange_soft_ceil_pct),
            hard_ceil_pct=float(highrange_hard_ceil_pct),
            floor_sigma=float(blackpoint_sigma),
            softclip_threshold=float(highrange_softclip_threshold),
            softclip_rolloff=float(highrange_softclip_rolloff),
        )
        # After high-range manager, normalize is redundant; but keep behavior if user asked.
        if normalize:
            mx = float(out.max())
            if mx > 0:
                out = out / mx

    return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)


def stretch_color_image(image: np.ndarray,
                        target_median: float,
                        linked: bool = True,
                        normalize: bool = False,
                        apply_curves: bool = False,
                        curves_boost: float = 0.0,
                        blackpoint_sigma: float = 5.0,
                        no_black_clip: bool = False,
                        hdr_compress: bool = False,
                        hdr_amount: float = 0.0,
                        hdr_knee: float = 0.75,
                        luma_only: bool = False,
                        luma_mode: str = "rec709",
                        luma_blend: float = 1.0, 
                        high_range: bool = False,
                        highrange_pedestal: float = 0.001,
                        highrange_soft_ceil_pct: float = 99.0,
                        highrange_hard_ceil_pct: float = 99.99,
                        highrange_softclip_threshold: float = 0.98,
                        highrange_softclip_rolloff: float = 2.0) -> np.ndarray:
    img = image.astype(np.float32, copy=False)

    # Mono/single-channel
    if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
        mono = img.squeeze()
        mono_out = stretch_mono_image(
            mono,
            target_median,
            normalize=normalize,
            apply_curves=apply_curves,
            curves_boost=curves_boost,
            blackpoint_sigma=blackpoint_sigma,
            hdr_compress=hdr_compress,
            hdr_amount=hdr_amount,
            hdr_knee=hdr_knee,
            high_range=high_range,
            highrange_pedestal=highrange_pedestal,
            highrange_soft_ceil_pct=highrange_soft_ceil_pct,
            highrange_hard_ceil_pct=highrange_hard_ceil_pct,
            highrange_softclip_threshold=highrange_softclip_threshold,
            highrange_softclip_rolloff=highrange_softclip_rolloff,
        )
        return np.stack([mono_out] * 3, axis=-1)

    sig = float(blackpoint_sigma)

    # ----- LUMA ONLY PATH (now with optional blending) -----
    if luma_only:
        b = float(np.clip(luma_blend, 0.0, 1.0))

        # --- A) Normal linked RGB stretch (same settings, but NOT luma-only) ---
        # Force linked=True here (matches "normal linked stretch" expectation)
        # We compute this first so b=0 is fast-ish if you later optimize.
        if no_black_clip:
            bp = float(img.min())
            med_img = float(np.median(img))
        else:
            bp, med_img = _compute_blackpoint_sigma(img, sig)

        denom = max(1.0 - bp, 1e-12)
        med_rescaled = (med_img - bp) / denom

        linked_out = numba_color_linked_from_img(img, bp, denom, float(med_rescaled), float(target_median))

        if apply_curves:
            linked_out = apply_curves_adjustment(linked_out, float(target_median), float(curves_boost))

        if hdr_compress and hdr_amount > 0.0:
            linked_out = hdr_compress_color_luminance(
                linked_out,
                amount=float(hdr_amount),
                knee=float(hdr_knee),
                luma_mode="rec709",
            )

        if high_range:
            linked_out = _high_range_rescale_and_softclip(
                linked_out,
                target_bg=float(target_median),
                pedestal=float(highrange_pedestal),
                soft_ceil_pct=float(highrange_soft_ceil_pct),
                hard_ceil_pct=float(highrange_hard_ceil_pct),
                floor_sigma=float(blackpoint_sigma),
                softclip_threshold=float(highrange_softclip_threshold),
                softclip_rolloff=float(highrange_softclip_rolloff),
            )

        if normalize:
            mx = float(linked_out.max())
            if mx > 0:
                linked_out = linked_out / mx

        linked_out = np.clip(linked_out, 0.0, 1.0).astype(np.float32, copy=False)

        # Short-circuit if blend is 0 (pure linked)
        if b <= 0.0:
            return linked_out

        # --- B) Your existing luma-only recombine stretch ---
        resolved_method, w, _profile_name = resolve_luma_profile_weights(luma_mode)

        ns = None
        if resolved_method == "snr":
            ns = _estimate_noise_sigma_per_channel(img)
        L = compute_luminance(img, method=resolved_method, weights=w, noise_sigma=ns)

        Ls = stretch_mono_image(
            L,
            target_median,
            normalize=False,
            apply_curves=apply_curves,
            curves_boost=curves_boost,
            blackpoint_sigma=sig,
            no_black_clip=no_black_clip,
            hdr_compress=False,
            hdr_amount=0.0,
            hdr_knee=hdr_knee,
            high_range=False,
        )

        if hdr_compress and hdr_amount > 0.0:
            Ls = hdr_compress_highlights(Ls, float(hdr_amount), knee=float(hdr_knee))

        if w is not None and np.asarray(w).size == 3:
            rw = np.asarray(w, dtype=np.float32)
            s = float(rw.sum())
            if s > 0:
                rw = rw / s
        else:
            if resolved_method == "rec601":
                rw = np.array([0.2990, 0.5870, 0.1140], dtype=np.float32)
            elif resolved_method == "rec2020":
                rw = np.array([0.2627, 0.6780, 0.0593], dtype=np.float32)
            else:
                rw = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

        luma_out = recombine_luminance_linear_scale(
            img,
            Ls,
            weights=rw,
            blend=1.0,
            highlight_soft_knee=0.0,
        )

        if high_range:
            luma_out = _high_range_rescale_and_softclip(
                luma_out,
                target_bg=float(target_median),
                pedestal=float(highrange_pedestal),
                soft_ceil_pct=float(highrange_soft_ceil_pct),
                hard_ceil_pct=float(highrange_hard_ceil_pct),
                floor_sigma=float(blackpoint_sigma),
                softclip_threshold=float(highrange_softclip_threshold),
                softclip_rolloff=float(highrange_softclip_rolloff),
            )

        if normalize:
            mx = float(luma_out.max())
            if mx > 0:
                luma_out = luma_out / mx

        luma_out = np.clip(luma_out, 0.0, 1.0).astype(np.float32, copy=False)

        # --- Final blend: exactly “blend two separate stretched images” ---
        out = (1.0 - b) * linked_out + b * luma_out
        return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)

    # ----- NORMAL RGB PATH -----
    if linked:
        if no_black_clip:
            bp = float(img.min())
            med_img = float(np.median(img))
        else:
            bp, med_img = _compute_blackpoint_sigma(img, sig)

        denom = max(1.0 - bp, 1e-12)
        med_rescaled = (med_img - bp) / denom

        out = numba_color_linked_from_img(img, bp, denom, float(med_rescaled), float(target_median))
    else:
        if no_black_clip:
            bp3 = np.array([float(img[...,0].min()),
                            float(img[...,1].min()),
                            float(img[...,2].min())], dtype=np.float32)
            med_img3 = np.median(img, axis=(0, 1)).astype(np.float32)
        else:
            bp3 = _compute_blackpoint_sigma_per_channel(img, sig).astype(np.float32, copy=False)
            med_img3 = np.median(img, axis=(0, 1)).astype(np.float32)

        denom3 = np.maximum(1.0 - bp3, 1e-12).astype(np.float32)
        meds_rescaled3 = (med_img3 - bp3) / denom3

        out = numba_color_unlinked_from_img(img, bp3, denom3, meds_rescaled3, float(target_median))


    if apply_curves:
        out = apply_curves_adjustment(out, float(target_median), float(curves_boost))

    if hdr_compress and hdr_amount > 0.0:
        # Compress highlights on luminance, then recombine via linear scaling (prevents star bloat)
        out = hdr_compress_color_luminance(
            out,
            amount=float(hdr_amount),
            knee=float(hdr_knee),
            luma_mode="rec709",
        )

    if normalize:
        mx = float(out.max())
        if mx > 0:
            out = out / mx

    if high_range:
        out = _high_range_rescale_and_softclip(
            out,
            target_bg=float(target_median),
            pedestal=float(highrange_pedestal),
            soft_ceil_pct=float(highrange_soft_ceil_pct),
            hard_ceil_pct=float(highrange_hard_ceil_pct),
            floor_sigma=float(blackpoint_sigma),
            softclip_threshold=float(highrange_softclip_threshold),
            softclip_rolloff=float(highrange_softclip_rolloff),
        )

    return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)



def siril_style_autostretch(image, sigma=3.0):
    """
    Perform a 'Siril-style histogram stretch' using MAD for robust contrast enhancement.
    
    Parameters:
        image (np.ndarray): Input image, assumed to be normalized to [0, 1] range.
        sigma (float): How many MADs to stretch from the median.
    
    Returns:
        np.ndarray: Stretched image in [0, 1] range.
    """
    def stretch_channel(channel):
        median = np.median(channel)
        mad = np.median(np.abs(channel - median))
        min_val = np.min(channel)
        max_val = np.max(channel)

        # Convert MAD to an equivalent of std (optional, keep raw MAD if preferred)
        mad_std_equiv = mad * 1.4826

        black_point = max(min_val, median - sigma * mad_std_equiv)
        white_point = min(max_val, median + sigma * mad_std_equiv)

        if white_point - black_point <= 1e-6:
            return np.zeros_like(channel)  # Avoid divide-by-zero

        stretched = (channel - black_point) / (white_point - black_point)
        return np.clip(stretched, 0, 1)

    if image.ndim == 2:
        return stretch_channel(image)
    elif image.ndim == 3 and image.shape[2] == 3:
        return np.stack([stretch_channel(image[..., c]) for c in range(3)], axis=-1)
    else:
        raise ValueError("Unsupported image format for histogram stretch.")
