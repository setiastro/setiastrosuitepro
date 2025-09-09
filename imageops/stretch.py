# imageops/stretch.py
from __future__ import annotations
import numpy as np

# ---- Try Numba kernels from legacy ----
try:
    from legacy.numba_utils import (
        numba_mono_final_formula,
        numba_color_final_formula_linked,
        numba_color_final_formula_unlinked,
    )
    _HAS_NUMBA = True
except Exception:
    _HAS_NUMBA = False

    # Vectorized fallbacks (no Numba)
    def numba_mono_final_formula(rescaled, median_rescaled, target_median):
        r = rescaled
        med = float(median_rescaled)
        num = (med - 1.0) * target_median * r
        den = med * (target_median + r - 1.0) - target_median * r
        den = np.where(np.abs(den) < 1e-12, 1e-12, den)
        return num / den

    def numba_color_final_formula_linked(rescaled, median_rescaled, target_median):
        r = rescaled
        med = float(median_rescaled)
        num = (med - 1.0) * target_median * r
        den = med * (target_median + r - 1.0) - target_median * r
        den = np.where(np.abs(den) < 1e-12, 1e-12, den)
        return num / den

    def numba_color_final_formula_unlinked(rescaled, medians_rescaled, target_median):
        r = rescaled
        med = np.asarray(medians_rescaled, dtype=np.float32).reshape((1, 1, 3))
        num = (med - 1.0) * target_median * r
        den = med * (target_median + r - 1.0) - target_median * r
        den = np.where(np.abs(den) < 1e-12, 1e-12, den)
        return num / den


# ---- Optional curves boost (gentle S-curve) ----
def apply_curves_adjustment(img: np.ndarray, target_median: float, strength: float) -> np.ndarray:
    """
    strength ∈ [0,1]. 0=no change, 1=strong S-curve centered ~ target_median.
    """
    if strength <= 0:
        return img

    x = np.clip(img, 0.0, 1.0).astype(np.float32, copy=False)
    m = float(target_median)
    k = 8.0 * float(strength)  # steepness

    # smoothstep-ish curve centered at m
    s = 1.0 / (1.0 + np.exp(-k * (x - m)))  # 0..1
    # Normalize so s(m) ~ 0.5 and blend with identity
    out = (1.0 - strength) * x + strength * s
    return out.astype(np.float32, copy=False)


# ---- Public API used by Pro ----
def stretch_mono_image(image: np.ndarray,
                       target_median: float,
                       normalize: bool = False,
                       apply_curves: bool = False,
                       curves_boost: float = 0.0) -> np.ndarray:
    """
    image: float32 preferred, ~[0..1]. Returns float32 in [0..1].
    """
    img = image.astype(np.float32, copy=False)

    # Black point from SASv2 logic
    med = float(np.median(img))
    std = float(np.std(img))
    bp = max(float(img.min()), med - 2.7 * std)
    denom = 1.0 - bp
    if abs(denom) < 1e-12:
        denom = 1e-12

    rescaled = (img - bp) / denom
    med_rescaled = float(np.median(rescaled))

    out = numba_mono_final_formula(rescaled, med_rescaled, float(target_median))

    if apply_curves:
        out = apply_curves_adjustment(out, float(target_median), float(curves_boost))
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
                        curves_boost: float = 0.0) -> np.ndarray:
    """
    image: float32 preferred, ~[0..1]. Returns float32 in [0..1].
    """
    img = image.astype(np.float32, copy=False)

    # Mono/single-channel → reuse mono path and broadcast to 3-ch for display
    if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
        mono = img.squeeze()
        mono_out = stretch_mono_image(mono, target_median, normalize=normalize,
                                      apply_curves=apply_curves, curves_boost=curves_boost)
        return np.stack([mono_out] * 3, axis=-1)

    # Color
    if linked:
        comb_med = float(np.median(img))
        comb_std = float(np.std(img))
        bp = max(float(img.min()), comb_med - 2.7 * comb_std)
        denom = 1.0 - bp
        if abs(denom) < 1e-12:
            denom = 1e-12

        rescaled = (img - bp) / denom
        med_rescaled = float(np.median(rescaled))
        out = numba_color_final_formula_linked(rescaled, med_rescaled, float(target_median))
    else:
        rescaled = np.empty_like(img, dtype=np.float32)
        meds = np.zeros(3, dtype=np.float32)
        for c in range(3):
            ch = img[..., c]
            ch_med = float(np.median(ch))
            ch_std = float(np.std(ch))
            bp = max(float(ch.min()), ch_med - 2.7 * ch_std)
            denom = 1.0 - bp
            if abs(denom) < 1e-12:
                denom = 1e-12
            rescaled[..., c] = (ch - bp) / denom
            meds[c] = float(np.median(rescaled[..., c]))

        out = numba_color_final_formula_unlinked(rescaled, meds, float(target_median))

    if apply_curves:
        out = apply_curves_adjustment(out, float(target_median), float(curves_boost))
    if normalize:
        mx = float(out.max())
        if mx > 0:
            out = out / mx

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
