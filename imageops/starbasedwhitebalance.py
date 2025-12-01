# imageops/starbasedwhitebalance.py
from __future__ import annotations

import numpy as np

# Optional deps
try:
    import cv2  # for ellipse overlay
except Exception:  # pragma: no cover
    cv2 = None

try:
    import sep  # Source Extractor
except Exception as e:  # pragma: no cover
    sep = None
    _sep_import_error = e
else:
    _sep_import_error = None

from typing import Tuple, Optional
from .stretch import stretch_color_image

# Shared utilities
from pro.widgets.image_utils import to_float01 as _to_float01

__all__ = ["apply_star_based_white_balance"]

# simple cache (reused when reuse_cached_sources=True)
cached_star_sources: Optional[np.ndarray] = None
cached_flux_radii: Optional[np.ndarray] = None


def _tone_preserve_bg_neutralize(rgb: np.ndarray) -> np.ndarray:
    """
    Neutralize background using the darkest grid patch in a tone-preserving way.
    Operates in-place on a copy; returns the neutralized image (float32 [0,1]).
    """
    h, w = rgb.shape[:2]
    patch_size = 10
    ph = max(1, h // patch_size)
    pw = max(1, w // patch_size)

    best = None
    best_sum = float("inf")
    for i in range(patch_size):
        for j in range(patch_size):
            y0, x0 = i * ph, j * pw
            y1, x1 = min(y0 + ph, h), min(x0 + pw, w)
            patch = rgb[y0:y1, x0:x1, :]
            med = np.median(patch, axis=(0, 1))
            s = float(np.sum(med))
            if s < best_sum:
                best_sum = s
                best = med

    out = rgb.copy()
    if best is not None:
        avg = float(np.mean(best))
        # “tone-preserving” shift+scale channel-wise toward avg
        for c in range(3):
            diff = float(best[c] - avg)
            denom = (1.0 - diff) if abs(1.0 - diff) > 1e-8 else 1e-8
            out[:, :, c] = np.clip((out[:, :, c] - diff) / denom, 0.0, 1.0)
    return out


def apply_star_based_white_balance(
    image: np.ndarray,
    threshold: float = 1.5,
    autostretch: bool = True,
    reuse_cached_sources: bool = False,
    return_star_colors: bool = False
) -> Tuple[np.ndarray, int, np.ndarray, np.ndarray, np.ndarray] | Tuple[np.ndarray, int, np.ndarray]:
    """
    Star-based white balance with background neutralization and an RGB overlay of detected stars.

    Parameters
    ----------
    image : np.ndarray
        RGB image (any dtype). Assumed RGB ordering.
    threshold : float
        SExtractor detection threshold (in background sigma).
    autostretch : bool
        If True, overlay is built from an autostretched view for visibility.
    reuse_cached_sources : bool
        If True, reuses star positions measured on a previous call (same scene).
    return_star_colors : bool
        If True, also returns (raw_star_pixels, after_star_pixels).

    Returns
    -------
    balanced_rgb : float32 RGB in [0,1]
    star_count   : int
    overlay_rgb  : float32 RGB in [0,1] with star ellipses drawn
    (optional) raw_star_pixels : (N,3) float array, colors sampled from ORIGINAL image
    (optional) after_star_pixels : (N,3) float array, colors sampled after WB
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("apply_star_based_white_balance: input must be an RGB image (H,W,3).")

    # 0) normalize
    img_rgb = _to_float01(image)

    # 1) first background neutralization (tone-preserving)
    bg_neutral = _tone_preserve_bg_neutralize(img_rgb)

    # 2) detect / reuse star positions
    if sep is None:
        raise ImportError(
            "apply_star_based_white_balance requires the 'sep' package. "
            f"Import error was: {_sep_import_error!r}"
        )

    gray = np.mean(bg_neutral, axis=2).astype(np.float32, copy=False)
    bkg = sep.Background(gray)
    data_sub = gray - bkg.back()
    err_val = float(bkg.globalrms)

    global cached_star_sources, cached_flux_radii

    if reuse_cached_sources and cached_star_sources is not None:
        sources = cached_star_sources
        r = cached_flux_radii
    else:
        sources = sep.extract(data_sub, threshold, err=err_val)
        if sources is None or len(sources) == 0:
            raise ValueError("No sources detected for Star-Based White Balance.")
        r, _ = sep.flux_radius(
            gray,
            sources["x"], sources["y"],
            2.0 * sources["a"], 0.2,
            normflux=sources["flux"],
            subpix=5
        )
        cached_star_sources = sources
        cached_flux_radii = r

    # filter: small-ish, star-like
    mask = (r > 0) & (r <= 10)
    sources = sources[mask]
    r = r[mask]
    if len(sources) == 0:
        raise ValueError("All detected sources were rejected as non-stellar (too large).")

    h, w = gray.shape
    # raw colors from ORIGINAL image - optimized vectorized extraction
    xs = sources["x"].astype(np.int32)
    ys = sources["y"].astype(np.int32)
    valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    raw_star_pixels = img_rgb[ys[valid], xs[valid], :]

    # 3) build overlay (autostretched if requested) and draw ellipses
    disp = stretch_color_image(bg_neutral.copy(), 0.25) if autostretch else bg_neutral.copy()

    if cv2 is not None:
        overlay_bgr = cv2.cvtColor((disp * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        for i in range(len(sources)):
            cx = float(sources["x"][i]); cy = float(sources["y"][i])
            a = float(sources["a"][i]); b = float(sources["b"][i])
            theta_deg = float(sources["theta"][i] * 180.0 / np.pi)
            center = (int(round(cx)), int(round(cy)))
            axes = (max(1, int(round(3 * a))), max(1, int(round(3 * b))))
            # red ellipse in BGR
            cv2.ellipse(overlay_bgr, center, axes, angle=theta_deg, startAngle=0, endAngle=360,
                        color=(0, 0, 255), thickness=1)
        overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    else:
        # fallback: no ellipses, just the display image
        overlay_rgb = disp.astype(np.float32, copy=False)

    # 4) compute WB scale using star colors sampled on bg_neutral image
    # Optimized: vectorized extraction instead of Python loop (10-50x faster)
    xs = sources["x"].astype(np.int32)
    ys = sources["y"].astype(np.int32)
    valid_mask = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    
    if not np.any(valid_mask):
        raise ValueError("No stellar samples available for white balance.")
    
    star_pixels = bg_neutral[ys[valid_mask], xs[valid_mask], :].astype(np.float32)
    avg_color = np.mean(star_pixels, axis=0)
    max_val = float(np.max(avg_color))
    # protect against divide-by-zero
    avg_color = np.where(avg_color <= 1e-8, 1e-8, avg_color)
    scaling = max_val / avg_color

    balanced = (bg_neutral * scaling.reshape((1, 1, 3))).clip(0.0, 1.0)

    # 5) second background neutralization pass on balanced image
    balanced = _tone_preserve_bg_neutralize(balanced)

    # 6) collect after-WB star samples - optimized vectorized extraction
    after_star_pixels = balanced[ys[valid_mask], xs[valid_mask], :]

    if return_star_colors:
        return (
            balanced.astype(np.float32, copy=False),
            int(len(star_pixels)),
            overlay_rgb.astype(np.float32, copy=False),
            np.asarray(raw_star_pixels, dtype=np.float32),
            np.asarray(after_star_pixels, dtype=np.float32),
        )

    return (
        balanced.astype(np.float32, copy=False),
        int(len(star_pixels)),
        overlay_rgb.astype(np.float32, copy=False),
    )


