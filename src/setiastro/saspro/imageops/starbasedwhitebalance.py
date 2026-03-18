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
from setiastro.saspro.widgets.image_utils import to_float01 as _to_float01
from setiastro.saspro.backgroundneutral import background_neutralize_rgb, auto_rect_50x50

__all__ = ["apply_star_based_white_balance"]

# Keep names for compatibility with any old imports / callers
cached_star_sources: Optional[np.ndarray] = None
cached_flux_radii: Optional[np.ndarray] = None
_cached_shape: Optional[tuple[int, int, int]] = None
_cached_threshold: Optional[float] = None


def _sample_star_circle_medians(
    rgb: np.ndarray,
    sources: np.ndarray,
    radius: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample each detected star as the median RGB value inside a small circular
    aperture centered on the rounded star centroid.

    Returns
    -------
    samples : (N,3) float32
    keep_idx : (N,) int indices into the input `sources`
    """
    h, w = rgb.shape[:2]
    rr2 = int(radius) * int(radius)

    samples = []
    keep_idx = []

    for i in range(len(sources)):
        cx = int(round(float(sources["x"][i])))
        cy = int(round(float(sources["y"][i])))

        x0 = max(0, cx - radius)
        x1 = min(w, cx + radius + 1)
        y0 = max(0, cy - radius)
        y1 = min(h, cy + radius + 1)

        if x1 <= x0 or y1 <= y0:
            continue

        yy, xx = np.ogrid[y0:y1, x0:x1]
        mask = ((xx - cx) ** 2 + (yy - cy) ** 2) <= rr2
        if not np.any(mask):
            continue

        patch = rgb[y0:y1, x0:x1, :]
        vals = patch[mask]
        if vals.size == 0:
            continue

        med = np.median(vals, axis=0).astype(np.float32)
        if not np.all(np.isfinite(med)):
            continue

        samples.append(med)
        keep_idx.append(i)

    if len(samples) == 0:
        raise ValueError("No valid stellar samples available for white balance.")

    return np.asarray(samples, dtype=np.float32), np.asarray(keep_idx, dtype=np.int32)


def apply_star_based_white_balance(
    image: np.ndarray,
    threshold: float = 1.5,
    autostretch: bool = True,
    reuse_cached_sources: bool = False,
    return_star_colors: bool = False
) -> Tuple[np.ndarray, int, np.ndarray, np.ndarray, np.ndarray] | Tuple[np.ndarray, int, np.ndarray]:
    """
    Star-based white balance using:
      1) shared Background Neutralization module
      2) SEP star detection on the BN image
      3) small circular median samples per star (radius=3 px)
      4) median-anchored white-point scaling
      5) NO second BN pass afterwards

    Parameters
    ----------
    image : np.ndarray
        RGB image (any dtype). Assumed RGB ordering.
    threshold : float
        SEP detection threshold (in background sigma).
    autostretch : bool
        If True, overlay is built from an autostretched BN view for visibility.
    reuse_cached_sources : bool
        Reuse detections only if image shape and threshold match.
    return_star_colors : bool
        If True, also returns (raw_star_pixels, after_star_pixels).

    Returns
    -------
    balanced_rgb : float32 RGB in [0,1]
    star_count   : int
    overlay_rgb  : float32 RGB in [0,1] with star ellipses drawn
    (optional) raw_star_pixels   : (N,3) float32
    (optional) after_star_pixels : (N,3) float32
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("apply_star_based_white_balance: input must be an RGB image (H,W,3).")

    if sep is None:
        raise ImportError(
            "apply_star_based_white_balance requires the 'sep' package. "
            f"Import error was: {_sep_import_error!r}"
        )

    img_rgb = _to_float01(image).astype(np.float32, copy=False)

    # ------------------------------------------------------------------
    # 1) Shared background neutralization
    # ------------------------------------------------------------------
    rect = auto_rect_50x50(img_rgb)
    bg_neutral = background_neutralize_rgb(img_rgb, rect).astype(np.float32, copy=False)

    # Use the post-BN sample-region medians as the anchor pivot.
    x, y, rw, rh = rect
    pivot = np.median(bg_neutral[y:y + rh, x:x + rw, :], axis=(0, 1)).astype(np.float32)

    # ------------------------------------------------------------------
    # 2) Detect stars on BN luminance
    # ------------------------------------------------------------------
    gray = np.mean(bg_neutral, axis=2).astype(np.float32, copy=False)
    bkg = sep.Background(gray)
    data_sub = gray - bkg.back()
    err_val = float(bkg.globalrms)

    global cached_star_sources, cached_flux_radii, _cached_shape, _cached_threshold

    use_cache = (
        bool(reuse_cached_sources)
        and cached_star_sources is not None
        and cached_flux_radii is not None
        and _cached_shape == tuple(img_rgb.shape)
        and _cached_threshold is not None
        and abs(float(_cached_threshold) - float(threshold)) < 1e-9
    )

    if use_cache:
        sources = cached_star_sources
        r = cached_flux_radii
    else:
        sources = sep.extract(data_sub, float(threshold), err=err_val)
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
        _cached_shape = tuple(img_rgb.shape)
        _cached_threshold = float(threshold)

    # Keep only small-ish star-like detections
    mask = (r > 0) & (r <= 10)
    sources = sources[mask]
    r = r[mask]

    if len(sources) == 0:
        raise ValueError("All detected sources were rejected as non-stellar (too large).")

    # ------------------------------------------------------------------
    # 3) Sample stars as circular medians, not center pixels
    # ------------------------------------------------------------------
    raw_star_pixels, keep_idx = _sample_star_circle_medians(img_rgb, sources, radius=3)
    sources = sources[keep_idx]
    r = r[keep_idx]

    bn_star_pixels, _ = _sample_star_circle_medians(bg_neutral, sources, radius=3)

    if len(bn_star_pixels) == 0:
        raise ValueError("No stellar samples remained after circular median sampling.")

    # ------------------------------------------------------------------
    # 4) Build preview overlay
    # ------------------------------------------------------------------
    disp = stretch_color_image(bg_neutral.copy(), 0.25) if autostretch else bg_neutral.copy()

    if cv2 is not None:
        overlay_bgr = cv2.cvtColor((disp * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        for i in range(len(sources)):
            cx = float(sources["x"][i])
            cy = float(sources["y"][i])
            a = float(sources["a"][i])
            b = float(sources["b"][i])
            theta_deg = float(sources["theta"][i] * 180.0 / np.pi)

            center = (int(round(cx)), int(round(cy)))
            axes = (max(1, int(round(3 * a))), max(1, int(round(3 * b))))

            cv2.ellipse(
                overlay_bgr,
                center,
                axes,
                angle=theta_deg,
                startAngle=0,
                endAngle=360,
                color=(0, 0, 255),
                thickness=1
            )

        overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    else:
        overlay_rgb = disp.astype(np.float32, copy=False)

    # ------------------------------------------------------------------
    # 5) WB math: anchor the BN background medians, adjust star white points
    # ------------------------------------------------------------------
    # Use the median stellar color across sampled stars for robustness.
    ref_color = np.median(bn_star_pixels, axis=0).astype(np.float32)
    ref_color = np.where(ref_color <= 1e-8, 1e-8, ref_color)

    target = float(np.max(ref_color))
    scaling = (target / ref_color).astype(np.float32)

    m = pivot.reshape((1, 1, 3)).astype(np.float32)
    g = scaling.reshape((1, 1, 3)).astype(np.float32)

    balanced = (bg_neutral.astype(np.float32) - m) * g + m
    balanced = np.clip(balanced, 0.0, 1.0).astype(np.float32, copy=False)

    # ------------------------------------------------------------------
    # 6) After-WB diagnostic samples, same circular-median method
    # ------------------------------------------------------------------
    after_star_pixels, _ = _sample_star_circle_medians(balanced, sources, radius=3)

    star_count = int(len(bn_star_pixels))

    if return_star_colors:
        return (
            balanced,
            star_count,
            overlay_rgb.astype(np.float32, copy=False),
            np.asarray(raw_star_pixels, dtype=np.float32),
            np.asarray(after_star_pixels, dtype=np.float32),
        )

    return (
        balanced,
        star_count,
        overlay_rgb.astype(np.float32, copy=False),
    )