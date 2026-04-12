#src/setiastro/saspro/imageops/starbasedwhitebalance.py
from __future__ import annotations

import numpy as np
import math
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
def _spatially_sample_sources(sources: np.ndarray, r: np.ndarray,
                               img_shape: tuple, grid: int = 3,
                               per_cell: int = 500) -> tuple[np.ndarray, np.ndarray]:
    """
    Divide the image into a grid x grid spatial grid and keep at most
    per_cell stars per cell, preferring the brightest (highest flux).
    Returns filtered (sources, r).
    """
    H, W = img_shape[:2]
    cell_h = H / grid
    cell_w = W / grid

    keep = []
    for row in range(grid):
        for col in range(grid):
            y0 = row * cell_h
            y1 = (row + 1) * cell_h
            x0 = col * cell_w
            x1 = (col + 1) * cell_w

            in_cell = np.where(
                (sources["x"] >= x0) & (sources["x"] < x1) &
                (sources["y"] >= y0) & (sources["y"] < y1)
            )[0]

            if len(in_cell) == 0:
                continue

            if len(in_cell) <= per_cell:
                keep.append(in_cell)
            else:
                cell_flux = sources["flux"][in_cell]
                top = np.argpartition(cell_flux, -per_cell)[-per_cell:]
                keep.append(in_cell[top])

    if not keep:
        return sources, r

    idx = np.concatenate(keep)
    return sources[idx], r[idx]

def _apply_color_matrix_wb(
    bg_neutral: np.ndarray,
    bn_star_pixels: np.ndarray,
    pivot: np.ndarray,
    *,
    target_slope: float = 0.4,
    tilt_strength: float = 0.7,
) -> np.ndarray:
    """
    Non-linear (color matrix) white balance.
    Solves for a 3x3 matrix that rotates the stellar scatter toward the
    blackbody locus slope while preserving the neutral point.

    The matrix is constrained so that (1,1,1) maps to (1,1,1) — neutrals stay neutral.
    """
    eps = 1e-8

    # ── 1) Build target star colors from blackbody locus ──────────────────
    # For each star, find its R/B ratio, then compute what G/B *should* be
    # if it sat on the blackbody locus (gb = target_slope * rb + intercept).
    # We anchor the locus through neutral: at rb=1, gb=1 → intercept = 1 - target_slope
    intercept = 1.0 - target_slope   # locus passes through (1,1)

    rb_actual = bn_star_pixels[:, 0] / (bn_star_pixels[:, 2] + eps)
    gb_actual = bn_star_pixels[:, 1] / (bn_star_pixels[:, 2] + eps)

    # Filter to reasonable range
    mask = (
        np.isfinite(rb_actual) & np.isfinite(gb_actual) &
        (rb_actual > 0.2) & (rb_actual < 3.0) &
        (gb_actual > 0.2) & (gb_actual < 3.0) &
        (bn_star_pixels[:, 2] > eps)
    )
    if np.sum(mask) < 10:
        return None  # not enough stars, caller falls back to linear

    src = bn_star_pixels[mask].astype(np.float64)   # (N, 3) — measured star colors
    rb_m = rb_actual[mask]
    gb_m = gb_actual[mask]

    # Target: keep R/B where it is, rotate G/B onto the locus
    # gb_target = slope * rb + intercept, blended with actual
    gb_target = target_slope * rb_m + intercept
    gb_target_blended = (1.0 - tilt_strength) * gb_m + tilt_strength * gb_target

    # Reconstruct target RGB (keep B=actual, R=actual, G adjusted)
    b_actual = src[:, 2]
    r_actual = src[:, 0]
    g_target  = gb_target_blended * b_actual

    dst = np.stack([r_actual, g_target, b_actual], axis=1)  # (N, 3)

    # ── 2) Solve for 3×3 color matrix via least squares ───────────────────
    # dst = src @ M.T  →  solve for M (3×3)
    # Constraint: neutral maps to neutral, i.e. M @ [1,1,1] = [1,1,1]
    # We enforce this by adding neutral as a high-weight sample
    NEUTRAL_WEIGHT = float(np.sum(mask)) * 2.0   # counts as 2× all stars
    neutral_src = np.ones((1, 3), dtype=np.float64)
    neutral_dst = np.ones((1, 3), dtype=np.float64)

    src_aug = np.vstack([src, neutral_src * NEUTRAL_WEIGHT])
    dst_aug = np.vstack([dst, neutral_dst * NEUTRAL_WEIGHT])

    # Solve row-by-row: each output channel = linear combo of input channels
    M = np.zeros((3, 3), dtype=np.float64)
    for ch in range(3):
        coeffs, _, _, _ = np.linalg.lstsq(src_aug, dst_aug[:, ch], rcond=None)
        M[ch] = coeffs

    # ── 3) Sanity check — matrix should be close to identity ──────────────
    # Reject if it's doing something crazy
    diag = np.diag(M)
    off  = M - np.diag(diag)
    if np.any(np.abs(diag) > 3.0) or np.any(np.abs(off) > 1.5):
        print(f"[CC WB] Color matrix rejected (out of bounds): diag={diag} off-diag max={np.abs(off).max():.3f}")
        return None

    print(f"[CC WB] Color matrix:\n{np.round(M, 4)}")
    print(f"[CC WB] Neutral check: {np.round(M @ np.array([1,1,1]), 4)}")

    # ── 4) Apply matrix to image ───────────────────────────────────────────
    img = bg_neutral.astype(np.float64)
    H, W = img.shape[:2]

    # Subtract pivot, apply matrix, re-add pivot
    p = pivot.astype(np.float64).reshape(1, 1, 3)
    shifted = img - p                              # (H, W, 3)
    flat = shifted.reshape(-1, 3)                  # (H*W, 3)
    out_flat = flat @ M.T                          # (H*W, 3)
    out = out_flat.reshape(H, W, 3) + p

    return np.clip(out, 0.0, 1.0).astype(np.float32)

def apply_star_based_white_balance(
    image: np.ndarray,
    threshold: float = 1.5,
    autostretch: bool = True,
    reuse_cached_sources: bool = False,
    return_star_colors: bool = False,
    use_color_matrix: bool = False,       # ← new
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

    pivot = np.median(bg_neutral[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2], :],
                      axis=(0, 1)).astype(np.float32)

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

        # Keep only star-like detections
        mask = (r > 0) & (r <= 10)
        sources = sources[mask]
        r = r[mask]

        if len(sources) == 0:
            raise ValueError("All detected sources were rejected as non-stellar (too large).")

        # Spatially uniform sampling — caps total stars on dense fields
        sources, r = _spatially_sample_sources(sources, r, img_rgb.shape, grid=3, per_cell=500)

        # Cache the already-filtered result
        cached_star_sources = sources
        cached_flux_radii = r
        _cached_shape = tuple(img_rgb.shape)
        _cached_threshold = float(threshold)

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
    # 5) WB math: shift center to neutral + tilt toward blackbody slope
    # ------------------------------------------------------------------
    # ---- Step 1: measure current star color ratios ----
    eps = 1e-8
    ref_color = np.median(bn_star_pixels, axis=0).astype(np.float32)
    ref_color = np.where(ref_color <= eps, eps, ref_color)

    rb_stars = bn_star_pixels[:, 0] / (bn_star_pixels[:, 2] + eps)
    gb_stars = bn_star_pixels[:, 1] / (bn_star_pixels[:, 2] + eps)

    finite_mask = (
        np.isfinite(rb_stars) & np.isfinite(gb_stars) &
        (rb_stars > 0.1) & (gb_stars > 0.1) &
        (rb_stars < 4.0) & (gb_stars < 4.0)
    )

    TARGET_SLOPE = 0.55
    TILT_STRENGTH = 0.7

    if np.sum(finite_mask) >= 10:
        rb_f = rb_stars[finite_mask]
        gb_f = gb_stars[finite_mask]
        A = np.stack([rb_f, np.ones_like(rb_f)], axis=1)
        result = np.linalg.lstsq(A, gb_f, rcond=None)
        current_slope = float(np.clip(result[0][0], 0.05, 3.0))
    else:
        current_slope = TARGET_SLOPE

    # ---- Step 2: compute slope correction gain ratio ----
    # new_slope = (kg/kr) * current_slope = TARGET_SLOPE
    # So kg/kr = TARGET_SLOPE / current_slope
    # Split symmetrically: kr = 1/sqrt(ratio), kg = sqrt(ratio)
    slope_ratio = float(np.clip(TARGET_SLOPE / current_slope, 0.3, 3.0))
    blended_ratio = 1.0 + TILT_STRENGTH * (slope_ratio - 1.0)

    kr_tilt = 1.0 / math.sqrt(blended_ratio)
    kg_tilt = math.sqrt(blended_ratio)
    kb_tilt = 1.0

    # ---- Step 3: apply tilt to star samples, then find neutral shift ----
    # We want the TILTED median to land at neutral (equal RGB)
    # So compute where the median ends up after tilt, then correct
    tilt = np.array([kr_tilt, kg_tilt, kb_tilt], dtype=np.float32)
    tilted_median = ref_color * tilt
    tilted_median = np.where(tilted_median <= eps, eps, tilted_median)

    # Neutral shift: scale so max channel = all channels (gray)
    neutral_target = float(np.max(tilted_median))
    neutral_scale = neutral_target / tilted_median  # per-channel

    # ---- Final combined scaling: tilt * neutral (computed together, not sequentially) ----
    final_scaling = (tilt * neutral_scale).astype(np.float32)

    m = pivot.reshape((1, 1, 3)).astype(np.float32)
    g = final_scaling.reshape((1, 1, 3)).astype(np.float32)

    balanced = (bg_neutral.astype(np.float32) - m) * g + m
    balanced = np.clip(balanced, 0.0, 1.0).astype(np.float32, copy=False)
    # ── 5b) Color matrix WB (advanced, non-linear) ────────────────────────
    if use_color_matrix:
        matrix_result = _apply_color_matrix_wb(
            bg_neutral,
            bn_star_pixels,
            pivot,
            target_slope=0.55,
            tilt_strength=0.7,
        )
        balanced = matrix_result if matrix_result is not None else balanced

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

from PyQt6.QtCore import QThread, pyqtSignal as _pyqtSignal

class StarDetectionWorker(QThread):
    finished = _pyqtSignal(object, int)
    failed   = _pyqtSignal(str)

    def __init__(self, image: np.ndarray, threshold: float, autostretch: bool, parent=None):
        super().__init__(parent)
        self._image = np.asarray(image, dtype=np.float32)
        self._threshold = float(threshold)
        self._autostretch = bool(autostretch)
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            result = apply_star_based_white_balance(
                self._image,
                threshold=self._threshold,
                autostretch=self._autostretch,
                reuse_cached_sources=False,
                return_star_colors=False,
            )
            if self._cancelled:
                return
            _, count, overlay = result
            self.finished.emit(overlay, int(count))
        except Exception as e:
            if not self._cancelled:
                self.failed.emit(str(e))