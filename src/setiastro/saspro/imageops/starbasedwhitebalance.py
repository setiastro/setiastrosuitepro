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
    sep.set_extract_pixstack(20000000)
except Exception as e:  # pragma: no cover
    sep = None
    _sep_import_error = e
else:
    _sep_import_error = None

from typing import Tuple, Optional, Callable, Sequence
from .stretch import stretch_color_image

# Shared utilities
from setiastro.saspro.widgets.image_utils import to_float01 as _to_float01
from setiastro.saspro.backgroundneutral import background_neutralize_rgb, auto_rect_50x50

__all__ = [
    "apply_star_based_white_balance",
    "detect_stars_for_wb",
    "build_wb_preview_overlay",
    "StarDetectionWorker",
    "DEFAULT_AUTO_SIGMA_LADDER",
    "DEFAULT_AUTO_SIGMA_MIN_STARS",
]

# ── Blackbody locus reference ──────────────────────────────────────────
# Target for the color-matrix rotation, in (R/B, G/B) ratio space.
#
# These are DERIVED, not guessed. They come from the same Planck spectrum →
# approximate CIE 1931 XYZ → linear sRGB chain that whitebalance.py's
# _blackbody_rgb_approx() uses to DRAW the locus on the diagnostic plot, so
# the math and the picture agree by construction.
#
#   T (K)      R/B      G/B   local d(G/B)/d(R/B)
#    5000   1.5864   1.2649   0.4978
#    6000   1.1403   1.0297   0.5615
#    6500   1.0066   0.9528   0.5890
#    7000   0.9057   0.8922   0.6139   <- reference
#    8000   0.7654   0.8031   0.6571
#   10000   0.6101   0.6963   0.7226
#
# The locus is CURVED, so the "right" slope depends on where you anchor it.
# 7000 K is the sensible anchor for a broadband stellar field: most field
# stars are A–G, and the locus crosses R/B = 1 at ~6529 K. Earlier code used
# 0.55 (roughly the 5000–9000 K secant) in the WB stage and 0.4 at the matrix
# call site — the two disagreed, and both undershot the real tangent.
LOCUS_REF_TEMP_K = 7000.0
LOCUS_TARGET_SLOPE = 0.6139        # d(G/B)/d(R/B) tangent at LOCUS_REF_TEMP_K

# G/B where the 7000 K tangent line crosses R/B = 1.0:
#     G/B = 0.6139*(R/B) + 0.3362  ->  0.9501 at R/B = 1
# (Was hardcoded 0.96. The true locus crosses R/B=1 at 6529 K, G/B=0.9490;
# 0.9501 is the self-consistent value for the line we actually target, so
# intercept = LOCUS_NEUTRAL_GB - LOCUS_TARGET_SLOPE lands on the real curve.)
LOCUS_NEUTRAL_GB = 0.9501


# ── Photometric clip-rejection defaults ────────────────────────────────
# Shared across every photometric fit site so slope and matrix solves don't
# lean on stars that have saturated in any channel (where the color info is
# gone) or are buried in the noise floor (where the ratio is pure noise).
_WB_CLIP_HIGH = 0.95
_WB_CLIP_LOW  = 0.005


def _photometric_keep_mask(
    samples: np.ndarray,
    *,
    clip_high: float = _WB_CLIP_HIGH,
    clip_low: float  = _WB_CLIP_LOW,
) -> np.ndarray:
    """
    Boolean keep-mask for stellar samples: True where NO channel is clipped
    high or buried in the noise floor. Vectorized across (N,3) or (N,K).

    This is what keeps the WB slope and the scatter-plot diagnostic from
    leaning on stars that have lost their color information.
    """
    a = np.asarray(samples, dtype=np.float32)
    finite = np.all(np.isfinite(a), axis=-1)
    below_hi = np.all(a < float(clip_high), axis=-1)
    above_lo = np.all(a > float(clip_low),  axis=-1)
    return finite & below_hi & above_lo


def _robust_slope_fit(
    rb: np.ndarray, gb: np.ndarray,
    *,
    sigma: float = 2.5,
    max_iter: int = 5,
    min_points: int = 10,
) -> tuple[float, float, np.ndarray]:
    """
    Iterative MAD-sigma-clipped fit of gb ≈ slope*rb + intercept.
    Caller is responsible for pre-rejecting saturated samples via
    _photometric_keep_mask BEFORE handing (rb, gb) in here.

    Returns (slope, intercept, final_keep_mask). Falls back to a plain
    lstsq if too few points survive.
    """
    rb = np.asarray(rb, dtype=np.float64).reshape(-1)
    gb = np.asarray(gb, dtype=np.float64).reshape(-1)
    keep = np.isfinite(rb) & np.isfinite(gb)

    if keep.sum() < min_points:
        if keep.sum() < 2:
            return 1.0, 0.0, keep
        s, i = np.polyfit(rb[keep], gb[keep], 1)
        return float(s), float(i), keep

    for _ in range(max_iter):
        xi, yi = rb[keep], gb[keep]
        A = np.stack([xi, np.ones_like(xi)], axis=1)
        (slope, intercept), *_ = np.linalg.lstsq(A, yi, rcond=None)
        resid = yi - (slope * xi + intercept)
        mad = float(np.median(np.abs(resid - np.median(resid))))
        scale = 1.4826 * mad if mad > 0 else float(np.std(resid))
        if scale <= 1e-9:
            break
        inlier = np.abs(resid) < sigma * scale
        if inlier.all():
            break
        idx = np.where(keep)[0]
        keep_new = np.zeros_like(keep)
        keep_new[idx[inlier]] = True
        if keep_new.sum() < min_points:
            break
        keep = keep_new

    xi, yi = rb[keep], gb[keep]
    A = np.stack([xi, np.ones_like(xi)], axis=1)
    (slope, intercept), *_ = np.linalg.lstsq(A, yi, rcond=None)
    return float(slope), float(intercept), keep


# ── Auto-σ waterfall defaults ──────────────────────────────────────────
# Highest σ first, stepping down to the most sensitive setting. The first
# rung that returns ≥ DEFAULT_AUTO_SIGMA_MIN_STARS detections wins. If no
# rung clears the bar, the last (most sensitive) rung is used so the pipeline
# still has something to work with on genuinely sparse fields.
DEFAULT_AUTO_SIGMA_LADDER: tuple[float, ...] = (200.0, 100.0, 50.0, 25.0, 12.0, 8.0, 5.0, 3.0)
DEFAULT_AUTO_SIGMA_MIN_STARS: int = 1200

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


def _waterfall_sep_extract(
    data_sub: np.ndarray,
    err_val: float,
    ladder: Sequence[float],
    min_stars: int,
    progress: Optional[Callable[[float, int, bool], None]] = None,
) -> tuple[np.ndarray, float]:
    """
    Run sep.extract against `data_sub` at each σ in `ladder` (highest first),
    returning the first result whose star count ≥ `min_stars`. If no rung
    clears the bar, the result at the lowest (most sensitive) σ is returned
    so callers still have something to work with on sparse fields.

    `progress`, when provided, is called with (sigma, count, settled) after
    every rung — `settled` is True on the one that is being returned.

    Returns
    -------
    sources : structured ndarray from sep.extract (may be empty on truly
              blank frames; callers must handle that case).
    used_sigma : the σ that produced `sources`.
    """
    if not ladder:
        raise ValueError("Auto-σ ladder is empty.")

    last_sources: Optional[np.ndarray] = None
    last_sigma: float = float(ladder[-1])

    for i, sig in enumerate(ladder):
        sig_f = float(sig)
        try:
            srcs = sep.extract(data_sub, sig_f, err=err_val)
        except Exception:
            # Treat a failure at this rung as "zero detections" and keep
            # descending — some rungs can blow the pixstack on dense fields
            srcs = np.empty(0, dtype=object)

        n = 0 if srcs is None else int(len(srcs))
        last_sources = srcs
        last_sigma = sig_f
        print(f"[Auto-σ] σ={sig_f:.3f} → {n} detections (min_stars={min_stars})")

        at_last = (i >= len(ladder) - 1)
        hit = (n >= int(min_stars))

        if progress is not None:
            try:
                progress(sig_f, n, bool(hit or at_last))
            except Exception:
                pass

        if hit:
            return srcs, sig_f

    # Ladder exhausted without hitting min_stars — fall back to the last try.
    return (last_sources if last_sources is not None
            else sep.extract(data_sub, float(ladder[-1]), err=err_val)), last_sigma


def detect_stars_for_wb(
    image: np.ndarray,
    threshold: float = 50.0,
    *,
    reuse_cached_sources: bool = False,
    auto_sigma: bool = False,
    auto_sigma_min_stars: int = DEFAULT_AUTO_SIGMA_MIN_STARS,
    auto_sigma_ladder: Sequence[float] = DEFAULT_AUTO_SIGMA_LADDER,
    on_sigma_resolved: Optional[Callable[[float], None]] = None,
) -> tuple[int, float, np.ndarray, np.ndarray]:
    """
    Detection-only pass for the WB dialog's open-time count and resolved σ.

    Runs the same BN + sep.Background + sep.extract waterfall + flux_radius
    filter + spatial subsampling as apply_star_based_white_balance, but
    skips the full-image overlay construction (stretch + cv2 ellipse draw)
    AND the WB math itself. On a 100 MP RGB frame that saves the overlay
    autostretch and the two full-image allocations the WB math makes
    (balanced = (bg_neutral - m) * g + m, then its clip), which is where
    the open-time cost of the preview actually lives — not the detector.

    Returns everything the dialog needs to:
      - report "σ=X, N stars" immediately
      - defer the overlay to a user-requested "Show Preview" (feed the
        returned bg_neutral + sources into build_wb_preview_overlay)
      - still populate the shared source cache for a later full
        apply_star_based_white_balance() call that uses reuse_cached_sources.

    Returns
    -------
    count : int
    used_threshold : float
        The σ the detector actually ran at. In manual mode this equals
        `threshold`; in auto-σ mode this is the ladder rung that won.
    sources : structured ndarray
    bg_neutral : (H,W,3) float32
        Already-neutralized image, so overlay rendering doesn't re-do BN.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("detect_stars_for_wb: input must be an RGB image (H,W,3).")
    if sep is None:
        raise ImportError(
            "detect_stars_for_wb requires the 'sep' package. "
            f"Import error was: {_sep_import_error!r}"
        )

    img_rgb = _to_float01(image).astype(np.float32, copy=False)

    rect = auto_rect_50x50(img_rgb)
    bg_neutral = background_neutralize_rgb(img_rgb, rect).astype(np.float32, copy=False)

    gray = np.mean(bg_neutral, axis=2).astype(np.float32, copy=False)
    bkg = sep.Background(gray)
    data_sub = gray - bkg.back()
    err_val = float(bkg.globalrms)

    global cached_star_sources, cached_flux_radii, _cached_shape, _cached_threshold

    # Cache gate:
    #   - manual mode: hit when caller asks AND cached σ matches input σ
    #   - auto mode:   hit when caller asks AND we have a cached result for
    #                  this shape — the waterfall is deterministic for a
    #                  given image, so the previously-resolved σ is the one
    #                  we'd resolve again. This is what makes a dialog
    #                  Preview → Apply round-trip run SEP exactly once.
    use_cache = (
        bool(reuse_cached_sources)
        and cached_star_sources is not None
        and cached_flux_radii is not None
        and _cached_shape == tuple(img_rgb.shape)
        and _cached_threshold is not None
        and (
            bool(auto_sigma)
            or abs(float(_cached_threshold) - float(threshold)) < 1e-9
        )
    )

    if use_cache:
        sources = cached_star_sources
        r = cached_flux_radii
        # In auto mode `threshold` is a stale user-set value — report back
        # the σ actually used (the cached one), not the slider value.
        used_threshold = float(_cached_threshold)
    else:
        if auto_sigma:
            sources, used_threshold = _waterfall_sep_extract(
                data_sub, err_val,
                ladder=auto_sigma_ladder,
                min_stars=auto_sigma_min_stars,
            )
        else:
            used_threshold = float(threshold)
            sources = sep.extract(data_sub, used_threshold, err=err_val)

        if sources is None or len(sources) == 0:
            raise ValueError("No sources detected for Star-Based White Balance.")

        r, _ = sep.flux_radius(
            gray,
            sources["x"], sources["y"],
            2.0 * sources["a"], 0.2,
            normflux=sources["flux"],
            subpix=5,
        )

        mask = (r > 0) & (r <= 10)
        sources = sources[mask]
        r = r[mask]

        if len(sources) == 0:
            raise ValueError("All detected sources were rejected as non-stellar (too large).")

        sources, r = _spatially_sample_sources(sources, r, img_rgb.shape, grid=3, per_cell=500)

        cached_star_sources = sources
        cached_flux_radii = r
        _cached_shape = tuple(img_rgb.shape)
        _cached_threshold = float(used_threshold)

    if on_sigma_resolved is not None:
        try:
            on_sigma_resolved(float(used_threshold))
        except Exception:
            pass

    return int(len(sources)), float(used_threshold), sources, bg_neutral


def build_wb_preview_overlay(
    bg_neutral: np.ndarray,
    sources: np.ndarray,
    autostretch: bool = True,
) -> np.ndarray:
    """
    Render the dialog's star-detection preview from an already-computed BN
    image and already-detected sources. Call from a worker thread — the
    autostretch allocates a full-image copy on big frames, which is the
    reason the preview was deferred in the first place.

    Parameters
    ----------
    bg_neutral : (H,W,3) float32 in [0,1]
    sources    : structured ndarray from sep.extract (needs x, y, a, b, theta)
    autostretch: if True, 0.25 target-median stretch on the display copy

    Returns
    -------
    overlay_rgb : (H,W,3) float32 in [0,1] with red SEP ellipses drawn
    """
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
                thickness=1,
            )

        overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    else:
        overlay_rgb = disp.astype(np.float32, copy=False)

    return overlay_rgb


def _apply_color_matrix_wb(
    bg_neutral: np.ndarray,
    bn_star_pixels: np.ndarray,
    pivot: np.ndarray,
    *,
    target_slope: float = LOCUS_TARGET_SLOPE,
    tilt_strength: float = 0.7,
) -> np.ndarray:
    """
    Non-linear (color matrix) white balance.
    Solves for a 3x3 matrix that rotates the stellar scatter toward the
    blackbody locus slope while preserving the locus-neutral point.

    Projects each star's (R/B, G/B) point orthogonally onto the target
    blackbody locus line, then solves for the 3x3 matrix that maps
    measured star colors to their locus-projected targets.

    The neutral point is anchored at the physically correct locus intercept
    (R/B=1, G/B≈0.96) rather than the mathematically convenient (1,1).
    """
    eps = 1e-8

    # Locus: gb = target_slope * rb + intercept, anchored through
    # (1, LOCUS_NEUTRAL_GB). Both constants are module-level and derived from
    # the Planck/CIE chain at LOCUS_REF_TEMP_K — see the table up top. A
    # neutral gray star is slightly green-deficient relative to equal RGB,
    # which is why LOCUS_NEUTRAL_GB is below 1.0.
    intercept = LOCUS_NEUTRAL_GB - target_slope

    rb_actual = bn_star_pixels[:, 0] / (bn_star_pixels[:, 2] + eps)
    gb_actual = bn_star_pixels[:, 1] / (bn_star_pixels[:, 2] + eps)

    # Reject stars saturated in ANY channel before fitting the matrix —
    # a clipped-high star contributes a meaningless (R/B, G/B) point that
    # pulls the solved rotation off the real blackbody locus.
    unsat_mask = _photometric_keep_mask(
        bn_star_pixels, clip_high=_WB_CLIP_HIGH, clip_low=_WB_CLIP_LOW
    )

    # Filter to reasonable range
    mask = (
        unsat_mask &
        np.isfinite(rb_actual) & np.isfinite(gb_actual) &
        (rb_actual > 0.2) & (rb_actual < 3.0) &
        (gb_actual > 0.2) & (gb_actual < 3.0) &
        (bn_star_pixels[:, 2] > eps)
    )
    n_sat = int(np.sum(~unsat_mask))
    print(f"[CC WB] matrix fit: {int(mask.sum())}/{len(bn_star_pixels)} stars "
          f"({n_sat} rejected as clipped), strength={tilt_strength:.2f}, "
          f"target slope={target_slope:.4f} ({LOCUS_REF_TEMP_K:.0f}K locus)")
    if np.sum(mask) < 10:
        return None

    src  = bn_star_pixels[mask].astype(np.float64)
    rb_m = rb_actual[mask]
    gb_m = gb_actual[mask]

    # Orthogonal projection of each (rb, gb) onto the locus line:
    #   rb_t = (rb + slope*(gb - intercept)) / (1 + slope^2)
    #   gb_t = slope * rb_t + intercept
    slope = float(target_slope)
    rb_t  = (rb_m + slope * (gb_m - intercept)) / (1.0 + slope ** 2)
    gb_t  = slope * rb_t + intercept

    # Blend toward projected target
    rb_target = (1.0 - tilt_strength) * rb_m + tilt_strength * rb_t
    gb_target = (1.0 - tilt_strength) * gb_m + tilt_strength * gb_t

    # Normalize each star by its own luminance so B isn't a fixed anchor
    lum_src = 0.2126 * src[:, 0] + 0.7152 * src[:, 1] + 0.0722 * src[:, 2]
    lum_src = np.where(lum_src <= eps, eps, lum_src)

    # Work directly in raw pixel space — no luminance normalization
    # This ensures the matrix is solved and applied in the same domain
    src_raw = src.copy()  # already float64, shape (N, 3)

    # Reconstruct target RGB from ratios, scaling B to match source luminance
    denom = 0.2126 * rb_target + 0.7152 * gb_target + 0.0722
    denom = np.where(denom <= eps, eps, denom)
    b_t = lum_src / denom
    r_t = rb_target * b_t
    g_t = gb_target * b_t
    dst_raw = np.stack([r_t, g_t, b_t], axis=1)

    # Neutral constraint in raw pixel space:
    # a mid-gray pixel (0.5, 0.5, 0.5) should map to (0.5, 0.48, 0.5)
    #
    # WEIGHTING: kept at 2N deliberately. It is tempting to argue that a
    # weight this heavy (w² = 4N² against the stars' combined N) drowns the
    # star rows and leaves the locus rotation as a rounding error — measured
    # on synthetic populations, that is NOT what happens. The neutral row
    # pins only ONE direction of each row's 3-parameter space, leaving 2 DOF
    # to fit the stars, so the rotation is essentially unaffected (70.9% of
    # the slope gap closed at 2N vs 70.5% at sqrt(N)) while neutral accuracy
    # is ~300x better (error 1e-5 vs 3e-3). Since the per-row re-anchor below
    # was removed, this weight is now the ONLY thing holding gray at gray —
    # which is exactly the "shift it back to neutral" half of the job the
    # diagonal stage can no longer do. Do not lower it.
    NEUTRAL_WEIGHT = float(np.sum(mask)) * 2.0
    neutral_src = np.array([[0.5, 0.5, 0.5]], dtype=np.float64)
    neutral_dst = np.array([[0.5, 0.5 * LOCUS_NEUTRAL_GB, 0.5]], dtype=np.float64)

    src_aug = np.vstack([src_raw, neutral_src * NEUTRAL_WEIGHT])
    dst_aug = np.vstack([dst_raw, neutral_dst * NEUTRAL_WEIGHT])

    # Solve row-by-row
    M = np.zeros((3, 3), dtype=np.float64)
    for ch in range(3):
        coeffs, _, _, _ = np.linalg.lstsq(src_aug, dst_aug[:, ch], rcond=None)
        M[ch] = coeffs

    # NOTE: the old per-row re-anchor (M[ch] *= target/neutral_out[ch]) was
    # REMOVED. Rescaling a solved row changes that channel's output scale,
    # which changes rb'/gb' — reintroducing a diagonal correction on top of
    # the rotation the lstsq just found, partially cancelling it. With the
    # neutral row now properly weighted above, the solve already satisfies
    # the neutral constraint; a third mechanism fighting the other two is
    # exactly what kept the scatter from rotating.

    # Sanity check — bound scaled to the correction actually being asked for.
    diag = np.diag(M)
    off  = M - np.diag(diag)

    # A real locus rotation REQUIRES substantial off-diagonals: rotating a
    # slope-0.20 scatter up to 0.40 is a ~2x correction and legitimately
    # produces off-diagonals near 1.0 (measured on real data: off-diag 0.942
    # with diag [0.63, 1.18, 0.66] — a perfectly sane rotation that the old
    # flat 0.5 ceiling threw away). Scale the bound by the correction being
    # asked for, with a floor that is still looser than the old ceiling and
    # a cap that catches genuinely degenerate (ill-conditioned / sample-
    # starved) solves.
    #
    # Base 0.8, not 0.5: the clamp has to hold on the slope measured HERE,
    # on the diagonal-balanced samples, which differs from the WB stage's
    # slope by kg/kr. A bound that only just clears the observed off-diag
    # would flip to reject on a small change in that ratio.
    measured_slope = float(np.clip(np.polyfit(rb_m, gb_m, 1)[0], 0.01, 5.0)) \
        if rb_m.size >= 2 else float(target_slope)
    correction = float(np.clip(abs(target_slope / max(measured_slope, 1e-3)), 1.0, 4.0))
    off_limit = float(np.clip(0.8 * correction, 0.8, 2.0))
    diag_limit = 2.5

    if np.any(np.abs(diag) > diag_limit) or np.any(np.abs(off) > off_limit):
        print(f"[CC WB] Color matrix rejected (out of bounds): "
              f"diag={np.round(diag,4)} off-diag max={np.abs(off).max():.3f} "
              f"(limits: diag {diag_limit:.2f}, off {off_limit:.2f}; "
              f"measured slope={measured_slope:.4f} → target={target_slope:.4f})")
        return None

    print(f"[CC WB] Color matrix (measured slope={measured_slope:.4f} → "
          f"target={target_slope:.4f}, off-limit={off_limit:.2f}):\n{np.round(M, 4)}")

    # ---- Verify BOTH constraints the 3x3 is supposed to satisfy ----------
    # (1) Neutral preserved: gray in → gray out (at the locus-neutral G/B).
    #     This is the "shift back to neutral" the diagonal stage can no
    #     longer do — the matrix has to own it, so prove it did.
    _n_out = M @ np.array([1.0, 1.0, 1.0])
    _n_rb = float(_n_out[0] / max(_n_out[2], eps))
    _n_gb = float(_n_out[1] / max(_n_out[2], eps))
    print(f"[CC WB] Neutral check: (1,1,1) → {np.round(_n_out, 4)}  "
          f"R/B={_n_rb:.4f} (want 1.000)  G/B={_n_gb:.4f} "
          f"(want {LOCUS_NEUTRAL_GB:.3f})")

    # (2) Rotation achieved: push the star samples through M and refit the
    #     slope. This is the only honest measure of whether the matrix
    #     actually rotated the scatter toward the locus, as opposed to
    #     solving something that cancels out downstream.
    try:
        _starred = src_raw @ M.T
        _rb_out = _starred[:, 0] / np.maximum(_starred[:, 2], eps)
        _gb_out = _starred[:, 1] / np.maximum(_starred[:, 2], eps)
        _ok = (np.isfinite(_rb_out) & np.isfinite(_gb_out) &
               (_rb_out > 0.05) & (_rb_out < 5.0) &
               (_gb_out > 0.05) & (_gb_out < 5.0))
        if int(_ok.sum()) >= 2:
            _achieved = float(np.polyfit(_rb_out[_ok], _gb_out[_ok], 1)[0])
            _closed = ((_achieved - measured_slope) /
                       max(target_slope - measured_slope, 1e-6)) * 100.0
            print(f"[CC WB] Slope achieved: {measured_slope:.4f} → "
                  f"{_achieved:.4f} (target {target_slope:.4f}, "
                  f"{_closed:.0f}% of the gap closed)")
    except Exception as _e:
        print(f"[CC WB] Slope-achieved check skipped: {_e}")

# Apply matrix directly (no pivot shift — matrix was solved in raw pixel space)
    img      = bg_neutral.astype(np.float64)
    flat     = img.reshape(-1, 3)
    out_flat = flat @ M.T
    out      = out_flat.reshape(img.shape)

    # Debug: report worst negative before any correction
    min_val = float(np.nanmin(out))
    if min_val < 0.0:
        neg_pixels = np.sum(out < 0.0)
        print(f"[CC WB] Post-matrix negatives: min={min_val:.4f}, "
              f"affected pixels={neg_pixels} ({100*neg_pixels/out.size:.2f}%)")

    result = out.astype(np.float32)

    # Per-pixel negative fix: blend toward the (already linearly WB'd)
    # input where the matrix went negative, weighted by how negative it
    # went. The previous whole-pixel additive lift manufactured luminance
    # at star cores (one channel went strongly negative, lifting the
    # other two above 1.0) which, combined with the ceiling compression
    # below, produced the teal-core artifact.
    pixel_mins = np.min(result, axis=-1, keepdims=True)  # (H, W, 1)
    needs_fix = pixel_mins < 0.0
    if np.any(needs_fix):
        severity = np.clip(-pixel_mins / 0.2, 0.0, 1.0).astype(np.float32)
        result = np.where(
            needs_fix,
            result * (1.0 - severity) + bg_neutral.astype(np.float32) * severity,
            result,
        )

    # Ceiling: single scalar compress to preserve inter-channel ratios
    mx = float(np.nanmax(result))
    if mx > 1.0:
        lum_pivot = float(np.median(
            0.2126 * result[..., 0] +
            0.7152 * result[..., 1] +
            0.0722 * result[..., 2]
        ))
        denom = max(mx - lum_pivot, 1e-8)
        a = (1.0 - lum_pivot) / denom
        result = lum_pivot + (result - lum_pivot) * a

    # Highlight protection: blend matrix output back toward the linearly
    # WB'd input for near-saturated pixels. The matrix is trained on
    # stellar scatter in the wings where color info lives; saturated
    # cores carry no color info and shouldn't be rotated.
    #
    # edge0 raised 0.80 → 0.92: stars ARE the bright things, so a 0.80 floor
    # reverted the rotation across most of the population the matrix was
    # trained on — and since after_star_pixels is sampled from this reverted
    # image, the After diagnostic plot then under-reported whatever rotation
    # survived. 0.92 still protects genuinely blown cores.
    L_in = (0.2126 * bg_neutral[..., 0]
            + 0.7152 * bg_neutral[..., 1]
            + 0.0722 * bg_neutral[..., 2]).astype(np.float32)
    edge0, edge1 = 0.92, 0.995
    t = np.clip((L_in - edge0) / (edge1 - edge0), 0.0, 1.0)
    w = (t * t * (3.0 - 2.0 * t))[..., None]
    result = (result * (1.0 - w)
              + bg_neutral.astype(np.float32) * w).astype(np.float32)

    return result
def apply_star_based_white_balance(
    image: np.ndarray,
    threshold: float = 1.5,
    autostretch: bool = True,
    reuse_cached_sources: bool = False,
    return_star_colors: bool = False,
    use_color_matrix: bool = False,       # ← new
    matrix_strength: float = 0.7,         # ← how hard the 3x3 rotates
    # ── Auto-σ waterfall ────────────────────────────────────────────────
    auto_sigma: bool = False,
    auto_sigma_min_stars: int = DEFAULT_AUTO_SIGMA_MIN_STARS,
    auto_sigma_ladder: Sequence[float] = DEFAULT_AUTO_SIGMA_LADDER,
    on_sigma_resolved: Optional[Callable[[float], None]] = None,
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
        SEP detection threshold (in background sigma). Ignored if
        `auto_sigma=True`.
    autostretch : bool
        If True, overlay is built from an autostretched BN view for visibility.
    reuse_cached_sources : bool
        Reuse detections only if image shape and threshold match. Disabled
        automatically when `auto_sigma=True` since the resolved σ isn't
        known until detection runs.
    return_star_colors : bool
        If True, also returns (raw_star_pixels, after_star_pixels).
    matrix_strength : float
        Only used when `use_color_matrix=True`. How far each star's measured
        (R/B, G/B) is blended toward its projection on the blackbody locus
        before the 3x3 is solved: 0.0 = no rotation at all, 1.0 = ask for
        full projection onto the locus. Clamped to [0, 1].

        The default 0.7 is deliberately conservative. The target for each
        star is built from that star's own measured color, so a pathological
        population (heavy nebulosity, emission-line objects, a bad clip
        threshold leaving junk in the fit) steers the targets as well as the
        solve. Holding back to 70% means such a population bends the result
        rather than dictating it. Push it higher on clean broadband fields
        with a healthy unclipped star count; leave it at or below the default
        on narrowband, duoband, or sparse data.

        Note this is what is ASKED for, not what lands: the matrix is a
        linear map solved in raw RGB against targets defined in ratio space,
        so it realizes roughly three quarters of the request even at 1.0.
    auto_sigma : bool
        If True, run the σ waterfall described in `auto_sigma_ladder`,
        picking the highest σ that yields ≥ `auto_sigma_min_stars` sources.
        The BN pass and sep.Background estimation are performed exactly
        once — only sep.extract itself is retried per rung — so the extra
        cost is modest even on big frames.
    auto_sigma_min_stars : int
        Target star count that qualifies a σ rung as "good enough".
    auto_sigma_ladder : Sequence[float]
        σ values tried in order (highest first).
    on_sigma_resolved : callable(float) | None
        Optional callback invoked with the σ that was ultimately used.
        Useful for UI code that wants to drive a slider to the auto value.

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

    # Cache gate — mirrors detect_stars_for_wb so that Preview → Apply on
    # the same image runs SEP exactly once, in either mode:
    #   - manual mode: hit when caller asks AND cached σ matches input σ
    #   - auto mode:   hit when caller asks AND we have a cached result for
    #                  this shape — the waterfall is deterministic for a
    #                  given image, so the previously-resolved σ is the σ
    #                  we would resolve again.
    use_cache = (
        bool(reuse_cached_sources)
        and cached_star_sources is not None
        and cached_flux_radii is not None
        and _cached_shape == tuple(img_rgb.shape)
        and _cached_threshold is not None
        and (
            bool(auto_sigma)
            or abs(float(_cached_threshold) - float(threshold)) < 1e-9
        )
    )

    if use_cache:
        sources = cached_star_sources
        r = cached_flux_radii
        # In auto mode `threshold` is the stale slider value — report back
        # the σ actually used (the cached one), not the input.
        used_threshold = float(_cached_threshold)
    else:
        if auto_sigma:
            # σ waterfall — BN already done, so this only re-runs sep.extract
            sources, used_threshold = _waterfall_sep_extract(
                data_sub, err_val,
                ladder=auto_sigma_ladder,
                min_stars=auto_sigma_min_stars,
            )
        else:
            used_threshold = float(threshold)
            sources = sep.extract(data_sub, used_threshold, err=err_val)

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

        # Cache the already-filtered result (keyed by whatever σ actually ran)
        cached_star_sources = sources
        cached_flux_radii = r
        _cached_shape = tuple(img_rgb.shape)
        _cached_threshold = float(used_threshold)

    # Report the σ we actually used — UI code can use this to drive a slider
    if on_sigma_resolved is not None:
        try:
            on_sigma_resolved(float(used_threshold))
        except Exception:
            pass

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
    eps = 1e-8

    # Reject clipped-high / noise-floor samples at the RAW sample level —
    # a star saturated in any channel has lost its color information, so
    # rb/gb computed from it is noise, not photometry. This mask gates
    # BOTH the neutral reference and the slope fit below.
    unsat_mask = _photometric_keep_mask(
        bn_star_pixels, clip_high=_WB_CLIP_HIGH, clip_low=_WB_CLIP_LOW
    )

    # Neutral reference from unsaturated stars only — otherwise clipped
    # stars pull the median toward the clip ceiling in whichever channel
    # saturates first, biasing the subsequent neutral shift.
    ref_source = bn_star_pixels[unsat_mask] if np.any(unsat_mask) else bn_star_pixels
    ref_color = np.median(ref_source, axis=0).astype(np.float32)
    ref_color = np.where(ref_color <= eps, eps, ref_color)

    rb_stars = bn_star_pixels[:, 0] / (bn_star_pixels[:, 2] + eps)
    gb_stars = bn_star_pixels[:, 1] / (bn_star_pixels[:, 2] + eps)

    finite_mask = (
        unsat_mask &
        np.isfinite(rb_stars) & np.isfinite(gb_stars) &
        (rb_stars > 0.1) & (gb_stars > 0.1) &
        (rb_stars < 4.0) & (gb_stars < 4.0)
    )

    # Single source of truth, shared with the matrix stage (was a separate
    # hardcoded 0.55 here against 0.4 at the matrix call site).
    TARGET_SLOPE = LOCUS_TARGET_SLOPE

    # TILT_STRENGTH removed from the diagonal stage. The old code built a
    # per-channel `tilt`, then computed neutral_scale = neutral_target /
    # (ref_color * tilt) and multiplied the two together — in which tilt_c
    # CANCELS ALGEBRAICALLY:
    #
    #     final_scaling_c = tilt_c * neutral_target / (ref_color_c * tilt_c)
    #                     = neutral_target / ref_color_c
    #
    # leaving pure median neutralization with a scalar (neutral_target) that
    # multiplies all three channels equally and so cannot change R/B or G/B
    # at all. TARGET_SLOPE and TILT_STRENGTH had NO effect on output ratios.
    #
    # This is not merely a bug — a diagonal gain has only 2 ratio DOF
    # (kr/kb, kg/kb), and neutralizing the star median consumes both, so no
    # diagonal WB can neutralize AND set the locus slope. Rotation needs the
    # 3x3, which has 9 DOF and solves both constraints jointly. So this stage
    # now does one job honestly: put the median star at neutral.
    n_total = int(bn_star_pixels.shape[0])
    n_kept  = int(np.sum(finite_mask))
    n_sat   = int(np.sum(~unsat_mask))
    print(f"[WB] photometric fit: {n_kept}/{n_total} stars "
          f"({n_sat} rejected as clipped)")

    if n_kept >= 10:
        rb_f = rb_stars[finite_mask]
        gb_f = gb_stars[finite_mask]
        current_slope, _intercept, slope_keep = _robust_slope_fit(
            rb_f, gb_f, sigma=2.5, max_iter=5, min_points=10
        )
        current_slope = float(np.clip(current_slope, 0.05, 3.0))
        n_slope_kept = int(np.sum(slope_keep))
        print(f"[WB] slope fit: {n_slope_kept}/{n_kept} after MAD sigma-clip, "
              f"slope={current_slope:.4f}")
    else:
        current_slope = TARGET_SLOPE
        print(f"[WB] slope fit: too few unclipped stars ({n_kept}), "
              f"falling back to TARGET_SLOPE={TARGET_SLOPE}")

    # ---- Step 2: neutral shift ----
    # Scale so the median star's channels all land on its brightest channel
    # (i.e. the median star becomes gray). This is the ONLY thing a diagonal
    # gain can do once neutrality is required — see the note above.
    ref_safe = np.where(ref_color <= eps, eps, ref_color)
    neutral_target = float(np.max(ref_safe))
    final_scaling = (neutral_target / ref_safe).astype(np.float32)

    # Report the slope the diagonal stage will leave behind. A diagonal gain
    # maps slope' = (kg/kr) * slope, so this is predictable in closed form —
    # and it is what the 3x3 stage (if enabled) then has to rotate.
    _kg_kr = float(final_scaling[1] / max(final_scaling[0], eps))
    print(f"[WB] diagonal stage: neutralizing median star; "
          f"slope {current_slope:.4f} → {current_slope * _kg_kr:.4f} "
          f"(locus target {TARGET_SLOPE:.2f}; rotation needs the 3x3 matrix)")

    m = pivot.reshape((1, 1, 3)).astype(np.float32)
    g = final_scaling.reshape((1, 1, 3)).astype(np.float32)

    balanced = (bg_neutral.astype(np.float32) - m) * g + m
    balanced = np.clip(balanced, 0.0, 1.0).astype(np.float32, copy=False)
    # ── 5b) Color matrix WB (advanced, non-linear) ────────────────────────
    if use_color_matrix:
        # Re-sample star pixels from the already linearly-balanced image
        # so the matrix corrects residual color rotation, not the raw cast
        balanced_star_pixels, _ = _sample_star_circle_medians(balanced, sources, radius=3)
        # Re-compute pivot from balanced image background region
        balanced_pivot = np.median(
            balanced[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2], :],
            axis=(0, 1)
        ).astype(np.float32)

        matrix_result = _apply_color_matrix_wb(
            balanced,               # ← work on already-balanced image
            balanced_star_pixels,   # ← star samples from balanced image
            balanced_pivot,         # ← pivot from balanced image
            target_slope=LOCUS_TARGET_SLOPE,   # was a hardcoded 0.4
            tilt_strength=float(np.clip(matrix_strength, 0.0, 1.0)),
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
    # overlay (or None if with_overlay=False), count, resolved_threshold
    #   `resolved_threshold` is the σ that was actually used — identical to
    #   the input threshold in manual mode, or the auto-picked value when
    #   `auto_sigma=True`. Dialogs can use it to drive a slider so the user
    #   sees what Auto chose.
    #
    #   `overlay` is None unless the dialog explicitly asked for the preview
    #   via with_overlay=True. The default is cheap-mode (detection only)
    #   because on big RGB frames the overlay autostretch + full-image WB
    #   math dominate the open-time cost and the preview label itself is
    #   too small for the ellipses to be useful anyway.
    finished = _pyqtSignal(object, int, float)
    failed   = _pyqtSignal(str)

    def __init__(
        self,
        image: np.ndarray,
        threshold: float,
        autostretch: bool,
        parent=None,
        *,
        auto_sigma: bool = False,
        auto_sigma_min_stars: int = DEFAULT_AUTO_SIGMA_MIN_STARS,
        auto_sigma_ladder: Sequence[float] = DEFAULT_AUTO_SIGMA_LADDER,
        with_overlay: bool = False,
    ):
        super().__init__(parent)
        self._image = np.asarray(image, dtype=np.float32)
        self._threshold = float(threshold)
        self._autostretch = bool(autostretch)
        self._cancelled = False
        self._auto_sigma = bool(auto_sigma)
        self._auto_sigma_min_stars = int(auto_sigma_min_stars)
        self._auto_sigma_ladder = tuple(float(x) for x in auto_sigma_ladder)
        self._with_overlay = bool(with_overlay)
        # Populated by detect_stars_for_wb via on_sigma_resolved as a
        # fallback — the direct return value is used first.
        self._resolved_threshold: float = float(threshold)

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            def _on_resolved(sig: float):
                self._resolved_threshold = float(sig)

            count, used_sigma, sources, bg_neutral = detect_stars_for_wb(
                self._image,
                threshold=self._threshold,
                reuse_cached_sources=False,
                auto_sigma=self._auto_sigma,
                auto_sigma_min_stars=self._auto_sigma_min_stars,
                auto_sigma_ladder=self._auto_sigma_ladder,
                on_sigma_resolved=_on_resolved,
            )
            if self._cancelled:
                return

            if self._with_overlay:
                overlay = build_wb_preview_overlay(
                    bg_neutral, sources, autostretch=self._autostretch
                )
                if self._cancelled:
                    return
            else:
                overlay = None

            self.finished.emit(overlay, int(count), float(used_sigma))
        except Exception as e:
            if not self._cancelled:
                self.failed.emit(str(e))