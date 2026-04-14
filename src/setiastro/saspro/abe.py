# pro/abe.py — SASpro Automatic (Dynamic) Background Extraction (ADBE)
# -----------------------------------------------------------------------------
# This module migrates the SASv2 ABE functionality into SASpro with:
#   • Polynomial background model (degree 1–6)
#   • Optional RBF refinement stage (multiquadric) with smoothing
#   • Smart sample-point generation (borders, corners, quartiles) with
#     gradient-descent-to-dim-spot and bright-region avoidance
#   • User-drawn exclusion polygons directly on the preview (image-space)
#   • Non‑destructive preview, commit with undo, optional background doc
#   • Mono and RGB float workflows (expects [0..1] float domain internally)
# -----------------------------------------------------------------------------
from __future__ import annotations

import os
import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

from PyQt6.QtCore import Qt, QSize, QEvent, QPointF, QTimer, QSettings, QByteArray
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QSpinBox,
    QCheckBox, QPushButton, QScrollArea, QWidget, QMessageBox, QComboBox,
    QGroupBox, QApplication, QToolBar, QToolButton
)
from PyQt6.QtGui import QImage, QPixmap, QPainter, QColor, QPen
from PyQt6 import sip

from scipy.interpolate import RBFInterpolator

from .doc_manager import ImageDocument
from setiastro.saspro.legacy.numba_utils import build_poly_terms, evaluate_polynomial
from .autostretch import autostretch as hard_autostretch
from setiastro.saspro.widgets.themed_buttons import themed_toolbtn

# =============================================================================
#                         Headless ABE Core (poly + RBF)
# =============================================================================

def _downsample_area(img: np.ndarray, scale: int) -> np.ndarray:
    if scale <= 1:
        return img
    if cv2 is None:
        return img[::scale, ::scale] if img.ndim == 2 else img[::scale, ::scale, :]
    h, w = img.shape[:2]
    return cv2.resize(img, (max(1, w // scale), max(1, h // scale)), interpolation=cv2.INTER_AREA)


def _upscale_bg(bg_small: np.ndarray, out_shape: tuple[int, int]) -> np.ndarray:
    oh, ow = out_shape
    if cv2 is None:
        ys = (np.linspace(0, bg_small.shape[0] - 1, oh)).astype(int)
        xs = (np.linspace(0, bg_small.shape[1] - 1, ow)).astype(int)
        if bg_small.ndim == 2:
            return bg_small[ys][:, xs]
        return np.stack([bg_small[..., c][ys][:, xs] for c in range(bg_small.shape[2])], axis=-1)
    if bg_small.ndim == 2:
        return cv2.resize(bg_small, (ow, oh), interpolation=cv2.INTER_LANCZOS4).astype(np.float32)
    return np.stack(
        [cv2.resize(bg_small[..., c], (ow, oh), interpolation=cv2.INTER_LANCZOS4) for c in range(bg_small.shape[2])],
        axis=-1
    ).astype(np.float32)


def _fit_poly_on_small(small: np.ndarray, points: np.ndarray, degree: int, patch_size: int = 15) -> np.ndarray:
    H, W = small.shape[:2]
    half = patch_size // 2
    pts = np.asarray(points, dtype=np.int32)
    xs = np.clip(pts[:, 0], 0, W - 1)
    ys = np.clip(pts[:, 1], 0, H - 1)

    A = build_poly_terms(xs.astype(np.float32), ys.astype(np.float32), degree).astype(np.float32)

    if small.ndim == 3 and small.shape[2] == 3:
        bg_small = np.zeros_like(small, dtype=np.float32)
        
        # Batch collect samples: (num_samples, 3)
        # We need N samples. z will be list of (3,) arrays
        
        # Pre-allocate Z: (N, 3)
        Z = np.zeros((len(xs), 3), dtype=np.float32)
        
        for k, (x, y) in enumerate(zip(xs, ys)):
            x0, x1 = max(0, x - half), min(W, x + half + 1)
            y0, y1 = max(0, y - half), min(H, y + half + 1)
            # Efficiently compute median for all channels in this patch
            patch = small[y0:y1, x0:x1, :]
            Z[k] = np.median(patch, axis=(0, 1))

        # Solve once: A is (N, terms), Z is (N, 3) -> coeffs is (terms, 3)
        coeffs_all, *_ = np.linalg.lstsq(A, Z, rcond=None)
        
        # Evaluate per channel
        for c in range(3):
            # coeffs_all[:, c] gives the terms for channel c
            bg_small[..., c] = evaluate_polynomial(H, W, coeffs_all[:, c].astype(np.float32), degree)
            
        return bg_small
    else:
        z = []
        for x, y in zip(xs, ys):
            x0, x1 = max(0, x - half), min(W, x + half + 1)
            y0, y1 = max(0, y - half), min(H, y + half + 1)
            z.append(np.median(small[y0:y1, x0:x1]))
        z = np.asarray(z, dtype=np.float32)
        coeffs, *_ = np.linalg.lstsq(A, z, rcond=None)
        return evaluate_polynomial(H, W, coeffs.astype(np.float32), degree)


def _divide_into_quartiles(image: np.ndarray):
    h, w = image.shape[:2]
    hh, ww = h // 2, w // 2
    return {
        "top_left":     (slice(0, hh),     slice(0, ww),     (0, 0)),
        "top_right":    (slice(0, hh),     slice(ww, w),     (ww, 0)),
        "bottom_left":  (slice(hh, h),     slice(0, ww),     (0, hh)),
        "bottom_right": (slice(hh, h),     slice(ww, w),     (ww, hh)),
    }


def _exclude_bright_regions(gray: np.ndarray, exclusion_fraction: float = 0.5) -> np.ndarray:
    flat = gray.ravel()
    thresh = np.percentile(flat, 100 * (1 - exclusion_fraction))
    return (gray < thresh)


def _to_luminance(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return np.dot(img[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.float32)


def _gradient_descent_to_dim_spot(image: np.ndarray, x: int, y: int, max_iter: int = 500, patch_size: int = 15) -> tuple[int, int]:
    half = patch_size // 2
    lum = _to_luminance(image)
    H, W = lum.shape

    def patch_median(px: int, py: int) -> float:
        x0, x1 = max(0, px - half), min(W, px + half + 1)
        y0, y1 = max(0, py - half), min(H, py + half + 1)
        return float(np.median(lum[y0:y1, x0:x1]))

    cx, cy = int(np.clip(x, 0, W - 1)), int(np.clip(y, 0, H - 1))
    for _ in range(max_iter):
        cur = patch_median(cx, cy)
        xs = range(max(0, cx - 1), min(W, cx + 2))
        ys = range(max(0, cy - 1), min(H, cy + 2))
        best = (cx, cy); best_val = cur
        for nx in xs:
            for ny in ys:
                if nx == cx and ny == cy:
                    continue
                val = patch_median(nx, ny)
                if val < best_val:
                    best_val = val; best = (nx, ny)
        if best == (cx, cy):
            break
        cx, cy = best
    return cx, cy


def _generate_sample_points(image: np.ndarray, num_points: int = 100, exclusion_mask: np.ndarray | None = None, patch_size: int = 15) -> np.ndarray:
    H, W = image.shape[:2]
    pts: list[tuple[int, int]] = []
    border = 10

    def allowed(x: int, y: int) -> bool:
        if exclusion_mask is None:
            return True
        return bool(exclusion_mask[min(max(0, y), H-1), min(max(0, x), W-1)])

    # corners
    corners = [(border, border), (W - border - 1, border), (border, H - border - 1), (W - border - 1, H - border - 1)]
    for x, y in corners:
        if not allowed(x, y):
            continue
        nx, ny = _gradient_descent_to_dim_spot(image, x, y, patch_size=patch_size)
        if allowed(nx, ny):
            pts.append((nx, ny))

    # borders
    xs = np.linspace(border, W - border - 1, 5, dtype=int)
    ys = np.linspace(border, H - border - 1, 5, dtype=int)
    for x in xs:
        if allowed(x, border):
            nx, ny = _gradient_descent_to_dim_spot(image, x, border, patch_size=patch_size)
            if allowed(nx, ny):
                pts.append((nx, ny))
        if allowed(x, H - border - 1):
            nx, ny = _gradient_descent_to_dim_spot(image, x, H - border - 1, patch_size=patch_size)
            if allowed(nx, ny):
                pts.append((nx, ny))
    for y in ys:
        if allowed(border, y):
            nx, ny = _gradient_descent_to_dim_spot(image, border, y, patch_size=patch_size)
            if allowed(nx, ny):
                pts.append((nx, ny))
        if allowed(W - border - 1, y):
            nx, ny = _gradient_descent_to_dim_spot(image, W - border - 1, y, patch_size=patch_size)
            if allowed(nx, ny):
                pts.append((nx, ny))

    # quartiles with bright-region avoidance and descent
    quarts = _divide_into_quartiles(image)
    for _, (yslc, xslc, (x0, y0)) in quarts.items():
        sub = image[yslc, xslc]
        gray = _to_luminance(sub)
        bright_mask = _exclude_bright_regions(gray, exclusion_fraction=0.5)
        if exclusion_mask is not None:
            bright_mask &= exclusion_mask[yslc, xslc]
        elig = np.argwhere(bright_mask)
        if elig.size == 0:
            continue
        k = min(len(elig), max(1, num_points // 4))
        sel = elig[np.random.choice(len(elig), k, replace=False)]
        for (yy, xx) in sel:
            gx, gy = x0 + int(xx), y0 + int(yy)
            nx, ny = _gradient_descent_to_dim_spot(image, gx, gy, patch_size=patch_size)
            if allowed(nx, ny):
                pts.append((nx, ny))

    if len(pts) == 0:
        # fallback grid
        grid = int(np.sqrt(max(9, num_points)))
        xs = np.linspace(border, W - border - 1, grid, dtype=int)
        ys = np.linspace(border, H - border - 1, grid, dtype=int)
        pts = [(x, y) for y in ys for x in xs if allowed(x, y)]
    return np.array(pts, dtype=np.int32)

def _fit_rbf_on_small(small: np.ndarray, points: np.ndarray, smooth: float = 0.1, patch_size: int = 15) -> np.ndarray:
    """
    RBF background fit using scipy.interpolate.RBFInterpolator (scipy 1.7+).
    Replaces the deprecated scipy.interpolate.Rbf.
    Runs in float32 throughout — sufficient precision for smooth gradient fitting.
    Hard-capped at 1MP to prevent OOM regardless of downsample factor.
    """
    H, W = small.shape[:2]
    half = patch_size // 2

    # Hard cap: reduce further for RBF only if still too large
    MAX_RBF_PIXELS = 1_000_000
    rbf_scale = 1
    if H * W > MAX_RBF_PIXELS:
        rbf_scale = int(np.ceil(np.sqrt(H * W / MAX_RBF_PIXELS)))
        if cv2 is not None:
            small_rbf = cv2.resize(
                small,
                (max(1, W // rbf_scale), max(1, H // rbf_scale)),
                interpolation=cv2.INTER_AREA
            ).astype(np.float32, copy=False)
        else:
            small_rbf = small[::rbf_scale, ::rbf_scale].astype(np.float32, copy=False)
    else:
        small_rbf = small.astype(np.float32, copy=False)

    H_r, W_r = small_rbf.shape[:2]

    pts = np.asarray(points, dtype=np.int32)
    xs = np.clip(np.round(pts[:, 0] / rbf_scale), 0, W_r - 1).astype(np.float32)
    ys = np.clip(np.round(pts[:, 1] / rbf_scale), 0, H_r - 1).astype(np.float32)

    # RBFInterpolator expects (N, 2) array of (y, x) coordinates
    sample_coords = np.column_stack([ys, xs])  # (N, 2) float32

    # Query grid: (H_r * W_r, 2)
    grid_y, grid_x = np.mgrid[0:H_r, 0:W_r]
    query_coords = np.column_stack([
        grid_y.ravel().astype(np.float32),
        grid_x.ravel().astype(np.float32)
    ])

    def _median_patch(arr2d, x, y):
        x0, x1 = max(0, int(x) - half), min(W_r, int(x) + half + 1)
        y0, y1 = max(0, int(y) - half), min(H_r, int(y) + half + 1)
        return float(np.median(arr2d[y0:y1, x0:x1]))

    if small_rbf.ndim == 3 and small_rbf.shape[2] == 3:
        bg_r = np.zeros((H_r, W_r, 3), dtype=np.float32)
        for c in range(3):
            z = np.array(
                [_median_patch(small_rbf[..., c], x, y) for x, y in zip(xs, ys)],
                dtype=np.float32
            )
            interp = RBFInterpolator(
                sample_coords, z,
                kernel='multiquadric',
                smoothing=float(smooth) * len(z),  # RBFInterpolator smoothing scales with N
                epsilon=1.0
            )
            bg_r[..., c] = interp(query_coords).reshape(H_r, W_r).astype(np.float32)
        return _upscale_bg(bg_r, (H, W)) if rbf_scale > 1 else bg_r
    else:
        z = np.array(
            [_median_patch(small_rbf, x, y) for x, y in zip(xs, ys)],
            dtype=np.float32
        )
        interp = RBFInterpolator(
            sample_coords, z,
            kernel='multiquadric',
            smoothing=float(smooth) * len(z),
            epsilon=1.0
        )
        bg_r = interp(query_coords).reshape(H_r, W_r).astype(np.float32)
        return _upscale_bg(bg_r, (H, W)) if rbf_scale > 1 else bg_r

def _legacy_stretch_unlinked(image: np.ndarray):
    was_single = False
    img = image
    if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
        was_single = True
        img = np.stack([img[..., 0] if img.ndim == 3 else img] * 3, axis=-1)

    img = img.astype(np.float32, copy=True)
    target_median = 0.25

    ch_mins: list[float] = []
    ch_meds: list[float] = []
    out = img.copy()

    for c in range(3):
        m0 = float(np.min(out[..., c]))
        ch_mins.append(m0)
        out[..., c] -= m0
        med = float(np.median(out[..., c]))
        ch_meds.append(med)
        if med != 0.0:
            num = (med - 1.0) * target_median * out[..., c]
            den = (med * (target_median + out[..., c] - 1.0) - target_median * out[..., c])
            den = np.where(den == 0.0, 1e-6, den)
            out[..., c] = num / den

    # *** NO np.clip here — bright stars legitimately exceed 1.0 in stretch domain ***
    return out, {"mins": ch_mins, "meds": ch_meds, "was_single": was_single}


def _legacy_unstretch_unlinked(image: np.ndarray, state: dict):
    mins = state["mins"]; meds = state["meds"]; was_single = state["was_single"]
    img = image.astype(np.float32, copy=True)

    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.ndim == 3 and img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)

    for c in range(3):
        ch_med = float(np.median(img[..., c]))
        orig_med = float(meds[c])
        if ch_med != 0.0 and orig_med != 0.0:
            num = (ch_med - 1.0) * orig_med * img[..., c]
            den = (ch_med * (orig_med + img[..., c] - 1.0) - orig_med * img[..., c])
            den = np.where(den == 0.0, 1e-6, den)
            img[..., c] = num / den
        img[..., c] += float(mins[c])

    # *** NO np.clip here — let abe_run do the single final rescale ***
    if was_single:
        return img[..., 0]
    return img

def _anchor_median_linear_rescale(img: np.ndarray, pivot: float, eps: float = 1e-8) -> np.ndarray:
    """
    Clip-free range normalization anchored at the median (pivot).
    Step 1: lift negatives (shift so min == 0, pivot adjusts accordingly).
    Step 2: compress ceiling if max > 1.0, keeping pivot fixed.
    No hard clipping — all data is preserved via linear mapping.
    """
    out = np.asarray(img, dtype=np.float32).copy()

    # Step 1: lift negatives
    mn = float(np.nanmin(out))
    if mn < 0.0:
        out -= mn
        pivot = float(pivot) - mn

    # Step 2: compress ceiling only if needed
    mx = float(np.nanmax(out))
    if not np.isfinite(mx) or mx <= 1.0:
        return out

    p = float(pivot) if np.isfinite(pivot) else 0.0
    denom = max(mx - p, eps)
    a = (1.0 - p) / denom
    out = p + (out - p) * a

    return out.astype(np.float32, copy=False)

def abe_run(
    image: np.ndarray,
    degree: int = 2,
    num_samples: int = 100,
    downsample: int = 4,
    patch_size: int = 15,
    use_rbf: bool = True,
    rbf_smooth: float = 0.1,
    exclusion_mask: np.ndarray | None = None,
    return_background: bool = True,
    progress_cb=None,
    legacy_prestretch: bool = True,
    manual_points: np.ndarray | None = None,   # NEW: Nx2 full-image coords
) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
    """Two-stage ABE (poly + optional RBF) with SASv2-compatible pre/post stretch.

    NOTE: For RGB images, always run ABE on the combined RGB image rather than
    on individual channels separately. The legacy stretch/unstretch stage
    operates on per-channel statistics — running it independently on R, G, B
    then combining can introduce small inter-channel offsets at saturated pixels
    that downstream tools (e.g. color matrix white balance) may amplify into
    visible color artifacts.
    """
    if image is None:
        raise ValueError("ABE: image is None")

    img_src = np.asarray(image).astype(np.float32, copy=False)
    mono = (img_src.ndim == 2) or (img_src.ndim == 3 and img_src.shape[2] == 1)
    # Preserve dynamic range: normalize only for internal modeling if needed,
    # then restore the original scale before returning.
    finite_mask = np.isfinite(img_src)
    work_scale = float(np.max(img_src[finite_mask])) if np.any(finite_mask) else 1.0
    if not np.isfinite(work_scale) or work_scale <= 0.0:
        work_scale = 1.0

    img_src_work = img_src / work_scale if work_scale > 1.0 else img_src
    # Work in RGB internally (even for mono) so pre/post stretch matches SASv2 behavior
    img_rgb = img_src_work if (img_src_work.ndim == 3 and img_src_work.shape[2] == 3) else np.stack(
        [img_src_work.squeeze()] * 3, axis=-1
    )

    # --- SASv2 modeling domain (optional) ---------------------------------
    stretch_state = None
    if legacy_prestretch:
        img_rgb, stretch_state = _legacy_stretch_unlinked(img_rgb)

    # IMPORTANT: compute original median ONCE in the modeling domain
    orig_med = float(np.median(img_rgb))

    # downsample & mask (for fitting only)
    if progress_cb: progress_cb("Downsampling image…")
    small = _downsample_area(img_rgb, downsample)
    mask_small = None
    if exclusion_mask is not None:
        if progress_cb: progress_cb("Downsampling exclusion mask…")
        mask_small = _downsample_area(exclusion_mask.astype(np.float32), downsample) >= 0.5
    manual_small = None
    if manual_points is not None:
        mp = np.asarray(manual_points, dtype=np.int32)
        if mp.ndim == 2 and mp.shape[1] == 2 and len(mp) > 0:
            manual_small = _scale_points_to_small(mp, img_rgb.shape[:2], small.shape[:2])

            # honor exclusion mask if present
            if mask_small is not None and len(manual_small) > 0:
                keep = []
                Hs, Ws = mask_small.shape[:2]
                for x, y in manual_small:
                    xx = int(np.clip(x, 0, Ws - 1))
                    yy = int(np.clip(y, 0, Hs - 1))
                    if bool(mask_small[yy, xx]):
                        keep.append((xx, yy))
                manual_small = np.asarray(keep, dtype=np.int32) if keep else np.zeros((0, 2), dtype=np.int32)
    # ---------- Polynomial stage (skip when degree == 0) ----------
    if degree <= 0:
        if progress_cb: progress_cb("Degree 0: skipping polynomial stage…")
        after_poly = img_rgb.copy()                         # nothing removed yet
        total_bg   = np.zeros_like(img_rgb, dtype=np.float32)
    else:
        if progress_cb: progress_cb("Sampling points (poly stage)…")
        if manual_small is not None and len(manual_small) > 0:
            pts = manual_small
        else:
            pts = _generate_sample_points(
                small,
                num_points=num_samples,
                exclusion_mask=mask_small,
                patch_size=patch_size
            )

        if progress_cb: progress_cb(f"Fitting polynomial (degree {degree})…")
        bg_poly_small = _fit_poly_on_small(small, pts, degree=degree, patch_size=patch_size)

        if progress_cb: progress_cb("Upscaling polynomial background…")
        bg_poly = _upscale_bg(bg_poly_small, img_rgb.shape[:2])

        if progress_cb: progress_cb("Subtracting polynomial background & re-centering…")
        after_poly = img_rgb - bg_poly
        med_after  = float(np.median(after_poly))
        after_poly = after_poly + (orig_med - med_after)
        after_poly = _anchor_median_linear_rescale(after_poly, orig_med)  # ← restored

        total_bg = bg_poly.astype(np.float32, copy=False)
    # ---------- RBF refinement --------------------------------------------
    if use_rbf:
        if progress_cb: progress_cb("Downsampling for RBF stage…")
        small_rbf = _downsample_area(after_poly, downsample)

        if progress_cb: progress_cb("Sampling points (RBF stage)…")
        if manual_points is not None and len(manual_points) > 0:
            pts_rbf = _scale_points_to_small(
                np.asarray(manual_points, dtype=np.int32),
                img_rgb.shape[:2],
                small_rbf.shape[:2],
            )

            if mask_small is not None and len(pts_rbf) > 0:
                keep = []
                Hs, Ws = small_rbf.shape[:2]
                # if mask_small shape differs slightly from small_rbf, rebuild a matching one
                if mask_small.shape[:2] != small_rbf.shape[:2]:
                    mask_rbf = _downsample_area(exclusion_mask.astype(np.float32), downsample) >= 0.5
                else:
                    mask_rbf = mask_small

                for x, y in pts_rbf:
                    xx = int(np.clip(x, 0, Ws - 1))
                    yy = int(np.clip(y, 0, Hs - 1))
                    if bool(mask_rbf[yy, xx]):
                        keep.append((xx, yy))
                pts_rbf = np.asarray(keep, dtype=np.int32) if keep else np.zeros((0, 2), dtype=np.int32)
        else:
            pts_rbf = _generate_sample_points(
                small_rbf,
                num_points=num_samples,
                exclusion_mask=mask_small,
                patch_size=patch_size
            )

        if progress_cb: progress_cb(f"Fitting RBF (smooth={rbf_smooth:.3f})…")
        bg_rbf_small = _fit_rbf_on_small(small_rbf, pts_rbf, smooth=rbf_smooth, patch_size=patch_size)

        if progress_cb: progress_cb("Upscaling RBF background…")
        bg_rbf = _upscale_bg(bg_rbf_small, img_rgb.shape[:2])

        if progress_cb: progress_cb("Combining backgrounds & finalizing…")
        total_bg = (total_bg + bg_rbf).astype(np.float32)
        corrected = img_rgb - total_bg
        med2 = float(np.median(corrected))
        corrected = corrected + (orig_med - med2)
        corrected = _anchor_median_linear_rescale(corrected, orig_med)  # ← restored
    else:
        if progress_cb: progress_cb("Finalizing…")
        corrected = after_poly

    # --- Undo SASv2 modeling domain if used -------------------------------
    if legacy_prestretch and stretch_state is not None:
        if progress_cb: progress_cb("Unstretching to source domain…")
        corrected = _legacy_unstretch_unlinked(corrected, stretch_state)
        total_bg  = _legacy_unstretch_unlinked(total_bg,  stretch_state)

        # Make sure types are float32
        corrected = corrected.astype(np.float32, copy=False)
        total_bg  = total_bg.astype(np.float32, copy=False)

        # If original was mono, squeeze to 2D
        if mono:
            if corrected.ndim == 3:
                corrected = corrected[..., 0]
            if total_bg.ndim == 3:
                total_bg  = total_bg[..., 0]
    else:
        # We stayed in RGB all along; if the source was mono, return mono
        if mono:
            corrected = corrected[..., 0]
            total_bg  = total_bg[..., 0]
    # Restore original input scale if we normalized for internal processing
    if work_scale > 1.0:
        corrected = corrected * work_scale
        total_bg = total_bg * work_scale

    if progress_cb: progress_cb("Ready")
    if return_background:
        return corrected.astype(np.float32, copy=False), total_bg.astype(np.float32, copy=False)
    return corrected.astype(np.float32, copy=False)



def mtf_style_autostretch(image: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    def stretch_channel(c):
        med = np.median(c); mad = np.median(np.abs(c - med))
        mad_std = mad * 1.4826
        mn, mx = float(c.min()), float(c.max())
        bp = max(mn, med - sigma * mad_std)
        wp = min(mx, med + 0.5*sigma * mad_std)
        if wp - bp <= 1e-8:
            return np.zeros_like(c, dtype=np.float32)
        out = (c - bp) / (wp - bp)
        return np.clip(out, 0.0, 1.0).astype(np.float32)

    if image.ndim == 2:
        return stretch_channel(image.astype(np.float32, copy=False))
    if image.ndim == 3 and image.shape[2] == 3:
        return np.stack([stretch_channel(image[..., i].astype(np.float32, copy=False))
                         for i in range(3)], axis=-1)
    raise ValueError("Unsupported image format for autostretch.")

def _scale_points_to_small(points: np.ndarray, src_hw: tuple[int, int], small_hw: tuple[int, int]) -> np.ndarray:
    """
    Scale image-space points from source resolution into downsampled 'small' resolution.
    points: Nx2 in full-image pixel coords
    """
    pts = np.asarray(points, dtype=np.float32)
    if pts.size == 0:
        return np.zeros((0, 2), dtype=np.int32)

    src_h, src_w = int(src_hw[0]), int(src_hw[1])
    sm_h, sm_w = int(small_hw[0]), int(small_hw[1])

    sx = float(sm_w) / float(max(1, src_w))
    sy = float(sm_h) / float(max(1, src_h))

    out = np.empty_like(pts, dtype=np.int32)
    out[:, 0] = np.clip(np.round(pts[:, 0] * sx), 0, sm_w - 1).astype(np.int32)
    out[:, 1] = np.clip(np.round(pts[:, 1] * sy), 0, sm_h - 1).astype(np.int32)
    return out



# =============================================================================
#                                   UI Dialog
# =============================================================================

def _asfloat32(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x)                  # zero-copy view when possible
    return a if a.dtype == np.float32 else a.astype(np.float32, copy=False)

class ABEDialog(QDialog):
    """
    Non-destructive preview with polygon exclusions and optional RBF stage.
    Apply commits to the document image with undo. Optionally spawns a
    background document containing the extracted gradient.
    """
    def __init__(self, parent, document: ImageDocument):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Automatic (Dynamic) Background Extraction (ADBE)"))

        # IMPORTANT: avoid “attached modal sheet” behavior on some Linux WMs
        self.setWindowFlag(Qt.WindowType.Window, True)
        # Non-modal: allow user to switch between images while dialog is open
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setModal(False)
        try:
            self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        except Exception:
            pass  # older PyQt6 versions

        self._main = parent
        self.doc = document
        self._manual_points: list[QPointF] = []

        self._connected_current_doc_changed = False
        if hasattr(self._main, "currentDocumentChanged"):
            try:
                self._main.currentDocumentChanged.connect(self._on_active_doc_changed)
                self._connected_current_doc_changed = True
            except Exception:
                self._connected_current_doc_changed = False

        self._preview_scale = 1.0
        self._preview_qimg = None
        self._last_preview = None  # backing ndarray for QImage lifetime
        self._overlay = None


        # image-space polygons: list[list[QPointF]] in ORIGINAL IMAGE COORDS
        self._polygons: list[list[QPointF]] = []
        self._drawing_poly: list[QPointF] | None = None
        self._panning = False
        self._pan_last = None
        self._preview_source_f01 = None 

        # ---------------- Controls ----------------
        self.sp_degree = QSpinBox(); self.sp_degree.setRange(0, 6); self.sp_degree.setValue(2)
        self.sp_samples = QSpinBox(); self.sp_samples.setRange(20, 10000); self.sp_samples.setSingleStep(20); self.sp_samples.setValue(120)
        self.sp_down = QSpinBox()
        self.sp_down.setRange(4, 32)  # minimum 2, not 1
        self.sp_down.setValue(8)
        self.sp_down.setToolTip(
            "Downsample factor for background fitting.\n"
            "Minimum 4 recommended — factor 1 (full resolution) will\n"
            "cause extreme memory use with RBF refinement enabled."
        )
        self.sp_patch = QSpinBox(); self.sp_patch.setRange(5, 151); self.sp_patch.setSingleStep(2); self.sp_patch.setValue(15)
        self.chk_use_rbf = QCheckBox(self.tr("Enable RBF refinement (after polynomial)")); self.chk_use_rbf.setChecked(True)
        self.sp_rbf = QSpinBox(); self.sp_rbf.setRange(0, 1000); self.sp_rbf.setValue(100)  # shown as ×0.01 below
        self.chk_make_bg_doc = QCheckBox(self.tr("Create background document")); self.chk_make_bg_doc.setChecked(False)
        self.chk_preview_bg   = QCheckBox(self.tr("Preview background instead of corrected")); self.chk_preview_bg.setChecked(False)
        self.cmb_sample_mode = QComboBox()
        self.cmb_sample_mode.addItems(["Auto", "Manual"])
        # ── Place Points group ──────────────────────────────────────────
        gb_place = QGroupBox(self.tr("Place Sample Points"))
        place_layout = QVBoxLayout(gb_place)

        # Grid row
        grid_row = QHBoxLayout()
        self.btn_place_grid = QPushButton(self.tr("Place Grid"))
        self.btn_place_grid.setToolTip(
            "Fill the image with a regular N×N grid of sample points.\n"
            "Switches to Manual mode so you can add/remove points before running."
        )
        self.sp_grid_size = QSpinBox()
        self.sp_grid_size.setRange(2, 50)
        self.sp_grid_size.setValue(10)
        self.sp_grid_size.setPrefix("Grid: ")
        self.sp_grid_size.setSuffix("×" + str(self.sp_grid_size.value()))
        self.sp_grid_size.setToolTip("Number of grid points along each axis (N×N total)")
        self.sp_grid_size.valueChanged.connect(
            lambda v: self.sp_grid_size.setSuffix(f"×{v}")
        )
        grid_row.addWidget(self.btn_place_grid)
        grid_row.addWidget(self.sp_grid_size)
        grid_row.addStretch(1)

        # Auto points row
        self.btn_place_auto = QPushButton(self.tr("Show Auto Points"))
        self.btn_place_auto.setToolTip(
            "Run the automatic gradient-descent sampler and show all chosen points.\n"
            "Switches to Manual mode so you can tweak before running."
        )

        place_layout.addLayout(grid_row)
        place_layout.addWidget(self.btn_place_auto)

        self.btn_clear_samples = QPushButton(self.tr("Clear Sample Points"))
        self.btn_clear_samples.clicked.connect(self._clear_manual_samples)

        self.btn_place_grid.clicked.connect(self._place_grid_points)
        self.btn_place_auto.clicked.connect(self._place_auto_points)

        # Preview area
        self.preview_label = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumSize(QSize(480, 360))
        self.preview_label.setScaledContents(False)
        self.preview_scroll = QScrollArea()
        self.preview_scroll.setWidgetResizable(False)
        self.preview_scroll.setWidget(self.preview_label)
        self.preview_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.preview_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        # Buttons
        self.btn_preview = QPushButton(self.tr("Preview"))
        self.btn_apply   = QPushButton(self.tr("Apply"))
        self.btn_close   = QPushButton(self.tr("Close"))
        self.btn_clear   = QPushButton(self.tr("Clear Exclusions"))
        self.btn_preview.clicked.connect(self._do_preview)
        self.btn_apply.clicked.connect(self._do_apply)
        self.btn_close.clicked.connect(self.close)
        self.btn_clear.clicked.connect(self._clear_polys)

        # Layout
        params = QFormLayout()
        params.addRow(self.tr("Polynomial degree:"), self.sp_degree)
        params.addRow(self.tr("# sample points:"),   self.sp_samples)
        params.addRow(self.tr("Downsample factor:"), self.sp_down)
        params.addRow(self.tr("Patch size (px):"),   self.sp_patch)
        params.addRow(self.tr("Sample mode:"), self.cmb_sample_mode)

        # ── Workflow note ───────────────────────────────────────────────
        note_lbl = QLabel(
            "💡 Tip: For best results with RGB data, combine R/G/B channels "
            "before running ADBE. Running ADBE on individual mono channels "
            "separately can cause color artifacts at saturated stars after combining."
        )
        note_lbl.setWordWrap(True)
        note_lbl.setStyleSheet(
            "font-size: 10px; color: palette(window-text); "
            "background: palette(base); "
            "border: 1px solid palette(mid); "
            "border-radius: 3px; padding: 4px;"
        )

        rbf_box = QGroupBox(self.tr("RBF Refinement"))
        rbf_form = QFormLayout()
        rbf_form.addRow(self.chk_use_rbf)
        rbf_form.addRow(self.tr("Smooth (x0.01):"), self.sp_rbf)
        rbf_box.setLayout(rbf_form)

        opts = QVBoxLayout()
        opts.addLayout(params)          # degree, samples, downsample, patch, sample mode
        opts.addWidget(note_lbl)
        opts.addWidget(gb_place)        # Place Grid / Show Auto Points — part of sampling setup
        opts.addWidget(self.btn_clear_samples)
        opts.addWidget(rbf_box)         # RBF refinement
        opts.addWidget(self.chk_make_bg_doc)
        opts.addWidget(self.chk_preview_bg)
        row = QHBoxLayout()
        row.addWidget(self.btn_preview)
        row.addWidget(self.btn_apply)
        row.addStretch(1)
        opts.addLayout(row)
        opts.addWidget(self.btn_clear)  # Clear Exclusions (polygon drawing tool)
        opts.addStretch(1)

        # ▼ New status label
        self.status_label = QLabel("Ready")
        self.status_label.setWordWrap(True)
        opts.addWidget(self.status_label)

        opts.addStretch(1)

        # ⬇️ New right-side stack: toolbar row ABOVE the preview
        right = QVBoxLayout()
        right.addLayout(self._build_toolbar())      # Zoom In / Out / Fit / Autostretch
        right.addWidget(self.preview_scroll, 1)     # Preview below the buttons

        main = QHBoxLayout(self)
        main.addLayout(opts, 0)                     # Left controls
        main.addLayout(right, 1)                    # Right: buttons above preview

        self._load_settings()

        self._base_pixmap = None  # clean, scaled image with no overlays
        self.preview_scroll.viewport().installEventFilter(self)
        self.preview_label.installEventFilter(self)
        self._install_zoom_filters()
        self._populate_initial_preview()
        self.sp_degree.valueChanged.connect(self._degree_changed) 
        self.sp_degree.valueChanged.connect(self._save_settings)
        self.sp_samples.valueChanged.connect(self._save_settings)
        self.sp_down.valueChanged.connect(self._save_settings)
        self.sp_patch.valueChanged.connect(self._save_settings)

        self.chk_use_rbf.toggled.connect(self._save_settings)
        self.sp_rbf.valueChanged.connect(self._save_settings)

        self.chk_make_bg_doc.toggled.connect(self._save_settings)
        self.chk_preview_bg.toggled.connect(self._save_settings)

        self._sample_mode_changed(self.cmb_sample_mode.currentText())

        self.cmb_sample_mode.currentTextChanged.connect(self._sample_mode_changed)
        self.cmb_sample_mode.currentTextChanged.connect(self._save_settings)

    def _load_settings(self):
        s = QSettings()

        # Core ABE params
        self.sp_degree.setValue(int(s.value("abe/degree", 2)))
        self.sp_samples.setValue(int(s.value("abe/samples", 120)))
        self.sp_down.setValue(int(s.value("abe/downsample", 4)))
        self.sp_patch.setValue(int(s.value("abe/patch_size", 15)))
        sample_mode = str(s.value("abe/sample_mode", "Auto"))
        if self.cmb_sample_mode.findText(sample_mode) >= 0:
            self.cmb_sample_mode.setCurrentText(sample_mode)
        # RBF
        self.chk_use_rbf.setChecked(bool(s.value("abe/use_rbf", True, type=bool)))
        self.sp_rbf.setValue(int(s.value("abe/rbf_smooth_x100", 100)))

        # Options
        self.chk_make_bg_doc.setChecked(bool(s.value("abe/make_bg_doc", False, type=bool)))
        self.chk_preview_bg.setChecked(bool(s.value("abe/preview_bg", False, type=bool)))

        # Optional preview prefs
        self._autostretch_on = bool(s.value("abe/preview_autostretch", True, type=bool))


    def _save_settings(self):
        s = QSettings()

        s.setValue("abe/degree", self.sp_degree.value())
        s.setValue("abe/samples", self.sp_samples.value())
        s.setValue("abe/downsample", self.sp_down.value())
        s.setValue("abe/patch_size", self.sp_patch.value())
        s.setValue("abe/sample_mode", self.cmb_sample_mode.currentText())

        s.setValue("abe/use_rbf", self.chk_use_rbf.isChecked())
        s.setValue("abe/rbf_smooth_x100", self.sp_rbf.value())

        s.setValue("abe/make_bg_doc", self.chk_make_bg_doc.isChecked())
        s.setValue("abe/preview_bg", self.chk_preview_bg.isChecked())

        s.setValue("abe/preview_autostretch", bool(getattr(self, "_autostretch_on", False)))

    def _manual_mode(self) -> bool:
        return self.cmb_sample_mode.currentText().lower() == "manual"


    def _sample_mode_changed(self, _text: str):
        manual = self._manual_mode()

        if manual:
            # Switching to Manual — clear exclusion polygons
            # (they were drawn in Auto context and don't make sense to carry over)
            if self._polygons or self._drawing_poly:
                self._polygons.clear()
                self._drawing_poly = None

        else:
            # Switching to Auto — clear manual sample points
            # (they were placed manually and Auto will generate its own)
            if self._manual_points:
                self._manual_points.clear()

        # In manual mode, # sample points is not used for fitting
        self.sp_samples.setEnabled(not manual)

        tip = (
            "Auto: ADBE picks sample points automatically.\n"
            "Manual: left-click to add sample squares, right-click to remove nearest one."
        )
        self.cmb_sample_mode.setToolTip(tip)
        self.btn_clear_samples.setEnabled(manual and len(self._manual_points) > 0)

        self._redraw_overlay()


    def _clear_manual_samples(self):
        self._manual_points.clear()
        self.btn_clear_samples.setEnabled(False)
        self._redraw_overlay()


    def _manual_points_array(self) -> np.ndarray | None:
        if not self._manual_points:
            return None
        pts = np.array([[int(round(p.x())), int(round(p.y()))] for p in self._manual_points], dtype=np.int32)
        return pts


    def _find_nearest_manual_point_index(self, img_pt: QPointF, max_dist_px: float = 20.0) -> int:
        if not self._manual_points:
            return -1

        best_idx = -1
        best_d2 = float("inf")
        x0, y0 = float(img_pt.x()), float(img_pt.y())

        for i, p in enumerate(self._manual_points):
            dx = float(p.x()) - x0
            dy = float(p.y()) - y0
            d2 = dx * dx + dy * dy
            if d2 < best_d2:
                best_d2 = d2
                best_idx = i

        if best_idx < 0:
            return -1

        return best_idx if best_d2 <= (max_dist_px * max_dist_px) else -1

    def _place_grid_points(self):
        """Generate a regular grid of sample points and switch to manual mode."""
        src = self._get_source_float()
        if src is None:
            return

        H, W = src.shape[:2]
        n = int(self.sp_grid_size.value())
        border = int(self.sp_patch.value()) // 2 + 2

        excl = self._build_exclusion_mask()

        xs = np.linspace(border, W - border - 1, n, dtype=int)
        ys = np.linspace(border, H - border - 1, n, dtype=int)

        new_pts = []
        for y in ys:
            for x in xs:
                # Skip points inside exclusion polygons
                if excl is not None and not excl[int(y), int(x)]:
                    continue
                new_pts.append(QPointF(float(x), float(y)))

        self._manual_points.clear()
        self._manual_points.extend(new_pts)

        # Switch to manual so user can edit
        self.cmb_sample_mode.setCurrentText("Manual")
        self.btn_clear_samples.setEnabled(len(self._manual_points) > 0)
        self._redraw_overlay()
        self._set_status(f"Placed {len(new_pts)} grid points ({n}×{n}). Edit as needed.")


    def _place_auto_points(self):
        """Run auto gradient-descent sampling and push results into manual points."""
        src = self._get_source_float()
        if src is None:
            return

        self._set_status("Running auto sampling with gradient descent…")
        QApplication.processEvents()

        dwn   = int(self.sp_down.value())
        npts  = int(self.sp_samples.value())
        patch = int(self.sp_patch.value())

        excl = self._build_exclusion_mask()

        # Downsample for speed (same as abe_run does)
        small = _downsample_area(src, dwn)
        mask_small = None
        if excl is not None:
            mask_small = _downsample_area(excl.astype(np.float32), dwn) >= 0.5

        # Run the same auto sampler used internally
        pts_small = _generate_sample_points(
            small,
            num_points=npts,
            exclusion_mask=mask_small,
            patch_size=patch,
        )

        if pts_small is None or len(pts_small) == 0:
            self._set_status("Auto sampling found no valid points.")
            return

        # Scale back up to full image coords
        H, W = src.shape[:2]
        sh, sw = small.shape[:2]
        sx = W / max(1, sw)
        sy = H / max(1, sh)

        self._manual_points.clear()
        for x, y in pts_small:
            fx = float(np.clip(x * sx, 0, W - 1))
            fy = float(np.clip(y * sy, 0, H - 1))
            self._manual_points.append(QPointF(fx, fy))

        # Switch to manual so user can edit
        self.cmb_sample_mode.setCurrentText("Manual")
        self.btn_clear_samples.setEnabled(len(self._manual_points) > 0)
        self._redraw_overlay()
        self._set_status(
            f"Placed {len(self._manual_points)} auto-sampled points. "
            "Right-click to remove, left-click to add."
        )

    def _post_init_fit_and_stretch(self) -> None:
        # No longer used — sequence moved entirely into showEvent
        pass

    def _set_status(self, text: str) -> None:
        try:
            lbl = getattr(self, "status_label", None)
            if lbl is None:
                return
            # Guard against wrapped C++ object deleted (dialog partially destroyed)
            from PyQt6 import sip
            if sip.isdeleted(lbl):
                return
            lbl.setText(text)
            QApplication.processEvents()
        except RuntimeError:
            pass

    def _build_toolbar(self):
        """
        Toolbar row: Zoom In, Zoom Out, Fit, Autostretch.
        """
        bar = QHBoxLayout()
        bar.setContentsMargins(0, 0, 0, 0)
        bar.setSpacing(6)

        self.btn_zoom_in  = themed_toolbtn("zoom-in", "Zoom In")
        self.btn_zoom_out = themed_toolbtn("zoom-out", "Zoom Out")
        self.btn_fit      = themed_toolbtn("zoom-fit-best", "Fit to Preview")

        # Use a plain QToolButton so it is always visible even if the theme icon is missing.
        self.btn_autostr = QToolButton(self)
        self.btn_autostr.setText("AutoStretch")
        self.btn_autostr.setCheckable(True)
        self.btn_autostr.setChecked(bool(getattr(self, "_autostretch_on", False)))
        self.btn_autostr.setToolTip("Autostretch preview")
        self.btn_autostr.setMinimumHeight(24)
        self.btn_autostr.setMinimumWidth(44)

        self.btn_zoom_in.clicked.connect(self.zoom_in)
        self.btn_zoom_out.clicked.connect(self.zoom_out)
        self.btn_fit.clicked.connect(self.fit_to_preview)
        self.btn_autostr.clicked.connect(self.autostretch_preview)

        # Keep all toolbar buttons together on the left
        bar.addWidget(self.btn_zoom_in)
        bar.addWidget(self.btn_zoom_out)
        bar.addWidget(self.btn_fit)
        bar.addWidget(self.btn_autostr)
        bar.addStretch(1)

        return bar

    # ----- active document change -----
    def _on_active_doc_changed(self, doc):
        if doc is None or getattr(doc, "image", None) is None:
            return
        self.doc = doc
        self._polygons.clear()
        self._drawing_poly = None
        self._manual_points.clear()
        self._preview_source_f01 = None
        self._populate_initial_preview()

    # ----- data helpers -----
    def _get_source_float(self) -> np.ndarray | None:
        src = np.asarray(self.doc.image)
        if src is None or src.size == 0:
            return None

        if np.issubdtype(src.dtype, np.integer):
            scale = float(np.iinfo(src.dtype).max)
            return src.astype(np.float32) / scale

        # float path: preserve full range; abe_run() will normalize internally if needed
        return src.astype(np.float32, copy=False)

    # ----- preview/applier -----
    def _run_abe(self, excl_mask: np.ndarray | None, progress=None):
        imgf = self._get_source_float()
        if imgf is None:
            return None, None

        deg   = int(self.sp_degree.value())
        npts  = int(self.sp_samples.value())
        dwn   = int(self.sp_down.value())
        patch = int(self.sp_patch.value())
        use_rbf = bool(self.chk_use_rbf.isChecked())
        rbf_smooth = float(self.sp_rbf.value()) * 0.01

        manual_pts = self._manual_points_array() if self._manual_mode() else None

        return abe_run(
            imgf,
            degree=deg,
            num_samples=npts,
            downsample=dwn,
            patch_size=patch,
            use_rbf=use_rbf,
            rbf_smooth=rbf_smooth,
            exclusion_mask=excl_mask,
            return_background=True,
            progress_cb=progress,
            manual_points=manual_pts,   # NEW
        )

    def _degree_changed(self, v: int):
        # Make it clear what 0 means, and default RBF on (can still be unchecked)
        if v == 0:
            self.chk_use_rbf.setChecked(True)
            if hasattr(self, "_set_status"):
                self._set_status("Polynomial disabled (degree 0) → RBF-only.")
        else:
            if hasattr(self, "_set_status"):
                self._set_status("Ready")

    def _populate_initial_preview(self):
        """Don't render anything yet — just store the source. showEvent handles the rest."""
        src = self._get_source_float()
        if src is None:
            return
        self._preview_source_f01 = np.clip(_asfloat32(src), 0.0, 1.0)
        # Leave _preview_qimg as None — placeholder shown in showEvent


    def _show_placeholder(self):
        if getattr(self, "_closing", False):
            return
        """Show a 'Computing preview…' message in the preview area while we work."""
        vp = self.preview_scroll.viewport()
        w, h = max(480, vp.width()), max(360, vp.height())

        # Dark background with centered text — matches the typical dark theme
        pm = QPixmap(w, h)
        pm.fill(QColor(30, 30, 30))

        painter = QPainter(pm)
        painter.setPen(QColor(180, 180, 180))
        font = painter.font()
        font.setPointSize(12)
        painter.setFont(font)
        painter.drawText(
            pm.rect(),
            Qt.AlignmentFlag.AlignCenter,
            "Computing preview…"
        )
        painter.end()

        self.preview_label.setPixmap(pm)
        self.preview_label.resize(pm.size())
        QApplication.processEvents()

    def _render_preview_from_source(self, stretch: bool = True):
        """
        Render self._preview_source_f01 into QImage/pixmap.
        stretch=True applies autostretch. Called after geometry is known.
        """
        src = getattr(self, "_preview_source_f01", None)
        if src is None:
            return

        if stretch and getattr(self, "_autostretch_on", False):
            disp = hard_autostretch(src, target_median=0.5, sigma=2,
                                    linked=False, use_24bit=True)
            disp = np.asarray(disp, dtype=np.float32)
        else:
            disp = np.clip(src, 0.0, 1.0).astype(np.float32)

        if disp.ndim == 2 or (disp.ndim == 3 and disp.shape[2] == 1):
            mono = disp if disp.ndim == 2 else disp[..., 0]
            buf8 = np.ascontiguousarray((mono * 255.0).astype(np.uint8))
            self._last_preview = np.ascontiguousarray(np.stack([buf8] * 3, axis=-1))
            h, w = buf8.shape
            self._preview_qimg = QImage(buf8.data, w, h, w,
                                        QImage.Format.Format_Grayscale8)
        else:
            buf8 = np.ascontiguousarray((disp * 255.0).astype(np.uint8))
            self._last_preview = buf8
            h, w, _ = buf8.shape
            self._preview_qimg = QImage(buf8.data, w, h, buf8.strides[0],
                                        QImage.Format.Format_RGB888)

        self._update_preview_scaled()
        self._redraw_overlay()

    def _do_preview(self):
        try:
            from PyQt6 import sip
            if sip.isdeleted(self):
                return
            self._set_status("Building exclusion mask…")
            excl = self._build_exclusion_mask()

            self._set_status("Running ABE preview…")
            corrected, bg = self._run_abe(excl, progress=self._set_status)
            if corrected is None:
                QMessageBox.information(self, "No image", "No image is loaded in the active document.")
                self._set_status("Ready")
                return

            show = bg if self.chk_preview_bg.isChecked() else corrected

            # ✅ If previewing the corrected image, honor the active mask
            if not self.chk_preview_bg.isChecked():
                srcf = self._get_source_float()
                show = self._blend_with_mask_float(show, srcf)

            self._set_status("Rendering preview…")
            self._set_preview_pixmap(show)
            self._set_status("Ready")
        except Exception as e:
            self._set_status("Error")
            QMessageBox.warning(self, "Preview failed", str(e))

    def _do_apply(self):
        try:
            self._set_status("Building exclusion mask…")
            excl = self._build_exclusion_mask()

            self._set_status("Running ABE (apply)…")
            corrected, bg = self._run_abe(excl, progress=self._set_status)
            if corrected is None:
                QMessageBox.information(self, "No image", "No image is loaded in the active document.")
                self._set_status("Ready")
                return

            # Preserve mono vs color shape w.r.t. source
            out = corrected
            if out.ndim == 3 and out.shape[2] == 3 and (self.doc.image.ndim == 2 or (self.doc.image.ndim == 3 and self.doc.image.shape[2] == 1)):
                out = out[..., 0]

            # ✅ Blend with active mask before committing
            srcf = self._get_source_float()
            out_masked = self._blend_with_mask_float(out, srcf)

            # Build step name for undo stack
            # Build step name + params for undo stack + Replay
            deg   = int(self.sp_degree.value())
            npts  = int(self.sp_samples.value())
            dwn   = int(self.sp_down.value())
            patch = int(self.sp_patch.value())
            use_rbf = bool(self.chk_use_rbf.isChecked())
            rbf_smooth = float(self.sp_rbf.value()) * 0.01
            make_bg_doc = bool(self.chk_make_bg_doc.isChecked())

            step_name = (
                f"ABE (deg={deg}, samples={npts}, ds={dwn}, patch={patch}, "
                f"rbf={'on' if use_rbf else 'off'}, s={rbf_smooth:.3f})"
            )

            # Normalized preset params (same schema as abe_preset.apply_abe_via_preset)
            params = {
                "degree": deg,
                "samples": npts,
                "downsample": dwn,
                "patch": patch,
                "rbf": use_rbf,
                "rbf_smooth": rbf_smooth,
                "make_background_doc": make_bg_doc,
            }
            params["sample_mode"] = self.cmb_sample_mode.currentText().lower()
            params["manual_sample_count"] = len(self._manual_points)

            # 🔁 Remember this as the last headless-style command for Replay
            mw = self.parent()
            try:
                remember = getattr(mw, "remember_last_headless_command", None)
                if remember is None:
                    remember = getattr(mw, "_remember_last_headless_command", None)
                if callable(remember):
                    remember("abe", params, description="Automatic Background Extraction")
                    try:
                        if hasattr(mw, "_log"):
                            mw._log(
                                f"[Replay] ABE UI apply stored: "
                                f"command_id='abe', preset_keys={list(params.keys())}"
                            )
                    except Exception:
                        pass
            except Exception:
                # don’t block the actual ABE apply if remembering fails
                pass

            # ✅ mask bookkeeping in metadata
            _marr, mid, mname = self._active_mask_layer()
            abe_meta = dict(params)
            abe_meta["exclusion"] = "polygons" if excl is not None else "none"
            abe_meta["manual_sample_count"] = len(self._manual_points)

            meta = {
                "step_name": "ABE",
                "abe": abe_meta,
                "masked": bool(mid),
                "mask_id": mid,
                "mask_name": mname,
                "mask_blend": "m*out + (1-m)*src",
            }

            self._set_status("Committing edit…")
            self.doc.apply_edit(
                out_masked.astype(np.float32, copy=False),
                step_name=step_name,
                metadata=meta,
            )


            if self.chk_make_bg_doc.isChecked() and bg is not None:
                self._set_status("Creating background document…")
                mw = self.parent()
                dm = getattr(mw, "docman", None)
                if dm is not None:
                    base = os.path.splitext(self.doc.display_name())[0]
                    meta = {
                        "bit_depth": "32-bit floating point",
                        "is_mono": (bg.ndim == 2),
                        "source": "ABE background",
                        "original_header": self.doc.metadata.get("original_header"),
                    }
                    doc_bg = dm.open_array(bg.astype(np.float32, copy=False), metadata=meta, title=f"{base}_ABE_BG")
                    if hasattr(mw, "_spawn_subwindow_for"):
                        mw._spawn_subwindow_for(doc_bg)

            # Preserve the current view's autostretch state: capture before/restore after
            mw = self.parent()
            prev_autostretch = False
            view = None
            try:
                if hasattr(mw, "mdi") and mw.mdi.activeSubWindow():
                    view = mw.mdi.activeSubWindow().widget()
                    prev_autostretch = bool(getattr(view, "autostretch_enabled", False))
            except Exception:
                prev_autostretch = False


            if hasattr(mw, "_log"):
                mw._log(step_name)

            # Restore autostretch state on the view (recompute display) so the
            # user's display-stretch choice survives the edit.
            try:
                if view is not None and hasattr(view, "set_autostretch") and callable(view.set_autostretch):
                    view.set_autostretch(prev_autostretch)
            except Exception:
                pass

            self._set_status("Done")
            # Dialog stays open so user can apply to other images
            # Refresh to use the now-active document for next operation
            self.close()
            return

        except Exception as e:
            self._set_status("Error")
            QMessageBox.critical(self, "Apply failed", str(e))

    def closeEvent(self, ev):
        self._closing = True
        try:
            if self._connected_current_doc_changed and hasattr(self._main, "currentDocumentChanged"):
                self._main.currentDocumentChanged.disconnect(self._on_active_doc_changed)
        except Exception:
            pass
        self._connected_current_doc_changed = False
        try:
            self._save_window_geometry()
        except Exception:
            pass
        try:
            if getattr(self, "_worker", None) is not None:
                try:
                    self._worker.requestInterruption()
                except Exception:
                    pass
            if getattr(self, "_thread", None) is not None:
                self._thread.quit()
                self._thread.wait(500)
        except Exception:
            pass
        super().closeEvent(ev)

    def _refresh_document_from_active(self):
        """
        Refresh the dialog's document reference to the currently active document.
        This allows reusing the same dialog on different images.
        """
        try:
            main = self.parent()
            if main and hasattr(main, "_active_doc"):
                new_doc = main._active_doc()
                if new_doc is not None and new_doc is not self.doc:
                    self.doc = new_doc
                    # Reset preview state for new document
                    self._preview_source_f01 = None
                    self._last_preview = None
                    self._preview_qimg = None
                    # Clear polygons since they were for old image
                    self._clear_polys()
                    self._manual_points.clear()
        except Exception:
            pass


    # ----- exclusion polygons & mask -----
    def _clear_polys(self):
        self._polygons.clear()
        self._drawing_poly = None
        # ✅ redraw from the clean base
        self._redraw_overlay()

    def _image_shape(self) -> tuple[int, int]:
        src = np.asarray(self.doc.image)
        if src.ndim == 2:
            return src.shape[0], src.shape[1]
        return src.shape[0], src.shape[1]

    def _build_exclusion_mask(self) -> np.ndarray | None:
        if not self._polygons:
            return None
        H, W = self._image_shape()
        mask = np.ones((H, W), dtype=np.uint8)
        if cv2 is None:
            # very slow pure-numpy fallback: fill polygon by bounding-box rasterization
            # (expect OpenCV to be available in SASpro)
            for poly in self._polygons:
                pts = np.array([[int(p.x()), int(p.y())] for p in poly], dtype=np.int32)
                minx, maxx = np.clip([pts[:,0].min(), pts[:,0].max()], 0, W-1)
                miny, maxy = np.clip([pts[:,1].min(), pts[:,1].max()], 0, H-1)
                for y in range(miny, maxy+1):
                    for x in range(minx, maxx+1):
                        # winding test approx omitted -> treat as box (coarse)
                        mask[y, x] = 0
        else:
            polys = [np.array([[int(p.x()), int(p.y())] for p in poly], dtype=np.int32) for poly in self._polygons]
            cv2.fillPoly(mask, polys, 0)  # 0 = excluded
        return mask.astype(bool)

    # ----- preview rendering helpers -----

    def _set_preview_pixmap(self, arr: np.ndarray):
        if arr is None or arr.size == 0:
            self.preview_label.clear(); self._overlay = None; self._preview_source_f01 = None
            return

        # keep the float source for autostretch toggling (no re-normalization)
        a = _asfloat32(arr)
        self._preview_source_f01 = a  # ← no np.clip here

        # show autostretched or raw; mtf_style_autostretch() already clips its result
        src_to_show = (hard_autostretch(self._preview_source_f01, target_median=0.5, sigma=2,
                                        linked=False, use_24bit=True)
                    if getattr(self, "_autostretch_on", False) else self._preview_source_f01)

        if src_to_show.ndim == 2 or (src_to_show.ndim == 3 and src_to_show.shape[2] == 1):
            # MONO path — match Crop: use Grayscale8 QImage; keep 3-ch backing for rebuild
            mono = src_to_show if src_to_show.ndim == 2 else src_to_show[..., 0]
            buf8_mono = (mono * 255.0).astype(np.uint8)               # ← no np.clip here
            buf8_mono = np.ascontiguousarray(buf8_mono)
            h, w = buf8_mono.shape

            # for the toggle/rebuild code which expects 3-ch bytes
            self._last_preview = np.ascontiguousarray(np.stack([buf8_mono]*3, axis=-1))

            qimg = QImage(buf8_mono.data, w, h, w, QImage.Format.Format_Grayscale8)
        else:
            # RGB path
            buf8 = (src_to_show * 255.0).astype(np.uint8)             # ← no np.clip here
            buf8 = np.ascontiguousarray(buf8)
            h, w, _ = buf8.shape
            self._last_preview = buf8
            qimg = QImage(buf8.data, w, h, buf8.strides[0], QImage.Format.Format_RGB888)

        self._preview_qimg = qimg
        self._update_preview_scaled()
        self._redraw_overlay()



    def _update_preview_scaled(self):
        try:
            if sip.isdeleted(self) or sip.isdeleted(self.preview_label):
                return
        except Exception:
            return

        if self._preview_qimg is None:
            self.preview_label.clear()
            return

        sw = max(1, int(self._preview_qimg.width()  * self._preview_scale))
        sh = max(1, int(self._preview_qimg.height() * self._preview_scale))

        scaled = self._preview_qimg.scaled(
            sw, sh,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

        if scaled.isNull():
            return

        self._base_pixmap = QPixmap.fromImage(scaled)
        if self._base_pixmap.isNull():
            return

        try:
            if not sip.isdeleted(self.preview_label):
                self.preview_label.setPixmap(self._base_pixmap)
                self.preview_label.resize(self._base_pixmap.size())
        except Exception:
            pass

    def _redraw_overlay(self):
        # Guard against being called during/after close
        try:
            if sip.isdeleted(self) or sip.isdeleted(self.preview_label):
                return
        except Exception:
            return

        pm_base = self._base_pixmap or self.preview_label.pixmap()
        if pm_base is None or pm_base.isNull():
            return

        # start from a fresh copy of the clean base
        composed = QPixmap(pm_base)
        if composed.isNull():
            return

        overlay = QPixmap(pm_base.size())
        if overlay.isNull():
            return
        overlay.fill(Qt.GlobalColor.transparent)

        painter = QPainter(overlay)
        if not painter.isActive():
            return  # paint device invalid — widget being destroyed

        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # map image-space polys to label-space
        img_w = self._preview_qimg.width() if self._preview_qimg else 1
        img_h = self._preview_qimg.height() if self._preview_qimg else 1
        lab_w = self.preview_label.width()
        lab_h = self.preview_label.height()
        sx = lab_w / img_w
        sy = lab_h / img_h

        # finalized polygons (green, semi-transparent)
        pen = QPen(QColor(0, 255, 0), 2)
        brush = QColor(0, 255, 0, 60)
        painter.setPen(pen)
        painter.setBrush(brush)
        for poly in self._polygons:
            if len(poly) >= 3:
                mapped = [QPointF(p.x() * sx, p.y() * sy) for p in poly]
                painter.drawPolygon(*mapped)

        # in-progress poly (red dashed)
        if self._drawing_poly and len(self._drawing_poly) >= 2:
            pen2 = QPen(QColor(255, 0, 0), 2, Qt.PenStyle.DashLine)
            painter.setPen(pen2)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            mapped = [QPointF(p.x() * sx, p.y() * sy) for p in self._drawing_poly]
            painter.drawPolyline(*mapped)

        # manual sample squares (yellow outlines)
        if self._manual_points:
            patch = int(max(1, self.sp_patch.value()))
            hx = 0.5 * patch * sx
            hy = 0.5 * patch * sy

            pen3 = QPen(QColor(255, 220, 0), 2)
            painter.setPen(pen3)
            painter.setBrush(Qt.BrushStyle.NoBrush)

            for p in self._manual_points:
                cx = p.x() * sx
                cy = p.y() * sy
                painter.drawRect(
                    int(round(cx - hx)),
                    int(round(cy - hy)),
                    int(round(2 * hx)),
                    int(round(2 * hy)),
                )


        painter.end()

        p = QPainter(composed)
        if not p.isActive():
            return
        p.drawPixmap(0, 0, overlay)
        p.end()

        try:
            if not sip.isdeleted(self.preview_label):
                self.preview_label.setPixmap(composed)
        except Exception:
            pass

    # ----- zoom/pan + polygon drawing -----
    def eventFilter(self, obj, ev):
        # ---- Mouse wheel zoom handling (Qt6-friendly) ----
        if ev.type() == QEvent.Type.Wheel and (
            obj is self.preview_label
            or obj is self.preview_scroll
            or obj is self.preview_scroll.viewport()
            or obj is self.preview_scroll.horizontalScrollBar()
            or obj is self.preview_scroll.verticalScrollBar()
        ):
            # Always stop the wheel from scrolling the scrollbars/scroll area.
            ev.accept()

            # Anchor at the mouse position in the viewport
            # (even if the event came from a scrollbar)
            vp = self.preview_scroll.viewport()
            anchor_vp = vp.mapFromGlobal(ev.globalPosition().toPoint())

            # Clamp to viewport rect (robust if the event originated on scrollbars)
            r = vp.rect()
            if not r.contains(anchor_vp):
                anchor_vp.setX(max(r.left(),  min(r.right(),  anchor_vp.x())))
                anchor_vp.setY(max(r.top(),   min(r.bottom(), anchor_vp.y())))

            # Smooth trackpad support first, then mouse wheel fallback
            dy = ev.pixelDelta().y()
            if dy != 0:
                abs_dy = abs(dy)
                ctrl_down = bool(ev.modifiers() & Qt.KeyboardModifier.ControlModifier)

                if abs_dy <= 3:
                    base_factor = 1.012 if ctrl_down else 1.010
                elif abs_dy <= 10:
                    base_factor = 1.025 if ctrl_down else 1.020
                else:
                    base_factor = 1.040 if ctrl_down else 1.030

                factor = base_factor if dy > 0 else 1.0 / base_factor
            else:
                dy = ev.angleDelta().y()
                if dy == 0:
                    return True

                ctrl_down = bool(ev.modifiers() & Qt.KeyboardModifier.ControlModifier)
                step = 1.25 if ctrl_down else 1.15
                factor = step if dy > 0 else 1.0 / step

            self._zoom_at(factor, anchor_vp)
            return True

        # ---- Existing polygon drawing on the label ----
        if obj is self.preview_label:
            if ev.type() == QEvent.Type.MouseButtonPress:
                # panning still works the same in either mode
                if ev.buttons() & Qt.MouseButton.MiddleButton or (
                    ev.buttons() & Qt.MouseButton.LeftButton and (ev.modifiers() & Qt.KeyboardModifier.ControlModifier)
                ):
                    self._panning = True
                    self._pan_last = ev.position().toPoint()
                    self.preview_label.setCursor(Qt.CursorShape.ClosedHandCursor)
                    return True

                # manual sample-point mode
                if self._manual_mode():
                    if ev.button() == Qt.MouseButton.LeftButton:
                        img_pt = self._label_to_image_coords(ev.position())
                        if img_pt is not None:
                            self._manual_points.append(img_pt)
                            self.btn_clear_samples.setEnabled(True)
                            self._redraw_overlay()
                            return True

                    if ev.button() == Qt.MouseButton.RightButton:
                        img_pt = self._label_to_image_coords(ev.position())
                        if img_pt is not None:
                            idx = self._find_nearest_manual_point_index(img_pt)
                            if idx >= 0:
                                self._manual_points.pop(idx)
                                self.btn_clear_samples.setEnabled(
                                    self._manual_mode() and len(self._manual_points) > 0
                                )
                                self._redraw_overlay()
                            return True

                # existing polygon mode
                if ev.buttons() & Qt.MouseButton.RightButton:
                    if self._drawing_poly and len(self._drawing_poly) >= 3:
                        self._polygons.append(self._drawing_poly)
                    self._drawing_poly = None
                    self._redraw_overlay()
                    return True

                if ev.buttons() & Qt.MouseButton.LeftButton:
                    img_pt = self._label_to_image_coords(ev.position())
                    if img_pt is not None:
                        if self._drawing_poly is None:
                            self._drawing_poly = [img_pt]
                        else:
                            self._drawing_poly.append(img_pt)
                        self._redraw_overlay()
                        return True

            elif ev.type() == QEvent.Type.MouseMove:
                if getattr(self, "_panning", False):
                    pos = ev.position().toPoint()
                    delta = pos - (self._pan_last or pos)
                    self._pan_last = pos
                    hsb = self.preview_scroll.horizontalScrollBar()
                    vsb = self.preview_scroll.verticalScrollBar()
                    hsb.setValue(hsb.value() - delta.x())
                    vsb.setValue(vsb.value() - delta.y())
                    return True
                if (not self._manual_mode()) and self._drawing_poly is not None and (ev.buttons() & Qt.MouseButton.LeftButton):
                    img_pt = self._label_to_image_coords(ev.position())
                    if img_pt is not None:
                        self._drawing_poly.append(img_pt)
                        self._redraw_overlay()
                        return True

            elif ev.type() == QEvent.Type.MouseButtonRelease:
                # finish panning
                if getattr(self, "_panning", False):
                    self._panning = False
                    self._pan_last = None
                    self.preview_label.unsetCursor()
                    return True

                # Close polygon on LEFT mouse release
                if (not self._manual_mode()) and ev.button() == Qt.MouseButton.LeftButton and self._drawing_poly is not None:
                    if len(self._drawing_poly) >= 3:
                        self._polygons.append(self._drawing_poly)
                    self._drawing_poly = None
                    self._redraw_overlay()
                    return True

        return super().eventFilter(obj, ev)

    def _ensure_scale_state(self):
        # internal guard so _zoom_at can be called even if _scale hasn't been set
        if not hasattr(self, "_scale"):
            self._scale = float(self.view.transform().m11()) if not self.view.transform().isIdentity() else 1.0

    def _zoom_at(self, factor: float, anchor_vp) -> None:
        """
        Zoom the preview by 'factor', keeping the content point under 'anchor_vp'
        (a QPoint in viewport coordinates) stationary.
        """
        old_scale = float(self._preview_scale)
        new_scale = max(0.05, min(old_scale * factor, 8.0))
        if abs(new_scale - old_scale) < 1e-6:
            return
        factor = new_scale / old_scale

        # content coordinates (relative to the QLabel) under the cursor BEFORE scaling
        hsb = self.preview_scroll.horizontalScrollBar()
        vsb = self.preview_scroll.verticalScrollBar()
        old_x = hsb.value() + anchor_vp.x()
        old_y = vsb.value() + anchor_vp.y()

        # apply scale
        self._preview_scale = new_scale
        self._update_preview_scaled()
        self._redraw_overlay()

        # desired scroll so the same content point stays under the cursor
        new_x = int(old_x * factor - anchor_vp.x())
        new_y = int(old_y * factor - anchor_vp.y())

        # clamp to valid range
        hsb.setValue(max(hsb.minimum(), min(new_x, hsb.maximum())))
        vsb.setValue(max(vsb.minimum(), min(new_y, vsb.maximum())))


    def zoom_in(self) -> None:
        vp = self.preview_scroll.viewport()
        self._zoom_at(1.25, vp.rect().center())

    def zoom_out(self) -> None:
        vp = self.preview_scroll.viewport()
        self._zoom_at(0.8, vp.rect().center())

    def fit_to_preview(self) -> None:
        """Set scale so the image fits inside the viewport (keeps aspect)."""
        if self._preview_qimg is None:
            return
        vp = self.preview_scroll.viewport()
        vw, vh = max(1, vp.width()), max(1, vp.height())
        iw, ih = self._preview_qimg.width(), self._preview_qimg.height()
        if iw == 0 or ih == 0:
            return
        scale = min(vw / iw, vh / ih)
        self._preview_scale = max(0.05, min(scale, 8.0))
        self._update_preview_scaled()
        self._redraw_overlay()

        # center after fit
        hsb = self.preview_scroll.horizontalScrollBar()
        vsb = self.preview_scroll.verticalScrollBar()
        hsb.setValue((hsb.maximum() - hsb.minimum()) // 2)
        vsb.setValue((vsb.maximum() - vsb.minimum()) // 2)



    def _label_to_image_coords(self, posf) -> QPointF | None:
        if self._preview_qimg is None:
            return None
        img_w = self._preview_qimg.width(); img_h = self._preview_qimg.height()
        lab_w = self.preview_label.width(); lab_h = self.preview_label.height()
        sx = img_w / max(1.0, lab_w); sy = img_h / max(1.0, lab_h)
        x_img = float(posf.x()) * sx; y_img = float(posf.y()) * sy
        # clamp to image
        x_img = max(0.0, min(x_img, img_w - 1.0))
        y_img = max(0.0, min(y_img, img_h - 1.0))
        return QPointF(x_img, y_img)

    def _install_zoom_filters(self):
        """Install event filters so Ctrl+Wheel works even when the cursor is over scrollbars."""
        self.preview_scroll.installEventFilter(self)
        self.preview_scroll.viewport().installEventFilter(self)
        self.preview_scroll.horizontalScrollBar().installEventFilter(self)
        self.preview_scroll.verticalScrollBar().installEventFilter(self)
        self.preview_label.installEventFilter(self)
        
    def _set_preview_from_float(self, arr: np.ndarray):
        if arr is None or arr.size == 0:
            return
        a = _asfloat32(arr)
        self._preview_source_f01 = a  # ← no np.clip

        src_to_show = (hard_autostretch(self._preview_source_f01, target_median=0.5, sigma=2,
                                        linked=False, use_24bit=True)
                    if getattr(self, "_autostretch_on", False) else self._preview_source_f01)

        if src_to_show.ndim == 2 or (src_to_show.ndim == 3 and src_to_show.shape[2] == 1):
            mono = src_to_show if src_to_show.ndim == 2 else src_to_show[..., 0]
            buf8_mono = (mono * 255.0).astype(np.uint8)               # ← no np.clip
            buf8_mono = np.ascontiguousarray(buf8_mono)
            self._last_preview = np.ascontiguousarray(np.stack([buf8_mono]*3, axis=-1))
            h, w = buf8_mono.shape
            qimg = QImage(buf8_mono.data, w, h, w, QImage.Format.Format_Grayscale8)
        else:
            buf8 = (src_to_show * 255.0).astype(np.uint8)             # ← no np.clip
            buf8 = np.ascontiguousarray(buf8)
            self._last_preview = buf8
            h, w, _ = buf8.shape
            qimg = QImage(buf8.data, w, h, buf8.strides[0], QImage.Format.Format_RGB888)

        self._preview_qimg = qimg
        self._update_preview_scaled()
        self._redraw_overlay()

    # --- mask helpers ---------------------------------------------------
    def _active_mask_layer(self):
        """Return (mask_float01, mask_id, mask_name) or (None, None, None)."""
        mid = getattr(self.doc, "active_mask_id", None)
        if not mid: return None, None, None
        layer = getattr(self.doc, "masks", {}).get(mid)
        if layer is None: return None, None, None
        m = np.asarray(getattr(layer, "data", None))
        if m is None or m.size == 0: return None, None, None
        m = m.astype(np.float32, copy=False)
        if m.dtype.kind in "ui":
            m /= float(np.iinfo(m.dtype).max)
        else:
            mx = float(m.max()) if m.size else 1.0
            if mx > 1.0: m /= mx
        return np.clip(m, 0.0, 1.0), mid, getattr(layer, "name", "Mask")

    def _resample_mask_if_needed(self, mask: np.ndarray, out_hw: tuple[int,int]) -> np.ndarray:
        """Nearest-neighbor resize via integer indexing."""
        mh, mw = mask.shape[:2]
        th, tw = out_hw
        if (mh, mw) == (th, tw): return mask
        yi = np.linspace(0, mh - 1, th).astype(np.int32)
        xi = np.linspace(0, mw - 1, tw).astype(np.int32)
        return mask[yi][:, xi]

    def _blend_with_mask_float(self, processed: np.ndarray, src: np.ndarray | None = None) -> np.ndarray:
        """
        m*out + (1-m)*src in float [0..1], mono or RGB.
        If src is None, uses the current document image (float [0..1]).
        """
        mask, _mid, _mname = self._active_mask_layer()
        if mask is None:
            return processed

        out = processed.astype(np.float32, copy=False)
        if src is None:
            src = self._get_source_float()
        else:
            src = src.astype(np.float32, copy=False)

        # match HxW
        m = self._resample_mask_if_needed(mask, out.shape[:2])

        # channel reconcile
        if out.ndim == 2 and src.ndim == 3:
            out = out[..., None]
        if src.ndim == 2 and out.ndim == 3:
            src = src[..., None]

        if out.ndim == 3 and out.shape[2] == 3 and m.ndim == 2:
            m = m[..., None]

        blended = (m * out + (1.0 - m) * src).astype(np.float32, copy=False)
        # squeeze back to mono if we expanded
        if blended.ndim == 3 and blended.shape[2] == 1:
            blended = blended[..., 0]
        return np.clip(blended, 0.0, 1.0)

    def autostretch_preview(self, checked: bool | None = None) -> None:
        """
        Toggle preview autostretch on/off and rebuild the viewport from the
        underlying float preview source every time.
        """
        if self._preview_source_f01 is None:
            return

        current = bool(getattr(self, "_autostretch_on", False))
        new_state = (not current) if checked is None else bool(checked)
        self._autostretch_on = new_state

        src = self._preview_source_f01

        if new_state:
            disp = hard_autostretch(
                src,
                target_median=0.5,
                sigma=2,
                linked=False,
                use_24bit=True
            )
        else:
            disp = src

        disp = np.asarray(disp, dtype=np.float32)

        if disp.ndim == 2 or (disp.ndim == 3 and disp.shape[2] == 1):
            mono = disp if disp.ndim == 2 else disp[..., 0]
            buf8_mono = (np.clip(mono, 0.0, 1.0) * 255.0).astype(np.uint8)
            buf8_mono = np.ascontiguousarray(buf8_mono)

            # keep 3-channel backing array for consistency with other preview code
            self._last_preview = np.ascontiguousarray(np.stack([buf8_mono] * 3, axis=-1))

            h, w = buf8_mono.shape
            self._preview_qimg = QImage(
                buf8_mono.data,
                w, h,
                w,
                QImage.Format.Format_Grayscale8
            )
        else:
            buf8 = (np.clip(disp, 0.0, 1.0) * 255.0).astype(np.uint8)
            buf8 = np.ascontiguousarray(buf8)
            self._last_preview = buf8

            h, w, _ = buf8.shape
            self._preview_qimg = QImage(
                buf8.data,
                w, h,
                buf8.strides[0],
                QImage.Format.Format_RGB888
            )

        self._update_preview_scaled()
        self._redraw_overlay()

        if hasattr(self, "btn_autostr"):
            self.btn_autostr.setChecked(new_state)
            self.btn_autostr.setToolTip(
                "Autostretch preview (On)" if new_state else "Autostretch preview"
            )

        self._save_settings()

    def _apply_autostretch_inplace(self, sigma: float = 3.0):
        self.autostretch_preview(True)

    def _restore_window_geometry(self):
        try:
            s = QSettings()
            g = s.value("abe_ui/window_geometry", None)   # ✅ unique, NOT stat_stretch
            if g is not None:
                self.restoreGeometry(g)
        except Exception:
            pass

    def _save_window_geometry(self):
        try:
            s = QSettings()
            s.setValue("abe_ui/window_geometry", self.saveGeometry())
        except Exception:
            pass        

    def showEvent(self, ev):
        super().showEvent(ev)

        if not getattr(self, "_geom_restored", False):
            self._geom_restored = True

            def _after_show():
                # 1) Restore geometry first so viewport size is correct
                self._restore_window_geometry()
                QApplication.processEvents()

                # 2) Show placeholder so user sees something while we compute
                self._show_placeholder()

                # 3) Render with stretch if enabled — this is the expensive step
                #    but now it happens with correct viewport size and after show
                self._render_preview_from_source(
                    stretch=bool(getattr(self, "_autostretch_on", False))
                )

                # 4) Fit to the now-correct viewport
                self.fit_to_preview()

            QTimer.singleShot(0, _after_show)
            return

        # Already restored — just refit if shown again
        QTimer.singleShot(0, self.fit_to_preview)