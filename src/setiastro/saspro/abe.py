# pro/abe.py — SASpro Automatic Background Extraction (ABE)
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

from PyQt6.QtCore import Qt, QSize, QEvent, QPointF, QTimer, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QSpinBox,
    QCheckBox, QPushButton, QScrollArea, QWidget, QMessageBox, QComboBox,
    QGroupBox, QApplication
)
from PyQt6.QtGui import QImage, QPixmap, QPainter, QColor, QPen
from PyQt6 import sip

from scipy.interpolate import Rbf

from .doc_manager import ImageDocument
from setiastro.saspro.legacy.numba_utils import build_poly_terms, evaluate_polynomial
from .autostretch import autostretch as hard_autostretch
from setiastro.saspro.widgets.themed_buttons import themed_toolbtn

# =============================================================================
#                         Headless ABE Core (poly + RBF)
# =============================================================================

def _asfloat32(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x)
    return a if a.dtype == np.float32 else a.astype(np.float32, copy=False)

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
    """Match SASv2 exactly: float64 for RBF inputs, multiquadric, epsilon=1.0."""
    H, W = small.shape[:2]
    half = patch_size // 2
    pts = np.asarray(points, dtype=np.int32)
    xs = np.clip(pts[:, 0], 0, W - 1).astype(np.int64)
    ys = np.clip(pts[:, 1], 0, H - 1).astype(np.int64)

    # Evaluate on a float64 meshgrid (same as SASv2)
    grid_x, grid_y = np.meshgrid(
        np.arange(W, dtype=np.float64),
        np.arange(H, dtype=np.float64),
    )

    def _median_patch(arr, x, y):
        x0, x1 = max(0, x - half), min(W, x + half + 1)
        y0, y1 = max(0, y - half), min(H, y + half + 1)
        return float(np.median(arr[y0:y1, x0:x1]))

    if small.ndim == 3 and small.shape[2] == 3:
        bg_small = np.zeros((H, W, 3), dtype=np.float32)
        for c in range(3):
            z = np.array([_median_patch(small[..., c], int(x), int(y)) for x, y in zip(xs, ys)], dtype=np.float64)
            rbf = Rbf(xs.astype(np.float64), ys.astype(np.float64), z,
                      function='multiquadric', smooth=float(smooth), epsilon=1.0)
            bg_small[..., c] = rbf(grid_x, grid_y).astype(np.float32)
        return bg_small
    else:
        z = np.array([_median_patch(small, int(x), int(y)) for x, y in zip(xs, ys)], dtype=np.float64)
        rbf = Rbf(xs.astype(np.float64), ys.astype(np.float64), z,
                  function='multiquadric', smooth=float(smooth), epsilon=1.0)
        return rbf(grid_x, grid_y).astype(np.float32)

def _legacy_stretch_unlinked(image: np.ndarray):
    """
    SASv2 stretch domain used for modeling: per-channel min shift + unlinked rational
    stretch to target median=0.25. Returns (stretched_rgb, state_dict).
    """
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

    out = np.clip(out, 0.0, 1.0)
    return out, {"mins": ch_mins, "meds": ch_meds, "was_single": was_single}


def _legacy_unstretch_unlinked(image: np.ndarray, state: dict):
    """
    Inverse of the SASv2 stretch above. Accepts mono or RGB; returns same ndim
    as input, except if original was single-channel it returns mono.
    """
    mins = state["mins"]; meds = state["meds"]; was_single = state["was_single"]
    img = image.astype(np.float32, copy=True)

    # Work as RGB internally
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

    img = np.clip(img, 0.0, 1.0)
    if was_single:
        # original was mono → return mono
        return img[..., 0]
    return img


def abe_run(
    image: np.ndarray,
    degree: int = 2,             # 0..6 (0 = skip polynomial)
    num_samples: int = 100,
    downsample: int = 4,
    patch_size: int = 15,
    use_rbf: bool = True,
    rbf_smooth: float = 0.1,      # numeric; UI can map 10 -> 0.10, 100 -> 1.0, etc.
    exclusion_mask: np.ndarray | None = None,
    return_background: bool = True,
    progress_cb=None,
    legacy_prestretch: bool = True,   # <-- SASv2 parity switch
) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
    """Two-stage ABE (poly + optional RBF) with SASv2-compatible pre/post stretch."""
    if image is None:
        raise ValueError("ABE: image is None")

    img_src = np.asarray(image).astype(np.float32, copy=False)
    mono = (img_src.ndim == 2) or (img_src.ndim == 3 and img_src.shape[2] == 1)

    # Work in RGB internally (even for mono) so pre/post stretch matches SASv2 behavior
    img_rgb = img_src if (img_src.ndim == 3 and img_src.shape[2] == 3) else np.stack(
        [img_src.squeeze()] * 3, axis=-1
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

    # ---------- Polynomial stage (skip when degree == 0) ----------
    if degree <= 0:
        if progress_cb: progress_cb("Degree 0: skipping polynomial stage…")
        after_poly = img_rgb.copy()                         # nothing removed yet
        total_bg   = np.zeros_like(img_rgb, dtype=np.float32)
    else:
        if progress_cb: progress_cb("Sampling points (poly stage)…")
        pts = _generate_sample_points(small, num_points=num_samples,
                                      exclusion_mask=mask_small, patch_size=patch_size)

        if progress_cb: progress_cb(f"Fitting polynomial (degree {degree})…")
        bg_poly_small = _fit_poly_on_small(small, pts, degree=degree, patch_size=patch_size)

        if progress_cb: progress_cb("Upscaling polynomial background…")
        bg_poly = _upscale_bg(bg_poly_small, img_rgb.shape[:2])

        if progress_cb: progress_cb("Subtracting polynomial background & re-centering…")
        after_poly = img_rgb - bg_poly
        med_after  = float(np.median(after_poly))
        after_poly = np.clip(after_poly + (orig_med - med_after), 0.0, 1.0)

        total_bg = bg_poly.astype(np.float32, copy=False)

    # ---------- RBF refinement --------------------------------------------
    if use_rbf:
        if progress_cb: progress_cb("Downsampling for RBF stage…")
        small_rbf = _downsample_area(after_poly, downsample)

        if progress_cb: progress_cb("Sampling points (RBF stage)…")
        pts_rbf = _generate_sample_points(small_rbf, num_points=num_samples,
                                          exclusion_mask=mask_small, patch_size=patch_size)

        if progress_cb: progress_cb(f"Fitting RBF (smooth={rbf_smooth:.3f})…")
        bg_rbf_small = _fit_rbf_on_small(small_rbf, pts_rbf, smooth=rbf_smooth, patch_size=patch_size)

        if progress_cb: progress_cb("Upscaling RBF background…")
        bg_rbf = _upscale_bg(bg_rbf_small, img_rgb.shape[:2])

        if progress_cb: progress_cb("Combining backgrounds & finalizing…")
        total_bg = (total_bg + bg_rbf).astype(np.float32)
        corrected = img_rgb - total_bg
        med2 = float(np.median(corrected))
        corrected = np.clip(corrected + (orig_med - med2), 0.0, 1.0)
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

    if progress_cb: progress_cb("Ready")
    if return_background:
        return corrected.astype(np.float32, copy=False), total_bg.astype(np.float32, copy=False)
    return corrected.astype(np.float32, copy=False)



def siril_style_autostretch(image: np.ndarray, sigma: float = 3.0) -> np.ndarray:
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





# =============================================================================
#                                   UI Dialog
# =============================================================================

class ABEWorker(QThread):
    """
    Worker thread for running ABE in the background.
    """
    finished = pyqtSignal(object, object)  # corrected, bg
    progress = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, imgf, deg, npts, dwn, patch, use_rbf, rbf_smooth, excl_mask):
        super().__init__()
        self.imgf = imgf
        self.deg = deg
        self.npts = npts
        self.dwn = dwn
        self.patch = patch
        self.use_rbf = use_rbf
        self.rbf_smooth = rbf_smooth
        self.excl_mask = excl_mask
    
    def run(self):
        try:
            # We must not touch GUI here. abe_run is pure numpy/math.
            corrected, bg = abe_run(
                self.imgf,
                degree=self.deg, num_samples=self.npts, downsample=self.dwn, patch_size=self.patch,
                use_rbf=self.use_rbf, rbf_smooth=self.rbf_smooth,
                exclusion_mask=self.excl_mask, return_background=True,
                progress_cb=self.progress.emit # Proxy progress callbacks to signal
            )
            self.finished.emit(corrected, bg)
        except Exception as e:
            self.error.emit(str(e))

class ABEDialog(QDialog):
    def __init__(self, main_window, doc: ImageDocument):
        super().__init__(main_window)
        self._main = main_window
        self.doc = doc
        self.setWindowTitle("Automatic Background Extraction (ABE)")
        self.resize(1000, 700)
        
        # State
        self._polygons = []
        self._drawing_poly = None
        self._preview_scale = 1.0
        self._panning = False
        self._pan_last = None
        
        self._preview_qimg: QImage | None = None
        self._base_pixmap: QPixmap | None = None
        self._preview_source_f01: np.ndarray | None = None
        self._last_preview: np.ndarray | None = None
        
        # Worker
        self._worker = None

        # UI Setup
        self._setup_ui()

        # Initial data
        self._populate_initial_preview()
        
        # Connections
        self._connected_current_doc_changed = True
        if hasattr(self._main, "currentDocumentChanged"):
            self._main.currentDocumentChanged.connect(self._on_active_doc_changed)

    def _setup_ui(self):
        main_layout = QHBoxLayout(self)
        
        # --- Left: Controls ---
        ctrl_panel = QWidget()
        ctrl_layout = QVBoxLayout(ctrl_panel)
        ctrl_panel.setFixedWidth(300)
        
        # Group: Model Parameters
        grp_param = QGroupBox("Model Parameters")
        form = QFormLayout(grp_param)
        
        self.sp_degree = QSpinBox()
        self.sp_degree.setRange(0, 6)
        self.sp_degree.setValue(2)
        self.sp_degree.setToolTip("Polynomial Degree (0 = Off)")
        self.sp_degree.valueChanged.connect(self._degree_changed)
        form.addRow("Degree:", self.sp_degree)
        
        self.sp_samples = QSpinBox()
        self.sp_samples.setRange(10, 10000)
        self.sp_samples.setValue(100)
        self.sp_samples.setSingleStep(10)
        form.addRow("Num Samples:", self.sp_samples)
        
        self.sp_down = QSpinBox()
        self.sp_down.setRange(1, 16)
        self.sp_down.setValue(4)
        form.addRow("Downsample:", self.sp_down)
        
        self.sp_patch = QSpinBox()
        self.sp_patch.setRange(5, 50)
        self.sp_patch.setValue(15)
        form.addRow("Patch Size:", self.sp_patch)
        
        ctrl_layout.addWidget(grp_param)
        
        # Group: RBF Refinement
        grp_rbf = QGroupBox("RBF Refinement")
        v_rbf = QVBoxLayout(grp_rbf)
        
        self.chk_use_rbf = QCheckBox("Enable RBF")
        self.chk_use_rbf.setChecked(True)
        v_rbf.addWidget(self.chk_use_rbf)
        
        bg_rbf = QWidget()
        form_rbf = QFormLayout(bg_rbf)
        self.sp_rbf = QSpinBox()
        self.sp_rbf.setRange(1, 500)
        self.sp_rbf.setValue(30) # 0.30
        self.sp_rbf.setSuffix(" / 100")
        form_rbf.addRow("Smoothness:", self.sp_rbf)
        v_rbf.addWidget(bg_rbf)
        
        ctrl_layout.addWidget(grp_rbf)
        
        # Group: Options
        grp_opts = QGroupBox("Output")
        v_opts = QVBoxLayout(grp_opts)
        
        self.chk_make_bg_doc = QCheckBox("Create background document")
        self.chk_make_bg_doc.setChecked(False)
        v_opts.addWidget(self.chk_make_bg_doc)
        
        self.chk_preview_bg = QCheckBox("Preview Background Model")
        self.chk_preview_bg.setChecked(False)
        self.chk_preview_bg.toggled.connect(self._do_preview)
        v_opts.addWidget(self.chk_preview_bg)
        
        ctrl_layout.addWidget(grp_opts)
        
        # Status
        self.lbl_status = QLabel("Ready")
        self.lbl_status.setStyleSheet("color: #888;")
        ctrl_layout.addWidget(self.lbl_status)
        
        ctrl_layout.addStretch()
        
        # Actions
        h_btn = QHBoxLayout()
        self.btn_preview = QPushButton("Update Preview")
        self.btn_preview.clicked.connect(self._do_preview)
        h_btn.addWidget(self.btn_preview)
        
        self.btn_apply = QPushButton("Apply")
        self.btn_apply.clicked.connect(self._do_apply)
        h_btn.addWidget(self.btn_apply)
        
        ctrl_layout.addLayout(h_btn)
        
        self.btn_close = QPushButton("Close")
        self.btn_close.clicked.connect(self.close)
        ctrl_layout.addWidget(self.btn_close)
        
        # --- Right: Preview ---
        preview_panel = QWidget()
        preview_layout = QVBoxLayout(preview_panel)
        self.preview_area = QWidget() # wrapper
        
        # Toolbar
        tbar = QHBoxLayout()
        self.btn_zoom_in = themed_toolbtn("magnify_plus", "Zoom In")
        self.btn_zoom_in.clicked.connect(self.zoom_in)
        self.btn_zoom_out = themed_toolbtn("magnify_minus", "Zoom Out")
        self.btn_zoom_out.clicked.connect(self.zoom_out)
        self.btn_fit = themed_toolbtn("arrow_expand_all", "Fit to View")
        self.btn_fit.clicked.connect(self.fit_to_preview)
        
        self.btn_autostr = QPushButton("Autostretch")
        self.btn_autostr.setCheckable(True)
        self.btn_autostr.clicked.connect(self.autostretch_preview)
        
        tbar.addWidget(QLabel("<b>Preview</b> (Draw polygons to exclude)"))
        tbar.addStretch()
        tbar.addWidget(self.btn_autostr)
        tbar.addWidget(self.btn_zoom_in)
        tbar.addWidget(self.btn_zoom_out)
        tbar.addWidget(self.btn_fit)
        
        preview_layout.addLayout(tbar)
        
        self.preview_scroll = QScrollArea()
        self.preview_scroll.setWidgetResizable(False) # we manage size
        self.preview_scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_scroll.setStyleSheet("background-color: #222;")
        
        self.preview_label = QLabel()
        self.preview_label.setMouseTracking(True)
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_scroll.setWidget(self.preview_label)
        
        preview_layout.addWidget(self.preview_scroll)
        
        main_layout.addWidget(ctrl_panel)
        main_layout.addWidget(preview_panel, stretch=1)
        
        self._install_zoom_filters()

    def _set_status(self, msg: str):
        self.lbl_status.setText(msg)
        QApplication.processEvents() # allow immediate update

    def _get_source_float(self) -> np.ndarray | None:
        """Return the current document's image as float32 [0..1]."""
        # Robust check: if we have a doc with an image, use it.
        # Strict validation against DocManager list can be flaky with proxies.
        if self.doc and hasattr(self.doc, "image") and self.doc.image is not None:
             return np.asarray(self.doc.image).astype(np.float32, copy=False)
        return None

    def _degree_changed(self, v: int):
        if v == 0:
            self.chk_use_rbf.setChecked(True)
            self._set_status("Polynomial disabled (degree 0) -> RBF-only.")
        else:
            self._set_status("Ready")

    def _populate_initial_preview(self):
        src = self._get_source_float()
        if src is not None:
             self._set_preview_from_float(src)
             self.fit_to_preview()

    def _on_active_doc_changed(self, doc):
        if doc is not self.doc:
            self._refresh_document_from_active()
    
    def _run_abe_async(self, is_preview: bool):
        imgf = self._get_source_float()
        if imgf is None:
            QMessageBox.information(self, "No image", "No image is loaded.")
            return

        # Disable UI
        self._set_ui_busy(True)
        self._set_status("Starting ABE worker...")

        # Params
        deg   = int(self.sp_degree.value())
        npts  = int(self.sp_samples.value())
        dwn   = int(self.sp_down.value())
        patch = int(self.sp_patch.value())
        use_rbf = bool(self.chk_use_rbf.isChecked())
        rbf_smooth = float(self.sp_rbf.value()) * 0.01
        
        excl = self._build_exclusion_mask() # This is fast enough to do on main thread? (It's drawing polygons on small mask)
        # Verify overhead of _build_exclusion_mask? It rasterizes polygons. Should be fast.

        self._worker = ABEWorker(imgf, deg, npts, dwn, patch, use_rbf, rbf_smooth, excl)
        
        if is_preview:
            self._worker.finished.connect(self._on_preview_finished)
        else:
            self._worker.finished.connect(self._on_apply_finished)
            
        self._worker.progress.connect(self._set_status)
        self._worker.error.connect(self._on_worker_error)
        
        # Ensure cleanup
        self._worker.finished.connect(lambda: self._set_ui_busy(False))
        self._worker.error.connect(lambda: self._set_ui_busy(False))
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker.error.connect(self._worker.deleteLater)
        
        self._worker.start()

    def _set_ui_busy(self, busy: bool):
        self.btn_apply.setEnabled(not busy)
        self.btn_preview.setEnabled(not busy)
        self.btn_close.setEnabled(not busy)
        
        self.sp_degree.setEnabled(not busy)
        self.sp_samples.setEnabled(not busy)
        self.sp_down.setEnabled(not busy)
        self.sp_patch.setEnabled(not busy)
        self.chk_use_rbf.setEnabled(not busy)
        self.sp_rbf.setEnabled(not busy)
        self.chk_make_bg_doc.setEnabled(not busy)
        self.chk_preview_bg.setEnabled(not busy)
        
        if busy:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        else:
            QApplication.restoreOverrideCursor()

    def _on_worker_error(self, msg):
        QMessageBox.warning(self, "ABE Error", f"Worker failed: {msg}")
        self._set_status("Error")

    def _on_preview_finished(self, corrected, bg):
        try:
            show = bg if self.chk_preview_bg.isChecked() else corrected
            if not self.chk_preview_bg.isChecked():
                srcf = self._get_source_float()
                show = self._blend_with_mask_float(show, srcf)

            self._set_status("Rendering preview...")
            self._set_preview_pixmap(show)
            self._set_status("Ready")
        except Exception as e:
             self._on_worker_error(str(e))

    def _on_apply_finished(self, corrected, bg):
        try:
            # Preserve mono vs color
            out = corrected
            if out.ndim == 3 and out.shape[2] == 3 and (self.doc.image.ndim == 2 or (self.doc.image.ndim == 3 and self.doc.image.shape[2] == 1)):
                out = out[..., 0]

            # Blend
            srcf = self._get_source_float()
            out_masked = self._blend_with_mask_float(out, srcf)

            # Metadata/Undo logic
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

            params = {
                "degree": deg,
                "samples": npts,
                "downsample": dwn,
                "patch": patch,
                "rbf": use_rbf,
                "rbf_smooth": rbf_smooth,
                "make_background_doc": make_bg_doc,
            }

            # Replay memory
            mw = self.parent()
            try:
                remember = getattr(mw, "remember_last_headless_command", None)
                if remember is None:
                    remember = getattr(mw, "_remember_last_headless_command", None)
                if callable(remember):
                    remember("abe", params, description="Automatic Background Extraction")
            except Exception:
                pass
            
            # Mask metadata
            _marr, mid, mname = self._active_mask_layer()
            abe_meta = dict(params)
            abe_meta["exclusion"] = "polygons" if self._polygons else "none"

            meta = {
                "step_name": "ABE",
                "abe": abe_meta,
                "masked": bool(mid),
                "mask_id": mid,
                "mask_name": mname,
                "mask_blend": "m*out + (1-m)*src",
            }
            
            if hasattr(self, "_set_status"):
                self._set_status("Committing edit...")
            
            self.doc.apply_edit(
                out_masked.astype(np.float32, copy=False),
                step_name=step_name,
                metadata=meta,
            )

            if make_bg_doc and bg is not None:
                if hasattr(self, "_set_status"):
                    self._set_status("Creating background document...")
                mw = self.parent()
                dm = getattr(mw, "docman", None)
                if dm is not None:
                    base = os.path.splitext(self.doc.display_name())[0] if self.doc else "Image"
                    meta = {
                        "bit_depth": "32-bit floating point",
                        "is_mono": (bg.ndim == 2),
                        "source": "ABE background",
                        "original_header": self.doc.metadata.get("original_header"),
                    }
                    doc_bg = dm.open_array(bg.astype(np.float32, copy=False), metadata=meta, title=f"{base}_ABE_BG")
                    if hasattr(mw, "_spawn_subwindow_for"):
                        mw._spawn_subwindow_for(doc_bg)

            # Restore autostretch
            mw = self.parent()
            prev_autostretch = False
            view = None
            try:
                if hasattr(mw, "mdi") and mw.mdi.activeSubWindow():
                    view = mw.mdi.activeSubWindow().widget()
                    prev_autostretch = bool(getattr(view, "autostretch_enabled", False))
            except Exception:
                pass

            if hasattr(mw, "_log"):
                mw._log(step_name)

            try:
                if view is not None and hasattr(view, "set_autostretch") and callable(view.set_autostretch):
                    view.set_autostretch(prev_autostretch)
            except Exception:
                pass

            if hasattr(self, "_set_status"):
                self._set_status("Done")
            self.close()

        except Exception as e:
            self._on_worker_error(str(e))

    def _do_preview(self):
        self._run_abe_async(is_preview=True)

    def _do_apply(self):
         self._run_abe_async(is_preview=False)

    def closeEvent(self, ev):
        # 1) Disconnect active-doc tracking (Fabio hook)
        try:
            if self._connected_current_doc_changed and hasattr(self._main, "currentDocumentChanged"):
                self._main.currentDocumentChanged.disconnect(self._on_active_doc_changed)
        except Exception:
            pass
        self._connected_current_doc_changed = False

        # 2) Stop any background preview worker/thread if you have one
        # (names may differ in your file; keep what matches your implementation)
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

        # show autostretched or raw; siril_style_autostretch() already clips its result
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

        # ✅ store a clean base without overlays
        self._base_pixmap = QPixmap.fromImage(scaled)
        self.preview_label.setPixmap(self._base_pixmap)
        self.preview_label.resize(self._base_pixmap.size())

    def _redraw_overlay(self):
        pm_base = self._base_pixmap or self.preview_label.pixmap()
        if pm_base is None:
            return

        # start from a fresh copy of the clean base
        composed = QPixmap(pm_base)
        overlay = QPixmap(pm_base.size())
        overlay.fill(Qt.GlobalColor.transparent)

        painter = QPainter(overlay)
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

        painter.end()

        p = QPainter(composed)
        p.drawPixmap(0, 0, overlay)
        p.end()

        self.preview_label.setPixmap(composed)

    # ----- zoom/pan + polygon drawing -----
    def eventFilter(self, obj, ev):
        # ---- Robust Ctrl+Wheel zoom handling (Qt6-friendly) ----
        if ev.type() == QEvent.Type.Wheel and (
            obj is self.preview_label
            or obj is self.preview_scroll
            or obj is self.preview_scroll.viewport()
            or obj is self.preview_scroll.horizontalScrollBar()
            or obj is self.preview_scroll.verticalScrollBar()
        ):
            # always stop the wheel from scrolling
            ev.accept()

            # Zoom only when Ctrl is held
            if ev.modifiers() & Qt.KeyboardModifier.ControlModifier:
                factor = 1.25 if ev.angleDelta().y() > 0 else 0.8

                # Anchor at the mouse position in the viewport (even if event came from a scrollbar)
                vp = self.preview_scroll.viewport()
                anchor_vp = vp.mapFromGlobal(ev.globalPosition().toPoint())

                # Clamp to viewport rect (robust if the event originated on scrollbars)
                r = vp.rect()
                if not r.contains(anchor_vp):
                    anchor_vp.setX(max(r.left(),  min(r.right(),  anchor_vp.x())))
                    anchor_vp.setY(max(r.top(),   min(r.bottom(), anchor_vp.y())))

                self._zoom_at(factor, anchor_vp)
            return True

        # ---- Existing polygon drawing on the label ----
        if obj is self.preview_label:
            if ev.type() == QEvent.Type.MouseButtonPress:
                if ev.buttons() & Qt.MouseButton.RightButton:
                    if self._drawing_poly and len(self._drawing_poly) >= 3:
                        self._polygons.append(self._drawing_poly)
                    self._drawing_poly = None
                    self._redraw_overlay()
                    return True
                if ev.buttons() & Qt.MouseButton.MiddleButton or (ev.buttons() & Qt.MouseButton.LeftButton and (ev.modifiers() & Qt.KeyboardModifier.ControlModifier)):
                    self._panning = True
                    self._pan_last = ev.position().toPoint()
                    self.preview_label.setCursor(Qt.CursorShape.ClosedHandCursor)
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
                if self._drawing_poly is not None and (ev.buttons() & Qt.MouseButton.LeftButton):
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
                if ev.button() == Qt.MouseButton.LeftButton and self._drawing_poly is not None:
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


    def autostretch_preview(self, sigma: float = 3.0) -> None:
        """
        Toggle Siril-style MAD autostretch on the *preview only* (non-destructive).
        First press applies; second press restores the original preview.
        Works from the float [0..1] preview source to avoid double-clipping.
        """
        if self._preview_source_f01 is None and self._last_preview is None:
            return

        # Lazy init toggle state
        if not hasattr(self, "_autostretch_on"):
            self._autostretch_on = False
        if not hasattr(self, "_orig_preview8"):
            self._orig_preview8 = None

        def _rebuild_from_last():
            h, w = self._last_preview.shape[:2]
            ptr = sip.voidptr(self._last_preview.ctypes.data)
            qimg = QImage(ptr, w, h, self._last_preview.strides[0], QImage.Format.Format_RGB888)
            self._preview_qimg = qimg
            self._update_preview_scaled()
            self._redraw_overlay()

        # Toggle OFF → restore original preview bytes
        if self._autostretch_on and self._orig_preview8 is not None:
            self._last_preview = np.ascontiguousarray(self._orig_preview8)
            _rebuild_from_last()
            self._autostretch_on = False
            if hasattr(self, "btn_autostr"):
                self.btn_autostr.setText("Autostretch")
            return

        # Toggle ON → cache original and apply stretch from float source
        if self._last_preview is not None:
            self._orig_preview8 = np.ascontiguousarray(self._last_preview)

        # Prefer float source (avoids 8-bit clipping); fall back to decoding _last_preview if needed
        arr = self._preview_source_f01 if self._preview_source_f01 is not None else (self._last_preview.astype(np.float32)/255.0)

        stretched = hard_autostretch(arr, target_median=0.5, sigma=2, linked=False, use_24bit=True)

        buf8 = (np.clip(stretched, 0.0, 1.0) * 255.0).astype(np.uint8)
        if buf8.ndim == 2:
            buf8 = np.stack([buf8] * 3, axis=-1)
        self._last_preview = np.ascontiguousarray(buf8)

        _rebuild_from_last()
        self._autostretch_on = True
        if hasattr(self, "btn_autostr"):
            self.btn_autostr.setText("Autostretch (On)")


    def _apply_autostretch_inplace(self, sigma: float = 3.0):
        # Apply autostretch directly from current float preview source without toggling state.
        if self._preview_source_f01 is None:
            return
        stretched = hard_autostretch(self._preview_source_f01, target_median=0.5, sigma=2,
                             linked=False, use_24bit=True)
        buf8 = (np.clip(stretched, 0.0, 1.0) * 255.0).astype(np.uint8)
        if buf8.ndim == 2:
            buf8 = np.stack([buf8] * 3, axis=-1)
        self._last_preview = np.ascontiguousarray(buf8)
        h, w = buf8.shape[:2]
        qimg = QImage(buf8.data, w, h, buf8.strides[0], QImage.Format.Format_RGB888)
        self._preview_qimg = qimg
        self._update_preview_scaled()
        self._redraw_overlay()
