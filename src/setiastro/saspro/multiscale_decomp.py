# pro/multiscale_decomp.py
from __future__ import annotations
import numpy as np
import cv2
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from PyQt6.QtCore import Qt, QTimer, QRect, QRectF, QSettings, QByteArray
from PyQt6.QtGui import QImage, QPixmap, QPen, QColor, QIcon, QMovie
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout,
    QPushButton, QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox,
    QTableWidget, QTableWidgetItem, QWidget, QLabel, QGraphicsView,
    QGraphicsScene, QGraphicsPixmapItem, QMessageBox, QToolButton, QSlider, QSplitter,
    QProgressDialog, QApplication
)
from contextlib import contextmanager
from setiastro.saspro.resources import get_resources
from setiastro.saspro.autostretch import autostretch
try:
    cv2.setUseOptimized(True)
    cv2.setNumThreads(0)  # 0 = let OpenCV decide
except Exception:
    pass
import math


_N_WORKERS = max(2, (os.cpu_count() or 4))

try:
    from scipy.ndimage import gaussian_filter as _scipy_gaussian
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

class _ZoomPanView(QGraphicsView):
    """
    QGraphicsView that supports wheel-zoom and click-drag panning.
    Calls on_view_changed() whenever viewport position/scale changes.
    """
    def __init__(self, *args, on_view_changed=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

        self._panning = False
        self._pan_start = None
        self._on_view_changed = on_view_changed  # callable or None

    def _notify(self):
        cb = self._on_view_changed
        if callable(cb):
            cb()

    def wheelEvent(self, ev):
        delta = ev.angleDelta().y()
        if delta == 0:
            return
        factor = 1.25 if delta > 0 else 0.8
        self.scale(factor, factor)
        ev.accept()
        self._notify()

    def mousePressEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            self._panning = True
            self._pan_start = ev.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            ev.accept()
            return
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if self._panning and self._pan_start is not None:
            delta = ev.pos() - self._pan_start
            self._pan_start = ev.pos()

            h = self.horizontalScrollBar()
            v = self.verticalScrollBar()
            h.setValue(h.value() - delta.x())
            v.setValue(v.value() - delta.y())
            ev.accept()
            # scrollbars will trigger _notify via their signals too, but harmless:
            self._notify()
            return

        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton and self._panning:
            self._panning = False
            self._pan_start = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
            ev.accept()
            return
        super().mouseReleaseEvent(ev)



# ─────────────────────────────────────────────
# Core math (your backbone)
# ─────────────────────────────────────────────

def _blur_gaussian(img: np.ndarray, sigma: float) -> np.ndarray:
    H, W = img.shape[:2]
    k_sz = int(max(3, 2 * round(3 * sigma) + 1)) | 1

    # Small images — not worth any overhead
    if H * W < 256 * 256 or sigma < 0.5:
        return cv2.GaussianBlur(img, (k_sz, k_sz), sigmaX=sigma, sigmaY=sigma,
                                borderType=cv2.BORDER_REFLECT)

    # Try torch path first — uses GPU if available, CPU otherwise.
    # Both are genuinely parallel and don't fight with ThreadPoolExecutor.
    try:
        result = _blur_gaussian_torch(img, sigma, k_sz)
        return result
    except Exception as e:
        print(f"[BLUR] torch path failed ({e}), falling back to tiled", flush=True)

    # Torch unavailable — fall back to tiled scipy/cv2
    return _blur_gaussian_tiled(img, sigma, k_sz)


def _blur_gaussian_torch(img: np.ndarray, sigma: float, k_sz: int) -> np.ndarray:
    """
    Gaussian blur via torch conv2d — uses whatever device torch has available.
    Separable: 1D horiz kernel then 1D vert kernel, much faster than 2D.
    """
    try:
        # Import torch the SASpro way — already loaded, zero overhead
        import torch
    except ImportError:
        raise RuntimeError("torch not available")

    # Build 1D Gaussian kernel matching cv2's kernel exactly
    k1d = torch.tensor(
        cv2.getGaussianKernel(k_sz, sigma).flatten(),
        dtype=torch.float32
    )

    # Pick device — CUDA if available, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    k1d = k1d.to(device)

    # Input tensor: (1, 1, H, W) for conv2d
    H, W = img.shape[:2]
    is_rgb = (img.ndim == 3)

    if is_rgb:
        # (H, W, C) -> (C, 1, H, W)
        t = torch.from_numpy(
            np.ascontiguousarray(img.transpose(2, 0, 1))
        ).float().unsqueeze(1).to(device)
        C = img.shape[2]
    else:
        # (H, W) -> (1, 1, H, W)
        t = torch.from_numpy(
            np.ascontiguousarray(img)
        ).float().unsqueeze(0).unsqueeze(0).to(device)
        C = 1

    pad = k_sz // 2

    # Horizontal pass: kernel shape (C, 1, 1, k_sz)
    kh = k1d.view(1, 1, 1, k_sz).expand(C, 1, 1, k_sz).contiguous()
    t = torch.nn.functional.pad(t, (pad, pad, 0, 0), mode='reflect')
    t = torch.nn.functional.conv2d(t, kh, groups=C)

    # Vertical pass: kernel shape (C, 1, k_sz, 1)
    kv = k1d.view(1, 1, k_sz, 1).expand(C, 1, k_sz, 1).contiguous()
    t = torch.nn.functional.pad(t, (0, 0, pad, pad), mode='reflect')
    t = torch.nn.functional.conv2d(t, kv, groups=C)

    # Back to numpy
    if is_rgb:
        # (C, 1, H, W) -> (H, W, C)
        result = t.squeeze(1).permute(1, 2, 0).cpu().numpy().astype(np.float32)
    else:
        result = t.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)

    return result


def _blur_gaussian_tiled(img: np.ndarray, sigma: float, k_sz: int) -> np.ndarray:
    """Tiled scipy fallback for when torch is unavailable."""
    if not _HAS_SCIPY:
        return cv2.GaussianBlur(img, (k_sz, k_sz), sigmaX=sigma, sigmaY=sigma,
                                borderType=cv2.BORDER_REFLECT)

    H, W = img.shape[:2]
    overlap = int(math.ceil(4.0 * sigma)) + 4
    n_tiles_target = _N_WORKERS * 2

    if H >= W:
        n_row = max(1, int(round(math.sqrt(n_tiles_target * H / max(W, 1)))))
        n_col = max(1, int(round(n_tiles_target / max(n_row, 1))))
    else:
        n_col = max(1, int(round(math.sqrt(n_tiles_target * W / max(H, 1)))))
        n_row = max(1, int(round(n_tiles_target / max(n_col, 1))))

    min_core = max(overlap * 2, 64)
    n_row = min(n_row, max(1, H // min_core))
    n_col = min(n_col, max(1, W // min_core))

    if n_row <= 1 and n_col <= 1:
        if img.ndim == 2:
            return _scipy_gaussian(img, sigma=sigma, mode='reflect').astype(np.float32)
        return _scipy_gaussian(img, sigma=(sigma, sigma, 0), mode='reflect').astype(np.float32)

    out = np.empty_like(img)
    row_edges = [int(round(H * i / n_row)) for i in range(n_row + 1)]
    col_edges = [int(round(W * j / n_col)) for j in range(n_col + 1)]

    def _process(rc):
        r, c = rc
        ry0, ry1 = row_edges[r], row_edges[r + 1]
        cx0, cx1 = col_edges[c], col_edges[c + 1]
        if ry1 <= ry0 or cx1 <= cx0:
            return
        py0 = max(0, ry0 - overlap); py1 = min(H, ry1 + overlap)
        px0 = max(0, cx0 - overlap); px1 = min(W, cx1 + overlap)
        crop = img[py0:py1, px0:px1]
        if crop.ndim == 2:
            blurred = _scipy_gaussian(crop, sigma=sigma, mode='reflect').astype(np.float32)
        else:
            blurred = _scipy_gaussian(crop, sigma=(sigma, sigma, 0), mode='reflect').astype(np.float32)
        oy0 = ry0 - py0; oy1 = oy0 + (ry1 - ry0)
        ox0 = cx0 - px0; ox1 = ox0 + (cx1 - cx0)
        if out.ndim == 2:
            out[ry0:ry1, cx0:cx1] = blurred[oy0:oy1, ox0:ox1]
        else:
            out[ry0:ry1, cx0:cx1, :] = blurred[oy0:oy1, ox0:ox1, :]

    with ThreadPoolExecutor(max_workers=_N_WORKERS) as ex:
        list(ex.map(_process, [(r, c) for r in range(n_row) for c in range(n_col)]))

    return out

def multiscale_decompose(img01: np.ndarray, layers: int, base_sigma: float = 1.0):
    img = img01.astype(np.float32, copy=False)
    is_rgb = (img.ndim == 3 and img.shape[2] >= 3)
    if not is_rgb:
        return _decompose_channel(img, layers, base_sigma)
    n_ch = img.shape[2]
    channel_imgs = [img[:, :, ch] for ch in range(n_ch)]
    with ThreadPoolExecutor(max_workers=min(n_ch, _N_WORKERS)) as ex:
        results = list(ex.map(
            lambda ch: _decompose_channel(channel_imgs[ch], layers, base_sigma),
            range(n_ch)
        ))
    H, W = channel_imgs[0].shape
    out_details = []
    for k in range(layers):
        arr = np.empty((H, W, n_ch), dtype=np.float32)
        for ch in range(n_ch):
            arr[:, :, ch] = results[ch][0][k]
        out_details.append(arr)
    residual = np.empty((H, W, n_ch), dtype=np.float32)
    for ch in range(n_ch):
        residual[:, :, ch] = results[ch][1]
    return out_details, residual

def _decompose_channel(img2d: np.ndarray, layers: int, base_sigma: float):
    """Serial pyramid for one channel — parallel blur within each layer."""
    c = img2d.astype(np.float32, copy=False)
    details = []
    for k in range(layers):
        c_next = _blur_gaussian(c, base_sigma * (2 ** k))
        details.append(c - c_next)
        c = c_next
    return details, c

def multiscale_reconstruct(details, residual):
    out = residual.astype(np.float32, copy=True)
    for w in details:
        out += w
    return out

def soft_threshold(x: np.ndarray, t: float):
    a = np.abs(x)
    return np.sign(x) * np.maximum(0.0, a - t)

def apply_layer_ops(
    w: np.ndarray,
    bias_gain: float,
    thr_sigma: float,
    amount: float,
    denoise_strength: float = 0.0,
    sigma: float | np.ndarray | None = None,
    layer_index: int | None = None,
    *,
    mode: str = "μ–σ Thresholding",
):
    w2 = w

    m = (mode or "").strip().lower()
    is_linear = m.startswith("linear")
    if is_linear:
        return w * bias_gain if abs(bias_gain - 1.0) > 1e-6 else w

    # 1) Noise reduction step (MMT-style NR)
    if denoise_strength > 0.0:
        if sigma is None:
            sigma = _robust_sigma(w2)
        sigma_f = float(sigma)

        i = int(layer_index or 0)

        # --- SMOOTH scaling option (pick ONE) ---
        # Option A: linear growth (very controllable)
        # scale = 1.0 + 0.75 * i

        # Option B: sqrt growth of 2^i (gentle, "natural")
        scale = (2.0 ** i) ** 0.5   # 1, 1.41, 2, 2.83, 4, ...

        # Base: 3σ at denoise=1 for layer 0, increases by scale
        t_dn = denoise_strength * 3.0 * scale * sigma_f

        if t_dn > 0.0:
            w_dn = soft_threshold(w2, t_dn)
            w2 = (1.0 - denoise_strength) * w2 + denoise_strength * w_dn

    # 2) Threshold in σ units + bias shaping
    if thr_sigma > 0.0:
        if sigma is None:
            sigma = _robust_sigma(w2)
        sigma_f = float(sigma)
        t = thr_sigma * sigma_f
        if t > 0.0:
            wt = soft_threshold(w2, t)
            w2 = (1.0 - amount) * w2 + amount * wt

    if abs(bias_gain - 1.0) > 1e-6:
        w2 = w2 * bias_gain
    return w2

def _robust_sigma(arr: np.ndarray) -> float:
    a = np.asarray(arr, dtype=np.float32).ravel()
    # Remove non-finite values
    a = a[np.isfinite(a)]
    if a.size == 0:
        return 1e-6

    # For large arrays, use a grid subsample instead of random.choice
    # Grid subsample is O(1) index math vs O(N) random permutation
    MAX_SAMPLES = 100_000
    if a.size > MAX_SAMPLES:
        step = a.size // MAX_SAMPLES
        a = a[::step]

    med = np.median(a)
    mad = np.median(np.abs(a - med))
    if mad <= 0:
        s = float(np.std(a))
        return s if s > 0 else 1e-6

    return float(1.4826 * mad)


# ─────────────────────────────────────────────
# Layer config
# ─────────────────────────────────────────────


@dataclass
class LayerCfg:
    enabled: bool = True
    bias_gain: float = 1.0        # 1.0 = unchanged
    thr: float = 0.0              # soft threshold in detail domain
    amount: float = 0.0           # 0..1 blend toward thresholded
    denoise: float = 0.0          # 0..1 additional noise reduction


# ─────────────────────────────────────────────
# Dialog
# ─────────────────────────────────────────────

class MultiscaleDecompDialog(QDialog):
    def __init__(self, parent, doc):
        super().__init__(parent)
        self.setWindowTitle("Multiscale Decomposition")
        self.setWindowFlag(Qt.WindowType.Window, True)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setModal(False)
        try:
            self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        except Exception:
            pass  # older PyQt6 versions        
        self.setMinimumSize(1050, 700)
        self.residual_enabled = True
        self._layer_noise = None  # list[float] per detail layer
        self._cached_coarse = None
        self._cached_img_id = None
        self._doc = doc
        base = getattr(doc, "image", None)
        if base is None:
            raise RuntimeError("Document has no image.")

        # normalize to float32 [0..1] ...
        img0 = np.asarray(base)
        is_int = (img0.dtype.kind in "ui")

        img = img0.astype(np.float32, copy=False)
        if is_int:
            maxv = float(np.nanmax(img0)) or 1.0
            img = img / max(1.0, maxv)
        img = np.clip(img, 0.0, 1.0).astype(np.float32, copy=False)

        self._orig_shape = img.shape
        self._orig_mono = (img.ndim == 2) or (img.ndim == 3 and img.shape[2] == 1)

        # Keep a mono working image for computation, 3ch only for display
        if img.ndim == 2:
            self._image_work = img.copy()          # (H, W) — used for decomp
            img3 = np.repeat(img[:, :, None], 3, axis=2)
        elif img.ndim == 3 and img.shape[2] == 1:
            self._image_work = img[:, :, 0].copy() # (H, W)
            img3 = np.repeat(img, 3, axis=2)
        else:
            self._image_work = img[:, :, :3].copy() # RGB stays as-is
            img3 = img[:, :, :3]

        self._image = img3.copy()      # 3ch for display only
        self._preview_img = img3.copy()

        # decomposition cache
        self._cached_layers = None
        self._cached_residual = None
        self._cached_key = None

        # per-layer configs
        self.layers = 4
        self.base_sigma = 1.0
        self.cfgs: list[LayerCfg] = [LayerCfg() for _ in range(self.layers)]

        # debounce preview updates
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.timeout.connect(self._rebuild_preview)

        self._build_ui()
        H, W = self._image.shape[:2]
        self.scene.setSceneRect(QRectF(0, 0, W, H))
        # ───── NEW: initialization busy dialog ─────
        prog = QProgressDialog("Initializing multiscale decomposition…", "", 0, 0, self)
        prog.setWindowTitle("Multiscale Decomposition")
        prog.setWindowModality(Qt.WindowModality.ApplicationModal)
        prog.setCancelButton(None)          # no cancel button, just a busy indicator
        prog.setMinimumDuration(0)          # show immediately
        prog.show()
        QApplication.processEvents()

        # heavy work (MADs, blurs, etc.)
        import time
        _t = time.perf_counter()
        self._recompute_decomp(force=True)
        print(f"[DECOMP] total _recompute_decomp: {(time.perf_counter()-_t)*1000:.1f}ms", flush=True)

        _t = time.perf_counter()
        self._rebuild_preview()
        print(f"[PREVIEW] _rebuild_preview: {(time.perf_counter()-_t)*1000:.1f}ms", flush=True)

        prog.close()
        # ─────────────── END NEW ───────────────

        QTimer.singleShot(0, self._fit_view)


    # ---------- UI ----------
    def _build_ui(self):
        root = QHBoxLayout(self)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(self.splitter)

        # ----- LEFT: preview -----
        left_widget = QWidget(self)
        left = QVBoxLayout(left_widget)

        self.scene = QGraphicsScene(self)

        self.view = _ZoomPanView(self.scene, on_view_changed=self._schedule_roi_preview)
        self.view.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Base full-image item (keeps zoom/pan working)
        self.pix_base = QGraphicsPixmapItem()
        self.pix_base.setOffset(0, 0)
        self.scene.addItem(self.pix_base)

        # ROI overlay item (updates fast)
        self.pix_roi = QGraphicsPixmapItem()
        self.pix_roi.setZValue(10)  # draw above base
        self.scene.addItem(self.pix_roi)

        left.addWidget(self.view)
        # Busy overlay (shown during recompute)
        self.busy_label = QLabel("Computing…", self.view.viewport())
        self.busy_label.setStyleSheet("""
            QLabel {
                background: rgba(0,0,0,140);
                color: white;
                padding: 6px 10px;
                border-radius: 8px;
                font-weight: 600;
            }
        """)
        self.busy_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.busy_label.hide()
        # --- Spinner (animated) ---
        self.busy_spinner = QLabel()
        self.busy_spinner.setFixedSize(20, 20)
        self.busy_spinner.setToolTip("Computing…")
        self.busy_spinner.setVisible(False)

        gif_path = get_resources().SPINNER_GIF  # <- canonical, works frozen/dev
        gif_path = os.path.normpath(gif_path)

        self._busy_movie = QMovie(gif_path)
        self._busy_movie.setScaledSize(self.busy_spinner.size())
        self.busy_spinner.setMovie(self._busy_movie)

        self._busy_show_timer = QTimer(self)
        self._busy_show_timer.setSingleShot(True)
        self._busy_show_timer.timeout.connect(self._show_busy_overlay)
        self._busy_depth = 0
        zoom_row = QHBoxLayout()

        self.zoom_out_btn = QToolButton()
        self.zoom_out_btn.setIcon(QIcon.fromTheme("zoom-out"))
        self.zoom_out_btn.setToolTip("Zoom Out")

        self.zoom_in_btn = QToolButton()
        self.zoom_in_btn.setIcon(QIcon.fromTheme("zoom-in"))
        self.zoom_in_btn.setToolTip("Zoom In")

        self.fit_btn = QToolButton()
        self.fit_btn.setIcon(QIcon.fromTheme("zoom-fit-best"))
        self.fit_btn.setToolTip("Fit to Preview")

        self.one_to_one_btn = QToolButton()
        self.one_to_one_btn.setIcon(QIcon.fromTheme("zoom-original"))
        self.one_to_one_btn.setToolTip("1:1")

        self.zoom_out_btn.clicked.connect(lambda: (self.view.scale(0.8, 0.8), self._schedule_roi_preview()))
        self.zoom_in_btn.clicked.connect(lambda: (self.view.scale(1.25, 1.25), self._schedule_roi_preview()))
        self.fit_btn.clicked.connect(self._fit_view)
        self.one_to_one_btn.clicked.connect(self._one_to_one)

        zoom_row.addStretch(1)
        zoom_row.addWidget(self.zoom_out_btn)
        zoom_row.addWidget(self.zoom_in_btn)
        zoom_row.addSpacing(10)
        zoom_row.addWidget(self.fit_btn)
        zoom_row.addWidget(self.one_to_one_btn)
        zoom_row.addSpacing(10)
        zoom_row.addWidget(self.busy_spinner)   # <-- add here
        zoom_row.addStretch(1)

        left.addLayout(zoom_row)

        # ----- RIGHT: controls -----
        right_widget = QWidget(self)
        right = QVBoxLayout(right_widget)

        gb_global = QGroupBox("Global")
        form = QFormLayout(gb_global)

        self.spin_layers = QSpinBox()
        self.spin_layers.setRange(1, 10)
        self.spin_layers.setValue(self.layers)

        self.spin_sigma = QDoubleSpinBox()
        self.spin_sigma.setRange(0.3, 5.0)
        self.spin_sigma.setSingleStep(0.1)
        self.spin_sigma.setValue(self.base_sigma)

        self.cb_linked_rgb = QCheckBox("Linked RGB (apply same params to all channels)")
        self.cb_linked_rgb.setChecked(True)

        # NEW: Fast ROI preview
        self.cb_fast_roi_preview = QCheckBox("Fast ROI preview (compute visible area only)")
        self.cb_fast_roi_preview.setChecked(True)
        self.cb_fast_roi_preview.setToolTip(
            "When enabled, preview only computes the currently visible region (with padding for blur).\n"
            "Apply/Send-to-Doc always computes the full image."
        )
        self.cb_autostretch = QCheckBox("Auto-stretch preview")
        self.cb_autostretch.setChecked(False)
        self.cb_autostretch.setToolTip(
            "Stretch the preview for visibility. Does not affect Apply or Send to Document — "
            "those always use linear data."
        )
        form.addRow(self.cb_autostretch)
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["μ–σ Thresholding", "Linear"])
        self.combo_mode.setCurrentText("μ–σ Thresholding")
        self.combo_mode.setToolTip(
            "Multiscale mode:\n"
            "• μ–σ Thresholding: σ-based thresholding + denoise and gain (nonlinear).\n"
            "• Linear: strictly linear multiscale transform; only Gain is applied."
        )

        self.combo_preview = QComboBox()
        self._refresh_preview_combo()

        form.addRow("Layers:", self.spin_layers)
        form.addRow("Base sigma:", self.spin_sigma)
        form.addRow(self.cb_linked_rgb)
        form.addRow(self.cb_fast_roi_preview)
        form.addRow("Mode:", self.combo_mode)
        form.addRow("Layer preview:", self.combo_preview)

        right.addWidget(gb_global)

        # Layers table
        gb_layers = QGroupBox("Layers")
        v = QVBoxLayout(gb_layers)
        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(
            ["On", "Layer", "Scale", "Gain", "Thr (σ)", "Amt", "NR", "Type"]
        )
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(self.table.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(self.table.SelectionMode.SingleSelection)
        v.addWidget(self.table)
        right.addWidget(gb_layers, stretch=1)

        # Per-layer editor...
        gb_edit = QGroupBox("Selected Layer")
        ef = QFormLayout(gb_edit)
        self.lbl_sel = QLabel("Layer: —")

        # --- Spin boxes ---
        self.spin_gain = QDoubleSpinBox()
        self.spin_gain.setRange(0.0, 10.0)
        self.spin_gain.setSingleStep(0.05)
        self.spin_gain.setValue(1.0)
        self.spin_gain.setToolTip(
            "Gain: multiplies the detail coefficients on this layer.\n"
            "1.0 = unchanged, >1.0 boosts detail, <1.0 reduces it."
        )

        self.spin_thr = QDoubleSpinBox()
        self.spin_thr.setRange(0.0, 10.0)       # N·σ
        self.spin_thr.setSingleStep(0.1)
        self.spin_thr.setDecimals(2)
        self.spin_thr.setToolTip(
            "Threshold (σ): soft threshold level in units of this layer's noise σ.\n"
            "0 = no thresholding; 1–3 ≈ mild to strong suppression of small coefficients."
        )

        self.spin_amt = QDoubleSpinBox()
        self.spin_amt.setRange(0.0, 1.0)
        self.spin_amt.setSingleStep(0.05)
        self.spin_amt.setToolTip(
            "Amount: blend factor toward the thresholded version of the layer.\n"
            "0 = ignore thresholding, 1 = fully use the thresholded layer."
        )

        self.spin_denoise = QDoubleSpinBox()
        self.spin_denoise.setRange(0.0, 1.0)
        self.spin_denoise.setSingleStep(0.05)
        self.spin_denoise.setValue(0.0)
        self.spin_denoise.setToolTip(
            "Denoise: extra multiscale noise reduction on this layer.\n"
            "0 = off, 1 = strong NR (≈3σ soft threshold blended in)."
        )

        # --- Sliders (int ranges, mapped to spins) ---
        self.slider_gain = QSlider(Qt.Orientation.Horizontal)
        self.slider_gain.setRange(0, 1000)         # 0..10.00
        self.slider_gain.setToolTip(self.spin_gain.toolTip())

        self.slider_thr = QSlider(Qt.Orientation.Horizontal)
        self.slider_thr.setRange(0, 1000)         # 0..10.00 σ (×0.01)
        self.slider_thr.setToolTip(self.spin_thr.toolTip())

        self.slider_amt = QSlider(Qt.Orientation.Horizontal)
        self.slider_amt.setRange(0, 100)          # 0..1.00
        self.slider_amt.setToolTip(self.spin_amt.toolTip())

        self.slider_denoise = QSlider(Qt.Orientation.Horizontal)
        self.slider_denoise.setRange(0, 100)      # 0..1.00
        self.slider_denoise.setToolTip(self.spin_denoise.toolTip())

        # Layout rows: label -> [slider | spinbox]
        ef.addRow(self.lbl_sel)

        gain_row = QHBoxLayout()
        gain_row.addWidget(self.slider_gain)
        gain_row.addWidget(self.spin_gain)
        ef.addRow("Gain:", gain_row)

        thr_row = QHBoxLayout()
        thr_row.addWidget(self.slider_thr)
        thr_row.addWidget(self.spin_thr)
        ef.addRow("Threshold (σ):", thr_row)

        amt_row = QHBoxLayout()
        amt_row.addWidget(self.slider_amt)
        amt_row.addWidget(self.spin_amt)
        ef.addRow("Amount:", amt_row)

        dn_row = QHBoxLayout()
        dn_row.addWidget(self.slider_denoise)
        dn_row.addWidget(self.spin_denoise)
        ef.addRow("Denoise:", dn_row)

        right.addWidget(gb_edit)

        # Buttons...
        btn_row = QHBoxLayout()
        self.btn_apply = QPushButton("Apply to Document")
        self.btn_detail_new = QPushButton("Send to New Document")
        self.btn_split_layers = QPushButton("Split Layers to Documents")
        self.btn_close = QPushButton("Close")

        btn_row.addStretch(1)
        btn_row.addWidget(self.btn_apply)
        btn_row.addWidget(self.btn_detail_new)
        btn_row.addWidget(self.btn_split_layers)
        btn_row.addWidget(self.btn_close)
        right.addLayout(btn_row)

        self.splitter.addWidget(left_widget)
        self.splitter.addWidget(right_widget)
        self.splitter.setStretchFactor(0, 2)
        self.splitter.setStretchFactor(1, 1)

        # ----- Signals -----
        self.spin_layers.valueChanged.connect(self._on_layers_changed)
        self.spin_sigma.valueChanged.connect(self._on_global_changed)
        self.combo_mode.currentIndexChanged.connect(self._on_mode_changed)
        self.combo_preview.currentIndexChanged.connect(self._schedule_preview)
        self.cb_fast_roi_preview.toggled.connect(self._schedule_roi_preview)
        self.cb_autostretch.toggled.connect(self._schedule_roi_preview)
        self.table.itemSelectionChanged.connect(self._on_table_select)

        self.spin_gain.valueChanged.connect(self._on_layer_editor_changed)
        self.spin_thr.valueChanged.connect(self._on_layer_editor_changed)
        self.spin_amt.valueChanged.connect(self._on_layer_editor_changed)
        self.spin_denoise.valueChanged.connect(self._on_layer_editor_changed)

        self.slider_gain.valueChanged.connect(self._on_gain_slider_changed)
        self.slider_thr.valueChanged.connect(self._on_thr_slider_changed)
        self.slider_amt.valueChanged.connect(self._on_amt_slider_changed)
        self.slider_denoise.valueChanged.connect(self._on_dn_slider_changed)

        self.btn_apply.clicked.connect(self._commit_to_doc)
        self.btn_detail_new.clicked.connect(self._send_detail_to_new_doc)
        self.btn_split_layers.clicked.connect(self._split_layers_to_docs)
        self.btn_close.clicked.connect(self._on_close_clicked)

        # Connect viewport scroll changes
        self._connect_viewport_signals()

    # ---------- Preview plumbing ----------
    def _spinner_on(self):
        if getattr(self, "_closing", False):
            return
        try:
            sp = getattr(self, "busy_spinner", None)
            if sp is None:
                return
            sp.setVisible(True)
            mv = getattr(self, "_busy_movie", None)
            if mv is not None and mv.state() != QMovie.MovieState.Running:
                mv.start()
        except RuntimeError:
            return

    def _spinner_off(self):
        try:
            sp = getattr(self, "busy_spinner", None)
            mv = getattr(self, "_busy_movie", None)
            if mv is not None:
                mv.stop()
            if sp is not None:
                sp.setVisible(False)
        except RuntimeError:
            return


    def _show_busy_overlay(self):
        try:
            self.busy_label.adjustSize()
            self.busy_label.move(12, 12)
            self.busy_label.show()
        except Exception:
            pass

    def _begin_busy(self):
        self._busy_depth += 1
        if self._busy_depth == 1:
            # show only if compute isn't instant
            self._busy_show_timer.start(120)
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

    def _end_busy(self):
        self._busy_depth = max(0, self._busy_depth - 1)
        if self._busy_depth == 0:
            self._busy_show_timer.stop()
            self.busy_label.hide()
            QApplication.restoreOverrideCursor()


    def _on_mode_changed(self, idx: int):
        # Re-enable/disable controls as needed
        self._update_param_widgets_for_mode()
        self._schedule_preview()

    def _schedule_preview(self):
        if getattr(self, "_closing", False):
            return
        self._preview_timer.start(60)

    def _schedule_roi_preview(self):
        if getattr(self, "_closing", False):
            return
        self._preview_timer.start(60)

    def _connect_viewport_signals(self):
        """
        Any pan/scroll should schedule ROI preview recompute.
        """
        try:
            self.view.horizontalScrollBar().valueChanged.connect(self._schedule_roi_preview)
            self.view.verticalScrollBar().valueChanged.connect(self._schedule_roi_preview)
        except Exception:
            pass

    def _recompute_decomp(self, force: bool = False):
        # at the top of _recompute_decomp, after the force/cache check:
        import time
        _t_decomp_start = time.perf_counter()        
        layers = int(self.spin_layers.value())
        base_sigma = float(self.spin_sigma.value())
        img_id = id(self._image)
        key = (base_sigma, img_id)

        if force or self._cached_key != key or self._cached_layers is None or self._cached_coarse is None:
            self.layers = layers
            self.base_sigma = base_sigma
            img = getattr(self, '_image_work', self._image).astype(np.float32, copy=False)
            is_rgb = (img.ndim == 3 and img.shape[2] >= 3)

            if is_rgb:
                n_ch = img.shape[2]
                channel_imgs = [img[:, :, ch] for ch in range(n_ch)]
                with ThreadPoolExecutor(max_workers=min(n_ch, _N_WORKERS)) as ex:
                    results = list(ex.map(
                        lambda ch: _decompose_channel(channel_imgs[ch], layers, base_sigma),
                        range(n_ch)
                    ))
                H, W = channel_imgs[0].shape
                details_list, coarse_list = [], []
                for k in range(layers):
                    d = np.empty((H, W, n_ch), dtype=np.float32)
                    c = np.empty((H, W, n_ch), dtype=np.float32)
                    for ch in range(n_ch):
                        d[:, :, ch] = results[ch][0][k]
                        # coarse[k] = sum of details[k+1:] + residual
                        c_ch = results[ch][1].copy()
                        for j in range(k + 1, layers):
                            c_ch += results[ch][0][j]
                        c[:, :, ch] = c_ch
                    details_list.append(d)
                    coarse_list.append(c)
                residual = np.empty((H, W, n_ch), dtype=np.float32)
                for ch in range(n_ch):
                    residual[:, :, ch] = results[ch][1]
            else:
                details_list, coarse_list = [], []
                c = img
                for k in range(layers):
                    _t = time.perf_counter()
                    c_next = _blur_gaussian(c, base_sigma * (2 ** k))
                    details_list.append(c - c_next)
                    coarse_list.append(c_next)
                    c = c_next
                residual = c

            self._cached_layers = details_list
            self._cached_coarse = coarse_list
            self._cached_residual = residual
            self._cached_key = key

            # For display-expanded mono (HWC where all 3 channels are identical),
            # only compute sigma on one channel — all three are the same data.
            def _sigma_for_layer(w):
                if not w.size:
                    return 1e-6
                if w.ndim == 3:
                    return _robust_sigma(w[:, :, 0])  # all channels identical for mono
                return _robust_sigma(w)

            self._layer_noise = [_sigma_for_layer(w) for w in self._cached_layers]

            self._sync_cfgs_and_ui()

            return

        old_layers = len(self._cached_layers)
        self.layers = layers
        self.base_sigma = base_sigma

        if layers == old_layers:
            self._sync_cfgs_and_ui()
            return

        if layers < old_layers:
            self._cached_layers = self._cached_layers[:layers]
            self._cached_coarse = self._cached_coarse[:layers]
            self._layer_noise   = self._layer_noise[:layers]
            self._cached_residual = (
                self._cached_coarse[layers - 1] if layers > 0
                else self._image.astype(np.float32, copy=False)
            )
            self._sync_cfgs_and_ui()
            return

        # Grow
        c = self._cached_residual
        is_rgb = (c.ndim == 3 and c.shape[2] >= 3)
        for k in range(old_layers, layers):
            sigma = base_sigma * (2 ** k)
            if is_rgb:
                n_ch = c.shape[2]
                with ThreadPoolExecutor(max_workers=min(n_ch, _N_WORKERS)) as ex:
                    blurred = list(ex.map(
                        lambda ch: _blur_gaussian(c[:, :, ch], sigma),
                        range(n_ch)
                    ))
                c_next = np.stack(blurred, axis=2).astype(np.float32, copy=False)
            else:
                c_next = _blur_gaussian(c, sigma)
            w = c - c_next
            self._cached_layers.append(w)
            self._cached_coarse.append(c_next)
            self._layer_noise.append(_robust_sigma(w) if w.size else 1e-6)
            c = c_next
        self._cached_residual = c
        self._sync_cfgs_and_ui()

    def _sync_cfgs_and_ui(self):
        # ensure cfg list matches layer count (your existing logic, just moved)
        if len(self.cfgs) != self.layers:
            old = self.cfgs[:]
            self.cfgs = [LayerCfg() for _ in range(self.layers)]
            for i in range(min(len(old), self.layers)):
                self.cfgs[i] = old[i]

        self._rebuild_table()
        self._refresh_preview_combo()

    def _build_tuned_layers(self):
        self._recompute_decomp(force=False)

        details = self._cached_layers
        residual = self._cached_residual
        if details is None or residual is None:
            return None, None

        mode = self.combo_mode.currentText()

        def do_one(i_w):
            i, w = i_w
            cfg = self.cfgs[i]
            if not cfg.enabled:
                return i, np.zeros_like(w)

            layer_sigma = self.base_sigma * (2 ** i)

            sigma = self._layer_noise[i] if self._layer_noise and i < len(self._layer_noise) else None
            out = apply_layer_ops(
                w,
                cfg.bias_gain,
                cfg.thr,
                cfg.amount,
                cfg.denoise,
                sigma,
                layer_index=i,
                mode=mode,
            )
            return i, out

        n = len(details)
        if n == 0:
            return [], residual

        max_workers = min(os.cpu_count() or 4, n)

        tuned = [None] * n
        # ThreadPoolExecutor is fine here because apply_layer_ops is numpy-heavy
        # (but real speed-up depends on GIL/OpenCV/BLAS behavior).
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for i, out in ex.map(do_one, enumerate(details)):
                tuned[i] = out

        return tuned, residual

    def _rebuild_preview(self):
        if getattr(self, "_closing", False):
            return
        self._spinner_on()
        QTimer.singleShot(0, self._rebuild_preview_impl)


    def _rebuild_preview_impl(self):
            if getattr(self, "_closing", False):
                return

            try:
                roi_ok = (
                    getattr(self, "cb_fast_roi_preview", None) is not None
                    and self.cb_fast_roi_preview.isChecked()
                    and not self.pix_base.pixmap().isNull()
                )

                if roi_ok:
                    roi_img, roi_rect = self._compute_preview_roi()
                    if roi_img is None:
                        return
                    self._refresh_pix_roi(roi_img, roi_rect)
                    return

                # Full-frame preview
                tuned, residual = self._build_tuned_layers()
                if tuned is None or residual is None:
                    return

                res = residual if self.residual_enabled else np.zeros_like(residual)
                out_raw = multiscale_reconstruct(tuned, res)
                out = np.clip(out_raw, 0.0, 1.0).astype(np.float32, copy=False)

                sel = self.combo_preview.currentData()
                if sel is None or sel == "final":
                    if not self.residual_enabled:
                        d = out_raw.astype(np.float32, copy=False)
                        vis = np.clip(0.5 + d * 4.0, 0.0, 1.0).astype(np.float32, copy=False)
                        if vis.ndim == 2:
                            vis = np.repeat(vis[:, :, None], 3, axis=2)
                        self._preview_img = vis
                    else:
                        # Autostretch applies here — on the linear reconstruct
                        self._preview_img = self._apply_preview_stretch(out)
                elif sel == "residual":
                    r = np.clip(residual, 0, 1)
                    self._preview_img = self._apply_preview_stretch(r)
                else:
                    # Detail layer view — skip stretch, mid-gray viz is already scaled
                    w = tuned[int(sel)]
                    vis = np.clip(0.5 + (w * 4.0), 0.0, 1.0).astype(np.float32, copy=False)
                    if vis.ndim == 2:
                        vis = np.repeat(vis[:, :, None], 3, axis=2)
                    self._preview_img = vis

                self._refresh_pix()

            finally:
                self._spinner_off()


    def _compute_preview_roi(self):
        vis = self._visible_image_rect()
        if vis is None:
            return None, None

        x0, y0, x1, y1 = vis

        MAX = 1400
        w = x1 - x0
        h = y1 - y0
        if w > MAX:
            cx = (x0 + x1) // 2
            x0 = max(0, cx - MAX // 2)
            x1 = min(self._image.shape[1], x0 + MAX)
        if h > MAX:
            cy = (y0 + y1) // 2
            y0 = max(0, cy - MAX // 2)
            y1 = min(self._image.shape[0], y0 + MAX)

        layers = int(self.spin_layers.value())
        base_sigma = float(self.spin_sigma.value())
        if layers <= 0:
            return None, None

        sigma_max = base_sigma * (2 ** (layers - 1))
        pad = int(np.ceil(3.0 * sigma_max)) + 2

        H, W = self._image.shape[:2]
        px0 = max(0, x0 - pad)
        py0 = max(0, y0 - pad)
        px1 = min(W, x1 + pad)
        py1 = min(H, y1 + pad)

        work = getattr(self, '_image_work', self._image)
        crop = work[py0:py1, px0:px1].astype(np.float32, copy=False)

        details, residual = multiscale_decompose(crop, layers=layers, base_sigma=base_sigma)
        layer_noise = [_robust_sigma(w) if w.size else 1e-6 for w in details]

        mode = self.combo_mode.currentText()

        def do_one(i_w):
            i, w = i_w
            cfg = self.cfgs[i]
            if not cfg.enabled:
                return i, np.zeros_like(w)
            return i, apply_layer_ops(
                w, cfg.bias_gain, cfg.thr, cfg.amount, cfg.denoise,
                layer_noise[i], layer_index=i, mode=mode
            )

        tuned = [None] * len(details)
        max_workers = min(os.cpu_count() or 4, len(details) or 1)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for i, out in ex.map(do_one, enumerate(details)):
                tuned[i] = out

        res = residual if self.residual_enabled else np.zeros_like(residual)
        out_raw = multiscale_reconstruct(tuned, res)

        sel = self.combo_preview.currentData()

        if sel is not None and sel not in ("final", "residual"):
            # Detail layer — skip stretch, mid-gray viz
            w = tuned[int(sel)]
            vis = np.clip(0.5 + (w * 4.0), 0.0, 1.0).astype(np.float32, copy=False)
            if vis.ndim == 2:
                vis = np.repeat(vis[:, :, None], 3, axis=2)
            cx0 = x0 - px0
            cy0 = y0 - py0
            return vis[cy0:cy0+(y1-y0), cx0:cx0+(x1-x0)], (x0, y0, x1, y1)

        if sel == "residual":
            out = np.clip(residual, 0.0, 1.0).astype(np.float32, copy=False)
        else:
            if not self.residual_enabled:
                out = np.clip(0.5 + out_raw * 4.0, 0.0, 1.0).astype(np.float32, copy=False)
            else:
                out = np.clip(out_raw, 0.0, 1.0).astype(np.float32, copy=False)

        # Apply stretch for display (on the ROI crop — fast)
        out = self._apply_preview_stretch(out)

        cx0 = x0 - px0
        cy0 = y0 - py0
        cx1 = cx0 + (x1 - x0)
        cy1 = cy0 + (y1 - y0)

        roi = out[cy0:cy1, cx0:cx1]
        return roi, (x0, y0, x1, y1)


    def _update_param_widgets_for_mode(self):
        linear = (self.combo_mode.currentText() == "Linear")

        # Always allow Gain in both modes
        gain_widgets = (self.spin_gain, self.slider_gain)

        # These are only meaningful in Mean mode
        nonlin_widgets = (
            self.spin_thr, self.slider_thr,
            self.spin_amt, self.slider_amt,
            self.spin_denoise, self.slider_denoise,
        )

        # For residual row we already disable everything in _load_layer_into_editor,
        # so here we just respect the current selection.
        idx = getattr(self, "_selected_layer", None)
        if idx is None or idx == self.layers:
            # Residual – handled in _load_layer_into_editor
            return

        for w in gain_widgets:
            w.setEnabled(True)

        for w in nonlin_widgets:
            w.setEnabled(not linear)

    def _apply_preview_stretch(self, img: np.ndarray) -> np.ndarray:
        """
        Apply autostretch for display only if the toggle is on.
        img is float32 [0..1], either (H,W) 2D or (H,W,3) 3ch.
        Always returns (H,W,3) float32 suitable for QPixmap.
        """
        # Ensure 3ch for display
        if img.ndim == 2:
            img3 = np.repeat(img[:, :, None], 3, axis=2)
        else:
            img3 = img

        if not self.cb_autostretch.isChecked():
            return img3

        try:
            stretched = autostretch(img3, target_median=0.25, linked=False)
            return stretched.astype(np.float32, copy=False)
        except Exception:
            return img3

    def _np_to_qpix(self, img: np.ndarray) -> QPixmap:
        arr = np.ascontiguousarray(np.clip(img * 255.0, 0, 255).astype(np.uint8))
        h, w = arr.shape[:2]
        if arr.ndim == 2:
            arr = np.repeat(arr[:, :, None], 3, axis=2)
        qimg = QImage(arr.data, w, h, 3*w, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qimg)

    def _refresh_pix(self):
        pm = self._np_to_qpix(self._preview_img)
        self.pix_base.setPixmap(pm)
        self.pix_base.setOffset(0, 0)

        # Optional: clear ROI overlay on full refresh
        self.pix_roi.setPixmap(QPixmap())
        self.pix_roi.setOffset(0, 0)

        H, W = self._image.shape[:2]
        self.scene.setSceneRect(QRectF(0, 0, W, H))

    def _fast_preview_enabled(self) -> bool:
        return bool(getattr(self, "cb_fast_roi_preview", None)) and self.cb_fast_roi_preview.isChecked()

    def _invalidate_full_decomp_cache(self):
        self._cached_layers = None
        self._cached_coarse = None
        self._cached_residual = None
        self._cached_key = None
        self._layer_noise = None


    def _fit_view(self):
        if self.pix_base.pixmap().isNull():
            return
        self.view.resetTransform()
        self.view.fitInView(self.pix_base, Qt.AspectRatioMode.KeepAspectRatio)
        self._schedule_roi_preview()

    def _one_to_one(self):
        self.view.resetTransform()
        self._schedule_roi_preview()

    # ---------- Table / layer editing ----------
    def _on_gain_slider_changed(self, v: int):
        # 0..300 -> 0.00..3.00
        val = v / 100.0
        self.spin_gain.blockSignals(True)
        self.spin_gain.setValue(val)
        self.spin_gain.blockSignals(False)
        self._on_layer_editor_changed()

    def _on_thr_slider_changed(self, v: int):
        # 0..1000 -> 0.00..10.00 σ
        val = v / 100.0
        self.spin_thr.blockSignals(True)
        self.spin_thr.setValue(val)
        self.spin_thr.blockSignals(False)
        self._on_layer_editor_changed()


    def _on_amt_slider_changed(self, v: int):
        # 0..100 -> 0.00..1.00
        val = v / 100.0
        self.spin_amt.blockSignals(True)
        self.spin_amt.setValue(val)
        self.spin_amt.blockSignals(False)
        self._on_layer_editor_changed()

    def _on_dn_slider_changed(self, v: int):
        # 0..100 -> 0.00..1.00
        val = v / 100.0
        self.spin_denoise.blockSignals(True)
        self.spin_denoise.setValue(val)
        self.spin_denoise.blockSignals(False)
        self._on_layer_editor_changed()


    def _rebuild_table(self):
        self.table.blockSignals(True)
        try:
            # +1 row for residual ("R")
            self.table.setRowCount(self.layers + 1)

            # detail rows
            for i in range(self.layers):
                cfg = self.cfgs[i]

                item_on = QTableWidgetItem("")
                item_on.setFlags(item_on.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item_on.setCheckState(Qt.CheckState.Checked if cfg.enabled else Qt.CheckState.Unchecked)
                self.table.setItem(i, 0, item_on)

                self.table.setItem(i, 1, QTableWidgetItem(str(i + 1)))
                self.table.setItem(i, 2, QTableWidgetItem(f"{self.base_sigma * (2**i):.2f}"))
                self.table.setItem(i, 3, QTableWidgetItem(f"{cfg.bias_gain:.2f}"))
                self.table.setItem(i, 4, QTableWidgetItem(f"{cfg.thr:.2f}"))   # N·σ
                self.table.setItem(i, 5, QTableWidgetItem(f"{cfg.amount:.2f}"))
                self.table.setItem(i, 6, QTableWidgetItem(f"{cfg.denoise:.2f}"))

                self.table.setItem(i, 7, QTableWidgetItem("D"))

            # residual row
            r = self.layers
            item_on = QTableWidgetItem("")
            item_on.setFlags(item_on.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item_on.setCheckState(
                Qt.CheckState.Checked if self.residual_enabled else Qt.CheckState.Unchecked
            )
            self.table.setItem(r, 0, item_on)

            self.table.setItem(r, 1, QTableWidgetItem("R"))
            self.table.setItem(r, 2, QTableWidgetItem("—"))
            self.table.setItem(r, 3, QTableWidgetItem("1.00"))
            self.table.setItem(r, 4, QTableWidgetItem("0.0000"))
            self.table.setItem(r, 5, QTableWidgetItem("0.00"))
            self.table.setItem(r, 6, QTableWidgetItem("0.00"))
            self.table.setItem(r, 7, QTableWidgetItem("R"))

        finally:
            self.table.blockSignals(False)

        # connect once (avoid stacking connects)
        try:
            self.table.itemChanged.disconnect(self._on_table_item_changed)
        except Exception:
            pass
        self.table.itemChanged.connect(self._on_table_item_changed)

        if self.layers > 0 and not self.table.selectedItems():
            self.table.selectRow(0)
            self._load_layer_into_editor(0)

    def _on_table_item_changed(self, item: QTableWidgetItem):
        r, c = item.row(), item.column()

        # Residual row
        if r == self.layers:
            if c == 0:
                self.residual_enabled = (item.checkState() == Qt.CheckState.Checked)
                self._schedule_preview()
            # ignore other edits for residual
            return

        if not (0 <= r < len(self.cfgs)):
            return

        cfg = self.cfgs[r]

        if c == 0:
            # On/off
            cfg.enabled = (item.checkState() == Qt.CheckState.Checked)
            self._schedule_preview()
            return

        # numeric columns: Gain(3), Thr(4), Amt(5), NR(6)
        try:
            text = item.text().strip()
            val = float(text) if text else 0.0
        except Exception:
            return

        if c == 3:
            cfg.bias_gain = val
        elif c == 4:
            cfg.thr = val
        elif c == 5:
            cfg.amount = val
        elif c == 6:
            cfg.denoise = val
        else:
            return

        # If this row is currently selected, update editor widgets too
        if getattr(self, "_selected_layer", None) == r:
            self._load_layer_into_editor(r)

        self._schedule_preview()

    @contextmanager
    def _busy_popup(self, text: str):
        dlg = QProgressDialog(text, "", 0, 0, self)
        dlg.setWindowTitle("Multiscale Decomposition")
        dlg.setWindowModality(Qt.WindowModality.ApplicationModal)
        dlg.setCancelButton(None)
        dlg.setMinimumDuration(0)
        dlg.show()

        self._spinner_on()
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QApplication.processEvents()

        try:
            yield dlg
        finally:
            try:
                dlg.close()
            except Exception:
                pass
            QApplication.restoreOverrideCursor()
            self._spinner_off()
            QApplication.processEvents()

    def _on_table_select(self):
        rows = {it.row() for it in self.table.selectedItems()}
        if not rows:
            return
        r = min(rows)
        self._load_layer_into_editor(r)

    def _load_layer_into_editor(self, idx: int):
        self._selected_layer = idx

        if idx == self.layers:
            self.lbl_sel.setText("Layer: R (Residual)")
            for w in (self.spin_gain, self.spin_thr, self.spin_amt, self.spin_denoise,
                    self.slider_gain, self.slider_thr, self.slider_amt, self.slider_denoise):
                w.setEnabled(False)
            return

        for w in (self.spin_gain, self.spin_thr, self.spin_amt, self.spin_denoise,
                self.slider_gain, self.slider_thr, self.slider_amt, self.slider_denoise):
            w.setEnabled(True)

        cfg = self.cfgs[idx]
        self.lbl_sel.setText(f"Layer: {idx+1} / {self.layers}")

        # spins + sliders in sync
        self.spin_gain.blockSignals(True)
        self.spin_thr.blockSignals(True)
        self.spin_amt.blockSignals(True)
        self.spin_denoise.blockSignals(True)

        self.slider_gain.blockSignals(True)
        self.slider_thr.blockSignals(True)
        self.slider_amt.blockSignals(True)
        self.slider_denoise.blockSignals(True)
        try:
            self.spin_gain.setValue(cfg.bias_gain)
            self.spin_thr.setValue(cfg.thr)           # thr is N·σ now
            self.spin_amt.setValue(cfg.amount)
            self.spin_denoise.setValue(cfg.denoise)

            self.slider_gain.setValue(int(round(cfg.bias_gain * 100.0)))
            self.slider_thr.setValue(int(round(cfg.thr * 100.0)))   # N·σ → 0..1000
            self.slider_amt.setValue(int(round(cfg.amount * 100.0)))
            self.slider_denoise.setValue(int(round(cfg.denoise * 100.0)))
        finally:
            self.spin_gain.blockSignals(False)
            self.spin_thr.blockSignals(False)
            self.spin_amt.blockSignals(False)
            self.spin_denoise.blockSignals(False)
            self.slider_gain.blockSignals(False)
            self.slider_thr.blockSignals(False)
            self.slider_amt.blockSignals(False)
            self.slider_denoise.blockSignals(False)
            self._update_param_widgets_for_mode()



    def _on_layer_editor_changed(self):
        idx = getattr(self, "_selected_layer", None)
        if idx is None or not (0 <= idx < len(self.cfgs)):
            return
        cfg = self.cfgs[idx]
        cfg.bias_gain = float(self.spin_gain.value())
        cfg.thr = float(self.spin_thr.value())
        cfg.amount = float(self.spin_amt.value())
        cfg.denoise = float(self.spin_denoise.value())

        # keep table in sync
        self.table.blockSignals(True)
        try:
            self.table.item(idx, 3).setText(f"{cfg.bias_gain:.2f}")
            self.table.item(idx, 4).setText(f"{cfg.thr:.2f}")      # N·σ
            self.table.item(idx, 5).setText(f"{cfg.amount:.2f}")
            self.table.item(idx, 6).setText(f"{cfg.denoise:.2f}")

        finally:
            self.table.blockSignals(False)

        self._schedule_preview()

    def _on_layers_changed(self):
        # Always update counts/UI
        self.layers = int(self.spin_layers.value())

        # Ensure cfgs length matches new layer count and table/combos update
        self._sync_cfgs_and_ui()

        if self._fast_preview_enabled():
            # Do NOT recompute full pyramid here; ROI preview will compute on-demand
            self._invalidate_full_decomp_cache()
            self._schedule_roi_preview()
            return

        # Old behavior for non-ROI mode
        self._recompute_decomp(force=True)
        self._schedule_preview()


    def _on_global_changed(self):
        self.base_sigma = float(self.spin_sigma.value())

        # Update table scale column text (it uses self.base_sigma)
        self._sync_cfgs_and_ui()

        if self._fast_preview_enabled():
            self._invalidate_full_decomp_cache()
            self._schedule_roi_preview()
            return

        self._recompute_decomp(force=True)
        self._schedule_preview()

    def _refresh_preview_combo(self):
        self.combo_preview.blockSignals(True)
        try:
            self.combo_preview.clear()
            self.combo_preview.addItem("Final", userData="final")
            self.combo_preview.addItem("R (Residual)", userData="residual")
            for i in range(self.layers):
                self.combo_preview.addItem(f"Detail Layer {i+1}", userData=i)
        finally:
            self.combo_preview.blockSignals(False)

    def _visible_image_rect(self) -> tuple[int, int, int, int] | None:
        # Use full image rect, NOT the pixmap bounds
        H, W = self._image.shape[:2]
        full_item_rect_scene = QRectF(0, 0, W, H)

        vr = self.view.viewport().rect()
        tl = self.view.mapToScene(vr.topLeft())
        br = self.view.mapToScene(vr.bottomRight())
        scene_rect = QRectF(tl, br).normalized()

        inter = scene_rect.intersected(full_item_rect_scene)
        if inter.isEmpty():
            return None

        x0 = int(np.floor(inter.left()))
        y0 = int(np.floor(inter.top()))
        x1 = int(np.ceil(inter.right()))
        y1 = int(np.ceil(inter.bottom()))

        x0 = max(0, min(W, x0))
        x1 = max(0, min(W, x1))
        y0 = max(0, min(H, y0))
        y1 = max(0, min(H, y1))

        if x1 <= x0 or y1 <= y0:
            return None
        return (x0, y0, x1, y1)

    def _np_to_qpix_roi_comp(self, img_rgb01: np.ndarray) -> QPixmap:
        """
        img_rgb01 is float32 RGB [0..1]
        """
        arr = np.ascontiguousarray(np.clip(img_rgb01 * 255.0, 0, 255).astype(np.uint8))
        h, w = arr.shape[:2]
        if arr.ndim == 2:
            arr = np.repeat(arr[:, :, None], 3, axis=2)

        bytes_per_line = arr.strides[0]
        qimg = QImage(arr.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qimg.copy())  # copy to detach from numpy buffer

    def _refresh_pix_roi(self, roi_img01: np.ndarray, roi_rect: tuple[int,int,int,int]):
        x0, y0, x1, y1 = roi_rect
        pm = self._np_to_qpix_roi_comp(roi_img01)

        self.pix_roi.setPixmap(pm)
        self.pix_roi.setOffset(x0, y0)

        # Keep scene bounds as full image, not ROI
        H, W = self._image.shape[:2]
        self.scene.setSceneRect(QRectF(0, 0, W, H))


    def _build_preview_roi(self):
        vis = self._visible_image_rect()
        if vis is None:
            return None

        x0,y0,x1,y1 = vis
        layers = int(self.spin_layers.value())
        base_sigma = float(self.spin_sigma.value())

        if layers <= 0:
            return None

        sigma_max = base_sigma * (2 ** (layers - 1))
        pad = int(np.ceil(3.0 * sigma_max)) + 2

        H, W = self._image.shape[:2]
        px0 = max(0, x0 - pad); py0 = max(0, y0 - pad)
        px1 = min(W, x1 + pad); py1 = min(H, y1 + pad)

        crop = self._image[py0:py1, px0:px1].astype(np.float32, copy=False)

        # Decompose crop
        details, residual = multiscale_decompose(crop, layers=layers, base_sigma=base_sigma)

        # noise per layer (crop-based) — good enough for preview
        layer_noise = [_robust_sigma(w) if w.size else 1e-6 for w in details]

        # Apply tuning per layer (can thread this like we discussed)
        mode = self.combo_mode.currentText()
        tuned = []
        for i,w in enumerate(details):
            cfg = self.cfgs[i]
            if not cfg.enabled:
                tuned.append(np.zeros_like(w))
            else:
                tuned.append(apply_layer_ops(w, cfg.bias_gain, cfg.thr, cfg.amount, cfg.denoise,
                                            layer_noise[i], mode=mode))

        res = residual if self.residual_enabled else np.zeros_like(residual)
        out_raw = multiscale_reconstruct(tuned, res)

        # Match your preview rules
        if not self.residual_enabled:
            out = np.clip(0.5 + out_raw * 4.0, 0.0, 1.0).astype(np.float32, copy=False)
        else:
            out = np.clip(out_raw, 0.0, 1.0).astype(np.float32, copy=False)

        # Crop back from padded-crop coords to visible ROI coords
        cx0 = x0 - px0; cy0 = y0 - py0
        cx1 = cx0 + (x1 - x0); cy1 = cy0 + (y1 - y0)
        return out[cy0:cy1, cx0:cx1], (x0,y0,x1,y1)


    # ---------- Apply to doc ----------

    def _commit_to_doc(self):
        try: self._save_window_state()
        except Exception: pass
        with self._busy_popup("Applying multiscale result to document…"):
            tuned, residual = self._build_tuned_layers()
            if tuned is None or residual is None:
                return
 
            res = residual if self.residual_enabled else np.zeros_like(residual)
            out_raw = multiscale_reconstruct(tuned, res)
 
            if not self.residual_enabled:
                d = out_raw.astype(np.float32, copy=False)
                out = np.clip(0.5 + d * 4.0, 0.0, 1.0).astype(np.float32, copy=False)
            else:
                out = np.clip(out_raw, 0.0, 1.0).astype(np.float32, copy=False)
 
            # out may be 2D (H,W) if we processed _image_work (mono)
            # or 3D (H,W,3) if RGB.
            # We need to restore to the original document shape.
            if self._orig_mono:
                # Flatten to 2D regardless of what came out
                if out.ndim == 3:
                    out2d = out[:, :, 0]
                else:
                    out2d = out
                # Restore to original mono shape
                if len(self._orig_shape) == 2:
                    out_final = out2d.astype(np.float32, copy=False)
                else:
                    # original was (H, W, 1)
                    out_final = out2d[:, :, None].astype(np.float32, copy=False)
            else:
                # RGB — ensure (H, W, 3)
                if out.ndim == 2:
                    out_final = np.repeat(out[:, :, None], 3, axis=2).astype(np.float32, copy=False)
                else:
                    out_final = out[:, :, :3].astype(np.float32, copy=False)
 
            try:
                if hasattr(self._doc, "set_image"):
                    self._doc.set_image(out_final, step_name="Multiscale Decomposition")
                elif hasattr(self._doc, "apply_numpy"):
                    self._doc.apply_numpy(out_final, step_name="Multiscale Decomposition")
                else:
                    self._doc.image = out_final
            except Exception as e:
                QMessageBox.critical(self, "Multiscale Decomposition", f"Failed to write to document:\n{e}")
                return
 
            if hasattr(self.parent(), "_refresh_active_view"):
                try:
                    self.parent()._refresh_active_view()
                except Exception:
                    pass
 
            self.accept()


    def _send_detail_to_new_doc(self):
        with self._busy_popup("Creating new document from multiscale result…"):
            self._recompute_decomp(force=False)
 
            details = self._cached_layers
            residual = self._cached_residual
            if details is None or residual is None:
                return
 
            dm = self._get_doc_manager()
            if dm is None:
                QMessageBox.warning(self, "Multiscale Decomposition",
                                    "No DocManager available to create a new document.")
                return
 
            mode = self.combo_mode.currentText()
            tuned = []
            for i, w in enumerate(details):
                cfg = self.cfgs[i]
                if not cfg.enabled:
                    tuned.append(np.zeros_like(w))
                else:
                    sigma = (self._layer_noise[i]
                             if self._layer_noise is not None and i < len(self._layer_noise)
                             else None)
                    tuned.append(apply_layer_ops(w, cfg.bias_gain, cfg.thr, cfg.amount,
                                                 cfg.denoise, sigma, mode=mode))
 
            res = residual if self.residual_enabled else np.zeros_like(residual)
            out_raw = multiscale_reconstruct(tuned, res)
 
            if not self.residual_enabled:
                d = out_raw.astype(np.float32, copy=False)
                out = np.clip(0.5 + d * 4.0, 0.0, 1.0).astype(np.float32, copy=False)
            else:
                out = np.clip(out_raw, 0.0, 1.0).astype(np.float32, copy=False)
 
            # Restore to original document shape
            if self._orig_mono:
                if out.ndim == 3:
                    out2d = out[:, :, 0]
                else:
                    out2d = out
                if len(self._orig_shape) == 2:
                    out_final = out2d.astype(np.float32, copy=False)
                else:
                    out_final = out2d[:, :, None].astype(np.float32, copy=False)
            else:
                if out.ndim == 2:
                    out_final = np.repeat(out[:, :, None], 3, axis=2).astype(np.float32, copy=False)
                else:
                    out_final = out[:, :, :3].astype(np.float32, copy=False)
 
            title = "Multiscale Result"
            meta = self._build_new_doc_metadata(title, out_final)
            try:
                dm.create_document(out_final, metadata=meta, name=title)
            except Exception as e:
                QMessageBox.critical(self, "Multiscale Decomposition",
                                     f"Failed to create new document:\n{e}")

    def _split_layers_to_docs(self):
        """
        Create a new document for each tuned detail layer *and* the residual.

        - Detail layers use the same mid-gray visualization as the per-layer preview:
              vis = 0.5 + layer*4.0
        - Residual layer is just the residual itself (0..1 clipped).
        """
        with self._busy_popup("Splitting layers into documents…") as prog:    
            self._recompute_decomp(force=False)

            details = self._cached_layers
            residual = self._cached_residual
            if details is None or residual is None:
                return

            dm = self._get_doc_manager()
            if dm is None:
                QMessageBox.warning(
                    self,
                    "Multiscale Decomposition",
                    "No DocManager available to create new documents."
                )
                return

            mode = self.combo_mode.currentText()
            # Build tuned layers just like everywhere else
            tuned = []
            for i, w in enumerate(details):
                cfg = self.cfgs[i]
                if not cfg.enabled:
                    tuned.append(np.zeros_like(w))
                else:
                    sigma = None
                    if self._layer_noise is not None and i < len(self._layer_noise):
                        sigma = self._layer_noise[i]
                    tuned.append(
                        apply_layer_ops(
                            w,
                            cfg.bias_gain,
                            cfg.thr,
                            cfg.amount,
                            cfg.denoise,
                            sigma,
                            mode=mode,
                        )
                    )

            # ---- 1) Detail layers ------------------------------------------
            for i, layer in enumerate(tuned):
                d = layer.astype(np.float32, copy=False)
                vis = np.clip(0.5 + d * 4.0, 0.0, 1.0).astype(np.float32, copy=False)

                if self._orig_mono:
                    mono = vis[..., 0]
                    if len(self._orig_shape) == 3 and self._orig_shape[2] == 1:
                        mono = mono[:, :, None]
                    out_final = mono.astype(np.float32, copy=False)
                else:
                    out_final = vis

                title = f"Multiscale Detail Layer {i+1}"
                meta = self._build_new_doc_metadata(title, out_final)

                try:
                    dm.create_document(out_final, metadata=meta, name=title)
                except Exception as e:
                    QMessageBox.critical(
                        self,
                        "Multiscale Decomposition",
                        f"Failed to create document for layer {i+1}:\n{e}"
                    )
                    # Don’t bail entirely on first error if you’d rather continue;
                    # right now we stop on first hard failure.
                    return

            # ---- 2) Residual layer -----------------------------------------
            try:
                res = residual.astype(np.float32, copy=False)
                res_img = np.clip(res, 0.0, 1.0)

                if self._orig_mono:
                    mono = res_img[..., 0]
                    if len(self._orig_shape) == 3 and self._orig_shape[2] == 1:
                        mono = mono[:, :, None]
                    res_final = mono.astype(np.float32, copy=False)
                else:
                    res_final = res_img

                r_title = "Multiscale Residual Layer"
                r_meta = self._build_new_doc_metadata(r_title, res_final)

                dm.create_document(res_final, metadata=r_meta, name=r_title)
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Multiscale Decomposition",
                    f"Failed to create residual-layer document:\n{e}"
                )



    def _get_doc_manager(self):
        """
        Best-effort: find the DocManager that owns the source document.
        Prefer the doc's own _doc_manager; fall back to parent.doc_manager.
        """
        doc = getattr(self, "_doc", None)
        dm = getattr(doc, "_doc_manager", None) if doc is not None else None

        if dm is None:
            parent = self.parent()
            dm = getattr(parent, "doc_manager", None) if parent is not None else None

        return dm

    def _build_new_doc_metadata(self, title: str, img: np.ndarray) -> dict:
        """
        Clone the source document's metadata and sanitize it for a brand-new doc.
        """
        base_doc = getattr(self, "_doc", None)
        base_meta = getattr(base_doc, "metadata", {}) or {}
        meta = dict(base_meta)

        # New display name
        if title:
            meta["display_name"] = title

        # Drop things that make it look linked/preview/ROI
        imi = dict(meta.get("image_meta") or {})
        for k in ("readonly", "view_kind", "derived_from", "layer", "layer_index", "linked"):
            imi.pop(k, None)
        meta["image_meta"] = imi

        # Remove any ROI-ish keys
        for k in list(meta.keys()):
            if k.startswith("_roi_") or k.endswith("_roi") or k == "roi":
                meta.pop(k, None)

        # For a brand-new doc, don't keep the original file_path
        meta.pop("file_path", None)

        # Normalize mono flag
        if isinstance(img, np.ndarray):
            meta["is_mono"] = (img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1))

        # Keep bit depth / headers / WCS as-is; DocManager.open_array() will
        # ensure bit_depth etc. are sane.
        return meta

    def _restore_window_state(self):
        try:
            s = QSettings()
            g = s.value("msdecomp/window_geometry", None)
            if g is not None:
                self.restoreGeometry(g)

            # splitter
            sp = s.value("msdecomp/splitter_state", None)
            if sp is not None and hasattr(self, "splitter"):
                self.splitter.restoreState(sp)

            # table header (column widths/order)
            hdr = self.table.horizontalHeader() if hasattr(self, "table") else None
            hs = s.value("msdecomp/table_header_state", None)
            if hs is not None and hdr is not None:
                hdr.restoreState(hs)

        except Exception:
            pass


    def _save_window_state(self):
        try:
            s = QSettings()
            s.setValue("msdecomp/window_geometry", self.saveGeometry())

            if hasattr(self, "splitter"):
                s.setValue("msdecomp/splitter_state", self.splitter.saveState())

            hdr = self.table.horizontalHeader() if hasattr(self, "table") else None
            if hdr is not None:
                s.setValue("msdecomp/table_header_state", hdr.saveState())
        except Exception:
            pass

    def showEvent(self, ev):
        super().showEvent(ev)
        if getattr(self, "_geom_restored", False):
            return
        self._geom_restored = True

        def _after():
            self._restore_window_state()
            # keep your existing behavior:
            self._fit_view()
        QTimer.singleShot(0, _after)

    def closeEvent(self, ev):
        self._closing = True
        try:
            self._save_window_state()
        except Exception:
            pass

        try:
            if hasattr(self, "_preview_timer"):
                self._preview_timer.stop()
            if hasattr(self, "_busy_show_timer"):
                self._busy_show_timer.stop()
            try:
                self.view.horizontalScrollBar().valueChanged.disconnect(self._schedule_roi_preview)
                self.view.verticalScrollBar().valueChanged.disconnect(self._schedule_roi_preview)
            except Exception:
                pass
        except Exception:
            pass

        super().closeEvent(ev)

    def _on_close_clicked(self):
        try: self._save_window_state()
        except Exception: pass
        self.reject()

class _MultiScaleDecompPresetDialog(QDialog):
    """
    Preset editor for Multiscale Decomposition (headless + shortcuts).
    """
    def __init__(self, parent=None, initial: dict | None = None):
        super().__init__(parent)
        self.setWindowTitle("Multiscale Decomposition — Preset")
        init = dict(initial or {})

        v = QVBoxLayout(self)

        # ---- Global ----
        gb = QGroupBox("Global")
        form = QFormLayout(gb)

        self.sp_layers = QSpinBox()
        self.sp_layers.setRange(1, 10)
        self.sp_layers.setValue(int(init.get("layers", 4)))

        self.sp_sigma = QDoubleSpinBox()
        self.sp_sigma.setRange(0.3, 5.0)
        self.sp_sigma.setDecimals(2)
        self.sp_sigma.setSingleStep(0.1)
        self.sp_sigma.setValue(float(init.get("base_sigma", 1.0)))

        self.cb_linked = QCheckBox("Linked RGB channels")
        self.cb_linked.setChecked(bool(init.get("linked_rgb", True)))

        form.addRow("Layers:", self.sp_layers)
        form.addRow("Base sigma:", self.sp_sigma)
        form.addRow("", self.cb_linked)

        v.addWidget(gb)

        # ---- Layers ----
        gb_layers = QGroupBox("Per-Layer Settings")
        lv = QVBoxLayout(gb_layers)

        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["On", "Layer", "Gain", "Thr (σ)", "Amount", "Denoise"]
        )

        self.table.verticalHeader().setVisible(False)
        lv.addWidget(self.table)

        v.addWidget(gb_layers)

        # ---- Buttons ----
        btns = QHBoxLayout()
        ok = QPushButton("OK")
        cancel = QPushButton("Cancel")
        btns.addStretch(1)
        btns.addWidget(ok)
        btns.addWidget(cancel)
        v.addLayout(btns)

        ok.clicked.connect(self.accept)
        cancel.clicked.connect(self.reject)

        self._populate_table(init)

    def _populate_table(self, init: dict):
        layers = int(self.sp_layers.value())
        cfgs = init.get("layers_cfg", [])

        self.table.setRowCount(layers)

        for i in range(layers):
            cfg = cfgs[i] if i < len(cfgs) else {}

            # Enabled
            chk = QTableWidgetItem("")
            chk.setFlags(chk.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            chk.setCheckState(
                Qt.CheckState.Checked if cfg.get("enabled", True)
                else Qt.CheckState.Unchecked
            )
            self.table.setItem(i, 0, chk)

            self.table.setItem(i, 1, QTableWidgetItem(str(i + 1)))
            self.table.setItem(i, 2, QTableWidgetItem(f"{float(cfg.get('gain',   1.0)):.2f}"))
            self.table.setItem(i, 3, QTableWidgetItem(f"{float(cfg.get('thr',    0.0)):.2f}"))  # N·σ
            self.table.setItem(i, 4, QTableWidgetItem(f"{float(cfg.get('amount', 0.0)):.2f}"))
            self.table.setItem(i, 5, QTableWidgetItem(f"{float(cfg.get('denoise',0.0)):.2f}"))



    def result_dict(self) -> dict:
        layers = int(self.sp_layers.value())
        out_layers = []

        for r in range(layers):
            enabled = self.table.item(r, 0).checkState() == Qt.CheckState.Checked
            gain = float(self.table.item(r, 2).text())
            thr = float(self.table.item(r, 3).text())
            amt = float(self.table.item(r, 4).text())
            try:
                dn = float(self.table.item(r, 5).text())
            except Exception:
                dn = 0.0

            out_layers.append({
                "enabled": enabled,
                "gain": gain,
                "thr": thr,
                "amount": amt,
                "denoise": dn,
            })


        return {
            "layers": layers,
            "base_sigma": float(self.sp_sigma.value()),
            "linked_rgb": bool(self.cb_linked.isChecked()),
            "layers_cfg": out_layers,
        }
def closeEvent(self, ev):
    self._closing = True
    try:
        if hasattr(self, "_preview_timer"):
            self._preview_timer.stop()
        if hasattr(self, "_busy_show_timer"):
            self._busy_show_timer.stop()
        # Optional: disconnect scrollbars to stop ROI scheduling
        try:
            self.view.horizontalScrollBar().valueChanged.disconnect(self._schedule_roi_preview)
            self.view.verticalScrollBar().valueChanged.disconnect(self._schedule_roi_preview)
        except Exception:
            pass
    except Exception:
        pass
    super().closeEvent(ev)
