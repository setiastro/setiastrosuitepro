# pro/wavescalede.py
from __future__ import annotations
import math
import numpy as np

from PyQt6.QtCore import Qt, QObject, pyqtSignal, QThread, QTimer
from PyQt6.QtGui import QImage, QPixmap, QIcon, QWheelEvent
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout, QLabel, QPushButton,
    QSlider, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QScrollArea,
    QMessageBox, QProgressBar, QMainWindow
)

# ─────────────────────────────────────────────────────────────────────────────
# Optional Numba color-space acceleration from legacy.numba_utils
# ─────────────────────────────────────────────────────────────────────────────
try:
    from legacy.numba_utils import (
        rgb_to_xyz_numba, xyz_to_lab_numba,
        lab_to_xyz_numba,  xyz_to_rgb_numba,
    )
    _HAVE_NUMBA = True
except Exception:
    _HAVE_NUMBA = False

# ─────────────────────────────────────────────────────────────────────────────
# Convolution (+gaussian fallback if SciPy is missing)
# ─────────────────────────────────────────────────────────────────────────────
try:
    from scipy.ndimage import convolve as _nd_convolve
    from scipy.ndimage import gaussian_filter as _nd_gauss

    def _conv_sep_reflect(image2d: np.ndarray, k1d: np.ndarray, axis: int) -> np.ndarray:
        if axis == 1:  # x
            return _nd_convolve(image2d, k1d.reshape(1, -1), mode="reflect")
        else:          # y
            return _nd_convolve(image2d, k1d.reshape(-1, 1), mode="reflect")

    def _gauss_blur(image2d: np.ndarray, sigma: float) -> np.ndarray:
        return _nd_gauss(image2d, sigma=sigma, mode="reflect")
except Exception:
    def _conv_sep_reflect(image2d: np.ndarray, k1d: np.ndarray, axis: int) -> np.ndarray:
        image2d = np.asarray(image2d, dtype=np.float32)
        k1d = np.asarray(k1d, dtype=np.float32)
        r = len(k1d) // 2
        if axis == 1:  # horizontal
            pad = np.pad(image2d, ((0, 0), (r, r)), mode="reflect")
            out = np.empty_like(image2d, dtype=np.float32)
            for i in range(image2d.shape[0]):
                out[i] = np.convolve(pad[i], k1d, mode="valid")
            return out
        else:          # vertical
            pad = np.pad(image2d, ((r, r), (0, 0)), mode="reflect")
            out = np.empty_like(image2d, dtype=np.float32)
            for j in range(image2d.shape[1]):
                out[:, j] = np.convolve(pad[:, j], k1d, mode="valid")
            return out

    def _gauss1d(sigma: float) -> np.ndarray:
        if sigma <= 0:
            return np.array([1.0], dtype=np.float32)
        # pragmatic kernel length
        radius = max(1, int(round(3.0 * sigma)))
        x = np.arange(-radius, radius + 1, dtype=np.float32)
        k = np.exp(-0.5 * (x / sigma)**2)
        k /= np.sum(k)
        return k.astype(np.float32)

    def _gauss_blur(image2d: np.ndarray, sigma: float) -> np.ndarray:
        k = _gauss1d(float(sigma))
        tmp = _conv_sep_reflect(image2d, k, axis=1)
        return _conv_sep_reflect(tmp, k, axis=0)

# ─────────────────────────────────────────────────────────────────────────────
# Core math (shared)
# ─────────────────────────────────────────────────────────────────────────────
_B3 = (np.array([1, 4, 6, 4, 1], dtype=np.float32) / 16.0)

def _build_spaced_kernel(kernel: np.ndarray, scale_idx: int) -> np.ndarray:
    if scale_idx == 0:
        return kernel.astype(np.float32, copy=False)
    step = 2 ** scale_idx
    spaced_len = len(kernel) + (len(kernel) - 1) * (step - 1)
    spaced = np.zeros(spaced_len, dtype=np.float32)
    spaced[0::step] = kernel
    return spaced

def _atrous_decompose(img2d: np.ndarray, n_scales: int, base_k: np.ndarray) -> list[np.ndarray]:
    current = img2d.astype(np.float32, copy=True)
    planes: list[np.ndarray] = []
    for s in range(n_scales):
        k = _build_spaced_kernel(base_k, s)
        tmp = _conv_sep_reflect(current, k, axis=1)
        smooth = _conv_sep_reflect(tmp, k, axis=0)
        planes.append(current - smooth)
        current = smooth
    planes.append(current)  # residual
    return planes

def _atrous_reconstruct(planes: list[np.ndarray]) -> np.ndarray:
    out = planes[-1].astype(np.float32, copy=True)
    for w in planes[:-1]:
        out += w
    return out

def _resize_mask_nn(mask2d: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    H, W = target_hw
    if mask2d.shape == (H, W):
        return mask2d.astype(np.float32, copy=False)
    yi = (np.linspace(0, mask2d.shape[0] - 1, H)).astype(np.int32)
    xi = (np.linspace(0, mask2d.shape[1] - 1, W)).astype(np.int32)
    return mask2d[yi][:, xi].astype(np.float32, copy=False)

# Color space helpers
def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    if _HAVE_NUMBA:
        xyz = rgb_to_xyz_numba(np.ascontiguousarray(rgb.astype(np.float32)))
        return xyz_to_lab_numba(xyz)
    # numpy fallback
    M = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]], dtype=np.float32)
    rgb = np.clip(rgb, 0.0, 1.0).astype(np.float32, copy=False)
    xyz = rgb.reshape(-1, 3) @ M.T
    xyz = xyz.reshape(rgb.shape); xyz[...,0] /= 0.95047; xyz[...,2] /= 1.08883
    delta = 6/29
    def f(t): return np.where(t > delta**3, np.cbrt(t), (t/(3*delta**2)) + (4/29))
    fx, fy, fz = f(xyz[...,0]), f(xyz[...,1]), f(xyz[...,2])
    L = 116*fy - 16; a = 500*(fx - fy); b = 200*(fy - fz)
    return np.stack([L, a, b], axis=-1)

def _lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    if _HAVE_NUMBA:
        xyz = lab_to_xyz_numba(np.ascontiguousarray(lab.astype(np.float32)))
        rgb = xyz_to_rgb_numba(xyz)
        return rgb.astype(np.float32, copy=False)
    M_inv = np.array([[ 3.2404542, -1.5371385, -0.4985314],
                      [-0.9692660,  1.8760108,  0.0415560],
                      [ 0.0556434, -0.2040259,  1.0572252]], dtype=np.float32)
    delta = 6/29
    fy = (lab[...,0] + 16.0)/116.0
    fx = fy + lab[...,1]/500.0
    fz = fy - lab[...,2]/200.0
    def finv(t): return np.where(t > delta, t**3, 3*delta**2*(t - 4/29))
    X = 0.95047*finv(fx); Y = finv(fy); Z = 1.08883*finv(fz)
    xyz = np.stack([X, Y, Z], axis=-1)
    rgb = xyz.reshape(-1,3) @ M_inv.T
    rgb = rgb.reshape(xyz.shape)
    return np.clip(rgb, 0.0, 1.0).astype(np.float32, copy=False)

# Darkness mask (scales 2–4, negative parts, mean → normalize → gamma → smooth → mild S-curve)
def _darkness_mask(L: np.ndarray, n_scales: int, base_k: np.ndarray, gamma: float) -> np.ndarray:
    planes = _atrous_decompose(L, n_scales, base_k)
    # mid-scales: 1:4 (0-based → skip 0)
    sel = planes[1:4]
    neg = [np.clip(-p, 0, None) for p in sel]
    if len(neg) == 0:
        m = np.zeros_like(L, dtype=np.float32)
    else:
        combined = np.mean(neg, axis=0).astype(np.float32)
        denom = float(np.max(combined) + 1e-8)
        m = combined / denom
    if gamma != 1.0:
        m = np.power(m, float(gamma), dtype=np.float32)
    m = _gauss_blur(m, sigma=3.0).astype(np.float32)
    # gentle brighten of mids
    m = np.clip(1.5 * m - 0.5 * (m * m), 0.0, 1.0).astype(np.float32)
    return m

# Main compute (mono or RGB)
def compute_wavescale_dse(image: np.ndarray,
                          n_scales: int = 6,
                          boost_factor: float = 5.0,
                          mask_gamma: float = 1.0,
                          iterations: int = 2,
                          base_kernel: np.ndarray = _B3,
                          decay_rate: float = 0.5,
                          external_mask: np.ndarray | None = None  # ← NEW
                          ) -> tuple[np.ndarray, np.ndarray]:
    """
    WaveScale Dark Enhancer.
    Returns (output_image, darkness_mask_used).
    If external_mask is provided (2-D [0..1]), it will be multiplied into the darkness mask.
    """
    arr = np.asarray(image, dtype=np.float32)

    # normalize external mask now
    ext = None
    if external_mask is not None:
        m = np.asarray(external_mask)
        if m.ndim == 3:  # collapse RGB(A)
            m = m.mean(axis=2)
        m = np.clip(m.astype(np.float32), 0.0, 1.0)
        ext = _resize_mask_nn(m, arr.shape[:2])

    if arr.ndim == 2 or (arr.ndim == 3 and arr.shape[2] == 1):
        L = arr.squeeze().astype(np.float32, copy=True)  # [0..1]
        mask = np.zeros_like(L, dtype=np.float32)  # define for return
        for it in range(int(iterations)):
            mask = _darkness_mask(L, n_scales, base_kernel, mask_gamma)
            if ext is not None:
                mask = np.clip(mask * ext, 0.0, 1.0)  # ← combine here

            planes = _atrous_decompose(L, n_scales, base_kernel)
            residual = planes.pop()
            for i in range(len(planes)):
                if i == 0:
                    continue  # skip highest frequency
                decay = decay_rate ** i
                neg = np.clip(-planes[i], 0, None)
                enhancement = neg * mask * (boost_factor - 1.0) * decay
                planes[i] = planes[i] - enhancement
            L = np.clip(_atrous_reconstruct(planes + [residual]), 0.0, 1.0)

        out = L.astype(np.float32, copy=False)
        return out, mask.astype(np.float32, copy=False)

    # RGB path
    rgb = np.clip(arr[:, :, :3], 0.0, 1.0).astype(np.float32, copy=False)
    lab = _rgb_to_lab(rgb)
    L = lab[..., 0].astype(np.float32, copy=True)
    mask = np.zeros(L.shape, dtype=np.float32)  # define for return
    for it in range(int(iterations)):
        mask = _darkness_mask(np.clip(L / 100.0, 0.0, 1.0), n_scales, base_kernel, mask_gamma)
        if ext is not None:
            mask = np.clip(mask * ext, 0.0, 1.0)  # ← combine here

        planes = _atrous_decompose(L, n_scales, base_kernel)
        residual = planes.pop()
        for i in range(len(planes)):
            if i == 0:
                continue
            decay = decay_rate ** i
            neg = np.clip(-planes[i], 0, None)
            enhancement = neg * mask * (boost_factor - 1.0) * decay
            planes[i] = planes[i] - enhancement
        L = np.clip(_atrous_reconstruct(planes + [residual]), 0.0, 100.0)

    lab[..., 0] = L
    out_rgb = _lab_to_rgb(lab)
    return out_rgb.astype(np.float32, copy=False), mask.astype(np.float32, copy=False)

# ─────────────────────────────────────────────────────────────────────────────
# Worker
# ─────────────────────────────────────────────────────────────────────────────
class DSEWorker(QObject):
    progress_update = pyqtSignal(str, int)
    finished = pyqtSignal(np.ndarray, np.ndarray)  # (output, mask)

    def __init__(self, image: np.ndarray, n_scales: int, boost: float, gamma: float,
                 base_kernel: np.ndarray, iterations: int,
                 external_mask: np.ndarray | None = None): 
        super().__init__()
        self.image = image
        self.n_scales = n_scales
        self.boost = boost
        self.gamma = gamma
        self.base_kernel = base_kernel
        self.iterations = iterations
        self.external_mask = external_mask

    def run(self):
        try:
            self.progress_update.emit("Analyzing dark structure…", 20)
            out, mask = compute_wavescale_dse(
                self.image, self.n_scales, self.boost, self.gamma,
                self.iterations, self.base_kernel,
                external_mask=self.external_mask             # ← NEW
            )
            self.progress_update.emit("Finalizing…", 95)
            self.finished.emit(out, mask)
        except Exception as e:
            print("WaveScale DSE error:", e)
            self.finished.emit(None, None)

# ─────────────────────────────────────────────────────────────────────────────
# Small mask window (fixed ~400×400, always shows a zoomed-out mask)
# ─────────────────────────────────────────────────────────────────────────────
class _MaskWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Dark Mask")
        self.setMinimumSize(300, 300)
        self.resize(400, 400)
        v = QVBoxLayout(self)
        self.lbl = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        v.addWidget(self.lbl)

    def set_mask(self, mask: np.ndarray):
        m = np.clip(mask, 0, 1).astype(np.float32)
        m8 = (m * 255.0).astype(np.uint8)
        if m8.ndim == 2:
            h, w = m8.shape
            q = QImage(m8.data, w, h, w, QImage.Format.Format_Grayscale8)
        else:
            h, w, _ = m8.shape
            q = QImage(m8.data, w, h, 3*w, QImage.Format.Format_RGB888)
        pm = QPixmap.fromImage(q)
        box = self.size()
        pm2 = pm.scaled(box, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.lbl.setPixmap(pm2)

# ─────────────────────────────────────────────────────────────────────────────
# Dialog
# ─────────────────────────────────────────────────────────────────────────────
class WaveScaleDarkEnhancerDialogPro(QDialog):
    def __init__(self, parent, doc, icon_path: str | None = None):
        super().__init__(parent)
        self.setWindowTitle("WaveScale Dark Enhancer")
        if icon_path:
            try: self.setWindowIcon(QIcon(icon_path))
            except Exception: pass
        self.resize(980, 700)

        self._doc = doc
        base = getattr(doc, "image", None)
        if base is None:
            raise RuntimeError("Active document has no image.")

        img = np.asarray(base, dtype=np.float32)
        if img.ndim == 2:
            self._was_mono = True
            self._mono_shape = img.shape
            rgb = np.repeat(img[:, :, None], 3, axis=2)
        elif img.ndim == 3 and img.shape[2] == 1:
            self._was_mono = True
            self._mono_shape = img.shape
            rgb = np.repeat(img, 3, axis=2)
        else:
            self._was_mono = False
            self._mono_shape = None
            rgb = img[:, :, :3]
        if img.dtype.kind in "ui":
            maxv = float(np.nanmax(rgb)) or 1.0
            rgb = rgb / max(1.0, maxv)
        rgb = np.clip(rgb, 0.0, 1.0).astype(np.float32, copy=False)

        self.original = rgb
        self.preview  = rgb.copy()

        # scene/view
        self.scene = QGraphicsScene(self)
        self.view  = QGraphicsView(self.scene)
        self.view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pix   = QGraphicsPixmapItem()
        self.scene.addItem(self.pix)
        self.scroll = QScrollArea(self); self.scroll.setWidgetResizable(True); self.scroll.setWidget(self.view)

        # zoom state
        self.zoom_factor = 1.0
        self.zoom_step   = 1.25
        self.zoom_min    = 0.1
        self.zoom_max    = 5.0

        # controls
        self.grp = QGroupBox("Dark Enhancer Controls")
        form = QFormLayout(self.grp)

        self.s_scales = QSlider(Qt.Orientation.Horizontal); self.s_scales.setRange(2, 10);  self.s_scales.setValue(6)
        self.s_boost  = QSlider(Qt.Orientation.Horizontal); self.s_boost.setRange(10, 1000); self.s_boost.setValue(500)  # 0.10..10.00
        self.s_gamma  = QSlider(Qt.Orientation.Horizontal); self.s_gamma.setRange(10, 1000); self.s_gamma.setValue(100)  # 0.10..10.00
        self.s_iters  = QSlider(Qt.Orientation.Horizontal); self.s_iters.setRange(1, 10);   self.s_iters.setValue(2)

        form.addRow("Number of Scales:", self.s_scales)
        form.addRow("Boost Factor:", self.s_boost)
        form.addRow("Mask Gamma:", self.s_gamma)
        form.addRow("Iterations:", self.s_iters)

        row = QHBoxLayout()
        self.btn_preview = QPushButton("Preview")
        self.btn_toggle  = QPushButton("Show Original"); self.btn_toggle.setCheckable(True)
        row.addWidget(self.btn_preview); row.addWidget(self.btn_toggle)
        form.addRow(row)

        # progress
        self.prog_grp = QGroupBox("Progress")
        vprog = QVBoxLayout(self.prog_grp)
        self.lbl_step = QLabel("Idle")
        self.bar = QProgressBar(); self.bar.setRange(0, 100); self.bar.setValue(0)
        vprog.addWidget(self.lbl_step); vprog.addWidget(self.bar)

        # bottom
        bot = QHBoxLayout()
        self.btn_apply = QPushButton("Apply to Document"); self.btn_apply.setEnabled(False)
        self.btn_reset = QPushButton("Reset")
        self.btn_close = QPushButton("Close")
        bot.addStretch(1); bot.addWidget(self.btn_apply); bot.addWidget(self.btn_reset); bot.addWidget(self.btn_close)

        # layout
        main = QVBoxLayout(self)
        main.addWidget(self.scroll)

        zoom_box = QGroupBox("Zoom Controls")
        zr = QHBoxLayout(zoom_box)
        self.btn_zin  = QPushButton("Zoom In")
        self.btn_zout = QPushButton("Zoom Out")
        self.btn_fit  = QPushButton("Fit to Preview")
        zr.addWidget(self.btn_zin); zr.addWidget(self.btn_zout); zr.addWidget(self.btn_fit)
        main.addWidget(zoom_box)

        h = QHBoxLayout()
        h.addWidget(self.grp, 3)
        h.addWidget(self.prog_grp, 1)
        main.addLayout(h)
        main.addLayout(bot)

        # mask window (show immediately)
        self.mask_win = _MaskWindow(self); self.mask_win.show()

        # kernel
        self.base_kernel = _B3

        # connections
        self.btn_preview.clicked.connect(self._start_preview)
        self.btn_apply.clicked.connect(self._apply_to_doc)
        self.btn_close.clicked.connect(self.reject)
        self.btn_reset.clicked.connect(self._reset)
        self.btn_toggle.clicked.connect(self._toggle)

        self.btn_zin.clicked.connect(self._zoom_in)
        self.btn_zout.clicked.connect(self._zoom_out)
        self.btn_fit.clicked.connect(self._fit_to_preview)

        # gamma debounce → live mask updates (250ms)
        self._gamma_timer = QTimer(self)
        self._gamma_timer.setSingleShot(True)
        self._gamma_timer.timeout.connect(self._update_mask_only)
        self.s_gamma.valueChanged.connect(lambda _v: self._gamma_timer.start(250))

        # init preview & initial mask
        self._set_pix(self.preview)
        self._update_mask_only()

    def _combine_with_doc_mask(self, op_mask: np.ndarray | None) -> np.ndarray | None:
        doc_m = self._get_doc_active_mask_2d()
        if doc_m is None:
            return op_mask
        if op_mask is None:
            return doc_m
        H, W = op_mask.shape[:2]
        if doc_m.shape != (H, W):
            yi = (np.linspace(0, doc_m.shape[0] - 1, H)).astype(np.int32)
            xi = (np.linspace(0, doc_m.shape[1] - 1, W)).astype(np.int32)
            doc_m = doc_m[yi][:, xi]
        return np.clip(op_mask * doc_m, 0.0, 1.0)

    def _get_doc_active_mask_2d(self) -> np.ndarray | None:
        """
        Return active document mask as 2-D float32 [0..1], resized to current image.
        """
        doc = getattr(self, "_doc", None)
        if doc is None:
            return None

        mid = getattr(doc, "active_mask_id", None)
        if not mid:
            return None

        masks = getattr(doc, "masks", {}) or {}
        layer = masks.get(mid)
        if layer is None:
            return None

        # pick first non-None payload without boolean 'or'
        data = None
        for attr in ("data", "mask", "image", "array"):
            if hasattr(layer, attr):
                val = getattr(layer, attr)
                if val is not None:
                    data = val
                    break
        if data is None and isinstance(layer, dict):
            for key in ("data", "mask", "image", "array"):
                if key in layer and layer[key] is not None:
                    data = layer[key]
                    break
        if data is None and isinstance(layer, np.ndarray):
            data = layer
        if data is None:
            return None

        m = np.asarray(data)
        if m.ndim == 3:
            m = m.mean(axis=2)

        m = m.astype(np.float32, copy=False)
        # normalize to [0,1] if needed
        mx = float(m.max()) if m.size else 1.0
        if mx > 1.0:
            m /= mx
        m = np.clip(m, 0.0, 1.0)

        # resize (nearest) to current image size
        H, W = self.original.shape[:2]
        if m.shape != (H, W):
            yi = (np.linspace(0, m.shape[0] - 1, H)).astype(np.int32)
            xi = (np.linspace(0, m.shape[1] - 1, W)).astype(np.int32)
            m = m[yi][:, xi]

        return m


    def _combine_with_doc_mask(self, algo_mask: np.ndarray) -> np.ndarray:
        m_doc = self._get_doc_active_mask_2d()
        if m_doc is None:
            return algo_mask
        return np.clip(algo_mask.astype(np.float32) * m_doc.astype(np.float32), 0.0, 1.0)


    # --- preview pixmap ---
    def _set_pix(self, rgb: np.ndarray):
        arr = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        h, w, _ = arr.shape
        q = QImage(arr.data, w, h, 3*w, QImage.Format.Format_RGB888)
        self.pix.setPixmap(QPixmap.fromImage(q))
        self.view.setSceneRect(self.pix.boundingRect())

    # --- toggle ---
    def _toggle(self):
        if self.btn_toggle.isChecked():
            self.btn_toggle.setText("Show Preview")
            self._set_pix(self.original)
        else:
            self.btn_toggle.setText("Show Original")
            self._set_pix(self.preview)

    # --- reset ---
    def _reset(self):
        self.s_scales.setValue(6)
        self.s_boost.setValue(500)
        self.s_gamma.setValue(100)
        self.s_iters.setValue(2)
        self.preview = self.original.copy()
        self._set_pix(self.preview)
        self.lbl_step.setText("Idle"); self.bar.setValue(0)
        self.btn_apply.setEnabled(False)
        self.btn_toggle.setChecked(False); self.btn_toggle.setText("Show Original")
        self._update_mask_only()

    # --- zoom + Ctrl+Wheel ---
    def wheelEvent(self, e: QWheelEvent):
        if e.modifiers() & Qt.KeyboardModifier.ControlModifier:
            if e.angleDelta().y() > 0: self._zoom_in()
            else: self._zoom_out()
            e.accept(); return
        super().wheelEvent(e)

    def _zoom_in(self):
        z = self.zoom_factor * self.zoom_step
        if z <= self.zoom_max:
            self.zoom_factor = z
            self._apply_zoom()

    def _zoom_out(self):
        z = self.zoom_factor / self.zoom_step
        if z >= self.zoom_min:
            self.zoom_factor = z
            self._apply_zoom()

    def _fit_to_preview(self):
        if not self.pix.pixmap().isNull():
            self.view.fitInView(self.pix, Qt.AspectRatioMode.KeepAspectRatio)
        self.zoom_factor = 1.0

    def _apply_zoom(self):
        self.view.resetTransform()
        self.view.scale(self.zoom_factor, self.zoom_factor)

    # --- live mask (no full recompute) ---
    def _update_mask_only(self):
        mgamma = float(self.s_gamma.value()) / 100.0
        base = self.original
        lab = _rgb_to_lab(base)
        L = lab[..., 0] / 100.0
        algo_mask = _darkness_mask(np.clip(L, 0.0, 1.0),
                                int(self.s_scales.value()),
                                self.base_kernel, mgamma)
        mask_comb = self._combine_with_doc_mask(algo_mask)
        self.mask_win.setWindowTitle(
            "Dark Mask (Algo × Active Mask)" if self._get_doc_active_mask_2d() is not None else "Dark Mask"
        )
        self.mask_win.set_mask(mask_comb)
    # --- threaded preview ---
    def _start_preview(self):
        self.btn_preview.setEnabled(False); self.btn_apply.setEnabled(False)
        n_scales = int(self.s_scales.value())
        boost    = float(self.s_boost.value()) / 100.0
        mgamma   = float(self.s_gamma.value()) / 100.0
        iters    = int(self.s_iters.value())
        docmask  = self._get_doc_active_mask_2d() 

        self.thread = QThread(self)
        self.worker = DSEWorker(self.original, n_scales, boost, mgamma,
                                self.base_kernel, iters,
                                external_mask=docmask)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress_update.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def _on_progress(self, step: str, pct: int):
        self.lbl_step.setText(step); self.bar.setValue(pct)

    def _on_finished(self, out: np.ndarray, mask: np.ndarray):
        self.btn_preview.setEnabled(True)
        if out is None:
            QMessageBox.critical(self, "WaveScale Dark Enhancer", "Processing failed.")
            return

        # Respect the document mask
        doc_m = self._get_doc_active_mask_2d()
        if out.ndim == 2:
            out_rgb = np.repeat(out[:, :, None], 3, axis=2)
        else:
            out_rgb = out

        if doc_m is not None:
            M3 = np.repeat(doc_m[:, :, None], 3, axis=2).astype(np.float32)
            self.preview = self.original * (1.0 - M3) + out_rgb * M3
        else:
            self.preview = out_rgb

        # show combined mask (internal darkness mask × doc mask)
        mask = self._combine_with_doc_mask(mask)

        self._set_pix(self.preview)
        self.mask_win.set_mask(mask)
        self.btn_apply.setEnabled(True)
        self.btn_toggle.setChecked(False); self.btn_toggle.setText("Show Original")
        self.lbl_step.setText("Preview ready"); self.bar.setValue(100)

    # --- apply back to doc ---
    def _apply_to_doc(self):
        out = self.preview
        if self._was_mono:
            mono = np.mean(out, axis=2, dtype=np.float32)
            if self._mono_shape and len(self._mono_shape) == 3 and self._mono_shape[2] == 1:
                mono = mono[:, :, None]
            out = mono
        out = np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)
        try:
            if hasattr(self._doc, "set_image"):
                self._doc.set_image(out, step_name="WaveScale Dark Enhancer")
            elif hasattr(self._doc, "apply_numpy"):
                self._doc.apply_numpy(out, step_name="WaveScale Dark Enhancer")
            else:
                self._doc.image = out
        except Exception as e:
            QMessageBox.critical(self, "WaveScale Dark Enhancer", f"Failed to write to document:\n{e}")
            return
        self.accept()

# ─────────────────────────────────────────────────────────────────────────────
# Installer helpers
# ─────────────────────────────────────────────────────────────────────────────
def install_wavescale_dark_enhancer(main_window: QMainWindow,
                                    dse_icon_path: str,
                                    *,
                                    command_id: str = "wavescale_dark_enhancer",
                                    menu_name: str = "Pro",
                                    toolbar_name: str = "Pro Tools"):
    """
    Creates the QAction, hooks it into menu+toolbar, and registers it
    with your ShortcutManager under `command_id`.
    Expects main_window to expose:
      • .docman.current_document() → returns doc with .image
      • ._spawn_subwindow_for(doc) (normal in your app)
      • .shortcut_manager (your ShortcutManager) — optional
    """
    # 1) QAction
    act = getattr(main_window, "act_wavescalede", None)
    if act is None:
        from PyQt6.QtGui import QAction
        act = QAction(QIcon(dse_icon_path), "WaveScale Dark Enhancer", main_window)
        act.setObjectName(command_id)
        act.setProperty("command_id", command_id)

        def _run_dialog():
            docman = getattr(main_window, "docman", None)
            doc = None
            if docman and hasattr(docman, "current_document"):
                doc = docman.current_document()
            if doc is None or getattr(doc, "image", None) is None:
                QMessageBox.warning(main_window, "WaveScale Dark Enhancer", "No active image.")
                return
            dlg = WaveScaleDarkEnhancerDialogPro(main_window, doc, icon_path=dse_icon_path)
            dlg.exec()

        act.triggered.connect(_run_dialog)
        setattr(main_window, "act_wavescalede", act)

    # 2) Menu hookup
    menubar = main_window.menuBar()
    menu = None
    for m in menubar.findChildren(type(menubar)):
        # best-effort: ignore; we’ll just create/find by title
        pass
    menu = None
    for i in range(menubar.actions().__len__()):
        if menubar.actions()[i].text().replace("&", "") == menu_name:
            menu = menubar.actions()[i].menu()
            break
    if menu is None:
        menu = menubar.addMenu(menu_name)
    menu.addAction(act)

    # 3) Toolbar hookup
    tb = None
    for t in main_window.findChildren(type(main_window.addToolBar("tmp"))):
        # naive scan (we won't rely on this); we'll create if needed
        pass
    tb = getattr(main_window, "_tb_" + toolbar_name.replace(" ", "_").lower(), None)
    if tb is None:
        tb = main_window.addToolBar(toolbar_name)
        setattr(main_window, "_tb_" + toolbar_name.replace(" ", "_").lower(), tb)
    tb.addAction(act)

    # 4) Register with ShortcutManager (if present)
    sm = getattr(main_window, "shortcut_manager", None)
    if sm and hasattr(sm, "register_action"):
        sm.register_action(command_id, act)

    return act
