# pro/morphology.py
from __future__ import annotations
import numpy as np
import cv2

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap, QIcon, QWheelEvent, QPainter
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QGridLayout,
    QLabel, QPushButton, QSlider, QComboBox, QGraphicsView,
    QGraphicsScene, QGraphicsPixmapItem, QMessageBox, QSpinBox
)

# ---------------- Zoomable view (same helper) ----------------
class ZoomableGraphicsView(QGraphicsView):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._zoom = 1.0
        self._step = 1.25
        self._min  = 0.10
        self._max  = 12.0
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)

    def wheelEvent(self, e: QWheelEvent):
        if e.modifiers() & Qt.KeyboardModifier.ControlModifier:
            dy = e.angleDelta().y()
            if dy == 0:
                e.accept(); return
            factor = self._step if dy > 0 else (1.0 / self._step)
            new_zoom = max(self._min, min(self._max, self._zoom * factor))
            factor = new_zoom / self._zoom
            if factor != 1.0:
                self.scale(factor, factor)
                self._zoom = new_zoom
            e.accept()
        else:
            super().wheelEvent(e)

    def zoom_in(self):
        self.wheelEvent(QWheelEvent(self.viewport().rect().center(), 120, Qt.MouseButton.NoButton,
                                    Qt.KeyboardModifier.ControlModifier, Qt.ScrollPhase.ScrollUpdate, False))

    def zoom_out(self):
        self.wheelEvent(QWheelEvent(self.viewport().rect().center(), -120, Qt.MouseButton.NoButton,
                                    Qt.KeyboardModifier.ControlModifier, Qt.ScrollPhase.ScrollUpdate, False))

    def fit_to_item(self, item):
        if not item or item.pixmap().isNull():
            return
        self._zoom = 1.0
        self.resetTransform()
        self.fitInView(item, Qt.AspectRatioMode.KeepAspectRatio)

# ---------------- Core (unchanged) ----------------
def apply_morphology(image: np.ndarray, *, operation: str = "erosion",
                     kernel_size: int = 3, iterations: int = 1) -> np.ndarray:
    # ... (existing body unchanged)
    if image is None:
        raise ValueError("image is None")
    img = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    was_hw1 = (img.ndim == 3 and img.shape[2] == 1)
    if kernel_size % 2 == 0: kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    def _do(u8):
        if operation == "erosion":  return cv2.erode(u8, kernel, iterations=iterations)
        if operation == "dilation": return cv2.dilate(u8, kernel, iterations=iterations)
        if operation == "opening":  return cv2.morphologyEx(u8, cv2.MORPH_OPEN, kernel, iterations=iterations)
        if operation == "closing":  return cv2.morphologyEx(u8, cv2.MORPH_CLOSE, kernel, iterations=iterations)
        raise ValueError(f"Unsupported morphological operation: {operation}")

    if img.ndim == 2 or was_hw1:
        mono = (img.squeeze() * 255.0).astype(np.uint8)
        out = _do(mono).astype(np.float32) / 255.0
        out = np.clip(out, 0.0, 1.0)
        return out[..., None] if was_hw1 else out

    if img.ndim == 3 and img.shape[2] == 3:
        u8 = (img * 255.0).astype(np.uint8)
        ch = cv2.split(u8)
        ch = [_do(c) for c in ch]
        out = cv2.merge(ch).astype(np.float32) / 255.0
        return np.clip(out, 0.0, 1.0)

    raise ValueError("Input image must be mono (H,W)/(H,W,1) or RGB (H,W,3).")

def apply_morphology_to_doc(doc, preset: dict | None):
    if doc is None or getattr(doc, "image", None) is None:
        raise RuntimeError("Document has no image.")

    img = np.asarray(doc.image)
    op   = (preset or {}).get("operation", "erosion")
    ker  = int((preset or {}).get("kernel", 3))
    it   = int((preset or {}).get("iterations", 1))

    out = apply_morphology(img, operation=str(op), kernel_size=ker, iterations=it)
    out = np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)

    # Blend with active mask if present
    H, W = out.shape[:2]
    m = _get_active_mask_resized(doc, H, W)
    if m is not None:
        base = np.asarray(doc.image, dtype=np.float32)
        if base.dtype.kind in "ui":
            maxv = float(np.iinfo(base.dtype).max)
            base = base / max(1.0, maxv)
        else:
            base = np.clip(base, 0.0, 1.0)
        out = _blend_with_mask(base, out, m).astype(np.float32, copy=False)

    if hasattr(doc, "set_image"): doc.set_image(out, step_name="Morphology")
    elif hasattr(doc, "apply_numpy"): doc.apply_numpy(out, step_name="Morphology")
    else: doc.image = out

def _nearest_resize_2d(m: np.ndarray, H: int, W: int) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    if m.shape == (H, W):
        return m
    try:
        return cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
    except Exception:
        yi = (np.linspace(0, m.shape[0] - 1, H)).astype(np.int32)
        xi = (np.linspace(0, m.shape[1] - 1, W)).astype(np.int32)
        return m[yi][:, xi].astype(np.float32, copy=False)

def _get_active_mask_resized(doc, H: int, W: int) -> np.ndarray | None:
    """
    Returns doc's active mask as a 2-D float32 array in [0..1], resized to (H, W).
    Handles layer objects, dicts, or raw ndarrays.
    """
    if doc is None:
        return None
    mid = getattr(doc, "active_mask_id", None)
    if not mid:
        return None
    masks = getattr(doc, "masks", {}) or {}
    layer = masks.get(mid)
    if layer is None:
        return None

    data = None
    # object-like
    for attr in ("data", "mask", "image", "array"):
        if hasattr(layer, attr):
            val = getattr(layer, attr)
            if val is not None:
                data = val
                break
    # dict-like
    if data is None and isinstance(layer, dict):
        for key in ("data", "mask", "image", "array"):
            if key in layer and layer[key] is not None:
                data = layer[key]
                break
    # raw ndarray
    if data is None and isinstance(layer, np.ndarray):
        data = layer
    if data is None:
        return None

    m = np.asarray(data)
    if m.ndim == 3:                      # collapse RGB(A) → gray
        m = m.mean(axis=2)
    m = m.astype(np.float32, copy=False)

    # normalize into [0..1]
    mx = float(m.max()) if m.size else 1.0
    if mx > 1.0:
        m = m / mx
    m = np.clip(m, 0.0, 1.0)

    return _nearest_resize_2d(m, H, W)

def _blend_with_mask(base: np.ndarray, out: np.ndarray, m2d: np.ndarray) -> np.ndarray:
    """
    Blend base and out using m2d in [0..1]. Shapes:
    - If out is RGB, broadcast mask & base to 3-ch as needed.
    - If out is mono (2-D), use 2-D mask.
    All inputs assumed float32 in [0..1].
    """
    base = np.asarray(base, dtype=np.float32)
    out  = np.asarray(out,  dtype=np.float32)
    m2d  = np.clip(np.asarray(m2d, dtype=np.float32), 0.0, 1.0)

    if out.ndim == 3:
        if base.ndim == 2:
            base = base[:, :, None].repeat(out.shape[2], axis=2)
        elif base.ndim == 3 and base.shape[2] == 1:
            base = base.repeat(out.shape[2], axis=2)
        M = m2d[:, :, None].repeat(out.shape[2], axis=2)
        return np.clip(base * (1.0 - M) + out * M, 0.0, 1.0)

    # out is mono
    if base.ndim == 3 and base.shape[2] == 1:
        base = base.squeeze(axis=2)
    return np.clip(base * (1.0 - m2d) + out * m2d, 0.0, 1.0)


# ---------------- Dialog ----------------
class MorphologyDialogPro(QDialog):
    OPS = ["Erosion", "Dilation", "Opening", "Closing"]
    OP_MAP = {"Erosion":"erosion","Dilation":"dilation","Opening":"opening","Closing":"closing"}

    def __init__(self, parent, doc, icon: QIcon | None = None, initial: dict | None = None):
        super().__init__(parent)
        self.setWindowTitle("Morphological Operations")
        if icon:
            try: self.setWindowIcon(icon)
            except Exception: pass

        self.doc  = doc
        self.orig = np.clip(np.asarray(doc.image, dtype=np.float32), 0.0, 1.0)

        disp = self.orig
        if disp.ndim == 2: disp = disp[..., None].repeat(3, axis=2)
        elif disp.ndim == 3 and disp.shape[2] == 1: disp = disp.repeat(3, axis=2)
        self._disp_base = disp

        v = QVBoxLayout(self)

        # ---- Params (unchanged) ----
        grp = QGroupBox("Morphological Parameters")
        grid = QGridLayout(grp)
        self.cb_op = QComboBox(); self.cb_op.addItems(self.OPS)
        self.sp_kernel = QSpinBox(); self.sp_kernel.setRange(1, 31); self.sp_kernel.setSingleStep(2)
        self.sp_iter   = QSpinBox(); self.sp_iter.setRange(1, 10)

        init = dict(initial or {})
        op_text = {v:k for k,v in self.OP_MAP.items()}.get(str(init.get("operation","erosion")).lower(), "Erosion")
        self.cb_op.setCurrentText(op_text)
        k = int(init.get("kernel", 3)); self.sp_kernel.setValue(k if k % 2 == 1 else k + 1)
        self.sp_iter.setValue(int(init.get("iterations", 1)))

        self.cb_op.currentTextChanged.connect(self._debounce)
        self.sp_kernel.valueChanged.connect(self._debounce)
        self.sp_iter.valueChanged.connect(self._debounce)

        grid.addWidget(QLabel("Operation:"), 0, 0); grid.addWidget(self.cb_op, 0, 1, 1, 2)
        grid.addWidget(QLabel("Kernel size:"), 1, 0); grid.addWidget(self.sp_kernel, 1, 1)
        grid.addWidget(QLabel("Iterations:"), 2, 0); grid.addWidget(self.sp_iter, 2, 1)
        v.addWidget(grp)

        # ---- Preview with zoom/pan ----
        self.scene = QGraphicsScene(self)
        self.view  = ZoomableGraphicsView(self.scene)
        self.view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pix   = QGraphicsPixmapItem()
        self.scene.addItem(self.pix)
        v.addWidget(self.view, 1)

        # ---- Zoom bar ----
        z = QHBoxLayout()
        btn_in  = QPushButton("Zoom In");  btn_in.clicked.connect(self.view.zoom_in)
        btn_out = QPushButton("Zoom Out"); btn_out.clicked.connect(self.view.zoom_out)
        btn_fit = QPushButton("Fit to Preview"); btn_fit.clicked.connect(lambda: self.view.fit_to_item(self.pix))
        z.addStretch(1); z.addWidget(btn_in); z.addWidget(btn_out); z.addWidget(btn_fit)
        v.addLayout(z)

        # ---- Buttons (unchanged) ----
        row = QHBoxLayout()
        btn_apply = QPushButton("Apply");  btn_apply.clicked.connect(self._apply)
        btn_reset = QPushButton("Reset");  btn_reset.clicked.connect(self._reset)
        btn_cancel= QPushButton("Cancel"); btn_cancel.clicked.connect(self.reject)
        row.addStretch(1); row.addWidget(btn_apply); row.addWidget(btn_reset); row.addWidget(btn_cancel)
        v.addLayout(row)

        self._timer = QTimer(self); self._timer.setSingleShot(True); self._timer.timeout.connect(self._update_preview)

        self._set_pix(self._disp_base)
        self._update_preview()
        # initial fit
        self.view.fit_to_item(self.pix)

    def _debounce(self): self._timer.start(200)

    def _set_pix(self, rgb):
        arr = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        h, w, _ = arr.shape
        q = QImage(arr.data, w, h, 3*w, QImage.Format.Format_RGB888)
        self.pix.setPixmap(QPixmap.fromImage(q))
        self.scene.setSceneRect(self.pix.boundingRect())

    def _params(self):
        op = self.OP_MAP[self.cb_op.currentText()]
        k  = int(self.sp_kernel.value())
        it = int(self.sp_iter.value())
        if k % 2 == 0: k += 1
        return op, k, it

    def _update_preview(self):
        op, k, it = self._params()
        try:
            out = apply_morphology(self._disp_base, operation=op, kernel_size=k, iterations=it)

            # Blend preview with active mask (preview is on _disp_base size)
            H, W = out.shape[:2]
            m = _get_active_mask_resized(self.doc, H, W)
            if m is not None:
                base = self._disp_base.astype(np.float32)
                out = _blend_with_mask(base, out.astype(np.float32), m)

            self._set_pix(out)
        except Exception as e:
            QMessageBox.warning(self, "Morphology", f"Preview failed:\n{e}")

    def _apply(self):
        op, k, it = self._params()
        try:
            out = apply_morphology(self.orig, operation=op, kernel_size=k, iterations=it)
            out = np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)

            # Blend with active mask at full resolution
            H, W = out.shape[:2]
            m = _get_active_mask_resized(self.doc, H, W)
            if m is not None:
                base = np.asarray(self.doc.image, dtype=np.float32)
                if base.dtype.kind in "ui":
                    maxv = float(np.iinfo(base.dtype).max)
                    base = base / max(1.0, maxv)
                else:
                    base = np.clip(base, 0.0, 1.0)
                out = _blend_with_mask(base, out, m).astype(np.float32, copy=False)

            if hasattr(self.doc, "set_image"): self.doc.set_image(out, step_name="Morphology")
            elif hasattr(self.doc, "apply_numpy"): self.doc.apply_numpy(out, step_name="Morphology")
            else: self.doc.image = out
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Morphology", f"Failed to apply:\n{e}")


    def _reset(self):
        self.cb_op.setCurrentText("Erosion")
        self.sp_kernel.setValue(3)
        self.sp_iter.setValue(1)
        self._set_pix(self._disp_base)
        self.view.fit_to_item(self.pix)
