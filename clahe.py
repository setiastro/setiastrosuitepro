# pro/clahe.py
from __future__ import annotations
import numpy as np
import cv2

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap, QIcon, QWheelEvent, QPainter
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QGridLayout,
    QLabel, QPushButton, QSlider, QGraphicsView, QGraphicsScene,
    QGraphicsPixmapItem, QMessageBox
)

# ----------------------- Zoomable view -----------------------
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

# ----------------------- Core -----------------------
def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    # ... (unchanged)
    if image is None:
        raise ValueError("image is None")
    arr = np.asarray(image, dtype=np.float32)
    arr = np.clip(arr, 0.0, 1.0)
    was_hw1 = (arr.ndim == 3 and arr.shape[2] == 1)
    if arr.ndim == 3 and arr.shape[2] == 3:
        lab = cv2.cvtColor((arr * 255.0).astype(np.uint8), cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tuple(tile_grid_size))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0
        return np.clip(enhanced, 0.0, 1.0)
    mono = arr.squeeze()
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tuple(tile_grid_size))
    cl = clahe.apply((mono * 255.0).astype(np.uint8)).astype(np.float32) / 255.0
    cl = np.clip(cl, 0.0, 1.0)
    if was_hw1:
        cl = cl[..., None]
    return cl

def _nearest_resize_2d(m: np.ndarray, H: int, W: int) -> np.ndarray:
    if m.shape == (H, W):
        return m.astype(np.float32, copy=False)
    try:
        return cv2.resize(m.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
    except Exception:
        yi = (np.linspace(0, m.shape[0] - 1, H)).astype(np.int32)
        xi = (np.linspace(0, m.shape[1] - 1, W)).astype(np.int32)
        return m[yi][:, xi].astype(np.float32, copy=False)

def _get_active_mask_resized(doc, H: int, W: int) -> np.ndarray | None:
    """
    Read doc.active_mask_id → masks[mid], pull first non-None payload among
    (data, mask, image, array), normalize to [0..1], and resize to (H, W).
    Returns 2-D float32 or None.
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
    if m.ndim == 3:  # collapse RGB(A) → gray
        m = m.mean(axis=2)
    m = m.astype(np.float32, copy=False)

    # normalize → [0..1]
    mx = float(m.max()) if m.size else 1.0
    if mx > 1.0:
        m = m / mx
    m = np.clip(m, 0.0, 1.0)

    return _nearest_resize_2d(m, H, W)

def apply_clahe_to_doc(doc, preset: dict | None):
    if doc is None or getattr(doc, "image", None) is None:
        raise RuntimeError("Document has no image.")

    img = np.asarray(doc.image)
    clip = float((preset or {}).get("clip_limit", 2.0))
    tile = int((preset or {}).get("tile", 8))

    out = apply_clahe(img, clip_limit=clip, tile_grid_size=(tile, tile))
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

        if out.ndim == 3:
            if base.ndim == 2:
                base = base[:, :, None].repeat(out.shape[2], axis=2)
            elif base.ndim == 3 and base.shape[2] == 1:
                base = base.repeat(out.shape[2], axis=2)
            M = np.repeat(m[:, :, None], out.shape[2], axis=2).astype(np.float32)
            out = np.clip(base * (1.0 - M) + out * M, 0.0, 1.0)
        else:
            if base.ndim == 3 and base.shape[2] == 1:
                base = base.squeeze(axis=2)
            out = np.clip(base * (1.0 - m) + out * m, 0.0, 1.0)

    if hasattr(doc, "set_image"):
        doc.set_image(out, step_name="CLAHE")
    elif hasattr(doc, "apply_numpy"):
        doc.apply_numpy(out, step_name="CLAHE")
    else:
        doc.image = out

# ----------------------- Dialog -----------------------
class CLAHEDialogPro(QDialog):
    def __init__(self, parent, doc, icon: QIcon | None = None):
        super().__init__(parent)
        self.setWindowTitle("CLAHE")
        if icon:
            try: self.setWindowIcon(icon)
            except Exception: pass

        self.doc = doc
        self.orig = np.clip(np.asarray(doc.image, dtype=np.float32), 0.0, 1.0)
        disp = self.orig
        if disp.ndim == 2: disp = disp[..., None].repeat(3, axis=2)
        elif disp.ndim == 3 and disp.shape[2] == 1: disp = disp.repeat(3, axis=2)
        self._disp_base = disp

        v = QVBoxLayout(self)

        # ---- Params (unchanged) ----
        grp = QGroupBox("CLAHE Parameters"); grid = QGridLayout(grp)
        self.s_clip = QSlider(Qt.Orientation.Horizontal); self.s_clip.setRange(1, 40); self.s_clip.setValue(20)
        self.lbl_clip = QLabel("2.0")
        self.s_clip.valueChanged.connect(lambda val: self.lbl_clip.setText(f"{val/10.0:.1f}"))
        self.s_clip.valueChanged.connect(self._debounce_preview)

        self.s_tile = QSlider(Qt.Orientation.Horizontal); self.s_tile.setRange(1, 32); self.s_tile.setValue(8)
        self.lbl_tile = QLabel("8")
        self.s_tile.valueChanged.connect(lambda val: self.lbl_tile.setText(str(val)))
        self.s_tile.valueChanged.connect(self._debounce_preview)

        grid.addWidget(QLabel("Clip Limit:"), 0, 0); grid.addWidget(self.s_clip, 0, 1); grid.addWidget(self.lbl_clip, 0, 2)
        grid.addWidget(QLabel("Tile Grid Size:"), 1, 0); grid.addWidget(self.s_tile, 1, 1); grid.addWidget(self.lbl_tile, 1, 2)
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
        self.btn_apply = QPushButton("Apply");  self.btn_apply.clicked.connect(self._apply)
        self.btn_reset = QPushButton("Reset");  self.btn_reset.clicked.connect(self._reset)
        self.btn_close = QPushButton("Cancel"); self.btn_close.clicked.connect(self.reject)
        row.addStretch(1); row.addWidget(self.btn_apply); row.addWidget(self.btn_reset); row.addWidget(self.btn_close)
        v.addLayout(row)

        self._timer = QTimer(self); self._timer.setSingleShot(True); self._timer.timeout.connect(self._update_preview)

        self._set_pix(self._disp_base)
        self._update_preview()
        # initial fit
        self.view.fit_to_item(self.pix)

    def _debounce_preview(self): self._timer.start(250)

    def _set_pix(self, rgb):
        arr = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        h, w, _ = arr.shape
        q = QImage(arr.data, w, h, 3*w, QImage.Format.Format_RGB888)
        self.pix.setPixmap(QPixmap.fromImage(q))
        self.scene.setSceneRect(self.pix.boundingRect())

    def _update_preview(self):
        clip = self.s_clip.value() / 10.0
        tile = self.s_tile.value()
        try:
            out = apply_clahe(self._disp_base, clip_limit=clip, tile_grid_size=(tile, tile))
            # Respect active mask, if present (preview works on _disp_base size)
            H, W = out.shape[:2]
            m = _get_active_mask_resized(self.doc, H, W)
            if m is not None:
                if out.ndim == 3:
                    M = np.repeat(m[:, :, None], out.shape[2], axis=2).astype(np.float32)
                else:
                    M = m.astype(np.float32)
                base = self._disp_base.astype(np.float32)
                out = np.clip(base * (1.0 - M) + out * M, 0.0, 1.0)

            self._set_pix(out)
            self._preview = out
        except Exception as e:
            QMessageBox.warning(self, "CLAHE", f"Preview failed:\n{e}")

    def _apply(self):
        try:
            clip = self.s_clip.value() / 10.0
            tile = self.s_tile.value()

            out = apply_clahe(self.orig, clip_limit=clip, tile_grid_size=(tile, tile))
            out = np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)

            # Mask-respectful commit
            H, W = out.shape[:2]
            m = _get_active_mask_resized(self.doc, H, W)
            if m is not None:
                base = np.asarray(self.doc.image, dtype=np.float32)
                # Normalize base into [0..1] for blending
                if base.dtype.kind in "ui":
                    maxv = float(np.iinfo(base.dtype).max)
                    base = base / max(1.0, maxv)
                else:
                    base = np.clip(base, 0.0, 1.0)

                if out.ndim == 3:
                    if base.ndim == 2:
                        base = base[:, :, None].repeat(out.shape[2], axis=2)
                    elif base.ndim == 3 and base.shape[2] == 1:
                        base = base.repeat(out.shape[2], axis=2)

                    M = np.repeat(m[:, :, None], out.shape[2], axis=2).astype(np.float32)
                    blended = np.clip(base * (1.0 - M) + out * M, 0.0, 1.0)
                else:
                    if base.ndim == 3 and base.shape[2] == 1:
                        base = base.squeeze(axis=2)
                    blended = np.clip(base * (1.0 - m) + out * m, 0.0, 1.0)

                out = blended.astype(np.float32, copy=False)

            if hasattr(self.doc, "set_image"):
                self.doc.set_image(out, step_name="CLAHE")
            elif hasattr(self.doc, "apply_numpy"):
                self.doc.apply_numpy(out, step_name="CLAHE")
            else:
                self.doc.image = out
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "CLAHE", f"Failed to apply:\n{e}")

    def _reset(self):
        self.s_clip.setValue(20); self.s_tile.setValue(8)
        self._set_pix(self._disp_base)
        self.view.fit_to_item(self.pix)
