# pro/history_explorer.py
from __future__ import annotations
from PyQt6.QtCore import Qt, QSize, QPointF, QEvent
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QListWidget, QPushButton, QLabel,
    QScrollArea, QWidget, QMessageBox, QSlider
)
from PyQt6.QtGui import QImage, QPixmap, QPainter
from PyQt6 import sip
import numpy as np

from .autostretch import autostretch

# ---------- helpers ----------
def _to_float01(img: np.ndarray) -> np.ndarray:
    if img is None:
        return None
    a = np.asarray(img)
    if a.dtype.kind in "ui":
        info = np.iinfo(a.dtype)
        return (a.astype(np.float32) / float(info.max)).clip(0, 1)
    if a.dtype.kind == "f":
        m = float(a.max()) if a.size else 1.0
        return (a.astype(np.float32) / (m if m > 0 else 1.0)).clip(0, 1)
    return a.astype(np.float32)

def _mk_qimage_rgb8(float01: np.ndarray) -> tuple[QImage, np.ndarray]:
    """Make a QImage (RGB888) and return it along with the backing uint8 buffer to keep alive."""
    f = float01
    if f.ndim == 2:
        f = np.stack([f]*3, axis=-1)
    elif f.ndim == 3 and f.shape[2] == 1:
        f = np.repeat(f, 3, axis=2)
    buf8 = (np.clip(f, 0.0, 1.0) * 255.0).astype(np.uint8, copy=False)
    buf8 = np.ascontiguousarray(buf8)
    h, w, _ = buf8.shape
    bpl = buf8.strides[0]
    ptr = sip.voidptr(buf8.ctypes.data)
    qimg = QImage(ptr, w, h, bpl, QImage.Format.Format_RGB888)
    return qimg, buf8

def _extract_undo_entries(doc):
    # Prefer the public getter we just added
    if hasattr(doc, "get_undo_stack"):
        return list(doc.get_undo_stack())

    # Fallbacks if needed
    for attr in ("_undo_stack", "undo_stack"):
        stack = getattr(doc, attr, None)
        if stack is None:
            continue
        out = []
        for item in stack:
            if isinstance(item, (list, tuple)):
                if len(item) >= 3:
                    img, meta, name = item[0], item[1] or {}, item[2] or "Unnamed"
                elif len(item) == 2:
                    img, meta = item
                    meta = meta or {}
                    name = meta.get("step_name", "Unnamed")
                else:
                    continue
                out.append((img, meta, str(name)))
        if out:
            return out
    return []



class HistoryExplorerDialog(QDialog):
    def __init__(self, document, parent=None):
        super().__init__(parent)
        self.setWindowTitle("History Explorer")
        self.setModal(False)
        self.doc = document

        self.setMinimumSize(700, 500)
        layout = QVBoxLayout(self)

        self.history_list = QListWidget()
        layout.addWidget(self.history_list)

        # Build a normalized list from the document
        self.undo_entries = _extract_undo_entries(self.doc)  # list[(img, meta, name)]
        self.items: list[tuple[np.ndarray, dict, str]] = []

        # Fill list (1-based) → undo states then current image
        for i, (img, meta, name) in enumerate(self.undo_entries):
            label = "1. Original Image" if i == 0 else f"{i+1}. {name or 'Unnamed'}"
            self.history_list.addItem(label)
            self.items.append((img, meta, name or "Unnamed"))

        # Append current image
        cur_img = getattr(self.doc, "image", None)
        cur_meta = getattr(self.doc, "metadata", {}) or {}
        self.history_list.addItem(f"{len(self.undo_entries)+1}. Current Image")
        self.items.append((cur_img, cur_meta, "Current Image"))

        self.history_list.itemDoubleClicked.connect(self._open_preview)

        row = QHBoxLayout()
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.close)
        row.addStretch(1)
        row.addWidget(btn_close)
        layout.addLayout(row)

    def _open_preview(self, item):
        row = self.history_list.row(item)
        if 0 <= row < len(self.items):
            img, meta, name = self.items[row]
            if img is None:
                QMessageBox.warning(self, "Preview", "No image stored for this step.")
                return
            pv = HistoryImagePreview(img, meta, self.doc, parent=self)
            pv.setWindowTitle(item.text())
            pv.show()
            mw = self._find_main_window()
            if mw and hasattr(mw, "_log"):
                mw._log(f"History: preview opened → {item.text()}")
        else:
            QMessageBox.warning(self, "Preview", "Invalid selection.")

    def _find_main_window(self):
        p = self.parent()
        while p is not None and not hasattr(p, "docman"):
            p = p.parent()
        return p


class HistoryImagePreview(QWidget):
    """
    Preview a single history entry with zoom/pan, optional display autostretch,
    compare vs current, and restore.
    """
    def __init__(self, image_data: np.ndarray, metadata: dict, document, parent=None):
        super().__init__(parent, Qt.WindowType.Window)
        self.doc = document
        self.image_data = image_data
        self.metadata = metadata or {}

        self.zoom = 1.0
        self._panning = False
        self._pan_start = QPointF()
        self._autostretch_on = False

        self._qimg_src = None
        self._buf8 = None

        # UI
        self.label = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        self.scroll = QScrollArea(widgetResizable=False)
        self.scroll.setWidget(self.label)
        self.scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scroll.viewport().installEventFilter(self)
        self.label.installEventFilter(self)

        # controls
        self.btn_stretch = QPushButton("Toggle AutoStretch")
        self.btn_stretch.clicked.connect(self._toggle_autostretch)

        self.btn_fit = QPushButton("Fit")
        self.btn_fit.clicked.connect(self._fit_to_view)

        self.btn_1to1 = QPushButton("1:1")
        self.btn_1to1.clicked.connect(lambda: self._set_zoom(1.0))

        self.btn_compare = QPushButton("Compare to Current…")
        self.btn_compare.clicked.connect(self._open_compare)

        self.btn_restore = QPushButton("Restore This Version")
        self.btn_restore.clicked.connect(self._restore)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(10, 800)
        self.slider.setValue(100)
        self.slider.valueChanged.connect(lambda v: self._set_zoom(v/100.0))

        ctrl = QHBoxLayout()
        ctrl.addWidget(self.btn_stretch)
        ctrl.addStretch(1)
        ctrl.addWidget(self.btn_fit)
        ctrl.addWidget(self.btn_1to1)
        ctrl.addWidget(self.slider)
        ctrl.addWidget(self.btn_compare)
        ctrl.addWidget(self.btn_restore)

        lay = QVBoxLayout(self)
        lay.addWidget(self.scroll, 1)
        lay.addLayout(ctrl)

        self._rebuild_source()
        self._fit_to_view()

    # data → qimage
    def _make_vis(self) -> np.ndarray:
        f = _to_float01(self.image_data)
        if f is None:
            return None
        if self._autostretch_on:
            try:
                return np.clip(autostretch(f, target_median=0.25, linked=False), 0, 1)
            except Exception:
                pass
        return np.clip(f, 0, 1)

    def _rebuild_source(self):
        vis = self._make_vis()
        if vis is None:
            self.label.clear(); self._qimg_src = None; self._buf8 = None
            return
        self._qimg_src, self._buf8 = _mk_qimage_rgb8(vis)
        self._update_scaled()

    # zoom/pan
    def _set_zoom(self, z: float):
        self.zoom = float(max(0.05, min(z, 8.0)))
        self.slider.blockSignals(True)
        self.slider.setValue(int(self.zoom * 100))
        self.slider.blockSignals(False)
        self._update_scaled()

    def _fit_to_view(self):
        if self._qimg_src is None:
            return
        vp = self.scroll.viewport().size()
        if self._qimg_src.width() == 0 or self._qimg_src.height() == 0:
            return
        s = min(vp.width() / self._qimg_src.width(), vp.height() / self._qimg_src.height())
        self._set_zoom(max(0.05, s))

    def _update_scaled(self):
        if self._qimg_src is None:
            return
        sw = max(1, int(self._qimg_src.width()  * self.zoom))
        sh = max(1, int(self._qimg_src.height() * self.zoom))
        scaled = self._qimg_src.scaled(sw, sh, Qt.AspectRatioMode.KeepAspectRatio,
                                       Qt.TransformationMode.SmoothTransformation)
        self.label.setPixmap(QPixmap.fromImage(scaled))
        self.label.resize(scaled.size())

    # actions
    def _toggle_autostretch(self):
        self._autostretch_on = not self._autostretch_on
        self._rebuild_source()

    def _open_compare(self):
        cur = getattr(self.doc, "image", None)
        if cur is None:
            QMessageBox.warning(self, "Compare", "No current image to compare.")
            return

        win = QWidget(self, Qt.WindowType.Window)
        win.setWindowTitle("Compare with Current")
        win.resize(900, 700)

        v = QVBoxLayout(win)
        self.slider_widget = ComparisonSlider(self.image_data, cur, parent=win)
        v.addWidget(self.slider_widget, 1)

        bar = QHBoxLayout()
        b_out = QPushButton("Zoom Out"); b_in = QPushButton("Zoom In")
        b_fit = QPushButton("Fit"); b_1  = QPushButton("1:1")
        b_st  = QPushButton("Toggle AutoStretch")
        b_out.clicked.connect(self.slider_widget.zoom_out)
        b_in.clicked.connect(self.slider_widget.zoom_in)
        b_fit.clicked.connect(self.slider_widget.fit_to_view)
        b_1.clicked.connect(lambda: self.slider_widget.set_zoom(1.0))
        b_st.clicked.connect(self.slider_widget.toggle_autostretch)
        bar.addWidget(b_out); bar.addWidget(b_in); bar.addWidget(b_fit); bar.addWidget(b_1)
        bar.addStretch(1); bar.addWidget(b_st)
        v.addLayout(bar)

        win.show()
        mw = self._find_main_window()
        if mw and hasattr(mw, "_log"):
            mw._log("History: opened Compare with Current.")

    def _restore(self):
        try:
            # Prefer a method that records step name if available
            if hasattr(self.doc, "set_image"):
                self.doc.set_image(self.image_data.copy(), {"step_name": "Restored from History"})
            elif hasattr(self.doc, "update_image"):
                self.doc.update_image(self.image_data.copy(), {"step_name": "Restored from History"})
            else:
                QMessageBox.critical(self, "Restore", "Document does not support setting image.")
                return
            mw = self._find_main_window()
            if mw and hasattr(mw, "_log"):
                mw._log("History: restored image from history.")
            self.close()
        except Exception as e:
            QMessageBox.critical(self, "Restore failed", str(e))

    def _find_main_window(self):
        p = self.parent()
        while p is not None and not hasattr(p, "docman"):
            p = p.parent()
        return p

    # input
    def eventFilter(self, obj, ev):
        if obj is self.scroll.viewport():
            if ev.type() == QEvent.Type.Wheel:
                if ev.modifiers() & Qt.KeyboardModifier.ControlModifier:
                    if self._qimg_src is None or self.label.pixmap() is None:
                        return True
                    factor = 1.25 if ev.angleDelta().y() > 0 else 1/1.25
                    pos_vp = ev.position()
                    pos_lb = self.label.mapFrom(self.scroll.viewport(), pos_vp.toPoint())
                    old = self.label.pixmap().size()
                    rel_x = pos_lb.x() / max(1, old.width())
                    rel_y = pos_lb.y() / max(1, old.height())
                    self._set_zoom(self.zoom * factor)
                    new = self.label.pixmap().size()
                    hbar = self.scroll.horizontalScrollBar()
                    vbar = self.scroll.verticalScrollBar()
                    hbar.setValue(int(rel_x * new.width()  - self.scroll.viewport().width()/2))
                    vbar.setValue(int(rel_y * new.height() - self.scroll.viewport().height()/2))
                    return True
                return False

        if obj is self.scroll.viewport() or obj is self.label:
            if ev.type() == QEvent.Type.MouseButtonPress and ev.button() == Qt.MouseButton.LeftButton:
                self._panning = True
                self._pan_start = ev.position()
                self.scroll.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
                return True
            if ev.type() == QEvent.Type.MouseMove and self._panning:
                d = ev.position() - self._pan_start
                hbar = self.scroll.horizontalScrollBar()
                vbar = self.scroll.verticalScrollBar()
                hbar.setValue(hbar.value() - int(d.x()))
                vbar.setValue(vbar.value() - int(d.y()))
                self._pan_start = ev.position()
                return True
            if ev.type() == QEvent.Type.MouseButtonRelease and ev.button() == Qt.MouseButton.LeftButton:
                self._panning = False
                self.scroll.viewport().setCursor(Qt.CursorShape.ArrowCursor)
                return True

        return super().eventFilter(obj, ev)


class ComparisonSlider(QWidget):
    """Before/after slider with Ctrl+wheel zoom, Fit, 1:1, optional display autostretch."""
    def __init__(self, before_image: np.ndarray, after_image: np.ndarray, parent=None):
        super().__init__(parent)
        self.before = np.asarray(before_image)
        self.after  = np.asarray(after_image)
        self.zoom = 1.0
        self.autostretch_on = False
        self.slider_pos = 0.5

        self._q_before = None; self._buf_before = None
        self._q_after  = None; self._buf_after  = None
        self.setMouseTracking(True)
        self.setMinimumSize(400, 300)
        self._rebuild()

    def _mk_vis(self, a: np.ndarray) -> np.ndarray:
        f = _to_float01(a)
        if self.autostretch_on:
            try:
                return np.clip(autostretch(f, target_median=0.25, linked=False), 0, 1)
            except Exception:
                pass
        return np.clip(f, 0, 1)

    def _rebuild(self):
        qb, bb = _mk_qimage_rgb8(self._mk_vis(self.before))
        qa, ba = _mk_qimage_rgb8(self._mk_vis(self.after))
        self._q_before, self._buf_before = qb, bb
        self._q_after,  self._buf_after  = qa, ba

    # public controls
    def set_zoom(self, z: float):
        self.zoom = float(max(0.05, min(z, 8.0))); self.update()
    def zoom_in(self):  self.set_zoom(self.zoom * 1.25)
    def zoom_out(self): self.set_zoom(self.zoom / 1.25)
    def fit_to_view(self):
        if not self._q_before: return
        W,H = self.width(), self.height()
        iw,ih = self._q_before.width(), self._q_before.height()
        if iw==0 or ih==0: return
        self.set_zoom(min(W/iw, H/ih))

    def toggle_autostretch(self):
        self.autostretch_on = not self.autostretch_on
        self._rebuild(); self.update()

    # painting & input
    def paintEvent(self, _ev):
        if not self._q_before or not self._q_after:
            return
        p = QPainter(self)
        W,H = self.width(), self.height()
        iw, ih = self._q_before.width(), self._q_before.height()
        if iw==0 or ih==0: return
        s = min(W/iw, H/ih) * self.zoom
        tw, th = int(iw*s), int(ih*s)
        b = self._q_before.scaled(tw, th, Qt.AspectRatioMode.KeepAspectRatio,
                                  Qt.TransformationMode.SmoothTransformation)
        a = self._q_after.scaled(tw, th, Qt.AspectRatioMode.KeepAspectRatio,
                                 Qt.TransformationMode.SmoothTransformation)
        ox = (W - b.width()) // 2
        oy = (H - b.height()) // 2
        cut = int(W * self.slider_pos)

        p.save(); p.setClipRect(0, 0, cut, H); p.drawImage(ox, oy, b); p.restore()
        p.save(); p.setClipRect(cut, 0, W-cut, H); p.drawImage(ox, oy, a); p.restore()

        p.setPen(Qt.GlobalColor.red); p.drawLine(cut, 0, cut, H)

    def mousePressEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            self._set_div(ev.position().x())
    def mouseMoveEvent(self, ev):
        if ev.buttons() & Qt.MouseButton.LeftButton:
            self._set_div(ev.position().x())
    def _set_div(self, x):
        self.slider_pos = min(max(x / max(1, self.width()), 0.0), 1.0); self.update()
    def wheelEvent(self, ev):
        if ev.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.set_zoom(self.zoom * (1.25 if ev.angleDelta().y() > 0 else 0.8))
            ev.accept()
        else:
            ev.ignore()
