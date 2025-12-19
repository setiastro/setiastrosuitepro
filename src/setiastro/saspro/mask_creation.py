# pro/mask_creation.py
from __future__ import annotations
import uuid
import numpy as np
import math

# Optional deps
try:
    import cv2
except Exception:
    cv2 = None
try:
    import sep
except Exception:
    sep = None

from PyQt6.QtCore import Qt, QPointF, QRectF, QTimer, QEvent
from PyQt6.QtGui import (
    QImage, QPixmap, QPainter, QColor, QPen, QBrush,
    QPainterPath, QWheelEvent, QPolygonF
)
from PyQt6.QtWidgets import (
    QInputDialog, QMessageBox, QFileDialog,   # QFileDialog only used if you later add “export”
    QDialog, QDialogButtonBox,
    QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QComboBox, QSlider, QCheckBox, QButtonGroup, QGroupBox,
    QScrollArea, QSizePolicy,
    QGraphicsView, QGraphicsScene, QGraphicsItem,
    QGraphicsPixmapItem, QGraphicsPathItem, QGraphicsPolygonItem,
    QGraphicsEllipseItem, QGraphicsRectItem, QMdiSubWindow, QLabel
)

from .masks_core import MaskLayer
from setiastro.saspro.widgets.themed_buttons import themed_toolbtn


# ---------- small utils ----------

def _to_qpixmap01(img01: np.ndarray) -> QPixmap:
    a = np.clip(img01, 0.0, 1.0)
    if a.ndim == 2:
        buf = (a * 255).astype(np.uint8)
        h, w = buf.shape
        qimg = QImage(buf.data, w, h, w, QImage.Format.Format_Grayscale8)
    else:
        buf = (a * 255).astype(np.uint8)
        h, w, _ = buf.shape
        qimg = QImage(buf.data, w, h, buf.strides[0], QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)


def _find_main_window(w):
    p = w
    from PyQt6.QtWidgets import QMainWindow
    while p is not None and not isinstance(p, QMainWindow):
        p = p.parent()
    return p

def _push_numpy_as_new_document(owner_widget, arr01: np.ndarray, default_name: str = "Mask") -> bool:
    mw = _find_main_window(owner_widget)
    if mw is None or not hasattr(mw, "docman"):
        QMessageBox.warning(owner_widget, "Cannot Create Document", "Main window / DocManager not found.")
        return False

    # Ask for the document name
    name, ok = QInputDialog.getText(owner_widget, "New Document Name", "Name:", text=default_name)
    if not ok:
        return False

    # Ensure float32 in [0..1]
    img = np.clip(arr01.astype(np.float32, copy=False), 0.0, 1.0)

    # This sets metadata['display_name'] via DocManager and emits documentAdded
    doc = mw.docman.open_array(img, title=name)

    # Nothing else required: AstroSuiteProMainWindow._open_subwindow_for_added_doc
    # will create/show the subwindow.
    if hasattr(mw, "_log"):
        mw._log(f"Created new document from mask: {doc.display_name()}")
    return True


# ---------- Interactive ellipse handles ----------
class HandleItem(QGraphicsRectItem):
    SIZE = 8
    def __init__(self, role: str, parent_ellipse: QGraphicsEllipseItem):
        super().__init__(-self.SIZE/2, -self.SIZE/2, self.SIZE, self.SIZE, parent_ellipse)
        self.role = role
        self.parent_ellipse = parent_ellipse
        self.setBrush(QColor(255, 0, 0))
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, False)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)

        cursors = {
            'top': Qt.CursorShape.SizeVerCursor,
            'bottom': Qt.CursorShape.SizeVerCursor,
            'left': Qt.CursorShape.SizeHorCursor,
            'right': Qt.CursorShape.SizeHorCursor,
            'rotate': Qt.CursorShape.OpenHandCursor,
        }
        self.setCursor(cursors[role])

        self._lastScenePos = None
        # extra state for rotation
        self._centerScene = None
        self._startAngle = None
        self._startRotation = None

    def mousePressEvent(self, ev):
        if self.role == 'rotate':
            # Store center of ellipse in scene coords
            rect = self.parent_ellipse.rect()
            center_item = rect.center()
            self._centerScene = self.parent_ellipse.mapToScene(center_item)

            # Starting angle from center → mouse
            p = ev.scenePos()
            dx = p.x() - self._centerScene.x()
            dy = p.y() - self._centerScene.y()
            self._startAngle = math.degrees(math.atan2(dy, dx))

            # Store current item rotation
            self._startRotation = self.parent_ellipse.rotation()

            # Optional: change cursor to "grabbing"
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            ev.accept()
            return

        # Non-rotate handles use the old dx/dy code path
        self._lastScenePos = ev.scenePos()
        ev.accept()

    def mouseMoveEvent(self, ev):
        if self.role == 'rotate':
            if self._centerScene is None or self._startAngle is None or self._startRotation is None:
                ev.accept()
                return

            p = ev.scenePos()
            dx = p.x() - self._centerScene.x()
            dy = p.y() - self._centerScene.y()

            # Current angle from center → mouse
            current_angle = math.degrees(math.atan2(dy, dx))

            # Delta relative to the original grab angle
            delta = current_angle - self._startAngle

            # Set absolute rotation: starting rotation + delta
            self.parent_ellipse.setRotation(self._startRotation + delta)
            ev.accept()
            return

        # Resize handles: same as before
        if self._lastScenePos is None:
            self._lastScenePos = ev.scenePos()
        dx = ev.scenePos().x() - self._lastScenePos.x()
        dy = ev.scenePos().y() - self._lastScenePos.y()
        self.parent_ellipse.interactiveResize(self.role, dx, dy)
        self._lastScenePos = ev.scenePos()
        ev.accept()

    def mouseReleaseEvent(self, ev):
        # Reset rotation state and cursor
        if self.role == 'rotate':
            self._centerScene = None
            self._startAngle = None
            self._startRotation = None
            self.setCursor(Qt.CursorShape.OpenHandCursor)

        self._lastScenePos = None
        ev.accept()



class InteractiveEllipseItem(QGraphicsEllipseItem):
    def __init__(self, rect: QRectF):
        super().__init__(rect)
        self._resizing = False
        self.setTransformOriginPoint(self.rect().center())
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable |
            QGraphicsItem.GraphicsItemFlag.ItemIsSelectable |
            QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )

        # Cosmetic pen: stays the same thickness on screen regardless of zoom
        pen = QPen(QColor(0, 255, 0), 2)
        pen.setCosmetic(True)
        self.setPen(pen)

        self.handles = {r: HandleItem(r, self) for r in ('top','bottom','left','right','rotate')}
        self.updateHandles()

    def updateHandles(self):
        r = self.rect()
        cx, cy = r.center().x(), r.center().y()
        for h in self.handles.values():
            h.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, False)
        positions = {
            'top':    QPointF(cx, r.top()),
            'bottom': QPointF(cx, r.bottom()),
            'left':   QPointF(r.left(), cy),
            'right':  QPointF(r.right(), cy),
            'rotate': QPointF(cx, r.top()-20)
        }
        for role, h in self.handles.items():
            h.setPos(self.mapFromScene(self.mapToScene(positions[role])))
        for h in self.handles.values():
            h.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)

    def itemChange(self, change, value):
        if change in (QGraphicsItem.GraphicsItemChange.ItemPositionChange,
                      QGraphicsItem.GraphicsItemChange.ItemTransformChange,
                      QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged):
            QTimer.singleShot(0, self.updateHandles)
        return super().itemChange(change, value)

    def interactiveResize(self, role: str, dx: float, dy: float):
        if self._resizing:
            return
        r = QRectF(self.rect())
        if role == 'top':
            r.setTop(r.top() + dy)
        elif role == 'bottom':
            r.setBottom(r.bottom() + dy)
        elif role == 'left':
            r.setLeft(r.left() + dx)
        elif role == 'right':
            r.setRight(r.right() + dx)
        elif role == 'rotate':
            # rotation is handled in HandleItem.mouseMoveEvent now
            return
        self._resizing = True
        self.prepareGeometryChange()
        self.setRect(r)
        self.updateHandles()
        self._resizing = False


# ---------- Canvas ----------

class MaskCanvas(QGraphicsView):
    def __init__(self, image01: np.ndarray, parent=None):
        super().__init__(parent)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)

        # scene + background image
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.bg_item = QGraphicsPixmapItem(_to_qpixmap01(image01))
        self.scene.addItem(self.bg_item)

        # --- NEW: basic zoom state ---
        self._zoom = 1.0
        self._min_zoom = 0.05
        self._max_zoom = 8.0
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        # Make sure the scene rect matches the image so fit works perfectly
        self.setSceneRect(self.bg_item.boundingRect())

        self.mode = 'polygon'
        self.temp_path: QGraphicsPathItem | None = None
        self.poly_points: list[QPointF] = []
        self.temp_ellipse: QGraphicsEllipseItem | None = None
        self.ellipse_origin: QPointF | None = None
        self.shapes: list[QGraphicsItem] = []

    # ------------------- NEW: Zoom API -------------------
    def set_zoom(self, z: float):
        """Absolute zoom setter (resets transform, then scales)."""
        self._zoom = max(self._min_zoom, min(float(z), self._max_zoom))
        self.resetTransform()
        self.scale(self._zoom, self._zoom)

    def zoom_in(self):
        self.set_zoom(self._zoom * 1.25)

    def zoom_out(self):
        self.set_zoom(self._zoom / 1.25)

    def fit_to_view(self):
        """Fit the background image into the viewport (Keeps aspect)."""
        pm = self.bg_item.pixmap()
        if pm.isNull():
            return
        vw = max(1, self.viewport().width())
        vh = max(1, self.viewport().height())
        iw = pm.width()
        ih = pm.height()
        if iw == 0 or ih == 0:
            return
        s = min(vw / iw, vh / ih)
        self.set_zoom(s)

    def wheelEvent(self, ev):
        """Ctrl + wheel → zoom; otherwise default scroll behavior."""
        if ev.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.set_zoom(self._zoom * (1.25 if ev.angleDelta().y() > 0 else 0.8))
            ev.accept()
            return
        super().wheelEvent(ev)
    # ----------------- END: Zoom API ---------------------

    def set_mode(self, mode: str):
        assert mode in ('polygon', 'ellipse', 'select')
        self.mode = mode

    def clear_shapes(self):
        for it in list(self.shapes):
            self.scene.removeItem(it)
        self.shapes.clear()

    def select_entire_image(self):
        self.clear_shapes()
        rect = self.bg_item.boundingRect()
        poly = QGraphicsPolygonItem(QPolygonF([rect.topLeft(), rect.topRight(),
                                            rect.bottomRight(), rect.bottomLeft()]))
        poly.setBrush(QColor(0, 255, 0, 50))
        pen = QPen(QColor(0, 255, 0), 2)
        pen.setCosmetic(True)
        poly.setPen(pen)
        poly.setFlags(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable |
                    QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        self.scene.addItem(poly); self.shapes.append(poly)


    def mousePressEvent(self, ev):
        pt = self.mapToScene(ev.pos())
        if self.mode == 'ellipse' and ev.button() == Qt.MouseButton.LeftButton:
            for it in self.items(ev.pos()):
                if isinstance(it, (InteractiveEllipseItem, HandleItem)):
                    return super().mousePressEvent(ev)

        if self.mode == 'polygon' and ev.button() == Qt.MouseButton.LeftButton:
            self.poly_points = [pt]
            path = QPainterPath(pt)
            self.temp_path = QGraphicsPathItem(path)
            pen = QPen(QColor(255, 0, 0), 2, Qt.PenStyle.DashLine)
            pen.setCosmetic(True)
            self.temp_path.setPen(pen)
            self.scene.addItem(self.temp_path)
            return

        if self.mode == 'ellipse' and ev.button() == Qt.MouseButton.LeftButton:
            self.ellipse_origin = pt
            self.temp_ellipse = QGraphicsEllipseItem(QRectF(pt, pt))
            pen = QPen(QColor(0, 255, 0), 2, Qt.PenStyle.DashLine)
            pen.setCosmetic(True)
            self.temp_ellipse.setPen(pen)
            self.scene.addItem(self.temp_ellipse)
            return

        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        pt = self.mapToScene(ev.pos())
        if self.mode == 'ellipse' and self.temp_ellipse is not None:
            self.temp_ellipse.setRect(QRectF(self.ellipse_origin, pt).normalized())
        elif self.mode == 'polygon' and self.temp_path:
            self.poly_points.append(pt)
            p = QPainterPath(self.poly_points[0])
            for q in self.poly_points[1:]:
                p.lineTo(q)
            self.temp_path.setPath(p)
        else:
            super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        if self.mode == 'ellipse' and self.temp_ellipse is not None:
            final_rect = self.temp_ellipse.rect().normalized()
            self.scene.removeItem(self.temp_ellipse); self.temp_ellipse = None
            if final_rect.width() > 4 and final_rect.height() > 4:
                local_rect = QRectF(0, 0, final_rect.width(), final_rect.height())
                ell = InteractiveEllipseItem(local_rect)
                ell.setBrush(QBrush(Qt.BrushStyle.NoBrush))
                ell.setZValue(1)
                ell.setPos(final_rect.topLeft())
                self.scene.addItem(ell); self.shapes.append(ell)
            return

        if self.mode == 'polygon' and self.temp_path:
            poly = QGraphicsPolygonItem(QPolygonF(self.poly_points))
            poly.setBrush(QColor(0, 255, 0, 50))
            pen = QPen(QColor(0, 255, 0), 2)
            pen.setCosmetic(True)
            poly.setPen(pen)
            poly.setFlags(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable |
                        QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
            self.scene.removeItem(self.temp_path); self.temp_path = None
            self.scene.addItem(poly); self.shapes.append(poly)
            return

        super().mouseReleaseEvent(ev)

    def create_mask(self) -> np.ndarray:
        if cv2 is None:
            raise RuntimeError("OpenCV (cv2) is required for mask creation.")
        h = self.bg_item.pixmap().height()
        w = self.bg_item.pixmap().width()
        mask = np.zeros((h, w), dtype=np.uint8)

        for s in self.shapes:
            if isinstance(s, QGraphicsPolygonItem):
                pts = s.polygon()
                arr = np.array([[p.x(), p.y()] for p in pts], np.int32)
                cv2.fillPoly(mask, [arr], 1)
            elif isinstance(s, InteractiveEllipseItem):
                r = s.rect()
                scenep = s.mapToScene(r.center())
                cx, cy = int(scenep.x()), int(scenep.y())
                rx = int(max(1, r.width() / 2))
                ry = int(max(1, r.height() / 2))
                angle = float(s.rotation())
                cv2.ellipse(mask, (cx, cy), (rx, ry), angle, 0, 360, 1, -1)

        return (mask > 0).astype(np.float32)

    # Fit once on first show (nice UX)
    def showEvent(self, ev):
        super().showEvent(ev)
        QTimer.singleShot(0, self.fit_to_view)



# ---------- Live preview ----------

class LivePreviewDialog(QDialog):
    def __init__(self, original_image01: np.ndarray, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Live Mask Preview"))
        self.label = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        lay = QVBoxLayout(self); lay.addWidget(self.label)
        self.resize(300, 300)
        self.base_pixmap = _to_qpixmap01(original_image01)
        self.max_alpha = 150

    def update_mask(self, mask01: np.ndarray):
        h, w = mask01.shape
        alpha = (np.clip(mask01, 0, 1) * self.max_alpha).astype(np.uint8)
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[..., 0] = 255  # red
        rgba[..., 3] = alpha
        overlay_qimg = QImage(rgba.data, w, h, 4*w, QImage.Format.Format_RGBA8888)
        overlay = QPixmap.fromImage(overlay_qimg)
        canvas = QPixmap(self.base_pixmap)
        p = QPainter(canvas); p.drawPixmap(0, 0, overlay); p.end()
        self.label.setPixmap(canvas.scaled(self.label.size(),
                                           Qt.AspectRatioMode.KeepAspectRatio,
                                           Qt.TransformationMode.SmoothTransformation))


# ---------- Preview (push-as-doc) ----------

class MaskPreviewDialog(QDialog):
    """Scrollable preview + 'Push as New Document…'."""
    def __init__(self, mask01: np.ndarray, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Mask Preview"))
        self.mask = np.clip(mask01, 0, 1).astype(np.float32)

        self.scroll = QScrollArea(self); self.scroll.setWidgetResizable(False)
        self.scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        self.label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.pixmap = self._to_pixmap(self.mask); self.label.setPixmap(self.pixmap)
        self.scroll.setWidget(self.label)

        btns = QHBoxLayout()
        b_in   = themed_toolbtn("zoom-in", "Zoom In")
        b_out  = themed_toolbtn("zoom-out", "Zoom Out")
        b_fit  = themed_toolbtn("zoom-fit-best", "Fit to Preview")


        b_push = QPushButton(self.tr("Push as New Document…"))
        b_in.clicked.connect(lambda: self._zoom(1.2))
        b_out.clicked.connect(lambda: self._zoom(1/1.2))
        b_fit.clicked.connect(self._fit)
        b_push.clicked.connect(self.push_as_new_document)
        for b in (b_in, b_out, b_fit, b_push):
            btns.addWidget(b)

        lay = QVBoxLayout(self); lay.addWidget(self.scroll); lay.addLayout(btns)
        self.scale = 1.0; self.setMinimumSize(600, 400)

    def _to_pixmap(self, mask01: np.ndarray) -> QPixmap:
        m8 = (np.clip(mask01, 0, 1) * 255).astype(np.uint8)
        h, w = m8.shape
        qimg = QImage(m8.data, w, h, w, QImage.Format.Format_Grayscale8)
        return QPixmap.fromImage(qimg)

    def _zoom(self, factor: float):
        self.scale *= factor
        scaled = self.pixmap.scaled(self.pixmap.size() * self.scale,
                                    Qt.AspectRatioMode.KeepAspectRatio,
                                    Qt.TransformationMode.SmoothTransformation)
        self.label.setPixmap(scaled); self.label.resize(scaled.size())

    def _fit(self):
        vp = self.scroll.viewport().size()
        if self.pixmap.width() and self.pixmap.height():
            s = min(vp.width()/self.pixmap.width(), vp.height()/self.pixmap.height())
            self.scale = max(0.05, s)
            self._zoom(1.0)

    def push_as_new_document(self):
        if self.mask is None:
            QMessageBox.warning(self, "No Mask", "No mask to push.")
            return

        # Walk up to the main window to reach DocManager
        host = self.parent()
        while host is not None and not hasattr(host, "docman"):
            host = host.parent()
        if host is None or not hasattr(host, "docman"):
            QMessageBox.warning(self, "No DocManager", "Could not find the document manager.")
            return

        # Ask for a friendly name
        name, ok = QInputDialog.getText(self, "New Document Name", "Name:", text="Mask")
        if not ok:
            return

        # Ensure float32; a mask is mono by definition
        img = self.mask.astype(np.float32, copy=False)
        meta = {
            "bit_depth": "32-bit floating point",
            "is_mono": True,
            "original_format": "fits",
        }

        # Create the doc → this emits documentAdded, which the main window now handles via _spawn_subwindow_for
        new_doc = host.docman.create_document(img, metadata=meta, name=(name or "Mask"))

        # Focus it
        try:
            sw = host._find_subwindow_for_doc(new_doc)
            if sw:
                host.mdi.setActiveSubWindow(sw)
        except Exception:
            pass

        self.accept()



# ---------- Mask dialog ----------

class MaskCreationDialog(QDialog):
    """Mask creation UI for SASpro documents (returns a np mask on OK)."""
    def __init__(self, image01: np.ndarray, parent=None, auto_push_on_ok: bool = True):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Mask Creation"))
        self.image = np.asarray(image01, dtype=np.float32).copy()
        self.mask: np.ndarray | None = None
        self.live_preview = LivePreviewDialog(self.image, parent=self)

        self.mask_type = "Binary"
        self.blur_amount = 0

        # <- this was missing
        self.auto_push_on_ok = auto_push_on_ok

        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # Mode toolbar
        mode_bar = QHBoxLayout()
        self.free_btn = QPushButton(self.tr("Freehand")); self.free_btn.setCheckable(True)
        self.ellipse_btn = QPushButton(self.tr("Ellipse")); self.ellipse_btn.setCheckable(True)
        self.select_btn = QPushButton(self.tr("Select Entire Image")); self.select_btn.setCheckable(True)
        group = QButtonGroup(self); group.setExclusive(True)
        for b in (self.free_btn, self.ellipse_btn, self.select_btn):
            b.setAutoExclusive(True); group.addButton(b)
            b.setStyleSheet("""
                QPushButton { padding:6px; border:1px solid #888; border-radius:4px; background:transparent; }
                QPushButton:checked { background-color:#0078d4; color:white; border-color:#005a9e; }
            """)
        for btn, mode in ((self.free_btn,'polygon'), (self.ellipse_btn,'ellipse'), (self.select_btn,'select')):
            btn.clicked.connect(lambda _=False, m=mode: self._set_mode(m))
            mode_bar.addWidget(btn)
        self.free_btn.setChecked(True)
        layout.addLayout(mode_bar)

        zoom_bar = QHBoxLayout()
        z_out = themed_toolbtn("zoom-out", "Zoom Out")
        z_in  = themed_toolbtn("zoom-in", "Zoom In")
        z_fit = themed_toolbtn("zoom-fit-best", "Fit to Preview")
        z_out.clicked.connect(lambda: self._zoom_canvas(1/1.25))
        z_in.clicked.connect(lambda: self._zoom_canvas(1.25))
        z_fit.clicked.connect(self._fit_canvas)
        zoom_bar.addWidget(z_out); zoom_bar.addWidget(z_in); zoom_bar.addWidget(z_fit)
        layout.addLayout(zoom_bar)

        # Canvas
        self.canvas = MaskCanvas(self.image)
        layout.addWidget(self.canvas, 1)

        # Mask type & blur
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Mask Type:"))
        self.type_dd = QComboBox()
        self.type_dd.addItems([
            "Binary","Range Selection","Lightness","Chrominance","Star Mask",
            "Color: Red","Color: Orange","Color: Yellow",
            "Color: Green","Color: Cyan","Color: Blue","Color: Magenta"
        ])
        self.type_dd.currentTextChanged.connect(lambda t: setattr(self, 'mask_type', t))
        controls.addWidget(self.type_dd)

        controls.addWidget(QLabel(self.tr("Edge Blur (px):")))
        self.blur_slider = QSlider(Qt.Orientation.Horizontal); self.blur_slider.setRange(0, 300)
        self.blur_slider.valueChanged.connect(lambda v: setattr(self, 'blur_amount', int(v)))
        controls.addWidget(self.blur_slider)
        self.blur_lbl = QLabel("0")
        self.blur_slider.valueChanged.connect(lambda v: self.blur_lbl.setText(str(v)))
        controls.addWidget(self.blur_lbl)
        layout.addLayout(controls)

        # Range Selection
        self.range_box = QGroupBox("Range Selection"); g = QGridLayout(self.range_box)
        def add_slider(row: int, name: str, maxv: int):
            g.addWidget(QLabel(name + ":"), row, 0)
            s = QSlider(Qt.Orientation.Horizontal); s.setRange(0, maxv)
            s.setValue(maxv if name == "Upper" else 0)
            lbl = QLabel(f"{(s.value()/maxv):.2f}")
            s.valueChanged.connect(lambda v, l=lbl, s=s: l.setText(f"{v/s.maximum():.2f}"))
            s.valueChanged.connect(self._update_live_preview)
            g.addWidget(s, row, 1); g.addWidget(lbl, row, 2)
            return s, lbl
        self.lower_sl, _ = add_slider(0, "Lower", 100)
        self.upper_sl, _ = add_slider(1, "Upper", 100)
        self.fuzz_sl,  _ = add_slider(2, "Transition", 100)
        g.addWidget(QLabel("Blur:"), 3, 0)
        self.smooth_sl = QSlider(Qt.Orientation.Horizontal)
        self.smooth_sl.setRange(1, 200)           # σ in pixels
        self.smooth_sl.setValue(3)                 # a sensible default
        g.addWidget(self.smooth_sl, 3, 1)

        self.smooth_lbl = QLabel("σ = 3 px")
        g.addWidget(self.smooth_lbl, 3, 2)

        # live label + live preview
        def _upd_smooth(v):
            self.smooth_lbl.setText(f"σ = {int(v)} px")
            self._update_live_preview()
        self.smooth_sl.valueChanged.connect(_upd_smooth)
        self.link_cb   = QCheckBox("Link limits"); g.addWidget(self.link_cb, 0, 3, 2, 1)
        self.screen_cb = QCheckBox("Screening");   g.addWidget(self.screen_cb, 4, 0, 1, 4)
        self.light_cb  = QCheckBox("Lightness");   g.addWidget(self.light_cb, 5, 0, 1, 4)
        self.invert_cb = QCheckBox("Invert");      g.addWidget(self.invert_cb, 6, 0, 1, 4)
        self.lower_sl.valueChanged.connect(self._on_linked)
        self.link_cb.toggled.connect(self._on_link_switch)
        layout.addWidget(self.range_box); self.range_box.hide()
        self.type_dd.currentTextChanged.connect(self._on_type_changed)

        # Preview & Clear
        rowb = QHBoxLayout()
        b_preview = QPushButton("Preview Mask"); b_preview.clicked.connect(self._preview_mask)
        b_clear = QPushButton("Clear Shapes");   b_clear.clicked.connect(self._clear_shapes)
        rowb.addWidget(b_preview); rowb.addWidget(b_clear)
        layout.addLayout(rowb)

        # OK / Cancel
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, parent=self)
        btns.accepted.connect(self._accept_apply); btns.rejected.connect(self.reject)
        layout.addWidget(btns)

        self.canvas.installEventFilter(self)

        self.resize(980, 640)

    # ---- callbacks
    def _set_mode(self, mode: str):
        self.canvas.set_mode(mode)
        if mode == 'select':
            self.canvas.select_entire_image()

    def _clear_shapes(self):
        self.canvas.clear_shapes()

    def _on_type_changed(self, txt: str):
        show = (txt == "Range Selection")
        self.range_box.setVisible(show)
        if show:
            if not self.live_preview.isVisible():
                self.live_preview.show()
            self._update_live_preview()
        else:
            if self.live_preview.isVisible():
                self.live_preview.close()

    def _on_link_switch(self, checked: bool):
        if checked:
            self.upper_sl.setValue(self.lower_sl.value())

    def _on_linked(self, v: int):
        if self.link_cb.isChecked():
            self.upper_sl.setValue(v)

    # ---- generators
    def _component_lightness(self) -> np.ndarray:
        if self.image.ndim == 3:
            return (self.image[..., 0]*0.2989 + self.image[..., 1]*0.5870 + self.image[..., 2]*0.1140).astype(np.float32)
        return self.image.astype(np.float32)

    def _range_selection_mask(self, comp01: np.ndarray, L, U, fuzz, smooth, screening, invert):
        m = np.zeros_like(comp01, dtype=np.float32)
        inside = (comp01 >= L) & (comp01 <= U); m[inside] = 1.0
        if fuzz > 0:
            ramp = (comp01 - (L - fuzz)) / max(fuzz, 1e-12); m += np.clip(ramp, 0, 1)
            ramp2 = ((U + fuzz) - comp01) / max(fuzz, 1e-12); m *= np.clip(ramp2, 0, 1)
        if screening: m *= comp01
        if smooth > 0 and cv2 is not None: m = cv2.GaussianBlur(m, (0, 0), float(smooth))
        if invert: m = 1.0 - m
        return np.clip(m, 0, 1)

    def _generate_color_mask(self, color: str) -> np.ndarray:
        if cv2 is None:
            QMessageBox.warning(self, "Missing OpenCV", "Color masks require OpenCV (cv2).")
            return np.zeros(self.image.shape[:2], dtype=np.float32)
        ranges = {
            "Red": [(0, 10), (350, 360)], "Orange": [(10, 40)], "Yellow": [(40, 70)],
            "Green": [(70, 170)], "Cyan": [(170, 200)], "Blue": [(200, 270)], "Magenta": [(270, 350)],
        }
        if color not in ranges or self.image.ndim != 3:
            return np.zeros(self.image.shape[:2], dtype=np.float32)
        rgb8 = (np.clip(self.image, 0, 1) * 255).astype(np.uint8)
        hls = cv2.cvtColor(rgb8, cv2.COLOR_RGB2HLS)
        hue = (hls[..., 0].astype(np.float32) / 180.0) * 360.0
        mask = np.zeros(hue.shape, dtype=np.float32)
        for lo, hi in ranges[color]:
            if lo < hi:
                mask = np.maximum(mask, ((hue >= lo) & (hue <= hi)).astype(np.float32))
            else:
                mask = np.maximum(mask, ((hue >= lo) | (hue <= hi)).astype(np.float32))
        return mask

    def _generate_chrominance(self) -> np.ndarray:
        if cv2 is None or self.image.ndim != 3:
            QMessageBox.warning(self, "Needs RGB + OpenCV", "Chrominance mask requires an RGB image and OpenCV.")
            return np.zeros(self.image.shape[:2], dtype=np.float32)
        rgb8 = (np.clip(self.image, 0, 1) * 255).astype(np.uint8)
        ycrcb = cv2.cvtColor(rgb8, cv2.COLOR_RGB2YCrCb)
        cb = ycrcb[..., 1].astype(np.float32) / 255.0
        cr = ycrcb[..., 2].astype(np.float32) / 255.0
        out = np.sqrt((cb - cb.mean())**2 + (cr - cr.mean())**2)
        return (out - out.min()) / (out.max() - out.min() + 1e-12)

    def _generate_star_mask(self) -> np.ndarray:
        if sep is None:
            QMessageBox.warning(self, "Missing SEP", "Star mask requires the 'sep' package.")
            return np.zeros(self.image.shape[:2], dtype=np.float32)
        data = self._component_lightness().astype(np.float32)
        bkg = sep.Background(data); data_sub = data - bkg.back()
        thresh = float(self.blur_amount) if self.blur_amount > 0 else 3.0
        objs = sep.extract(data_sub, thresh=thresh, err=bkg.globalrms)
        h, w = data.shape; out = np.zeros((h, w), dtype=np.float32)
        if cv2 is None: return out
        MAX_RADIUS = 10
        for o in objs:
            x, y = int(o['x']), int(o['y'])
            r = int(max(o['a'], o['b']) * 1.5)
            if r <= MAX_RADIUS:
                cv2.circle(out, (x, y), max(1, r), 1.0, -1)
        return np.clip(out, 0, 1)

    def generate_mask(self) -> np.ndarray | None:
        try:
            base = self.canvas.create_mask()
        except RuntimeError as e:
            QMessageBox.warning(self, "Mask creation failed", str(e)); return None

        t = self.mask_type
        if t == "Binary":
            m = base
        elif t == "Range Selection":
            comp = self._component_lightness() if self.light_cb.isChecked() else self._component_lightness()
            L = self.lower_sl.value() / self.lower_sl.maximum()
            U = self.upper_sl.value() / self.upper_sl.maximum()
            fuzz = self.fuzz_sl.value() / self.fuzz_sl.maximum()
            smooth = float(self.smooth_sl.value())
            rs = self._range_selection_mask(comp, L, U, fuzz, smooth,
                                            self.screen_cb.isChecked(), self.invert_cb.isChecked())
            m = base * rs
        elif t == "Lightness":
            m = np.where(base > 0, self._component_lightness(), 0.0)
        elif t == "Chrominance":
            m = np.where(base > 0, self._generate_chrominance(), 0.0)
        elif t == "Star Mask":
            m = np.where(base > 0, self._generate_star_mask(), 0.0)
        elif t.startswith("Color:"):
            color = t.split(":", 1)[1].strip()
            m = np.where(base > 0, self._generate_color_mask(color), 0.0)
        else:
            m = base

        if self.blur_amount > 0 and cv2 is not None:
            k = max(1, int(self.blur_amount) * 2 + 1)
            m = cv2.GaussianBlur(m, (k, k), 0.0)
        return np.clip(m, 0.0, 1.0)

    def _update_live_preview(self, *_):
        m = self.generate_mask()
        if m is None: return
        if not self.live_preview.isVisible(): self.live_preview.show()
        self.live_preview.update_mask(m)

    def _preview_mask(self):
        m = self.generate_mask()
        if m is None: return
        MaskPreviewDialog(m, self).exec()

    def _accept_apply(self):
        m = self.generate_mask()
        if m is None:
            return

        # always store it on the dialog for callers
        self.mask = m

        # if this dialog was opened in "tool" mode, push it as a new doc
        if self.auto_push_on_ok:
            _push_numpy_as_new_document(self, m, default_name="Mask")

        self.accept()

    def closeEvent(self, ev):
        if self.live_preview and self.live_preview.isVisible():
            self.live_preview.close()
        super().closeEvent(ev)

    # --- NEW: generic zoom helpers -----------------------------------------
    def _zoom_canvas(self, factor: float):
        """
        Try several zoom APIs so we work with different MaskCanvas versions.
        """
        c = self.canvas
        try:
            if hasattr(c, "zoom_in") and factor > 1.0:
                c.zoom_in()
                return
            if hasattr(c, "zoom_out") and factor < 1.0:
                c.zoom_out()
                return
            if hasattr(c, "set_zoom"):
                z = getattr(c, "_zoom", 1.0)
                c.set_zoom(max(0.05, min(z * float(factor), 8.0)))
                return
            # If it's a QGraphicsView or similar, scale its view transform
            if isinstance(c, QGraphicsView):
                c.scale(float(factor), float(factor))
                return
        except Exception:
            pass  # fall through to friendly message
        QMessageBox.information(self, "Zoom", "Zoom is not supported by this canvas build.")

    def _fit_canvas(self):
        c = self.canvas
        try:
            if hasattr(c, "fit_to_view"):
                c.fit_to_view()
                return
            if isinstance(c, QGraphicsView):
                # Fit the full scene rect (keep aspect)
                r = c.sceneRect() if hasattr(c, "sceneRect") else None
                if r and r.isValid():
                    c.fitInView(r, Qt.AspectRatioMode.KeepAspectRatio)
                    return
        except Exception:
            pass
        QMessageBox.information(self, "Fit", "Fit-to-preview is not supported by this canvas build.")

    # --- NEW: Ctrl+Wheel zoom passthrough -----------------------------------
    def eventFilter(self, obj, ev):
        # Let the canvas keep its own interactions;
        # only intercept Ctrl+Wheel to trigger our zoom.
        if obj is self.canvas and ev.type() == QEvent.Type.Wheel:
            if isinstance(ev, QWheelEvent) and (ev.modifiers() & Qt.KeyboardModifier.ControlModifier):
                self._zoom_canvas(1.25 if ev.angleDelta().y() > 0 else 1/1.25)
                return True
        return super().eventFilter(obj, ev)


# ---------- Integration helper ----------

def create_mask_and_attach(parent, document) -> bool:
    if document is None or getattr(document, "image", None) is None:
        QMessageBox.information(parent, "No image", "Open an image first.")
        return False

    # NOW we let the dialog auto-push when user hits OK
    dlg = MaskCreationDialog(document.image, parent=parent, auto_push_on_ok=True)
    if dlg.exec() != QDialog.DialogCode.Accepted:
        return False

    mask = getattr(dlg, "mask", None)
    if mask is None:
        QMessageBox.information(parent, "No mask", "No mask was generated.")
        return False

    # since we already pushed a mask doc, just attach it quietly
    layer = MaskLayer(
        id=uuid.uuid4().hex,
        name="Mask",                     # keep it simple; matches preview default
        data=np.clip(mask.astype(np.float32, copy=False), 0.0, 1.0),
        invert=False,
        opacity=1.0,
        mode="affect",
        visible=True,
    )
    document.add_mask(layer, make_active=True)

    try:
        if hasattr(parent, "_log"):
            parent._log(f"Added mask '{layer.name}' and set active (and pushed as document).")
    except Exception:
        pass

    return True


