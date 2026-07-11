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
    QPainterPath, QWheelEvent, QPolygonF, QMouseEvent,QShortcut, QKeySequence
)
from PyQt6.QtWidgets import (
    QApplication,
    QInputDialog, QMessageBox, QFileDialog,
    QDialog, QDialogButtonBox,
    QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QComboBox, QSlider, QCheckBox, QButtonGroup, QGroupBox,
    QScrollArea, QSizePolicy,
    QGraphicsView, QGraphicsScene, QGraphicsItem,
    QGraphicsPixmapItem, QGraphicsPathItem, QGraphicsPolygonItem,
    QGraphicsEllipseItem, QGraphicsRectItem, QMdiSubWindow, QLabel, QWidget
)

from .masks_core import MaskLayer
from setiastro.saspro.widgets.themed_buttons import themed_toolbtn
from setiastro.saspro.imageops.stretch import stretch_color_image

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


def _display_stretch(img01: np.ndarray) -> np.ndarray:
    a = np.asarray(img01, dtype=np.float32)
    a = np.clip(a, 0.0, 1.0)
    if a.ndim == 3 and a.shape[2] == 3 and stretch_color_image is not None:
        try:
            return np.clip(
                stretch_color_image(a, 0.25, linked=False, normalize=False),
                0.0, 1.0
            ).astype(np.float32)
        except Exception:
            pass
    m = float(np.nanmedian(a))
    if not np.isfinite(m):
        return a.astype(np.float32, copy=False)
    target = 0.25
    eps = 1e-8
    scale = target / max(m, eps)
    out = np.clip(a * scale, 0.0, 1.0)
    out = np.sqrt(out)
    return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)


def _find_main_window(w):
    p = w
    from PyQt6.QtWidgets import QMainWindow
    while p is not None and not isinstance(p, QMainWindow):
        p = p.parent()
    return p


def _push_numpy_as_new_document(
    owner_widget, arr01: np.ndarray, default_name: str = "Mask"
) -> bool:
    mw = _find_main_window(owner_widget)
    if mw is None or not hasattr(mw, "docman"):
        QMessageBox.warning(
            owner_widget, "Cannot Create Document",
            "Main window / DocManager not found."
        )
        return False
    name, ok = QInputDialog.getText(
        owner_widget, "New Document Name", "Name:", text=default_name
    )
    if not ok:
        return False
    img = np.clip(arr01.astype(np.float32, copy=False), 0.0, 1.0)
    doc = mw.docman.open_array(img, title=name)
    if hasattr(mw, "_log"):
        mw._log(f"Created new document from mask: {doc.display_name()}")
    return True


def _downsample_for_preview(
    img: np.ndarray, max_dim: int = 1000
) -> tuple[np.ndarray, float]:
    a = np.asarray(img, dtype=np.float32)
    h, w = a.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return a, 1.0
    scale = float(max_dim) / float(longest)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    if cv2 is not None:
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        out = cv2.resize(a, (new_w, new_h), interpolation=interp)
        return out.astype(np.float32, copy=False), scale
    ys = np.linspace(0, h - 1, new_h).astype(np.int32)
    xs = np.linspace(0, w - 1, new_w).astype(np.int32)
    out = a[ys][:, xs] if a.ndim == 2 else a[ys][:, xs, :]
    return out.astype(np.float32, copy=False), scale

class _ZoomableImageView(QGraphicsView):
    """
    A QGraphicsView that displays a single pixmap with:
    - Ctrl+Wheel or plain wheel zoom
    - Middle-button or right-button drag pan
    - Keyboard +/- zoom, F fit
    - zoom_to_rect(scene_rect) for syncing with the canvas
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        self._pix_item = QGraphicsPixmapItem()
        self._pix_item.setTransformationMode(
            Qt.TransformationMode.SmoothTransformation
        )
        self._scene.addItem(self._pix_item)

        self._zoom = 1.0
        self._min_zoom = 0.02
        self._max_zoom = 32.0
        self._pan_active = False
        self._pan_start = None

    # ── pixmap API ────────────────────────────────────────────────────────────

    def set_pixmap(self, pm: QPixmap):
        self._pix_item.setPixmap(pm)
        self._scene.setSceneRect(self._pix_item.boundingRect())

    def pixmap(self) -> QPixmap:
        return self._pix_item.pixmap()

    # ── zoom API ──────────────────────────────────────────────────────────────

    def set_zoom(self, z: float):
        self._zoom = max(self._min_zoom, min(float(z), self._max_zoom))
        self.resetTransform()
        self.scale(self._zoom, self._zoom)
        self._update_zoom_label()

    def _update_zoom_label(self):
        try:
            lbl = getattr(self.parent(), "_zoom_lbl", None)
            if lbl is not None:
                lbl.setText(f"{int(round(self._zoom * 100))}%")
        except Exception:
            pass

    def zoom_in(self):
        self.set_zoom(self._zoom * 1.25)

    def zoom_out(self):
        self.set_zoom(self._zoom / 1.25)

    def fit_to_view(self):
        pm = self._pix_item.pixmap()
        if pm.isNull():
            return
        vw = max(1, self.viewport().width())
        vh = max(1, self.viewport().height())
        iw, ih = pm.width(), pm.height()
        if iw == 0 or ih == 0:
            return
        s = min(vw / iw, vh / ih)
        self.set_zoom(s)

    def sync_view_from(self, other: "QGraphicsView"):
        """
        Copy the zoom and center from another QGraphicsView so both
        canvas and live preview stay locked together.
        """
        try:
            t = other.transform()
            # Extract uniform scale from the transform (m11 = x scale)
            self.set_zoom(t.m11())
            center = other.mapToScene(other.viewport().rect().center())
            self.centerOn(center)
        except Exception:
            pass

    # ── events ────────────────────────────────────────────────────────────────

    def wheelEvent(self, ev):
        dy = ev.pixelDelta().y()
        if dy != 0:
            abs_dy = abs(dy)
            if abs_dy <= 3:
                base = 1.015
            elif abs_dy <= 10:
                base = 1.03
            else:
                base = 1.06
            factor = base if dy > 0 else 1.0 / base
        else:
            ad = ev.angleDelta().y()
            if ad == 0:
                ev.accept()
                return
            factor = 1.25 if ad > 0 else 0.8
        self.set_zoom(self._zoom * factor)
        ev.accept()

    def mousePressEvent(self, ev):
        if ev.button() in (
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.MiddleButton,
            Qt.MouseButton.RightButton
        ):
            self._pan_active = True
            self._pan_start = ev.position().toPoint()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            ev.accept()
            return
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if self._pan_active and self._pan_start is not None:
            delta = ev.position().toPoint() - self._pan_start
            self._pan_start = ev.position().toPoint()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x()
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y()
            )
            ev.accept()
            return
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        if ev.button() in (
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.MiddleButton,
            Qt.MouseButton.RightButton
        ):
            self._pan_active = False
            self._pan_start = None
            self.unsetCursor()
            ev.accept()
            return
        super().mouseReleaseEvent(ev)

    def keyPressEvent(self, ev):
        if ev.key() in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
            self.zoom_in(); ev.accept()
        elif ev.key() == Qt.Key.Key_Minus:
            self.zoom_out(); ev.accept()
        elif ev.key() == Qt.Key.Key_F:
            self.fit_to_view(); ev.accept()
        else:
            super().keyPressEvent(ev)

    def showEvent(self, ev):
        super().showEvent(ev)
        QTimer.singleShot(0, self.fit_to_view)
        QTimer.singleShot(50, self._update_zoom_label)

class LivePreviewDialog(QDialog):
    """
    Floating live mask preview — zoomable/pannable, stays in sync with
    the main canvas zoom when sync_with_canvas() is called.
    """

    def __init__(self, original_image01: np.ndarray, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Live Mask Preview"))
        self.setWindowFlag(Qt.WindowType.Tool, True)
        self.resize(480, 400)

        self._base_image01 = np.asarray(original_image01, dtype=np.float32)
        self._current_mask01: np.ndarray | None = None
        self.max_alpha = 150

        # ── view FIRST so toolbar buttons can connect to it ────────────────
        self._view = _ZoomableImageView(self)
        self._view.set_pixmap(_to_qpixmap01(self._base_image01))

        # ── toolbar ────────────────────────────────────────────────────────
        tb = QHBoxLayout()
        b_out = themed_toolbtn("zoom-out",      "Zoom Out  (scroll wheel)")
        b_in  = themed_toolbtn("zoom-in",       "Zoom In   (scroll wheel)")
        b_fit = themed_toolbtn("zoom-fit-best", "Fit to Window  (F)")

        b_out.clicked.connect(self._view.zoom_out)
        b_in.clicked.connect(self._view.zoom_in)
        b_fit.clicked.connect(self._view.fit_to_view)

        for b in (b_out, b_in, b_fit):
            tb.addWidget(b)
        tb.addStretch()

        self._zoom_lbl = QLabel("100%")
        tb.addWidget(self._zoom_lbl)

        # ── layout ─────────────────────────────────────────────────────────
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(2)
        lay.addLayout(tb)
        lay.addWidget(self._view, 1)

    # ── public API ────────────────────────────────────────────────────────────

    def update_mask(self, mask01: np.ndarray):
        self._current_mask01 = mask01
        self._render()

    def set_base_image(self, image01: np.ndarray):
        self._base_image01 = np.asarray(image01, dtype=np.float32)
        if self._current_mask01 is not None:
            self._render()
        else:
            self._view.set_pixmap(_to_qpixmap01(self._base_image01))

    def sync_with_canvas(self, canvas: "MaskCanvas"):
        """Lock zoom/pan to match the drawing canvas."""
        self._view.sync_view_from(canvas)

    # ── internal ──────────────────────────────────────────────────────────────

    def _render(self):
        if self._current_mask01 is None:
            return
        h, w = self._current_mask01.shape[:2]
        alpha = (
            np.clip(self._current_mask01, 0, 1) * self.max_alpha
        ).astype(np.uint8)

        # Build base RGB from stored image
        base = np.clip(self._base_image01, 0, 1)
        if base.ndim == 2:
            base_rgb = np.stack([base, base, base], axis=2)
        else:
            base_rgb = base[..., :3]
        base8 = (base_rgb * 255).astype(np.uint8)

        # Resize base to match mask if they differ (preview vs full)
        if base8.shape[:2] != (h, w):
            if cv2 is not None:
                base8 = cv2.resize(base8, (w, h), interpolation=cv2.INTER_LINEAR)
            else:
                base8 = base8  # best effort

        # Red overlay composited onto base
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[..., :3] = base8
        rgba[..., 3] = 255

        # Screen-blend red channel where mask > 0
        mask_f = np.clip(self._current_mask01, 0, 1)[..., None]
        red_overlay = np.array([255, 50, 50], dtype=np.float32)
        blended = base8.astype(np.float32) * (1 - mask_f * 0.55) + red_overlay * (mask_f * 0.55)
        rgba[..., :3] = np.clip(blended, 0, 255).astype(np.uint8)

        qimg = QImage(
            rgba.data, w, h, 4 * w, QImage.Format.Format_RGBA8888
        )
        pm = QPixmap.fromImage(qimg)

        old_zoom = self._view._zoom
        old_center = self._view.mapToScene(
            self._view.viewport().rect().center()
        )
        self._view.set_pixmap(pm)
        # restore view position
        self._view.set_zoom(old_zoom)
        self._view.centerOn(old_center)

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

        self._base_image01 = np.asarray(image01, dtype=np.float32)
        self._display_stretch_enabled = False

        # scene + background image
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        self.bg_item = QGraphicsPixmapItem(_to_qpixmap01(self._base_image01))
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

    def set_display_stretch_enabled(self, enabled: bool):
        enabled = bool(enabled)
        if enabled == self._display_stretch_enabled:
            return
        self._display_stretch_enabled = enabled
        self._refresh_background_pixmap(keep_view=True)

    def display_stretch_enabled(self) -> bool:
        return bool(self._display_stretch_enabled)

    def current_display_image01(self) -> np.ndarray:
        """Returns the image currently used for *display* (not for mask math)."""
        if self._display_stretch_enabled:
            return _display_stretch(self._base_image01)
        return self._base_image01

    def _refresh_background_pixmap(self, keep_view: bool = True):
        # Preserve current view transform/center so toggling doesn't “jump”
        old_transform = self.transform()
        old_center = self.mapToScene(self.viewport().rect().center())

        disp = self.current_display_image01()
        self.bg_item.setPixmap(_to_qpixmap01(disp))

        # Ensure scene rect still matches image pixels
        self.setSceneRect(self.bg_item.boundingRect())

        if keep_view:
            self.setTransform(old_transform)
            self.centerOn(old_center)


    def set_mode(self, mode: str):
        assert mode in ('polygon', 'ellipse', 'select')
        self.mode = mode

    def clear_shapes(self):
        for it in list(self.shapes):
            try:
                self.scene.removeItem(it)
            except Exception:
                pass
        self.shapes.clear()

    def undo_last_shape(self) -> bool:
        if not self.shapes:
            return False
        it = self.shapes.pop()
        try:
            self.scene.removeItem(it)
        except Exception:
            pass
        return True


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

# ---------- Preview (push-as-doc) ----------

class MaskPreviewDialog(QDialog):
    """Full-resolution mask preview with zoom/pan + push-as-document."""

    def __init__(self, mask01: np.ndarray, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Mask Preview"))
        self.mask = np.clip(mask01, 0, 1).astype(np.float32)

        self._view = _ZoomableImageView(self)
        self._view.set_pixmap(self._make_pixmap(self.mask))

        # toolbar
        tb = QHBoxLayout()
        b_out  = themed_toolbtn("zoom-out",      "Zoom Out")
        b_in   = themed_toolbtn("zoom-in",       "Zoom In")
        b_fit  = themed_toolbtn("zoom-fit-best", "Fit to Window")
        b_push = QPushButton(self.tr("Push as New Document…"))

        b_out.clicked.connect(self._view.zoom_out)
        b_in.clicked.connect(self._view.zoom_in)
        b_fit.clicked.connect(self._view.fit_to_view)
        b_push.clicked.connect(self.push_as_new_document)

        for w in (b_out, b_in, b_fit, b_push):
            tb.addWidget(w)
        tb.addStretch()

        self._zoom_lbl = QLabel("")
        tb.addWidget(self._zoom_lbl)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(2)
        lay.addLayout(tb)
        lay.addWidget(self._view, 1)

        self.setMinimumSize(640, 480)

    def _make_pixmap(self, mask01: np.ndarray) -> QPixmap:
        m8 = (np.clip(mask01, 0, 1) * 255).astype(np.uint8)
        h, w = m8.shape
        qimg = QImage(m8.data, w, h, w, QImage.Format.Format_Grayscale8)
        return QPixmap.fromImage(qimg)

    def push_as_new_document(self):
        if self.mask is None:
            QMessageBox.warning(self, "No Mask", "No mask to push.")
            return
        host = self.parent()
        while host is not None and not hasattr(host, "docman"):
            host = host.parent()
        if host is None or not hasattr(host, "docman"):
            QMessageBox.warning(self, "No DocManager",
                                "Could not find the document manager.")
            return
        name, ok = QInputDialog.getText(
            self, "New Document Name", "Name:", text="Mask"
        )
        if not ok:
            return
        img = self.mask.astype(np.float32, copy=False)
        meta = {
            "bit_depth": "32-bit floating point",
            "is_mono": True,
            "original_format": "fits",
        }
        new_doc = host.docman.create_document(img, metadata=meta, name=(name or "Mask"))
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
    def __init__(self, image01: np.ndarray, parent=None,
                 auto_push_on_ok: bool = True, wcs=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Mask Creation"))
        # Optional WCS enables the Gaia Local DB star-mask augment. Silently
        # unavailable (checkbox greyed) if None or not celestial.
        self.wcs = wcs if (wcs is not None and getattr(wcs, "has_celestial", False)) else None
        self.setWindowFlag(Qt.WindowType.Window, True)
        import platform
        if platform.system() == "Darwin":
            self.setWindowFlag(Qt.WindowType.Tool, True)  
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setModal(False)
        try:
            self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        except Exception:
            pass  # older PyQt6 versions        
        self.image = np.asarray(image01, dtype=np.float32).copy()
        self.preview_image, self._preview_scale_factor = _downsample_for_preview(self.image, max_dim=1000)

        self.mask: np.ndarray | None = None
        self.live_preview = LivePreviewDialog(self.preview_image, parent=self)
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(120)   # 80-150 ms usually feels good
        self._preview_timer.timeout.connect(self._update_live_preview)
        self._cached_base_mask_full = None
        self._cached_base_mask_full_dirty = True

        self._cached_base_mask_preview = None
        self._cached_base_mask_preview_dirty = True       
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

        # Display stretch toggle (display-only; never modifies image data)
        self.btn_disp_stretch = QPushButton(self.tr("Toggle Display Stretch"))
        self.btn_disp_stretch.setCheckable(True)
        self.btn_disp_stretch.setToolTip(
            "Display-only stretch for easier masking on linear images.\n"
            "This does NOT change the image data or the generated mask."
        )
        self.btn_disp_stretch.toggled.connect(self._toggle_display_stretch)
        self.btn_disp_stretch.setChecked(False)
        self.btn_disp_stretch.setText("Enable Display Stretch")
        zoom_bar.addWidget(self.btn_disp_stretch)

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
            s.valueChanged.connect(self._schedule_live_preview)
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
            self._schedule_live_preview()
        self.smooth_sl.valueChanged.connect(_upd_smooth)
        self.link_cb   = QCheckBox("Link limits"); g.addWidget(self.link_cb, 0, 3, 2, 1)
        self.screen_cb = QCheckBox("Screening");   g.addWidget(self.screen_cb, 4, 0, 1, 4)
        self.light_cb  = QCheckBox("Lightness");   g.addWidget(self.light_cb, 5, 0, 1, 4)
        self.invert_cb = QCheckBox("Invert");      g.addWidget(self.invert_cb, 6, 0, 1, 4)
        self.lower_sl.valueChanged.connect(self._on_linked)
        self.link_cb.toggled.connect(self._on_link_switch)
        layout.addWidget(self.range_box); self.range_box.hide()
        self._build_star_thresh_controls(layout)
        self.type_dd.currentTextChanged.connect(self._on_type_changed)


        # Preview & Clear
        # Preview & Clear
        rowb = QHBoxLayout()
        b_preview = QPushButton("Preview Mask"); b_preview.clicked.connect(self._preview_mask)

        b_undo = QPushButton("Undo Shape"); b_undo.clicked.connect(self._undo_shape)

        b_clear = QPushButton("Clear Shapes");   b_clear.clicked.connect(self._clear_shapes)

        rowb.addWidget(b_preview)
        rowb.addWidget(b_undo)
        rowb.addWidget(b_clear)
        layout.addLayout(rowb)

        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #cfcfcf;")
        layout.addWidget(self.status_label)

        # OK / Cancel
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, parent=self)
        btns.accepted.connect(self._accept_apply); btns.rejected.connect(self.reject)
        layout.addWidget(btns)
        
        self._undo_shortcut = QShortcut(QKeySequence("Ctrl+Z"), self)
        self._undo_shortcut.activated.connect(self._undo_shape)


        self.canvas.installEventFilter(self)

        self.resize(980, 640)

    def _on_mask_type_changed(self, mask_type: str):
        is_star = (mask_type == "Star Mask")  # match your exact string
        self._star_thresh_row.setVisible(is_star)
        self._btn_trial_detect.setVisible(is_star)
        if hasattr(self, "_gaia_augment_row"):
            self._gaia_augment_row.setVisible(is_star)

    def _build_star_thresh_controls(self, parent_layout):
        """Call this during __init__ to build the hidden threshold row."""
        from PyQt6.QtWidgets import QHBoxLayout, QLabel, QDoubleSpinBox, QPushButton

        # Threshold row
        self._star_thresh_row = QWidget()
        row_layout = QHBoxLayout(self._star_thresh_row)
        row_layout.setContentsMargins(0, 0, 0, 0)

        lbl = QLabel("Detection Threshold (σ):")
        lbl.setStyleSheet("color:#eaeaea;font-size:11px;")

        self._spin_star_thresh = QDoubleSpinBox()
        self._spin_star_thresh.setRange(0.5, 50.0)
        self._spin_star_thresh.setSingleStep(0.5)
        self._spin_star_thresh.setValue(3.0)
        self._spin_star_thresh.setFixedWidth(70)
        self._spin_star_thresh.setToolTip(
            "Lower = detect more/fainter stars (may over-detect).\n"
            "Higher = only bright stars detected.\n"
            "Use Trial Detect to test before generating the mask."
        )
        self._spin_star_thresh.valueChanged.connect(
            lambda v: setattr(self, "_star_thresh", float(v))
        )

        row_layout.addWidget(lbl)
        row_layout.addWidget(self._spin_star_thresh)
        row_layout.addStretch()
        parent_layout.addWidget(self._star_thresh_row)

        # Gaia augment: SEP finds what it can, Gaia fills in the catalogued
        # stars SEP missed (saturated cores, dim ones below threshold).
        # Requires plate-solved WCS and the astrometric library installed.
        self._gaia_augment_row = QWidget()
        gar_layout = QHBoxLayout(self._gaia_augment_row)
        gar_layout.setContentsMargins(0, 0, 0, 0)

        self._chk_gaia_augment = QCheckBox("Augment with Gaia Local DB")
        self._chk_gaia_augment.setStyleSheet("color:#eaeaea;font-size:11px;")
        try:
            from PyQt6.QtCore import QSettings
            self._chk_gaia_augment.setChecked(
                QSettings("SetiAstro", "SASpro").value(
                    "mask/gaia_augment", False, type=bool))
        except Exception:
            pass

        def _persist_augment(on):
            try:
                from PyQt6.QtCore import QSettings
                QSettings("SetiAstro", "SASpro").setValue(
                    "mask/gaia_augment", bool(on))
            except Exception:
                pass
        self._chk_gaia_augment.toggled.connect(_persist_augment)

        # Availability gate — grey out with an explanatory tooltip when it
        # can't work, rather than failing silently or throwing at generate.
        gaia_ok, gaia_reason = self._check_gaia_augment_available()
        self._chk_gaia_augment.setEnabled(gaia_ok)
        if not gaia_ok:
            self._chk_gaia_augment.setChecked(False)
            self._chk_gaia_augment.setToolTip(gaia_reason)
        else:
            self._chk_gaia_augment.setToolTip(
                "Add Gaia Local DB catalogued stars to the mask, sized to match\n"
                "the SEP-detected stars. Handles saturated cores SEP misses.\n"
                "Requires a plate-solved image and an installed astrometric band."
            )

        # Max-G limit so wide-field frames don't spend a minute masking
        # 10,000 faint stars nobody cares about.
        lbl_g = QLabel("Max G:")
        lbl_g.setStyleSheet("color:#eaeaea;font-size:11px;")
        self._spin_gaia_maxg = QDoubleSpinBox()
        self._spin_gaia_maxg.setRange(6.0, 20.0)
        self._spin_gaia_maxg.setSingleStep(0.5)
        self._spin_gaia_maxg.setDecimals(1)
        self._spin_gaia_maxg.setValue(17.0)
        self._spin_gaia_maxg.setFixedWidth(60)
        self._spin_gaia_maxg.setToolTip(
            "Skip Gaia stars fainter than this G magnitude.\n"
            "17 is a good default; raise for very deep/narrowband data."
        )

        gar_layout.addWidget(self._chk_gaia_augment)
        gar_layout.addSpacing(12)
        gar_layout.addWidget(lbl_g)
        gar_layout.addWidget(self._spin_gaia_maxg)
        gar_layout.addStretch()
        parent_layout.addWidget(self._gaia_augment_row)

        # Trial Detect button
        self._btn_trial_detect = QPushButton("🔍 Trial Detect")
        self._btn_trial_detect.setFixedHeight(26)
        self._btn_trial_detect.setStyleSheet(
            "QPushButton{background:#0f3460;color:#eaeaea;border-radius:4px;font-size:11px;}"
            "QPushButton:hover{background:#e94560;color:#fff;}"
        )
        self._btn_trial_detect.setToolTip(
            "Run star detection with the current threshold and report how many\n"
            "stars were found — without generating the full mask."
        )
        self._btn_trial_detect.clicked.connect(self._run_trial_detect)
        parent_layout.addWidget(self._btn_trial_detect)

        # Result label
        self._lbl_trial_result = QLabel("")
        self._lbl_trial_result.setStyleSheet("color:#888;font-size:10px;")
        self._lbl_trial_result.setWordWrap(True)
        parent_layout.addWidget(self._lbl_trial_result)

        # Hide by default
        self._star_thresh_row.setVisible(False)
        self._btn_trial_detect.setVisible(False)
        self._lbl_trial_result.setVisible(False)
        if hasattr(self, "_gaia_augment_row"):
            self._gaia_augment_row.setVisible(False)

    def _run_trial_detect(self):
        import sep
        import numpy as np
        from PyQt6.QtWidgets import QApplication
        from PyQt6.QtCore import Qt

        img = getattr(self, "image", None)
        if img is None:
            self._lbl_trial_result.setText("⚠  No image loaded.")
            self._lbl_trial_result.setVisible(True)
            return

        thresh = float(getattr(self, "_star_thresh", 3.0))

        self._btn_trial_detect.setEnabled(False)
        self._lbl_trial_result.setText("Detecting…")
        self._lbl_trial_result.setVisible(True)
        QApplication.processEvents()

        try:
            if img.ndim == 3:
                data = img.mean(axis=2).astype(np.float32)
            else:
                data = img.astype(np.float32)

            data = np.ascontiguousarray(data)
            bkg = sep.Background(data)
            data_sub = data - bkg

            sep.set_extract_pixstack(5000000)
            objs = sep.extract(data_sub, thresh=thresh, err=bkg.globalrms)
            n = len(objs)

            if n == 0:
                msg = f"⚠  No stars detected at σ={thresh:.1f} — try lowering the threshold."
                color = "#ffc107"
            elif n > 100000:
                msg = (f"⚠  {n:,} objects detected at σ={thresh:.1f} — "
                       f"threshold is probably too low. Try raising it.")
                color = "#ff9800"
            elif n > 20000:
                msg = (f"⚠  {n:,} stars detected at σ={thresh:.1f} — "
                       f"quite high, consider raising the threshold.")
                color = "#ffc107"
            else:
                msg = f"✓  {n:,} stars detected at σ={thresh:.1f}"
                color = "#4caf50"

            self._lbl_trial_result.setText(msg)
            self._lbl_trial_result.setStyleSheet(
                f"color:{color};font-size:10px;"
            )

        except Exception as e:
            err = str(e)
            if "pixel buffer full" in err or "pixstack" in err.lower():
                msg = (f"⚠  Too many active pixels at σ={thresh:.1f}. "
                       f"Raise the threshold significantly.")
                color = "#e94560"
            else:
                msg = f"⚠  Detection error: {err}"
                color = "#e94560"
            self._lbl_trial_result.setText(msg)
            self._lbl_trial_result.setStyleSheet(f"color:{color};font-size:10px;")

        finally:
            self._btn_trial_detect.setEnabled(True)

    def _set_status(self, text: str):
        self.status_label.setText(text)
        QApplication.processEvents()

    def _invalidate_base_mask(self):
        self._cached_base_mask_full_dirty = True
        self._cached_base_mask_preview_dirty = True

    def _get_base_mask_full(self) -> np.ndarray | None:
        if self._cached_base_mask_full is None or self._cached_base_mask_full_dirty:
            try:
                h, w = self.image.shape[:2]
                self._cached_base_mask_full = self._create_mask_from_shapes_at_size(h, w)
                self._cached_base_mask_full_dirty = False
            except RuntimeError as e:
                QMessageBox.warning(self, "Mask creation failed", str(e))
                return None
        return self._cached_base_mask_full


    def _get_base_mask_preview(self) -> np.ndarray | None:
        if self._cached_base_mask_preview is None or self._cached_base_mask_preview_dirty:
            try:
                h, w = self.preview_image.shape[:2]
                self._cached_base_mask_preview = self._create_mask_from_shapes_at_size(h, w)
                self._cached_base_mask_preview_dirty = False
            except RuntimeError as e:
                QMessageBox.warning(self, "Mask creation failed", str(e))
                return None
        return self._cached_base_mask_preview

    def _create_mask_from_shapes_at_size(self, out_h: int, out_w: int) -> np.ndarray:
        if cv2 is None:
            raise RuntimeError("OpenCV (cv2) is required for mask creation.")

        full_h, full_w = self.image.shape[:2]
        sx = float(out_w) / float(full_w)
        sy = float(out_h) / float(full_h)

        mask = np.zeros((out_h, out_w), dtype=np.uint8)

        for s in self.canvas.shapes:
            if isinstance(s, QGraphicsPolygonItem):
                pts = s.polygon()
                arr = np.array(
                    [[int(round(p.x() * sx)), int(round(p.y() * sy))] for p in pts],
                    dtype=np.int32
                )
                if len(arr) >= 3:
                    cv2.fillPoly(mask, [arr], 1)

            elif isinstance(s, InteractiveEllipseItem):
                r = s.rect()
                scenep = s.mapToScene(r.center())

                cx = int(round(scenep.x() * sx))
                cy = int(round(scenep.y() * sy))
                rx = max(1, int(round((r.width() * 0.5) * sx)))
                ry = max(1, int(round((r.height() * 0.5) * sy)))
                angle = float(s.rotation())

                cv2.ellipse(mask, (cx, cy), (rx, ry), angle, 0, 360, 1, -1)

        return (mask > 0).astype(np.float32)

    def _schedule_live_preview(self, *_):
        if self.range_box.isVisible():
            self._preview_timer.start()

    def _undo_shape(self):
        if not self.canvas.undo_last_shape():
            return
        self._invalidate_base_mask()
        if hasattr(self, "range_box") and self.range_box.isVisible():
            self._schedule_live_preview()


    # ---- callbacks
    def _set_mode(self, mode: str):
        self.canvas.set_mode(mode)
        if mode == 'select':
            self.canvas.select_entire_image()
            self._invalidate_base_mask()
            self._schedule_live_preview()

    def _clear_shapes(self):
        self.canvas.clear_shapes()
        self._invalidate_base_mask()
        if hasattr(self, "range_box") and self.range_box.isVisible():
            self._schedule_live_preview()

    def _on_type_changed(self, txt: str):
        show_range = (txt == "Range Selection")
        self.range_box.setVisible(show_range)
        if show_range:
            if not self.live_preview.isVisible():
                self.live_preview.show()
            self._schedule_live_preview()
        else:
            if self.live_preview.isVisible():
                self.live_preview.close()

        is_star = (txt == "Star Mask")
        self._star_thresh_row.setVisible(is_star)
        self._btn_trial_detect.setVisible(is_star)
        self._lbl_trial_result.setVisible(is_star and self._lbl_trial_result.text() != "")
        if hasattr(self, "_gaia_augment_row"):
            self._gaia_augment_row.setVisible(is_star)

    def _on_link_switch(self, checked: bool):
        if checked:
            self.upper_sl.setValue(self.lower_sl.value())

    def _on_linked(self, v: int):
        if self.link_cb.isChecked():
            self.upper_sl.setValue(v)

    def _toggle_display_stretch(self, enabled: bool):
        try:
            self.canvas.set_display_stretch_enabled(bool(enabled))

            # keep button label in sync
            self.btn_disp_stretch.setText(
                self.tr("Disable Display Stretch") if enabled else self.tr("Enable Display Stretch")
            )

            # Keep the live preview background in sync (Range Selection uses it)
            if hasattr(self, "live_preview") and self.live_preview is not None:
                self.live_preview.set_base_image(self.canvas.current_display_image01())
                if self.live_preview.isVisible():
                    self._update_live_preview()
        except Exception:
            pass


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

    def _check_gaia_augment_available(self) -> tuple[bool, str]:
        """Availability + reason string for the Gaia augment checkbox."""
        if getattr(self, "wcs", None) is None:
            return (False,
                    "Gaia augment requires a plate-solved image.\n"
                    "Run Plate Solve on the image first.")
        try:
            from setiastro.saspro.gaia_database import get_astro_library
            if not get_astro_library().installed_bands():
                return (False,
                        "No Gaia Local DB astrometric bands installed.\n"
                        "Open Settings → Gaia Local DB Library → Astrometry to install.")
        except Exception as e:
            return (False, f"Gaia astrometric library unavailable:\n{e}")
        return (True, "")

    def _augment_star_mask_with_gaia(self, out: np.ndarray, sep_objs) -> np.ndarray:
        """
        Add Gaia Local DB catalogued stars to a SEP-derived star mask. Radius
        per Gaia star is derived from a fit of SEP radius vs. G-mag on
        stars that appear in both — so the drawn circles visually match
        what SEP produced.
        """
        if not getattr(self, "_chk_gaia_augment", None) \
                or not self._chk_gaia_augment.isChecked():
            return out
        if getattr(self, "wcs", None) is None or cv2 is None:
            return out

        import numpy as np
        try:
            from setiastro.saspro.gaia_database import get_astro_library
            astro_lib = get_astro_library()
            if not astro_lib.installed_bands():
                return out
        except Exception as e:
            print(f"[MaskGaia] library open failed: {e}")
            return out

        h, w = out.shape[:2]
        wcs = self.wcs

        # Frame extent in sky coords via the four corners.
        try:
            corners_px = np.array([[0, 0], [w - 1, 0],
                                   [w - 1, h - 1], [0, h - 1]], dtype=np.float64)
            sky = wcs.pixel_to_world(corners_px[:, 0], corners_px[:, 1])
            ra_arr  = np.asarray(sky.ra.deg,  dtype=np.float64)
            dec_arr = np.asarray(sky.dec.deg, dtype=np.float64)
        except Exception as e:
            print(f"[MaskGaia] WCS corner mapping failed: {e}")
            return out

        dec_lo, dec_hi = float(dec_arr.min()), float(dec_arr.max())
        # Wraparound-aware RA range: if the span > 180° we've straddled 0/360.
        ra_min, ra_max = float(ra_arr.min()), float(ra_arr.max())
        if ra_max - ra_min > 180.0:
            ra_lo, ra_hi = ra_max, ra_min       # find_stars_in_box handles wrap
        else:
            ra_lo, ra_hi = ra_min, ra_max

        try:
            max_g = float(self._spin_gaia_maxg.value())
        except Exception:
            max_g = 17.0

        try:
            stars = astro_lib.find_stars_in_box(
                ra_lo, ra_hi, dec_lo, dec_hi, max_mag=max_g)
        except Exception as e:
            print(f"[MaskGaia] catalog query failed: {e}")
            return out
        if not stars:
            print("[MaskGaia] no catalog stars in frame")
            return out

        # Sky → pixel for all Gaia stars in one call.
        try:
            g_ra  = np.array([s.ra  for s in stars], dtype=np.float64)
            g_dec = np.array([s.dec for s in stars], dtype=np.float64)
            xs, ys = wcs.world_to_pixel_values(g_ra, g_dec)
            xs = np.asarray(xs, dtype=np.float64)
            ys = np.asarray(ys, dtype=np.float64)
        except Exception as e:
            print(f"[MaskGaia] world_to_pixel failed: {e}")
            return out

        in_frame = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        xs, ys = xs[in_frame], ys[in_frame]
        gmags = np.array([s.gmag for s in stars], dtype=np.float64)[in_frame]

        # Build the radius vs. G-mag mapping from SEP results if we have
        # enough coverage; otherwise fall back to a mag-based formula.
        radius_fn = self._fit_gaia_radius_model(sep_objs, gmags)

        MAX_RADIUS = 10   # matches _generate_star_mask's cap
        drawn = 0
        skipped_duplicate = 0
        # 3-px duplicate skip: if SEP already put a mask disc where this
        # Gaia star is, skip (avoids double-drawing bright stars that both
        # found — Gaia's job is filling gaps, not painting over SEP).
        for x, y, g in zip(xs, ys, gmags):
            ix, iy = int(round(x)), int(round(y))
            if 0 <= iy < h and 0 <= ix < w and out[iy, ix] > 0:
                skipped_duplicate += 1
                continue
            r = int(max(1, min(MAX_RADIUS, round(radius_fn(g)))))
            cv2.circle(out, (ix, iy), r, 1.0, -1)
            drawn += 1

        print(f"[MaskGaia] added {drawn} Gaia stars "
              f"({skipped_duplicate} already SEP-masked, "
              f"{len(stars) - int(in_frame.sum())} out of frame, "
              f"G<={max_g})")
        return out

    def _fit_gaia_radius_model(self, sep_objs, gmags_reference):
        """
        Returns a callable f(gmag) -> pixel_radius that matches how SEP
        drew radii for stars in this same image. Fits from SEP objects
        when there are enough to be meaningful; falls back to a static
        model otherwise so the augment still works on very sparse frames.
        """
        import numpy as np

        # Fallback: cheap monotone map — bright stars bigger, faint ~2px.
        def fallback(g):
            g = float(g)
            if g <= 6.0:  return 10.0
            if g >= 15.0: return 2.0
            return 10.0 - (g - 6.0) * (8.0 / 9.0)

        if sep_objs is None or len(sep_objs) < 15:
            return fallback

        try:
            sep_r = np.array([float(max(o["a"], o["b"])) * 1.5
                              for o in sep_objs], dtype=np.float64)
            # Rank both by size and use as a rank-to-mag pseudo-fit. We
            # don't know the SEP objects' magnitudes without re-measuring,
            # so anchor by percentile: brightest SEP radius ~= brightest
            # Gaia G in frame, faintest to faintest.
            r_bright = float(np.percentile(sep_r, 95))
            r_faint  = float(np.percentile(sep_r,  5))
            g_bright = float(np.percentile(gmags_reference,  5))
            g_faint  = float(np.percentile(gmags_reference, 95))
            if g_faint - g_bright < 0.5 or r_bright <= r_faint:
                return fallback

            slope = (r_faint - r_bright) / (g_faint - g_bright)
            intercept = r_bright - slope * g_bright

            def fitted(g):
                r = slope * float(g) + intercept
                return max(1.0, min(15.0, r))
            return fitted
        except Exception:
            return fallback

    def _generate_star_mask(self):
        import sep
        import numpy as np

        img = self.image
        if img is None:
            return None

        # Use mono for detection
        if img.ndim == 3:
            data = img.mean(axis=2).astype(np.float32)
        else:
            data = img.astype(np.float32)

        data = np.ascontiguousarray(data)
        thresh = float(getattr(self, "_star_thresh", 3.0))

        try:
            bkg = sep.Background(data)
            data_sub = data - bkg
            # Increase pixstack limit before extraction — default 300000 is too low
            # for wide-field images with many/large stars
            sep.set_extract_pixstack(5000000)
            objs = sep.extract(data_sub, thresh=thresh, err=bkg.globalrms)
        except Exception as e:
            raise RuntimeError(str(e))
        h, w = data.shape; out = np.zeros((h, w), dtype=np.float32)
        if cv2 is None: return out
        MAX_RADIUS = 10
        for o in objs:
            x, y = int(o['x']), int(o['y'])
            r = int(max(o['a'], o['b']) * 1.5)
            if r <= MAX_RADIUS:
                cv2.circle(out, (x, y), max(1, r), 1.0, -1)

        out = self._augment_star_mask_with_gaia(out, objs)
        return np.clip(out, 0, 1)

    def generate_preview_mask(self) -> np.ndarray | None:
        base = self._get_base_mask_preview()
        if base is None:
            return None

        t = self.mask_type
        img = self.preview_image

        if t == "Binary":
            m = base

        elif t == "Range Selection":
            comp = (
                (img[..., 0]*0.2989 + img[..., 1]*0.5870 + img[..., 2]*0.1140).astype(np.float32)
                if img.ndim == 3 else img.astype(np.float32)
            )
            L = self.lower_sl.value() / self.lower_sl.maximum()
            U = self.upper_sl.value() / self.upper_sl.maximum()
            fuzz = self.fuzz_sl.value() / self.fuzz_sl.maximum()
            preview_smooth = max(0.5, float(self.smooth_sl.value()) * self._preview_scale_factor)

            rs = self._range_selection_mask(
                comp, L, U, fuzz, preview_smooth,
                self.screen_cb.isChecked(), self.invert_cb.isChecked()
            )
            m = base * rs

        elif t == "Lightness":
            comp = (
                (img[..., 0]*0.2989 + img[..., 1]*0.5870 + img[..., 2]*0.1140).astype(np.float32)
                if img.ndim == 3 else img.astype(np.float32)
            )
            m = np.where(base > 0, comp, 0.0)

        else:
            # For now, preview path can still use full logic later if needed,
            # but Range Selection is the main win.
            return self.generate_mask()

        if self.blur_amount > 0 and cv2 is not None:
            k = max(1, int(self.blur_amount) * 2 + 1)
            m = cv2.GaussianBlur(m, (k, k), 0.0)

        return np.clip(m, 0.0, 1.0)

    def generate_mask(self) -> np.ndarray | None:
        try:
            base = self._get_base_mask_full()
            if base is None:
                return None
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
        m = self.generate_preview_mask()
        if m is None:
            return
        if not self.live_preview.isVisible():
            self.live_preview.show()
        self.live_preview.update_mask(m)

    def _preview_mask(self):
        self._set_status("Building full-resolution mask preview...")
        try:
            m = self.generate_mask()
            if m is None:
                return
            MaskPreviewDialog(m, self).exec()
        finally:
            self._set_status("")

    def _accept_apply(self):
        self._set_status("Creating full-resolution mask document...")
        try:
            m = self.generate_mask()
            if m is None:
                return

            # always store it on the dialog for callers
            self.mask = m

            # if this dialog was opened in "tool" mode, push it as a new doc
            if self.auto_push_on_ok:
                self._set_status("Calculating full mask and creating document...")
                _push_numpy_as_new_document(self, m, default_name="Mask")

            self.accept()
        finally:
            self._set_status("")

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
        # only intercept wheel to trigger our zoom.
        if obj is self.canvas and ev.type() == QEvent.Type.Wheel:
            if isinstance(ev, QWheelEvent):
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
                        ev.accept()
                        return True

                    ctrl_down = bool(ev.modifiers() & Qt.KeyboardModifier.ControlModifier)
                    step = 1.25 if ctrl_down else 1.15
                    factor = step if dy > 0 else 1.0 / step

                self._zoom_canvas(factor)
                ev.accept()
                return True

        return super().eventFilter(obj, ev)

# ---------- Integration helper ----------

def create_mask_and_attach(parent, document) -> bool:
    if document is None or getattr(document, "image", None) is None:
        QMessageBox.information(parent, "No image", "Open an image first.")
        return False

    # Pull WCS from the doc for the Gaia augment. Match WIMI's extraction
    # order: wcs_header wins (it's the WCS-only header the solver writes),
    # then original_header. Use naxis=2 + .celestial to strip stub 3rd
    # axes on RGB images — the plain WCS(hdr) path produces naxis=3 for
    # colour cubes and fails has_celestial even on a valid solve.
    wcs = None
    meta = getattr(document, "metadata", {}) or {}
    try:
        from astropy.wcs import WCS
        hdr = meta.get("wcs_header") or meta.get("original_header")
        if hdr is not None:
            w = WCS(hdr, naxis=2, relax=True)
            if w.naxis > 2:
                w = w.celestial
            if w.naxis == 2 and getattr(w, "has_celestial", False):
                wcs = w
    except Exception as e:
        print(f"[MaskGaia] WCS extraction failed: {e}")
        wcs = None
    print(f"[MaskGaia] WCS available for augment: "
          f"{wcs is not None} (keys present: "
          f"wcs_header={'wcs_header' in meta}, "
          f"original_header={'original_header' in meta})")

    # NOW we let the dialog auto-push when user hits OK
    dlg = MaskCreationDialog(document.image, parent=parent,
                             auto_push_on_ok=True, wcs=wcs)
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


