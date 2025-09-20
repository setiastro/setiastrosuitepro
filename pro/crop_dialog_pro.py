# pro/crop_dialog_pro.py
from __future__ import annotations

import math, numpy as np, cv2
from typing import Optional

from PyQt6.QtCore import Qt, QEvent, QPointF, QRectF, pyqtSignal, QPoint
from PyQt6.QtGui import QPixmap, QImage, QPen, QBrush, QColor
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox, QToolButton,
    QMessageBox, QGraphicsScene, QGraphicsView, QGraphicsRectItem, QGraphicsEllipseItem,
    QGraphicsItem, QGraphicsPixmapItem
)

# -------- util: Siril-style preview stretch (non-destructive) ----------
def siril_style_autostretch(image: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    def stretch_channel(c):
        med = np.median(c); mad = np.median(np.abs(c - med))
        mad_std = mad * 1.4826
        mn, mx = float(c.min()), float(c.max())
        bp = max(mn, med - sigma * mad_std)
        wp = min(mx, med + sigma * mad_std)
        if wp - bp <= 1e-8: return np.zeros_like(c)
        out = (c - bp) / (wp - bp)
        return np.clip(out, 0, 1)

    if image.ndim == 2:
        return stretch_channel(image)
    if image.ndim == 3 and image.shape[2] == 3:
        return np.stack([stretch_channel(image[..., i]) for i in range(3)], axis=-1)
    raise ValueError("Unsupported image format for autostretch.")

HANDLE_SIZE = 8  # screen pixels (handles stay constant size)
EDGE_GRAB_PX = 12  # screen-pixel tolerance for grabbing edges when zoomed out

class ResizableRotatableRectItem(QGraphicsRectItem):
    def __init__(self, rect: QRectF, parent=None):
        super().__init__(rect, parent)
        pen = QPen(Qt.GlobalColor.green, 2); pen.setCosmetic(True)
        self.setPen(pen)
        self.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsSelectable |
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable |
            QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.setAcceptHoverEvents(True)
        self._fixed_ar: Optional[float] = None
        self._handles: dict[str, QGraphicsEllipseItem] = {}
        self._active: Optional[str] = None
        self._rotating = False
        self._angle0 = 0.0
        self._pivot_scene = QPointF()

        self._grab_pad = 20              # ← extra hit slop in screen px
        self._edge_pad_px = EDGE_GRAB_PX
        self.setZValue(100)             # ← keep above pixmap

        self._mk_handles()
        self.setTransformOriginPoint(self.rect().center())

    def setFixedAspectRatio(self, ratio: Optional[float]):
        self._fixed_ar = ratio

    def _scene_tolerance(self, px: float) -> float:
        """Convert a pixel tolerance into scene/item units using the first view."""
        sc = self.scene()
        if not sc:
            return float(px)
        views = sc.views()
        if not views:
            return float(px)
        v = views[0]
        p0 = v.mapToScene(QPoint(0, 0))
        p1 = v.mapToScene(QPoint(int(px), 0))
        dx = p1.x() - p0.x()
        dy = p1.y() - p0.y()
        return math.hypot(dx, dy)

    def _edge_under_cursor(self, scene_pos: QPointF) -> Optional[str]:
        """
        Return 'l', 'r', 't', or 'b' if the pointer is near an edge (within px-tolerance),
        else None. Works at any zoom/rotation.
        """
        tol = self._scene_tolerance(self._edge_pad_px)
        r = self.rect()
        p = self.mapFromScene(scene_pos)  # local coords (rotation handled)

        # Distance to each edge in item units
        d = {
            "l": abs(p.x() - r.left()),
            "r": abs(p.x() - r.right()),
            "t": abs(p.y() - r.top()),
            "b": abs(p.y() - r.bottom()),
        }
        m = min(d.values())
        if m > tol:
            return None

        # Must also be within the span of the opposite axis (with tolerance)
        if d["l"] == m or d["r"] == m:
            if (r.top() - tol) <= p.y() <= (r.bottom() + tol):
                return "l" if d["l"] <= d["r"] else "r"
        else:  # top/bottom
            if (r.left() - tol) <= p.x() <= (r.right() + tol):
                return "t" if d["t"] <= d["b"] else "b"

        return None


    def _mk_handles(self):
        pen = QPen(Qt.GlobalColor.green, 2); pen.setCosmetic(True)
        brush = QBrush(Qt.GlobalColor.white)
        for name in ("tl", "tr", "br", "bl"):
            h = QGraphicsEllipseItem(0, 0, HANDLE_SIZE, HANDLE_SIZE, self)
            h.setPen(pen); h.setBrush(brush)
            h.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)
            h.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)  # constant-size on screen
            h.setAcceptedMouseButtons(Qt.MouseButton.NoButton)   # ← let parent receive mouse events
            h.setAcceptHoverEvents(False)
            h.setZValue(self.zValue() + 1)
            self._handles[name] = h
        self._sync_handles()

    def _handle_hit(self, h: QGraphicsEllipseItem, scene_pos: QPointF) -> bool:
        """
        True if scene_pos is within the handle ellipse *plus* padding.
        Because the handle ignores view transforms, this padding is in screen px.
        """
        p = h.mapFromScene(scene_pos)
        r = h.rect().adjusted(-self._grab_pad, -self._grab_pad, self._grab_pad, self._grab_pad)
        return r.contains(p)

    def _sync_handles(self):
        r = self.rect(); s = HANDLE_SIZE
        pos = {
            "tl": QPointF(r.left()-s/2,  r.top()-s/2),
            "tr": QPointF(r.right()-s/2, r.top()-s/2),
            "br": QPointF(r.right()-s/2, r.bottom()-s/2),
            "bl": QPointF(r.left()-s/2,  r.bottom()-s/2),
        }
        for k, it in self._handles.items():
            it.setPos(pos[k])

    def hoverMoveEvent(self, e):
        # Corner handles take priority
        for k, h in self._handles.items():
            if self._handle_hit(h, e.scenePos()):
                self.setCursor({
                    "tl": Qt.CursorShape.SizeFDiagCursor,
                    "br": Qt.CursorShape.SizeFDiagCursor,
                    "tr": Qt.CursorShape.SizeBDiagCursor,
                    "bl": Qt.CursorShape.SizeBDiagCursor,
                }.get(k, Qt.CursorShape.ArrowCursor))
                return

        # Edges next
        edge = self._edge_under_cursor(e.scenePos())
        if edge:
            self.setCursor(
                Qt.CursorShape.SizeHorCursor if edge in ("l", "r")
                else Qt.CursorShape.SizeVerCursor
            )
            return

        # Otherwise move
        self.setCursor(Qt.CursorShape.SizeAllCursor)
        super().hoverMoveEvent(e)

    def mousePressEvent(self, e):
        if e.modifiers() == Qt.KeyboardModifier.ShiftModifier:
            self._rotating = True
            self._pivot_scene = self.mapToScene(self.rect().center())
            v0 = e.scenePos() - self._pivot_scene
            self._angle_ref = math.degrees(math.atan2(v0.y(), v0.x()))
            self._angle0 = self.rotation()
            e.accept(); return

        # padded corner hit
        for k, h in self._handles.items():
            if self._handle_hit(h, e.scenePos()):
                self._active = k
                e.accept(); return

        # edge hit
        edge = self._edge_under_cursor(e.scenePos())
        if edge:
            self._active = edge
            e.accept(); return

        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if self._rotating:
            v = e.scenePos() - self._pivot_scene
            ang = math.degrees(math.atan2(v.y(), v.x()))
            self.setRotation(self._angle0 + (ang - self._angle_ref))
            e.accept(); return
        if self._active:
            self._resize_via_handle(e.scenePos()); e.accept(); return
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        self._rotating = False; self._active = None
        super().mouseReleaseEvent(e)

    def itemChange(self, change, value):
        if change in (
            QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged,
            QGraphicsItem.GraphicsItemChange.ItemRotationHasChanged,
            QGraphicsItem.GraphicsItemChange.ItemTransformHasChanged,
        ):
            self._sync_handles()
        return super().itemChange(change, value)

    def _resize_via_handle(self, scene_pt: QPointF):
        r = self.rect()
        p = self.mapFromScene(scene_pt)

        # Corners
        if   self._active == "tl": r.setTopLeft(p)
        elif self._active == "tr": r.setTopRight(p)
        elif self._active == "br": r.setBottomRight(p)
        elif self._active == "bl": r.setBottomLeft(p)
        # Edges
        elif self._active == "l":  r.setLeft(p.x())
        elif self._active == "r":  r.setRight(p.x())
        elif self._active == "t":  r.setTop(p.y())
        elif self._active == "b":  r.setBottom(p.y())

        # Aspect ratio maintenance
        if self._fixed_ar:
            r = r.normalized()
            cx, cy = r.center().x(), r.center().y()
            if self._active in ("l", "r"):  # horizontal resize → adjust height
                w = r.width()
                h = w / self._fixed_ar
                r.setTop(cy - h/2); r.setBottom(cy + h/2)
            elif self._active in ("t", "b"):  # vertical resize → adjust width
                h = r.height()
                w = h * self._fixed_ar
                r.setLeft(cx - w/2); r.setRight(cx + w/2)
            else:  # corner behaves like before
                w = r.width(); h = w / self._fixed_ar
                if self._active in ("tl", "tr"):
                    r.setTop(r.bottom() - h)
                else:
                    r.setBottom(r.top() + h)

        r = r.normalized()
        self.setRect(r)
        self._sync_handles()


class CropDialogPro(QDialog):
    """SASpro crop/rotate dialog working on a Document."""
    crop_applied = pyqtSignal(np.ndarray)

    # persistent “Load Previous”
    _prev_rect: Optional[QRectF] = None
    _prev_angle: float = 0.0
    _prev_pos: QPointF = QPointF()

    def __init__(self, parent, document):
        super().__init__(parent)
        self.setWindowTitle("Crop Tool")
        self.doc = document
        self._rect_item: Optional[ResizableRotatableRectItem] = None
        self._pix_item: Optional[QGraphicsPixmapItem] = None
        self._drawing = False
        self._origin = QPointF()
        self._autostretch_on = False

        # ---------- layout ----------
        main = QVBoxLayout(self)

        info = QLabel(
            "• Click–drag to draw a crop\n"
            "• Drag corner handles to resize\n"
            "• Shift + drag on box to rotate"
        ); info.setStyleSheet("color: gray; font-style: italic;")
        main.addWidget(info)

        # aspect row
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(QLabel("Aspect Ratio:"))
        self.cmb_ar = QComboBox()
        self.cmb_ar.addItems(["Free", "Original", "1:1", "16:9", "9:16", "4:3"])
        row.addWidget(self.cmb_ar)
        row.addStretch(1)
        main.addLayout(row)

        # graphics view
        self.scene = QGraphicsScene(self)
        self.view  = QGraphicsView(self.scene)
        self.view.setRenderHints(self.view.renderHints())
        self.view.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.view.viewport().installEventFilter(self)
        main.addWidget(self.view, 1)

        self._zoom = 1.0            # manual zoom factor
        self._fit_mode = True       # start in Fit-to-View mode

        # nicer zoom behavior
        self.view.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.view.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)  # pan with mouse-drag

        zoom_row = QHBoxLayout()
        self.btn_zoom_out = QToolButton(); self.btn_zoom_out.setText("−")
        self.btn_zoom_in  = QToolButton(); self.btn_zoom_in.setText("+")
        self.btn_zoom_100 = QToolButton(); self.btn_zoom_100.setText("100%")
        self.btn_zoom_fit = QToolButton(); self.btn_zoom_fit.setText("Fit")

        zoom_row.addStretch(1)
        for b in (self.btn_zoom_out, self.btn_zoom_in, self.btn_zoom_100, self.btn_zoom_fit):
            zoom_row.addWidget(b)
        zoom_row.addStretch(1)
        main.addLayout(zoom_row)

        # wire zoom buttons
        self.btn_zoom_in.clicked.connect(lambda: self._zoom_by(1.25))
        self.btn_zoom_out.clicked.connect(lambda: self._zoom_by(1/1.25))
        self.btn_zoom_100.clicked.connect(self._zoom_reset_100)
        self.btn_zoom_fit.clicked.connect(self._fit_view)

        # buttons
        btn_row = QHBoxLayout()
        self.btn_autostretch = QPushButton("Toggle Autostretch")
        self.btn_prev = QPushButton("Load Previous Crop")
        self.btn_apply = QPushButton("Apply")
        self.btn_batch = QPushButton("Batch Crop (all open)")
        self.btn_close = QToolButton(); self.btn_close.setText("Close")
        for b in (self.btn_autostretch, self.btn_prev, self.btn_apply, self.btn_batch, self.btn_close):
            btn_row.addWidget(b)
        main.addLayout(btn_row)

        # wire
        self.cmb_ar.currentTextChanged.connect(self._on_ar_changed)
        self.btn_autostretch.clicked.connect(self._toggle_autostretch)
        self.btn_prev.clicked.connect(self._load_previous)
        self.btn_apply.clicked.connect(self._apply_one)
        self.btn_batch.clicked.connect(self._apply_batch)
        self.btn_close.clicked.connect(self.accept)

        # seed image
        self._load_from_doc()
        self.resize(1000, 720)

    # ---------- image plumbing ----------
    def _img01_from_doc(self) -> np.ndarray:
        arr = np.asarray(self.doc.image)
        if arr.dtype.kind in "ui":
            arr = arr.astype(np.float32) / np.iinfo(self.doc.image.dtype).max
        else:
            arr = arr.astype(np.float32, copy=False)
        # ⬇️ Treat mono with a trailing channel as true mono
        if arr.ndim == 3 and arr.shape[2] == 1:
            arr = arr[..., 0]
        return np.clip(arr, 0.0, 1.0)

    def _load_from_doc(self):
        self._full01 = self._img01_from_doc()
        self._orig_h, self._orig_w = self._full01.shape[:2]
        self._preview01 = self._full01 if not self._autostretch_on else siril_style_autostretch(self._full01)

        self.scene.clear()
        q = self._to_qimage(self._preview01)
        pm = QPixmap.fromImage(q)
        self._pix_item = QGraphicsPixmapItem(pm)
        self._pix_item.setZValue(-1)
        self.scene.addItem(self._pix_item)
        self._apply_zoom_transform()

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        if self._fit_mode:
            self._apply_zoom_transform()

    @staticmethod
    def _to_qimage(img01: np.ndarray) -> QImage:
        # Ensure shapes we expect
        if img01.ndim == 3 and img01.shape[2] == 1:
            img01 = img01[..., 0]

        if img01.ndim == 2:
            buf = np.ascontiguousarray((img01 * 255).astype(np.uint8))
            h, w = buf.shape
            bpl = buf.strides[0]  # == w for contiguous grayscale
            return QImage(buf.tobytes(), w, h, bpl, QImage.Format.Format_Grayscale8)

        if img01.ndim == 3 and img01.shape[2] == 3:
            buf = np.ascontiguousarray((img01 * 255).astype(np.uint8))
            h, w, _ = buf.shape
            bpl = buf.strides[0]  # == 3*w for contiguous RGB
            return QImage(buf.tobytes(), w, h, bpl, QImage.Format.Format_RGB888)

        raise ValueError(f"Unsupported image shape for preview: {img01.shape}")

    # ---------- aspect ratio ----------
    def _on_ar_changed(self, txt: str):
        if not self._rect_item: return
        if txt == "Free":
            ar = None
        elif txt == "Original":
            ar = self._orig_w / self._orig_h
        else:
            a, b = map(float, txt.split(":")); ar = a / b
        self._rect_item.setFixedAspectRatio(ar)
        if ar is not None:
            r = self._rect_item.rect()
            w = r.width(); h = w / ar
            c = r.center()
            nr = QRectF(c.x()-w/2, c.y()-h/2, w, h)
            self._rect_item.setRect(nr)
            self._rect_item.setTransformOriginPoint(nr.center())

    # ---------- drawing / interaction ----------
    def eventFilter(self, src, e):
        if src is self.view.viewport():
            if e.type() == QEvent.Type.Wheel and (e.modifiers() & Qt.KeyboardModifier.ControlModifier):
                delta = e.angleDelta().y()
                self._zoom_by(1.25 if delta > 0 else 1/1.25)
                return True            
            if e.type() in (QEvent.Type.MouseButtonPress, QEvent.Type.MouseMove, QEvent.Type.MouseButtonRelease):
                scene_pt = self.view.mapToScene(e.pos())

            if self._rect_item is None:
                if e.type() == QEvent.Type.MouseButtonPress and e.button() == Qt.MouseButton.LeftButton:
                    self._drawing = True; self._origin = scene_pt; return True

                if e.type() == QEvent.Type.MouseMove and self._drawing:
                    r = QRectF(self._origin, scene_pt).normalized()
                    r = self._apply_ar_to_rect(r, live=True, scene_pt=scene_pt)
                    self._draw_live_rect(r); return True

                if e.type() == QEvent.Type.MouseButtonRelease and e.button() == Qt.MouseButton.LeftButton and self._drawing:
                    self._drawing = False
                    r = QRectF(self._origin, scene_pt).normalized()
                    r = self._apply_ar_to_rect(r, live=False, scene_pt=scene_pt)
                    self._clear_live_rect()
                    self._rect_item = ResizableRotatableRectItem(r)
                    self._rect_item.setZValue(10)
                    self._rect_item.setFixedAspectRatio(self._current_ar_value())
                    self.scene.addItem(self._rect_item)

                    # remember for “Load Previous”
                    CropDialogPro._prev_rect = QRectF(r)
                    CropDialogPro._prev_angle = self._rect_item.rotation()
                    CropDialogPro._prev_pos = self._rect_item.pos()
                    return True

            return False
        return super().eventFilter(src, e)
    
    def _apply_zoom_transform(self):
        """Apply current zoom or fit mode to the view."""
        if not self._pix_item:
            return
        if self._fit_mode:
            # Fit pixmap + rect overlay
            rect = self._pix_item.mapRectToScene(self._pix_item.boundingRect())
            self.view.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
            self.view.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)
            self.view.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        else:
            self.view.resetTransform()
            self.view.scale(self._zoom, self._zoom)

    def _fit_view(self):
        self._fit_mode = True
        self._apply_zoom_transform()

    def _zoom_reset_100(self):
        self._fit_mode = False
        self._zoom = 1.0
        self._apply_zoom_transform()

    def _zoom_by(self, factor: float):
        self._fit_mode = False
        # clamp zoom
        newz = min(16.0, max(0.05, self._zoom * float(factor)))
        if abs(newz - self._zoom) < 1e-4:
            return
        self._zoom = newz
        self._apply_zoom_transform()


    def _current_ar_value(self) -> Optional[float]:
        txt = self.cmb_ar.currentText()
        if txt == "Free": return None
        if txt == "Original": return self._orig_w / self._orig_h
        a, b = map(float, txt.split(":")); return a / b

    def _apply_ar_to_rect(self, r: QRectF, live: bool, scene_pt: QPointF) -> QRectF:
        txt = self.cmb_ar.currentText()
        if txt == "Free": return r
        ar = self._orig_w / self._orig_h if txt == "Original" else (lambda a,b: a/b)(*map(float, txt.split(":")))
        w = r.width(); h = w / ar
        if scene_pt.y() < self._origin.y(): r.setTop(r.bottom() - h)
        else:                                 r.setBottom(r.top() + h)
        return r.normalized()

    def _draw_live_rect(self, r: QRectF):
        if hasattr(self, "_live_rect") and self._live_rect:
            self.scene.removeItem(self._live_rect)
        pen = QPen(QColor(0,255,0), 2, Qt.PenStyle.DashLine); pen.setCosmetic(True)
        self._live_rect = self.scene.addRect(r, pen)

    def _clear_live_rect(self):
        if hasattr(self, "_live_rect") and self._live_rect:
            self.scene.removeItem(self._live_rect); self._live_rect = None

    # ---------- preview toggles ----------
    def _toggle_autostretch(self):
        self._autostretch_on = not self._autostretch_on
        saved = self._snapshot_rect_state()
        self._load_from_doc()
        self._restore_rect_state(saved)

    def _snapshot_rect_state(self):
        if not self._rect_item: return None
        return (QRectF(self._rect_item.rect()),
                float(self._rect_item.rotation()),
                QPointF(self._rect_item.pos()))

    def _restore_rect_state(self, state):
        if not state: return
        r, ang, pos = state
        self._rect_item = ResizableRotatableRectItem(r)
        self._rect_item.setZValue(10)
        self._rect_item.setFixedAspectRatio(self._current_ar_value())
        self._rect_item.setRotation(ang)
        self._rect_item.setPos(pos)
        self._rect_item.setTransformOriginPoint(r.center())
        self.scene.addItem(self._rect_item)

    def _load_previous(self):
        if CropDialogPro._prev_rect is None:
            QMessageBox.information(self, "No Previous", "No previous crop stored.")
            return
        if self._rect_item:
            self.scene.removeItem(self._rect_item)
        r = QRectF(CropDialogPro._prev_rect)
        self._rect_item = ResizableRotatableRectItem(r)
        self._rect_item.setZValue(10)
        self._rect_item.setFixedAspectRatio(self._current_ar_value())
        self._rect_item.setRotation(CropDialogPro._prev_angle)
        self._rect_item.setPos(CropDialogPro._prev_pos)
        self._rect_item.setTransformOriginPoint(r.center())
        self.scene.addItem(self._rect_item)

    # ---------- apply ----------
    def _corners_scene(self):
        rl = self._rect_item.rect()
        loc = [rl.topLeft(), rl.topRight(), rl.bottomRight(), rl.bottomLeft()]
        return [self._rect_item.mapToScene(p) for p in loc]

    def _scene_to_img_pixels(self, pt_scene: QPointF, w_img: int, h_img: int):
        pm = self._pix_item.pixmap()
        sx, sy = w_img / pm.width(), h_img / pm.height()
        return np.array([pt_scene.x() * sx, pt_scene.y() * sy], dtype=np.float32)

    def _apply_one(self):
        if not self._rect_item:
            QMessageBox.warning(self, "No Selection", "Draw & finalize a crop first.")
            return

        corners = self._corners_scene()
        w_img, h_img = self._orig_w, self._orig_h
        src = np.array([self._scene_to_img_pixels(p, w_img, h_img) for p in corners], dtype=np.float32)

        width  = np.linalg.norm(src[1] - src[0])
        height = np.linalg.norm(src[3] - src[0])
        dst = np.array([[0,0],[width,0],[width,height],[0,height]], dtype=np.float32)

        M = cv2.getPerspectiveTransform(src, dst)
        out = cv2.warpPerspective(self._full01, M, (int(round(width)), int(round(height))), flags=cv2.INTER_LINEAR)

        # record previous
        CropDialogPro._prev_rect = QRectF(self._rect_item.rect())
        CropDialogPro._prev_angle = float(self._rect_item.rotation())
        CropDialogPro._prev_pos = QPointF(self._rect_item.pos())

        # push back to document (float [0..1])
        try:
            self.doc.apply_edit(out.copy(), metadata={"step_name": "Crop"}, step_name="Crop")
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Apply failed", str(e))

    def _apply_batch(self):
        if not self._rect_item:
            QMessageBox.warning(self, "No Selection", "Draw & finalize a crop first.")
            return

        # normalize the crop polygon to THIS image size
        corners = self._corners_scene()
        src_this = np.array([self._scene_to_img_pixels(p, self._orig_w, self._orig_h) for p in corners], dtype=np.float32)
        norm = src_this / np.array([self._orig_w, self._orig_h], dtype=np.float32)

        # collect all open documents from the MDI
        win = self.parent()
        subs = getattr(win, "mdi", None).subWindowList() if hasattr(win, "mdi") else []
        docs = []
        for sw in subs:
            vw = sw.widget()
            d = getattr(vw, "document", None)
            if d is not None:
                docs.append(d)

        if not docs:
            QMessageBox.information(self, "No Images", "No open images to crop.")
            return

        ok = QMessageBox.question(
            self, "Confirm Batch",
            f"Apply this crop to {len(docs)} open image(s)?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if ok != QMessageBox.StandardButton.Yes:
            return

        for d in docs:
            img = np.asarray(d.image)
            if img.dtype.kind in "ui":
                src01 = img.astype(np.float32) / np.iinfo(d.image.dtype).max
            else:
                src01 = img.astype(np.float32, copy=False)
            h, w = src01.shape[:2]
            src_pts = norm * np.array([w, h], dtype=np.float32)

            w_out  = int(round(np.linalg.norm(src_pts[1] - src_pts[0])))
            h_out  = int(round(np.linalg.norm(src_pts[3] - src_pts[0])))
            dst = np.array([[0,0],[w_out,0],[w_out,h_out],[0,h_out]], dtype=np.float32)

            M = cv2.getPerspectiveTransform(src_pts.astype(np.float32), dst)
            cropped = cv2.warpPerspective(src01, M, (w_out, h_out), flags=cv2.INTER_LINEAR)

            try:
                d.apply_edit(cropped.copy(), metadata={"step_name":"Crop"}, step_name="Crop")
            except Exception:
                pass

        QMessageBox.information(self, "Batch Crop", "Applied crop to all open images.")
        self.accept()
