# pro/crop_dialog_pro.py
from __future__ import annotations

import math
import os
import numpy as np
import cv2
from typing import Optional
import platform
from PyQt6.QtCore import Qt, QEvent, QPointF, QRectF, pyqtSignal, QPoint, QTimer, QSettings, QByteArray
from PyQt6.QtGui import QPixmap, QImage, QPen, QBrush, QColor, QPainterPath
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox, QToolButton,
    QMessageBox, QGraphicsScene, QGraphicsView, QGraphicsRectItem, QGraphicsEllipseItem,
    QGraphicsItem, QGraphicsPixmapItem, QSpinBox, QGraphicsPathItem,
    QGroupBox, QCheckBox, QScrollArea, QWidget, QSizePolicy, QFrame,
)

from setiastro.saspro.wcs_update import update_wcs_after_crop
from setiastro.saspro.widgets.themed_buttons import themed_toolbtn

# -------- util: s_-style preview stretch (non-destructive) ----------
def histogram_style_autostretch(image: np.ndarray, sigma: float = 3.0) -> np.ndarray:
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


# -------- blend mode compositing -----------------------------------
_BLEND_MODES = ["Screen", "Average", "Min", "Max", "Difference"]

# Colorize palette — six perceptually distinct hues cycling for N images.
# Each entry is an RGB [0,1] multiplier applied to the luminance of the layer.
_OVERLAP_COLORS: list[tuple[float, float, float]] = [
    (0.30, 0.55, 1.00),   # blue
    (1.00, 0.30, 0.30),   # red
    (0.30, 1.00, 0.40),   # green
    (1.00, 0.30, 1.00),   # magenta
    (0.20, 0.95, 0.95),   # cyan
    (1.00, 0.90, 0.10),   # yellow
]


def _colorize_layer(layer: np.ndarray, color: tuple[float, float, float]) -> np.ndarray:
    """
    Tint a HxWx3 float32 layer with a solid hue.
    We convert to luminance first so mono and colour images both tint uniformly,
    then multiply that luma by the RGB color vector.
    Result is clamped to [0,1].
    """
    lum = (0.2126 * layer[..., 0] +
           0.7152 * layer[..., 1] +
           0.0722 * layer[..., 2])[..., np.newaxis]          # H×W×1
    tinted = lum * np.array(color, dtype=np.float32)         # broadcast to H×W×3
    return np.clip(tinted, 0.0, 1.0).astype(np.float32)

def _to_rgb01(arr: np.ndarray) -> np.ndarray:
    """Normalise any doc image array to float32 HxWx3 in [0,1]."""
    a = arr.astype(np.float32, copy=False)
    if a.dtype.kind in "ui":
        a = a / np.iinfo(arr.dtype).max
    if a.ndim == 2:
        a = np.stack([a] * 3, axis=-1)
    elif a.ndim == 3 and a.shape[2] == 1:
        a = np.repeat(a, 3, axis=2)
    elif a.ndim == 3 and a.shape[2] > 3:
        a = a[..., :3]
    return np.clip(a, 0.0, 1.0)


def _blend_images(layers: list[np.ndarray], mode: str) -> np.ndarray:
    """
    Blend a list of HxWx3 float32 [0,1] arrays using the chosen mode.
    All layers must be the same shape (caller's responsibility to resize).
    """
    if not layers:
        raise ValueError("No layers to blend")
    if len(layers) == 1:
        return layers[0].copy()

    stack = np.stack(layers, axis=0)   # (N, H, W, 3)

    if mode == "Average":
        return stack.mean(axis=0).astype(np.float32)
    if mode == "Min":
        return stack.min(axis=0).astype(np.float32)
    if mode == "Max":
        return stack.max(axis=0).astype(np.float32)
    if mode == "Difference":
        # pairwise max-abs difference from first layer
        base = stack[0]
        diff = np.max(np.abs(stack - base), axis=0)
        return np.clip(diff, 0.0, 1.0).astype(np.float32)
    # Screen (default): 1 - prod(1 - layer_i)
    result = np.ones_like(stack[0])
    for layer in layers:
        result = result * (1.0 - layer)
    return np.clip(1.0 - result, 0.0, 1.0).astype(np.float32)


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
        self._bounds_scene: QRectF | None = None
        self._clamp_eps_deg = 0.25
        self._grab_pad = 20
        self._edge_pad_px = EDGE_GRAB_PX
        # center crosshair / X
        self._crosshair = QGraphicsPathItem(self)
        pen_x = QPen(QColor(0, 255, 0), 1, Qt.PenStyle.SolidLine)
        pen_x.setCosmetic(True)
        self._crosshair.setPen(pen_x)
        self._crosshair.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        self._crosshair.setZValue(self.zValue() + 2)
        self.setZValue(100)

        self._mk_handles()
        self.setTransformOriginPoint(self.rect().center())
        self._sync_crosshair()

    def _sync_crosshair(self):
        r = self.rect()
        c = r.center()
        s = 0.12 * min(r.width(), r.height())
        s = max(10.0, min(40.0, float(s)))
        p = QPainterPath()
        p.moveTo(c.x() - s, c.y()); p.lineTo(c.x() + s, c.y())
        p.moveTo(c.x(), c.y() - s); p.lineTo(c.x(), c.y() + s)
        p.moveTo(c.x() - s, c.y() - s); p.lineTo(c.x() + s, c.y() + s)
        p.moveTo(c.x() - s, c.y() + s); p.lineTo(c.x() + s, c.y() - s)
        self._crosshair.setPath(p)

    def set_crosshair_visible(self, on: bool):
        try:
            self._crosshair.setVisible(bool(on))
        except Exception:
            pass

    def setFixedAspectRatio(self, ratio: Optional[float]):
        self._fixed_ar = ratio

    def _scene_tolerance(self, px: float) -> float:
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

    def setBoundsSceneRect(self, r: QRectF | None):
        self._bounds_scene = QRectF(r) if r is not None else None

    def _is_unrotated(self) -> bool:
        a = float(self.rotation()) % 360.0
        if a > 180.0:
            a -= 360.0
        return abs(a) < self._clamp_eps_deg

    def _bounds_local(self) -> QRectF | None:
        if self._bounds_scene is None:
            return None
        tl = self.mapFromScene(self._bounds_scene.topLeft())
        br = self.mapFromScene(self._bounds_scene.bottomRight())
        return QRectF(tl, br).normalized()

    def _edge_under_cursor(self, scene_pos: QPointF) -> Optional[str]:
        tol = self._scene_tolerance(self._edge_pad_px)
        r = self.rect()
        p = self.mapFromScene(scene_pos)
        d = {
            "l": abs(p.x() - r.left()),
            "r": abs(p.x() - r.right()),
            "t": abs(p.y() - r.top()),
            "b": abs(p.y() - r.bottom()),
        }
        m = min(d.values())
        if m > tol:
            return None
        if d["l"] == m or d["r"] == m:
            if (r.top() - tol) <= p.y() <= (r.bottom() + tol):
                return "l" if d["l"] <= d["r"] else "r"
        else:
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
            h.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
            h.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            h.setAcceptHoverEvents(False)
            h.setZValue(self.zValue() + 1)
            self._handles[name] = h
        self._sync_handles()

    def _handle_hit(self, h: QGraphicsEllipseItem, scene_pos: QPointF) -> bool:
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
        self._sync_crosshair()

    def hoverMoveEvent(self, e):
        for k, h in self._handles.items():
            if self._handle_hit(h, e.scenePos()):
                self.setCursor({
                    "tl": Qt.CursorShape.SizeFDiagCursor,
                    "br": Qt.CursorShape.SizeFDiagCursor,
                    "tr": Qt.CursorShape.SizeBDiagCursor,
                    "bl": Qt.CursorShape.SizeBDiagCursor,
                }.get(k, Qt.CursorShape.ArrowCursor))
                return
        edge = self._edge_under_cursor(e.scenePos())
        if edge:
            self.setCursor(
                Qt.CursorShape.SizeHorCursor if edge in ("l", "r")
                else Qt.CursorShape.SizeVerCursor
            )
            return
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
        for k, h in self._handles.items():
            if self._handle_hit(h, e.scenePos()):
                self._active = k
                e.accept(); return
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

        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange:
            if self._bounds_scene is not None and self._is_unrotated():
                new_pos = QPointF(value)
                sr0 = self.mapRectToScene(self.rect())
                d = new_pos - self.pos()
                sr = sr0.translated(d)
                b = self._bounds_scene
                dx = 0.0; dy = 0.0
                if sr.left() < b.left():   dx = b.left()  - sr.left()
                elif sr.right() > b.right(): dx = b.right() - sr.right()
                if sr.top() < b.top():     dy = b.top()   - sr.top()
                elif sr.bottom() > b.bottom(): dy = b.bottom() - sr.bottom()
                if dx != 0.0 or dy != 0.0:
                    return new_pos + QPointF(dx, dy)
                return new_pos

        return super().itemChange(change, value)

    def _resize_via_handle(self, scene_pt: QPointF):
        r = self.rect()
        p = self.mapFromScene(scene_pt)
        if self._bounds_scene is not None and self._is_unrotated():
            bL = self._bounds_local()
            if bL is not None:
                px = min(max(p.x(), bL.left()), bL.right())
                py = min(max(p.y(), bL.top()),  bL.bottom())
                p = QPointF(px, py)
        if   self._active == "tl": r.setTopLeft(p)
        elif self._active == "tr": r.setTopRight(p)
        elif self._active == "br": r.setBottomRight(p)
        elif self._active == "bl": r.setBottomLeft(p)
        elif self._active == "l":  r.setLeft(p.x())
        elif self._active == "r":  r.setRight(p.x())
        elif self._active == "t":  r.setTop(p.y())
        elif self._active == "b":  r.setBottom(p.y())
        if self._fixed_ar:
            r = r.normalized()
            cx, cy = r.center().x(), r.center().y()
            if self._active in ("l", "r"):
                w = r.width(); h = w / self._fixed_ar
                r.setTop(cy - h/2); r.setBottom(cy + h/2)
            elif self._active in ("t", "b"):
                h = r.height(); w = h * self._fixed_ar
                r.setLeft(cx - w/2); r.setRight(cx + w/2)
            else:
                w = r.width(); h = w / self._fixed_ar
                if self._active in ("tl", "tr"):
                    r.setTop(r.bottom() - h)
                else:
                    r.setBottom(r.top() + h)
        r = r.normalized()
        self.setRect(r)
        self._sync_handles()


# =============================================================================
# Multi-image overlap panel
# =============================================================================

class OverlapPanel(QWidget):
    """
    Collapsible side panel that lists all open MDI documents as checkboxes,
    provides a blend-mode dropdown, and exposes the composite result.
    """
    composite_changed = pyqtSignal()   # emitted whenever the composite should be refreshed

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMaximumWidth(280)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)

        self._docs: list = []          # list of document objects
        self._checks: list[QCheckBox] = []
        self._composite_mode = False   # True = composite preview active
        self._colorize = False         # True = tint each layer before blending

        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        # ── header ──────────────────────────────────────────────────────────
        hdr = QHBoxLayout()
        self.lbl_title = QLabel("<b>Multi-Image Overlap</b>")
        hdr.addWidget(self.lbl_title)
        hdr.addStretch(1)
        root.addLayout(hdr)

        sep = QFrame(); sep.setFrameShape(QFrame.Shape.HLine)
        root.addWidget(sep)

        # ── blend mode ──────────────────────────────────────────────────────
        blend_row = QHBoxLayout()
        blend_row.addWidget(QLabel("Blend:"))
        self.cmb_blend = QComboBox()
        self.cmb_blend.addItems(_BLEND_MODES)
        self.cmb_blend.setCurrentText("Screen")
        blend_row.addWidget(self.cmb_blend, 1)
        root.addLayout(blend_row)

        # ── colorize toggle ─────────────────────────────────────────────────
        self.chk_colorize = QCheckBox("Colorize images (blue / red / green / …)")
        self.chk_colorize.setChecked(False)
        self.chk_colorize.setToolTip(
            "Tint each image a distinct hue before blending.\n"
            "Overlap regions converge toward white/grey; frame edges\n"
            "show their assigned color — makes coverage gaps obvious."
        )
        root.addWidget(self.chk_colorize)

        # ── image list (scrollable) ─────────────────────────────────────────
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._list_widget = QWidget()
        self._list_layout = QVBoxLayout(self._list_widget)
        self._list_layout.setContentsMargins(2, 2, 2, 2)
        self._list_layout.setSpacing(2)
        self._list_layout.addStretch(1)
        self.scroll.setWidget(self._list_widget)
        root.addWidget(self.scroll, 1)

        # ── action buttons ──────────────────────────────────────────────────
        self.btn_refresh = QPushButton("Refresh Image List")
        self.btn_apply   = QPushButton("Apply Composite Preview")
        self.btn_apply.setCheckable(True)
        self.btn_apply.setToolTip(
            "Toggle composite preview on the canvas.\n"
            "The crop rectangle and Apply/Batch buttons still work normally."
        )
        self.btn_clear   = QPushButton("Back to Single Image")
        self.btn_clear.setEnabled(False)

        root.addWidget(self.btn_refresh)
        root.addWidget(self.btn_apply)
        root.addWidget(self.btn_clear)

        # ── wiring ──────────────────────────────────────────────────────────
        self.cmb_blend.currentTextChanged.connect(self._on_blend_changed)
        self.chk_colorize.stateChanged.connect(self._on_colorize_changed)
        self.btn_refresh.clicked.connect(lambda _: self.refresh_docs())
        self.btn_apply.clicked.connect(self._on_apply_toggled)
        self.btn_clear.clicked.connect(self._on_clear)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def refresh_docs(self, all_docs: list | None = None):
        """
        Populate the checkbox list.  Pass a list of document objects,
        or call with no args to re-fetch from the parent CropDialogPro.
        """
        if all_docs is not None:
            self._docs = list(all_docs)
        else:
            # Re-fetch from the parent crop dialog if available
            parent_dlg = self.parent()
            if hasattr(parent_dlg, "_collect_all_docs"):
                self._docs = parent_dlg._collect_all_docs()
        self._rebuild_list()

    def checked_docs(self) -> list:
        """Return documents whose checkboxes are ticked."""
        return [
            d for d, cb in zip(self._docs, self._checks) if cb.isChecked()
        ]

    def blend_mode(self) -> str:
        return self.cmb_blend.currentText()

    def is_composite_active(self) -> bool:
        return self._composite_mode

    def build_composite(
        self,
        target_shape: tuple[int, int],
        autostretch: bool,
    ) -> np.ndarray | None:
        """
        Build and return the blended HxWx3 float32 array at *target_shape* (H, W).
        Returns None if fewer than 2 images are checked.
        """
        docs = self.checked_docs()
        if len(docs) < 2:
            return None

        H, W = target_shape
        colorize = self._colorize
        layers = []
        for idx, d in enumerate(docs):
            raw = _to_rgb01(np.asarray(d.image))
            # resize to match the primary image canvas
            dh, dw = raw.shape[:2]
            if (dh, dw) != (H, W):
                raw = cv2.resize(raw, (W, H), interpolation=cv2.INTER_AREA)
            if autostretch:
                raw = histogram_style_autostretch(raw)
            if colorize:
                color = _OVERLAP_COLORS[idx % len(_OVERLAP_COLORS)]
                raw = _colorize_layer(raw, color)
            layers.append(raw.astype(np.float32))

        return _blend_images(layers, self.blend_mode())

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _rebuild_list(self):
        # clear existing checkboxes
        for cb in self._checks:
            cb.deleteLater()
        self._checks.clear()

        # remove all but the trailing stretch
        while self._list_layout.count() > 1:
            item = self._list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for doc in self._docs:
            # ImageDocument.display_name() is a method; fall back through
            # metadata keys that DocManager actually populates
            title = None
            try:
                fn = getattr(doc, "display_name", None)
                if callable(fn):
                    title = fn()
            except Exception:
                pass
            if not title:
                meta = getattr(doc, "metadata", {}) or {}
                title = (
                    meta.get("display_name")
                    or meta.get("file_path")
                    or getattr(doc, "title", None)
                    or getattr(doc, "filename", None)
                    or "Untitled"
                )
                # if file_path, show just the basename
                if title and os.path.sep in title:
                    title = os.path.basename(title)
            cb = QCheckBox(str(title))
            cb.setChecked(True)
            cb.stateChanged.connect(self._on_check_changed)
            self._list_layout.insertWidget(self._list_layout.count() - 1, cb)
            self._checks.append(cb)

    def _on_check_changed(self, _):
        if self._composite_mode:
            self.composite_changed.emit()

    def _on_blend_changed(self, _):
        if self._composite_mode:
            self.composite_changed.emit()

    def _on_colorize_changed(self, state: int):
        self._colorize = bool(state)
        if self._composite_mode:
            self.composite_changed.emit()

    def _on_apply_toggled(self, checked: bool):
        self._composite_mode = bool(checked)
        self.btn_clear.setEnabled(self._composite_mode)
        self.btn_apply.setText(
            "✓ Composite Active" if self._composite_mode else "Apply Composite Preview"
        )
        self.composite_changed.emit()

    def _on_clear(self):
        self._composite_mode = False
        self.btn_apply.setChecked(False)
        self.btn_apply.setText("Apply Composite Preview")
        self.btn_clear.setEnabled(False)
        self.composite_changed.emit()


# =============================================================================
# Main crop dialog
# =============================================================================

class CropDialogPro(QDialog):
    """SASpro crop/rotate dialog working on a Document."""
    crop_applied = pyqtSignal(np.ndarray)

    # persistent "Load Previous"
    _prev_rect: Optional[QRectF] = None
    _prev_angle: float = 0.0
    _prev_pos: QPointF = QPointF()

    def __init__(self, parent, document):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Crop Tool"))
        self.setWindowFlag(Qt.WindowType.Window, True)
        if platform.system() == "Darwin":
            self.setWindowFlag(Qt.WindowType.Tool, True)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setModal(False)
        self._main = parent
        self.doc = document
        self._live_cross = None

        self._follow_conn = False
        if hasattr(self._main, "currentDocumentChanged"):
            try:
                self._main.currentDocumentChanged.connect(self._on_active_doc_changed)
                self._follow_conn = True
            except Exception:
                self._follow_conn = False

        self.finished.connect(self._cleanup_connections)
        try:
            self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        except Exception:
            pass
        self._rect_item: Optional[ResizableRotatableRectItem] = None
        self._pix_item: Optional[QGraphicsPixmapItem] = None
        self._drawing = False
        self._origin = QPointF()
        self._autostretch_on = True

        # ── outer horizontal split: controls+canvas on left, panel on right ──
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ── left side (original layout) ───────────────────────────────────
        left_widget = QWidget()
        main = QVBoxLayout(left_widget)
        outer.addWidget(left_widget, 1)

        # ── right side: overlap panel (hidden until user opens it) ────────
        self._overlap_panel = OverlapPanel(self)
        self._overlap_panel.setVisible(False)
        self._overlap_panel.composite_changed.connect(self._on_composite_changed)
        outer.addWidget(self._overlap_panel, 0)

        # ── info label ───────────────────────────────────────────────────
        info = QLabel(self.tr(
            "• Click–drag to draw a crop\n"
            "• Drag corner handles to resize\n"
            "• Shift + drag on box to rotate"
        )); info.setStyleSheet("color: gray; font-style: italic;")
        main.addWidget(info)

        # aspect row
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(QLabel(self.tr("Aspect Ratio:")))
        self.cmb_ar = QComboBox()
        self.cmb_ar.addItems([
            self.tr("Free"), self.tr("Original"),
            "1:1",
            "3:2", "2:3",
            "4:3", "3:4",
            "4:5", "5:4",
            "16:9", "9:16",
            "21:9", "9:21",
            "2:1", "1:2",
            "3:5", "5:3",
        ])
        row.addWidget(self.cmb_ar)
        row.addStretch(1)
        main.addLayout(row)

        # typed margins
        margins_row = QHBoxLayout()
        margins_row.addStretch(1)
        margins_row.addWidget(QLabel(self.tr("Margins (px):")))
        self.sb_top    = QSpinBox(); self.sb_top.setSuffix(" px")
        self.sb_right  = QSpinBox(); self.sb_right.setSuffix(" px")
        self.sb_bottom = QSpinBox(); self.sb_bottom.setSuffix(" px")
        self.sb_left   = QSpinBox(); self.sb_left.setSuffix(" px")
        for sb in (self.sb_top, self.sb_bottom, self.sb_left, self.sb_right):
            sb.setRange(0, 1_000_000)
        margins_row.addWidget(QLabel(self.tr("Top")));    margins_row.addWidget(self.sb_top)
        margins_row.addSpacing(8)
        margins_row.addWidget(QLabel(self.tr("Right")));  margins_row.addWidget(self.sb_right)
        margins_row.addSpacing(8)
        margins_row.addWidget(QLabel(self.tr("Bottom"))); margins_row.addWidget(self.sb_bottom)
        margins_row.addSpacing(8)
        margins_row.addWidget(QLabel(self.tr("Left")));   margins_row.addWidget(self.sb_left)
        margins_row.addStretch(1)
        main.addLayout(margins_row)

        self._suppress_margin_sync = False
        def _on_margin_changed(_):
            if self._suppress_margin_sync:
                return
            self._apply_margin_inputs()
        for sb in (self.sb_top, self.sb_right, self.sb_bottom, self.sb_left):
            sb.valueChanged.connect(_on_margin_changed)

        # graphics view
        self.scene = QGraphicsScene(self)
        self.view  = QGraphicsView(self.scene)
        self.view.setRenderHints(self.view.renderHints())
        self.view.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.view.viewport().installEventFilter(self)
        main.addWidget(self.view, 1)

        self._zoom = 1.0
        self._fit_mode = True
        self.view.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.view.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)

        zoom_row = QHBoxLayout()
        zoom_row.addStretch(1)
        self.btn_zoom_out = themed_toolbtn("zoom-out",      self.tr("Zoom Out"))
        self.btn_zoom_in  = themed_toolbtn("zoom-in",       self.tr("Zoom In"))
        self.btn_zoom_100 = themed_toolbtn("zoom-original", self.tr("Zoom 100%"))
        self.btn_zoom_fit = themed_toolbtn("zoom-fit-best", self.tr("Fit to View"))
        for b in (self.btn_zoom_out, self.btn_zoom_in, self.btn_zoom_100, self.btn_zoom_fit):
            zoom_row.addWidget(b)
        zoom_row.addStretch(1)
        main.addLayout(zoom_row)

        dim_row = QHBoxLayout()
        dim_row.addStretch(1)
        self.lbl_dims = QLabel(self.tr("Selection: —"))
        self.lbl_dims.setStyleSheet("color: gray;")
        dim_row.addWidget(self.lbl_dims)
        dim_row.addStretch(1)
        main.addLayout(dim_row)

        self.btn_zoom_in.clicked.connect(lambda: self._zoom_by(1.25))
        self.btn_zoom_out.clicked.connect(lambda: self._zoom_by(1/1.25))
        self.btn_zoom_100.clicked.connect(self._zoom_reset_100)
        self.btn_zoom_fit.clicked.connect(self._fit_view)

        # buttons row
        btn_row = QHBoxLayout()
        self.btn_autostretch = QPushButton(self.tr("Toggle Autostretch"))
        self.btn_prev        = QPushButton(self.tr("Load Previous Crop"))
        self.btn_apply       = QPushButton(self.tr("Apply"))
        self.btn_batch       = QPushButton(self.tr("Batch Crop (all open)"))
        self.btn_overlap     = QPushButton(self.tr("⊞ Multi-Image Overlap"))
        self.btn_overlap.setCheckable(True)
        self.btn_overlap.setToolTip(
            "Open the multi-image overlap panel.\n"
            "Select which open images to composite together so you can\n"
            "find the best shared crop region before applying."
        )
        self.btn_close = QToolButton(); self.btn_close.setText(self.tr("Close"))
        for b in (self.btn_autostretch, self.btn_prev, self.btn_apply,
                  self.btn_batch, self.btn_overlap, self.btn_close):
            btn_row.addWidget(b)
        main.addLayout(btn_row)

        # composite status label (hidden when not in composite mode)
        self._lbl_composite_status = QLabel()
        self._lbl_composite_status.setStyleSheet(
            "color: #c8a000; font-style: italic; font-size: 11px;"
        )
        self._lbl_composite_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl_composite_status.setVisible(False)
        main.addWidget(self._lbl_composite_status)

        # wire
        self.cmb_ar.currentTextChanged.connect(self._on_ar_changed)
        self.btn_autostretch.clicked.connect(self._toggle_autostretch)
        self.btn_prev.clicked.connect(self._load_previous)
        self.btn_apply.clicked.connect(self._apply_one)
        self.btn_batch.clicked.connect(self._apply_batch)
        self.btn_overlap.clicked.connect(self._toggle_overlap_panel)
        self.btn_close.clicked.connect(self.close)

        # seed image
        self._load_from_doc()
        self._update_margin_spin_ranges()
        self.resize(1100, 720)
        self._deferred_fit()

    # =========================================================================
    # Overlap panel integration
    # =========================================================================

    def _toggle_overlap_panel(self, checked: bool):
        self._overlap_panel.setVisible(checked)
        if checked:
            # populate with all open docs
            self._overlap_panel.refresh_docs(self._collect_all_docs())

    def _collect_all_docs(self) -> list:
        """Gather all open document objects from the MDI."""
        docs = []
        try:
            subs = getattr(self._main, "mdi", None)
            if subs is not None:
                for sw in subs.subWindowList():
                    vw = sw.widget()
                    d = getattr(vw, "document", None)
                    if d is not None and getattr(d, "image", None) is not None:
                        docs.append(d)
        except Exception:
            pass
        return docs

    def _on_composite_changed(self):
        """Called whenever the overlap panel wants a canvas refresh."""
        if self._overlap_panel.is_composite_active():
            self._show_composite()
        else:
            self._show_single()

    def _show_composite(self):
        """Build composite and update the scene pixmap."""
        composite = self._overlap_panel.build_composite(
            target_shape=(self._orig_h, self._orig_w),
            autostretch=self._autostretch_on,
        )
        if composite is None:
            self._lbl_composite_status.setText(
                self.tr("Select at least 2 images in the overlap panel.")
            )
            self._lbl_composite_status.setVisible(True)
            self._show_single()
            return

        n = len(self._overlap_panel.checked_docs())
        mode = self._overlap_panel.blend_mode()
        self._lbl_composite_status.setText(
            self.tr("Composite preview: {0} images · {1} blend").format(n, mode)
        )
        self._lbl_composite_status.setVisible(True)

        saved = self._snapshot_rect_state()
        self._set_preview_pixmap(composite)
        self._restore_rect_state(saved)

    def _show_single(self):
        """Restore single-image preview."""
        self._lbl_composite_status.setVisible(False)
        saved = self._snapshot_rect_state()
        self._set_preview_pixmap(self._preview01)
        self._restore_rect_state(saved)

    def _set_preview_pixmap(self, img01: np.ndarray):
        """Replace the scene pixmap without touching the crop rectangle."""
        q  = self._to_qimage(img01)
        pm = QPixmap.fromImage(q)

        if self._pix_item is not None:
            self._pix_item.setPixmap(pm)
        else:
            self._pix_item = QGraphicsPixmapItem(pm)
            self._pix_item.setZValue(-1)
            self.scene.addItem(self._pix_item)

        if self._fit_mode:
            self._apply_zoom_transform()

    # =========================================================================
    # Deferred fit / show
    # =========================================================================

    def _deferred_fit(self):
        if self._fit_mode:
            QTimer.singleShot(0, self._fit_view)

    def showEvent(self, ev):
        super().showEvent(ev)
        if not getattr(self, "_geom_restored", False):
            self._geom_restored = True
            def _after_restore():
                self._restore_window_geometry()
                self._deferred_fit()
            QTimer.singleShot(0, _after_restore)
            return
        self._deferred_fit()

    # =========================================================================
    # Image plumbing
    # =========================================================================

    def _quad_is_axis_aligned(self, pts: np.ndarray, tol: float = 1e-2) -> bool:
        if pts.shape != (4, 2):
            return False
        left_dx  = abs(pts[0,0] - pts[3,0])
        right_dx = abs(pts[1,0] - pts[2,0])
        top_dy   = abs(pts[0,1] - pts[1,1])
        bot_dy   = abs(pts[2,1] - pts[3,1])
        return (left_dx < tol and right_dx < tol and top_dy < tol and bot_dy < tol)

    def _int_bounds_from_quad(self, pts: np.ndarray, W: int, H: int) -> tuple[int,int,int,int] | None:
        if pts.size != 8:
            return None
        xs = pts[:,0]; ys = pts[:,1]
        x0 = int(np.floor(xs.min() + 1e-6))
        y0 = int(np.floor(ys.min() + 1e-6))
        x1 = int(np.ceil (xs.max() - 1e-6))
        y1 = int(np.ceil (ys.max() - 1e-6))
        x0 = max(0, min(W, x0)); x1 = max(0, min(W, x1))
        y0 = max(0, min(H, y0)); y1 = max(0, min(H, y1))
        if x1 <= x0 or y1 <= y0:
            return None
        return x0, x1, y0, y1

    def _img01_from_doc(self) -> np.ndarray:
        arr = np.asarray(self.doc.image)
        if arr.dtype.kind in "ui":
            arr = arr.astype(np.float32) / np.iinfo(self.doc.image.dtype).max
        else:
            arr = arr.astype(np.float32, copy=False)
        if arr.ndim == 3 and arr.shape[2] == 1:
            arr = arr[..., 0]
        return np.clip(arr, 0.0, 1.0)

    def _on_active_doc_changed(self, doc):
        if doc is None or getattr(doc, "image", None) is None:
            return
        self.doc = doc
        self._rect_item = None
        self._load_from_doc()
        # refresh panel doc list if open
        if self._overlap_panel.isVisible():
            self._overlap_panel.refresh_docs(self._collect_all_docs())

    def _load_from_doc(self):
        self._full01   = self._img01_from_doc()
        self._orig_h, self._orig_w = self._full01.shape[:2]
        self._preview01 = (
            self._full01 if not self._autostretch_on
            else histogram_style_autostretch(self._full01)
        )

        self.scene.clear()
        self._pix_item = None
        self._rect_item = None
        self._set_preview_pixmap(self._preview01)
        self._deferred_fit()
        self._set_dim_label_none()

        # if composite was active, rebuild it against the new primary image size
        if self._overlap_panel.is_composite_active():
            QTimer.singleShot(50, self._show_composite)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        if self._fit_mode:
            self._apply_zoom_transform()

    # =========================================================================
    # Selection dims label
    # =========================================================================

    def _set_dim_label_none(self):
        if hasattr(self, "lbl_dims"):
            self.lbl_dims.setText(self.tr("Selection: —"))

    def _update_dim_label_from_corners(self, corners_scene):
        if not hasattr(self, "lbl_dims") or not corners_scene or not self._pix_item:
            self._set_dim_label_none()
            return
        w_img, h_img = self._orig_w, self._orig_h
        src = np.array(
            [self._scene_to_img_pixels(p, w_img, h_img) for p in corners_scene],
            dtype=np.float32,
        )
        width  = float(np.linalg.norm(src[1] - src[0]))
        height = float(np.linalg.norm(src[3] - src[0]))
        self.lbl_dims.setText(
            self.tr("Selection: {0}×{1} px").format(int(round(height)), int(round(width)))
        )

    def _update_dim_label_from_rect_item(self):
        if not self._rect_item:
            self._set_dim_label_none()
            return
        corners = self._corners_scene()
        self._update_dim_label_from_corners(corners)

    @staticmethod
    def _to_qimage(img01: np.ndarray) -> QImage:
        if img01.ndim == 3 and img01.shape[2] == 1:
            img01 = img01[..., 0]
        if img01.ndim == 2:
            buf = np.ascontiguousarray((img01 * 255).astype(np.uint8))
            h, w = buf.shape
            return QImage(buf.tobytes(), w, h, buf.strides[0], QImage.Format.Format_Grayscale8)
        if img01.ndim == 3 and img01.shape[2] == 3:
            buf = np.ascontiguousarray((img01 * 255).astype(np.uint8))
            h, w, _ = buf.shape
            return QImage(buf.tobytes(), w, h, buf.strides[0], QImage.Format.Format_RGB888)
        raise ValueError(f"Unsupported image shape for preview: {img01.shape}")

    # =========================================================================
    # Aspect ratio
    # =========================================================================

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

    # =========================================================================
    # Drawing / interaction
    # =========================================================================

    def eventFilter(self, src, e):
        if src is self.view.viewport():
            if e.type() == QEvent.Type.Wheel:
                dy = e.pixelDelta().y()
                if dy != 0:
                    abs_dy = abs(dy)
                    ctrl_down = bool(e.modifiers() & Qt.KeyboardModifier.ControlModifier)
                    if abs_dy <= 3:   base_factor = 1.012 if ctrl_down else 1.010
                    elif abs_dy <= 10: base_factor = 1.025 if ctrl_down else 1.020
                    else:              base_factor = 1.040 if ctrl_down else 1.030
                    factor = base_factor if dy > 0 else 1.0 / base_factor
                else:
                    dy = e.angleDelta().y()
                    if dy == 0: return True
                    ctrl_down = bool(e.modifiers() & Qt.KeyboardModifier.ControlModifier)
                    step = 1.25 if ctrl_down else 1.15
                    factor = step if dy > 0 else 1.0 / step
                self._zoom_by(factor)
                e.accept()
                return True

            if e.type() in (QEvent.Type.MouseButtonPress, QEvent.Type.MouseMove, QEvent.Type.MouseButtonRelease):
                scene_pt = self.view.mapToScene(e.pos())

            if e.type() == QEvent.Type.MouseMove and self._rect_item is not None:
                self._update_dim_label_from_rect_item()

            if self._rect_item is None:
                if e.type() == QEvent.Type.MouseButtonPress and e.button() == Qt.MouseButton.LeftButton:
                    self._drawing = True; self._origin = scene_pt; return True

                if e.type() == QEvent.Type.MouseMove and self._drawing:
                    r = QRectF(self._origin, scene_pt).normalized()
                    r = self._apply_ar_to_rect(r, live=True, scene_pt=scene_pt)
                    r = self._clamp_rect_to_pixmap(r)
                    self._draw_live_rect(r)
                    corners = [r.topLeft(), r.topRight(), r.bottomRight(), r.bottomLeft()]
                    self._update_dim_label_from_corners(corners)
                    return True

                if e.type() == QEvent.Type.MouseButtonRelease and e.button() == Qt.MouseButton.LeftButton and self._drawing:
                    self._drawing = False
                    r = QRectF(self._origin, scene_pt).normalized()
                    r = self._apply_ar_to_rect(r, live=False, scene_pt=scene_pt)
                    r = self._clamp_rect_to_pixmap(r)
                    self._clear_live_rect()
                    self._rect_item = ResizableRotatableRectItem(r)
                    self._rect_item.setZValue(10)
                    self._rect_item.setBoundsSceneRect(self._bounds_scene_rect())
                    self._rect_item.setFixedAspectRatio(self._current_ar_value())
                    self.scene.addItem(self._rect_item)
                    CropDialogPro._prev_rect  = QRectF(r)
                    CropDialogPro._prev_angle = self._rect_item.rotation()
                    CropDialogPro._prev_pos   = self._rect_item.pos()
                    self._update_dim_label_from_rect_item()
                    return True

            return False
        return super().eventFilter(src, e)

    def _apply_zoom_transform(self):
        if not self._pix_item:
            return
        if self._fit_mode:
            rect = self._pix_item.mapRectToScene(self._pix_item.boundingRect())
            self.view.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
            self.view.fitInView(rect.adjusted(-1,-1,1,1), Qt.AspectRatioMode.KeepAspectRatio)
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
        newz = min(16.0, max(0.05, self._zoom * float(factor)))
        if abs(newz - self._zoom) < 1e-4: return
        self._zoom = newz
        self._apply_zoom_transform()

    # =========================================================================
    # Typed margins
    # =========================================================================

    def _update_margin_spin_ranges(self):
        h, w = int(self._orig_h), int(self._orig_w)
        self.sb_top.setRange(0, max(0, h))
        self.sb_bottom.setRange(0, max(0, h))
        self.sb_left.setRange(0, max(0, w))
        self.sb_right.setRange(0, max(0, w))

    def _apply_margin_inputs(self):
        t = int(self.sb_top.value());    r = int(self.sb_right.value())
        b = int(self.sb_bottom.value()); l = int(self.sb_left.value())
        self._set_rect_from_margins(t, r, b, l)

    def _set_rect_from_margins(self, top: int, right: int, bottom: int, left: int):
        w_img, h_img = float(self._orig_w), float(self._orig_h)
        left   = max(0, min(int(left),   int(w_img)))
        right  = max(0, min(int(right),  int(w_img)))
        top    = max(0, min(int(top),    int(h_img)))
        bottom = max(0, min(int(bottom), int(h_img)))
        x = float(left); y = float(top)
        w = max(1.0, w_img - (left + right))
        h = max(1.0, h_img - (top + bottom))
        r = QRectF(x, y, w, h)
        if self._rect_item is None:
            self._rect_item = ResizableRotatableRectItem(r)
            self._rect_item.setZValue(10)
            self._rect_item.setBoundsSceneRect(self._bounds_scene_rect())
            self.scene.addItem(self._rect_item)
        else:
            self._rect_item.setRotation(0.0)
            self._rect_item.setPos(QPointF(0, 0))
            self._rect_item.setRect(r)
        self._rect_item.setTransformOriginPoint(r.center())
        self._update_dim_label_from_rect_item()

    def _current_ar_value(self) -> Optional[float]:
        txt = self.cmb_ar.currentText()
        if txt == self.tr("Free"): return None
        if txt == self.tr("Original"): return self._orig_w / self._orig_h
        a, b = map(float, txt.split(":")); return a / b

    def _apply_ar_to_rect(self, r: QRectF, live: bool, scene_pt: QPointF) -> QRectF:
        ar = self._current_ar_value()
        if ar is None:
            return r
        w = r.width(); h = w / ar
        if scene_pt.y() < self._origin.y():
            r.setTop(r.bottom() - h)
        else:
            r.setBottom(r.top() + h)
        return r.normalized()

    def _draw_live_rect(self, r: QRectF):
        if hasattr(self, "_live_rect") and self._live_rect:
            self.scene.removeItem(self._live_rect)
        if getattr(self, "_live_cross", None):
            self.scene.removeItem(self._live_cross)
            self._live_cross = None
        pen = QPen(QColor(0,255,0), 2, Qt.PenStyle.DashLine); pen.setCosmetic(True)
        self._live_rect = self.scene.addRect(r, pen)
        s = 0.12 * min(r.width(), r.height())
        s = max(10.0, min(40.0, float(s)))
        c = r.center()
        p = QPainterPath()
        p.moveTo(c.x()-s, c.y()); p.lineTo(c.x()+s, c.y())
        p.moveTo(c.x(), c.y()-s); p.lineTo(c.x(), c.y()+s)
        p.moveTo(c.x()-s, c.y()-s); p.lineTo(c.x()+s, c.y()+s)
        p.moveTo(c.x()-s, c.y()+s); p.lineTo(c.x()+s, c.y()-s)
        self._live_cross = self.scene.addPath(p, QPen(QColor(0,255,0), 1))

    def _clear_live_rect(self):
        if hasattr(self, "_live_rect") and self._live_rect:
            self.scene.removeItem(self._live_rect); self._live_rect = None
        if getattr(self, "_live_cross", None):
            self.scene.removeItem(self._live_cross); self._live_cross = None

    def _pixmap_scene_rect(self) -> QRectF | None:
        if not self._pix_item:
            return None
        return self._pix_item.mapRectToScene(self._pix_item.boundingRect())

    def _clamp_rect_to_pixmap(self, r: QRectF) -> QRectF:
        bounds = self._pixmap_scene_rect()
        if bounds is None:
            return r.normalized()
        rr = r.normalized().intersected(bounds)
        if rr.isNull() or rr.width() <= 1e-6 or rr.height() <= 1e-6:
            x = min(max(r.center().x(), bounds.left()), bounds.right())
            y = min(max(r.center().y(), bounds.top()),  bounds.bottom())
            rr = QRectF(x, y, 1.0, 1.0)
        return rr.normalized()

    def _bounds_scene_rect(self) -> QRectF | None:
        if not self._pix_item:
            return None
        return self._pix_item.mapRectToScene(self._pix_item.boundingRect())

    # =========================================================================
    # Preview toggles
    # =========================================================================

    def _toggle_autostretch(self):
        self._autostretch_on = not self._autostretch_on
        self._preview01 = (
            self._full01 if not self._autostretch_on
            else histogram_style_autostretch(self._full01)
        )
        saved = self._snapshot_rect_state()
        if self._overlap_panel.is_composite_active():
            self._show_composite()
        else:
            self._set_preview_pixmap(self._preview01)
        self._restore_rect_state(saved)
        self._deferred_fit()

    def _snapshot_rect_state(self):
        if not self._rect_item: return None
        return (QRectF(self._rect_item.rect()),
                float(self._rect_item.rotation()),
                QPointF(self._rect_item.pos()))

    def _restore_rect_state(self, state):
        if not state: return
        r, ang, pos = state
        # Remove old rect from scene only if it's still tracked
        if self._rect_item is not None:
            try:
                self.scene.removeItem(self._rect_item)
            except Exception:
                pass
        self._rect_item = ResizableRotatableRectItem(r)
        self._rect_item.setZValue(10)
        self._rect_item.setBoundsSceneRect(self._bounds_scene_rect())
        self._rect_item.setFixedAspectRatio(self._current_ar_value())
        self._rect_item.setRotation(ang)
        self._rect_item.setPos(pos)
        self._rect_item.setTransformOriginPoint(r.center())
        self.scene.addItem(self._rect_item)
        self._update_dim_label_from_rect_item()

    def _load_previous(self):
        if CropDialogPro._prev_rect is None:
            QMessageBox.information(self, self.tr("No Previous"), self.tr("No previous crop stored."))
            return
        if self._rect_item:
            self.scene.removeItem(self._rect_item)
        r = QRectF(CropDialogPro._prev_rect)
        self._rect_item = ResizableRotatableRectItem(r)
        self._rect_item.setZValue(10)
        self._rect_item.setBoundsSceneRect(self._bounds_scene_rect())
        self._rect_item.setFixedAspectRatio(self._current_ar_value())
        self._rect_item.setRotation(CropDialogPro._prev_angle)
        self._rect_item.setPos(CropDialogPro._prev_pos)
        self._rect_item.setTransformOriginPoint(r.center())
        self.scene.addItem(self._rect_item)
        self._update_dim_label_from_rect_item()

    # =========================================================================
    # Apply
    # =========================================================================

    def _corners_scene(self):
        rl = self._rect_item.rect()
        loc = [rl.topLeft(), rl.topRight(), rl.bottomRight(), rl.bottomLeft()]
        return [self._rect_item.mapToScene(p) for p in loc]

    def _scene_to_img_pixels(self, pt_scene: QPointF, w_img: int, h_img: int):
        pm = self._pix_item.pixmap()
        sx, sy = w_img / pm.width(), h_img / pm.height()
        return np.array([pt_scene.x() * sx, pt_scene.y() * sy], dtype=np.float32)

    def _apply_one(self):
        try:
            self._save_window_geometry()
        except Exception:
            pass
        if not self._rect_item:
            QMessageBox.warning(self, self.tr("No Selection"), self.tr("Draw & finalize a crop first."))
            return

        corners = self._corners_scene()
        w_img, h_img = self._orig_w, self._orig_h
        src = np.array([self._scene_to_img_pixels(p, w_img, h_img) for p in corners], dtype=np.float32)

        width  = np.linalg.norm(src[1] - src[0])
        height = np.linalg.norm(src[3] - src[0])

        H_img, W_img = self._orig_h, self._orig_w
        axis_aligned = self._quad_is_axis_aligned(src)

        if axis_aligned:
            bounds = self._int_bounds_from_quad(src, W_img, H_img)
            if bounds is None:
                QMessageBox.critical(self, self.tr("Apply failed"), self.tr("Invalid crop bounds."))
                return
            x0, x1, y0, y1 = bounds
            out = self._full01[y0:y1, x0:x1].copy()
            M = np.array([[1.0, 0.0, -float(x0)],
                          [0.0, 1.0, -float(y0)],
                          [0.0, 0.0,  1.0]], dtype=np.float32)
            w_out, h_out = (x1 - x0), (y1 - y0)
        else:
            dst = np.array([[0,0],[width,0],[width,height],[0,height]], dtype=np.float32)
            M = cv2.getPerspectiveTransform(src, dst)
            w_out = int(round(width)); h_out = int(round(height))
            if w_out <= 0 or h_out <= 0:
                QMessageBox.critical(self, self.tr("Apply failed"), self.tr("Invalid crop size."))
                return
            out = cv2.warpPerspective(self._full01, M, (w_out, h_out), flags=cv2.INTER_LANCZOS4)

        new_meta = dict(self.doc.metadata or {})
        try:
            if update_wcs_after_crop is not None:
                new_meta = update_wcs_after_crop(new_meta, M_src_to_dst=M, out_w=w_out, out_h=h_out)
        except Exception:
            pass

        CropDialogPro._prev_rect  = QRectF(self._rect_item.rect())
        CropDialogPro._prev_angle = float(self._rect_item.rotation())
        CropDialogPro._prev_pos   = QPointF(self._rect_item.pos())

        try:
            self.doc.apply_edit(out.copy(), metadata={**new_meta, "step_name": "Crop"}, step_name="Crop")
            self._maybe_notify_wcs_update(new_meta)
            self.crop_applied.emit(out)
            self.close()
        except Exception as e:
            QMessageBox.critical(self, self.tr("Apply failed"), str(e))

    def _apply_batch(self):
        try:
            self._save_window_geometry()
        except Exception:
            pass
        if not self._rect_item:
            QMessageBox.warning(self, self.tr("No Selection"), self.tr("Draw & finalize a crop first."))
            return

        # ── decide which docs to crop ─────────────────────────────────────
        # When composite is active, use exactly the checked images.
        # Otherwise fall back to the original "all open" behaviour.
        if self._overlap_panel.is_composite_active():
            docs = self._overlap_panel.checked_docs()
            if not docs:
                QMessageBox.information(
                    self, self.tr("No Images Selected"),
                    self.tr("No images are checked in the overlap panel.")
                )
                return
            source_label = self.tr(
                "Apply this crop to {0} checked image(s) from the overlap panel?"
            ).format(len(docs))
        else:
            # original behaviour: all open MDI docs
            win = self.parent()
            subs = getattr(win, "mdi", None).subWindowList() if hasattr(win, "mdi") else []
            docs = []
            for sw in subs:
                vw = sw.widget()
                d = getattr(vw, "document", None)
                if d is not None:
                    docs.append(d)
            if not docs:
                QMessageBox.information(self, self.tr("No Images"), self.tr("No open images to crop."))
                return
            source_label = self.tr("Apply this crop to {0} open image(s)?").format(len(docs))

        # Normalise crop polygon to THIS (primary) image size
        corners  = self._corners_scene()
        src_this = np.array(
            [self._scene_to_img_pixels(p, self._orig_w, self._orig_h) for p in corners],
            dtype=np.float32,
        )
        norm = src_this / np.array([self._orig_w, self._orig_h], dtype=np.float32)

        ok = QMessageBox.question(
            self, self.tr("Confirm Batch"), source_label,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if ok != QMessageBox.StandardButton.Yes:
            return

        last_cropped = None
        for d in docs:
            img = np.asarray(d.image)
            if img.dtype.kind in "ui":
                src01 = img.astype(np.float32) / np.iinfo(d.image.dtype).max
            else:
                src01 = img.astype(np.float32, copy=False)

            h, w    = src01.shape[:2]
            src_pts = norm * np.array([w, h], dtype=np.float32)

            axis_aligned = self._quad_is_axis_aligned(src_pts)
            if axis_aligned:
                b = self._int_bounds_from_quad(src_pts, w, h)
                if b is None: continue
                x0, x1, y0, y1 = b
                cropped = src01[y0:y1, x0:x1].copy()
                w_out, h_out = (x1 - x0), (y1 - y0)
                M = np.array([[1.0, 0.0, -float(x0)],
                              [0.0, 1.0, -float(y0)],
                              [0.0, 0.0,  1.0]], dtype=np.float32)
            else:
                w_out = int(round(np.linalg.norm(src_pts[1] - src_pts[0])))
                h_out = int(round(np.linalg.norm(src_pts[3] - src_pts[0])))
                if w_out <= 0 or h_out <= 0: continue
                dst = np.array([[0,0],[w_out,0],[w_out,h_out],[0,h_out]], dtype=np.float32)
                M = cv2.getPerspectiveTransform(src_pts.astype(np.float32), dst)
                cropped = cv2.warpPerspective(src01, M, (w_out, h_out), flags=cv2.INTER_LANCZOS4)

            meta_this = dict(d.metadata or {})
            try:
                if update_wcs_after_crop is not None:
                    meta_this = update_wcs_after_crop(meta_this, M_src_to_dst=M, out_w=w_out, out_h=h_out)
            except Exception:
                pass
            try:
                d.apply_edit(cropped.copy(), metadata={**meta_this, "step_name": "Crop"}, step_name="Crop")
                last_cropped = cropped
            except Exception:
                pass

        QMessageBox.information(
            self, self.tr("Batch Crop"),
            self.tr("Applied crop to {0} image(s). Any Astrometric Solutions have been updated.").format(len(docs))
        )
        if last_cropped is not None:
            self.crop_applied.emit(last_cropped)
        self.close()

    # =========================================================================
    # WCS notification
    # =========================================================================

    def _maybe_notify_wcs_update(self, meta: dict, batch_note: str | None = None):
        dbg = (meta or {}).get("__wcs_debug__")
        if not dbg:
            return
        try:
            before = dbg.get("before", {}); after = dbg.get("after", {}); fit = dbg.get("fit", {})
            b_ra, b_dec = before.get("crval_deg", (float("nan"), float("nan")))
            a_ra, a_dec = after.get("crval_deg",  (float("nan"), float("nan")))
            rms  = fit.get("rms_arcsec", float("nan"))
            p95  = fit.get("p95_arcsec", float("nan"))
            sip  = after.get("sip_degree")
            size = after.get("size")
            sip_txt  = f"TAN-SIP (deg={sip})" if sip is not None else "TAN"
            size_txt = f"{size[0]}×{size[1]}" if size else "?"
            extra = f"\n{batch_note}" if batch_note else ""
            msg = (
                self.tr("Astrometric solution updated ✔️\n\n") +
                self.tr("Model: {0}   Image: {1}\n").format(sip_txt, size_txt) +
                self.tr("CRVAL: ({0:.6f}, {1:.6f}) → ({2:.6f}, {3:.6f})\n").format(b_ra, b_dec, a_ra, a_dec) +
                self.tr("Fit residuals: RMS {0:.3f}\"  (p95 {1:.3f}\")").format(rms, p95) +
                f"{extra}"
            )
            QMessageBox.information(self, self.tr("WCS Updated"), msg)
        except Exception:
            pass

    # =========================================================================
    # Cleanup / geometry
    # =========================================================================

    def _cleanup_connections(self):
        try:
            if self._follow_conn and hasattr(self._main, "currentDocumentChanged"):
                self._main.currentDocumentChanged.disconnect(self._on_active_doc_changed)
        except Exception:
            pass
        self._follow_conn = False

    def _restore_window_geometry(self):
        try:
            s = QSettings()
            g = s.value("crop/window_geometry", None)
            if g is not None:
                self.restoreGeometry(g)
        except Exception:
            pass

    def _save_window_geometry(self):
        try:
            s = QSettings()
            s.setValue("crop/window_geometry", self.saveGeometry())
        except Exception:
            pass

    def closeEvent(self, ev):
        try:
            self._save_window_geometry()
        except Exception:
            pass
        self._cleanup_connections()
        super().closeEvent(ev)