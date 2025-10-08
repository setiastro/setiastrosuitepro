# pro/signature_insert.py
from __future__ import annotations
import math
import numpy as np

from PyQt6.QtCore import Qt, QTimer, QRectF, QPointF
from PyQt6.QtGui import (
    QImage, QPixmap, QPainter, QColor, QPen, QTransform, QIcon
)
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox, QLabel, QPushButton,
    QSlider, QCheckBox, QColorDialog, QComboBox, QFileDialog, QInputDialog, QMenu,
    QMessageBox, QWidget, QGraphicsView, QGraphicsScene, QGraphicsItem,
    QGraphicsPixmapItem, QGraphicsEllipseItem, QGraphicsRectItem, QSpinBox
)


def _np_to_qimage_rgb(a: np.ndarray) -> QImage:
    a = np.asarray(a, dtype=np.float32)
    a = np.clip(a, 0.0, 1.0)
    if a.ndim == 2:
        a = a[..., None].repeat(3, axis=2)
    if a.shape[2] != 3:
        a = a[:, :, :3]
    u8 = (a * 255.0).astype(np.uint8)
    h, w = u8.shape[:2]
    return QImage(u8.data, w, h, w*3, QImage.Format.Format_RGB888).copy()

def _qimage_to_np_rgba(img: QImage) -> np.ndarray:
    q = img.convertToFormat(QImage.Format.Format_RGBA8888)
    w, h = q.width(), q.height()
    ptr = q.bits(); ptr.setsize(h * q.bytesPerLine())
    buf = np.frombuffer(ptr, dtype=np.uint8).reshape((h, q.bytesPerLine()))
    arr = buf[:, :w*4].reshape((h, w, 4)).astype(np.float32) / 255.0
    return arr

def _anchor_point(base_w: int, base_h: int, ins_w: int, ins_h: int,
                  key: str, mx: int, my: int) -> QPointF:
    # compute top-left anchor by key + margins
    left   = 0 + mx
    right  = base_w - ins_w - mx
    top    = 0 + my
    bottom = base_h - ins_h - my
    center_x = (base_w - ins_w) / 2
    center_y = (base_h - ins_h) / 2
    table = {
        "top_left":      QPointF(left, top),
        "top_center":    QPointF(center_x, top),
        "top_right":     QPointF(right, top),
        "middle_left":   QPointF(left, center_y),
        "center":        QPointF(center_x, center_y),
        "middle_right":  QPointF(right, center_y),
        "bottom_left":   QPointF(left, bottom),
        "bottom_center": QPointF(center_x, bottom),
        "bottom_right":  QPointF(right, bottom),
    }
    return table.get(key, QPointF(right, bottom))  # default BR

def apply_signature_preset_to_doc(doc, preset: dict) -> np.ndarray:
    """
    Headless apply of signature/insert using a preset.
    Preset fields (all optional except file_path):
      - file_path: str (PNG recommended; alpha preserved)
      - position:  str in {"top_left","top_center","top_right",
                           "middle_left","center","middle_right",
                           "bottom_left","bottom_center","bottom_right"}
      - margin_x:  int pixels (default 20)
      - margin_y:  int pixels (default 20)
      - scale:     percent (default 100)
      - rotation:  degrees (default 0)
      - opacity:   percent (default 100)
    Returns: RGB float32 image in [0,1]
    """
    fp = str(preset.get("file_path", "")).strip()
    if not fp:
        raise ValueError("Preset missing 'file_path'.")

    # base → RGB
    base = np.asarray(getattr(doc, "image", None), dtype=np.float32)
    if base is None:
        raise RuntimeError("Document has no image.")
    if base.ndim == 2:
        base_rgb = np.repeat(base[:, :, None], 3, axis=2)
    elif base.ndim == 3 and base.shape[2] == 1:
        base_rgb = np.repeat(base, 3, axis=2)
    else:
        base_rgb = base[:, :, :3]
    base_rgb = np.clip(base_rgb, 0, 1)

    # canvas (ARGB32 so we can keep alpha while painting)
    canvas = QImage(base_rgb.shape[1], base_rgb.shape[0], QImage.Format.Format_ARGB32)
    canvas.fill(Qt.GlobalColor.transparent)
    p = QPainter(canvas)
    # draw base first (opaque)
    p.drawImage(QPointF(0, 0), _np_to_qimage_rgb(base_rgb))

    # load insert (alpha preserved)
    ins_img = QImage(fp)
    if ins_img.isNull():
        p.end()
        raise ValueError(f"Could not load insert image: {fp}")

    # parameters
    pos_key = str(preset.get("position", "bottom_right"))
    mx = int(preset.get("margin_x", 20))
    my = int(preset.get("margin_y", 20))
    scale = float(preset.get("scale", 100)) / 100.0
    rotation = float(preset.get("rotation", 0.0))
    opacity = max(0.0, min(1.0, float(preset.get("opacity", 100)) / 100.0))

    # transform: scale + rotate around center, then translate to anchor
    iw, ih = ins_img.width(), ins_img.height()
    aw = max(1, int(round(iw * scale)))
    ah = max(1, int(round(ih * scale)))

    # NOTE: we can let QPainter scale it by world transform (keeps alpha)
    # Compute anchor for the post-transform bounding rect.
    # For rotation, the item’s visual bbox changes; the usual UX expectation is:
    # "put the visual center on the anchor then offset by half of its size".
    # We do: transform about center, then translate so the *visual* top-left hits the anchor.
    t = QTransform()
    t.translate(aw/2, ah/2)
    t.rotate(rotation)
    t.scale(scale, scale)  # scale first or last works since we rotate around center
    t.translate(-iw/2, -ih/2)

    # Find visual bbox of transformed image to compute margins correctly
    transformed_rect = t.mapRect(QRectF(0, 0, iw, ih))
    vis_w, vis_h = transformed_rect.width(), transformed_rect.height()

    anchor = _anchor_point(base_rgb.shape[1], base_rgb.shape[0], int(round(vis_w)), int(round(vis_h)), pos_key, mx, my)

    # Now shift so that the transformed visual top-left lands at anchor
    t2 = QTransform(t)
    t2.translate(anchor.x() - transformed_rect.left(), anchor.y() - transformed_rect.top())

    p.setOpacity(opacity)
    p.setWorldTransform(t2, combine=False)
    p.drawImage(QPointF(0, 0), ins_img)
    p.end()

    # back to numpy (drop alpha → RGB)
    out_rgba = _qimage_to_np_rgba(canvas)
    out_rgb = out_rgba[:, :, :3]
    out_rgb = np.clip(out_rgb, 0.0, 1.0).astype(np.float32, copy=False)
    return out_rgb

# --------------------------- Graphics helpers ---------------------------

class TransformHandle(QGraphicsEllipseItem):
    """
    A small circular handle that sits on the top-right of the pixmap's local
    bounds and allows scale+rotation by dragging.
    """
    def __init__(self, parent_item: QGraphicsPixmapItem, scene: QGraphicsScene):
        super().__init__(-5, -5, 10, 10)
        self.parent_item = parent_item
        self.scene = scene

        self.setBrush(QColor("blue"))
        self.setPen(QPen(Qt.PenStyle.SolidLine))
        self.setCursor(Qt.CursorShape.SizeAllCursor)
        self.setZValue(2)

        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable |
            QGraphicsItem.GraphicsItemFlag.ItemIsFocusable |
            QGraphicsItem.GraphicsItemFlag.ItemIgnoresParentOpacity |
            QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
        )
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)
        self.setAcceptHoverEvents(True)

        self.initial_distance = None
        self.initial_angle = None
        self.initial_scale = parent_item.scale()

        scene.addItem(self)
        self.update_position()

    def update_position(self):
        corner = self.parent_item.boundingRect().topRight()
        scene_corner = self.parent_item.mapToScene(corner)
        self.setPos(scene_corner)

    def mousePressEvent(self, e):
        center = self.parent_item.mapToScene(self.parent_item.boundingRect().center())
        delta = self.scenePos() - center
        self.initial_distance = math.hypot(delta.x(), delta.y())
        self.initial_angle = math.degrees(math.atan2(delta.y(), delta.x()))
        self.initial_scale = self.parent_item.scale()
        e.accept()

    def mouseMoveEvent(self, e):
        center = self.parent_item.mapToScene(self.parent_item.boundingRect().center())
        new_pos = self.mapToScene(e.pos())
        delta = new_pos - center
        dist = math.hypot(delta.x(), delta.y())
        ang  = math.degrees(math.atan2(delta.y(), delta.x()))

        # scale
        s = (dist / self.initial_distance) if self.initial_distance else 1.0
        self.parent_item.setScale(max(0.05, self.initial_scale * s))

        # rotate
        self.parent_item.setRotation(ang - self.initial_angle)
        self.update_position()
        e.accept()

    def mouseReleaseEvent(self, e):
        self.initial_distance = None
        self.initial_angle = None
        self.initial_scale = self.parent_item.scale()
        e.accept()


class InsertView(QGraphicsView):
    """Pannable view + Ctrl+wheel zoom, with a right-click menu on inserts."""
    def __init__(self, scene: QGraphicsScene, owner: "SignatureInsertDialogPro"):
        super().__init__(scene)
        self.owner = owner
        self.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.zoom_factor = 1.0
        self.min_zoom, self.max_zoom = 0.10, 10.0

    # --- zoom/pan ---
    def wheelEvent(self, e):
        if e.modifiers() & Qt.KeyboardModifier.ControlModifier:
            step = 1.15 if e.angleDelta().y() > 0 else 1/1.15
            self.set_zoom(self.zoom_factor * step)
            e.accept()
            return
        super().wheelEvent(e)

    def set_zoom(self, z):
        z = max(self.min_zoom, min(self.max_zoom, z))
        self.zoom_factor = z
        self.setTransform(QTransform().scale(z, z))

    def zoom_in(self):  self.set_zoom(self.zoom_factor * 1.15)
    def zoom_out(self): self.set_zoom(self.zoom_factor / 1.15)
    def fit_to_view(self):
        r = self.scene().itemsBoundingRect()
        if r.isEmpty(): return
        self.fitInView(r, Qt.AspectRatioMode.KeepAspectRatio)
        self.zoom_factor = 1.0  # logical reset

    # --- context menu to snap inserts ---
    def contextMenuEvent(self, e):
        scene_pos = self.mapToScene(e.pos())
        item = self.scene().itemAt(scene_pos, self.transform())

        # If user clicked the child rect, use the parent pixmap
        if isinstance(item, QGraphicsRectItem) and item.parentItem() in self.owner.inserts:
            item = item.parentItem()

        if isinstance(item, QGraphicsPixmapItem) and item in self.owner.inserts:
            m = QMenu(self)
            pos = {
                "Top-Left":"top_left", "Top-Center":"top_center", "Top-Right":"top_right",
                "Middle-Left":"middle_left","Center":"center","Middle-Right":"middle_right",
                "Bottom-Left":"bottom_left","Bottom-Center":"bottom_center","Bottom-Right":"bottom_right"
            }
            for label, key in pos.items():
                m.addAction(label, lambda k=key, it=item: self.owner.send_insert_to_position(it, k))
            m.exec(e.globalPos())
        else:
            super().contextMenuEvent(e)


# --------------------------- Main dialog ---------------------------

class SignatureInsertDialogPro(QDialog):
    """
    Add one or more overlays (“signatures/inserts”) on top of the active doc,
    transform them interactively, then bake into the doc.
    """
    def __init__(self, parent, doc, icon: QIcon | None = None):
        super().__init__(parent)
        self.setWindowTitle("Signature / Insert")
        if icon:
            try: self.setWindowIcon(icon)
            except Exception: pass

        self.doc = doc
        self.scene = QGraphicsScene(self)
        self.view  = InsertView(self.scene, self)

        self.inserts: list[QGraphicsPixmapItem] = []
        self.bounding_boxes: list[QGraphicsRectItem] = []
        self.bounding_boxes_enabled = True
        self.bounding_box_pen = QPen(QColor("red"), 2, Qt.PenStyle.DashLine)

        # Handle sync timer (keeps the handle parked on the item corner)
        self._timer = QTimer(self); self._timer.timeout.connect(self._sync_handles); self._timer.start(16)

        self._build_ui()
        self._update_base_image()
        self.resize(1000, 680)

    # -------- UI ----------
    def _build_ui(self):
        root = QHBoxLayout(self)

        # ---- LEFT COLUMN ------------------------------------------------------
        col = QVBoxLayout()

        # Alpha hint (always visible – simple, clear)
        alpha_hint = QLabel("Tip: Transparent signatures — use “Load from File” to preserve PNG alpha. "
                            "Loading from View uses RGB (no alpha).")
        alpha_hint.setStyleSheet("color:#e0b000;")
        alpha_hint.setWordWrap(True)
        col.addWidget(alpha_hint)

        # Load controls
        row_load = QHBoxLayout()
        b_from_view = QPushButton("Load Insert from View…"); b_from_view.clicked.connect(self._load_from_view)
        b_from_file = QPushButton("Load Insert from File…"); b_from_file.clicked.connect(self._load_from_file)
        row_load.addWidget(b_from_view); row_load.addWidget(b_from_file)
        col.addLayout(row_load)

        # Transform group
        grp = QGroupBox("Transform")
        g = QGridLayout(grp)
        b_rot = QPushButton("Rotate +90°"); b_rot.clicked.connect(self._rotate_selected)
        g.addWidget(b_rot, 0, 0, 1, 2)

        g.addWidget(QLabel("Scale (%)"), 1, 0)
        self.sl_scale = QSlider(Qt.Orientation.Horizontal); self.sl_scale.setRange(10, 400); self.sl_scale.setValue(100)
        self.sl_scale.valueChanged.connect(self._scale_selected)
        g.addWidget(self.sl_scale, 1, 1)

        g.addWidget(QLabel("Opacity (%)"), 2, 0)
        self.sl_opacity = QSlider(Qt.Orientation.Horizontal); self.sl_opacity.setRange(0, 100); self.sl_opacity.setValue(100)
        self.sl_opacity.valueChanged.connect(self._opacity_selected)
        g.addWidget(self.sl_opacity, 2, 1)
        col.addWidget(grp)

        # Bounding boxes
        self.cb_draw = QCheckBox("Draw Bounding Box"); self.cb_draw.setChecked(True); self.cb_draw.stateChanged.connect(self._toggle_boxes)
        col.addWidget(self.cb_draw)

        grp_box = QGroupBox("Bounding Box Style")
        gb = QGridLayout(grp_box)
        self.b_color = QPushButton("Color…"); self.b_color.clicked.connect(self._pick_box_color)
        self.sl_thick = QSlider(Qt.Orientation.Horizontal); self.sl_thick.setRange(1, 10); self.sl_thick.setValue(2); self.sl_thick.valueChanged.connect(self._update_box_pen)
        self.cmb_style = QComboBox(); self.cmb_style.addItems(["Solid","Dash","Dot","DashDot","DashDotDot"]); self.cmb_style.currentIndexChanged.connect(self._update_box_pen)
        gb.addWidget(self.b_color, 0, 0, 1, 2)
        gb.addWidget(QLabel("Thickness"), 1, 0); gb.addWidget(self.sl_thick, 1, 1)
        gb.addWidget(QLabel("Style"), 2, 0); gb.addWidget(self.cmb_style, 2, 1)
        col.addWidget(grp_box)

        # --- Snap with margins -------------------------------------------------
        snap_grp = QGroupBox("Send to position")
        sg = QGridLayout(snap_grp)

        # margins
        sg.addWidget(QLabel("Margin X (px)"), 0, 0)
        self.sp_margin_x = QSpinBox(); self.sp_margin_x.setRange(0, 5000); self.sp_margin_x.setValue(20)
        sg.addWidget(self.sp_margin_x, 0, 1)

        sg.addWidget(QLabel("Margin Y (px)"), 0, 2)
        self.sp_margin_y = QSpinBox(); self.sp_margin_y.setRange(0, 5000); self.sp_margin_y.setValue(20)
        sg.addWidget(self.sp_margin_y, 0, 3)

        # 3x3 snap buttons
        def s(key):  # helper to create buttons
            btn = QPushButton(key.replace('_', ' ').title())
            btn.setMinimumWidth(105)
            btn.clicked.connect(lambda _, k=key: self._send_selected(k))
            return btn

        sg.addWidget(s("top_left"),      1, 0)
        sg.addWidget(s("top_center"),    1, 1)
        sg.addWidget(s("top_right"),     1, 2)
        sg.addWidget(s("middle_left"),   2, 0)
        sg.addWidget(s("center"),        2, 1)
        sg.addWidget(s("middle_right"),  2, 2)
        sg.addWidget(s("bottom_left"),   3, 0)
        sg.addWidget(s("bottom_center"), 3, 1)
        sg.addWidget(s("bottom_right"),  3, 2)
        col.addWidget(snap_grp)

        # Zoom
        row_zoom = QHBoxLayout()
        b_zo  = QPushButton("–"); b_zo.clicked.connect(self.view.zoom_out)
        b_zi  = QPushButton("+"); b_zi.clicked.connect(self.view.zoom_in)
        b_fit = QPushButton("Fit"); b_fit.clicked.connect(self.view.fit_to_view)
        row_zoom.addWidget(QLabel("Zoom (Ctrl+Wheel):")); row_zoom.addWidget(b_zo); row_zoom.addWidget(b_zi); row_zoom.addWidget(b_fit); row_zoom.addStretch(1)
        col.addLayout(row_zoom)

        col.addStretch(1)

        # Commit/Clear
        row_commit = QHBoxLayout()
        b_affix = QPushButton("Affix Inserts"); b_affix.clicked.connect(self._affix_inserts)
        b_clear = QPushButton("Clear All");     b_clear.clicked.connect(self._clear_inserts)
        row_commit.addWidget(b_affix); row_commit.addWidget(b_clear); row_commit.addStretch(1)
        col.addLayout(row_commit)

        left = QWidget(); left.setLayout(col)
        root.addWidget(left, 0)
        root.addWidget(self.view, 1)

    # -------- Scene / items ----------
    def _sync_handles(self):
        for it in self.scene.items():
            if isinstance(it, TransformHandle):
                it.update_position()

    def _update_base_image(self):
        self.scene.clear()
        arr = np.asarray(self.doc.image, dtype=np.float32)
        if arr is None: return
        qimg = self._numpy_to_qimage(arr)
        bg = QGraphicsPixmapItem(QPixmap.fromImage(qimg))
        bg.setZValue(0)
        self.scene.addItem(bg)

    def _load_from_file(self):
        fp, _ = QFileDialog.getOpenFileName(self, "Select Insert Image", "", "Images (*.png *.jpg *.jpeg *.tif *.tiff)")
        if not fp: return
        pm = QPixmap(fp)
        if pm.isNull():
            QMessageBox.warning(self, "Load Failed", "Could not load image.")
            return
        self._add_insert(pm)

    def _load_from_view(self):
        # list all open views via a helper the app already uses elsewhere (fallback to active only)
        candidates = []
        if hasattr(self.parent(), "_subwindow_docs"):
            for title, d in self.parent()._subwindow_docs():
                if d is self.doc:  # skip self
                    continue
                if getattr(d, "image", None) is not None:
                    candidates.append((title, d))
        if not candidates:
            QMessageBox.information(self, "Insert", "No other image windows found.")
            return

        names = [t for (t, _) in candidates]
        choice, ok = QInputDialog.getItem(self, "Load Insert from View", "Choose:", names, 0, False)
        if not ok: return
        d = candidates[names.index(choice)][1]
        pm = QPixmap.fromImage(self._numpy_to_qimage(np.asarray(d.image, dtype=np.float32)))
        self._add_insert(pm)

    def _add_insert(self, pm: QPixmap):
        it = QGraphicsPixmapItem(pm)
        it.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable |
            QGraphicsItem.GraphicsItemFlag.ItemIsSelectable |
            QGraphicsItem.GraphicsItemFlag.ItemIsFocusable |
            QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        it.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        it.setTransformOriginPoint(it.boundingRect().center())
        it.setZValue(1)
        it.setOpacity(1.0)
        self.scene.addItem(it)
        self.inserts.append(it)
        TransformHandle(it, self.scene)

        if self.bounding_boxes_enabled:
            rect = QGraphicsRectItem(it.boundingRect())
            rect.setParentItem(it)
            rect.setPen(self.bounding_box_pen)
            rect.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            rect.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresParentOpacity, True)
            rect.setZValue(it.zValue() + 0.1)
            self.scene.addItem(rect)
            self.bounding_boxes.append(rect)

    def _send_selected(self, key: str):
        for it in self.inserts:
            if it.isSelected():
                self.send_insert_to_position(it, key)

    # -------- Commands ----------
    def _rotate_selected(self):
        for it in self.inserts:
            if it.isSelected():
                it.setRotation(it.rotation() + 90)

    def _scale_selected(self, val):
        s = val / 100.0
        for it in self.inserts:
            if it.isSelected():
                it.setScale(s)
                # keep the child rect matching the pixmap's local bounds
                for box in self.bounding_boxes:
                    if box.parentItem() == it:
                        box.setRect(it.boundingRect())

    def _opacity_selected(self, val):
        o = val / 100.0
        for it in self.inserts:
            if it.isSelected():
                it.setOpacity(o)

    def _toggle_boxes(self, state):
        self.bounding_boxes_enabled = bool(state)
        for r in self.bounding_boxes:
            r.setVisible(self.bounding_boxes_enabled)

    def _pick_box_color(self):
        c = QColorDialog.getColor()
        if c.isValid():
            self.bounding_box_pen.setColor(c)
            self._refresh_all_boxes()

    def _update_box_pen(self):
        style_map = {
            "Solid": Qt.PenStyle.SolidLine,
            "Dash": Qt.PenStyle.DashLine,
            "Dot": Qt.PenStyle.DotLine,
            "DashDot": Qt.PenStyle.DashDotLine,
            "DashDotDot": Qt.PenStyle.DashDotDotLine
        }
        self.bounding_box_pen.setWidth(self.sl_thick.value())
        self.bounding_box_pen.setStyle(style_map[self.cmb_style.currentText()])
        self._refresh_all_boxes()

    def _refresh_all_boxes(self):
        for r in self.bounding_boxes:
            r.setPen(self.bounding_box_pen)

    # snap an insert to one of 9 standard positions inside the base image
    def send_insert_to_position(self, item: QGraphicsPixmapItem, key: str):
        base = next((i for i in self.scene.items()
                    if isinstance(i, QGraphicsPixmapItem) and i.zValue() == 0), None)
        if not base: return

        mx = self.sp_margin_x.value()
        my = self.sp_margin_y.value()

        br = base.boundingRect()
        ir = item.boundingRect()
        size = ir.size()

        table = {
            "top_left":      QPointF(br.left() + mx,  br.top() + my),
            "top_center":    QPointF(br.center().x() - size.width()/2, br.top() + my),
            "top_right":     QPointF(br.right() - size.width() - mx, br.top() + my),
            "middle_left":   QPointF(br.left() + mx,  br.center().y() - size.height()/2),
            "center":        QPointF(br.center().x() - size.width()/2, br.center().y() - size.height()/2),
            "middle_right":  QPointF(br.right() - size.width() - mx, br.center().y() - size.height()/2),
            "bottom_left":   QPointF(br.left() + mx,  br.bottom() - size.height() - my),
            "bottom_center": QPointF(br.center().x() - size.width()/2, br.bottom() - size.height() - my),
            "bottom_right":  QPointF(br.right() - size.width() - mx, br.bottom() - size.height() - my),
        }
        pt = table.get(key)
        if pt is None: return
        item.setPos(base.mapToScene(pt))


    # bake overlays into the doc
    def _affix_inserts(self):
        if not self.inserts:
            QMessageBox.information(self, "Signature / Insert", "No inserts to affix.")
            return

        # deselect to avoid selection outline; honor the checkbox visibility
        for it in self.inserts: it.setSelected(False)
        hidden_boxes = []
        if not self.bounding_boxes_enabled:
            for r in self.bounding_boxes:
                r.setVisible(False); hidden_boxes.append(r)

        # gather relevant items (background + inserts + (maybe) boxes)
        items = []
        for it in self.scene.items():
            if isinstance(it, QGraphicsPixmapItem) and (it.zValue()==0 or it in self.inserts):
                items.append(it)
            elif self.bounding_boxes_enabled and isinstance(it, QGraphicsRectItem):
                items.append(it)

        # compute bbox
        bbox = QRectF()
        for it in items:
            bbox = bbox.united(it.sceneBoundingRect())
        bbox = bbox.normalized()
        x, y = int(bbox.left()), int(bbox.top())
        w, h = int(bbox.right()) - x, int(bbox.bottom()) - y
        if w <= 0 or h <= 0: return

        # temporarily hide others
        hidden = []
        for it in self.scene.items():
            if it not in items:
                it.setVisible(False); hidden.append(it)

        # render
        out = QImage(w, h, QImage.Format.Format_ARGB32); out.fill(Qt.GlobalColor.transparent)
        p = QPainter(out)
        self.scene.render(p, target=QRectF(0, 0, w, h), source=QRectF(x, y, w, h))
        p.end()

        # restore
        for it in hidden: it.setVisible(True)
        for r in hidden_boxes: r.setVisible(True)

        # back to numpy (drop alpha), push to doc
        arr = self._qimage_to_numpy(out)
        if arr.shape[2] == 4:
            arr = arr[:, :, :3]
        arr = np.clip(arr, 0.0, 1.0).astype(np.float32, copy=False)

        if hasattr(self.doc, "set_image"):
            self.doc.set_image(arr, step_name="Signature / Insert")
        elif hasattr(self.doc, "apply_numpy"):
            self.doc.apply_numpy(arr, step_name="Signature / Insert")
        else:
            self.doc.image = arr

        # cleanup
        self._clear_inserts()
        self._update_base_image()

    def _clear_inserts(self):
        for it in self.inserts: self.scene.removeItem(it)
        self.inserts.clear()
        for r in self.bounding_boxes: self.scene.removeItem(r)
        self.bounding_boxes.clear()

    # ------------------ numpy/QImage bridges ------------------
    def _numpy_to_qimage(self, a: np.ndarray) -> QImage:
        a = np.asarray(a, dtype=np.float32)
        a = np.clip(a, 0.0, 1.0)
        if a.ndim == 2:
            a = a[..., None].repeat(3, axis=2)
        if a.shape[2] == 3:
            fmt, ch = QImage.Format.Format_RGB888, 3
        elif a.shape[2] == 4:
            fmt, ch = QImage.Format.Format_RGBA8888, 4
        else:
            raise ValueError(f"Unsupported shape {a.shape}")
        u8 = (a * 255.0).astype(np.uint8)
        u8 = np.ascontiguousarray(u8)
        h, w = u8.shape[:2]
        return QImage(u8.data, w, h, w*ch, fmt).copy()

    def _qimage_to_numpy(self, img: QImage) -> np.ndarray:
        q = img.convertToFormat(QImage.Format.Format_RGBA8888)
        w, h = q.width(), q.height()
        ptr = q.bits(); ptr.setsize(h * q.bytesPerLine())
        buf = np.frombuffer(ptr, dtype=np.uint8).reshape((h, q.bytesPerLine()))
        arr = buf[:, :w*4].reshape((h, w, 4)).astype(np.float32)/255.0
        return arr
