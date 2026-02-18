# pro/signature_insert.py
from __future__ import annotations
import math
import numpy as np

from PyQt6.QtCore import Qt, QTimer, QRectF, QPointF, QSettings
from PyQt6.QtGui import (
    QImage, QPixmap, QPainter, QColor, QPen, QTransform, QIcon, QFont, QPainterPath, QFontMetricsF, QFontDatabase, QTextCursor, QTextCharFormat, QBrush
)
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox, QLabel, QPushButton,
    QSlider, QCheckBox, QColorDialog, QComboBox, QFileDialog, QInputDialog, QMenu,
    QMessageBox, QWidget, QGraphicsView, QGraphicsScene, QGraphicsItem,QFontComboBox, QGraphicsTextItem,
    QGraphicsPixmapItem, QGraphicsEllipseItem, QGraphicsRectItem, QSpinBox, QScrollArea
)
from setiastro.saspro.widgets.themed_buttons import themed_toolbtn

def _qcolor_to_rgba(c: QColor) -> str:
    # store as "#AARRGGBB" so alpha is preserved if you ever want it
    return c.name(QColor.NameFormat.HexArgb)

def _rgba_to_qcolor(s: str, fallback: QColor) -> QColor:
    try:
        c = QColor(s)
        return c if c.isValid() else QColor(fallback)
    except Exception:
        return QColor(fallback)


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

# --------------------------- Header helpers (DocManager truth) ---------------------------

def _header_items(header_obj):
    """
    Yield (key, value) pairs from whatever DocManager stored as the header.
    Supports:
      - astropy Header-like (has .cards OR .keys/.get) without importing astropy
      - dict-like (has .items)
      - list/tuple of (k, v) pairs
    """
    if header_obj is None:
        return []

    # dict-like
    items = getattr(header_obj, "items", None)
    if callable(items):
        try:
            return list(header_obj.items())
        except Exception:
            pass

    # astropy Header-like: prefer cards if present
    cards = getattr(header_obj, "cards", None)
    if cards is not None:
        try:
            out = []
            for c in cards:
                k = getattr(c, "keyword", None)
                v = getattr(c, "value", None)
                if k is not None:
                    out.append((k, v))
            return out
        except Exception:
            pass

    # Header-like: keys()+get()
    keys = getattr(header_obj, "keys", None)
    getv = getattr(header_obj, "get", None)
    if callable(keys) and callable(getv):
        try:
            out = []
            for k in header_obj.keys():
                try:
                    out.append((k, header_obj.get(k)))
                except Exception:
                    continue
            return out
        except Exception:
            pass

    # list of pairs
    if isinstance(header_obj, (list, tuple)):
        try:
            out = []
            for it in header_obj:
                if isinstance(it, (list, tuple)) and len(it) >= 2:
                    out.append((it[0], it[1]))
            return out
        except Exception:
            pass

    return []


def _header_to_dict(header_obj) -> dict[str, str]:
    """
    Normalize any header-ish object into {KEY: VALUE_STR}.
    DOES NOT read from disk. Uses what DocManager already has.
    """
    out: dict[str, str] = {}
    for k, v in _header_items(header_obj):
        if k is None:
            continue
        ks = str(k).strip()
        if not ks:
            continue

        # Skip non-meaningful values
        if v is None:
            continue

        vs = str(v).strip()
        if not vs:
            continue

        out[ks] = vs
    return out


def _get_docmanager_header_dict(doc) -> dict[str, str]:
    """
    Pull the *current* header already living in doc.metadata, as maintained by DocManager.
    Priority should match your save logic expectations:
      - wcs_header (if it's a real header object)
      - fits_header
      - original_header
      - header
    """
    meta = getattr(doc, "metadata", None) or {}

    # IMPORTANT: This order matches what you WANT for display:
    # show WCS-enhanced header if present, otherwise fall back.
    for key in ("wcs_header", "fits_header", "original_header", "header"):
        hdr_obj = meta.get(key)
        if hdr_obj is None:
            continue

        d = _header_to_dict(hdr_obj)
        if d:
            return d

    return {}


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
    A small circular handle that sits on the top-right of the item's local
    bounds and allows scale+rotation by dragging. Works for any QGraphicsItem
    that has boundingRect()/setScale()/setRotation().
    """
    def __init__(self, parent_item: QGraphicsItem, scene: QGraphicsScene):
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
        self.initial_scale = max(0.05, float(self.parent_item.scale()) if hasattr(self.parent_item, "scale") else 1.0)

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
        sc = getattr(self.parent_item, "scale", None)
        self.initial_scale = sc() if callable(sc) else 1.0
        e.accept()

    def mouseMoveEvent(self, e):
        center = self.parent_item.mapToScene(self.parent_item.boundingRect().center())
        new_pos = self.mapToScene(e.pos())
        delta = new_pos - center
        dist = math.hypot(delta.x(), delta.y())
        ang  = math.degrees(math.atan2(delta.y(), delta.x()))

        # scale
        s = (dist / self.initial_distance) if self.initial_distance else 1.0
        new_scale = max(0.05, float(self.initial_scale) * s)
        if hasattr(self.parent_item, "setScale"):
            self.parent_item.setScale(new_scale)

        # rotate
        if hasattr(self.parent_item, "setRotation"):
            self.parent_item.setRotation(ang - self.initial_angle)

        self.update_position()
        e.accept()

    def mouseReleaseEvent(self, e):
        self.initial_distance = None
        self.initial_angle = None
        sc = getattr(self.parent_item, "scale", None)
        self.initial_scale = sc() if callable(sc) else 1.0
        e.accept()

class OutlinedTextItem(QGraphicsTextItem):
    """
    Text item that paints a solid fill with an optional outline.
    It still supports selection/transform/opacity like other items.
    """
    def __init__(self, text: str, font: QFont, fill: QColor, outline: QColor | None, outline_w: float = 0.0):
        super().__init__(text)
        self._font = font
        self._fill = QColor(fill)
        self._outline = QColor(outline) if outline else None
        self._outline_w = float(max(0.0, outline_w))
        self.setFont(font)
        self.setDefaultTextColor(self._fill)

        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable |
            QGraphicsItem.GraphicsItemFlag.ItemIsSelectable |
            QGraphicsItem.GraphicsItemFlag.ItemIsFocusable |
            QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.setTransformOriginPoint(self.boundingRect().center())
        self.setZValue(1)

    # simple multi-line path builder
    def _text_path(self) -> QPainterPath:
        path = QPainterPath()
        fm = QFontMetricsF(self._font)
        lh = fm.lineSpacing()
        y = 0.0
        for i, line in enumerate(self.toPlainText().splitlines() or [""]):
            # baseline at +ascent for each line
            path.addText(0, y + fm.ascent(), self._font, line)
            y += lh
        return path

    def paint(self, painter, option, widget=None):
        # draw fill as normal (fast)
        if not self._outline or self._outline_w <= 0.0:
            return super().paint(painter, option, widget)

        # with outline: draw a vector path so the stroke is crisp after scaling
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        path = self._text_path()

        # center origin like the base class would—translate so (0,0) is our item’s top-left
        painter.translate(self.boundingRect().topLeft())

        pen = QPen(self._outline, max(0.0, self._outline_w), Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(QBrush(self._fill))
        painter.drawPath(path)

        painter.restore()

    # accessors used by the controls
    def set_fill(self, c: QColor):
        self._fill = QColor(c)
        self.setDefaultTextColor(self._fill)
        self.update()

    def set_outline(self, c: QColor | None, w: float):
        self._outline = QColor(c) if c else None
        self._outline_w = float(max(0.0, w))
        self.update()

    def set_font(self, f: QFont):
        self._font = f
        self.setFont(f)
        self.update()



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

    def contextMenuEvent(self, e):
        scene_pos = self.mapToScene(e.pos())
        item = self.scene().itemAt(scene_pos, self.transform())

        if item is None:
            return super().contextMenuEvent(e)

        # 1) If user clicked the child rect (bounding box) -> use the parent pixmap insert
        if isinstance(item, QGraphicsRectItem) and item.parentItem() in self.owner.inserts:
            item = item.parentItem()

        # 2) If user clicked *inside* a TechCard (bg/border/text) -> walk up to TechCardItem
        # (TechCardItem is a parent of its bg/border/text children)
        p = item
        while p is not None and not isinstance(p, TechCardItem):
            p = p.parentItem()
        if isinstance(p, TechCardItem):
            item = p

        # 3) Determine if this is a snap-eligible insert
        is_pix_insert = isinstance(item, QGraphicsPixmapItem) and item in self.owner.inserts
        is_text_insert = isinstance(item, QGraphicsTextItem)  # includes OutlinedTextItem
        is_tech_card = isinstance(item, TechCardItem)

        if not (is_pix_insert or is_text_insert or is_tech_card):
            return super().contextMenuEvent(e)

        # 4) Build one menu
        m = QMenu(self)
        pos = {
            "Top-Left":      "top_left",
            "Top-Center":    "top_center",
            "Top-Right":     "top_right",
            "Middle-Left":   "middle_left",
            "Center":        "center",
            "Middle-Right":  "middle_right",
            "Bottom-Left":   "bottom_left",
            "Bottom-Center": "bottom_center",
            "Bottom-Right":  "bottom_right",
        }
        for label, key in pos.items():
            m.addAction(label, lambda k=key, it=item: self.owner.send_insert_to_position(it, k))

        m.exec(e.globalPos())
        e.accept()


class TechCardItem(QGraphicsItem):
    def __init__(self, text_item: OutlinedTextItem):
        super().__init__()
        self.bg = QGraphicsRectItem(self)
        self.border = QGraphicsRectItem(self)
        self.text = text_item
        self.text.setParentItem(self)

        self.padding = 16
        self.bg_color = QColor(0, 0, 0)
        self.bg_opacity = 0.55

        self.border_enabled = True
        self.border_pen = QPen(QColor("white"), 2, Qt.PenStyle.SolidLine)
        self.border_pen.setCosmetic(True)

        # ✅ correct: QPen/QBrush objects
        self.bg.setPen(QPen(Qt.PenStyle.NoPen))
        self.bg.setBrush(QBrush(self.bg_color))
        self.bg.setOpacity(self.bg_opacity)

        self.border.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        self.border.setPen(self.border_pen)
        self.border.setVisible(self.border_enabled)

        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable |
            QGraphicsItem.GraphicsItemFlag.ItemIsSelectable |
            QGraphicsItem.GraphicsItemFlag.ItemIsFocusable |
            QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.setZValue(1)
        self._rebuild_geometry()


    def boundingRect(self) -> QRectF:
        # union of child rects (border covers bg; include text)
        r = QRectF()
        r = r.united(self.bg.rect())
        r = r.united(self.border.rect())
        r = r.united(self.text.mapRectToParent(self.text.boundingRect()))
        return r

    def paint(self, painter, option, widget=None):
        # children paint themselves
        pass

    def set_padding(self, px: int):
        self.padding = max(0, int(px))
        self._rebuild_geometry()

    def set_background(self, c: QColor, opacity: float):
        self.bg_color = QColor(c)
        self.bg_opacity = max(0.0, min(1.0, float(opacity)))
        self.bg.setBrush(QBrush(self.bg_color))
        self.bg.setOpacity(self.bg_opacity)

    def set_border(self, enabled: bool, pen: QPen):
        self.border_enabled = bool(enabled)
        self.border_pen = QPen(pen)
        self.border_pen.setCosmetic(True)
        self.border.setVisible(self.border_enabled)
        self.border.setPen(self.border_pen)

    def set_text(self, s: str):
        self.text.setPlainText(s)
        self._rebuild_geometry()

    def _rebuild_geometry(self):
        self.prepareGeometryChange()  # <-- move to top

        self.text.setPos(QPointF(self.padding, self.padding))
        tr = self.text.mapRectToParent(self.text.boundingRect())

        w = tr.width() + 2 * self.padding
        h = tr.height() + 2 * self.padding

        rect = QRectF(0, 0, w, h)
        self.bg.setRect(rect)
        self.border.setRect(rect)

        self.setTransformOriginPoint(rect.center())
        self.update()



# --------------------------- Main dialog ---------------------------

class SignatureInsertDialogPro(QDialog):
    """
    Add one or more overlays (“signatures/inserts”) on top of the active doc,
    transform them interactively, then bake into the doc.
    """
    def __init__(self, parent, doc, icon: QIcon | None = None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Signature / Insert"))
        self.setWindowFlag(Qt.WindowType.Window, True)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setModal(False)
        self.settings = QSettings()  # match the rest of your app if you already use QSettings elsewhere
        self._persist_prefix = "signature_insert"        # settings namespace for this dialog        
        try:
            self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        except Exception:
            pass  # older PyQt6 versions        
        if icon:
            try: self.setWindowIcon(icon)
            except Exception as e:
                import logging
                logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")

        self.doc = doc
        self.scene = QGraphicsScene(self)
        self.view  = InsertView(self.scene, self)

        self.inserts: list[QGraphicsPixmapItem] = []
        self.bounding_boxes: list[QGraphicsRectItem] = []
        self.bounding_boxes_enabled = True
        self.bounding_box_pen = QPen(QColor("red"), 2, Qt.PenStyle.DashLine)
        self.bounding_box_pen.setCosmetic(True)
        self.text_inserts: list[OutlinedTextItem] = []
        self.scene.selectionChanged.connect(self._on_selection_changed)
        # Handle sync timer (keeps the handle parked on the item corner)
        self._timer = QTimer(self); self._timer.timeout.connect(self._sync_handles); self._timer.start(16)
        self.tech_card_item: TechCardItem | None = None
        self.tech_card_text: OutlinedTextItem | None = None
        self.tech_fields_order: list[str] = []
        self.tech_fields_enabled: set[str] = set()
        self.tech_bg_color = QColor(0, 0, 0)
        self.tech_border_color = QColor("white")
        self.tech_text_fill = QColor("white")
        self.tech_text_outline = QColor("black")

        self._build_ui()
        self._load_persistent_ui()
        self._wire_persistence_signals()
        
        self._update_base_image()
        self._tech_init_catalog()
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

        # --- Text controls ----------------------------------------------------
        txt_grp = QGroupBox("Text")
        tg = QGridLayout(txt_grp)

        self.btn_add_text = QPushButton("Add Text…")
        self.btn_edit_text = QPushButton("Edit Selected…"); self.btn_edit_text.setEnabled(False)
        self.btn_add_text.clicked.connect(self._add_text_dialog)
        self.btn_edit_text.clicked.connect(self._edit_selected_text)

        tg.addWidget(self.btn_add_text, 0, 0)
        tg.addWidget(self.btn_edit_text, 0, 1)

        self.font_box = QFontComboBox(); self.font_box.setCurrentFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.GeneralFont))
        self.font_size = QSpinBox(); self.font_size.setRange(4, 512); self.font_size.setValue(36)
        self.chk_bold = QCheckBox("Bold")
        self.chk_italic = QCheckBox("Italic")

        self.btn_fill = QPushButton("Fill Color…")
        self.btn_outline = QPushButton("Outline Color…")
        self.outline_w = QSpinBox(); self.outline_w.setRange(0, 30); self.outline_w.setValue(0)

        # wire style changes
        self.font_box.currentFontChanged.connect(lambda _: self._apply_text_controls_to_selected())
        self.font_size.valueChanged.connect(lambda _: self._apply_text_controls_to_selected())
        self.chk_bold.stateChanged.connect(lambda _: self._apply_text_controls_to_selected())
        self.chk_italic.stateChanged.connect(lambda _: self._apply_text_controls_to_selected())
        self.btn_fill.clicked.connect(self._pick_text_fill)
        self.btn_outline.clicked.connect(self._pick_text_outline)
        self.outline_w.valueChanged.connect(lambda _: self._apply_text_controls_to_selected())

        tg.addWidget(QLabel("Font"), 1, 0); tg.addWidget(self.font_box, 1, 1)
        tg.addWidget(QLabel("Size"), 2, 0); tg.addWidget(self.font_size, 2, 1)
        tg.addWidget(self.chk_bold, 3, 0);  tg.addWidget(self.chk_italic, 3, 1)
        tg.addWidget(self.btn_fill, 4, 0);  tg.addWidget(self.btn_outline, 4, 1)
        tg.addWidget(QLabel("Outline (px)"), 5, 0); tg.addWidget(self.outline_w, 5, 1)

        col.addWidget(txt_grp)

        # --- Tech Card ---------------------------------------------------------
        tech_grp = QGroupBox("Technical Card")
        tc = QGridLayout(tech_grp)

        self.cb_tech = QCheckBox("Enable Tech Card")
        self.cb_tech.stateChanged.connect(self._tech_toggle)

        self.btn_tech_build = QPushButton("Build / Update")
        self.btn_tech_build.clicked.connect(self._tech_rebuild)

        self.btn_tech_reset = QPushButton("Reset")
        self.btn_tech_reset.clicked.connect(self._tech_reset_defaults)

        tc.addWidget(self.cb_tech, 0, 0, 1, 2)
        tc.addWidget(self.btn_tech_build, 0, 2)
        tc.addWidget(self.btn_tech_reset, 0, 3)

        # field list (simple: checklist combo)
        self.cmb_tech_field = QComboBox()
        self.btn_tech_add_field = QPushButton("Add Field")
        self.btn_tech_remove_field = QPushButton("Remove Field")
        self.btn_tech_up = QPushButton("Up")
        self.btn_tech_down = QPushButton("Down")

        self.btn_tech_add_field.clicked.connect(self._tech_add_field)
        self.btn_tech_remove_field.clicked.connect(self._tech_remove_field)
        self.btn_tech_up.clicked.connect(lambda: self._tech_move_field(-1))
        self.btn_tech_down.clicked.connect(lambda: self._tech_move_field(+1))

        tc.addWidget(QLabel("Fields"), 1, 0)
        tc.addWidget(self.cmb_tech_field, 1, 1, 1, 2)
        tc.addWidget(self.btn_tech_add_field, 1, 3)

        self.cmb_tech_order = QComboBox()  # shows current ordered selection
        tc.addWidget(QLabel("Order"), 2, 0)
        tc.addWidget(self.cmb_tech_order, 2, 1, 1, 2)
        btns = QHBoxLayout()
        btns.addWidget(self.btn_tech_remove_field)
        btns.addWidget(self.btn_tech_up)
        btns.addWidget(self.btn_tech_down)
        w_btns = QWidget(); w_btns.setLayout(btns)
        tc.addWidget(w_btns, 2, 3)

        self.cb_tech_hide_empty = QCheckBox("Hide empty fields")
        self.cb_tech_hide_empty.setChecked(True)
        self.cb_tech_hide_empty.stateChanged.connect(lambda: self._tech_rebuild(live=True))
        tc.addWidget(self.cb_tech_hide_empty, 3, 0, 1, 2)

        # style controls
        self.sp_tech_padding = QSpinBox(); self.sp_tech_padding.setRange(0, 200); self.sp_tech_padding.setValue(16)
        self.sp_tech_padding.valueChanged.connect(lambda: self._tech_rebuild(live=True))

        self.sl_tech_bg_opacity = QSlider(Qt.Orientation.Horizontal); self.sl_tech_bg_opacity.setRange(0, 100); self.sl_tech_bg_opacity.setValue(55)
        self.sl_tech_bg_opacity.valueChanged.connect(lambda: self._tech_rebuild(live=True))

        self.cb_tech_border = QCheckBox("Border"); self.cb_tech_border.setChecked(True)
        self.cb_tech_border.stateChanged.connect(lambda: self._tech_rebuild(live=True))

        self.sp_tech_border_w = QSpinBox(); self.sp_tech_border_w.setRange(0, 30); self.sp_tech_border_w.setValue(2)
        self.sp_tech_border_w.valueChanged.connect(lambda: self._tech_rebuild(live=True))

        self.cmb_tech_border_style = QComboBox()
        self.cmb_tech_border_style.addItems(["Solid","Dash","Dot","DashDot","DashDotDot"])
        self.cmb_tech_border_style.currentIndexChanged.connect(lambda: self._tech_rebuild(live=True))

        self.btn_tech_bg = QPushButton("BG Color…"); self.btn_tech_bg.clicked.connect(self._tech_pick_bg)
        self.btn_tech_border_color = QPushButton("Border Color…"); self.btn_tech_border_color.clicked.connect(self._tech_pick_border)

        tc.addWidget(QLabel("Padding"), 4, 0); tc.addWidget(self.sp_tech_padding, 4, 1)
        tc.addWidget(QLabel("BG Opacity"), 5, 0); tc.addWidget(self.sl_tech_bg_opacity, 5, 1, 1, 3)
        tc.addWidget(self.btn_tech_bg, 4, 2); tc.addWidget(self.btn_tech_border_color, 4, 3)
        tc.addWidget(self.cb_tech_border, 6, 0); tc.addWidget(QLabel("Width"), 6, 1); tc.addWidget(self.sp_tech_border_w, 6, 2); tc.addWidget(self.cmb_tech_border_style, 6, 3)

        col.addWidget(tech_grp)


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
        b_affix = QPushButton("Affix Inserts");   b_affix.clicked.connect(self._affix_inserts)
        b_clear_sel = QPushButton("Clear Selected"); b_clear_sel.clicked.connect(self._clear_selected)
        b_clear = QPushButton("Clear All");       b_clear.clicked.connect(self._clear_inserts)
        row_commit.addWidget(b_affix)
        row_commit.addWidget(b_clear_sel)   # ← NEW
        row_commit.addWidget(b_clear)
        row_commit.addStretch(1)
        col.addLayout(row_commit)

        left = QWidget()
        left.setLayout(col)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setWidget(left)

        # Optional: make it feel less “webby”
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)

        root.addWidget(scroll, 0)   # instead of root.addWidget(left, 0)
        root.addWidget(self.view, 1)

    def _k(self, key: str) -> str:
        return f"{self._persist_prefix}/{key}"

    def _load_persistent_ui(self):
        s = self.settings

        # ---- Text controls ----
        font_family = s.value(self._k("text/font_family"), "", type=str)
        if font_family:
            try:
                self.font_box.setCurrentFont(QFont(font_family))
            except Exception:
                pass

        self.font_size.setValue(int(s.value(self._k("text/font_size"), self.font_size.value(), type=int)))
        self.chk_bold.setChecked(bool(s.value(self._k("text/bold"), self.chk_bold.isChecked(), type=bool)))
        self.chk_italic.setChecked(bool(s.value(self._k("text/italic"), self.chk_italic.isChecked(), type=bool)))
        self.outline_w.setValue(int(s.value(self._k("text/outline_w"), self.outline_w.value(), type=int)))

        # Persist your last-picked colors (also used by tech card)
        self._text_fill_color = _rgba_to_qcolor(
            s.value(self._k("text/fill_rgba"), _qcolor_to_rgba(QColor("white")), type=str),
            QColor("white")
        )
        self._text_outline_color = _rgba_to_qcolor(
            s.value(self._k("text/outline_rgba"), _qcolor_to_rgba(QColor("black")), type=str),
            QColor("black")
        )

        # ---- Transform controls ----
        self.sl_scale.setValue(int(s.value(self._k("xform/scale"), self.sl_scale.value(), type=int)))
        self.sl_opacity.setValue(int(s.value(self._k("xform/opacity"), self.sl_opacity.value(), type=int)))

        # ---- Bounding box controls ----
        self.cb_draw.setChecked(bool(s.value(self._k("bbox/enabled"), self.cb_draw.isChecked(), type=bool)))
        self.sl_thick.setValue(int(s.value(self._k("bbox/thickness"), self.sl_thick.value(), type=int)))
        self.cmb_style.setCurrentText(str(s.value(self._k("bbox/style"), self.cmb_style.currentText(), type=str)))

        self.bounding_box_pen.setColor(
            _rgba_to_qcolor(
                s.value(self._k("bbox/color_rgba"), _qcolor_to_rgba(self.bounding_box_pen.color()), type=str),
                self.bounding_box_pen.color()
            )
        )
        # apply the pen based on restored thickness/style
        self._update_box_pen()

        # ---- Snap margins ----
        self.sp_margin_x.setValue(int(s.value(self._k("snap/margin_x"), self.sp_margin_x.value(), type=int)))
        self.sp_margin_y.setValue(int(s.value(self._k("snap/margin_y"), self.sp_margin_y.value(), type=int)))

        # ---- Tech card controls/state ----
        self.cb_tech.setChecked(bool(s.value(self._k("tech/enabled"), self.cb_tech.isChecked(), type=bool)))
        self.cb_tech_hide_empty.setChecked(bool(s.value(self._k("tech/hide_empty"), self.cb_tech_hide_empty.isChecked(), type=bool)))
        self.sp_tech_padding.setValue(int(s.value(self._k("tech/padding"), self.sp_tech_padding.value(), type=int)))
        self.sl_tech_bg_opacity.setValue(int(s.value(self._k("tech/bg_opacity"), self.sl_tech_bg_opacity.value(), type=int)))
        self.cb_tech_border.setChecked(bool(s.value(self._k("tech/border_enabled"), self.cb_tech_border.isChecked(), type=bool)))
        self.sp_tech_border_w.setValue(int(s.value(self._k("tech/border_w"), self.sp_tech_border_w.value(), type=int)))
        self.cmb_tech_border_style.setCurrentText(str(s.value(self._k("tech/border_style"), self.cmb_tech_border_style.currentText(), type=str)))

        self.tech_bg_color = _rgba_to_qcolor(
            s.value(self._k("tech/bg_rgba"), _qcolor_to_rgba(self.tech_bg_color), type=str),
            self.tech_bg_color
        )
        self.tech_border_color = _rgba_to_qcolor(
            s.value(self._k("tech/border_rgba"), _qcolor_to_rgba(self.tech_border_color), type=str),
            self.tech_border_color
        )
        self.tech_text_fill = _rgba_to_qcolor(
            s.value(self._k("tech/text_fill_rgba"), _qcolor_to_rgba(self.tech_text_fill), type=str),
            self.tech_text_fill
        )
        self.tech_text_outline = _rgba_to_qcolor(
            s.value(self._k("tech/text_outline_rgba"), _qcolor_to_rgba(self.tech_text_outline), type=str),
            self.tech_text_outline
        )

        # Order/enabled lists
        self.tech_fields_order = list(s.value(self._k("tech/fields_order"), [], type=list) or [])
        self.tech_fields_enabled = set(s.value(self._k("tech/fields_enabled"), [], type=list) or [])

        # If tech fields were never saved, your existing _tech_init_catalog() will set defaults.
        # But if they WERE saved, we want to keep them.
        # _tech_init_catalog() is called in _update_base_image(); we just ensure our saved lists exist before then.

    def _save_persistent_ui(self):
        s = self.settings

        # ---- Text controls ----
        try:
            s.setValue(self._k("text/font_family"), self.font_box.currentFont().family())
        except Exception:
            pass
        s.setValue(self._k("text/font_size"), int(self.font_size.value()))
        s.setValue(self._k("text/bold"), bool(self.chk_bold.isChecked()))
        s.setValue(self._k("text/italic"), bool(self.chk_italic.isChecked()))
        s.setValue(self._k("text/outline_w"), int(self.outline_w.value()))

        # Save last-picked colors (we keep them in attributes)
        s.setValue(self._k("text/fill_rgba"), _qcolor_to_rgba(getattr(self, "_text_fill_color", QColor("white"))))
        s.setValue(self._k("text/outline_rgba"), _qcolor_to_rgba(getattr(self, "_text_outline_color", QColor("black"))))

        # ---- Transform controls ----
        s.setValue(self._k("xform/scale"), int(self.sl_scale.value()))
        s.setValue(self._k("xform/opacity"), int(self.sl_opacity.value()))

        # ---- Bounding box ----
        s.setValue(self._k("bbox/enabled"), bool(self.cb_draw.isChecked()))
        s.setValue(self._k("bbox/thickness"), int(self.sl_thick.value()))
        s.setValue(self._k("bbox/style"), str(self.cmb_style.currentText()))
        s.setValue(self._k("bbox/color_rgba"), _qcolor_to_rgba(self.bounding_box_pen.color()))

        # ---- Snap margins ----
        s.setValue(self._k("snap/margin_x"), int(self.sp_margin_x.value()))
        s.setValue(self._k("snap/margin_y"), int(self.sp_margin_y.value()))

        # ---- Tech card ----
        s.setValue(self._k("tech/enabled"), bool(self.cb_tech.isChecked()))
        s.setValue(self._k("tech/hide_empty"), bool(self.cb_tech_hide_empty.isChecked()))
        s.setValue(self._k("tech/padding"), int(self.sp_tech_padding.value()))
        s.setValue(self._k("tech/bg_opacity"), int(self.sl_tech_bg_opacity.value()))
        s.setValue(self._k("tech/border_enabled"), bool(self.cb_tech_border.isChecked()))
        s.setValue(self._k("tech/border_w"), int(self.sp_tech_border_w.value()))
        s.setValue(self._k("tech/border_style"), str(self.cmb_tech_border_style.currentText()))

        s.setValue(self._k("tech/bg_rgba"), _qcolor_to_rgba(self.tech_bg_color))
        s.setValue(self._k("tech/border_rgba"), _qcolor_to_rgba(self.tech_border_color))
        s.setValue(self._k("tech/text_fill_rgba"), _qcolor_to_rgba(self.tech_text_fill))
        s.setValue(self._k("tech/text_outline_rgba"), _qcolor_to_rgba(self.tech_text_outline))

        s.setValue(self._k("tech/fields_order"), list(self.tech_fields_order))
        s.setValue(self._k("tech/fields_enabled"), list(self.tech_fields_enabled))

        try:
            s.sync()
        except Exception:
            pass

    def _wire_persistence_signals(self):
        # Save often; QSettings writes are cheap.
        def save():
            self._save_persistent_ui()

        # text controls
        self.font_box.currentFontChanged.connect(lambda *_: save())
        self.font_size.valueChanged.connect(lambda *_: save())
        self.chk_bold.stateChanged.connect(lambda *_: save())
        self.chk_italic.stateChanged.connect(lambda *_: save())
        self.outline_w.valueChanged.connect(lambda *_: save())

        # transform controls
        self.sl_scale.valueChanged.connect(lambda *_: save())
        self.sl_opacity.valueChanged.connect(lambda *_: save())

        # bbox controls
        self.cb_draw.stateChanged.connect(lambda *_: save())
        self.sl_thick.valueChanged.connect(lambda *_: save())
        self.cmb_style.currentIndexChanged.connect(lambda *_: save())

        # margins
        self.sp_margin_x.valueChanged.connect(lambda *_: save())
        self.sp_margin_y.valueChanged.connect(lambda *_: save())

        # tech controls
        self.cb_tech.stateChanged.connect(lambda *_: save())
        self.cb_tech_hide_empty.stateChanged.connect(lambda *_: save())
        self.sp_tech_padding.valueChanged.connect(lambda *_: save())
        self.sl_tech_bg_opacity.valueChanged.connect(lambda *_: save())
        self.cb_tech_border.stateChanged.connect(lambda *_: save())
        self.sp_tech_border_w.valueChanged.connect(lambda *_: save())
        self.cmb_tech_border_style.currentIndexChanged.connect(lambda *_: save())


    def _tech_active_doc(self):
        dm = getattr(self, "doc_manager", None)
        if dm is not None:
            try:
                d = dm.get_active_document()
                if d is not None:
                    return d
            except Exception:
                pass
        return getattr(self, "doc", None) or getattr(self, "_doc", None)

    def _tech_pick_header_obj(self, doc):
        meta = getattr(doc, "metadata", None) or {}
        for key in ("wcs_header", "fits_header", "original_header", "header"):
            h = meta.get(key)
            if h is not None:
                return h
        return None

    def _tech_header_to_dict(self, hdr_obj) -> dict[str, str]:
        """
        Convert whatever header object DocManager stored into {KEY: VALUE_STR}.
        No disk IO. No astropy import.
        """
        if hdr_obj is None:
            return {}

        # dict-like
        items = getattr(hdr_obj, "items", None)
        if callable(items):
            try:
                return {str(k).strip(): str(v).strip() for k, v in hdr_obj.items()
                        if str(k).strip() and str(v).strip() and str(v).strip() != "None"}
            except Exception:
                pass

        # astropy Header-like: cards iterable with .keyword/.value
        cards = getattr(hdr_obj, "cards", None)
        if cards is not None:
            out = {}
            try:
                for c in cards:
                    k = getattr(c, "keyword", None)
                    v = getattr(c, "value", None)
                    if k is None or v is None:
                        continue
                    ks = str(k).strip()
                    vs = str(v).strip()
                    if ks and vs and vs != "None":
                        out[ks] = vs
                if out:
                    return out
            except Exception:
                pass

        # Header-like: keys()+get()
        keys = getattr(hdr_obj, "keys", None)
        getv = getattr(hdr_obj, "get", None)
        if callable(keys) and callable(getv):
            out = {}
            try:
                for k in hdr_obj.keys():
                    try:
                        v = hdr_obj.get(k)
                    except Exception:
                        continue
                    if v is None:
                        continue
                    ks = str(k).strip()
                    vs = str(v).strip()
                    if ks and vs and vs != "None":
                        out[ks] = vs
                return out
            except Exception:
                pass

        return {}


    def _tech_init_catalog(self):
        catalog = self._build_tech_field_catalog()
        keys = list(catalog.keys())
        self.cmb_tech_field.clear()
        self.cmb_tech_field.addItems(keys)

        # defaults: populate order + enabled if empty
        if not self.tech_fields_order:
            defaults = ["OBJECT","DATE-OBS","EXPTIME","FILTER","GAIN","OFFSET","INSTRUME","TELESCOP","FOCALLEN","XPIXSZ","BAYERPAT","Size","Bit Depth","Mono"]
            self.tech_fields_order = [k for k in defaults if k in catalog]
            self.tech_fields_enabled = set(self.tech_fields_order)

        self._tech_refresh_order_combo()

    def _tech_refresh_order_combo(self):
        self.cmb_tech_order.blockSignals(True)
        self.cmb_tech_order.clear()
        self.cmb_tech_order.addItems(self.tech_fields_order)
        self.cmb_tech_order.blockSignals(False)

    def _tech_toggle(self, state):
        enabled = bool(state)
        if enabled:
            self._tech_ensure_item()
            self._tech_rebuild()
        else:
            self._tech_remove_item()

    def _tech_ensure_item(self):
        if self.tech_card_item is not None:
            return

        # Create a text item for the card (outlined)
        f = self._current_qfont()
        self.tech_card_text = OutlinedTextItem("", f, self.tech_text_fill, self.tech_text_outline, outline_w=float(self.outline_w.value()))
        self.tech_card_text.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)  # card text is not editable directly
        self.tech_card_text.setZValue(1)

        self.tech_card_item = TechCardItem(self.tech_card_text)
        self.tech_card_item.setZValue(1)
        self.scene.addItem(self.tech_card_item)

        TransformHandle(self.tech_card_item, self.scene)

        # drop near bottom-right by default using your snap
        self.tech_card_item.setSelected(True)
        self.send_insert_to_position(self.tech_card_item, "top_left")

    def _tech_remove_item(self):
        if self.tech_card_item is None:
            return
        # Remove handle tied to tech card
        for it in list(self.scene.items()):
            if isinstance(it, TransformHandle) and getattr(it, "parent_item", None) is self.tech_card_item:
                try: self.scene.removeItem(it)
                except Exception: pass

        try:
            self.scene.removeItem(self.tech_card_item)
        except Exception:
            pass

        self.tech_card_item = None
        self.tech_card_text = None

    def _tech_pick_bg(self):
        c = QColorDialog.getColor(self.tech_bg_color, self, "Tech Card Background")
        if c.isValid():
            self.tech_bg_color = c
            self._tech_rebuild(live=True)
            self._save_persistent_ui()

    def _tech_pick_border(self):
        c = QColorDialog.getColor(self.tech_border_color, self, "Tech Card Border")
        if c.isValid():
            self.tech_border_color = c
            self._tech_rebuild(live=True)
            self._save_persistent_ui()


    def _tech_add_field(self):
        k = self.cmb_tech_field.currentText().strip()
        if not k:
            return
        if k not in self.tech_fields_order:
            self.tech_fields_order.append(k)
        self.tech_fields_enabled.add(k)
        self._tech_refresh_order_combo()
        self._tech_rebuild(live=True)

    def _tech_remove_field(self):
        k = self.cmb_tech_order.currentText().strip()
        if not k:
            return
        if k in self.tech_fields_order:
            self.tech_fields_order.remove(k)
        self.tech_fields_enabled.discard(k)
        self._tech_refresh_order_combo()
        self._tech_rebuild(live=True)

    def _tech_move_field(self, delta: int):
        k = self.cmb_tech_order.currentText().strip()
        if not k or k not in self.tech_fields_order:
            return
        i = self.tech_fields_order.index(k)
        j = max(0, min(len(self.tech_fields_order) - 1, i + int(delta)))
        if i == j:
            return
        self.tech_fields_order.insert(j, self.tech_fields_order.pop(i))
        self._tech_refresh_order_combo()
        self.cmb_tech_order.setCurrentText(k)
        self._tech_rebuild(live=True)

    def _tech_reset_defaults(self):
        self.tech_fields_order = []
        self.tech_fields_enabled = set()
        self.tech_bg_color = QColor(0, 0, 0)
        self.tech_border_color = QColor("white")
        self.tech_text_fill = QColor("white")
        self.tech_text_outline = QColor("black")
        self.sp_tech_padding.setValue(16)
        self.sl_tech_bg_opacity.setValue(55)
        self.cb_tech_border.setChecked(True)
        self.sp_tech_border_w.setValue(2)
        self.cmb_tech_border_style.setCurrentText("Solid")
        self.cb_tech_hide_empty.setChecked(True)
        self._tech_init_catalog()
        self._tech_rebuild()

    def _tech_format_text(self) -> str:
        catalog = self._build_tech_field_catalog()
        hide_empty = self.cb_tech_hide_empty.isChecked()

        lines = []
        for k in self.tech_fields_order:
            if k not in self.tech_fields_enabled:
                continue
            v = str(catalog.get(k, "")).strip()
            if hide_empty and not v:
                continue
            # Friendly: strip full file path by default (optional)
            if k == "File" and v:
                try:
                    import os
                    v = os.path.basename(v)
                except Exception:
                    pass
            lines.append(f"{k}: {v}" if v else f"{k}:")

        return "\n".join(lines) if lines else "No fields selected."

    def _tech_border_pen(self) -> QPen:
        style_map = {
            "Solid": Qt.PenStyle.SolidLine,
            "Dash": Qt.PenStyle.DashLine,
            "Dot": Qt.PenStyle.DotLine,
            "DashDot": Qt.PenStyle.DashDotLine,
            "DashDotDot": Qt.PenStyle.DashDotDotLine
        }
        pen = QPen(self.tech_border_color, float(self.sp_tech_border_w.value()), style_map[self.cmb_tech_border_style.currentText()])
        pen.setCosmetic(True)
        return pen

    def _tech_rebuild(self, live: bool = False):
        if not self.cb_tech.isChecked():
            return
        self._tech_ensure_item()
        if self.tech_card_item is None or self.tech_card_text is None:
            return

        # Update text styling from the existing text controls (so UI stays consistent)
        f = self._current_qfont()
        self.tech_card_text.set_font(f)
        self.tech_card_text.set_fill(self.tech_text_fill)
        ow = float(self.outline_w.value())
        if ow > 0:
            self.tech_card_text.set_outline(self.tech_text_outline, ow)
        else:
            self.tech_card_text.set_outline(None, 0.0)

        # Update card style
        pad = int(self.sp_tech_padding.value())
        bg_op = float(self.sl_tech_bg_opacity.value()) / 100.0
        self.tech_card_item.set_padding(pad)
        self.tech_card_item.set_background(self.tech_bg_color, bg_op)

        border_on = self.cb_tech_border.isChecked() and self.sp_tech_border_w.value() > 0
        self.tech_card_item.set_border(border_on, self._tech_border_pen())

        # Update text content
        txt = self._tech_format_text()
        self.tech_card_item.set_text(txt)

        # Keep transform handle correct
        self._sync_handles()


    def _selected_text_items(self):
        return [it for it in self.scene.selectedItems() if isinstance(it, QGraphicsTextItem)]

    def _selected_pixmap_items(self):
        return [it for it in self.scene.selectedItems() if isinstance(it, QGraphicsPixmapItem)]

    def _add_text_item(self, text: str, font: QFont, color: QColor):
        ti = OutlinedTextItem(text, font, color, outline=None, outline_w=0.0)
        ti.setTextInteractionFlags(Qt.TextInteractionFlag.TextEditorInteraction)
        ti.setZValue(1)
        ti.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        ti.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        ti.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsFocusable, True)
        ti.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        ti.setTransformOriginPoint(ti.boundingRect().center())
        self.scene.addItem(ti)

        TransformHandle(ti, self.scene)
        self.text_inserts.append(ti)
        ti.setSelected(True)
        return ti

    def _add_text_dialog(self):
        # wrapper to match your signal connect
        self._on_add_text()

    def _add_text_dialog(self):
        # wrapper to match your signal connect
        self._on_add_text()

    def _edit_selected_text(self):
        items = self._selected_text_items()
        if not items:
            return
        ti = items[0]
        existing = ti.toPlainText()
        txt, ok = QInputDialog.getMultiLineText(self, "Edit Text", "Text:", existing)
        if ok:
            ti.setPlainText(txt)

    def _apply_text_controls_to_selected(self):
        f = self._current_qfont()
        w = self.outline_w.value()
        for ti in self._selected_text_items():
            if isinstance(ti, OutlinedTextItem):
                ti.set_font(f)
                # only adjust outline width here; color comes from the outline color picker
                if w <= 0:
                    ti.set_outline(ti._outline, 0.0)
                else:
                    ti.set_outline(ti._outline or QColor("black"), float(w))
            else:
                ti.setFont(f)

    def _pick_text_fill(self):
        c = QColorDialog.getColor(getattr(self, "_text_fill_color", QColor("white")), self, "Text Fill Color")
        if not c.isValid():
            return
        self._text_fill_color = c
        for ti in self._selected_text_items():
            if isinstance(ti, OutlinedTextItem):
                ti.set_fill(c)
            else:
                ti.setDefaultTextColor(c)
        self._save_persistent_ui()

    def _pick_text_outline(self):
        c = QColorDialog.getColor(getattr(self, "_text_outline_color", QColor("black")), self, "Text Outline Color")
        if not c.isValid():
            return
        self._text_outline_color = c
        w = self.outline_w.value()
        for ti in self._selected_text_items():
            if isinstance(ti, OutlinedTextItem):
                ti.set_outline(c, float(w))
        self._save_persistent_ui()


    def _clear_text_selection(self, ti: QGraphicsTextItem):
        cur = ti.textCursor()
        if cur.hasSelection():
            cur.clearSelection()
            ti.setTextCursor(cur)

    def _remove_item_and_accessories(self, item: QGraphicsItem):
        # Remove child bounding box if present & tracked
        if isinstance(item, QGraphicsPixmapItem):
            # child rect we added lives as parentItem(item)
            for r in list(self.bounding_boxes):
                if r.parentItem() is item:
                    try:
                        self.scene.removeItem(r)
                    except Exception:
                        pass
                    self.bounding_boxes.remove(r)

        # Remove any TransformHandle bound to this item
        for it in list(self.scene.items()):
            if isinstance(it, TransformHandle) and getattr(it, "parent_item", None) is item:
                try:
                    self.scene.removeItem(it)
                except Exception:
                    pass

        # Remove from our tracking lists
        if isinstance(item, QGraphicsPixmapItem) and item in self.inserts:
            self.inserts.remove(item)
        if isinstance(item, QGraphicsTextItem) and item in self.text_inserts:
            self.text_inserts.remove(item)

        # Finally remove the item itself
        try:
            self.scene.removeItem(item)
        except Exception:
            pass

    def _clear_selected(self):
        for it in list(self.scene.selectedItems()):
            # only user inserts (pixmaps) and text inserts are removable
            if (isinstance(it, QGraphicsPixmapItem) and it in self.inserts) or isinstance(it, QGraphicsTextItem):
                self._remove_item_and_accessories(it)


    def _on_selection_changed(self):
        texts = self._selected_text_items()
        self.btn_edit_text.setEnabled(bool(texts))
        if texts:
            ti = texts[0]
            f = ti.font()
            self.font_box.setCurrentFont(f)
            ps = f.pointSize() if f.pointSize() > 0 else 36
            self.font_size.setValue(int(ps))
            self.chk_bold.setChecked(f.bold())
            self.chk_italic.setChecked(f.italic())
            if isinstance(ti, OutlinedTextItem):
                self.outline_w.setValue(int(round(ti._outline_w)))

        # ── NEW: when a text item becomes unselected, clear any in-text highlight
        selected_set = set(texts)
        for it in self.text_inserts:
            if it not in selected_set:
                self._clear_text_selection(it)


    def _current_qfont(self) -> QFont:
        f = self.font_box.currentFont()
        f.setPointSize(self.font_size.value())
        f.setBold(self.chk_bold.isChecked())
        f.setItalic(self.chk_italic.isChecked())
        return f

    def _apply_font_to_selection(self):
        f = self._current_qfont()
        for ti in self._selected_text_items():
            ti.setFont(f)

    def _apply_color_to_selection(self, color: QColor):
        for ti in self._selected_text_items():
            ti.setDefaultTextColor(color)

    def _on_add_text(self):
        txt, ok = QInputDialog.getMultiLineText(self, "Add Text", "Enter text:")
        if not ok or not txt.strip():
            return
        f = self._current_qfont()
        c = QColor("white")  # default
        ti = self._add_text_item(txt, f, c)
        # drop it near center
        base = next((i for i in self.scene.items()
                    if isinstance(i, QGraphicsPixmapItem) and i.zValue() == 0), None)
        if base:
            center_scene = base.mapToScene(base.boundingRect().center())
            ti.setPos(center_scene - ti.boundingRect().center())

    def _on_text_color(self):
        c = QColorDialog.getColor()
        if c.isValid():
            self._apply_color_to_selection(c)

    def _on_font_changed(self, _):
        self._apply_font_to_selection()

    def _on_font_size(self, _):
        self._apply_font_to_selection()

    def _on_font_bold(self, _):
        self._apply_font_to_selection()

    def _on_font_italic(self, _):
        self._apply_font_to_selection()


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
        self._tech_init_catalog()
        if self.cb_tech.isChecked():
            self._tech_rebuild(live=True)


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
        # pixmaps
        for it in self.inserts:
            if it.isSelected():
                self.send_insert_to_position(it, key)
        # text
        for ti in self._selected_text_items():
            self.send_insert_to_position(ti, key)
        # tech card
        if self.tech_card_item is not None and self.tech_card_item.isSelected():
            self.send_insert_to_position(self.tech_card_item, key)


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
        for it in (self._selected_pixmap_items() + self._selected_text_items()):
            it.setOpacity(o)

    def _toggle_boxes(self, state):
        self.bounding_boxes_enabled = bool(state)
        for r in self.bounding_boxes:
            r.setVisible(self.bounding_boxes_enabled)

    def _pick_box_color(self):
        c = QColorDialog.getColor(self.bounding_box_pen.color(), self, "Bounding Box Color")
        if c.isValid():
            self.bounding_box_pen.setColor(c)
            self._refresh_all_boxes()
            self._save_persistent_ui()


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
        self.bounding_box_pen.setCosmetic(True)
        self._refresh_all_boxes()

    def _refresh_all_boxes(self):
        for r in self.bounding_boxes:
            r.setPen(self.bounding_box_pen)

    # snap an insert to one of 9 standard positions inside the base image
    def send_insert_to_position(self, item: QGraphicsItem, key: str):
        """Snap a selected insert (pixmap or text) to one of 9 standard positions."""
        base = next((i for i in self.scene.items()
                    if isinstance(i, QGraphicsPixmapItem) and i.zValue() == 0), None)
        if not base:
            return

        mx = self.sp_margin_x.value()
        my = self.sp_margin_y.value()

        br = base.boundingRect()
        # item's *local* bounding rect
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
        if pt is None:
            return

        # map the desired *base* point into scene coords, then move item so its local
        # top-left (0,0) maps onto that scene point.
        scene_pt = base.mapToScene(pt)
        item.setPos(scene_pt)

    def _scrub_text_highlights_for_render(self):
        """
        Collapse any QTextCursor selections inside QGraphicsTextItem so no
        character-range highlight can be painted by Qt during scene.render().
        Also disable editing and selection temporarily to be extra safe.
        """
        self._text_restore = []  # stash state to restore after render

        for it in self.scene.items():
            if isinstance(it, QGraphicsTextItem):
                # Save the minimal state we need to restore
                self._text_restore.append((
                    it,
                    it.textInteractionFlags(),
                    it.flags(),
                    it.hasFocus()
                ))

                # 1) Collapse any in-text selection (this is the blue 'N' you see)
                cur = it.textCursor()
                # Force a definite collapse (some cases cur.hasSelection() is False
                # but an anchor remains; resetting both positions removes it):
                pos = cur.position()
                cur.setPosition(pos, QTextCursor.MoveMode.MoveAnchor)
                cur.setPosition(pos, QTextCursor.MoveMode.KeepAnchor)  # set, then collapse
                cur.clearSelection()
                it.setTextCursor(cur)

                # 2) Fully exit editing state
                it.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
                it.clearFocus()

                # 3) Make sure the item itself cannot be “selected” while we paint
                it.setSelected(False)
                it.setFlags(it.flags() & ~QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)

                # 4) Ensure a repaint with the new state
                it.update()

    def _restore_text_state_after_render(self):
        if not hasattr(self, "_text_restore"):
            return
        for it, flags, item_flags, had_focus in self._text_restore:
            it.setTextInteractionFlags(flags)
            it.setFlags(item_flags)
            if had_focus:
                it.setFocus()
        self._text_restore = []


    # bake overlays into the doc
    def _affix_inserts(self):
        if not (self.inserts or self._selected_text_items() or any(isinstance(i, QGraphicsTextItem) for i in self.scene.items())):
            QMessageBox.information(self, "Signature / Insert", "Nothing to affix.")
            return

        # Deselect everything to avoid selection outlines in the render
        for it in self.scene.selectedItems():
            it.setSelected(False)

        # honor box visibility
        hidden_boxes = []
        if not self.bounding_boxes_enabled:
            for r in self.bounding_boxes:
                r.setVisible(False); hidden_boxes.append(r)

        # gather background + pixmap inserts + text + (maybe) boxes
        items = []
        for it in self.scene.items():
            if isinstance(it, QGraphicsPixmapItem) and it.zValue() == 0:
                items.append(it)  # background
            elif isinstance(it, QGraphicsPixmapItem) and it in self.inserts:
                items.append(it)
            elif isinstance(it, QGraphicsTextItem):
                items.append(it)
            elif self.bounding_boxes_enabled and isinstance(it, QGraphicsRectItem):
                items.append(it)
            elif isinstance(it, TechCardItem):
                items.append(it)

        # compute scene bbox
        bbox = QRectF()
        for it in items:
            bbox = bbox.united(it.sceneBoundingRect())
        bbox = bbox.normalized()
        x, y = int(bbox.left()), int(bbox.top())
        w, h = int(bbox.right()) - x, int(bbox.bottom()) - y
        if w <= 0 or h <= 0:
            return

        # Temporarily suppress in-text selection highlights for text items
        text_states = []
        for it in self.scene.items():
            if isinstance(it, QGraphicsTextItem):
                text_states.append((
                    it,
                    it.textInteractionFlags(),
                    it.textCursor(),
                    it.hasFocus()
                ))
                # clear any selection highlight and disable editing visuals
                cur = it.textCursor()
                if cur.hasSelection():
                    cur.clearSelection()
                    it.setTextCursor(cur)
                it.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
                it.clearFocus()

        # temporarily hide non-items
        hidden = []
        for it in self.scene.items():
            if it not in items:
                it.setVisible(False); hidden.append(it)

        self._scrub_text_highlights_for_render()        

        # --- render ---
        out = QImage(w, h, QImage.Format.Format_ARGB32)
        out.fill(Qt.GlobalColor.transparent)
        p = QPainter(out)
        self.scene.render(p, target=QRectF(0, 0, w, h), source=QRectF(x, y, w, h))
        p.end()

        self._restore_text_state_after_render()

        # restore hidden things
        for it in hidden: it.setVisible(True)
        for r in hidden_boxes: r.setVisible(True)

        # restore text editability / state
        for it, flags, cursor, had_focus in text_states:
            it.setTextInteractionFlags(flags)
            it.setTextCursor(cursor)
            if had_focus:
                it.setFocus()

        # drop alpha → RGB, write back to doc
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
        # remove all user pixmap inserts
        for it in list(self.inserts):
            self._remove_item_and_accessories(it)
        self.inserts.clear()

        # remove all text inserts
        for ti in list(self.text_inserts):
            self._remove_item_and_accessories(ti)
        self.text_inserts.clear()

        # any stray boxes that weren't parented/cleaned
        for r in list(self.bounding_boxes):
            try:
                self.scene.removeItem(r)
            except Exception:
                pass
        self.bounding_boxes.clear()

    def _build_tech_field_catalog(self) -> dict[str, str]:
        """
        Returns a merged catalog of:
        - header-derived fields (OBJECT/DATE-OBS/etc) from the in-memory doc header
        - convenience fields (File/Size/Bit Depth/Mono)
        """
        doc = self._tech_active_doc()
        if doc is None:
            return {}

        meta = getattr(doc, "metadata", None) or {}

        # --- header from DocManager (in-memory) ---
        hdr_obj = self._tech_pick_header_obj(doc)
        H = self._tech_header_to_dict(hdr_obj)

        # Case-insensitive getter for FITS keys
        # (FITS keys are usually uppercase, but play safe)
        def hget(key: str) -> str:
            if not key:
                return ""
            if key in H:
                return H.get(key, "") or ""
            up = key.upper()
            if up in H:
                return H.get(up, "") or ""
            # last resort: scan once (small headers, OK)
            for k, v in H.items():
                if str(k).upper() == up:
                    return v or ""
            return ""

        # --- computed fields ---
        # File: prefer actual file path from metadata
        file_path = meta.get("file_path") or ""
        # Size: prefer header if available (NAXIS1/2), fallback to image shape
        naxis1 = hget("NAXIS1")
        naxis2 = hget("NAXIS2")
        if naxis1 and naxis2:
            size_str = f"{naxis1}×{naxis2}"
        else:
            img = getattr(doc, "image", None)
            try:
                if img is not None and hasattr(img, "shape"):
                    if img.ndim == 2:
                        h, w = img.shape[:2]
                    else:
                        h, w = img.shape[:2]
                    size_str = f"{w}×{h}"
                else:
                    size_str = ""
            except Exception:
                size_str = ""

        bit_depth = meta.get("bit_depth") or ""
        # Mono: metadata uses is_mono in your loader; your save uses "mono" sometimes; handle both
        mono_val = meta.get("is_mono", meta.get("mono", None))
        if mono_val is None:
            img = getattr(doc, "image", None)
            mono_val = bool(img.ndim == 2) if hasattr(img, "ndim") else False
        mono_str = "Yes" if bool(mono_val) else "No"

        # --- build catalog ---
        catalog: dict[str, str] = {}

        # 1) Header keys: include all FITS keys as selectable fields
        # (This is why your dropdown will finally show everything in that big list.)
        for k, v in H.items():
            ks = str(k).strip()
            if not ks:
                continue
            catalog[ks] = str(v).strip()

        # 2) Friendly extras (these are the ones you have in defaults)
        catalog.setdefault("File", str(file_path))
        catalog["Size"] = size_str
        catalog["Bit Depth"] = str(bit_depth)
        catalog["Mono"] = mono_str

        return catalog

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

    def closeEvent(self, e):
        try:
            self._save_persistent_ui()
        except Exception:
            pass
        super().closeEvent(e)
