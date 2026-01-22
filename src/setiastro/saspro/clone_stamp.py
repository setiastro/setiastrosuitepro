# src/setiastro/saspro/clone_stamp.py
from __future__ import annotations

import numpy as np
from typing import Optional, Tuple

from PyQt6.QtCore import Qt, QEvent, QPointF,QTimer
from PyQt6.QtGui import QImage, QPixmap, QPen, QBrush, QAction, QKeySequence, QColor, QWheelEvent, QPainter
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox, QLabel, QPushButton, QSlider,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsEllipseItem, QMessageBox,
    QScrollArea, QCheckBox, QDoubleSpinBox, QGraphicsLineItem, QWidget, QFrame, QSizePolicy
)

from setiastro.saspro.imageops.stretch import stretch_color_image, stretch_mono_image
from setiastro.saspro.widgets.themed_buttons import themed_toolbtn


def _circle_mask(radius: int, feather: float) -> np.ndarray:
    """
    Returns float32 mask (2r+1, 2r+1) in [0..1], with optional feather falloff.
    feather: 0..1 (0=hard edge, 1=soft)
    """
    r = int(max(1, radius))
    y, x = np.ogrid[-r:r+1, -r:r+1]
    d = np.sqrt(x*x + y*y).astype(np.float32)
    m = (d <= r).astype(np.float32)

    if feather <= 0:
        return m

    # Soft edge: inner radius where mask=1 then falls to 0 at r
    # feather=1 => inner radius 0, feather small => thin falloff band
    inner = float(r) * (1.0 - float(np.clip(feather, 0.0, 1.0)))
    if inner < 0.5:
        inner = 0.5

    # Linear falloff in [inner..r]
    fall = np.clip((float(r) - d) / max(1e-6, float(r) - inner), 0.0, 1.0)
    return np.where(d <= inner, 1.0, np.where(d <= r, fall, 0.0)).astype(np.float32)

def _blend_clone_inplace(
    img: np.ndarray,
    tx: int, ty: int,
    sx: int, sy: int,
    mask_full: np.ndarray,   # (2r+1,2r+1)
    r: int,
    opacity: float,
) -> None:
    """
    In-place clone dab. img is float32 HxWx3 in [0..1].
    mask_full already includes feather falloff (0..1). We multiply by opacity here.
    """
    h, w = img.shape[:2]

    # target bounds
    x0 = tx - r; x1 = tx + r + 1
    y0 = ty - r; y1 = ty + r + 1

    # clip target
    cx0 = max(0, x0); cx1 = min(w, x1)
    cy0 = max(0, y0); cy1 = min(h, y1)
    if cx0 >= cx1 or cy0 >= cy1:
        return

    # map to mask coords
    mx0 = cx0 - x0
    my0 = cy0 - y0
    tw = cx1 - cx0
    th = cy1 - cy0

    # source aligned bounds for same shape
    sx0 = (sx - r) + mx0
    sy0 = (sy - r) + my0
    sx1 = sx0 + tw
    sy1 = sy0 + th

    # clip both if source out of bounds
    adj_cx0, adj_cy0, adj_cx1, adj_cy1 = cx0, cy0, cx1, cy1
    if sx0 < 0:
        d = -sx0
        sx0 = 0
        adj_cx0 += d
    if sy0 < 0:
        d = -sy0
        sy0 = 0
        adj_cy0 += d
    if sx1 > w:
        d = sx1 - w
        sx1 = w
        adj_cx1 -= d
    if sy1 > h:
        d = sy1 - h
        sy1 = h
        adj_cy1 -= d

    if adj_cx0 >= adj_cx1 or adj_cy0 >= adj_cy1:
        return

    tw = adj_cx1 - adj_cx0
    th = adj_cy1 - adj_cy0
    if tw <= 0 or th <= 0:
        return

    # recompute mask slice indices after clipping shift
    mx0 = adj_cx0 - x0
    my0 = adj_cy0 - y0

    tgt = img[adj_cy0:adj_cy1, adj_cx0:adj_cx1, :]
    src = img[sy0:sy0+th, sx0:sx0+tw, :]

    a = (mask_full[my0:my0+th, mx0:mx0+tw] * float(opacity)).astype(np.float32)
    if a.max() <= 0:
        return

    a3 = a[:, :, None]
    # in-place blend
    tgt[:] = (1.0 - a3) * tgt + a3 * src


class CloneStampDialogPro(QDialog):
    """
    Interactive Clone Stamp:
    - Ctrl+Click sets source point.
    - Left-drag paints source onto target with classic offset-follow behavior.
    Writes back to the provided document when 'Apply' is pressed.
    """
    def __init__(self, parent, doc):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Clone Stamp"))
        self.setWindowFlag(Qt.WindowType.Window, True)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setModal(False)
        self.setMinimumSize(900, 650)
        self._mask_cache_key = None
        self._mask_cache = None

        self._doc = doc
        base = getattr(doc, "image", None)
        if base is None:
            raise RuntimeError("Document has no image.")

        # normalize to float32 [0..1]
        self._orig_shape = base.shape
        self._orig_mono = (base.ndim == 2) or (base.ndim == 3 and base.shape[2] == 1)

        img = np.asarray(base, dtype=np.float32)
        if img.dtype.kind in "ui":
            maxv = float(np.nanmax(img)) or 1.0
            img = img / max(1.0, maxv)
        img = np.clip(img, 0.0, 1.0).astype(np.float32, copy=False)

        # display/working is 3-channel
        if img.ndim == 2:
            img3 = np.repeat(img[:, :, None], 3, axis=2)
        elif img.ndim == 3 and img.shape[2] == 1:
            img3 = np.repeat(img, 3, axis=2)
        elif img.ndim == 3 and img.shape[2] >= 3:
            img3 = img[:, :, :3]
        else:
            raise ValueError(f"Unsupported image shape: {img.shape}")

        self._image   = img3.copy()      # linear working
        self._display = self._image.copy()

        # --- stroke state ---
        self._has_source = False
        self._src_point: Tuple[int, int] = (0, 0)     # absolute source point (current anchor)
        self._offset: Tuple[int, int] = (0, 0)        # src - tgt at stroke start
        self._painting = False
        self._last_tgt: Optional[Tuple[int, int]] = None

        # ── Scene/View
        self.scene = QGraphicsScene(self)
        self.view  = QGraphicsView(self.scene)
        self.view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pix   = QGraphicsPixmapItem()
        self.scene.addItem(self.pix)

        # Brush circle (green, always visible on move)
        self.circle = QGraphicsEllipseItem()
        self.circle.setPen(QPen(QColor(0, 255, 0), 2, Qt.PenStyle.SolidLine))
        self.circle.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        self.circle.setVisible(False)
        self.scene.addItem(self.circle)

        # Source X (two line items)
        self.src_x1 = QGraphicsLineItem()
        self.src_x2 = QGraphicsLineItem()
        pen_src = QPen(QColor(0, 255, 0), 2, Qt.PenStyle.SolidLine)
        self.src_x1.setPen(pen_src)
        self.src_x2.setPen(pen_src)
        self.src_x1.setVisible(False)
        self.src_x2.setVisible(False)
        self.scene.addItem(self.src_x1)
        self.scene.addItem(self.src_x2)

        # Optional line from target to source
        self.link = QGraphicsLineItem()
        self.link.setPen(QPen(QColor(0, 255, 0), 1, Qt.PenStyle.DotLine))
        self.link.setVisible(False)
        self.scene.addItem(self.link)

        # scroll container
        self.scroll = QScrollArea(self)
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.view)
        self.scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        # Zoom controls
        self._zoom = 1.0
        self.btn_zoom_out = themed_toolbtn("zoom-out", "Zoom Out")
        self.btn_zoom_in  = themed_toolbtn("zoom-in", "Zoom In")
        self.btn_zoom_fit = themed_toolbtn("zoom-fit-best", "Fit to Preview")

        # ── Controls
        ctrls = QGroupBox(self.tr("Controls"))
        form  = QFormLayout(ctrls)

        self.s_radius  = QSlider(Qt.Orientation.Horizontal); self.s_radius.setRange(1, 900); self.s_radius.setValue(24)
        self.s_feather = QSlider(Qt.Orientation.Horizontal); self.s_feather.setRange(0, 100); self.s_feather.setValue(50)
        self.s_opacity = QSlider(Qt.Orientation.Horizontal); self.s_opacity.setRange(0, 100); self.s_opacity.setValue(100)

        form.addRow(self.tr("Radius:"),  self.s_radius)
        form.addRow(self.tr("Feather:"), self.s_feather)
        form.addRow(self.tr("Opacity:"), self.s_opacity)

        self.brush_preview = QLabel(self)
        self.brush_preview.setFixedSize(90, 90)
        self.brush_preview.setStyleSheet("background:#000; border:1px solid #333;")
        form.addRow(self.tr("Brush preview:"), self.brush_preview)

        self.lbl_help = QLabel(self.tr(
            "Ctrl+Click to set source. Then Left-drag to paint.\n"
            "Source follows the cursor (classic clone stamp)."
        ))
        self.lbl_help.setStyleSheet("color:#888;")
        self.lbl_help.setWordWrap(True)
        form.addRow(self.lbl_help)

        self.btn_clear_src = QPushButton(self.tr("Clear Source"))
        self.btn_clear_src.clicked.connect(self._clear_source)
        form.addRow(self.btn_clear_src)

        # Preview autostretch (display only)
        self.cb_autostretch = QCheckBox(self.tr("Auto-stretch preview"))
        self.cb_autostretch.setChecked(False)
        form.addRow(self.cb_autostretch)

        self.s_target_median = QDoubleSpinBox()
        self.s_target_median.setRange(0.01, 0.60)
        self.s_target_median.setSingleStep(0.01)
        self.s_target_median.setDecimals(3)
        self.s_target_median.setValue(0.25)
        form.addRow(self.tr("Target median:"), self.s_target_median)

        self.cb_linked = QCheckBox(self.tr("Linked color channels"))
        self.cb_linked.setChecked(True)
        form.addRow(self.cb_linked)

        self.cb_autostretch.toggled.connect(self._update_display_autostretch)
        self.s_target_median.valueChanged.connect(self._update_display_autostretch)
        self.cb_linked.toggled.connect(self._update_display_autostretch)
        self.cb_autostretch.toggled.connect(lambda on: (self.s_target_median.setEnabled(on),
                                                        self.cb_linked.setEnabled(on)))

        # buttons
        bb = QHBoxLayout()
        self.btn_undo  = QPushButton(self.tr("Undo"))
        self.btn_redo  = QPushButton(self.tr("Redo"))
        self.btn_apply = QPushButton(self.tr("Apply to Document"))
        self.btn_close = QPushButton(self.tr("Close"))

        self.btn_undo.setEnabled(False)
        self.btn_redo.setEnabled(False)

        bb.addStretch()
        bb.addWidget(self.btn_undo)
        bb.addWidget(self.btn_redo)
        bb.addSpacing(12)
        bb.addWidget(self.btn_apply)
        bb.addWidget(self.btn_close)

        # ─────────────────────────────────────────────────────────────
        # Layout: Left = Preview, Right = Zoom + Controls + Buttons
        # ─────────────────────────────────────────────────────────────
        root = QHBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(10)

        # ---- LEFT: Preview ----
        left = QVBoxLayout()
        left.setSpacing(8)

        left.addWidget(self.scroll, 1)  # preview expands


        # optional vertical separator (nice on wide screens)
        sep = QFrame(self)
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)

        # ---- RIGHT: Zoom + Controls + Buttons ----
        right = QVBoxLayout()
        right.setSpacing(10)

        # Zoom group (top of right column)
        zoom_box = QGroupBox(self.tr("Zoom"))
        zoom_lay = QHBoxLayout(zoom_box)
        zoom_lay.addStretch(1)
        zoom_lay.addWidget(self.btn_zoom_out)
        zoom_lay.addWidget(self.btn_zoom_in)
        zoom_lay.addWidget(self.btn_zoom_fit)
        zoom_lay.addStretch(1)

        right.addWidget(zoom_box)

        # Controls group (already built as `ctrls`)
        right.addWidget(ctrls, 0)

        # Bottom buttons row (already built as `bb`)
        btn_row = QWidget(self)
        btn_row.setLayout(bb)
        right.addWidget(btn_row, 0)

        right.addStretch(1)

        # Wrap right column in a scroll area so small monitors don't get cramped
        right_widget = QWidget(self)
        right_widget.setLayout(right)

        right_scroll = QScrollArea(self)
        right_scroll.setWidgetResizable(True)
        right_scroll.setMinimumWidth(320)   # adjust if you want (300–360 is good)
        right_scroll.setWidget(right_widget)

        root.addWidget(right_scroll, 0)  # RIGHT column becomes LEFT side now
        root.addWidget(sep)
        root.addLayout(left, 1)          # preview becomes RIGHT side now

        # behavior
        self.view.setMouseTracking(True)
        self.view.viewport().installEventFilter(self)

        self.btn_zoom_out.clicked.connect(lambda: self._set_zoom(self._zoom / 1.25))
        self.btn_zoom_in.clicked.connect(lambda: self._set_zoom(self._zoom * 1.25))
        self.btn_zoom_fit.clicked.connect(self._fit_view)

        self.btn_apply.clicked.connect(self._commit_to_doc)
        self.btn_close.clicked.connect(self.reject)
        self.btn_undo.clicked.connect(self._undo_step)
        self.btn_redo.clicked.connect(self._redo_step)
        self.s_radius.valueChanged.connect(self._update_brush_preview)
        self.s_feather.valueChanged.connect(self._update_brush_preview)
        self.s_opacity.valueChanged.connect(self._update_brush_preview)
        self._undo, self._redo = [], []
        self._update_undo_redo_buttons()

        self._update_display_autostretch()
        self._fit_view()
        self._stroke_last_pos: Optional[Tuple[float, float]] = None
        self._stroke_spacing_frac = 0.25  # dab spacing = radius * this

        self._paint_refresh_pending = False
        self._paint_refresh_timer = QTimer(self)
        self._paint_refresh_timer.setSingleShot(True)
        self._paint_refresh_timer.timeout.connect(self._do_paint_refresh)

        # shortcuts
        a_undo = QAction(self); a_undo.setShortcut(QKeySequence.StandardKey.Undo); a_undo.triggered.connect(self._undo_step)
        a_redo = QAction(self); a_redo.setShortcut(QKeySequence.StandardKey.Redo); a_redo.triggered.connect(self._redo_step)
        self.addAction(a_undo); self.addAction(a_redo)
        self._update_brush_preview()

    def _schedule_paint_refresh(self):
        if self._paint_refresh_pending:
            return
        self._paint_refresh_pending = True
        self._paint_refresh_timer.start(33)  # ~30 FPS

    def _do_paint_refresh(self):
        self._paint_refresh_pending = False
        self._display = self._image
        self._refresh_pix()


    # ──────────────────────────────────────────────────────────────────────────
    def _get_mask(self, r: int, feather: float) -> np.ndarray:
        key = (int(r), float(feather))
        if self._mask_cache_key != key or self._mask_cache is None:
            self._mask_cache_key = key
            self._mask_cache = _circle_mask(r, feather)
        return self._mask_cache


    def _clear_source(self):
        self._has_source = False
        self._src_point = (0, 0)
        self.src_x1.setVisible(False)
        self.src_x2.setVisible(False)
        self.link.setVisible(False)

    def _update_undo_redo_buttons(self):
        self.btn_undo.setEnabled(len(self._undo) > 0)
        self.btn_redo.setEnabled(len(self._redo) > 0)

    def _update_display_autostretch(self):
        src = self._image
        if not self.cb_autostretch.isChecked():
            self._display = src.astype(np.float32, copy=False)
            self._refresh_pix()
            return

        tm = float(self.s_target_median.value())
        if not self._orig_mono:
            disp = stretch_color_image(src, target_median=tm, linked=self.cb_linked.isChecked(),
                                      normalize=False, apply_curves=False)
        else:
            mono = src[..., 0]
            mono_st = stretch_mono_image(mono, target_median=tm, normalize=False, apply_curves=False)
            disp = np.stack([mono_st]*3, axis=-1)

        self._display = disp.astype(np.float32, copy=False)
        self._refresh_pix()

    def _update_brush_preview(self):
        w = self.brush_preview.width()
        h = self.brush_preview.height()

        # Use a fixed "preview radius" so the graphic always fills nicely.
        # Feather/opacity are the main teaching tools here.
        r = min(w, h) * 0.42

        feather = float(self.s_feather.value()) / 100.0
        opacity = float(self.s_opacity.value()) / 100.0

        # Build a tiny mask using the same logic as the real stamp.
        # Convert radius -> int pixel radius for the preview buffer.
        pr = int(max(1, round(r)))
        mask = _circle_mask(pr, feather)  # (2pr+1, 2pr+1) float32

        # Make preview canvas
        canvas = np.zeros((h, w), dtype=np.float32)

        # Center the mask
        cy, cx = h // 2, w // 2
        y0 = cy - pr
        x0 = cx - pr
        y1 = y0 + mask.shape[0]
        x1 = x0 + mask.shape[1]

        # Clip just in case
        yy0 = max(0, y0); xx0 = max(0, x0)
        yy1 = min(h, y1); xx1 = min(w, x1)

        my0 = yy0 - y0; mx0 = xx0 - x0
        my1 = my0 + (yy1 - yy0); mx1 = mx0 + (xx1 - xx0)

        canvas[yy0:yy1, xx0:xx1] = mask[my0:my1, mx0:mx1] * opacity

        # Convert to QPixmap (grayscale)
        arr8 = np.ascontiguousarray(np.clip(canvas * 255.0, 0, 255).astype(np.uint8))
        qimg = QImage(arr8.data, w, h, w, QImage.Format.Format_Grayscale8)
        self.brush_preview.setPixmap(QPixmap.fromImage(qimg))


    # ── Event filter
    def eventFilter(self, src, ev):
        if src is self.view.viewport():
            if ev.type() == QEvent.Type.MouseMove:
                pos = self.view.mapToScene(ev.position().toPoint())
                self._on_mouse_move(pos, ev)
                return True

            if ev.type() == QEvent.Type.MouseButtonPress:
                pos = self.view.mapToScene(ev.position().toPoint())
                if ev.button() == Qt.MouseButton.LeftButton:
                    mods = ev.modifiers()
                    if mods & Qt.KeyboardModifier.ControlModifier:
                        self._set_source_at(pos)
                    else:
                        self._start_paint_at(pos)
                    return True

            if ev.type() == QEvent.Type.MouseButtonRelease:
                if ev.button() == Qt.MouseButton.LeftButton:
                    self._end_paint()
                    return True

            if ev.type() == QEvent.Type.Wheel:
                self._wheel_zoom(ev)
                return True

        return super().eventFilter(src, ev)

    def _on_mouse_move(self, scene_pos: QPointF, ev):
        x, y = float(scene_pos.x()), float(scene_pos.y())
        r = int(self.s_radius.value())
        self.circle.setRect(x - r, y - r, 2*r, 2*r)
        self.circle.setVisible(True)

        if self._painting and self._has_source:
            self._paint_segment(scene_pos)

        # Update source overlay while hovering (and while painting)
        self._update_source_overlay(scene_pos)

    def _set_source_at(self, scene_pos: QPointF):
        x, y = int(round(scene_pos.x())), int(round(scene_pos.y()))
        if not (0 <= x < self._image.shape[1] and 0 <= y < self._image.shape[0]):
            return
        self._has_source = True
        self._src_point = (x, y)
        self._last_tgt = None  # reset stroke history
        self._painting = False

        # show X at the chosen source (no offset yet)
        self._draw_source_x(x, y, size=8)
        self.src_x1.setVisible(True)
        self.src_x2.setVisible(True)
        self.link.setVisible(False)

    def _start_paint_at(self, scene_pos: QPointF):
        if not self._has_source:
            QMessageBox.information(self, "Clone Stamp", "Ctrl+Click to set a source point first.")
            return

        tx, ty = int(round(scene_pos.x())), int(round(scene_pos.y()))
        if not (0 <= tx < self._image.shape[1] and 0 <= ty < self._image.shape[0]):
            return

        # Start of stroke: lock offset = src - tgt
        sx, sy = self._src_point
        self._offset = (sx - tx, sy - ty)

        # Push undo snapshot once per stroke
        self._undo.append(self._image.copy())
        self._redo.clear()
        self._update_undo_redo_buttons()

        self._painting = True
        self._stroke_last_pos = (scene_pos.x(), scene_pos.y())
        self._paint_segment(scene_pos)  # do first dab

    def _end_paint(self):
        self._painting = False
        self._last_tgt = None
        self._stroke_last_pos = None
        # Now do the expensive autostretch once:
        self._update_display_autostretch()

    def _paint_segment(self, scene_pos: QPointF):
        if not (self._painting and self._has_source):
            return

        x1, y1 = float(scene_pos.x()), float(scene_pos.y())
        if self._stroke_last_pos is None:
            self._stroke_last_pos = (x1, y1)

        x0, y0 = self._stroke_last_pos
        dx = x1 - x0
        dy = y1 - y0
        dist = float((dx*dx + dy*dy) ** 0.5)

        radius  = int(self.s_radius.value())
        feather = float(self.s_feather.value()) / 100.0
        opacity = float(self.s_opacity.value()) / 100.0

        # spacing: smaller = smoother (but more compute)
        spacing = max(1.0, radius * self._stroke_spacing_frac)
        steps = max(1, int(dist / spacing))

        mask = self._get_mask(radius, feather)
        ox, oy = self._offset

        h, w = self._image.shape[:2]

        # stamp along the line
        for i in range(1, steps + 1):
            t = i / steps
            xt = x0 + dx * t
            yt = y0 + dy * t
            tx, ty = int(round(xt)), int(round(yt))
            if not (0 <= tx < w and 0 <= ty < h):
                continue
            sx, sy = tx + ox, ty + oy
            _blend_clone_inplace(self._image, tx, ty, sx, sy, mask, radius, opacity)

        self._stroke_last_pos = (x1, y1)

        # During painting: do NOT autostretch every dab.
        # Just show linear buffer quickly.
        self._schedule_paint_refresh()


    def _paint_at(self, scene_pos: QPointF):
        tx, ty = int(round(scene_pos.x())), int(round(scene_pos.y()))
        h, w = self._image.shape[:2]
        if not (0 <= tx < w and 0 <= ty < h):
            return

        # spacing: don’t spam dabs if mouse barely moved
        if self._last_tgt is not None:
            lx, ly = self._last_tgt
            if (tx - lx)*(tx - lx) + (ty - ly)*(ty - ly) < 2:  # ~1px
                return
        self._last_tgt = (tx, ty)

        ox, oy = self._offset
        sx, sy = tx + ox, ty + oy

        radius  = int(self.s_radius.value())
        feather = float(self.s_feather.value()) / 100.0
        opacity = float(self.s_opacity.value()) / 100.0

        self._image = _blend_clone(
            self._image, (tx, ty), (sx, sy),
            radius=radius, feather=feather, opacity=opacity
        ).astype(np.float32, copy=False)

        # update display quickly without recomputing expensive stuff every dab:
        # - if autostretch is ON, we still need a rebuild (can be heavy),
        #   but in practice it’s fine; if it’s too slow, we can add a 60–120ms debounce.
        self._update_display_autostretch()

        # update overlays immediately
        self._update_source_overlay(scene_pos)

    def _update_source_overlay(self, tgt_scene_pos: QPointF):
        if not self._has_source:
            self.src_x1.setVisible(False)
            self.src_x2.setVisible(False)
            self.link.setVisible(False)
            return

        tx, ty = float(tgt_scene_pos.x()), float(tgt_scene_pos.y())

        if self._painting:
            # live source follows the cursor via offset
            sx = tx + float(self._offset[0])
            sy = ty + float(self._offset[1])

            self._draw_source_x(sx, sy, size=8)
            self.src_x1.setVisible(True)
            self.src_x2.setVisible(True)

            self.link.setLine(tx, ty, sx, sy)
            self.link.setVisible(True)
        else:
            # not painting: show X at the anchored source point
            sx, sy = self._src_point
            self._draw_source_x(float(sx), float(sy), size=8)
            self.src_x1.setVisible(True)
            self.src_x2.setVisible(True)
            self.link.setVisible(False)

    def _draw_source_x(self, x: float, y: float, size: int = 8):
        s = float(size)
        self.src_x1.setLine(x - s, y - s, x + s, y + s)
        self.src_x2.setLine(x - s, y + s, x + s, y - s)

    # ── Zoom
    def _wheel_zoom(self, ev: QWheelEvent):
        step = 1.25 if ev.angleDelta().y() > 0 else 1/1.25
        self._set_zoom(self._zoom * step)

    def _set_zoom(self, z: float):
        z = float(max(0.05, min(4.0, z)))
        if abs(z - self._zoom) < 1e-4:
            return
        self._zoom = z
        self.view.resetTransform()
        self.view.scale(self._zoom, self._zoom)

    def _fit_view(self):
        if self.pix is None or self.pix.pixmap().isNull():
            return
        br = self.pix.boundingRect()
        if br.isNull():
            return
        self.scene.setSceneRect(br)
        self.view.resetTransform()
        self.view.fitInView(br, Qt.AspectRatioMode.KeepAspectRatio)
        t = self.view.transform()
        self._zoom = t.m11()

    # ── Undo/Redo
    def _undo_step(self):
        if not self._undo:
            return
        self._redo.append(self._image.copy())
        self._image = self._undo.pop()
        self._update_display_autostretch()
        self._update_undo_redo_buttons()

    def _redo_step(self):
        if not self._redo:
            return
        self._undo.append(self._image.copy())
        self._image = self._redo.pop()
        self._update_display_autostretch()
        self._update_undo_redo_buttons()

    # ── Commit
    def _commit_to_doc(self):
        out = self._image
        if self._orig_mono:
            mono = np.mean(out, axis=2, dtype=np.float32)
            if len(self._orig_shape) == 3 and self._orig_shape[2] == 1:
                mono = mono[:, :, None]
            out = mono.astype(np.float32, copy=False)
        else:
            if out.ndim == 2:
                out = np.repeat(out[:, :, None], 3, axis=2)
            elif out.ndim == 3 and out.shape[2] >= 3:
                out = out[:, :, :3]

        out = np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)

        applied = False
        try:
            if hasattr(self._doc, "set_image"):
                self._doc.set_image(out, step_name="Clone Stamp"); applied = True
            elif hasattr(self._doc, "apply_numpy"):
                self._doc.apply_numpy(out, step_name="Clone Stamp"); applied = True
            elif hasattr(self._doc, "image"):
                self._doc.image = out; applied = True
        except Exception as e:
            QMessageBox.critical(self, "Clone Stamp", f"Failed to write to document:\n{e}")
            return

        if applied and hasattr(self.parent(), "_refresh_active_view"):
            try:
                self.parent()._refresh_active_view()
            except Exception:
                pass

        self.accept()

    # ── display helpers
    def _np_to_qpix(self, img: np.ndarray) -> QPixmap:
        arr = np.ascontiguousarray(np.clip(img * 255.0, 0, 255).astype(np.uint8))
        if arr.ndim == 2:
            h, w = arr.shape
            arr = np.repeat(arr[:, :, None], 3, axis=2)
            qimg = QImage(arr.data, w, h, 3*w, QImage.Format.Format_RGB888)
        else:
            h, w, _ = arr.shape
            qimg = QImage(arr.data, w, h, 3*w, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qimg)

    def _refresh_pix(self):
        self.pix.setPixmap(self._np_to_qpix(self._display))
        self.circle.setVisible(False)
