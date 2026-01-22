# src/setiastro/saspro/clone_stamp.py
from __future__ import annotations

import numpy as np
from typing import Optional, Tuple

from PyQt6.QtCore import Qt, QEvent, QPointF
from PyQt6.QtGui import QImage, QPixmap, QPen, QBrush, QAction, QKeySequence, QColor, QWheelEvent
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox, QLabel, QPushButton, QSlider,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsEllipseItem, QMessageBox,
    QScrollArea, QCheckBox, QDoubleSpinBox, QGraphicsLineItem
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


def _blend_clone(
    img: np.ndarray,
    tgt_xy: Tuple[int, int],
    src_xy: Tuple[int, int],
    radius: int,
    feather: float,
    opacity: float,
) -> np.ndarray:
    """
    Clone-dab: copy from src patch into tgt patch with feathered mask and opacity.
    img is float32 HxWx3 in [0..1].
    Returns new img (copy-on-write for the modified region).
    """
    h, w = img.shape[:2]
    tx, ty = int(tgt_xy[0]), int(tgt_xy[1])
    sx, sy = int(src_xy[0]), int(src_xy[1])
    r = int(max(1, radius))

    # Bounding boxes
    x0 = tx - r; x1 = tx + r + 1
    y0 = ty - r; y1 = ty + r + 1

    # Clip to image
    cx0 = max(0, x0); cx1 = min(w, x1)
    cy0 = max(0, y0); cy1 = min(h, y1)
    if cx0 >= cx1 or cy0 >= cy1:
        return img

    # Corresponding source region (same shape as clipped target)
    # Compute offset from unclipped target window
    ox0 = cx0 - x0
    oy0 = cy0 - y0

    # Source window aligned
    sx0 = (sx - r) + ox0
    sy0 = (sy - r) + oy0
    sx1 = sx0 + (cx1 - cx0)
    sy1 = sy0 + (cy1 - cy0)

    # If source window extends out of bounds, clip both source and target accordingly
    # (keep them same shape)
    adj_cx0, adj_cy0, adj_cx1, adj_cy1 = cx0, cy0, cx1, cy1

    if sx0 < 0:
        shift = -sx0
        sx0 = 0
        adj_cx0 += shift
    if sy0 < 0:
        shift = -sy0
        sy0 = 0
        adj_cy0 += shift
    if sx1 > w:
        shift = sx1 - w
        sx1 = w
        adj_cx1 -= shift
    if sy1 > h:
        shift = sy1 - h
        sy1 = h
        adj_cy1 -= shift

    if adj_cx0 >= adj_cx1 or adj_cy0 >= adj_cy1:
        return img

    # Recompute final shapes
    tw = adj_cx1 - adj_cx0
    th = adj_cy1 - adj_cy0
    if tw <= 0 or th <= 0:
        return img

    tgt_patch = img[adj_cy0:adj_cy1, adj_cx0:adj_cx1, :]
    src_patch = img[sy0:sy0+th, sx0:sx0+tw, :]

    # Mask slice for this clipped patch
    mask_full = _circle_mask(r, feather)
    mx0 = adj_cx0 - x0
    my0 = adj_cy0 - y0
    mask = mask_full[my0:my0+th, mx0:mx0+tw].astype(np.float32)

    a = float(np.clip(opacity, 0.0, 1.0)) * mask
    if a.max() <= 0:
        return img

    out = img.copy()
    # broadcast mask to 3 channels
    a3 = a[:, :, None]
    out[adj_cy0:adj_cy1, adj_cx0:adj_cx1, :] = (1.0 - a3) * tgt_patch + a3 * src_patch
    return out


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
        self.cb_autostretch.setChecked(True)
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

        main = QVBoxLayout(self)
        main.addWidget(self.scroll)

        zoom_bar = QHBoxLayout()
        zoom_bar.addStretch()
        zoom_bar.addWidget(self.btn_zoom_out)
        zoom_bar.addWidget(self.btn_zoom_in)
        zoom_bar.addWidget(self.btn_zoom_fit)
        zoom_bar.addStretch()
        main.addLayout(zoom_bar)

        main.addWidget(ctrls)
        main.addLayout(bb)

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

        self._undo, self._redo = [], []
        self._update_undo_redo_buttons()

        self._update_display_autostretch()
        self._fit_view()

        # shortcuts
        a_undo = QAction(self); a_undo.setShortcut(QKeySequence.StandardKey.Undo); a_undo.triggered.connect(self._undo_step)
        a_redo = QAction(self); a_redo.setShortcut(QKeySequence.StandardKey.Redo); a_redo.triggered.connect(self._redo_step)
        self.addAction(a_undo); self.addAction(a_redo)

    # ──────────────────────────────────────────────────────────────────────────

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
            self._paint_at(scene_pos)

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
        self._last_tgt = None
        self._paint_at(scene_pos)

    def _end_paint(self):
        self._painting = False
        self._last_tgt = None
        # refresh display (autostretch) once at end too
        self._update_display_autostretch()

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
