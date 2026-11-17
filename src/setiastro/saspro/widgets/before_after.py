# saspro/widgets/before_after.py
"""
Before/after comparison for tool preview panes, plus a display stretch that
both sides share.

The host keeps its own QGraphicsView subclass (Convo's view picks the
Larson-Sekanina centre on Shift+click; Multiscale's view drives its ROI
preview from pan/zoom). BeforeAfterCompare layers onto that view rather than
replacing it:

  * a "before" pixmap underneath the host's own "after" item(s),
  * a clip rect the after items are reparented into, for the split wipe,
  * a divider you can drag in the preview, and
  * a toolbar: After / Before / Split / Side by side, Hold to compare (or hold
    Space), divider position and axis.

Mode, divider position and axis persist under the host's settings key.
CLAHE (clahe.py) predates this module and keeps its own built-in copy.
"""
from __future__ import annotations

import math

import numpy as np
from PyQt6.QtCore import (
    Qt, QObject, QEvent, QPointF, QRectF, QSettings, QCoreApplication, pyqtSignal,
)
from PyQt6.QtGui import QBrush, QColor, QCursor, QPainterPath, QPen, QPixmap
from PyQt6.QtWidgets import (
    QButtonGroup, QGraphicsEllipseItem, QGraphicsItem, QGraphicsLineItem,
    QGraphicsPathItem, QGraphicsPixmapItem, QGraphicsRectItem, QHBoxLayout,
    QLabel, QSlider, QToolButton, QWidget,
)


# ---------------------------------------------------------------------------
# Display stretch shared by both sides of a comparison
# ---------------------------------------------------------------------------
def display_stretch_params(ref, target_median: float = 0.25, linked: bool = True,
                           blackpoint_sigma: float = 5.0, ignore_zeros: bool = False):
    """Measure a statistical display stretch on `ref`, once.

    Same math as imageops.stretch.stretch_mono_image / stretch_color_image on
    their default path (no curves, HDR, normalise or high-range), split into
    measure + apply so a before/after pair goes through ONE stretch. Stretching
    each side on its own statistics makes a result look brighter or darker
    than the original for no real reason (a denoise lowers the noise, which
    moves the black point), and the comparison stops meaning anything.

    ignore_zeros leaves 0.0 (no-data) pixels out of the statistics, as the main
    view's autostretch does.
    """
    from setiastro.saspro.imageops.stretch import _compute_blackpoint_sigma

    def _measure(x):
        v = np.asarray(x, dtype=np.float32)
        if ignore_zeros:
            nz = v[v > 0.0]
            if nz.size:
                v = nz
        bp, med = _compute_blackpoint_sigma(v, float(blackpoint_sigma))
        return float(bp), float(med)

    a = np.asarray(ref, dtype=np.float32)
    if a.ndim == 3 and a.shape[2] == 1:
        a = a[..., 0]
    t = float(target_median)
    if a.ndim == 2:
        return ("mono",) + _measure(a) + (t,)
    a = a[..., :3]
    if linked:
        return ("linked",) + _measure(a) + (t,)
    per = [_measure(a[..., c]) for c in range(a.shape[2])]
    bp3 = np.array([p[0] for p in per], dtype=np.float32)
    med3 = np.array([p[1] for p in per], dtype=np.float32)
    return ("unlinked", bp3, med3, t)


def apply_display_stretch(img, params) -> np.ndarray:
    """Apply display_stretch_params() output to `img`.

    The curve is per pixel, so mono parameters also serve a 3-channel display
    copy of a mono image. Keeps the input layout; float32 in [0, 1].
    """
    from setiastro.saspro.imageops import stretch as _st

    kind, bp, med, t = params
    a = np.asarray(img, dtype=np.float32)
    if a.ndim == 3 and a.shape[2] == 1:
        a = a[..., 0]
    # The public stretch functions hold this lock around the same kernels:
    # two numba parallel regions entered concurrently abort the process.
    with _st._parallel_lock():
        if kind == "unlinked":
            bp3 = np.asarray(bp, dtype=np.float32)
            if a.ndim == 2:
                a = np.repeat(a[..., None], bp3.size, axis=2)
            denom3 = np.maximum(1.0 - bp3, 1e-12).astype(np.float32)
            meds3 = ((np.asarray(med, dtype=np.float32) - bp3) / denom3).astype(np.float32)
            out = _st.numba_color_unlinked_from_img(
                np.ascontiguousarray(a[..., :bp3.size]), bp3, denom3, meds3, t)
        else:
            denom = max(1.0 - float(bp), 1e-12)
            med_r = (float(med) - float(bp)) / denom
            if a.ndim == 2:
                out = _st.numba_mono_from_img(
                    np.ascontiguousarray(a), float(bp), denom, med_r, t)
            else:
                out = _st.numba_color_linked_from_img(
                    np.ascontiguousarray(a[..., :3]), float(bp), denom, med_r, t)
    return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Comparison controller
# ---------------------------------------------------------------------------
MODE_AFTER, MODE_BEFORE, MODE_SPLIT, MODE_SBS = 0, 1, 2, 3

_SBS_GAP = 4.0                        # scene px between the halves (as CLAHE)
_Z_BEFORE, _Z_AFTER = -3.0, -2.0      # under host overlays such as the LS crosshair
_Z_DIVIDER = 50.0


def _tr(text: str) -> str:
    return QCoreApplication.translate("BeforeAfterCompare", text)


class BeforeAfterCompare(QObject):
    """Before/after comparison layered onto an existing view.

    Host contract:
      * pass the item(s) that show the result as `after_items`; they are
        reparented into a clip group (scene coordinates do not change),
      * put `toolbar` somewhere in the layout,
      * call set_image_size(w, h) instead of setting the scene rect yourself
        (side by side needs room for the second copy),
      * call set_before(pixmap) with the untouched image through the SAME
        display stretch as the result, or None when there is nothing to
        compare yet,
      * connect fitRequested to the host's fit action, and call detach() from
        closeEvent before anything clears the scene.
    """

    fitRequested = pyqtSignal()   # side by side switched on/off: the layout changed

    def __init__(self, view, scene, after_items, *, settings_key: str, host: QWidget):
        super().__init__(host)
        self._view = view
        self._viewport = view.viewport()
        self._scene = scene
        self._host = host
        self._skey = str(settings_key)
        self._w = 0
        self._h = 0
        self._has_before = False
        self._hold = False
        self._mode = MODE_AFTER
        self._split_frac = 0.5
        self._vertical_line = True        # vertical divider = wipe left/right
        self._drag = False
        self._cursor_saved = None         # (viewport had its own cursor, QCursor)
        self._detached = False

        self._before = QGraphicsPixmapItem()
        self._before.setZValue(_Z_BEFORE)
        self._before_sbs = QGraphicsPixmapItem()     # left copy for side by side
        self._before_sbs.setZValue(_Z_BEFORE)
        for it in (self._before, self._before_sbs):
            it.setVisible(False)
            scene.addItem(it)

        # The after items live inside this rect. Clipping is only switched on in
        # split mode, so moving the divider is a setRect() with no pixel work.
        self._clip = QGraphicsRectItem()
        self._clip.setPen(QPen(Qt.PenStyle.NoPen))
        self._clip.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        self._clip.setZValue(_Z_AFTER)
        scene.addItem(self._clip)
        for it in after_items:
            it.setParentItem(self._clip)

        # Cosmetic pens stay one screen pixel at any zoom; the dark line under
        # the light one keeps the divider visible on bright and dark frames.
        self._divider_bg = QGraphicsLineItem()
        self._divider_bg.setPen(QPen(QColor(0, 0, 0, 190), 3))
        self._divider_bg.setZValue(_Z_DIVIDER)
        self._divider = QGraphicsLineItem()
        self._divider.setPen(QPen(QColor(255, 255, 255, 230), 0))
        self._divider.setZValue(_Z_DIVIDER + 1)
        self._grip = QGraphicsEllipseItem(-13.0, -13.0, 26.0, 26.0)
        self._grip.setBrush(QBrush(QColor(255, 255, 255, 235)))
        self._grip.setPen(QPen(QColor(0, 0, 0, 190), 1))
        self._grip.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
        self._grip.setZValue(_Z_DIVIDER + 2)
        chev = QGraphicsPathItem(self._grip)
        path = QPainterPath()
        path.moveTo(-8.0, 0.0); path.lineTo(-3.0, -5.0); path.lineTo(-3.0, 5.0)
        path.closeSubpath()
        path.moveTo(8.0, 0.0); path.lineTo(3.0, -5.0); path.lineTo(3.0, 5.0)
        path.closeSubpath()
        chev.setPath(path)
        chev.setBrush(QBrush(QColor(25, 25, 25)))
        chev.setPen(QPen(Qt.PenStyle.NoPen))
        for it in (self._divider_bg, self._divider, self._grip):
            it.setVisible(False)
            scene.addItem(it)

        self.toolbar = self._build_toolbar()
        self._restore()
        self._viewport.installEventFilter(self)
        host.installEventFilter(self)
        self._apply()

    # ---------------- toolbar ----------------
    def _build_toolbar(self) -> QWidget:
        bar = QWidget()
        row = QHBoxLayout(bar)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)

        self._grp = QButtonGroup(bar)
        self._grp.setExclusive(True)
        for mid, text, tip in (
            (MODE_AFTER,  _tr("After"),        _tr("Show the processed result")),
            (MODE_BEFORE, _tr("Before"),       _tr("Show the untouched image")),
            (MODE_SPLIT,  _tr("Split"),        _tr("One frame, divider you can drag")),
            (MODE_SBS,    _tr("Side by side"), _tr("Before and after, left and right")),
        ):
            b = QToolButton(bar)
            b.setText(text)
            b.setToolTip(tip)
            b.setCheckable(True)
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            self._grp.addButton(b, mid)
            row.addWidget(b)
        self._grp.idClicked.connect(self._on_mode_clicked)

        self._btn_hold = QToolButton(bar)
        self._btn_hold.setText(_tr("Hold to compare"))
        self._btn_hold.setToolTip(_tr("Press and hold (or hold Space) to peek at the original"))
        self._btn_hold.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._btn_hold.pressed.connect(lambda: self.set_hold(True))
        self._btn_hold.released.connect(lambda: self.set_hold(False))
        row.addSpacing(10)
        row.addWidget(self._btn_hold)

        self._lbl_split = QLabel(_tr("Divider:"), bar)
        self._s_split = QSlider(Qt.Orientation.Horizontal, bar)
        self._s_split.setRange(0, 1000)
        self._s_split.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._s_split.setMaximumWidth(140)
        self._s_split.valueChanged.connect(self._on_split_slider)
        self._s_split.sliderReleased.connect(self._save)
        self._btn_axis = QToolButton(bar)
        self._btn_axis.setCheckable(True)
        self._btn_axis.setToolTip(_tr("Divider orientation — vertical wipes left/right, "
                                      "horizontal wipes top/bottom"))
        self._btn_axis.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._btn_axis.toggled.connect(self._on_axis_toggled)
        row.addSpacing(10)
        row.addWidget(self._lbl_split)
        row.addWidget(self._s_split)
        row.addWidget(self._btn_axis)
        # Keeps the buttons packed left when the divider controls are hidden
        # and the bar is wider than its contents.
        row.addStretch(1)
        return bar

    # ---------------- settings ----------------
    def _restore(self):
        mode, frac, vertical = MODE_AFTER, 0.5, True
        try:
            s = QSettings()
            mode = int(s.value(f"{self._skey}/mode", MODE_AFTER))
            frac = float(s.value(f"{self._skey}/split", 0.5))
            vertical = s.value(f"{self._skey}/split_vertical", True, type=bool)
        except Exception:
            pass
        self._mode = mode if mode in (MODE_AFTER, MODE_BEFORE, MODE_SPLIT, MODE_SBS) else MODE_AFTER
        self._split_frac = max(0.0, min(1.0, frac))
        self._vertical_line = bool(vertical)
        self._grp.button(self._mode).setChecked(True)
        for w in (self._s_split, self._btn_axis):
            w.blockSignals(True)
        self._s_split.setValue(int(round(self._split_frac * 1000)))
        self._btn_axis.setChecked(not self._vertical_line)
        for w in (self._s_split, self._btn_axis):
            w.blockSignals(False)
        self._sync_axis_text()

    def _save(self):
        try:
            s = QSettings()
            s.setValue(f"{self._skey}/mode", int(self._mode))
            s.setValue(f"{self._skey}/split", float(self._split_frac))
            s.setValue(f"{self._skey}/split_vertical", bool(self._vertical_line))
        except Exception:
            pass

    # ---------------- host API ----------------
    @property
    def has_before(self) -> bool:
        return self._has_before

    def set_image_size(self, w: int, h: int):
        """Image size in scene px; owns the scene rect from here on."""
        if self._detached:
            return
        self._w, self._h = max(0, int(w)), max(0, int(h))
        self._before_sbs.setPos(-(self._w + _SBS_GAP), 0.0)
        self._apply()

    def set_before(self, pixmap: QPixmap | None):
        """The untouched image, or None when there is nothing to compare yet."""
        if self._detached:
            return
        pm = pixmap if (pixmap is not None and not pixmap.isNull()) else QPixmap()
        had = self._has_before
        self._before.setPixmap(pm)
        self._before_sbs.setPixmap(pm)     # QPixmap is implicitly shared: no copy
        self._has_before = not pm.isNull()
        if not self._has_before:
            self._hold = False
        self._apply()
        if had != self._has_before and self._mode == MODE_SBS:
            self.fitRequested.emit()

    def set_hold(self, on: bool):
        on = bool(on) and self._has_before
        if self._detached or on == self._hold:
            return
        self._hold = on
        self._apply()

    def detach(self):
        """Stop listening and drop the before pixmap. Call from closeEvent,
        before anything clears the scene; every other call is a no-op after."""
        if self._detached:
            return
        self._save()
        self._drag = False
        for obj in (self._viewport, self._host):
            try:
                obj.removeEventFilter(self)
            except Exception:
                pass
        try:
            self._restore_cursor()
            self._before.setPixmap(QPixmap())
            self._before_sbs.setPixmap(QPixmap())
        except Exception:
            pass
        self._detached = True

    # ---------------- composition ----------------
    def _apply(self):
        if self._detached:
            return
        sel = self._mode if self._has_before else MODE_AFTER
        eff = MODE_BEFORE if (self._hold and self._has_before) else sel
        split = (eff == MODE_SPLIT)
        self._before_sbs.setVisible(sel == MODE_SBS)
        self._before.setVisible(eff in (MODE_BEFORE, MODE_SPLIT))
        self._clip.setVisible(eff != MODE_BEFORE)
        self._clip.setFlag(QGraphicsItem.GraphicsItemFlag.ItemClipsChildrenToShape, split)
        for it in (self._divider_bg, self._divider, self._grip):
            it.setVisible(split)
        if split:
            self._update_divider()
        else:
            self._drag = False
            self._restore_cursor()
        self._update_scene_rect()
        self._sync_controls()

    def _update_scene_rect(self):
        w, h = float(self._w), float(self._h)
        if w <= 0 or h <= 0:
            return
        if self._has_before and self._mode == MODE_SBS:
            r = QRectF(-(w + _SBS_GAP), 0.0, 2.0 * w + _SBS_GAP, h)
        else:
            r = QRectF(0.0, 0.0, w, h)
        if self._scene.sceneRect() != r:
            self._scene.setSceneRect(r)

    def _update_divider(self):
        """Geometry only, so dragging stays smooth at any image size."""
        w, h = float(self._w), float(self._h)
        if w <= 0 or h <= 0:
            return
        f = max(0.0, min(1.0, float(self._split_frac)))
        if self._vertical_line:
            x = f * w
            self._clip.setRect(x, 0.0, w - x, h)
            for ln in (self._divider, self._divider_bg):
                ln.setLine(x, 0.0, x, h)
            self._grip.setPos(x, h * 0.5)
            self._grip.setRotation(0.0)
        else:
            y = f * h
            self._clip.setRect(0.0, y, w, h - y)
            for ln in (self._divider, self._divider_bg):
                ln.setLine(0.0, y, w, y)
            self._grip.setPos(w * 0.5, y)
            self._grip.setRotation(90.0)

    def _sync_controls(self):
        on = self._has_before
        for b in self._grp.buttons():
            b.setEnabled(on)
        self._btn_hold.setEnabled(on)
        show_split = (self._mode == MODE_SPLIT)
        for w in (self._lbl_split, self._s_split, self._btn_axis):
            w.setVisible(show_split)
            w.setEnabled(on)

    def _sync_axis_text(self):
        self._btn_axis.setText(_tr("Vertical") if self._vertical_line else _tr("Horizontal"))

    # ---------------- toolbar slots ----------------
    def _on_mode_clicked(self, mid: int):
        was_sbs = (self._mode == MODE_SBS)
        self._mode = int(mid)
        self._save()
        self._apply()
        if self._has_before and was_sbs != (self._mode == MODE_SBS):
            self.fitRequested.emit()

    def _on_split_slider(self, val: int):
        self._split_frac = float(val) / 1000.0
        self._update_divider()
        if not self._s_split.isSliderDown():
            self._save()

    def _on_axis_toggled(self, horizontal_line: bool):
        self._vertical_line = not bool(horizontal_line)
        self._sync_axis_text()
        self._save()
        self._update_divider()

    # ---------------- events ----------------
    def eventFilter(self, obj, ev):
        try:
            et = ev.type()
            if obj is self._host:
                if et in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease) \
                        and ev.key() == Qt.Key.Key_Space and not ev.isAutoRepeat() \
                        and self._has_before:
                    self.set_hold(et == QEvent.Type.KeyPress)
                    return True
                if et == QEvent.Type.WindowDeactivate:
                    self.set_hold(False)      # a release we never see must not stick
                return False
            if obj is self._viewport and et in (QEvent.Type.MouseButtonPress,
                                                QEvent.Type.MouseMove,
                                                QEvent.Type.MouseButtonRelease):
                return self._divider_mouse(ev, et)
        except Exception:
            pass
        return False

    def _divider_mouse(self, ev, et) -> bool:
        """Drag the split divider directly in the preview.

        Runs on the viewport before the host view sees the event, so the
        host's pan/zoom and Shift+click handling are untouched unless the
        cursor is on the divider.
        """
        split = self._has_before and not self._hold and self._mode == MODE_SPLIT
        if not split or self._w <= 0 or self._h <= 0:
            self._drag = False
            return False
        w, h = float(self._w), float(self._h)
        vert = self._vertical_line
        anchor = (QPointF(self._split_frac * w, h * 0.5) if vert
                  else QPointF(w * 0.5, self._split_frac * h))
        va = self._view.mapFromScene(anchor)
        pos = ev.position()
        dx, dy = pos.x() - va.x(), pos.y() - va.y()
        near = (abs(dx) if vert else abs(dy)) <= 10.0 or math.hypot(dx, dy) <= 18.0

        if et == QEvent.Type.MouseButtonPress:
            if (ev.button() == Qt.MouseButton.LeftButton and near
                    and not (ev.modifiers() & Qt.KeyboardModifier.ShiftModifier)):
                self._drag = True
                return True
            return False
        if et == QEvent.Type.MouseMove:
            if self._drag:
                sp = self._view.mapToScene(pos.toPoint())
                f = max(0.0, min(1.0, float(sp.x() / w if vert else sp.y() / h)))
                self._split_frac = f
                self._s_split.blockSignals(True)
                self._s_split.setValue(int(round(f * 1000)))
                self._s_split.blockSignals(False)
                self._update_divider()
                return True
            if ev.buttons() == Qt.MouseButton.NoButton:
                self._hover_cursor(near)
            return False
        if et == QEvent.Type.MouseButtonRelease and self._drag:
            self._drag = False
            self._save()
            return True
        return False

    def _hover_cursor(self, near: bool):
        vp = self._viewport
        if not near:
            self._restore_cursor()
            return
        if self._cursor_saved is None:
            self._cursor_saved = (vp.testAttribute(Qt.WidgetAttribute.WA_SetCursor),
                                  QCursor(vp.cursor()))
        shape = (Qt.CursorShape.SplitHCursor if self._vertical_line
                 else Qt.CursorShape.SplitVCursor)
        if vp.cursor().shape() != shape:
            vp.setCursor(QCursor(shape))

    def _restore_cursor(self):
        """Hand the viewport back the cursor the host view had set (e.g. the
        open hand of ScrollHandDrag), or none at all if it had none."""
        saved, self._cursor_saved = self._cursor_saved, None
        if saved is None:
            return
        had_own, cur = saved
        try:
            if had_own:
                self._viewport.setCursor(cur)
            else:
                self._viewport.unsetCursor()
        except Exception:
            pass
