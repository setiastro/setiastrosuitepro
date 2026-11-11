# saspro/mdi_canvas.py
"""
A pannable workspace for the MDI area.

QMdiArea's own scrolling is not usable for this. Measured, on Qt 6:

  * its scroll range is the union of the SUBWINDOW rects, not a canvas, so
    it collapses back to nothing the moment you drag a window home -- it
    can let you reach a stray window but it cannot give you room to lay
    anything out;
  * scrolling rewrites subwindow geometry, so positions are viewport
    coordinates that shift under everything else;
  * and it moves subwindows only. The shortcut chips live on a transparent
    overlay widget parented to the viewport, which would stay nailed to
    the screen while the views scrolled out from under them.

So the canvas is ours. This controller owns an offset -- the canvas
coordinate currently at the viewport's top-left -- and pans by translating
both populations together: every subwindow by -delta, and the chip overlay
(one widget, sized to the whole canvas) by the same amount.

Coordinates
-----------
A subwindow's canvas position is ``widget pos + offset``. The chip overlay
is sized to the canvas and placed at ``-offset``, so a chip's position
inside it IS its canvas position, with no per-chip bookkeeping and no
change to how chips are stored.

That last equivalence is also why **the canvas origin stays pinned at
(0, 0)**: if the origin could move, the overlay's origin would move with
it and every chip would have to be re-registered against the canvas
underneath. The workspace still has to be able to grow at that corner,
though -- a view dragged left has to go somewhere -- so it grows there by
*renumbering*. The overshoot is inserted AT the corner: the offset goes
up, which carries every subwindow's canvas position with it (``widget pos
+ offset``), and the chips are moved by hand, because theirs are stored as
canvas coordinates. ``home_shift`` is the running total of those
insertions. Nothing moves on screen -- offset and overlay shift in step --
the canvas gets wider rather than negative, and the dragged view ends up
at the corner with the rest of the workspace further from it, which is
what dragging it there meant.

The canvas grows on demand at every edge -- push a window or a chip
toward one and the bounds extend past it -- and never shrinks on its own,
because a canvas that collapsed whenever you tidied up would make the
scrollbars jump. Tile and Cascade reset it deliberately, and since both
reposition every view they are also the way back from a view dragged off
into the dark.

Verified against QMdiArea before building on it: subwindow moves are never
clamped by Qt in any direction, setViewportMargins is honoured (so the
scrollbars are real rather than floating over the canvas), and QMdiArea
does not reposition subwindows behind our back when the area resizes.
"""
from __future__ import annotations

from PyQt6.QtCore import QEvent, QObject, QPoint, QRect, QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QCursor
from PyQt6.QtWidgets import QApplication, QMdiSubWindow, QRubberBand


class CanvasController(QObject):
    """Owns the pannable workspace behind a QMdiArea."""

    # offset, bounds or content changed -- the minimap listens to this
    canvasChanged = pyqtSignal()

    # Room added beyond the content whenever the canvas has to grow, so
    # there is always somewhere further to drag to.
    GROW_MARGIN = 240

    # Wheel notch -> pixels panned.
    WHEEL_STEP = 90

    def __init__(self, mdi, parent=None):
        super().__init__(parent or mdi)
        self.mdi = mdi
        self._offset = QPoint(0, 0)
        self._size = QSize(1, 1)          # canvas size; origin is always 0,0
        self._home_shift = QPoint(0, 0)   # room inserted at the home corner
        self._layer = None                # the chip overlay
        self._layer_provider = None
        self._panning = False
        self._pan_anchor = QPoint()
        self._pan_offset0 = QPoint()
        self._suspend = 0
        self._watched = set()

        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(0)
        self._refresh_timer.timeout.connect(self.refresh)

        vp = self.mdi.viewport()
        if vp is not None:
            vp.installEventFilter(self)
        self.mdi.installEventFilter(self)
        try:
            self.mdi.subWindowActivated.connect(self._rewatch)
        except Exception:
            pass

        self._size = QSize(self._viewport_size())
        QTimer.singleShot(0, self.refresh)

    # ── wiring ───────────────────────────────────────────────────────────

    def set_layer_provider(self, fn):
        """Supply a callable returning the current chip overlay.

        A provider rather than a fixed reference because loading a project
        rebuilds the ShortcutManager, and with it the overlay; this way the
        canvas re-adopts the new one by itself instead of needing every
        rebuild site to remember to call back.
        """
        self._layer_provider = fn
        self.schedule_refresh()

    def _resolve_layer(self):
        layer = self._layer
        try:
            if layer is not None and layer.parent() is not None:
                return layer
        except RuntimeError:
            layer = None
        self._layer = None
        if self._layer_provider is not None:
            try:
                layer = self._layer_provider()
            except Exception:
                layer = None
            if layer is not None:
                self._layer = layer
        return self._layer

    def _rewatch(self, *_):
        """Watch subwindows for moves, so the canvas can grow under them."""
        for sw in self.mdi.subWindowList():
            if sw not in self._watched:
                try:
                    sw.installEventFilter(self)
                    self._watched.add(sw)
                except Exception:
                    pass
        self.schedule_refresh()

    # ── geometry ─────────────────────────────────────────────────────────

    def offset(self) -> QPoint:
        return QPoint(self._offset)

    def canvas_rect(self) -> QRect:
        return QRect(QPoint(0, 0), self._size)

    def home_shift(self) -> QPoint:
        """How much room has been inserted at the home corner so far.

        Every canvas coordinate on the workspace -- chips included -- is
        that much higher than it was when the workspace was last at home.
        It is live bookkeeping rather than a placement anyone made, so
        whatever persists chip positions takes it back out again (see
        ``ShortcutManager.save_shortcuts``); a tidy-up returns the chips
        and clears it.
        """
        return QPoint(self._home_shift)

    def _viewport_size(self) -> QSize:
        vp = self.mdi.viewport()
        return QSize(vp.size() if vp is not None else self.mdi.size())

    def viewport_on_canvas(self) -> QRect:
        return QRect(QPoint(self._offset), self._viewport_size())

    def content_rect(self) -> QRect:
        """Union of everything on the canvas, in canvas coords."""
        out = QRect()
        for sw in self.mdi.subWindowList():
            try:
                if not sw.isVisible():
                    continue
                r = sw.geometry().translated(self._offset)
            except RuntimeError:
                continue
            out = r if out.isNull() else out.united(r)

        layer = self._resolve_layer()
        if layer is not None:
            try:
                # The overlay's local coords ARE canvas coords.
                for ch in layer.children():
                    if not (getattr(ch, "isWidgetType", None)
                            and ch.isWidgetType() and ch.isVisible()):
                        continue
                    r = ch.geometry()
                    if r.width() <= 1 or r.height() <= 1:
                        continue
                    # Measure where the chip WANTS to be, not where it is
                    # currently clamped to. A chip restored from a project
                    # beyond the fresh canvas is clamped inside it on the
                    # way in; reading the clamped position would leave the
                    # canvas too small to ever let it back out.
                    want = getattr(ch, "_desired_pos", None)
                    if want is not None:
                        r = QRect(want, r.size())
                    out = r if out.isNull() else out.united(r)
            except RuntimeError:
                pass
        return out

    # ── the canvas itself ────────────────────────────────────────────────

    def schedule_refresh(self):
        if self._suspend:
            return
        self._refresh_timer.start()

    def refresh(self):
        """Keep views inside the canvas, grow it to fit, resync everything."""
        if self._suspend:
            return
        self._suspend += 1
        try:
            self._make_room_at_home()
            vp = self._viewport_size()
            need_w = self._offset.x() + vp.width()
            need_h = self._offset.y() + vp.height()
            content = self.content_rect()
            if not content.isNull():
                # The margin is headroom for the NEXT drag, so it applies
                # only where the content has actually reached the edge.
                # Adding it unconditionally would pin the canvas at
                # content+margin forever and leave the scrollbars up even
                # after a Tile had brought everything back inside.
                # Strictly past, not merely touching: reset_bounds shrinks
                # the canvas to exactly content+1, and >= would regrow it by
                # the margin on the very next refresh -- scrollbars that
                # never go away after a tidy.
                if content.right() + 1 > self._size.width():
                    need_w = max(need_w, content.right() + self.GROW_MARGIN)
                if content.bottom() + 1 > self._size.height():
                    need_h = max(need_h, content.bottom() + self.GROW_MARGIN)
            # Grow only. A canvas that shrank whenever you tidied up would
            # make the scrollbars jump; Tile and Cascade reset it.
            self._size = QSize(max(self._size.width(), int(need_w)),
                               max(self._size.height(), int(need_h)))
            self._clamp_offset()
            self._sync_layer()
            self._sync_scrollbars()
        finally:
            self._suspend -= 1
        self.canvasChanged.emit()

    def _make_room_at_home(self):
        """Grow the canvas under a view dragged past the home corner.

        Qt clamps an interactive drag so the grabbed point of the title
        bar cannot leave the viewport, which already stops the window a
        grip-width outside it. Refusing the rest -- pinning the view at
        canvas 0, as this used to -- is what makes the corner feel like a
        wall: the only way to get a view left of the others was to drag,
        scroll, drag again.

        So take the overshoot and insert it at the corner instead. The
        view keeps the position Qt gave it and the workspace moves up
        around it: the same relative result, reached by growing the
        canvas rather than by going negative, where no scrollbar could
        follow.
        """
        ox, oy = self._offset.x(), self._offset.y()
        left = top = 0
        for sw in self.mdi.subWindowList():
            try:
                if not sw.isVisible():
                    continue
                left = min(left, sw.x() + ox)
                top = min(top, sw.y() + oy)
            except RuntimeError:
                pass
        if left < 0 or top < 0:
            self._insert_room_at_home(-min(0, left), -min(0, top))

    def _insert_room_at_home(self, dx: int, dy: int):
        """Add (dx, dy) of canvas at the origin without moving anything.

        Raising the offset moves every subwindow's canvas position by the
        same amount for free, because a subwindow is stored in viewport
        coordinates. Chips are not: the overlay's local coordinates are
        canvas coordinates, so each chip has to be moved by hand -- and
        its remembered placement with it, or the next reflow would pull
        it back out of step with the views.
        """
        dx, dy = max(0, int(dx)), max(0, int(dy))
        if not (dx or dy):
            return
        self._size = QSize(self._size.width() + dx, self._size.height() + dy)
        self._offset += QPoint(dx, dy)
        self._home_shift += QPoint(dx, dy)
        self._move_chips(dx, dy)
        self._sync_layer()

    def _move_chips(self, dx: int, dy: int):
        """Shift every chip on the overlay by (dx, dy) in canvas coords."""
        layer = self._resolve_layer()
        if layer is None:
            return
        try:
            for ch in layer.children():
                if not (getattr(ch, "isWidgetType", None)
                        and ch.isWidgetType()):
                    continue
                # Not the rubber band: it is placed from scratch every
                # time it is used, and it is not part of the layout.
                if isinstance(ch, QRubberBand):
                    continue
                ch.move(max(0, ch.x() + dx), max(0, ch.y() + dy))
                want = getattr(ch, "_desired_pos", None)
                if want is not None:
                    ch._desired_pos = QPoint(max(0, want.x() + dx),
                                             max(0, want.y() + dy))
        except RuntimeError:
            self._layer = None

    def _return_chips_home(self):
        """Give back the inserted room, now that the views are coming home.

        A tidy-up lays every view out from the origin again, so the room
        inserted at the corner while one of them was dragged left of it
        is gone -- and if the chips kept their share of it, they would
        sit that far right of the views they were parked beside.
        """
        dx, dy = self._home_shift.x(), self._home_shift.y()
        if not (dx or dy):
            return
        self._move_chips(-dx, -dy)
        self._home_shift = QPoint(0, 0)

    def _clamp_offset(self):
        vp = self._viewport_size()
        max_x = max(0, self._size.width() - vp.width())
        max_y = max(0, self._size.height() - vp.height())
        self._offset = QPoint(max(0, min(self._offset.x(), max_x)),
                              max(0, min(self._offset.y(), max_y)))

    def reset_bounds(self):
        """Collapse the canvas back to what is actually in use."""
        self._suspend += 1
        try:
            self._return_chips_home()
            vp = self._viewport_size()
            content = self.content_rect()
            w = self._offset.x() + vp.width()
            h = self._offset.y() + vp.height()
            if not content.isNull():
                w = max(w, content.right() + 1)
                h = max(h, content.bottom() + 1)
            self._size = QSize(int(w), int(h))
        finally:
            self._suspend -= 1
        self.refresh()

    def _sync_layer(self):
        layer = self._resolve_layer()
        if layer is None:
            return
        try:
            want = QRect(QPoint(-self._offset.x(), -self._offset.y()),
                         self._size)
            if layer.geometry() != want:
                layer.setGeometry(want)
        except RuntimeError:
            self._layer = None

    def _sync_scrollbars(self):
        setter = getattr(self.mdi, "sync_canvas_scrollbars", None)
        if setter is None:
            return
        vp = self._viewport_size()
        setter(
            h_max=max(0, self._size.width() - vp.width()),
            h_val=self._offset.x(),
            h_page=vp.width(),
            v_max=max(0, self._size.height() - vp.height()),
            v_val=self._offset.y(),
            v_page=vp.height(),
        )

    # ── panning ──────────────────────────────────────────────────────────

    def pan_to(self, x: int, y: int):
        vp = self._viewport_size()
        max_x = max(0, self._size.width() - vp.width())
        max_y = max(0, self._size.height() - vp.height())
        x = max(0, min(int(x), max_x))
        y = max(0, min(int(y), max_y))
        self.pan_by(x - self._offset.x(), y - self._offset.y())

    def pan_by(self, dx: int, dy: int):
        dx, dy = int(dx), int(dy)
        if dx == 0 and dy == 0:
            return
        self._suspend += 1
        try:
            for sw in self.mdi.subWindowList():
                try:
                    sw.move(sw.x() - dx, sw.y() - dy)
                except RuntimeError:
                    pass
            self._offset += QPoint(dx, dy)
            self._sync_layer()
        finally:
            self._suspend -= 1
        self.refresh()

    def center_on(self, canvas_pt: QPoint):
        vp = self._viewport_size()
        self.pan_to(int(canvas_pt.x()) - vp.width() // 2,
                    int(canvas_pt.y()) - vp.height() // 2)

    def ensure_visible(self, sub: QMdiSubWindow, margin: int = 40):
        """Pan so a given view is on screen, moving as little as possible.

        The rescue path for a view parked off the visible area.
        """
        try:
            r = sub.geometry()
        except RuntimeError:
            return
        vp = QRect(QPoint(0, 0), self._viewport_size())
        dx = dy = 0
        if r.left() < vp.left() + margin:
            dx = r.left() - (vp.left() + margin)
        elif r.right() > vp.right() - margin:
            dx = min(r.left() - (vp.left() + margin),
                     r.right() - (vp.right() - margin))
        if r.top() < vp.top() + margin:
            dy = r.top() - (vp.top() + margin)
        elif r.bottom() > vp.bottom() - margin:
            dy = min(r.top() - (vp.top() + margin),
                     r.bottom() - (vp.bottom() - margin))
        self.pan_by(dx, dy)

    # ── gestures ─────────────────────────────────────────────────────────

    def _begin_pan(self, global_pos):
        self._panning = True
        self._pan_anchor = QPoint(global_pos)
        self._pan_offset0 = QPoint(self._offset)
        QApplication.setOverrideCursor(QCursor(Qt.CursorShape.ClosedHandCursor))

    def _end_pan(self):
        if not self._panning:
            return
        self._panning = False
        try:
            QApplication.restoreOverrideCursor()
        except Exception:
            pass
        self.refresh()

    def eventFilter(self, obj, ev):
        t = ev.type()

        # A view moved or resized -- the canvas may need to grow.
        if t in (QEvent.Type.Move, QEvent.Type.Resize) and \
                isinstance(obj, QMdiSubWindow):
            self.schedule_refresh()
            return False

        if obj is self.mdi and t == QEvent.Type.Resize:
            self.schedule_refresh()
            return False

        # Middle-drag anywhere on the background pans. Taken on the
        # viewport AND on the chip overlay, which covers the viewport and
        # would otherwise swallow it.
        if t == QEvent.Type.MouseButtonPress and \
                ev.button() == Qt.MouseButton.MiddleButton:
            self._begin_pan(ev.globalPosition().toPoint())
            return True

        if t == QEvent.Type.MouseMove and self._panning:
            delta = self._pan_anchor - ev.globalPosition().toPoint()
            target = self._pan_offset0 + delta
            self.pan_to(target.x(), target.y())
            return True

        if t == QEvent.Type.MouseButtonRelease and self._panning and \
                ev.button() == Qt.MouseButton.MiddleButton:
            self._end_pan()
            return True

        return False

    def wheel_pan(self, ev) -> bool:
        """Pan from a wheel event that reached the background.

        Returns True if it was used. Events over a view never get here --
        the view consumes them for its own scrolling and zoom.
        """
        d = ev.angleDelta()
        if d.isNull():
            return False
        steps_y = d.y() / 120.0
        steps_x = d.x() / 120.0
        if ev.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            steps_x, steps_y = steps_y, 0.0
        if steps_x == 0.0 and steps_y == 0.0:
            return False
        before = QPoint(self._offset)
        self.pan_by(int(-steps_x * self.WHEEL_STEP),
                    int(-steps_y * self.WHEEL_STEP))
        return self._offset != before

    # ── scrollbar callbacks ──────────────────────────────────────────────

    def on_hscroll(self, value: int):
        if self._suspend:
            return
        self.pan_to(int(value), self._offset.y())

    def on_vscroll(self, value: int):
        if self._suspend:
            return
        self.pan_to(self._offset.x(), int(value))
