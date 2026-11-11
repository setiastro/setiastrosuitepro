# saspro/widgets/minimap_dock.py
"""
Overview of the workspace canvas.

QMdiArea cannot scale its children, so there is no way to zoom the
workspace out and look at the whole thing. This is the substitute: the
canvas drawn to scale, with every view as a rectangle, the chips as dots,
and the part you are currently looking at as a box you can drag.

  * drag the viewport box (or anywhere on the canvas) to pan
  * click a view to activate it
  * double-click a view to centre the canvas on it

It is a plain QWidget repainted from the controller's canvasChanged
signal, plus a slow timer while visible -- QMdiSubWindow has no "I moved"
signal, so a drag in progress has to be sampled.
"""
from __future__ import annotations

import os

from PyQt6.QtCore import QPoint, QPointF, QRect, QRectF, Qt, QTimer
from PyQt6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QDockWidget, QSizePolicy, QWidget


def minimap_icon(size: int = 24) -> QIcon:
    """The dock's rail icon.

    The artwork is `minimappanel.png` with the rest of the rail family,
    so an icon pack themes it like any other panel. The painted version
    below is the fallback, and is what this looked like before the
    artwork landed -- keep it working: it is what a stripped or
    half-installed copy falls back to.
    """
    try:
        from setiastro.saspro.resources import get_icon_path
        path = get_icon_path("minimappanel")
        if path and os.path.exists(path):
            ic = QIcon(path)
            if not ic.isNull():
                return ic
    except Exception:
        pass

    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    pen = QPen(QColor(200, 200, 200))
    pen.setWidthF(1.4)
    p.setPen(pen)
    m = size * 0.12
    p.drawRect(QRectF(m, m, size - 2 * m, size - 2 * m))
    p.fillRect(QRectF(size * 0.22, size * 0.24, size * 0.3, size * 0.26),
               QColor(150, 150, 150))
    p.fillRect(QRectF(size * 0.46, size * 0.5, size * 0.3, size * 0.26),
               QColor(110, 110, 110))
    pen.setColor(QColor(80, 190, 160))
    pen.setWidthF(1.6)
    p.setPen(pen)
    p.drawRect(QRectF(size * 0.3, size * 0.34, size * 0.46, size * 0.4))
    p.end()
    return QIcon(pm)


class MinimapView(QWidget):
    PAD = 6

    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self._mw = main_window
        self._drag = None               # canvas point grabbed, or None
        self._rects = []                # [(QRect canvas, subwindow)]
        self.setMinimumSize(140, 110)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self.setToolTip(
            "Workspace overview.\n\n"
            "Drag to pan the canvas, click a view to activate it, "
            "double-click to centre on it.")

        self._timer = QTimer(self)
        self._timer.setInterval(400)
        self._timer.timeout.connect(self._tick)

    # ── plumbing ─────────────────────────────────────────────────────────

    def _canvas(self):
        mdi = getattr(self._mw, "mdi", None)
        return getattr(mdi, "canvas", None) if mdi is not None else None

    def showEvent(self, e):
        super().showEvent(e)
        self._timer.start()

    def hideEvent(self, e):
        super().hideEvent(e)
        self._timer.stop()

    def _tick(self):
        # Subwindows move without telling anyone; sample while visible.
        if self.isVisible():
            self.update()

    # ── mapping ──────────────────────────────────────────────────────────

    def _scale_and_origin(self):
        c = self._canvas()
        if c is None:
            return None
        bounds = c.canvas_rect()
        if bounds.width() <= 0 or bounds.height() <= 0:
            return None
        w = max(1, self.width() - 2 * self.PAD)
        h = max(1, self.height() - 2 * self.PAD)
        k = min(w / bounds.width(), h / bounds.height())
        ox = self.PAD + (w - bounds.width() * k) / 2.0
        oy = self.PAD + (h - bounds.height() * k) / 2.0
        return bounds, k, QPointF(ox, oy)

    def _to_widget(self, canvas_rect, m) -> QRectF:
        bounds, k, o = m
        return QRectF(
            o.x() + (canvas_rect.x() - bounds.x()) * k,
            o.y() + (canvas_rect.y() - bounds.y()) * k,
            max(1.0, canvas_rect.width() * k),
            max(1.0, canvas_rect.height() * k))

    def _to_canvas(self, pt, m) -> QPoint:
        bounds, k, o = m
        return QPoint(int(bounds.x() + (pt.x() - o.x()) / k),
                      int(bounds.y() + (pt.y() - o.y()) / k))

    # ── painting ─────────────────────────────────────────────────────────

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        pal = self.palette()
        p.fillRect(self.rect(), pal.base())

        c = self._canvas()
        m = self._scale_and_origin()
        if c is None or m is None:
            p.setPen(pal.placeholderText().color())
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                       "No workspace")
            return

        bounds, _k, _o = m
        canvas_r = self._to_widget(bounds, m)
        p.setPen(QPen(pal.mid().color(), 1))
        p.setBrush(pal.window())
        p.drawRect(canvas_r)

        mdi = self._mw.mdi
        try:
            active = mdi.activeSubWindow()
        except Exception:
            active = None

        # chips, as faint dots
        layer = getattr(c, "_layer", None)
        if layer is not None:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(140, 140, 140, 120))
            try:
                off = c.offset()
                for ch in layer.children():
                    if not (getattr(ch, "isWidgetType", None)
                            and ch.isWidgetType() and ch.isVisible()):
                        continue
                    r = ch.geometry().translated(layer.pos()).translated(off)
                    if r.width() <= 1 or r.height() <= 1:
                        continue
                    wr = self._to_widget(r, m)
                    p.drawEllipse(wr.center(), 2.2, 2.2)
            except RuntimeError:
                pass

        # views
        self._rects = []
        accent = QColor(61, 191, 159)          # SASpro teal
        for sw in mdi.subWindowList():
            try:
                if not sw.isVisible():
                    continue
                cr = sw.geometry().translated(c.offset())
            except RuntimeError:
                continue
            self._rects.append((cr, sw))
            wr = self._to_widget(cr, m)
            is_active = (sw is active)
            p.setBrush(QColor(accent.red(), accent.green(), accent.blue(),
                              90 if is_active else 45))
            p.setPen(QPen(accent if is_active else pal.mid().color(),
                          1.6 if is_active else 1.0))
            p.drawRect(wr)

        # the part of the canvas on screen
        vp = c.viewport_on_canvas()
        vr = self._to_widget(vp, m)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(pal.highlight().color(), 2))
        p.drawRect(vr)
        p.end()

    # ── interaction ──────────────────────────────────────────────────────

    def _hit(self, pos):
        m = self._scale_and_origin()
        if m is None:
            return None
        for cr, sw in reversed(self._rects):
            if self._to_widget(cr, m).contains(QPointF(pos)):
                return sw
        return None

    def mousePressEvent(self, e):
        if e.button() != Qt.MouseButton.LeftButton:
            return
        c = self._canvas()
        m = self._scale_and_origin()
        if c is None or m is None:
            return
        sw = self._hit(e.position().toPoint())
        if sw is not None:
            try:
                self._mw.mdi.setActiveSubWindow(sw)
            except Exception:
                pass
        self._drag = self._to_canvas(e.position(), m)
        c.center_on(self._drag)
        self.update()

    def mouseMoveEvent(self, e):
        if self._drag is None:
            return
        c = self._canvas()
        m = self._scale_and_origin()
        if c is None or m is None:
            return
        c.center_on(self._to_canvas(e.position(), m))
        self.update()

    def mouseReleaseEvent(self, _e):
        self._drag = None

    def mouseDoubleClickEvent(self, e):
        c = self._canvas()
        sw = self._hit(e.position().toPoint())
        if c is None or sw is None:
            return
        try:
            self._mw.mdi.setActiveSubWindow(sw)
            c.center_on(sw.geometry().translated(c.offset()).center())
        except Exception:
            pass
        self.update()


class MinimapDock(QDockWidget):
    def __init__(self, main_window):
        super().__init__("Workspace Map", main_window)
        self.setObjectName("MinimapDock")
        self.view = MinimapView(main_window, self)
        self.setWidget(self.view)
        self.setWindowIcon(minimap_icon())
        self.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)

        mdi = getattr(main_window, "mdi", None)
        canvas = getattr(mdi, "canvas", None) if mdi is not None else None
        if canvas is not None:
            try:
                canvas.canvasChanged.connect(self.view.update)
            except Exception:
                pass
