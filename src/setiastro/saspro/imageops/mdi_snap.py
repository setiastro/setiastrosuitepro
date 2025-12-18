# pro/mdi_snap.py
from __future__ import annotations
from PyQt6.QtCore import Qt, QRect, QPoint, QObject, QEvent, QSize
from PyQt6.QtGui import QPainter, QPen, QPalette
from PyQt6.QtWidgets import QWidget, QMdiArea, QMdiSubWindow

def _dpi_scaled(widget: QWidget, px: int) -> int:
    try:
        ratio = float(widget.devicePixelRatioF())
    except Exception:
        ratio = 1.0
    return max(1, int(round(px * ratio)))

class _GuideOverlay(QWidget):
    """Thin, non-interactive overlay that draws snap guides on the MDI viewport."""
    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self._rects: list[QRect] = []
        self.hide()

    def set_guides(self, rects: list[QRect]):
        self._rects = rects or []
        if self._rects:
            if self.isHidden():
                self.show()
        else:
            if self.isVisible():
                self.hide()
        self.update()

    def paintEvent(self, _ev):
        if not self._rects:
            return
        p = QPainter(self)
        pen = QPen()
        pen.setWidth(1)
        pen.setColor(self.palette().color(QPalette.ColorRole.Highlight))
        pen.setStyle(Qt.PenStyle.SolidLine)
        p.setPen(pen)
        for r in self._rects:
            p.drawRect(r)

class MdiSnapController(QObject):
    """
    Adds 'sticky' snapping for QMdiSubWindow edges inside a QMdiArea (Qt6-safe).
    - Snaps to sibling subwindow edges and viewport edges.
    - Shows faint guide lines while snapping.
    - Hold Alt to temporarily disable snapping.
    """
    def __init__(self, mdi: QMdiArea, threshold_px: int = 8, show_guides: bool = False):
        super().__init__(mdi)
        self.mdi = mdi
        self.view = mdi.viewport()  # geometry space of subwindows
        self.overlay = _GuideOverlay(self.view)
        self.threshold = max(1, int(threshold_px))
        self._active: QMdiSubWindow | None = None
        self._snap_enabled = True
        self._show_guides = bool(show_guides)
        self._install()

    def set_show_guides(self, enabled: bool):
        self._show_guides = bool(enabled)
        if not self._show_guides:
            self.overlay.set_guides([])

    # --- public knobs ---
    def set_threshold(self, px: int):
        self.threshold = max(1, int(px))

    def install_on(self, sub: QMdiSubWindow):
        # Avoid double-install
        sub.removeEventFilter(self)
        sub.installEventFilter(self)

    # --- internals ---
    def _install(self):
        # Track active subwindow
        self.mdi.subWindowActivated.connect(self._on_activated)

        # Watch existing subs
        for sw in self.mdi.subWindowList():
            self.install_on(sw)

        # Keep overlay matched to viewport size/visibility
        self.view.installEventFilter(self)
        self.overlay.setGeometry(self.view.rect())

        # Periodically re-check list on activation (new windows)
        self.mdi.subWindowActivated.connect(lambda _sw: self._refresh_watch_list())

    def _refresh_watch_list(self):
        for sw in self.mdi.subWindowList():
            self.install_on(sw)

    def _on_activated(self, sw: QMdiSubWindow | None):
        self._active = sw

    # Gather candidate snap edges (x/y positions) from siblings & viewport
    def _collect_edges(self, ignore: QMdiSubWindow | None):
        siblings = [s for s in self.mdi.subWindowList() if s is not ignore]
        vp = self.view.rect()

        xs, ys = set(), set()
        rects = []
        for s in siblings:
            r = s.geometry()  # already in viewport coords
            rects.append(r)
            xs.update([r.left(), r.right()])
            ys.update([r.top(), r.bottom()])

        # viewport edges (no center)
        xs.update([vp.left(), vp.right()])
        ys.update([vp.top(), vp.bottom()])
        return sorted(xs), sorted(ys), rects, vp

    @staticmethod
    def _nearest(value: int, candidates: list[int], tol: int) -> tuple[bool, int]:
        """
        Return (True, snapped_value) if any candidate is within tol of value,
        otherwise (False, value).
        """
        best_val = value
        best_d = tol + 1
        for c in candidates:
            d = abs(c - value)
            if d < best_d:
                best_d = d
                best_val = c
        if best_d <= tol:
            return True, best_val
        return False, value

    def _build_guides(self, snap_rect: QRect, vp: QRect) -> list[QRect]:
        """Horizontal and vertical guides along the snapped rect edges."""
        lines: list[QRect] = []
        w = _dpi_scaled(self.view, 2)
        # horizontal
        lines.append(QRect(vp.left(), snap_rect.top(), vp.width(), w))
        lines.append(QRect(vp.left(), snap_rect.bottom(), vp.width(), w))
        # vertical
        lines.append(QRect(snap_rect.left(), vp.top(), w, vp.height()))
        lines.append(QRect(snap_rect.right(), vp.top(), w, vp.height()))
        return lines

    def _snap_geometry(
        self,
        g: QRect,
        xs: list[int],
        ys: list[int],
        tol: int,
        size_snap: bool
    ) -> tuple[QRect, list[QRect]]:
        """
        - If size_snap is False (Move): keep W/H fixed and move the rect so the
          nearest edges line up with candidates.
        - If size_snap is True (Resize): keep top/left fixed and adjust W/H so
          right/bottom edges snap to nearby candidates.
        """
        L, T = g.left(), g.top()
        W, H = g.width(), g.height()
        R, B = L + W - 1, T + H - 1

        vp = self.view.rect()
        snapped = False

        if not size_snap:
            # --- MOVE MODE: translate the rect, no size change ---
            okL, snapL = self._nearest(L, xs, tol)
            okR, snapR = self._nearest(R, xs, tol)

            dx = 0
            if okL and okR:
                dL = snapL - L
                dR = snapR - R
                dx = dL if abs(dL) <= abs(dR) else dR
                snapped = True
            elif okL:
                dx = snapL - L
                snapped = True
            elif okR:
                dx = snapR - R
                snapped = True

            okT, snapT = self._nearest(T, ys, tol)
            okB, snapB = self._nearest(B, ys, tol)

            dy = 0
            if okT and okB:
                dT = snapT - T
                dB = snapB - B
                dy = dT if abs(dT) <= abs(dB) else dB
                snapped = True
            elif okT:
                dy = snapT - T
                snapped = True
            elif okB:
                dy = snapB - B
                snapped = True

            new_L = L + dx
            new_T = T + dy
            g2 = QRect(QPoint(new_L, new_T), QSize(W, H))

        else:
            # --- RESIZE MODE: keep L/T fixed, snap R/B by changing W/H ---
            okR, snapR = self._nearest(R, xs, tol)
            okB, snapB = self._nearest(B, ys, tol)

            new_W = W
            new_H = H

            if okR:
                new_W = max(1, (snapR - L + 1))
                snapped = True
            if okB:
                new_H = max(1, (snapB - T + 1))
                snapped = True

            g2 = QRect(QPoint(L, T), QSize(new_W, new_H))

        guides = (
            self._build_guides(g2, vp)
            if (snapped and self._show_guides)
            else []
        )
        return g2, guides
    
    # --- Event filter on each subwindow + viewport ---
    def eventFilter(self, obj: QObject, ev: QEvent) -> bool:
        t = ev.type()

        # Keep overlay sized to the viewport
        if obj is self.view:
            if t == QEvent.Type.Resize:
                self.overlay.setGeometry(self.view.rect())
                self.overlay.update()
            elif t in _CLEAR_EVENTS:
                self.overlay.set_guides([])
            return False

        if not isinstance(obj, QMdiSubWindow):
            return super().eventFilter(obj, ev)

        # Alt disables snapping while held
        try:
            mods = obj.window().keyboardModifiers()
            self._snap_enabled = not bool(mods & Qt.KeyboardModifier.AltModifier)
        except Exception:
            self._snap_enabled = True

        if t in (QEvent.Type.Move, QEvent.Type.Resize):
            if not self._snap_enabled:
                self.overlay.set_guides([])
                return super().eventFilter(obj, ev)

            xs, ys, _rects, _vp = self._collect_edges(ignore=obj)
            tol = _dpi_scaled(self.view, self.threshold)
            cur = obj.geometry()

            # <<< key change: size_snap only during Resize >>>
            size_snap = (t == QEvent.Type.Resize)
            snapped_rect, guides = self._snap_geometry(
                cur, xs, ys, tol, size_snap=size_snap
            )

            if snapped_rect != cur:
                # Minimize feedback loops
                obj.blockSignals(True)
                try:
                    obj.setGeometry(snapped_rect)
                finally:
                    obj.blockSignals(False)

            self.overlay.set_guides(guides if (self._show_guides and guides) else [])
            return False

        # Qt6-safe "clear overlay" conditions
        if t in (QEvent.Type.Hide, QEvent.Type.Leave):
            if self._show_guides:
                self.overlay.set_guides([])

        return super().eventFilter(obj, ev)


# ---- Qt6-safe clear events set (no MoveAboutToBeAnimated) -------------------
_CLEAR_EVENTS = set()
for _name in ("Hide", "Leave", "MouseButtonRelease", "WindowDeactivate", "FocusOut"):
    _val = getattr(QEvent.Type, _name, None)
    if _val is not None:
        _CLEAR_EVENTS.add(_val)
