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
            xs.update([r.left(), r.right(), r.center().x()])
            ys.update([r.top(), r.bottom(), r.center().y()])

        # viewport edges and center
        xs.update([vp.left(), vp.right(), vp.center().x()])
        ys.update([vp.top(), vp.bottom(), vp.center().y()])
        return sorted(xs), sorted(ys), rects, vp

    @staticmethod
    def _nearest(value: int, candidates: list[int], tol: int) -> tuple[bool, int]:
        best = None
        best_d = tol + 1
        for c in candidates:
            d = abs(c - value)
            if d < best_d:
                best_d = d
                best = c
        return (best is not None and best_d <= tol), (value if best is None else best)

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

    def _snap_geometry(self, g: QRect, xs: list[int], ys: list[int], tol: int, size_snap: bool) -> tuple[QRect, list[QRect]]:
        L, T, R, B = g.left(), g.top(), g.right(), g.bottom()
        snapped = False

        ok, nx = self._nearest(L, xs, tol);  L = nx if ok else L; snapped = snapped or ok
        ok, nx = self._nearest(R, xs, tol);  R = nx if ok else R; snapped = snapped or ok
        ok, ny = self._nearest(T, ys, tol);  T = ny if ok else T; snapped = snapped or ok
        ok, ny = self._nearest(B, ys, tol);  B = ny if ok else B; snapped = snapped or ok

        # QRect from left/top + size is least error-prone with setGeometry
        new_pos = QPoint(min(L, R), min(T, B))
        new_size = QSize(abs(R - L) + 1, abs(B - T) + 1)
        g2 = QRect(new_pos, new_size)

        # Optional: width/height match is implicitly achieved when edges line up
        _ = size_snap  # reserved for future refinements

        guides = (self._build_guides(g2, self.view.rect()) if (snapped and self._show_guides) else [])
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
            snapped_rect, guides = self._snap_geometry(cur, xs, ys, tol, size_snap=True)

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
