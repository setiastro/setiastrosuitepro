# pro/window_shelf.py
from __future__ import annotations
from PyQt6.QtCore import Qt, QEvent, QTimer, QSize, QObject, QRect
from PyQt6.QtWidgets import QDockWidget, QListWidget, QListWidgetItem, QMdiSubWindow, QWidget
from PyQt6.QtGui import QIcon, QPixmap

WINDOW_SHELF_DEBUG = False  # flip to True to log capture/restore details

def _dbg(owner, msg: str):
    if not WINDOW_SHELF_DEBUG:
        return
    p = owner.parent()
    if p and hasattr(p, "_log") and callable(getattr(p, "_log")):
        p._log(f"[Shelf] {msg}")
    else:
        print(f"[Shelf] {msg}")

class WindowShelf(QDockWidget):
    def __init__(self, parent=None):
        super().__init__("Minimized Views", parent)

        # PyQt6 dock area enum
        self.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)

        self.list = QListWidget(self)
        self.list.setUniformItemSizes(False)
        self.list.setIconSize(QSize(32, 32))
        self.setWidget(self.list)

        # map: QListWidgetItem -> subwindow
        # use item id() as key because QListWidgetItem is unhashable
        self._item2sub: dict[int, QMdiSubWindow] = {}
        # map: subwindow -> {"geom": QRect, "max": bool}
        self._saved_state: dict[QMdiSubWindow, dict] = {}

        self.list.itemClicked.connect(self._restore)

    # ---- public API used by the interceptor ----
    def pre_capture_state(self, sub: QMdiSubWindow):
        """Capture normal geometry/max state BEFORE we hide/minimize."""
        if not sub:
            return
        was_max = sub.isMaximized()
        # If was maximized, normalGeometry() holds the pre-max rect; otherwise use geometry()
        g = sub.normalGeometry() if was_max else sub.geometry()
        if not g.isValid():
            g = sub.geometry()
        self._saved_state[sub] = {"geom": QRect(g), "max": bool(was_max)}
        _dbg(self, f"CAPTURE for '{sub.windowTitle()}': max={was_max}, geom={g}")

    def add_entry(self, sub: QMdiSubWindow):
        """Add a button to the shelf for `sub` (state must be pre-captured)."""
        if sub is None or sub.widget() is None:
            return

        title = sub.windowTitle() or "Untitled"
        # strip leading dot and Active prefix for the shelf display text only

        # Remove any number of leading glyphs like ■ ● ◆ ▲ etc.
        while len(title) >= 2 and title[1] == " " and title[0] in "■●◆▲▪▫•◼◻◾◽":
            title = title[2:]

        # Remove leading 'Active View: ' if present
        if title.startswith("Active View: "):
            title = title[len("Active View: "):]

        # Best-effort thumbnail from the view's QLabel (if present)
        icon = QIcon()
        w = sub.widget()
        pm = getattr(getattr(w, "label", None), "pixmap", lambda: None)()
        if isinstance(pm, QPixmap) and not pm.isNull():
            icon = QIcon(pm.scaled(
                64, 64,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            ))
        else:
            icon = QIcon.fromTheme("image-x-generic")

        item = QListWidgetItem(icon, title)
        # store the subwindow via item data (so QListWidgetItem doesn't have to be a dict key)
        item.setData(Qt.ItemDataRole.UserRole, sub)
        self._item2sub[id(item)] = sub
        self.list.addItem(item)
        self.show()
        _dbg(self, f"ADD entry for '{title}' (items={self.list.count()})")

    # ---- restore flow ----
    def _restore(self, item: QListWidgetItem):
        sub = item.data(Qt.ItemDataRole.UserRole)
        if not sub:
            return

        # Remove the shelf button first
        row = self.list.row(item)
        self.list.takeItem(row)
        self._item2sub.pop(id(item), None)

        st = self._saved_state.get(sub, None)
        title = sub.windowTitle()
        _dbg(self, f"RESTORE '{title}': have_state={bool(st)}")

        try:
            if st and st.get("max", False):
                _dbg(self, f" → showMaximized()")
                sub.showMaximized()
            else:
                # normal window → restore the exact rectangle
                r = QRect()
                if st and isinstance(st.get("geom"), QRect):
                    r = QRect(st["geom"])
                _dbg(self, f" → target rect={r} (valid={r.isValid()})")

                def apply_rect():
                    if r.isValid() and not sub.isMaximized():
                        # Apply both ways; some styles ignore one or the other during layout churn
                        sub.setWindowState(Qt.WindowState.WindowNoState)
                        sub.resize(r.size())
                        sub.move(r.topLeft())
                        sub.setGeometry(r)
                        _dbg(self, f"   reapplied rect now={sub.geometry()}")

                # Pre-apply (helps avoid the tiny default)
                if r.isValid():
                    sub.setWindowState(Qt.WindowState.WindowNoState)
                    sub.setGeometry(r)
                    sub.resize(r.size())
                    sub.move(r.topLeft())

                sub.showNormal()
                # Once MDI has activated and re-laid out, re-apply a couple of times
                QTimer.singleShot(0, apply_rect)
                QTimer.singleShot(30, apply_rect)
                QTimer.singleShot(120, apply_rect)

            sub.raise_()
            sub.activateWindow()
        finally:
            if self.list.count() == 0:
                self.hide()

    def remove_for_subwindow(self, sub):
        if not sub:
            return
        for i in range(self.list.count()):
            item = self.list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) is sub:
                self._item2sub.pop(id(item), None)
                self.list.takeItem(i)
                break
        self._saved_state.pop(sub, None)   # ← also forget geometry for that sub
        if self.list.count() == 0:
            self.hide()

    def clear_all(self):
        """Remove all thumbnails and forget saved window states."""
        try:
            self.list.blockSignals(True)
            self.list.clear()
        finally:
            self.list.blockSignals(False)
        self._item2sub.clear()
        self._saved_state.clear()
        self.hide()

class MinimizeInterceptor(QObject):
    """Redirect native minimize → shelf entry, capturing geometry BEFORE hiding."""
    def __init__(self, shelf: WindowShelf, parent: QWidget | None = None):
        super().__init__(parent)
        self.shelf = shelf

    def eventFilter(self, obj, ev):
        if isinstance(obj, QMdiSubWindow) and ev.type() == QEvent.Type.WindowStateChange:
            if obj.windowState() & Qt.WindowState.WindowMinimized:
                # Capture state FIRST, then cancel minimize and hide.
                self.shelf.pre_capture_state(obj)
                QTimer.singleShot(0, lambda o=obj: self._redirect(o))
                return True
        return False

    def _redirect(self, sub: QMdiSubWindow):
        # Clear the minimized bit and hide, then add shelf entry
        sub.setWindowState(sub.windowState() & ~Qt.WindowState.WindowMinimized)
        sub.hide()
        self.shelf.add_entry(sub)
