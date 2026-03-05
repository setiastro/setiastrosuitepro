# pro/window_shelf.py
from __future__ import annotations
from PyQt6.QtCore import Qt, QEvent, QTimer, QSize, QObject, QRect
from PyQt6.QtWidgets import QDockWidget, QListWidget, QListWidgetItem, QMdiSubWindow, QWidget
from PyQt6.QtGui import QIcon, QPixmap
import uuid
from PyQt6 import sip # ✅ PyQt6 ships sip; this is what we use

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
        super().__init__(self.tr("Minimized Views"), parent)

        # PyQt6 dock area enum
        self.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)

        self.list = QListWidget(self)
        self.list.setUniformItemSizes(False)
        self.list.setIconSize(QSize(32, 32))
        self.setWidget(self.list)

        # map: token(str) -> QPointer(QMdiSubWindow)
        # map: token(str) -> {"geom": QRect, "max": bool}
        self._tok2sub: dict[str, QMdiSubWindow] = {}
        self._saved_state: dict[str, dict] = {}

        self.list.itemClicked.connect(self._restore)

    # ---- public API used by the interceptor ----
    def pre_capture_state(self, sub: QMdiSubWindow):
        if self._is_dead(sub):
            return

        tok = getattr(sub, "_shelf_token", None)
        if not tok:
            tok = uuid.uuid4().hex
            sub._shelf_token = tok

        was_max = sub.isMaximized()
        g = sub.normalGeometry() if was_max else sub.geometry()
        if not g.isValid():
            g = sub.geometry()

        self._saved_state[tok] = {"geom": QRect(g), "max": bool(was_max)}

    def add_entry(self, sub: QMdiSubWindow):
        if self._is_dead(sub) or sub.widget() is None:
            return

        tok = getattr(sub, "_shelf_token", None)
        if not tok:
            tok = uuid.uuid4().hex
            sub._shelf_token = tok

        # store mapping
        self._tok2sub[tok] = sub

        # auto-remove when Qt deletes the subwindow (Explorer close, etc.)
        try:
            sub.destroyed.connect(lambda *_ , t=tok: self._on_sub_destroyed(t))
        except Exception:
            pass

        title = sub.windowTitle() or self.tr("Untitled")
        while len(title) >= 2 and title[1] == " " and title[0] in "■●◆▲▪▫•◼◻◾◽":
            title = title[2:]
        if title.startswith("Active View: "):
            title = title[len("Active View: "):]

        icon = QIcon()
        w = sub.widget()
        pm = getattr(getattr(w, "label", None), "pixmap", lambda: None)()
        if isinstance(pm, QPixmap) and not pm.isNull():
            icon = QIcon(pm.scaled(64, 64, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        else:
            icon = QIcon.fromTheme("image-x-generic")

        item = QListWidgetItem(icon, title)
        item.setData(Qt.ItemDataRole.UserRole, tok)  # ✅ store token, not sub
        self.list.addItem(item)
        self.show()

    def _on_sub_destroyed(self, tok: str):
        # Called when the subwindow is deleted by Qt
        self._remove_item_for_token(tok)
        self._tok2sub.pop(tok, None)
        self._saved_state.pop(tok, None)
        if self.list.count() == 0:
            self.hide()

    def _remove_item_for_token(self, tok: str):
        for i in range(self.list.count()):
            it = self.list.item(i)
            if it.data(Qt.ItemDataRole.UserRole) == tok:
                self.list.takeItem(i)
                break

    # ---- restore flow ----
    def _restore(self, item: QListWidgetItem):
        tok = item.data(Qt.ItemDataRole.UserRole)
        if not tok:
            return

        # remove shelf item first
        row = self.list.row(item)
        self.list.takeItem(row)

        sub = self._tok2sub.get(tok)

        # if deleted/closed, just cleanup
        if self._is_dead(sub):
            self._tok2sub.pop(tok, None)
            self._saved_state.pop(tok, None)
            if self.list.count() == 0:
                self.hide()
            return

        st = self._saved_state.get(tok, None)

        try:
            if st and st.get("max", False):
                sub.showMaximized()
            else:
                r = QRect(st["geom"]) if (st and isinstance(st.get("geom"), QRect)) else QRect()

                def apply_rect():
                    if self._is_dead(sub):
                        return
                    if r.isValid() and not sub.isMaximized():
                        sub.setWindowState(Qt.WindowState.WindowNoState)
                        sub.resize(r.size())
                        sub.move(r.topLeft())
                        sub.setGeometry(r)

                if r.isValid():
                    sub.setWindowState(Qt.WindowState.WindowNoState)
                    sub.setGeometry(r)
                    sub.resize(r.size())
                    sub.move(r.topLeft())

                sub.showNormal()
                QTimer.singleShot(0, apply_rect)
                QTimer.singleShot(30, apply_rect)
                QTimer.singleShot(120, apply_rect)

            mdi = sub.mdiArea()
            if mdi is not None and not self._is_dead(mdi):
                mdi.setActiveSubWindow(sub)

            sub.raise_()
            sub.activateWindow()
        finally:
            # cleanup saved state for that token once restored
            self._tok2sub.pop(tok, None)
            self._saved_state.pop(tok, None)

            if self.list.count() == 0:
                self.hide()

    def remove_for_subwindow(self, sub):
        if self._is_dead(sub):
            return
        tok = getattr(sub, "_shelf_token", None)
        if not tok:
            return
        self._remove_item_for_token(tok)
        self._tok2sub.pop(tok, None)
        self._saved_state.pop(tok, None)
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

    def _is_dead(self, obj) -> bool:
        try:
            return obj is None or sip.isdeleted(obj)
        except Exception:
            # if sip can’t inspect it, be conservative
            return obj is None

from PyQt6.QtWidgets import QMdiArea

class MinimizeInterceptor(QObject):
    """Redirect native minimize → shelf entry, capturing geometry BEFORE hiding."""
    def __init__(self, shelf: WindowShelf, parent: QWidget | None = None):
        super().__init__(parent)
        self.shelf = shelf

    def eventFilter(self, obj, ev):
        if isinstance(obj, QMdiSubWindow) and ev.type() == QEvent.Type.WindowStateChange:
            if obj.windowState() & Qt.WindowState.WindowMinimized:
                self.shelf.pre_capture_state(obj)
                QTimer.singleShot(0, lambda o=obj: self._redirect(o))
                return True
        return False

    def _redirect(self, sub: QMdiSubWindow):
        # Clear the minimized bit and hide, then add shelf entry
        sub.setWindowState(sub.windowState() & ~Qt.WindowState.WindowMinimized)
        sub.hide()
        self.shelf.add_entry(sub)

        # NEW: pick a new active subwindow (so the hidden one isn't "active")
        QTimer.singleShot(0, lambda s=sub: self._activate_next_visible(s))

    def _activate_next_visible(self, hidden_sub: QMdiSubWindow):
        mdi = hidden_sub.mdiArea()
        if mdi is None:
            return

        # Prefer an order that feels stable to users
        try:
            subs = mdi.subWindowList(QMdiArea.WindowOrder.CreationOrder)
        except Exception:
            subs = mdi.subWindowList()

        # Only candidates that can actually be "active"
        cand = []
        for sw in subs:
            if sw is None or sw is hidden_sub:
                continue
            # hidden subwindows won't accept activation
            if not sw.isVisible():
                continue
            # sanity: ignore minimized (shouldn't happen since you intercept)
            if sw.windowState() & Qt.WindowState.WindowMinimized:
                continue
            cand.append(sw)

        if not cand:
            # Nothing else to activate
            try:
                mdi.setActiveSubWindow(None)  # may be ignored on some platforms
            except Exception:
                pass
            return

        # Pick "next" relative to where the hidden window was in the list
        try:
            idx = subs.index(hidden_sub)
        except Exception:
            idx = -1

        # Rotate forward to the next candidate
        picked = None
        if idx >= 0:
            for off in range(1, len(subs) + 1):
                sw = subs[(idx + off) % len(subs)]
                if sw in cand:
                    picked = sw
                    break
        if picked is None:
            picked = cand[0]

        mdi.setActiveSubWindow(picked)
        picked.raise_()
        picked.widget().setFocus(Qt.FocusReason.OtherFocusReason)
