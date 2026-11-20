# saspro/gui/command_palette.py
"""
Centred command palette for Ctrl+Shift+P and Ctrl+/ (Command on macOS).

Search Commands in Help (macOS) or About (Windows/Linux) opens this popup.
The Command Search dock is the normal home of command search, but Ctrl+Shift+P
used to do nothing at all unless that dock's search box was already on screen:
closed, collapsed into its rail, or tabbed behind another panel, the shortcut
focused an invisible line edit. This palette is what the shortcut opens in
those cases -- one search box over the middle of the main window, the same
commands from the same model, ranked by name, description and fuzzy matches.

It is a Qt Popup, so a click anywhere else closes it, as does Esc or pressing
Ctrl+Shift+P a second time. It owns its own result list rather than a
QCompleter: a completer's popup is itself a Popup, and nesting one inside
another is exactly the kind of focus juggling this should not depend on.
"""
from __future__ import annotations

from PyQt6.QtCore import Qt, QEvent, QSize, QModelIndex
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import (
    QFrame, QVBoxLayout, QLineEdit, QListView, QAbstractItemView, QApplication,
)

from setiastro.saspro.gui.command_search import (
    CommandSearchProxy, MAX_QUERY_LENGTH,
)

_VISIBLE_ROWS = 12
PALETTE_SHORTCUTS = ("Ctrl+/", "Ctrl+Num+/")


def palette_action(host, callback):
    """Keep one window-owned shortcut action across menu rebuilds.

    Register directly on the main window as well as in its menu. Numeric
    keypad division carries a distinct KeypadModifier on Windows/Linux.
    """
    action = host.__dict__.get("_command_palette_action")
    if action is None:
        action = QAction(host)
        action.setShortcuts([QKeySequence(key) for key in PALETTE_SHORTCUTS])
        action.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
        action.setAutoRepeat(False)
        action.setMenuRole(QAction.MenuRole.NoRole)
        action.triggered.connect(callback)
        host.addAction(action)
        host._command_palette_action = action
    action.setText(host.tr("Search Commands..."))
    return action


class CommandPalette(QFrame):
    """
    `model` is the main window's command model (one QStandardItem per QAction).
    `run_index(source_index)` runs the command a source-model index points at;
    the palette has already closed itself by the time it is called, so a
    command that opens a modal dialog does not open it underneath the popup.
    """

    def __init__(self, parent, model, run_index):
        super().__init__(
            parent, Qt.WindowType.Popup | Qt.WindowType.FramelessWindowHint,
        )
        self.setObjectName("CommandPalette")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet(
            "QFrame#CommandPalette { border: 1px solid palette(highlight); }"
        )
        self._run_index = run_index

        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)

        self.edit = QLineEdit(self)
        self.edit.setMaxLength(MAX_QUERY_LENGTH)
        self.edit.setToolTip(self.tr(
            "Search command names and descriptions (up to 256 characters)."
        ))
        self.edit.setPlaceholderText(self.tr(
            "Search commands...  (Enter runs, Esc closes)"
        ))
        self.edit.setClearButtonEnabled(True)
        lay.addWidget(self.edit)

        self.proxy = CommandSearchProxy(self)
        self.proxy.setSourceModel(model)

        self.list = QListView(self)
        self.list.setModel(self.proxy)
        self.list.setIconSize(QSize(18, 18))
        self.list.setUniformItemSizes(True)
        self.list.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.list.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.list.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        # Typing never leaves the search box; the list only shows and clicks.
        self.list.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        lay.addWidget(self.list, 1)

        self.edit.textChanged.connect(self._on_text_changed)
        self.edit.installEventFilter(self)
        self.list.clicked.connect(self._run_proxy_index)
        self.proxy.modelReset.connect(self._select_best)
        self.proxy.rowsInserted.connect(self._select_best)
        self.proxy.rowsRemoved.connect(self._select_best)

    # ---- open / close -------------------------------------------------------
    def popup_centered(self):
        """Show over the parent window, search box empty and focused."""
        mw = self.parentWidget()
        self.edit.clear()                       # resets the filter too
        self._select_best()

        row_h = (max(self.list.sizeHintForRow(0), 20)
                 if self.proxy.rowCount() else 24)
        list_h = row_h * _VISIBLE_ROWS + 2 * self.list.frameWidth() + 4
        w = 640
        if mw is not None:
            w = int(max(420, min(640, mw.width() * 0.6)))
        h = self.edit.sizeHint().height() + list_h + 8 + 8 + 6
        screen = (mw.screen() if mw is not None
                  else QApplication.primaryScreen())
        scr = screen.availableGeometry() if screen is not None else None
        if scr is not None:
            w = min(w, scr.width())
            h = min(h, scr.height())
        self.resize(w, h)

        if mw is not None:
            c = mw.mapToGlobal(mw.rect().center())
            x, y = c.x() - w // 2, c.y() - h // 2
            if scr is not None:
                x = max(scr.left(), min(x, scr.right() - w + 1))
                y = max(scr.top(), min(y, scr.bottom() - h + 1))
            self.move(x, y)

        self.show()
        self.raise_()
        self.activateWindow()
        self.edit.setFocus(Qt.FocusReason.PopupFocusReason)

    # ---- filtering ----------------------------------------------------------
    def _on_text_changed(self, text: str):
        self.proxy.set_query(text or "")
        self._select_best()

    def _select_best(self):
        """Select the best result, preserving proxy/source mapping."""
        n = self.proxy.rowCount()
        if n == 0:
            self.list.clearSelection()
            return
        needle = (self.edit.text() or "").strip().casefold()
        row = 0
        if needle:
            for r in range(n):
                t = (self.proxy.index(r, 0).data(
                    Qt.ItemDataRole.DisplayRole) or "").strip()
                if t.casefold() == needle:
                    row = r
                    break
        idx = self.proxy.index(row, 0)
        self.list.setCurrentIndex(idx)
        self.list.scrollTo(idx)

    def _move(self, step: int):
        n = self.proxy.rowCount()
        if n == 0:
            return
        cur = self.list.currentIndex()
        row = cur.row() if cur.isValid() else 0
        row = max(0, min(n - 1, row + step))
        idx = self.proxy.index(row, 0)
        self.list.setCurrentIndex(idx)
        self.list.scrollTo(idx)

    # ---- running ------------------------------------------------------------
    def _run_proxy_index(self, idx: QModelIndex):
        if not self.isVisible() or not idx.isValid():
            return
        src = self.proxy.mapToSource(idx)
        if not src.isValid():
            return
        self.hide()
        self._run_index(src)

    def eventFilter(self, obj, ev):
        if obj is self.edit and ev.type() == QEvent.Type.KeyPress:
            key = ev.key()
            if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                if not ev.isAutoRepeat():
                    self._run_proxy_index(self.list.currentIndex())
                return True
            if key == Qt.Key.Key_Escape:
                self.hide()
                return True
            if key == Qt.Key.Key_Down:
                self._move(1)
                return True
            if key == Qt.Key.Key_Up:
                self._move(-1)
                return True
            if key == Qt.Key.Key_PageDown:
                self._move(_VISIBLE_ROWS)
                return True
            if key == Qt.Key.Key_PageUp:
                self._move(-_VISIBLE_ROWS)
                return True
            # The popup holds the keyboard, so the main window's Ctrl+Shift+P
            # shortcut cannot fire while it is open. Handle the toggle here.
            seq = QKeySequence(ev.keyCombination())
            toggles = ("Ctrl+Shift+P", *PALETTE_SHORTCUTS)
            if any(
                seq.matches(QKeySequence(keys))
                == QKeySequence.SequenceMatch.ExactMatch
                for keys in toggles
            ):
                if not ev.isAutoRepeat():
                    self.hide()
                return True
        return super().eventFilter(obj, ev)
