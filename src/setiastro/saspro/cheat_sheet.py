# pro/cheat_sheet.py
"""
Keyboard shortcut cheat sheet dialog.

Displays all keyboard shortcuts and mouse gestures in a tabbed dialog.
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, QWidget,
    QTableWidget, QTableWidgetItem, QHeaderView, QPushButton,
    QApplication, QMessageBox
)
from PyQt6.QtGui import QAction, QShortcut, QKeySequence


def _qs_to_str(seq: QKeySequence) -> str:
    """Convert a QKeySequence to a human-readable string."""
    return seq.toString(QKeySequence.SequenceFormat.NativeText).strip()


def _clean_text(text: str) -> str:
    """Remove common Unicode decoration from text."""
    if not text:
        return ""
    # Remove common ellipsis, arrows, etc.
    return text.replace("…", "").replace("→", "->").replace("←", "<-").strip()


def _uniq_keep_order(items):
    """Return unique items preserving order."""
    seen = set()
    out = []
    for x in items:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def _seqs_for_action(act: QAction):
    """Get non-empty key sequences for an action."""
    seqs = [s for s in act.shortcuts() or []] or ([act.shortcut()] if act.shortcut() else [])
    return [s for s in seqs if not s.isEmpty()]


def _where_for_action(act: QAction) -> str:
    """Determine where an action is available (Menus/Toolbar or Window)."""
    if act.parent():
        pn = act.parent().__class__.__name__
        if pn.startswith("QMenu") or pn.startswith("QToolBar"):
            from PyQt6.QtCore import QCoreApplication
            return QCoreApplication.translate("CheatSheet", "Menus/Toolbar")
    return QCoreApplication.translate("CheatSheet", "Window")


def _describe_action(act: QAction) -> str:
    """Get a human-readable description for an action."""
    desc = _clean_text(act.statusTip() or act.toolTip() or act.text() or act.objectName() or "Action")
    if desc == "Action":
        from PyQt6.QtCore import QCoreApplication
        desc = QCoreApplication.translate("CheatSheet", "Action")
    return desc


def _describe_shortcut(sc: QShortcut) -> str:
    """Get a human-readable description for a shortcut."""
    desc = _clean_text(sc.property("hint") or sc.whatsThis() or sc.objectName() or "Shortcut")
    if desc == "Shortcut":
        from PyQt6.QtCore import QCoreApplication
        desc = QCoreApplication.translate("CheatSheet", "Shortcut")
    return desc


def _where_for_shortcut(sc: QShortcut) -> str:
    """Determine where a shortcut is available."""
    par = sc.parent()
    from PyQt6.QtCore import QCoreApplication
    return par.__class__.__name__ if par is not None else QCoreApplication.translate("CheatSheet", "Window")


class CheatSheetDialog(QDialog):
    """
    Dialog showing all keyboard shortcuts and mouse gestures.
    
    Displays two tabs:
    - Keyboard shortcuts (from QActions)
    - Mouse/drag gestures
    """
    
    def __init__(self, parent, keyboard_rows, gesture_rows):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Keyboard Shortcut Cheat Sheet"))
        self.resize(780, 520)
        
        self._keyboard_rows = keyboard_rows
        self._gesture_rows = gesture_rows

        tabs = QTabWidget(self)

        # --- Keyboard tab ---
        pg_keys = QWidget(tabs)
        v1 = QVBoxLayout(pg_keys)
        tbl_keys = QTableWidget(0, 3, pg_keys)
        tbl_keys.setHorizontalHeaderLabels([self.tr("Shortcut"), self.tr("Action"), self.tr("Where")])
        tbl_keys.verticalHeader().setVisible(False)
        tbl_keys.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        tbl_keys.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        tbl_keys.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        tbl_keys.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        tbl_keys.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        tbl_keys.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        v1.addWidget(tbl_keys)

        # Populate keyboard shortcuts
        for s, action, where in keyboard_rows:
            r = tbl_keys.rowCount()
            tbl_keys.insertRow(r)
            tbl_keys.setItem(r, 0, QTableWidgetItem(s))
            tbl_keys.setItem(r, 1, QTableWidgetItem(action))
            tbl_keys.setItem(r, 2, QTableWidgetItem(where))

        # --- Mouse/Drag tab ---
        pg_mouse = QWidget(tabs)
        v2 = QVBoxLayout(pg_mouse)
        tbl_mouse = QTableWidget(0, 3, pg_mouse)
        tbl_mouse.setHorizontalHeaderLabels([self.tr("Gesture"), self.tr("Context"), self.tr("Effect")])
        tbl_mouse.verticalHeader().setVisible(False)
        tbl_mouse.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        tbl_mouse.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        tbl_mouse.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        tbl_mouse.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        tbl_mouse.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        tbl_mouse.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        v2.addWidget(tbl_mouse)

        # Populate mouse gestures
        for gesture, context, effect in gesture_rows:
            r = tbl_mouse.rowCount()
            tbl_mouse.insertRow(r)
            tbl_mouse.setItem(r, 0, QTableWidgetItem(gesture))
            tbl_mouse.setItem(r, 1, QTableWidgetItem(context))
            tbl_mouse.setItem(r, 2, QTableWidgetItem(effect))

        tabs.addTab(pg_keys, self.tr("Base Keyboard"))
        tabs.addTab(pg_mouse, self.tr("Additional & Mouse & Drag"))

        # Buttons
        btns = QHBoxLayout()
        btns.addStretch(1)

        b_copy = QPushButton(self.tr("Copy"))
        b_copy.clicked.connect(self._copy_all)
        b_close = QPushButton(self.tr("Close"))
        b_close.clicked.connect(self.accept)
        btns.addWidget(b_copy)
        btns.addWidget(b_close)

        top = QVBoxLayout(self)
        top.addWidget(tabs)
        top.addLayout(btns)

    def _copy_all(self):
        """Copy all shortcuts to clipboard as plain text."""
        lines = []
        lines.append("== Keyboard ==")
        for s, a, w in self._keyboard_rows:
            lines.append(f"{s:20}  {a}  [{w}]")
        lines.append("")
        lines.append("== Mouse & Drag ==")
        for g, c, e in self._gesture_rows:
            lines.append(f"{g:24}  {c:18}  {e}")
        QApplication.clipboard().setText("\n".join(lines))
        QMessageBox.information(self, self.tr("Copied"), self.tr("Cheat sheet copied to clipboard."))


# Legacy alias for backward compatibility
_CheatSheetDialog = CheatSheetDialog
