# src/setiastro/saspro/stacking_master_picks.py
# SetiAstro Suite Pro  ·  Franklin Marek  ·  www.setiastro.com
"""
The Dark and Flat rows the Stacking Suite shows under each group in its Flats
and Lights tabs, in Quick Stack's style: a list holding the master that group
will use (Auto, a loaded master, None, or add one) and a badge saying why.

The rows, and each group's "N frames" row, are extra items in those trees.
They carry ROLE_CONTROL, have no file path and cannot be selected; code that
walks a tree for its frames skips them with is_control_item().
"""
from __future__ import annotations

import html
import os

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QTreeWidgetItem, QWidget

# A row's list: its first entry, the way out, and "add one".
AUTO, NONE, BROWSE = "\x00auto", "\x00none", "\x00browse"
BADGE_COLOUR = {"ok": "#3fb950", "warn": "#d29922", "info": "#58a6ff"}

ROLE_CONTROL = int(Qt.ItemDataRole.UserRole) + 40       # "pick" or "frames" on a control row
ROLE_ROWS = int(Qt.ItemDataRole.UserRole) + 41          # on a group: what its control rows were made for
ROLE_FRAMES_SHOWN = int(Qt.ItemDataRole.UserRole) + 42  # on a group: are its frames listed


def norm_path(path) -> str:
    return os.path.normcase(os.path.normpath(str(path or "")))


def is_control_item(item) -> bool:
    """A Dark / Flat row or an "N frames" row, not a group or a frame."""
    try:
        return bool(item is not None and item.data(0, ROLE_CONTROL))
    except RuntimeError:
        return False


def badge_html(parts) -> str:
    """parts: (colour, text) pairs, colour one of BADGE_COLOUR's keys."""
    return "  ".join(f'<span style="color:{BADGE_COLOUR[c]}">{html.escape(t)}</span>' for c, t in parts)


def insert_control_item(parent: QTreeWidgetItem, index: int, kind: str, text: str = "") -> QTreeWidgetItem:
    """A row spanning every column, under `parent` (which must be in a tree)."""
    it = QTreeWidgetItem([text])
    it.setFlags(Qt.ItemFlag.ItemIsEnabled)          # a control, not a selectable row
    it.setData(0, ROLE_CONTROL, kind)
    parent.insertChild(index, it)
    it.setFirstColumnSpanned(True)
    return it


class MasterPickRow(QWidget):
    """One group's Dark or Flat: label, optional tag (the session it is for),
    the list of masters and the badge."""

    picked = pyqtSignal(object, object)     # the row's key; AUTO, NONE, BROWSE or a master's path

    def __init__(self, key, label: str, *, label_width: int = 0, tag: str = "", tag_width: int = 0,
                 tooltip: str = "", parent=None):
        super().__init__(parent)
        self.key = key
        lay = QHBoxLayout(self)
        lay.setContentsMargins(2, 1, 6, 1)
        lay.setSpacing(8)
        self.label = QLabel(label)
        if label_width:
            self.label.setMinimumWidth(label_width)
        lay.addWidget(self.label)
        self.tag = None
        if tag_width:
            self.tag = QLabel(tag)
            self.tag.setMinimumWidth(tag_width)
            self.tag.setStyleSheet("color: palette(mid);")
            lay.addWidget(self.tag)
        self.combo = QComboBox()
        self.combo.setMinimumContentsLength(34)
        self.combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        if tooltip:
            self.combo.setToolTip(tooltip)
        self.badge = QLabel()
        self.badge.setTextFormat(Qt.TextFormat.RichText)
        lay.addWidget(self.combo)
        lay.addWidget(self.badge, 1)
        self.combo.activated.connect(self._on_activated)

    def _on_activated(self, index: int):
        self.picked.emit(self.key, self.combo.itemData(index))

    def set_choices(self, *, auto_text: str, masters, current, none_text: str, browse_text: str,
                    empty_text: str):
        """masters: (text, path, enabled, tooltip) for each loaded master.
        current: AUTO, NONE or a master's path. With no masters loaded the
        list is just empty_text and browse_text."""
        c = self.combo
        c.blockSignals(True)
        c.clear()
        if not masters:
            c.addItem(empty_text, AUTO)
        else:
            c.addItem(auto_text, AUTO)
            c.insertSeparator(c.count())
            for text, path, enabled, tip in masters:
                c.addItem(text, path)
                i = c.count() - 1
                if not enabled:
                    c.model().item(i).setEnabled(False)
                if tip:
                    c.setItemData(i, tip, Qt.ItemDataRole.ToolTipRole)
            c.addItem(none_text, NONE)
        c.insertSeparator(c.count())
        c.addItem(browse_text, BROWSE)
        c.setCurrentIndex(max(0, self._index_of(current)))
        c.blockSignals(False)

    def _index_of(self, data) -> int:
        if data in (AUTO, NONE, None):
            return self.combo.findData(AUTO if data is None else data)
        want = norm_path(data)
        for i in range(self.combo.count()):
            d = self.combo.itemData(i)
            if isinstance(d, str) and not d.startswith("\x00") and norm_path(d) == want:
                return i
        return -1

    def set_badge(self, parts, tooltip: str = ""):
        self.badge.setText(badge_html(parts))
        self.badge.setToolTip(tooltip)
