# src/setiastro/saspro/icon_table.py
"""
Icon Table — PixInsight-style categorised shortcut panel.

A floating window that lets the user organise process icons into named,
collapsible categories.  Every icon in the table:
  - Runs its command (with any attached preset) on double-click.
  - Can be dragged to a view / canvas / anywhere the existing MIME_CMD
    consumers accept.  Dropping on the MDI creates a canvas shortcut;
    dropping directly on an image applies headlessly — same as the
    toolbar buttons do today.
  - Has a right-click menu: Run, Edit Preset…, Rename…, Copy to Canvas,
    Move to Category…, Remove from Table.

Adding items:
  - The "+ Add Item" button below any category opens a searchable menu of
    every registered command; picking one appends it to that category.
  - Dragging a toolbar button (which emits MIME_CMD or MIME_ACTION) onto
    a category also adds it there.

Reordering / moving:
  - Drag an item within its category to reorder.
  - Drag it onto another category's header (or body) to move it.

Categories:
  - "+ Add Category" button at the bottom of the window creates a new one.
  - Right-click a category header for Rename / Delete / Move Up / Move
    Down / Collapse.

Persistence:
  - Serialised as JSON under QSettings key `icon_table/v1`.

Wire-up (in main_window.py):
  - Store an instance:      self._icon_table = None
  - Menu handler:
        def _open_icon_table(self):
            from setiastro.saspro.icon_table import open_icon_table
            self._icon_table = open_icon_table(self, self.shortcuts, self._icon_table)
"""
from __future__ import annotations
import json
import platform
from typing import Optional, List, Dict, Any

from PyQt6.QtCore import Qt, QSize, QMimeData, QPoint, QRect, QSettings
from PyQt6.QtGui import (
    QAction, QIcon, QPixmap, QDrag, QDragEnterEvent, QDropEvent,
    QMouseEvent, QContextMenuEvent,
)
from PyQt6.QtWidgets import (
    QDialog, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QToolButton,
    QMenu, QMessageBox, QInputDialog, QPushButton, QScrollArea, QFrame,
    QSizePolicy, QLayout, QStyle, QLineEdit, QGridLayout,
)

# Reuse the same MIME vocabulary the toolbar / canvas already speak.
try:
    from setiastro.saspro.dnd_mime import MIME_CMD, MIME_ACTION
except Exception:
    # Fallback strings if that module isn't importable in dev — must match
    # the actual mime types shortcuts.py uses at runtime for drop-target
    # interop.
    MIME_CMD = "application/x-saspro-cmd"
    MIME_ACTION = "application/x-saspro-action"

# --- QSettings namespace --------------------------------------------------
_QS_KEY = "icon_table/v1"
_DEFAULT_ICON_PX = 32


# --- pack/unpack for MIME_CMD (mirrors _pack_cmd_payload in shortcuts.py) --
def _pack_cmd_payload(command_id: str, preset: dict | None = None) -> bytes:
    payload = {"command_id": command_id, "preset": preset or {}}
    return json.dumps(payload).encode("utf-8")


# =========================================================================
# Flow layout — reflows child widgets on resize (classic Qt example)
# =========================================================================
class _FlowLayout(QLayout):
    def __init__(self, parent=None, margin=6, h_spacing=6, v_spacing=6):
        super().__init__(parent)
        if parent is not None:
            self.setContentsMargins(margin, margin, margin, margin)
        self._h = h_spacing
        self._v = v_spacing
        self._items: list = []

    def __del__(self):
        while self._items:
            self._items.pop()

    def addItem(self, item):
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, idx):
        if 0 <= idx < len(self._items):
            return self._items[idx]
        return None

    def takeAt(self, idx):
        if 0 <= idx < len(self._items):
            return self._items.pop(idx)
        return None

    def expandingDirections(self):
        return Qt.Orientation(0)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width: int) -> int:
        return self._do_layout(QRect(0, 0, width, 0), test_only=True)

    def setGeometry(self, rect: QRect):
        super().setGeometry(rect)
        self._do_layout(rect, test_only=False)

    def sizeHint(self):
        # The natural size for a flow layout depends on the target width,
        # which we don't know here.  Report the widest single item as the
        # width hint and let heightForWidth() drive the actual height.
        w = 0
        h = 0
        for item in self._items:
            hint = item.sizeHint()
            w = max(w, hint.width())
            h = max(h, hint.height())
        m = self.contentsMargins()
        # At minimum, give parent a size that fits one row.  The real height
        # comes from heightForWidth() once the parent has been given a width.
        return QSize(w + m.left() + m.right(), h + m.top() + m.bottom())

    def minimumSize(self):
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        m = self.contentsMargins()
        size += QSize(m.left() + m.right(), m.top() + m.bottom())
        return size

    def _do_layout(self, rect: QRect, test_only: bool) -> int:
        m = self.contentsMargins()
        eff = rect.adjusted(+m.left(), +m.top(), -m.right(), -m.bottom())
        x = eff.x()
        y = eff.y()
        line_h = 0
        for item in self._items:
            w = item.widget()
            # Use isHidden() (True only when setHidden(True) was called
            # explicitly) rather than `not isVisible()`.  A freshly added
            # widget hasn't been through its first showEvent yet, so
            # isVisible() would return False and we'd skip positioning it
            # — leaving it at (0, 0), overlapping earlier icons.
            if w is not None and w.isHidden():
                continue
            hint = item.sizeHint()
            next_x = x + hint.width() + self._h
            if next_x - self._h > eff.right() and line_h > 0:
                x = eff.x()
                y = y + line_h + self._v
                next_x = x + hint.width() + self._h
                line_h = 0
            if not test_only:
                item.setGeometry(QRect(QPoint(x, y), hint))
            x = next_x
            line_h = max(line_h, hint.height())
        return y + line_h - rect.y() + m.bottom()


# =========================================================================
# A single icon in the table
# =========================================================================
class _TableIcon(QToolButton):
    """Icon+text button that lives inside a category body.

    Left-drag  → moves the item within/between categories or drops on the
                 canvas / a view (using MIME_CMD so it works with the
                 existing ShortcutCanvas handler).
    Double-click → runs the command (with preset if any).
    Right-click  → context menu (Run / Edit Preset / Move / etc).
    """
    _CELL_W = 74            # visual width of the whole icon cell
    _CELL_H = 74            # visual height (icon + 1 line of text)
    _LABEL_MAX_W = 66       # px available for the text (cell − margins)

    def __init__(self, panel: "IconTablePanel", command_id: str,
                 preset: dict | None, label: str,
                 icon: QIcon, parent: QWidget):
        super().__init__(parent)
        self._panel = panel
        self.command_id = command_id
        self._preset: dict | None = dict(preset) if preset else None
        self._label = label
        self.setIcon(icon)
        self.setIconSize(QSize(_DEFAULT_ICON_PX, _DEFAULT_ICON_PX))
        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        self.setAutoRaise(True)
        self.setFixedSize(self._CELL_W, self._CELL_H)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._context_menu)
        # Full label goes in the tooltip; visible text is elided to fit.
        self.setToolTip(
            f"{label}\n"
            f"• Double-click: run\n"
            f"• Drag onto a view: apply headlessly\n"
            f"• Drag onto the canvas: create desktop shortcut\n"
            f"• Right-click: options"
        )
        self._apply_label_text()
        self._press_pos: QPoint | None = None

    def _apply_label_text(self):
        """Set the button's visible text, elided so it fits the cell width.

        QToolButton doesn't clip its own text painting; long labels
        would otherwise render past the button's boundary and visually
        collide with the neighbour.
        """
        try:
            from PyQt6.QtGui import QFontMetrics
            fm = QFontMetrics(self.font())
            elided = fm.elidedText(self._label,
                                   Qt.TextElideMode.ElideRight,
                                   self._LABEL_MAX_W)
        except Exception:
            elided = self._label
        self.setText(elided)

    def sizeHint(self) -> QSize:
        # The flow layout uses this to compute positioning.  Return the
        # actual cell size so items don't overlap even when the text
        # would want more room.
        return QSize(self._CELL_W, self._CELL_H)

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()

    # ---- preset access ------------------------------------------------
    def preset(self) -> dict | None:
        return dict(self._preset) if self._preset else None

    def set_preset(self, preset: dict | None):
        self._preset = dict(preset) if preset else None
        self._panel.save()

    def label(self) -> str:
        return self._label

    def set_label(self, label: str):
        self._label = label
        self._apply_label_text()
        # Keep the full name available in the tooltip
        self.setToolTip(
            f"{label}\n"
            f"• Double-click: run\n"
            f"• Drag onto a view: apply headlessly\n"
            f"• Drag onto the canvas: create desktop shortcut\n"
            f"• Right-click: options"
        )
        self._panel.save()

    # ---- interactions -------------------------------------------------
    def mouseDoubleClickEvent(self, ev: QMouseEvent):
        # Single-click activation runs from mouseReleaseEvent below.
        # QAbstractButton's default double-click routes back through
        # mousePressEvent, which would re-capture _press_pos and trigger
        # a SECOND _run() on the trailing release. Accept the event here
        # to swallow it cleanly — the first release of a double-click
        # sequence already ran the tool once, that's enough.
        ev.accept()

    def mousePressEvent(self, ev: QMouseEvent):
        if ev.button() == Qt.MouseButton.LeftButton:
            self._press_pos = ev.position().toPoint()
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev: QMouseEvent):
        if not (ev.buttons() & Qt.MouseButton.LeftButton) or self._press_pos is None:
            return super().mouseMoveEvent(ev)
        d = (ev.position().toPoint() - self._press_pos).manhattanLength()
        if d < QApplication_startDragDistance():
            return
        self._start_drag()
        self._press_pos = None

    def mouseReleaseEvent(self, ev: QMouseEvent):
        # Left-click without a drag → single-click activates the tool.
        # `_press_pos` was captured in mousePressEvent and cleared by
        # mouseMoveEvent the moment a drag actually started, so a
        # non-None value on release means the press-release cycle never
        # crossed the drag-start threshold. That's a genuine click.
        if (ev.button() == Qt.MouseButton.LeftButton
                and self._press_pos is not None):
            self._press_pos = None
            self._run()
            ev.accept()
            return
        self._press_pos = None
        super().mouseReleaseEvent(ev)

    # ---- drag ---------------------------------------------------------
    def _start_drag(self):
        md = QMimeData()
        # Full command payload — ShortcutCanvas + view drop handlers speak this
        md.setData(MIME_CMD, _pack_cmd_payload(self.command_id, self._preset))
        # Also action-id so plain shortcut creation still works if that's all
        # a target reads
        md.setData(MIME_ACTION, self.command_id.encode("utf-8"))
        # Internal marker so drop-targets INSIDE the table can distinguish
        # a table-originated drag (for reorder/move) from an external drag
        md.setData(_INTRA_TABLE_MIME, self._panel._uid_for(self).encode("utf-8"))

        drag = QDrag(self)
        drag.setMimeData(md)
        pm = self.icon().pixmap(_DEFAULT_ICON_PX, _DEFAULT_ICON_PX)
        if pm.isNull():
            pm = QPixmap(_DEFAULT_ICON_PX, _DEFAULT_ICON_PX)
            pm.fill(Qt.GlobalColor.darkGray)
        drag.setPixmap(pm)
        drag.setHotSpot(pm.rect().center())
        drag.exec(Qt.DropAction.CopyAction | Qt.DropAction.MoveAction)

    # ---- context menu -------------------------------------------------
    def _context_menu(self, pos: QPoint):
        m = QMenu(self)
        m.addAction(self.tr("Run"), self._run)
        m.addSeparator()
        m.addAction(self.tr("Edit Preset…"), self._edit_preset)
        if self._preset:
            m.addAction(self.tr("Clear Preset"), lambda: self.set_preset(None))
        m.addAction(self.tr("Rename…"), self._rename)
        m.addSeparator()
        m.addAction(self.tr("Copy to Canvas"), self._copy_to_canvas)
        move_m = m.addMenu(self.tr("Move to Category"))
        cats = self._panel.categories()
        my_cat = self._panel.category_of(self)
        for c in cats:
            act = move_m.addAction(c.name())
            act.setEnabled(c is not my_cat)
            act.triggered.connect(lambda _=False, target=c: self._panel.move_icon(self, target))
        m.addSeparator()
        m.addAction(self.tr("Remove from Table"), self._delete)
        m.exec(self.mapToGlobal(pos))

    # ---- actions ------------------------------------------------------
    def _run(self):
        mgr = self._panel.manager
        if mgr is None:
            return
        if self._preset:
            try:
                mgr.trigger_with_preset(self.command_id, self._preset)
                return
            except Exception:
                pass
        mgr.trigger(self.command_id)

    def _edit_preset(self):
        """Open the same preset editor the canvas shortcuts use, if any."""
        try:
            from setiastro.saspro.shortcuts import _open_preset_editor_for_command
        except Exception:
            _open_preset_editor_for_command = None
        if _open_preset_editor_for_command is None:
            QMessageBox.information(self, "Edit Preset",
                                    "No preset editor available for this command.")
            return
        try:
            new_preset = _open_preset_editor_for_command(
                self._panel, self.command_id, self._preset or {}
            )
        except Exception as e:
            QMessageBox.warning(self, "Edit Preset", f"Editor failed:\n{e}")
            return
        if new_preset is not None:
            self.set_preset(new_preset)

    def _rename(self):
        new_name, ok = QInputDialog.getText(
            self, self.tr("Rename Icon"), self.tr("Label:"), text=self._label,
        )
        if ok and new_name.strip():
            self.set_label(new_name.strip())

    def _copy_to_canvas(self):
        mgr = self._panel.manager
        if mgr is None:
            return
        try:
            # place near the top-left of the MDI viewport so the user sees it
            pos = QPoint(40, 40)
            mgr.add_shortcut(self.command_id, pos, label=self._label)
            if self._preset:
                # Attach preset to the newly created canvas shortcut, using the
                # same mechanism the drag-and-drop payload path uses.
                for sid, w in mgr.widgets.items():
                    if getattr(w, "command_id", None) == self.command_id and w.pos() == pos:
                        try:
                            w._save_preset(self._preset)
                        except Exception:
                            pass
                        break
                mgr.save_shortcuts()
        except Exception as e:
            QMessageBox.warning(self, "Copy to Canvas", f"Failed:\n{e}")

    def _delete(self):
        self._panel.remove_icon(self)


def QApplication_startDragDistance() -> int:
    """Lazy import to avoid a top-level QApplication import for this constant."""
    from PyQt6.QtWidgets import QApplication
    return QApplication.startDragDistance()


# Marker mime for intra-table drags (holds the icon's runtime uid so the
# panel can identify which item is being moved).
_INTRA_TABLE_MIME = "application/x-saspro-icon-table-item"


# =========================================================================
# Category header + body (collapsible group)
# =========================================================================
class _CategoryHeader(QLabel):
    """Clickable centered label; toggles the body's visibility."""
    def __init__(self, panel: "IconTablePanel", cat: "_Category", parent=None):
        super().__init__(parent)
        self._panel = panel
        self._cat = cat
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setContentsMargins(6, 4, 6, 4)
        self.setStyleSheet(
            "QLabel {"
            "  background-color: rgba(90,110,150,140);"
            "  color: #eaf1ff;"
            "  font-weight: bold;"
            "  border-radius: 3px;"
            "  padding: 4px 2px;"
            "}"
            "QLabel:hover { background-color: rgba(110,135,180,170); }"
        )
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAcceptDrops(True)
        self.refresh_text()

    def refresh_text(self):
        arrow = "" if self._cat.expanded else "  ▸"
        self.setText(f"{self._cat.name()}{arrow}")

    def mousePressEvent(self, ev: QMouseEvent):
        if ev.button() == Qt.MouseButton.LeftButton:
            self._cat.set_expanded(not self._cat.expanded)
            self.refresh_text()
            ev.accept()
            return
        super().mousePressEvent(ev)

    def contextMenuEvent(self, ev: QContextMenuEvent):
        m = QMenu(self)
        m.addAction(self.tr("Rename…"), self._rename)
        m.addSeparator()
        m.addAction(self.tr("Move Up"), lambda: self._panel.move_category(self._cat, -1))
        m.addAction(self.tr("Move Down"), lambda: self._panel.move_category(self._cat, +1))
        m.addSeparator()
        m.addAction(
            self.tr("Collapse") if self._cat.expanded else self.tr("Expand"),
            lambda: (self._cat.set_expanded(not self._cat.expanded), self.refresh_text()),
        )
        m.addSeparator()
        m.addAction(self.tr("Delete Category"), self._delete)
        m.exec(ev.globalPos())

    def _rename(self):
        new_name, ok = QInputDialog.getText(
            self, self.tr("Rename Category"), self.tr("Name:"),
            text=self._cat.name(),
        )
        if ok and new_name.strip():
            self._cat.set_name(new_name.strip())
            self.refresh_text()

    def _delete(self):
        if len(self._cat._icons) > 0:
            ret = QMessageBox.question(
                self, self.tr("Delete Category"),
                self.tr(f"Delete “{self._cat.name()}” and its "
                        f"{len(self._cat._icons)} icon(s)?"),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if ret != QMessageBox.StandardButton.Yes:
                return
        self._panel.delete_category(self._cat)

    # ---- drop-on-header: move item into this category ----------------
    def dragEnterEvent(self, ev: QDragEnterEvent):
        if ev.mimeData().hasFormat(MIME_CMD) or ev.mimeData().hasFormat(MIME_ACTION):
            ev.acceptProposedAction()
        else:
            ev.ignore()

    def dropEvent(self, ev: QDropEvent):
        self._panel._handle_drop_on_category(self._cat, ev)


class _CategoryBody(QFrame):
    """The area beneath a category header that hosts the icons."""
    def __init__(self, panel: "IconTablePanel", cat: "_Category", parent=None):
        super().__init__(parent)
        self._panel = panel
        self._cat = cat
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setAcceptDrops(True)
        self._flow = _FlowLayout(self, margin=6, h_spacing=6, v_spacing=6)
        self.setLayout(self._flow)
        # CRITICAL: without a height-for-width size policy, the surrounding
        # QVBoxLayout ignores heightForWidth() and gives us a 0-tall slot —
        # every icon then lands at (0,0) on top of each other.
        sp = QSizePolicy(QSizePolicy.Policy.Expanding,
                        QSizePolicy.Policy.Minimum)
        sp.setHeightForWidth(True)
        self.setSizePolicy(sp)
        # Reserve at least one icon row so a freshly created (empty)
        # category still has a visible drop target.
        self.setMinimumHeight(_DEFAULT_ICON_PX + 24)

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, w: int) -> int:
        # Forward to the flow layout so the parent QVBoxLayout can size us
        # correctly for the current width.
        return self._flow.heightForWidth(w)

    def add_icon_widget(self, icon: _TableIcon):
        self._flow.addWidget(icon)
        # Force an immediate show so the first _do_layout pass positions
        # it correctly.  Without this, the widget's first showEvent may
        # not fire until AFTER the layout pass that added it, leaving it
        # at its default (0, 0) position on top of the first icon.
        icon.show()
        # New icon → recompute our size hint so the parent QVBoxLayout
        # gives us more vertical space if needed.
        self.updateGeometry()

    def remove_icon_widget(self, icon: _TableIcon):
        # QLayout.removeWidget only unparents from the layout; explicit hide
        # + setParent(None) is required to actually detach.
        self._flow.removeWidget(icon)
        icon.setParent(None)
        self.updateGeometry()

    # ---- drop-on-body: same handling as header (append to this cat) --
    def dragEnterEvent(self, ev: QDragEnterEvent):
        if ev.mimeData().hasFormat(MIME_CMD) or ev.mimeData().hasFormat(MIME_ACTION):
            ev.acceptProposedAction()
        else:
            ev.ignore()

    def dragMoveEvent(self, ev):
        if ev.mimeData().hasFormat(MIME_CMD) or ev.mimeData().hasFormat(MIME_ACTION):
            ev.acceptProposedAction()

    def dropEvent(self, ev: QDropEvent):
        # Pass the drop position so the handler can compute an insertion
        # index for intra-table reorders. Widget-local coords — icon
        # geometries live in the same frame.
        self._panel._handle_drop_on_category(self._cat, ev, at_pos=ev.position().toPoint())


# =========================================================================
# Category — data + widget owner
# =========================================================================
class _Category:
    def __init__(self, panel: "IconTablePanel", name: str, expanded: bool = True):
        self._panel = panel
        self._name = name
        self.expanded = expanded
        self._icons: list[_TableIcon] = []
        self._header: _CategoryHeader | None = None
        self._body: _CategoryBody | None = None
        self._add_btn: QToolButton | None = None
        self._container: QWidget | None = None

    # ---- data ----
    def name(self) -> str:
        return self._name

    def set_name(self, s: str):
        self._name = s
        if self._header is not None:
            self._header.refresh_text()
        self._panel.save()

    def set_expanded(self, on: bool):
        self.expanded = bool(on)
        if self._body is not None:
            self._body.setVisible(self.expanded)
        if self._add_btn is not None:
            self._add_btn.setVisible(self.expanded)
        self._panel.save()

    def icons(self) -> list[_TableIcon]:
        return list(self._icons)

    def build_widget(self, parent: QWidget) -> QWidget:
        self._container = QWidget(parent)
        v = QVBoxLayout(self._container)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(2)

        self._header = _CategoryHeader(self._panel, self, self._container)
        v.addWidget(self._header)

        self._body = _CategoryBody(self._panel, self, self._container)
        v.addWidget(self._body)

        # Per-category "add" button — appears at the end of the body
        self._add_btn = QToolButton(self._container)
        self._add_btn.setText(self._panel.tr("+ Add item"))
        self._add_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self._add_btn.setStyleSheet("QToolButton { color: #8fb8ff; padding: 2px 6px; }")
        self._add_btn.clicked.connect(lambda: self._panel._prompt_add_item(self))
        v.addWidget(self._add_btn, 0, Qt.AlignmentFlag.AlignLeft)

        # Re-attach existing icons if this is a rebuild
        for ic in self._icons:
            self._body.add_icon_widget(ic)

        self._body.setVisible(self.expanded)
        self._add_btn.setVisible(self.expanded)
        return self._container

    def append_icon(self, icon: _TableIcon):
        self._icons.append(icon)
        if self._body is not None:
            self._body.add_icon_widget(icon)

    def remove_icon(self, icon: _TableIcon):
        if icon in self._icons:
            self._icons.remove(icon)
        if self._body is not None:
            self._body.remove_icon_widget(icon)

    def insert_icon(self, icon: _TableIcon, index: int):
        index = max(0, min(index, len(self._icons)))
        self._icons.insert(index, icon)
        # We rebuild the body to respect ordering; QLayout doesn't have a
        # cheap insertAt for flow layouts.
        if self._body is not None:
            # Detach all, re-add in order.
            for ic in self._icons:
                self._body._flow.removeWidget(ic)
            for ic in self._icons:
                self._body.add_icon_widget(ic)


# =========================================================================
# The panel itself
# =========================================================================
class IconTablePanel(QDialog):
    """Floating window holding categorised process icons.

    Constructed with a reference to the running ShortcutManager so it can
    look up icons + trigger commands without needing to reach through the
    main window.
    """
    def __init__(self, main_window, manager, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.manager = manager
        self.setWindowTitle(self.tr("Function Icon Table"))
        self.setWindowFlag(Qt.WindowType.Window, True)
        if platform.system() == "Darwin":
            self.setWindowFlag(Qt.WindowType.Tool, True)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setModal(False)
        self.setAcceptDrops(True)

        self._categories: list[_Category] = []
        # Runtime uid → _TableIcon, for identifying intra-table drags
        self._uid_map: dict[str, _TableIcon] = {}
        self._next_uid = 0

        self._build_ui()
        self._load_from_settings()
        if not self._categories:
            self._install_defaults()
        self._rebuild()

        self.resize(360, 720)

    # -----------------------------------------------------------------
    # Public API — used by the icons/headers/callers
    # -----------------------------------------------------------------
    def categories(self) -> list[_Category]:
        return list(self._categories)

    def category_of(self, icon: _TableIcon) -> _Category | None:
        for c in self._categories:
            if icon in c._icons:
                return c
        return None

    def save(self):
        """Persist the current state to QSettings."""
        try:
            s = QSettings()
            s.setValue(_QS_KEY, json.dumps(self._to_json()))
            s.sync()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Window geometry + visibility persistence.  Called by main_window
    # on shutdown / startup so the panel reopens where the user left it.
    # ------------------------------------------------------------------
    def save_window_state(self, was_visible: bool | None = None):
        """Persist geometry + visibility so the next launch can restore it.

        `was_visible` overrides the runtime isVisible() check — useful
        during shutdown because Qt may already have hidden the widget
        by the time this is called.
        """
        try:
            s = QSettings()
            if was_visible is None:
                was_visible = bool(self.isVisible())
            s.setValue("icon_table/visible", bool(was_visible))
            s.setValue("icon_table/geometry", self.saveGeometry())
            s.sync()
        except Exception:
            pass

    def restore_window_state(self) -> bool:
        """Restore geometry (if saved) and return whether the panel was
        visible in the previous session.  Doesn't show/hide — caller
        decides based on the return value."""
        try:
            s = QSettings()
            geo = s.value("icon_table/geometry", None)
            if geo is not None and len(geo) > 0:
                self.restoreGeometry(geo)
            return bool(s.value("icon_table/visible", False, type=bool))
        except Exception:
            return False

    # Persist "visible" flag on every user-driven show/hide + close so
    # we don't lose the state if the app crashes before closeEvent fires.
    # HOWEVER — during app shutdown, save_main_window_state() has already
    # recorded the correct value; the hideEvent/closeEvent that fire as
    # Qt tears widgets down would otherwise overwrite it with False.
    # We check the main window's _shutting_down flag to suppress those
    # spurious writes.
    def _shutdown_in_progress(self) -> bool:
        mw = self.main_window
        try:
            return bool(getattr(mw, "_shutting_down", False))
        except Exception:
            return False

    def hideEvent(self, ev):
        if not self._shutdown_in_progress():
            try:
                QSettings().setValue("icon_table/visible", False)
            except Exception:
                pass
        super().hideEvent(ev)

    def showEvent(self, ev):
        if not self._shutdown_in_progress():
            try:
                QSettings().setValue("icon_table/visible", True)
            except Exception:
                pass
        super().showEvent(ev)

    def closeEvent(self, ev):
        if not self._shutdown_in_progress():
            try:
                s = QSettings()
                s.setValue("icon_table/visible", False)
                s.setValue("icon_table/geometry", self.saveGeometry())
                s.sync()
            except Exception:
                pass
        super().closeEvent(ev)

    def _uid_for(self, icon: _TableIcon) -> str:
        """Return the runtime uid registered for this icon (assigned at
        creation time in _make_icon)."""
        for uid, w in self._uid_map.items():
            if w is icon:
                return uid
        return ""

    # -----------------------------------------------------------------
    # UI construction
    # -----------------------------------------------------------------
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        # Scrollable body — categories stack vertically
        self._scroll = QScrollArea(self)
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll_inner = QWidget()
        self._scroll_layout = QVBoxLayout(self._scroll_inner)
        self._scroll_layout.setContentsMargins(2, 2, 2, 2)
        self._scroll_layout.setSpacing(4)
        self._scroll_layout.addStretch(1)  # trailing stretch pushes everything up
        self._scroll.setWidget(self._scroll_inner)
        root.addWidget(self._scroll, 1)

        # Bottom row: [+ Add Category] [Load ▾] [Save…] [Clear]   [Close]
        btn_row = QHBoxLayout()
        self.btn_add_cat = QPushButton(self.tr("+ Add Category"))
        self.btn_add_cat.clicked.connect(self._on_add_category)
        btn_row.addWidget(self.btn_add_cat)

        self.btn_load = QToolButton()
        self.btn_load.setText(self.tr("Load ▾"))
        self.btn_load.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.btn_load.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.btn_load.setToolTip(self.tr(
            "Load a saved icon table.  The menu also lets you delete saved\n"
            "tables you no longer need."
        ))
        self.btn_load.clicked.connect(self._show_load_menu)
        btn_row.addWidget(self.btn_load)

        self.btn_save = QPushButton(self.tr("Save…"))
        self.btn_save.setToolTip(self.tr(
            "Save the current table layout under a name so you can restore it\n"
            "later.  Existing names are offered as an autocomplete when saving."
        ))
        self.btn_save.clicked.connect(self._on_save_table)
        btn_row.addWidget(self.btn_save)

        self.btn_clear = QPushButton(self.tr("Clear"))
        self.btn_clear.setToolTip(self.tr(
            "Remove all categories and icons from the current table.\n"
            "This does not delete any saved tables."
        ))
        self.btn_clear.clicked.connect(self._on_clear_table)
        btn_row.addWidget(self.btn_clear)

        btn_row.addStretch(1)
        self.btn_close = QPushButton(self.tr("Close"))
        self.btn_close.clicked.connect(self.hide)
        btn_row.addWidget(self.btn_close)
        root.addLayout(btn_row)

    def _rebuild(self):
        """Re-render every category from scratch."""
        # Remove existing category widgets (keep the trailing stretch)
        while self._scroll_layout.count() > 1:
            item = self._scroll_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()

        # Rebuild — this constructs fresh header/body widgets AND
        # re-parents the existing _TableIcon widgets into the new bodies.
        for cat in self._categories:
            container = cat.build_widget(self._scroll_inner)
            # Insert BEFORE the trailing stretch (index count-1)
            self._scroll_layout.insertWidget(self._scroll_layout.count() - 1, container)

    # -----------------------------------------------------------------
    # Defaults — installed only when no persisted layout exists yet
    # -----------------------------------------------------------------
    def _install_defaults(self):
        """Seed with a sensible starter set of categories (empty)."""
        for name in ("Calibration", "Preprocessing", "Linear",
                     "Stretching", "Non-Linear", "Utilities"):
            self._categories.append(_Category(self, name, expanded=True))

    # -----------------------------------------------------------------
    # JSON persistence
    # -----------------------------------------------------------------
    def _to_json(self) -> dict:
        cats = []
        for c in self._categories:
            items = []
            for ic in c._icons:
                items.append({
                    "command_id": ic.command_id,
                    "label": ic.label(),
                    "preset": ic.preset(),
                })
            cats.append({
                "name": c.name(),
                "expanded": bool(c.expanded),
                "items": items,
            })
        return {"categories": cats}

    def _load_from_settings(self):
        try:
            s = QSettings()
            raw = s.value(_QS_KEY, "", type=str) or ""
            if not raw:
                return
            data = json.loads(raw)
        except Exception:
            return
        try:
            for cd in data.get("categories", []):
                cat = _Category(self, str(cd.get("name", "Untitled")),
                                expanded=bool(cd.get("expanded", True)))
                for item in cd.get("items", []):
                    cid = str(item.get("command_id", "") or "")
                    if not cid:
                        continue
                    label = str(item.get("label", "") or "")
                    preset = item.get("preset") or None
                    icon = self._make_icon(cid, preset, label)
                    if icon is not None:
                        cat._icons.append(icon)
                self._categories.append(cat)
        except Exception:
            # If persisted state is corrupt, start clean
            self._categories = []

    # -----------------------------------------------------------------
    # Icon construction — asks the ShortcutManager for the QAction/icon
    # -----------------------------------------------------------------
    def _make_icon(self, command_id: str, preset: dict | None,
                   label: str | None) -> _TableIcon | None:
        mgr = self.manager
        if mgr is None:
            return None
        act = mgr.registry.get(command_id)
        if act is None:
            # Registration might come later; still create a stub so the user's
            # layout isn't silently dropped. Uses a blank icon + command_id
            # as label.
            ico = QIcon()
            lbl = label or command_id
        else:
            try:
                ico = mgr._icon_for_command(command_id, act)
            except Exception:
                ico = act.icon() if act is not None else QIcon()
            lbl = (label or (act.text() or command_id)).strip() or command_id
        icon_w = _TableIcon(self, command_id, preset, lbl, ico, self._scroll_inner)
        # register a uid so intra-table drags can identify this widget
        self._next_uid += 1
        uid = f"it_{self._next_uid}"
        self._uid_map[uid] = icon_w
        return icon_w

    # -----------------------------------------------------------------
    # Save / load / clear whole tables
    # -----------------------------------------------------------------
    _SAVED_KEY_PREFIX = "icon_table/saved/"    # value = JSON blob (same schema as _to_json)
    _SAVED_INDEX_KEY  = "icon_table/saved_names"   # ordered list of saved-table names

    def _load_saved_index(self) -> list[str]:
        """Return the ordered list of saved-table names."""
        try:
            s = QSettings()
            raw = s.value(self._SAVED_INDEX_KEY, "", type=str) or ""
            if not raw:
                return []
            names = json.loads(raw)
            return [str(n) for n in names if isinstance(n, str)]
        except Exception:
            return []

    def _write_saved_index(self, names: list[str]):
        try:
            s = QSettings()
            s.setValue(self._SAVED_INDEX_KEY, json.dumps(names))
            s.sync()
        except Exception:
            pass

    def _on_save_table(self):
        """Save the current layout under a user-supplied name."""
        existing = self._load_saved_index()
        # QInputDialog doesn't have autocomplete out of the box, but the
        # existing names appear in the Load menu so users can see them.
        default = ""
        name, ok = QInputDialog.getText(
            self, self.tr("Save Table"),
            self.tr("Name for this table:") + (
                "\n\n" + self.tr("Existing: ") + ", ".join(existing) if existing else ""
            ),
            text=default,
        )
        if not ok:
            return
        name = name.strip()
        if not name:
            return

        # Confirm overwrite
        if name in existing:
            ret = QMessageBox.question(
                self, self.tr("Overwrite Table"),
                self.tr(f"“{name}” already exists.  Overwrite it?"),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if ret != QMessageBox.StandardButton.Yes:
                return

        # Write blob + update index
        try:
            s = QSettings()
            s.setValue(f"{self._SAVED_KEY_PREFIX}{name}", json.dumps(self._to_json()))
            if name not in existing:
                existing.append(name)
                # Keep the index sorted for a predictable menu order
                existing.sort(key=str.lower)
                self._write_saved_index(existing)
            s.sync()
        except Exception as e:
            QMessageBox.warning(self, self.tr("Save Table"),
                                self.tr(f"Save failed:\n{e}"))
            return

        # Bookkeeping
        try:
            if hasattr(self.main_window, "_log"):
                self.main_window._log(f"Icon Table: saved “{name}”")
        except Exception:
            pass

    def _show_load_menu(self):
        """Popup the list of saved tables — pick one to replace the current
        layout, or use the Manage submenu to delete."""
        names = self._load_saved_index()
        m = QMenu(self)

        if not names:
            act = m.addAction(self.tr("(no saved tables yet)"))
            act.setEnabled(False)
        else:
            for name in names:
                act = m.addAction(name)
                act.triggered.connect(lambda _=False, n=name: self._load_table(n))
            m.addSeparator()
            del_m = m.addMenu(self.tr("Delete saved table"))
            for name in names:
                dact = del_m.addAction(name)
                dact.triggered.connect(lambda _=False, n=name: self._delete_saved_table(n))

        gp = self.btn_load.mapToGlobal(self.btn_load.rect().bottomLeft())
        m.exec(gp)

    def _load_table(self, name: str):
        """Replace the current layout with the saved table `name`."""
        try:
            s = QSettings()
            raw = s.value(f"{self._SAVED_KEY_PREFIX}{name}", "", type=str) or ""
            if not raw:
                QMessageBox.warning(self, self.tr("Load Table"),
                                    self.tr(f"“{name}” could not be found."))
                return
            data = json.loads(raw)
        except Exception as e:
            QMessageBox.warning(self, self.tr("Load Table"),
                                self.tr(f"Load failed:\n{e}"))
            return

        # Wipe current state (widgets + uid map) — but don't touch saved
        # blobs.  This mirrors _load_from_settings but works from the
        # already-parsed dict.
        self._wipe_current_state()
        try:
            for cd in data.get("categories", []):
                cat = _Category(self, str(cd.get("name", "Untitled")),
                                expanded=bool(cd.get("expanded", True)))
                for item in cd.get("items", []):
                    cid = str(item.get("command_id", "") or "")
                    if not cid:
                        continue
                    label = str(item.get("label", "") or "")
                    preset = item.get("preset") or None
                    icon = self._make_icon(cid, preset, label)
                    if icon is not None:
                        cat._icons.append(icon)
                self._categories.append(cat)
        except Exception as e:
            QMessageBox.warning(self, self.tr("Load Table"),
                                self.tr(f"Layout parse failed:\n{e}"))
            self._categories = []
        self._rebuild()
        self.save()   # persist the newly-loaded layout as "current"

        try:
            if hasattr(self.main_window, "_log"):
                self.main_window._log(f"Icon Table: loaded “{name}”")
        except Exception:
            pass

    def _delete_saved_table(self, name: str):
        ret = QMessageBox.question(
            self, self.tr("Delete Saved Table"),
            self.tr(f"Delete the saved table “{name}”?  This cannot be undone."),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if ret != QMessageBox.StandardButton.Yes:
            return
        try:
            s = QSettings()
            s.remove(f"{self._SAVED_KEY_PREFIX}{name}")
            names = [n for n in self._load_saved_index() if n != name]
            self._write_saved_index(names)
            s.sync()
        except Exception as e:
            QMessageBox.warning(self, self.tr("Delete Saved Table"),
                                self.tr(f"Delete failed:\n{e}"))
            return
        try:
            if hasattr(self.main_window, "_log"):
                self.main_window._log(f"Icon Table: deleted saved “{name}”")
        except Exception:
            pass

    def _on_clear_table(self):
        """Wipe all icons from every category.  Categories themselves are
        preserved — clearing is about resetting the *contents*, not the
        user's organisational structure."""
        total_icons = sum(len(c._icons) for c in self._categories)
        if total_icons == 0:
            return
        ret = QMessageBox.question(
            self, self.tr("Clear Table"),
            self.tr(
                f"Remove all {total_icons} icon(s) from the current table?\n\n"
                "Categories will be kept.  Saved tables are not affected — "
                "you can restore them from Load ▾."
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if ret != QMessageBox.StandardButton.Yes:
            return

        # Detach every icon widget but leave the categories in place.
        for cat in self._categories:
            for ic in list(cat._icons):
                for uid, w in list(self._uid_map.items()):
                    if w is ic:
                        del self._uid_map[uid]
                cat.remove_icon(ic)
                ic.setParent(None)
                ic.deleteLater()
        self._next_uid = 0
        self._uid_map.clear()
        self.save()

    def _wipe_current_state(self):
        """Detach every icon widget and drop every category — used by
        Load and Clear before installing a fresh layout."""
        for cat in self._categories:
            for ic in list(cat._icons):
                for uid, w in list(self._uid_map.items()):
                    if w is ic:
                        del self._uid_map[uid]
                ic.setParent(None)
                ic.deleteLater()
            cat._icons.clear()
        self._categories.clear()
        # Reset uid counter so numbers stay tidy across load cycles
        self._next_uid = 0
        self._uid_map.clear()

    # -----------------------------------------------------------------
    # Category management
    # -----------------------------------------------------------------
    def _on_add_category(self):
        name, ok = QInputDialog.getText(
            self, self.tr("New Category"), self.tr("Category name:"),
            text=self.tr("New Category"),
        )
        if not ok or not name.strip():
            return
        cat = _Category(self, name.strip(), expanded=True)
        self._categories.append(cat)
        self._rebuild()
        self.save()

    def delete_category(self, cat: _Category):
        # Remove icons' uid registrations first
        for ic in list(cat._icons):
            for uid, w in list(self._uid_map.items()):
                if w is ic:
                    del self._uid_map[uid]
            ic.setParent(None)
            ic.deleteLater()
        if cat in self._categories:
            self._categories.remove(cat)
        self._rebuild()
        self.save()

    def move_category(self, cat: _Category, delta: int):
        try:
            i = self._categories.index(cat)
        except ValueError:
            return
        j = i + delta
        if j < 0 or j >= len(self._categories):
            return
        self._categories[i], self._categories[j] = self._categories[j], self._categories[i]
        self._rebuild()
        self.save()

    # -----------------------------------------------------------------
    # Icon add / move / remove
    # -----------------------------------------------------------------
    def _prompt_add_item(self, cat: _Category):
        """Popup a searchable menu of every registered command."""
        mgr = self.manager
        if mgr is None:
            return
        # Build items list
        items: list[tuple[str, str, QIcon]] = []  # (label, cid, icon)
        for cid, act in mgr.registry.items():
            try:
                ico = mgr._icon_for_command(cid, act)
            except Exception:
                ico = act.icon() if act is not None else QIcon()
            lbl = (act.text() or cid).strip() or cid
            items.append((lbl, cid, ico))
        items.sort(key=lambda t: t[0].lower())

        # Simple searchable dialog: a QLineEdit + a menu-like listing.
        # For a minimal footprint we use a QMenu, and let the user just scroll.
        m = QMenu(self)
        m.setStyleSheet("QMenu { menu-scrollable: 1; }")
        m.addAction(self.tr("(pick a command to add)")).setEnabled(False)
        m.addSeparator()
        for lbl, cid, ico in items:
            act = m.addAction(ico, lbl)
            act.triggered.connect(lambda _=False, c=cid, l=lbl:
                                  self.add_item(cat, c, preset=None, label=l))
        # Show near the "+ Add item" button of that category
        anchor = cat._add_btn if cat._add_btn is not None else self
        gp = anchor.mapToGlobal(anchor.rect().bottomLeft())
        m.exec(gp)

    def add_item(self, cat: _Category, command_id: str,
                 preset: dict | None = None, label: str | None = None):
        icon = self._make_icon(command_id, preset, label)
        if icon is None:
            return
        cat.append_icon(icon)
        self.save()

    def remove_icon(self, icon: _TableIcon):
        cat = self.category_of(icon)
        if cat is None:
            icon.setParent(None)
            icon.deleteLater()
            return
        # unregister uid
        for uid, w in list(self._uid_map.items()):
            if w is icon:
                del self._uid_map[uid]
        cat.remove_icon(icon)
        icon.setParent(None)
        icon.deleteLater()
        self.save()

    def move_icon(self, icon: _TableIcon, target: _Category, index: int | None = None):
        cur = self.category_of(icon)
        if cur is None or target is None:
            return
        if cur is target:
            # Reorder within the same category — actually move the icon
            # this time. `index` is what the caller derived from the drop
            # position (see _compute_insert_index). None still means "no
            # positional info was provided", which for a same-category
            # drag would be a no-op anyway.
            if index is None:
                return
            try:
                cur_idx = cur._icons.index(icon)
            except ValueError:
                return
            # Two no-op cases: dropping exactly at your own position, or
            # at the adjacent slot that would leave you exactly where you
            # started after the remove-then-insert dance. Skip the layout
            # rebuild and the save in those cases.
            if index == cur_idx or index == cur_idx + 1:
                return
            cur._icons.remove(icon)
            # Removing shifts positions ABOVE the source down by one, so
            # a requested target past the source needs to compensate to
            # land where the user visually dropped.
            if index > cur_idx:
                index -= 1
            cur.insert_icon(icon, index)
        else:
            cur.remove_icon(icon)
            if index is None:
                target.append_icon(icon)
            else:
                target.insert_icon(icon, index)
        self.save()

    def _compute_insert_index(self, cat: _Category, pos) -> int:
        """Return the index at which to insert an icon dropped at `pos`
        (in the category body's local coordinates).

        Finds the icon whose geometric centre is closest to `pos`, then
        decides before/after based on which side of the icon's horizontal
        midpoint the drop landed on. Handles the flow layout's multiple
        rows implicitly — closest-by-Euclidean-distance naturally picks
        the icon on the same row when there is one, and falls to the
        nearest icon on an adjacent row when the drop is between rows.

        Returns 0 for an empty category, or len(icons) if the drop is
        past the last icon."""
        icons = cat._icons
        if not icons:
            return 0
        best_i = 0
        best_d = None
        for i, ic in enumerate(icons):
            r = ic.geometry()
            cx = r.center().x()
            cy = r.center().y()
            d = (pos.x() - cx) ** 2 + (pos.y() - cy) ** 2
            if best_d is None or d < best_d:
                best_d = d
                best_i = i
        r = icons[best_i].geometry()
        return best_i if pos.x() < r.center().x() else best_i + 1

    # -----------------------------------------------------------------
    # Panel-level drop (external drops go to the last category)
    # -----------------------------------------------------------------
    def dragEnterEvent(self, ev: QDragEnterEvent):
        md = ev.mimeData()
        if md.hasFormat(MIME_CMD) or md.hasFormat(MIME_ACTION):
            ev.acceptProposedAction()

    def dropEvent(self, ev: QDropEvent):
        # Drop landing on empty space in the panel — append to last cat
        if not self._categories:
            self._categories.append(_Category(self, self.tr("Uncategorised"), expanded=True))
            self._rebuild()
        self._handle_drop_on_category(self._categories[-1], ev)

    def _handle_drop_on_category(self, cat: _Category, ev: QDropEvent, at_pos=None):
        md = ev.mimeData()

        # 1) Intra-table drag (reorder or move between categories).
        # If a drop position was passed (body drop), turn it into an
        # insertion index — that's what actually enables drag-reorder
        # within a group. Header drops and panel-empty-space drops pass
        # at_pos=None and keep the old append-to-category behaviour.
        if md.hasFormat(_INTRA_TABLE_MIME):
            uid = bytes(md.data(_INTRA_TABLE_MIME)).decode("utf-8", "ignore")
            src_icon = self._uid_map.get(uid)
            if src_icon is not None:
                insert_index = None
                if at_pos is not None:
                    insert_index = self._compute_insert_index(cat, at_pos)
                self.move_icon(src_icon, cat, index=insert_index)
                ev.acceptProposedAction()
                return

        # 2) External MIME_CMD (full payload)
        if md.hasFormat(MIME_CMD):
            try:
                payload = json.loads(bytes(md.data(MIME_CMD)).decode("utf-8"))
                cid = str(payload.get("command_id", "") or "")
                if cid:
                    preset = payload.get("preset") or None
                    self.add_item(cat, cid, preset=preset)
                    ev.acceptProposedAction()
                    return
            except Exception:
                pass

        # 3) External MIME_ACTION (bare command_id)
        if md.hasFormat(MIME_ACTION):
            cid = bytes(md.data(MIME_ACTION)).decode("utf-8", "ignore").strip()
            if cid:
                self.add_item(cat, cid)
                ev.acceptProposedAction()
                return


# =========================================================================
# Entry point wired from main
# =========================================================================
def open_icon_table(main_window, manager, existing: IconTablePanel | None = None):
    """Open (or raise) the Icon Table panel.

    Pass in whatever reference the main window currently holds so we don't
    make duplicates — the caller stores the returned instance back for
    reuse.
    """
    if existing is not None:
        try:
            existing.show(); existing.raise_(); existing.activateWindow()
            return existing
        except Exception:
            pass
    dlg = IconTablePanel(main_window, manager, parent=main_window)
    try:
        # Optional: attach the same window icon the toolbar uses if there's one
        from setiastro.saspro.resources import shortcuts_path
        dlg.setWindowIcon(QIcon(shortcuts_path))
    except Exception:
        pass
    dlg.show()
    return dlg