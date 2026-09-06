# pro/gui/mixins/dock_mixin.py
"""
Dock management mixin for AstroSuiteProMainWindow.

This mixin contains all dock-related functionality: initialization, 
visibility management, and registration in menus.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QDockWidget, QPlainTextEdit, QTreeWidget, QTreeWidgetItem,
    QVBoxLayout, QWidget, QTextEdit, QListWidget, QListWidgetItem,
    QAbstractItemView, QApplication, QLineEdit, QMenu, QMainWindow
)

from setiastro.saspro.dock_host_window import DockHostWindow
from PyQt6.QtGui import QTextCursor, QAction, QGuiApplication

if TYPE_CHECKING:
    from PyQt6.QtWidgets import QAction

import os


from PyQt6.QtWidgets import QStyledItemDelegate
from PyQt6.QtGui import QPainter, QColor
from PyQt6.QtCore import QRect

class ActiveDocDelegate(QStyledItemDelegate):
    """Draws a teal left-bar accent on the row whose doc is the active document."""

    def __init__(self, get_active_fn, parent=None):
        super().__init__(parent)
        self._get_active = get_active_fn   # callable → current active doc

    def paint(self, painter: QPainter, option, index):
        super().paint(painter, option, index)
        doc = index.data(Qt.ItemDataRole.UserRole)
        if doc is None:
            return
        active = self._get_active()
        if active is None:
            return
        if doc is active or getattr(doc, "_base_doc", doc) is getattr(active, "_base_doc", active):
            painter.save()
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor("#3dbf9f"))   # teal accent
            bar = QRect(option.rect.left(), option.rect.top(), 3, option.rect.height())
            painter.drawRect(bar)
            painter.restore()

GLYPHS = "■●◆▲▪▫•◼◻◾◽🔗"

def _strip_ui_decorations(text: str) -> str:
    """
    Strip UI-only decorations from titles:
    - Qt mnemonics (&)
    - link badges like "[LINK]"
    - your glyph badges
    - file extension (optional, but nice for Explorer)
    """
    if not text:
        return ""
    s = str(text)

    # remove mnemonics
    s = s.replace("&", "")

    # remove common prefixes/badges
    s = s.replace("[LINK]", "").strip()

    # remove glyph badges
    s = s.translate({ord(ch): None for ch in GLYPHS})

    # collapse whitespace
    s = " ".join(s.split())

    return s



class DockMixin:
    """
    Mixin for dock widget management.
    
    Provides methods for creating, managing, and synchronizing dock widgets
    in the main window.
    """
    
    def _init_log_dock(self):
        from setiastro.saspro.system_log_dock import SystemLogDock
        self.system_log_dock = SystemLogDock(self)
        self.log_text = self.system_log_dock.log_text  # keeps _append_log_text compat
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.system_log_dock)
        self.act_toggle_log = self.system_log_dock.toggleViewAction()
        self.act_toggle_log.setText(self.tr("Show System Log Panel"))
    
    def _ensure_dock_host(self):
        if getattr(self, "dock_host", None) is None:
            self.dock_host = DockHostWindow(None)  # None = proper top-level, own taskbar entry
            try:
                self.dock_host.setWindowIcon(self.app_icon)
            except Exception:
                pass
        return self.dock_host

    def _secondary_host_candidate_docks(self) -> list[QDockWidget]:
        docks = []
        for name in (
            "explorer_dock",
            "console_dock",
            "status_log_dock",
            "layers_dock",
            "header_dock",
            "log_dock",
            "system_log_dock",   # ← add this
            "window_shelf",
            "resource_monitor_dock",
        ):
            d = getattr(self, name, None)
            if isinstance(d, QDockWidget):
                docks.append(d)
        return docks

    def _append_log_text(self, text: str):
        try:
            self.system_log_dock.append_text(text)
        except Exception:
            pass
    
    def _hook_stdout_stderr(self):
        """Hook stdout/stderr to redirect to the system log dock."""
        import sys
        from setiastro.saspro.mdi_widgets import QtLogStream

        # Remember original streams so we still print to real console.
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr

        self._qt_stdout = QtLogStream(self._orig_stdout, self)
        self._qt_stderr = QtLogStream(self._orig_stderr, self)

        self._qt_stdout.text_emitted.connect(self._append_log_text)
        self._qt_stderr.text_emitted.connect(self._append_log_text)

        sys.stdout = self._qt_stdout
        sys.stderr = self._qt_stderr
    
    def _register_dock_in_view_menu(self, dock: QDockWidget, action: QAction | None = None):
        """
        Register a dock widget's toggle action in the View → Panels menu.
        
        Args:
            dock: The QDockWidget to register
            action: Optional custom action (if None, uses dock.toggleViewAction())
        """
        if not hasattr(self, "_view_panels_menu"):
            return
        
        if action is None:
            action = dock.toggleViewAction()
        
        self._view_panels_menu.addAction(action)
    
    def _remove_dock_from_view_menu(self, action: QAction):
        """
        Remove a dock's toggle action from the View → Panels menu.
        
        Args:
            action: The action to remove
        """
        if not hasattr(self, "_view_panels_menu"):
            return
        
        self._view_panels_menu.removeAction(action)
    
    def _init_explorer_dock(self):
        host = QWidget(self)
        lay = QVBoxLayout(host)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(4)

        # Optional filter box (super useful)
        self.explorer_filter = QLineEdit(host)
        self.explorer_filter.setPlaceholderText(self.tr("Filter open documents…"))
        self.explorer_filter.textChanged.connect(self._explorer_apply_filter)
        lay.addWidget(self.explorer_filter)

        self.explorer = QTreeWidget(host)
        self.explorer.setObjectName("ExplorerTree")
        self.explorer.setColumnCount(3)
        self.explorer.setHeaderLabels([self.tr("Document"), self.tr("Dims"), self.tr("Type")])

        # Sorting
        self.explorer.setSortingEnabled(True)
        self.explorer.header().setSortIndicatorShown(True)
        self.explorer.sortByColumn(0, Qt.SortOrder.AscendingOrder)

        # Selection/activation behavior
        self.explorer.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.explorer.itemActivated.connect(self._activate_or_open_from_explorer)

        # Inline rename support
        self.explorer.setEditTriggers(
            QAbstractItemView.EditTrigger.EditKeyPressed |
            QAbstractItemView.EditTrigger.SelectedClicked
        )
        self.explorer.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.explorer.customContextMenuRequested.connect(self._on_explorer_context_menu)
        self.explorer.itemChanged.connect(self._on_explorer_item_changed)

        lay.addWidget(self.explorer)

        self.explorer_dock = QDockWidget(self.tr("Explorer"), self)
        self.explorer_dock.setWidget(host)
        self.explorer_dock.setObjectName("ExplorerDock")
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.explorer_dock)

        self._active_doc_delegate = ActiveDocDelegate(
            lambda: getattr(self, "_current_active_doc", None),
            self.explorer
        )
        self.explorer.setItemDelegate(self._active_doc_delegate)

    def _init_console_dock(self):
        self.console = QListWidget()

        # Allow multi-row selection so Select All actually highlights everything
        self.console.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)

        # Right-click context menu
        self.console.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.console.customContextMenuRequested.connect(self._on_console_context_menu)

        self.console_dock = QDockWidget(self.tr("Console / Status"), self)
        self.console_dock.setWidget(self.console)
        self.console_dock.setObjectName("ConsoleDock")
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.console_dock)

    def _init_status_log_dock(self):
        from setiastro.saspro.status_log_dock import StatusLogDock
        from setiastro.saspro.log_bus import LogBus
        
        # Create the dock
        self.status_log_dock = StatusLogDock(self)            # your dock widget class
        self.status_log_dock.setObjectName("StatusLogDock")   # stable name for restoreState/menu
        self.status_log_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea
            | Qt.DockWidgetArea.RightDockWidgetArea
            | Qt.DockWidgetArea.BottomDockWidgetArea
            | Qt.DockWidgetArea.TopDockWidgetArea
        )
        # Default area (will be overridden by restoreState if present)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.status_log_dock)

        # Expose the dock globally so dialogs can show/raise it
        app = QApplication.instance()
        app._sasd_status_console = self.status_log_dock

        # Ensure a global log bus and wire bus -> dock (queued; thread-safe)
        if not hasattr(app, "_sasd_log_bus"):
            app._sasd_log_bus = LogBus()
        app._sasd_log_bus.posted.connect(
            self.status_log_dock.append_line,
            type=Qt.ConnectionType.QueuedConnection
        )

        # First-run placement (only if no prior saved layout)
        self._first_place_status_log_if_needed()

    def _init_layers_dock(self):
        from setiastro.saspro.layers_dock import LayersDock
        
        self.layers_dock = LayersDock(self)
        self.layers_dock.setObjectName("LayersDock") 
        # put it on the right, *below* header viewer if you dock that at right too
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.layers_dock)
        # ensure the header viewer sits above, layers below
        try:
            self.splitDockWidget(self.header_dock, self.layers_dock, Qt.Orientation.Vertical)
        except Exception:
            pass

    def _init_header_viewer_dock(self):
        from setiastro.saspro.header_viewer import HeaderViewerDock
        
        self.header_dock = HeaderViewerDock(self)
        self.header_dock.setObjectName("HeaderViewerDock")
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.header_dock)

        self.header_dock.attach_doc_manager(self.doc_manager)

        try:
            self.header_dock.set_document(self.doc_manager.get_active_document())
        except Exception:
            pass

    def _init_resource_monitor_dock(self):
        """Initialize the System Resource Monitor as a standard utility dock."""
        try:
            from setiastro.saspro.widgets.resource_monitor import ResourceMonitorDock
        except Exception as e:
            print(f"WARNING: Could not initialize System Monitor dock: {e}")
            self.resource_monitor_dock = None
            self.resource_monitor = None
            return

        self.resource_monitor_dock = ResourceMonitorDock(self)
        self.resource_monitor_dock.setObjectName("ResourceMonitorDock")
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea,
                           self.resource_monitor_dock)

        # Tab it alongside the existing right-side stack if present, so it
        # doesn't force a tall new column on first run. restoreState() will
        # override this with the user's saved layout when there is one.
        try:
            anchor = getattr(self, "header_dock", None) or getattr(self, "layers_dock", None)
            if anchor is not None:
                self.tabifyDockWidget(anchor, self.resource_monitor_dock)
        except Exception:
            pass

        # Legacy alias: the floating-overlay attribute is gone. Keep the name
        # defined (None) so any lingering `self.resource_monitor` truthiness
        # checks elsewhere just skip cleanly instead of raising.
        self.resource_monitor = None

    # Back-compat: older call sites (e.g. MainWindow setup) may still invoke
    # the previous overlay entry point. It now builds the dock.
    _init_resource_monitor_overlay = _init_resource_monitor_dock

    def _toggle_resource_monitor(self, checked: bool):
        """Show/hide the resource-monitor dock. Kept for any legacy menu wiring;
        the dock's own toggleViewAction now drives it from View -> Panels."""
        dock = getattr(self, "resource_monitor_dock", None)
        if dock is not None:
            dock.setVisible(bool(checked))

    def _update_monitor_position(self):
        """No-op. The monitor is a docked panel now, not a floating overlay;
        retained so any resizeEvent hook that still calls it won't crash."""
        return

    # We need to hook resizeEvent to call _update_monitor_position.
    # Since this is a mixin, we can't easily override resizeEvent of the MainWindow without being careful.
    # Best way: install an event filter on self, or since we are a mixin mixed into MainWindow, 
    # we can rely on MainWindow calling a specific method or we can patch it... 
    # Actually, MainWindow likely has resizeEvent. 
    # simpler: QTimer check? No.
    # Correct way for Mixin: The MainWindow class should call something. 
    # BUT, I can just installEventFilter(self) ? No, infinite loop risk.
    # 
    # Let's use the 'GeometryMixin' or just add a standard method `_on_resize_for_monitor` 
    # and assume I can hook it in MainWindow.py.


        # âŒ Remove this old line; it let random mouse-over updates hijack the dock:
        # self.currentDocumentChanged.disconnect(self.header_viewer.set_document)  # if previously connected
        # (If you prefer to keep the signal for explicit tab switches, it's fine to leave
        #  it connected--the dock's new guard will ignore non-active/hover docs.)

    def _sync_explorer_to_active_doc(self, doc):
        """Called whenever the active MDI document changes."""
        # Normalize to base doc
        base = self._normalize_base_doc(doc) if doc else None
        self._current_active_doc = base

        # Repaint explorer so the delegate re-draws the accent bar
        self.explorer.viewport().update()

        # Also scroll the active row into view (but don't change selection)
        if base is None:
            return
        for i in range(self.explorer.topLevelItemCount()):
            it = self.explorer.topLevelItem(i)
            if it.data(0, Qt.ItemDataRole.UserRole) is base:
                self.explorer.scrollToItem(it, QAbstractItemView.ScrollHint.EnsureVisible)
                break

    def _all_known_docks(self) -> list[QDockWidget]:
        docks = []
        for name in (
            "explorer_dock",
            "console_dock",
            "status_log_dock",
            "layers_dock",
            "header_dock",
            "log_dock",
            "system_log_dock",   # ← add this
            "window_shelf",
            "resource_monitor_dock",
        ):
            d = getattr(self, name, None)
            if isinstance(d, QDockWidget):
                docks.append(d)

        out = []
        seen = set()
        for d in docks:
            if id(d) not in seen:
                seen.add(id(d))
                out.append(d)
        return out

    def _populate_view_panels_menu(self):
        """Rebuild 'View Panels' with all current dock widgets (ordered nicely)."""
        menu = self._ensure_view_panels_menu()
        menu.clear()
        self._view_panels_actions = {}

        # Collect every QDockWidget that exists right now
        docks = self._all_known_docks()

        # Friendly ordering for common ones; others follow alphabetically.
        order_hint = {
            self.tr("Explorer"): 10,
            self.tr("Console / Status"): 20,
            self.tr("Header Viewer"): 30,
            self.tr("Layers"): 40,
            self.tr("Window Shelf"): 50,
            self.tr("System Monitor"): 55,
            self.tr("Command Search"): 60,
            self.tr("System Log"): 70,     # ← add this
            self.tr("Stacking Log"): 80,   # ← and this for status_log_dock
        }
        
        # The resource monitor is a normal utility dock now, so it is registered
        # through the standard dock loop below (it's listed in _all_known_docks).
        # Keep act_toggle_monitor pointing at its toggle for any legacy callers.
        rmd = getattr(self, "resource_monitor_dock", None)
        self.act_toggle_monitor = rmd.toggleViewAction() if rmd is not None else None

        def key_fn(d: QDockWidget):
            t = d.windowTitle()
            return (order_hint.get(t, 1000), t.lower())

        for dock in sorted(docks, key=key_fn):
            self._register_dock_in_view_menu(dock)
        # System Monitor is included in the loop above, like every other dock.

    def _add_doc_to_explorer(self, doc):
        base = self._normalize_base_doc(doc)

        # de-dupe by identity on base
        for i in range(self.explorer.topLevelItemCount()):
            it = self.explorer.topLevelItem(i)
            if it.data(0, Qt.ItemDataRole.UserRole) is base:
                self._refresh_explorer_row(it, base)
                return

        it = QTreeWidgetItem()
        it.setData(0, Qt.ItemDataRole.UserRole, base)

        # Make name editable; other columns read-only
        it.setFlags(it.flags() | Qt.ItemFlag.ItemIsEditable | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)

        self._refresh_explorer_row(it, base)

        fp = (base.metadata or {}).get("file_path")
        if fp:
            it.setToolTip(0, fp)

        self.explorer.addTopLevelItem(it)

        # keep row label in sync with edits/resizes/renames
        try:
            base.changed.connect(lambda *_, d=base: self._update_explorer_item_for_doc(d))
        except Exception:
            pass


    def _remove_doc_from_explorer(self, doc):
        base = self._normalize_base_doc(doc)
        for i in range(self.explorer.topLevelItemCount()):
            it = self.explorer.topLevelItem(i)
            d = it.data(0, Qt.ItemDataRole.UserRole)
            if d is doc or d is base:
                self.explorer.takeTopLevelItem(i)
                break


    def _update_explorer_item_for_doc(self, doc):
        for i in range(self.explorer.topLevelItemCount()):
            it = self.explorer.topLevelItem(i)
            if it.data(0, Qt.ItemDataRole.UserRole) is doc:
                self._refresh_explorer_row(it, doc)
                return

    def _refresh_explorer_row(self, item, doc):
        # Column 0: display name (NO glyph decorations)
        name = _strip_ui_decorations(doc.display_name() or "Untitled")

        name_no_ext, _ext = os.path.splitext(name)
        if name_no_ext:
            name = name_no_ext

        item.setText(0, name)

        # Column 1: dims
        dims = ""
        try:
            import numpy as np
            arr = getattr(doc, "image", None)
            if isinstance(arr, np.ndarray) and arr.size:
                h, w = arr.shape[:2]
                c = arr.shape[2] if arr.ndim == 3 else 1
                dims = f"{h}×{w}×{c}"
        except Exception:
            pass
        item.setText(1, dims)

        # Column 2: type/bit-depth (whatever you have available)
        md = (doc.metadata or {})
        bit = md.get("bit_depth") or md.get("dtype") or ""
        kind = md.get("format") or md.get("doc_type") or ""
        t = " / ".join([s for s in (str(kind), str(bit)) if s and s != "None"])
        item.setText(2, t)

    def _on_explorer_item_changed(self, item, col: int):
        if col != 0:
            return

        doc = item.data(0, Qt.ItemDataRole.UserRole)
        if doc is None:
            return

        new_name = (item.text(0) or "").strip()
        if not new_name:
            # revert to current doc name
            self._refresh_explorer_row(item, doc)
            return

        # Avoid infinite loops: only apply if changed
        cur = _strip_ui_decorations(doc.display_name() or "Untitled")
        cur_no_ext, _ = os.path.splitext(cur)
        cur = cur_no_ext or cur
        if new_name == cur:
            return

        try:
            doc.metadata["display_name"] = new_name
        except Exception:
            # if metadata missing or immutable, revert
            self._refresh_explorer_row(item, doc)
            return

        try:
            doc.changed.emit()
        except Exception:
            pass

    def _on_explorer_context_menu(self, pos):
        it = self.explorer.itemAt(pos)
        if it is None:
            return
        doc = it.data(0, Qt.ItemDataRole.UserRole)
        if doc is None:
            return

        menu = QMenu(self.explorer)
        a_rename = menu.addAction(self.tr("Rename Document…"))
        a_close  = menu.addAction(self.tr("Close Document"))
        menu.addSeparator()
        a_copy_path = menu.addAction(self.tr("Copy File Path"))
        a_reveal = menu.addAction(self.tr("Reveal in File Manager"))
        menu.addSeparator()
        a_send_shelf = menu.addAction(self.tr("Send View to Shelf"))  # acts on active view for this doc

        act = menu.exec(self.explorer.viewport().mapToGlobal(pos))
        if act == a_rename:
            # Start inline editing
            self.explorer.editItem(it, 0)

        elif act == a_close:
            # close only if no other subwindows show it: you already do that in _on_view_about_to_close,
            # but Explorer close is explicit; just close all views of this doc then docman.close_document.
            try:
                self._close_all_views_for_doc(doc)
            except Exception:
                pass

        elif act == a_copy_path:
            fp = (doc.metadata or {}).get("file_path", "")
            if fp:
                QGuiApplication.clipboard().setText(fp)

        elif act == a_reveal:
            fp = (doc.metadata or {}).get("file_path", "")
            if fp:
                self._reveal_in_file_manager(fp)

        elif act == a_send_shelf:
            sw = self._find_subwindow_for_doc(doc)
            if sw and hasattr(sw.widget(), "_send_to_shelf"):
                try:
                    sw.widget()._send_to_shelf()
                except Exception:
                    pass

    def _close_all_views_for_doc(self, doc):
        base = self._normalize_base_doc(doc)
        subs = list(self.mdi.subWindowList())
        for sw in subs:
            w = sw.widget()
            if getattr(w, "base_document", None) is base:
                try:
                    # NEW: make sure the shelf cannot keep a stale reference
                    if hasattr(self, "window_shelf") and self.window_shelf:
                        self.window_shelf.remove_for_subwindow(sw)
                except Exception:
                    pass

                try:
                    sw.close()
                except Exception:
                    pass

        try:
            self.docman.close_document(base)
        except Exception:
            pass

    def _reveal_in_file_manager(self, path: str):
        import sys, os, subprocess
        try:
            if sys.platform.startswith("win"):
                subprocess.Popen(["explorer", "/select,", os.path.normpath(path)])
            elif sys.platform == "darwin":
                subprocess.Popen(["open", "-R", path])
            else:
                # best-effort on Linux
                subprocess.Popen(["xdg-open", os.path.dirname(path)])
        except Exception:
            pass

    def _explorer_apply_filter(self, text: str):
        t = (text or "").strip().lower()
        for i in range(self.explorer.topLevelItemCount()):
            it = self.explorer.topLevelItem(i)
            name = (it.text(0) or "").lower()
            fp = (it.toolTip(0) or "").lower()
            hide = bool(t) and (t not in name) and (t not in fp)
            it.setHidden(hide)

    def _move_dock_to_host(self, dock: QDockWidget, area=Qt.DockWidgetArea.LeftDockWidgetArea):
        if dock is None:
            return

        host = self._ensure_dock_host()

        try:
            self.removeDockWidget(dock)
        except Exception:
            pass

        try:
            old_win = dock.window()
            if isinstance(old_win, QMainWindow) and old_win is not self and old_win is not host:
                old_win.removeDockWidget(dock)
            elif isinstance(old_win, QMainWindow) and old_win is host:
                old_win.removeDockWidget(dock)
        except Exception:
            pass

        host.addDockWidget(area, dock)
        dock.show()
        host.show()
        host.raise_()
        host.activateWindow()

    def _move_dock_to_main(self, dock: QDockWidget, area=Qt.DockWidgetArea.LeftDockWidgetArea):
        if dock is None:
            return

        try:
            old_win = dock.window()
            if isinstance(old_win, QMainWindow) and old_win is not self:
                old_win.removeDockWidget(dock)
        except Exception:
            pass

        self.addDockWidget(area, dock)
        dock.show()
        self.raise_()
        self.activateWindow()

    def _dock_is_in_host(self, dock: QDockWidget) -> bool:
        host = getattr(self, "dock_host", None)
        if dock is None or host is None:
            return False
        try:
            return dock.window() is host
        except Exception:
            return False

    def _send_default_docks_to_host(self):
        host = self._ensure_dock_host()

        # Save the current main window dock layout BEFORE moving docks out
        # so we can restore it if the user returns panels to main
        try:
            s = self.settings
            k = self._mw_key()
            s.setValue(f"{k}/pre_host_dock_state", self.saveState())
            s.sync()
        except Exception:
            pass

        mapping = [
            (getattr(self, "explorer_dock", None), Qt.DockWidgetArea.LeftDockWidgetArea),
            (getattr(self, "layers_dock", None), Qt.DockWidgetArea.RightDockWidgetArea),
            (getattr(self, "header_dock", None), Qt.DockWidgetArea.RightDockWidgetArea),
            (getattr(self, "console_dock", None), Qt.DockWidgetArea.BottomDockWidgetArea),
            (getattr(self, "status_log_dock", None), Qt.DockWidgetArea.BottomDockWidgetArea),
            (getattr(self, "log_dock", None), Qt.DockWidgetArea.BottomDockWidgetArea),
            (getattr(self, "resource_monitor_dock", None), Qt.DockWidgetArea.RightDockWidgetArea),
        ]

        for dock, area in mapping:
            if isinstance(dock, QDockWidget):
                self._move_dock_to_host(dock, area)

        try:
            if getattr(self, "header_dock", None) and getattr(self, "layers_dock", None):
                host.splitDockWidget(self.header_dock, self.layers_dock, Qt.Orientation.Vertical)
        except Exception:
            pass

    def _show_dock_host(self):
        host = self._ensure_dock_host()
        # If minimized, restore it properly
        if host.isMinimized():
            host.setWindowState(
                host.windowState() & ~Qt.WindowState.WindowMinimized
                | Qt.WindowState.WindowActive
            )
        host.show()
        host.raise_()
        host.activateWindow()

    def save_dock_host_state(self):
        host = getattr(self, "dock_host", None)
        s = self.settings
        k = self._mw_key()

        # Check if any docks are actually in the host right now
        in_host = []
        if host is not None:
            for dock in self._all_known_docks():
                if self._dock_is_in_host(dock):
                    in_host.append(dock.objectName())

        if not in_host:
            # No docks in host — mark as inactive so we don't restore it next launch
            s.setValue(f"{k}/dock_host/active", False)
            s.sync()
            return

        # Docks are in host — save everything
        try:
            s.setValue(f"{k}/dock_host/active", True)
            s.setValue(f"{k}/dock_host/geometry", host.saveGeometry())
            s.setValue(f"{k}/dock_host/state", host.saveState())
        except Exception:
            pass

        s.setValue(f"{k}/dock_host/docks", in_host)
        s.sync()

    def restore_dock_host_state(self):
        s = self.settings
        k = self._mw_key()

        was_active = s.value(f"{k}/dock_host/active", False, type=bool)
        if not was_active:
            return

        in_host = s.value(f"{k}/dock_host/docks", [], type=list)
        if not in_host:
            return

        name_map = {d.objectName(): d for d in self._all_known_docks()}
        host = self._ensure_dock_host()
        host.hide()

        for name in in_host:
            dock = name_map.get(name)
            if dock is not None:
                self._move_dock_to_host(dock)

        geom = s.value(f"{k}/dock_host/geometry", None)
        if geom is not None and len(geom) > 0:
            try:
                host.restoreGeometry(geom)
            except Exception:
                pass

        # Show first, then restoreState — Qt needs window visible for layout
        host.show()

        state = s.value(f"{k}/dock_host/state", None)
        if state is not None and len(state) > 0:
            try:
                host.restoreState(state)
            except Exception:
                pass

    def _return_all_host_docks_to_main(self):
        default_areas = {
            "ExplorerDock": Qt.DockWidgetArea.LeftDockWidgetArea,
            "ConsoleDock": Qt.DockWidgetArea.BottomDockWidgetArea,
            "StatusLogDock": Qt.DockWidgetArea.RightDockWidgetArea,
            "LayersDock": Qt.DockWidgetArea.RightDockWidgetArea,
            "HeaderViewerDock": Qt.DockWidgetArea.RightDockWidgetArea,
            "LogDock": Qt.DockWidgetArea.BottomDockWidgetArea,
            "ResourceMonitorDock": Qt.DockWidgetArea.RightDockWidgetArea,
        }

        for dock in self._all_known_docks():
            if self._dock_is_in_host(dock):
                area = default_areas.get(dock.objectName(), Qt.DockWidgetArea.LeftDockWidgetArea)
                self._move_dock_to_main(dock, area)

        try:
            if getattr(self, "header_dock", None) and getattr(self, "layers_dock", None):
                self.splitDockWidget(self.header_dock, self.layers_dock, Qt.Orientation.Vertical)
        except Exception:
            pass

        # Restore the main window dock layout from before panels were sent to host
        try:
            s = self.settings
            k = self._mw_key()
            pre_state = s.value(f"{k}/pre_host_dock_state", None)
            if pre_state is not None and len(pre_state) > 0:
                self.restoreState(pre_state)
        except Exception:
            pass

        # Close the secondary dock host — it's empty now
        try:
            host = getattr(self, "dock_host", None)
            if host is not None:
                host.hide()
                host.close()
                self.dock_host = None
        except Exception:
            pass