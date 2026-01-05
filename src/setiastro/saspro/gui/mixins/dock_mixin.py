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
    QAbstractItemView, QApplication, QLineEdit, QMenu
)
from PyQt6.QtGui import QTextCursor, QAction, QGuiApplication

if TYPE_CHECKING:
    from PyQt6.QtWidgets import QAction

import os

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
        """Initialize the system log dock widget."""
        self.log_dock = QDockWidget(self.tr("System Log"), self)
        self.log_dock.setObjectName("LogDock")
        self.log_dock.setAllowedAreas(
            Qt.DockWidgetArea.BottomDockWidgetArea
            | Qt.DockWidgetArea.TopDockWidgetArea
        )

        self.log_text = QPlainTextEdit(self.log_dock)
        self.log_text.setReadOnly(True)
        self.log_text.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.log_dock.setWidget(self.log_text)

        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.log_dock)

        self.act_toggle_log = self.log_dock.toggleViewAction()
        self.act_toggle_log.setText(self.tr("Show System Log Panel"))
    
    def _append_log_text(self, text: str):
        """Append text to the system log dock."""
        if not text:
            return
        # Append to the bottom and keep view scrolled
        cursor = self.log_text.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(text)
        self.log_text.setTextCursor(cursor)
        self.log_text.ensureCursorVisible()
    
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

        dock = QDockWidget(self.tr("Explorer"), self)
        dock.setWidget(host)
        dock.setObjectName("ExplorerDock")
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)

    def _init_console_dock(self):
        self.console = QListWidget()

        # Allow multi-row selection so Select All actually highlights everything
        self.console.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)

        # Right-click context menu
        self.console.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.console.customContextMenuRequested.connect(self._on_console_context_menu)

        dock = QDockWidget(self.tr("Console / Status"), self)
        dock.setWidget(self.console)
        dock.setObjectName("ConsoleDock")
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock)

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
        
        self.header_viewer = HeaderViewerDock(self)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.header_viewer)

        # Bind the dock to DocManager so it tracks the ACTIVE subwindow only.
        # Make sure self.doc_manager.set_mdi_area(mdi) was already called.
        self.header_viewer.attach_doc_manager(self.doc_manager)

        # Optional: keep it strictly active-only (default). Flip to True to restore hover-follow behavior.
        # self.header_viewer.set_follow_hover(False)

        # Seed once with whatever is currently active.
        try:
            self.header_viewer.set_document(self.doc_manager.get_active_document())
        except Exception:
            pass

    def _init_resource_monitor_overlay(self):
        """Initialize the QML System Resource Monitor as a floating overlay."""
        try:
            from setiastro.saspro.widgets.resource_monitor import SystemMonitorWidget
            
            # Create as a child of the central widget or self to sit on top
            # Using self (QMainWindow) allows it to float over everything including status bar if we want,
            # but usually we want it over MDI area. Let's try self first for "floating" feel.
            self.resource_monitor = SystemMonitorWidget(self)
            self.resource_monitor.setObjectName("ResourceMonitorOverlay")
            
            # Make it a proper independent window to allow true transparency (translucent background)
            # without black artifacts from parent composition.
            # Fixed: Removed WindowStaysOnTopHint to allow it to be obscured by other apps (Alt-Tab support)
            self.resource_monitor.setWindowFlags(
                Qt.WindowType.Window |
                Qt.WindowType.FramelessWindowHint |
                Qt.WindowType.Tool
            )
            
            # Sizing and Transparency
            self.resource_monitor.setFixedSize(200, 60)
            # self.resource_monitor.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True) # Optional: if we want click-through
            
            
            # Initial placement (will be updated by resizeEvent)
            self._update_monitor_position()
            
            # Defer visibility to MainWindow.showEvent to prevent appearing before main window
            # visible = self.settings.value("ui/resource_monitor_visible", True, type=bool)
            # if visible:
            #    self.resource_monitor.show()
            # else:
            #    self.resource_monitor.hide()
        except Exception as e:
            print(f"WARNING: Could not initialize System Monitor overlay: {e}")
            self.resource_monitor = None

    def _toggle_resource_monitor(self, checked: bool):
        """Toggle floating monitor visibility."""
        if hasattr(self, 'resource_monitor') and self.resource_monitor:
            if checked:
                self.resource_monitor.show()
                self._update_monitor_position()
            else:
                self.resource_monitor.hide()
            self.settings.setValue("ui/resource_monitor_visible", checked)

    def _update_monitor_position(self):
        """Snap monitor to bottom-right corner or restore saved position."""
        if hasattr(self, 'resource_monitor') and self.resource_monitor:
            from PyQt6.QtCore import QPoint
            
            # Check for saved position first
            saved_x = self.settings.value("ui/resource_monitor_pos_x", type=int)
            saved_y = self.settings.value("ui/resource_monitor_pos_y", type=int)
            
            if saved_x != 0 and saved_y != 0: # Basic validity check (0,0 is unlikely to be desired but also default if missing)
                 # Actually 0,0 is valid but type=int returns 0 if missing. 
                 # Let's check string existence to be safer or just accept 0 if set.
                 # Checking existence via `contains` is better but value() logic is ok for now.
                 if self.settings.contains("ui/resource_monitor_pos_x"):
                     self.resource_monitor.move(saved_x, saved_y)
                     self.resource_monitor.raise_()
                     return

            m = 5  # margin

            screen = self.screen()
            geom = screen.availableGeometry()

            mw = self.resource_monitor.width()
            mh = self.resource_monitor.height()

            x = geom.x() + geom.width()  - mw - m
            y = geom.y() + geom.height() - mh - m

            self.resource_monitor.move(x, y)
            self.resource_monitor.raise_()

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

    def _populate_view_panels_menu(self):
        """Rebuild 'View Panels' with all current dock widgets (ordered nicely)."""
        menu = self._ensure_view_panels_menu()
        menu.clear()
        self._view_panels_actions = {}

        # Collect every QDockWidget that exists right now
        docks: list[QDockWidget] = self.findChildren(QDockWidget)

        # Friendly ordering for common ones; others follow alphabetically.
        order_hint = {
            self.tr("Explorer"): 10,
            self.tr("Console / Status"): 20,
            self.tr("Header Viewer"): 30,
            self.tr("Layers"): 40,
            self.tr("Window Shelf"): 50,
            self.tr("Command Search"): 60,
        }
        
        # Add special action for overlay monitor
        mon_act = QAction(self.tr("System Monitor"), self)
        mon_act.setCheckable(True)
        mon_act.setChecked(self.settings.value("ui/resource_monitor_visible", True, type=bool))
        mon_act.triggered.connect(self._toggle_resource_monitor)
        
        # We need to insert it into the logic that populates the menu.
        # But 'dock_mixin' automates menu from self.findChildren(QDockWidget).
        # So we have to manually inject this action into the "Panels" menu if possible
        # or expose it such that main_window can add it.
        # 
        # Easier: allow main_window to add it, or ...
        # If I can't easily see where menu is built, I'll bind it to self.act_toggle_monitor = mon_act
        self.act_toggle_monitor = mon_act

        def key_fn(d: QDockWidget):
            t = d.windowTitle()
            return (order_hint.get(t, 1000), t.lower())

        for dock in sorted(docks, key=key_fn):
            self._register_dock_in_view_menu(dock)
            
        if hasattr(self, "act_toggle_monitor"):
             menu.addSeparator()
             menu.addAction(self.act_toggle_monitor)

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
                    sw.close()
                except Exception:
                    pass
        # If none left (or even if close failed), try docman close defensively
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
