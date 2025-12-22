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
    QAbstractItemView, QApplication
)
from PyQt6.QtGui import QTextCursor, QAction

if TYPE_CHECKING:
    from PyQt6.QtWidgets import QAction


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
        self.explorer = QListWidget()
        # Enter/Return or single-activation: focus if open, else open
        self.explorer.itemActivated.connect(self._activate_or_open_from_explorer)
        # Double-click: same behavior
        self.explorer.itemDoubleClicked.connect(self._activate_or_open_from_explorer)

        dock = QDockWidget(self.tr("Explorer"), self)
        dock.setWidget(self.explorer)
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
        """Snap monitor to bottom-right corner."""
        if hasattr(self, 'resource_monitor') and self.resource_monitor:
            from PyQt6.QtCore import QPoint
            m = 5 # margin
            # Position relative to the main window geometry
            w = self.resource_monitor.width()
            h = self.resource_monitor.height()
            
            # Anchor to bottom-right of the window
            x = self.width() - w - m
            y = self.height() - h - m
            
            # Map local MainWindow coordinates to Global Screen coordinates
            # This is required because resource_monitor is a Top-Level Window (for transparency)
            global_pos = self.mapToGlobal(QPoint(x, y))
            self.resource_monitor.move(global_pos)
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
            "Explorer": 10,
            "Console / Status": 20,
            "Header Viewer": 30,
            "Layers": 40,
            "Window Shelf": 50,
            "Command Search": 60,
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
        for i in range(self.explorer.count()):
            it = self.explorer.item(i)
            if it.data(Qt.ItemDataRole.UserRole) is base:
                # refresh text in case dims/name changed
                it.setText(self._format_explorer_title(base))
                return

        item = QListWidgetItem(self._format_explorer_title(base))
        item.setData(Qt.ItemDataRole.UserRole, base)
        fp = (base.metadata or {}).get("file_path")
        if fp:
            item.setToolTip(fp)
        self.explorer.addItem(item)

        # keep row label in sync with edits/resizes/renames
        try:
            base.changed.connect(lambda *_: self._update_explorer_item_for_doc(base))
        except Exception:
            pass

    def _remove_doc_from_explorer(self, doc):
        """
        Remove either the exact doc or its base (handles ROI proxies).
        """
        base = self._normalize_base_doc(doc)
        for i in range(self.explorer.count()):
            it = self.explorer.item(i)
            d = it.data(Qt.ItemDataRole.UserRole)
            if d is doc or d is base:
                self.explorer.takeItem(i)
                break

