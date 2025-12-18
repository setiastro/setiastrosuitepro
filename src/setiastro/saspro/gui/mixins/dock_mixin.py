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
from PyQt6.QtGui import QTextCursor

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
        self.log_dock = QDockWidget("System Log", self)
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
        self.act_toggle_log.setText("Show System Log Panel")
    
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

        dock = QDockWidget("Explorer", self)
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

        dock = QDockWidget("Console / Status", self)
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

        def key_fn(d: QDockWidget):
            t = d.windowTitle()
            return (order_hint.get(t, 1000), t.lower())

        for dock in sorted(docks, key=key_fn):
            self._register_dock_in_view_menu(dock)

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

