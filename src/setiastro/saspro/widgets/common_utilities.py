"""
Common UI utilities and shared components.

This module provides centralized implementations of commonly used
UI components and utilities to avoid code duplication across the codebase.

Included:
- AboutDialog: Standard about dialog
- ProjectSaveWorker: Background thread for saving projects
- install_crash_handlers: Global exception/crash handling setup
- UI decoration helpers: DECOR_GLYPHS, strip_ui_decorations
"""

from __future__ import annotations

import sys
import os
import threading
import traceback
import logging
import atexit
from typing import TYPE_CHECKING, Optional, List, Any

from PyQt6.QtCore import QThread, pyqtSignal, QTimer, Qt
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QMessageBox

if TYPE_CHECKING:
    from PyQt6.QtWidgets import QApplication, QMdiArea


# ---------------------------------------------------------------------------
# Version and Build Info (should be imported from main module)
# ---------------------------------------------------------------------------

def get_version() -> str:
    """Get application version from main module."""
    try:
        from setiastrosuitepro import VERSION
        return VERSION
    except ImportError:
        return "Unknown"


def get_build_timestamp() -> str:
    """Get human-friendly build timestamp from main module."""
    try:
        from setiastrosuitepro import BUILD_TIMESTAMP
    except ImportError:
        return "Unknown"

    if BUILD_TIMESTAMP == "dev":
        # No generated build_info → running from local source checkout
        return "Running locally from source code"
    return BUILD_TIMESTAMP


# ---------------------------------------------------------------------------
# About Dialog
# ---------------------------------------------------------------------------

class AboutDialog(QDialog):
    """
    Standard About dialog for Seti Astro Suite.
    """

    def __init__(self, parent: Optional[Any] = None, version: str = "", build_timestamp: str = ""):
        super().__init__(parent)
        self.setWindowTitle(self.tr("About Seti Astro Suite"))

        # Get version info if not provided
        if not version:
            version = get_version()

        # Normalize build_timestamp
        if not build_timestamp:
            build_timestamp = get_build_timestamp()
        else:
            # If someone passed the raw sentinel from the main module
            if build_timestamp == "dev":
                build_timestamp = "Running locally from source code"

        layout = QVBoxLayout()

        # Build about text with optional build timestamp
        about_lines = [
            f"<h2>Seti Astro's Suite Pro {version}</h2>",
            f"<p>{self.tr('Written by Franklin Marek')}</p>",
            f"<p>{self.tr('Collaborators: Fabio Tempera')}</p>",
            f"<p>{self.tr('Copyright © 2025 Seti Astro')}</p>",
        ]

        if build_timestamp and build_timestamp != "Unknown":
            about_lines.append(f"<p><b>{self.tr('Build:')}</b> {build_timestamp}</p>")

        about_lines.extend([
            f"<p>{self.tr('Website:')} <a href='http://www.setiastro.com'>www.setiastro.com</a></p>",
            f"<p>{self.tr('Donations:')} <a href='https://www.setiastro.com/checkout/donate?donatePageId=65ae7e7bac20370d8c04c1ab'>{self.tr('Click here to donate')}</a></p>",
        ])

        about_text = "".join(about_lines)

        label = QLabel(about_text)
        label.setTextFormat(Qt.TextFormat.RichText)
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        label.setOpenExternalLinks(True)

        layout.addWidget(label)
        self.setLayout(layout)



# ---------------------------------------------------------------------------
# Project Save Worker
# ---------------------------------------------------------------------------

class ProjectSaveWorker(QThread):
    """
    Background thread for saving projects.
    
    Emits 'ok' signal on success, 'error' signal with message on failure.
    
    Args:
        path: Path to save the project to
        docs: List of documents to save
        shortcuts: Shortcuts configuration
        mdi: MDI area reference
        compress: Whether to compress the project
        window_shelf: Optional window shelf reference
        parent: Parent QObject
        
    Example:
        worker = ProjectSaveWorker(path, docs, shortcuts, mdi, compress=True)
        worker.ok.connect(on_save_complete)
        worker.error.connect(on_save_error)
        worker.start()
    """
    
    ok = pyqtSignal()
    error = pyqtSignal(str)
    
    def __init__(
        self,
        path: str,
        docs: List[Any],
        shortcuts: Any,
        mdi: 'QMdiArea',
        compress: bool,
        window_shelf: Optional[Any] = None,
        parent: Optional[Any] = None
    ):
        super().__init__(parent)
        self.path = path
        self.docs = docs
        self.shortcuts = shortcuts
        self.mdi = mdi
        self.compress = compress
        self.window_shelf = window_shelf
    
    def run(self) -> None:
        """Execute the save operation in background thread."""
        try:
            from setiastro.saspro.project_io import ProjectWriter
            ProjectWriter.write(
                self.path,
                docs=self.docs,
                shortcuts=self.shortcuts,
                mdi=self.mdi,
                compress=self.compress,
                shelf=self.window_shelf,
            )
            self.ok.emit()
        except Exception as e:
            self.error.emit(str(e))


# ---------------------------------------------------------------------------
# UI Decoration Helpers
# ---------------------------------------------------------------------------

DECOR_GLYPHS = "■●◆▲▪▫•◼◻◾◽"

def strip_ui_decorations(s: str) -> str:
    """
    Remove UI decoration glyphs and prefixes from a string.
    
    Strips:
    - Leading glyph characters followed by space
    - "Active View: " prefix
    
    Args:
        s: String to clean
        
    Returns:
        Cleaned string without decorations
        
    Example:
        >>> strip_ui_decorations("● Active View: My Image.fits")
        'My Image.fits'
    """
    s = s or ""
    
    # Strip any number of leading glyph+space
    while len(s) >= 2 and s[1] == " " and s[0] in DECOR_GLYPHS:
        s = s[2:]
    
    # Strip leading Active prefix if present
    ACTIVE = "Active View: "
    if s.startswith(ACTIVE):
        s = s[len(ACTIVE):]
    
    return s


# Alias for backward compatibility
_strip_ui_decorations = strip_ui_decorations


# ---------------------------------------------------------------------------
# Crash/Exception Handlers
# ---------------------------------------------------------------------------

def install_crash_handlers(app: 'QApplication') -> None:
    """
    Install global crash and exception handlers for the application.
    
    Sets up:
    1. faulthandler for hard crashes (segfaults) → saspro_crash.log
    2. sys.excepthook for uncaught main thread exceptions
    3. threading.excepthook for uncaught background thread exceptions
    
    All exceptions are logged and displayed in a dialog to the user.
    
    Args:
        app: The QApplication instance
        
    Example:
        app = QApplication(sys.argv)
        install_crash_handlers(app)
    """
    import faulthandler
    
    # 1) Hard crashes (segfaults, access violations) → saspro_crash.log
    try:
        _crash_log = open("saspro_crash.log", "w", encoding="utf-8", errors="replace")
        faulthandler.enable(file=_crash_log, all_threads=True)
        atexit.register(_crash_log.close)
    except Exception:
        logging.exception("Failed to enable faulthandler")
    
    def _show_dialog(title: str, head: str, details: str) -> None:
        """Show error dialog marshaled to main thread."""
        def _ui():
            m = QMessageBox(app.activeWindow())
            m.setIcon(QMessageBox.Icon.Critical)
            m.setWindowTitle(title)
            m.setText(head)
            from PyQt6.QtCore import QCoreApplication
            m.setInformativeText(QCoreApplication.translate("CrashHandler", "Details are available below and in saspro.log."))
            if details:
                m.setDetailedText(details)
            m.setStandardButtons(QMessageBox.StandardButton.Ok)
            m.exec()
        QTimer.singleShot(0, _ui)
    
    # 2) Any uncaught exception on the main thread
    def _excepthook(exc_type, exc_value, exc_tb):
        tb = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        logging.error("Uncaught exception:\n%s", tb)
        _show_dialog(
            QCoreApplication.translate("CrashHandler", "Unhandled Exception"),
            f"{exc_type.__name__}: {exc_value}",
            tb
        )
    
    sys.excepthook = _excepthook
    
    # 3) Any uncaught exception in background threads (Py3.8+)
    def _threadhook(args: threading.ExceptHookArgs):
        tb = "".join(traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback))
        logging.error("Uncaught thread exception (%s):\n%s", args.thread.name, tb)
        _show_dialog(
            QCoreApplication.translate("CrashHandler", "Unhandled Thread Exception"),
            f"{args.exc_type.__name__}: {args.exc_value}",
            tb
        )
    
    try:
        threading.excepthook = _threadhook  # type: ignore[attr-defined]
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

__all__ = [
    'AboutDialog',
    'ProjectSaveWorker',
    'DECOR_GLYPHS',
    'strip_ui_decorations',
    '_strip_ui_decorations',  # Backward compatibility
    'install_crash_handlers',
    'get_version',
    'get_build_timestamp',
]
