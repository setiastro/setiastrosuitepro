#src/setiastro/saspro/widgets/common_utilities.py
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

from PyQt6.QtCore import QThread, pyqtSignal, QTimer, Qt, QCoreApplication
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                             QMessageBox, QPushButton, QPlainTextEdit)

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
# Licensing
# ---------------------------------------------------------------------------

SOURCE_URL  = "https://github.com/setiastro/setiastrosuitepro"
GPL_URL     = "https://www.gnu.org/licenses/gpl-3.0.html"


def _license_dir():
    """
    Directory holding LICENSE (GPLv3 text) and license.txt (bundled
    third-party licences).

    Frozen build: the installer drops both alongside the executable.
    Source checkout: walk up from this file until LICENSE turns up.
    """
    from pathlib import Path

    if getattr(sys, "frozen", False):
        base = Path(sys.executable).resolve().parent
        if (base / "LICENSE").exists() or (base / "license.txt").exists():
            return base
        mei = getattr(sys, "_MEIPASS", None)      # onefile fallback
        if mei:
            return Path(mei)
        return base

    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "LICENSE").exists():
            return parent
    return here.parent


def find_license_file(name: str) -> Optional[str]:
    """Absolute path to a licence file if it shipped, else None."""
    from pathlib import Path
    try:
        p = Path(_license_dir()) / name
        return str(p) if p.is_file() else None
    except Exception:
        return None


class LicenseViewer(QDialog):
    """
    Plain-text licence viewer.

    Deliberately in-app rather than QDesktopServices.openUrl: LICENSE has no
    file extension, so handing it to the shell prompts "how do you want to
    open this file?" on Windows and does something unpredictable elsewhere.
    """

    def __init__(self, parent, title: str, path: str):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(780, 640)

        lay = QVBoxLayout(self)

        view = QPlainTextEdit(self)
        view.setReadOnly(True)
        view.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        mono = QFont("Courier New")
        mono.setStyleHint(QFont.StyleHint.Monospace)
        mono.setPointSize(9)
        view.setFont(mono)
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                view.setPlainText(fh.read())
        except OSError as e:
            view.setPlainText(f"Could not read {path}\n\n{e}")
        lay.addWidget(view, 1)

        row = QHBoxLayout()
        row.addStretch(1)
        btn = QPushButton(self.tr("Close"))
        btn.clicked.connect(self.accept)
        row.addWidget(btn)
        lay.addLayout(row)

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
            f"<p>{self.tr('By Franklin Marek')}</p>",
            f"<p>{self.tr('Copyright © 2026 Seti Astro')}</p>",
        ]

        if build_timestamp and build_timestamp != "Unknown":
            about_lines.append(f"<p><b>{self.tr('Build:')}</b> {build_timestamp}</p>")

        about_lines.extend([
            f"<p>{self.tr('Website:')} <a href='https://www.setiastro.com'>www.setiastro.com</a></p>",
            f"<p>{self.tr('Source code:')} <a href='{SOURCE_URL}'>{SOURCE_URL}</a></p>",
            f"<p>{self.tr('Donations:')} <a href='https://www.setiastro.com/checkout/donate?donatePageId=65ae7e7bac20370d8c04c1ab'>{self.tr('Click here to donate')}</a></p>",
        ])

        # GPLv3 notice. Kept as the standard wording so it reads as the licence
        # notice it is, not a paraphrase.
        gpl_para = self.tr(
            "Seti Astro Suite Pro is free software: you can redistribute it and/or "
            "modify it under the terms of the GNU General Public License as published "
            "by the Free Software Foundation, version 3 of the License."
        )
        warranty_para = self.tr(
            "This program is distributed in the hope that it will be useful, but "
            "WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY "
            "or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License "
            "for more details."
        )
        gpl_link_text = self.tr("Read the GNU General Public License v3")

        about_lines.extend([
            "<hr>",
            f"<p style='font-size:11px;'>{gpl_para}</p>",
            f"<p style='font-size:11px;'>{warranty_para}</p>",
            f"<p style='font-size:11px;'><a href='{GPL_URL}'>{gpl_link_text}</a></p>",
        ])

        about_text = "".join(about_lines)

        label = QLabel(about_text)
        label.setTextFormat(Qt.TextFormat.RichText)
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        label.setOpenExternalLinks(True)
        label.setWordWrap(True)

        layout.addWidget(label)

        # Buttons for the licence files that shipped with this build. Shown only
        # when the file is actually present, so a source checkout or an install
        # that missed them doesn't offer a dead button.
        btn_row = QHBoxLayout()

        _lic = find_license_file("LICENSE")
        if _lic:
            b_lic = QPushButton(self.tr("License"))
            b_lic.setToolTip(self.tr("GNU General Public License v3"))
            b_lic.clicked.connect(
                lambda _checked=False, p=_lic: LicenseViewer(
                    self, self.tr("License"), p).exec())
            btn_row.addWidget(b_lic)

        _third = find_license_file("license.txt")
        if _third:
            b_third = QPushButton(self.tr("Third-Party Licenses"))
            b_third.setToolTip(self.tr("Licences for the bundled libraries"))
            b_third.clicked.connect(
                lambda _checked=False, p=_third: LicenseViewer(
                    self, self.tr("Third-Party Licenses"), p).exec())
            btn_row.addWidget(b_third)

        btn_row.addStretch(1)
        b_close = QPushButton(self.tr("Close"))
        b_close.clicked.connect(self.accept)
        btn_row.addWidget(b_close)

        layout.addLayout(btn_row)
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
    import faulthandler
    import tempfile
    from pathlib import Path

    def _get_crash_log_path() -> str:
        try:
            if hasattr(sys, "_MEIPASS"):
                if sys.platform.startswith("win"):
                    log_dir = Path(os.path.expandvars("%APPDATA%")) / "SetiAstroSuitePro" / "logs"
                elif sys.platform.startswith("darwin"):
                    log_dir = Path.home() / "Library" / "Logs" / "SetiAstroSuitePro"
                else:
                    log_dir = Path.home() / ".local" / "share" / "SetiAstroSuitePro" / "logs"
            else:
                # dev fallback
                log_dir = Path("logs")

            log_dir.mkdir(parents=True, exist_ok=True)
            return str(log_dir / "saspro_crash.log")
        except Exception:
            return str(Path(tempfile.gettempdir()) / "saspro_crash.log")

    # 1) Hard crashes → saspro_crash.log
    # faulthandler writes to a raw file fd, so RotatingFileHandler can't manage it.
    # Instead: if the existing crash log has grown past a cap, roll it to .1 and
    # start clean; then append within this session. Bounds the file without
    # losing the most recent prior crash.
    try:
        crash_path = _get_crash_log_path()

        _CRASH_MAX_BYTES = 512_000  # ~0.5 MB cap
        try:
            if os.path.exists(crash_path) and os.path.getsize(crash_path) > _CRASH_MAX_BYTES:
                _prev = crash_path + ".1"
                try:
                    if os.path.exists(_prev):
                        os.remove(_prev)
                    os.replace(crash_path, _prev)  # keep one previous generation
                except Exception:
                    # if the roll fails (locked, perms), fall back to truncation
                    open(crash_path, "w", encoding="utf-8", errors="replace").close()
        except Exception:
            pass

        _crash_log = open(crash_path, "a", encoding="utf-8", errors="replace")
        # Session banner so multiple runs in one file are distinguishable
        try:
            from datetime import datetime as _dt
            _crash_log.write(f"\n===== faulthandler session {_dt.now().isoformat(timespec='seconds')} =====\n")
            _crash_log.flush()
        except Exception:
            pass
        faulthandler.enable(file=_crash_log, all_threads=True)
        atexit.register(_crash_log.close)
        logging.info("Faulthandler crash log: %s", crash_path)
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
    'LicenseViewer',
    'find_license_file',
    'SOURCE_URL',
    'GPL_URL',
    'ProjectSaveWorker',
    'DECOR_GLYPHS',
    'strip_ui_decorations',
    '_strip_ui_decorations',  # Backward compatibility
    'install_crash_handlers',
    'get_version',
    'get_build_timestamp',
]
