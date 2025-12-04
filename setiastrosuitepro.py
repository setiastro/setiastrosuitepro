#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Seti Astro Suite Pro - Main Entry Point
"""

# Show splash screen IMMEDIATELY before any heavy imports

import sys
import os

# Only run splash logic when executed as main script
if __name__ == "__main__":
    # Minimal imports for splash screen
    from PyQt6.QtWidgets import QApplication, QWidget
    from PyQt6.QtCore import Qt, QCoreApplication, QRect
    from PyQt6.QtGui import QGuiApplication, QIcon, QPixmap, QColor, QPainter, QFont, QLinearGradient
    
    # Set application attributes before creating QApplication
    try:
        QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )
    except Exception:
        pass
    
    QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_DontShowIconsInMenus, False)
    QCoreApplication.setOrganizationName("SetiAstro")
    QCoreApplication.setOrganizationDomain("setiastrosuite.pro")
    QCoreApplication.setApplicationName("Seti Astro Suite Pro")
    
    # Create QApplication FIRST
    _app = QApplication(sys.argv)
    
    # Determine icon paths early
    def _find_icon_path():
        """Find the best available icon path."""
        if hasattr(sys, '_MEIPASS'):
            base = sys._MEIPASS
        else:
            base = os.path.dirname(os.path.abspath(__file__))
        
        candidates = [
            os.path.join(base, "images", "astrosuitepro.png"),
            os.path.join(base, "images", "astrosuitepro.ico"),
            os.path.join(base, "images", "astrosuite.png"),
            os.path.join(base, "images", "astrosuite.ico"),
        ]
        for p in candidates:
            if os.path.exists(p):
                return p
        return candidates[0]  # fallback
    
    _early_icon_path = _find_icon_path()
    
    # =========================================================================
    # PhotoshopStyleSplash - Custom splash screen widget
    # =========================================================================
    class _EarlySplash(QWidget):
        """
        A modern, Photoshop-style splash screen shown immediately on startup.
        """
        def __init__(self, logo_path: str):
            super().__init__()
            self._version = "1.5.6"  # Hardcoded for early display
            self._build = ""
            self.current_message = "Starting..."
            self.progress_value = 0
            
            # Window setup
            self.setWindowFlags(
                Qt.WindowType.SplashScreen |
                Qt.WindowType.FramelessWindowHint |
                Qt.WindowType.WindowStaysOnTopHint
            )
            self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
            self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
            
            # Splash dimensions
            self.splash_width = 600
            self.splash_height = 400
            self.setFixedSize(self.splash_width, self.splash_height)
            
            # Center on screen
            screen = QGuiApplication.primaryScreen()
            if screen:
                screen_geo = screen.availableGeometry()
                x = (screen_geo.width() - self.splash_width) // 2 + screen_geo.x()
                y = (screen_geo.height() - self.splash_height) // 2 + screen_geo.y()
                self.move(x, y)
            
            # Load and scale logo
            self.logo_pixmap = self._load_logo(logo_path)
            
            # Fonts
            self.title_font = QFont("Segoe UI", 28, QFont.Weight.Bold)
            self.subtitle_font = QFont("Segoe UI", 11)
            self.message_font = QFont("Segoe UI", 9)
            self.copyright_font = QFont("Segoe UI", 8)
        
        def _load_logo(self, path: str) -> QPixmap:
            """Load the logo and scale appropriately."""
            if not path or not os.path.exists(path):
                return QPixmap()
            
            ext = os.path.splitext(path)[1].lower()
            if ext == ".ico":
                ic = QIcon(path)
                pm = ic.pixmap(256, 256)
                if pm.isNull():
                    pm = QPixmap(path)
            else:
                pm = QPixmap(path)
                if pm.isNull():
                    pm = QIcon(path).pixmap(256, 256)
            
            if not pm.isNull():
                pm = pm.scaled(
                    180, 180,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
            return pm
        
        def setMessage(self, message: str):
            """Update the loading message."""
            self.current_message = message
            self.repaint()
            _app.processEvents()
        
        def setProgress(self, value: int):
            """Update progress (0-100)."""
            self.progress_value = max(0, min(100, value))
            self.repaint()
            _app.processEvents()
        
        def setBuildInfo(self, version: str, build: str):
            """Update version and build info once available."""
            self._version = version
            self._build = build
            self.repaint()
        
        def paintEvent(self, event):
            """Custom paint for the splash screen."""
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
            
            w, h = self.splash_width, self.splash_height
            
            # --- Background gradient (deep space theme) ---
            gradient = QLinearGradient(0, 0, 0, h)
            gradient.setColorAt(0.0, QColor(15, 15, 25))
            gradient.setColorAt(0.5, QColor(25, 25, 45))
            gradient.setColorAt(1.0, QColor(10, 10, 20))
            painter.fillRect(0, 0, w, h, gradient)
            
            # --- Subtle border ---
            painter.setPen(QColor(60, 60, 80))
            painter.drawRect(0, 0, w - 1, h - 1)
            
            # --- Logo (centered upper area) ---
            if not self.logo_pixmap.isNull():
                logo_x = (w - self.logo_pixmap.width()) // 2
                logo_y = 40
                painter.drawPixmap(logo_x, logo_y, self.logo_pixmap)
            
            # --- Title ---
            painter.setFont(self.title_font)
            painter.setPen(QColor(255, 255, 255))
            title_rect = QRect(0, 230, w, 40)
            painter.drawText(title_rect, Qt.AlignmentFlag.AlignCenter, "Seti Astro Suite Pro")
            
            # --- Subtitle with version ---
            painter.setFont(self.subtitle_font)
            painter.setPen(QColor(180, 180, 200))
            subtitle_text = f"Version {self._version}"
            if self._build and self._build != "dev":
                subtitle_text += f"  •  Build {self._build}"
            subtitle_rect = QRect(0, 270, w, 25)
            painter.drawText(subtitle_rect, Qt.AlignmentFlag.AlignCenter, subtitle_text)
            
            # --- Progress bar ---
            bar_margin = 50
            bar_height = 4
            bar_y = h - 70
            bar_width = w - (bar_margin * 2)
            
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(40, 40, 60))
            painter.drawRoundedRect(bar_margin, bar_y, bar_width, bar_height, 2, 2)
            
            if self.progress_value > 0:
                fill_width = int(bar_width * self.progress_value / 100)
                bar_gradient = QLinearGradient(bar_margin, 0, bar_margin + bar_width, 0)
                bar_gradient.setColorAt(0.0, QColor(80, 140, 220))
                bar_gradient.setColorAt(1.0, QColor(140, 180, 255))
                painter.setBrush(bar_gradient)
                painter.drawRoundedRect(bar_margin, bar_y, fill_width, bar_height, 2, 2)
            
            # --- Loading message ---
            painter.setFont(self.message_font)
            painter.setPen(QColor(150, 150, 180))
            msg_rect = QRect(bar_margin, bar_y + 10, bar_width, 20)
            painter.drawText(msg_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, 
                           self.current_message)
            
            # --- Copyright ---
            painter.setFont(self.copyright_font)
            painter.setPen(QColor(100, 100, 130))
            copyright_text = "© 2024-2025 Franklin Marek (Seti Astro)  •  All Rights Reserved"
            copyright_rect = QRect(0, h - 30, w, 20)
            painter.drawText(copyright_rect, Qt.AlignmentFlag.AlignCenter, copyright_text)
            
            painter.end()
        
        def finish(self):
            """Hide and cleanup the splash."""
            self.hide()
            self.close()
            self.deleteLater()
    
    # --- Show splash IMMEDIATELY ---
    _splash = _EarlySplash(_early_icon_path)
    _splash.show()
    _splash.setMessage("Initializing Python runtime...")
    _splash.setProgress(2)
    _app.processEvents()


# =============================================================================
# Now proceed with all the heavy imports (splash is visible)
# =============================================================================

# Helper to update splash during imports (only when running as main)
def _update_splash(msg: str, progress: int):
    if __name__ == "__main__":
        _splash.setMessage(msg)
        _splash.setProgress(progress)

_update_splash("Loading PyTorch runtime...", 5)

from pro.runtime_torch import (
    add_runtime_to_sys_path,
    _ban_shadow_torch_paths,
    _purge_bad_torch_from_sysmodules,
)

add_runtime_to_sys_path(status_cb=lambda *_: None)
_ban_shadow_torch_paths(status_cb=lambda *_: None)
_purge_bad_torch_from_sysmodules(status_cb=lambda *_: None)

_update_splash("Loading standard libraries...", 10)

# ----------------------------------------
# Standard library imports (consolidated)
# ----------------------------------------
import importlib
import json
import logging
import math
import multiprocessing
import os
import re
import sys
import threading
import time
import traceback
import warnings
import webbrowser

from collections import defaultdict
from datetime import datetime
from decimal import getcontext
from io import BytesIO
from itertools import combinations
from math import isnan
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import quote, quote_plus

_update_splash("Loading NumPy...", 15)

# ----------------------------------------
# Third-party imports
# ----------------------------------------
import numpy as np

_update_splash("Loading image libraries...", 20)
from tifffile import imwrite
from xisf import XISF

_update_splash("Configuring matplotlib...", 25)
from pro.config_bootstrap import ensure_mpl_config_dir
_MPL_CFG_DIR = ensure_mpl_config_dir()

# Apply metadata patches for frozen builds
from pro.metadata_patcher import apply_metadata_patches
apply_metadata_patches()
# ----------------------------------------

warnings.filterwarnings(
    "ignore",
    message=r"Call to deprecated function \(or staticmethod\) _destroy\.",
    category=DeprecationWarning,
)

os.environ['LIGHTKURVE_STYLE'] = 'default'

# ----------------------------------------
# Matplotlib configuration
# ----------------------------------------
import matplotlib
matplotlib.use("QtAgg") 

# Configure stdout encoding
if (sys.stdout is not None) and (hasattr(sys.stdout, "reconfigure")):
    sys.stdout.reconfigure(encoding='utf-8')

# --- Lazy imports for heavy dependencies (performance optimization) ---
# photutils: loaded on first use
_photutils_isophote = None
def _get_photutils_isophote():
    """Lazy loader for photutils.isophote module."""
    global _photutils_isophote
    if _photutils_isophote is None:
        try:
            from photutils import isophote as _isophote_module
            _photutils_isophote = _isophote_module
        except Exception:
            _photutils_isophote = False  # Mark as failed
    return _photutils_isophote if _photutils_isophote else None

def get_Ellipse():
    """Get photutils.isophote.Ellipse, loading lazily."""
    mod = _get_photutils_isophote()
    return mod.Ellipse if mod else None

def get_EllipseGeometry():
    """Get photutils.isophote.EllipseGeometry, loading lazily."""
    mod = _get_photutils_isophote()
    return mod.EllipseGeometry if mod else None

def get_build_ellipse_model():
    """Get photutils.isophote.build_ellipse_model, loading lazily."""
    mod = _get_photutils_isophote()
    return mod.build_ellipse_model if mod else None

# lightkurve: loaded on first use
_lightkurve_module = None
def get_lightkurve():
    """Lazy loader for lightkurve module."""
    global _lightkurve_module
    if _lightkurve_module is None:
        try:
            import lightkurve as _lk
            _lk.MPLSTYLE = None
            _lightkurve_module = _lk
        except Exception:
            _lightkurve_module = False  # Mark as failed
    return _lightkurve_module if _lightkurve_module else None
# --- End lazy imports ---

_update_splash("Loading UI utilities...", 30)

# Shared UI utilities (avoiding code duplication)
from pro.widgets.common_utilities import (
    AboutDialog,
    ProjectSaveWorker as _ProjectSaveWorker,
    DECOR_GLYPHS,
    _strip_ui_decorations,
    install_crash_handlers,
)

_update_splash("Loading reproject library...", 35)

# Reproject for WCS-based alignment
try:
    from reproject import reproject_interp
except ImportError:
    reproject_interp = None  # fallback if not installed

_update_splash("Loading OpenCV...", 40)

# OpenCV for transform estimation & warping
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False


_update_splash("Loading PyQt6 components...", 45)

#################################
# PyQt6 Imports
#################################
from PyQt6 import sip

# ----- QtWidgets -----
from PyQt6.QtWidgets import (QDialog, QApplication, QMainWindow, QWidget, QHBoxLayout, QFileDialog, QMessageBox, QSizePolicy, QToolBar, QPushButton, QAbstractItemDelegate,
    QLineEdit, QMenu, QListWidget, QListWidgetItem, QSplashScreen, QDockWidget, QListView, QCompleter, QMdiArea, QMdiSubWindow, QWidgetAction, QAbstractItemView,
    QInputDialog, QVBoxLayout, QLabel, QCheckBox, QProgressBar, QProgressDialog, QGraphicsItem, QTabWidget, QTableWidget, QHeaderView, QTableWidgetItem, QToolButton, QPlainTextEdit
)

# ----- QtGui -----
from PyQt6.QtGui import (QPixmap, QColor, QIcon, QKeySequence, QShortcut, QGuiApplication, QStandardItemModel, QStandardItem, QAction, QPalette, QBrush, QActionGroup, QDesktopServices, QFont, QTextCursor
)

# ----- QtCore -----
from PyQt6.QtCore import (Qt, pyqtSignal, QCoreApplication, QTimer, QSize, QSignalBlocker, QModelIndex, QThread, QUrl, QSettings, QEvent, QByteArray, QObject
)

from PyQt6.QtNetwork import QNetworkAccessManager, QNetworkRequest, QNetworkReply


try:
    from pro._generated.build_info import BUILD_TIMESTAMP
except Exception:
    BUILD_TIMESTAMP = "dev"


VERSION = "1.5.7"

_update_splash("Loading resources...", 50)

# Icon paths are now centralized in pro.resources module
from pro.resources import (
    icon_path, windowslogo_path, green_path, neutral_path, whitebalance_path,
    morpho_path, clahe_path, starnet_path, staradd_path, LExtract_path,
    LInsert_path, slot0_path, slot1_path, slot2_path, slot3_path, slot4_path,
    rgbcombo_path, rgbextract_path, copyslot_path, graxperticon_path,
    cropicon_path, openfile_path, abeicon_path, undoicon_path, redoicon_path,
    blastericon_path, hdr_path, invert_path, fliphorizontal_path,
    flipvertical_path, rotateclockwise_path, rotatecounterclockwise_path,
    rotate180_path, maskcreate_path, maskapply_path, maskremove_path,
    slot5_path, slot6_path, slot7_path, slot8_path, slot9_path, pixelmath_path,
    histogram_path, mosaic_path, rescale_path, staralign_path, mask_path,
    platesolve_path, psf_path, supernova_path, starregistration_path,
    stacking_path, pedestal_icon_path, starspike_path, aperture_path,
    jwstpupil_path, signature_icon_path, livestacking_path, hrdiagram_path,
    convoicon_path, spcc_icon_path, sasp_data_path, exoicon_path, peeker_icon,
    dse_icon_path, astrobin_filters_csv_path, isophote_path, statstretch_path,
    starstretch_path, curves_path, disk_path, uhs_path, blink_path, ppp_path,
    nbtorgb_path, freqsep_path, contsub_path, halo_path, cosmic_path,
    satellite_path, imagecombine_path, wrench_path, eye_icon_path,
    disk_icon_path, nuke_path, hubble_path, collage_path, annotated_path,
    colorwheel_path, font_path, csv_icon_path, spinner_path, wims_path,
    wimi_path, linearfit_path, debayer_path, aberration_path,
    functionbundles_path, viewbundles_path, selectivecolor_path, rgbalign_path,
)


_update_splash("Configuring Qt message handler...", 55)

from PyQt6.QtCore import qInstallMessageHandler, QtMsgType

def _qt_msg_handler(mode, ctx, msg):
    lvl = {
        QtMsgType.QtDebugMsg:    logging.DEBUG,
        QtMsgType.QtInfoMsg:     logging.INFO,
        QtMsgType.QtWarningMsg:  logging.WARNING,
        QtMsgType.QtCriticalMsg: logging.ERROR,
        QtMsgType.QtFatalMsg:    logging.CRITICAL,
    }.get(mode, logging.ERROR)
    logging.log(lvl, "Qt: %s (%s:%s)", msg, getattr(ctx, "file", "?"), getattr(ctx, "line", -1))

qInstallMessageHandler(_qt_msg_handler)

_update_splash("Loading MDI widgets...", 60)

# MDI widgets imported from pro.mdi_widgets
from pro.mdi_widgets import (
    MdiArea, ViewLinkController, ConsoleListWidget, QtLogStream, _DocProxy,
    ROLE_ACTION as _ROLE_ACTION,
)

# Helper functions imported from pro.main_helpers
from pro.main_helpers import (
    safe_join_dir_and_name as _safe_join_dir_and_name,
    normalize_save_path_chosen_filter as _normalize_save_path_chosen_filter,
    display_name as _display_name,
    best_doc_name as _best_doc_name,
    doc_looks_like_table as _doc_looks_like_table,
    is_alive as _is_alive,
    safe_widget as _safe_widget,
)

# AboutDialog, DECOR_GLYPHS, _strip_ui_decorations imported from pro.widgets.common_utilities

# File utilities imported from pro.file_utils
from pro.file_utils import (
    _normalize_ext,
    _sanitize_filename,
    _exts_from_filter,
    REPLACE_SPACES_WITH_UNDERSCORES as _REPLACE_SPACES_WITH_UNDERSCORES,
    WIN_RESERVED_NAMES as _WIN_RESERVED,
)

_update_splash("Loading main window module...", 65)

from pro.gui.main_window import AstroSuiteProMainWindow

_update_splash("Modules loaded, finalizing...", 70)

if __name__ == "__main__":
    # Update splash with build info now that we have VERSION and BUILD_TIMESTAMP
    _splash.setBuildInfo(VERSION, BUILD_TIMESTAMP)
    _splash.setMessage("Setting up logging...")
    _splash.setProgress(72)
    
    # --- Logging (catch unhandled exceptions to a file) ---
    import tempfile
    from pathlib import Path
 
    # Cross-platform log file location
    def get_log_file_path():
        """Get appropriate log file path for the current platform."""
        
        if hasattr(sys, '_MEIPASS'):
            # Running in PyInstaller bundle - use platform-appropriate user directory
            if sys.platform.startswith('win'):
                # Windows: %APPDATA%\SetiAstroSuitePro\logs\
                log_dir = Path(os.path.expandvars('%APPDATA%')) / 'SetiAstroSuitePro' / 'logs'
            elif sys.platform.startswith('darwin'):
                # macOS: ~/Library/Logs/SetiAstroSuitePro/
                log_dir = Path.home() / 'Library' / 'Logs' / 'SetiAstroSuitePro'
            else:
                # Linux: ~/.local/share/SetiAstroSuitePro/logs/
                log_dir = Path.home() / '.local' / 'share' / 'SetiAstroSuitePro' / 'logs'
            
            # Create directory if it doesn't exist
            try:
                log_dir.mkdir(parents=True, exist_ok=True)
                log_file = log_dir / 'saspro.log'
            except (OSError, PermissionError):
                # Fallback to temp directory if user directory fails
                log_file = Path(tempfile.gettempdir()) / 'saspro.log'
        else:
            # Development mode - use logs folder in project
            log_dir = Path('logs')
            try:
                log_dir.mkdir(parents=True, exist_ok=True)
                log_file = log_dir / 'saspro.log'
            except (OSError, PermissionError):
                log_file = Path('saspro.log')
        
        return str(log_file)
    
    # Configure logging with cross-platform path
    log_file_path = get_log_file_path()

    try:
        logging.basicConfig(
            filename=log_file_path,
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            filemode='a'  # Append mode
        )
        logging.info(f"Logging to: {log_file_path}")
        logging.info(f"Platform: {sys.platform}")
        logging.info(f"PyInstaller bundle: {hasattr(sys, '_MEIPASS')}")
    except Exception as e:
        # Ultimate fallback - console only logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler(sys.stdout)]
        )
        print(f"Warning: Could not write to log file {log_file_path}: {e}")
        print("Using console-only logging")
        

    # Setup crash handlers and app icon
    _splash.setMessage("Installing crash handlers...")
    _splash.setProgress(75)
    install_crash_handlers(_app) 
    _app.setWindowIcon(QIcon(windowslogo_path if os.path.exists(windowslogo_path) else icon_path))

    # --- Windows exe / multiprocessing friendly ---
    _splash.setMessage("Configuring multiprocessing...")
    _splash.setProgress(78)
    try:
        multiprocessing.freeze_support()
        try:
            multiprocessing.set_start_method("spawn", force=True)
        except RuntimeError:
            # Already set in this interpreter
            pass
    except Exception:
        logging.exception("Multiprocessing init failed (continuing).")

    try:
        _splash.setMessage("Loading image manager...")
        _splash.setProgress(80)
        from legacy.image_manager import ImageManager
        
        _splash.setMessage("Suppressing warnings...")
        _splash.setProgress(82)
        from matplotlib import MatplotlibDeprecationWarning
        warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)

        _splash.setMessage("Creating image manager...")
        _splash.setProgress(85)
        imgr = ImageManager(max_slots=100)
        
        _splash.setMessage("Building main window...")
        _splash.setProgress(90)
        win = AstroSuiteProMainWindow(
            image_manager=imgr,
            version=VERSION,
            build_timestamp=BUILD_TIMESTAMP,
        )
        
        _splash.setMessage("Showing main window...")
        _splash.setProgress(95)
        win.show()

        # Start background Numba warmup after UI is visible
        try:
            from pro.numba_warmup import start_background_warmup
            start_background_warmup()
        except Exception:
            pass  # Non-critical if warmup fails

        _splash.setMessage("Ready!")
        _splash.setProgress(100)
        _app.processEvents()
        
        # Small delay to show "Ready!" before closing
        import time
        time.sleep(0.3)
        _app.processEvents()

        # Ensure the splash cannot resurrect later:
        try:
            _splash.finish()
        finally:
            _splash.hide()
            _splash.close()
            _splash.deleteLater()
        
        print(f"Seti Astro Suite Pro v{VERSION} (build {BUILD_TIMESTAMP}) up and running!")
        sys.exit(_app.exec())

    except Exception:
        import traceback
        try:
            _splash.hide()
            _splash.close()
            _splash.deleteLater()
        except Exception:
            pass
        tb = traceback.format_exc()
        logging.error("Unhandled exception occurred\n%s", tb)
        msg = QMessageBox(None)
        msg.setIcon(QMessageBox.Icon.Critical)
        msg.setWindowTitle("Application Error")
        msg.setText("An unexpected error occurred.")
        msg.setInformativeText(tb.splitlines()[-1] if tb else "See details.")
        msg.setDetailedText(tb)
        msg.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg.exec()
        sys.exit(1)
