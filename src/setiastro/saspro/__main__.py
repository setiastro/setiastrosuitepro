# -*- coding: utf-8 -*-
"""
Seti Astro Suite Pro - Main Entry Point Module

This module contains the main application entry point logic.
It can be executed directly via `python -m setiastro.saspro` or
called via the `main()` function when invoked as an entry point.
"""

# Show splash screen IMMEDIATELY before any heavy imports

import sys
import os
from PyQt6.QtCore import QCoreApplication

# ---- Linux Qt stability guard (must run BEFORE any PyQt6 import) ----
# Default behavior: DO NOT override Wayland.
# If a user needs the "safe" path, they can opt-in by setting:
#   SASPRO_QT_SAFE=1
#
# This avoids punishing all Wayland users for one bad driver/Qt stack.
if sys.platform.startswith("linux"):
    if os.environ.get("SASPRO_QT_SAFE", "").strip() in ("1", "true", "yes", "on"):
        # Prefer X11/xcb unless user explicitly set a platform plugin
        os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

        # Prefer software GL unless user explicitly set something else
        os.environ.setdefault("QT_OPENGL", "software")
        os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")

# Global variables for splash screen and app
_splash = None
_app = None

# Flag to track if splash was initialized
_splash_initialized = False

from setiastro.saspro.versioning import get_app_version
_EARLY_VERSION = get_app_version("setiastrosuitepro")

VERSION = _EARLY_VERSION

def _init_splash():
    """Initialize the splash screen. Safe to call multiple times."""
    global _splash, _app, _splash_initialized
    
    if _splash_initialized:
        return
    
    # Minimal imports for splash screen
    from PyQt6.QtWidgets import QApplication, QWidget
    from PyQt6.QtCore import Qt, QCoreApplication, QRect, QPropertyAnimation, QEasingCurve
    import time
    from PyQt6.QtGui import QGuiApplication, QIcon, QPixmap, QColor, QPainter, QFont, QLinearGradient
    

    # If we're forcing software OpenGL, do it *before* QApplication is created.
    if sys.platform.startswith("linux"):
        if os.environ.get("SASPRO_QT_SAFE", "").strip() in ("1", "true", "yes", "on"):
            if os.environ.get("QT_OPENGL", "").lower() == "software":
                try:
                    QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_UseSoftwareOpenGL, True)
                except Exception:
                    pass

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

    # Create QApplication
    _app = QApplication(sys.argv)
    
    if sys.platform.startswith("linux"):
        try:
            print("Qt platform:", _app.platformName())
            print("XDG_SESSION_TYPE:", os.environ.get("XDG_SESSION_TYPE"))
            print("QT_QPA_PLATFORM:", os.environ.get("QT_QPA_PLATFORM"))
            print("QT_OPENGL:", os.environ.get("QT_OPENGL"))
        except Exception:
            pass

    # Determine icon paths early
    # Determine icon paths early
    def _find_icon_path():
        """Legacy fallback if resources import fails."""
        if hasattr(sys, '_MEIPASS'):
            base = sys._MEIPASS
        else:
            try:
                import setiastro
                package_dir = os.path.dirname(os.path.abspath(setiastro.__file__))
                package_parent = os.path.dirname(package_dir)
                images_dir_installed = os.path.join(package_parent, 'images')
                if os.path.exists(images_dir_installed):
                    base = package_parent
                else:
                    base = os.path.dirname(
                        os.path.dirname(
                            os.path.dirname(
                                os.path.dirname(os.path.abspath(__file__))
                            )
                        )
                    )
            except (ImportError, AttributeError):
                base = os.path.dirname(
                    os.path.dirname(
                        os.path.dirname(
                            os.path.dirname(os.path.abspath(__file__))
                        )
                    )
                )

        candidates = [
            os.path.join(base, "images", "astrosuitepro.png"),
            os.path.join(base, "images", "astrosuitepro.ico"),
            os.path.join(base, "images", "astrosuite.png"),
            os.path.join(base, "images", "astrosuite.ico"),
        ]
        for p in candidates:
            if os.path.exists(p):
                return p
        return ""  # nothing found

    # NEW: Prefer centralized resources resolver
    try:
        from setiastro.saspro.resources import icon_path, background_startup_path
        _early_icon_path = icon_path
        if not os.path.exists(_early_icon_path):
            # fall back to legacy search if for some reason this is missing
            _early_icon_path = _find_icon_path()
        
        # Load startup background path
        _startup_bg_path = background_startup_path
        if not os.path.exists(_startup_bg_path):
             _startup_bg_path = None
             
    except Exception:
        _early_icon_path = _find_icon_path()
        _startup_bg_path = None

    
    # =========================================================================
    # PhotoshopStyleSplash - Custom splash screen widget
    # =========================================================================
    class _EarlySplash(QWidget):
        """
        A modern, Photoshop-style splash screen shown immediately on startup.
        """
        def __init__(self, logo_path: str):
            super().__init__()
            self._version = _EARLY_VERSION
            self._build = ""
            self.current_message = QCoreApplication.translate("Splash", "Starting...")
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
            
            # Load background image
            self.bg_image_pixmap = QPixmap()
            if _startup_bg_path:
                self.bg_image_pixmap = QPixmap(_startup_bg_path)

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
            if _app:
                _app.processEvents()
        
        def setProgress(self, value: int):
            """Update progress (0-100) with smooth animation."""
            target = max(0, min(100, value))
            start = self.progress_value
            
            # If jumping backwards or small change, just set it
            if target <= start or (target - start) < 1:
                self.progress_value = target
                self.repaint()
                if _app: _app.processEvents()
                return

            # Animate forward
            steps = 15  # number of frames for the slide
            # We want the total slide to take ~100-150ms max to feel responsive but smooth
            dt = 0.005  # 5ms per frame
            
            for i in range(1, steps + 1):
                # Ease out interpolator
                t = i / steps
                # Quadratic ease out: f(t) = -t*(t-2)
                factor = -t * (t - 2)
                
                cur = start + (target - start) * factor
                self.progress_value = cur
                self.repaint()
                if _app: _app.processEvents()
                time.sleep(dt)

            self.progress_value = target
            self.repaint()
            if _app:
                _app.processEvents()
        
        def setBuildInfo(self, version: str, build: str):
            """Update version and build info once available."""
            self._version = _EARLY_VERSION
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
            
            # --- Background Image (Centered with Fade Out) ---
            if not self.bg_image_pixmap.isNull():
                # Create a temporary pixmap to handle the masking
                temp = QPixmap(w, h)
                temp.fill(Qt.GlobalColor.transparent)
                
                ptmp = QPainter(temp)
                ptmp.setRenderHint(QPainter.RenderHint.Antialiasing)
                ptmp.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
                
                # Scale image to cover the entire splash screen
                scaled = self.bg_image_pixmap.scaled(
                    w, h, 
                    Qt.AspectRatioMode.KeepAspectRatioByExpanding, 
                    Qt.TransformationMode.SmoothTransformation
                )
                
                # Center the image
                sx = (w - scaled.width()) // 2
                sy = (h - scaled.height()) // 2
                ptmp.drawPixmap(sx, sy, scaled)
                
                # Apply Fade Out Mask (Gradient Alpha)
                ptmp.setCompositionMode(QPainter.CompositionMode.CompositionMode_DestinationIn)
                fade_gradient = QLinearGradient(0, 0, 0, h)
                # Keep top half fully visible (subject to global opacity)
                fade_gradient.setColorAt(0.0, QColor(0, 0, 0, 255)) 
                fade_gradient.setColorAt(0.5, QColor(0, 0, 0, 255)) 
                # Fade out completely at the bottom
                fade_gradient.setColorAt(1.0, QColor(0, 0, 0, 0))   
                ptmp.fillRect(0, 0, w, h, fade_gradient)
                ptmp.end()
                
                # Draw combined result with 50% opacity
                painter.save()
                painter.setOpacity(0.25)
                painter.drawPixmap(0, 0, temp)
                painter.restore()

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
            subtitle_text = QCoreApplication.translate("Splash", "Version {0}").format(self._version)

            if self._build:
                if self._build == "dev":
                    # No build_info → running from source checkout
                    subtitle_text += QCoreApplication.translate("Splash", "  •  Running locally from source code")
                else:
                    subtitle_text += QCoreApplication.translate("Splash", "  •  Build {0}").format(self._build)

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

        def start_fade_out(self):
            """Smoothly fade out the splash screen."""
            self._anim = QPropertyAnimation(self, b"windowOpacity")
            self._anim.setDuration(1000)
            self._anim.setStartValue(1.0)
            self._anim.setEndValue(0.0)
            self._anim.setEasingCurve(QEasingCurve.Type.OutQuad)
            self._anim.finished.connect(self.finish)
            self._anim.start()
    
        def start_fade_in(self):
            """Smoothly fade in the splash screen."""
            self.setWindowOpacity(0.0)
            self._anim = QPropertyAnimation(self, b"windowOpacity")
            self._anim.setDuration(800)
            self._anim.setStartValue(0.0)
            self._anim.setEndValue(1.0)
            self._anim.setEasingCurve(QEasingCurve.Type.InQuad)
            self._anim.start()

    # --- Show splash IMMEDIATELY ---
    _splash = _EarlySplash(_early_icon_path)
    _splash.start_fade_in()
    _splash.show()
    
    # Block briefly to allow fade-in to progress smoothly before heavy imports start
    # We use a busy loop with processEvents to keep the UI responsive during fade
    t_start = time.time()
    while time.time() - t_start < 0.85:  # slightly longer than animation
        _app.processEvents()
        if _splash.windowOpacity() >= 0.99:
            break
        time.sleep(0.01)

    _splash.setMessage(QCoreApplication.translate("Splash", "Initializing Python runtime..."))
    _splash.setProgress(2)
    _app.processEvents()
    
    # Load translation BEFORE any other widgets are created
    try:
        from setiastro.saspro.i18n import load_language
        load_language(app=_app)
    except Exception:
        pass  # Translations not critical - continue without them
    
    _splash_initialized = True


# Initialize splash immediately before any heavy imports
# This ensures the splash is visible while PyTorch, NumPy, etc. are loading
_init_splash()


# =============================================================================
# Now proceed with all the heavy imports (splash is visible)
# =============================================================================

# Helper to update splash during imports
def _update_splash(msg: str, progress: int):
    global _splash
    if _splash is not None:
        _splash.setMessage(msg)
        _splash.setProgress(progress)

_update_splash(QCoreApplication.translate("Splash", "Loading PyTorch runtime..."), 5)

from setiastro.saspro.runtime_torch import (
    add_runtime_to_sys_path,
    _ban_shadow_torch_paths,
    _purge_bad_torch_from_sysmodules,
)

add_runtime_to_sys_path(status_cb=lambda *_: None)
_ban_shadow_torch_paths(status_cb=lambda *_: None)
_purge_bad_torch_from_sysmodules(status_cb=lambda *_: None)

_update_splash(QCoreApplication.translate("Splash", "Loading standard libraries..."), 10)

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

_update_splash(QCoreApplication.translate("Splash", "Loading NumPy..."), 15)

# ----------------------------------------
# Third-party imports
# ----------------------------------------
import numpy as np

_update_splash(QCoreApplication.translate("Splash", "Loading image libraries..."), 20)
from tifffile import imwrite
from setiastro.saspro.xisf import XISF

_update_splash(QCoreApplication.translate("Splash", "Configuring matplotlib..."), 25)
from setiastro.saspro.config_bootstrap import ensure_mpl_config_dir
_MPL_CFG_DIR = ensure_mpl_config_dir()

# Apply metadata patches for frozen builds
from setiastro.saspro.metadata_patcher import apply_metadata_patches
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

_update_splash(QCoreApplication.translate("Splash", "Loading UI utilities..."), 30)

# Shared UI utilities (avoiding code duplication)
from setiastro.saspro.widgets.common_utilities import (
    AboutDialog,
    ProjectSaveWorker as _ProjectSaveWorker,
    DECOR_GLYPHS,
    _strip_ui_decorations,
    install_crash_handlers,
)

_update_splash(QCoreApplication.translate("Splash", "Loading reproject library..."), 35)

# Reproject for WCS-based alignment
try:
    from reproject import reproject_interp
except ImportError:
    reproject_interp = None  # fallback if not installed

_update_splash(QCoreApplication.translate("Splash", "Loading OpenCV..."), 40)

# OpenCV for transform estimation & warping
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False


_update_splash(QCoreApplication.translate("Splash", "Loading PyQt6 components..."), 45)

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
from PyQt6.QtCore import (Qt, pyqtSignal, QCoreApplication, QTimer, QSize, QSignalBlocker, QModelIndex, QThread, QUrl, QSettings, QEvent, QByteArray, QObject,
    QPropertyAnimation, QEasingCurve
)

from PyQt6.QtNetwork import QNetworkAccessManager, QNetworkRequest, QNetworkReply


try:
    from setiastro.saspro._generated.build_info import BUILD_TIMESTAMP
except Exception:
    # No generated build info → running from local source checkout
    BUILD_TIMESTAMP = "dev"



_update_splash(QCoreApplication.translate("Splash", "Loading resources..."), 50)

# Icon paths are now centralized in setiastro.saspro.resources module
from setiastro.saspro.resources import (
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


_update_splash(QCoreApplication.translate("Splash", "Configuring Qt message handler..."), 55)

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

_update_splash(QCoreApplication.translate("Splash", "Loading MDI widgets..."), 60)

# MDI widgets imported from setiastro.saspro.mdi_widgets
from setiastro.saspro.mdi_widgets import (
    MdiArea, ViewLinkController, ConsoleListWidget, QtLogStream, _DocProxy,
    ROLE_ACTION as _ROLE_ACTION,
)

# Helper functions imported from setiastro.saspro.main_helpers
from setiastro.saspro.main_helpers import (
    safe_join_dir_and_name as _safe_join_dir_and_name,
    normalize_save_path_chosen_filter as _normalize_save_path_chosen_filter,
    display_name as _display_name,
    best_doc_name as _best_doc_name,
    doc_looks_like_table as _doc_looks_like_table,
    is_alive as _is_alive,
    safe_widget as _safe_widget,
)

# AboutDialog, DECOR_GLYPHS, _strip_ui_decorations imported from setiastro.saspro.widgets.common_utilities

# File utilities imported from setiastro.saspro.file_utils
from setiastro.saspro.file_utils import (
    _normalize_ext,
    _sanitize_filename,
    _exts_from_filter,
    REPLACE_SPACES_WITH_UNDERSCORES as _REPLACE_SPACES_WITH_UNDERSCORES,
    WIN_RESERVED_NAMES as _WIN_RESERVED,
)

_update_splash(QCoreApplication.translate("Splash", "Loading main window module..."), 65)

from setiastro.saspro.gui.main_window import AstroSuiteProMainWindow

_update_splash(QCoreApplication.translate("Splash", "Modules loaded, finalizing..."), 70)


def main():
    """
    Main entry point for Seti Astro Suite Pro.
    
    This function can be called from:
    - The package entry point (setiastrosuitepro command)
    - Direct import and call
    - When running as a module: python -m setiastro.saspro
    """
    global _splash, _app, _splash_initialized
    
    # Initialize splash if not already done
    if not _splash_initialized:
        _init_splash()
    
    # Update splash with build info now that we have VERSION and BUILD_TIMESTAMP
    if _splash:
        _splash.setBuildInfo(VERSION, BUILD_TIMESTAMP)
        _splash.setMessage(QCoreApplication.translate("Splash", "Setting up logging..."))
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
    if _splash:
        _splash.setMessage(QCoreApplication.translate("Splash", "Installing crash handlers..."))
        _splash.setProgress(75)
    install_crash_handlers(_app) 
    _app.setWindowIcon(QIcon(windowslogo_path if os.path.exists(windowslogo_path) else icon_path))

    # --- Windows exe / multiprocessing friendly ---
    if _splash:
        _splash.setMessage(QCoreApplication.translate("Splash", "Configuring multiprocessing..."))
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
        if _splash:
            _splash.setMessage(QCoreApplication.translate("Splash", "Loading image manager..."))
            _splash.setProgress(80)
        from setiastro.saspro.legacy.image_manager import ImageManager
        
        if _splash:
            _splash.setMessage(QCoreApplication.translate("Splash", "Suppressing warnings..."))
            _splash.setProgress(82)
        from matplotlib import MatplotlibDeprecationWarning
        warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)

        if _splash:
            _splash.setMessage(QCoreApplication.translate("Splash", "Creating image manager..."))
            _splash.setProgress(85)
        imgr = ImageManager(max_slots=100)
        
        if _splash:
            _splash.setMessage(QCoreApplication.translate("Splash", "Building main window..."))
            _splash.setProgress(90)
        win = AstroSuiteProMainWindow(
            image_manager=imgr,
            version=VERSION,
            build_timestamp=BUILD_TIMESTAMP,
        )
        
        if _splash:
            _splash.setMessage(QCoreApplication.translate("Splash", "Showing main window..."))
            _splash.setProgress(95)
        
        # --- Smooth Transition: App Fade In + Splash Fade Out ---
        # MITIGATION: Prevent "White Flash" on startup
        # 1. Force a dark background immediately so if opacity lags, it's dark not white
        win.setStyleSheet("QMainWindow { background-color: #0F0F19; }") 
        # 2. Ensure native window handle exists so setWindowOpacity works immediately
        win.winId()
        # 3. Set opacity to 0
        win.setWindowOpacity(0.0)
        
        win.show()
        
        # 1. Animate Main Window Fade In
        anim_app = QPropertyAnimation(win, b"windowOpacity")
        anim_app.setDuration(1200)
        anim_app.setStartValue(0.0)
        anim_app.setEndValue(1.0)
        anim_app.setEasingCurve(QEasingCurve.Type.OutQuad)
        
        # Cleanup temp stylesheet upon completion to avoid interfering with ThemeMixin
        def _on_fade_in_finished():
            win.setStyleSheet("")
            if hasattr(win, "on_fade_in_complete"):
                win.on_fade_in_complete()
            
        anim_app.finished.connect(_on_fade_in_finished)
        anim_app.start()

        # Start background Numba warmup after UI is visible
        try:
            from setiastro.saspro.numba_warmup import start_background_warmup
            start_background_warmup()
        except Exception:
            pass  # Non-critical if warmup fails

        if _splash:
            _splash.setMessage(QCoreApplication.translate("Splash", "Ready!"))
            _splash.setProgress(100)
            _app.processEvents()
            
            # Small delay to ensure "Ready!" is seen briefly before fade starts
            import time
            time.sleep(0.1)
            
            # 2. Animate Splash Fade Out
            # Note: We do NOT use finish() directly here. The animation calls it when done.
            _splash.start_fade_out()
            
            # NOTE: We keep a reference to _splash (global) so it doesn't get GC'd during animation.
            # It will deleteLater() itself.
        
        if BUILD_TIMESTAMP == "dev":
            build_label = "running from local source code"
        else:
            build_label = f"build {BUILD_TIMESTAMP}"

        print(f"Seti Astro Suite Pro v{VERSION} ({build_label}) up and running!")
        sys.exit(_app.exec())

    except Exception:
        import traceback
        if _splash:
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
        msg.setWindowTitle(QCoreApplication.translate("Main", "Application Error"))
        msg.setText(QCoreApplication.translate("Main", "An unexpected error occurred."))
        msg.setInformativeText(tb.splitlines()[-1] if tb else "See details.")
        msg.setDetailedText(tb)
        msg.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg.exec()
        sys.exit(1)


# When run as a module, execute main()
if __name__ == "__main__":
    main()
