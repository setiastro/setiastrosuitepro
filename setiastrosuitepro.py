#setiastrosuitepro.py
from pro.runtime_torch import add_runtime_to_sys_path, _ban_shadow_torch_paths, _purge_bad_torch_from_sysmodules
add_runtime_to_sys_path(status_cb=lambda *_: None)
_ban_shadow_torch_paths(status_cb=lambda *_: None)
_purge_bad_torch_from_sysmodules(status_cb=lambda *_: None)

import time, traceback
# ----------------------------------------
# Matplotlib bootstrap (for frozen + for prewarmed cache)
# ----------------------------------------
import os
import sys

def _ensure_mpl_config_dir() -> str:
    """
    Make matplotlib use a known, writable folder.

    Frozen (PyInstaller): <folder-with-exe>/mpl_config
    Dev / IDE:            <repo-folder>/mpl_config

    This matches the pre-warm script that will build the font cache there.
    """
    if getattr(sys, "frozen", False):
        base = os.path.dirname(sys.executable)
    else:
        base = os.path.dirname(os.path.abspath(__file__))

    mpl_cfg = os.path.join(base, "mpl_config")
    try:
        os.makedirs(mpl_cfg, exist_ok=True)
    except OSError:
        # worst case: let matplotlib pick its default
        return mpl_cfg

    # only set if user / env didn't force something else
    os.environ.setdefault("MPLCONFIGDIR", mpl_cfg)
    return mpl_cfg

_MPL_CFG_DIR = _ensure_mpl_config_dir()
# (optional) print once in dev:
# print("MPLCONFIGDIR ->", _MPL_CFG_DIR)
# ----------------------------------------



import importlib

if getattr(sys, 'frozen', False):
    # 1) Attempt to import both metadata modules
    try:
        std_md = importlib.import_module('importlib.metadata')
    except ImportError:
        std_md = None

    try:
        back_md = importlib.import_module('importlib_metadata')
    except ImportError:
        back_md = None

    # 2) Ensure that any "import importlib.metadata" or
    #    "import importlib_metadata" picks up our loaded module
    if std_md:
        sys.modules['importlib.metadata'] = std_md
        setattr(importlib, 'metadata', std_md)
    if back_md:
        sys.modules['importlib_metadata'] = back_md

    # 3) Pick whichever is available for defaults (prefer stdlib)
    meta = std_md or back_md
    if not meta:
        # nothing to patch
        sys.exit(0)

    # 4) Save originals
    orig_version      = getattr(meta, 'version', None)
    orig_distribution = getattr(meta, 'distribution', None)

    # 5) Define safe fallbacks
    def safe_version(pkg, *args, **kwargs):
        try:
            return orig_version(pkg, *args, **kwargs)
        except Exception:
            return "0.0.0"

    class DummyDist:
        version = "0.0.0"
        metadata = {}

    def safe_distribution(pkg, *args, **kwargs):
        try:
            return orig_distribution(pkg, *args, **kwargs)
        except Exception:
            return DummyDist()

    # 6) Patch both modules (stdlib and back-port) if they exist
    for m in (std_md, back_md):
        if not m:
            continue
        if orig_version:
            m.version = safe_version
        if orig_distribution:
            m.distribution = safe_distribution



import warnings
warnings.filterwarnings(
    "ignore",
    message=r"Call to deprecated function \(or staticmethod\) _destroy\.",
    category=DeprecationWarning
)

# Standard library imports
from itertools import combinations
from tifffile import imwrite

from math import isnan
import re, threading, webbrowser

os.environ['LIGHTKURVE_STYLE'] = 'default'

import json
import logging

from decimal import getcontext
from urllib.parse import quote
from urllib.parse import quote_plus

from xisf import XISF

from pathlib import Path

from typing import List, Tuple, Dict, Set, Optional
import time
from datetime import datetime

from io import BytesIO

# scipy.ndimage imports removed - not used in main module
# gaussian_filter, laplace, zoom loaded on demand in specific modules

import matplotlib
matplotlib.use("QtAgg") 

import numpy as np


#if running in IDE which runs ipython or jupiter in backend reconfigure may not be available
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


from numba_utils import *

# Shared UI utilities (avoiding code duplication)
from pro.widgets.common_utilities import (
    AboutDialog,
    ProjectSaveWorker as _ProjectSaveWorker,
    DECOR_GLYPHS,
    _strip_ui_decorations,
    install_crash_handlers,
)


# Reproject for WCS-based alignment
try:
    from reproject import reproject_interp
except ImportError:
    reproject_interp = None  # fallback if not installed

# OpenCV for transform estimation & warping
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False


# Third-party library imports

import numpy as np

import cv2




#################################
# PyQt6 Imports
#################################
from collections import defaultdict

from PyQt6 import sip

# ----- QtWidgets -----
from PyQt6.QtWidgets import (QDialog, QApplication, QMainWindow, QWidget, QHBoxLayout, QFileDialog, QMessageBox, QSizePolicy, QToolBar, QPushButton, QAbstractItemDelegate,
    QLineEdit, QMenu, QListWidget, QListWidgetItem, QSplashScreen, QDockWidget, QListView, QCompleter, QMdiArea, QMdiArea, QMdiSubWindow, QWidgetAction, QAbstractItemView,
    QInputDialog, QVBoxLayout, QLabel, QCheckBox, QProgressBar, QProgressDialog, QGraphicsItem, QTabWidget, QTableWidget, QHeaderView, QTableWidgetItem, QToolButton, QPlainTextEdit
)

# ----- QtGui -----
from PyQt6.QtGui import (QPixmap, QColor, QIcon, QKeySequence, QShortcut, QGuiApplication, QStandardItemModel, QStandardItem, QAction, QPalette, QBrush, QActionGroup, QDesktopServices, QFont, QTextCursor
)

# ----- QtCore -----
from PyQt6.QtCore import (Qt, pyqtSignal, QCoreApplication, QTimer, QSize, QSignalBlocker,  QModelIndex, QThread, QUrl, QSettings, QEvent, QByteArray, QObject
)

from PyQt6.QtNetwork import QNetworkAccessManager, QNetworkRequest, QNetworkReply


# Math functions

import math


from pro.doc_manager import DocManager
from pro.subwindow import ImageSubWindow, TableSubWindow
from legacy.image_manager import ImageManager
from pro.header_viewer import HeaderViewerDock
from pro.batch_convert import BatchConvertDialog
from pro.autostretch import autostretch
from pro.stat_stretch import StatisticalStretchDialog
from pro.save_options import SaveOptionsDialog
from pro.history_explorer import HistoryExplorerDialog
from pro.star_stretch import StarStretchDialog
from pro.histogram import HistogramDialog
from pro.curve_editor_pro import CurvesDialogPro
from pro.ghs_dialog_pro import GhsDialogPro
from pro.crop_dialog_pro import CropDialogPro
from pro.blink_comparator_pro import BlinkComparatorPro
from pro.perfect_palette_picker import PerfectPalettePicker
from pro.nbtorgb_stars import NBtoRGBStars
from pro.frequency_separation import FrequencySeperationTab
from pro.shortcuts import DraggableToolBar, ShortcutManager, _StatStretchPresetDialog
from pro.shortcuts import _unpack_cmd_payload
from pro.continuum_subtract import ContinuumSubtractTab
from pro.abe import ABEDialog
from ops.settings import SettingsDialog
from pro.mask_creation import create_mask_and_attach
from pro.dnd_mime import MIME_VIEWSTATE, MIME_CMD, MIME_MASK, MIME_ASTROMETRY, MIME_LINKVIEW
from pro.graxpert import remove_gradient_with_graxpert
from pro.remove_stars import remove_stars
from pro.add_stars import add_stars 
from pro.window_shelf import WindowShelf, MinimizeInterceptor
from pro.pedestal import remove_pedestal
from pro.remove_green import open_remove_green_dialog, apply_remove_green_preset_to_doc
from pro.backgroundneutral import BackgroundNeutralizationDialog, apply_background_neutral_to_doc
from pro.luminancerecombine import apply_recombine_to_doc, compute_luminance, _to_float01_strict, _estimate_noise_sigma_per_channel, _LUMA_REC709, _LUMA_REC601, _LUMA_REC2020
from pro.sfcc import SFCCDialog
from pro.rgb_extract import extract_rgb_channels 
from pro.rgb_combination import RGBCombinationDialogPro
from pro.blemish_blaster import BlemishBlasterDialogPro
from pro.wavescale_hdr import WaveScaleHDRDialogPro, compute_wavescale_hdr
from pro.wavescalede import install_wavescale_dark_enhancer
from pro.clahe import CLAHEDialogPro
from pro.morphology import MorphologyDialogPro
from pro.pixelmath import PixelMathDialogPro
from pro.signature_insert import SignatureInsertDialogPro
from pro.cosmicclarity import CosmicClarityDialogPro, CosmicClaritySatelliteDialogPro
from legacy.numba_utils import (
    rescale_image_numba,
    flip_horizontal_numba,
    flip_vertical_numba,
    rotate_90_clockwise_numba,
    rotate_90_counterclockwise_numba,
    invert_image_numba,
    rotate_180_numba,
)
from pro.wcs_update import update_wcs_after_crop
from pro.project_io import ProjectWriter, ProjectReader
from pro.psf_viewer import PSFViewer
from pro.plate_solver import plate_solve_doc_inplace, PlateSolverDialog
from pro.star_alignment import StellarAlignmentDialog, StarRegistrationWindow, MosaicMasterDialog
from pro.image_peeker_pro import ImagePeekerDialogPro
# Lazy imports for heavy dialogs (loaded on demand)
# from pro.live_stacking import LiveStackWindow  # loaded on demand
# from pro.stacking_suite import StackingSuiteDialog  # loaded on demand
# from pro.supernovaasteroidhunter import SupernovaAsteroidHunterDialog  # loaded on demand
from pro.star_spikes import StarSpikesDialogPro
# Lazy imports for modules that load lightkurve (~12s)
# from pro.exoplanet_detector import ExoPlanetWindow  # loaded on demand
# from wimi import WIMIDialog  # loaded on demand
from pro.isophote import IsophoteModelerDialog
from wims import WhatsInMySkyDialog
from pro.fitsmodifier import FITSModifier
from pro.batch_renamer import BatchRenamerDialog
from pro.astrobin_exporter import AstrobinExporterDialog
from pro.linear_fit import LinearFitDialog
from pro.debayer import DebayerDialog, apply_debayer_preset_to_doc
from pro.copyastro import CopyAstrometryDialog
from pro.layers_dock import LayersDock
try:
    from pro._generated.build_info import BUILD_TIMESTAMP
except Exception:
    BUILD_TIMESTAMP = "dev"
from pro.aberration_ai import AberrationAIDialog
from pro.view_bundle import show_view_bundles
from pro.function_bundle import show_function_bundles
from pro.ghs_preset import open_ghs_with_preset
from pro.curves_preset import open_curves_with_preset
from pro.save_options import _normalize_ext
from pro.status_log_dock import StatusLogDock
from pro.log_bus import LogBus
from imageops.mdi_snap import MdiSnapController
from pro.fitsmodifier import BatchFITSHeaderDialog
from pro.autostretch import autostretch as _autostretch
from ops.scripts import ScriptManager


VERSION = "1.5.5"



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

import faulthandler

def _install_crash_logging():
    faulthandler.enable(all_threads=True)
    def _excepthook(t, v, tb):
        logging.critical("Uncaught exception", exc_info=(t, v, tb))
        try:
            faulthandler.dump_traceback(file=sys.stderr)
        except Exception:
            pass
    sys.excepthook = _excepthook

_install_crash_logging()


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



from pro.gui.main_window import AstroSuiteProMainWindow

if __name__ == "__main__":
    # --- Logging (catch unhandled exceptions to a file) ---
    import logging, sys, os, multiprocessing
    import tempfile
    from PyQt6.QtCore import Qt, QCoreApplication, QSettings
    from PyQt6.QtGui import QGuiApplication, QIcon, QPixmap, QColor
    from PyQt6.QtWidgets import QApplication, QSplashScreen, QMessageBox


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
            # Development mode - use current directory
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
        

    # --- Qt app (make splash ASAP) ---
    try:
        # Small nicety on some PyQt6 builds; ignore if not available.
        QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )
    except Exception:
        pass

    QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_DontShowIconsInMenus, False)

    QCoreApplication.setOrganizationName("SetiAstro")
    QCoreApplication.setOrganizationDomain("setiastrosuite.pro")
    QCoreApplication.setApplicationName("Seti Astro Suite Pro")

    app = QApplication(sys.argv)
    install_crash_handlers(app) 
    app.setWindowIcon(QIcon(windowslogo_path if os.path.exists(windowslogo_path) else icon_path))

    # Helper: build QPixmap for the splash from icon_path (fallback to windowslogo_path)
    def _splash_pixmap():
        p = icon_path if os.path.exists(icon_path) else (windowslogo_path if os.path.exists(windowslogo_path) else icon_path)
        ext = os.path.splitext(p)[1].lower()
        if ext == ".ico":
            ic = QIcon(p)
            pm = ic.pixmap(512, 512)
            if pm.isNull():
                pm = QPixmap(p)
            return pm
        pm = QPixmap(p)
        if pm.isNull():
            pm = QIcon(p).pixmap(512, 512)
        return pm

    # --- Splash FIRST ---
    _pm = _splash_pixmap()
    # Use the SplashScreen window type; keep frameless; ensure it will be deleted.
    splash = QSplashScreen(_pm, Qt.WindowType.SplashScreen)
    splash.setWindowFlag(Qt.WindowType.FramelessWindowHint, True)
    splash.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)

    splash.show()
    splash.showMessage(
        "Starting Seti Astro Suite Pro...",
        Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignLeft,
        QColor("white"),
    )
    app.processEvents()
    # --- Windows exe / multiprocessing friendly (after splash is visible) ---
    try:
        import multiprocessing
        multiprocessing.freeze_support()
        try:
            multiprocessing.set_start_method("spawn", force=True)
        except RuntimeError:
            # Already set in this interpreter
            pass
    except Exception:
        # If something here is slow/complains, at least the splash is up.
        logging.exception("Multiprocessing init failed (continuing).")

    try:
        # If you have heavy imports/numba warmups, you can update the splash here:
        splash.showMessage(
            "Initializing UI...",
            Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignLeft,
            QColor("white"),
        )
        app.processEvents()

        import warnings
        from matplotlib import MatplotlibDeprecationWarning
        warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)


        # Your image manager + main window
        imgr = ImageManager(max_slots=100)
        win = AstroSuiteProMainWindow(image_manager=imgr)
        win.show()

        # Start background Numba warmup after UI is visible
        try:
            from pro.numba_warmup import start_background_warmup
            start_background_warmup()
        except Exception:
            pass  # Non-critical if warmup fails

        # Ensure the splash cannot resurrect later:
        try:
            splash.finish(win)   # hide and wait for first paint of win
        finally:
            splash.hide()
            splash.close()       # triggers WA_DeleteOnClose if set
            splash.deleteLater() # in case close doesn't destroy immediately
            splash = None        # drop all references
        print(f"Seti Astro Suite Pro v{VERSION} (build {BUILD_TIMESTAMP}) up and running!")
        sys.exit(app.exec())

    except Exception:
        import traceback
        try:
            if splash is not None:
                splash.hide()
                splash.close()
                splash.deleteLater()
        except Exception:
            pass
        tb = traceback.format_exc()
        logging.error("Unhandled exception occurred\n%s", tb)
        msg = QMessageBox(None)
        msg.setIcon(QMessageBox.Icon.Critical)
        msg.setWindowTitle("Application Error")
        # Short headline:
        msg.setText("An unexpected error occurred.")
        # One-line summary:
        msg.setInformativeText(tb.splitlines()[-1] if tb else "See details.")
        # Full traceback:
        msg.setDetailedText(tb)
        msg.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg.exec()
        sys.exit(1)
