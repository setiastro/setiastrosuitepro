#pro.gui.main_window.py
from pro.runtime_torch import add_runtime_to_sys_path, _ban_shadow_torch_paths, _purge_bad_torch_from_sysmodules
add_runtime_to_sys_path(status_cb=lambda *_: None)
_ban_shadow_torch_paths(status_cb=lambda *_: None)
_purge_bad_torch_from_sysmodules(status_cb=lambda *_: None)

# ============================================================================
# Standard Library Imports
# ============================================================================
import importlib
import json
import logging
import math
import os
import re
import sys
import threading
import time
import traceback
import warnings
import webbrowser
from datetime import datetime
from decimal import getcontext
from io import BytesIO
from itertools import combinations
from math import isnan
from pathlib import Path
from typing import List, Tuple, Dict, Set, Optional
from urllib.parse import quote, quote_plus

# ============================================================================
# Third-Party Imports
# ============================================================================
import numpy as np
import matplotlib
from tifffile import imwrite
from xisf import XISF

# ============================================================================
# Bootstrap Configuration (must run early)
# ============================================================================
from pro.config_bootstrap import ensure_mpl_config_dir
from pro.metadata_patcher import apply_metadata_patches

# Apply matplotlib configuration
_MPL_CFG_DIR = ensure_mpl_config_dir()

# Apply metadata patches for frozen builds
apply_metadata_patches()

# Configure matplotlib backend
matplotlib.use("QtAgg")

# Configure warnings
warnings.filterwarnings(
    "ignore",
    message=r"Call to deprecated function \(or staticmethod\) _destroy\.",
    category=DeprecationWarning
)

# Configure lightkurve style
os.environ['LIGHTKURVE_STYLE'] = 'default'

# Configure stdout encoding if available
if (sys.stdout is not None) and (hasattr(sys.stdout, "reconfigure")):
    sys.stdout.reconfigure(encoding='utf-8')

# ============================================================================
# Lazy Imports for Heavy Dependencies
# ============================================================================
from pro.lazy_imports import (
    get_photutils_isophote,
    get_Ellipse,
    get_EllipseGeometry,
    get_build_ellipse_model,
    get_lightkurve,
    get_reproject_interp,
    lazy_cv2,
)

# scipy.ndimage imports removed - not used in main module
# gaussian_filter, laplace, zoom loaded on demand in specific modules

# ============================================================================
# Shared UI Utilities
# ============================================================================
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
from PyQt6.QtGui import (QPixmap, QColor, QIcon, QKeySequence, QShortcut,
     QGuiApplication, QStandardItemModel, QStandardItem, QAction, QPalette,
     QBrush, QActionGroup, QDesktopServices, QFont, QTextCursor, QPainter
)

# ----- QtCore -----
from PyQt6.QtCore import (Qt, pyqtSignal, QCoreApplication, QTimer, QSize, QSignalBlocker,  QModelIndex, QThread, QUrl, QSettings, QEvent, QByteArray, QObject
)

from PyQt6.QtNetwork import QNetworkAccessManager, QNetworkRequest, QNetworkReply


# Math functions

import math


#from pro.subwindow import ImageSubWindow, TableSubWindow
#from legacy.image_manager import ImageManager


from pro.autostretch import autostretch
from pro.autostretch import autostretch as _autostretch
from pro.rgb_extract import extract_rgb_channels



from legacy.numba_utils import (
    rescale_image_numba,
    flip_horizontal_numba,
    flip_vertical_numba,
    rotate_90_clockwise_numba,
    rotate_90_counterclockwise_numba,
    invert_image_numba,
    rotate_180_numba,
)

try:
    from pro._generated.build_info import BUILD_TIMESTAMP
except Exception:
    BUILD_TIMESTAMP = "dev"








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

# GUI Mixins for modular code organization
from pro.gui.mixins import (
    DockMixin, MenuMixin, ToolbarMixin, FileMixin,
    ThemeMixin, GeometryMixin, ViewMixin, HeaderMixin, MaskMixin, UpdateMixin
)


class AstroSuiteProMainWindow(
    DockMixin, MenuMixin, ToolbarMixin, FileMixin,
    ThemeMixin, GeometryMixin, ViewMixin, HeaderMixin, MaskMixin, UpdateMixin,
    QMainWindow
):
    currentDocumentChanged = pyqtSignal(object)  # ImageDocument | None

    def __init__(self, image_manager=None, parent=None,
                 version: str = "dev", build_timestamp: str = "dev"):
        super().__init__(parent)
        from pro.doc_manager import DocManager
        from pro.window_shelf import WindowShelf, MinimizeInterceptor
        from imageops.mdi_snap import MdiSnapController
        from ops.scripts import ScriptManager
        self._version = version
        self._build_timestamp = build_timestamp
        self.setWindowTitle(f"Seti Astro Suite Pro v{self._version}")
        self.resize(1400, 900)
        self._ensure_network_manager()
        self.app_icon = QIcon(windowslogo_path if os.path.exists(windowslogo_path) else icon_path)
        self.setWindowIcon(self.app_icon)
        self._doc = None
        self._force_close_all = False
        self.settings = QSettings()  # reuse everywhere
        self._last_active_view = None
        self._current_active_sw = None
        self._last_active_sw = None
        self._suspend_dock_sync = False        # pause action<->dock syncing while minimized
        self._dock_visibility_snapshot = {}     # objectName -> bool
        self._pre_minimize_state = None
        self._dock_vis_intended: dict[str, bool] = {}   # last known intended visibility per dock
        self._last_good_state: QByteArray | None = None
        auto_on = self.settings.value("view/auto_fit_on_resize", False, type=bool)
        self._auto_fit_on_resize = bool(auto_on)
        self._last_headless_command: dict | None = None
        self._headless_history: list[dict] = []  # newest at the end
        self._headless_history_max = 150

        # -- Recent files / projects ---------------------------------------
        self._recent_max = 20
        self._recent_image_paths: list[str] = []
        self._recent_project_paths: list[str] = []
        self._load_recent_lists()

        # Debounce timer for auto-fit on resize
        self._auto_fit_timer = QTimer(self)
        self._auto_fit_timer.setSingleShot(True)
        self._auto_fit_timer.setInterval(200)  # ms: tweak if you want snappier/slower
        self._auto_fit_timer.timeout.connect(self._apply_auto_fit_resize)
        # Core
        self.doc_manager = DocManager(image_manager=image_manager, parent=self)
        self.docman = self.doc_manager  # legacy alias for older code
        self.docman.imageRegionUpdated.connect(self._on_doc_region_updated)

        # MDI workspace
        self.mdi = MdiArea()
        self.mdi.setViewMode(QMdiArea.ViewMode.SubWindowView)

        # Absolute path to the background image
        bg_path = os.path.abspath(os.path.join("images", "background.png"))
        self._bg_pixmap = QPixmap(bg_path)

        def _draw_transparent_bg(event):
            painter = QPainter(self.mdi.viewport())
            
            painter.fillRect(self.mdi.rect(), QColor("#1e1e1e"))

            if not self._bg_pixmap.isNull():
                opacity_percent = self.settings.value("display/bg_opacity", 50, type=int)
                opacity_float = opacity_percent / 100.0
                painter.setOpacity(opacity_float)
                x = (self.mdi.width() - self._bg_pixmap.width()) // 2
                y = (self.mdi.height() - self._bg_pixmap.height()) // 2
                painter.drawPixmap(x, y, self._bg_pixmap)

        self.mdi.paintEvent = _draw_transparent_bg

        self.mdi.subWindowActivated.connect(self._remember_active_pair)
        self.mdi.backgroundDoubleClicked.connect(self.open_files)   # <- new
        QShortcut(QKeySequence("Ctrl+PgDown"), self, activated=self._toggle_last_active_view)
        QShortcut(QKeySequence("Ctrl+PgUp"), self, activated=self._toggle_last_active_view)
        self.setCentralWidget(self.mdi)
        self._snap = MdiSnapController(self.mdi, threshold_px=8)

        self.window_shelf = WindowShelf(self)
        self.window_shelf.setObjectName("WindowShelfDock") 
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.window_shelf)
        self.window_shelf.hide()

        self._minimize_interceptor = MinimizeInterceptor(self.window_shelf, self)
        self.currentDocumentChanged.connect(self._sync_docman_active)
        self.scriptman = ScriptManager(self)
        self.scriptman.load_registry()
        # Docks
        self._init_explorer_dock()
        self._init_console_dock()
        self._init_header_viewer_dock()
        self._init_layers_dock()
        self._shutting_down = False
        self._init_status_log_dock()
        self._init_log_dock()
        self._hook_stdout_stderr()

        # Toolbar / actions
        self._create_actions()
        self._init_menubar()
        self._init_toolbar()
        self._install_command_search()

        # Keep explorer in sync
        self.docman.documentAdded.connect(self._add_doc_to_explorer)
        self.docman.documentRemoved.connect(self._remove_doc_from_explorer)
        #self.mdi.viewStateDropped.connect(self._handle_viewstate_drop)
        self.docman.documentAdded.connect(self._on_doc_added_for_header_sync)
        self.docman.documentRemoved.connect(self._on_doc_removed_for_header_sync)
        self.mdi.subWindowActivated.connect(lambda _sw: self._hdr_refresh_timer.start(0))
        self.mdi.subWindowActivated.connect(self._on_subwindow_activated)
        self.mdi.commandDropped.connect(self._handle_command_drop)
        self.docman.documentAdded.connect(self._open_subwindow_for_added_doc)
        self.mdi.maskDropped.connect(self._handle_mask_drop)
        self.mdi.astrometryDropped.connect(self._on_astrometry_drop)
        self.docman.documentAdded.connect(lambda _d: self._refresh_mask_action_states())
        self.docman.documentRemoved.connect(lambda _d: self._refresh_mask_action_states())
        self.docman.documentAdded.connect(self._on_document_added)
        self.mdi.viewStateDropped.connect(self._on_mdi_viewstate_drop)
        self.mdi.linkViewDropped.connect(self._on_linkview_drop)

        self.doc_manager.set_mdi_area(self.mdi)

        # Keep the toolbar in sync whenever anything relevant changes
        self.doc_manager.documentAdded.connect(lambda *_: self.update_undo_redo_action_labels())
        self.doc_manager.documentRemoved.connect(lambda *_: self.update_undo_redo_action_labels())
        self.doc_manager.imageRegionUpdated.connect(lambda *_: self.update_undo_redo_action_labels())
        self.doc_manager.previewRepaintRequested.connect(lambda *_: self.update_undo_redo_action_labels())

        # Also refresh when the active subwindow changes
        try:
            self.mdi.subWindowActivated.connect(lambda *_: self.update_undo_redo_action_labels())
        except Exception:
            pass

        try:
            QApplication.instance().focusChanged.connect(
                lambda *_: QTimer.singleShot(0, self.update_undo_redo_action_labels)
            )
        except Exception:
            pass

        self.shortcuts.load_shortcuts()
        self._ensure_persistent_names() 
        self._restore_window_placement()
        try:
            from pro.function_bundle import restore_function_bundle_chips
            restore_function_bundle_chips(self)
        except Exception:
            pass

        try:
            from pro.view_bundle import restore_view_bundle_chips
            restore_view_bundle_chips(self)
        except Exception:
            pass
        self._updates_url = self.settings.value(
            "updates/url",
            "https://raw.githubusercontent.com/setiastro/setiastrosuitepro/main/updates.json",
            type=str
        )

        app = QApplication.instance()

        # Re-entrancy guard + debounce
        self._theme_guard = False
        self._theme_debounce = QTimer(self)
        self._theme_debounce.setSingleShot(True)
        self._theme_debounce.timeout.connect(self._apply_theme_safely)

        # Build a safe list of theme-relevant event types for this Qt build
        _names = [
            "ApplicationPaletteChange",  # present on all Qt6
            "PaletteChange",             # older / extra signal from widgets
            "ColorSchemeChange",         # newer Qt (6.5+)
            # "ThemeChange",             # NOT reliable--do NOT include
            # "StyleChange",             # optional; usually noisy, so we skip
        ]
        _types = []
        for n in _names:
            t = getattr(QEvent.Type, n, None)
            if t is not None:
                _types.append(t)
        self._theme_events = tuple(_types)

        # Listen only on the app object
        app.installEventFilter(self)

        self.apply_theme_from_settings()
        self._populate_view_panels_menu()
        # Startup check (no lambdas)
        if self.settings.value("updates/check_on_startup", True, type=bool):
            QTimer.singleShot(1500, self.check_for_updates_startup)

        self._hdr_refresh_timer = QTimer(self)
        self._hdr_refresh_timer.setSingleShot(True)
        self._hdr_refresh_timer.timeout.connect(lambda: self._refresh_header_viewer(self._active_doc()))

        try:
            self.docman.imageRegionUpdated.connect(self._on_image_region_updated_global)
        except Exception:
            pass

        try:
            self._last_good_state = self.saveState()
        except Exception:
            pass

        self.linker = ViewLinkController(self.mdi)

        # attach any already-open subwindows
        for sw in self.mdi.subWindowList():
            try:
                self.linker.attach_view(sw.widget())
            except Exception:
                pass

        self.mdi.subWindowActivated.connect(self._on_sw_activated)
        self.mdi.subWindowActivated.connect(lambda _=None: self._sync_link_action_state())  
        self._link_views_enabled = QSettings().value("view/link_scroll_zoom", True, type=bool)

        self.status_log_dock.hide()

    # _init_log_dock, _hook_stdout_stderr, and _append_log_text are now in DockMixin


    def _on_sw_activated(self, sw):
        if not sw:
            return
        view = sw.widget()
        try:
            self.linker.attach_view(view)
        except Exception:
            pass

    def _on_document_opened(self, doc):
        try:
            doc.changed.connect(self.update_undo_redo_action_labels)
        except Exception:
            pass
        self.update_undo_redo_action_labels()

    def _promote_roi_preview_to_real_doc(self, st: dict, preview_doc) -> None:
        """
        Turn a ROI preview doc into a real ImageDocument with correct WCS,
        using the preview's already-cropped pixels and roi_header.

        preview_doc: the RoiViewDocument we dragged from.
        """
        dm = self.doc_manager

        # ---- 1) Get pixels from the preview doc ----

        arr = getattr(preview_doc, "image", None)
        if arr is None:
            
            return

        H, W = arr.shape[:2]


        roi = st.get("roi") or [0, 0, W, H]


        if len(roi) != 4:

            x = y = 0
            w, h = W, H
        else:
            x, y, w, h = map(int, roi)


        # NOTE: roi in the drag state is in *full-frame* coords; the preview is
        # already cropped. Cropping again with (x, y) would be wrong and often OOB,
        # so we just use the entire preview image as the final crop.
        crop = arr.copy()


        # ---- 2) Build metadata from the preview, stripping preview/ROI flags ----
        pmeta = getattr(preview_doc, "metadata", {}) or {}


        meta = {
            k: v
            for k, v in pmeta.items()
            if k not in (
                "is_preview", "roi", "preview_name",
                "roi_wcs", "roi_header",
                "base_doc_uid",
                "wcs", "original_wcs", "sip_wcs",
                "__header_snapshot__",
            )
        }


        # Mark that this document is a *promoted ROI* doc so later DnD
        # knows it's already a standalone image and should just duplicate it.
        meta["is_roi_doc"] = True


        # Mono / bit-depth flags
        meta["is_mono"] = bool(
            crop.ndim == 2 or (crop.ndim == 3 and crop.shape[2] == 1)
        )
        meta["bit_depth"] = meta.get("bit_depth", "32-bit floating point")


        # ---- 3) Build a nice display name without "(Preview)" chained on ----
        # e.g. "andromedasolved.fit (Preview) [ROI 1793,1067,1132Ã--954]"
        disp = preview_doc.display_name() if hasattr(preview_doc, "display_name") else ""


        # Strip any existing "[ROI ...]" suffix
        if "[ROI" in disp:
            disp = disp.split("[ROI", 1)[0].rstrip()

        # Strip " (Preview)" if present
        if " (Preview)" in disp:
            disp = disp.split(" (Preview)", 1)[0].rstrip()

        if not disp:
            disp = pmeta.get("display_name") or "Untitled"


        meta["display_name"] = f"{disp} [ROI {x},{y},{w}Ã--{h}]"


        # ---- 4) Use the preview's ROI header as the *primary* header ----
        from astropy.wcs import WCS

        roi_hdr = pmeta.get("roi_header")
        base_hdr = (
            pmeta.get("original_header")
            or pmeta.get("fits_header")
            or pmeta.get("header")
        )


        if roi_hdr is not None:

            # We have a true ROI header created by the preview machinery.
            # Work on a copy so we don't mutate the original in-place.
            hdr = roi_hdr.copy()
            hdr["NAXIS1"] = int(W)
            hdr["NAXIS2"] = int(H)

            meta["original_header"] = hdr
            meta["fits_header"] = hdr
            meta["header"] = hdr  # HeaderViewer sees this


            # Build WCS directly from this already-cropped header
            try:
                meta["wcs"] = WCS(hdr)

            except Exception as e:

                # Fallback: reuse any existing WCS object if one was stored
                w_existing = (
                    pmeta.get("roi_wcs")
                    or pmeta.get("wcs")
                    or pmeta.get("original_wcs")
                )

                if w_existing is not None:
                    meta["wcs"] = w_existing


            # Optional: snapshot for project I/O / header viewer
            try:
                from pro.doc_manager import _dm_json_sanitize
                meta["__header_snapshot__"] = {
                    "format": "dict",
                    "items": {str(k): _dm_json_sanitize(v) for k, v in hdr.items()},
                }

            except Exception as e:

                meta.pop("__header_snapshot__", None)

        else:

            # No dedicated roi_header: this is either a "plain" doc or a
            # promoted ROI doc. Do NOT try to re-derive WCS from the header,
            # since it may contain non-WCS strings like:
            #   "Calibrated: bias/dark sub, flat division."
            #
            # Instead, just:
            #   - propagate any existing WCS object
            #   - copy whatever header we have and fix NAXIS1/2
            w_existing = (
                pmeta.get("roi_wcs")
                or pmeta.get("wcs")
                or pmeta.get("original_wcs")
            )

            if w_existing is not None:
                meta["wcs"] = w_existing


            if base_hdr is not None:
                hdr = base_hdr.copy()
                hdr["NAXIS1"] = int(W)
                hdr["NAXIS2"] = int(H)

                meta["original_header"] = hdr
                meta["fits_header"] = hdr
                meta["header"] = hdr


                # Snapshot is optional here; no WCS rebuild
                try:
                    from pro.doc_manager import _dm_json_sanitize
                    meta["__header_snapshot__"] = {
                        "format": "dict",
                        "items": {str(k): _dm_json_sanitize(v) for k, v in hdr.items()},
                    }

                except Exception as e:

                    meta.pop("__header_snapshot__", None)

        # ---- 5) Create a real ImageDocument so Explorer sees it as a normal doc ----

        new_doc = dm.open_array(crop, metadata=meta, title=meta.get("display_name"))


        # IMPORTANT: do NOT call any "rebuild cropped WCS" helpers here.
        # We already have a correct, final ROI header.

        # ---- 6) Find the subwindow that doc_manager already spawned ----
        sw = None
        try:

            for sub in self.mdi.subWindowList():
                w = sub.widget() if hasattr(sub, "widget") else None
                d = getattr(w, "document", None)
                # DEBUG: print each candidate

                if d is new_doc:
                    sw = sub

                    break
            if sw is None:
                pass
        except Exception as e:
            print("[Main] ROI promotion: error searching for subwindow:", e)

        # ---- 7) Apply viewstate to that subwindow ----
        if sw and hasattr(sw, "widget"):

            wv = sw.widget()
            if st.get("autostretch") and hasattr(wv, "set_autostretch"):

                wv.set_autostretch(True)
                if hasattr(wv, "set_autostretch_target"):
                    wv.set_autostretch_target(float(st.get("autostretch_target", 0.25)))
            if hasattr(wv, "set_view_transform"):

                wv.set_view_transform(
                    float(st.get("scale", 1.0)),
                    int(st.get("hval", 0)),
                    int(st.get("vval", 0)),
                    from_link=False,
                )
        else:
            pass




    def _on_mdi_viewstate_drop(self, st: dict, target_sw: object | None):
        dm = self.doc_manager
        doc = None



        # Prefer *stable* identifiers over the proxy pointer
        uid     = st.get("doc_uid")
        doc_ptr = st.get("doc_ptr")
        fpath   = (st.get("file_path") or "").strip()


        # ----------------------------------------
        # 1) First try: look up by UID in DocManager.
        #    This gives us the real ImageDocument,
        #    not the dynamic _DocProxy tied to tabs.
        # ----------------------------------------
        if uid and hasattr(dm, "lookup_by_uid"):

            try:
                doc = dm.lookup_by_uid(uid)

            except Exception as e:

                doc = None
        else:
            if uid:
                pass

        # ----------------------------------------
        # 2) Fallback: DocManager pointer-based helpers
        #    (still better than going through proxies).
        # ----------------------------------------
        if doc is None and doc_ptr is not None:
            if hasattr(dm, "get_document_by_ptr"):

                try:
                    doc = dm.get_document_by_ptr(doc_ptr)

                except Exception as e:
                    pass
            else:
                pass

        if doc is None and doc_ptr is not None:
            if hasattr(dm, "lookup_by_python_id"):

                try:
                    doc = dm.lookup_by_python_id(doc_ptr)

                except Exception as e:
                    pass
            else:
                pass

        # ----------------------------------------
        # 3) Last-ditch: scan subwindows. We *still*
        #    try UID / file_path first here, and only
        #    finally fall back to matching the proxy.
        # ----------------------------------------
        if doc is None:
            
            try:
                for sw in self.mdi.subWindowList():
                    w = sw.widget() if hasattr(sw, "widget") else None
                    d = getattr(w, "document", None)
                    if d is None:
                        continue

                    meta = getattr(d, "metadata", {}) or {}
                    d_uid = getattr(d, "uid", None)
                    d_path = (meta.get("file_path") or "").strip()


                    # 3a) UID match on the underlying doc (if present)
                    if uid and d_uid == uid:

                        doc = d
                        break

                    # 3b) file_path match
                    meta = getattr(d, "metadata", {}) or {}
                    d_path = (meta.get("file_path") or "").strip()

                    # DEBUG print is already there in your version:
                    # print("  [VIEWSTATE_DROP] candidate doc:", d, "uid:", d_uid, "file_path:", d_path)

                    # Only use file_path when there is NO doc_uid in the drag state.
                    # If uid is present, we want to keep scanning for the ROI doc whose uid matches.
                    if fpath and d_path == fpath and not uid:

                        doc = d
                        break

                    # 3c) absolute last resort -> pointer to proxy
                    if doc_ptr is not None and id(d) == doc_ptr:

                        doc = d
                        break
            except Exception as e:
                pass

        if doc is None:
            print("[Main] viewstate_drop: could NOT resolve document; aborting.")
            print("[VIEWSTATE_DROP] EXIT (no doc)")
            return



        # ----------------------------------------
        # 4) Peek at metadata to see if this is a
        #    preview-of-ROI situation.
        #    NOTE: doc is now a *real* document, not
        #    the _DocProxy, so this metadata is stable
        #    and no longer depends on tabs.
        # ----------------------------------------
        pmeta        = getattr(doc, "metadata", {}) or {}
        base_uid     = pmeta.get("base_doc_uid")
        roi_base_doc = None


        if base_uid and hasattr(dm, "lookup_by_uid"):

            try:
                roi_base_doc = dm.lookup_by_uid(base_uid)

            except Exception as e:

                roi_base_doc = None
        elif base_uid:
            pass

        # ----------------------------------------
        # 5) Decide behavior: new view vs copy transform
        # ----------------------------------------
        force_new   = (target_sw is None)
        source_kind = st.get("source_kind")
        roi         = st.get("roi")
        is_preview  = (source_kind in ("preview", "roi-preview")) or bool(roi)



        # If this looks like a *preview of an already-promoted ROI doc*,
        # we don't want to re-promote. Instead, treat it as a normal
        # ROI image: duplicate the ROI base doc and ignore ROI coords.
        if is_preview and roi_base_doc is not None:
            base_meta = getattr(roi_base_doc, "metadata", {}) or {}

            if base_meta.get("is_roi_doc"):

                # preview-of-ROI -> behave like a plain ROI image doc
                doc = roi_base_doc
                is_preview = False
                roi = None

        # ----------------------------------------
        # 6) ROI promotion block (only for genuine
        #    previews of the *full* base doc).
        # ----------------------------------------
        if force_new and is_preview and roi and len(roi) == 4:

            # If this doc is already a promoted ROI doc, do NOT re-promote it.
            # (Catches the case where the preview doc itself carries the flag.)
            pmeta = getattr(doc, "metadata", {}) or {}

            if not pmeta.get("is_roi_doc"):
                try:
                    x, y, w, h = map(int, roi)  # sanity / logging

                    self._promote_roi_preview_to_real_doc(st, doc)

                    return
                except Exception as e:
                    try:
                        self._log(f"[viewstate-drop ROI] failed: {e}")
                    except Exception:
                        print("[Main] viewstate_drop: ROI promotion failed:", e)
                    # fall through to default behavior

            else:
                pass

        # ----------------------------------------
        # 7) Default behavior:
        #    - Background drop (force_new=True) -> new
        #      subwindow for *this* doc (full or ROI).
        #    - Drop onto existing subwindow -> just
        #      copy the view transform.
        # ----------------------------------------
        # ----------------------------------------
        # 7) Default behavior:
        #    - Background drop (force_new=True) -> new
        #      *independent document* (full or ROI),
        #      same as legacy _duplicate_view_from_state.
        #    - Drop onto existing subwindow -> just
        #      copy the view transform.
        # ----------------------------------------
        if force_new:
            # We're here only if:
            #  - it's NOT a preview (normal full or promoted ROI), or
            #  - ROI promotion didn't apply and we fell through.
            base_doc = doc

            # 1) Duplicate the underlying document
            try:
                base_name = ""
                try:
                    base_name = base_doc.display_name() or "Untitled"
                except Exception:
                    base_name = "Untitled"

                try:
                    base_name = _strip_ui_decorations(base_name)
                except Exception:
                    # minimal fallback: remove known glyph prefixes and "Active View: "
                    while len(base_name) >= 2 and base_name[1] == " " and base_name[0] in "â- â--â--†â-²â-ªâ-«â€¢â--¼â--»â--¾â--½":
                        base_name = base_name[2:]
                    if base_name.startswith("Active View: "):
                        base_name = base_name[len("Active View: "):]

                new_doc = self.docman.duplicate_document(
                    base_doc, new_name=f"{base_name}_duplicate"
                )
            except Exception as e:
                print("[Main] viewstate_drop: duplicate_document failed, falling back to original doc:", e)
                new_doc = base_doc  # worst-case: still just reuse

            # 2) Let doc_manager's documentAdded handler create the subwindow.
            #    We just wait for it to show up and then apply the view state.
            from PyQt6.QtCore import QTimer

            def _apply_when_ready():
                sw = self._find_subwindow_for_doc(new_doc)
                if not sw:
                    QTimer.singleShot(0, _apply_when_ready)
                    return

                view = sw.widget()
                try:
                    # Reuse the same helper the legacy code used
                    self._apply_view_state_to_view(view, st)
                except Exception:
                    pass

                self.mdi.setActiveSubWindow(sw)
                if hasattr(self, "_log"):
                    try:
                        self._log(f"Duplicated as independent document -> '{new_doc.display_name()}'")
                    except Exception:
                        pass

            QTimer.singleShot(0, _apply_when_ready)

        else:
            # Dropped onto an existing subwindow -> just copy view transform
            tgt = target_sw.widget() if hasattr(target_sw, "widget") else None

            if tgt and hasattr(tgt, "set_view_transform"):
                tgt.set_view_transform(
                    float(st.get("scale", 1.0)),
                    int(st.get("hval", 0)),
                    int(st.get("vval", 0)),
                    from_link=False,
                )
            if tgt and st.get("autostretch") and hasattr(tgt, "set_autostretch"):
                tgt.set_autostretch(True)
                if hasattr(tgt, "set_autostretch_target"):
                    tgt.set_autostretch_target(
                        float(st.get("autostretch_target", 0.25))
                    )





    def _open_from_explorer(self, doc):
        sw = self._find_subwindow_for_doc(doc)
        if sw:
            # robust raise path (covers minimized/hidden)
            try:
                sw.show(); sw.widget().show()
                st = sw.windowState()
                if st & Qt.WindowState.WindowMinimized:
                    sw.setWindowState(st & ~Qt.WindowState.WindowMinimized)
                self.mdi.setActiveSubWindow(sw)
                sw.raise_()
            except Exception:
                pass
            return
        self._spawn_subwindow_for(doc, force_new=False)


    def _on_doc_region_updated(self, doc, roi):
        """
        Called after DocManager applies an edit to a doc (full image or ROI).
        Refresh the visible view for that doc; if a Preview tab is active, rebuild it.
        """
        sw = self._find_subwindow_for_doc(doc)
        if not sw:
            return
        vw = sw.widget()

        # If your ImageSubWindow exposes targeted preview refresh:
        if roi is not None:
            # Prefer a region-aware hook if present
            if hasattr(vw, "refresh_preview_region"):
                try:
                    vw.refresh_preview_region(roi)  # expects (x,y,w,h)
                    return
                except Exception:
                    pass

        # If a preview tab is active, ask it to rebuild from the document
        if hasattr(vw, "has_active_preview") and callable(vw.has_active_preview):
            try:
                if vw.has_active_preview():
                    # common helper names; use whichever you have
                    for m in ("rebuild_active_preview", "refresh_from_document", "update_pixmap_from_doc"):
                        if hasattr(vw, m):
                            getattr(vw, m)()
                            return
            except Exception:
                pass

        # Fallback: refresh the main image view
        for m in ("refresh_from_document", "update_pixmap_from_doc"):
            if hasattr(vw, m):
                try:
                    getattr(vw, m)()
                    return
                except Exception:
                    pass

        # Last resort: repaint
        try:
            vw.update()
            sw.update()
        except Exception:
            pass


    def _alive(self, obj) -> bool:
        if obj is None:
            return False
        try:
            _ = obj.metaObject()  # raises if C++ object already deleted
            return True
        except RuntimeError:
            return False

    def _on_document_added(self, doc):
        # Helpful debug:
        try:
            is_table = (getattr(doc, "metadata", {}).get("doc_type") == "table") or \
                    (hasattr(doc, "rows") and hasattr(doc, "headers"))
            self._log(f"[documentAdded] {type(doc).__name__}  table={is_table}  name={getattr(doc, 'display_name', lambda:'?')()}")
        except Exception:
            pass

        self._spawn_subwindow_for(doc)

    # --- UI scaffolding ---
    def _on_console_context_menu(self, pos):
        lw = self.console
        global_pos = lw.viewport().mapToGlobal(pos)

        menu = QMenu(lw)
        act_copy_selected = menu.addAction("Copy Selected")
        act_copy_all      = menu.addAction("Copy All")
        menu.addSeparator()
        act_select_all    = menu.addAction("Select All Lines")
        act_clear         = menu.addAction("Clear Console")

        action = menu.exec(global_pos)
        if action is None:
            return

        if action is act_select_all:
            # thanks to ExtendedSelection this will highlight every row
            lw.selectAll()

        elif action is act_copy_selected:
            items = lw.selectedItems()
            # if nothing is selected, fall back to the row under the cursor
            if not items:
                item = lw.itemAt(pos)
                if item is not None:
                    items = [item]
            if items:
                text = "\n".join(i.text() for i in items)
                QGuiApplication.clipboard().setText(text)

        elif action is act_copy_all:
            lines = [lw.item(i).text() for i in range(lw.count())]
            if lines:
                QGuiApplication.clipboard().setText("\n".join(lines))

        elif action is act_clear:
            lw.clear()


    def _first_place_status_log_if_needed(self):
        s = self.settings  # QSettings you already have
        flag_key = "ui/status_log/placed_v1"

        # If we've already placed it once, or a full window state exists, do nothing
        if s.value(flag_key, False, type=bool):
            return

        # If you have a "main window state" key, use it to detect prior layouts:
        has_main_state = s.contains("mainwindow/state") or s.contains("ui/window_state")
        if has_main_state:
            return

        # Defer until after the window has a real geometry
        def _place():
            # OPTION A: float it centered on first run (recommended)
            self.status_log_dock.setFloating(True)

            g = self.frameGeometry()  # screen coords incl. frame
            w = min(900, int(g.width() * 0.6))
            h = min(320, int(g.height() * 0.3))
            x = g.x() + (g.width() - w) // 2
            y = g.y() + 60

            self.status_log_dock.resize(w, h)
            self.status_log_dock.move(x, y)
            self.status_log_dock.show()
            self.status_log_dock.raise_()

            # Remember we've "introduced" it once
            s.setValue(flag_key, True)
            s.sync()

            # Optional: immediately persist the whole layout so next launch restores it
            try:
                s.setValue("mainwindow/state", self.saveState())
            except Exception:
                pass

        QTimer.singleShot(0, _place)


    def _open_user_scripts_github(self):
        # User script examples on GitHub
        url = QUrl("https://github.com/setiastro/setiastrosuitepro/tree/main/scripts")
        QDesktopServices.openUrl(url)

    def _open_scripts_discord_forum(self):
        # Scripts Discord forum
        url = QUrl("https://discord.gg/vvYH82C82f")
        QDesktopServices.openUrl(url)

    # ----------------------------------------
    # Recent images / projects
    # ----------------------------------------
    def _clear_recent_images(self):
        self._recent_image_paths = []
        self._save_recent_lists()
        self._rebuild_recent_menus()

    def _clear_recent_projects(self):
        self._recent_project_paths = []
        self._save_recent_lists()
        self._rebuild_recent_menus()

    def _on_exit(self):
        # Funnel through closeEvent so your confirmation + state saves run
        self.close()

    # --- Link UI helpers (methods on AstroSuiteProMainWindow) ---
    def _cycle_group_for_active(self):
        g = self._current_group_of_active()
        order = [None, "A", "B", "C", "D"]  # how we cycle
        try:
            i = order.index(g)
        except ValueError:
            i = 0
        nxt = order[(i + 1) % len(order)]
        self._set_group_for_active(nxt)

    def _refresh_group_badges(self):
        # Update all window titles to include [A]/[B]/...
        for sw in self.mdi.subWindowList():
            v = sw.widget()
            try:
                g = self.linker.group_of(v)
            except Exception:
                g = None
            # Have each view rebuild its title with group suffix
            try:
                base = v.base_doc_title()
            except Exception:
                base = sw.windowTitle()
            suffix = f"  [Group {g}]" if g else ""
            try:
                v._rebuild_title(base=base + suffix)
            except Exception:
                # fallback: set directly
                sw.setWindowTitle((base or "Untitled") + suffix)
                sw.setToolTip(sw.windowTitle())


    def _set_group_for_active(self, name_or_none):
        sw = self.mdi.activeSubWindow()
        if not sw:
            return
        view = sw.widget()
        if hasattr(self, "linker"):
            self.linker.set_view_group(view, name_or_none)
        self._sync_link_action_state()
        self._refresh_group_badges()
        # Status bar toast
        sb = self.statusBar() if hasattr(self, "statusBar") else None
        if sb:
            msg = "Link: None" if not name_or_none else f"Link: Group {name_or_none}"
            sb.showMessage(msg, 2500)

    def _current_group_of_active(self):
        sw = self.mdi.activeSubWindow()
        if not sw:
            return None
        return self.linker.group_of(sw.widget()) if hasattr(self, "linker") else None

    def _on_linkview_drop(self, payload: dict, target_sw: QMdiSubWindow | None):
        if not target_sw:
            return
        target_view = target_sw.widget()
        if not hasattr(target_view, "set_view_transform"):
            return

        src_id = payload.get("source_view_id")
        if src_id is None:
            return

        # find the source by id(self) that was serialized
        src_view = None
        for sw in self.mdi.subWindowList():
            w = sw.widget()
            if id(w) == src_id:
                src_view = w
                break
        if src_view is None or src_view is target_view:
            return

        self._link_views_bidirectional(src_view, target_view, payload.get("modes", {}))

    def _link_views_bidirectional(self, a, b, modes):
        # Create a small registry to avoid duplicate connections
        if not hasattr(self, "_view_links"):
            self._view_links = set()
        key = tuple(sorted((id(a), id(b))))
        if key in self._view_links:
            return
        self._view_links.add(key)

        # Copy autostretch state once (optional)
        if modes.get("autostretch_once", True):
            try:
                b.set_autostretch(a.autostretch_enabled)
                b.set_autostretch_profile(a.autostretch_profile)
                b.set_autostretch_target(a.autostretch_target)
                b.set_autostretch_sigma(a.autostretch_sigma)
            except Exception:
                pass

        # Live pan/zoom both ways
        def apply_to(dst):
            return lambda scale, h, v: dst.set_view_transform(scale, h, v, from_link=True)

        a.viewTransformChanged.connect(apply_to(b))
        b.viewTransformChanged.connect(apply_to(a))

        # Immediately snap B to A so they look linked right away
        try:
            s, h, v = a._current_transform()
            b.set_view_transform(s, h, v, from_link=True)
        except Exception:
            pass

        # Optional: toast/status to confirm
        try:
            self.statusBar().showMessage(f"Linked views: {a.base_doc_title()} <-> {b.base_doc_title()}", 4000)
        except Exception:
            pass


    # --- Shortcuts -> View Panels (dynamic) --------------------------------------
    def _auto_fit_all_subwindows(self):
        """Apply auto-fit to every visible subwindow when the mode is enabled."""
        if not getattr(self, "_auto_fit_on_resize", False):
            return

        subs = self._visible_subwindows()
        if not subs:
            return

        # Remember current active so we can restore it
        prev_active = self.mdi.activeSubWindow()

        for sw in subs:
            # Make this subwindow active so _zoom_active_fit() works on it
            self.mdi.setActiveSubWindow(sw)
            self._zoom_active_fit()

        # Restore previously active subwindow if still around
        if prev_active and prev_active in subs:
            self.mdi.setActiveSubWindow(prev_active)


    def _visible_subwindows(self):
        # Only arrange visible, non-minimized views
        subs = [sw for sw in self.mdi.subWindowList()
                if sw.isVisible() and not (sw.windowState() & Qt.WindowState.WindowMinimized)]
        return subs

    def _cascade_views(self):
        self.mdi.cascadeSubWindows()
        self._auto_fit_all_subwindows()

    def _tile_views(self):
        self.mdi.tileSubWindows()
        self._auto_fit_all_subwindows()

    def _tile_views_direction(self, direction: str):
        """direction: 'v' for vertical columns, 'h' for horizontal rows"""
        subs = self._visible_subwindows()
        if not subs:
            return
        area = self.mdi.viewport().rect()
        # account for MDI viewport origin in global coords
        off = self.mdi.viewport().mapTo(self.mdi, area.topLeft())
        origin_x, origin_y = off.x(), off.y()

        n = len(subs)
        if direction == "v":  # columns
            col_w = max(1, area.width() // n)
            for i, sw in enumerate(subs):
                sw.setGeometry(origin_x + i*col_w, origin_y, col_w, area.height())
        else:  # rows
            row_h = max(1, area.height() // n)
            for i, sw in enumerate(subs):
                sw.setGeometry(origin_x, origin_y + i*row_h, area.width(), row_h)

        self._auto_fit_all_subwindows()


    def _tile_views_grid(self):
        """Arrange near-square grid across the MDI area."""
        subs = self._visible_subwindows()
        if not subs:
            return
        area = self.mdi.viewport().rect()
        off = self.mdi.viewport().mapTo(self.mdi, area.topLeft())
        origin_x, origin_y = off.x(), off.y()

        n = len(subs)
        # rows x cols ~ square
        cols = int(max(1, math.ceil(math.sqrt(n))))
        rows = int(max(1, math.ceil(n / cols)))

        cell_w = max(1, area.width() // cols)
        cell_h = max(1, area.height() // rows)

        for idx, sw in enumerate(subs):
            r = idx // cols
            c = idx % cols
            sw.setGeometry(origin_x + c*cell_w, origin_y + r*cell_h, cell_w, cell_h)

        self._auto_fit_all_subwindows()

    def _ensure_view_panels_menu(self):
        if getattr(self, "_shutting_down", False):
            return getattr(self, "_menu_view_panels", None)

        # if cached menu died (e.g., after repolish), drop it
        if not self._alive(getattr(self, "_menu_view_panels", None)):
            self._menu_view_panels = None

        if self._menu_view_panels:
            return self._menu_view_panels

        shortcuts_menu = None
        for act in self.menuBar().actions():
            m = act.menu()
            if m and (m.title().replace("&", "").strip().lower() == "shortcuts"):
                shortcuts_menu = m
                break
        if shortcuts_menu is None:
            shortcuts_menu = self.menuBar().addMenu("&Shortcuts")

        self._menu_view_panels = shortcuts_menu.addMenu("View Panels")
        self._view_panels_actions = {}
        return self._menu_view_panels

    def _is_inactive_or_minimized(self) -> bool:
        app = QApplication.instance()
        try:
            inactive = app.applicationState() != Qt.ApplicationState.ApplicationActive
        except Exception:
            inactive = False
        return bool(self.windowState() & Qt.WindowState.WindowMinimized) or inactive

    def _register_dock_in_view_menu(self, dock: QDockWidget):
        if dock is None:
            return
        if not dock.objectName():
            dock.setObjectName(dock.windowTitle().replace(" ", "") + "Dock")

        menu = self._ensure_view_panels_menu()
        name = dock.objectName()
        title = dock.windowTitle() or name

        act = self._view_panels_actions.get(name)
        if act is None:
            act = QAction(title, self, checkable=True)
            self._view_panels_actions[name] = act
            menu.addAction(act)

            def _on_action_toggled(checked, d=dock, n=name):
                if self._suspend_dock_sync or self._is_inactive_or_minimized():
                    return
                self._dock_vis_intended[n] = bool(checked)
                d.setVisible(bool(checked))
                # capture a last-good layout whenever the user changes vis while active
                try:
                    self._last_good_state = self.saveState()
                except Exception:
                    pass

            act.toggled.connect(_on_action_toggled)

            def _on_dock_vis_changed(vis, a=act, n=name):
                # Ignore layout churn during minimize/restore; don't let "False" uncheck the action.
                if self._suspend_dock_sync or self._is_inactive_or_minimized():
                    return
                a.setChecked(bool(vis))
                self._dock_vis_intended[n] = bool(vis)
                try:
                    self._last_good_state = self.saveState()
                except Exception:
                    pass

            dock.visibilityChanged.connect(_on_dock_vis_changed)

            dock.destroyed.connect(lambda _=None, n=name: self._remove_dock_from_view_menu(n))

        act.setChecked(dock.isVisible())
        self._dock_vis_intended[name] = dock.isVisible()

    # --- File pickers ---
    def _export_shortcuts_dialog(self):
        from PyQt6.QtWidgets import QFileDialog
        fn, _ = QFileDialog.getSaveFileName(
            self, "Export Shortcuts",
            "shortcuts.sass",
            "SAS Shortcuts (*.sass);;JSON (*.json);;All Files (*)"
        )
        if not fn:
            return
        ok, msg = self.shortcuts.export_to_file(fn)
        if not ok:
            try: self._log(f"Export failed: {msg}")
            except Exception as e:
                import logging
                logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")

    def _import_shortcuts_dialog(self):
        from PyQt6.QtWidgets import QFileDialog, QMessageBox
        fn, _ = QFileDialog.getOpenFileName(
            self, "Import Shortcuts",
            "", "SAS Shortcuts (*.sass *.json);;All Files (*)"
        )
        if not fn:
            return

        # Ask merge vs replace
        btn = QMessageBox.question(
            self, "Import Shortcuts",
            "Replace existing shortcuts? (Choose No to merge.)",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        replace = (btn == QMessageBox.StandardButton.Yes)

        ok, msg = self.shortcuts.import_from_file(fn, replace_existing=replace)
        if not ok:
            try: self._log(f"Import failed: {msg}")
            except Exception as e:
                import logging
                logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")

    def changeEvent(self, ev):
        super().changeEvent(ev)
        if ev.type() == QEvent.Type.WindowStateChange:
            if self.windowState() & Qt.WindowState.WindowMinimized:
                # entering minimized -- just guard; do not snapshot false vis
                self._suspend_dock_sync = True
                return

            # leaving minimized / other state change
            if self._suspend_dock_sync:
                try:
                    # Restore the last good layout if we have it
                    if self._last_good_state:
                        self.restoreState(self._last_good_state)
                except Exception:
                    pass

                # Enforce intended vis for each known dock (in case restoreState wasn't enough)
                for name, want in dict(self._dock_vis_intended).items():
                    d = self.findChild(QDockWidget, name)
                    if d:
                        if want: d.show()
                        else:    d.hide()

                # Re-sync menu checks
                for name, act in getattr(self, "_view_panels_actions", {}).items():
                    dock = self.findChild(QDockWidget, name)
                    if dock:
                        act.setChecked(dock.isVisible())

                # Resume normal syncing
                self._suspend_dock_sync = False

                # Capture a fresh last-good layout now that we're back
                try:
                    self._last_good_state = self.saveState()
                except Exception:
                    pass



    def _remove_dock_from_view_menu(self, name: str):
        if getattr(self, "_shutting_down", False):
            self._view_panels_actions.pop(name, None)
            return
        act = self._view_panels_actions.pop(name, None)
        if not act:
            return
        m = self._ensure_view_panels_menu()
        if not self._alive(m):
            return
        try:
            m.removeAction(act)
        except RuntimeError:
            pass
        try:
            act.deleteLater()
        except Exception:
            pass


    def _open_view_bundles(self):
        from pro.view_bundle import show_view_bundles
        try:

            show_view_bundles(self)
        except Exception as e:
            QMessageBox.warning(self, "View Bundles", f"Open failed:\n{e}")

    def _open_function_bundles(self):
        from pro.function_bundle import show_function_bundles
        try:

            show_function_bundles(self)
        except Exception as e:
            QMessageBox.warning(self, "Function Bundles", f"Open failed:\n{e}")

    def _open_scripts_folder(self):
        if hasattr(self, "scriptman"):
            self.scriptman.open_scripts_folder()

    def _reload_scripts(self):
        if not hasattr(self, "scriptman"):
            return
        self.scriptman.load_registry()
        if hasattr(self, "menu_scripts") and self.menu_scripts:
            self.scriptman.rebuild_menu(self.menu_scripts)
        self._log("[Scripts] Reload complete.")

    def _create_sample_script(self):
        if not hasattr(self, "scriptman"):
            return
        self.scriptman.create_sample_script()

    def _show_script_editor(self):
        if not hasattr(self, "script_editor_dock") or self.script_editor_dock is None:
            from ops.script_editor import ScriptEditorDock
            self.script_editor_dock = ScriptEditorDock(self, parent=self)
            self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.script_editor_dock)
        self.script_editor_dock.show()
        self.script_editor_dock.raise_()
        self.script_editor_dock.activateWindow()

    def _build_cheats_keyboard_rows(self):
        rows = []

        # 1) All QActions (you already have a collector; fall back if missing)
        try:
            actions = self._collect_all_qactions()
        except Exception:
            actions = self.findChildren(QAction)
        for act in actions:
            for seq in _seqs_for_action(act):
                rows.append((_qs_to_str(seq), _describe_action(act), _where_for_action(act)))

        # 2) Ad-hoc QShortcuts created in code
        for sc in self.findChildren(QShortcut):
            seq = sc.key()
            if seq and not seq.isEmpty():
                rows.append((_qs_to_str(seq), _describe_shortcut(sc), _where_for_shortcut(sc)))

        # De-duplicate and sort by shortcut text
        rows = _uniq_keep_order(rows)
        rows.sort(key=lambda r: (r[0].lower(), r[1].lower()))
        return rows

    def _build_cheats_gesture_rows(self):
        # Manual list (extend anytime). Format: (Gesture, Context, Effect)
        rows = [
            # Command search
            ("A", "Display Stretch", "Toggle Display Auto-Stretch"),
            ("Ctrl+I", "Invert", "Invert the Image"),
            ("Ctrl+Shift+P", "Command Search", "Focus the command search bar; Enter runs first match"),

            # View Icon
            ("Drag view -> Off to Canvas", "View", "Duplicate Image"),
            ("Drag view -> On to Other Image", "View", "Copy Zoom and Pan"),
            ("Shift+Drag -> On to Other Image", "View", "Apply that image to the other as a mask"), 
            ("Ctrl+Drag -> On to Other Image", "View", "Copy Astrometric Solution"),            

            # View zoom
            ("Ctrl+1", "View", "Zoom to 100% (1:1)"),
            ("Ctrl+0", "View", "Fit image to current window"),
            ("Ctrl++", "View", "Zoom In"),
            ("Ctrl+-", "View", "Zoom Out"),

            # Window switching
            ("Ctrl+PgDown", "MDI", "Switch to previously active view"),
            ("Ctrl+PgUp",   "MDI", "Switch to next active view"),

            # Shortcuts canvas + buttons
            ("Alt+Drag (toolbar button)", "Toolbar", "Create a desktop shortcut for that action"),
            ("Alt+Drag (shortcut button -> view)", "Shortcuts", "Headless apply the shortcut's command/preset to a view"),
            ("Ctrl/Shift+Click", "Shortcuts", "Multi-select shortcut buttons"),
            ("Drag (selection)", "Shortcuts", "Move selected shortcut buttons"),
            ("Delete / Backspace", "Shortcuts", "Delete selected shortcut buttons"),
            ("Ctrl+A", "Shortcuts", "Select all shortcut buttons"),
            ("Double-click empty area", "MDI background", "Open files dialog"),

            # Layers dock
            ("Drag view -> Layers list", "Layers", "Add dragged view as a new layer (on top)"),
            ("Shift+Drag mask -> Layers list", "Layers", "Attach dragged image as mask to the selected layer"),

            # Crop tool
            ("Click-drag", "Crop Tool", "Draw a crop rectangle"),
            ("Drag corner handles", "Crop Tool", "Resize crop rectangle"),
            ("Shift+Drag on box", "Crop Tool", "Rotate crop rectangle"),
        ]
        return rows

    def _show_cheat_sheet(self):
        kb = self._build_cheats_keyboard_rows()
        gs = self._build_cheats_gesture_rows()
        dlg = _CheatSheetDialog(self, kb, gs)
        dlg.exec()

    def _doc_by_ptr(self, ptr: int):
        dm = getattr(self, "doc_manager", None) or getattr(self, "docman", None)
        if dm and hasattr(dm, "all_documents"):
            for d in dm.all_documents():
                if id(d) == ptr:
                    return d
        return None

    def _close_all_subwindows(self):
        for sw in list(self.mdi.subWindowList()):
            try:
                sw.close()
            except Exception:
                pass

    def _clear_all_documents(self):
        dm = getattr(self, "doc_manager", None)
        if not dm:
            return
        # Make a copy because closing views may mutate _docs via signals
        for doc in list(dm._docs):
            try:
                self._safe_close_doc(doc)
            except Exception:
                # fallback: force drop
                try: dm.close_document(doc)
                except Exception as e:
                    import logging
                    logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")

    def _clear_minimized_shelf(self):
        try:
            if hasattr(self, "window_shelf") and self.window_shelf:
                self.window_shelf.clear_all()
        except Exception:
            pass

    def _confirm_discard(self, title="New Project", msg="This will close all views and clear desktop shortcuts. Continue?"):
        btn = QMessageBox.question(self, title, msg,
                                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                QMessageBox.StandardButton.No)
        return btn == QMessageBox.StandardButton.Yes

    def _clear_views_keep_shortcuts(self):
        if not self._confirm_discard(
            title="Clear All Views",
            msg="Close all views and documents? Desktop shortcuts will be preserved."
        ):
            return

        # Close views + docs + minimized shelf (same as _new_project)
        self._close_all_subwindows()
        self._clear_all_documents()
        self._clear_minimized_shelf()

        # DO NOT clear shortcuts or their persisted layout.
        # Just bring the canvas forward so users can keep working.
        try:
            if getattr(self, "shortcuts", None):
                self.shortcuts.canvas.raise_()
                self.shortcuts.canvas.show()
                self.shortcuts.canvas.setFocus()
        except Exception:
            pass

        self._log("Cleared all views (shortcuts preserved).")

    def _ask_project_compress(self) -> bool:
        """
        Returns True if the user wants compression.
        Respects a remembered preference in QSettings.
        """
        s = QSettings()
        has_pref = s.contains("projects/compress")
        if has_pref:
            # read as bool
            return s.value("projects/compress", True, type=bool)

        msg = QMessageBox(self)
        msg.setWindowTitle("Save Project")
        msg.setText("Compress project file for smaller size?\n\n"
                    "â€¢ Yes = smaller file, slower save\n"
                    "â€¢ No  = larger file, faster save")
        msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        msg.setDefaultButton(QMessageBox.StandardButton.Yes)

        remember = QCheckBox("Remember my choice")
        msg.setCheckBox(remember)

        choice = msg.exec()
        compress = (choice == QMessageBox.StandardButton.Yes)

        if remember.isChecked():
            s.setValue("projects/compress", compress)
            s.sync()

        return compress

    def _prepare_for_project_load(self, title: str = "Load Project") -> bool:
        """Confirm and clear current desktop before loading a project."""
        if getattr(self, "doc_manager", None) and self.doc_manager._docs:
            if not self._confirm_discard(
                title=title,
                msg=(
                    "Loading a project will close current views and replace desktop shortcuts.\n"
                    "Continue?"
                ),
            ):
                return False
            self._close_all_subwindows()
            self._clear_all_documents()
            self._clear_minimized_shelf()
            if getattr(self, "shortcuts", None):
                try:
                    self.shortcuts.canvas.raise_()
                    self.shortcuts.canvas.show()
                    self.shortcuts.canvas.setFocus()
                except Exception:
                    pass
        return True

    def _do_load_project_path(self, path: str):
        from pro.project_io import ProjectWriter, ProjectReader
        """Internal helper to actually load a .sas file."""
        if not path:
            return

        # ensure DocManager exists
        if not hasattr(self, "doc_manager") or self.doc_manager is None:
            from pro.doc_manager import DocManager
            self.doc_manager = DocManager(
                image_manager=getattr(self, "image_manager", None), parent=self
            )

        # progress ("thinking") dialog
        dlg = QProgressDialog("Loading project...", "", 0, 0, self)
        dlg.setWindowTitle("Loading")
        try:
            dlg.setWindowModality(Qt.WindowModality.ApplicationModal)
        except AttributeError:  # PyQt5 fallback
            dlg.setWindowModality(Qt.ApplicationModal)
        try:
            dlg.setCancelButton(None)
        except TypeError:
            dlg.setCancelButtonText("")
        dlg.setAutoClose(False)
        dlg.setAutoReset(False)
        dlg.show()

        try:
            ProjectReader(self).read(path)
            self._log("Project loaded.")
            self._add_recent_project(path)   # âœ... track in MRU
        except Exception as e:
            QMessageBox.critical(self, "Load Project", f"Failed to load:\n{e}")
        finally:
            dlg.close()


    def _show_mask_overlay(self):
        vw = self._active_view()
        if not vw:
            return
        # require an active mask on this doc
        doc = getattr(vw, "document", None)
        has_mask = bool(doc and getattr(doc, "active_mask_id", None))
        if not has_mask:
            QMessageBox.information(self, "Mask Overlay", "No active mask on this image.")
            return
        vw.show_mask_overlay = True
        # ensure visuals are up-to-date immediately
        try:
            vw._set_mask_highlight(True)
        except Exception:
            pass
        vw._render(rebuild=True)
        self._refresh_mask_action_states()

    def _hide_mask_overlay(self):
        vw = self._active_view()
        if not vw:
            return
        vw.show_mask_overlay = False
        vw._render(rebuild=True)
        self._refresh_mask_action_states()

    def _invert_mask(self):
        import numpy as np
        doc = self._active_doc()
        if not doc:
            return
        mid = getattr(doc, "active_mask_id", None)
        if not mid:
            return
        layer = (getattr(doc, "masks", {}) or {}).get(mid)
        if layer is None or getattr(layer, "data", None) is None:
            return

        m = np.asarray(layer.data)
        if m.size == 0:
            return

        # invert (preserve dtype)
        if m.dtype.kind in "ui":
            maxv = np.iinfo(m.dtype).max
            layer.data = (maxv - m).astype(m.dtype, copy=False)
        else:
            layer.data = (1.0 - m.astype(np.float32, copy=False)).clip(0.0, 1.0)

        # notify listeners (triggers ImageSubWindow.render via your existing hookup)
        if hasattr(doc, "changed"):
            doc.changed.emit()

        # and explicitly refresh the active view overlay right now
        vw = self._active_view()
        if vw and hasattr(vw, "refresh_mask_overlay"):
            vw.refresh_mask_overlay()

        # keep menu states tidy
        if hasattr(self, "_refresh_mask_action_states"):
            self._refresh_mask_action_states()


    def _open_settings(self):
        from ops.settings import SettingsDialog
        dlg = SettingsDialog(self, self.settings)
        if dlg.exec():
            # (Optional) react to changes if needed
            pass

    def graxpert_path(self) -> str:
        return self.settings.value("paths/graxpert", "", type=str)

    def cosmic_clarity_path(self) -> str:
        return self.settings.value("paths/cosmic_clarity", "", type=str)

    def starnet_path(self) -> str:
        return self.settings.value("paths/starnet", "", type=str)

    def _unwrap_history_doc(self, d):
        """
        Return the *real* document that owns the history stack for `d`.
        Unwraps proxies/ROI/preview wrappers when present.
        """
        seen = set()
        while d is not None and id(d) not in seen:
            seen.add(id(d))

            # Common wrappers: ROI/proxy sets _parent_doc or base_document
            for key in ("history_document", "get_history_document", "base_document", "_parent_doc"):
                try:
                    v = getattr(d, key, None)
                    if callable(v):
                        v = v()
                    if v is not None:
                        d = v
                        break
                except Exception:
                    pass
            else:
                # _DocProxy pattern: _target() returns current doc
                try:
                    tgt = getattr(d, "_target", None)
                    if callable(tgt):
                        t = tgt()
                        if t is not None and t is not d:
                            d = t
                            continue
                except Exception:
                    pass
                # Nothing else to unwrap
                break
        return d

    def _active_history_doc(self):
        dm = getattr(self, "doc_manager", None) or getattr(self, "docman", None)
        if not dm:
            return None

        # Prefer the currently active subwindow doc, then fall back to doc_manager's active doc.
        d = None
        try:
            sw = self.mdi.activeSubWindow()
            if sw is not None and sw.widget() is not None:
                d = getattr(sw.widget(), "document", None)
        except Exception:
            d = None
        if d is None:
            d = dm.get_active_document()

        return self._unwrap_history_doc(d)

    def _subwindow_for_history_doc(self, hist_doc):
        """
        Return the QMdiSubWindow showing a view whose base/history doc resolves to `hist_doc`.
        """
        if hist_doc is None:
            return None
        try:
            for sw in self.mdi.subWindowList():
                try:
                    vw = sw.widget()
                except RuntimeError:
                    continue
                if vw is None:
                    continue
                # Try explicit base link first
                base = getattr(vw, "base_document", None)
                if base is hist_doc:
                    return sw
                # Then unwrap the view's document to its history root
                vdoc = getattr(vw, "document", None)
                if vdoc is not None and self._unwrap_history_doc(vdoc) is hist_doc:
                    return sw
        except Exception:
            pass
        return None

    # ---------- actions ----------
    def _current_document(self):
        sw = self.mdi.activeSubWindow()
        if not sw:
            return None
        vw = sw.widget()
        return getattr(vw, "document", None)

    def _find_subwindow_for_doc(self, doc):
        """
        Return the QMdiSubWindow showing `doc`, if any.

        Matching rules (in order):
        1) view.base_document is `doc` (identity)
        2) view.document resolves (via _target if proxy) to `doc` (identity)
        We do NOT match by display_name or file_path to avoid aliasing duplicates.
        """
        try:
            for sw in self.mdi.subWindowList():
                # Be defensive about deleted wrappers
                try:
                    w = sw.widget()
                except RuntimeError:
                    continue
                if w is None:
                    continue

                # 1) Prefer explicit base handle installed by _spawn_subwindow_for
                base = getattr(w, "base_document", None)
                if base is doc:
                    return sw

                # 2) Fall back to the view's document (unwrap proxy if present)
                vdoc = getattr(w, "document", None)
                # If this is our _DocProxy, resolve to its current target
                try:
                    if hasattr(vdoc, "_target") and callable(vdoc._target):
                        vdoc = vdoc._target()
                except Exception:
                    pass

                if vdoc is doc:
                    return sw

            # No identity match
            return None
        except Exception:
            return None


    def _normalize_base_doc(self, doc):
        """If doc is an ROI/proxy, return its base/parent; else return doc."""
        return getattr(doc, "_parent_doc", None) or doc

    def _open_subwindow_for_added_doc(self, doc):
        """
        Called when DocManager emits documentAdded(doc).
        Avoid dupes, create a subwindow, connect close hook, and activate it.
        """
        base = self._normalize_base_doc(doc)

        # Avoid duplicate views if one already exists for this *base* doc
        sw_existing = self._find_subwindow_for_doc(base)
        if sw_existing:
            # still ensure the explorer row text is fresh
            try:
                self._update_explorer_item_for_doc(base)
            except Exception:
                pass
            QTimer.singleShot(0, lambda: self.mdi.setActiveSubWindow(sw_existing))
            return

        try:
            sw = self._spawn_subwindow_for(base)  # ensure you pass base
            if sw:
                w = sw.widget()
                # Wire the close hook once
                if hasattr(w, "aboutToClose"):
                    try:
                        # Avoid multiple connections if re-spawned somehow
                        w.aboutToClose.disconnect(self._on_view_about_to_close)
                    except Exception:
                        pass
                    w.aboutToClose.connect(self._on_view_about_to_close)

                # Activate on next tick
                QTimer.singleShot(0, lambda: self.mdi.setActiveSubWindow(sw))
        except Exception as e:
            # Safe fallback: show a very simple subwindow so user sees *something*
            try:
                if hasattr(self, "_log"):
                    self._log(f"Failed to open subwindow for {base.display_name()}: {e}")
            except Exception:
                pass
            from PyQt6.QtWidgets import QLabel, QMdiSubWindow
            w = QLabel(base.display_name()); setattr(w, "document", base)
            wrapper = QMdiSubWindow(self); wrapper.setWidget(w)
            self.mdi.addSubWindow(wrapper); wrapper.show()


    def _action_create_mask(self):
        from pro.mask_creation import create_mask_and_attach
        doc = self._current_document()
        if doc is None or getattr(doc, "image", None) is None:
            QMessageBox.information(self, "No image", "Open an image first.")
            return
        created = create_mask_and_attach(self, doc)
        # Optional toast/log
        if created and hasattr(self, "_log"):
            self._log("Mask created and set active.")

    def _format_explorer_title(self, doc) -> str:
        name = _strip_ui_decorations(doc.display_name() or "Untitled")

        dims = ""
        try:
            import numpy as np
            arr = getattr(doc, "image", None)
            if isinstance(arr, np.ndarray) and arr.size:
                h, w = arr.shape[:2]
                c = arr.shape[2] if arr.ndim == 3 else 1
                dims = f"  --  {h}x{w}x{c}"
        except Exception:
            pass

        return f"{name}{dims}"

    def _update_explorer_item_for_doc(self, doc):
        for i in range(self.explorer.count()):
            it = self.explorer.item(i)
            if it.data(Qt.ItemDataRole.UserRole) is doc:
                it.setText(self._format_explorer_title(doc))
                return
    #-----------FUNCTIONS----------------

    # --- WCS summary popup ----------------------------------------
    def _show_wcs_update_popup(self, debug_summary: dict, step_name: str):
        from pro.wcs_update import update_wcs_after_crop
        """
        Show a small WCS update summary dialog using the debug payload
        from update_wcs_after_crop (stored under '__wcs_debug__').
        """
        import math

        before = debug_summary.get("before", {}) or {}
        after  = debug_summary.get("after", {}) or {}
        fit    = debug_summary.get("fit", {}) or {}

        def _fmt_pair(p, fmt="{:.6f}"):
            if not isinstance(p, (list, tuple)) or len(p) != 2:
                return "n/a"
            a, b = p
            def _one(x):
                try:
                    if x is None or not math.isfinite(float(x)):
                        return "n/a"
                    return fmt.format(float(x))
                except Exception:
                    return "n/a"
            return f"({_one(a)}, {_one(b)})"

        def _fmt_one(x, fmt="{:.3f}"):
            try:
                if x is None or not math.isfinite(float(x)):
                    return "n/a"
                return fmt.format(float(x))
            except Exception:
                return "n/a"

        msg_lines = [
            f"{step_name}: WCS updated.",
            "",
            "BEFORE:",
            f"  CRVAL (deg):    { _fmt_pair(before.get('crval_deg')) }",
            f"  CRPIX (pix):    { _fmt_pair(before.get('crpix_pix'), fmt='{:.2f}') }",
            f"  Scale (as/px):  { _fmt_pair(before.get('scale_as_per_pix'), fmt='{:.3f}') }",
            f"  Rotation (deg): { _fmt_one(before.get('rot_deg'), fmt='{:.3f}') }",
            "",
            "AFTER:",
            f"  CRVAL (deg):    { _fmt_pair(after.get('crval_deg')) }",
            f"  CRPIX (pix):    { _fmt_pair(after.get('crpix_pix'), fmt='{:.2f}') }",
            f"  Scale (as/px):  { _fmt_pair(after.get('scale_as_per_pix'), fmt='{:.3f}') }",
            f"  Rotation (deg): { _fmt_one(after.get('rot_deg'), fmt='{:.3f}') }",
        ]

        size = after.get("size")
        if isinstance(size, (list, tuple)) and len(size) == 2:
            msg_lines.append(f"  Image size:     {int(size[0])} Ã-- {int(size[1])}")

        rms  = _fmt_one(fit.get("rms_arcsec"), fmt="{:.3f}")
        p50  = _fmt_one(fit.get("p50_arcsec"), fmt="{:.3f}")
        p95  = _fmt_one(fit.get("p95_arcsec"), fmt="{:.3f}")
        msg_lines += [
            "",
            "Fit residuals (arcsec):",
            f"  RMS: {rms}   median: {p50}   p95: {p95}",
        ]

        coerced = debug_summary.get("coerced_to_2d", None)
        if coerced is True:
            msg_lines.append("")
            msg_lines.append("Note: WCS was coerced from 3-D to 2-D for refitting.")

        text = "\n".join(msg_lines)
        QMessageBox.information(self, "WCS Updated", text)

    def _bake_display_stretch(self):
        """Apply the current Display-Stretch to the image data (undoable, non-replayable)."""
        sw = self.mdi.activeSubWindow()
        if not sw:
            QMessageBox.information(self, "Display-Stretch", "No active image window.")
            return

        view = sw.widget()
        doc = getattr(view, "document", None)
        if doc is None or getattr(doc, "image", None) is None:
            QMessageBox.information(self, "Display-Stretch", "Active window has no image.")
            return

        img = getattr(doc, "image", None)
        a = np.asarray(img)
        if a.size == 0:
            QMessageBox.information(self, "Display-Stretch", "Image is empty.")
            return

        # --- Get the *current* display-stretch parameters ---
        # start from global defaults
        target = float(self.settings.value("display/target", 0.30, type=float))
        sigma  = float(self.settings.value("display/sigma", 5.0, type=float))
        linked = bool(self.settings.value("display/stretch_linked", False, type=bool))
        use_16 = self.settings.value("display/autostretch_16bit", True, type=bool)

        # if your view exposes per-view overrides, prefer those
        if hasattr(view, "autostretch_target"):
            try:
                target = float(view.autostretch_target)
            except Exception:
                pass
        if hasattr(view, "autostretch_sigma"):
            try:
                sigma = float(view.autostretch_sigma)
            except Exception:
                pass
        if hasattr(view, "stretch_linked"):
            try:
                linked = bool(view.stretch_linked)
            except Exception:
                pass

        # --- Run the same autostretch math used for display ---
        try:
            stretched01 = _autostretch(
                a,
                target_median=target,
                linked=linked,
                sigma=sigma,
                use_16bit=use_16,
            )
        except Exception as e:
            QMessageBox.warning(self, "Display-Stretch", f"Failed to apply autostretch:\n{e}")
            return

        # --- Convert back to original dtype ---
        if np.issubdtype(a.dtype, np.integer):
            info = np.iinfo(a.dtype)
            out = (np.clip(stretched01, 0.0, 1.0) * float(info.max)).astype(a.dtype, copy=False)
        else:
            # float images: bake 0-1 stretched data into same float dtype
            out = np.clip(stretched01, 0.0, 1.0).astype(a.dtype, copy=False)

        # --- Commit to document with undo metadata (no command_id -> non-replayable) ---
        meta = {
            "step_name": "Display-Stretch (baked)",
            "autostretch_target": float(target),
            "autostretch_sigma": float(sigma),
            "autostretch_linked": bool(linked),
        }

        try:
            if hasattr(doc, "set_image"):
                # your Document.set_image already manages undo/redo
                doc.set_image(out, meta)
            elif hasattr(doc, "update_image"):
                doc.update_image(out, meta)
            else:
                # last-resort fallback (no undo)
                doc.image = out
        except Exception as e:
            QMessageBox.critical(self, "Display-Stretch", f"Failed to update image:\n{e}")
            return

        # Turn OFF display-stretch so the baked image looks exactly like the preview did
        if hasattr(view, "set_autostretch"):
            view.set_autostretch(False)
        self._sync_autostretch_action(False)

        try:
            self._log(
                f"Display-Stretch baked into image (target={target:.3f}, "
                f"sigma={sigma:.2f}, linked={'on' if linked else 'off'}) "
                f"-> {sw.windowTitle()}"
            )
        except Exception:
            pass


    def _open_histogram(self):
        from pro.histogram import HistogramDialog
        sw = self.mdi.activeSubWindow()
        if not sw:
            QMessageBox.information(self, "Histogram", "No active image window.")
            return

        doc = sw.widget().document

        # make sure we have a place to hold dialogs
        if not hasattr(self, "_open_histograms"):
            self._open_histograms = []

        dlg = HistogramDialog(self, doc)
        dlg.setWindowTitle(f"Histogram -- {sw.windowTitle()}")
        try:
            dlg.setWindowIcon(QIcon(histogram_path))
        except Exception:
            pass

        # ðŸ'‡ this is the key: stay on top
        dlg.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)

        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

        # keep it alive
        self._open_histograms.append(dlg)

        # optional: prune on close
        dlg.finished.connect(lambda _: self._open_histograms.remove(dlg))

        if hasattr(self, "_log"):
            self._log(f"Opened Histogram for {doc.display_name()}")


    def _open_crop_dialog(self):
        from pro.crop_dialog_pro import CropDialogPro
        sw = self.mdi.activeSubWindow()
        if not sw:
            QMessageBox.information(self, "Crop", "No active image window.")
            return
        doc = sw.widget().document
        if doc is None or getattr(doc, "image", None) is None:
            QMessageBox.information(self, "Crop", "Active document has no image.")
            return
        dlg = CropDialogPro(self, doc)
        try:
            dlg.setWindowIcon(QIcon(cropicon_path))
        except Exception:
            pass

        dlg.crop_applied.connect(lambda *_: QTimer.singleShot(0, self._zoom_active_fit))
        dlg.show()

    def _open_statistical_stretch(self):
        from pro.stat_stretch import StatisticalStretchDialog
        sw = self.mdi.activeSubWindow()
        if not sw:
            QMessageBox.information(self, "No image", "Open an image first.")
            return
        view = sw.widget()
        # ROI-aware: always resolve via DocManager for THIS view
        doc = self.doc_manager.get_document_for_view(view)

        dlg = StatisticalStretchDialog(self, doc)
        try:
            dlg.setWindowIcon(QIcon(statstretch_path))
        except Exception:
            pass
        dlg.resize(900, 600)
        dlg.show()


    def _open_star_stretch(self):
        from pro.star_stretch import StarStretchDialog
        sw = self.mdi.activeSubWindow()
        if not sw:
            QMessageBox.information(self, "No image", "Open an image first.")
            return
        doc = sw.widget().document
        dlg = StarStretchDialog(self, doc)
        try:
            dlg.setWindowIcon(QIcon(starstretch_path))
        except Exception:
            pass
        dlg.resize(1000, 650)
        dlg.show()
        self._log("Functions: opened Star Stretch.")

    def _open_curves_editor(self):
        from pro.curve_editor_pro import CurvesDialogPro
        sw = self.mdi.activeSubWindow()
        if not sw:
            QMessageBox.information(self, "No image", "Open an image first.")
            return
        doc = sw.widget().document

        dlg = CurvesDialogPro(self, doc)
        try:
            dlg.setWindowIcon(QIcon(curves_path))
        except Exception:
            pass        
        dlg.resize(1000, 650)
        dlg.show()   # non-modal; you can open one per subwindow

    def _open_hyperbolic(self):
        from pro.ghs_dialog_pro import GhsDialogPro
        sw = self.mdi.activeSubWindow()
        if not sw:
            QMessageBox.information(self, "No image", "Open an image first.")
            return
        doc = sw.widget().document
        dlg = GhsDialogPro(self, doc)  # class below
        try:
            dlg.setWindowIcon(QIcon(uhs_path))
        except Exception:
            pass        
        dlg.resize(1000, 650)
        dlg.show()

    def _remove_stars(self, doc=None):
        from pro.remove_stars import remove_stars
        """
        Wrapper so both the menu and Replay Last Action can call star removal
        on a specific document (ROI, base, etc.).
        """
        # If replay passed a specific doc, use it.
        if doc is None:
            sw = self.mdi.activeSubWindow()
            if not sw:
                QMessageBox.information(self, "No image", "Open an image first.")
                return
            doc = sw.widget().document

        remove_stars(self, doc)

    def _add_stars(self, doc=None):
        from pro.add_stars import add_stars
        """
        Wrapper so both the menu and Replay Last Action can call add_stars
        on a specific document (ROI, base, etc.).
        """
        # If replay passed a specific doc, use it.
        if doc is None:
            sw = self.mdi.activeSubWindow()
            if not sw:
                QMessageBox.information(self, "No image", "Open an image first.")
                return

        add_stars(self)

    def _open_graxpert(self):
        from pro.graxpert import remove_gradient_with_graxpert
        """Open GraXpert for the active document (same style as Star Stretch)."""
        sw = self.mdi.activeSubWindow()
        if not sw:
            QMessageBox.information(self, "GraXpert", "Open an image first.")
            return

        view = sw.widget()
        doc = getattr(view, "document", None)
        if doc is None or getattr(doc, "image", None) is None:
            QMessageBox.information(self, "GraXpert", "Active document has no image.")
            return

        # Let pro.graxpert handle the UI + progress and apply_edit back to this doc
        remove_gradient_with_graxpert(self, target_doc=doc)

        try:
            self._log("Functions: ran GraXpert on active document.")
        except Exception:
            pass

    def _open_abe_tool(self):        
        from pro.abe import ABEDialog
        sw = self.mdi.activeSubWindow()
        if not sw:
            QMessageBox.information(self, "No image", "Open an image first.")
            return
        doc = sw.widget().document
        dlg = ABEDialog(self, doc)
        try:
            dlg.setWindowIcon(QIcon(abeicon_path))
        except Exception:
            pass  
        dlg.resize(980, 620)
        dlg.show()

    def _open_background_neutral(self):
        from pro.backgroundneutral import BackgroundNeutralizationDialog
        sw = self.mdi.activeSubWindow()
        if not sw:
            QMessageBox.information(self, "No image", "Open an image first.")
            return
        doc = sw.widget().document
        try:

            dlg = BackgroundNeutralizationDialog(self, doc, icon=QIcon(neutral_path))
        except Exception as e:
            QMessageBox.warning(self, "Background Neutralization", f"Failed to open dialog:\n{e}")
            return
        dlg.resize(900, 600)
        dlg.show()

    def _apply_background_neutral_preset_to_doc(self, doc, preset: dict):
        from pro.backgroundneutral import apply_background_neutral_to_doc
        apply_background_neutral_to_doc(doc, preset or {"mode": "auto"})

    def _open_white_balance(self):
        sw = self.mdi.activeSubWindow()
        if not sw:
            QMessageBox.information(self, "No image", "Open an image first.")
            return
        doc = sw.widget().document
        try:
            from pro.whitebalance import WhiteBalanceDialog
            dlg = WhiteBalanceDialog(self, doc, icon=QIcon(whitebalance_path))
            dlg.show()
        except Exception as e:
            QMessageBox.warning(self, "White Balance", f"Failed to open dialog:\n{e}")

    def _apply_white_balance_preset_to_doc(self, doc, preset: dict | None):
        from pro.whitebalance import apply_white_balance_to_doc
        apply_white_balance_to_doc(doc, preset or {"mode": "star", "threshold": 50.0})

    def _apply_convo_preset_to_doc(self, doc, preset: dict):
        """
        Headless apply of Convo/Deconvo/TV to a specific document (ROI or base),
        and record it as the last headless command for Replay Last Action.
        """
        from pro.convo_preset import apply_convo_via_preset

        if doc is None or getattr(doc, "image", None) is None:
            return

        # Actually apply
        apply_convo_via_preset(self, doc, preset or {})

        # Record for replay-last-action
        try:
            op = (preset or {}).get("op", "convolution")
            self._last_headless_command = {
                "cid": "convo",
                "preset": dict(preset or {}),
            }
            if hasattr(self, "_log"):
                name = doc.display_name() if hasattr(doc, "display_name") else "Image"
                self._log(f"Convo/Deconvo preset ({op}) applied to '{name}'")
        except Exception:
            pass


    def _open_remove_green(self, doc=None):
        """
        Open Remove Green (SCNR) for the current view's document.
        If doc is passed (e.g. from Replay), use that; otherwise use the
        active subwindow's document (ROI or full).
        """
        from pro.remove_green import RemoveGreenDialog

        sw = self.mdi.activeSubWindow()
        if not sw:
            QMessageBox.information(self, "No image", "Open an image first.")
            return
        doc = sw.widget().document

        try:
            dlg = RemoveGreenDialog(self, doc, parent=self)
            dlg.show()
        except Exception as e:
            QMessageBox.warning(self, "Remove Green", f"Failed to open dialog:\n{e}")


    def SFCC_show(self):
        from pro.sfcc import SFCCDialog
        from pro.doc_manager import DocManager
        if getattr(self, "SFCC_window", None) and self.SFCC_window.isVisible():
            self.SFCC_window.raise_()
            self.SFCC_window.activateWindow()
            return

        # ensure we have a DocManager (if you already create it, keep yours)
        if not hasattr(self, "doc_manager") or self.doc_manager is None:
            self.doc_manager = DocManager(image_manager=getattr(self, "image_manager", None), parent=self)

        self.SFCC_window = SFCCDialog(
            doc_manager=self.doc_manager,
            sasp_data_path=sasp_data_path,
            parent=self
        )
        try:
            self.SFCC_window.setWindowIcon(QIcon(spcc_icon_path))
        except Exception:
            pass

        try:
            self.SFCC_window.destroyed.connect(lambda _=None: setattr(self, "SFCC_window", None))
        except Exception:
            pass
        self.SFCC_window.show()

    def show_convo_deconvo(self, doc=None):
        # Reuse existing dialog if it's already open
        sw = self.mdi.activeSubWindow()
        if not sw:
            QMessageBox.information(self, "No image", "Open an image first.")
            return
        doc = sw.widget().document

        from pro.convo import ConvoDeconvoDialog
        self.convo_window = ConvoDeconvoDialog(
            doc_manager=self.doc_manager,
            parent=self,
            doc=doc,  # <- KEY: bind dialog to this Document instance
        )
        try:
            self.convo_window.setWindowIcon(QIcon(convoicon_path))
        except Exception:
            pass
        try:
            self.convo_window.destroyed.connect(
                lambda _=None: setattr(self, "convo_window", None)
            )
        except Exception:
            pass

        self.convo_window.show()

    

    def _apply_extract_luminance_preset_to_doc(self, doc, preset=None):

        from PyQt6.QtWidgets import QMessageBox
        from pro.luminancerecombine import compute_luminance, _LUMA_REC709, _LUMA_REC601, _LUMA_REC2020
        from pro.headless_utils import normalize_headless_main, unwrap_docproxy

        doc = unwrap_docproxy(doc)
        p = dict(preset or {})
        mode = (p.get("mode") or "rec709").lower()

        if doc is None or getattr(doc, "image", None) is None:
            QMessageBox.information(self, "Extract Luminance", "No target image.")
            return

        img = np.asarray(doc.image)

        # pick weights
        if mode == "rec601":
            w = _LUMA_REC601
        elif mode == "rec2020":
            w = _LUMA_REC2020
        elif mode == "max":
            w = None
        else:
            w = _LUMA_REC709

        L = compute_luminance(img, method=mode, weights=w)

        dm = getattr(self, "doc_manager", None)
        if dm is None:
            # headless fallback: just overwrite active doc
            doc.apply_edit(L.astype(np.float32), step_name="Extract Luminance")
            return

        # normal behavior: create a new mono document
        try:
            new_doc = dm.create_document_from_array(
                L.astype(np.float32),
                name=f"{doc.display_name()} -- Luminance ({mode})",
                is_mono=True,
                metadata={"step_name":"Extract Luminance", "luma_method":mode}
            )
            dm.add_document(new_doc)
        except Exception:
            # safe fallback
            doc.apply_edit(L.astype(np.float32), step_name="Extract Luminance")


    def _extract_luminance(self, doc=None, preset: dict | None = None):
        from pro.luminancerecombine import _LUMA_REC709, _LUMA_REC601, _LUMA_REC2020
        """
        If doc is None, uses the active subwindow's document.
        Otherwise, run on the provided doc (for drag-and-drop to a specific view).
        Creates a new mono document (float32, [0..1]) and spawns a subwindow.

        Preset schema:
        {
            "mode": "rec709" | "rec601" | "rec2020" | "max" | "snr" | "equal" | "median",
            # aliases accepted: method, luma_method, nb_max -> "max", snr_unequal -> "snr"
        }
        """
        # 1) resolve source document
        sw = None
        if doc is None:
            sw = self.mdi.activeSubWindow()
            if not sw:
                QMessageBox.information(self, "Extract Luminance", "No active image window.")
                return
            vw = sw.widget()
            doc = getattr(vw, "document", None)

        if doc is None or getattr(doc, "image", None) is None:
            QMessageBox.information(self, "Extract Luminance", "Active document has no image.")
            return

        img = np.asarray(doc.image)
        if img.ndim != 3 or img.shape[2] != 3:
            QMessageBox.information(self, "Extract Luminance", "Luminance extraction requires an RGB image.")
            return

        # 2) normalize to [0,1] float32
        a = img.astype(np.float32, copy=False)
        if a.size:
            m = float(np.nanmax(a))
            if np.isfinite(m) and m > 1.0:
                a = a / m
        a = np.clip(a, 0.0, 1.0)

        # 3) choose luminance method
        p = dict(preset or {})
        method = str(
            p.get("mode",
            p.get("method",
            p.get("luma_method",
                getattr(self, "luma_method", "rec709"))))
        ).strip().lower()

        # aliases
        alias = {
            "rec.709": "rec709",
            "rec-709": "rec709",
            "rgb": "rec709",
            "k": "rec709",
            "rec.601": "rec601",
            "rec-601": "rec601",
            "rec.2020": "rec2020",
            "rec-2020": "rec2020",
            "nb_max": "max",
            "narrowband": "max",
            "snr_unequal": "snr",
            "unequal_noise": "snr",
        }
        method = alias.get(method, method)

        # 4) compute luminance per selected method
        luma_weights = None
        if method == "rec601":
            luma_weights = _LUMA_REC601
            y = np.tensordot(a, _LUMA_REC601, axes=([2],[0]))
        elif method == "rec2020":
            luma_weights = _LUMA_REC2020
            y = np.tensordot(a, _LUMA_REC2020, axes=([2],[0]))
        elif method == "max":
            y = a.max(axis=2)
        elif method == "median":
            y = np.median(a, axis=2)
        elif method == "equal":
            luma_weights = np.array([1/3, 1/3, 1/3], dtype=np.float32)
            y = a.mean(axis=2)
        elif method == "snr":
            from pro.luminancerecombine import _estimate_noise_sigma_per_channel
            sigma = _estimate_noise_sigma_per_channel(a)
            w = 1.0 / (sigma[:3]**2 + 1e-12)
            w = w / w.sum()
            luma_weights = w.astype(np.float32)
            y = np.tensordot(a[..., :3], luma_weights, axes=([2],[0]))
        else:  # "rec709" default
            method = "rec709"
            luma_weights = _LUMA_REC709
            y = np.tensordot(a, _LUMA_REC709, axes=([2],[0]))

        y = np.clip(y.astype(np.float32, copy=False), 0.0, 1.0)

        # 5) metadata & title
        base_meta = {}
        try:
            base_meta = dict(getattr(doc, "metadata", {}) or {})
        except Exception:
            pass

        meta = {
            **base_meta,
            "source": "ExtractLuminance",
            "is_mono": True,
            "bit_depth": "32f",
            "luma_method": method,
        }
        if luma_weights is not None:
            meta["luma_weights"] = np.asarray(luma_weights, dtype=np.float32).tolist()

        base_title = sw.windowTitle() if sw else (getattr(doc, "title", getattr(doc, "name", "")) or "Untitled")
        title = f"{base_title} -- Luminance"

        dm = getattr(self, "docman", None)
        if dm is None:
            QMessageBox.critical(self, "Extract Luminance", "DocManager not available.")
            return

        try:
            if hasattr(dm, "open_array"):
                new_doc = dm.open_array(y, metadata=meta, title=title)
            elif hasattr(dm, "open_numpy"):
                new_doc = dm.open_numpy(y, metadata=meta, title=title)
            elif hasattr(dm, "create_document"):
                new_doc = dm.create_document(image=y, metadata=meta, name=title)
            else:
                raise RuntimeError("DocManager lacks open_array/open_numpy/create_document")
        except Exception as e:
            QMessageBox.critical(self, "Extract Luminance", f"Failed to create document:\n{e}")
            return

        try:
            self._spawn_subwindow_for(new_doc)
            sub = self.mdi.activeSubWindow()
            if sub:
                sub.setWindowIcon(QIcon(LExtract_path))
        except Exception:
            pass

        # ðŸ" Remember for Replay (optional but consistent)
        try:
            remember = getattr(self, "remember_last_headless_command", None) or getattr(self, "_remember_last_headless_command", None)
            if callable(remember):
                remember("extract_luminance", {"mode": method}, description="Extract Luminance")
        except Exception:
            pass

        if hasattr(self, "_log"):
            self._log(f"Extract Luminance ({method}) -> new mono document created.")

        return new_doc

    def _subwindow_docs(self):
        docs = []
        for sw in self.mdi.subWindowList():
            w = sw.widget()
            d = getattr(w, "document", None)
            if d is not None and getattr(d, "image", None) is not None:
                docs.append((sw.windowTitle(), d))
        return docs

    def _recombine_luminance_ui(self, target_doc=None):
        """If target_doc is None, use active; then pick a luminance source and overwrite target."""
        # resolve target
        if target_doc is None:
            sw = self.mdi.activeSubWindow()
            if not sw:
                QMessageBox.information(self, "Recombine Luminance", "No active image window.")
                return
            target_doc = getattr(sw.widget(), "document", None)
            if target_doc is None or getattr(target_doc, "image", None) is None:
                QMessageBox.information(self, "Recombine Luminance", "Active window has no image.")
                return

        tgt_img = np.asarray(target_doc.image)
        if tgt_img.ndim != 3 or tgt_img.shape[2] != 3:
            QMessageBox.warning(self, "Recombine Luminance", "Target image must be RGB.")
            return

        # gather candidates (all other open docs)
        candidates = []
        for title, d in self._subwindow_docs():
            if d is target_doc:
                continue
            img = getattr(d, "image", None)
            if img is None:
                continue
            # accept mono directly, or RGB (we'll extract Y')
            if img.ndim == 2 or (img.ndim == 3 and img.shape[2] in (1,3)):
                candidates.append((title, d))

        if not candidates:
            QMessageBox.information(self, "Recombine Luminance", "Open a luminance (mono) view or any image to use as L.")
            return

        # auto-pick if only one
        if len(candidates) == 1:
            sel_title, src_doc = candidates[0]
        else:
            titles = [t for (t, _) in candidates]
            item, ok = QInputDialog.getItem(self, "Select Luminance Source",
                                            "Choose the image to use as luminance (mono accepted; RGB will be converted):",
                                            titles, 0, False)
            if not ok:
                return
            idx = titles.index(item)
            sel_title, src_doc = candidates[idx]

        try:
            from pro.luminancerecombine import _to_float01_strict, _LUMA_REC601, _LUMA_REC2020
            src_img = _to_float01_strict(np.asarray(src_doc.image))

            # Prefer the source doc's stored method/weights (for perfect round-trip),
            # otherwise fall back to current menu selection.
            meta = dict(getattr(src_doc, "metadata", {}) or {})
            method = meta.get("luma_method", getattr(self, "luma_method", "rec709"))
            weights = None
            noise_sigma = None

            if "luma_weights" in meta:
                lw = np.asarray(meta["luma_weights"], dtype=np.float32)
                if lw.size == 3:
                    weights = lw
            else:
                if method == "rec601":
                    weights = _LUMA_REC601
                elif method == "rec2020":
                    weights = _LUMA_REC2020
                elif method == "snr":
                    noise_sigma = None  # will estimate inside apply if src_img is RGB

            # Optional knobs: you can expose these in a small dialog later
            blend = 1.0       # exact replace
            soft_knee = 0.0   # no highlight protection by default
            from pro.luminancerecombine import apply_recombine_to_doc
            apply_recombine_to_doc(
                target_doc,
                luminance_source_img=src_img,
                method=method,
                weights=weights,
                noise_sigma=noise_sigma,
                blend=blend,
                soft_knee=soft_knee
            )

            try:
                self._log(f"Recombine Luminance: '{sel_title}' -> '{target_doc.display_name()}' [{method}]")
            except Exception:
                pass

        except Exception as e:
            QMessageBox.critical(self, "Recombine Luminance", f"Failed: {e}")

    def _rgb_extract_active(self):
        sw = self.mdi.activeSubWindow()
        if not sw:
            QMessageBox.information(self, "RGB Extract", "No active image window.")
            return
        view = sw.widget()
        doc = getattr(view, "document", None)
        if doc is None or getattr(doc, "image", None) is None:
            QMessageBox.information(self, "RGB Extract", "Active document has no image.")
            return
        self._rgb_extract_on_doc(doc, base_title=sw.windowTitle())

    def _rgb_extract_on_doc(self, doc, base_title: str | None = None):
        img = getattr(doc, "image", None)
        if img is None:
            QMessageBox.information(self, "RGB Extract", "No image to extract.")
            return
        if img.ndim != 3 or img.shape[2] != 3:
            QMessageBox.information(self, "RGB Extract", "Image is not a 3-channel RGB image.")
            return

        try:
            r, g, b = extract_rgb_channels(img)
        except Exception as e:
            QMessageBox.critical(self, "RGB Extract", f"Failed to split channels:\n{e}")
            return

        dm = getattr(self, "docman", None)
        if not dm:
            QMessageBox.critical(self, "RGB Extract", "Document manager not available.")
            return

        # derive base name for the three windows
        base = base_title or (getattr(doc, "display_name", lambda: None)() or "RGB")

        def _open(arr, suffix):
            meta = {
                "source": "RGB Extract",
                "is_mono": True,
                "bit_depth": "32-bit floating point",
                "parent_title": base,
            }
            title = f"{base}_{suffix}"
            try:
                if hasattr(dm, "open_array"):
                    newdoc = dm.open_array(arr, metadata=meta, title=title)
                elif hasattr(dm, "open_numpy"):
                    newdoc = dm.open_numpy(arr, metadata=meta, title=title)
                else:
                    newdoc = dm.create_document(image=arr, metadata=meta, name=title)
                self._spawn_subwindow_for(newdoc)
            except Exception as ex:
                QMessageBox.critical(self, "RGB Extract", f"Failed to open '{title}':\n{ex}")

        _open(r, "R")
        _open(g, "G")
        _open(b, "B")

        # optional log
        if hasattr(self, "_log"):
            self._log(f"RGB Extract -> created '{base}_R', '{base}_G', '{base}_B'")

    def _list_open_docs_for_rgb(self):
        items = []
        for sw in self.mdi.subWindowList():
            w = sw.widget()
            doc = getattr(w, "document", None)
            if doc is not None:
                items.append((sw.windowTitle(), doc))
        return items

    def _open_rgb_combination(self):
        from pro.rgb_combination import RGBCombinationDialogPro
        dlg = RGBCombinationDialogPro(
            parent=self,
            list_open_docs_fn=self._list_open_docs_for_rgb,
            doc_manager=getattr(self, "docman", None)
        )
        try:
            dlg.setWindowIcon(QIcon(rgbcombo_path))
        except Exception:
            pass
        dlg.resize(600, 360)
        dlg.show()

    def _open_blemish_blaster(self):
        from pro.blemish_blaster import BlemishBlasterDialogPro
        sw = self.mdi.activeSubWindow()
        if not sw:
            QMessageBox.information(self, "Blemish Blaster", "No active image window.")
            return
        view = sw.widget()
        doc  = getattr(view, "document", None)
        if doc is None or getattr(doc, "image", None) is None:
            QMessageBox.information(self, "Blemish Blaster", "Active document has no image.")
            return
        dlg = BlemishBlasterDialogPro(self, doc)
        try:
            dlg.setWindowIcon(QIcon(blastericon_path))
        except Exception:
            pass
        dlg.resize(900, 650)
        dlg.show()

    def _open_wavescale_hdr(self):
        from pro.wavescale_hdr import WaveScaleHDRDialogPro
        sw = self.mdi.activeSubWindow()
        if not sw:
            QMessageBox.information(self, "WaveScale HDR", "No active image window.")
            return
        view = sw.widget()
        doc = getattr(view, "document", None)
        if doc is None or getattr(doc, "image", None) is None:
            QMessageBox.information(self, "WaveScale HDR", "Active document has no image.")
            return

        dlg = WaveScaleHDRDialogPro(self, doc, icon_path=hdr_path)

        # -- NEW: capture preset for replay when user clicks Apply ------
        def _on_applied(doc_obj, preset: dict):
            try:
                # Whatever helper you used for Curves / GHS:
                # cid is what command-drop & replay will call.
                self._register_replay_action(
                    cid="wavescale_hdr",
                    label="WaveScale HDR",
                    preset=dict(preset or {}),
                    target_doc=doc_obj,
                )
            except Exception:
                pass

        try:
            dlg.applied_preset.connect(_on_applied)
        except Exception:
            pass
        # ----------------------------------------

        dlg.resize(980, 700)
        dlg.show()


    def _open_wavescale_dark_enhance(self):
        sw = self.mdi.activeSubWindow()
        if not sw:
            QMessageBox.information(self, "WaveScale Dark Enhancer", "No active image window.")
            return
        view = sw.widget()
        doc  = getattr(view, "document", None)
        if doc is None or getattr(doc, "image", None) is None:
            QMessageBox.information(self, "WaveScale Dark Enhancer", "Active document has no image.")
            return

        try:
            # Prefer the Pro dialog name; fall back to the non-Pro name if that's what you used.
            from pro.wavescalede import WaveScaleDarkEnhancerDialogPro as _Dlg
        except Exception:
            try:
                from pro.wavescalede import WaveScaleDarkEnhanceDialog as _Dlg
            except Exception as e:
                QMessageBox.warning(self, "WaveScale Dark Enhancer", f"Failed to import dialog:\n{e}")
                return

        try:
            dlg = _Dlg(self, doc, icon_path=dse_icon_path)  # matches our Pro dialogs' __init__(parent, doc, icon_path)
        except TypeError:
            # if your ctor is (image_manager,parent) like the SASv2 snippet:
            dlg = _Dlg(image_manager=getattr(self, "image_manager", None), parent=self)
        try:
            dlg.setWindowIcon(QIcon(dse_icon_path))
        except Exception:
            pass
        dlg.resize(900, 650)
        dlg.show()

    def _open_clahe(self):
        sw = self.mdi.activeSubWindow()
        if not sw:
            QMessageBox.information(self, "CLAHE", "Open an image first.")
            return
        view = sw.widget()
        doc = getattr(view, "document", None)
        if doc is None or getattr(doc, "image", None) is None:
            QMessageBox.information(self, "CLAHE", "Active document has no image.")
            return
        try:
            from pro.clahe import CLAHEDialogPro
            dlg = CLAHEDialogPro(self, doc, icon=QIcon(clahe_path))
            dlg.resize(900, 650)
            dlg.show()
        except Exception as e:
            QMessageBox.warning(self, "CLAHE", f"Failed to open dialog:\n{e}")

    def _apply_clahe_preset_to_doc(self, doc, preset: dict | None):
        """
        Headless CLAHE apply on a document using a preset dict.
        Expected keys: clip_limit (float, e.g. 2.0), tile (int, e.g. 8)
        """
        try:
            from pro.clahe import apply_clahe_to_doc
            p = dict(preset or {"clip_limit": 2.0, "tile": 8})
            apply_clahe_to_doc(doc, p)

            # -- also register as last_headless_command for replay ------
            try:
                payload = {
                    "command_id": "clahe",
                    "preset": dict(p),
                }
                setattr(self, "_last_headless_command", payload)
            except Exception:
                pass
            # ----------------------------------------

        except Exception as e:
            raise RuntimeError(f"CLAHE apply failed: {e}")

    def _apply_morphology_preset_to_doc(self, doc, preset: dict | None):
        """
        Headless Morphology apply on a document using a preset dict.
        Expected keys:
          â€¢ operation: "erosion" | "dilation" | "opening" | "closing"
          â€¢ kernel: odd int (3,5,7,...)
          â€¢ iterations: int
        """
        try:
            from pro.morphology import apply_morphology_to_doc
            p = dict(preset or {})
            apply_morphology_to_doc(doc, p)

            # -- also register as last_headless_command for replay ------
            try:
                payload = {
                    "command_id": "morphology",
                    "preset": dict(p),
                }
                setattr(self, "_last_headless_command", payload)
            except Exception:
                pass
            # ----------------------------------------

        except Exception as e:
            raise RuntimeError(f"Morphology apply failed: {e}")


    def _open_morphology(self, preset: dict | None = None):
        sw = self.mdi.activeSubWindow()
        if not sw:
            QMessageBox.information(self, "Morphology", "No active image window.")
            return
        doc = getattr(sw.widget(), "document", None)
        if doc is None or getattr(doc, "image", None) is None:
            QMessageBox.information(self, "Morphology", "Active document has no image.")
            return
        from pro.morphology import MorphologyDialogPro
        dlg = MorphologyDialogPro(self, doc, icon=QIcon(morpho_path), initial=preset or {})
        dlg.resize(900, 600)
        dlg.show()

    def _open_morphology_with_preset(self, preset: dict | None):
        self._open_morphology(preset or {})

    def _apply_pixelmath_preset_to_doc(self, doc, preset: dict | None):
        """
        Headless Pixel Math apply on a document using a preset dict.

        Preset fields:
          â€¢ mode: 'single' or 'rgb' (optional, informational)
          â€¢ expr:   single-expression mode (string)
          â€¢ expr_r / expr_g / expr_b: per-channel expressions (strings)
        """
        from pro.pixelmath import apply_pixel_math_to_doc

        p = dict(preset or {})
        apply_pixel_math_to_doc(self, doc, p)

        # Also register as last_headless_command so replay uses this
        try:
            payload = {
                "command_id": "pixel_math",
                "preset": dict(p),
            }
            setattr(self, "_last_headless_command", payload)
        except Exception:
            pass


    def _open_pixel_math(self):
        sw = self.mdi.activeSubWindow()
        if not sw:
            QMessageBox.information(self, "Pixel Math", "No active image window.")
            return
        doc = getattr(sw.widget(), "document", None)
        if doc is None or getattr(doc, "image", None) is None:
            QMessageBox.information(self, "Pixel Math", "Active document has no image.")
            return
        from pro.pixelmath import PixelMathDialogPro
        dlg = PixelMathDialogPro(self, doc, icon=QIcon(pixelmath_path))
        dlg.resize(820, 560)
        dlg.show()

    def _open_signature_insert(self):
        sw = self.mdi.activeSubWindow()
        if not sw:
            QMessageBox.information(self, "Signature / Insert", "No active image window."); return
        doc = getattr(sw.widget(), "document", None)
        if doc is None or getattr(doc, "image", None) is None:
            QMessageBox.information(self, "Signature / Insert", "Active document has no image."); return
        from pro.signature_insert import SignatureInsertDialogPro
        dlg = SignatureInsertDialogPro(self, doc, icon=QIcon(signature_icon_path))
        dlg.show()

    def _open_halo_b_gon(self):
        sw = self.mdi.activeSubWindow()
        if not sw: 
            QMessageBox.information(self, "Halo-B-Gon", "No active image view."); return
        doc = getattr(sw.widget(), "document", None)
        if doc is None or getattr(doc, "image", None) is None:
            QMessageBox.information(self, "Halo-B-Gon", "Active view has no image."); return
        from pro.halobgon import HaloBGonDialogPro
        dlg = HaloBGonDialogPro(self, doc, icon=QIcon(halo_path))
        dlg.show()

    def _apply_halobgon_preset_to_doc(self, doc, preset: dict | None):
        """
        Headless Halo-B-Gon apply on a document using a preset dict.

        Preset keys:
          â€¢ reduction: int 0..3
          â€¢ linear: bool
        """
        from pro.halobgon import apply_halo_b_gon_to_doc

        p = dict(preset or {})
        apply_halo_b_gon_to_doc(self, doc, p)

        # Also register as last_headless_command so replay uses this
        try:
            payload = {
                "command_id": "halo_b_gon",
                "preset": dict(p),
            }
            setattr(self, "_last_headless_command", payload)
        except Exception:
            pass


    def _open_aberration_ai(self):
        from pro.aberration_ai import AberrationAIDialog
        sw = self.mdi.activeSubWindow()
        if not sw:
            QMessageBox.information(self, "Aberration Correction", "No active image view.")
            return

        w = sw.widget() if hasattr(sw, "widget") else None
        doc = getattr(w, "document", None)
        if doc is None or getattr(doc, "image", None) is None:
            QMessageBox.information(self, "Aberration Correction", "Active view has no image.")
            return

        try:
            # pass a callable that returns the current doc (matches your dialog signature)
            dlg = AberrationAIDialog(self, self.docman, get_active_doc_callable=lambda: doc, icon=QIcon(aberration_path))
            dlg.exec()
        except Exception as e:
            print(f"Failed to open Aberration AI: {e}")

    def _open_cosmic_clarity_ui(self):
        print("Opening Cosmic Clarity UI...")
        try:
            from pro import cosmicclarity as cc
            CosmicClarityDialogPro = cc.CosmicClarityDialogPro
        except Exception as e:
            import traceback
            print("Failed to import pro.cosmicclarity:", e)
            traceback.print_exc()
            QMessageBox.critical(self, "Cosmic Clarity",
                                f"Failed to import Cosmic Clarity module:\n{e}")
            return

        sw = self.mdi.activeSubWindow()
        if not sw:
            QMessageBox.information(self, "Cosmic Clarity", "No active image view.")
            return

        w = sw.widget() if hasattr(sw, "widget") else None
        doc = getattr(w, "document", None)
        if doc is None or getattr(doc, "image", None) is None:
            QMessageBox.information(self, "Cosmic Clarity", "Active view has no image.")
            return

        # 🔸 Clear any stale headless flag when user explicitly opens the UI
        try:
            s = QSettings()
            s.remove("cc/headless_in_progress")
        except Exception:
            pass

        try:
            print("Creating CosmicClarityDialogPro (interactive)...")
            dlg = CosmicClarityDialogPro(
                self,
                doc,
                icon=QIcon(cosmic_path),
                headless=False,
                bypass_guard=True,          # <-- key change
            )
            print("Dialog created, calling exec()...")
            result = dlg.exec()
            print(f"Cosmic Clarity dialog returned code {result}")
        except Exception as e:
            import traceback
            print("Failed to open Cosmic Clarity UI:", e)
            traceback.print_exc()
            QMessageBox.critical(self, "Cosmic Clarity",
                                f"Failed to open Cosmic Clarity UI:\n{e}")


    def _open_cosmic_clarity_satellite(self):
        from pro.cosmicclarity import CosmicClaritySatelliteDialogPro
        # It's OK if there is no active subwindow or no image.
        sw = self.mdi.activeSubWindow() if hasattr(self, "mdi") else None
        doc = None
        if sw is not None and hasattr(sw, "widget") and sw.widget() is not None:
            # Only try to grab a document if it exists
            w = sw.widget()
            doc = getattr(w, "document", None)

        try:
            # Use the Satellite dialog here (not the main CC dialog)
            dlg = CosmicClaritySatelliteDialogPro(self, doc, icon=QIcon(satellite_path))
            dlg.exec()
        except Exception as e:
            print(f"Failed to open Cosmic Clarity Satellite: {e}")
            QMessageBox.critical(self, "Cosmic Clarity Satellite",
                                f"Failed to open Cosmic Clarity Satellite:\n{e}")

    def _open_history_explorer(self):
        from pro.history_explorer import HistoryExplorerDialog
        sw = self.mdi.activeSubWindow()
        doc = sw.widget().document if sw else None
        if not doc:
            QMessageBox.information(self, "History", "No active document.")
            return

        dlg = HistoryExplorerDialog(doc, parent=self)
        sub_title = sw.windowTitle() if sw else doc.display_name()
        dlg.setWindowTitle(f"History Explorer -- {sub_title}")
        dlg.show()
        self._log("History: opened History Explorer.")

    def _open_blink_tool(self):
        from pro.blink_comparator_pro import BlinkComparatorPro
        dlg = BlinkComparatorPro(doc_manager=self.docman)
        dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        dlg.setWindowTitle("Blink Comparator")
        try:
            dlg.setWindowIcon(QIcon(blink_path))
        except Exception:
            pass

        # pass blink dialog back so we can bring it to front afterward
        dlg.sendToStacking.connect(
            lambda paths, target, d=dlg: self._on_blink_send_to_stacking(paths, target, d)
        )

        dlg.show()

    def _open_ppp_tool(self):
        from pro.perfect_palette_picker import PerfectPalettePicker
        w = PerfectPalettePicker(doc_manager=self.docman)  # parent gives access to _spawn_subwindow_for
        w.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        w.setWindowTitle("Perfect Palette Picker")
        try:
            w.setWindowIcon(QIcon(ppp_path))
        except Exception:
            pass        
        w.show()   

    def _open_nbtorgb_tool(self):
        from pro.nbtorgb_stars import NBtoRGBStars
        w = NBtoRGBStars(doc_manager=self.docman)
        w.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        w.setWindowTitle("NB -> RGB Stars")
        try:
            w.setWindowIcon(QIcon(nbtorgb_path))
        except Exception:
            pass
        w.show()

    def _open_selective_color_tool(self):
        # get active document, same pattern you use elsewhere
        doc = None
        if hasattr(self, "mdi") and self.mdi.activeSubWindow():
            sw = self.mdi.activeSubWindow().widget()
            doc = getattr(sw, "document", None)
        if doc is None and getattr(self, "docman", None) and self.docman._docs:
            doc = self.docman._docs[-1]
        if doc is None or getattr(doc, "image", None) is None:
            QMessageBox.information(self, "No image", "Open an image first."); return

        from pro.selective_color import SelectiveColorCorrection
        w = SelectiveColorCorrection(doc_manager=self.docman, document=doc, parent=self,
                                    window_icon=QIcon(selectivecolor_path))
        w.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        w.show()

    def _open_freqsep_tool(self):
        from pro.frequency_separation import FrequencySeperationTab
        # get the active ImageDocument (same pattern you use elsewhere)
        doc = None
        if hasattr(self, "mdi") and self.mdi.activeSubWindow():
            sw = self.mdi.activeSubWindow().widget()
            doc = getattr(sw, "document", None)

        # fallback to last opened document if needed
        if doc is None and getattr(self, "docman", None) and self.docman._docs:
            doc = self.docman._docs[-1]

        w = FrequencySeperationTab(doc_manager=self.docman, document=doc)
        w.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        w.setWindowTitle("Frequency Separation")

        try:
            w.setWindowIcon(QIcon(freqsep_path))
        except Exception:
            pass         
        # If we have a document, preload its image/metadata before showing
        if doc is not None and getattr(doc, "image", None) is not None:
            w.set_image_from_doc(doc.image, doc.metadata)

        w.show()

    def _open_contsub_tool(self):
        from pro.continuum_subtract import ContinuumSubtractTab
        w = ContinuumSubtractTab(doc_manager=self.docman)
        w.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        w.setWindowTitle("Continuum Subtract")
        try:
            w.setWindowIcon(QIcon(contsub_path))
        except Exception:
            pass
        w.show()

    def _open_image_combine(self):
        from pro.image_combine import ImageCombineDialog
        w = ImageCombineDialog(self)   # <- only pass self
        w.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        w.setWindowTitle("Image Combination")
        try:
            w.setWindowIcon(QIcon(imagecombine_path))
        except Exception:
            pass        
        w.resize(900, 650)
        w.show()                       # <- modeless; no exec(), no result_preset()


    def _apply_image_combine_from_preset(self, preset: dict, *, target_doc=None):
        """
        Headless apply (supports drag+drop) or dialog-OK path.
        If target_doc is provided (drop onto a view), it's A; otherwise use preset/doc chooser.
        """
        dm = getattr(self, "doc_manager", None) or getattr(self, "dm", None)
        if dm is None:
            QMessageBox.warning(self, "Image Combine", "No document manager."); return

        docs = self._list_open_docs()
        if not docs:
            QMessageBox.information(self, "Image Combine", "No open images."); return

        mode   = preset.get("mode", "Blend")
        alpha  = float(preset.get("opacity", 1.0))
        lonly  = bool(preset.get("luma_only", False))
        output = preset.get("output", "replace")

        # Resolve A/B
        A = target_doc or next((d for d in docs if id(d) == preset.get("docA_id")), self._active_doc())
        B = next((d for d in docs if id(d) == preset.get("docB_id")), None)

        # fallback for B by title (useful for shortcut presets across sessions)
        if B is None:
            title = (preset.get("docB_title") or "").strip()
            if title:
                for d in docs:
                    if _display_name(d).strip() == title:
                        B = d; break

        # if still None and exactly two docs, pick the other
        if (A is not None) and (B is None) and len(docs) == 2:
            B = docs[0] if docs[1] is A else docs[1]

        if A is None or B is None:
            QMessageBox.warning(self, "Image Combine", "Could not resolve Source A and B."); return

        imgA = np.asarray(getattr(A, "image", None), dtype=np.float32)
        imgB = np.asarray(getattr(B, "image", None), dtype=np.float32)
        if imgA is None or imgB is None:
            QMessageBox.warning(self, "Image Combine", "One of the sources has no image."); return
        if imgA.shape[:2] != imgB.shape[:2]:
            QMessageBox.warning(self, "Image Combine", "Image sizes must match."); return

        from pro.image_combine import _blend_dispatch, _rgb_to_luma, _recombine_luma_into_rgb, _to_float01
        try:
            if lonly:
                if imgA.ndim != 3 or (imgA.shape[2] != 3):
                    QMessageBox.warning(self, "Luminance Blend", "Source A must be RGB."); return
                YA = _rgb_to_luma(imgA)
                YB = _rgb_to_luma(imgB)
                Ymix = _blend_dispatch(YA[..., None], YB[..., None], mode, alpha)[..., 0]
                result = _recombine_luma_into_rgb(Ymix, imgA)
                step = f"Luminance {mode}"
            else:
                A3 = imgA if imgA.ndim == 3 else imgA[..., None]
                B3 = imgB if imgB.ndim == 3 else imgB[..., None]
                result = _blend_dispatch(A3, B3, mode, alpha)
                if imgA.ndim == 2: result = result[..., 0]
                step = f"{mode} Combine"

            result = _to_float01(result)

            if output == "replace":
                if hasattr(A, "set_image"):
                    A.set_image(result, step_name=f"Image Combine: {step}")
                else:
                    A.image = result; A.changed.emit()
                self._log(f"Image Combine -> replaced '{_display_name(A)}' ({step})")
            else:
                newdoc = dm.create_document(result, metadata={
                    "display_name": f"Combined ({step})",
                    "bit_depth": "32-bit floating point",
                    "is_mono": (result.ndim == 2),
                    "source": f"Combine: {step}",
                }, name=f"Combined ({step})")
                self._spawn_subwindow_for(newdoc)
                self._log(f"Image Combine -> new view '{newdoc.display_name()}' ({step})")

        except Exception as e:
            QMessageBox.critical(self, "Image Combine", f"Failed:\n{e}")

    def _open_psf_viewer(self, preset: dict | None = None):
        from pro.psf_viewer import PSFViewer
        sw = self.mdi.activeSubWindow()
        if not sw:
            QMessageBox.information(self, "Pixel Math", "No active image window.")
            return

        dlg = PSFViewer(self.mdi.activeSubWindow().widget(), parent=self)
        dlg.setWindowIcon(QIcon(psf_path))

        # Optional preset support: {"threshold": int, "mode": "PSF"|"Flux", "log": bool, "zoom": int}
        if isinstance(preset, dict):
            if "threshold" in preset:
                try: dlg.threshold_slider.setValue(int(preset["threshold"]))
                except Exception as e:
                    import logging
                    logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")
            if preset.get("mode") == "Flux":
                try: dlg.toggleHistogramMode()
                except Exception as e:
                    import logging
                    logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")
            if bool(preset.get("log", False)) != bool(dlg.log_scale):
                try: dlg.log_toggle_button.setChecked(bool(preset.get("log", False)))
                except Exception as e:
                    import logging
                    logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")
            if "zoom" in preset:
                try: dlg.zoom_slider.setValue(int(preset["zoom"]))
                except Exception as e:
                    import logging
                    logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")

        dlg.show()

    def _open_image_peeker(self):
        from pro.image_peeker_pro import ImagePeekerDialogPro
        sw = self.mdi.activeSubWindow()
        if not sw:
            QMessageBox.information(self, "Image Peaker", "No active image window.")
            return

        dlg = ImagePeekerDialogPro(parent=self, document=sw, settings=self.settings)
        dlg.setWindowIcon(QIcon(peeker_icon))
        dlg.show()

    def _open_image_peeker_for_doc(self, doc, title_hint=None):
        from pro.image_peeker_pro import ImagePeekerDialogPro
        dlg = ImagePeekerDialogPro(parent=self, document=doc, settings=self.settings)
        try: dlg.setWindowIcon(QIcon(peeker_icon))
        except Exception as e:
            import logging
            logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")
        dlg.show()
        if hasattr(self, "_log"):
            self._log(f"Opened Image Peeker for '{title_hint or getattr(doc, 'display_name', lambda:'view')()}'")


    def _open_plate_solver(self):
        from pro.plate_solver import plate_solve_doc_inplace, PlateSolverDialog
        dlg = PlateSolverDialog(self.settings, parent=self)
        dlg.setWindowIcon(QIcon(platesolve_path))
        dlg.show()  # modal; keeps the dialog (and its QProcess) alive until done

        # After modal returns, refresh header viewer just in case it changed
        try:
            doc = self._active_doc()
            if doc:
                self._hdr_refresh_timer.start(0)
        except Exception:
            pass

        # Optional: if you have a tree/list that depends on metadata, refresh it too
        try:
            if hasattr(self, "_refresh_treebox"):
                self._refresh_treebox()
        except Exception:
            pass


    def _open_stellar_alignment(self):
        from pro.star_alignment import StellarAlignmentDialog
        dlg = StellarAlignmentDialog(
            parent=self,
            settings=self.settings,
            doc_manager=self.doc_manager,            # <- so Apply/New use undo/redo + creation
            list_open_docs_fn=self._list_open_docs   # <- same helper used by RGB dialog
        )
        dlg.setWindowIcon(QIcon(staralign_path))
        dlg.show()   # modal (keeps workers/dialog alive)
        try:
            doc = self._active_doc()
            if doc:
                self._hdr_refresh_timer.start(0)
        except Exception:
            pass
        try:
            if hasattr(self, "_refresh_treebox"):
                self._refresh_treebox()
        except Exception:
            pass

    def _open_stellar_registration(self):
        # If we still have a handle, make sure it's alive before using it
        win = getattr(self, "_starreg_win", None)
        if win is not None:
            try:
                if win.isVisible():
                    win.raise_()
                    win.activateWindow()
                    return
            except RuntimeError:
                # C++ object was deleted; drop the stale Python handle
                self._starreg_win = None

        # Create a fresh window
        from pro.star_alignment import StarRegistrationWindow
        self._starreg_win = StarRegistrationWindow(parent=self)
        self._starreg_win.setWindowFlag(Qt.WindowType.Window, True)
        self._starreg_win.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        # When the window actually dies, clear our handle so future calls recreate it
        self._starreg_win.destroyed.connect(lambda: setattr(self, "_starreg_win", None))
        self._starreg_win.setWindowIcon(QIcon(starregistration_path))
        self._starreg_win.show()

    def _open_rgb_align(self):
        sw = self.mdi.activeSubWindow()
        if not sw:
            QMessageBox.information(self, "RGB Align", "No active image window.")
            return

        view = sw.widget()
        from pro.rgbalign import RGBAlignDialog
        dlg = RGBAlignDialog(parent=self, document=view)
        dlg.setWindowIcon(QIcon(rgbalign_path))
        dlg.show()

    def _open_mosaic_master(self):
        from pro.star_alignment import MosaicMasterDialog
        dlg = MosaicMasterDialog(
            settings=self.settings,
            parent=self,
            image_manager=getattr(self, "image_manager", None),
            doc_manager=getattr(self, "doc_manager", None),
            wrench_path=wrench_path,
            spinner_path=spinner_path,
            list_open_docs_fn=getattr(self, "_list_open_docs", None),  # <- add this
        )
        dlg.setWindowFlag(Qt.WindowType.Window, True)
        dlg.setWindowIcon(QIcon(mosaic_path))
        dlg.show()

    def _open_live_stacking(self):
        from pro.live_stacking import LiveStackWindow
        dlg = LiveStackWindow(
            parent=self,
            doc_manager=getattr(self, "doc_manager", None),   # pass doc_manager (not image_manager)
            wrench_path=wrench_path,                          # optional: for the settings button icon
            spinner_path=spinner_path                         # optional: if you want to reuse
        )
        dlg.setWindowFlag(Qt.WindowType.Window, True)
        dlg.setWindowIcon(QIcon(livestacking_path))
        dlg.show()

    def _open_stacking_suite(self):
        # Reuse if we already have one
        dlg = getattr(self, "_stacking_suite", None)
        if dlg is not None:
            try:
                if not dlg.isVisible():
                    dlg.show()
                dlg.raise_()
                dlg.activateWindow()
                return dlg     # ðŸ'ˆ return existing
            except RuntimeError:
                self._stacking_suite = None  # C++ deleted, recreate

        from pro.stacking_suite import StackingSuiteDialog
        dlg = StackingSuiteDialog(
            parent=self,
            wrench_path=wrench_path,
            spinner_path=spinner_path,
        )
        dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        dlg.destroyed.connect(lambda _=None: setattr(self, "_stacking_suite", None))

        dlg.setWindowFlag(Qt.WindowType.Window, True)
        dlg.setWindowIcon(QIcon(stacking_path))
        dlg.show()

        self._stacking_suite = dlg
        return dlg 

    def _on_blink_send_to_stacking(self, paths: list[str], target: str, blink_dlg=None):
        # 1) open / focus stacking first
        dlg = self._open_stacking_suite()
        # 2) push the files in
        dlg.ingest_paths_from_blink(paths, target)
        # 3) then re-raise blink so it doesn't get lost
        if blink_dlg is not None:
            try:
                blink_dlg.show()
                blink_dlg.raise_()
                blink_dlg.activateWindow()
            except Exception:
                pass

    def _on_stackingsuite_relaunch(self, old_dir: str, new_dir: str):
        # Optional: respond to dialog's relaunch request
        try:
            if getattr(self, "_stacking_suite", None):
                self._stacking_suite.close()
        except Exception:
            pass
        self._stacking_suite = None

        # re-open
        self._open_stacking_suite()
        # if your dialog exposes a setter, apply the new directory:
        if hasattr(self._stacking_suite, "set_stacking_directory"):
            self._stacking_suite.set_stacking_directory(new_dir)

    def _open_supernova_hunter(self):
        from pro.supernovaasteroidhunter import SupernovaAsteroidHunterDialog
        dlg = SupernovaAsteroidHunterDialog(
            parent=self,
            settings=getattr(self, "settings", None),
            image_manager=getattr(self, "image_manager", None),
            doc_manager=getattr(self, "doc_manager", None),
            supernova_path=supernova_path,         # for the window icon
            wrench_path=wrench_path,               # optional if you want a settings icon later
            spinner_path=spinner_path              # optional
        )
        dlg.setWindowFlag(Qt.WindowType.Window, True)
        dlg.setWindowIcon(QIcon(supernova_path))
        dlg.show()

    def _open_star_spikes(self, *, doc=None, preset: dict | None = None, title_hint: str | None = None):
        from pro.star_spikes import StarSpikesDialogPro
        dlg = StarSpikesDialogPro(
            parent=self,
            doc_manager=getattr(self, "docman", None),
            initial_doc=doc,
            jwstpupil_path=jwstpupil_path,
            aperture_help_path=aperture_path,
            spinner_path=spinner_path,  # optional; used if you want
        )
        if preset:
            dlg.apply_preset(preset)
        if title_hint:
            dlg.setWindowTitle(f"Diffraction Spikes -- {title_hint}")
        dlg.setWindowFlag(Qt.WindowType.Window, True)
        dlg.setWindowIcon(QIcon(starspike_path))
        dlg.show()

    def _open_astrospike(self, *, doc=None):
        """Open the AstroSpike dialog with advanced diffraction effects."""
        from pro.astrospike import AstroSpikeDialog
        from pro.resources import Icons
        from pro.headless_utils import unwrap_docproxy
        
        # Get the active document first
        active_doc = doc
        if active_doc is None:
            sw = self.mdi.activeSubWindow()
            if sw:
                view = sw.widget()
                if hasattr(view, "document"):
                    active_doc = view.document
        
        # Unwrap DocProxy if needed - call _target() if it exists
        if active_doc and hasattr(active_doc, "_target"):
            active_doc = active_doc._target()
        else:
            active_doc = unwrap_docproxy(active_doc)
        
        # Print debug info
        print(f"[AstroSpike] Active document (unwrapped): {active_doc}")
        if active_doc and hasattr(active_doc, "image"):
            img = active_doc.image
            print(f"[AstroSpike] Active image shape: {img.shape if img is not None else 'None'}")
        
        # Create a callback to get the current image from the active document
        def get_image_callback():
            """Get image from active document if available."""
            if active_doc and hasattr(active_doc, "image"):
                img = active_doc.image
                if img is not None:
                    print(f"[AstroSpike] Callback: Retrieved image shape {img.shape}, dtype {img.dtype}")
                    # Ensure correct format for the script
                    if img.dtype == np.uint8:
                        return img.astype(np.float32) / 255.0
                    elif img.dtype == np.float32:
                        if img.max() > 1.0:
                            return img / 255.0
                        return img
                    else:
                        return img.astype(np.float32)
                else:
                    print("[AstroSpike] Callback: image is None")
            else:
                print("[AstroSpike] Callback: active_doc or image attribute not available")
            return None
        
        # Create a callback to set the image back to the document
        def set_image_callback(image_data, step_name):
            """Apply the result image back to the active document."""
            if active_doc and hasattr(active_doc, "set_image"):
                print(f"[AstroSpike] Setting image back to document, shape: {image_data.shape}")
                # Pass metadata as empty dict and step_name separately
                active_doc.set_image(image_data, metadata={}, step_name=step_name)
            elif active_doc and hasattr(active_doc, "image"):
                print(f"[AstroSpike] Setting image directly, shape: {image_data.shape}")
                active_doc.image = image_data
            else:
                print("[AstroSpike] Cannot set image - active_doc or methods not available")
        
        icon_path = Icons().ASTRO_SPIKE
        dlg = AstroSpikeDialog(parent=self, icon_path=icon_path, get_image_callback=get_image_callback, set_image_callback=set_image_callback)
        dlg.showMaximized()
        dlg.exec()

    def _open_exo_detector(self):
        # Lazy import to avoid loading lightkurve at startup (~12s)
        from pro.exoplanet_detector import ExoPlanetWindow
        dlg = ExoPlanetWindow(
            parent=self,
            wrench_path=wrench_path,
        )
        dlg.setWindowFlag(Qt.WindowType.Window, True)
        dlg.setWindowIcon(QIcon(exoicon_path))

        # Optional WIMI wiring (safe guards so we don't assume exact API names)
        if hasattr(self, "wimi_tab"):
            # WIMI -> dialog: emit RA/Dec once solved
            if hasattr(self.wimi_tab, "wcsCoordinatesAvailable"):
                self.wimi_tab.wcsCoordinatesAvailable.connect(dlg.receive_wcs_coordinates)
            # dialog -> WIMI: send a reference FITS path to solve
            # (rename this to whatever your WIMI expects; kept guarded)
            if hasattr(self.wimi_tab, "open_reference_path"):
                dlg.referenceSelected.connect(self.wimi_tab.open_reference_path)

        dlg.show()

    def _open_isophote(self):
        from pro.isophote import IsophoteModelerDialog
        # Mirror PSF opener: use MDI active subwindow -> widget -> document -> image
        sw = self.mdi.activeSubWindow()
        if not sw:
            QMessageBox.information(self, "GLIMR", "No active image window.")
            return

        view = sw.widget()
        doc = getattr(view, "document", None) or view

        # Grab image from the doc (same style as PSF)
        img = getattr(doc, "image", None)
        if img is None:
            QMessageBox.information(self, "GLIMR", "Active view has no image data.")
            return

        # Ensure ndarray
        try:
            arr = np.asarray(img)
        except Exception:
            QMessageBox.information(self, "GLIMR", "Could not read image data from the active view.")
            return

        # Coerce to mono float32 in [0,1]
        if arr.ndim == 3 and arr.shape[-1] in (3, 4):
            mono = arr[..., :3].mean(axis=2)
        else:
            mono = arr
        mono = mono.astype(np.float32, copy=False)

        if np.issubdtype(arr.dtype, np.integer):
            info = np.iinfo(arr.dtype)
            rng = max(1, info.max - info.min)
            mono = (mono - float(info.min)) / float(rng)
        else:
            # assume already roughly normalized; clamp into display range
            mono = np.clip(mono, 0.0, 1.0)

        # doc_manager only used for pushing new docs/views from the dialog
        dm = getattr(self, "doc_manager", None)

        dlg = IsophoteModelerDialog(
            mono_image=mono,
            parent=self,
            title_hint="GLIMR -- Isophote Modeler",
            doc_manager=dm,
        )
        dlg.setWindowFlag(Qt.WindowType.Window, True)
        dlg.setWindowIcon(QIcon(isophote_path))
        dlg.show()

    def _open_whats_in_my_sky(self):
        from wims import WhatsInMySkyDialog
        dlg = WhatsInMySkyDialog(
            parent=self,
            wims_path=wims_path,          # window icon
            wrench_path=wrench_path       # optional settings icon
        )
        dlg.setWindowFlag(Qt.WindowType.Window, True)
        dlg.setWindowIcon(QIcon(wims_path))
        dlg.show()

    def _open_wimi(self):
        # Lazy import to avoid loading lightkurve at startup (~12s)
        from wimi import WIMIDialog
        dlg = WIMIDialog(
            parent=self,
            settings=getattr(self, "settings", None),
            doc_manager=getattr(self, "doc_manager", None),
            wimi_path=wimi_path,       # window icon
            wrench_path=wrench_path    # optional for settings button
        )
        dlg.setWindowFlag(Qt.WindowType.Window, True)
        dlg.setWindowIcon(QIcon(wimi_path))
        dlg.show()


    def _open_fits_modifier(self):
        from pro.fitsmodifier import FITSModifier
        doc = self.doc_manager.get_active_document()
        if not doc:
            QMessageBox.information(self, "FITS Header Editor", "No active image window.")
            return

        file_path = doc.metadata.get("file_path")
        header    = doc.metadata.get("original_header") or {}

        dlg = FITSModifier(
            file_path=file_path if (file_path and os.path.isfile(file_path)) else None,
            header=header,
            doc_manager=self.doc_manager,
            active_document=doc,
            parent=self,
        )
        # dlg.setWindowIcon(QIcon("..."))  # optional
        dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        dlg.show()

    def _open_fits_batch_modifier(self):
        from pro.fitsmodifier import BatchFITSHeaderDialog
        """
        doc = self.doc_manager.get_active_document()
        if not doc:
            QMessageBox.information(self, "FITS Header Editor", "No active image window.")
            return
        file_path = doc.metadata.get("file_path")
        header    = doc.metadata.get("original_header") or {}
        """
        dlg = BatchFITSHeaderDialog(
            parent=self,
        )
        # dlg.setWindowIcon(QIcon("..."))  # optional
        dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        dlg.show()



    def _open_batch_renamer(self):
        from pro.batch_renamer import BatchRenamerDialog
        dlg = BatchRenamerDialog(parent=self)
        dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        dlg.show()

    def _open_astrobin_exporter(self):
        from pro.astrobin_exporter import AstrobinExporterDialog
        # you said this is defined in the parent main UI:
        # astrobin_filters_csv_path = os.path.join(sys._MEIPASS, 'astrobin_filters.csv')

        dlg = AstrobinExporterDialog(self, offline_filters_csv=astrobin_filters_csv_path)
        dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        dlg.show()

    def _open_batch_convert(self):
        from pro.batch_convert import BatchConvertDialog
        dlg = BatchConvertDialog(self)
        dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        dlg.show()

    def _open_copy_astrometry(self):
        from pro.copyastro import CopyAstrometryDialog
        sw = self.mdi.activeSubWindow()
        if not sw:
            QMessageBox.information(self, "Copy Astrometric Solution", "No active image window.")
            return

        dlg = CopyAstrometryDialog(parent=self, target=sw)
        dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        dlg.show()

    def _open_linear_fit(self):
        from pro.linear_fit import LinearFitDialog
        dm = getattr(self, "doc_manager", None) or getattr(self, "docman", None)
        if dm is None:
            QMessageBox.information(self, "Linear Fit", "No document manager available.")
            return

        sw = self.mdi.activeSubWindow()
        if not sw:
            QMessageBox.information(self, "No image", "Open an image first.")
            return

        view = sw.widget()
        active_doc = None

        # Prefer ROI-aware resolution from DocManager
        try:
            if hasattr(dm, "get_document_for_view"):
                active_doc = dm.get_document_for_view(view)
        except Exception:
            active_doc = None

        # Fallback to the view's base document
        if active_doc is None:
            try:
                active_doc = getattr(view, "document", None)
            except Exception:
                active_doc = None

        if active_doc is None or getattr(active_doc, "image", None) is None:
            QMessageBox.information(self, "Linear Fit", "No active image.")
            return

        dlg = LinearFitDialog(self, dm, active_doc)  # <-- pass ROI-aware doc + DM
        try:
            dlg.setWindowIcon(QIcon(self._icon_path("linear_fit")))
        except Exception:
            pass
        dlg.resize(900, 600)
        dlg.show()


    def _open_debayer(self):
        from pro.debayer import DebayerDialog
        sw = self.mdi.activeSubWindow()
        if not sw:
            QMessageBox.information(self, "No image", "Open an image first.")
            return
        try:
            doc = sw.widget().document
        except Exception:
            QMessageBox.information(self, "Debayer", "No active image.")
            return
        dm = getattr(self, "doc_manager", None) or getattr(self, "docman", None)
        if dm is None:
            QMessageBox.information(self, "Debayer", "No document manager available.")
            return
        dlg = DebayerDialog(self, dm, doc)
        try:
            dlg.setWindowIcon(QIcon(self._icon_path("debayer")))
        except Exception:
            pass
        dlg.resize(700, 420)
        dlg.show()


    def _about(self):
        dlg = AboutDialog(
            parent=self,
            version=getattr(self, "_version", ""),
            build_timestamp=getattr(self, "_build_timestamp", ""),
        )
        dlg.exec()

    #######-------COMMAND DROPS-------#################
    def remember_last_headless_command(
        self,
        command_id: str,
        preset: dict | None = None,
        description: str = "",
    ):
        """
        Store the last headless-style command so subwindows can ask to replay it.
        Also appends it to a rolling history for the replay dropdown.
        Shape matches what _handle_command_drop expects.
        """
        payload = {
            "command_id": command_id,
            "preset": dict(preset or {}),
        }
        # Keep old single-slot behavior for "Replay last"
        self._last_headless_command = payload

        # NEW: append to history with a human label
        try:
            desc = (description or command_id).strip()
        except Exception:
            desc = command_id

        entry = {
            "command_id": command_id,
            "preset": dict(preset or {}),
            "description": desc,
        }

        hist = getattr(self, "_headless_history", None)
        if hist is None:
            self._headless_history = hist = []

        hist.append(entry)

        # Cap the list
        max_len = getattr(self, "_headless_history_max", 50) or 0
        if max_len and len(hist) > max_len:
            del hist[:-max_len]

        # Logging as before
        try:
            self._log(f"[Replay] Last action stored: {desc} (command_id={command_id})")
        except Exception:
            print(f"[Replay] Last action stored: {desc} (command_id={command_id})")



    def _remember_last_headless_command(self, command_id: str, preset: dict | None = None, description: str = ""):
        """
        Private alias so older/newer call sites can use the underscored name.
        """
        return self.remember_last_headless_command(command_id, preset, description)

    def get_headless_history(self) -> list[dict]:
        """
        Return a *copy* of the headless history list.
        Each entry: {"command_id", "preset", "description"}.
        Newest is last.
        """
        return list(getattr(self, "_headless_history", []) or [])

    def replay_headless_history_entry_on_base(self, index: int, target_sw=None):
        """
        Replay a specific history entry on the base doc of target_sw,
        reusing replay_last_action_on_base for all the special cases.
        """
        hist = getattr(self, "_headless_history", [])
        if not hist:
            QMessageBox.information(self, "Replay Action", "There are no actions in history yet.")
            return

        try:
            entry = hist[index]
        except IndexError:
            QMessageBox.warning(self, "Replay Action", "Selected history item is no longer available.")
            return

        # Build a payload in the same schema replay_last_action_on_base expects
        payload = {
            "command_id": entry.get("command_id"),
            "preset": dict(entry.get("preset") or {}),
        }

        # Temporarily override _last_headless_command so we can reuse
        # your big replay_last_action_on_base() switchboard unchanged.
        old = getattr(self, "_last_headless_command", None)
        try:
            self._last_headless_command = payload
            self.replay_last_action_on_base(target_sw=target_sw)
        finally:
            self._last_headless_command = old


    def replay_last_action_on_subwindow(self, target_sw=None):
        """
        Called by subwindow(s) when the Replay button is clicked.
        Uses the stored headless command and routes it through _handle_command_drop.
        """
        payload = getattr(self, "_last_headless_command", None)

        # DEBUG
        try:
            self._log(
                f"[Replay] replay_last_action_on_subwindow: payload={bool(payload)}, "
                f"target_sw={id(target_sw) if target_sw else None}"
            )
        except Exception:
            print(
                f"[Replay] replay_last_action_on_subwindow: payload={bool(payload)}, "
                f"target_sw={id(target_sw) if target_sw else None}"
            )

        if not payload:
            QMessageBox.information(
                self, "Replay Last Action",
                "There is no previous action to replay yet."
            )
            return

        # Resolve target subwindow
        if target_sw is None and hasattr(self, "mdi"):
            target_sw = self.mdi.activeSubWindow()

        if target_sw is None:
            QMessageBox.information(
                self, "Replay Last Action",
                "No active image view to apply the action to."
            )
            return

        try:
            self._handle_command_drop(dict(payload), target_sw=target_sw)
        except Exception as e:
            QMessageBox.critical(self, "Replay Last Action", f"Replay failed:\n{e}")

    def replay_last_action_on_base(self, target_sw=None):
        """
        Replay last headless command, but target the *base* document behind the view.
        Used by preview tabs that want "do this again on the full image".
        """
        payload = getattr(self, "_last_headless_command", None) or {}

        # DEBUG
        try:
            self._log(
                f"[Replay] replay_last_action_on_base: payload={bool(payload)}, "
                f"target_sw={id(target_sw) if target_sw else None}"
            )
        except Exception:
            print(
                f"[Replay] replay_last_action_on_base: payload={bool(payload)}, "
                f"target_sw={id(target_sw) if target_sw else None}"
            )

        if not payload:
            QMessageBox.information(
                self, "Replay Last Action",
                "There is no previous action to replay yet."
            )
            return

        # Resolve target subwindow
        if target_sw is None and hasattr(self, "mdi"):
            target_sw = self.mdi.activeSubWindow()

        if target_sw is None:
            QMessageBox.information(
                self, "Replay Last Action",
                "No active image view to apply the action to."
            )
            return

        # Resolve the *base* document for this subwindow
        base_doc = self._target_doc_from_subwindow(target_sw) if hasattr(self, "_target_doc_from_subwindow") else None
        if base_doc is None or getattr(base_doc, "image", None) is None:
            QMessageBox.information(self, "Replay Last Action", "No base image to apply the action to.")
            return

        # Small debug about which doc we're hitting
        try:
            view = target_sw.widget()
            cur_doc = getattr(view, "document", None)
            self._log(
                f"[Replay] base_doc id={id(base_doc)}, "
                f"view.document id={id(cur_doc)}, "
                f"same={base_doc is cur_doc}"
            )
        except Exception:
            pass

        # ---- Extract cid + preset from payload (support both old + new schemas) ----
        cid_raw = payload.get("command_id")
        if cid_raw is None:
            cid_raw = payload.get("cid")
        cid = str(cid_raw or "").strip().lower()

        preset = payload.get("preset") or {}
        if not isinstance(preset, dict):
            try:
                preset = dict(preset)
            except Exception:
                preset = {}

        # ---- SPECIAL CASES: always run on base_doc ----
        if cid == "stat_stretch":
            try:
                self._apply_stat_stretch_preset_to_doc(base_doc, preset)
                try:
                    self._log(f"[Replay] Applied Statistical Stretch preset to base of '{target_sw.windowTitle()}'")
                except Exception:
                    pass
            except Exception as e:
                QMessageBox.warning(self, "Preset apply failed", str(e))
            return

        if cid == "pedestal":
            try:
                from pro.pedestal import remove_pedestal
                remove_pedestal(self, target_doc=base_doc)
                try:
                    self._log(f"[Replay] Applied Pedestal Removal to base of '{target_sw.windowTitle()}'")
                except Exception:
                    pass
            except Exception as e:
                QMessageBox.warning(self, "Pedestal Removal", str(e))
            return

        if cid == "linear_fit":
            try:
                from pro.linear_fit import apply_linear_fit_to_doc
                apply_linear_fit_to_doc(self, base_doc, preset)
                try:
                    self._log(f"[Replay] Applied Linear Fit preset to base of '{target_sw.windowTitle()}'")
                except Exception:
                    pass
            except Exception as e:
                try:
                    QMessageBox.warning(self, "Linear Fit", f"Replay-on-base failed:\n{e}")
                except Exception:
                    pass
            return

        if cid == "star_stretch":
            try:
                self._apply_star_stretch_preset_to_doc(base_doc, preset)
                try:
                    self._log(f"[Replay] Applied Star Stretch preset to base of '{target_sw.windowTitle()}'")
                except Exception:
                    pass
            except Exception as e:
                try:
                    QMessageBox.warning(self, "Star Stretch", f"Replay-on-base failed:\n{e}")
                except Exception:
                    pass
            return

        if cid == "curves":
            try:
                # preset = payload.get("preset") from above
                preset_dict = preset if isinstance(preset, dict) else {}
                op = preset_dict.get("_ops")

                if op:
                    # New, exact replay: use the headless op engine strictly on base_doc
                    from pro.curve_editor_pro import apply_curves_ops
                    ok = apply_curves_ops(base_doc, op)
                    if not ok:
                        raise RuntimeError("apply_curves_ops() returned False")

                    try:
                        self._log(
                            f"[Replay] Applied Curves (ops) to base of "
                            f"'{target_sw.windowTitle()}'"
                        )
                    except Exception:
                        pass
                    return

                # Fallback for older payloads without _ops: use preset-style helper
                from pro.curves_preset import apply_curves_via_preset
                apply_curves_via_preset(self, base_doc, preset_dict)

                try:
                    self._log(
                        f"[Replay] Applied Curves (preset) to base of "
                        f"'{target_sw.windowTitle()}'"
                    )
                except Exception:
                    pass
                return

            except Exception as e:
                try:
                    QMessageBox.warning(
                        self, "Curves", f"Replay-on-base failed:\n{e}"
                    )
                except Exception:
                    print("Replay-on-base Curves failed:", e)
            return

        if cid == "ghs":
            try:
                from pro.ghs_preset import apply_ghs_via_preset

                # Normalize payload -> dict
                preset_dict = preset if isinstance(preset, dict) else {}

                # DEBUG: what did we actually get?
                try:
                    self._log(
                        f"[Replay] GHS replay-on-base: "
                        f"preset_keys={list(preset_dict.keys())}"
                    )
                except Exception:
                    print(
                        "[Replay] GHS replay-on-base: preset_keys=",
                        list(preset_dict.keys()),
                    )

                apply_ghs_via_preset(self, base_doc, preset_dict or {})

                try:
                    self._log(
                        f"[Replay] Applied GHS preset to base of "
                        f"'{target_sw.windowTitle()}'"
                    )
                except Exception:
                    pass

            except Exception as e:
                try:
                    QMessageBox.warning(self, "GHS", f"Apply failed:\n{e}")
                except Exception:
                    print("GHS replay-on-base failed:", e)
            return

        if cid == "abe":
            try:
                from pro.abe_preset import apply_abe_via_preset

                # Normalize payload -> dict
                preset_dict = preset if isinstance(preset, dict) else {}

                # DEBUG
                try:
                    self._log(
                        f"[Replay] ABE replay-on-base: "
                        f"preset_keys={list(preset_dict.keys())}"
                    )
                except Exception:
                    print(
                        "[Replay] ABE replay-on-base: preset_keys=",
                        list(preset_dict.keys()),
                    )

                apply_abe_via_preset(self, base_doc, preset_dict or {})

                try:
                    self._log(
                        f"[Replay] Applied ABE preset to base of "
                        f"'{target_sw.windowTitle()}' (no exclusions)"
                    )
                except Exception:
                    pass

            except Exception as e:
                try:
                    QMessageBox.warning(
                        self, "ABE", f"Replay-on-base failed:\n{e}"
                    )
                except Exception:
                    print("ABE replay-on-base failed:", e)
            return
        if cid == "graxpert":
            try:
                from pro.graxpert_preset import run_graxpert_via_preset

                preset_dict = preset if isinstance(preset, dict) else {}
                op = str(preset_dict.get("op", "background")).lower()
                gpu_val = bool(preset_dict.get("gpu", True))

                # Normalize for logging
                if op == "denoise":
                    strength_raw = preset_dict.get("strength", 0.50)
                    try:
                        strength_val = float(strength_raw)
                    except Exception:
                        strength_val = 0.50
                    ai_ver = preset_dict.get("ai_version") or "latest"
                    log_msg = (
                        f"GraXpert Denoise "
                        f"(strength={strength_val:.2f}, model={ai_ver}, "
                        f"gpu={'on' if gpu_val else 'off'})"
                    )
                else:
                    smooth_raw = preset_dict.get("smoothing", 0.10)
                    try:
                        smooth_val = float(smooth_raw)
                    except Exception:
                        smooth_val = 0.10
                    log_msg = (
                        f"GraXpert Gradient Removal "
                        f"(smoothing={smooth_val:.2f}, gpu={'on' if gpu_val else 'off'})"
                    )

                # ðŸ" Re-run GraXpert on the *base* document
                run_graxpert_via_preset(self, preset_dict, target_doc=base_doc)

                try:
                    self._log(
                        f"[Replay] Applied {log_msg} to base of "
                        f"'{target_sw.windowTitle()}'"
                    )
                except Exception:
                    pass

            except Exception as e:
                try:
                    QMessageBox.warning(self, "GraXpert", f"Replay-on-base failed:\n{e}")
                except Exception:
                    print("GraXpert replay-on-base failed:", e)
            return
        if cid == "remove_stars":
            try:
                from pro.remove_stars_preset import run_remove_stars_via_preset

                # Normalize payload -> dict
                preset_dict = preset if isinstance(preset, dict) else {}

                # ðŸ" Re-run Remove Stars on the *base* document
                run_remove_stars_via_preset(self, preset_dict, target_doc=base_doc)

                # Logging (mirror command-drop logging but with replay/base info)
                tool = str(preset_dict.get("tool", "starnet")).lower()
                try:
                    if tool.startswith("star"):
                        lin = bool(preset_dict.get("linear", True))
                        self._log(
                            f"[Replay] Ran Remove Stars on base "
                            f"(tool=StarNet, linear={'yes' if lin else 'no'}) "
                            f"for '{target_sw.windowTitle()}'"
                        )
                    else:
                        mode   = preset_dict.get("mode", "unscreen")
                        stride = int(preset_dict.get("stride", 512))
                        gpu    = not bool(preset_dict.get("disable_gpu", False))
                        show   = bool(preset_dict.get("show_extracted_stars", True))
                        self._log(
                            f"[Replay] Ran Remove Stars on base "
                            f"(tool=DarkStar, mode={mode}, stride={stride}, "
                            f"gpu={'on' if gpu else 'off'}, "
                            f"stars={'on' if show else 'off'}) "
                            f"for '{target_sw.windowTitle()}'"
                        )
                except Exception:
                    # Logging should never break replay
                    pass

            except Exception as e:
                try:
                    QMessageBox.warning(
                        self, "Remove Stars",
                        f"Replay-on-base failed:\n{e}"
                    )
                except Exception:
                    print("Remove Stars replay-on-base failed:", e)
            return
        if cid == "background_neutral":
            try:
                # Normalize payload -> dict
                preset_dict = preset if isinstance(preset, dict) else {}

                # Re-run BN on the *base* document
                self._apply_background_neutral_preset_to_doc(base_doc, preset_dict)

                try:
                    mode = str(preset_dict.get("mode", "auto")).lower()
                    self._log(
                        f"[Replay] Applied Background Neutralization "
                        f"(mode={mode}) to base of '{target_sw.windowTitle()}'"
                    )
                except Exception:
                    pass

            except Exception as e:
                try:
                    QMessageBox.warning(
                        self,
                        "Background Neutralization",
                        f"Replay-on-base failed:\n{e}",
                    )
                except Exception:
                    print("Background Neutralization replay-on-base failed:", e)
            return
        if cid == "white_balance":
            try:
                # Normalize payload -> dict
                preset_dict = preset if isinstance(preset, dict) else {}

                # Re-run WB on the *base* document via the existing helper
                self._apply_white_balance_preset_to_doc(base_doc, preset_dict)

                # Optional: nice logging describing the mode/params
                try:
                    mode = str(preset_dict.get("mode", "star")).lower()
                    if mode == "manual":
                        r = float(preset_dict.get("r_gain", 1.0))
                        g = float(preset_dict.get("g_gain", 1.0))
                        b = float(preset_dict.get("b_gain", 1.0))
                        detail = f"manual (R={r:.3f}, G={g:.3f}, B={b:.3f})"
                    elif mode == "auto":
                        detail = "auto"
                    else:
                        thr = float(preset_dict.get("threshold", 50.0))
                        reuse = bool(preset_dict.get("reuse_cached_sources", True))
                        detail = (
                            f"star-based (thr={thr:.1f}, "
                            f"reuse={'yes' if reuse else 'no'})"
                        )

                    self._log(
                        f"[Replay] Applied White Balance {detail} "
                        f"to base of '{target_sw.windowTitle()}'"
                    )
                except Exception:
                    # Logging should never break replay
                    pass

            except Exception as e:
                try:
                    QMessageBox.warning(
                        self,
                        "White Balance",
                        f"Replay-on-base failed:\n{e}",
                    )
                except Exception:
                    print("White Balance replay-on-base failed:", e)
            return
        if cid == "remove_green":
            try:
                from pro.remove_green import apply_remove_green_preset_to_doc

                preset_dict = preset if isinstance(preset, dict) else {}

                # Re-run Remove Green on the *base* document
                apply_remove_green_preset_to_doc(self, base_doc, preset_dict)

                try:
                    amt = float(preset_dict.get(
                        "amount",
                        preset_dict.get("strength",
                                        preset_dict.get("value", 1.0))
                    ))
                    mode = str(preset_dict.get(
                        "mode",
                        preset_dict.get("neutral_mode", "avg")
                    )).lower()
                    preserve = bool(preset_dict.get(
                        "preserve_lightness",
                        preset_dict.get("preserve", True)
                    ))
                    self._log(
                        f"[Replay] Applied Remove Green to base of "
                        f"'{target_sw.windowTitle()}' "
                        f"(amount={amt:.2f}, mode={mode}, "
                        f"preserve_lightness={'yes' if preserve else 'no'})"
                    )
                except Exception:
                    pass

            except Exception as e:
                try:
                    QMessageBox.warning(
                        self,
                        "Remove Green",
                        f"Replay-on-base failed:\n{e}",
                    )
                except Exception:
                    print("Remove Green replay-on-base failed:", e)
            return
        if cid == "convo":
            try:
                self._apply_convo_preset_to_doc(base_doc, preset)
                try:
                    op = str(preset.get("op", "convolution"))
                    self._log(
                        f"[Replay] Applied Convo/Deconvo ({op}) preset to base of "
                        f"'{target_sw.windowTitle()}'"
                    )
                except Exception:
                    pass
            except Exception as e:
                try:
                    QMessageBox.warning(
                        self,
                        "Convo / Deconvo",
                        f"Replay-on-base failed:\n{e}",
                    )
                except Exception:
                    print("Convo replay-on-base failed:", e)
            return  # <- IMPORTANT: don't fall through to _handle_command_drop
        if cid == "wavescale_hdr":
            try:
                from pro.wavescale_hdr_preset import run_wavescale_hdr_via_preset

                # Normalize payload -> dict
                preset_dict = preset if isinstance(preset, dict) else {}

                # DEBUG (optional)
                try:
                    self._log(
                        f"[Replay] WaveScale HDR replay-on-base: "
                        f"preset_keys={list(preset_dict.keys())}"
                    )
                except Exception:
                    print(
                        "[Replay] WaveScale HDR replay-on-base: preset_keys=",
                        list(preset_dict.keys()),
                    )

                # ðŸ" Re-run WaveScale HDR on the *base* document
                run_wavescale_hdr_via_preset(self, preset_dict, target_doc=base_doc)

                # Logging similar to the command-drop handler
                try:
                    ns = int(preset_dict.get("n_scales", 5))
                    try:
                        comp = float(preset_dict.get("compression_factor", 1.5))
                    except Exception:
                        comp = 1.5
                    try:
                        mg = float(preset_dict.get("mask_gamma", 5.0))
                    except Exception:
                        mg = 5.0

                    self._log(
                        f"[Replay] Applied WaveScale HDR to base of "
                        f"'{target_sw.windowTitle()}' "
                        f"(n_scales={ns}, compression={comp:.2f}, mask_gamma={mg:.2f})"
                    )
                except Exception:
                    pass

            except Exception as e:
                try:
                    QMessageBox.warning(
                        self,
                        "WaveScale HDR",
                        f"Replay-on-base failed:\n{e}",
                    )
                except Exception:
                    print("WaveScale HDR replay-on-base failed:", e)
            return
        if cid == "wavescale_dark_enhance":
            try:
                # Normalize payload -> dict
                preset_dict = preset if isinstance(preset, dict) else {}

                n_scales   = int(preset_dict.get("n_scales", 6))
                boost      = float(preset_dict.get("boost_factor", 5.0))
                mask_gamma = float(preset_dict.get("mask_gamma", 1.0))
                iters      = int(preset_dict.get("iterations", 2))

                # Prefer helper if it exists (handles masks / blending)
                if hasattr(self, "_apply_wavescale_dark_enhance_preset_to_doc"):
                    self._apply_wavescale_dark_enhance_preset_to_doc(base_doc, {
                        "n_scales": n_scales,
                        "boost_factor": boost,
                        "mask_gamma": mask_gamma,
                        "iterations": iters,
                    })
                else:
                    # Fallback: direct compute, similar to _handle_command_drop
                    from pro.wavescalede import compute_wavescale_dse
                    import numpy as np

                    img = np.asarray(getattr(base_doc, "image", None), dtype=np.float32)
                    if img.size:
                        mx = float(np.nanmax(img))
                        if np.isfinite(mx) and mx > 1.0:
                            img = img / mx
                    img = np.clip(img, 0.0, 1.0).astype(np.float32, copy=False)

                    out, _ = compute_wavescale_dse(
                        img,
                        n_scales=n_scales,
                        boost_factor=boost,
                        mask_gamma=mask_gamma,
                        iterations=iters,
                    )
                    out = np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)

                    if hasattr(base_doc, "set_image"):
                        base_doc.set_image(out, step_name="WaveScale Dark Enhancer")
                    elif hasattr(base_doc, "apply_numpy"):
                        base_doc.apply_numpy(out, step_name="WaveScale Dark Enhancer")
                    else:
                        base_doc.image = out

                try:
                    self._log(
                        f"[Replay] WaveScale Dark Enhancer applied to base of "
                        f"'{target_sw.windowTitle()}' "
                        f"(n_scales={n_scales}, boost={boost}, "
                        f"gamma={mask_gamma}, iter={iters})"
                    )
                except Exception:
                    pass
            except Exception as e:
                try:
                    QMessageBox.warning(
                        self,
                        "WaveScale Dark Enhancer",
                        f"Replay-on-base failed:\n{e}",
                    )
                except Exception:
                    print("WaveScale Dark Enhancer replay-on-base failed:", e)
            return
        if cid == "clahe":
            try:
                # normalize preset
                p = preset if isinstance(preset, dict) else {}
                # prefer your helper (respects masks & bit depth)
                if hasattr(self, "_apply_clahe_preset_to_doc"):
                    self._apply_clahe_preset_to_doc(base_doc, p)
                else:
                    from pro.clahe import apply_clahe_to_doc
                    apply_clahe_to_doc(base_doc, p)

                # optional logging
                try:
                    clip = p.get("clip_limit", 2.0)
                    tile = p.get("tile", 8)
                    if hasattr(self, "_log"):
                        self._log(
                            f"[Replay] CLAHE applied to base of "
                            f"'{target_sw.windowTitle()}' "
                            f"(clip_limit={clip}, tile={tile})"
                        )
                except Exception:
                    pass
            except Exception as e:
                try:
                    QMessageBox.warning(self, "CLAHE", f"Replay-on-base failed:\n{e}")
                except Exception:
                    print("CLAHE replay-on-base failed:", e)
            return
        if cid == "morphology":
            try:
                p = preset if isinstance(preset, dict) else {}

                # Prefer our helper that also maintains replay state
                if hasattr(self, "_apply_morphology_preset_to_doc"):
                    self._apply_morphology_preset_to_doc(base_doc, p)
                else:
                    from pro.morphology import apply_morphology_to_doc
                    apply_morphology_to_doc(base_doc, p)

                # optional logging
                try:
                    op   = p.get("operation", "erosion")
                    kern = p.get("kernel", 3)
                    it   = p.get("iterations", 1)
                    if hasattr(self, "_log"):
                        self._log(
                            f"[Replay] Morphology applied to base of "
                            f"'{target_sw.windowTitle()}' "
                            f"(op={op}, kernel={kern}, iter={it})"
                        )
                except Exception:
                    pass
            except Exception as e:
                try:
                    QMessageBox.warning(self, "Morphology", f"Replay-on-base failed:\n{e}")
                except Exception:
                    print("Morphology replay-on-base failed:", e)
            return
        if cid == "pixel_math":
            try:
                p = dict(preset or {})

                # Prefer helper that keeps replay state in sync
                if hasattr(self, "_apply_pixelmath_preset_to_doc"):
                    self._apply_pixelmath_preset_to_doc(base_doc, p)
                else:
                    from pro.pixelmath import apply_pixel_math_to_doc
                    apply_pixel_math_to_doc(self, base_doc, p)

                expr = (p.get("expr") or "").strip()
                if expr:
                    desc = expr
                else:
                    desc = (
                        f"R:{p.get('expr_r', '')} "
                        f"G:{p.get('expr_g', '')} "
                        f"B:{p.get('expr_b', '')}"
                    )

                try:
                    if hasattr(self, "_log"):
                        self._log(
                            f"[Replay] Pixel Math applied to base of "
                            f"'{target_sw.windowTitle()}' -> {desc}"
                        )
                except Exception:
                    pass
            except Exception as e:
                try:
                    QMessageBox.warning(self, "Pixel Math", f"Replay-on-base failed:\n{e}")
                except Exception:
                    print("Pixel Math replay-on-base failed:", e)
            return
        if cid == "halo_b_gon":
            try:
                p = dict(preset or {})

                if hasattr(self, "_apply_halobgon_preset_to_doc"):
                    self._apply_halobgon_preset_to_doc(base_doc, p)
                else:
                    from pro.halobgon import apply_halo_b_gon_to_doc
                    apply_halo_b_gon_to_doc(self, base_doc, p)

                lvl = int(p.get("reduction", 0))
                lin = bool(p.get("linear", False))
                try:
                    if hasattr(self, "_log"):
                        self._log(
                            f"[Replay] Halo-B-Gon applied to base of "
                            f"'{target_sw.windowTitle()}' "
                            f"(level={lvl}, linear={lin})"
                        )
                except Exception:
                    pass
            except Exception as e:
                try:
                    QMessageBox.warning(self, "Halo-B-Gon", f"Replay-on-base failed:\n{e}")
                except Exception:
                    print("Halo-B-Gon replay-on-base failed:", e)
            return
        if cid == "aberrationai":
            try:
                from pro.aberration_ai_preset import run_aberration_ai_via_preset
                # Apply the same preset, but explicitly on the base_doc
                run_aberration_ai_via_preset(self, preset or {}, doc=base_doc)

                pp = preset or {}
                auto = bool(pp.get("auto_gpu", True))
                prov = pp.get("provider", "auto" if auto else "CPUExecutionProvider")
                patch = int(pp.get("patch", 512))
                overlap = int(pp.get("overlap", 64))
                border = int(pp.get("border_px", 10))

                if hasattr(self, "_log"):
                    self._log(
                        f"[Replay] Aberration AI applied to base of "
                        f"'{target_sw.windowTitle()}' "
                        f"(patch={patch}, overlap={overlap}, border={border}px, provider={prov})"
                    )
            except Exception as e:
                try:
                    QMessageBox.warning(self, "Aberration AI", f"Replay-on-base failed:\n{e}")
                except Exception:
                    print("Replay Aberration AI failed:", e)
            return
        if cid == "cosmic_clarity":
            try:
                from pro.cosmicclarity_preset import run_cosmicclarity_via_preset

                # Normalize preset -> dict
                preset_dict = preset if isinstance(preset, dict) else {}
                run_cosmicclarity_via_preset(self, preset_dict, doc=base_doc)

                try:
                    m = preset_dict.get("mode", "sharpen")
                    self._log(
                        f"[Replay] Replayed Cosmic Clarity (mode={m}) "
                        f"on base of '{target_sw.windowTitle()}'"
                    )
                except Exception:
                    pass
            except Exception as e:
                try:
                    QMessageBox.warning(
                        self, "Cosmic Clarity",
                        f"Replay-on-base failed:\n{e}"
                    )
                except Exception:
                    print("Cosmic Clarity replay-on-base failed:", e)
            return

        # ---- For everything else, fall back to the normal command-drop behavior ----
        try:
            self._handle_command_drop(dict(payload), target_sw=target_sw)
        except Exception as e:
            QMessageBox.critical(self, "Replay Last Action", f"Replay failed:\n{e}")



    def _on_view_replay_last_requested(self, view):
        """
        Slot for ImageSubWindow.replayOnBaseRequested(view).
        Find the QMdiSubWindow that wraps this view and forward
        to replay_last_action_on_base().
        """
        target_sw = None
        if hasattr(self, "mdi"):
            for sw in self.mdi.subWindowList():
                if sw.widget() is view:
                    target_sw = sw
                    break

        # DEBUG
        try:
            self._log(
                f"[Replay] _on_view_replay_last_requested: view id={id(view)}, "
                f"found_subwindow={bool(target_sw)}"
            )
        except Exception:
            print(
                f"[Replay] _on_view_replay_last_requested: view id={id(view)}, "
                f"found_subwindow={bool(target_sw)}"
            )

        # ðŸ" For preview-tab replay -> run on base doc
        self.replay_last_action_on_base(target_sw=target_sw)



    # --- Command drop handling ----------------------------------------


    def _handle_command_drop(self, payload: dict, target_sw):
        # --- Debug: track raw calls ---
        cid_raw = (payload or {}).get("command_id")
        ts = time.monotonic()
        target_id = id(target_sw) if target_sw is not None else None

        # --- end debug header ---
        cid = payload.get("command_id")
        preset = payload.get("preset") or {}
    
        payload = payload or {}

        # 🔍 Global debug sniffer
        try:
            print(
                f"[HCD] DROP cid={cid!r}, target_sw={repr(target_sw)}, payload_keys={list((payload or {}).keys())}",
                flush=True,
            )
        except Exception:
            pass
        try:
            QApplication.processEvents()
        except Exception:
            pass
        def _extract_cid(p):
            # accept several shapes: "command_id": str | dict | list, or "command": {...}
            c = p.get("command_id")
            if isinstance(c, dict):
                c = c.get("id") or c.get("name") or c.get("command_id")
            elif isinstance(c, (list, tuple)):
                c = c[0] if c else None

            if not c:
                cmd = p.get("command")
                if isinstance(cmd, dict):
                    c = cmd.get("id") or cmd.get("name") or cmd.get("command_id")

            # last-ditch: stringify non-strings
            if c is None:
                c = ""
            if not isinstance(c, str):
                c = str(c)
            return c

        cid_raw = _extract_cid(payload)
        preset = payload.get("preset")
        if not isinstance(preset, dict):
            preset = {}

        def _cid_norm(c: str) -> str:
            c = (c or "").strip().lower()
            aliases = {
                # geometry short <->" long ids
                "flip_horizontal": "geom_flip_horizontal",
                "geom_flip_h": "geom_flip_horizontal",
                "geom_flip_horizontal": "geom_flip_horizontal",

                "flip_vertical": "geom_flip_vertical",
                "geom_flip_v": "geom_flip_vertical",
                "geom_rotate_clockwise": "geom_rotate_clockwise",
                "rotate_clockwise": "geom_rotate_clockwise",
                "geom_rot_cw": "geom_rotate_clockwise",

                "rotate_counterclockwise": "geom_rotate_counterclockwise",
                "geom_rot_ccw": "geom_rotate_counterclockwise",
                "geom_rotate_counterclockwise": "geom_rotate_counterclockwise",

                "rotate_180": "geom_rotate_180",
                "geom_rotate_180": "geom_rotate_180",

                "invert": "geom_invert",
                "geom_invert": "geom_invert",

                "rescale": "geom_rescale",
                "geom_rescale": "geom_rescale",

                "ghs": "ghs",
                "hyperbolic_stretch": "ghs",
                "universal_hyperbolic_stretch": "ghs",

                "abe": "abe",
                "automatic_background_extraction": "abe",

                "graxpert": "graxpert",
                "grax": "graxpert",
                "remove_gradient_graxpert": "graxpert",

                "remove_stars": "remove_stars",
                "star_removal": "remove_stars",
                "starnet": "remove_stars",
                "darkstar": "remove_stars",

                "aberrationai": "aberrationai",
                "aberration": "aberrationai",
                "ai_aberration": "aberrationai",

                "cosmic": "cosmic_clarity",
                "cosmicclarity": "cosmic_clarity",
                "cosmic_clarity": "cosmic_clarity",

                "crop": "crop",
                "geom_crop": "crop",

                "wavescale_hdr": "wavescale_hdr",
                "wavescalehdr": "wavescale_hdr",
                "wavescale": "wavescale_hdr",

                "wavescale_dark_enhance": "wavescale_dark_enhance",
                "wavescale_dark_enhancer": "wavescale_dark_enhance",
                "wsde": "wavescale_dark_enhance",
                "dark_enhancer": "wavescale_dark_enhance",

                "star_alignment": "star_align",
                "align_stars": "star_align",
                "align": "star_align",

                "convo": "convo",
                "convolution": "convo",
                "deconvolution": "convo",
                "convo_deconvo": "convo",
            }
            return aliases.get(c, c)

        cid = _cid_norm(cid_raw)

        def _call_any(method_names: list[str], *args, **kwargs) -> bool:
            for name in method_names:
                m = getattr(self, name, None)
                if callable(m):
                    m(*args, **kwargs)
                    return True
            return False

        # ----- Function bundle: run a sequence of steps on the target view(s) -----
        # ----- Function bundle: run a sequence of steps on the target view(s) -----
        if cid in ("function_bundle", "bundle_functions"):
            from PyQt6.QtWidgets import QApplication

            payload = payload or {}
            steps   = list((payload or {}).get("steps") or [])
            inherit = bool((payload or {}).get("inherit_target", True))

            print(
                f"[HCD] ENTER function_bundle: inherit={inherit}, target_sw={repr(target_sw)}, steps={len(steps)}, payload={payload!r}",
                flush=True,
            )
            QApplication.processEvents()

            if not steps:
                print("[HCD] function_bundle: NO STEPS, returning", flush=True)
                QApplication.processEvents()
                return

            # If user dropped onto the background (no direct target), support 'targets'
            if target_sw is None:
                targets = (payload or {}).get("targets")
                print(f"[HCD] function_bundle: NO target_sw, targets={targets!r}", flush=True)
                QApplication.processEvents()

                if targets == "all_open":
                    print("[HCD] function_bundle: apply to ALL OPEN subwindows", flush=True)
                    QApplication.processEvents()
                    for sw in list(self.mdi.subWindowList()):
                        print(f"[HCD]   all_open -> sw={repr(sw)}", flush=True)
                        QApplication.processEvents()
                        for i, st in enumerate(steps, start=1):
                            if not isinstance(st, dict) or not st.get("command_id"):
                                print(f"[HCD]     skip step[{i}]: bad payload {st!r}", flush=True)
                                QApplication.processEvents()
                                continue
                            cid2 = st.get("command_id")
                            print(f"[HCD]     BEGIN step[{i}/{len(steps)}] cid={cid2}", flush=True)
                            QApplication.processEvents()
                            try:
                                self._handle_command_drop(st, target_sw=sw)
                                print(f"[HCD]     END   step[{i}/{len(steps)}] cid={cid2} OK", flush=True)
                            except Exception as e:
                                print(f"[HCD]     END   step[{i}/{len(steps)}] cid={cid2} ERROR={e!r}", flush=True)
                            QApplication.processEvents()
                    print("[HCD] EXIT function_bundle (all_open)", flush=True)
                    QApplication.processEvents()
                    return

                if isinstance(targets, (list, tuple)):
                    print(f"[HCD] function_bundle: apply to explicit targets={targets!r}", flush=True)
                    QApplication.processEvents()
                    for ptr in targets:
                        try:
                            doc, sw = self._find_doc_by_id(int(ptr))
                            print(f"[HCD]   _find_doc_by_id({ptr}) -> sw={repr(sw)}", flush=True)
                        except Exception as e:
                            print(f"[HCD]   _find_doc_by_id({ptr}) ERROR={e!r}", flush=True)
                            sw = None
                        QApplication.processEvents()
                        if sw:
                            for i, st in enumerate(steps, start=1):
                                if not isinstance(st, dict) or not st.get("command_id"):
                                    print(f"[HCD]     skip step[{i}]: bad payload {st!r}", flush=True)
                                    QApplication.processEvents()
                                    continue
                                cid2 = st.get("command_id")
                                print(f"[HCD]     BEGIN step[{i}/{len(steps)}] cid={cid2}", flush=True)
                                QApplication.processEvents()
                                try:
                                    self._handle_command_drop(st, target_sw=sw)
                                    print(f"[HCD]     END   step[{i}/{len(steps)}] cid={cid2} OK", flush=True)
                                except Exception as e:
                                    print(f"[HCD]     END   step[{i}/{len(steps)}] cid={cid2} ERROR={e!r}", flush=True)
                                QApplication.processEvents()
                    print("[HCD] EXIT function_bundle (explicit targets)", flush=True)
                    QApplication.processEvents()
                    return

                # No target info -> open Function Bundles UI
                print("[HCD] function_bundle: no target info, opening Function Bundles UI", flush=True)
                QApplication.processEvents()
                try:
                    from pro.function_bundle import show_function_bundles
                    show_function_bundles(self)
                except Exception as e:
                    print(f"[HCD] show_function_bundles ERROR={e!r}", flush=True)
                    QApplication.processEvents()
                print("[HCD] EXIT function_bundle (UI path)", flush=True)
                QApplication.processEvents()
                return

            # We DO have an explicit target subwindow -> run the sequence there.
            print(f"[HCD] function_bundle: USING target_sw={repr(target_sw)}", flush=True)
            QApplication.processEvents()

            for i, st in enumerate(steps, start=1):
                if not isinstance(st, dict) or not st.get("command_id"):
                    print(f"[HCD]   skip step[{i}]: bad payload {st!r}", flush=True)
                    QApplication.processEvents()
                    continue

                cid2 = st.get("command_id")
                is_cc = str(cid2).lower().startswith("cosmic")
                print(
                    f"[HCD]   BEGIN step[{i}/{len(steps)}]{' (CC)' if is_cc else ''} cid={cid2}, payload={st!r}",
                    flush=True,
                )
                QApplication.processEvents()

                try:
                    self._handle_command_drop(st, target_sw=target_sw if inherit else None)
                    print(
                        f"[HCD]   END   step[{i}/{len(steps)}]{' (CC)' if is_cc else ''} cid={cid2} OK",
                        flush=True,
                    )
                except Exception as e:
                    print(
                        f"[HCD]   END   step[{i}/{len(steps)}]{' (CC)' if is_cc else ''} cid={cid2} ERROR={e!r}",
                        flush=True,
                    )

                QApplication.processEvents()

            print("[HCD] EXIT function_bundle (explicit target)", flush=True)
            QApplication.processEvents()
            return
        # --- Bundle runner ----------------------------------------
        if cid in ("bundle", "__bundle_exec__"):
            steps = list((payload or {}).get("steps") or [])
            if not steps:
                return

            # Resolve targets
            targets = (payload or {}).get("targets", None)
            def _iter_bundle_targets():
                # explicit list of doc pointers
                if isinstance(targets, (list, tuple)):
                    for ptr in targets:
                        try:
                            d, sw = self._find_doc_by_id(int(ptr))
                        except Exception:
                            d, sw = None, None
                        if sw is not None:
                            yield sw
                    return
                # special keywords
                if isinstance(targets, str) and targets.lower() == "all_open":
                    for sw in self.mdi.subWindowList():
                        if sw and sw.isVisible():
                            yield sw
                    return
                # default: the explicit drop target, else the active subwindow
                if target_sw is not None:
                    yield target_sw
                    return
                sw = self.mdi.activeSubWindow()
                if sw:
                    yield sw

            stop_on_error = bool((payload or {}).get("stop_on_error", False))

            # Optional: light progress text in Console
            try: self._log(f"Bundle: {len(steps)} step(s) -> {targets or 'target view'}")
            except Exception as e:
                import logging
                logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")


            for sw in _iter_bundle_targets():
                if sw is None: 
                    continue
                title = getattr(sw, "windowTitle", lambda: "view")()
                for i, sp in enumerate(steps, start=1):
                    try:
                        # allow nested bundles but guard against weird payloads
                        if not isinstance(sp, dict) or "command_id" not in sp:
                            continue
                        QCoreApplication.processEvents()
                        # Reuse this dispatcher on the specific subwindow target
                        self._handle_command_drop(sp, sw)
                        QCoreApplication.processEvents()
                        try: self._log(f"Bundle [{i}/{len(steps)}] on '{title}' -> {sp.get('command_id')}")
                        except Exception as e:
                            import logging
                            logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")
                    except Exception as e:
                        try: self._log(f"Bundle step failed on '{title}': {e}")
                        except Exception as e:
                            import logging
                            logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")
                        if stop_on_error:
                            QMessageBox.warning(self, "Bundle", f"Stopped on error:\n{e}")
                            return
                        # else continue to next step

            try: self._log("Bundle complete.")
            except Exception as e:
                import logging
                logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")
            return

        # ------------------- No target subwindow -> open UIs / active ops -------------------
        if target_sw is None:
            if cid == "stat_stretch":
                self._open_statistical_stretch_with_preset(preset); return
            if cid == "star_stretch":
                self._open_star_stretch_with_preset(preset); return
            if cid == "remove_green":
                from pro.remove_green import open_remove_green_dialog
                open_remove_green_dialog(self, preset); return
            if cid == "extract_luminance":
                self._extract_luminance(doc=None); return
            if cid == "recombine_luminance":
                self._recombine_luminance_ui(target_doc=None); return
            if cid == "wavescale_hdr":
                self._open_wavescale_hdr(); return
            if cid == "wavescale_dark_enhance":
                self._open_wavescale_dark_enhance(); return
            if cid == "clahe":
                self._open_clahe(); return
            if cid == "pixel_math":
                self._open_pixel_math(); return
            if cid == "halo_b_gon":
                self._open_halo_b_gon(); return
            if cid == "rgb_combine":
                self._open_rgb_combination(); return
            if cid == "curves":
                from pro.curves_preset import open_curves_with_preset
                open_curves_with_preset(self, preset)
                return
            if cid == "crop":
                try:
                    from pro.crop_preset import run_crop_via_preset
                    run_crop_via_preset(self, preset or {}, target_doc=None)
                    self._log("Ran Crop headlessly on active view.")
                except Exception as e:
                    QMessageBox.warning(self, "Crop", f"Apply failed:\n{e}")
                return
            
            if cid == "star_align":
                try:
                    from pro.star_alignment_preset import run_star_alignment_via_preset
                    run_star_alignment_via_preset(self, preset or {}, target_doc=doc)
                    rp = preset or {}
                    rm = str(rp.get("ref_mode", "active"))
                    rn = (rp.get("ref_name") or os.path.basename(rp.get("ref_file","")) or
                        (str(rp.get("ref_ptr")) if rp.get("ref_ptr") is not None else "active"))
                    self._log(f"Ran Star Alignment (ref={rm}:{rn}, overwrite={'yes' if rp.get('overwrite', False) else 'no'})")
                except Exception as e:
                    
                    QMessageBox.warning(self, "Star Alignment", f"Apply failed:\n{e}")
                return            
            if target_sw is None:
                if cid == "ghs":
                    from pro.ghs_preset import open_ghs_with_preset
                    open_ghs_with_preset(self, preset)
                    return

            # Fallback: trigger QAction by cid (ok when no target)
            act = self._find_action_by_cid(cid)
            if act:
                act.trigger()
            return


        # ------------------- Dropped on a specific subwindow -> HEADLESS APPLY -------------------
        view = target_sw.widget()
        doc = getattr(view, "document", None)

        # NEW: resolve the base document (for Preview tabs, this is the full-frame doc)
        base_doc = getattr(view, "base_document", None)
        if base_doc is None:
            base_doc = doc

        # --- Existing image-processing blocks (unchanged) ---
        if cid == "stat_stretch":
            try:
                self._apply_stat_stretch_preset_to_doc(doc, preset)
                self._log(f"Applied Statistical Stretch preset to '{target_sw.windowTitle()}'")
            except Exception as e:
                QMessageBox.warning(self, "Preset apply failed", str(e))
            return

        if cid == "star_stretch":
            try:
                self._apply_star_stretch_preset_to_doc(doc, preset)
                self._log(f"Applied Star Stretch preset to '{target_sw.windowTitle()}'")
            except Exception as e:
                QMessageBox.warning(self, "Preset apply failed", str(e))
            return

        if cid == "curves":
            try:
                from pro.curves_preset import apply_curves_via_preset
                apply_curves_via_preset(self, doc, preset or {})
                self._log(f"Applied Curves preset to '{target_sw.windowTitle()}'")
            except Exception as e:
                
                QMessageBox.warning(self, "Curves", f"Apply failed:\n{e}")
            return

        if cid == "ghs":
            try:
                from pro.ghs_preset import apply_ghs_via_preset
                apply_ghs_via_preset(self, doc, preset or {})
                self._log(f"Applied GHS preset to '{target_sw.windowTitle()}'")
            except Exception as e:
                
                QMessageBox.warning(self, "GHS", f"Apply failed:\n{e}")
            return
        if cid == "crop":
            try:
                from pro.crop_preset import apply_crop_via_preset
                apply_crop_via_preset(self, doc, preset or {})
                self._log(f"Applied Crop preset to '{target_sw.windowTitle()}'")
            except Exception as e:
                QMessageBox.warning(self, "Crop", f"Apply failed:\n{e}")
            return

        if cid == "pedestal":
            from pro.pedestal import remove_pedestal
            remove_pedestal(self, target_doc=doc)
            return

        if cid == "abe":
            try:
                from pro.abe_preset import apply_abe_via_preset
                apply_abe_via_preset(self, doc, preset or {})
                self._log(f"Applied ABE preset to '{target_sw.windowTitle()}' (no exclusions)")
            except Exception as e:
                QMessageBox.warning(self, "ABE", f"Apply failed:\n{e}")
            return

        if cid == "graxpert":
            from pro.graxpert_preset import run_graxpert_via_preset

            if doc is None or getattr(doc, "image", None) is None:
                QMessageBox.warning(self, "GraXpert", "Target document has no image.")
                return

            p = preset or {}
            op = str(p.get("op", "background")).lower()  # "background" or "denoise"

            # Run headless on this specific document (ROI or full frame)
            run_graxpert_via_preset(self, p, target_doc=doc)

            # Logging
            gpu_val = bool(p.get("gpu", True))
            if op == "denoise":
                s_raw = p.get("strength", 0.50)
                try:
                    s_val = float(s_raw)
                except Exception:
                    s_val = 0.50
                ai_ver = p.get("ai_version") or "latest"
                self._log(
                    f"Ran GraXpert Denoise "
                    f"(strength={s_val:.2f}, model={ai_ver}, gpu={'on' if gpu_val else 'off'})"
                )
            else:
                s_raw = p.get("smoothing", 0.10)
                try:
                    s_val = float(s_raw)
                except Exception:
                    s_val = 0.10
                self._log(
                    f"Ran GraXpert Gradient Removal "
                    f"(smoothing={round(s_val, 2)}, gpu={'on' if gpu_val else 'off'})"
                )
            return


        if cid == "convo":
            try:
                self._apply_convo_preset_to_doc(doc, preset or {})
            except Exception as e:
                QMessageBox.warning(self, "Convo/Deconvo", f"Apply failed:\n{e}")
            return

        if cid == "remove_stars":
            from pro.remove_stars_preset import run_remove_stars_via_preset

            if doc is None or getattr(doc, "image", None) is None:
                QMessageBox.warning(self, "Remove Stars", "Target document has no image.")
                return

            # Run headless on this specific document (ROI or full frame)
            run_remove_stars_via_preset(self, preset or {}, target_doc=doc)

            # safe logging (no nested format specs)
            tool = str((preset or {}).get("tool", "starnet"))
            if tool.lower().startswith("star"):
                lin = bool((preset or {}).get("linear", True))
                self._log(f"Ran Remove Stars (tool=StarNet, linear={'yes' if lin else 'no'})")
            else:
                mode = (preset or {}).get("mode", "unscreen")
                stride = int((preset or {}).get("stride", 512))
                gpu = not bool((preset or {}).get("disable_gpu", False))
                show = bool((preset or {}).get("show_extracted_stars", True))
                self._log(
                    f"Ran Remove Stars (tool=DarkStar, mode={mode}, "
                    f"stride={stride}, gpu={'on' if gpu else 'off'}, "
                    f"stars={'on' if show else 'off'})"
                )
            return


        if cid == "aberrationai":
            from pro.aberration_ai_preset import run_aberration_ai_via_preset
            run_aberration_ai_via_preset(self, preset or {})
            # safe, simple log
            pp = preset or {}
            auto = bool(pp.get("auto_gpu", True))
            prov = pp.get("provider", "auto" if auto else "CPUExecutionProvider")
            self._log(f"Ran Aberration AI (patch={int(pp.get('patch',512))}, overlap={int(pp.get('overlap',64))}, border={int(pp.get('border_px',10))}px, provider={prov})")
            return

        if cid == "cosmic_clarity":
            from pro.cosmicclarity_preset import run_cosmicclarity_via_preset

            # resolve doc from the target subwindow if present
            doc = None
            try:
                if target_sw is not None:
                    vw = target_sw.widget()
                    doc = getattr(vw, "document", None)
            except Exception:
                doc = None

            run_cosmicclarity_via_preset(self, preset or {}, doc=doc)  # <-- pass doc

            try:
                m = (preset or {}).get("mode", "sharpen")
                self._log(f"Ran Cosmic Clarity (mode={m})")
            except Exception:
                pass
            return

        if cid == "linear_fit":
            try:
                doc = self.doc_manager.get_active_document()
                from pro.linear_fit import apply_linear_fit_via_preset
                apply_linear_fit_via_preset(self, self.doc_manager, doc, preset or {})
                self._log("Applied Linear Fit")
            except Exception as e:
                QMessageBox.warning(self, "Linear Fit", f"Apply failed:\n{e}")
            return

        if cid == "remove_green":
            try:
                from pro.remove_green import apply_remove_green_preset_to_doc

                # Normalize any legacy keys into a canonical preset
                raw = preset if isinstance(preset, dict) else {}
                amt = float(raw.get("amount",
                                    raw.get("strength",
                                            raw.get("value", 1.0))))
                mode = str(raw.get("mode",
                                   raw.get("neutral_mode", "avg"))).lower()
                preserve = bool(raw.get("preserve_lightness",
                                        raw.get("preserve", True)))

                preset_dict = {
                    "amount": amt,
                    "mode": mode,
                    "preserve_lightness": preserve,
                }

                # Apply to the current doc (ROI or full view)
                apply_remove_green_preset_to_doc(self, doc, preset_dict)

                # Record for Replay Last Action so preview -> base works
                try:
                    self._last_headless_command = {
                        "command_id": "remove_green",
                        "preset": dict(preset_dict),
                    }
                    if hasattr(self, "_log"):
                        self._log(
                            f"[Replay] Recorded Remove Green preset from command drop "
                            f"(amount={amt:.2f}, mode={mode}, "
                            f"preserve_lightness={'yes' if preserve else 'no'})"
                        )
                except Exception:
                    # Don't let logging break the command
                    pass

            except Exception as e:
                QMessageBox.warning(self, "Remove Green", str(e))
            return


        if cid == "star_align":
            try:
                from pro.star_alignment_preset import run_star_alignment_via_preset
                run_star_alignment_via_preset(self, preset or {}, target_doc=doc)
                # simple, robust logging
                rp = preset or {}
                rm = rp.get("ref_mode", "active")
                rn = rp.get("ref_name") or os.path.basename(rp.get("ref_file","")) or "active"
                self._log(f"Ran Star Alignment (ref={rm}:{rn}, overwrite={'yes' if rp.get('overwrite', False) else 'no'})")
            except Exception as e:
                
                QMessageBox.warning(self, "Star Alignment", f"Apply failed:\n{e}")
            return

        if cid == "rgb_align":
            # 1) find the document
            doc = None
            view = None
            if target_sw is not None:
                view = target_sw.widget()
                doc = getattr(view, "document", None)
            if doc is None and hasattr(self, "_active_doc"):
                # your existing helper
                doc = self._active_doc()

            if not doc or getattr(doc, "image", None) is None:
                QMessageBox.information(self, "RGB Align", "No image in the dropped/active view.")
                return

            # 2) if we were called WITH a preset -> headless
            #    (this is what shortcuts / Alt-drag will supply)
            if preset:
                self._log(f"Running RGB Align headlessly on '{target_sw.windowTitle()}'")
                QApplication.processEvents()
                try:
                    from pro.rgbalign import run_rgb_align_headless
                    
                    run_rgb_align_headless(self, doc, preset)
                except Exception as e:
                    QMessageBox.critical(self, "RGB Align", f"Headless failed:\n{e}")
                return

            # 3) otherwise -> open the dialog like before
            try:
                from pro.rgbalign import RGBAlignDialog
                dlg = RGBAlignDialog(parent=self, document=view or doc)
                try:
                    dlg.setWindowIcon(QIcon(rgbalign_path))
                except Exception:
                    pass
                dlg.show()
            except Exception as e:
                QMessageBox.critical(self, "RGB Align", f"Open failed:\n{e}")
            return


        if cid == "background_neutral":
            try:
                self._apply_background_neutral_preset_to_doc(doc, preset)
                self._log(f"Background Neutralization applied to '{target_sw.windowTitle()}'")
            except Exception as e:
                QMessageBox.warning(self, "Background Neutralization", str(e))
            return

        if cid == "white_balance":
            try:
                # Normalize preset -> dict and supply sensible defaults
                preset_dict = preset if isinstance(preset, dict) else {}
                if not preset_dict:
                    preset_dict = {
                        "mode": "star",
                        "threshold": 50.0,
                        "reuse_cached_sources": True,
                    }

                # Apply to the *current* doc (ROI or full), just like before
                self._apply_white_balance_preset_to_doc(doc, preset_dict)

                # Record for Replay Last Action so preview -> base replay works
                try:
                    self._last_headless_command = {
                        "command_id": "white_balance",
                        "preset": preset_dict,
                    }

                    # Optional: nice logging
                    mode = str(preset_dict.get("mode", "star")).lower()
                    if mode == "manual":
                        r = float(preset_dict.get("r_gain", 1.0))
                        g = float(preset_dict.get("g_gain", 1.0))
                        b = float(preset_dict.get("b_gain", 1.0))
                        self._log(
                            f"[Replay] Recorded White Balance preset from command drop "
                            f"(mode=manual, R={r:.3f}, G={g:.3f}, B={b:.3f})"
                        )
                    elif mode == "auto":
                        self._log(
                            "[Replay] Recorded White Balance preset from command drop (mode=auto)"
                        )
                    else:
                        thr = float(preset_dict.get("threshold", 50.0))
                        reuse = bool(preset_dict.get("reuse_cached_sources", True))
                        self._log(
                            f"[Replay] Recorded White Balance preset from command drop "
                            f"(mode=star, threshold={thr:.1f}, "
                            f"reuse={'yes' if reuse else 'no'})"
                        )
                except Exception:
                    # Recording/logging must never break the command
                    pass

                # Existing log about the actual apply
                self._log(f"White Balance applied to '{target_sw.windowTitle()}'")

            except Exception as e:
                QMessageBox.warning(self, "White Balance", str(e))
            return


        if cid == "wavescale_hdr":
            from pro.wavescale_hdr_preset import run_wavescale_hdr_via_preset
            run_wavescale_hdr_via_preset(self, preset or {}, target_doc=doc)
            # safe, readable log
            pp = preset or {}
            ns = int(pp.get("n_scales", 5))
            try:
                comp = float(pp.get("compression_factor", 1.5))
            except Exception:
                comp = 1.5
            try:
                mg = float(pp.get("mask_gamma", 5.0))
            except Exception:
                mg = 5.0
            self._log(f"Ran WaveScale HDR (n_scales={ns}, compression={comp:.2f}, mask_gamma={mg:.2f})")
            return

        if cid == "wavescale_dark_enhance":
            try:
                from pro.wavescalede_preset import run_wavescalede_via_preset
                run_wavescalede_via_preset(self, preset or {}, target_doc=doc)
                pp = preset or {}
                ns   = int(pp.get("n_scales", 6))
                try: bf = float(pp.get("boost_factor", 5.0))
                except Exception: bf = 5.0
                try: mg = float(pp.get("mask_gamma", 1.0))
                except Exception: mg = 1.0
                it   = int(pp.get("iterations", 2))
                self._log(f"Ran WaveScale Dark Enhancer (n_scales={ns}, boost={bf:.2f}, mask_gamma={mg:.2f}, iters={it})")
            except Exception as e:
                
                QMessageBox.warning(self, "WaveScale Dark Enhancer", f"Apply failed:\n{e}")
            return


        if cid == "extract_luminance":
            try:
                self._extract_luminance(doc=doc)
                self._log(f"Extract Luminance -> new doc from '{target_sw.windowTitle()}'")
            except Exception as e:
                QMessageBox.warning(self, "Extract Luminance", str(e))
            return

        if cid == "recombine_luminance":
            self._recombine_luminance_ui(target_doc=doc)
            self._log(f"Recombined Luminance -> doc '{target_sw.windowTitle()}'")
            return

        if cid == "rgb_extract":
            self._rgb_extract_on_doc(doc, base_title=target_sw.windowTitle())
            self._log(f"Extracted R, G, and B channels from '{target_sw.windowTitle()}'")
            return

        if cid == "blemish_blaster":
            from pro.blemish_blaster import BlemishBlasterDialogPro
            dlg = BlemishBlasterDialogPro(self, doc)
            try: dlg.setWindowIcon(QIcon(blastericon_path))
            except Exception as e:
                import logging
                logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")
            dlg.resize(900, 650)
            dlg.show()
            return



        if cid == "wavescale_hdr":
            # (unchanged block)
            from pro.wavescale_hdr import compute_wavescale_hdr
            try:
                img = np.asarray(doc.image, dtype=np.float32)
                if img.ndim == 2:
                    base_rgb = np.repeat(img[:, :, None], 3, axis=2); was_mono, mono_shape = True, img.shape
                elif img.ndim == 3 and img.shape[2] == 1:
                    base_rgb = np.repeat(img, 3, axis=2); was_mono, mono_shape = True, img.shape
                else:
                    base_rgb = img[:, :, :3]; was_mono, mono_shape = False, None

                n_scales = int(preset.get("n_scales", 5))
                compression_factor = float(preset.get("compression_factor", 1.5))
                mask_gamma = float(preset.get("mask_gamma", 5.0))

                transformed, mask = compute_wavescale_hdr(
                    np.clip(base_rgb, 0, 1),
                    n_scales=n_scales, compression_factor=compression_factor, mask_gamma=mask_gamma
                )
                m3 = np.repeat(mask[..., None], 3, axis=2)
                blended = base_rgb * (1.0 - m3) + transformed * m3

                if was_mono:
                    out = np.mean(blended, axis=2, dtype=np.float32)
                    if len(mono_shape) == 3 and mono_shape[2] == 1:
                        out = out[:, :, None]
                else:
                    out = blended

                out = np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)
                if hasattr(doc, "set_image"): doc.set_image(out, step_name="WaveScale HDR")
                elif hasattr(doc, "apply_numpy"): doc.apply_numpy(out, step_name="WaveScale HDR")
                else: doc.image = out

                self._log(f"WaveScale HDR applied to '{target_sw.windowTitle()}' (n_scales={n_scales}, comp={compression_factor}, gamma={mask_gamma})")
            except Exception as e:
                QMessageBox.warning(self, "WaveScale HDR", f"Preset apply failed:\n{e}")
            return

        if cid == "wavescale_dark_enhance":
            n_scales     = int(preset.get("n_scales", 6))
            boost_factor = float(preset.get("boost_factor", 5.0))
            mask_gamma   = float(preset.get("mask_gamma", 1.0))
            iterations   = int(preset.get("iterations", 2))
            try:
                if hasattr(self, "_apply_wavescale_dark_enhance_preset_to_doc"):
                    self._apply_wavescale_dark_enhance_preset_to_doc(doc, {
                        "n_scales": n_scales, "boost_factor": boost_factor,
                        "mask_gamma": mask_gamma, "iterations": iterations,
                    })
                else:
                    from pro.wavescalede import compute_wavescale_dse
                    img = np.asarray(doc.image, dtype=np.float32)
                    if img.size:
                        mx = float(np.nanmax(img))
                        if np.isfinite(mx) and mx > 1.0:
                            img = img / mx
                    img = np.clip(img, 0.0, 1.0).astype(np.float32, copy=False)
                    out, _ = compute_wavescale_dse(
                        img, n_scales=n_scales, boost_factor=boost_factor,
                        mask_gamma=mask_gamma, iterations=iterations
                    )
                    out = np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)
                    if hasattr(doc, "set_image"): doc.set_image(out, step_name="WaveScale Dark Enhance")
                    elif hasattr(doc, "apply_numpy"): doc.apply_numpy(out, step_name="WaveScale Dark Enhance")
                    else: doc.image = out
                self._log(f"WaveScale Dark Enhancer applied to '{target_sw.windowTitle()}' (n_scales={n_scales}, boost={boost_factor}, gamma={mask_gamma}, iter={iterations})")
            except Exception as e:
                QMessageBox.warning(self, "WaveScale Dark Enhancer", f"Preset apply failed:\n{e}")
            return

        if cid == "clahe":
            try:
                self._apply_clahe_preset_to_doc(doc, preset)
                self._log(f"CLAHE applied to '{target_sw.windowTitle()}'")
            except Exception as e:
                QMessageBox.warning(self, "CLAHE", str(e))
            return

        if cid == "morphology":
            try:
                if hasattr(self, "_apply_morphology_preset_to_doc"):
                    self._apply_morphology_preset_to_doc(doc, preset)
                else:
                    from pro.morphology import apply_morphology_to_doc
                    apply_morphology_to_doc(doc, preset)

                self._log(f"Morphology applied to '{target_sw.windowTitle()}'")
            except Exception as e:
                QMessageBox.warning(self, "Morphology", str(e))
            return

        if cid == "pixel_math":
            try:
                p = dict(preset or {})

                if hasattr(self, "_apply_pixelmath_preset_to_doc"):
                    self._apply_pixelmath_preset_to_doc(doc, p)
                else:
                    from pro.pixelmath import apply_pixel_math_to_doc
                    apply_pixel_math_to_doc(self, doc, p)

                expr = (p.get("expr") or "").strip()
                if expr:
                    desc = expr
                else:
                    desc = (
                        f"R:{p.get('expr_r', '')} "
                        f"G:{p.get('expr_g', '')} "
                        f"B:{p.get('expr_b', '')}"
                    )

                self._log(f"Pixel Math applied to '{target_sw.windowTitle()}' -> {desc}")
            except Exception as e:
                QMessageBox.warning(self, "Pixel Math", f"Preset apply failed:\n{e}")
            return


        if cid == "signature_insert":
            try:
                from pro.signature_insert import apply_signature_preset_to_doc
                out = apply_signature_preset_to_doc(doc, preset)
                if hasattr(doc, "set_image"): doc.set_image(out, step_name="Signature / Insert")
                elif hasattr(doc, "apply_numpy"): doc.apply_numpy(out, step_name="Signature / Insert")
                else: doc.image = out
                fp = preset.get("file_path", "<file>"); pos = preset.get("position", "bottom_right")
                self._log(f"Signature preset applied to '{target_sw.windowTitle()}' (file={fp}, pos={pos}, scale={preset.get('scale',100)}%, rot={preset.get('rotation',0)}Â deg, op={preset.get('opacity',100)}%)")
            except Exception as e:
                QMessageBox.warning(self, "Signature / Insert", f"Preset apply failed:\n{e}")
            return

        if cid == "halo_b_gon":
            try:
                p = dict(preset or {})

                if hasattr(self, "_apply_halobgon_preset_to_doc"):
                    self._apply_halobgon_preset_to_doc(doc, p)
                else:
                    from pro.halobgon import apply_halo_b_gon_to_doc
                    apply_halo_b_gon_to_doc(self, doc, p)

                lvl = int(p.get("reduction", 0))
                lin = bool(p.get("linear", False))
                self._log(
                    f"Halo-B-Gon applied to '{target_sw.windowTitle()}' "
                    f"(level={lvl}, linear={lin})"
                )
            except Exception as e:
                QMessageBox.warning(self, "Halo-B-Gon", f"Preset apply failed:\n{e}")
            return


        # ----- Geometry (headless; accept all old/new ids) -----
        if cid == "geom_invert":
            try:
                _call_any(["_apply_geom_invert_to_doc"], doc)
                self._log(f"Inverted '{target_sw.windowTitle()}'")
            except Exception as e:
                QMessageBox.warning(self, "Invert", str(e))
            return

        if cid == "geom_flip_horizontal":
            try:
                called = _call_any(["_apply_geom_flip_h_to_doc", "_apply_geom_flip_horizontal_to_doc"], doc)
                if not called:
                    raise RuntimeError("No flip-horizontal apply method found")
                self._log(f"Flip Horizontal applied to '{target_sw.windowTitle()}'")
            except Exception as e:
                QMessageBox.warning(self, "Flip Horizontal", str(e))
            return

        if cid == "geom_flip_vertical":
            try:
                called = _call_any(["_apply_geom_flip_v_to_doc", "_apply_geom_flip_vertical_to_doc"], doc)
                if not called:
                    raise RuntimeError("No flip-vertical apply method found")
                self._log(f"Flip Vertical applied to '{target_sw.windowTitle()}'")
            except Exception as e:
                QMessageBox.warning(self, "Flip Vertical", str(e))
            return

        if cid == "geom_rotate_clockwise":
            try:
                called = _call_any(["_apply_geom_rot_cw_to_doc", "_apply_geom_rotate_cw_to_doc"], doc)
                if not called:
                    raise RuntimeError("No rotate-cw apply method found")
                self._log(f"Rotate 90Â deg CW applied to '{target_sw.windowTitle()}'")
            except Exception as e:
                QMessageBox.warning(self, "Rotate 90Â deg CW", str(e))
            return

        if cid == "geom_rotate_counterclockwise":
            try:
                called = _call_any(["_apply_geom_rot_ccw_to_doc", "_apply_geom_rotate_ccw_to_doc"], doc)
                if not called:
                    raise RuntimeError("No rotate-ccw apply method found")
                self._log(f"Rotate 90Â deg CCW applied to '{target_sw.windowTitle()}'")
            except Exception as e:
                QMessageBox.warning(self, "Rotate 90Â deg CCW", str(e))
            return

        if cid == "geom_rotate_180":
            try:
                called = _call_any(["_apply_geom_rot_180_to_doc"], doc)
                if not called:
                    raise RuntimeError("No rotate-180 apply method found")
                self._log(f"Rotate 180Â deg applied to '{target_sw.windowTitle()}'")
            except Exception as e:
                QMessageBox.warning(self, "Rotate 180Â deg", str(e))
            return

        if cid == "geom_rescale":
            try:
                factor = float(preset.get("factor", 1.0))
                # support both names you've used
                called = _call_any(
                    ["_apply_rescale_preset_to_doc", "_apply_geom_rescale_to_doc"],
                    doc, {"factor": factor} if "_apply_rescale_preset_to_doc" in dir(self) else factor
                )
                if not called:
                    # last resort: try signature (doc, preset)
                    _call_any(["_apply_rescale_preset_to_doc"], doc, {"factor": factor})
                self._log(f"Rescale Ã--{factor:g} applied to '{target_sw.windowTitle()}'")
            except Exception as e:
                QMessageBox.warning(self, "Rescale", str(e))
            return
        if cid == "debayer":
            try:
                # Resolve a document from (a) the provided 'view' (DnD) or (b) the active subwindow.
                def _resolve_doc(v):
                    if v is None:
                        return None
                    # direct document?
                    if hasattr(v, "document"):
                        return v.document
                    # some wrappers expose a callable or attr named 'widget'
                    wattr = getattr(v, "widget", None)
                    if callable(wattr):
                        try:
                            w = wattr()
                            if hasattr(w, "document"):
                                return w.document
                        except Exception:
                            pass
                    elif wattr is not None and hasattr(wattr, "document"):
                        return wattr.document
                    # some subwindows use 'view' instead of 'widget'
                    vattr = getattr(v, "view", None)
                    if vattr is not None and hasattr(vattr, "document"):
                        return vattr.document
                    return None

                # 1) try the view passed by the shortcuts manager (DnD path)
                doc = _resolve_doc(view)

                # 2) fallback to active subwindow
                if doc is None:
                    sw = self.mdi.activeSubWindow()
                    doc = _resolve_doc(sw)

                if doc is None or getattr(doc, "image", None) is None:
                    QMessageBox.information(self, "Debayer", "No active image.")
                    return

                dm = getattr(self, "doc_manager", None) or getattr(self, "docman", None)
                from pro.debayer import apply_debayer_preset_to_doc  # ensure imported
                pattern_used, _ = apply_debayer_preset_to_doc(dm, doc, preset or {})

                # get a friendly title for logs
                title = None
                dn = getattr(doc, "display_name", None)
                if callable(dn):
                    title = dn()
                if not title:
                    title = getattr(doc, "name", None) or getattr(doc, "title", None) or "Untitled"

                self._log(f"Applied Debayer ({pattern_used}) to '{title}'")
            except Exception as e:
                QMessageBox.warning(self, "Debayer failed", str(e))
            return   
        if cid == "image_combine":
            if target_sw is None:
                # Open the dialog (use preset defaults if present)
                self._open_image_combine()
                return
            # Dropped on a specific subwindow -> treat that doc as A, resolve B from preset/heuristics
            view = target_sw.widget()
            docA = getattr(view, "document", None)
            if docA is None or getattr(docA, "image", None) is None:
                QMessageBox.information(self, "Image Combine", "Target view has no image."); return
            try:
                p = dict(preset or {})
                p.setdefault("output", p.get("output", "replace"))   # default drop replaces A
                self._apply_image_combine_from_preset(p, target_doc=docA)
                return
            except Exception as e:
                QMessageBox.warning(self, "Image Combine", f"Preset apply failed:\n{e}")
                return
        if cid == "psf_viewer":
            # Open viewer; optional preset keys: threshold/mode/log/zoom
            try:
                self._open_psf_viewer(preset)
                self._log("Opened PSF Viewer" + (f" with preset {preset}" if preset else ""))
            except Exception as e:
                QMessageBox.warning(self, "PSF Viewer", f"Open failed:\n{e}")
            return            
        if cid == "plate_solve":
            # headless: solve the active document in-place
            doc = self._active_doc() if hasattr(self, "_active_doc") else None
            if not doc or getattr(doc, "image", None) is None:
                QMessageBox.information(self, "Plate Solver", "No active image view.")
                return

            # optional: quick sanity check for ASTAP path
            astap_path = self.settings.value("paths/astap", "", type=str)
            if not astap_path or not os.path.exists(astap_path):
                QMessageBox.information(
                    self, "Plate Solver",
                    "ASTAP executable not set. Go to Preferences -> ASTAP executable."
                )
                return

            try:       
                from pro.plate_solver import plate_solve_doc_inplace         
                ok, hdr_or_err = plate_solve_doc_inplace(self, doc, self.settings)
                if ok:
                    h = hdr_or_err  # astropy.io.fits.Header

                    # build a nice one-line summary
                    def _ff(x):
                        try: return float(x)
                        except Exception: return None

                    ra  = _ff(h.get("CRVAL1"))
                    dec = _ff(h.get("CRVAL2"))
                    cd11 = _ff(h.get("CD1_1")); cd21 = _ff(h.get("CD2_1"))
                    cd11 = cd11 if cd11 is not None else _ff(h.get("CDELT1"))
                    cd21 = cd21 if cd21 is not None else _ff(h.get("CDELT2"))
                    scale = None
                    if cd11 is not None or cd21 is not None:
                        a = cd11 or 0.0; b = cd21 or 0.0
                        scale = (a*a + b*b) ** 0.5 * 3600.0  # "/px

                    msg = "Plate solve complete"
                    if ra is not None and dec is not None:
                        msg += f" | RA={ra:.6f}Â deg, Dec={dec:.6f}Â deg"
                    if scale is not None:
                        msg += f" | scale~{scale:.3f}\"/px"

                    if hasattr(self, "_log"):
                        self._log(msg)
                    else:
                        print(msg)

                    # views will already refresh via doc.changed in plate_solve_doc_inplace
                else:
                    QMessageBox.warning(self, "Plate Solver", f"Plate solve failed:\n{hdr_or_err}")
            except Exception as e:
                QMessageBox.critical(self, "Plate Solver", f"Unhandled error:\n{e}")
            self._hdr_refresh_timer.start(0)
            return
        if cid == "star_align":
            # For alignment we need user choice of source/target, so open the dialog.
            # (We map DnD -> QAction via reg("star_align", self.act_star_align) already,
            # but keep this for parity with your explicit cid switch.)
            try:
                self._open_stellar_alignment()
            except Exception as e:
                QMessageBox.critical(self, "Stellar Alignment", f"Unhandled error:\n{e}")
            return
        if cid == "star_register":
            try:
                self._open_stellar_registration()
            except Exception as e:
                QMessageBox.critical(self, "Stellar Register", f"Unhandled error:\n{e}")
            return
        if cid == "image_peeker":
            doc = None
            if target_sw is not None:
                view = target_sw.widget()
                doc = getattr(view, "document", None)
            if doc is None and hasattr(self, "_active_doc"):
                doc = self._active_doc()
            if not doc or getattr(doc, "image", None) is None:
                QMessageBox.information(self, "Image Peeker", "No image in the dropped/active view.")
                return
            self._open_image_peeker_for_doc(doc, title_hint=getattr(target_sw, "windowTitle", lambda:"view")())
            return
        if cid == "star_spikes":
            doc = None
            if target_sw is not None:
                view = target_sw.widget()
                doc = getattr(view, "document", None)
            if doc is None and hasattr(self, "_active_doc"):
                # if you keep a helper; otherwise use docman.get_active_document()
                try:
                    doc = self._active_doc()
                except Exception:
                    pass
            if doc is None and hasattr(self, "docman"):
                doc = self.docman.get_active_document()
            if not doc or getattr(doc, "image", None) is None:
                QMessageBox.information(self, "Diffraction Spikes", "No image in the dropped/active view.")
                return
            self._open_star_spikes(doc=doc, preset=payload.get("preset") or {},
                                title_hint=getattr(target_sw, "windowTitle", lambda:"view")())
            return

        # ---------- Unknown cid with a doc: try a generic _apply_{cid}_to_doc  ----------
        generic = getattr(self, f"_apply_{cid}_to_doc", None)
        if callable(generic):
            try:
                # try passing preset if the method accepts it; otherwise call with (doc)
                try:
                    generic(doc, preset)
                except TypeError:
                    generic(doc)
                self._log(f"{cid} applied to '{target_sw.windowTitle()}'")
            except Exception as e:
                QMessageBox.warning(self, cid.replace("_", " ").title(), str(e))
            return

    def _find_action_by_cid(self, cid: str) -> QAction | None:
        if not cid:
            return None
        # We registered command ids on actions via register_action(...)
        for a in self.findChildren(QAction):
            if a.property("command_id") == cid or a.objectName() == cid:
                return a
        return None

    # in AstroSuiteProMainWindow (or wherever your drop handlers are)
    def _resolve_doc_from_payload(self, payload: dict, *, prefer_base: bool = True):
        """
        Resolve an ImageDocument from a DnD payload. Prefer the backing/base doc
        (full image) over a preview/proxy when requested.
        """
        dm = getattr(self, "docman", None) or getattr(self, "doc_manager", None)
        if dm is None:
            return None

        # 1) Try by uid (prefer base)
        base_uid = payload.get("base_doc_uid")
        doc_uid  = payload.get("doc_uid")
        if prefer_base and base_uid:
            try:
                d = dm.get_document_by_uid(base_uid)
                if d is not None:
                    return d
            except Exception:
                pass
        if doc_uid:
            try:
                d = dm.get_document_by_uid(doc_uid)
                if d is not None:
                    return d
            except Exception:
                pass

        # 2) Fallback: try file path match
        fp = (payload.get("file_path") or "").strip()
        if fp:
            try:
                for d in dm.all_documents():
                    meta = getattr(d, "metadata", {}) or {}
                    if (meta.get("file_path") or "").strip() == fp:
                        return d
            except Exception:
                pass

        # 3) Last resort: pointer from same-process drag (may be proxy)
        ptr = payload.get("wcs_from_doc_ptr")
        if ptr and hasattr(dm, "get_document_by_ptr"):
            try:
                d = dm.get_document_by_ptr(ptr)
                if d is not None:
                    # If this is a preview/proxy, map it back to the base if possible
                    if hasattr(dm, "get_base_document_for"):
                        b = dm.get_base_document_for(d) or d
                        return b
                    return d
            except Exception:
                pass

        return None


    def _target_doc_from_subwindow(self, subwin) -> object | None:
        """
        Resolve the *base* target document for a QMdiSubWindow.

        For ROI/preview-aware views, this prefers view.base_document so that
        "apply to base" operations don't accidentally hit the preview/proxy doc.
        """
        if subwin is None:
            return None

        w = subwin.widget()
        dm = getattr(self, "docman", None) or getattr(self, "doc_manager", None)

        # 0) NEW: prefer explicit base_document on the view
        base = getattr(w, "base_document", None)
        if base is not None:
            return base

        # 1) OLD behavior: ask DocManager (may return a proxy/ROI doc on older views)
        if dm and hasattr(dm, "get_document_for_view"):
            try:
                d = dm.get_document_for_view(w)
                if d is not None:
                    return d
            except Exception:
                pass

        # 2) Fallbacks
        if hasattr(w, "document"):
            return w.document

        if hasattr(self, "get_active_document"):
            try:
                return self.get_active_document()
            except Exception:
                pass

        return None



    def _on_astrometry_drop(self, payload: dict, target_subwindow):
        """
        Handle MIME_ASTROMETRY drops. Copies WCS/SIP from the *base* source doc
        into the true target doc (base if a preview tab is active).
        """
        dm = getattr(self, "docman", None) or getattr(self, "doc_manager", None)
        if dm is None:
            QMessageBox.information(self, "Copy Astrometry", "No document manager.")
            return

        # 1) Resolve source (prefer base doc over preview/proxy)
        src_doc = self._resolve_doc_from_payload(payload, prefer_base=True)
        if src_doc is None:
            QMessageBox.information(self, "Copy Astrometry", "Source view not found.")
            return

        # 2) Resolve target (map preview tab -> ROI doc -> base doc as needed)
        tgt_doc = self._target_doc_from_subwindow(target_subwindow)
        if tgt_doc is None:
            QMessageBox.information(self, "Copy Astrometry", "No target image.")
            return

        # 3) Extract WCS dict from source
        if hasattr(self, "_extract_wcs_dict"):
            try:
                wcs = self._extract_wcs_dict(src_doc)
            except Exception:
                wcs = {}
        else:
            wcs = {}

        if not wcs:
            QMessageBox.information(self, "Copy Astrometry", "Source has no WCS/SIP solution.")
            return

        # 4) Apply to target
        ok = False
        if hasattr(self, "_apply_wcs_dict_to_doc"):
            try:
                ok = bool(self._apply_wcs_dict_to_doc(tgt_doc, dict(wcs)))
            except Exception:
                ok = False

        if not ok:
            QMessageBox.warning(self, "Copy Astrometry", "Failed to apply astrometric solution.")
            return

        # 5) Refresh UI bits immediately
        try:
            if hasattr(self, "_refresh_header_viewer"):
                self._refresh_header_viewer(tgt_doc)
            if hasattr(self, "currentDocumentChanged"):
                self.currentDocumentChanged.emit(tgt_doc)
        except Exception:
            pass

        try:
            sname = getattr(src_doc, "display_name", lambda: None)() or "Source"
            tname = getattr(tgt_doc, "display_name", lambda: None)() or "Target"
            QMessageBox.information(self, "Copy Astrometry",
                                    f"Copied solution from '{sname}' to '{tname}'.")
        except Exception:
            pass

    def _on_remove_pedestal(self):
        from pro.pedestal import remove_pedestal

        # Let remove_pedestal resolve the correct target via DocManager
        remove_pedestal(self, target_doc=None)

        # remember for replay - no preset payload needed, just the id
        self._remember_last_headless_command(
            "pedestal",
            preset={},                          # no parameters for this one
            description="Pedestal Removal"
        )

    def _open_statistical_stretch_with_preset(self, preset: dict):
        """
        Open the Statistical Stretch dialog and prefill its controls from preset.
        """
        sw = self.mdi.activeSubWindow()
        if not sw:
            
            QMessageBox.information(self, "No image", "Open an image first.")
            return

        doc = sw.widget().document
        from pro.stat_stretch import StatisticalStretchDialog  # adjust import if needed

        dlg = StatisticalStretchDialog(self, doc)

        # prefill if keys exist (ignore missing keys gracefully)
        try:
            if "target_median" in preset:
                dlg.spin_target.setValue(float(preset["target_median"]))
            if "linked" in preset:
                dlg.chk_linked.setChecked(bool(preset["linked"]))
            if "normalize" in preset:
                dlg.chk_normalize.setChecked(bool(preset["normalize"]))
            if "apply_curves" in preset:
                dlg.chk_curves.setChecked(bool(preset["apply_curves"]))
            if "curves_boost" in preset:
                dlg.sld_curves.setValue(int(round(float(preset["curves_boost"]) * 100)))
        except Exception:
            pass  # never block the dialog if a value is off-type

        try:

            dlg.setWindowIcon(QIcon(statstretch_path))
        except Exception:
            pass

        dlg.resize(900, 600)
        dlg.show()


    def _apply_stat_stretch_preset_to_doc(self, doc, preset: dict):
        """
        Headless apply of Statistical Stretch using the dialog's own apply path.
        We instantiate the dialog, set controls from the preset, and call its Apply.
        """

        # --- Re-entry guard: prevent double-apply per event ---
        if getattr(self, "_stat_stretch_apply_in_progress", False):
            print("[Replay] _apply_stat_stretch_preset_to_doc: re-entry suppressed")
            return
        self._stat_stretch_apply_in_progress = True
        try:
            from pro.stat_stretch import StatisticalStretchDialog  # adjust import if needed

            dlg = StatisticalStretchDialog(self, doc)

            # fill controls
            try:
                if "target_median" in preset:
                    dlg.spin_target.setValue(float(preset["target_median"]))
                if "linked" in preset:
                    dlg.chk_linked.setChecked(bool(preset["linked"]))
                if "normalize" in preset:
                    dlg.chk_normalize.setChecked(bool(preset["normalize"]))
                if "apply_curves" in preset and hasattr(dlg, "chk_curves"):
                    dlg.chk_curves.setChecked(bool(preset["apply_curves"]))
                if "curves_boost" in preset and hasattr(dlg, "sld_curves"):
                    dlg.sld_curves.setValue(int(round(float(preset["curves_boost"]) * 100)))
            except Exception:
                pass

            # directly run the dialog's apply slot (reuses your edit/undo naming, etc.)
            dlg._do_apply()
            try:
                dlg.close()
            except Exception:
                pass
        finally:
            self._stat_stretch_apply_in_progress = False



    def _open_star_stretch_with_preset(self, preset: dict):
        from pro.star_stretch import StarStretchDialog
        """Background drop -> open Star Stretch dialog with controls preloaded."""
        sw = self.mdi.activeSubWindow()
        if not sw:
            QMessageBox.information(self, "Star Stretch", "No active image window.")
            return
        doc = sw.widget().document
        if doc is None or getattr(doc, "image", None) is None:
            QMessageBox.information(self, "Star Stretch", "Active document has no image.")
            return

        dlg = StarStretchDialog(self, doc)

        # Accept synonyms, fall back to dialog defaults
        amt = preset.get("stretch_factor",
            preset.get("stretch_amount",
            preset.get("amount", None)))
        if amt is not None:
            try: dlg.sld_st.setValue(int(float(amt) * 100.0))
            except Exception as e:
                import logging
                logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")

        sat = preset.get("color_boost", preset.get("saturation", None))
        if sat is not None:
            try: dlg.sld_sat.setValue(int(float(sat) * 100.0))
            except Exception as e:
                import logging
                logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")

        scnr = preset.get("scnr_green", preset.get("scnr", None))
        if scnr is not None:
            try: dlg.chk_scnr.setChecked(bool(scnr))
            except Exception as e:
                import logging
                logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")

        dlg.resize(1000, 650)
        dlg.show()
        if hasattr(self, "_log"):
            self._log("Star Stretch: opened dialog with preset.")

    def _apply_star_stretch_preset_to_doc(self, doc, preset: dict):
        """
        Drop on a specific subwindow -> apply Star Stretch headlessly to that doc.
        Uses your Numba kernel (applyPixelMath_numba). If unavailable, raises.
        """
        import numpy as np
        try:
            from legacy.numba_utils import applyPixelMath_numba, applySCNR_numba
            _has_numba = True
        except Exception:
            _has_numba = False
            applyPixelMath_numba = None
            # lightweight SCNR fallback (same as in pro/star_stretch.py)
            def applySCNR_numba(image_array: np.ndarray) -> np.ndarray:
                img = image_array.astype(np.float32, copy=False)
                if img.ndim != 3 or img.shape[2] != 3:
                    return img
                r = img[..., 0]; g = img[..., 1]; b = img[..., 2]
                g2 = np.minimum(g, 0.5 * (r + b))
                out = img.copy()
                out[..., 1] = g2
                return np.clip(out, 0.0, 1.0)

        img = getattr(doc, "image", None)
        if img is None:
            raise RuntimeError("Document has no image.")

        # read preset (accept a few aliases)
        amount = float(preset.get("stretch_factor",
                preset.get("stretch_amount",
                preset.get("amount", 5.0))))  # 0..8
        sat    = float(preset.get("color_boost", preset.get("saturation", 1.0)))  # 0..2
        scnr   = bool(preset.get("scnr_green", preset.get("scnr", False)))

        # convert to float 0..1
        a = np.asarray(img)
        if a.dtype.kind in "ui":
            a = a.astype(np.float32) / float(np.iinfo(a.dtype).max)
        elif a.dtype.kind == "f":
            mx = float(a.max()) if a.size else 1.0
            a = a.astype(np.float32) / (mx if mx > 1.0 else 1.0)
        else:
            a = a.astype(np.float32)

        # mono -> 3ch temporarily
        need_collapse = False
        if a.ndim == 2:
            a = np.stack([a]*3, axis=-1); need_collapse = True
        elif a.ndim == 3 and a.shape[2] == 1:
            a = np.repeat(a, 3, axis=2); need_collapse = True

        if applyPixelMath_numba is None:
            raise RuntimeError("Star Stretch requires Numba kernel (applyPixelMath_numba) for headless run.")

        out = applyPixelMath_numba(a, amount)

        # color boost
        if out.ndim == 3 and out.shape[2] == 3 and abs(sat - 1.0) > 1e-6:
            mean = out.mean(axis=2, keepdims=True)
            out = mean + (out - mean) * sat
            out = np.clip(out, 0.0, 1.0)

        # SCNR
        if scnr and out.ndim == 3 and out.shape[2] == 3:
            out = applySCNR_numba(out.astype(np.float32, copy=False))

        if need_collapse:
            out = out[..., 0]

        meta = {
            "step_name": "Star Stretch",
            "star_stretch": {
                "stretch_factor": amount,
                "color_boost": sat,
                "scnr_green": scnr,
                "numba": _has_numba,
            }
        }
        doc.apply_edit(out.astype(np.float32, copy=False), metadata=meta, step_name="Star Stretch")


    # --- Command Search (palette) ----------------------------------------
    def _strip_menu_text(self, s: str) -> str:
        return s.replace("&", "").replace("...", "").strip()

    # --- Command palette helpers (safe: no receivers/emit on Qt signals) ---

    def _is_action_commandy(self, act: QAction) -> bool:
        """Heuristic: action is a real, user-triggerable command."""
        if act is None:
            return False
        if act.isSeparator():
            return False
        # Exclude submenu containers -- we want leaf commands only
        if act.menu() is not None:
            return False
        # Text visible to users?
        txt = (act.text() or "").strip().replace("&", "")
        if not txt:
            return False
        # Optional heuristics:
        if not act.isEnabled():
            return False
        # Allow opt-out per action if you ever need it:
        if bool(act.property("cmdp_exclude")):
            return False
        return True

    def _collect_all_qactions(self) -> list[QAction]:
        """Gather every 'leaf' QAction from menubar + toolbars without using receivers()."""
        from PyQt6.QtWidgets import QToolBar, QMenu
        seen, out = set(), []

        # Menus: recurse into each top-level QMenu
        for top_act in self.menuBar().actions():
            m = top_act.menu()
            if m is None:
                # Top-level action without a submenu (rare) -- include if commandy
                if self._is_action_commandy(top_act) and id(top_act) not in seen:
                    out.append(top_act); seen.add(id(top_act))
                continue
            for act in self._iter_menu_actions(m):
                if self._is_action_commandy(act) and id(act) not in seen:
                    out.append(act); seen.add(id(act))

        # Toolbars
        for tb in self.findChildren(QToolBar):
            for act in tb.actions():
                if self._is_action_commandy(act) and id(act) not in seen:
                    out.append(act); seen.add(id(act))

        return out


    def _install_command_search(self):

        # Clean up any old placement (corner widget / toolbar / dock)
        mb = self.menuBar()
        mb.setNativeMenuBar(False)
        old = mb.cornerWidget(Qt.Corner.TopRightCorner)
        if old:
            old.deleteLater()
            mb.setCornerWidget(None, Qt.Corner.TopRightCorner)

        tb = getattr(self, "_search_tb", None)
        if tb:
            try: self.removeToolBar(tb)
            except Exception as e:
                import logging
                logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")
            tb.deleteLater()
            self._search_tb = None

        old_dock = getattr(self, "_search_dock", None)
        if old_dock:
            try: old_dock.hide(); old_dock.setParent(None)
            except Exception as e:
                import logging
                logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")
            old_dock.deleteLater()
            self._search_dock = None

        # --- Right-side mini dock with the search box ---
        self._search_dock = QDockWidget("Command Search", self)
        self._search_dock.setObjectName("CommandSearchDock")
        # âœ... Allow moving/closing like other panels
        self._search_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea |
            Qt.DockWidgetArea.RightDockWidgetArea |
            Qt.DockWidgetArea.TopDockWidgetArea |
            Qt.DockWidgetArea.BottomDockWidgetArea
        )
        self._search_dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable |
            QDockWidget.DockWidgetFeature.DockWidgetClosable |
            QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )

        holder = QWidget(self._search_dock)
        lay = QHBoxLayout(holder)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(6)

        self._cmd_edit = QLineEdit(holder)
        self._cmd_edit.setPlaceholderText("Search commands...  (Ctrl+Shift+P)")
        self._cmd_edit.setClearButtonEnabled(True)
        self._cmd_edit.setMinimumWidth(240)
        self._cmd_edit.setMaximumWidth(700)
        self._cmd_edit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        lay.addWidget(self._cmd_edit, 1)

        holder.setMaximumHeight(44)  # keep the dock short
        self._search_dock.setWidget(holder)

        # Add to the RIGHT area
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._search_dock)

        # Make sure it's ABOVE Layers/Header
        layers_dock = getattr(self, "layers_dock", None) or getattr(self, "_layers_dock", None)
        header_dock = getattr(self, "header_dock", None) or getattr(self, "_header_viewer_dock", None)
        try:
            if layers_dock:
                # split so Layers goes BELOW search
                self.splitDockWidget(self._search_dock, layers_dock, Qt.Orientation.Vertical)
            if header_dock and layers_dock:
                # keep header under layers (or whatever arrangement you prefer)
                # If you tabify layers/header, comment the line below and use tabifyDockWidget
                self.splitDockWidget(layers_dock, header_dock, Qt.Orientation.Vertical)
        except Exception:
            pass

        # ---- Completer + model (same behavior as before) ----
        self._cmd_model = QStandardItemModel(self)
        self._cmd_completer = QCompleter(self._cmd_model, self)
        self._cmd_completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self._cmd_completer.setFilterMode(Qt.MatchFlag.MatchContains)

        popup = QListView(self)
        popup.setMinimumWidth(420)
        popup.setIconSize(QSize(18, 18))
        self._cmd_completer.setPopup(popup)

        self._cmd_edit.setCompleter(self._cmd_completer)
        self._cmd_completer.activated[QModelIndex].connect(self._run_selected_completion)

        # Shortcut to focus it
        QShortcut(QKeySequence("Ctrl+Shift+P"), self, activated=self._focus_command_search)


        # Enter runs the first visible completion
        self._cmd_edit.returnPressed.connect(self._trigger_first_visible_completion)

        # Initial population
        self._build_command_model()

        # After window state restore, re-pin it to the top of the right area
        if not self.settings.value("window_layout/restored", False, type=bool):
            try:
                layers_dock = getattr(self, "layers_dock", None) or getattr(self, "_layers_dock", None)
                header_dock = getattr(self, "header_dock", None) or getattr(self, "_header_viewer_dock", None)
                if layers_dock:
                    self.splitDockWidget(self._search_dock, layers_dock, Qt.Orientation.Vertical)
                if header_dock and layers_dock:
                    self.splitDockWidget(layers_dock, header_dock, Qt.Orientation.Vertical)
            except Exception:
                pass



    def _build_command_model(self):
        self._cmd_model.clear()
        self._cmd_model.setColumnCount(1)

        actions = self._collect_all_qactions()
        actions.sort(key=lambda a: self._strip_menu_text(a.text()).lower())

        for act in actions:
            title = self._strip_menu_text(act.text())
            if not title:
                continue
            item = QStandardItem(act.icon(), title)
            item.setEditable(False)
            item.setData(act, _ROLE_ACTION)
            self._cmd_model.appendRow(item)

        self._cmd_completer.setCompletionColumn(0)
        self._cmd_completer.popup().setModelColumn(0)

    def _run_selected_completion(self, index: QModelIndex):
        if index.isValid():
            act = index.data(_ROLE_ACTION)
            if isinstance(act, QAction) and act.isEnabled():
                act.trigger()

    def _trigger_first_visible_completion(self):
        popup = self._cmd_completer.popup()
        m = popup.model()
        if m and m.rowCount() > 0:
            self._run_selected_completion(m.index(0, 0))

    def _focus_command_search(self):
        self._cmd_edit.setFocus(Qt.FocusReason.ShortcutFocusReason)
        self._cmd_edit.selectAll()

    # --- Actions ----------------------------------------
    def set_document(self, doc):
        self._doc = doc
        if doc is None:
            self._clear()                      # <- must fully clear visible content
            self.setWindowTitle("Header (No image)")
            return
        self._populate_from(doc)
        self.setWindowTitle("Header")

    # --- Helpers ---
    def _remember_active_pair(self, new_sw):
        """Remember last and current active subwindow so we can â€˜bounce'."""
        if new_sw is None:
            return
        if new_sw is self._current_active_sw:
            return
        # only remember last if it still exists
        if self._current_active_sw in self.mdi.subWindowList():
            self._last_active_sw = self._current_active_sw
        else:
            self._last_active_sw = None
        self._current_active_sw = new_sw

    def _toggle_last_active_view(self):
        last = getattr(self, "_last_active_sw", None)
        if not last or last not in self.mdi.subWindowList() or not last.isVisible():
            return  # nothing to bounce to
        self.mdi.setActiveSubWindow(last)

    # In AstroSuiteProMainWindow
    def _on_image_region_updated_global(self, doc, roi_tuple_or_none):
        """
        Fan-out: when DocManager pastes an edit (full or ROI),
        refresh every view that shows `doc`. If the view is on a Preview tab,
        refresh only if its ROI intersects the changed region.
        """
        def _roi_intersects(a, b):
            ax, ay, aw, ah = map(int, a)
            bx, by, bw, bh = map(int, b)
            if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
                return False
            return not (ax + aw <= bx or bx + bw <= ax or ay + ah <= by or by + bh <= ay)

        for sw in self.mdi.subWindowList():
            view = getattr(sw, "widget", lambda: None)()
            if view is None:
                continue

            base = getattr(view, "base_document", None) or getattr(view, "document", None)
            if base is not doc:
                # Some views replace .document with a proxy; compare by identity if possible
                # If your proxy holds a ._base or ._doc, you can also try to unwrap here.
                continue

            # If not on a Preview tab, just force a repaint.
            if not (hasattr(view, "has_active_preview") and view.has_active_preview()):
                if hasattr(view, "refresh_from_docman"):
                    view.refresh_from_docman()
                elif hasattr(view, "_render"):
                    view._render(rebuild=True)
                continue

            # Preview tab active -> refresh only if the changed region overlaps our ROI
            try:
                my_roi = view.current_preview_roi()  # (x,y,w,h)
            except Exception:
                my_roi = None

            if my_roi is None or roi_tuple_or_none is None or _roi_intersects(my_roi, roi_tuple_or_none):
                if hasattr(view, "refresh_from_docman"):
                    view.refresh_from_docman()
                elif hasattr(view, "_render"):
                    view._render(rebuild=True)

    def _hook_preview_awareness(self, view):
        """
        Make the main window react when the user switches to/from the Preview tab
        or changes the selected ROI, so toolbars/undo/etc. reflect the ROI doc.
        """
        from PyQt6.QtCore import QObject

        def _on_any_preview_change(*_):
            # Re-resolve active doc using DocManager's ROI logic and refresh UI
            try:
                dm = getattr(self, "doc_manager", None) or getattr(self, "docman", None)
                if dm and hasattr(dm, "get_document_for_view"):
                    resolved = dm.get_document_for_view(view)  # ROI-aware
                    # mark that resolved doc as active so _active_doc() reflects it
                    if hasattr(dm, "set_active_document"):
                        dm.set_active_document(resolved)
            except Exception:
                pass

            try: self.update_undo_redo_action_labels()
            except Exception as e:
                import logging
                logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")
            try: self._refresh_mask_action_states()
            except Exception as e:
                import logging
                logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")
            try:
                # any other UI you keep in sync (histogram, header viewer, etc.)
                if hasattr(self, "_hdr_refresh_timer") and self._hdr_refresh_timer is not None:
                    self._hdr_refresh_timer.start(0)
            except Exception:
                pass

        # Prefer explicit signals if your ImageSubWindow exposes them
        hooked = False
        for sig_name in ("previewActivated", "previewDeactivated", "previewROIChanged", "previewTabChanged"):
            if hasattr(view, sig_name):
                try:
                    getattr(view, sig_name).connect(_on_any_preview_change)
                    hooked = True
                except Exception:
                    pass

        # Generic fallback: watch the tab widget if present
        try:
            tabs = getattr(view, "tabs", None)
            if tabs and hasattr(tabs, "currentChanged"):
                tabs.currentChanged.connect(_on_any_preview_change)
                hooked = True
        except Exception:
            pass

        # Ultra-fallback: install an event filter on the view to catch tab changes
        if not hooked:
            class _PreviewEventFilter(QObject):
                def eventFilter(self, _obj, ev):
                    et = getattr(ev, "type", lambda: None)()
                    #  CurrentChanged on QTabBar/QTabWidget routes through various event types;
                    #  repaint + layout + focus-in when user clicks is a good cheap proxy.
                    if et in (12, 14, 24):  # Paint, LayoutRequest, FocusIn
                        _on_any_preview_change()
                    return False
            try:
                ef = _PreviewEventFilter(view)
                view.installEventFilter(ef)
                # keep a ref so it doesn't get GC'd
                if not hasattr(view, "_preview_event_filters"):
                    view._preview_event_filters = []
                view._preview_event_filters.append(ef)
            except Exception:
                pass

    def _pretty_title(self, doc, *, linked: bool | None = None) -> str:
        name = getattr(doc, "display_name", lambda: "Untitled")()
        name = name.replace("[LINK] ", "").strip()
        if linked is None:
            linked = hasattr(doc, "_parent_doc")  # ROI proxy -> linked
        return f"[LINK] {name}" if linked else name

    def _build_subwindow_title_for_doc(self, doc) -> str:
        """
        Build a unique, human-friendly title for a QMdiSubWindow
        that shows this document. If multiple views exist for the
        same doc, append [View N].

        IMPORTANT: This is called *after* the new subwindow has been
        added to the QMdiArea, so the first view will already give
        count == 1.
        """
        # Base label (reuse pretty_title logic, but keep it unlinked
        # like your old _spawn_subwindow_for did)
        base = self._pretty_title(doc, linked=False)

        # Count how many existing views show this *same* doc
        count = 0
        try:
            for sw in self.mdi.subWindowList():
                w = sw.widget() if hasattr(sw, "widget") else None
                if w is None:
                    continue
                # For image views we have base_document; for others fall back
                d = getattr(w, "base_document", None) or getattr(w, "document", None)
                if d is doc:
                    count += 1
        except Exception:
            pass

        # If this is the only view (count == 1), or something weird (0),
        # just show the base title with no [View N] suffix.
        if count <= 1:
            return base

        # Subsequent views -> base + [View N], where N == count
        return f"{base} [View {count}]"

    def _unique_window_title(self, base: str) -> str:
        """
        Return a window title based on `base` that is not already used by
        any QMdiSubWindow. If `base` is free, use it; otherwise append
        ' [View N]' with the first free N.
        """
        base = (base or "Untitled").strip()

        existing = set()
        try:
            for sw in self.mdi.subWindowList():
                if sw is None:
                    continue
                t = sw.windowTitle() or ""
                if t:
                    existing.add(t)
        except Exception:
            pass

        # First use: plain base
        if base not in existing:
            return base

        # Subsequent uses: base [View 2], [View 3], ...
        n = 2
        while True:
            cand = f"{base} [View {n}]"
            if cand not in existing:
                return cand
            n += 1


    def _spawn_subwindow_for(self, doc, *, force_new: bool = False):
        """
        Open a subwindow for `doc`. If one already exists and force_new=False,
        simply raise/show the existing one. If force_new=True, always spawn a
        brand-new view (useful for 'duplicate view' drops on the MDI background).
        """
        # -- 0) Reuse existing unless caller explicitly wants a new one
        if not force_new:
            existing = self._find_subwindow_for_doc(doc)
            if existing:
                try:
                    # ensure really visible and focused
                    existing.show()
                    wst = existing.windowState()
                    if wst & Qt.WindowState.WindowMinimized:
                        existing.setWindowState(wst & ~Qt.WindowState.WindowMinimized)
                    self.mdi.setActiveSubWindow(existing)
                    existing.raise_()
                except Exception:
                    pass
                return existing

        # Track whether there were any visible subwindows before we add this one
        first_window = False
        try:
            subs = [s for s in self.mdi.subWindowList() if s.isVisible()]
            first_window = (len(subs) == 0)
        except Exception:
            # be conservative on error - just don't special-case it
            first_window = False

        # -- 1) Import view classes
        try:
            from pro.subwindow import ImageSubWindow, TableSubWindow
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            QMessageBox.critical(self, "View Import Error",
                                f"Failed to import view classes from pro.subwindow:\n{e}\n\n{tb}")
            from PyQt6.QtWidgets import QLabel
            w = QLabel(doc.display_name()); setattr(w, "document", doc)
            wrapper = self.mdi.addSubWindow(w)
            wrapper.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
            wrapper.show()
            return wrapper

        # -- 2) Table vs Image detection (unchanged)
        md = (getattr(doc, "metadata", {}) or {})
        is_table = (md.get("doc_type") == "table") or (hasattr(doc, "rows") and hasattr(doc, "headers"))

        # -- 3) Construct the view (log all errors, fall back to label)
        try:
            view = TableSubWindow(doc) if is_table else ImageSubWindow(doc)
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            try:
                self._log(f"[spawn] View construction failed: {e}\n{tb}")
            except Exception:
                pass
            from PyQt6.QtWidgets import QLabel
            view = QLabel(doc.display_name()); setattr(view, "document", doc)

        # -- 4) DocManager wiring (prefer self.doc_manager, fall back to self.docman)
        dm = getattr(self, "doc_manager", None) or getattr(self, "docman", None)
        if hasattr(view, "set_doc_manager") and dm is not None:
            try:
                view.set_doc_manager(dm)
            except Exception:
                pass

        # -- 5) ROI-aware proxy: keep base handle, expose live proxy at view.document
        try:
            setattr(view, "base_document", doc)  # explicit base (non-ROI) doc
            if dm is not None:
                view.document = _DocProxy(dm, view, doc)
            else:
                setattr(view, "document", doc)
        except Exception as e:
            print(f"Failed to install DocProxy: {e}")
            try:
                self._log(f"[spawn] Failed to install DocProxy: {e}")
            except Exception:
                pass
            try:
                setattr(view, "document", doc)
            except Exception:
                pass

        # ðŸ"-- REPLAY: connect ImageSubWindow -> MainWindow (support old + new signal names)
        replay_sig = None
        sig_name_used = None
        for name in ("replayOnBaseRequested", "replayLastRequested"):
            s = getattr(view, name, None)
            if s is not None:
                replay_sig = s
                sig_name_used = name
                break

        if replay_sig is not None:
            try:
                replay_sig.connect(self._on_view_replay_last_requested)
                try:
                    self._log(f"[Replay] Connected {sig_name_used} for view id={id(view)}")
                except Exception:
                    print(f"[Replay] Connected {sig_name_used} for view id={id(view)}")
            except Exception as e:
                try:
                    self._log(f"[Replay] FAILED to connect {sig_name_used} for view id={id(view)}: {e}")
                except Exception:
                    print(f"[Replay] FAILED to connect {sig_name_used} for view id={id(view)}: {e}")

        self._hook_preview_awareness(view)
        base_title = self._pretty_title(doc, linked=False)
        final_title = self._unique_window_title(base_title)

        # -- 6) Add subwindow and set chrome
        sw = self.mdi.addSubWindow(view)
        sw.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        if hasattr(view, "resized"):
            view.resized.connect(self._on_view_resized)
        sw.setWindowIcon(self.app_icon)
        sw.setWindowTitle(final_title)

        # Apply standard window flags
        flags = sw.windowFlags()
        flags |= Qt.WindowType.WindowCloseButtonHint
        flags |= Qt.WindowType.WindowMinimizeButtonHint
        flags |= Qt.WindowType.WindowMaximizeButtonHint
        sw.setWindowFlags(flags)

        # âŒ removed the "fill MDI viewport" block - we *don't* want full-monitor first window

        # Show / activate
        sw.show()
        sw.raise_()
        self.mdi.setActiveSubWindow(sw)
        # (no second setWindowTitle() here)

        # Optional minimize/restore interceptor
        if hasattr(self, "_minimize_interceptor"):
            try:
                sw.installEventFilter(self._minimize_interceptor)
            except Exception:
                pass

        # Shelf cleanup on destroy
        try:
            sw.destroyed.connect(
                lambda _=None, s=sw:
                hasattr(self, "window_shelf") and self.window_shelf.remove_for_subwindow(s)
            )
        except Exception:
            pass

        # Close handling hooks
        if hasattr(view, "aboutToClose"):
            try:
                # avoid accidental double connections if respawned
                try:
                    view.aboutToClose.disconnect()
                except Exception:
                    pass
                # forward the *base* doc to the slot
                view.aboutToClose.connect(lambda _=None, d=doc: self._on_view_about_to_close(d))
            except Exception:
                pass
        else:
            # Worst-case fallback: if the view doesn't expose aboutToClose, use destroyed
            try:
                sw.destroyed.connect(lambda _=None, d=doc: self._on_view_about_to_close(d))
            except Exception:
                pass

        # Keep undo/redo labels in sync
        try:
            doc.changed.connect(self.update_undo_redo_action_labels)
        except Exception:
            pass

        # -- 7) Initial sizing + scale
        if is_table:
            # Tables still get a reasonable fixed size
            sw.resize(1000, 700)
        else:
            # Image docs: just do a one-time fit-to-window into whatever
            # geometry Qt gave this subwindow. This avoids hard 900x700 boxes.
            try:
                if self.mdi.activeSubWindow() is not sw:
                    self.mdi.setActiveSubWindow(sw)
                self._zoom_active_fit()
            except Exception:
                pass

        # -- 8) Sync autostretch UI state
        if hasattr(view, "autostretch_enabled"):
            try:
                self._sync_autostretch_action(view.autostretch_enabled)
            except Exception:
                pass

        # -- 9) View-level duplicate signal -> route to handler
        if hasattr(view, "requestDuplicate"):
            try:
                view.requestDuplicate.connect(self._duplicate_view_from_signal)
            except Exception:
                pass

        # -- 10) Log if image missing (non-table)
        if not is_table and getattr(doc, "image", None) is None:
            try:
                self._log(f"[spawn] No image and not recognized as table; metadata keys: "
                        f"{list((getattr(doc,'metadata',{}) or {}).keys())}")
            except Exception:
                pass

        # activate hook
        try:
            self._on_subwindow_activated(sw)
        except Exception:
            pass

        # -- 11) If this is the first window and it's an image, mimic "Cascade Views"
        try:
            if first_window and not is_table:
                # Prefer our helper (does cascade + auto-fit), else raw Qt cascade
                if hasattr(self, "_cascade_views") and callable(self._cascade_views):
                    self._cascade_views()
                else:
                    self.mdi.cascadeSubWindows()
        except Exception:
            pass

        return sw



    def _connect_view_signals(self, view):
        # Make sure all views we create are wired to our handlers
        view.aboutToClose.connect(self._on_view_about_to_close)
        view.requestDuplicate.connect(self._duplicate_view_from_signal)

    def _on_view_about_to_close(self, doc):
        """
        Invoked by ImageSubWindow.closeEvent() before teardown.
        Remove the *base* document from DocManager only if no other subwindow
        is still showing it.
        """
        sender_view = self.sender()
        base = self._normalize_base_doc(doc)

        # IMPORTANT: compare by each view's base_document (not .document which is a proxy)
        still_open = [
            sw for sw in self.mdi.subWindowList()
            if getattr(sw.widget(), "base_document", None) is base
            and (sender_view is None or sw.widget() is not sender_view)
        ]

        if not still_open:
            try:
                self.docman.close_document(base)   # emits documentRemoved(base)
                if hasattr(self, "_log"):
                    self._log(f"Closed: {base.display_name()}")
            except Exception:
                pass

        from PyQt6.QtCore import QTimer
        QTimer.singleShot(0, self._maybe_clear_ui_after_close)


    def _maybe_clear_ui_after_close(self):
        from pro.header_viewer import HeaderViewerDock
        # If no subwindows remain, clear all "active doc" UI bits, including header
        if not self.mdi.subWindowList():
            self.currentDocumentChanged.emit(None)   # drives HeaderViewerDock.set_document(None)
            self.update_undo_redo_action_labels()
            self._hdr_refresh_timer.start(0)       # belt-and-suspenders for manual widgets
            # If your dock has its own set_document, call it explicitly too
            hv = getattr(self, "header_viewer", None)
            if hv and hasattr(hv, "set_document"):
                try:
                    hv.set_document(None)
                except Exception:
                    pass

    def _duplicate_view_from_signal(self, source_view):
        print("Duplicating view from signal...")
        from PyQt6.QtCore import QTimer  # safe even if unused on some code paths

        # 0) Resolve the *base* document (avoid DocProxy/ROI targets)
        doc = getattr(source_view, "document", None)
        if doc is None:
            return

        base_doc = getattr(source_view, "base_document", None)
        if base_doc is None:
            # If doc is a _DocProxy, try to resolve; otherwise fall back to doc
            try:
                base_doc = doc._target()  # may exist on our proxy
            except Exception:
                base_doc = doc

        if getattr(base_doc, "image", None) is None:
            return

        # 1) Capture source view state
        hbar = source_view.scroll.horizontalScrollBar()
        vbar = source_view.scroll.verticalScrollBar()
        state = {
            "scale": float(getattr(source_view, "scale", 1.0)),
            "hval": int(hbar.value()),
            "vval": int(vbar.value()),
            "autostretch": bool(getattr(source_view, "autostretch_enabled", False)),
            "autostretch_target": float(getattr(source_view, "autostretch_target", 0.25)),
        }

        # 2) New name (strip UI decorations if any)
        base_name = ""
        try:
            base_name = base_doc.display_name() or "Untitled"
        except Exception:
            base_name = "Untitled"

        try:
            base_name = _strip_ui_decorations(base_name)
        except Exception:
            # minimal fallback: remove our known prefix/glyphs
            while len(base_name) >= 2 and base_name[1] == " " and base_name[0] in "â- â--â--†â-²â-ªâ-«â€¢â--¼â--»â--¾â--½":
                base_name = base_name[2:]
            if base_name.startswith("Active View: "):
                base_name = base_name[len("Active View: "):]

        # 3) Duplicate the *base* document (not the ROI proxy)
        #    NOTE: your project uses `self.docman` elsewhere for duplication.
        new_doc = self.docman.duplicate_document(base_doc, new_name=f"{base_name}_duplicate")
        print(f"  Duplicated document ID {id(base_doc)} -> {id(new_doc)}")

        # 4) Ensure the duplicate starts mask-free (so we don't inherit mask UI state)
        try:
            mid = getattr(new_doc, "active_mask_id", None)
            if mid and hasattr(new_doc, "remove_mask"):
                new_doc.remove_mask(mid)
            if hasattr(new_doc, "masks") and getattr(new_doc, "masks", None):
                try:
                    new_doc.masks.clear()
                except Exception:
                    new_doc.masks = {}
            new_doc.active_mask_id = None
            if hasattr(new_doc, "changed"):
                new_doc.changed.emit()
        except Exception:
            try:
                new_doc.active_mask_id = None
                if hasattr(new_doc, "masks"):
                    new_doc.masks = {}
                if hasattr(new_doc, "changed"):
                    new_doc.changed.emit()
            except Exception:
                pass

        # 5) Spawn the subwindow *now* (don't rely on an external documentAdded handler)
        sw = self._spawn_subwindow_for(new_doc, force_new=True)
        print(f"  Spawned subwindow for duplicated document ID {id(new_doc)}")
        if not sw:
            # Extremely defensive: try once more on the next tick, then give up
            def _retry_spawn():
                sw2 = self._spawn_subwindow_for(new_doc, force_new=True)
                if not sw2 and hasattr(self, "_log"):
                    self._log("[duplicate] failed to spawn subwindow for duplicated document")
            QTimer.singleShot(0, _retry_spawn)
            return

        view = sw.widget()

        # If the view was constructed before we cleared the doc masks, force-clear the UI dot.
        if hasattr(view, "_mask_dot_enabled") and view._mask_dot_enabled:
            view._mask_dot_enabled = False
            try:
                view._rebuild_title()
            except Exception:
                # last resort: set a clean title
                t = getattr(view, "base_doc_title", lambda: new_doc.display_name())()
                sw.setWindowTitle(t)
                sw.setToolTip(t)

        # 6) Apply the saved view state (zoom/pan/autostretch)
        try:
            self._apply_view_state_to_view(view, state)
        except Exception:
            pass

        # 7) Focus the new window and log
        self.mdi.setActiveSubWindow(sw)
        print(f"  Activated subwindow for duplicated document ID {id(new_doc)}")
        if hasattr(self, "_log"):
            self._log(f"Duplicated as independent document -> '{new_doc.display_name()}'")




    def _find_doc_by_id(self, doc_ptr):
        if doc_ptr is None:
            return None, None
        for sw in self.mdi.subWindowList():
            w = sw.widget()
            d = getattr(w, "document", None)
            if d is not None and id(d) == doc_ptr:
                return d, sw
        return None, None

    def _handle_viewstate_drop(self, state: dict, target_sw):
        """
        Legacy handler:
        - target_sw is None -> duplicate view on background
        - target_sw is a QMdiSubWindow -> copy zoom/pan/stretch to it

        NOTE: If the payload came from a Preview/ROI, we let the ROI-aware
        handler (_on_mdi_viewstate_drop) take over for background drops.
        """
        is_preview = (state.get("source_kind") == "preview") or bool(state.get("roi"))

        # Background drop of a preview -> ROI logic lives in _on_mdi_viewstate_drop
        if target_sw is None and is_preview:
            return

        if target_sw is None:
            self._duplicate_view_from_state(state)
        else:
            self._apply_view_state_to_sub(state, target_sw)

    def _duplicate_view_from_state(self, state: dict):
        doc, source_sw = self._find_doc_by_id(state.get("doc_ptr"))
        if not doc:
            return

        orig_title = source_sw.windowTitle() if source_sw else doc.display_name()
        new_doc = self.docman.duplicate_document(doc, new_name=f"{orig_title}_duplicate")

        def _apply_when_ready():
            sw = self._find_subwindow_for_doc(new_doc)
            if not sw:
                QTimer.singleShot(0, _apply_when_ready)
                return
            view = sw.widget()
            self._apply_view_state_to_view(view, state)
            self.mdi.setActiveSubWindow(sw)
            if hasattr(self, "_log"):
                self._log(f"Duplicated as independent document -> '{new_doc.display_name()}'")

        QTimer.singleShot(0, _apply_when_ready)

    def _apply_view_state_to_view(self, view, state: dict):
        if not hasattr(view, "scroll"):
            return
        try:
            view.set_autostretch(state.get("autostretch", getattr(view, "autostretch_enabled", False)))
            view.set_autostretch_target(state.get("autostretch_target", getattr(view, "autostretch_target", 0.25)))
            view.set_scale(state.get("scale", getattr(view, "scale", 1.0)))
            QApplication.processEvents()
            view.scroll.horizontalScrollBar().setValue(int(state.get("hval", 0)))
            view.scroll.verticalScrollBar().setValue(int(state.get("vval", 0)))
        except Exception:
            pass


    def _apply_view_state_to_sub(self, state: dict, target_sw):
        view = target_sw.widget()
        if not hasattr(view, "scroll"):
            return
        view.set_autostretch(state.get("autostretch", getattr(view, "autostretch_enabled", False)))
        view.set_autostretch_target(state.get("autostretch_target", getattr(view, "autostretch_target", 0.25)))
        view.set_scale(state.get("scale", getattr(view, "scale", 1.0)))
        QApplication.processEvents()
        try:
            view.scroll.horizontalScrollBar().setValue(int(state.get("hval", 0)))
            view.scroll.verticalScrollBar().setValue(int(state.get("vval", 0)))
        except Exception:
            pass

        if hasattr(self, "_log"):
            self._log(f"Copied view state -> '{target_sw.windowTitle()}'")


    def _activate_or_open_from_explorer(self, item):
        doc = item.data(Qt.ItemDataRole.UserRole)
        base = self._normalize_base_doc(doc)

        # 1) Try to focus an existing view for this base
        for sw in self.mdi.subWindowList():
            w = sw.widget()
            if getattr(w, "base_document", None) is base:
                try:
                    sw.show(); w.show()
                    st = sw.windowState()
                    if st & Qt.WindowState.WindowMinimized:
                        sw.setWindowState(st & ~Qt.WindowState.WindowMinimized)
                    self.mdi.setActiveSubWindow(sw)
                    sw.raise_()
                except Exception:
                    pass
                return

        # 2) None exists -> open one
        self._open_subwindow_for_added_doc(base)

    def _set_linked_stretch_from_action(self, checked: bool):
        # persist as the default for *new* views
        self.settings.setValue("display/stretch_linked", bool(checked))

        # apply to the current view immediately (if any)
        sw = self.mdi.activeSubWindow()
        if not sw:
            return
        view = sw.widget()
        if hasattr(view, "set_autostretch_linked"):
            view.set_autostretch_linked(bool(checked))
            # If stretch is off, turn it on so the user sees the effect right away
            if not getattr(view, "autostretch_enabled", False):
                view.set_autostretch(True)
                self._sync_autostretch_action(True)

        self._log(f"Display-Stretch mode -> {'LINKED' if checked else 'UNLINKED'}")



    def _on_subwindow_activated(self, sw):
        # -- Clear previous active marker (guard dead wrappers)
        prev = getattr(self, "_last_active_view", None)
        if _is_alive(prev):
            try:
                # hasattr on dead wrappers can raise; gate with _is_alive first
                if hasattr(prev, "set_active_highlight"):
                    prev.set_active_highlight(False)
            except RuntimeError:
                pass
            except Exception:
                pass

        # -- Resolve the newly activated view safely
        new_view = _safe_widget(sw)
        if _is_alive(new_view):
            try:
                if hasattr(new_view, "set_active_highlight"):
                    new_view.set_active_highlight(True)
            except RuntimeError:
                pass
            except Exception:
                pass

        # Remember for next time (store None if dead/missing)
        self._last_active_view = new_view if _is_alive(new_view) else None

        # -- Safely pull the document and emit signal
        doc = None
        try:
            w = _safe_widget(sw)
            doc = w.document if _is_alive(w) and hasattr(w, "document") else None
        except RuntimeError:
            doc = None
        except Exception:
            doc = None

        try:
            self.currentDocumentChanged.emit(doc)
        except Exception:
            pass

        # Sync DocManager active, guarded
        try:
            self._sync_docman_active(doc)
        except Exception:
            pass

        # Toolbar/menu states
        try:
            if hasattr(self, "act_zoom_1_1"):
                self.act_zoom_1_1.setEnabled(bool(new_view))
        except Exception:
            pass

        # Autostretch checkbox reflect active view
        try:
            from PyQt6.QtCore import QSignalBlocker
            if hasattr(self, "act_autostretch"):
                block = QSignalBlocker(self.act_autostretch)
                if _is_alive(new_view) and hasattr(new_view, "autostretch_enabled"):
                    self.act_autostretch.setChecked(bool(getattr(new_view, "autostretch_enabled")))
                else:
                    self.act_autostretch.setChecked(False)
        except Exception:
            pass

        # Linked stretch action
        try:
            if hasattr(self, "act_stretch_linked"):
                if not _is_alive(new_view):
                    self.act_stretch_linked.setEnabled(False)
                else:
                    is_mono = False
                    try:
                        if hasattr(new_view, "is_mono"):
                            is_mono = bool(new_view.is_mono())
                    except Exception:
                        is_mono = False
                    self.act_stretch_linked.setEnabled(not is_mono)
                    if hasattr(new_view, "is_autostretch_linked"):
                        from PyQt6.QtCore import QSignalBlocker
                        with QSignalBlocker(self.act_stretch_linked):
                            try:
                                self.act_stretch_linked.setChecked(bool(new_view.is_autostretch_linked()))
                            except Exception:
                                self.act_stretch_linked.setChecked(False)
        except Exception:
            pass

        # Misc UI refreshes (guarded)
        try:
            self.update_undo_redo_action_labels()
        except Exception:
            pass
        try:
            if hasattr(self, "_hdr_refresh_timer") and self._hdr_refresh_timer is not None:
                self._hdr_refresh_timer.start(0)
        except Exception:
            pass
        try:
            self._refresh_mask_action_states()
        except Exception:
            pass

    def _sync_docman_active(self, doc):
        dm = self.doc_manager
        try:
            if hasattr(dm, "set_active_document") and callable(dm.set_active_document):
                dm.set_active_document(doc)
            else:
                # best-effort fallback
                setattr(dm, "_active_document", doc)
        except Exception:
            pass

    def _list_open_docs(self):
        docs = []
        for sw in self.mdi.subWindowList():
            d = getattr(sw.widget(), "document", None)
            if d:
                docs.append(d)
        return docs

    def _active_view(self):
        sw = self.mdi.activeSubWindow()
        return sw.widget() if sw else None

    def _active_doc(self):
        dm = getattr(self, "doc_manager", None) or getattr(self, "docman", None)
        if not dm:
            return None
        return dm.get_active_document()

    def _document_has_edits(self, doc) -> bool:
        # Prefer a dedicated 'dirty' indicator if your ImageDocument exposes one
        dirty_attr = getattr(doc, "dirty", None)
        if callable(dirty_attr):
            try:
                return bool(dirty_attr())
            except Exception:
                pass
        elif isinstance(dirty_attr, bool):
            return dirty_attr

        # Fallback heuristic: anything to undo = user edited since load/reset
        try:
            return bool(doc.can_undo())
        except Exception:
            return False

    def _log(self, msg):
        self.console.addItem(msg)
        self.console.scrollToBottom()

    def _safe_close_doc(self, doc):
        # skip if app is shutting down or docman already gone
        if self._shutting_down:
            return
        dm = getattr(self, "docman", None)
        if dm is None:
            return
        try:
            if sip.isdeleted(dm):
                return
        except Exception:
            # sip may not be able to inspect; fall back and hope for the best
            pass
        dm.close_document(doc)

    def _restore_window_placement(self):
        s = self.settings
        try:
            geo = s.value("ui/main/geometry")
            st  = s.value("ui/main/state")
            is_max = s.value("ui/main/maximized", False, type=bool)

            if geo is not None:
                self.restoreGeometry(geo)
            if st is not None:
                self.restoreState(st, version=1)

            # Make sure we're on a visible screen
            from PyQt6.QtGui import QGuiApplication
            r = self.frameGeometry()
            scr = QGuiApplication.screenAt(r.center()) or QGuiApplication.primaryScreen()
            if scr:
                ag = scr.availableGeometry()
                if not ag.intersects(r):
                    # Center on the available geometry
                    self.move(ag.center() - self.rect().center())

            if is_max:
                self.showMaximized()
        except Exception:
            pass

    def _ensure_persistent_names(self):
        def _safe_name(title: str, prefix: str):
            t = title or ""
            t = "".join(ch if ch.isalnum() else "_" for ch in t)
            t = t.strip("_") or "Untitled"
            return f"{prefix}_{t}"

        # Docks
        for dock in self.findChildren(QDockWidget):
            if not dock.objectName():
                dock.setObjectName(_safe_name(dock.windowTitle(), "Dock"))

        # Toolbars
        for tb in self.findChildren(QToolBar):
            if not tb.objectName():
                tb.setObjectName(_safe_name(tb.windowTitle(), "Toolbar"))

    def _clear_view_bundles_for_next_launch(self):
        """
        On app exit, wipe any saved doc_ptrs in View Bundles so they don't point
        to stale objects on next run. We keep bundle names/uuids/file_paths, just
        empty the 'doc_ptrs' lists.

        This handles both the old v2 and the new v3 stores.
        """
        try:
            s = QSettings()

            def _scrub_doc_ptrs(key: str):
                raw = s.value(key, "[]", type=str) or "[]"
                try:
                    data = json.loads(raw)
                except Exception:
                    data = []
                if isinstance(data, list):
                    for b in data:
                        if isinstance(b, dict):
                            b["doc_ptrs"] = []  # drop all view pointers
                    s.setValue(key, json.dumps(data, ensure_ascii=False))
                else:
                    s.setValue(key, "[]")

            # scrub both generations
            _scrub_doc_ptrs("viewbundles/v2")
            _scrub_doc_ptrs("viewbundles/v3")

            # nuke legacy v1 entirely
            s.setValue("viewbundles/v1", "[]")

            s.sync()
        except Exception:
            # last-resort: write empty arrays everywhere
            try:
                s = QSettings()
                s.setValue("viewbundles/v1", "[]")
                s.setValue("viewbundles/v2", "[]")
                s.setValue("viewbundles/v3", "[]")
                s.sync()
            except Exception:
                pass


    def showEvent(self, ev):
        super().showEvent(ev)
        # when returning from the taskbar, some platforms don't emit WindowStateChange reliably
        if self._suspend_dock_sync:
            QTimer.singleShot(0, lambda: self.changeEvent(QEvent(QEvent.Type.WindowStateChange)))

    def closeEvent(self, e):

        try:
            if hasattr(self, "_orig_stdout") and self._orig_stdout is not None:
                sys.stdout = self._orig_stdout
            if hasattr(self, "_orig_stderr") and self._orig_stderr is not None:
                sys.stderr = self._orig_stderr
        except Exception:
            pass        
        self._shutting_down = True
        # Gather open docs
        docs = []
        for sw in self.mdi.subWindowList():
            vw = sw.widget()
            d = getattr(vw, "document", None)
            if d and d not in docs:
                docs.append(d)

        edited = [d for d in docs if self._document_has_edits(d)]
        msg = "Exit Seti Astro Suite Pro?"
        detail = []
        if docs:
            detail.append(f"Open images: {len(docs)}")
        if edited:
            detail.append(f"Edited since open: {len(edited)}")
        if detail:
            msg += "\n\n" + "\n".join(detail)

        # --- stay-on-top message box ---
        mbox = QMessageBox(self)
        mbox.setIcon(QMessageBox.Icon.Question)
        mbox.setWindowTitle("Confirm Exit")
        mbox.setText(msg)
        mbox.setStandardButtons(
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        mbox.setDefaultButton(QMessageBox.StandardButton.No)
        # ðŸ'‡ key line
        mbox.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
        mbox.raise_()
        mbox.activateWindow()
        btn = mbox.exec()

        if btn != QMessageBox.StandardButton.Yes:
            e.ignore()
            return

        # User confirmed: prevent per-subwindow prompts and proceed
        self._force_close_all = True
        self._shutting_down = True

        # Save UI layout/placement
        self._ensure_persistent_names()
        try:
            if self.isMaximized():
                self.settings.setValue("ui/main/maximized", True)
                self.showNormal()
                self.settings.setValue("ui/main/geometry", self.saveGeometry())
                self.showMaximized()
            else:
                self.settings.setValue("ui/main/maximized", False)
                self.settings.setValue("ui/main/geometry", self.saveGeometry())

            self.settings.setValue("ui/main/state", self.saveState(version=1))
        except Exception:
            pass

        # save shortcuts
        try:
            save_on_exit = self.settings.value("shortcuts/save_on_exit", True, type=bool)
        except Exception:
            save_on_exit = True
        if save_on_exit and hasattr(self, "shortcuts"):
            try:
                self.shortcuts.save_shortcuts()
            except Exception:
                pass

        # wait on bg threads
        try:
            for t in list(getattr(self, "_bg_threads", [])):
                try:
                    t.wait(3000)
                except Exception:
                    pass
        except Exception:
            pass

        self._clear_view_bundles_for_next_launch()

        super().closeEvent(e)

# CheatSheet dialog and helper functions imported from pro.cheat_sheet
from pro.cheat_sheet import (
    CheatSheetDialog as _CheatSheetDialog,
    _qs_to_str,
    _clean_text,
    _uniq_keep_order,
    _seqs_for_action,
    _where_for_action,
    _describe_action,
    _describe_shortcut,
    _where_for_shortcut,
)

# _ProjectSaveWorker and install_crash_handlers imported from pro.widgets.common_utilities

