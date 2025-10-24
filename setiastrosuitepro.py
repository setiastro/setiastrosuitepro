#setiastrosuitepro.py
from pro.runtime_torch import add_runtime_to_sys_path, _ban_shadow_torch_paths, _purge_bad_torch_from_sysmodules
add_runtime_to_sys_path(status_cb=lambda *_: None)
_ban_shadow_torch_paths(status_cb=lambda *_: None)
_purge_bad_torch_from_sysmodules(status_cb=lambda *_: None)


import sys
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
import os
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


from scipy.ndimage import gaussian_filter, laplace


from scipy.ndimage import zoom
import multiprocessing
import matplotlib
matplotlib.use("QtAgg") 

import numpy as np


#if running in IDE which runs ipython or jupiter in backend reconfigure may not be available
if (sys.stdout is not None) and (hasattr(sys.stdout, "reconfigure")):
    sys.stdout.reconfigure(encoding='utf-8')

try:
    from photutils.isophote import Ellipse, EllipseGeometry, build_ellipse_model
except Exception as e:
    Ellipse = EllipseGeometry = build_ellipse_model = None


from numba_utils import *


import lightkurve as lk


lk.MPLSTYLE = None



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
from PyQt6.QtWidgets import (QDialog, QApplication, QMainWindow, QWidget, QHBoxLayout, QFileDialog, QMessageBox, QSizePolicy, QToolBar, QPushButton,
    QLineEdit, QMenu, QListWidget, QListWidgetItem, QSplashScreen, QDockWidget, QListView, QCompleter, QMdiArea, QMdiArea, QMdiSubWindow, 
    QInputDialog, QVBoxLayout, QLabel, QCheckBox, QProgressBar, QProgressDialog, QGraphicsItem, QTabWidget, QTableWidget, QHeaderView, QTableWidgetItem, QToolButton
)

# ----- QtGui -----
from PyQt6.QtGui import (QPixmap, QColor, QIcon, QKeySequence, QShortcut, QGuiApplication, QStandardItemModel, QStandardItem, QAction, QPalette, QBrush
)

# ----- QtCore -----
from PyQt6.QtCore import (Qt, pyqtSignal, QCoreApplication, QTimer, QSize, QSignalBlocker,  QModelIndex, QThread, QUrl, QSettings, QEvent, QByteArray
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
from pro.dnd_mime import MIME_VIEWSTATE, MIME_CMD, MIME_MASK, MIME_ASTROMETRY
from pro.graxpert import remove_gradient_with_graxpert
from pro.remove_stars import remove_stars
from pro.add_stars import add_stars 
from pro.window_shelf import WindowShelf, MinimizeInterceptor
from pro.pedestal import remove_pedestal
from pro.remove_green import open_remove_green_dialog, apply_remove_green_preset_to_doc
from pro.backgroundneutral import BackgroundNeutralizationDialog, apply_background_neutral_to_doc
from pro.luminancerecombine import apply_recombine_to_doc, compute_luminance, _to_float01
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
from numba_utils import (
    rescale_image_numba,
    flip_horizontal_numba,
    flip_vertical_numba,
    rotate_90_clockwise_numba,
    rotate_90_counterclockwise_numba,
    invert_image_numba,
)
from pro.project_io import ProjectWriter, ProjectReader
from pro.psf_viewer import PSFViewer
from pro.plate_solver import plate_solve_doc_inplace, PlateSolverDialog
from pro.star_alignment import StellarAlignmentDialog, StarRegistrationWindow, MosaicMasterDialog
from pro.image_peeker_pro import ImagePeekerDialogPro
from pro.live_stacking import LiveStackWindow
from pro.stacking_suite import StackingSuiteDialog
from pro.supernovaasteroidhunter import SupernovaAsteroidHunterDialog
from pro.star_spikes import StarSpikesDialogPro
from pro.exoplanet_detector import ExoPlanetWindow
from pro.isophote import IsophoteModelerDialog
from wims import WhatsInMySkyDialog
from wimi import WIMIDialog 
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


VERSION = "1.3.18"



if hasattr(sys, '_MEIPASS'):
    # PyInstaller path
    icon_path = os.path.join(sys._MEIPASS, 'astrosuitepro.png')
    windowslogo_path = os.path.join(sys._MEIPASS, 'astrosuitepro.ico')
    green_path = os.path.join(sys._MEIPASS, 'green.png')
    neutral_path = os.path.join(sys._MEIPASS, 'neutral.png')
    whitebalance_path = os.path.join(sys._MEIPASS, 'whitebalance.png')
    morpho_path = os.path.join(sys._MEIPASS, 'morpho.png')
    clahe_path = os.path.join(sys._MEIPASS, 'clahe.png')
    starnet_path = os.path.join(sys._MEIPASS, 'starnet.png')
    staradd_path = os.path.join(sys._MEIPASS, 'staradd.png')
    LExtract_path = os.path.join(sys._MEIPASS, 'LExtract.png')
    LInsert_path = os.path.join(sys._MEIPASS, 'LInsert.png')
    slot0_path = os.path.join(sys._MEIPASS, 'slot0.png')
    slot1_path = os.path.join(sys._MEIPASS, 'slot1.png')
    slot2_path = os.path.join(sys._MEIPASS, 'slot2.png')
    slot3_path = os.path.join(sys._MEIPASS, 'slot3.png')
    slot4_path = os.path.join(sys._MEIPASS, 'slot4.png')
    rgbcombo_path = os.path.join(sys._MEIPASS, 'rgbcombo.png')
    rgbextract_path = os.path.join(sys._MEIPASS, 'rgbextract.png')
    copyslot_path = os.path.join(sys._MEIPASS, 'copyslot.png')
    graxperticon_path = os.path.join(sys._MEIPASS, 'graxpert.png')
    cropicon_path = os.path.join(sys._MEIPASS, 'cropicon.png')
    openfile_path = os.path.join(sys._MEIPASS, 'openfile.png')
    abeicon_path = os.path.join(sys._MEIPASS, 'abeicon.png')    
    undoicon_path = os.path.join(sys._MEIPASS, 'undoicon.png')  
    redoicon_path = os.path.join(sys._MEIPASS, 'redoicon.png')  
    blastericon_path = os.path.join(sys._MEIPASS, 'blaster.png')
    hdr_path = os.path.join(sys._MEIPASS, 'hdr.png')  
    invert_path = os.path.join(sys._MEIPASS, 'invert.png')  
    fliphorizontal_path = os.path.join(sys._MEIPASS, 'fliphorizontal.png')
    flipvertical_path = os.path.join(sys._MEIPASS, 'flipvertical.png')
    rotateclockwise_path = os.path.join(sys._MEIPASS, 'rotateclockwise.png')
    rotatecounterclockwise_path = os.path.join(sys._MEIPASS, 'rotatecounterclockwise.png')
    maskcreate_path = os.path.join(sys._MEIPASS, 'maskcreate.png')
    maskapply_path = os.path.join(sys._MEIPASS, 'maskapply.png')
    maskremove_path = os.path.join(sys._MEIPASS, 'maskremove.png')
    slot5_path = os.path.join(sys._MEIPASS, 'slot5.png')
    slot6_path = os.path.join(sys._MEIPASS, 'slot6.png')
    slot7_path = os.path.join(sys._MEIPASS, 'slot7.png')
    slot8_path = os.path.join(sys._MEIPASS, 'slot8.png')
    slot9_path = os.path.join(sys._MEIPASS, 'slot9.png') 
    pixelmath_path = os.path.join(sys._MEIPASS, 'pixelmath.png')   
    histogram_path = os.path.join(sys._MEIPASS, 'histogram.png') 
    mosaic_path = os.path.join(sys._MEIPASS, 'mosaic.png')
    rescale_path = os.path.join(sys._MEIPASS, 'rescale.png')
    staralign_path = os.path.join(sys._MEIPASS, 'staralign.png')
    mask_path = os.path.join(sys._MEIPASS, 'maskapply.png')
    platesolve_path = os.path.join(sys._MEIPASS, 'platesolve.png')
    psf_path = os.path.join(sys._MEIPASS, 'psf.png')
    supernova_path = os.path.join(sys._MEIPASS, 'supernova.png')
    starregistration_path = os.path.join(sys._MEIPASS, 'starregistration.png')
    stacking_path = os.path.join(sys._MEIPASS, 'stacking.png')
    pedestal_icon_path = os.path.join(sys._MEIPASS, 'pedestal.png')
    starspike_path = os.path.join(sys._MEIPASS, 'starspike.png')
    aperture_path = os.path.join(sys._MEIPASS, 'aperture.png')
    jwstpupil_path = os.path.join(sys._MEIPASS, 'jwstpupil.png')
    signature_icon_path = os.path.join(sys._MEIPASS, 'pen.png')
    livestacking_path = os.path.join(sys._MEIPASS, 'livestacking.png')
    hrdiagram_path = os.path.join(sys._MEIPASS, 'HRDiagram.png')
    convoicon_path = os.path.join(sys._MEIPASS, 'convo.png')
    spcc_icon_path = os.path.join(sys._MEIPASS, 'spcc.png')
    sasp_data_path = os.path.join(sys._MEIPASS, 'SASP_data.fits')
    exoicon_path = os.path.join(sys._MEIPASS, 'exoicon.png')
    peeker_icon = os.path.join(sys._MEIPASS, 'gridicon.png')
    dse_icon_path = os.path.join(sys._MEIPASS, 'dse.png')
    astrobin_filters_csv_path = os.path.join(sys._MEIPASS, 'astrobin_filters.csv')
    isophote_path = os.path.join(sys._MEIPASS, 'isophote.png')
    statstretch_path = os.path.join(sys._MEIPASS, 'statstretch.png')
    starstretch_path = os.path.join(sys._MEIPASS, 'starstretch.png')
    curves_path = os.path.join(sys._MEIPASS, 'curves.png')
    disk_path = os.path.join(sys._MEIPASS, 'disk.png')
    uhs_path = os.path.join(sys._MEIPASS, 'uhs.png')
    blink_path = os.path.join(sys._MEIPASS, 'blink.png')
    ppp_path = os.path.join(sys._MEIPASS, 'ppp.png')
    nbtorgb_path = os.path.join(sys._MEIPASS, 'nbtorgb.png')
    freqsep_path = os.path.join(sys._MEIPASS, 'freqsep.png')
    contsub_path = os.path.join(sys._MEIPASS, 'contsub.png')
    halo_path = os.path.join(sys._MEIPASS, 'halo.png')
    cosmic_path = os.path.join(sys._MEIPASS, 'cosmic.png')
    satellite_path= os.path.join(sys._MEIPASS, 'cosmicsat.png')
    imagecombine_path = os.path.join(sys._MEIPASS, 'imagecombine.png')
    wrench_path = os.path.join(sys._MEIPASS, 'wrench_icon.png')
    eye_icon_path = os.path.join(sys._MEIPASS, 'eye.png')
    disk_icon_path = os.path.join(sys._MEIPASS, 'disk.png')
    nuke_path = os.path.join(sys._MEIPASS, 'nuke.png')  
    hubble_path = os.path.join(sys._MEIPASS, 'hubble.png') 
    collage_path = os.path.join(sys._MEIPASS, 'collage.png') 
    annotated_path = os.path.join(sys._MEIPASS, 'annotated.png') 
    colorwheel_path = os.path.join(sys._MEIPASS, 'colorwheel.png')
    font_path = os.path.join(sys._MEIPASS, 'font.png')
    csv_icon_path = os.path.join(sys._MEIPASS, 'cvs.png')
    spinner_path = os.path.join(sys._MEIPASS, 'spinner.gif')
    wims_path = os.path.join(sys._MEIPASS, 'wims.png')
    wimi_path = os.path.join(sys._MEIPASS, 'wimi_icon_256x256.png')
    linearfit_path= os.path.join(sys._MEIPASS, 'linearfit.png')
    debayer_path = os.path.join(sys._MEIPASS, 'debayer.png')
    aberration_path = os.path.join(sys._MEIPASS, 'aberration.png')
    functionbundles_path = os.path.join(sys._MEIPASS, 'functionbundle.png')
    viewbundles_path = os.path.join(sys._MEIPASS, 'viewbundle.png')
else:
    # Development path
    icon_path = 'astrosuitepro.png'
    windowslogo_path = 'astrosuitepro.ico'
    green_path = 'green.png'
    neutral_path = 'neutral.png'
    whitebalance_path = 'whitebalance.png'
    morpho_path = 'morpho.png'
    clahe_path = 'clahe.png'
    starnet_path = 'starnet.png'
    staradd_path = 'staradd.png'
    LExtract_path = 'LExtract.png'
    LInsert_path = 'LInsert.png'
    slot1_path = 'slot1.png'
    slot0_path = 'slot0.png'
    slot2_path = 'slot2.png'
    slot3_path  = 'slot3.png'
    slot4_path  = 'slot4.png'
    rgbcombo_path = 'rgbcombo.png'
    rgbextract_path = 'rgbextract.png'
    copyslot_path = 'copyslot.png'
    graxperticon_path = 'graxpert.png'
    cropicon_path = 'cropicon.png'
    openfile_path = 'openfile.png'
    abeicon_path = 'abeicon.png'
    undoicon_path = 'undoicon.png'
    redoicon_path = 'redoicon.png'
    blastericon_path = 'blaster.png'
    hdr_path = 'hdr.png'
    invert_path = 'invert.png'
    fliphorizontal_path = 'fliphorizontal.png'
    flipvertical_path = 'flipvertical.png'
    rotateclockwise_path = 'rotateclockwise.png'
    rotatecounterclockwise_path = 'rotatecounterclockwise.png'
    maskcreate_path = 'maskcreate.png'
    maskapply_path = 'maskapply.png'
    maskremove_path = 'maskremove.png'
    slot5_path = 'slot5.png'
    slot6_path = 'slot6.png'
    slot7_path = 'slot7.png'
    slot8_path  = 'slot8.png'
    slot9_path  = 'slot9.png'
    pixelmath_path = 'pixelmath.png'
    histogram_path = 'histogram.png'
    mosaic_path = 'mosaic.png'
    rescale_path = 'rescale.png'
    staralign_path = 'staralign.png'
    mask_path = 'maskapply.png'
    platesolve_path = 'platesolve.png'
    psf_path = 'psf.png'
    supernova_path = 'supernova.png'
    starregistration_path = 'starregistration.png'
    stacking_path = 'stacking.png'
    pedestal_icon_path = 'pedestal.png'
    starspike_path = 'starspike.png'
    aperture_path = 'aperture.png'
    jwstpupil_path = 'jwstpupil.png'
    signature_icon_path = 'pen.png'
    livestacking_path = 'livestacking.png'
    hrdiagram_path = 'HRDiagram.png'
    convoicon_path = 'convo.png'
    spcc_icon_path = 'spcc.png'
    sasp_data_path = 'SASP_data.fits'
    exoicon_path = 'exoicon.png'
    peeker_icon = 'gridicon.png'
    dse_icon_path = 'dse.png'
    astrobin_filters_csv_path = 'astrobin_filters.csv'
    isophote_path = 'isophote.png'
    statstretch_path = 'statstretch.png'
    starstretch_path = 'starstretch.png'
    curves_path = 'curves.png'
    disk_path = 'disk.png'
    uhs_path = 'uhs.png'
    blink_path = 'blink.png'
    ppp_path = 'ppp.png'
    nbtorgb_path = 'nbtorgb.png'
    freqsep_path = 'freqsep.png'
    contsub_path = 'contsub.png'
    halo_path = 'halo.png'
    cosmic_path = 'cosmic.png'
    satellite_path = 'cosmicsat.png'
    imagecombine_path = 'imagecombine.png'
    wrench_path = 'wrench_icon.png'  # Path for running as a script
    eye_icon_path = 'eye.png'  # Path for running as a script
    disk_icon_path = 'disk.png'   
    nuke_path = 'nuke.png' 
    hubble_path = 'hubble.png'
    collage_path = 'collage.png'
    annotated_path = 'annotated.png'
    colorwheel_path = 'colorwheel.png'
    font_path = 'font.png'
    csv_icon_path = 'cvs.png'
    spinner_path = 'spinner.gif'
    wims_path = 'wims.png'
    wimi_path = 'wimi_icon_256x256.png'
    linearfit_path = 'linearfit.png'
    debayer_path = 'debayer.png'
    aberration_path = 'aberration.png'
    functionbundles_path = 'functionbundle.png'
    viewbundles_path = 'viewbundle.png'

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

class MdiArea(QMdiArea):
    backgroundDoubleClicked = pyqtSignal()
    viewStateDropped = pyqtSignal(dict, object)   # (state_dict, target_subwindow or None)
    commandDropped   = pyqtSignal(dict, object)   # ({"command_id","preset"}, target_subwindow or None)
    maskDropped = pyqtSignal(dict, object)  # (payload, target_subwindow or None)
    astrometryDropped = pyqtSignal(dict, object)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, e):
        if (e.mimeData().hasFormat("application/x-sas-viewstate")
                or e.mimeData().hasFormat(MIME_CMD)
                or e.mimeData().hasFormat(MIME_MASK)
                or e.mimeData().hasFormat(MIME_ASTROMETRY)):
            e.acceptProposedAction()
        else:
            super().dragEnterEvent(e)

    def dropEvent(self, e):
        pos = e.position().toPoint()

        # Map the event position from the MdiArea into the viewport's coords.
        vp = self.viewport()
        vp_pos = vp.mapFrom(self, pos) if vp is not None else pos

        # Get subwindows in real z-order (back → front). Qt6: WindowOrder.StackingOrder
        try:
            order_enum = getattr(QMdiArea, "WindowOrder", None)
            subwins = self.subWindowList(order_enum.StackingOrder) if order_enum else self.subWindowList()
        except Exception:
            subwins = self.subWindowList()

        # Pick the visually top-most window under the cursor
        target = None
        for sw in reversed(subwins):            # reversed: front-most first
            if sw.isVisible() and sw.geometry().contains(vp_pos):
                target = sw
                break

        # 1) existing: view-state payload
        if e.mimeData().hasFormat("application/x-sas-viewstate"):
            try:
                raw = bytes(e.mimeData().data("application/x-sas-viewstate"))
                state = json.loads(raw.decode("utf-8"))
            except Exception:
                e.ignore(); return
            self.viewStateDropped.emit(state, target)  # None ⇒ duplicate on background
            e.acceptProposedAction()
            return

        # 2) command + preset payload (from shortcuts)
        if e.mimeData().hasFormat(MIME_CMD):
            try:
                payload = _unpack_cmd_payload(bytes(e.mimeData().data(MIME_CMD)))
            except Exception:
                e.ignore(); return
            self.commandDropped.emit(payload, target)
            e.acceptProposedAction()
            return

        # 3) mask payload (from subwindow DnD)
        if e.mimeData().hasFormat(MIME_MASK):
            try:
                raw = bytes(e.mimeData().data(MIME_MASK))
                payload = json.loads(raw.decode("utf-8"))
            except Exception:
                e.ignore(); return
            self.maskDropped.emit(payload, target)
            e.acceptProposedAction()
            return
        # 4) Astrometric payload (from subwindow DnD)
        if e.mimeData().hasFormat(MIME_ASTROMETRY):
            try:
                raw = bytes(e.mimeData().data(MIME_ASTROMETRY))
                payload = json.loads(raw.decode("utf-8"))
            except Exception:
                e.ignore(); return
            self.astrometryDropped.emit(payload, target)
            e.acceptProposedAction()
            return

        # fallback
        super().dropEvent(e)

    def mouseDoubleClickEvent(self, event):
        pt = event.position().toPoint() if hasattr(event, "position") else event.pos()
        for sw in self.subWindowList():
            if sw.geometry().contains(pt):
                return super().mouseDoubleClickEvent(event)

        self.backgroundDoubleClicked.emit()
        event.accept()

_ROLE_ACTION = Qt.ItemDataRole.UserRole + 1

class AboutDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("About Seti Astro Suite")
        layout = QVBoxLayout()

        about_text = (
            f"<h2>Seti Astro's Suite Pro {VERSION}</h2>"
            "<p>Written by Franklin Marek</p>"
            "<p>Copyright © 2025 Seti Astro</p>"
            f"<p><b>Build:</b> {BUILD_TIMESTAMP}</p>"
            "<p>Website: <a href='http://www.setiastro.com'>www.setiastro.com</a></p>"
            "<p>Donations: <a href='https://www.setiastro.com/checkout/donate?donatePageId=65ae7e7bac20370d8c04c1ab'>Click here to donate</a></p>"
        )
        label = QLabel(about_text)
        label.setTextFormat(Qt.TextFormat.RichText)
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        label.setOpenExternalLinks(True)

        layout.addWidget(label)
        self.setLayout(layout)

DECOR_GLYPHS = "■●◆▲▪▫•◼◻◾◽"
def _strip_ui_decorations(s: str) -> str:
    s = s or ""
    # strip any number of leading glyph+space
    while len(s) >= 2 and s[1] == " " and s[0] in DECOR_GLYPHS:
        s = s[2:]
    # strip leading Active prefix if present
    ACTIVE = "Active View: "
    if s.startswith(ACTIVE):
        s = s[len(ACTIVE):]
    return s

def _exts_from_filter(selected_filter: str) -> list[str]:
    """
    Extract extensions from a Qt name filter string like:
      "TIFF (*.tif *.tiff)"
    Returns normalized (e.g., 'tiff' -> 'tif', 'jpeg' -> 'jpg').
    """
    exts = [m.group(1).lower() for m in re.finditer(r"\*\.\s*([A-Za-z0-9]+)", selected_filter)]
    if not exts:
        return []
    # normalize & uniquify while preserving order
    seen = set()
    out = []
    for e in exts:
        n = _normalize_ext(e)
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out

# --- filename/path hardening helpers ---

import os, re, sys
from pathlib import Path

# If you want to keep spaces, set to False
_REPLACE_SPACES_WITH_UNDERSCORES = True

_WIN_RESERVED = {
    "CON","PRN","AUX","NUL",
    *(f"COM{i}" for i in range(1,10)),
    *(f"LPT{i}" for i in range(1,10)),
}

def _sanitize_filename(basename: str, replace_spaces: bool = _REPLACE_SPACES_WITH_UNDERSCORES) -> str:
    """
    Returns a safe file basename (no directories), with:
      - collapsed/trimmed whitespace
      - optional spaces→underscores
      - illegal characters removed (Windows/macOS/Linux superset)
      - no leading/trailing dots/spaces
      - Windows reserved device names avoided by appending '_'
    """
    name = (basename or "").strip()

    # Split name/ext carefully (may be empty if user typed only ext)
    stem, ext = os.path.splitext(name)

    # collapse weird whitespace in stem
    stem = re.sub(r"\s+", " ", stem).strip()

    if replace_spaces:
        stem = stem.replace(" ", "_")

    # Remove characters illegal or risky on common platforms
    # Windows: <>:"/\|?* ; also control chars; mac legacy ':' ; keep unicode letters/numbers
    stem = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "", stem)
    stem = stem.replace(":", "")  # legacy macOS

    # On POSIX, '/' is the only illegal path char, but we already stripped it above.

    # Strip trailing dots/spaces (Windows)
    stem = stem.rstrip(" .")

    # Avoid empty or dot-only names
    if not stem or set(stem) == {"."}:
        stem = "untitled"

    # Avoid Windows reserved device names (case-insensitive)
    if sys.platform.startswith("win") and stem.upper() in _WIN_RESERVED:
        stem = stem + "_"

    # Guard overall length (very conservative 200 for basename)
    if len(stem) > 200:
        stem = stem[:200]

    # Clean extension too (just in case user typed garbage)
    ext = re.sub(r'[<>:"/\\|?*\s\x00-\x1F]', "", ext)

    # Final safety: if ext is just a dot or empty, drop it here
    if ext in (".", ""):
        ext = ""

    return stem + ext


def _safe_join_dir_and_name(directory: str, basename: str) -> str:
    """
    Join directory + sanitized basename. Ensures the directory exists or raises a clear error.
    """
    safe_name = _sanitize_filename(basename)
    final_dir = directory or ""
    if final_dir and not os.path.isdir(final_dir):
        # attempt to create; ignore if it fails (docman.save_document may still error, which we catch)
        try:
            os.makedirs(final_dir, exist_ok=True)
        except Exception:
            pass
    return os.path.join(final_dir, safe_name)


def _exts_from_filter(selected_filter: str) -> list[str]:
    # unchanged; your existing helper
    exts = re.findall(r"\*\.(\w+)", selected_filter or "")
    return [e.lower() for e in exts] if exts else []


def _normalize_ext(ext: str) -> str:
    # unchanged; your existing helper
    return ext.lower().lstrip(".")


def _normalize_save_path_chosen_filter(path: str, selected_filter: str) -> tuple[str, str]:
    """
    Returns (final_path, final_ext_norm). Ensures:
      - appends extension if missing (from chosen filter)
      - avoids double extensions (*.png.png)
      - if user provided a conflicting ext, enforce the chosen filter’s default
      - sanitizes the basename (spaces, illegal chars, trailing dots)
    """
    raw_path = (path or "").strip().rstrip(".")
    allowed = _exts_from_filter(selected_filter) or ["png"]  # safe fallback
    default_ext = allowed[0]

    # split dir + basename (sanitize only the basename)
    directory, base = os.path.split(raw_path)
    if not base:
        base = "untitled"

    # if the user typed something like "name.png" but selected TIFF, we’ll fix after sanitization
    base_stem, base_ext = os.path.splitext(base)
    typed = _normalize_ext(base_ext) if base_ext else ""

    # remove repeated extension in stem (e.g. "foo.png" then + ".png")
    def strip_trailing_allowed(stem: str) -> str:
        lowered = stem.lower()
        for a in allowed:
            suf = "." + a
            if lowered.endswith(suf):
                return stem[:-len(suf)]
        return stem

    base_stem = strip_trailing_allowed(base_stem)

    # choose final extension
    if not typed:
        final_ext = default_ext
    else:
        final_ext = typed if typed in allowed else default_ext

    # rebuild name with the chosen extension, then sanitize the WHOLE basename
    basename_target = f"{base_stem}.{final_ext}"
    basename_safe = _sanitize_filename(basename_target, replace_spaces=_REPLACE_SPACES_WITH_UNDERSCORES)

    # final join (create dir if missing)
    final_path = _safe_join_dir_and_name(directory, basename_safe)
    return final_path, final_ext


def _display_name(doc) -> str:
    """Best-effort title for any doc-like object."""
    # Prefer a method
    for attr in ("display_name", "title", "name"):
        v = getattr(doc, attr, None)
        if callable(v):
            try:
                s = v()
                if isinstance(s, str) and s.strip():
                    return s
            except Exception:
                pass
        elif isinstance(v, str) and v.strip():
            return v

    # Metadata fallbacks
    md = getattr(doc, "metadata", {}) or {}
    if isinstance(md, dict):
        for k in ("display_name", "title", "name", "filename", "basename"):
            s = md.get(k)
            if isinstance(s, str) and s.strip():
                return s

    # Last resort: id snippet
    return f"Document-{id(doc) & 0xFFFF:04X}"

def _best_doc_name(doc) -> str:
    # try common attributes in order
    for attr in ("display_name", "name", "title"):
        v = getattr(doc, attr, None)
        if callable(v):
            try:
                v = v()
            except Exception:
                v = None
        if isinstance(v, str) and v.strip():
            return v.strip()

    # fallback: derive from original path if we have it
    try:
        meta = getattr(doc, "metadata", {}) or {}
        fp = meta.get("file_path")
        if isinstance(fp, str) and fp:
            return os.path.splitext(os.path.basename(fp))[0]
    except Exception:
        pass

    return "untitled"

def _doc_looks_like_table(doc) -> bool:
    md = getattr(doc, "metadata", {}) or {}

    # explicit type hints from your own pipeline
    if str(md.get("doc_type", "")).lower() in {"table", "catalog", "fits_table"}:
        return True
    if str(md.get("fits_hdu_type", "")).lower().endswith("tablehdu"):
        return True
    if str(md.get("hdu_class", "")).lower().endswith("tablehdu"):
        return True

    # FITS header inspection (common with astropy)
    hdr = md.get("original_header") or md.get("fits_header") or {}
    try:
        xt = str(hdr.get("XTENSION", "")).upper()
        if xt in {"TABLE", "BINTABLE", "ASCIITABLE"}:
            return True
    except Exception:
        pass

    # structural hints from the doc
    if hasattr(doc, "table"):
        return True
    if hasattr(doc, "columns"):
        return True
    if hasattr(doc, "rows") or hasattr(doc, "headers"):
        return True

    # last resort: no image but we clearly have column metadata
    if getattr(doc, "image", None) is None and isinstance(md.get("columns"), (list, tuple)):
        return True

    return False

class AstroSuiteProMainWindow(QMainWindow):
    currentDocumentChanged = pyqtSignal(object)  # ImageDocument | None

    def __init__(self, image_manager=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Seti Astro Suite Pro v{VERSION}")
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
        

        # Core
        self.doc_manager = DocManager(image_manager=image_manager, parent=self)
        self.docman = self.doc_manager  # legacy alias for older code

        # MDI workspace
        self.mdi = MdiArea()
        self.mdi.setViewMode(QMdiArea.ViewMode.SubWindowView)
        self.mdi.subWindowActivated.connect(self._on_subwindow_activated)
        self.mdi.subWindowActivated.connect(self._remember_active_pair)
        self.mdi.backgroundDoubleClicked.connect(self.open_files)   # ← new
        QShortcut(QKeySequence("Ctrl+PgDown"), self, activated=self._toggle_last_active_view)
        QShortcut(QKeySequence("Ctrl+PgUp"), self, activated=self._toggle_last_active_view)
        self.setCentralWidget(self.mdi)

        self.window_shelf = WindowShelf(self)
        self.window_shelf.setObjectName("WindowShelfDock") 
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.window_shelf)
        self.window_shelf.hide()

        self._minimize_interceptor = MinimizeInterceptor(self.window_shelf, self)
        self.currentDocumentChanged.connect(self._sync_docman_active)

        # Docks
        self._init_explorer_dock()
        self._init_console_dock()
        self._init_header_viewer_dock()
        self._init_layers_dock()
        self._shutting_down = False
        self._init_status_log_dock()


        # Toolbar / actions
        self._create_actions()
        self._init_menubar()
        self._init_toolbar()
        self._install_command_search()

        # Keep explorer in sync
        self.docman.documentAdded.connect(self._add_doc_to_explorer)
        self.docman.documentRemoved.connect(self._remove_doc_from_explorer)
        self.mdi.viewStateDropped.connect(self._handle_viewstate_drop)
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

        self.doc_manager.set_mdi_area(self.mdi)

        self.shortcuts.load_shortcuts()
        self._ensure_persistent_names() 
        self._restore_window_placement()

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
            # "ThemeChange",             # NOT reliable—do NOT include
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
            self._last_good_state = self.saveState()
        except Exception:
            pass

        self.status_log_dock.hide()

    def _alive(self, obj) -> bool:
        if obj is None:
            return False
        try:
            _ = obj.metaObject()  # raises if C++ object already deleted
            return True
        except RuntimeError:
            return False

    # ---------- THEME API ----------
    def _apply_workspace_theme(self):
        """Retint the QMdiArea background + viewport to current theme colors."""
        pal = QApplication.palette()
        # Use Base for light, Window for dark (looks better with your palettes)
        role = QPalette.ColorRole.Base if self._theme_mode() == "light" else QPalette.ColorRole.Window
        col  = pal.color(role)

        # 1) Tell QMdiArea to use a flat color background
        try:
            self.mdi.setBackground(QBrush(col))
        except Exception:
            pass

        # 2) Also set the viewport palette (some styles ignore setBackground)
        try:
            vp = self.mdi.viewport()
            vp.setAutoFillBackground(True)
            p = vp.palette()
            p.setColor(QPalette.ColorRole.Window, col)
            vp.setPalette(p)
            vp.update()
        except Exception:
            pass

        # 3) Ensure the overlay canvas stays transparent and refreshes
        try:
            if hasattr(self, "shortcuts") and self.shortcuts and getattr(self.shortcuts, "canvas", None):
                c = self.shortcuts.canvas
                c.setStyleSheet("background: transparent;")
                c.update()

        except Exception:
            pass

    def _on_document_added(self, doc):
        # Helpful debug:
        try:
            is_table = (getattr(doc, "metadata", {}).get("doc_type") == "table") or \
                    (hasattr(doc, "rows") and hasattr(doc, "headers"))
            self._log(f"[documentAdded] {type(doc).__name__}  table={is_table}  name={getattr(doc, 'display_name', lambda:'?')()}")
        except Exception:
            pass

        self._spawn_subwindow_for(doc)


    def apply_theme_from_settings(self):
        mode = self._theme_mode()
        app = QApplication.instance()
        color_scheme = app.styleHints().colorScheme()

        if mode == "system":
            if color_scheme == Qt.ColorScheme.Dark:
                print("System is in Dark Mode")
                mode = "dark"
            else:
                print("System is in Light Mode")
                mode = "light"
        if mode == "dark":
            app.setStyle("Fusion")
            app.setPalette(self._dark_palette())
            app.setStyleSheet("QToolTip { color: #ffffff; background-color: #2a2a2a; border: 1px solid #5a5a5a; }")
        elif mode == "light":
            app.setStyle("Fusion")
            app.setPalette(self._light_palette())
            # Tooltips readable in light mode
            app.setStyleSheet(
                "QToolTip { color: #141414; background-color: #ffffee; border: 1px solid #c8c8c8; }"
            )
        else:  # system
            app.setStyle(None)
            app.setPalette(QApplication.style().standardPalette())
            app.setStyleSheet("")

        # Nudge widgets to pick up role changes
        self._repolish_top_levels()
        self._apply_workspace_theme()
        self._style_mdi_titlebars()
        self._menu_view_panels = None
        self._populate_view_panels_menu()

        try:
            vp = self.mdi.viewport()
            vp.setAutoFillBackground(True)
            vp.setPalette(QApplication.palette())
            vp.update()
            #self.mdi.setStyleSheet(f"QMdiArea {{ background: {col.name()}; }}")
        except Exception:
            pass

    def _repolish_top_levels(self):
        app = QApplication.instance()
        for w in app.topLevelWidgets():
            w.setUpdatesEnabled(False)
            w.style().unpolish(w)
            w.style().polish(w)
            w.setUpdatesEnabled(True)

    def _style_mdi_titlebars(self):
        if self._theme_mode() == "dark":
            base = "#1b1b1b"     # inactive titlebar
            active = "#242424"   # active titlebar
            fg = "#dcdcdc"       # text color
            self.mdi.setStyleSheet(f"""
                QMdiSubWindow::titlebar        {{ background: {base};  color: {fg}; }}
                QMdiSubWindow::titlebar:active {{ background: {active}; color: {fg}; }}
            """)
        else:
            # No override in light mode
            self.mdi.setStyleSheet("")

    def _dark_palette(self) -> QPalette:
        p = QPalette()

        # Bases
        bg      = QColor(18, 18, 18)   # editor / view backgrounds (Base)
        panel   = QColor(27, 27, 27)   # window / panels (Window, Button)
        altbase = QColor(33, 33, 33)
        text    = QColor(220, 220, 220)
        dis     = QColor(140, 140, 140)
        hi      = QColor(30, 144, 255)  # highlight (dodger blue)

        p.setColor(QPalette.ColorRole.Window,        panel)
        p.setColor(QPalette.ColorRole.WindowText,    text)
        p.setColor(QPalette.ColorRole.Base,          bg)
        p.setColor(QPalette.ColorRole.AlternateBase, altbase)
        p.setColor(QPalette.ColorRole.ToolTipBase,   panel)
        p.setColor(QPalette.ColorRole.ToolTipText,   text)
        p.setColor(QPalette.ColorRole.Text,          text)
        p.setColor(QPalette.ColorRole.Button,        panel)
        p.setColor(QPalette.ColorRole.ButtonText,    text)
        p.setColor(QPalette.ColorRole.BrightText,    QColor(255, 0, 0))
        p.setColor(QPalette.ColorRole.Highlight,     hi)
        p.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))   # ← readable on blue
        p.setColor(QPalette.ColorRole.Link,          QColor(90, 160, 255))
        p.setColor(QPalette.ColorRole.LinkVisited,   QColor(160, 140, 255))
        # Qt6: explicit placeholder color helps avoid faint-on-faint
        try:
            p.setColor(QPalette.ColorRole.PlaceholderText, QColor(160, 160, 160))
        except Exception:
            pass

        # Disabled
        p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text,       dis)
        p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, dis)
        p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, dis)
        p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Base,       QColor(24, 24, 24))
        p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Highlight,  QColor(60, 60, 60))
        p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.HighlightedText, QColor(210, 210, 210))

        return p

    def _light_palette(self) -> QPalette:
        p = QPalette()

        # Light neutrals
        window  = QColor(246, 246, 246)  # panels/docks
        base    = QColor(255, 255, 255)  # text fields, editors
        altbase = QColor(242, 242, 242)  # alternating rows
        text    = QColor(20, 20, 20)     # primary text
        btn     = QColor(246, 246, 246)  # buttons same as window
        dis     = QColor(140, 140, 140)  # disabled text
        link    = QColor(25, 100, 210)   # link blue
        linkv   = QColor(120, 70, 200)   # visited
        hi      = QColor(43, 120, 228)   # selection blue (Windows-like)
        hitxt   = QColor(255, 255, 255)  # text over selection

        # Core roles
        p.setColor(QPalette.ColorRole.Window,          window)
        p.setColor(QPalette.ColorRole.WindowText,      text)
        p.setColor(QPalette.ColorRole.Base,            base)
        p.setColor(QPalette.ColorRole.AlternateBase,   altbase)
        p.setColor(QPalette.ColorRole.ToolTipBase,     QColor(255, 255, 238))  # soft yellow tooltip
        p.setColor(QPalette.ColorRole.ToolTipText,     text)
        p.setColor(QPalette.ColorRole.Text,            text)
        p.setColor(QPalette.ColorRole.Button,          btn)
        p.setColor(QPalette.ColorRole.ButtonText,      text)
        p.setColor(QPalette.ColorRole.BrightText,      QColor(180, 0, 0))
        p.setColor(QPalette.ColorRole.Highlight,       hi)
        p.setColor(QPalette.ColorRole.HighlightedText, hitxt)
        p.setColor(QPalette.ColorRole.Link,            link)
        p.setColor(QPalette.ColorRole.LinkVisited,     linkv)

        # Helps line edits/placeholders avoid too-faint gray
        try:
            p.setColor(QPalette.ColorRole.PlaceholderText, QColor(110, 110, 110))
        except Exception:
            pass

        # Disabled group (keep contrasts sane)
        p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text,            dis)
        p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText,      dis)
        p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText,      dis)
        p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Highlight,       QColor(200, 200, 200))
        p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.HighlightedText, QColor(120, 120, 120))
        p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Base,            QColor(248, 248, 248))

        return p

    def _apply_theme_safely(self):
        if self._theme_guard:
            return
        self._theme_guard = True
        try:
            self.apply_theme_from_settings()
        finally:
            QTimer.singleShot(0, lambda: setattr(self, "_theme_guard", False))

    # Follow OS theme only when "system" is selected
    def _theme_mode(self) -> str:
        return (self.settings.value("ui/theme", "system", type=str) or "system").lower()

    def eventFilter(self, obj, ev):
        if obj is QApplication.instance() and ev.type() in getattr(self, "_theme_events", ()):
            if self._theme_mode() == "system" and not self._theme_guard:
                # debounce to collapse bursts
                self._theme_debounce.start(100)
            return False
        return super().eventFilter(obj, ev)


    # --- UI scaffolding ---
    def _init_explorer_dock(self):
        self.explorer = QListWidget()
        self.explorer.itemActivated.connect(self._activate_from_explorer)

        dock = QDockWidget("Explorer", self)
        dock.setWidget(self.explorer)
        dock.setObjectName("ExplorerDock")
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)

    def _init_console_dock(self):
        self.console = QListWidget()
        dock = QDockWidget("Console / Status", self)
        dock.setWidget(self.console)
        dock.setObjectName("ConsoleDock")
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock)

    def _init_status_log_dock(self):
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

        # Ensure a global log bus and wire bus → dock (queued; thread-safe)
        if not hasattr(app, "_sasd_log_bus"):
            app._sasd_log_bus = LogBus()
        app._sasd_log_bus.posted.connect(
            self.status_log_dock.append_line,
            type=Qt.ConnectionType.QueuedConnection
        )

        # First-run placement (only if no prior saved layout)
        self._first_place_status_log_if_needed()

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

            # Remember we’ve “introduced” it once
            s.setValue(flag_key, True)
            s.sync()

            # Optional: immediately persist the whole layout so next launch restores it
            try:
                s.setValue("mainwindow/state", self.saveState())
            except Exception:
                pass

        QTimer.singleShot(0, _place)


    def _init_toolbar(self):
        # View toolbar (Undo / Redo / Display-Stretch)
        tb = DraggableToolBar("View", self)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, tb)

        tb.addAction(self.act_open)
        tb.addAction(self.act_save)
        tb.addSeparator()
        tb.addAction(self.act_undo)
        tb.addAction(self.act_redo)
        tb.addSeparator()

        # Put Display-Stretch on the bar first so we can attach a menu to its button
        tb.addAction(self.act_autostretch)
        tb.addAction(self.act_zoom_1_1)
        tb.addAction(self.act_zoom_fit)

        # Style
        btn = tb.widgetForAction(self.act_autostretch)
        if isinstance(btn, QToolButton):
            # split-button menu with Linked + Hard profile
            menu = QMenu(btn)
            menu.addAction(self.act_stretch_linked)
            menu.addAction(self.act_hardstretch)   # keep your hard profile here too
            btn.setMenu(menu)
            btn.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)

            btn.setStyleSheet("""
                QToolButton { color: #dcdcdc; }
                QToolButton:checked { color: #DAA520; font-weight: 600; }
            """)


        # Functions toolbar
        tb_fn = DraggableToolBar("Functions", self)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, tb_fn)
        tb_fn.addAction(self.act_crop) 
        tb_fn.addAction(self.act_histogram) 
        tb_fn.addAction(self.act_pedestal)
        tb_fn.addAction(self.act_linear_fit)
        tb_fn.addAction(self.act_stat_stretch)
        tb_fn.addAction(self.act_star_stretch)
        tb_fn.addAction(self.act_curves)
        tb_fn.addAction(self.act_ghs)
        tb_fn.addAction(self.act_abe)
        tb_fn.addAction(self.act_graxpert)
        tb_fn.addAction(self.act_remove_stars)
        tb_fn.addAction(self.act_add_stars)
        tb_fn.addAction(self.act_background_neutral)
        tb_fn.addAction(self.act_white_balance)
        tb_fn.addAction(self.act_sfcc)
        tb_fn.addAction(self.act_remove_green)
        tb_fn.addAction(self.act_convo)
        tb_fn.addAction(self.act_extract_luma)
        tb_fn.addAction(self.act_recombine_luma)
        tb_fn.addAction(self.act_rgb_extract)
        tb_fn.addAction(self.act_rgb_combine)
        tb_fn.addAction(self.act_blemish)
        tb_fn.addAction(self.act_wavescale_hdr)
        tb_fn.addAction(self.act_wavescale_de)
        tb_fn.addAction(self.act_clahe)
        tb_fn.addAction(self.act_morphology)
        tb_fn.addAction(self.act_pixelmath)
        tb_fn.addAction(self.act_signature) 
        tb_fn.addAction(self.act_halobgon)

        tbCosmic = DraggableToolBar("Cosmic Clarity", self)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, tbCosmic)
        tbCosmic.addAction(self.actAberrationAI)        
        tbCosmic.addAction(self.actCosmicUI)
        tbCosmic.addAction(self.actCosmicSat)


        tb_tl = DraggableToolBar("Tools", self)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, tb_tl)
        tb_tl.addAction(self.act_blink) # Tools start here; Blink shows with QIcon(blink_path)
        tb_tl.addAction(self.act_ppp)   # Perfect Palette Picker
        tb_tl.addAction(self.act_nbtorgb)
        tb_tl.addAction(self.act_freqsep)
        tb_tl.addAction(self.act_contsub)
        tb_tl.addAction(self.act_image_combine)

        tb_geom = DraggableToolBar("Geometry", self)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, tb_geom)
        tb_geom.addAction(self.act_geom_invert)
        tb_geom.addSeparator()
        tb_geom.addAction(self.act_geom_flip_h)
        tb_geom.addAction(self.act_geom_flip_v)
        tb_geom.addSeparator()
        tb_geom.addAction(self.act_geom_rot_cw)
        tb_geom.addAction(self.act_geom_rot_ccw)
        tb_geom.addSeparator()
        tb_geom.addAction(self.act_geom_rescale)
        tb_geom.addSeparator()
        tb_geom.addAction(self.act_debayer)

        tb_star = DraggableToolBar("Star Stuff", self)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, tb_star)
        tb_star.addAction(self.act_image_peeker)
        tb_star.addAction(self.act_psf_viewer)
        tb_star.addAction(self.act_stacking_suite)
        tb_star.addAction(self.act_live_stacking)
        tb_star.addAction(self.act_plate_solve)
        tb_star.addAction(self.act_star_align)
        tb_star.addAction(self.act_star_register)
        tb_star.addAction(self.act_mosaic_master)
        tb_star.addAction(self.act_supernova_hunter)
        tb_star.addAction(self.act_star_spikes)
        tb_star.addAction(self.act_exo_detector)
        tb_star.addAction(self.act_isophote)  

        tb_msk = DraggableToolBar("Masks", self)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, tb_msk)
        tb_msk.addAction(self.act_create_mask)
        tb_msk.addAction(self.act_apply_mask)
        tb_msk.addAction(self.act_remove_mask)

        tb_wim = DraggableToolBar("What's In My...", self)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, tb_wim)   
        tb_wim.addAction(self.act_whats_in_my_sky)     
        tb_wim.addAction(self.act_wimi)

        tb_bundle = DraggableToolBar("Bundles", self)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, tb_bundle)
        tb_bundle.addAction(self.act_view_bundles)
        tb_bundle.addAction(self.act_function_bundles)

    def _create_actions(self):
        # File actions
        self.act_open = QAction(QIcon(openfile_path), "Open…", self)
        self.act_open.setIconVisibleInMenu(True)
        self.act_open.setShortcut(QKeySequence.StandardKey.Open)
        self.act_open.setStatusTip("Open image(s)")
        self.act_open.triggered.connect(self.open_files)


        self.act_project_new  = QAction("New Project", self)
        self.act_project_save = QAction("Save Project…", self)
        self.act_project_load = QAction("Load Project…", self)

        self.act_project_new.setStatusTip("Close all views and clear shortcuts")
        self.act_project_save.setStatusTip("Save all views, histories, and shortcuts to a .sas file")
        self.act_project_load.setStatusTip("Load a .sas project (views, histories, shortcuts)")

        self.act_project_new.triggered.connect(self._new_project)
        self.act_project_save.triggered.connect(self._save_project)
        self.act_project_load.triggered.connect(self._load_project)

        self.act_clear_views = QAction("Clear All Views", self)
        self.act_clear_views.setStatusTip("Close all views and documents, keep desktop shortcuts")
        # optional shortcut (pick anything you like or omit)
        # self.act_clear_views.setShortcut(QKeySequence("Ctrl+Shift+W"))
        self.act_clear_views.triggered.connect(self._clear_views_keep_shortcuts)

        self.act_save = QAction(QIcon(disk_path), "Save As…", self)
        self.act_save.setIconVisibleInMenu(True)
        self.act_save.setShortcut(QKeySequence.StandardKey.SaveAs)
        self.act_save.setStatusTip("Save the active image")
        self.act_save.triggered.connect(self.save_active)

        self.act_cascade = QAction("Cascade Views", self)
        self.act_cascade.setStatusTip("Cascade all subwindows")
        self.act_cascade.setShortcut(QKeySequence("Ctrl+Shift+C"))
        self.act_cascade.triggered.connect(self._cascade_views)

        self.act_tile = QAction("Tile Views", self)
        self.act_tile.setStatusTip("Tile all subwindows")
        self.act_tile.setShortcut(QKeySequence("Ctrl+Shift+T"))
        self.act_tile.triggered.connect(self._tile_views)

        self.act_tile_vert = QAction("Tile Vertically", self)
        self.act_tile_vert.setStatusTip("Split the workspace into equal vertical columns")
        self.act_tile_vert.triggered.connect(lambda: self._tile_views_direction("v"))

        self.act_tile_horiz = QAction("Tile Horizontally", self)
        self.act_tile_horiz.setStatusTip("Split the workspace into equal horizontal rows")
        self.act_tile_horiz.triggered.connect(lambda: self._tile_views_direction("h"))

        self.act_tile_grid = QAction("Smart Grid", self)
        self.act_tile_grid.setStatusTip("Arrange subwindows in a near-square grid")
        self.act_tile_grid.triggered.connect(self._tile_views_grid)

        self.act_undo = QAction(QIcon(undoicon_path), "Undo", self)
        self.act_redo = QAction(QIcon(redoicon_path), "Redo", self)
        self.act_undo.setShortcut(QKeySequence.StandardKey.Undo)               # Ctrl+Z
        self.act_redo.setShortcuts([QKeySequence.StandardKey.Redo, "Ctrl+Y"])  # Shift+Ctrl+Z / Ctrl+Y
        self.act_undo.setIconVisibleInMenu(True)
        self.act_redo.setIconVisibleInMenu(True)
        self.act_undo.triggered.connect(self._undo_active)
        self.act_redo.triggered.connect(self._redo_active)

        # View-ish action (toolbar toggle)
        self.act_autostretch = QAction("Display-Stretch", self, checkable=True)
        self.act_autostretch.setStatusTip("Toggle display auto-stretch for the active window")
        self.act_autostretch.toggled.connect(self._toggle_autostretch)

        self.act_hardstretch = QAction("Hard-Display-Stretch", self, checkable=True)
        self.addAction(self.act_hardstretch)
        self.act_hardstretch.setShortcut(QKeySequence("H"))
        self.act_hardstretch.setStatusTip("Toggle hard profile for Display-Stretch (H)")

        # use toggled(bool), not triggered()
        self.act_hardstretch.toggled.connect(self._set_hard_autostretch_from_action)

        # NEW: Linked/Unlinked toggle (global default via QSettings, per-view runtime)
        self.act_stretch_linked = QAction("Link RGB channels", self, checkable=True)
        self.act_stretch_linked.setStatusTip("Apply the same stretch to all RGB channels")
        self.act_stretch_linked.setShortcut(QKeySequence("Ctrl+Shift+L"))
        self.act_stretch_linked.setChecked(
            self.settings.value("display/stretch_linked", False, type=bool)
        )
        self.act_stretch_linked.toggled.connect(self._set_linked_stretch_from_action)

        self.act_zoom_1_1 = QAction("1:1", self)
        self.act_zoom_1_1.setStatusTip("Zoom to 100% (pixel-for-pixel)")
        self.act_zoom_1_1.setShortcut(QKeySequence("Ctrl+1"))    
        self.act_zoom_1_1.triggered.connect(self._zoom_active_1_1)

        self.act_zoom_fit = QAction("Fit", self)
        self.act_zoom_fit.setStatusTip("Fit image to current window")
        self.act_zoom_fit.setShortcut(QKeySequence("Ctrl+0"))
        self.act_zoom_fit.triggered.connect(self._zoom_active_fit)

        # View state copy/paste (optional quick commands)
        self._copied_view_state = None
        self.act_copy_view = QAction("Copy View (zoom/pan)", self)
        self.act_paste_view = QAction("Paste View", self)
        self.act_copy_view.setShortcut("Ctrl+Shift+C")
        self.act_paste_view.setShortcut("Ctrl+Shift+V")
        self.act_copy_view.triggered.connect(self._copy_active_view)
        self.act_paste_view.triggered.connect(self._paste_active_view)

        # Functions
        self.act_crop = QAction(QIcon(cropicon_path), "Crop…", self)
        self.act_crop.setStatusTip("Crop / rotate with handles")
        self.act_crop.setIconVisibleInMenu(True)
        self.act_crop.triggered.connect(self._open_crop_dialog)

        self.act_histogram = QAction(QIcon(histogram_path), "Histogram…", self)
        self.act_histogram.setStatusTip("View histogram and basic stats for the active image")
        self.act_histogram.setIconVisibleInMenu(True)
        self.act_histogram.triggered.connect(self._open_histogram)

        self.act_stat_stretch = QAction(QIcon(statstretch_path), "Statistical Stretch…", self)
        self.act_stat_stretch.setStatusTip("Stretch the image using median/SD statistics")
        self.act_stat_stretch.setIconVisibleInMenu(True)
        self.act_stat_stretch.triggered.connect(self._open_statistical_stretch)

        self.act_star_stretch = QAction(QIcon(starstretch_path), "Star Stretch…", self)
        self.act_star_stretch.setStatusTip("Arcsinh star stretch with optional SCNR and color boost")
        self.act_star_stretch.setIconVisibleInMenu(True)
        self.act_star_stretch.triggered.connect(self._open_star_stretch)

        self.act_curves = QAction(QIcon(curves_path), "Curves Editor…", self)  # add an icon later if you want
        self.act_curves.setStatusTip("Open the Curves Editor for the active image")
        self.act_curves.setIconVisibleInMenu(True)
        self.act_curves.triggered.connect(self._open_curves_editor)

        self.act_ghs = QAction(QIcon(uhs_path), "Hyperbolic Stretch…", self)
        self.act_ghs.setStatusTip("Generalized hyperbolic stretch (α/β/γ, LP/HP, pivot)")
        self.act_ghs.setIconVisibleInMenu(True)
        self.act_ghs.triggered.connect(self._open_hyperbolic)

        self.act_abe = QAction(QIcon(abeicon_path), "ABE…", self)
        self.act_abe.setStatusTip("Automatic Background Extraction")
        self.act_abe.setIconVisibleInMenu(True)
        self.act_abe.triggered.connect(self._open_abe_tool)

        self.act_graxpert = QAction(QIcon(graxperticon_path), "Remove Gradient (GraXpert)…", self)
        self.act_graxpert.setIconVisibleInMenu(True)
        self.act_graxpert.setStatusTip("Run GraXpert background extraction on the active image")
        self.act_graxpert.triggered.connect(lambda: remove_gradient_with_graxpert(self))

        self.act_remove_stars = QAction(QIcon(starnet_path), "Remove Stars…", self)
        self.act_remove_stars.setIconVisibleInMenu(True)
        self.act_remove_stars.setStatusTip("Run star removal on the active image")
        self.act_remove_stars.triggered.connect(lambda: remove_stars(self))

        self.act_add_stars = QAction(QIcon(staradd_path), "Add Stars…", self)
        self.act_add_stars.setStatusTip("Blend a starless view with a stars-only view")
        self.act_add_stars.setIconVisibleInMenu(True)
        self.act_add_stars.triggered.connect(lambda: add_stars(self))

        self.act_pedestal = QAction(QIcon(pedestal_icon_path), "Remove Pedestal", self)
        self.act_pedestal.setToolTip("Subtract per-channel minimum.\nClick: active view\nAlt+Drag: drop onto a view")
        self.act_pedestal.setShortcut("Ctrl+P")
        self.act_pedestal.triggered.connect(lambda: remove_pedestal(self))

        self.act_linear_fit = QAction(QIcon(linearfit_path),"Linear Fit…", self)
        self.act_linear_fit.setIconVisibleInMenu(True)
        self.act_linear_fit.setStatusTip("Match image levels using Linear Fit")
        # optional shortcut; change if you already use it elsewhere
        self.act_linear_fit.setShortcut("Ctrl+L")
        self.act_linear_fit.triggered.connect(self._open_linear_fit)

        self.act_remove_green = QAction(QIcon(green_path), "Remove Green...", self)
        self.act_remove_green.setToolTip("SCNR-style green channel removal.")
        self.act_remove_green.setIconVisibleInMenu(True)
        self.act_remove_green.triggered.connect(lambda: open_remove_green_dialog(self, preset=None))

        self.act_background_neutral = QAction(QIcon(neutral_path), "Background Neutralization…", self)
        self.act_background_neutral.setStatusTip("Neutralize background color balance using a sampled region")
        self.act_background_neutral.setIconVisibleInMenu(True)
        self.act_background_neutral.triggered.connect(self._open_background_neutral)

        self.act_white_balance = QAction(QIcon(whitebalance_path), "White Balance…", self)
        self.act_white_balance.setStatusTip("Apply white balance (Star-Based, Manual, or Auto)")
        self.act_white_balance.triggered.connect(self._open_white_balance)

        self.act_sfcc = QAction(QIcon(spcc_icon_path), "Spectral Flux Color Calibration…", self)
        self.act_sfcc.setObjectName("sfcc")
        self.act_sfcc.setToolTip("Open SFCC (Pickles + Filters + Sensor QE)")
        self.act_sfcc.triggered.connect(self.SFCC_show)

        self.act_convo = QAction(QIcon(convoicon_path), "Convolution / Deconvolution…", self)
        self.act_convo.setObjectName("convo_deconvo")
        self.act_convo.setToolTip("Open Convolution / Deconvolution")
        self.act_convo.triggered.connect(self.show_convo_deconvo)

        self.act_extract_luma = QAction(QIcon(LExtract_path), "Extract Luminance", self)
        self.act_extract_luma.setStatusTip("Create a new mono document from Rec.709 luminance of the active image")
        self.act_extract_luma.setIconVisibleInMenu(True)
        self.act_extract_luma.triggered.connect(lambda: self._extract_luminance(doc=None))

        self.act_recombine_luma = QAction(QIcon(LInsert_path), "Recombine Luminance…", self)
        self.act_recombine_luma.setStatusTip("Replace the active image's luminance from another view")
        self.act_recombine_luma.setIconVisibleInMenu(True)
        self.act_recombine_luma.triggered.connect(lambda: self._recombine_luminance_ui(target_doc=None))

        self.act_rgb_extract = QAction(QIcon(rgbextract_path), "RGB Extract", self)
        self.act_rgb_extract.setIconVisibleInMenu(True)
        self.act_rgb_extract.setStatusTip("Extract R/G/B as three mono documents")
        self.act_rgb_extract.triggered.connect(self._rgb_extract_active)

        self.act_rgb_combine = QAction(QIcon(rgbcombo_path), "RGB Combination…", self)
        self.act_rgb_combine.setIconVisibleInMenu(True)
        self.act_rgb_combine.setStatusTip("Combine three mono images into RGB")
        self.act_rgb_combine.triggered.connect(self._open_rgb_combination)

        self.act_blemish = QAction(QIcon(blastericon_path), "Blemish Blaster…", self)
        self.act_blemish.setIconVisibleInMenu(True)
        self.act_blemish.setStatusTip("Interactive blemish removal on the active view")
        self.act_blemish.triggered.connect(self._open_blemish_blaster)

        self.act_wavescale_hdr = QAction(QIcon(hdr_path), "WaveScale HDR…", self)
        self.act_wavescale_hdr.setStatusTip("Wave-scale HDR with luminance-masked starlet")
        self.act_wavescale_hdr.setIconVisibleInMenu(True)
        self.act_wavescale_hdr.triggered.connect(self._open_wavescale_hdr)

        self.act_wavescale_de = QAction(QIcon(dse_icon_path), "WaveScale Dark Enhancer…", self)
        self.act_wavescale_de.setStatusTip("Enhance faint/dark structures with wavelet-guided masking")
        self.act_wavescale_de.setIconVisibleInMenu(True)
        self.act_wavescale_de.triggered.connect(self._open_wavescale_dark_enhance)

        self.act_clahe = QAction(QIcon(clahe_path), "CLAHE…", self)
        self.act_clahe.setStatusTip("Contrast Limited Adaptive Histogram Equalization")
        self.act_clahe.setIconVisibleInMenu(True)
        self.act_clahe.triggered.connect(self._open_clahe)

        self.act_morphology = QAction(QIcon(morpho_path), "Morphological Operations…", self)
        self.act_morphology.setStatusTip("Erosion, dilation, opening, and closing.")
        self.act_morphology.setIconVisibleInMenu(True)
        self.act_morphology.triggered.connect(self._open_morphology)

        self.act_pixelmath = QAction(QIcon(pixelmath_path), "Pixel Math…", self)
        self.act_pixelmath.setStatusTip("Evaluate expressions using open view names")
        self.act_pixelmath.setIconVisibleInMenu(True)
        self.act_pixelmath.triggered.connect(self._open_pixel_math)

        self.act_signature = QAction(QIcon(signature_icon_path), "Signature / Insert…", self)
        self.act_signature.setIconVisibleInMenu(True)
        self.act_signature.setStatusTip("Add signatures/overlays and bake them into the active image")
        self.act_signature.triggered.connect(self._open_signature_insert)

        self.act_halobgon = QAction(QIcon(halo_path), "Halo-B-Gon...", self)
        self.act_halobgon.setIconVisibleInMenu(True)
        self.act_halobgon.setStatusTip("Remove those pesky halos around your stars")
        self.act_halobgon.triggered.connect(self._open_halo_b_gon)

        self.act_image_combine = QAction(QIcon(imagecombine_path), "Image Combine…", self)
        self.act_image_combine.setIconVisibleInMenu(True)
        self.act_image_combine.setStatusTip("Blend two open images (replace A or create new)")
        self.act_image_combine.triggered.connect(self._open_image_combine)

        # --- Geometry ---
        self.act_geom_invert = QAction(QIcon(invert_path), "Invert", self)
        self.act_geom_invert.setIconVisibleInMenu(True)
        self.act_geom_invert.setStatusTip("Invert image colors")
        self.act_geom_invert.triggered.connect(self._exec_geom_invert)

        self.act_geom_flip_h = QAction(QIcon(fliphorizontal_path), "Flip Horizontal", self)
        self.act_geom_flip_h.setIconVisibleInMenu(True)
        self.act_geom_flip_h.setStatusTip("Flip image left↔right")
        self.act_geom_flip_h.triggered.connect(self._exec_geom_flip_h)

        self.act_geom_flip_v = QAction(QIcon(flipvertical_path), "Flip Vertical", self)
        self.act_geom_flip_v.setIconVisibleInMenu(True)
        self.act_geom_flip_v.setStatusTip("Flip image top↕bottom")
        self.act_geom_flip_v.triggered.connect(self._exec_geom_flip_v)

        self.act_geom_rot_cw = QAction(QIcon(rotateclockwise_path), "Rotate 90° Clockwise", self)
        self.act_geom_rot_cw.setIconVisibleInMenu(True)
        self.act_geom_rot_cw.setStatusTip("Rotate image 90° clockwise")
        self.act_geom_rot_cw.triggered.connect(self._exec_geom_rot_cw)

        self.act_geom_rot_ccw = QAction(QIcon(rotatecounterclockwise_path), "Rotate 90° Counterclockwise", self)
        self.act_geom_rot_ccw.setIconVisibleInMenu(True)
        self.act_geom_rot_ccw.setStatusTip("Rotate image 90° counterclockwise")
        self.act_geom_rot_ccw.triggered.connect(self._exec_geom_rot_ccw)

        self.act_geom_rescale = QAction(QIcon(rescale_path), "Rescale…", self)
        self.act_geom_rescale.setIconVisibleInMenu(True)
        self.act_geom_rescale.setStatusTip("Rescale image by a factor")
        self.act_geom_rescale.triggered.connect(self._exec_geom_rescale)

        self.act_debayer = QAction(QIcon(debayer_path), "Debayer…", self)
        self.act_debayer.setObjectName("debayer")
        self.act_debayer.setProperty("command_id", "debayer")
        self.act_debayer.setStatusTip("Demosaic a Bayer-mosaic mono image to RGB")
        self.act_debayer.triggered.connect(self._open_debayer)

        # (Optional example shortcuts; uncomment if you want)
        self.act_geom_invert.setShortcut("Ctrl+I")
        
        # self.act_geom_flip_h.setShortcut("H")
        # self.act_geom_flip_v.setShortcut("V")
        # self.act_geom_rot_cw.setShortcut("]")
        # self.act_geom_rot_ccw.setShortcut("[")
        # self.act_geom_rescale.setShortcut("Ctrl+R")


        # actions (use your actual icon paths if you have them)
        try:
            cosmic_icon = QIcon(cosmic_path)  # define cosmic_path like your other icons (same pattern as halo_path)
        except Exception:
            cosmic_icon = QIcon()

        try:
            sat_icon = QIcon(satellite_path)  # optional icon for satellite
        except Exception:
            sat_icon = QIcon()

        self.actCosmicUI  = QAction(cosmic_icon, "Cosmic Clarity UI…", self)
        self.actCosmicSat = QAction(sat_icon,    "Cosmic Clarity Satellite…", self)

        self.actCosmicUI.triggered.connect(self._open_cosmic_clarity_ui)
        self.actCosmicSat.triggered.connect(self._open_cosmic_clarity_satellite)


        ab_icon = QIcon(aberration_path)  # falls back if file missing

        self.actAberrationAI = QAction(ab_icon, "Aberration Correction (AI)…", self)
        self.actAberrationAI.triggered.connect(self._open_aberration_ai)



        #Tools
        self.act_blink = QAction(QIcon(blink_path), "Blink Comparator…", self)
        self.act_blink.setStatusTip("Compare a stack of images by blinking")
        self.act_blink.triggered.connect(self._open_blink_tool)        

        self.act_ppp = QAction(QIcon(ppp_path), "Perfect Palette Picker…", self)
        self.act_ppp.setStatusTip("Pick the perfect palette for your image")
        self.act_ppp.triggered.connect(self._open_ppp_tool) 

        self.act_nbtorgb = QAction(QIcon(nbtorgb_path), "NB→RGB Stars…", self)
        self.act_nbtorgb.setStatusTip("Combine narrowband to RGB with optional OSC stars")
        self.act_nbtorgb.setIconVisibleInMenu(True)
        self.act_nbtorgb.triggered.connect(self._open_nbtorgb_tool)

        # NEW: Frequency Separation
        self.act_freqsep = QAction(QIcon(freqsep_path), "Frequency Separation…", self)
        self.act_freqsep.setStatusTip("Split into LF/HF and enhance HF (scale, wavelet, denoise)")
        self.act_freqsep.setIconVisibleInMenu(True)
        self.act_freqsep.triggered.connect(self._open_freqsep_tool)

        self.act_contsub = QAction(QIcon(contsub_path), "Continuum Subtract…", self)
        self.act_contsub.setStatusTip("Continuum subtract (NB – scaled broadband)")
        self.act_contsub.setIconVisibleInMenu(True)
        self.act_contsub.triggered.connect(self._open_contsub_tool)

        # History
        self.act_history_explorer = QAction("History Explorer…", self)
        self.act_history_explorer.setStatusTip("Inspect and restore from the slot’s history")
        self.act_history_explorer.triggered.connect(self._open_history_explorer)


        #STAR STUFF
        self.act_image_peeker = QAction(QIcon(peeker_icon), "Image Peeker…", self)
        self.act_image_peeker.setIconVisibleInMenu(True)
        self.act_image_peeker.setStatusTip("Image Inspector and Focal Plane Analysis")
        self.act_image_peeker.triggered.connect(self._open_image_peeker)

        self.act_psf_viewer = QAction(QIcon(psf_path), "PSF Viewer...", self)
        self.act_psf_viewer.setIconVisibleInMenu(True)
        self.act_psf_viewer.setStatusTip("Inspect star PSF/HFR and flux histograms (SEP)")
        self.act_psf_viewer.triggered.connect(self._open_psf_viewer)        

        self.act_stacking_suite = QAction(QIcon(stacking_path), "Stacking Suite...", self)
        self.act_stacking_suite.setIconVisibleInMenu(True)
        self.act_stacking_suite.setStatusTip("Stacking! Darks, Flats, Lights, Calibration, Drizzle, and more!!")
        self.act_stacking_suite.triggered.connect(self._open_stacking_suite)

        self.act_live_stacking = QAction(QIcon(livestacking_path), "Live Stacking...", self)
        self.act_live_stacking.setIconVisibleInMenu(True)
        self.act_live_stacking.setStatusTip("Live monitor and stack incoming frames")
        self.act_live_stacking.triggered.connect(self._open_live_stacking)

        self.act_plate_solve = QAction(QIcon(platesolve_path), "Plate Solver...", self)
        self.act_plate_solve.setIconVisibleInMenu(True)
        self.act_plate_solve.setStatusTip("Solve WCS/SIP for the active image or a file")
        self.act_plate_solve.triggered.connect(self._open_plate_solver)

        self.act_star_align = QAction(QIcon(staralign_path), "Stellar Alignment...", self)
        self.act_star_align.setIconVisibleInMenu(True)
        self.act_star_align.setStatusTip("Align images via astroalign / triangles")
        self.act_star_align.triggered.connect(self._open_stellar_alignment)

        self.act_star_register = QAction(QIcon(starregistration_path), "Stellar Register...", self)
        self.act_star_register.setIconVisibleInMenu(True)
        self.act_star_register.setStatusTip("Batch-align frames to a reference")
        self.act_star_register.triggered.connect(self._open_stellar_registration)

        self.act_mosaic_master = QAction(QIcon(mosaic_path), "Mosaic Master...", self)
        self.act_mosaic_master.setIconVisibleInMenu(True)
        self.act_mosaic_master.setStatusTip("Build mosaics from overlapping frames")
        self.act_mosaic_master.triggered.connect(self._open_mosaic_master)

        self.act_supernova_hunter = QAction(QIcon(supernova_path), "Supernova / Asteroid Hunter...", self)
        self.act_supernova_hunter.setIconVisibleInMenu(True)
        self.act_supernova_hunter.setStatusTip("Find transients/anomalies across frames")
        self.act_supernova_hunter.triggered.connect(self._open_supernova_hunter)

        self.act_star_spikes = QAction(QIcon(starspike_path), "Diffraction Spikes...", self)
        self.act_star_spikes.setIconVisibleInMenu(True)
        self.act_star_spikes.setStatusTip("Add diffraction spikes to detected stars")
        self.act_star_spikes.triggered.connect(self._open_star_spikes)

        self.act_exo_detector = QAction(QIcon(exoicon_path), "Exoplanet Detector...", self)
        self.act_exo_detector.setIconVisibleInMenu(True)
        self.act_exo_detector.setStatusTip("Detect exoplanet transits from time-series subs")
        self.act_exo_detector.triggered.connect(self._open_exo_detector)

        self.act_isophote = QAction(QIcon(isophote_path), "GLIMR — Isophote Modeler…", self)
        self.act_isophote.setIconVisibleInMenu(True)
        self.act_isophote.setStatusTip("Fit galaxy isophotes and reveal residuals")
        self.act_isophote.triggered.connect(self._open_isophote)

        self.act_whats_in_my_sky = QAction(QIcon(wims_path), "What's In My Sky...", self)
        self.act_whats_in_my_sky.setIconVisibleInMenu(True)
        self.act_whats_in_my_sky.setStatusTip("Plan targets by altitude, transit time, and lunar separation")
        self.act_whats_in_my_sky.triggered.connect(self._open_whats_in_my_sky)

        self.act_wimi = QAction(QIcon(wimi_path), "What's In My Image...", self)
        self.act_wimi.setIconVisibleInMenu(True)
        self.act_wimi.setStatusTip("Identify objects in a plate-solved frame")
        self.act_wimi.triggered.connect(self._open_wimi)

        # --- FITS Header Modifier action ---
        self.act_fits_modifier = QAction("FITS Header Modifier…", self)
        # self.act_fits_modifier.setIcon(QIcon(path_to_icon))  # (optional) icon goes here later
        self.act_fits_modifier.setIconVisibleInMenu(True)
        self.act_fits_modifier.setStatusTip("View/Edit FITS headers; save or batch modify")
        self.act_fits_modifier.triggered.connect(self._open_fits_modifier)

        self.act_batch_renamer = QAction("Batch Rename from FITS…", self)
        # self.act_batch_renamer.setIcon(QIcon(batch_renamer_icon_path))  # (optional icon)
        self.act_batch_renamer.triggered.connect(self._open_batch_renamer)

        self.act_astrobin_exporter = QAction("AstroBin Exporter…", self)
        # self.act_astrobin_exporter.setIcon(QIcon(astrobin_icon_path))  # optional icon
        self.act_astrobin_exporter.triggered.connect(self._open_astrobin_exporter)

        self.act_batch_convert = QAction("Batch Converter…", self)
        # self.act_batch_convert.setIcon(QIcon("path/to/icon.svg"))  # optional later
        self.act_batch_convert.triggered.connect(self._open_batch_convert)

        self.act_copy_astrometry = QAction("Copy Astrometric Solution…", self)
        self.act_copy_astrometry.triggered.connect(self._open_copy_astrometry)

        # Create Mask
        self.act_create_mask = QAction(QIcon(maskcreate_path), "Create Mask…", self)
        self.act_create_mask.setIconVisibleInMenu(True)
        self.act_create_mask.setStatusTip("Create a mask from the active image")
        self.act_create_mask.triggered.connect(self._action_create_mask)

        # --- Masks ---
        self.act_apply_mask = QAction(QIcon(maskapply_path), "Apply Mask", self)
        self.act_apply_mask.setStatusTip("Apply a mask document to the active image")
        self.act_apply_mask.triggered.connect(self._apply_mask_menu)

        self.act_remove_mask = QAction(QIcon(maskremove_path), "Remove Active Mask", self)
        self.act_remove_mask.setStatusTip("Remove the active mask from the active image")
        self.act_remove_mask.triggered.connect(self._remove_mask_menu)

        self.act_show_mask = QAction("Show Mask Overlay", self)
        self.act_hide_mask = QAction("Hide Mask Overlay", self)
        self.act_show_mask.triggered.connect(self._show_mask_overlay)
        self.act_hide_mask.triggered.connect(self._hide_mask_overlay)

        self.act_invert_mask = QAction("Invert Mask", self)
        self.act_invert_mask.triggered.connect(self._invert_mask)
        self.act_invert_mask.setShortcut("Ctrl+Shift+I")

        self.act_check_updates = QAction("Check for Updates…", self)
        self.act_check_updates.triggered.connect(self.check_for_updates_now)

        self.act_view_bundles = QAction(QIcon(viewbundles_path), "View Bundles…", self)
        self.act_view_bundles.setStatusTip("Create bundles of views; drop shortcuts to apply to all")
        self.act_view_bundles.triggered.connect(self._open_view_bundles)

        self.act_function_bundles = QAction(QIcon(functionbundles_path), "Function Bundles…", self)
        self.act_function_bundles.setStatusTip("Create and run bundles of functions/shortcuts")
        self.act_function_bundles.triggered.connect(self._open_function_bundles)

        # give each action a stable id and register
        def reg(cid, act):
            act.setProperty("command_id", cid)
            act.setObjectName(cid)  # also becomes default if we ever need it
            self.shortcuts.register_action(cid, act)

        # create manager once MDI exists
        if not hasattr(self, "shortcuts"):
            # self.mdi is your QMdiArea used elsewhere (stat_stretch uses it)
            self.shortcuts = ShortcutManager(self.mdi, self)

        # register whatever you want draggable/launchable
        reg("open",           self.act_open)
        reg("save_as",        self.act_save)
        reg("undo",           self.act_undo)
        reg("redo",           self.act_redo)
        reg("autostretch",    self.act_autostretch)
        reg("zoom_1_1",       self.act_zoom_1_1)
        reg("crop",           self.act_crop)
        reg("histogram",      self.act_histogram)
        reg("stat_stretch",   self.act_stat_stretch)
        reg("star_stretch",   self.act_star_stretch)
        reg("curves",         self.act_curves)
        reg("ghs",            self.act_ghs)
        reg("blink",          self.act_blink)
        reg("ppp",            self.act_ppp)
        reg("nbtorgb",       self.act_nbtorgb)
        reg("freqsep",       self.act_freqsep)
        reg("contsub",      self.act_contsub)
        reg("abe",          self.act_abe)
        reg("create_mask", self.act_create_mask)
        reg("graxpert", self.act_graxpert)
        reg("remove_stars", self.act_remove_stars)
        reg("add_stars", self.act_add_stars)
        reg("pedestal",       self.act_pedestal)
        reg("remove_green",   self.act_remove_green)
        reg("background_neutral", self.act_background_neutral)
        reg("white_balance", self.act_white_balance)
        reg("sfcc",    self.act_sfcc)
        reg("convo", self.act_convo)
        reg("extract_luminance", self.act_extract_luma)
        reg("recombine_luminance", self.act_recombine_luma)
        reg("rgb_extract", self.act_rgb_extract)
        reg("rgb_combine", self.act_rgb_combine)
        reg("blemish_blaster", self.act_blemish)
        reg("wavescale_hdr", self.act_wavescale_hdr)
        reg("wavescale_dark_enhance", self.act_wavescale_de)
        reg("clahe", self.act_clahe)
        reg("morphology", self.act_morphology)
        reg("pixel_math", self.act_pixelmath)
        reg("signature_insert", self.act_signature) 
        reg("halo_b_gon", self.act_halobgon)
        reg("geom_invert",                 self.act_geom_invert)
        reg("geom_flip_horizontal",        self.act_geom_flip_h)
        reg("geom_flip_vertical",          self.act_geom_flip_v)
        reg("geom_rotate_clockwise",       self.act_geom_rot_cw)
        reg("geom_rotate_counterclockwise",self.act_geom_rot_ccw)
        reg("geom_rescale",                self.act_geom_rescale)        
        reg("project_new",  self.act_project_new)
        reg("project_save", self.act_project_save)
        reg("project_load", self.act_project_load)     
        reg("image_combine", self.act_image_combine)   
        reg("psf_viewer", self.act_psf_viewer)
        reg("plate_solve", self.act_plate_solve)
        reg("star_align", self.act_star_align)
        reg("star_register", self.act_star_register)
        reg("mosaic_master", self.act_mosaic_master)
        reg("image_peeker", self.act_image_peeker)
        reg("live_stacking", self.act_live_stacking)
        reg("stacking_suite", self.act_stacking_suite)
        reg("supernova_hunter", self.act_supernova_hunter)
        reg("star_spikes", self.act_star_spikes)
        reg("exo_detector", self.act_exo_detector)
        reg("isophote", self.act_isophote) 
        reg("whats_in_my_sky", self.act_whats_in_my_sky)
        reg("whats_in_my_image", self.act_wimi)
        reg("linear_fit", self.act_linear_fit)
        reg("debayer", self.act_debayer)
        reg("cosmicclarity", self.actCosmicUI)
        reg("cosmicclaritysat", self.actCosmicSat)
        reg("aberrationai", self.actAberrationAI)
        reg("view_bundles", self.act_view_bundles)
        reg("function_bundles", self.act_function_bundles)

    def _init_menubar(self):
        mb = self.menuBar()

        # File
        m_file = mb.addMenu("&File")
        m_file.addAction(self.act_open)
        m_file.addSeparator()
        m_file.addAction(self.act_save)
        m_file.addSeparator()
        m_file.addAction(self.act_clear_views) 
        m_file.addSeparator()
        m_file.addAction(self.act_project_new)
        m_file.addAction(self.act_project_save)
        m_file.addAction(self.act_project_load)

        # Edit (with icons)
        m_edit = mb.addMenu("&Edit")
        m_edit.addAction(self.act_undo)
        m_edit.addAction(self.act_redo)

        # Functions
        m_fn = mb.addMenu("&Functions")
        m_fn.addAction(self.act_crop) 
        m_fn.addAction(self.act_histogram)   
        m_fn.addAction(self.act_pedestal)
        m_fn.addAction(self.act_linear_fit)
        m_fn.addAction(self.act_stat_stretch)
        m_fn.addAction(self.act_star_stretch) 
        m_fn.addAction(self.act_curves) 
        m_fn.addAction(self.act_ghs)
        m_fn.addAction(self.act_abe)
        m_fn.addAction(self.act_graxpert)
        m_fn.addAction(self.act_remove_stars)
        m_fn.addAction(self.act_add_stars)
        m_fn.addAction(self.act_background_neutral)
        m_fn.addAction(self.act_white_balance)
        m_fn.addAction(self.act_sfcc)
        m_fn.addAction(self.act_remove_green)
        m_fn.addAction(self.act_convo)
        m_fn.addAction(self.act_extract_luma)
        m_fn.addAction(self.act_recombine_luma)
        m_fn.addAction(self.act_rgb_extract)
        m_fn.addAction(self.act_rgb_combine)
        m_fn.addAction(self.act_blemish)
        m_fn.addAction(self.act_wavescale_hdr)
        m_fn.addAction(self.act_wavescale_de)
        m_fn.addAction(self.act_clahe)
        m_fn.addAction(self.act_morphology)
        m_fn.addAction(self.act_pixelmath)
        m_fn.addAction(self.act_signature)
        m_fn.addAction(self.act_halobgon)

        mCosmic = mb.addMenu("&Smart Tools")
        mCosmic.addAction(self.actAberrationAI)
        mCosmic.addAction(self.actCosmicUI)
        mCosmic.addAction(self.actCosmicSat)
        mCosmic.addAction(self.act_graxpert)
        mCosmic.addAction(self.act_remove_stars)

        m_tools = mb.addMenu("&Tools")
        m_tools.addAction(self.act_blink)
        m_tools.addAction(self.act_ppp)
        m_tools.addAction(self.act_nbtorgb)
        m_tools.addAction(self.act_freqsep)
        m_tools.addAction(self.act_contsub)
        m_tools.addAction(self.act_image_combine)
        m_tools.addSeparator()
        m_tools.addAction(self.act_view_bundles) 
        m_tools.addAction(self.act_function_bundles)

        m_geom = mb.addMenu("&Geometry")
        m_geom.addAction(self.act_geom_invert)
        m_geom.addSeparator()
        m_geom.addAction(self.act_geom_flip_h)
        m_geom.addAction(self.act_geom_flip_v)
        m_geom.addSeparator()
        m_geom.addAction(self.act_geom_rot_cw)
        m_geom.addAction(self.act_geom_rot_ccw)
        m_geom.addSeparator()
        m_geom.addAction(self.act_geom_rescale)
        m_geom.addSeparator()
        m_geom.addAction(self.act_debayer)

        m_star = mb.addMenu("&Star Stuff")
        m_star.addAction(self.act_image_peeker)
        m_star.addAction(self.act_psf_viewer)
        m_star.addAction(self.act_stacking_suite)
        m_star.addAction(self.act_live_stacking)
        m_star.addAction(self.act_plate_solve)         
        m_star.addAction(self.act_star_align)
        m_star.addAction(self.act_star_register)
        m_star.addAction(self.act_mosaic_master)
        m_star.addAction(self.act_supernova_hunter)
        m_star.addAction(self.act_star_spikes)
        m_star.addAction(self.act_exo_detector)
        m_star.addAction(self.act_isophote)
        

        m_masks = mb.addMenu("&Masks")
        m_masks.addAction(self.act_create_mask)
        m_masks.addSeparator()
        m_masks.addAction(self.act_apply_mask)
        m_masks.addAction(self.act_remove_mask)
        m_masks.addSeparator()
        m_masks.addAction(self.act_show_mask)
        m_masks.addAction(self.act_hide_mask)
        m_masks.addSeparator()
        m_masks.addAction(self.act_invert_mask)

        m_wim = mb.addMenu("&What's In My...")
        m_wim.addAction(self.act_whats_in_my_sky)
        m_wim.addAction(self.act_wimi)

        m_header = mb.addMenu("&Header Mods && Misc")
        m_header.addAction(self.act_fits_modifier)
        m_header.addAction(self.act_batch_renamer)
        m_header.addAction(self.act_astrobin_exporter)
        m_header.addAction(self.act_batch_convert)
        m_header.addAction(self.act_copy_astrometry)

        m_hist = mb.addMenu("&History")
        m_hist.addAction(self.act_history_explorer)

        m_short = mb.addMenu("&Shortcuts")

        act_cheats = QAction("Keyboard Shortcut Cheat Sheet…", self)
        act_cheats.triggered.connect(self._show_cheat_sheet)
        m_short.addAction(act_cheats)

        act_save_sc = QAction("Save Shortcuts Now", self, triggered=self.shortcuts.save_shortcuts)
        act_clear_sc = QAction("Clear All Shortcuts", self, triggered=self.shortcuts.clear)
        m_short.addAction(act_save_sc)
        m_short.addAction(act_clear_sc)

        m_view = mb.addMenu("&View")
        m_view.addAction(self.act_cascade)
        m_view.addAction(self.act_tile)
        m_view.addAction(self.act_tile_vert)
        m_view.addAction(self.act_tile_horiz)
        m_view.addAction(self.act_tile_grid)  

        m_settings = mb.addMenu("&Settings")
        m_settings.addAction("Preferences…", self._open_settings)

        m_about = mb.addMenu("&About")
        m_about.addAction("About...", self._about)
        m_about.addAction(self.act_check_updates)

        # initialize enabled state + names
        self.update_undo_redo_action_labels()

    # --- Shortcuts -> View Panels (dynamic) --------------------------------------
    def _visible_subwindows(self):
        # Only arrange visible, non-minimized views
        subs = [sw for sw in self.mdi.subWindowList()
                if sw.isVisible() and not (sw.windowState() & Qt.WindowState.WindowMinimized)]
        return subs

    def _cascade_views(self):
        self.mdi.cascadeSubWindows()

    def _tile_views(self):
        self.mdi.tileSubWindows()

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

    def _tile_views_grid(self):
        """Arrange near-square grid across the MDI area."""
        subs = self._visible_subwindows()
        if not subs:
            return
        area = self.mdi.viewport().rect()
        off = self.mdi.viewport().mapTo(self.mdi, area.topLeft())
        origin_x, origin_y = off.x(), off.y()

        n = len(subs)
        # rows x cols ≈ square
        cols = int(max(1, math.ceil(math.sqrt(n))))
        rows = int(max(1, math.ceil(n / cols)))

        cell_w = max(1, area.width() // cols)
        cell_h = max(1, area.height() // rows)

        for idx, sw in enumerate(subs):
            r = idx // cols
            c = idx % cols
            sw.setGeometry(origin_x + c*cell_w, origin_y + r*cell_h, cell_w, cell_h)


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
                # Ignore layout churn during minimize/restore; don’t let “False” uncheck the action.
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

    def changeEvent(self, ev):
        super().changeEvent(ev)
        if ev.type() == QEvent.Type.WindowStateChange:
            if self.windowState() & Qt.WindowState.WindowMinimized:
                # entering minimized — just guard; do not snapshot false vis
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

                # Enforce intended vis for each known dock (in case restoreState wasn’t enough)
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

                # Capture a fresh last-good layout now that we’re back
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



    def _open_view_bundles(self):
        try:

            show_view_bundles(self)
        except Exception as e:
            QMessageBox.warning(self, "View Bundles", f"Open failed:\n{e}")

    def _open_function_bundles(self):
        try:

            show_function_bundles(self)
        except Exception as e:
            QMessageBox.warning(self, "Function Bundles", f"Open failed:\n{e}")

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
            ("Drag view → Off to Canvas", "View", "Duplicate Image"),
            ("Drag view → On to Other Image", "View", "Copy Zoom and Pan"),
            ("Shift+Drag → On to Other Image", "View", "Apply that image to the other as a mask"), 
            ("Ctrl+Drag → On to Other Image", "View", "Copy Astrometric Solution"),            

            # View zoom
            ("Ctrl+1", "View", "Zoom to 100% (1:1)"),
            ("Ctrl+0", "View", "Fit image to current window"),

            # Window switching
            ("Ctrl+PgDown", "MDI", "Switch to previously active view"),
            ("Ctrl+PgUp",   "MDI", "Switch to next active view"),

            # Shortcuts canvas + buttons
            ("Alt+Drag (toolbar button)", "Toolbar", "Create a desktop shortcut for that action"),
            ("Alt+Drag (shortcut button → view)", "Shortcuts", "Headless apply the shortcut’s command/preset to a view"),
            ("Ctrl/Shift+Click", "Shortcuts", "Multi-select shortcut buttons"),
            ("Drag (selection)", "Shortcuts", "Move selected shortcut buttons"),
            ("Delete / Backspace", "Shortcuts", "Delete selected shortcut buttons"),
            ("Ctrl+A", "Shortcuts", "Select all shortcut buttons"),
            ("Double-click empty area", "MDI background", "Open files dialog"),

            # Layers dock
            ("Drag view → Layers list", "Layers", "Add dragged view as a new layer (on top)"),
            ("Shift+Drag mask → Layers list", "Layers", "Attach dragged image as mask to the selected layer"),

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

    def _parse_version_tuple(self, s: str) -> tuple[int, ...]:
        nums = re.findall(r"\d+", s or "")
        return tuple(int(n) for n in nums) if nums else (0,)

    def _ensure_network_manager(self):
        if getattr(self, "_nam", None) is None:
            self._nam = QNetworkAccessManager(self)
            self._nam.finished.connect(self._on_update_reply)

    def _kick_update_check(self, *, interactive: bool):
        self._ensure_network_manager()   # ← ensure _nam exists
        url_str = self.settings.value("updates/url", self._updates_url, type=str) or self._updates_url
        req = QNetworkRequest(QUrl(url_str))
        req.setRawHeader(b"User-Agent", f"SASPro/{globals().get('VERSION','0.0.0')}".encode("utf-8"))
        reply = self._nam.get(req)
        reply.setProperty("interactive", interactive)

        # Optional: if you ever hit SSL errors on some systems, you can ignore or log:
        # reply.sslErrors.connect(lambda errs, r=reply: r.ignoreSslErrors())


    def check_for_updates_now(self):
        if self.statusBar():
            self.statusBar().showMessage("Checking for updates…")
        self._kick_update_check(interactive=True)

    def check_for_updates_startup(self):
        self._kick_update_check(interactive=False)

    def _on_update_reply(self, reply: QNetworkReply):
        interactive = bool(reply.property("interactive"))
        try:
            if reply.error() != QNetworkReply.NetworkError.NoError:
                err = reply.errorString()
                if self.statusBar():
                    self.statusBar().showMessage("Update check failed.", 5000)
                if interactive:
                    QMessageBox.warning(self, "Update Check Failed",
                                        f"Unable to check for updates.\n\n{err}")
                else:
                    print(f"[updates] check failed: {err}")
                return

            raw = bytes(reply.readAll())
            try:
                data = json.loads(raw.decode("utf-8"))
            except Exception as je:
                if self.statusBar():
                    self.statusBar().showMessage("Update check failed (bad JSON).", 5000)
                if interactive:
                    QMessageBox.warning(self, "Update Check Failed",
                                        f"Update JSON is invalid.\n\n{je}")
                else:
                    print(f"[updates] bad JSON: {je}")
                return

            latest_str = str(data.get("version", "")).strip()
            notes = str(data.get("notes", "") or "")
            downloads = data.get("downloads", {}) or {}

            if not latest_str:
                if self.statusBar():
                    self.statusBar().showMessage("Update check failed (no 'version').", 5000)
                if interactive:
                    QMessageBox.warning(self, "Update Check Failed",
                                        "Update JSON missing the 'version' field.")
                else:
                    print("[updates] JSON missing 'version'")
                return

            cur_tuple = self._parse_version_tuple(globals().get("VERSION", "0.0.0"))
            latest_tuple = self._parse_version_tuple(latest_str)
            available = bool(latest_tuple and latest_tuple > cur_tuple)

            if available:
                if self.statusBar():
                    self.statusBar().showMessage(f"Update available: {latest_str}", 5000)
                msg_box = QMessageBox(self)
                msg_box.setIcon(QMessageBox.Icon.Information)
                msg_box.setWindowTitle("Update Available")
                msg_box.setText(f"A new version ({latest_str}) is available!")
                if notes:
                    msg_box.setInformativeText(f"Release Notes:\n{notes}")
                msg_box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                msg_box.setDefaultButton(QMessageBox.StandardButton.Yes)

                if downloads:
                    details = "\n".join([f"{k}: {v}" for k, v in downloads.items()])
                    msg_box.setDetailedText(details)

                if msg_box.exec() == QMessageBox.StandardButton.Yes:
                    plat = sys.platform
                    link = downloads.get(
                        "Windows" if plat.startswith("win") else
                        "macOS" if plat.startswith("darwin") else
                        "Linux" if plat.startswith("linux") else "", ""
                    )
                    if link:
                        webbrowser.open(link)
                    else:
                        QMessageBox.warning(self, "Download", "No download link available for this platform.")
            else:
                if self.statusBar():
                    self.statusBar().showMessage("You’re up to date.", 3000)
                if interactive:
                    QMessageBox.information(self, "Up to Date",
                                            "You’re already running the latest version.")
        finally:
            reply.deleteLater()


    def _doc_by_ptr(self, ptr: int):
        dm = getattr(self, "doc_manager", None) or getattr(self, "docman", None)
        if dm and hasattr(dm, "all_documents"):
            for d in dm.all_documents():
                if id(d) == ptr:
                    return d
        return None

    # --- WCS helpers (put inside AstroSuiteProMainWindow) -------------------


    # Exact keys we always consider WCS
    _WCS_KEY_SET = {
        "WCSAXES", "CTYPE1", "CTYPE2", "CUNIT1", "CUNIT2",
        "CRPIX1", "CRPIX2", "CRVAL1", "CRVAL2",
        "CD1_1", "CD1_2", "CD2_1", "CD2_2",
        "PC1_1", "PC1_2", "PC2_1", "PC2_2",
        "CDELT1", "CDELT2",
        "LONPOLE", "LATPOLE",
        "RADESYS", "RADECSYS", "EQUINOX", "EPOCH",
        "NAXIS1", "NAXIS2"  # useful context for UIs/solvers
    }
    # Prefixes we treat as WCS/SIP
    _WCS_PREFIXES = ("PV", "PROJP", "A_", "B_", "AP_", "BP_", "LTM", "LTV")

    def _ensure_header_map(self, doc):
        meta = getattr(doc, "metadata", None)
        if meta is None:
            return None
        hdr = meta.get("original_header")
        if not isinstance(hdr, dict):
            hdr = {}
            meta["original_header"] = hdr
        return hdr

    # --- WCS copy helpers (MainWindow) ------------------------------------

    def _coerce_wcs_numbers(self, d: dict) -> dict:
        """Convert common WCS/SIP values to int/float where appropriate."""
        import re
        numeric = {
            "CRPIX1","CRPIX2","CRVAL1","CRVAL2","CDELT1","CDELT2",
            "CD1_1","CD1_2","CD2_1","CD2_2","PC1_1","PC1_2","PC2_1","PC2_2",
            "CROTA1","CROTA2","EQUINOX","WCSAXES","A_ORDER","B_ORDER","AP_ORDER","BP_ORDER",
            "LONPOLE","LATPOLE"
        }
        out = {}
        for k, v in d.items():
            K = str(k).upper()
            try:
                if K in numeric or re.match(r"^(A|B|AP|BP)_\d+_\d+$", K or ""):
                    if isinstance(v, str):
                        s = v.strip()
                        # int if clean integer, else float
                        out[K] = int(s) if re.fullmatch(r"[+-]?\d+", s) else float(s)
                    else:
                        out[K] = v
                else:
                    out[K] = v
            except Exception:
                out[K] = v
        return out

    def _extract_wcs_dict(self, doc) -> dict:
        """Collect a complete WCS/SIP dict from the doc’s header/meta."""
        if doc is None:
            return {}
        src = (getattr(doc, "metadata", {}) or {}).get("original_header")

        # accept astropy.io.fits.Header or dict (or string blobs you may have converted)
        wcs = {}
        if src is None:
            pass
        else:
            try:
                # Header-like
                for k, v in dict(src).items():
                    K = str(k).upper()
                    if (K.startswith(("CRPIX","CRVAL","CDELT","CD","PC","CROTA","CTYPE","CUNIT",
                                    "WCSAXES","LONPOLE","LATPOLE","EQUINOX","PV")) or
                        K in {"RADECSYS","RADESYS","NAXIS1","NAXIS2","RADECSYS","RADESYS"} or
                        K.startswith(("A_","B_","AP_","BP_"))):
                        wcs[K] = v
            except Exception:
                pass

        # Also accept any mirror you previously stored
        meta = getattr(doc, "metadata", {}) or {}
        imgmeta = meta.get("image_meta") or meta.get("WCS") or {}
        if isinstance(imgmeta, dict):
            sub = imgmeta.get("WCS", imgmeta)
            if isinstance(sub, dict):
                for k, v in sub.items():
                    K = str(k).upper()
                    if (K.startswith(("CRPIX","CRVAL","CDELT","CD","PC","CROTA","CTYPE","CUNIT",
                                    "WCSAXES","LONPOLE","LATPOLE","EQUINOX","PV")) or
                        K in {"RADECSYS","RADESYS","NAXIS1","NAXIS2"} or
                        K.startswith(("A_","B_","AP_","BP_"))):
                        wcs.setdefault(K, v)

        # sensible defaults/parity
        if any(k.startswith(("A_","B_","AP_","BP_")) for k in wcs):
            wcs.setdefault("CUNIT1", "deg")
            wcs.setdefault("CUNIT2", "deg")
            # TAN-SIP labels if SIP present
            c1 = str(wcs.get("CTYPE1","RA---TAN"))
            c2 = str(wcs.get("CTYPE2","DEC--TAN"))
            if not c1.endswith("-SIP"): wcs["CTYPE1"] = "RA---TAN-SIP"
            if not c2.endswith("-SIP"): wcs["CTYPE2"] = "DEC--TAN-SIP"

        if "RADECSYS" in wcs and "RADESYS" not in wcs:
            wcs["RADESYS"] = wcs["RADECSYS"]
        if "WCSAXES" not in wcs and {"CTYPE1","CTYPE2"} <= wcs.keys():
            wcs["WCSAXES"] = 2

        return self._coerce_wcs_numbers(wcs)

    # --- WCS merge for "Copy Astrometric Solution" -------------------------
    def _ensure_header_for_doc(self, doc):
        """Return an astropy Header for doc.metadata['original_header'] (creating one if needed)."""
        from astropy.io.fits import Header
        meta = getattr(doc, "metadata", None)
        if not isinstance(meta, dict):
            setattr(doc, "metadata", {})
            meta = doc.metadata

        hdr_like = meta.get("original_header")
        # Already a Header?
        if isinstance(hdr_like, Header):
            hdr = hdr_like
        elif isinstance(hdr_like, dict):
            # coerce dict → Header (similar to plate_solver._as_header)
            import re
            hdr = Header()
            int_keys = {"A_ORDER", "B_ORDER", "AP_ORDER", "BP_ORDER", "WCSAXES", "NAXIS", "NAXIS1", "NAXIS2", "NAXIS3"}
            for k, v in dict(hdr_like).items():
                K = str(k).upper()
                try:
                    if K in int_keys:
                        hdr[K] = int(float(str(v).strip().split()[0]))
                    elif re.match(r"^(?:A|B|AP|BP)_\d+_\d+$", K) or \
                        re.match(r"^(?:CRPIX|CRVAL|CDELT|CD|PC|CROTA|LATPOLE|LONPOLE|EQUINOX)\d?_?\d*$", K):
                        hdr[K] = float(str(v).strip().split()[0])
                    else:
                        hdr[K] = v
                except Exception:
                    pass
        else:
            hdr = Header()

        # Ensure basic axis cards exist (needed for non-FITS sources)
        try:
            import numpy as np
            img = getattr(doc, "image", None)
            if img is not None:
                a = np.asarray(img)
                H = int(a.shape[0]) if a.ndim >= 2 else 1
                W = int(a.shape[1]) if a.ndim >= 2 else 1
                C = int(a.shape[2]) if a.ndim == 3 else 1
                # Only set when missing (don’t clobber real FITS headers)
                if "NAXIS" not in hdr:
                    hdr["NAXIS"]  = 2 if a.ndim != 3 else 3
                if "NAXIS1" not in hdr: hdr["NAXIS1"] = W
                if "NAXIS2" not in hdr: hdr["NAXIS2"] = H
                if a.ndim == 3 and "NAXIS3" not in hdr:
                    hdr["NAXIS3"] = C
                if "SIMPLE" not in hdr: hdr["SIMPLE"] = True
                if "BITPIX" not in hdr: hdr["BITPIX"] = -32
                if "BZERO"  not in hdr: hdr["BZERO"]  = 0.0
                if "BSCALE" not in hdr: hdr["BSCALE"] = 1.0
        except Exception:
            pass

        meta["original_header"] = hdr
        return hdr

    def _normalize_wcs_dict(self, wcs_dict: dict) -> dict:
        """Match plate_solver’s numeric coercions and CTYPE defaults."""
        import re
        d = {}
        numeric = {
            "CRPIX1","CRPIX2","CRVAL1","CRVAL2","CDELT1","CDELT2",
            "CD1_1","CD1_2","CD2_1","CD2_2","PC1_1","PC1_2","PC2_1","PC2_2",
            "CROTA1","CROTA2","EQUINOX","WCSAXES","A_ORDER","B_ORDER","AP_ORDER","BP_ORDER",
            "LONPOLE","LATPOLE"
        }
        for k, v in (wcs_dict or {}).items():
            K = str(k).upper()
            try:
                if K in numeric or re.match(r"^(A|B|AP|BP)_\d+_\d+$", K):
                    if isinstance(v, str):
                        s = v.strip()
                        v2 = int(s) if re.fullmatch(r"[+-]?\d+", s) else float(s)
                    else:
                        v2 = v
                    d[K] = v2
                else:
                    d[K] = v
            except Exception:
                d[K] = v

        # Ensure CTYPEs
        if "CTYPE1" not in d: d["CTYPE1"] = "RA---TAN"
        if "CTYPE2" not in d: d["CTYPE2"] = "DEC--TAN"
        # If SIP present, force TAN-SIP
        if any(K.startswith(("A_","B_","AP_","BP_")) for K in d):
            if not str(d["CTYPE1"]).endswith("-SIP"): d["CTYPE1"] = "RA---TAN-SIP"
            if not str(d["CTYPE2"]).endswith("-SIP"): d["CTYPE2"] = "DEC--TAN-SIP"
        if "RADECSYS" in d and "RADESYS" not in d:
            d["RADESYS"] = d["RADECSYS"]
        if "WCSAXES" not in d and {"CTYPE1","CTYPE2"} <= d.keys():
            d["WCSAXES"] = 2
        return d

    def _apply_wcs_dict_to_doc(self, doc, wcs_dict: dict) -> bool:
        """
        Write **separate** WCS/SIP cards directly into doc.metadata['original_header'] (astropy Header).
        Also build/attach a WCS object, set HasAstrometricSolution flags, refresh header dock.
        """
        if doc is None or not wcs_dict:
            return False

        # 1) Ensure we have a real FITS Header to write into
        hdr = self._ensure_header_for_doc(doc)

        # 2) Normalize/coerce values exactly like plate_solver
        w = self._normalize_wcs_dict(wcs_dict)

        # 3) Merge keys into header (no nesting)
        changed = False
        for k, v in w.items():
            try:
                old = hdr.get(k)
                if old != v:
                    hdr[k] = v
                    changed = True
            except Exception:
                # skip weird keys (overlong etc.)
                pass

        # 4) Mark solution flags in header and at top-level (for quick checks)
        try:
            if hdr.get("HasAstrometricSolution") is not True:
                hdr["HasAstrometricSolution"] = True
                changed = True
        except Exception:
            pass
        meta = doc.metadata
        if meta.get("HasAstrometricSolution") is not True:
            meta["HasAstrometricSolution"] = True
            changed = True

        # 5) (Optional) also keep a mirror under image_meta — but not as the primary store
        im = meta.get("image_meta")
        if not isinstance(im, dict):
            im = {}
        im["WCS"] = dict(w)  # mirror only
        meta["image_meta"] = im

        # 6) Build a WCS object from the header (best effort)
        try:
            from astropy.wcs import WCS
            meta["wcs"] = WCS(hdr)
        except Exception:
            pass

        # 7) Notify UI
        if changed and hasattr(doc, "changed"):
            try: doc.changed.emit()
            except Exception: pass
        if hasattr(self, "_refresh_header_viewer"):
            self._hdr_refresh_timer.start(0)
        if hasattr(self, "currentDocumentChanged"):
            try: self.currentDocumentChanged.emit(doc)
            except Exception: pass
        return True



    # expose constants to methods above
    _WCS_KEY_SET = _WCS_KEY_SET
    _WCS_PREFIXES = _WCS_PREFIXES



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
                except Exception: pass

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

    def _new_project(self):
        if not self._confirm_discard(title="New Project",
                                    msg="Start a new project? This closes all views and clears desktop shortcuts."):
            return

        # Close views + docs + shelf
        self._close_all_subwindows()
        self._clear_all_documents()
        self._clear_minimized_shelf()

        # Clear desktop shortcuts (widgets + persisted positions)
        try:
            if getattr(self, "shortcuts", None):
                self.shortcuts.clear()
            else:
                # Fallback: wipe persisted layout so nothing reloads later
                from PyQt6.QtCore import QSettings
                from pro.shortcuts import SET_KEY_V1, SET_KEY_V2
                s = QSettings()
                s.setValue(SET_KEY_V2, "[]")
                s.remove(SET_KEY_V1)
                s.sync()
        except Exception:
            pass

        # (Optional) keep canvas ready for fresh adds
        try:
            if getattr(self, "shortcuts", None):
                self.shortcuts.canvas.raise_()
                self.shortcuts.canvas.show()
                self.shortcuts.canvas.setFocus()
        except Exception:
            pass

        self._log("New project workspace ready.")

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

    def _collect_open_documents(self):
        # Prefer DocManager if present
        dm = getattr(self, "doc_manager", None) or getattr(self, "docman", None)
        if dm is not None and getattr(dm, "_docs", None) is not None:
            return list(dm._docs)

        # Fallback: harvest from open subwindows
        docs = []
        for sw in self.mdi.subWindowList():
            try:
                view = sw.widget()
                doc = getattr(view, "document", None)
                if doc is not None:
                    docs.append(doc)
            except Exception:
                pass
        return docs

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
                    "• Yes = smaller file, slower save\n"
                    "• No  = larger file, faster save")
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

    def _save_project(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Project", "", "SetiAstro Project (*.sas)"
        )
        if not path:
            return
        if not path.lower().endswith(".sas"):
            path += ".sas"

        docs = self._collect_open_documents()
        if not docs:
            QMessageBox.warning(self, "Save Project", "No documents to save.")
            return

        try:
            compress = self._ask_project_compress()  # your existing yes/no dialog

            # Busy dialog (indeterminate)
            dlg = QProgressDialog("Saving project…", "", 0, 0, self)
            dlg.setWindowTitle("Saving")
            # PyQt6 (with PyQt5 fallback if you ever run it there)
            try:
                dlg.setWindowModality(Qt.WindowModality.ApplicationModal)
            except AttributeError:
                dlg.setWindowModality(Qt.ApplicationModal)  # PyQt5

            # Hide the cancel button (API differs across versions)
            try:
                dlg.setCancelButton(None)
            except TypeError:
                dlg.setCancelButtonText("")

            dlg.setAutoClose(False)
            dlg.setAutoReset(False)
            dlg.show()

            # Threaded save
            self._proj_save_worker = _ProjectSaveWorker(
                path,
                docs,
                getattr(self, "shortcuts", None),
                getattr(self, "mdi", None),
                compress,
                parent=self,
            )
            self._proj_save_worker.ok.connect(lambda: (dlg.close(), self._log("Project saved.")))
            self._proj_save_worker.error.connect(lambda msg: (dlg.close(), QMessageBox.critical(self, "Save Project", f"Failed to save:\n{msg}")))
            self._proj_save_worker.finished.connect(lambda: setattr(self, "_proj_save_worker", None))  # GC
            self._proj_save_worker.start()

        except Exception as e:
            QMessageBox.critical(self, "Save Project", f"Failed to save:\n{e}")


    def _load_project(self):
        # warn / clear current desktop
        if getattr(self, "doc_manager", None) and self.doc_manager._docs:
            if not self._confirm_discard(
                title="Load Project",
                msg=("Loading a project will close current views and replace desktop shortcuts.\n"
                    "Continue?")
            ):
                return
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

        path, _ = QFileDialog.getOpenFileName(
            self, "Load Project", "", "SetiAstro Project (*.sas)"
        )
        if not path:
            return

        # ensure DocManager exists
        if not hasattr(self, "doc_manager") or self.doc_manager is None:
            from pro.doc_manager import DocManager
            self.doc_manager = DocManager(image_manager=getattr(self, "image_manager", None), parent=self)

        # progress (“thinking”) dialog
        dlg = QProgressDialog("Loading project…", "", 0, 0, self)
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
            # ✅ instantiate the reader
            ProjectReader(self).read(path)
            self._log("Project loaded.")
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

    def _undo_active(self):
        doc = self._active_doc()
        if doc and doc.can_undo():
            name = doc.undo()
            if name:
                self._log(f"Undo: {name}")
        self.update_undo_redo_action_labels()

    def _redo_active(self):
        doc = self._active_doc()
        if doc and doc.can_redo():
            name = doc.redo()
            if name:
                self._log(f"Redo: {name}")
        self.update_undo_redo_action_labels()

    def update_undo_redo_action_labels(self):
        doc = self._active_doc()
        if not hasattr(self, "act_undo"):
            return

        if doc:
            undo_name = doc.last_undo_name()
            redo_name = doc.last_redo_name()

            # Button text
            self.act_undo.setText(f"Undo {undo_name}" if undo_name else "Undo")
            self.act_redo.setText(f"Redo {redo_name}" if redo_name else "Redo")

            # Tooltips / Status tips
            undo_tip = ("Nothing to undo" if not doc.can_undo()
                        else (f"Undo: {undo_name}" if undo_name else "Undo last action"))
            redo_tip = ("Nothing to redo" if not doc.can_redo()
                        else (f"Redo: {redo_name}" if redo_name else "Redo last action"))

            self.act_undo.setToolTip(undo_tip)
            self.act_redo.setToolTip(redo_tip)
            self.act_undo.setStatusTip(undo_tip)
            self.act_redo.setStatusTip(redo_tip)

            # Enabled state
            self.act_undo.setEnabled(doc.can_undo())
            self.act_redo.setEnabled(doc.can_redo())
        else:
            self.act_undo.setText("Undo")
            self.act_redo.setText("Redo")
            self.act_undo.setToolTip("Nothing to undo")
            self.act_redo.setToolTip("Nothing to redo")
            self.act_undo.setStatusTip("Nothing to undo")
            self.act_redo.setStatusTip("Nothing to redo")
            self.act_undo.setEnabled(False)
            self.act_redo.setEnabled(False)


    def _current_document(self):
        sw = self.mdi.activeSubWindow()
        if not sw:
            return None
        vw = sw.widget()
        return getattr(vw, "document", None)

    def _find_subwindow_for_doc(self, doc):
        """Find the QMdiSubWindow that hosts this exact doc object.
        Falls back to a tagged id() property if present."""
        want_ptr = id(doc)
        for sw in self.mdi.subWindowList():
            vw = sw.widget()
            # primary: exact object identity
            if getattr(vw, "document", None) is doc:
                return sw
            # fallback: compare a tagged pointer if the widget exposes it
            try:
                if int(vw.property("doc_ptr") or 0) == want_ptr:
                    return sw
            except Exception:
                pass
        return None

    def _open_subwindow_for_added_doc(self, doc):
        # Avoid duplicate views if something else already created one
        if self._find_subwindow_for_doc(doc):
            return
        try:
            # Create it and use the returned subwindow directly
            sw = self._spawn_subwindow_for(doc)
            if sw:
                # Activate on the next loop tick so Qt has applied flags/layout
                from PyQt6.QtCore import QTimer
                QTimer.singleShot(0, lambda: self.mdi.setActiveSubWindow(sw))
        except Exception as e:
            # Safe fallback: show *something* and log why
            try:
                if hasattr(self, "_log"):
                    self._log(f"Failed to open subwindow for {doc.display_name()}: {e}")
            except Exception:
                pass
            from PyQt6.QtWidgets import QLabel, QMdiSubWindow
            w = QLabel(doc.display_name()); setattr(w, "document", doc)
            wrapper = QMdiSubWindow(self); wrapper.setWidget(w)
            self.mdi.addSubWindow(wrapper); wrapper.show()



    def _action_create_mask(self):
        doc = self._current_document()
        if doc is None or getattr(doc, "image", None) is None:
            QMessageBox.information(self, "No image", "Open an image first.")
            return
        created = create_mask_and_attach(self, doc)
        # Optional toast/log
        if created and hasattr(self, "_log"):
            self._log("Mask created and set active.")

    def _header_widget(self):
        """Return the widget inside the header dock (whatever you created)."""
        # Adjust this if you used a different name in _init_header_viewer_dock
        return getattr(self, "header_viewer", None)

    def _clear_header_viewer(self, msg: str = "No image"):
        w = self._header_widget()
        if w is None:
            return
        # QTableWidget?
        try:
            from PyQt6.QtWidgets import QTableWidget
            if isinstance(w, QTableWidget):
                w.setRowCount(0)
                w.setColumnCount(3)
                w.setHorizontalHeaderLabels(["Key", "Value", "Comment"])
                return
        except Exception:
            pass
        # QListWidget?
        try:
            from PyQt6.QtWidgets import QListWidget
            if isinstance(w, QListWidget):
                w.clear()
                w.addItem(msg)
                return
        except Exception:
            pass
        # QTextEdit-like?
        if hasattr(w, "clear") and hasattr(w, "setPlainText"):
            w.clear()
            w.setPlainText(msg)

    def _refresh_header_viewer(self, doc=None):
        """Rebuild the header dock from the given (or active) doc — never raises."""
        try:
            doc = doc or self._active_doc()
            hv = getattr(self, "header_viewer", None)

            # If your dock widget has a native API, try it but don't trust it.
            if hv and hasattr(hv, "set_document"):
                try:
                    hv.set_document(doc)
                    return
                except Exception as e:
                    print("[header] set_document suppressed:", e)

            # Fallback path: extract → populate, all guarded.
            rows = self._extract_header_pairs(doc)
            if not rows:
                self._clear_header_viewer("No header" if doc else "No image")
            else:
                self._populate_header_viewer(rows)
        except Exception as e:
            print("[header] refresh suppressed:", e)
            self._clear_header_viewer("")


    def _extract_header_pairs(self, doc):
        """
        Return list[(key, value, comment)].
        Prefers a JSON-safe snapshot if present, otherwise best-effort parsing.
        Never raises.
        """
        try:
            if not doc:
                return []

            meta = getattr(doc, "metadata", {}) or {}

            # 1) Prefer a snapshot if any writer/loader provided it.
            snap = meta.get("__header_snapshot__")
            if isinstance(snap, dict):
                fmt = snap.get("format")
                if fmt == "fits-cards":
                    cards = snap.get("cards") or []
                    out = []
                    for it in cards:
                        try:
                            k, v, c = it
                        except Exception:
                            # tolerate weird shapes
                            k, v, c = (str(it[0]) if it else "",
                                    "" if len(it) < 2 else str(it[1]),
                                    "" if len(it) < 3 else str(it[2]))
                        out.append((str(k), str(v), str(c)))
                    return out
                if fmt == "dict":
                    items = snap.get("items") or {}
                    out = []
                    for k, v in items.items():
                        if isinstance(v, dict):
                            out.append((str(k), str(v.get("value","")), str(v.get("comment",""))))
                        else:
                            out.append((str(k), str(v), ""))
                    return out
                if fmt == "repr":
                    return [("Header", str(snap.get("text","")), "")]

            # 2) Live header object(s) (can be astropy, dict, or random).
            hdr = (meta.get("original_header")
                or meta.get("fits_header")
                or meta.get("header"))

            if hdr is None:
                return []

            # astropy.io.fits.Header (optional; no hard dependency)
            try:
                from astropy.io.fits import Header  # type: ignore
            except Exception:
                Header = None  # type: ignore

            if Header is not None:
                try:
                    if isinstance(hdr, Header):
                        out = []
                        for k in hdr.keys():
                            try: val = hdr[k]
                            except Exception: val = ""
                            try: cmt = hdr.comments[k]
                            except Exception: cmt = ""
                            out.append((str(k), str(val), str(cmt)))
                        return out
                except Exception as e:
                    print("[header] astropy parse suppressed:", e)

            # dict-ish header (e.g., XISF-like)
            if isinstance(hdr, dict):
                out = []
                for k, v in hdr.items():
                    if isinstance(v, dict):
                        out.append((str(k), str(v.get("value","")), str(v.get("comment",""))))
                    else:
                        # avoid huge array dumps
                        try:
                            import numpy as _np
                            if isinstance(v, _np.ndarray):
                                v = f"ndarray{tuple(v.shape)}"
                        except Exception:
                            pass
                        out.append((str(k), str(v), ""))
                return out

            # Fallback: string repr
            return [("Header", str(hdr), "")]
        except Exception as e:
            print("[header] extract suppressed:", e)
            return []


    def _populate_header_viewer(self, rows):
        """Render rows into whatever widget you expose; never raises."""
        try:
            w = self._header_widget()
        except Exception as e:
            print("[header] _header_widget suppressed:", e)
            return
        if w is None:
            return

        # Table widget path
        try:
            from PyQt6.QtWidgets import QTableWidget, QTableWidgetItem
            if isinstance(w, QTableWidget):
                try:
                    w.setRowCount(0)
                    w.setColumnCount(3)
                    w.setHorizontalHeaderLabels(["Key", "Value", "Comment"])
                    for r, (k, v, c) in enumerate(rows):
                        w.insertRow(r)
                        w.setItem(r, 0, QTableWidgetItem(k))
                        w.setItem(r, 1, QTableWidgetItem(v))
                        w.setItem(r, 2, QTableWidgetItem(c))
                    return
                except Exception as e:
                    print("[header] table populate suppressed:", e)
        except Exception:
            pass

        # List widget path
        try:
            from PyQt6.QtWidgets import QListWidget
            if isinstance(w, QListWidget):
                try:
                    w.clear()
                    for k, v, c in rows:
                        w.addItem(f"{k} = {v}" + (f"  / {c}" if c else ""))
                    return
                except Exception as e:
                    print("[header] list populate suppressed:", e)
        except Exception:
            pass

        # Plain text-ish
        try:
            if hasattr(w, "setPlainText"):
                w.setPlainText("\n".join(
                    f"{k} = {v}" + (f"  / {c}" if c else "") for (k, v, c) in rows
                ))
                return
            if hasattr(w, "setText"):
                w.setText("\n".join(
                    f"{k} = {v}" + (f"  / {c}" if c else "") for (k, v, c) in rows
                ))
                return
        except Exception as e:
            print("[header] text populate suppressed:", e)


    def _clear_header_viewer(self, msg=""):
        """Clear the dock safely (any widget type)."""
        try:
            w = self._header_widget()
            if not w:
                return
            if hasattr(w, "clear"):
                w.clear()
            if hasattr(w, "setPlainText"):
                w.setPlainText(str(msg))
            elif hasattr(w, "setText"):
                w.setText(str(msg))
        except Exception as e:
            print("[header] clear suppressed:", e)


    def _clear_header_viewer(self, message: str = ""):
        """Clear header viewer content quietly—no dialogs."""
        w = self._header_widget()
        if w is None:
            return
        try:
            # QTableWidget
            from PyQt6.QtWidgets import QTableWidget  # type: ignore
            if isinstance(w, QTableWidget):
                w.setRowCount(0)
                w.setColumnCount(3)
                w.setHorizontalHeaderLabels(["Key", "Value", "Comment"])
                return
        except Exception:
            pass
        try:
            # QListWidget
            from PyQt6.QtWidgets import QListWidget  # type: ignore
            if isinstance(w, QListWidget):
                w.clear()
                if message:
                    w.addItem(message)
                return
        except Exception:
            pass
        # QTextEdit-like
        if hasattr(w, "setPlainText"):
            try:
                w.setPlainText(message or "")
            except Exception:
                pass


    def _header_widget(self):
        """
        Find the concrete widget used to display header text/table.
        Never raises; returns None if nothing sensible is found.
        """
        # Common setups:
        hv = getattr(self, "header_viewer", None) or getattr(self, "metadata_dock", None)
        if hv is None:
            return None

        # If it's a dock widget (QDockWidget-like), pull its child widget
        try:
            if hasattr(hv, "widget") and callable(hv.widget):
                inner = hv.widget()
                if inner is not None:
                    return inner
        except Exception:
            pass

        # It might already be the actual widget
        return hv

    def _on_doc_added_for_header_sync(self, doc):
        # Update header when the *active* doc changes
        try:
            doc.changed.connect(self._maybe_refresh_header_on_doc_change)
        except Exception:
            pass

    def _on_doc_removed_for_header_sync(self, doc):
        # If the removed doc was the active one, clear header
        if doc is self._active_doc():
            self._clear_header_viewer("No image")
            hv = getattr(self, "header_viewer", None)
            if hv and hasattr(hv, "set_document"):
                try:
                    hv.set_document(None)
                except Exception:
                    pass

        # If there are no more subwindows, force a global clear too
        if not self.mdi.subWindowList():
            self.currentDocumentChanged.emit(None)
            self._hdr_refresh_timer.start(0)

    def _maybe_refresh_header_on_doc_change(self):
        sender = self.sender()
        if sender is self._active_doc():
            self._hdr_refresh_timer.start(0)

    def _format_explorer_title(self, doc) -> str:
        name = _strip_ui_decorations(doc.display_name() or "Untitled")

        dims = ""
        try:
            import numpy as np
            arr = getattr(doc, "image", None)
            if isinstance(arr, np.ndarray) and arr.size:
                h, w = arr.shape[:2]
                c = arr.shape[2] if arr.ndim == 3 else 1
                dims = f"  —  {h}x{w}x{c}"
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


    def _exec_geom_invert(self):
        sw = self.mdi.activeSubWindow() if hasattr(self, "mdi") else None
        view = sw.widget() if sw else None
        doc = getattr(view, "document", None)
        if doc is None or getattr(doc, "image", None) is None:
            QMessageBox.information(self, "Invert", "Active view has no image.")
            return
        try:
            self._apply_geom_invert_to_doc(doc)
            self._log("Invert applied to active view")
        except Exception as e:
            QMessageBox.critical(self, "Invert", str(e))

    def _exec_geom_flip_h(self):
        sw = self.mdi.activeSubWindow() if hasattr(self, "mdi") else None
        view = sw.widget() if sw else None
        doc = getattr(view, "document", None)
        if doc is None or getattr(doc, "image", None) is None:
            QMessageBox.information(self, "Flip Horizontal", "Active view has no image.")
            return
        try:
            self._apply_geom_flip_h_to_doc(doc)
            self._log("Flip Horizontal applied to active view")
        except Exception as e:
            QMessageBox.critical(self, "Flip Horizontal", str(e))

    def _exec_geom_flip_v(self):
        sw = self.mdi.activeSubWindow() if hasattr(self, "mdi") else None
        view = sw.widget() if sw else None
        doc = getattr(view, "document", None)
        if doc is None or getattr(doc, "image", None) is None:
            QMessageBox.information(self, "Flip Vertical", "Active view has no image.")
            return
        try:
            self._apply_geom_flip_v_to_doc(doc)
            self._log("Flip Vertical applied to active view")
        except Exception as e:
            QMessageBox.critical(self, "Flip Vertical", str(e))

    def _exec_geom_rot_cw(self):
        sw = self.mdi.activeSubWindow() if hasattr(self, "mdi") else None
        view = sw.widget() if sw else None
        doc = getattr(view, "document", None)
        if doc is None or getattr(doc, "image", None) is None:
            QMessageBox.information(self, "Rotate 90° CW", "Active view has no image.")
            return
        try:
            self._apply_geom_rot_cw_to_doc(doc)
            self._log("Rotate 90° CW applied to active view")
        except Exception as e:
            QMessageBox.critical(self, "Rotate 90° CW", str(e))

    def _exec_geom_rot_ccw(self):
        sw = self.mdi.activeSubWindow() if hasattr(self, "mdi") else None
        view = sw.widget() if sw else None
        doc = getattr(view, "document", None)
        if doc is None or getattr(doc, "image", None) is None:
            QMessageBox.information(self, "Rotate 90° CCW", "Active view has no image.")
            return
        try:
            self._apply_geom_rot_ccw_to_doc(doc)
            self._log("Rotate 90° CCW applied to active view")
        except Exception as e:
            QMessageBox.critical(self, "Rotate 90° CCW", str(e))

    def _exec_geom_rescale(self):
        sw = self.mdi.activeSubWindow() if hasattr(self, "mdi") else None
        view = sw.widget() if sw else None
        doc = getattr(view, "document", None)
        if doc is None or getattr(doc, "image", None) is None:
            QMessageBox.information(self, "Rescale Image", "Active view has no image.")
            return

        # remember last value (like other tools that recall settings)
        if not hasattr(self, "_last_rescale_factor"):
            self._last_rescale_factor = 1.0

        dlg = QInputDialog(self)
        dlg.setWindowTitle("Rescale Image")
        dlg.setLabelText("Enter scaling factor (e.g., 0.5 for 50%, 2 for 200%):")
        dlg.setInputMode(QInputDialog.InputMode.DoubleInput)
        dlg.setDoubleRange(0.1, 10.0)
        dlg.setDoubleDecimals(2)
        dlg.setDoubleValue(self._last_rescale_factor)

        # make sure it’s a true window so the icon shows on all platforms
        dlg.setWindowFlag(Qt.WindowType.Window, True)

        # set the icon from rescale_path if available
        icon_path = getattr(self, "rescale_path", None)
        if icon_path is None:
            try:
                icon_path = rescale_path  # module/global fallback if you keep it there
            except NameError:
                icon_path = None
        if icon_path:
            dlg.setWindowIcon(QIcon(icon_path))

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        factor = dlg.doubleValue()


        try:
            self._apply_geom_rescale_to_doc(doc, factor=factor)
            self._last_rescale_factor = factor
            self._log(f"Rescale ({factor:g}×) applied to active view")
        except Exception as e:
            QMessageBox.critical(self, "Rescale Image", str(e))

# --- Geometry: headless apply-to-doc helpers (like apply_remove_green_preset_to_doc) ---

    def _apply_geom_invert_to_doc(self, doc):
        arr = np.asarray(doc.image, dtype=np.float32)
        out = invert_image_numba(arr)
        if hasattr(doc, "set_image"):
            doc.set_image(out, step_name="Invert")
        else:
            doc.image = out

    def _apply_geom_flip_h_to_doc(self, doc):
        arr = np.asarray(doc.image, dtype=np.float32)
        out = flip_horizontal_numba(arr)
        if hasattr(doc, "set_image"):
            doc.set_image(out, step_name="Flip Horizontal")
        else:
            doc.image = out

    def _apply_geom_flip_v_to_doc(self, doc):
        arr = np.asarray(doc.image, dtype=np.float32)
        out = flip_vertical_numba(arr)
        if hasattr(doc, "set_image"):
            doc.set_image(out, step_name="Flip Vertical")
        else:
            doc.image = out

    def _apply_geom_rot_cw_to_doc(self, doc):
        arr = np.asarray(doc.image, dtype=np.float32)
        out = rotate_90_clockwise_numba(arr)
        if hasattr(doc, "set_image"):
            doc.set_image(out, step_name="Rotate 90° Clockwise")
        else:
            doc.image = out

    def _apply_geom_rot_ccw_to_doc(self, doc):
        arr = np.asarray(doc.image, dtype=np.float32)
        out = rotate_90_counterclockwise_numba(arr)
        if hasattr(doc, "set_image"):
            doc.set_image(out, step_name="Rotate 90° Counterclockwise")
        else:
            doc.image = out

    def _apply_geom_rescale_to_doc(self, doc, *, factor: float):
        factor = float(max(0.1, min(10.0, factor)))
        arr = np.asarray(doc.image, dtype=np.float32)
        out = rescale_image_numba(arr, factor)
        if hasattr(doc, "set_image"):
            doc.set_image(out, step_name=f"Rescale ({factor:g}×)")
        else:
            doc.image = out

    def _apply_geom_rescale_preset_to_doc(self, doc, preset):
        """
        Accepts flexible presets:
        - dict with 'factor' or 'scale'
        - a lone float/int
        - a '0.5x'/'2x' string
        - (factor, ...) tuple/list
        Falls back to 1.0 if unparsable.
        """
        factor = None
        try:
            if isinstance(preset, dict):
                factor = preset.get("factor", preset.get("scale", None))
            elif isinstance(preset, (float, int)):
                factor = float(preset)
            elif isinstance(preset, str):
                s = preset.strip().lower().replace("×", "x")
                if s.endswith("x"):
                    s = s[:-1]
                factor = float(s)
            elif isinstance(preset, (list, tuple)) and preset:
                factor = float(preset[0])
        except Exception:
            factor = None

        if factor is None:
            factor = getattr(self, "_last_rescale_factor", 1.0) or 1.0

        self._apply_geom_rescale_to_doc(doc, factor=factor)

    def _apply_rescale_preset_to_doc(self, doc, preset: dict):
        """
        Headless rescale for drag-and-drop / shortcut preset application.
        Expects preset like {"factor": 1.25}.
        """
        factor = float(preset.get("factor", 1.0))
        if not (0.10 <= factor <= 10.0):
            raise ValueError("Rescale factor must be between 0.10 and 10.0")

        if getattr(doc, "image", None) is None:
            raise RuntimeError("Target document has no image")

        # Make sure you have this import at module top with your other numba utils:
        # from numba_utils import rescale_image_numba
        src = np.asarray(doc.image, dtype=np.float32, order="C")
        out = rescale_image_numba(src, factor)

        if hasattr(doc, "set_image"):
            doc.set_image(out, step_name=f"Rescale ×{factor:.2f}")
        elif hasattr(doc, "apply_numpy"):
            doc.apply_numpy(out, step_name=f"Rescale ×{factor:.2f}")
        else:
            doc.image = out

    def _open_histogram(self):
        sw = self.mdi.activeSubWindow()
        if not sw:
            QMessageBox.information(self, "Histogram", "No active image window.")
            return
        doc = sw.widget().document
        dlg = HistogramDialog(self, doc)
        # Title includes the subwindow name so multiple dialogs are clear
        dlg.setWindowTitle(f"Histogram — {sw.windowTitle()}")
        try:
            dlg.setWindowIcon(QIcon(histogram_path))
        except Exception:
            pass
        dlg.show()
        if hasattr(self, "_log"):
            self._log(f"Opened Histogram for {doc.display_name()}")

    def _open_crop_dialog(self):
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
        sw = self.mdi.activeSubWindow()
        if not sw:
            QMessageBox.information(self, "No image", "Open an image first.")
            return
        doc = sw.widget().document
        dlg = StatisticalStretchDialog(self, doc)
        try:
            dlg.setWindowIcon(QIcon(statstretch_path))
        except Exception:
            pass
        dlg.resize(900, 600)  # comfy preview area
        dlg.show()

    def _open_star_stretch(self):
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

    def _open_abe_tool(self):
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

    def SFCC_show(self):
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

    def show_convo_deconvo(self):
        if getattr(self, "convo_window", None) and self.convo_window.isVisible():
            self.convo_window.raise_()
            self.convo_window.activateWindow()
            return

        # make sure we have a DocManager (same pattern as SFCC_show)
        if not hasattr(self, "doc_manager") or self.doc_manager is None:
            # if you keep an image_manager around, pass it through like SFCC_show does
            self.doc_manager = DocManager(image_manager=getattr(self, "image_manager", None), parent=self)

        from pro.convo import ConvoDeconvoDialog
        self.convo_window = ConvoDeconvoDialog(doc_manager=self.doc_manager, parent=self)
        try:
            self.convo_window.setWindowIcon(QIcon(convoicon_path))
        except Exception:
            pass
        try:
            self.convo_window.destroyed.connect(lambda _=None: setattr(self, "convo_window", None))
        except Exception:
            pass
        self.convo_window.show()

    def _extract_luminance(self, doc=None):
        """
        If doc is None, uses the active subwindow's document.
        Otherwise, run on the provided doc (for drag-and-drop to a specific view).
        Creates a new mono document (float32, [0..1]) and spawns a subwindow.
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

        # 3) Rec.709 linear luma
        # Y' = 0.2126 R + 0.7152 G + 0.0722 B
        y = (a[..., 0] * 0.2126 + a[..., 1] * 0.7152 + a[..., 2] * 0.0722).astype(np.float32, copy=False)
        y = np.clip(y, 0.0, 1.0)

        # 4) metadata & title
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
        }

        # Try to use the existing subwindow title as base; otherwise fall back
        base_title = sw.windowTitle() if sw else (getattr(doc, "title", getattr(doc, "name", "")) or "Untitled")
        title = f"{base_title} — Luminance"

        # 5) create a real document via DocManager
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

        # 6) show subwindow and set the luminance icon
        try:
            self._spawn_subwindow_for(new_doc)
            sub = self.mdi.activeSubWindow()
            if sub:
                sub.setWindowIcon(QIcon(LExtract_path))
        except Exception:
            pass

        if hasattr(self, "_log"):
            self._log("Functions: Extract Luminance → new mono document created.")

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
            # accept mono directly, or RGB (we'll extract Y’)
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
            src_img = _to_float01(np.asarray(src_doc.image))
            # compute L from src (works for mono or RGB)
            L = compute_luminance(src_img)
            apply_recombine_to_doc(target_doc, L)
            try:
                self._log(f"Recombine Luminance: '{sel_title}' → '{target_doc.display_name()}'")
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
            self._log(f"RGB Extract → created '{base}_R', '{base}_G', '{base}_B'")

    def _list_open_docs_for_rgb(self):
        items = []
        for sw in self.mdi.subWindowList():
            w = sw.widget()
            doc = getattr(w, "document", None)
            if doc is not None:
                items.append((sw.windowTitle(), doc))
        return items

    def _open_rgb_combination(self):
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
            # Prefer the Pro dialog name; fall back to the non-Pro name if that’s what you used.
            from pro.wavescalede import WaveScaleDarkEnhancerDialogPro as _Dlg
        except Exception:
            try:
                from pro.wavescalede import WaveScaleDarkEnhanceDialog as _Dlg
            except Exception as e:
                QMessageBox.warning(self, "WaveScale Dark Enhancer", f"Failed to import dialog:\n{e}")
                return

        try:
            dlg = _Dlg(self, doc, icon_path=dse_icon_path)  # matches our Pro dialogs’ __init__(parent, doc, icon_path)
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
            apply_clahe_to_doc(doc, preset or {"clip_limit": 2.0, "tile": 8})
        except Exception as e:
            raise RuntimeError(f"CLAHE apply failed: {e}")

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



    def _open_aberration_ai(self):
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
        sw = self.mdi.activeSubWindow()
        if not sw:
            QMessageBox.information(self, "Cosmic Clarity", "No active image view.")
            return

        w = sw.widget() if hasattr(sw, "widget") else None
        doc = getattr(w, "document", None)
        if doc is None or getattr(doc, "image", None) is None:
            QMessageBox.information(self, "Cosmic Clarity", "Active view has no image.")
            return

        try:
            dlg = CosmicClarityDialogPro(self, doc, icon=QIcon(cosmic_path))
            dlg.exec()
        except Exception as e:
            print(f"Failed to open Cosmic Clarity UI: {e}")

    def _open_cosmic_clarity_satellite(self):
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
        sw = self.mdi.activeSubWindow()
        doc = sw.widget().document if sw else None
        if not doc:
            QMessageBox.information(self, "History", "No active document.")
            return

        dlg = HistoryExplorerDialog(doc, parent=self)
        sub_title = sw.windowTitle() if sw else doc.display_name()
        dlg.setWindowTitle(f"History Explorer — {sub_title}")
        dlg.show()
        self._log("History: opened History Explorer.")

    def _zoom_active_1_1(self):
        sw = self.mdi.activeSubWindow()
        if not sw:
            return
        view = sw.widget()
        if hasattr(view, "_zoom_at_anchor"):
            # Anchor to viewport center for a nice experience
            factor = 1.0 / max(getattr(view, "scale", 1.0), 1e-12)
            # Temporarily move the cursor anchor to the viewport center by faking it:
            # easiest approach: just call set_scale if you prefer; _zoom_at_anchor keeps things stable
            try:
                view._zoom_at_anchor(factor)
                return
            except Exception:
                pass  # fall back below if anything odd happens

        # Fallback: simple set + center content
        if hasattr(view, "set_scale"):
            view.set_scale(1.0)
        try:
            vp = view.scroll.viewport()
            hbar = view.scroll.horizontalScrollBar()
            vbar = view.scroll.verticalScrollBar()
            cx = max(0, view.label.width()  // 2 - vp.width()  // 2)
            cy = max(0, view.label.height() // 2 - vp.height() // 2)
            hbar.setValue(min(hbar.maximum(), cx))
            vbar.setValue(min(vbar.maximum(), cy))
        except Exception:
            pass

    def _infer_image_size(self, view):
        """Return (img_w, img_h) in device-independent pixels (ints), best-effort."""
        # Preferred: from the label's pixmap
        try:
            pm = getattr(view, "label", None).pixmap() if hasattr(view, "label") else None
            if pm and not pm.isNull():
                dpr = max(1.0, float(pm.devicePixelRatio()))
                return int(round(pm.width() / dpr)), int(round(pm.height() / dpr))
        except Exception:
            pass

        # Next: from the document image
        try:
            doc = getattr(view, "document", None)
            if doc and getattr(doc, "image", None) is not None:
                import numpy as np
                h, w = np.asarray(doc.image).shape[:2]
                return int(w), int(h)
        except Exception:
            pass

        # Fallback: from attributes some views keep
        for w_key, h_key in (("image_width","image_height"), ("_img_w","_img_h")):
            w = getattr(view, w_key, None); h = getattr(view, h_key, None)
            if isinstance(w, (int,float)) and isinstance(h, (int,float)) and w>0 and h>0:
                return int(w), int(h)

        return None, None


    def _viewport_widget(self, view):
        """Return the viewport widget used to display the image."""
        try:
            if hasattr(view, "scroll") and hasattr(view.scroll, "viewport"):
                return view.scroll.viewport()
            # Some views are QGraphicsView/QAbstractScrollArea-like
            if hasattr(view, "viewport"):
                return view.viewport()
        except Exception:
            pass
        # Worst case: the view itself
        return view

    def _zoom_active_fit(self):
        sw = self.mdi.activeSubWindow()
        if not sw:
            return
        view = sw.widget()
        self._zoom_active_1_1()
        # Get sizes
        img_w, img_h = self._infer_image_size(view)
        if not img_w or not img_h:
            return

        vp = self._viewport_widget(view)
        vw, vh = max(1, vp.width()), max(1, vp.height())

        # Compute uniform scale (minus a hair to avoid scrollbars fighting)
        scale = min((vw - 2) / img_w, (vh - 2) / img_h)
        # Clamp to sane bounds
        scale = max(1e-4, min(32.0, scale))

        # Apply using view API if available
        if hasattr(view, "set_scale") and callable(view.set_scale):
            try:
                view.set_scale(float(scale))
                self._center_view(view)
                return
            except Exception:
                pass

        # Fallback: relative zoom using _zoom_at_anchor
        try:
            cur = float(getattr(view, "scale", 1.0))
            factor = scale / max(cur, 1e-12)
            if hasattr(view, "_zoom_at_anchor") and callable(view._zoom_at_anchor):
                view._zoom_at_anchor(float(factor))  # most implementations anchor on cursor/center
                self._center_view(view)
                return
        except Exception:
            pass

    def _center_view(self, view):
        """Center the content after a zoom change, if possible."""
        try:
            vp = self._viewport_widget(view)
            hbar = view.scroll.horizontalScrollBar() if hasattr(view, "scroll") else None
            vbar = view.scroll.verticalScrollBar() if hasattr(view, "scroll") else None
            lbl  = getattr(view, "label", None)
            if vp and hbar and vbar and lbl:
                cx = max(0, lbl.width()  // 2 - vp.width()  // 2)
                cy = max(0, lbl.height() // 2 - vp.height() // 2)
                hbar.setValue(min(hbar.maximum(), cx))
                vbar.setValue(min(vbar.maximum(), cy))
        except Exception:
            pass

    def _toggle_autostretch(self, on: bool):
        sw = self.mdi.activeSubWindow()
        if sw:
            sw.widget().set_autostretch(on)
            self._log(f"Display-Stretch {'ON' if on else 'OFF'} → {sw.windowTitle()}")

    def _set_hard_autostretch_from_action(self, checked: bool):
        sw = self.mdi.activeSubWindow()
        if not sw:
            return
        view = sw.widget()

        # mirror the action's check to the view profile
        if hasattr(view, "set_autostretch_profile"):
            view.set_autostretch_profile("hard" if checked else "normal")

        # ensure it's visible
        if not getattr(view, "autostretch_enabled", False):
            view.set_autostretch(True)
            self._sync_autostretch_action(True)

        self._log(f"Display-Stretch profile → {'HARD' if checked else 'NORMAL'}  ({sw.windowTitle()})")

    def _toggle_hard_autostretch(self):
        sw = self.mdi.activeSubWindow()
        if not sw:
            return
        view = sw.widget()
        # flip profile
        new_profile = "hard" if not getattr(view, "is_hard_autostretch", lambda: False)() else "normal"
        if hasattr(view, "set_autostretch_profile"):
            view.set_autostretch_profile(new_profile)
        # ensure autostretch is ON so the change is visible immediately
        if not getattr(view, "autostretch_enabled", False):
            view.set_autostretch(True)
            self._sync_autostretch_action(True)

        # ✅ reflect in toolbar button
        with QSignalBlocker(self.act_hardstretch):
            self.act_hardstretch.setChecked(new_profile == "hard")

        self._log(f"Display-Stretch profile → {new_profile.upper()}  ({sw.windowTitle()})")


    def _sync_autostretch_action(self, on: bool):
        if hasattr(self, "act_autostretch"):
            block = QSignalBlocker(self.act_autostretch)
            self.act_autostretch.setChecked(bool(on))

    def _init_tools_menu(self):
        tools = self.menuBar().addMenu("Tools")
        act_batch = QAction("Batch Convert…", self)
        act_batch.triggered.connect(self._open_batch_convert)
        tools.addAction(act_batch)



    #------------Tools-----------------
    def _open_blink_tool(self):
        dlg = BlinkComparatorPro(doc_manager=self.docman)
        dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        dlg.setWindowTitle("Blink Comparator")
        try:
            dlg.setWindowIcon(QIcon(blink_path))
        except Exception:
            pass        
        dlg.show()

    def _open_ppp_tool(self):
        w = PerfectPalettePicker(doc_manager=self.docman)  # parent gives access to _spawn_subwindow_for
        w.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        w.setWindowTitle("Perfect Palette Picker")
        try:
            w.setWindowIcon(QIcon(ppp_path))
        except Exception:
            pass        
        w.show()   

    def _open_nbtorgb_tool(self):

        w = NBtoRGBStars(doc_manager=self.docman)
        w.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        w.setWindowTitle("NB → RGB Stars")
        try:
            w.setWindowIcon(QIcon(nbtorgb_path))
        except Exception:
            pass
        w.show()

    def _open_freqsep_tool(self):
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
        # if we have a document, preload its image/metadata (like Stat Stretch does)
        if doc is not None and getattr(doc, "image", None) is not None:
            w.set_image_from_doc(doc.image, doc.metadata)
       
        w.show()

    def _open_contsub_tool(self):
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
        w = ImageCombineDialog(self)   # ← only pass self
        w.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        w.setWindowTitle("Image Combination")
        try:
            w.setWindowIcon(QIcon(imagecombine_path))
        except Exception:
            pass        
        w.resize(900, 650)
        w.show()                       # ← modeless; no exec(), no result_preset()


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
                self._log(f"Image Combine → replaced '{_display_name(A)}' ({step})")
            else:
                newdoc = dm.create_document(result, metadata={
                    "display_name": f"Combined ({step})",
                    "bit_depth": "32-bit floating point",
                    "is_mono": (result.ndim == 2),
                    "source": f"Combine: {step}",
                }, name=f"Combined ({step})")
                self._spawn_subwindow_for(newdoc)
                self._log(f"Image Combine → new view '{newdoc.display_name()}' ({step})")

        except Exception as e:
            QMessageBox.critical(self, "Image Combine", f"Failed:\n{e}")

    def _open_psf_viewer(self, preset: dict | None = None):
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
                except Exception: pass
            if preset.get("mode") == "Flux":
                try: dlg.toggleHistogramMode()
                except Exception: pass
            if bool(preset.get("log", False)) != bool(dlg.log_scale):
                try: dlg.log_toggle_button.setChecked(bool(preset.get("log", False)))
                except Exception: pass
            if "zoom" in preset:
                try: dlg.zoom_slider.setValue(int(preset["zoom"]))
                except Exception: pass

        dlg.show()

    def _open_image_peeker(self):
        sw = self.mdi.activeSubWindow()
        if not sw:
            QMessageBox.information(self, "Image Peaker", "No active image window.")
            return

        dlg = ImagePeekerDialogPro(parent=self, document=sw, settings=self.settings)
        dlg.setWindowIcon(QIcon(peeker_icon))
        dlg.show()

    def _open_image_peeker_for_doc(self, doc, title_hint=None):

        dlg = ImagePeekerDialogPro(parent=self, document=doc, settings=self.settings)
        try: dlg.setWindowIcon(QIcon(peeker_icon))
        except Exception: pass
        dlg.show()
        if hasattr(self, "_log"):
            self._log(f"Opened Image Peeker for '{title_hint or getattr(doc, 'display_name', lambda:'view')()}'")


    def _open_plate_solver(self):
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
        dlg = StellarAlignmentDialog(
            parent=self,
            settings=self.settings,
            doc_manager=self.doc_manager,            # ← so Apply/New use undo/redo + creation
            list_open_docs_fn=self._list_open_docs   # ← same helper used by RGB dialog
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

    def _open_mosaic_master(self):
        dlg = MosaicMasterDialog(
            settings=self.settings,
            parent=self,
            image_manager=getattr(self, "image_manager", None),
            doc_manager=getattr(self, "doc_manager", None),
            wrench_path=wrench_path,
            spinner_path=spinner_path,
            list_open_docs_fn=getattr(self, "_list_open_docs", None),  # ← add this
        )
        dlg.setWindowFlag(Qt.WindowType.Window, True)
        dlg.setWindowIcon(QIcon(mosaic_path))
        dlg.show()

    def _open_live_stacking(self):
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
                # If it still exists, make sure it's visible
                if not dlg.isVisible():
                    dlg.show()
                dlg.raise_()
                dlg.activateWindow()
                return
            except RuntimeError:
                # Wrapped C++ object was deleted; clear the stale ref and fall through to recreate
                self._stacking_suite = None

        # Create a new one
        dlg = StackingSuiteDialog(
            parent=self,
            wrench_path=wrench_path,
            spinner_path=spinner_path,
        )
        # Ensure proper lifecycle: clear our ref when it closes
        dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        dlg.destroyed.connect(lambda _=None: setattr(self, "_stacking_suite", None))
        # (Optional and also fine)
        # dlg.finished.connect(lambda _code: setattr(self, "_stacking_suite", None))

        dlg.setWindowFlag(Qt.WindowType.Window, True)
        dlg.setWindowIcon(QIcon(stacking_path))
        dlg.show()

        self._stacking_suite = dlg

    def _on_stackingsuite_relaunch(self, old_dir: str, new_dir: str):
        # Optional: respond to dialog’s relaunch request
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
            dlg.setWindowTitle(f"Diffraction Spikes — {title_hint}")
        dlg.setWindowFlag(Qt.WindowType.Window, True)
        dlg.setWindowIcon(QIcon(starspike_path))
        dlg.show()

    def _open_exo_detector(self):
        dlg = ExoPlanetWindow(
            parent=self,
            wrench_path=wrench_path,
        )
        dlg.setWindowFlag(Qt.WindowType.Window, True)
        dlg.setWindowIcon(QIcon(exoicon_path))

        # Optional WIMI wiring (safe guards so we don’t assume exact API names)
        if hasattr(self, "wimi_tab"):
            # WIMI → dialog: emit RA/Dec once solved
            if hasattr(self.wimi_tab, "wcsCoordinatesAvailable"):
                self.wimi_tab.wcsCoordinatesAvailable.connect(dlg.receive_wcs_coordinates)
            # dialog → WIMI: send a reference FITS path to solve
            # (rename this to whatever your WIMI expects; kept guarded)
            if hasattr(self.wimi_tab, "open_reference_path"):
                dlg.referenceSelected.connect(self.wimi_tab.open_reference_path)

        dlg.show()

    def _open_isophote(self):
        # Mirror PSF opener: use MDI active subwindow → widget → document → image
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
            title_hint="GLIMR — Isophote Modeler",
            doc_manager=dm,
        )
        dlg.setWindowFlag(Qt.WindowType.Window, True)
        dlg.setWindowIcon(QIcon(isophote_path))
        dlg.show()

    def _open_whats_in_my_sky(self):
        dlg = WhatsInMySkyDialog(
            parent=self,
            wims_path=wims_path,          # window icon
            wrench_path=wrench_path       # optional settings icon
        )
        dlg.setWindowFlag(Qt.WindowType.Window, True)
        dlg.setWindowIcon(QIcon(wims_path))
        dlg.show()

    def _open_wimi(self):
        
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

    def _open_batch_renamer(self):
        dlg = BatchRenamerDialog(parent=self)
        dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        dlg.show()

    def _open_astrobin_exporter(self):
        # you said this is defined in the parent main UI:
        # astrobin_filters_csv_path = os.path.join(sys._MEIPASS, 'astrobin_filters.csv')

        dlg = AstrobinExporterDialog(self, offline_filters_csv=astrobin_filters_csv_path)
        dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        dlg.show()

    def _open_batch_convert(self):
        dlg = BatchConvertDialog(self)
        dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        dlg.show()

    def _open_copy_astrometry(self):
        sw = self.mdi.activeSubWindow()
        if not sw:
            QMessageBox.information(self, "Copy Astrometric Solution", "No active image window.")
            return

        dlg = CopyAstrometryDialog(parent=self, target=sw)
        dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        dlg.show()

    def _open_linear_fit(self):
        sw = self.mdi.activeSubWindow()
        if not sw:
            QMessageBox.information(self, "No image", "Open an image first.")
            return
        try:
            active_doc = sw.widget().document
        except Exception:
            QMessageBox.information(self, "Linear Fit", "No active image.")
            return

        dm = getattr(self, "doc_manager", None) or getattr(self, "docman", None)
        if dm is None:
            QMessageBox.information(self, "Linear Fit", "No document manager available.")
            return

        dlg = LinearFitDialog(self, dm, active_doc)  # <-- pass doc manager
        try:
            dlg.setWindowIcon(QIcon(self._icon_path("linear_fit")))
        except Exception:
            pass
        dlg.resize(900, 600)
        dlg.show()

    def _open_debayer(self):
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
        dlg = AboutDialog(self)
        dlg.show()

    #######-------COMMAND DROPS-------#################

    # --- Command drop handling ------------------------------------------------
    # --- Command drop handling ------------------------------------------------
    def _handle_command_drop(self, payload: dict, target_sw):
        cid = (payload or {}).get("command_id") or ""
        preset = (payload or {}).get("preset") or {}

        # ---------- helpers ----------
        def _cid_norm(c: str) -> str:
            aliases = {
                # geometry short ↔ long ids
                "flip_horizontal": "geom_flip_horizontal",
                "geom_flip_h": "geom_flip_horizontal",
                "geom_flip_horizontal": "geom_flip_horizontal",

                "flip_vertical": "geom_flip_vertical",
                "geom_flip_v": "geom_flip_vertical",
                "geom_flip_vertical": "geom_flip_vertical",

                "rotate_clockwise": "geom_rotate_clockwise",
                "geom_rot_cw": "geom_rotate_clockwise",
                "geom_rotate_clockwise": "geom_rotate_clockwise",

                "rotate_counterclockwise": "geom_rotate_counterclockwise",
                "geom_rot_ccw": "geom_rotate_counterclockwise",
                "geom_rotate_counterclockwise": "geom_rotate_counterclockwise",

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

        def _call_any(method_names: list[str], *args, **kwargs) -> bool:
            for name in method_names:
                m = getattr(self, name, None)
                if callable(m):
                    m(*args, **kwargs)
                    return True
            return False

        cid = _cid_norm(cid)

        # ----- Function bundle: run a sequence of steps on the target view(s) -----
        if cid in ("function_bundle", "bundle_functions"):
            steps   = list((payload or {}).get("steps") or [])
            inherit = bool((payload or {}).get("inherit_target", True))  # NEW: default True

            # If user dropped onto the background (no direct target), support 'targets'
            if target_sw is None:
                targets = (payload or {}).get("targets")

                # Apply to all open subwindows
                if targets == "all_open":
                    for sw in list(self.mdi.subWindowList()):
                        for st in steps:
                            self._handle_command_drop(st, target_sw=sw)
                    return

                # Apply to explicit list of doc_ptrs
                if isinstance(targets, (list, tuple)):
                    for ptr in targets:
                        try:
                            doc, sw = self._find_doc_by_id(int(ptr))
                        except Exception:
                            sw = None
                        if sw:
                            for st in steps:
                                self._handle_command_drop(st, target_sw=sw)
                    return

                # No target info → open Function Bundles UI
                try:
                    from pro.function_bundle import show_function_bundles
                    show_function_bundles(self)
                except Exception:
                    pass
                return

            # We DO have an explicit target subwindow → run the sequence there.
            # If inherit=True (default), forward the SAME target to each child step.
            for st in steps:
                if not isinstance(st, dict) or not st.get("command_id"):
                    continue
                self._handle_command_drop(st, target_sw=target_sw if inherit else None)
            return

        # --- Bundle runner -----------------------------------------------------------
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
            try: self._log(f"Bundle: {len(steps)} step(s) → {targets or 'target view'}")
            except Exception: pass


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
                        try: self._log(f"Bundle [{i}/{len(steps)}] on '{title}' → {sp.get('command_id')}")
                        except Exception: pass
                    except Exception as e:
                        try: self._log(f"Bundle step failed on '{title}': {e}")
                        except Exception: pass
                        if stop_on_error:
                            QMessageBox.warning(self, "Bundle", f"Stopped on error:\n{e}")
                            return
                        # else continue to next step

            try: self._log("Bundle complete.")
            except Exception: pass
            return

        # ------------------- No target subwindow → open UIs / active ops -------------------
        if target_sw is None:
            if cid == "stat_stretch":
                self._open_statistical_stretch_with_preset(preset); return
            if cid == "star_stretch":
                self._open_star_stretch_with_preset(preset); return
            if cid == "remove_green":
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
                    
                    open_ghs_with_preset(self, preset)
                    return

            # Fallback: trigger QAction by cid (ok when no target)
            act = self._find_action_by_cid(cid)
            if act:
                act.trigger()
            return

        # ------------------- Dropped on a specific subwindow → HEADLESS APPLY -------------------
        view = target_sw.widget()
        doc = getattr(view, "document", None)

        # Early out if no doc
        if doc is None:
            return

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
            run_graxpert_via_preset(self, preset or {})
            s_raw = (preset or {}).get("smoothing", 0.10)
            try:
                s_val = float(s_raw)
            except Exception:
                s_val = 0.10
            gpu_val = bool((preset or {}).get("gpu", True))

            self._log(f"Ran GraXpert (smoothing={round(s_val, 2)}, gpu={'on' if gpu_val else 'off'})")
            return

        if cid == "convo":
            try:
                from pro.convo_preset import apply_convo_via_preset
                apply_convo_via_preset(self, doc, preset or {})
                op = (preset or {}).get("op", "convolution")
                self._log(f"Applied Convo/Deconvo preset ({op}) to '{target_sw.windowTitle()}'")
            except Exception as e:
                
                QMessageBox.warning(self, "Convo/Deconvo", f"Apply failed:\n{e}")
            return

        if cid == "remove_stars":
            from pro.remove_stars_preset import run_remove_stars_via_preset
            run_remove_stars_via_preset(self, preset or {})
            # safe logging (no nested format specs)
            tool = str((preset or {}).get("tool","starnet"))
            if tool.lower().startswith("star"):
                lin = bool((preset or {}).get("linear", True))
                self._log(f"Ran Remove Stars (tool=StarNet, linear={'yes' if lin else 'no'})")
            else:
                mode = (preset or {}).get("mode","unscreen")
                stride = int((preset or {}).get("stride", 512))
                gpu = not bool((preset or {}).get("disable_gpu", False))
                show = bool((preset or {}).get("show_extracted_stars", True))
                self._log(f"Ran Remove Stars (tool=DarkStar, mode={mode}, stride={stride}, gpu={'on' if gpu else 'off'}, stars={'on' if show else 'off'})")
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
            apply_remove_green_preset_to_doc(self, doc, preset)
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

        if cid == "background_neutral":
            try:
                self._apply_background_neutral_preset_to_doc(doc, preset)
                self._log(f"Background Neutralization applied to '{target_sw.windowTitle()}'")
            except Exception as e:
                QMessageBox.warning(self, "Background Neutralization", str(e))
            return

        if cid == "white_balance":
            try:
                self._apply_white_balance_preset_to_doc(doc, preset)
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
                self._log(f"Extract Luminance → new doc from '{target_sw.windowTitle()}'")
            except Exception as e:
                QMessageBox.warning(self, "Extract Luminance", str(e))
            return

        if cid == "recombine_luminance":
            self._recombine_luminance_ui(target_doc=doc)
            self._log(f"Recombined Luminance → doc '{target_sw.windowTitle()}'")
            return

        if cid == "rgb_extract":
            self._rgb_extract_on_doc(doc, base_title=target_sw.windowTitle())
            self._log(f"Extracted R, G, and B channels from '{target_sw.windowTitle()}'")
            return

        if cid == "blemish_blaster":
            dlg = BlemishBlasterDialogPro(self, doc)
            try: dlg.setWindowIcon(QIcon(blastericon_path))
            except Exception: pass
            dlg.resize(900, 650)
            dlg.show()
            return



        if cid == "wavescale_hdr":
            # (unchanged block)
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
                from pro.morphology import apply_morphology_to_doc
                apply_morphology_to_doc(doc, preset)
                self._log(f"Morphology applied to '{target_sw.windowTitle()}'")
            except Exception as e:
                QMessageBox.warning(self, "Morphology", str(e))
            return

        if cid == "pixel_math":
            try:
                from pro.pixelmath import apply_pixel_math_to_doc
                apply_pixel_math_to_doc(self, doc, preset)
                desc = preset.get("expr", None) or f"R:{preset.get('expr_r','')} G:{preset.get('expr_g','')} B:{preset.get('expr_b','')}"
                self._log(f"Pixel Math applied to '{target_sw.windowTitle()}' → {desc}")
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
                self._log(f"Signature preset applied to '{target_sw.windowTitle()}' (file={fp}, pos={pos}, scale={preset.get('scale',100)}%, rot={preset.get('rotation',0)}°, op={preset.get('opacity',100)}%)")
            except Exception as e:
                QMessageBox.warning(self, "Signature / Insert", f"Preset apply failed:\n{e}")
            return

        if cid == "halo_b_gon":
            try:
                from pro.halobgon import apply_halo_b_gon_to_doc
                apply_halo_b_gon_to_doc(self, doc, preset)
                lvl = int((preset or {}).get("reduction", 0))
                lin = bool((preset or {}).get("linear", False))
                self._log(f"Halo-B-Gon applied to '{target_sw.windowTitle()}' (level={lvl}, linear={lin})")
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
                self._log(f"Rotate 90° CW applied to '{target_sw.windowTitle()}'")
            except Exception as e:
                QMessageBox.warning(self, "Rotate 90° CW", str(e))
            return

        if cid == "geom_rotate_counterclockwise":
            try:
                called = _call_any(["_apply_geom_rot_ccw_to_doc", "_apply_geom_rotate_ccw_to_doc"], doc)
                if not called:
                    raise RuntimeError("No rotate-ccw apply method found")
                self._log(f"Rotate 90° CCW applied to '{target_sw.windowTitle()}'")
            except Exception as e:
                QMessageBox.warning(self, "Rotate 90° CCW", str(e))
            return

        if cid == "geom_rescale":
            try:
                factor = float(preset.get("factor", 1.0))
                # support both names you’ve used
                called = _call_any(
                    ["_apply_rescale_preset_to_doc", "_apply_geom_rescale_to_doc"],
                    doc, {"factor": factor} if "_apply_rescale_preset_to_doc" in dir(self) else factor
                )
                if not called:
                    # last resort: try signature (doc, preset)
                    _call_any(["_apply_rescale_preset_to_doc"], doc, {"factor": factor})
                self._log(f"Rescale ×{factor:g} applied to '{target_sw.windowTitle()}'")
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
            # Dropped on a specific subwindow → treat that doc as A, resolve B from preset/heuristics
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
                    "ASTAP executable not set. Go to Preferences → ASTAP executable."
                )
                return

            try:                
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
                        scale = (a*a + b*b) ** 0.5 * 3600.0  # ″/px

                    msg = "Plate solve complete"
                    if ra is not None and dec is not None:
                        msg += f" | RA={ra:.6f}°, Dec={dec:.6f}°"
                    if scale is not None:
                        msg += f" | scale≈{scale:.3f}\"/px"

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
            # (We map DnD → QAction via reg("star_align", self.act_star_align) already,
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

    def _handle_mask_drop(self, payload: dict, target_sw):
        if target_sw is None:
            # applying a mask requires a target view
            return
        mask_ptr = payload.get("mask_doc_ptr")
        if mask_ptr is None:
            return

        mask_doc, _ = self._find_doc_by_id(mask_ptr)
        if mask_doc is None:
            QMessageBox.warning(self, "Mask", "Could not resolve mask document.")
            return

        target_view = target_sw.widget()
        target_doc  = getattr(target_view, "document", None)
        if target_doc is None:
            return

        mode    = str(payload.get("mode", "replace"))
        invert  = bool(payload.get("invert", False))
        feather = float(payload.get("feather", 0.0))
        name    = payload.get("name") or mask_doc.display_name() or "Mask"

        if self._attach_mask_to_document(target_doc, mask_doc, name=name, mode=mode, invert=invert, feather=feather):
            if hasattr(self, "_log"):
                self._log(f"Mask '{name}' applied to '{target_sw.windowTitle()}'")

            # ✅ Make the drop target the active subwindow immediately
            def _activate():
                try:
                    self.mdi.setActiveSubWindow(target_sw)
                    target_sw.activateWindow()
                    target_sw.raise_()
                    target_sw.widget().setFocus(Qt.FocusReason.MouseFocusReason)
                except Exception:
                    pass
                self._refresh_mask_action_states()

            _activate()
            # Also schedule once for the next tick to be extra robust after the drop event
            QTimer.singleShot(0, _activate)


    def _find_action_by_cid(self, cid: str) -> QAction | None:
        if not cid:
            return None
        # We registered command ids on actions via register_action(...)
        for a in self.findChildren(QAction):
            if a.property("command_id") == cid or a.objectName() == cid:
                return a
        return None

    def _on_astrometry_drop(self, payload: dict, target_sw):
        # Resolve source doc from ptr
        src_ptr = int(payload.get("wcs_from_doc_ptr", 0))
        src = self._doc_by_ptr(src_ptr)
        if src is None:
            QMessageBox.information(self, "Copy Astrometry", "Source view not found.")
            return

        # Resolve target doc from the QMdiSubWindow we got
        if target_sw is None or not hasattr(target_sw, "widget"):
            QMessageBox.information(self, "Copy Astrometry", "Drop onto a view to copy its solution.")
            return
        target_view = target_sw.widget()
        tgt = getattr(target_view, "document", None)
        if tgt is None:
            QMessageBox.information(self, "Copy Astrometry", "Target view has no document.")
            return

        # Extract + apply
        wcs = self._extract_wcs_dict(src)
        if not wcs:
            QMessageBox.information(self, "Copy Astrometry", "No astrometric solution found in the source view.")
            return

        ok = self._apply_wcs_dict_to_doc(tgt, wcs)
        if not ok:
            QMessageBox.warning(self, "Copy Astrometry", "Failed to apply astrometric solution to target.")
            return

        # Make the drop target active so the header dock tracks it,
        # and force-refresh the header viewer.
        def _activate_and_refresh():
            try:
                self.mdi.setActiveSubWindow(target_sw)
                target_sw.activateWindow()
                target_sw.raise_()
                target_sw.widget().setFocus(Qt.FocusReason.MouseFocusReason)
            except Exception:
                pass
            try:
                # If your header dock attaches to the active doc via signals,
                # this explicit refresh guarantees the UI updates immediately.
                self._hdr_refresh_timer.start(0)
            except Exception:
                pass
            try:
                # If other parts of the UI listen to this signal
                self.currentDocumentChanged.emit(tgt)
            except Exception:
                pass

        _activate_and_refresh()
        QTimer.singleShot(0, _activate_and_refresh)  # run again next tick (post-drop)

        # Optional: push undo step if your doc manager supports it
        dm = getattr(self, "doc_manager", None) or getattr(self, "docman", None)
        push = getattr(dm, "push_history", None) or getattr(dm, "add_history", None)
        if callable(push):
            try:
                src_name = getattr(src, "display_name", lambda: None)() or "Source"
                push(tgt, f"Copy Astrometric Solution (from {src_name})")
            except Exception:
                pass

        # Log
        try:
            self._log(f"Copied astrometric solution from '{src.display_name()}' to '{tgt.display_name()}'.")
        except Exception:
            pass

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
        Headless apply of Statistical Stretch using the dialog’s own apply path.
        We instantiate the dialog, set controls from the preset, and call its Apply.
        """
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
            if "apply_curves" in preset:
                dlg.chk_curves.setChecked(bool(preset["apply_curves"]))
            if "curves_boost" in preset:
                dlg.sld_curves.setValue(int(round(float(preset["curves_boost"]) * 100)))
        except Exception:
            pass

        # directly run the dialog's apply slot (reuses your edit/undo naming, etc.)
        dlg._do_apply()
        try:
            dlg.close()
        except Exception:
            pass

    def _open_star_stretch_with_preset(self, preset: dict):
        """Background drop → open Star Stretch dialog with controls preloaded."""
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
            except Exception: pass

        sat = preset.get("color_boost", preset.get("saturation", None))
        if sat is not None:
            try: dlg.sld_sat.setValue(int(float(sat) * 100.0))
            except Exception: pass

        scnr = preset.get("scnr_green", preset.get("scnr", None))
        if scnr is not None:
            try: dlg.chk_scnr.setChecked(bool(scnr))
            except Exception: pass

        dlg.resize(1000, 650)
        dlg.show()
        if hasattr(self, "_log"):
            self._log("Star Stretch: opened dialog with preset.")

    def _apply_star_stretch_preset_to_doc(self, doc, preset: dict):
        """
        Drop on a specific subwindow → apply Star Stretch headlessly to that doc.
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

        # mono → 3ch temporarily
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


    # --- Command Search (palette) -----------------------------------------------
    def _strip_menu_text(self, s: str) -> str:
        return s.replace("&", "").replace("…", "").strip()

    def _iter_menu_actions(self, menu: QMenu):
        for act in menu.actions():
            if act.isSeparator():
                continue
            if act.menu() is not None:
                yield from self._iter_menu_actions(act.menu())
            else:
                yield act

    # --- Command palette helpers (safe: no receivers/emit on Qt signals) ---

    def _iter_menu_actions(self, menu: QMenu):
        """Depth-first iterator over all actions inside a QMenu tree."""
        for act in menu.actions():
            yield act
            sub = act.menu()
            if sub is not None:
                yield from self._iter_menu_actions(sub)

    def _is_action_commandy(self, act: QAction) -> bool:
        """Heuristic: action is a real, user-triggerable command."""
        if act is None:
            return False
        if act.isSeparator():
            return False
        # Exclude submenu containers — we want leaf commands only
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
                # Top-level action without a submenu (rare) — include if commandy
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
            except Exception: pass
            tb.deleteLater()
            self._search_tb = None

        old_dock = getattr(self, "_search_dock", None)
        if old_dock:
            try: old_dock.hide(); old_dock.setParent(None)
            except Exception: pass
            old_dock.deleteLater()
            self._search_dock = None

        # --- Right-side mini dock with the search box ---
        self._search_dock = QDockWidget("Command Search", self)
        self._search_dock.setObjectName("CommandSearchDock")
        # ✅ Allow moving/closing like other panels
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
        self._cmd_edit.setPlaceholderText("Search commands…  (Ctrl+Shift+P)")
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
        try:
            self._populate_view_panels_menu()
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

    # --- Actions ---------------------------------------------------------
    def open_files(self):
        # One-stop “All Supported” plus focused groups the user can switch to
        filters = (
            "All Supported (*.png *.jpg *.jpeg *.tif *.tiff "
            "*.fits *.fit *.fits.gz *.fit.gz *.fz *.xisf "
            "*.cr2 *.cr3 *.nef *.arw *.dng *.orf *.rw2 *.pef);;"
            "Astro (FITS/XISF) (*.xisf *.fits *.fit *.fits.gz *.fit.gz *.fz);;"
            "RAW Images (*.cr2 *.cr3 *.nef *.arw *.dng *.orf *.rw2 *.pef);;"
            "Common Images (*.png *.jpg *.jpeg *.tif *.tiff);;"
            "All Files (*)"
        )

        # read last dir; validate it still exists
        last_dir = self.settings.value("paths/last_open_dir", "", type=str) or ""
        if last_dir and not os.path.isdir(last_dir):
            last_dir = ""

        paths, _ = QFileDialog.getOpenFileNames(self, "Open Images", last_dir, filters)
        if not paths:
            return

        # store the directory of the first picked file
        try:
            self.settings.setValue("paths/last_open_dir", os.path.dirname(paths[0]))
        except Exception:
            pass

        # open each path (doc_manager should emit documentAdded; no manual spawn)
        for p in paths:
            try:
                doc = self.docman.open_path(p)   # this emits documentAdded
                # self._spawn_subwindow_for(doc)  # ← keep disabled; handled by signal
                self._log(f"Opened: {p}")
            except Exception as e:
                QMessageBox.warning(self, "Open failed", f"{p}\n\n{e}")


    def save_active(self):
        doc = self._active_doc()
        if not doc:
            return

        filters = (
            "FITS (*.fits *.fit);;"
            "XISF (*.xisf);;"
            "TIFF (*.tif *.tiff);;"
            "PNG (*.png);;"
            "JPEG (*.jpg *.jpeg)"
        )

        # --- Determine initial directory nicely -----------------------------
        # 1) Try the document's original file path (strip any "::HDU" or "::XISF[...]" suffix)
        orig_path = (doc.metadata or {}).get("file_path", "") or ""
        if "::" in orig_path:
            # e.g. "/foo/bar/file.fits::HDU 2" or "...::XISF[3]"
            orig_path_fs = orig_path.split("::", 1)[0]
        else:
            orig_path_fs = orig_path

        candidate_dir = ""
        try:
            if orig_path_fs:
                pdir = os.path.dirname(orig_path_fs)
                if pdir and os.path.isdir(pdir):
                    candidate_dir = pdir
        except Exception:
            candidate_dir = ""

        # 2) Else, fall back to last save dir setting
        if not candidate_dir:
            candidate_dir = self.settings.value("paths/last_save_dir", "", type=str) or ""

        # 3) Else, home directory
        if not candidate_dir or not os.path.isdir(candidate_dir):
            from pathlib import Path
            candidate_dir = str(Path.home())

        # --- Suggest a sane filename ---------------------------------------
        suggested = _best_doc_name(doc)
        suggested = os.path.splitext(suggested)[0]               # remove any ext
        suggested_safe = _sanitize_filename(suggested)
        suggested_path = os.path.join(candidate_dir, suggested_safe)

        # --- Open dialog ----------------------------------------------------
        path, selected_filter = QFileDialog.getSaveFileName(self, "Save As", suggested_path, filters)
        if not path:
            return

        before = path
        path, ext_norm = _normalize_save_path_chosen_filter(path, selected_filter)

        # If we changed the path (e.g., sanitized), inform once
        if before != path:
            self._log(f"Adjusted filename for safety:\n  {before}\n→ {path}")

        # --- Bit depth selection -------------------------------------------
        from pro.save_options import SaveOptionsDialog
        current_bd = doc.metadata.get("bit_depth")
        dlg = SaveOptionsDialog(self, ext_norm, current_bd)
        if dlg.exec() != dlg.DialogCode.Accepted:
            return
        chosen_bd = dlg.selected_bit_depth()

        # --- Save & remember folder ----------------------------------------
        try:
            self.docman.save_document(doc, path, bit_depth_override=chosen_bd)
            self._log(f"Saved: {path} ({chosen_bd})")
            self.settings.setValue("paths/last_save_dir", os.path.dirname(path))
        except Exception as e:
            QMessageBox.critical(self, "Save failed", str(e))




    def _init_layers_dock(self):
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
        self.header_viewer = HeaderViewerDock(self)
        # right side feels natural for metadata
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.header_viewer)
        # keep it in sync with selection
        self.currentDocumentChanged.connect(self.header_viewer.set_document)

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
        """Remember last and current active subwindow so we can ‘bounce’."""
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

    def _spawn_subwindow_for(self, doc):
        existing = self._find_subwindow_for_doc(doc)
        if existing:
            self.mdi.setActiveSubWindow(existing)
            return existing

        # --- choose the right view type (keep your detection) ---
        try:
            from pro.subwindow import ImageSubWindow, TableSubWindow
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            QMessageBox.critical(self, "View Import Error",
                                f"Failed to import view classes from pro.subwindow:\n{e}\n\n{tb}")
            # limp to a label so *something* shows:
            from PyQt6.QtWidgets import QLabel, QMdiSubWindow
            w = QLabel(doc.display_name()); setattr(w, "document", doc)
            wrapper = self.mdi.addSubWindow(w); wrapper.show()
            return wrapper

        # your table detector
        is_table = (
            (getattr(doc, "metadata", {}) or {}).get("doc_type") == "table" or
            hasattr(doc, "rows") and hasattr(doc, "headers")
        )

        # ---- construct the view; LOG ALL ERRORS ----
        try:
            view = TableSubWindow(doc) if is_table else ImageSubWindow(doc)
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            self._log(f"[spawn] View construction failed: {e}\n{tb}")
            QMessageBox.critical(self, "View Construction Error",
                                f"Could not create {'Table' if is_table else 'Image'} view:\n{e}\n\n{tb}")
            # safe fallback:
            from PyQt6.QtWidgets import QLabel
            view = QLabel(doc.display_name()); setattr(view, "document", doc)

        sw = self.mdi.addSubWindow(view)
        sw.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        sw.setWindowTitle(doc.display_name())
        sw.setWindowIcon(self.app_icon)
        sw.show()

        # Window flags (apply then show)
        flags = sw.windowFlags()
        flags |= Qt.WindowType.WindowCloseButtonHint
        flags |= Qt.WindowType.WindowMinimizeButtonHint
        flags |= Qt.WindowType.WindowMaximizeButtonHint
        sw.setWindowFlags(flags)
        sw.show()

        # Intercept minimize → Shelf (if you have one)
        if hasattr(self, "_minimize_interceptor"):
            sw.installEventFilter(self._minimize_interceptor)
        try:
            sw.destroyed.connect(lambda _=None, s=sw:
                hasattr(self, "window_shelf") and self.window_shelf.remove_for_subwindow(s))
        except Exception:
            pass

        # Close bookkeeping
        if hasattr(view, "aboutToClose"):
            try:
                view.aboutToClose.connect(lambda d=doc: self._safe_close_doc(d))
                view.aboutToClose.connect(lambda d=doc: self._log(f"Closed: {d.display_name()}"))
            except Exception:
                pass
        else:
            try:
                sw.destroyed.connect(lambda _=None, d=doc: self._safe_close_doc(d))
            except Exception:
                pass

        # Doc change → undo/redo labels
        try:
            doc.changed.connect(self.update_undo_redo_action_labels)
        except Exception:
            pass

        # Size/scale
        if is_table:
            sw.resize(1000, 700)
        else:
            # try to size to content
            w = h = None
            try:
                img = getattr(doc, "image", None)
                if isinstance(img, np.ndarray) and img.size:
                    h, w = img.shape[:2]
            except Exception:
                pass
            if w and h and w <= 900 and h <= 700 and hasattr(view, "set_scale"):
                view.set_scale(1.0)
                from PyQt6.QtCore import QTimer
                QTimer.singleShot(0, sw.adjustSize)
            else:
                sw.resize(900, 700)
                if w and h and hasattr(view, "set_scale"):
                    view.set_scale(min(900 / float(w), 700 / float(h)))

        # Keep toolbar toggle in sync if this is an image view
        if hasattr(view, "autostretch_enabled"):
            try:
                self._sync_autostretch_action(view.autostretch_enabled)
            except Exception:
                pass

        if hasattr(view, "requestDuplicate"):
            try:
                view.requestDuplicate.connect(self._duplicate_view_from_signal)
            except Exception:
                pass

        if not is_table and getattr(doc, "image", None) is None:
            try:
                self._log(f"[spawn] No image and not recognized as table; metadata keys: {list((getattr(doc,'metadata',{}) or {}).keys())}")
            except Exception:
                pass

        self._on_subwindow_activated(sw)
        return sw


    def _connect_view_signals(self, view):
        # Make sure all views we create are wired to our handlers
        view.aboutToClose.connect(self._on_view_about_to_close)
        view.requestDuplicate.connect(self._duplicate_view_from_signal)

    def _on_view_about_to_close(self, doc):
        """
        Called by ImageSubWindow.closeEvent() before teardown.
        Only remove the document from DocManager if NO OTHER subwindow
        is still showing the same doc.
        """
        sender_view = self.sender()
        # Are there any other subwindows still showing this doc?
        still_open = [
            sw for sw in self.mdi.subWindowList()
            if getattr(sw.widget(), "document", None) is doc and sw.widget() is not sender_view
        ]
        if not still_open:
            # This will emit documentRemoved → Explorer auto-updates
            self.docman.close_document(doc)
            if hasattr(self, "_log"):
                self._log(f"Closed: {doc.display_name()}")
        QTimer.singleShot(0, self._maybe_clear_ui_after_close)        

    def _maybe_clear_ui_after_close(self):
        # If no subwindows remain, clear all “active doc” UI bits, including header
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
        doc = getattr(source_view, "document", None)
        if not doc or doc.image is None:
            return

        # capture source view state
        hbar = source_view.scroll.horizontalScrollBar()
        vbar = source_view.scroll.verticalScrollBar()
        state = {
            "scale": float(getattr(source_view, "scale", 1.0)),
            "hval": int(hbar.value()), "vval": int(vbar.value()),
            "autostretch": bool(getattr(source_view, "autostretch_enabled", False)),
            "autostretch_target": float(getattr(source_view, "autostretch_target", 0.25)),
        }

        base = doc.display_name() or "Untitled"
        # strip any UI decorations that might have leaked into a name string
        while len(base) >= 2 and base[1] == " " and base[0] in "■●◆▲▪▫•◼◻◾◽":
            base = base[2:]
        if base.startswith("Active View: "):
            base = base[len("Active View: "):]

        # 1) duplicate the document
        new_doc = self.docman.duplicate_document(doc, new_name=f"{base}_duplicate")

        # --- IMPORTANT: ensure the duplicate doc starts mask-free ---
        try:
            mid = getattr(new_doc, "active_mask_id", None)
            if mid and hasattr(new_doc, "remove_mask"):
                new_doc.remove_mask(mid)
            # if duplicate carried other masks, nuke them
            if hasattr(new_doc, "masks") and getattr(new_doc, "masks", None):
                try:
                    new_doc.masks.clear()
                except Exception:
                    new_doc.masks = {}
            new_doc.active_mask_id = None
            # announce the clean state
            if hasattr(new_doc, "changed"):
                new_doc.changed.emit()
        except Exception:
            # best-effort fallback
            try:
                new_doc.active_mask_id = None
                if hasattr(new_doc, "masks"):
                    new_doc.masks = {}
                if hasattr(new_doc, "changed"):
                    new_doc.changed.emit()
            except Exception:
                pass
        # ------------------------------------------------------------------

        # 2) after the subwindow is created by our documentAdded handler, apply view state
        def _apply_when_ready():
            sw = self._find_subwindow_for_doc(new_doc)
            if not sw:
                QTimer.singleShot(0, _apply_when_ready)
                return
            view = sw.widget()

            # If the view was constructed before we cleared the doc, forcibly reset the dot.
            if hasattr(view, "_mask_dot_enabled") and view._mask_dot_enabled:
                view._mask_dot_enabled = False
                # rebuild the title from flags + base (no dot)
                try:
                    view._rebuild_title()
                except Exception:
                    # last resort: hard set clean title
                    sub = sw
                    t = getattr(view, "base_doc_title", lambda: new_doc.display_name())()
                    sub.setWindowTitle(t)
                    sub.setToolTip(t)

            # now restore zoom/pan/etc
            self._apply_view_state_to_view(view, state)

            self.mdi.setActiveSubWindow(sw)
            if hasattr(self, "_log"):
                self._log(f"View duplicated as independent doc '{new_doc.display_name()}'")

        QTimer.singleShot(0, _apply_when_ready)




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
        """Drop handler from MdiArea:
        - target_sw is None → duplicate view on background
        - target_sw is a QMdiSubWindow → copy zoom/pan/stretch to it
        """
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
                self._log(f"Duplicated as independent document → '{new_doc.display_name()}'")

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
            self._log(f"Copied view state ➜ '{target_sw.windowTitle()}'")



    def _add_doc_to_explorer(self, doc):
        # de-dupe by identity
        for i in range(self.explorer.count()):
            it = self.explorer.item(i)
            if it.data(Qt.ItemDataRole.UserRole) is doc:
                # refresh text in case dims changed
                it.setText(self._format_explorer_title(doc))
                return

        item = QListWidgetItem(self._format_explorer_title(doc))
        item.setData(Qt.ItemDataRole.UserRole, doc)
        fp = (doc.metadata or {}).get("file_path")
        if fp:
            item.setToolTip(fp)
        self.explorer.addItem(item)

        # keep the row label in sync with edits/resizes/anything
        try:
            doc.changed.connect(lambda *_: self._update_explorer_item_for_doc(doc))
        except Exception:
            pass

    def _remove_doc_from_explorer(self, doc):
        for i in range(self.explorer.count()):
            it = self.explorer.item(i)
            if it.data(Qt.ItemDataRole.UserRole) is doc:
                self.explorer.takeItem(i)
                break

    def _activate_from_explorer(self, item):
        doc = item.data(Qt.ItemDataRole.UserRole)
        for sw in self.mdi.subWindowList():
            w = sw.widget()
            if getattr(w, "document", None) is doc:
                self.mdi.setActiveSubWindow(sw)
                return

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

        self._log(f"Display-Stretch mode → {'LINKED' if checked else 'UNLINKED'}")

    def _on_subwindow_activated(self, sw):
        # Clear previous active marker
        prev = getattr(self, "_last_active_view", None)
        if prev and hasattr(prev, "set_active_highlight"):
            try:
                prev.set_active_highlight(False)
            except Exception:
                pass

        # Mark new active view
        new_view = sw.widget() if sw else None
        if new_view and hasattr(new_view, "set_active_highlight"):
            try:
                new_view.set_active_highlight(True)
            except Exception:
                pass

        # Remember for next time
        self._last_active_view = new_view

        # ... keep your existing logic below ...
        doc = sw.widget().document if sw else None
        self.currentDocumentChanged.emit(doc)
        self._sync_docman_active(doc)
        if hasattr(self, "act_zoom_1_1"):
            self.act_zoom_1_1.setEnabled(bool(sw))

        if hasattr(self, "act_autostretch"):
            from PyQt6.QtCore import QSignalBlocker
            block = QSignalBlocker(self.act_autostretch)
            if sw and hasattr(sw.widget(), "autostretch_enabled"):
                self.act_autostretch.setChecked(bool(sw.widget().autostretch_enabled))
            else:
                self.act_autostretch.setChecked(False)

        view = sw.widget() if sw else None
        # sync the Linked action with the active view (and enable/disable for mono)
        if hasattr(self, "act_stretch_linked"):
            if not view:
                self.act_stretch_linked.setEnabled(False)
                return
            is_mono = getattr(view, "is_mono", lambda: False)()
            self.act_stretch_linked.setEnabled(not is_mono)
            if hasattr(view, "is_autostretch_linked"):
                with QSignalBlocker(self.act_stretch_linked):
                    self.act_stretch_linked.setChecked(bool(view.is_autostretch_linked()))

        self.update_undo_redo_action_labels()
        self._hdr_refresh_timer.start(0)
        self._refresh_mask_action_states()     

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

    def _refresh_mask_action_states(self):
        active_doc = self._active_doc()

        can_apply = bool(active_doc and self._list_candidate_mask_sources(exclude_doc=active_doc))
        can_remove = bool(active_doc and getattr(active_doc, "active_mask_id", None))

        if hasattr(self, "act_apply_mask"):
            self.act_apply_mask.setEnabled(can_apply)
        if hasattr(self, "act_remove_mask"):
            self.act_remove_mask.setEnabled(can_remove)

        # NEW: enable/disable Invert
        if hasattr(self, "act_invert_mask"):
            self.act_invert_mask.setEnabled(can_remove)

        vw = self._active_view()
        overlay_on = bool(getattr(vw, "show_mask_overlay", False)) if vw else False
        has_mask   = bool(active_doc and getattr(active_doc, "active_mask_id", None))

        if hasattr(self, "act_show_mask"):
            self.act_show_mask.setEnabled(has_mask and not overlay_on)
        if hasattr(self, "act_hide_mask"):
            self.act_hide_mask.setEnabled(has_mask and overlay_on)

    def _list_open_docs(self):
        docs = []
        for sw in self.mdi.subWindowList():
            d = getattr(sw.widget(), "document", None)
            if d:
                docs.append(d)
        return docs

    def _list_candidate_mask_sources(self, exclude_doc=None):
        # any other open document can serve as a mask source
        return [d for d in self._list_open_docs() if d is not exclude_doc]

    # Reuse the attach helper from earlier answer; include here if not already present:
    def _prepare_mask_array(self, src_img, target_hw, invert=False, feather_px=0.0):
        import numpy as np
        a = np.asarray(src_img)
        if a.ndim == 3:
            a = (0.2126*a[...,0] + 0.7152*a[...,1] + 0.0722*a[...,2])
        elif a.ndim == 3 and a.shape[2] == 1:
            a = a[...,0]
        a = a.astype(np.float32, copy=False)
        if a.dtype.kind in "ui":
            a /= float(np.iinfo(a.dtype).max)
        else:
            mx = float(a.max()) if a.size else 1.0
            if mx > 1.0: a /= mx
        a = np.clip(a, 0.0, 1.0)

        th, tw = target_hw
        sh, sw = a.shape[:2]
        if (sh, sw) != (th, tw):
            yi = (np.linspace(0, sh-1, th)).astype(np.int32)
            xi = (np.linspace(0, sw-1, tw)).astype(np.int32)
            a = a[yi][:, xi]
        if invert:
            a = 1.0 - a
        if feather_px and feather_px > 0.5:
            k = max(1, min(int(round(feather_px)), 64))
            import numpy as np
            w = np.ones((k,), dtype=np.float32) / float(k)
            a = np.apply_along_axis(lambda r: np.convolve(r, w, mode='same'), 1, a)
            a = np.apply_along_axis(lambda c: np.convolve(c, w, mode='same'), 0, a)
            a = np.clip(a, 0.0, 1.0)
        return a.astype(np.float32, copy=False)

    def _attach_mask_to_document(self, target_doc, mask_doc, *, name="Mask", mode="replace", invert=False, feather=0.0):
        import numpy as np
        if getattr(target_doc, "image", None) is None or getattr(mask_doc, "image", None) is None:
            return False
        th, tw = target_doc.image.shape[:2]
        mask_arr = self._prepare_mask_array(mask_doc.image, (th, tw), invert=invert, feather_px=feather)

        try:
            from pro.masks_core import MaskLayer
        except Exception:
            from uuid import uuid4
            class MaskLayer:
                def __init__(self, name, data, mode="replace", opacity=1.0):
                    self.id = f"mask-{uuid4().hex[:8]}"
                    self.name = name
                    self.data = data
                    self.mode = mode
                    self.opacity = opacity

        layer = MaskLayer(id=name, name=name, data=mask_arr, mode=mode, opacity=1.0)
        try:
            target_doc.add_mask(layer, make_active=True)
        except Exception:
            if not hasattr(target_doc, "masks"): target_doc.masks = {}
            target_doc.masks[layer.id] = layer
            target_doc.active_mask_id = layer.id
            target_doc.changed.emit()

        md = target_doc.metadata.setdefault("masks_meta", {})
        md[layer.id] = {"name": name, "mode": mode, "invert": bool(invert), "feather": float(feather)}
        target_doc.changed.emit()
        return True

    def _apply_mask_menu(self):
        target_doc = self._active_doc()
        if not target_doc:
            QMessageBox.information(self, "Mask", "No active document.")
            return

        candidates = self._list_candidate_mask_sources(exclude_doc=target_doc)
        if not candidates:
            QMessageBox.information(self, "Mask", "Open another image to use as a mask.")
            return

        # If there are multiple, ask which one to use
        mask_doc = None
        if len(candidates) == 1:
            mask_doc = candidates[0]
        else:
            from PyQt6.QtWidgets import QInputDialog
            names = [f"{i+1}. {d.display_name()}" for i, d in enumerate(candidates)]
            choice, ok = QInputDialog.getItem(self, "Choose Mask Image",
                                            "Use this image as mask:", names, 0, False)
            if not ok:
                return
            idx = names.index(choice)
            mask_doc = candidates[idx]

        name = mask_doc.display_name() or "Mask"
        ok = self._attach_mask_to_document(target_doc, mask_doc,
                                        name=name, mode="replace",
                                        invert=False, feather=0.0)
        if ok and hasattr(self, "_log"):
            self._log(f"Mask '{name}' applied to '{target_doc.display_name()}'")

        # NEW: force views to update title/overlay immediately
        if ok:
            try:
                target_doc.changed.emit()
            except Exception:
                pass

        self._refresh_mask_action_states()


    def _remove_mask_menu(self):
        doc = self._active_doc()
        if not doc:
            return
        mid = getattr(doc, "active_mask_id", None)
        if not mid:
            QMessageBox.information(self, "Mask", "No active mask to remove.")
            return
        try:
            doc.remove_mask(mid)
            doc.changed.emit()
            if hasattr(self, "_log"):
                self._log(f"Removed active mask from '{doc.display_name()}'")
        except Exception:
            ...
        # NEW: if overlay was on, hide it now
        vw = self._active_view()
        if vw and getattr(vw, "show_mask_overlay", False):
            vw.show_mask_overlay = False
            vw._render(rebuild=True)

        self._refresh_mask_action_states()



    def _active_view(self):
        sw = self.mdi.activeSubWindow()
        return sw.widget() if sw else None

    def _copy_active_view(self):
        vw = self._active_view()
        if not vw or not hasattr(vw, "get_view_state"):
            return
        self._copied_view_state = vw.get_view_state()
        self._log("View: copied zoom/pan from active window.")

    def _paste_active_view(self):
        vw = self._active_view()
        if not vw or self._copied_view_state is None:
            return
        try:
            vw.apply_view_state(self._copied_view_state)
            self._log("View: pasted zoom/pan into active window.")
        except Exception as e:
            self._log(f"View: paste failed: {e}")

    def _active_doc(self):
        sw = self.mdi.activeSubWindow()
        return sw.widget().document if sw else None

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

            # Make sure we’re on a visible screen
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
        to stale objects on next run. We keep bundle names/uuids, just empty lists.
        If you prefer a full wipe (remove all bundles), see the commented lines.
        """
        try:
            s = QSettings()
            # ---- OPTION A: keep bundles, clear their views (recommended) ----
            raw = s.value("viewbundles/v2", "[]", type=str) or "[]"
            try:
                data = json.loads(raw)
            except Exception:
                data = []

            if isinstance(data, list):
                for b in data:
                    if isinstance(b, dict):
                        b["doc_ptrs"] = []  # drop all pointers
                s.setValue("viewbundles/v2", json.dumps(data, ensure_ascii=False))
            else:
                s.setValue("viewbundles/v2", "[]")

            # also neutralize any legacy store just in case
            s.setValue("viewbundles/v1", "[]")

            # ---- OPTION B: nuke everything (uncomment to prefer a clean slate) ----
            # s.setValue("viewbundles/v2", "[]")
            # s.setValue("viewbundles/v1", "[]")

            s.sync()
        except Exception:
            # last-resort: try to at least write empty arrays
            try:
                s = QSettings()
                s.setValue("viewbundles/v2", "[]")
                s.setValue("viewbundles/v1", "[]")
                s.sync()
            except Exception:
                pass

    def showEvent(self, ev):
        super().showEvent(ev)
        # when returning from the taskbar, some platforms don’t emit WindowStateChange reliably
        if self._suspend_dock_sync:
            QTimer.singleShot(0, lambda: self.changeEvent(QEvent(QEvent.Type.WindowStateChange)))

    def closeEvent(self, e):
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
        if docs:   detail.append(f"Open images: {len(docs)}")
        if edited: detail.append(f"Edited since open: {len(edited)}")
        if detail:
            msg += "\n\n" + "\n".join(detail)

        btn = QMessageBox.question(
            self, "Confirm Exit", msg,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
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
                # Remember we were maximized, and also remember the normal geometry
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

        # NEW: only save shortcuts if enabled in settings (default True)
        try:
            save_on_exit = self.settings.value("shortcuts/save_on_exit", True, type=bool)
        except Exception:
            save_on_exit = True
        if save_on_exit and hasattr(self, "shortcuts"):
            try:
                self.shortcuts.save_shortcuts()
            except Exception:
                pass

        try:
            for t in list(getattr(self, "_bg_threads", [])):
                try:
                    t.wait(3000)  # don't block forever
                except Exception:
                    pass
        except Exception:
            pass

        self._clear_view_bundles_for_next_launch()

        super().closeEvent(e)

def _qs_to_str(seq: QKeySequence) -> str:
    # Native text gives familiar “Ctrl+Shift+P” etc.
    return seq.toString(QKeySequence.SequenceFormat.NativeText).strip()

def _clean_text(txt: str) -> str:
    # Strip ampersand accelerators (&File → File)
    return (txt or "").replace("&", "").strip()

class _CheatSheetDialog(QDialog):
    def __init__(self, parent, keyboard_rows, gesture_rows):
        super().__init__(parent)
        self.setWindowTitle("Keyboard Shortcut Cheat Sheet")
        self.resize(780, 520)

        tabs = QTabWidget(self)

        # --- Keyboard tab ---
        pg_keys = QWidget(tabs)
        v1 = QVBoxLayout(pg_keys)
        tbl_keys = QTableWidget(0, 3, pg_keys)
        tbl_keys.setHorizontalHeaderLabels(["Shortcut", "Action", "Where"])
        tbl_keys.verticalHeader().setVisible(False)
        tbl_keys.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        tbl_keys.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        tbl_keys.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        tbl_keys.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        tbl_keys.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        tbl_keys.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        v1.addWidget(tbl_keys)

        # populate
        for s, action, where in keyboard_rows:
            r = tbl_keys.rowCount()
            tbl_keys.insertRow(r)
            tbl_keys.setItem(r, 0, QTableWidgetItem(s))
            tbl_keys.setItem(r, 1, QTableWidgetItem(action))
            tbl_keys.setItem(r, 2, QTableWidgetItem(where))

        # --- Mouse/Drag tab ---
        pg_mouse = QWidget(tabs)
        v2 = QVBoxLayout(pg_mouse)
        tbl_mouse = QTableWidget(0, 3, pg_mouse)
        tbl_mouse.setHorizontalHeaderLabels(["Gesture", "Context", "Effect"])
        tbl_mouse.verticalHeader().setVisible(False)
        tbl_mouse.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        tbl_mouse.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        tbl_mouse.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        tbl_mouse.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        tbl_mouse.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        tbl_mouse.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        v2.addWidget(tbl_mouse)

        for gesture, context, effect in gesture_rows:
            r = tbl_mouse.rowCount()
            tbl_mouse.insertRow(r)
            tbl_mouse.setItem(r, 0, QTableWidgetItem(gesture))
            tbl_mouse.setItem(r, 1, QTableWidgetItem(context))
            tbl_mouse.setItem(r, 2, QTableWidgetItem(effect))

        tabs.addTab(pg_keys, "Base Keyboard")
        tabs.addTab(pg_mouse, "Additional & Mouse & Drag")

        # Buttons
        btns = QHBoxLayout()
        btns.addStretch(1)

        def _copy_all():
            # Copy as plain text (simple columns)
            lines = []
            lines.append("== Keyboard ==")
            for s, a, w in keyboard_rows:
                lines.append(f"{s:20}  {a}  [{w}]")
            lines.append("")
            lines.append("== Mouse & Drag ==")
            for g, c, e in gesture_rows:
                lines.append(f"{g:24}  {c:18}  {e}")
            QApplication.clipboard().setText("\n".join(lines))
            QMessageBox.information(self, "Copied", "Cheat sheet copied to clipboard.")

        b_copy = QPushButton("Copy")
        b_copy.clicked.connect(_copy_all)
        b_close = QPushButton("Close")
        b_close.clicked.connect(self.accept)
        btns.addWidget(b_copy)
        btns.addWidget(b_close)

        top = QVBoxLayout(self)
        top.addWidget(tabs)
        top.addLayout(btns)

def _uniq_keep_order(items):
    seen = set(); out = []
    for x in items:
        if x in seen: continue
        seen.add(x); out.append(x)
    return out

def _seqs_for_action(act: QAction):
    seqs = [s for s in act.shortcuts() or []] or ([act.shortcut()] if act.shortcut() else [])
    return [s for s in seqs if not s.isEmpty()]

def _where_for_action(act: QAction):
    # Rough “where”: menu/toolbar if attached, else object type
    if act.parent():
        pn = act.parent().__class__.__name__
        if pn.startswith("QMenu") or pn.startswith("QToolBar"):
            return "Menus/Toolbar"
    return "Window"

def _describe_action(act: QAction):
    return _clean_text(act.statusTip() or act.toolTip() or act.text() or act.objectName() or "Action")

def _describe_shortcut(sc: QShortcut):
    return _clean_text(sc.property("hint") or sc.whatsThis() or sc.objectName() or "Shortcut")

def _where_for_shortcut(sc: QShortcut):
    par = sc.parent()
    return par.__class__.__name__ if par is not None else "Window"

class _ProjectSaveWorker(QThread):
    ok = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, path, docs, shortcuts, mdi, compress, parent=None):
        super().__init__(parent)
        self.path = path
        self.docs = docs
        self.shortcuts = shortcuts
        self.mdi = mdi
        self.compress = compress

    def run(self):
        try:
            from pro.project_io import ProjectWriter
            ProjectWriter.write(
                self.path,
                docs=self.docs,
                shortcuts=self.shortcuts,
                mdi=self.mdi,
                compress=self.compress,
                shelf=getattr(self, "window_shelf", None),
            )
            self.ok.emit()
        except Exception as e:
            self.error.emit(str(e))

# --- Global crash/exception handlers (Qt-safe) ---
def install_crash_handlers(app):
    import sys, threading, traceback, faulthandler, atexit, logging
    
    from PyQt6.QtCore import QTimer

    # 1) Hard crashes (segfaults, access violations) → saspro_crash.log
    try:
        _crash_log = open("saspro_crash.log", "w", encoding="utf-8", errors="replace")
        faulthandler.enable(file=_crash_log, all_threads=True)
        atexit.register(_crash_log.close)
    except Exception:
        logging.exception("Failed to enable faulthandler")

    def _show_dialog(title: str, head: str, details: str):
        # Always marshal UI to the main thread
        def _ui():
            m = QMessageBox(app.activeWindow())
            m.setIcon(QMessageBox.Icon.Critical)
            m.setWindowTitle(title)
            m.setText(head)
            m.setInformativeText("Details are available below and in saspro.log.")
            if details:
                m.setDetailedText(details)
            m.setStandardButtons(QMessageBox.StandardButton.Ok)
            m.exec()
        QTimer.singleShot(0, _ui)

    # 2) Any uncaught exception on the main thread
    def _excepthook(exc_type, exc_value, exc_tb):
        tb = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        logging.error("Uncaught exception:\n%s", tb)
        _show_dialog("Unhandled Exception",
                     f"{exc_type.__name__}: {exc_value}",
                     tb)
    sys.excepthook = _excepthook

    # 3) Any uncaught exception in background threads (Py3.8+)
    def _threadhook(args: threading.ExceptHookArgs):
        tb = "".join(traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback))
        logging.error("Uncaught thread exception (%s):\n%s", args.thread.name, tb)
        _show_dialog("Unhandled Thread Exception",
                     f"{args.exc_type.__name__}: {args.exc_value}",
                     tb)
    try:
        threading.excepthook = _threadhook  # type: ignore[attr-defined]
    except Exception:
        pass


if __name__ == "__main__":
    # --- Logging (catch unhandled exceptions to a file) ---
    import logging, sys, os, multiprocessing
    from PyQt6.QtCore import Qt, QCoreApplication, QSettings
    from PyQt6.QtGui import QGuiApplication, QIcon, QPixmap, QColor
    from PyQt6.QtWidgets import QApplication, QSplashScreen, QMessageBox


    logging.basicConfig(
        filename="saspro.log",
        level=logging.ERROR,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

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
        "Starting Seti Astro Suite Pro…",
        Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignLeft,
        QColor("white"),
    )
    app.processEvents()
    # --- Windows exe / multiprocessing friendly (after splash is visible) ---
    try:
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
            "Initializing UI…",
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


