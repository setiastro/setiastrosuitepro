import sys
import typing_extensions
import urllib
# Standard library imports
from itertools import combinations
from tifffile import imwrite

import pickle
import os
os.environ['LIGHTKURVE_STYLE'] = 'default'
import tempfile
import time
import json
import logging
import math
from datetime import datetime, timedelta, timezone
from decimal import getcontext
from urllib.parse import quote
from urllib.parse import quote_plus
import webbrowser
import warnings
import shutil
import subprocess
from xisf import XISF
import requests
import csv
import lz4.block
import zstandard
import base64
import ast
import platform
from pathlib import Path
import glob
from typing import List, Tuple, Dict, Set
import time
from datetime import datetime
import pywt
from io import BytesIO
import io
import re
from collections import defaultdict
from scipy.spatial import Delaunay, KDTree
from scipy.ndimage import gaussian_filter, laplace
import scipy.ndimage as ndi
import plotly.graph_objects as go
from scipy.ndimage import zoom
import multiprocessing
import matplotlib
matplotlib.use("QtAgg") 
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import hsv_to_rgb
import numpy as np
from typing import List, Tuple, Optional
from skimage.restoration import richardson_lucy, denoise_bilateral, denoise_tv_chambolle
from skimage.color import rgb2gray, rgb2lab, lab2rgb
from skimage.transform import warp_polar, warp
from skimage import img_as_float32
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from typing import Optional, Tuple, Callable
from scipy.signal import medfilt
import matplotlib.ticker as mtick
from scipy.interpolate import RBFInterpolator
import random
import inspect
#if running in IDE which runs ipython or jupiter in backend reconfigure may not be available
if (sys.stdout is not None) and (hasattr(sys.stdout, "reconfigure")):
    sys.stdout.reconfigure(encoding='utf-8')

try:
    from photutils.isophote import Ellipse, EllipseGeometry, build_ellipse_model
except Exception as e:
    Ellipse = EllipseGeometry = build_ellipse_model = None

from astropy.stats import sigma_clipped_stats
from collections.abc import MutableMapping
from astropy.io.votable import parse_single_table
from astropy.timeseries import LombScargle, BoxLeastSquares
from photutils.detection import DAOStarFinder
from scipy.spatial import ConvexHull
from astropy.table import Table, vstack
from numba import njit, prange
from scipy.optimize import curve_fit
import exifread
import astroalign
import sqlite3
from datetime import datetime
import traceback
import sep
from astroquery.mast import Tesscut

from lightkurve import TessTargetPixelFile
import oktopus
import lightkurve as lk
from scipy.interpolate import griddata
from astropy.io.fits import Header as FitsHeader
lk.MPLSTYLE = None

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

from astropy.wcs.utils import skycoord_to_pixel
from astropy.coordinates import SkyCoord
from astropy import units as u

from astropy.io.fits import Header
from pyvo.dal.exceptions import DALServiceError

# Reproject for WCS-based alignment
try:
    from reproject import reproject_interp
except ImportError:
    reproject_interp = None  # fallback if not installed

# OpenCV for transform estimation & warping

OPENCV_AVAILABLE = True



# Third-party library imports
import requests
import numpy as np
import pandas as pd
import cv2
from PIL import Image, ImageDraw, ImageFont

# Astropy and Astroquery imports
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, get_body, get_sun
import astropy.units as u
from astropy.wcs import WCS
from astroquery.simbad import Simbad
from astroquery.mast import Mast
from astroquery.vizier import Vizier
import tifffile as tiff

from astropy.utils.data import conf

from scipy.interpolate import PchipInterpolator
from scipy.interpolate import Rbf
from scipy.ndimage import median_filter
from scipy.ndimage import convolve
from scipy.signal import fftconvolve
from scipy.interpolate import interp1d


import numpy.ma as ma

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Circle
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from pro.plate_solver import plate_solve_doc_inplace, _active_doc_from_parent, _get_seed_mode, _set_seed_mode, _as_header

#################################
# PyQt6 Imports
#################################
from collections import defaultdict
import fnmatch
import pyqtgraph as pg
import psutil
from PyQt6.QtGui import QIntValidator
# ----- QtWidgets -----
from PyQt6.QtWidgets import (    QApplication,    QMainWindow,    QWidget,    QVBoxLayout,    QHBoxLayout,    QLabel,    QPushButton,    QFileDialog,    QGraphicsView,    QGraphicsScene,    QMessageBox,    QInputDialog,    QTreeWidget,
    QTreeWidgetItem,    QGraphicsPolygonItem,    QFrame,    QToolTip,    QCheckBox,    QDialog,    QFormLayout,    QSpinBox,    QDialogButtonBox,    QGridLayout,    QGraphicsEllipseItem,    QGraphicsLineItem,    QGraphicsRectItem,
    QGraphicsPathItem,    QDoubleSpinBox,    QColorDialog,    QFontDialog,    QStyle,    QSlider,    QTabWidget,    QScrollArea,    QSizePolicy,    QSpacerItem,    QAbstractItemView,    QToolBar,    QGraphicsPixmapItem,    QRubberBand,
    QGroupBox,    QGraphicsTextItem,    QComboBox,    QLineEdit,    QRadioButton,    QButtonGroup,    QHeaderView,    QStackedWidget,    QSplitter,    QMenuBar,    QTextEdit,    QPlainTextEdit,          QProgressBar,    QGraphicsItem,
    QToolButton,    QStatusBar,    QMenu,    QTableWidget,    QTableWidgetItem,    QListWidget,    QListWidgetItem,    QSplashScreen,    QProgressDialog,     QDockWidget,    QAbstractItemView,    QStyledItemDelegate,    QListView,    QCompleter    , QDateTimeEdit
)

# ----- QtGui -----
from PyQt6.QtGui import (QPixmap,QImage,    QPainter,    QPen,    QColor,    QTransform,    QIcon,    QPainterPath,    QKeySequence,    QFont,    QMovie,    QCursor,    QBrush,    QShortcut,    QPolygon,
    QFontMetrics,        QPolygonF,    QKeyEvent,    QPalette,     QWheelEvent,     QDoubleValidator,    QFontDatabase,    QGuiApplication,    QStandardItemModel,    QStandardItem,    QAction  # NOTE: In PyQt6, QAction is in QtGui (moved from QtWidgets)
)

# ----- QtCore -----
from PyQt6.QtCore import (    Qt,    QRectF,    QLineF,    QPointF,    QThread,    pyqtSignal,    QCoreApplication,    QPoint,    QTimer,    QRect,    QFileSystemWatcher,    QEvent,    pyqtSlot,    QLocale,    QProcess,    QSize,
    QObject,    QSettings,    QRunnable,    QThreadPool,    QSignalBlocker,    QStandardPaths,    QModelIndex,    QMetaObject, QDateTime, QEventLoop
)


# Math functions
from math import sqrt
import math

from legacy.image_manager import load_image, save_image
from imageops.stretch import stretch_color_image, stretch_mono_image
from pro import minorbodycatalog as mbc


# Determine if running inside a PyInstaller bundle
if hasattr(sys, '_MEIPASS'):
    # Set path for PyInstaller bundle
    data_path = os.path.join(sys._MEIPASS, "astroquery", "simbad", "data")
else:
    # Set path using astroquery package location (cross-platform)
    try:
        import astroquery.simbad
        data_path = os.path.join(os.path.dirname(astroquery.simbad.__file__), "data")
    except Exception:
        # Fallback: try to find in site-packages
        import site
        for sp in site.getsitepackages():
            candidate = os.path.join(sp, "astroquery", "simbad", "data")
            if os.path.isdir(candidate):
                data_path = candidate
                break
        else:
            data_path = ""  # Will fail gracefully if not found

# Ensure the final path doesn't contain 'data/data' duplication
if 'data/data' in data_path:
    data_path = data_path.replace('data/data', 'data')

if data_path:
    conf.dataurl = f'file://{data_path}/'

# Access wrench_icon.png, adjusting for PyInstaller executable
if hasattr(sys, '_MEIPASS'):
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
    hrdiagram_path = os.path.join(sys._MEIPASS, 'HRDiagram.png')    
else:
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
    hrdiagram_path = 'HRDiagram.png'    

# Constants for comoving radial distance calculation
H0 = 69.6  # Hubble constant in km/s/Mpc
WM = 0.286  # Omega(matter)
WV = 0.714  # Omega(vacuum)
c = 299792.458  # speed of light in km/s
Tyr = 977.8  # coefficient to convert 1/H into Gyr
Mpc_to_Gly = 3.262e-3  # Conversion from Mpc to Gly

otype_long_name_lookup = {
    "ev": "transient event",
    "Rad": "Radio-source",
    "mR": "metric Radio-source",
    "cm": "centimetric Radio-source",
    "mm": "millimetric Radio-source",
    "smm": "sub-millimetric source",
    "HI": "HI (21cm) source",
    "rB": "radio Burst",
    "Mas": "Maser",
    "IR": "Infra-Red source",
    "FIR": "Far-Infrared source",
    "MIR": "Mid-Infrared source",
    "NIR": "Near-Infrared source",
    "blu": "Blue object",
    "UV": "UV-emission source",
    "X": "X-ray source",
    "UX?": "Ultra-luminous X-ray candidate",
    "ULX": "Ultra-luminous X-ray source",
    "gam": "gamma-ray source",
    "gB": "gamma-ray Burst",
    "err": "Not an object (error, artefact, ...)",
    "grv": "Gravitational Source",
    "Lev": "(Micro)Lensing Event",
    "LS?": "Possible gravitational lens System",
    "Le?": "Possible gravitational lens",
    "LI?": "Possible gravitationally lensed image",
    "gLe": "Gravitational Lens",
    "gLS": "Gravitational Lens System (lens+images)",
    "GWE": "Gravitational Wave Event",
    "..?": "Candidate objects",
    "G?": "Possible Galaxy",
    "SC?": "Possible Supercluster of Galaxies",
    "C?G": "Possible Cluster of Galaxies",
    "Gr?": "Possible Group of Galaxies",
    "**?": "Physical Binary Candidate",
    "EB?": "Eclipsing Binary Candidate",
    "Sy?": "Symbiotic Star Candidate",
    "CV?": "Cataclysmic Binary Candidate",
    "No?": "Nova Candidate",
    "XB?": "X-ray binary Candidate",
    "LX?": "Low-Mass X-ray binary Candidate",
    "HX?": "High-Mass X-ray binary Candidate",
    "Pec?": "Possible Peculiar Star",
    "Y*?": "Young Stellar Object Candidate",
    "TT?": "T Tau star Candidate",
    "C*?": "Possible Carbon Star",
    "S*?": "Possible S Star",
    "OH?": "Possible Star with envelope of OH/IR type",
    "WR?": "Possible Wolf-Rayet Star",
    "Be?": "Possible Be Star",
    "Ae?": "Possible Herbig Ae/Be Star",
    "HB?": "Possible Horizontal Branch Star",
    "RR?": "Possible Star of RR Lyr type",
    "Ce?": "Possible Cepheid",
    "WV?": "Possible Variable Star of W Vir type",
    "RB?": "Possible Red Giant Branch star",
    "sg?": "Possible Supergiant star",
    "s?r": "Possible Red supergiant star",
    "s?y": "Possible Yellow supergiant star",
    "s?b": "Possible Blue supergiant star",
    "AB?": "Asymptotic Giant Branch Star candidate",
    "LP?": "Long Period Variable candidate",
    "Mi?": "Mira candidate",
    "pA?": "Post-AGB Star Candidate",
    "BS?": "Candidate blue Straggler Star",
    "HS?": "Hot subdwarf candidate",
    "WD?": "White Dwarf Candidate",
    "N*?": "Neutron Star Candidate",
    "BH?": "Black Hole Candidate",
    "SN?": "SuperNova Candidate",
    "LM?": "Low-mass star candidate",
    "BD?": "Brown Dwarf Candidate",
    "mul": "Composite object",
    "reg": "Region defined in the sky",
    "vid": "Underdense region of the Universe",
    "SCG": "Supercluster of Galaxies",
    "ClG": "Cluster of Galaxies",
    "GrG": "Group of Galaxies",
    "CGG": "Compact Group of Galaxies",
    "PaG": "Pair of Galaxies",
    "IG": "Interacting Galaxies",
    "C?*": "Possible (open) star cluster",
    "Gl?": "Possible Globular Cluster",
    "Cl*": "Cluster of Stars",
    "GlC": "Globular Cluster",
    "OpC": "Open (galactic) Cluster",
    "As*": "Association of Stars",
    "St*": "Stellar Stream",
    "MGr": "Moving Group",
    "**": "Double or multiple star",
    "EB*": "Eclipsing binary",
    "Al*": "Eclipsing binary of Algol type",
    "bL*": "Eclipsing binary of beta Lyr type",
    "WU*": "Eclipsing binary of W UMa type",
    "SB*": "Spectroscopic binary",
    "El*": "Ellipsoidal variable Star",
    "Sy*": "Symbiotic Star",
    "CV*": "Cataclysmic Variable Star",
    "DQ*": "CV DQ Her type (intermediate polar)",
    "AM*": "CV of AM Her type (polar)",
    "NL*": "Nova-like Star",
    "No*": "Nova",
    "DN*": "Dwarf Nova",
    "XB*": "X-ray Binary",
    "LXB": "Low Mass X-ray Binary",
    "HXB": "High Mass X-ray Binary",
    "ISM": "Interstellar matter",
    "PoC": "Part of Cloud",
    "PN?": "Possible Planetary Nebula",
    "CGb": "Cometary Globule",
    "bub": "Bubble",
    "EmO": "Emission Object",
    "Cld": "Cloud",
    "GNe": "Galactic Nebula",
    "DNe": "Dark Cloud (nebula)",
    "RNe": "Reflection Nebula",
    "MoC": "Molecular Cloud",
    "glb": "Globule (low-mass dark cloud)",
    "cor": "Dense core",
    "SFR": "Star forming region",
    "HVC": "High-velocity Cloud",
    "HII": "HII (ionized) region",
    "PN": "Planetary Nebula",
    "sh": "HI shell",
    "SR?": "SuperNova Remnant Candidate",
    "SNR": "SuperNova Remnant",
    "of?": "Outflow candidate",
    "out": "Outflow",
    "HH": "Herbig-Haro Object",
    "*": "Star",
    "V*?": "Star suspected of Variability",
    "Pe*": "Peculiar Star",
    "HB*": "Horizontal Branch Star",
    "Y*O": "Young Stellar Object",
    "Ae*": "Herbig Ae/Be star",
    "Em*": "Emission-line Star",
    "Be*": "Be Star",
    "BS*": "Blue Straggler Star",
    "RG*": "Red Giant Branch star",
    "AB*": "Asymptotic Giant Branch Star (He-burning)",
    "C*": "Carbon Star",
    "S*": "S Star",
    "sg*": "Evolved supergiant star",
    "s*r": "Red supergiant star",
    "s*y": "Yellow supergiant star",
    "s*b": "Blue supergiant star",
    "HS*": "Hot subdwarf",
    "pA*": "Post-AGB Star (proto-PN)",
    "WD*": "White Dwarf",
    "LM*": "Low-mass star (M<1solMass)",
    "BD*": "Brown Dwarf (M<0.08solMass)",
    "N*": "Confirmed Neutron Star",
    "OH*": "OH/IR star",
    "TT*": "T Tau-type Star",
    "WR*": "Wolf-Rayet Star",
    "PM*": "High proper-motion Star",
    "HV*": "High-velocity Star",
    "V*": "Variable Star",
    "Ir*": "Variable Star of irregular type",
    "Or*": "Variable Star of Orion Type",
    "Er*": "Eruptive variable Star",
    "RC*": "Variable Star of R CrB type",
    "RC?": "Variable Star of R CrB type candidate",
    "Ro*": "Rotationally variable Star",
    "a2*": "Variable Star of alpha2 CVn type",
    "Psr": "Pulsar",
    "BY*": "Variable of BY Dra type",
    "RS*": "Variable of RS CVn type",
    "Pu*": "Pulsating variable Star",
    "RR*": "Variable Star of RR Lyr type",
    "Ce*": "Cepheid variable Star",
    "dS*": "Variable Star of delta Sct type",
    "RV*": "Variable Star of RV Tau type",
    "WV*": "Variable Star of W Vir type",
    "bC*": "Variable Star of beta Cep type",
    "cC*": "Classical Cepheid (delta Cep type)",
    "gD*": "Variable Star of gamma Dor type",
    "SX*": "Variable Star of SX Phe type (subdwarf)",
    "LP*": "Long-period variable star",
    "Mi*": "Variable Star of Mira Cet type",
    "SN*": "SuperNova",
    "su*": "Sub-stellar object",
    "Pl?": "Extra-solar Planet Candidate",
    "Pl": "Extra-solar Confirmed Planet",
    "G": "Galaxy",
    "PoG": "Part of a Galaxy",
    "GiC": "Galaxy in Cluster of Galaxies",
    "BiC": "Brightest galaxy in a Cluster (BCG)",
    "GiG": "Galaxy in Group of Galaxies",
    "GiP": "Galaxy in Pair of Galaxies",
    "rG": "Radio Galaxy",
    "H2G": "HII Galaxy",
    "LSB": "Low Surface Brightness Galaxy",
    "AG?": "Possible Active Galaxy Nucleus",
    "Q?": "Possible Quasar",
    "Bz?": "Possible Blazar",
    "BL?": "Possible BL Lac",
    "EmG": "Emission-line galaxy",
    "SBG": "Starburst Galaxy",
    "bCG": "Blue compact Galaxy",
    "LeI": "Gravitationally Lensed Image",
    "LeG": "Gravitationally Lensed Image of a Galaxy",
    "LeQ": "Gravitationally Lensed Image of a Quasar",
    "AGN": "Active Galaxy Nucleus",
    "LIN": "LINER-type Active Galaxy Nucleus",
    "SyG": "Seyfert Galaxy",
    "Sy1": "Seyfert 1 Galaxy",
    "Sy2": "Seyfert 2 Galaxy",
    "Bla": "Blazar",
    "BLL": "BL Lac - type object",
    "OVV": "Optically Violently Variable object",
    "QSO": "Quasar"
}


# ────────────────────────────────────────────────
# 1a) Map each SIMBAD otype → one of our high-level categories
# ────────────────────────────────────────────────
OTYPE_TO_CATEGORY = {
    # Transient & Explosive Events
    "ev":   "Transient & Explosive",
    "rB":   "Transient & Explosive",
    "gB":   "Transient & Explosive",
    "GWE":  "Transient & Explosive",
    "SN*":  "Transient & Explosive",
    "SN?":  "Transient & Explosive",
    "SR?":  "Transient & Explosive",
    "SNR":  "Transient & Explosive",
    "Lev":  "Transient & Explosive",

    # High-Energy / X-ray / γ-ray
    "X":    "High-Energy / X-ray / γ-ray",
    "UX?":  "High-Energy / X-ray / γ-ray",
    "ULX":  "High-Energy / X-ray / γ-ray",
    "gam":  "High-Energy / X-ray / γ-ray",
    "grv":  "High-Energy / X-ray / γ-ray",
    "Psr":  "High-Energy / X-ray / γ-ray",
    "N*?":  "High-Energy / X-ray / γ-ray",
    "BH?":  "High-Energy / X-ray / γ-ray",

    # Radio & sub-millimetric
    "Rad":  "Radio & Sub-mm",
    "mR":   "Radio & Sub-mm",
    "cm":   "Radio & Sub-mm",
    "mm":   "Radio & Sub-mm",
    "smm":  "Radio & Sub-mm",
    "HI":   "Radio & Sub-mm",
    "Mas":  "Radio & Sub-mm",

    # IR / Optical / UV / Blue
    "IR":   "IR / Optical / UV",
    "FIR":  "IR / Optical / UV",
    "MIR":  "IR / Optical / UV",
    "NIR":  "IR / Optical / UV",
    "UV":   "IR / Optical / UV",
    "blu":  "IR / Optical / UV",

    # Gravitational Lensing & Microlensing
    "Lev":  "Gravitational Lensing",
    "LS?":  "Gravitational Lensing",
    "Le?":  "Gravitational Lensing",
    "LI?":  "Gravitational Lensing",
    "gLe":  "Gravitational Lensing",
    "gLS":  "Gravitational Lensing",

    # Stars & Stellar Objects
    "*":    "Stars & Stellar",
    "V*":   "Stars & Stellar",
    "Pe*":  "Stars & Stellar",
    "HB*":  "Stars & Stellar",
    "Y*O":  "Stars & Stellar",
    "Ae*":  "Stars & Stellar",
    "Em*":  "Stars & Stellar",
    "Be*":  "Stars & Stellar",
    "BS*":  "Stars & Stellar",
    "RG*":  "Stars & Stellar",
    "AB*":  "Stars & Stellar",
    "C*":   "Stars & Stellar",
    "S*":   "Stars & Stellar",
    "sg*":  "Stars & Stellar",
    "s*r":  "Stars & Stellar",
    "s*y":  "Stars & Stellar",
    "s*b":  "Stars & Stellar",
    "HS*":  "Stars & Stellar",
    "pA*":  "Stars & Stellar",
    "WD*":  "Stars & Stellar",
    "LM*":  "Stars & Stellar",
    "BD*":  "Stars & Stellar",
    "N*":   "Stars & Stellar",
    "OH*":  "Stars & Stellar",
    "TT*":  "Stars & Stellar",
    "WR*":  "Stars & Stellar",
    "PM*":  "Stars & Stellar",
    "HV*":  "Stars & Stellar",
    "C?*":  "Stars & Stellar",
    "Pec?": "Stars & Stellar",
    "Y*?":  "Stars & Stellar",
    "TT?":  "Stars & Stellar",
    "C*?":  "Stars & Stellar",
    "S*?":  "Stars & Stellar",
    "OH?":  "Stars & Stellar",
    "WR?":  "Stars & Stellar",
    "Be?":  "Stars & Stellar",
    "Ae?":  "Stars & Stellar",
    "HB?":  "Stars & Stellar",
    "RB?":  "Stars & Stellar",
    "sg?":  "Stars & Stellar",
    "s?r":  "Stars & Stellar",
    "s?y":  "Stars & Stellar",
    "s?b":  "Stars & Stellar",
    "pA?":  "Stars & Stellar",
    "BS?":  "Stars & Stellar",
    "HS?":  "Stars & Stellar",
    "WD?":  "Stars & Stellar",    

    # Binaries & Multiples / Variables
    "**":   "Binaries & Variables",
    "EB*":  "Binaries & Variables",
    "Al*":  "Binaries & Variables",
    "bL*":  "Binaries & Variables",
    "WU*":  "Binaries & Variables",
    "SB*":  "Binaries & Variables",
    "El*":  "Binaries & Variables",
    "Sy*":  "Binaries & Variables",
    "CV*":  "Binaries & Variables",
    "DQ*":  "Binaries & Variables",
    "AM*":  "Binaries & Variables",
    "NL*":  "Binaries & Variables",
    "No*":  "Binaries & Variables",
    "DN*":  "Binaries & Variables",
    "XB*":  "Binaries & Variables",
    "LXB":  "Binaries & Variables",
    "HXB":  "Binaries & Variables",
    "Pl?":  "Binaries & Variables",
    "Ce?":  "Binaries & Variables",
    "Ce*":  "Binaries & Variables",
    "cC*":  "Binaries & Variables",
    "**?":  "Binaries & Variables",
    "EB?":  "Binaries & Variables",
    "Sy?":  "Binaries & Variables",
    "CV?":  "Binaries & Variables",
    "No?":  "Binaries & Variables",
    "XB?":  "Binaries & Variables",
    "LX?":  "Binaries & Variables",
    "HX?":  "Binaries & Variables",
    "RR?":  "Binaries & Variables",
    "WV?":  "Binaries & Variables",
    "LP?":  "Binaries & Variables",
    "Mi?":  "Binaries & Variables",
    "Ce?":  "Binaries & Variables",
    "cC*":  "Binaries & Variables",
    "Pl?":  "Binaries & Variables",    

    # Clusters & Associations
    "Cl*":  "Clusters & Associations",
    "GlC":  "Clusters & Associations",
    "OpC":  "Clusters & Associations",
    "As*":  "Clusters & Associations",
    "St*":  "Clusters & Associations",
    "MGr":  "Clusters & Associations",
    "C?*":  "Clusters & Associations",
    "Gl?":  "Clusters & Associations",    

    # Nebulae & Interstellar Matter
    "PN":   "Nebulae & ISM",
    "PN?":  "Nebulae & ISM",
    "EmO":  "Nebulae & ISM",
    "GNe":  "Nebulae & ISM",
    "DNe":  "Nebulae & ISM",
    "RNe":  "Nebulae & ISM",
    "MoC":  "Nebulae & ISM",
    "Cld":  "Nebulae & ISM",
    "bub":  "Nebulae & ISM",
    "Cld":  "Nebulae & ISM",
    "Cld":  "Nebulae & ISM",
    "sh":   "Nebulae & ISM",
    "SFR":  "Nebulae & ISM",
    "HVC":  "Nebulae & ISM",
    "CGb":  "Nebulae & ISM",
    "PoC":  "Nebulae & ISM",
    "glb":  "Nebulae & ISM",
    "cor":  "Nebulae & ISM",
    "out":  "Nebulae & ISM",
    "HH":   "Nebulae & ISM",
    "HII":  "Nebulae & ISM",
    "ISM":  "Nebulae & ISM",
    "of?":  "Nebulae & ISM",    

    # Galaxies & Active Nuclei
    "G":    "Galaxies & AGN",
    "PoG":  "Galaxies & AGN",
    "EmG":  "Galaxies & AGN",
    "SBG":  "Galaxies & AGN",
    "LSB":  "Galaxies & AGN",
    "AGN":  "Galaxies & AGN",
    "LIN":  "Galaxies & AGN",
    "SyG":  "Galaxies & AGN",
    "Sy1":  "Galaxies & AGN",
    "Sy2":  "Galaxies & AGN",
    "Bla":  "Galaxies & AGN",
    "BLL":  "Galaxies & AGN",
    "OVV":  "Galaxies & AGN",
    "QSO":  "Galaxies & AGN",
    "Q?":   "Galaxies & AGN",
    "AG?":  "Galaxies & AGN",
    "G?":   "Galaxies & AGN",
    "IG":   "Galaxies & AGN",
    "GiC":  "Galaxies & AGN",
    "BiC":  "Galaxies & AGN",
    "GiP":  "Galaxies & AGN",
    "rG":   "Galaxies & AGN",
    "H2G":  "Galaxies & AGN",
    "Bz?":  "Galaxies & AGN",
    "BL?":  "Galaxies & AGN",    


    # Large-Scale Structure: clusters, superclusters, voids
    "ClG":  "Large-Scale Structure",
    "GrG":  "Large-Scale Structure",
    "CGG":  "Large-Scale Structure",
    "PaG":  "Large-Scale Structure",
    "SCG":  "Large-Scale Structure",
    "SC?":  "Large-Scale Structure",
    "C?G":  "Large-Scale Structure",
    "Gr?":  "Large-Scale Structure",
    "vid":  "Large-Scale Structure",
    "GiG":  "Large-Scale Structure",

    # Regions, Clouds & Artefacts
    "reg":  "Regions & Clouds",
    "mul":  "Regions & Clouds",

    # Errors & Artefacts / Unknown
    "err":  "Errors & Artefacts",
    "..?":  "Errors & Artefacts",  
}

# ────────────────────────────────────────────────
# 1b) Assign each category a distinct QColor
# ────────────────────────────────────────────────
CATEGORY_TO_COLOR = {
    "Transient & Explosive":        QColor(255,   0,   0),  # red
    "High-Energy / X-ray / γ-ray":  QColor(255, 165,   0),  # orange
    "Radio & Sub-mm":               QColor(128,   0, 128),  # purple
    "IR / Optical / UV":            QColor(  0, 128,   0),  # green
    "Gravitational Lensing":        QColor(218, 112, 214),  # orchid
    "Stars & Stellar":              QColor(  0,   0, 255),  # blue
    "Binaries & Variables":         QColor(255, 255,   0),  # yellow
    "Clusters & Associations":      QColor(165,  42,  42),  # brown
    "Nebulae & ISM":                QColor(  0, 128, 128),  # teal
    "Galaxies & AGN":               QColor(255,   0, 255),  # magenta
    "Large-Scale Structure":        QColor( 200,  200,  200),  # dark gray
    "Regions & Clouds":             QColor( 95, 158, 160),  # cadet blue
    "Errors & Artefacts":           QColor(128, 128, 128),  # gray
}



Simbad.ROW_LIMIT = 0  # Remove row limit for full results
Simbad.TIMEOUT = 300  # Increase timeout for long queries

# Astrometry.net API constants
ASTROMETRY_API_URL = "http://nova.astrometry.net/api/"
ASTROMETRY_API_KEY_FILE = "astrometry_api_key.txt"

settings = QSettings("Seti Astro", "Seti Astro Suite")

def save_api_key(api_key):
    settings.setValue("astrometry_api_key", api_key)  # Save to QSettings
    print("API key saved.")

def load_api_key():
    api_key = settings.value("astrometry_api_key", "")  # Load from QSettings
    if api_key:
        print("API key loaded.")
    return api_key

def _find_main_window(w):
    p = w.parent()
    while p and not (hasattr(p, "doc_manager") or hasattr(p, "docman")):
        p = p.parent()
    return p


# --- MarkerLayer: one lightweight item that draws all markers efficiently ---
from PyQt6.QtWidgets import QGraphicsItem
from PyQt6.QtCore import QRectF, QPointF, Qt
from PyQt6.QtGui import QPainter, QPen, QColor

class MarkerLayer(QGraphicsItem):
    def __init__(self, image_w, image_h, show_names_fn, color_fn, style_fn,
                 selected_name_fn=lambda: None, radius_px=6, cell=64, parent=None):
        super().__init__(parent)
        self._w, self._h = image_w, image_h
        self._bounds = QRectF(0, 0, self._w, self._h)
        self._radius = radius_px
        self._cell = cell
        self._grid = {}
        self._show_names = show_names_fn
        self._color = color_fn              # fallback color
        self._style = style_fn
        self._selected_name = selected_name_fn
        self.setZValue(100)
        self.setCacheMode(QGraphicsItem.CacheMode.DeviceCoordinateCache)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemUsesExtendedStyleOption, True)
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)

    def boundingRect(self) -> QRectF:
        return self._bounds

    def resize(self, w, h):
        if w == self._w and h == self._h:
            return
        self.prepareGeometryChange()
        self._w, self._h = w, h
        self._bounds = QRectF(0, 0, w, h)

    def set_points(self, pts):
        try:
            if self.scene() is None:
                return
        except RuntimeError:
            return

        self._grid.clear()
        c = self._cell
        for p in pts:
            x, y = p["x"], p["y"]
            gx, gy = int(x // c), int(y // c)
            self._grid.setdefault((gx, gy), []).append(p)
        try:
            self.update()
        except RuntimeError:
            pass


    def paint(self, p: QPainter, option, widget):
        vr = option.exposedRect.adjusted(-self._cell, -self._cell, self._cell, self._cell)
        c = self._cell

        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        base_pen = QPen()
        base_pen.setCosmetic(True)
        base_pen.setWidth(1)

        r = self._radius
        show_names = self._show_names()
        style = self._style()
        selected_name = self._selected_name()

        gx0, gy0 = int(max(0, int(vr.left() // c))),  int(max(0, int(vr.top() // c)))
        gx1, gy1 = int(int(vr.right() // c)),         int(int(vr.bottom() // c))

        for gy in range(gy0, gy1 + 1):
            for gx in range(gx0, gx1 + 1):
                for pt in self._grid.get((gx, gy), ()):
                    x, y = pt["x"], pt["y"]
                    if not vr.contains(QPointF(x, y)):
                        continue

                    # pick color: green if selected, else per-point color, else fallback
                    col = QColor(0, 255, 0) if selected_name and pt.get("name") == selected_name \
                          else pt.get("color", self._color())
                    base_pen.setColor(col)
                    p.setPen(base_pen)


                    if style == "Crosshair":
                        # use QLineF to avoid int casting everywhere
                        p.drawLine(QLineF(x - r, y,     x + r, y))
                        p.drawLine(QLineF(x,     y - r, x,     y + r))
                    else:
                        p.drawEllipse(QPointF(x, y), r, r)

                    if show_names and pt.get("name"):
                        text_pen = QPen(QColor(255, 255, 255))
                        text_pen.setCosmetic(True)
                        p.setPen(text_pen)
                        name_str = str(pt["name"])  # make sure it’s a plain str
                        # either cast to ints...
                        p.drawText(int(x + r + 2), int(y - r - 2), name_str)
                        # ...or equivalently:
                        # p.drawText(QPointF(x + r + 2, y - r - 2), name_str)
                        p.setPen(base_pen)

def _qt_is_alive(obj) -> bool:
    if obj is None:
        return False
    try:
        import shiboken6
        return shiboken6.isValid(obj)
    except Exception:
        pass
    try:
        from PyQt6 import sip
        return not sip.isdeleted(obj)
    except Exception:
        pass
    try:
        _ = obj.scene()
        return True
    except RuntimeError:
        return False


class CustomGraphicsView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.setMouseTracking(True)  # Enable mouse tracking
        self.setDragMode(QGraphicsView.DragMode.NoDrag)  # Disable default drag mode to avoid hand cursor
        self.setCursor(Qt.CursorShape.ArrowCursor)  # Set default cursor to arrow
        self.drawing_item = None
        self.start_pos = None     
        self.annotation_items = []  # Store annotation items  
        self.drawing_measurement = False
        self.measurement_start = QPointF()    
        self._search_circle_item = None
         

        self.selected_object = None  # Initialize selected_object to None
        self.show_names = False 

        # Variables for drawing the circle
        self.circle_center = None
        self.circle_radius = 0
        self.drawing_circle = False  # Flag to check if we're currently drawing a circle
        self.dragging = False  # Flag to manage manual dragging


    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if event.modifiers() == Qt.KeyboardModifier.ControlModifier:
                # Start annotation mode with the current tool
                self.start_pos = self.mapToScene(event.pos())

                # Check which tool is currently selected
                if self.parent.current_tool == "Ellipse":
                    self.drawing_item = QGraphicsEllipseItem()
                    self.drawing_item.setPen(QPen(self.parent.selected_color, 2))
                    self.parent.main_scene.addItem(self.drawing_item)

                elif self.parent.current_tool == "Rectangle":
                    self.drawing_item = QGraphicsRectItem()
                    self.drawing_item.setPen(QPen(self.parent.selected_color, 2))
                    self.parent.main_scene.addItem(self.drawing_item)

                elif self.parent.current_tool == "Arrow":
                    self.drawing_item = QGraphicsLineItem()
                    self.drawing_item.setPen(QPen(self.parent.selected_color, 2))
                    self.parent.main_scene.addItem(self.drawing_item)

                elif self.parent.current_tool == "Freehand":
                    self.drawing_item = QGraphicsPathItem()
                    path = QPainterPath(self.start_pos)
                    self.drawing_item.setPath(path)
                    self.drawing_item.setPen(QPen(self.parent.selected_color, 2))
                    self.parent.main_scene.addItem(self.drawing_item)

                elif self.parent.current_tool == "Text":
                    text, ok = QInputDialog.getText(self, "Add Text", "Enter text:")
                    if ok and text:
                        text_item = QGraphicsTextItem(text)
                        text_item.setPos(self.start_pos)
                        text_item.setDefaultTextColor(self.parent.selected_color)  # Use selected color
                        text_item.setFont(self.parent.selected_font)  # Use selected font
                        self.parent.main_scene.addItem(text_item)
                        
                        # Store as ('text', text, position, color)
                        self.annotation_items.append(('text', text, self.start_pos, self.parent.selected_color))


                elif self.parent.current_tool == "Compass":
                    self.place_celestial_compass(self.start_pos)

            elif event.modifiers() == Qt.KeyboardModifier.ShiftModifier:
                # Start drawing a circle for Shift+Click
                self.drawing_circle = True
                self.circle_center = self.mapToScene(event.pos())
                self.circle_radius = 0
                self.parent.status_label.setText("Drawing circle: Shift + Drag")
                self.update_circle()

            elif event.modifiers() == Qt.KeyboardModifier.AltModifier:
                # Start celestial measurement for Alt+Click
                self.measurement_start = self.mapToScene(event.pos())
                self.drawing_measurement = True
                self.drawing_item = None  # Clear any active annotation item
    

            else:
                # Detect if an object circle was clicked without Shift or Ctrl
                scene_pos = self.mapToScene(event.pos())
                clicked_object = self.get_object_at_position(scene_pos)
                
                if clicked_object:
                    # Select the clicked object and redraw
                    self.parent.selected_object = clicked_object
                    self.select_object(clicked_object)
                    self.draw_query_results()
                    self.update_mini_preview()
                    
                    # Highlight the corresponding row in the TreeWidget
                    for i in range(self.parent.results_tree.topLevelItemCount()):
                        item = self.parent.results_tree.topLevelItem(i)
                        if item.text(2) == clicked_object["name"]:  # Assuming third element is 'Name'
                            self.parent.results_tree.setCurrentItem(item)
                            break
                else:
                    # Start manual dragging if no modifier is held
                    self.dragging = True
                    self.setCursor(Qt.CursorShape.ClosedHandCursor)  # Use closed hand cursor to indicate dragging
                    self.drag_start_pos = event.pos()  # Store starting position

        super().mousePressEvent(event)


    def mouseDoubleClickEvent(self, event):
        """Handle double-click event on an object in the main image to open SIMBAD or NED URL based on source."""
        scene_pos = self.mapToScene(event.pos())
        clicked_object = self.get_object_at_position(scene_pos)

        if clicked_object:
            object_name = clicked_object.get("name")  # Access 'name' key from the dictionary
            ra = float(clicked_object.get("ra"))  # Ensure RA is a float for precision
            dec = float(clicked_object.get("dec"))  # Ensure Dec is a float for precision
            source = clicked_object.get("source", "Simbad")  # Default to "Simbad" if source not specified

            if source == "Simbad" and object_name:
                # Open Simbad URL with encoded object name
                encoded_name = quote(object_name)
                url = f"https://simbad.cds.unistra.fr/simbad/sim-basic?Ident={encoded_name}&submit=SIMBAD+search"
                webbrowser.open(url)
            elif source == "Vizier":
                # Format the NED search URL with proper RA, Dec, and radius
                radius = 5 / 60  # Radius in arcminutes (5 arcseconds)
                dec_sign = "%2B" if dec >= 0 else "-"  # Determine sign for declination
                ned_url = (
                    f"http://ned.ipac.caltech.edu/conesearch?search_type=Near%20Position%20Search"
                    f"&ra={ra:.6f}d&dec={dec_sign}{abs(dec):.6f}d&radius={radius:.3f}"
                    "&in_csys=Equatorial&in_equinox=J2000.0"
                )
                webbrowser.open(ned_url)
            elif source == "Mast":
                # Open MAST URL using RA and Dec with a small radius for object lookup
                mast_url = f"https://mast.stsci.edu/portal/Mashup/Clients/Mast/Portal.html?searchQuery={ra}%2C{dec}%2Cradius%3D0.0006"
                webbrowser.open(mast_url)                
        else:
            super().mouseDoubleClickEvent(event)

    def mouseMoveEvent(self, event):
        scene_pos = self.mapToScene(event.pos())

        if self.drawing_circle:
            # Update the circle radius as the mouse moves
            self.circle_radius = np.sqrt(
                (scene_pos.x() - self.circle_center.x()) ** 2 +
                (scene_pos.y() - self.circle_center.y()) ** 2
            )
            self.update_circle()

        elif self.drawing_measurement:
            # Update the measurement line dynamically as the mouse moves
            if self.drawing_item:
                self.parent.main_scene.removeItem(self.drawing_item)  # Remove previous line if exists
            self.drawing_item = QGraphicsLineItem(QLineF(self.measurement_start, scene_pos))
            self.drawing_item.setPen(QPen(Qt.GlobalColor.green, 2, Qt.PenStyle.DashLine))  # Use green dashed line for measurement
            self.parent.main_scene.addItem(self.drawing_item)

        elif self.drawing_item:
            # Update the current drawing item based on the selected tool and mouse position
            if isinstance(self.drawing_item, QGraphicsEllipseItem) and self.parent.current_tool == "Ellipse":
                # For Ellipse tool, update the ellipse dimensions
                rect = QRectF(self.start_pos, scene_pos).normalized()
                self.drawing_item.setRect(rect)

            elif isinstance(self.drawing_item, QGraphicsRectItem) and self.parent.current_tool == "Rectangle":
                # For Rectangle tool, update the rectangle dimensions
                rect = QRectF(self.start_pos, scene_pos).normalized()
                self.drawing_item.setRect(rect)

            elif isinstance(self.drawing_item, QGraphicsLineItem) and self.parent.current_tool == "Arrow":
                # For Arrow tool, set the line from start_pos to current mouse position
                line = QLineF(self.start_pos, scene_pos)
                self.drawing_item.setLine(line)

            elif isinstance(self.drawing_item, QGraphicsPathItem) and self.parent.current_tool == "Freehand":
                # For Freehand tool, add a line to the path to follow the mouse movement
                path = self.drawing_item.path()
                path.lineTo(scene_pos)
                self.drawing_item.setPath(path)

        elif self.dragging:
            # Handle manual dragging by scrolling the view
            delta = event.pos() - self.drag_start_pos
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            self.drag_start_pos = event.pos()
        else:
            # Update RA/Dec display as the cursor moves
            self.parent.update_ra_dec_from_mouse(event)
            
        super().mouseMoveEvent(event)
                

    def mouseReleaseEvent(self, event):
        if self.drawing_circle and event.button() == Qt.MouseButton.LeftButton:
            # Stop drawing the circle
            self.drawing_circle = False
            self.parent.circle_center = self.circle_center
            self.parent.circle_radius = self.circle_radius

            # Calculate RA/Dec for the circle center
            ra, dec = self.parent.calculate_ra_dec_from_pixel(self.circle_center.x(), self.circle_center.y())
            if ra is not None and dec is not None:
                self.parent.ra_label.setText(f"RA: {self.parent.convert_ra_to_hms(ra)}")
                self.parent.dec_label.setText(f"Dec: {self.parent.convert_dec_to_dms(dec)}")

                if self.parent.pixscale:
                    radius_arcmin = self.circle_radius * self.parent.pixscale / 60.0
                    self.parent.status_label.setText(
                        f"Circle set at center RA={ra:.6f}, Dec={dec:.6f}, radius={radius_arcmin:.2f} arcmin"
                    )
                else:
                    self.parent.status_label.setText("Pixscale not available for radius calculation.")
            else:
                self.parent.status_label.setText("Unable to determine RA/Dec due to missing WCS.")

            # Update circle data and redraw
            self.parent.update_circle_data()
            self.update_circle()

        elif self.drawing_measurement and event.button() == Qt.MouseButton.LeftButton:
            # Complete the measurement when the mouse is released
            self.drawing_measurement = False
            measurement_end = self.mapToScene(event.pos())

            # Calculate celestial distance between start and end points
            ra1, dec1 = self.parent.calculate_ra_dec_from_pixel(self.measurement_start.x(), self.measurement_start.y())
            ra2, dec2 = self.parent.calculate_ra_dec_from_pixel(measurement_end.x(), measurement_end.y())
            
            if ra1 is not None and dec1 is not None and ra2 is not None and dec2 is not None:
                # Compute the angular distance
                angular_distance = self.parent.calculate_angular_distance(ra1, dec1, ra2, dec2)
                distance_text = self.parent.format_distance_as_dms(angular_distance)

                # Create and add the line item for display
                measurement_line_item = QGraphicsLineItem(QLineF(self.measurement_start, measurement_end))
                measurement_line_item.setPen(QPen(Qt.GlobalColor.green, 2, Qt.PenStyle.DashLine))
                self.parent.main_scene.addItem(measurement_line_item)

                # Create a midpoint position for the distance text
                midpoint = QPointF(
                    (self.measurement_start.x() + measurement_end.x()) / 2,
                    (self.measurement_start.y() + measurement_end.y()) / 2
                )

                # Create and add the text item at the midpoint
                text_item = QGraphicsTextItem(distance_text)
                text_item.setPos(midpoint)
                text_item.setDefaultTextColor(Qt.GlobalColor.green)
                text_item.setFont(self.parent.selected_font)  # Use the selected font
                self.parent.main_scene.addItem(text_item)

                # Store the line and text in annotation items for future reference
                measurement_line = QLineF(self.measurement_start, measurement_end)
                self.annotation_items.append(('line', measurement_line))  # Store QLineF, not QGraphicsLineItem
                self.annotation_items.append(('text', distance_text, midpoint, Qt.GlobalColor.green))

            # Clear the temporary measurement line item without removing the final line
            self.drawing_item = None



        elif self.drawing_item and event.button() == Qt.MouseButton.LeftButton:
            # Finalize the shape drawing and add its properties to annotation_items
            if isinstance(self.drawing_item, QGraphicsEllipseItem):
                rect = self.drawing_item.rect()
                color = self.drawing_item.pen().color()
                self.annotation_items.append(('ellipse', rect, color))
            elif isinstance(self.drawing_item, QGraphicsRectItem):
                rect = self.drawing_item.rect()
                color = self.drawing_item.pen().color()
                self.annotation_items.append(('rect', rect, color))
            elif isinstance(self.drawing_item, QGraphicsLineItem):
                line = self.drawing_item.line()
                color = self.drawing_item.pen().color()
                self.annotation_items.append(('line', line, color))
            elif isinstance(self.drawing_item, QGraphicsTextItem):
                pos = self.drawing_item.pos()
                text = self.drawing_item.toPlainText()
                color = self.drawing_item.defaultTextColor()
                self.annotation_items.append(('text', pos, text, color))
            elif isinstance(self.drawing_item, QGraphicsPathItem):  # Handle Freehand
                path = self.drawing_item.path()
                color = self.drawing_item.pen().color()
                self.annotation_items.append(('freehand', path, color))        

            # Clear the temporary drawing item
            self.drawing_item = None

        # Stop manual dragging and reset cursor to arrow
        self.dragging = False
        self.setCursor(Qt.CursorShape.ArrowCursor)
        
        # Update the mini preview to reflect any changes
        self.update_mini_preview()

        super().mouseReleaseEvent(event)


    def draw_measurement_line_and_label(self, distance_ddmmss):
        """Draw the measurement line and label with the celestial distance."""
        # Draw line
        line_item = QGraphicsLineItem(
            QLineF(self.measurement_start, self.measurement_end)
        )
        line_item.setPen(QPen(QColor(0, 255, 255), 2))  # Cyan color for measurement
        self.parent.main_scene.addItem(line_item)

        # Place distance text at the midpoint of the line
        midpoint = QPointF(
            (self.measurement_start.x() + self.measurement_end.x()) / 2,
            (self.measurement_start.y() + self.measurement_end.y()) / 2
        )
        text_item = QGraphicsTextItem(distance_ddmmss)
        text_item.setDefaultTextColor(QColor(0, 255, 255))  # Same color as line
        text_item.setPos(midpoint)
        self.parent.main_scene.addItem(text_item)
        
        # Append both line and text to annotation_items
        self.annotation_items.append(('line', line_item))
        self.annotation_items.append(('text', midpoint, distance_ddmmss, QColor(0, 255, 255)))


    
    def wheelEvent(self, event):
        """Handle zoom in and out with the mouse wheel."""
        if event.angleDelta().y() > 0:
            self.parent.zoom_in()
        else:
            self.parent.zoom_out()        

    def update_circle(self):
        """Draws the search circle on the main scene if circle_center and circle_radius are set."""
        if self.parent.main_image and self.circle_center is not None and self.circle_radius > 0:
            # Clear the main scene and add the main image back
            self.parent.main_scene.clear()
            self.parent._marker_layer = None  # <--- add this line
            self.parent.main_scene.addPixmap(self.parent.main_image)

            # Redraw all shapes and annotations from stored properties
            for item in self.annotation_items:
                if item[0] == 'ellipse':
                    rect = item[1]
                    color = item[2]
                    ellipse = QGraphicsEllipseItem(rect)
                    ellipse.setPen(QPen(color, 2))
                    self.parent.main_scene.addItem(ellipse)
                elif item[0] == 'rect':
                    rect = item[1]
                    color = item[2]
                    rect_item = QGraphicsRectItem(rect)
                    rect_item.setPen(QPen(color, 2))
                    self.parent.main_scene.addItem(rect_item)
                elif item[0] == 'line':
                    line = item[1]
                    color = item[2]
                    line_item = QGraphicsLineItem(line)
                    line_item.setPen(QPen(color, 2))
                    self.parent.main_scene.addItem(line_item)
                elif item[0] == 'text':
                    text = item[1]            # The text string
                    pos = item[2]             # A QPointF for the position
                    color = item[3]           # The color for the text

                    text_item = QGraphicsTextItem(text)
                    text_item.setPos(pos)
                    text_item.setDefaultTextColor(color)
                    text_item.setFont(self.parent.selected_font)
                    self.parent.main_scene.addItem(text_item)

                elif item[0] == 'freehand':  # Redraw Freehand
                    path = item[1]
                    color = item[2]
                    freehand_item = QGraphicsPathItem(path)
                    freehand_item.setPen(QPen(color, 2))
                    self.parent.main_scene.addItem(freehand_item)        

                elif item[0] == 'compass':
                    compass = item[1]
                    # North Line
                    north_line_coords = compass['north_line']
                    north_line_item = QGraphicsLineItem(
                        north_line_coords[0], north_line_coords[1], north_line_coords[2], north_line_coords[3]
                    )
                    north_line_item.setPen(QPen(Qt.GlobalColor.red, 2))
                    self.parent.main_scene.addItem(north_line_item)
                    
                    # East Line
                    east_line_coords = compass['east_line']
                    east_line_item = QGraphicsLineItem(
                        east_line_coords[0], east_line_coords[1], east_line_coords[2], east_line_coords[3]
                    )
                    east_line_item.setPen(QPen(Qt.GlobalColor.blue, 2))
                    self.parent.main_scene.addItem(east_line_item)
                    
                    # North Label
                    text_north = QGraphicsTextItem(compass['north_label'][2])
                    text_north.setPos(compass['north_label'][0], compass['north_label'][1])
                    text_north.setDefaultTextColor(Qt.GlobalColor.red)
                    self.parent.main_scene.addItem(text_north)
                    
                    # East Label
                    text_east = QGraphicsTextItem(compass['east_label'][2])
                    text_east.setPos(compass['east_label'][0], compass['east_label'][1])
                    text_east.setDefaultTextColor(Qt.GlobalColor.blue)
                    self.parent.main_scene.addItem(text_east)

                elif item[0] == 'measurement':  # Redraw celestial measurement line
                    line = item[1]
                    color = item[2]
                    text_position = item[3]
                    distance_text = item[4]
                    
                    # Draw the measurement line
                    measurement_line_item = QGraphicsLineItem(line)
                    measurement_line_item.setPen(QPen(color, 2, Qt.PenStyle.DashLine))  # Dashed line for measurement
                    self.parent.main_scene.addItem(measurement_line_item)
                    
                    # Draw the distance text label
                    text_item = QGraphicsTextItem(distance_text)
                    text_item.setPos(text_position)
                    text_item.setDefaultTextColor(color)
                    text_item.setFont(self.parent.selected_font)
                    self.parent.main_scene.addItem(text_item)                                
                        
            
            # Draw the search circle
            # >>> Recreate + repopulate the marker layer so query results persist
            self.parent._ensure_marker_layer()            # <--- add this line
            self.parent._set_marker_points_from_results() # <--- add this line

            # Draw the search circle
            pen_circle = QPen(QColor(255, 0, 0), 2)
            pen_circle.setCosmetic(True)                  # stays same width while zooming (optional)
            circle = self.parent.main_scene.addEllipse(
                int(self.circle_center.x() - self.circle_radius),
                int(self.circle_center.y() - self.circle_radius),
                int(self.circle_radius * 2),
                int(self.circle_radius * 2),
                pen_circle
            )
            circle.setZValue(10_000)                      # keep circle above markers/pixmap (optional)

            self.update_mini_preview()
        else:
            # If circle is disabled, restore base layers and markers too
            self.parent.main_scene.clear()
            self.parent._marker_layer = None              # <--- add this line
            self.parent.main_scene.addPixmap(self.parent.main_image)
            self.parent._ensure_marker_layer()            # <--- add this line
            self.parent._set_marker_points_from_results() # <--- add this line

    def delete_selected_object(self):
        if self.selected_object is None:
            self.parent.status_label.setText("No object selected to delete.")
            return

        # Remove the selected object from the results list
        self.parent.results = [obj for obj in self.parent.results if obj != self.selected_object]

        # Remove the corresponding row from the TreeBox
        for i in range(self.parent.results_tree.topLevelItemCount()):
            item = self.parent.results_tree.topLevelItem(i)
            if item.text(2) == self.selected_object["name"]:  # Match the name in the third column
                self.parent.results_tree.takeTopLevelItem(i)
                break

        # Clear the selection
        self.selected_object = None
        self.parent.results_tree.clearSelection()

        # Redraw the main and mini previews without the deleted marker
        self.draw_query_results()
        self.update_mini_preview()

        # Update the status label
        self.parent.status_label.setText("Selected object and marker removed.")

    def delete_selected_objects(self):
        items = self.parent.results_tree.selectedItems()
        if not items:
            self.parent.status_label.setText("No objects selected to delete.")
            return

        # Collect names from the selected rows (column 2 is "Name")
        names_to_delete = {it.text(2) for it in items if it.text(2)}

        # Remove from results
        before = len(self.parent.results)
        self.parent.results = [obj for obj in self.parent.results if obj.get("name") not in names_to_delete]
        after = len(self.parent.results)

        # Remove the rows from the tree (handle both top-level and children just in case)
        for it in items:
            parent = it.parent()
            if parent is None:
                idx = self.parent.results_tree.indexOfTopLevelItem(it)
                if idx != -1:
                    self.parent.results_tree.takeTopLevelItem(idx)
            else:
                parent.removeChild(it)

        # Clear selection & selected highlight
        self.selected_object = None
        if hasattr(self.parent, "_set_selected_name"):
            self.parent._set_selected_name(None)

        # Update marker layer + mini preview (no scene clears, so we don’t delete the layer)
        self.parent._ensure_marker_layer()
        self.parent._set_marker_points_from_results()
        if self.parent._marker_layer:
            self.parent._marker_layer.update()
        self.update_mini_preview()

        # Update counters / status
        self.parent.object_count_label.setText(f"Objects Found: {after}")
        removed = before - after
        self.parent.status_label.setText(f"Removed {removed} object(s).")


    def scrollContentsBy(self, dx, dy):
        """Called whenever the main preview scrolls, ensuring the green box updates in the mini preview."""
        super().scrollContentsBy(dx, dy)
        self.parent.update_green_box()

    def update_mini_preview(self):
        """Update the mini preview with the current view rectangle and any additional mirrored elements."""
        if self.parent.main_image:
            # Scale the main image to fit in the mini preview
            mini_pixmap = self.parent.main_image.scaled(
                self.parent.mini_preview.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            mini_painter = QPainter(mini_pixmap)

            try:
                # Define scale factors based on main image dimensions
                if self.parent.main_image.width() > 0 and self.parent.main_image.height() > 0:
                    scale_factor_x = mini_pixmap.width() / self.parent.main_image.width()
                    scale_factor_y = mini_pixmap.height() / self.parent.main_image.height()

                    # Draw the search circle if it's defined
                    if self.circle_center is not None and self.circle_radius > 0:
                        pen_circle = QPen(QColor(255, 0, 0), 2)
                        mini_painter.setPen(pen_circle)
                        mini_painter.drawEllipse(
                            int(self.circle_center.x() * scale_factor_x - self.circle_radius * scale_factor_x),
                            int(self.circle_center.y() * scale_factor_y - self.circle_radius * scale_factor_y),
                            int(self.circle_radius * 2 * scale_factor_x),
                            int(self.circle_radius * 2 * scale_factor_y)
                        )

                    # Draw the green box representing the current view
                    mini_painter.setPen(QPen(QColor(0, 255, 0), 2))
                    view_rect = self.parent.main_preview.mapToScene(
                        self.parent.main_preview.viewport().rect()
                    ).boundingRect()
                    mini_painter.drawRect(
                        int(view_rect.x() * scale_factor_x),
                        int(view_rect.y() * scale_factor_y),
                        int(view_rect.width() * scale_factor_x),
                        int(view_rect.height() * scale_factor_y)
                    )


                    # Draw dots for each result with a color based on selection status
                    for obj in self.parent.results:
                        ra, dec = obj['ra'], obj['dec']
                        x, y = self.parent.calculate_pixel_from_ra_dec(ra, dec)
                        if x is not None and y is not None:
                            # Change color to green if this is the selected object
                            dot_color = QColor(0, 255, 0) if obj == getattr(self.parent, 'selected_object', None) else QColor(255, 0, 0)
                            mini_painter.setPen(QPen(dot_color, 4))
                            mini_painter.drawPoint(
                                int(x * scale_factor_x),
                                int(y * scale_factor_y)
                            )

                    # Redraw annotation items on the mini preview
                    for item in self.annotation_items:
                        pen = QPen(self.parent.selected_color, 1)  # Use a thinner pen for mini preview
                        mini_painter.setPen(pen)

                        # Interpret item type and draw accordingly
                        if item[0] == 'ellipse':
                            rect = item[1]
                            mini_painter.drawEllipse(
                                int(rect.x() * scale_factor_x), int(rect.y() * scale_factor_y),
                                int(rect.width() * scale_factor_x), int(rect.height() * scale_factor_y)
                            )
                        elif item[0] == 'rect':
                            rect = item[1]
                            mini_painter.drawRect(
                                int(rect.x() * scale_factor_x), int(rect.y() * scale_factor_y),
                                int(rect.width() * scale_factor_x), int(rect.height() * scale_factor_y)
                            )
                        elif item[0] == 'line':
                            line = item[1]
                            mini_painter.drawLine(
                                int(line.x1() * scale_factor_x), int(line.y1() * scale_factor_y),
                                int(line.x2() * scale_factor_x), int(line.y2() * scale_factor_y)
                            )
                        elif item[0] == 'text':
                            text = item[1]            # The text string
                            pos = item[2]             # A QPointF for the position
                            color = item[3]           # The color for the text

                            # Create a smaller font for the mini preview
                            mini_font = QFont(self.parent.selected_font)
                            mini_font.setPointSize(int(self.parent.selected_font.pointSize() * 0.2))  # Scale down font size

                            mini_painter.setFont(mini_font)
                            mini_painter.setPen(color)  # Set the color for the text
                            mini_painter.drawText(
                                int(pos.x() * scale_factor_x), int(pos.y() * scale_factor_y),
                                text
                            )

                        elif item[0] == 'freehand':
                            # Scale the freehand path and draw it
                            path = item[1]
                            scaled_path = QPainterPath()
                            
                            # Scale each point in the path to fit the mini preview
                            for i in range(path.elementCount()):
                                point = path.elementAt(i)
                                if i == 0:
                                    scaled_path.moveTo(point.x * scale_factor_x, point.y * scale_factor_y)
                                else:
                                    scaled_path.lineTo(point.x * scale_factor_x, point.y * scale_factor_y)

                            mini_painter.drawPath(scaled_path)

                        elif item[0] == 'compass':
                            compass = item[1]
                            # Draw the North line
                            mini_painter.setPen(QPen(Qt.GlobalColor.red, 1))
                            north_line = compass["north_line"]
                            mini_painter.drawLine(
                                int(north_line[0] * scale_factor_x), int(north_line[1] * scale_factor_y),
                                int(north_line[2] * scale_factor_x), int(north_line[3] * scale_factor_y)
                            )

                            # Draw the East line
                            mini_painter.setPen(QPen(Qt.GlobalColor.blue, 1))
                            east_line = compass["east_line"]
                            mini_painter.drawLine(
                                int(east_line[0] * scale_factor_x), int(east_line[1] * scale_factor_y),
                                int(east_line[2] * scale_factor_x), int(east_line[3] * scale_factor_y)
                            )

                            # Draw North and East labels
                            mini_painter.setPen(QPen(Qt.GlobalColor.red, 1))
                            north_label = compass["north_label"]
                            mini_painter.drawText(
                                int(north_label[0] * scale_factor_x), int(north_label[1] * scale_factor_y), north_label[2]
                            )

                            mini_painter.setPen(QPen(Qt.GlobalColor.blue, 1))
                            east_label = compass["east_label"]
                            mini_painter.drawText(
                                int(east_label[0] * scale_factor_x), int(east_label[1] * scale_factor_y), east_label[2]
                            )                            

            finally:
                mini_painter.end()  # Ensure QPainter is properly ended

            self.parent.mini_preview.setPixmap(mini_pixmap)

    def place_celestial_compass(self, center):
        """Draw a celestial compass at a given point aligned with celestial North and East."""
        compass_radius = 50  # Length of the compass lines

        # Get the orientation in radians (assuming `self.parent.orientation` is in degrees)
        orientation_radians = math.radians(self.parent.orientation)

        # Calculate North vector (upwards, adjusted for orientation)
        north_dx = math.sin(orientation_radians) * compass_radius
        north_dy = -math.cos(orientation_radians) * compass_radius

        # Calculate East vector (rightwards, adjusted for orientation)
        east_dx = math.cos(orientation_radians) * -compass_radius
        east_dy = math.sin(orientation_radians) * -compass_radius

        # Draw North line
        north_line = QGraphicsLineItem(
            center.x(), center.y(),
            center.x() + north_dx, center.y() + north_dy
        )
        north_line.setPen(QPen(Qt.GlobalColor.red, 2))
        self.parent.main_scene.addItem(north_line)

        # Draw East line
        east_line = QGraphicsLineItem(
            center.x(), center.y(),
            center.x() + east_dx, center.y() + east_dy
        )
        east_line.setPen(QPen(Qt.GlobalColor.blue, 2))
        self.parent.main_scene.addItem(east_line)

        # Add labels for North and East
        text_north = QGraphicsTextItem("N")
        text_north.setDefaultTextColor(Qt.GlobalColor.red)
        text_north.setPos(center.x() + north_dx - 10, center.y() + north_dy - 10)
        self.parent.main_scene.addItem(text_north)

        text_east = QGraphicsTextItem("E")
        text_east.setDefaultTextColor(Qt.GlobalColor.blue)
        text_east.setPos(center.x() + east_dx - 15, center.y() + east_dy - 10)
        self.parent.main_scene.addItem(text_east)

        # Append all compass components as a tuple to annotation_items for later redrawing
        self.annotation_items.append((
            "compass", {
                "center": center,
                "north_line": (center.x(), center.y(), center.x() + north_dx, center.y() + north_dy),
                "east_line": (center.x(), center.y(), center.x() + east_dx, center.y() + east_dy),
                "north_label": (center.x() + north_dx - 10, center.y() + north_dy - 10, "N"),
                "east_label": (center.x() + east_dx - 15, center.y() + east_dy - 10, "E"),
                "orientation": self.parent.orientation
            }
        ))

    def _set_marker_points_from_results(self):
        self._ensure_marker_layer()
        if not _qt_is_alive(self._marker_layer):
            return
        pts = []
        for obj in getattr(self, "results", []):
            ra, dec = obj.get("ra"), obj.get("dec")
            xy = self.calculate_pixel_from_ra_dec(ra, dec)
            if not xy: continue
            x, y = xy
            if x is None or y is None: continue
            pts.append({"x": float(x), "y": float(y),
                        "name": obj.get("name"),
                        "color": obj.get("color", QColor(255,255,255))})
        try:
            self._marker_layer.set_points(pts)
        except RuntimeError:
            self._marker_layer = None
            self._ensure_marker_layer()
            if _qt_is_alive(self._marker_layer):
                self._marker_layer.set_points(pts)


    def zoom_to_coordinates(self, ra, dec):
        """Zoom to the specified RA/Dec coordinates and center the view on that position."""
        # Calculate the pixel position from RA and Dec
        pixel_x, pixel_y = self.parent.calculate_pixel_from_ra_dec(ra, dec)
        
        if pixel_x is not None and pixel_y is not None:
            # Center the view on the calculated pixel position
            self.centerOn(pixel_x, pixel_y)
            
            # Reset the zoom level to 1.0 by adjusting the transformation matrix
            self.resetTransform()
            self.scale(1.0, 1.0)

            # Optionally, update the mini preview to reflect the new zoom and center
            self.update_mini_preview()

    def draw_query_results(self):
        """Draw query results with or without names based on the show_names setting."""
        if self.parent.main_image:
            # Clear the main scene and re-add the main image
            self.parent.main_scene.clear()
            self.parent.main_scene.addPixmap(self.parent.main_image)

            self.parent._marker_layer = None
            self.parent._ensure_marker_layer()
            self.parent._set_marker_points_from_results()

            # Redraw all shapes and annotations from stored properties
            for item in self.annotation_items:
                if item[0] == 'ellipse':
                    rect = item[1]
                    color = item[2]
                    ellipse = QGraphicsEllipseItem(rect)
                    ellipse.setPen(QPen(color, 2))
                    self.parent.main_scene.addItem(ellipse)
                elif item[0] == 'rect':
                    rect = item[1]
                    color = item[2]
                    rect_item = QGraphicsRectItem(rect)
                    rect_item.setPen(QPen(color, 2))
                    self.parent.main_scene.addItem(rect_item)
                elif item[0] == 'line':
                    line = item[1]
                    color = item[2]
                    line_item = QGraphicsLineItem(line)
                    line_item.setPen(QPen(color, 2))
                    self.parent.main_scene.addItem(line_item)
                elif item[0] == 'text':
                    text = item[1]            # The text string
                    pos = item[2]             # A QPointF for the position
                    color = item[3]           # The color for the text

                    text_item = QGraphicsTextItem(text)
                    text_item.setPos(pos)
                    text_item.setDefaultTextColor(color)
                    text_item.setFont(self.parent.selected_font)
                    self.parent.main_scene.addItem(text_item)

                elif item[0] == 'freehand':  # Redraw Freehand
                    path = item[1]
                    color = item[2]
                    freehand_item = QGraphicsPathItem(path)
                    freehand_item.setPen(QPen(color, 2))
                    self.parent.main_scene.addItem(freehand_item)                      
                elif item[0] == 'measurement':  # Redraw celestial measurement line
                    line = item[1]
                    color = item[2]
                    text_position = item[3]
                    distance_text = item[4]
                    
                    # Draw the measurement line
                    measurement_line_item = QGraphicsLineItem(line)
                    measurement_line_item.setPen(QPen(color, 2, Qt.PenStyle.DashLine))  # Dashed line for measurement
                    self.parent.main_scene.addItem(measurement_line_item)
                    
                    # Draw the distance text label
                    text_item = QGraphicsTextItem(distance_text)
                    text_item.setPos(text_position)
                    text_item.setDefaultTextColor(color)
                    text_item.setFont(self.parent.selected_font)
                    self.parent.main_scene.addItem(text_item)        
                elif item[0] == 'compass':
                    compass = item[1]
                    # North Line
                    north_line_coords = compass['north_line']
                    north_line_item = QGraphicsLineItem(
                        north_line_coords[0], north_line_coords[1], north_line_coords[2], north_line_coords[3]
                    )
                    north_line_item.setPen(QPen(Qt.GlobalColor.red, 2))
                    self.parent.main_scene.addItem(north_line_item)
                    
                    # East Line
                    east_line_coords = compass['east_line']
                    east_line_item = QGraphicsLineItem(
                        east_line_coords[0], east_line_coords[1], east_line_coords[2], east_line_coords[3]
                    )
                    east_line_item.setPen(QPen(Qt.GlobalColor.blue, 2))
                    self.parent.main_scene.addItem(east_line_item)
                    
                    # North Label
                    text_north = QGraphicsTextItem(compass['north_label'][2])
                    text_north.setPos(compass['north_label'][0], compass['north_label'][1])
                    text_north.setDefaultTextColor(Qt.GlobalColor.red)
                    self.parent.main_scene.addItem(text_north)
                    
                    # East Label
                    text_east = QGraphicsTextItem(compass['east_label'][2])
                    text_east.setPos(compass['east_label'][0], compass['east_label'][1])
                    text_east.setDefaultTextColor(Qt.GlobalColor.blue)
                    self.parent.main_scene.addItem(text_east)                               
            # Ensure the search circle is drawn if circle data is available
            #if self.circle_center is not None and self.circle_radius > 0:
            #    self.update_circle()

            # Draw object markers (circle or crosshair)
            for obj in self.parent.results:
                ra, dec, name = obj["ra"], obj["dec"], obj["name"]
                x, y = self.parent.calculate_pixel_from_ra_dec(ra, dec)
                if x is not None and y is not None:
                    # Determine color: green if selected, red otherwise
                    base_color = obj["color"]
                    pen_color  = QColor(0,255,0) if obj is self.selected_object else base_color
                    pen = QPen(pen_color, 2)

                    if self.parent.marker_style == "Circle":
                        # Draw a circle around the object
                        self.parent.main_scene.addEllipse(int(x - 5), int(y - 5), 10, 10, pen)
                    elif self.parent.marker_style == "Crosshair":
                        # Draw crosshair with a 5-pixel gap in the middle
                        crosshair_size = 10
                        gap = 5
                        line1 = QLineF(x - crosshair_size, y, x - gap, y)
                        line2 = QLineF(x + gap, y, x + crosshair_size, y)
                        line3 = QLineF(x, y - crosshair_size, x, y - gap)
                        line4 = QLineF(x, y + gap, x, y + crosshair_size)
                        for line in [line1, line2, line3, line4]:
                            crosshair_item = QGraphicsLineItem(line)
                            crosshair_item.setPen(pen)
                            self.parent.main_scene.addItem(crosshair_item)
                    if self.parent.show_names:
                        #print(f"Drawing name: {name} at ({x}, {y})")  # Debugging statement
                        text_color = obj.get("color", QColor(Qt.GlobalColor.white))
                        text_item = QGraphicsTextItem(name)
                        text_item.setPos(x + 10, y + 10)  # Offset to avoid overlapping the marker
                        text_item.setDefaultTextColor(text_color)
                        text_item.setFont(self.parent.selected_font)
                        self.parent.main_scene.addItem(text_item)                            
    

    def clear_query_results(self):
        """Clear query markers from the main image without removing annotations."""
        # Clear the main scene and add the main image back
        self.parent.main_scene.clear()
        if self.parent.main_image:
            self.parent.main_scene.addPixmap(self.parent.main_image)

        self.parent._marker_layer = None
        self.parent._ensure_marker_layer()
        self.parent._set_marker_points_from_results()

        # Redraw the stored annotation items
        for item in self.annotation_items:
            if item[0] == 'ellipse':
                rect = item[1]
                color = item[2]
                ellipse = QGraphicsEllipseItem(rect)
                ellipse.setPen(QPen(color, 2))
                self.parent.main_scene.addItem(ellipse)
            elif item[0] == 'rect':
                rect = item[1]
                color = item[2]
                rect_item = QGraphicsRectItem(rect)
                rect_item.setPen(QPen(color, 2))
                self.parent.main_scene.addItem(rect_item)
            elif item[0] == 'line':
                line = item[1]
                color = item[2]
                line_item = QGraphicsLineItem(line)
                line_item.setPen(QPen(color, 2))
                self.parent.main_scene.addItem(line_item)
            elif item[0] == 'text':
                text = item[1]            # The text string
                pos = item[2]             # A QPointF for the position
                color = item[3]           # The color for the text

                text_item = QGraphicsTextItem(text)
                text_item.setPos(pos)
                text_item.setDefaultTextColor(color)
                text_item.setFont(self.parent.selected_font)
                self.parent.main_scene.addItem(text_item)

            elif item[0] == 'freehand':  # Redraw Freehand
                path = item[1]
                color = item[2]
                freehand_item = QGraphicsPathItem(path)
                freehand_item.setPen(QPen(color, 2))
                self.parent.main_scene.addItem(freehand_item)  
            elif item[0] == 'measurement':  # Redraw celestial measurement line
                line = item[1]
                color = item[2]
                text_position = item[3]
                distance_text = item[4]
                
                # Draw the measurement line
                measurement_line_item = QGraphicsLineItem(line)
                measurement_line_item.setPen(QPen(color, 2, Qt.PenStyle.DashLine))  # Dashed line for measurement
                self.parent.main_scene.addItem(measurement_line_item)
                
                # Draw the distance text label
                text_item = QGraphicsTextItem(distance_text)
                text_item.setPos(text_position)
                text_item.setDefaultTextColor(color)
                text_item.setFont(self.parent.selected_font)
                self.parent.main_scene.addItem(text_item)       
            elif item[0] == 'compass':
                compass = item[1]
                # North line
                north_line_item = QGraphicsLineItem(
                    compass['north_line'][0], compass['north_line'][1],
                    compass['north_line'][2], compass['north_line'][3]
                )
                north_line_item.setPen(QPen(Qt.GlobalColor.red, 2))
                self.parent.main_scene.addItem(north_line_item)
                # East line
                east_line_item = QGraphicsLineItem(
                    compass['east_line'][0], compass['east_line'][1],
                    compass['east_line'][2], compass['east_line'][3]
                )
                east_line_item.setPen(QPen(Qt.GlobalColor.blue, 2))
                self.parent.main_scene.addItem(east_line_item)
                # North label
                text_north = QGraphicsTextItem(compass['north_label'][2])
                text_north.setPos(compass['north_label'][0], compass['north_label'][1])
                text_north.setDefaultTextColor(Qt.GlobalColor.red)
                self.parent.main_scene.addItem(text_north)
                # East label
                text_east = QGraphicsTextItem(compass['east_label'][2])
                text_east.setPos(compass['east_label'][0], compass['east_label'][1])
                text_east.setDefaultTextColor(Qt.GlobalColor.blue)
                self.parent.main_scene.addItem(text_east)
        
        # Update the circle data, if any
        self.parent.update_circle_data()
                        

    def set_query_results(self, results):
        """Store results, assign each object a category & color, then redraw."""
        self.parent.results = results

        for obj in self.parent.results:
            # use the same key you populated in MainWindow.query_simbad
            short_type = obj.get("short_type", "")
            category   = OTYPE_TO_CATEGORY.get(short_type, "Errors & Artefacts")
            obj["category"] = category

            # lookup the QColor for that category (fallback to white)
            obj["color"]    = CATEGORY_TO_COLOR.get(category, QColor(255,255,255))

        self.draw_query_results()

    def get_object_at_position(self, pos):
        """Find the object at the given position in the main preview."""
        for obj in self.parent.results:
            ra, dec = obj["ra"], obj["dec"]
            x, y = self.parent.calculate_pixel_from_ra_dec(ra, dec)
            if x is not None and y is not None:
                if abs(pos.x() - x) <= 5 and abs(pos.y() - y) <= 5:
                    return obj
        return None


    def select_object(self, selected_obj):
        self.selected_object = selected_obj if self.selected_object != selected_obj else None
        sel_name = self.selected_object["name"] if self.selected_object else None
        # tell the dialog (and thus the MarkerLayer)
        self.parent._set_selected_name(sel_name)

        # Update the TreeWidget selection in MainWindow
        for i in range(self.parent.results_tree.topLevelItemCount()):
            item = self.parent.results_tree.topLevelItem(i)
            if item.text(2) == selected_obj["name"]:
                self.parent.results_tree.setCurrentItem(item if self.selected_object else None)
                break

    def undo_annotation(self):
        """Remove the last annotation item from the scene and annotation_items list."""
        if self.annotation_items:
            # Remove the last item from annotation_items
            self.annotation_items.pop()

            # Clear the scene and redraw all annotations except the last one
            self.parent.main_scene.clear()
            if self.parent.main_image:
                self.parent.main_scene.addPixmap(self.parent.main_image)

            # NEW:
            self.parent._marker_layer = None
            self.parent._ensure_marker_layer()
            self.parent._set_marker_points_from_results()

            # Redraw remaining annotations
            self.redraw_annotations()

            # Optionally, update the mini preview to reflect changes
            self.update_mini_preview()

    def clear_annotations(self):
        """Clear all annotation items from the scene and annotation_items list."""
        # Clear all items in annotation_items and update the scene
        self.annotation_items.clear()
        self.parent.main_scene.clear()

        self.parent._marker_layer = None
        self.parent._ensure_marker_layer()
        self.parent._set_marker_points_from_results()

        # Redraw only the main image
        if self.parent.main_image:
            self.parent.main_scene.addPixmap(self.parent.main_image)

        # Optionally, update the mini preview to reflect changes
        self.update_mini_preview()

    def redraw_annotations(self):
        """Helper function to redraw all annotations from annotation_items."""
        for item in self.annotation_items:
            if item[0] == 'ellipse':
                rect = item[1]
                color = item[2]
                ellipse = QGraphicsEllipseItem(rect)
                ellipse.setPen(QPen(color, 2))
                self.parent.main_scene.addItem(ellipse)
            elif item[0] == 'rect':
                rect = item[1]
                color = item[2]
                rect_item = QGraphicsRectItem(rect)
                rect_item.setPen(QPen(color, 2))
                self.parent.main_scene.addItem(rect_item)
            elif item[0] == 'line':
                line = item[1]
                color = item[2]
                line_item = QGraphicsLineItem(line)
                line_item.setPen(QPen(color, 2))
                self.parent.main_scene.addItem(line_item)
            elif item[0] == 'text':
                text = item[1]            # The text string
                pos = item[2]             # A QPointF for the position
                color = item[3]           # The color for the text

                text_item = QGraphicsTextItem(text)
                text_item.setPos(pos)
                text_item.setDefaultTextColor(color)
                text_item.setFont(self.parent.selected_font)
                self.parent.main_scene.addItem(text_item)

            elif item[0] == 'freehand':  # Redraw Freehand
                path = item[1]
                color = item[2]
                freehand_item = QGraphicsPathItem(path)
                freehand_item.setPen(QPen(color, 2))
                self.parent.main_scene.addItem(freehand_item) 
            elif item[0] == 'measurement':  # Redraw celestial measurement line
                line = item[1]
                color = item[2]
                text_position = item[3]
                distance_text = item[4]
                
                # Draw the measurement line
                measurement_line_item = QGraphicsLineItem(line)
                measurement_line_item.setPen(QPen(color, 2, Qt.PenStyle.DashLine))  # Dashed line for measurement
                self.parent.main_scene.addItem(measurement_line_item)
                
                # Draw the distance text label
                text_item = QGraphicsTextItem(distance_text)
                text_item.setPos(text_position)
                text_item.setDefaultTextColor(color)
                text_item.setFont(self.parent.selected_font)
                self.parent.main_scene.addItem(text_item)                                        
            elif item[0] == 'compass':
                compass = item[1]
                # Redraw north line
                north_line_item = QGraphicsLineItem(
                    compass['north_line'][0], compass['north_line'][1],
                    compass['north_line'][2], compass['north_line'][3]
                )
                north_line_item.setPen(QPen(Qt.GlobalColor.red, 2))
                self.parent.main_scene.addItem(north_line_item)
                
                # Redraw east line
                east_line_item = QGraphicsLineItem(
                    compass['east_line'][0], compass['east_line'][1],
                    compass['east_line'][2], compass['east_line'][3]
                )
                east_line_item.setPen(QPen(Qt.GlobalColor.blue, 2))
                self.parent.main_scene.addItem(east_line_item)
                
                # Redraw labels
                text_north = QGraphicsTextItem(compass['north_label'][2])
                text_north.setPos(compass['north_label'][0], compass['north_label'][1])
                text_north.setDefaultTextColor(Qt.GlobalColor.red)
                self.parent.main_scene.addItem(text_north)
                
                text_east = QGraphicsTextItem(compass['east_label'][2])
                text_east.setPos(compass['east_label'][0], compass['east_label'][1])
                text_east.setDefaultTextColor(Qt.GlobalColor.blue)
                self.parent.main_scene.addItem(text_east)        

class ThreeDSettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("3D Model Settings")
        layout = QVBoxLayout(self)

        # Image Plane Style
        layout.addWidget(QLabel("Image Plane Style:"))
        self.plane_style_cb = QComboBox()
        self.plane_style_cb.addItems([
            "Mesh RGB Scatter Plane",
            "Smooth Grayscale Image Plane"
        ])
        layout.addWidget(self.plane_style_cb)

        # Resolution
        res_layout = QHBoxLayout()
        res_layout.addWidget(QLabel("Resolution:"))
        self.res_spin = QSpinBox()
        self.res_spin.setRange(50, 2000)
        self.res_spin.setSingleStep(50)
        self.res_spin.setValue(500)
        res_layout.addWidget(self.res_spin)
        layout.addLayout(res_layout)

        # Z-Axis Range Options (Min-Max/Custom as before)
        layout.addWidget(QLabel("Z-Axis Range:"))
        self.zaxis_cb = QComboBox()
        self.zaxis_cb.addItems(["Default", "Min-Max", "Custom"])
        layout.addWidget(self.zaxis_cb)

        self.custom_widget = QWidget()
        cl = QHBoxLayout(self.custom_widget)
        cl.addWidget(QLabel("Min:"))
        self.zmin_spin = QDoubleSpinBox()
        self.zmin_spin.setRange(-1e6, 1e6)
        self.zmin_spin.setValue(0.0)
        cl.addWidget(self.zmin_spin)
        cl.addWidget(QLabel("Max:"))
        self.zmax_spin = QDoubleSpinBox()
        self.zmax_spin.setRange(-1e6, 1e6)
        self.zmax_spin.setValue(10.0)
        cl.addWidget(self.zmax_spin)
        layout.addWidget(self.custom_widget)
        self.custom_widget.setVisible(False)
        self.zaxis_cb.currentIndexChanged.connect(
            lambda idx: self.custom_widget.setVisible(self.zaxis_cb.currentText() == "Custom")
        )

        # Z-Axis Scale: Log vs Linear
        layout.addWidget(QLabel("Z-Axis Scale:"))
        self.zscale_cb = QComboBox()
        self.zscale_cb.addItems(["Logarithmic", "Linear"])
        layout.addWidget(self.zscale_cb)

        # Linear max input
        self.linear_widget = QWidget()
        ll = QHBoxLayout(self.linear_widget)
        ll.addWidget(QLabel("Linear Z-Max:"))
        self.linear_max_spin = QDoubleSpinBox()
        self.linear_max_spin.setRange(0.1, 1e12)
        self.linear_max_spin.setValue(1e4)
        ll.addWidget(self.linear_max_spin)
        layout.addWidget(self.linear_widget)
        self.linear_widget.setVisible(False)
        self.zscale_cb.currentIndexChanged.connect(
            lambda idx: self.linear_widget.setVisible(self.zscale_cb.currentText() == "Linear")
        )

        self.reverse_cb = QCheckBox("Reverse Z-Axis")
        self.reverse_cb.setChecked(False)
        layout.addWidget(self.reverse_cb)

        # Object Color
        layout.addWidget(QLabel("Object Color:"))
        self.color_cb = QComboBox()
        self.color_cb.addItems(["Image-Based", "Legend Color", "Solid (Custom)"])
        layout.addWidget(self.color_cb)

        self.color_btn = QPushButton("Choose Color…")
        self.custom_color = QColor(255, 0, 0)
        self.color_btn.setVisible(False)
        layout.addWidget(self.color_btn)
        self.color_btn.clicked.connect(self._choose_color)
        self.color_cb.currentIndexChanged.connect(
            lambda idx: self.color_btn.setVisible(self.color_cb.currentText() == "Solid (Custom)")
        )

        # Z-Axis Height control
        layout.addWidget(QLabel("Z-Axis Height (aspect ratio z):"))
        self.zheight_spin = QDoubleSpinBox()
        self.zheight_spin.setRange(0.1, 10.0)
        self.zheight_spin.setSingleStep(0.1)
        self.zheight_spin.setValue(0.5)
        layout.addWidget(self.zheight_spin)

        # ─── Show Connector Lines ─────────────────────────────
        self.lines_cb = QCheckBox("Show Connector Lines")
        self.lines_cb.setChecked(True)
        layout.addWidget(self.lines_cb)

        # OK / Cancel
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _choose_color(self):
        col = QColorDialog.getColor(self.custom_color, self, "Select Object Color")
        if col.isValid():
            self.custom_color = col

    def getSettings(self):
        if self.exec() == QDialog.DialogCode.Accepted:
            return {
                "plane_style": self.plane_style_cb.currentText(),
                "resolution": self.res_spin.value(),
                "z_option":   self.zaxis_cb.currentText(),
                "z_min":      self.zmin_spin.value(),
                "z_max":      self.zmax_spin.value(),
                "z_scale":    self.zscale_cb.currentText(),
                "linear_max": self.linear_max_spin.value(),
                "object_color": self.color_cb.currentText(),
                "custom_color": self.custom_color,      # QColor
                "z_height":   self.zheight_spin.value(),
                "show_lines": self.lines_cb.isChecked(),
                "reverse_z":   self.reverse_cb.isChecked(),
            }
        return None

class HRSettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("H-R Diagram Settings")
        layout = QVBoxLayout(self)

        # 1) Star Color Mode
        layout.addWidget(QLabel("Star Color Mode:"))
        self.color_mode_cb = QComboBox()
        self.color_mode_cb.addItems(["Realistic (blackbody)", "Solid (Custom)"])
        layout.addWidget(self.color_mode_cb)

        self.color_btn = QPushButton("Choose Solid Color…")
        self.custom_color = QColor(255, 255, 255)
        self.color_btn.setVisible(False)
        layout.addWidget(self.color_btn)
        self.color_btn.clicked.connect(self._choose_color)
        self.color_mode_cb.currentIndexChanged.connect(
            lambda idx: self.color_btn.setVisible(
                self.color_mode_cb.currentText().startswith("Solid")
            )
        )

        # 2) Background Choice
        layout.addWidget(QLabel("Background:"))
        self.bg_mode_cb = QComboBox()
        self.bg_mode_cb.addItems(["HR Diagram Image", "Solid Black"])
        layout.addWidget(self.bg_mode_cb)

        # 3) Axis Range Mode
        layout.addWidget(QLabel("Axis Range:"))
        self.range_mode_cb = QComboBox()
        self.range_mode_cb.addItems(["Default (–0.3→2.25, –9→19)", "Custom"])
        layout.addWidget(self.range_mode_cb)

        self.custom_range_widget = QWidget()
        cr_layout = QHBoxLayout(self.custom_range_widget)
        cr_layout.addWidget(QLabel("X Min:"))
        self.xmin_spin = QDoubleSpinBox()
        self.xmin_spin.setRange(-10.0, 10.0)
        self.xmin_spin.setDecimals(3)
        self.xmin_spin.setValue(-0.3)
        cr_layout.addWidget(self.xmin_spin)

        cr_layout.addWidget(QLabel("X Max:"))
        self.xmax_spin = QDoubleSpinBox()
        self.xmax_spin.setRange(-10.0, 10.0)
        self.xmax_spin.setDecimals(3)
        self.xmax_spin.setValue(2.25)
        cr_layout.addWidget(self.xmax_spin)

        cr_layout.addWidget(QLabel("Y Min:"))
        self.ymin_spin = QDoubleSpinBox()
        self.ymin_spin.setRange(-50.0, 50.0)
        self.ymin_spin.setDecimals(3)
        self.ymin_spin.setValue(-9.0)
        cr_layout.addWidget(self.ymin_spin)

        cr_layout.addWidget(QLabel("Y Max:"))
        self.ymax_spin = QDoubleSpinBox()
        self.ymax_spin.setRange(-50.0, 50.0)
        self.ymax_spin.setDecimals(3)
        self.ymax_spin.setValue(19.0)
        cr_layout.addWidget(self.ymax_spin)

        layout.addWidget(self.custom_range_widget)
        self.custom_range_widget.setVisible(False)
        self.range_mode_cb.currentIndexChanged.connect(
            lambda idx: self.custom_range_widget.setVisible(
                self.range_mode_cb.currentText().startswith("Custom")
            )
        )

        layout.addWidget(QLabel("Show Sun:"))
        self.show_sun_cb = QCheckBox("Include Sun on diagram")
        self.show_sun_cb.setChecked(True)       # default ON
        layout.addWidget(self.show_sun_cb)

        # 4) OK / Cancel Buttons
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _choose_color(self):
        col = QColorDialog.getColor(self.custom_color, self, "Select Marker Color")
        if col.isValid():
            self.custom_color = col

    def getSettings(self):
        """
        Pops up the dialog. Returns a dict:
        {
            "color_mode":    "Realistic (blackbody)" or "Solid (Custom)",
            "custom_color":  QColor,
            "bg_mode":       "HR Diagram Image" or "Solid Black",
            "range_mode":    "Default" or "Custom",
            "x_min":         float,
            "x_max":         float,
            "y_min":         float,
            "y_max":         float
        }
        or None if the user canceled.
        """
        if self.exec() == QDialog.DialogCode.Accepted:
            return {
                "color_mode": self.color_mode_cb.currentText(),
                "custom_color": self.custom_color,
                "bg_mode": self.bg_mode_cb.currentText(),
                "range_mode": self.range_mode_cb.currentText(),
                "x_min": self.xmin_spin.value(),
                "x_max": self.xmax_spin.value(),
                "y_min": self.ymin_spin.value(),
                "y_max": self.ymax_spin.value(),
                "show_sun":      self.show_sun_cb.isChecked(),
            }
        return None

class LegendDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Object Type Legend")
        self.swatches = {}
        layout = QVBoxLayout(self)

        # Build one row per category
        for category, color in CATEGORY_TO_COLOR.items():
            row = QHBoxLayout()

            # color swatch
            swatch = QLabel()
            swatch.setFixedSize(16, 16)
            swatch.setStyleSheet(f"background-color: {color.name()}; border:1px solid #000;")
            row.addWidget(swatch)
            self.swatches[category] = swatch

            # category name
            row.addWidget(QLabel(category))

            # edit‐color button
            btn = QPushButton("Edit…")
            btn.clicked.connect(lambda _, cat=category: self.change_color(cat))
            row.addWidget(btn)

            row.addStretch()
            layout.addLayout(row)

        # OK / Cancel buttons
        btn_row = QHBoxLayout()
        ok = QPushButton("OK")
        ok.clicked.connect(self.accept)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        btn_row.addStretch()
        btn_row.addWidget(ok)
        btn_row.addWidget(cancel)
        layout.addLayout(btn_row)

    def change_color(self, category):
        """Open a QColorDialog, update the swatch and CATEGORY_TO_COLOR."""
        initial = CATEGORY_TO_COLOR[category]
        c = QColorDialog.getColor(initial, self, f"Select color for {category}")
        if c.isValid():
            CATEGORY_TO_COLOR[category] = c
            sw = self.swatches[category]
            sw.setStyleSheet(f"background-color: {c.name()}; border:1px solid #000;")

def kelvin_to_rgb(T):
    """Approximate conversion from black‐body temperature (K) to sRGB tuple."""
    # based on Tanner Helland's approximation
    # Clamp input
    T = max(1000, min(T, 40000)) / 100.0
    # red
    if T <= 66:
        r = 255
    else:
        r = 329.698727446 * ((T - 60) ** -0.1332047592)
    # green
    if T <= 66:
        g = 99.4708025861 * math.log(T) - 161.1195681661
    else:
        g = 288.1221695283 * ((T - 60) ** -0.0755148492)
    # blue
    if T >= 66:
        b = 255
    elif T <= 19:
        b = 0
    else:
        b = 138.5177312231 * math.log(T - 10) - 305.0447927307

    # clamp and return CSS‐style rgb()
    def clamp(x): return int(max(0, min(x, 255)))
    return f"rgb({clamp(r)},{clamp(g)},{clamp(b)})"

def resource_path(relative_path):
    """ Get the absolute path to the resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temporary folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

# Import centralized widgets
from pro.widgets.spinboxes import CustomSpinBox, CustomDoubleSpinBox


class _NoopSignal:
    def emit(self, *_, **__): pass

class _WIMIAdapterDoc:
    """
    Minimal doc shim so plate_solve_doc_inplace can write into .metadata.
    """
    def __init__(self, image: np.ndarray, metadata: dict | None = None):
        self.image = image
        self.metadata = metadata if isinstance(metadata, dict) else {}
        self.changed = _NoopSignal()

import numpy.ma as ma

def _mask_safe_float(val):
    """
    Convert SIMBAD / Astropy table values to a normal float or None.
    - Returns None for masked, None, or NaN.
    """
    if val is None:
        return None
    if ma.is_masked(val):
        return None
    try:
        f = float(val)
    except Exception:
        return None
    if np.isnan(f):
        return None
    return f

class MinorBodyDownloadWorker(QThread):
    """
    Background worker to download/update the minor-body catalog.

    Emits:
        finished_ok(db_path: str, version: str, asteroids: int, comets: int)
        failed(error_message: str)
    """
    finished_ok = pyqtSignal(str, str, int, int)   # db_path, version, ast, com
    failed = pyqtSignal(str)

    def __init__(self, data_dir: Path, force_refresh: bool = False, parent=None):
        super().__init__(parent)
        self.data_dir = Path(data_dir)
        self.force_refresh = force_refresh

    def run(self):
        try:
            from pro import minorbodycatalog as mbc
            db_path, manifest = mbc.ensure_minor_body_db(
                data_dir=self.data_dir,
                manifest_url=mbc.MANIFEST_URL,
                force_refresh=self.force_refresh,
            )
            # Emit db path, version and counts so the UI can show nice stats
            self.finished_ok.emit(
                str(db_path),
                str(manifest.version),
                int(manifest.counts_asteroids),
                int(manifest.counts_comets),
            )
        except Exception as e:
            self.failed.emit(str(e))


class WIMIDialog(QDialog):
    def __init__(self, parent=None, settings=None, doc_manager=None, wimi_path: Optional[str] = None, wrench_path: Optional[str] = None):
        super().__init__(parent)
        self.setWindowTitle("What's In My Image")
        if wimi_path:
            self.setWindowIcon(QIcon(wimi_path))

        self.settings = settings or QSettings()
        self._doc_manager = doc_manager  # <— we’ll use this to list “views”

        # Track the theme status
        self.is_dark_mode = True
        self.metadata = {}
        self.circle_center = None
        self.circle_radius = 0    
        self.show_names = False  # Boolean to toggle showing names on the main image
        self.max_results = 100  # Default maximum number of query results     
        self.current_tool = None  # Track the active annotation tool
        self.header = Header()
        # store optional original FITS header when available (pre-seed for plate solver)
        self.original_header = None
        self.marker_style = "Circle" 
        self.settings = QSettings() 
            

        main_layout = QHBoxLayout()

        # Left Column Layout
        left_panel = QVBoxLayout()

        # Load the image using the resource_path function
        wimilogo_path = resource_path("wimilogo.png")

        # Create a QLabel to display the logo
        self.logo_label = QLabel()

        # Set the logo image to the label
        logo_pixmap = QPixmap(wimilogo_path)

        # Scale the pixmap to fit within a desired size, maintaining the aspect ratio
        scaled_pixmap = logo_pixmap.scaled(100, 50, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)

        # Set the scaled pixmap to the label
        self.logo_label.setPixmap(scaled_pixmap)

        # Set alignment to center the logo horizontally
        self.logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Optionally, you can set a fixed size for the label (this is for layout purposes)
        #self.logo_label.setFixedSize(200, 100)  # Adjust the size as needed

        # Add the logo_label to your layout
        left_panel.addWidget(self.logo_label)
       
        button_layout = QHBoxLayout()
        
        # Load button
        self.load_button = QPushButton("Load Image File")
        self.load_button.setIcon(QApplication.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogStart))
        self.load_button.clicked.connect(self.open_image)

        self.load_from_view_btn = QToolButton()
        self.load_from_view_btn.setText("Load from View")
        self.load_from_view_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.load_from_view_btn.setIcon(QApplication.style().standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon))
        self.load_from_view_menu = QMenu(self)
        self.load_from_view_btn.setMenu(self.load_from_view_menu)
        self.load_from_view_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.load_from_view_menu.aboutToShow.connect(self._refresh_views_menu)
        button_layout.addWidget(self.load_from_view_btn)

        # AutoStretch button
        self.auto_stretch_button = QPushButton("AutoStretch")
        self.auto_stretch_button.clicked.connect(self.toggle_autostretch)

        # Add both buttons to the horizontal layout
        button_layout.addWidget(self.load_button)
        button_layout.addWidget(self.auto_stretch_button)

        # Add the button layout to the left panel
        left_panel.addLayout(button_layout)

        # Create the instruction QLabel for search region
        search_region_instruction_label = QLabel("Shift+Click to define a search region")
        search_region_instruction_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        search_region_instruction_label.setStyleSheet("font-size: 15px; color: gray;")

        # Add this QLabel to your layout at the appropriate position above RA/Dec
        left_panel.addWidget(search_region_instruction_label)  



        # Query Simbad button
        self.query_button = QPushButton("Query Simbad")
        self.query_button.setIcon(QApplication.style().standardIcon(QStyle.StandardPixmap.SP_DialogApplyButton))
        left_panel.addWidget(self.query_button)
        self.query_button.clicked.connect(lambda: self.query_simbad(self.get_defined_radius()))

        self.legend_button = QPushButton("Legend")
        self.legend_button.clicked.connect(self.show_legend)
        left_panel.addWidget(self.legend_button)

        # Create a horizontal layout for the show names checkbox and clear results button
        show_clear_layout = QHBoxLayout()

        # Create the Show Object Names checkbox
        self.show_names_checkbox = QCheckBox("Show Object Names")
        self.show_names_checkbox.stateChanged.connect(self.toggle_object_names)  # Connect to a function to toggle names
        show_clear_layout.addWidget(self.show_names_checkbox)

        # Create the Clear Results button
        self.clear_results_button = QPushButton("Clear Results")
        self.clear_results_button.setIcon(QApplication.style().standardIcon(QStyle.StandardPixmap.SP_DialogCloseButton))
        self.clear_results_button.clicked.connect(self.clear_search_results)  # Connect to a function to clear results
        show_clear_layout.addWidget(self.clear_results_button)

        # Add this horizontal layout to the left panel layout (or wherever you want it to appear)
        left_panel.addLayout(show_clear_layout)   

        # Create a horizontal layout for the two buttons
        button_layout = QHBoxLayout()

        # Show Visible Objects Only button
        self.toggle_visible_objects_button = QPushButton("Show Visible Objects Only")
        self.toggle_visible_objects_button.setCheckable(True)  # Toggle button state
        self.toggle_visible_objects_button.setIcon(QIcon(eye_icon_path))
        self.toggle_visible_objects_button.clicked.connect(self.filter_visible_objects)
        self.toggle_visible_objects_button.setToolTip("Toggle the visibility of objects based on brightness.")
        button_layout.addWidget(self.toggle_visible_objects_button)

        # Save CSV button
        self.save_csv_button = QPushButton("Save CSV")
        self.save_csv_button.setIcon(QIcon(csv_icon_path))
        self.save_csv_button.clicked.connect(self.save_results_as_csv)
        button_layout.addWidget(self.save_csv_button)

        # Add the button layout to the left panel or main layout
        left_panel.addLayout(button_layout)  

        # Advanced Search Button
        self.advanced_search_button = QPushButton("Advanced Search")
        self.advanced_search_button.setIcon(QApplication.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogDetailedView))
        self.advanced_search_button.setCheckable(True)
        self.advanced_search_button.clicked.connect(self.toggle_advanced_search)
        left_panel.addWidget(self.advanced_search_button)

        # Advanced Search Panel (initially hidden)
        self.advanced_search_panel = QVBoxLayout()
        self.advanced_search_panel_widget = QWidget()
        self.advanced_search_panel_widget.setLayout(self.advanced_search_panel)
        self.advanced_search_panel_widget.setFixedWidth(300)
        self.advanced_search_panel_widget.setVisible(False)  # Hide initially        

        # Status label
        self.status_label = QLabel("Status: Ready")
        left_panel.addWidget(self.status_label)

        # Create a horizontal layout
        button_layout = QHBoxLayout()

        # Copy RA/Dec to Clipboard button
        self.copy_button = QPushButton("Copy RA/Dec to Clipboard", self)
        self.copy_button.setIcon(QApplication.style().standardIcon(QStyle.StandardPixmap.SP_CommandLink))
        self.copy_button.clicked.connect(self.copy_ra_dec_to_clipboard)
        button_layout.addWidget(self.copy_button)

        # Settings button (wrench icon)
        self.settings_button = QPushButton()
        self.settings_button.setIcon(QIcon(wrench_path))  # Adjust icon path as needed
        self.settings_button.clicked.connect(self.open_settings_dialog)
        button_layout.addWidget(self.settings_button)

        # Add the horizontal layout to the main layout or the desired parent layout
        left_panel.addLayout(button_layout)
        
         # Save Plate Solved Fits Button
        self.save_plate_solved_button = QPushButton("Save Plate Solved Fits")
        self.save_plate_solved_button.setIcon(QIcon(disk_icon_path))
        self.save_plate_solved_button.clicked.connect(self.save_plate_solved_fits)
        left_panel.addWidget(self.save_plate_solved_button)       

        # RA/Dec Labels
        ra_dec_layout = QHBoxLayout()
        self.ra_label = QLabel("RA: N/A")
        self.dec_label = QLabel("Dec: N/A")
        self.orientation_label = QLabel("Orientation: N/A°")
        ra_dec_layout.addWidget(self.ra_label)
        ra_dec_layout.addWidget(self.dec_label)
        ra_dec_layout.addWidget(self.orientation_label)
        left_panel.addLayout(ra_dec_layout)

        # Mini Preview
        self.mini_preview = QLabel("Mini Preview")
        self.mini_preview.setMaximumSize(300, 300)
        self.mini_preview.mousePressEvent = self.on_mini_preview_press
        self.mini_preview.mouseMoveEvent = self.on_mini_preview_drag
        self.mini_preview.mouseReleaseEvent = self.on_mini_preview_release
        left_panel.addWidget(self.mini_preview)

  


        # Right Column Layout
        right_panel = QVBoxLayout()

        # Zoom buttons above the main preview
        zoom_controls_layout = QHBoxLayout()
        self.zoom_in_button = QPushButton("Zoom In")
        self.zoom_in_button.clicked.connect(self.zoom_in)
        self.zoom_out_button = QPushButton("Zoom Out")
        self.zoom_out_button.clicked.connect(self.zoom_out)
        zoom_controls_layout.addWidget(self.zoom_in_button)
        zoom_controls_layout.addWidget(self.zoom_out_button)
        right_panel.addLayout(zoom_controls_layout)        

        # Main Preview
        self.main_preview = CustomGraphicsView(self)
        self.main_scene = QGraphicsScene(self.main_preview)
        self.main_preview.setScene(self.main_scene)
        self.main_scene.setItemIndexMethod(QGraphicsScene.ItemIndexMethod.BspTreeIndex)
        self.main_preview.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.SmartViewportUpdate)
        self.main_preview.setOptimizationFlag(QGraphicsView.OptimizationFlag.DontSavePainterState, True)
        self.main_preview.setOptimizationFlag(QGraphicsView.OptimizationFlag.DontAdjustForAntialiasing, True)
        self.main_preview.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        #try:
        #    from PyQt6.QtOpenGLWidgets import QOpenGLWidget  # optional boost
        #    self.main_preview.setViewport(QOpenGLWidget())
        #except Exception:
        #    pass

        # Our marker layer handle
        self._marker_layer = None

        # ---- monkey-patch existing calls on CustomGraphicsView to our fast paths ----
        self.main_preview.set_query_results = lambda results: self._cg_set_query_results_proxy(results)
        self.main_preview.draw_query_results = lambda: self._cg_draw_query_results_proxy()
        self.main_preview.clear_query_results = lambda: self._cg_clear_query_results_proxy()
        self.main_preview.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        right_panel.addWidget(self.main_preview)

        # Save Annotated Image and Save Collage of Objects Buttons in a Horizontal Layout between main image and treebox
        save_buttons_layout = QHBoxLayout()

        # Button to toggle annotation tools section
        self.show_annotations_button = QPushButton("Show Annotation Tools")
        self.show_annotations_button.setIcon(QApplication.style().standardIcon(QStyle.StandardPixmap.SP_DialogResetButton))
        self.show_annotations_button.clicked.connect(self.toggle_annotation_tools)
        save_buttons_layout.addWidget(self.show_annotations_button)
        
        self.save_annotated_button = QPushButton("Save Annotated Image")
        self.save_annotated_button.setIcon(QIcon(annotated_path))
        self.save_annotated_button.clicked.connect(self.save_annotated_image)
        save_buttons_layout.addWidget(self.save_annotated_button)
        
        self.save_collage_button = QPushButton("Save Collage of Objects")
        self.save_collage_button.setIcon(QIcon(collage_path))
        self.save_collage_button.clicked.connect(self.save_collage_of_objects)
        save_buttons_layout.addWidget(self.save_collage_button)

        # New 3D View Button
        self.show_3d_view_button = QPushButton("3D Distance Model")
        self.show_3d_view_button.clicked.connect(self.show_3d_model_view)
        self.show_3d_view_button.setIcon(    QApplication.style().standardIcon(QStyle.StandardPixmap.SP_TitleBarNormalButton))
        save_buttons_layout.addWidget(self.show_3d_view_button)

        self.show_hr_button = QPushButton("H-R Diagram")
        # Optionally give it an icon:
        self.show_hr_button.setIcon(QApplication.style().standardIcon(
            QStyle.StandardPixmap.SP_DesktopIcon))
        self.show_hr_button.clicked.connect(self.show_hr_diagram)
        save_buttons_layout.addWidget(self.show_hr_button)

        right_panel.addLayout(save_buttons_layout)        

        # Connect scroll events to update the green box in the mini preview
        self.main_preview.verticalScrollBar().valueChanged.connect(self.main_preview.update_mini_preview)
        self.main_preview.horizontalScrollBar().valueChanged.connect(self.main_preview.update_mini_preview)

        # Create a horizontal layout for the labels
        label_layout = QHBoxLayout()

        # Create the label to display the count of objects
        self.object_count_label = QLabel("Objects Found: 0")

        # Create the label with instructions
        self.instructions_label = QLabel("Right Click a Row for More Options")

        # Add both labels to the horizontal layout
        label_layout.addWidget(self.object_count_label)
        label_layout.addWidget(self.instructions_label)

        # Add the horizontal layout to the main panel layout
        right_panel.addLayout(label_layout)

        self.results_tree = QTreeWidget()
        self.results_tree.setHeaderLabels(["RA", "Dec", "Name", "Diameter", "Type", "Long Type", "Redshift/Parallax (z/mas)", "Comoving Radial Distance (GLy)"])
        self.results_tree.setFixedHeight(150)
        self.results_tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.results_tree.customContextMenuRequested.connect(self.open_context_menu)
        self.results_tree.itemClicked.connect(self.on_tree_item_clicked)
        self.results_tree.itemDoubleClicked.connect(self.on_tree_item_double_clicked)
        self.results_tree.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.results_tree.setSortingEnabled(True)
        right_panel.addWidget(self.results_tree)

        self.annotation_buttons = []

        # Annotation Tools Section (initially hidden)
        self.annotation_tools_section = QWidget()
        annotation_tools_layout = QGridLayout(self.annotation_tools_section)

        annotation_instruction_label = QLabel("Ctrl+Click to add items, Alt+Click to measure distance")
        annotation_instruction_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        annotation_instruction_label.setStyleSheet("font-size: 10px; color: gray;")        

        self.draw_ellipse_button = QPushButton("Draw Ellipse")
        self.draw_ellipse_button.tool_name = "Ellipse"
        self.draw_ellipse_button.clicked.connect(lambda: self.set_tool("Ellipse"))
        self.annotation_buttons.append(self.draw_ellipse_button)

        self.freehand_button = QPushButton("Freehand (Lasso)")
        self.freehand_button.tool_name = "Freehand"
        self.freehand_button.clicked.connect(lambda: self.set_tool("Freehand"))
        self.annotation_buttons.append(self.freehand_button)

        self.draw_rectangle_button = QPushButton("Draw Rectangle")
        self.draw_rectangle_button.tool_name = "Rectangle"
        self.draw_rectangle_button.clicked.connect(lambda: self.set_tool("Rectangle"))
        self.annotation_buttons.append(self.draw_rectangle_button)

        self.draw_arrow_button = QPushButton("Draw Arrow")
        self.draw_arrow_button.tool_name = "Arrow"
        self.draw_arrow_button.clicked.connect(lambda: self.set_tool("Arrow"))
        self.annotation_buttons.append(self.draw_arrow_button)

        self.place_compass_button = QPushButton("Place Celestial Compass")
        self.place_compass_button.tool_name = "Compass"
        self.place_compass_button.clicked.connect(lambda: self.set_tool("Compass"))
        self.annotation_buttons.append(self.place_compass_button)

        self.add_text_button = QPushButton("Add Text")
        self.add_text_button.tool_name = "Text"
        self.add_text_button.clicked.connect(lambda: self.set_tool("Text"))
        self.annotation_buttons.append(self.add_text_button)

        # Add Color and Font buttons
        self.color_button = QPushButton("Select Color")
        self.color_button.setIcon(QIcon(colorwheel_path))
        self.color_button.clicked.connect(self.select_color)

        self.font_button = QPushButton("Select Font")
        self.font_button.setIcon(QIcon(font_path))
        self.font_button.clicked.connect(self.select_font)

        # Undo button
        self.undo_button = QPushButton("Undo")
        self.undo_button.setIcon(QApplication.style().standardIcon(QStyle.StandardPixmap.SP_ArrowLeft))  # Left arrow icon for undo
        self.undo_button.clicked.connect(self.main_preview.undo_annotation)  # Connect to undo_annotation in CustomGraphicsView

        # Clear Annotations button
        self.clear_annotations_button = QPushButton("Clear Annotations")
        self.clear_annotations_button.setIcon(QApplication.style().standardIcon(QStyle.StandardPixmap.SP_TrashIcon))  # Trash icon
        self.clear_annotations_button.clicked.connect(self.main_preview.clear_annotations)  # Connect to clear_annotations in CustomGraphicsView

        # Delete Selected Object button
        self.delete_selected_object_button = QPushButton("Delete Selected Object(s)")
        self.delete_selected_object_button.setIcon(QApplication.style().standardIcon(QStyle.StandardPixmap.SP_DialogCloseButton))  # Trash icon
        self.delete_selected_object_button.clicked.connect(self.main_preview.delete_selected_objects)

        # Add the instruction label to the top of the grid layout (row 0, spanning multiple columns)
        annotation_tools_layout.addWidget(annotation_instruction_label, 0, 0, 1, 4)  # Span 5 columns to center it

        # Shift all other widgets down by one row
        annotation_tools_layout.addWidget(self.draw_ellipse_button, 1, 0)
        annotation_tools_layout.addWidget(self.freehand_button, 1, 1)
        annotation_tools_layout.addWidget(self.draw_rectangle_button, 2, 0)
        annotation_tools_layout.addWidget(self.draw_arrow_button, 2, 1)
        annotation_tools_layout.addWidget(self.place_compass_button, 3, 0)
        annotation_tools_layout.addWidget(self.add_text_button, 3, 1)
        annotation_tools_layout.addWidget(self.color_button, 4, 0)
        annotation_tools_layout.addWidget(self.font_button, 4, 1)
        annotation_tools_layout.addWidget(self.undo_button, 1, 4)
        annotation_tools_layout.addWidget(self.clear_annotations_button, 2, 4)
        annotation_tools_layout.addWidget(self.delete_selected_object_button, 3, 4)

        self.annotation_tools_section.setVisible(False)  # Initially hidden
        right_panel.addWidget(self.annotation_tools_section)

        # Advanced Search Panel
        self.advanced_param_label = QLabel("Advanced Search Parameters")
        self.advanced_search_panel.addWidget(self.advanced_param_label)

        # TreeWidget for object types
        self.object_tree = QTreeWidget()
        self.object_tree.setHeaderLabels(["Object Type", "Description"])
        self.object_tree.setColumnWidth(0, 150)
        self.object_tree.setSortingEnabled(True)

        # Populate the TreeWidget with object types from otype_long_name_lookup
        for obj_type, description in otype_long_name_lookup.items():
            item = QTreeWidgetItem([obj_type, description])
            item.setCheckState(0, Qt.CheckState.Checked)  # Start with all items unchecked
            self.object_tree.addTopLevelItem(item)

        self.advanced_search_panel.addWidget(self.object_tree)

        # Buttons for toggling selections
        toggle_buttons_layout = QHBoxLayout()

        # Toggle All
        self.toggle_all_button = QPushButton("Toggle All")
        self.toggle_all_button.clicked.connect(self.toggle_all_items)
        toggle_buttons_layout.addWidget(self.toggle_all_button)

        # Save Custom List
        self.save_list_button = QPushButton("Save List…")
        self.save_list_button.clicked.connect(self.save_custom_list)
        toggle_buttons_layout.addWidget(self.save_list_button)

        # Load Custom List
        self.load_list_button = QPushButton("Load List…")
        self.load_list_button.clicked.connect(self.load_custom_list)
        toggle_buttons_layout.addWidget(self.load_list_button)

        self.advanced_search_panel.addLayout(toggle_buttons_layout)   

        # Add Simbad Search buttons below the toggle buttons
        search_button_layout = QHBoxLayout()

        self.simbad_defined_region_button = QPushButton("Search Defined Region")
        self.simbad_defined_region_button.clicked.connect(self.search_defined_region)
        search_button_layout.addWidget(self.simbad_defined_region_button)

        self.simbad_entire_image_button = QPushButton("Search Entire Image")
        self.simbad_entire_image_button.clicked.connect(self.search_entire_image)
        search_button_layout.addWidget(self.simbad_entire_image_button)

        self.advanced_search_panel.addLayout(search_button_layout)

        # ─────────────────────────────
        # Minor Planets / Comets block
        # ─────────────────────────────
        self.minor_group = QGroupBox("Minor Planets / Comets")
        minor_layout = QGridLayout(self.minor_group)

        # --- DB info + buttons ---
        self.minor_db_label = QLabel("Database: not downloaded")
        self.minor_db_label.setStyleSheet("font-size: 10px; color: gray;")

        self.btn_minor_download = QPushButton("Download Catalog")
        self.btn_minor_download.clicked.connect(self.download_minor_body_catalog)

        self.btn_minor_search = QPushButton("Search Minor Bodies")
        self.btn_minor_search.clicked.connect(self.perform_minor_body_search)

        # Row 0: status label across full width
        minor_layout.addWidget(self.minor_db_label, 0, 0, 1, 4)

        # Row 1: download + search buttons
        minor_layout.addWidget(self.btn_minor_download, 1, 0, 1, 2)
        minor_layout.addWidget(self.btn_minor_search,   1, 2, 1, 2)

        # --- Search scope (Defined Circle vs Entire Image) ---
        scope_label = QLabel("Search scope:")
        self.minor_scope_combo = QComboBox()
        self.minor_scope_combo.addItems([
            "Defined Region",
            "Entire Image",
        ])

        minor_layout.addWidget(scope_label,          2, 0)
        minor_layout.addWidget(self.minor_scope_combo, 2, 1, 1, 3)

        # --- Limits row 1: asteroid H_max + max count ---
        ast_H_label = QLabel("Asteroid H \u2264")
        self.minor_ast_H_spin = QDoubleSpinBox()
        self.minor_ast_H_spin.setRange(0.0, 40.0)
        self.minor_ast_H_spin.setDecimals(1)
        self.minor_ast_H_spin.setSingleStep(0.5)
        self.minor_ast_H_spin.setValue(
            float(self.settings.value("wimi/minor/asteroid_H_max", 20.0))
        )

        ast_max_label = QLabel("Max asteroids:")
        self.minor_ast_max_spin = QSpinBox()
        self.minor_ast_max_spin.setRange(100, 2000000)
        self.minor_ast_max_spin.setSingleStep(1000)
        self.minor_ast_max_spin.setValue(
            int(self.settings.value("wimi/minor/asteroid_max", 20000))
        )

        minor_layout.addWidget(ast_H_label,          3, 0)
        minor_layout.addWidget(self.minor_ast_H_spin, 3, 1)
        minor_layout.addWidget(ast_max_label,        3, 2)
        minor_layout.addWidget(self.minor_ast_max_spin, 3, 3)

        # --- Limits row 2: comet H_max + max count ---
        com_H_label = QLabel("Comet H \u2264")
        self.minor_com_H_spin = QDoubleSpinBox()
        self.minor_com_H_spin.setRange(0.0, 40.0)
        self.minor_com_H_spin.setDecimals(1)
        self.minor_com_H_spin.setSingleStep(0.5)
        self.minor_com_H_spin.setValue(
            float(self.settings.value("wimi/minor/comet_H_max", 15.0))
        )

        com_max_label = QLabel("Max comets:")
        self.minor_com_max_spin = QSpinBox()
        self.minor_com_max_spin.setRange(100, 100000)
        self.minor_com_max_spin.setSingleStep(500)
        self.minor_com_max_spin.setValue(
            int(self.settings.value("wimi/minor/comet_max", 5000))
        )

        minor_layout.addWidget(com_H_label,          4, 0)
        minor_layout.addWidget(self.minor_com_H_spin, 4, 1)
        minor_layout.addWidget(com_max_label,        4, 2)
        minor_layout.addWidget(self.minor_com_max_spin, 4, 3)

        # --- Optional specific target (designation / name) ---
        target_label = QLabel("Target (optional):")
        self.minor_target_edit = QLineEdit()
        self.minor_target_edit.setPlaceholderText("e.g. 584, Semiramis, C/2023 A3...")

        minor_layout.addWidget(target_label,          5, 0)
        minor_layout.addWidget(self.minor_target_edit, 5, 1, 1, 3)

        self.advanced_search_panel.addWidget(self.minor_group)


        # Try to pick up an already-downloaded DB on startup
        self._load_minor_db_path()

        # Adding the "Deep Vizier Search" button below the other search buttons
        self.deep_vizier_button = QPushButton("Caution - Deep Vizier Search")
        self.deep_vizier_button.setIcon(QIcon(nuke_path))  # Assuming `nuke_path` is the correct path for the icon
        self.deep_vizier_button.setToolTip("Perform a deep search with Vizier. Caution: May return large datasets.")

        # Connect the button to a placeholder method for the deep Vizier search
        self.deep_vizier_button.clicked.connect(self.perform_deep_vizier_search)

        # Add the Deep Vizier button to the advanced search layout
        self.advanced_search_panel.addWidget(self.deep_vizier_button)

        self.mast_search_button = QPushButton("Search M.A.S.T Database")
        self.mast_search_button.setIcon(QIcon(hubble_path))
        self.mast_search_button.clicked.connect(self.perform_mast_search)
        self.mast_search_button.setToolTip("Search Hubble, JWST, Spitzer, TESS and More.")
        self.advanced_search_panel.addWidget(self.mast_search_button)                        

        # Combine left and right panels
        main_layout.addLayout(left_panel)
        main_layout.addLayout(right_panel)
        main_layout.addWidget(self.advanced_search_panel_widget)
        
        delete_shortcut = QShortcut(QKeySequence.StandardKey.Delete, self.results_tree)
        delete_shortcut.activated.connect(self.main_preview.delete_selected_objects)

        self.setLayout(main_layout)

        self.image_path = None
        self.zoom_level = 1.0
        self.main_image = None
        self.green_box = None
        self.dragging = False
        self.center_ra = None
        self.center_dec = None
        self.pixscale = None
        self.orientation = None
        self.parity = None  
        self.circle_center = None
        self.circle_radius = 0  
        self.results = []
        self._selected_name = None
        self.wcs = None  # Initialize WCS to None
        # Initialize selected color and font with default values
        self.selected_color = QColor(Qt.GlobalColor.red)  # Default annotation color
        self.selected_font = QFont("Arial", 12)  # Default font for text annotations   
        self.populate_object_tree()     
        # Minor-planet bits
        # Minor-body initial state
        self.minor_db_path = self.settings.value(
            "wimi/minorbody_db_path", "", type=str
        )
        self._load_minor_db_path()
        if self.minor_db_path:
            p = Path(self.minor_db_path)
            if p.is_file():
                # Try to read local manifest for nicer label
                try:
                    data_dir = p.parent
                    manifest_path = data_dir / mbc.DEFAULT_MANIFEST_BASENAME
                    manifest = mbc.load_local_manifest(manifest_path)
                    if manifest is not None:
                        ast = f"{manifest.counts_asteroids:,}"
                        com = f"{manifest.counts_comets:,}"
                        self.minor_db_label.setText(
                            f"Database: v{manifest.version} — {ast} asteroids, {com} comets"
                        )
                    else:
                        self.minor_db_label.setText(f"Database: {p.name}")
                except Exception:
                    self.minor_db_label.setText(f"Database: {p.name}")
            else:
                self.minor_db_label.setText("Database: not downloaded")
        else:
            self.minor_db_label.setText("Database: not downloaded")      
        #self._legend_dock = QDockWidget("Object Type Legend", self)
        legend = LegendDialog(self)
        legend.setModal(False)
    

    def toggle_object_names(self, state=None):
        self.show_names = self.show_names_checkbox.isChecked() if state is None else bool(state)
        self._ensure_marker_layer()
        if self._marker_layer:
            self._marker_layer.setCacheMode(QGraphicsItem.CacheMode.NoCache)
            self._marker_layer.update(self._marker_layer.boundingRect())
            self.main_scene.invalidate(self.main_scene.sceneRect(), QGraphicsScene.SceneLayer.AllLayers)
            self._marker_layer.setCacheMode(QGraphicsItem.CacheMode.DeviceCoordinateCache)

    def _set_selected_name(self, name: Optional[str]):
        self._selected_name = name
        if self._marker_layer:
            self._marker_layer.update()

    def _ensure_marker_layer(self):
        # need a scene and a pixmap
        if getattr(self, "main_scene", None) is None or self.main_image is None:
            return

        w = int(self.main_image.width())
        h = int(self.main_image.height())

        # drop dead/dangling layers
        if getattr(self, "_marker_layer", None) is not None and not _qt_is_alive(self._marker_layer):
            self._marker_layer = None

        if getattr(self, "_marker_layer", None) is None:
            self._marker_layer = MarkerLayer(
                image_w=w, image_h=h,
                show_names_fn=lambda: self.show_names,
                color_fn=lambda: self.selected_color,
                style_fn=lambda: self.marker_style,
                selected_name_fn=lambda: self._selected_name
            )
            self.main_scene.addItem(self._marker_layer)
        else:
            self._marker_layer.resize(w, h)

    def _get_selected_name(self):
        return self._selected_name

    def _set_marker_points_from_results(self):
        if self._marker_layer is None or self.wcs is None:
            return
        pts = []
        for obj in self.results:
            ra, dec = obj.get('ra'), obj.get('dec')
            if ra is None or dec is None:
                continue
            x, y = self.calculate_pixel_from_ra_dec(ra, dec)
            if x is None or y is None:
                continue
            name_val = obj.get("name")
            if not name_val:
                name_val = obj.get("Name")  # fallback if key differs
            pts.append({
                "x": x, "y": y,
                "name": str(name_val or ""),
                "color": obj.get("color")
            })
        self._marker_layer.set_points(pts)

    def _set_marker_points_from_results(self):
        self._ensure_marker_layer()
        if not _qt_is_alive(self._marker_layer):
            return
        pts = []
        for obj in getattr(self, "results", []):
            ra, dec = obj.get("ra"), obj.get("dec")
            xy = self.calculate_pixel_from_ra_dec(ra, dec)
            if not xy: continue
            x, y = xy
            if x is None or y is None: continue
            pts.append({"x": float(x), "y": float(y),
                        "name": obj.get("name"),
                        "color": obj.get("color", QColor(255,255,255))})
        try:
            self._marker_layer.set_points(pts)
        except RuntimeError:
            self._marker_layer = None
            self._ensure_marker_layer()
            if _qt_is_alive(self._marker_layer):
                self._marker_layer.set_points(pts)

    # ---- drop-in replacements (proxies) that your existing code already calls ----
    def _cg_set_query_results_proxy(self, results):
        # Recreate the color/category tagging that used to happen in CustomGraphicsView.set_query_results
        for obj in results:
            short_type = obj.get("short_type", "")
            category = OTYPE_TO_CATEGORY.get(short_type, "Errors & Artefacts")
            obj["category"] = category
            obj["color"] = CATEGORY_TO_COLOR.get(category, QColor(255, 255, 255))
        self.results = results

        self._ensure_marker_layer()
        self._set_marker_points_from_results()

    def _cg_draw_query_results_proxy(self):
        """Keeps existing call sites working: self.main_preview.draw_query_results()"""
        self._ensure_marker_layer()
        # Re-derive points (cheap) in case self.results changed elsewhere
        self._set_marker_points_from_results()
        if self._marker_layer:
            self._marker_layer.update()

    def _cg_clear_query_results_proxy(self):
        """Keeps existing call sites working: self.main_preview.clear_query_results()"""
        self.results = []
        if self._marker_layer:
            self._marker_layer.set_points([])


    def _doc_for_solver(self):
        # Prefer the real pro document if we loaded from a view
        if getattr(self, "_loaded_doc", None) is not None:
            return self._loaded_doc

        # Otherwise build an adapter around the image currently in WIMI
        if getattr(self, "image_data", None) is None:
            return None

        meta = {}
        # pack whatever we already know so ASTAP can seed if possible
        if getattr(self, "original_header", None):
            meta["original_header"] = self.original_header
        if getattr(self, "wcs", None) is not None:
            try:
                meta["wcs_header"] = self.wcs.to_header(relax=True)
            except Exception:
                pass
        return _WIMIAdapterDoc(self.image_data, meta)


    def show_legend(self):
        # keep a persistent reference so it doesn't get garbage-collected
        if not hasattr(self, "_legend_dialog"):
            self._legend_dialog = LegendDialog(self)
            self._legend_dialog.setModal(False)
        self._legend_dialog.show()
        self._legend_dialog.raise_()
        self._legend_dialog.activateWindow()

    def _refresh_views_menu(self):
        self.load_from_view_menu.clear()
        docs = []
        try:
            if self._doc_manager is not None:
                docs = list(self._doc_manager.all_documents())
        except Exception:
            docs = []

        if not docs:
            a = self.load_from_view_menu.addAction("No open views")
            a.setEnabled(False)
            return

        # most-recent last → show newest at bottom (or reverse if you prefer)
        for doc in docs:
            title = getattr(doc, "display_name", lambda: "Untitled")()
            act = self.load_from_view_menu.addAction(title)
            act.triggered.connect(lambda _, d=doc: self._load_from_view(d))

    def _ensure_wcs_on_doc(self, doc) -> bool:
        """
        Return True if WCS already present or successfully solved and written back to doc.metadata.
        Uses the shared plate solver (ASTAP first, then Astrometry.net) and shows status in the UI.
        If the document has no usable seed header, temporarily forces a blind solve for this run.
        """
        meta = getattr(doc, "metadata", {}) or {}
        # already solved?
        if self.check_astrometry_data(meta.get("original_header")) or \
        self.check_astrometry_data(meta.get("wcs_header")):
            return True

        # decide whether to force blind for this one run (no seed header)
        seed_hdr = _as_header(meta.get("original_header") or meta.get("wcs_header"))
        force_blind = (seed_hdr is None)

        mw = _find_main_window(self) or self.parent()
        settings = getattr(mw, "settings", QSettings())

        # status hint
        try:
            if hasattr(self, "status_label"):
                self.status_label.setText("Status: Solving (ASTAP)…")
                QApplication.processEvents()
        except Exception:
            pass

        prev_mode = _get_seed_mode(settings)
        try:
            if force_blind and prev_mode != "none":
                _set_seed_mode(settings, "none")  # blind just for this solve
            ok, _ = plate_solve_doc_inplace(mw, doc, settings)
        finally:
            # restore user preference
            if force_blind and prev_mode != "none":
                _set_seed_mode(settings, prev_mode)

        if ok:
            # double-check we now have WCS on the doc
            meta = getattr(doc, "metadata", {}) or {}
            got = self.check_astrometry_data(meta.get("original_header")) or \
                self.check_astrometry_data(meta.get("wcs_header"))
            try:
                if hasattr(self, "status_label"):
                    self.status_label.setText("Status: Solve complete.")
            except Exception:
                pass
            return bool(got)

        # unified solver already tried ASTAP and Astrometry.net; just report failure
        try:
            QMessageBox.critical(self, "Solve Failed",
                                "Automatic plate solve failed (ASTAP → Astrometry.net).")
            if hasattr(self, "status_label"):
                self.status_label.setText("Status: Solve failed.")
        except Exception:
            pass
        return False

    def populate_object_tree(self):
        self.object_tree.blockSignals(True)
        self.object_tree.clear()

        # 1) Build reverse map: category → list of short codes
        cat_to_types = defaultdict(list)

        # Pre-sort patterns so more specific (longer) come first
        patterns = list(OTYPE_TO_CATEGORY.items())
        patterns.sort(key=lambda x: len(x[0]), reverse=True)

        for code, longname in otype_long_name_lookup.items():
            # first try exact
            cat = OTYPE_TO_CATEGORY.get(code)
            if cat is None:
                # then try wildcard patterns (but skip the lone "*" pattern)
                for pat, candidate_cat in patterns:
                    if any(c in pat for c in "*?") and pat != "*" and fnmatch.fnmatch(code, pat):
                        cat = candidate_cat
                        break
            if cat is None:
                cat = "Errors & Artefacts"
            cat_to_types[cat].append(code)

        # 2) Populate tree
        for category, codes in cat_to_types.items():
            color  = CATEGORY_TO_COLOR.get(category, QColor(200,200,200))
            parent = QTreeWidgetItem(self.object_tree, [category, ""])
            parent.setFlags(parent.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            parent.setCheckState(0, Qt.CheckState.Checked)
            parent.setForeground(0, QBrush(color))
            parent.setFirstColumnSpanned(True)

            for code in sorted(codes):
                desc  = otype_long_name_lookup[code]
                child = QTreeWidgetItem(parent, [code, desc])
                child.setFlags(child.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                child.setCheckState(0, Qt.CheckState.Checked)
                pix = QPixmap(12,12)
                pix.fill(color)
                child.setIcon(0, QIcon(pix))

        self.object_tree.blockSignals(False)

        # wire up a very simple parent→children handler (no partial logic)
        try:
            self.object_tree.itemChanged.disconnect(self.on_object_tree_item_changed)
        except TypeError:
            pass
        self.object_tree.itemChanged.connect(self.on_object_tree_item_changed)


    def get_selected_object_types(self) -> list:
        """
        Return all the otype codes (children) whose checkboxes are checked.
        """
        checked = []
        root = self.object_tree.invisibleRootItem()
        # iterate over each category
        for i in range(root.childCount()):
            category_item = root.child(i)
            # iterate over that category's children
            for j in range(category_item.childCount()):
                child = category_item.child(j)
                if child.checkState(0) == Qt.CheckState.Checked:
                    checked.append(child.text(0))
        return checked


    def update_object_count(self):
        count = self.results_tree.topLevelItemCount()
        self.object_count_label.setText(f"Objects Found: {count}")

    def open_context_menu(self, position):
        
        # Get the item at the mouse position
        item = self.results_tree.itemAt(position)
        if not item:
            return  # If no item is clicked, do nothing
        
        self.on_tree_item_clicked(item)

        # Create the context menu
        menu = QMenu(self)

        # Define actions
        open_website_action = QAction("Open Website", self)
        open_website_action.triggered.connect(lambda: self.results_tree.itemDoubleClicked.emit(item, 0))
        menu.addAction(open_website_action)

        zoom_to_object_action = QAction("Zoom to Object", self)
        zoom_to_object_action.triggered.connect(lambda: self.zoom_to_object(item))
        menu.addAction(zoom_to_object_action)

        copy_info_action = QAction("Copy Object Information", self)
        copy_info_action.triggered.connect(lambda: self.copy_object_information(item))
        menu.addAction(copy_info_action)

        # Display the context menu at the cursor position
        menu.exec(self.results_tree.viewport().mapToGlobal(position))

    def toggle_autostretch(self):
        if not hasattr(self, 'original_image'):
            # Store the original image the first time AutoStretch is applied
            self.original_image = self.image_data.copy()
        
        # Determine if the image is mono or color based on the number of dimensions
        if self.image_data.ndim == 2:
            # Call stretch_mono_image if the image is mono

            stretched_image = stretch_mono_image(self.image_data, target_median=0.25, normalize=True)
        else:
            # Call stretch_color_image if the image is color

            stretched_image = stretch_color_image(self.image_data, target_median=0.25, linked=True, normalize=True)
        
        # If the AutoStretch is toggled off (using the same button), restore the original image
        if self.auto_stretch_button.text() == "AutoStretch":
            # Store the stretched image and update the button text to indicate it's on
            self.stretched_image = stretched_image
            self.auto_stretch_button.setText("Turn Off AutoStretch")
        else:
            # Revert to the original image and update the button text to indicate it's off
            stretched_image = self.original_image
            self.auto_stretch_button.setText("AutoStretch")
        

        stretched_image = (stretched_image * 255).astype(np.uint8)


        # Update the display with the stretched image (or original if toggled off)

        height, width = stretched_image.shape[:2]
        bytes_per_line = 3 * width

        # Ensure the image has 3 channels (RGB)
        if stretched_image.ndim == 2:
            stretched_image = np.stack((stretched_image,) * 3, axis=-1)
        elif stretched_image.shape[2] == 1:
            stretched_image = np.repeat(stretched_image, 3, axis=2)



        qimg = QImage(stretched_image.tobytes(), width, height, bytes_per_line, QImage.Format.Format_RGB888)
        if qimg.isNull():
            print("Failed to create QImage")
            return

        pixmap = QPixmap.fromImage(qimg)
        if pixmap.isNull():
            print("Failed to create QPixmap")
            return

        self.main_image = pixmap
        scaled_pixmap = pixmap.scaled(self.mini_preview.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.mini_preview.setPixmap(scaled_pixmap)

        self.main_scene.clear()
        self.main_scene.addPixmap(pixmap)
        self._marker_layer = None
        self._ensure_marker_layer()
        self._set_marker_points_from_results()        
        self.main_preview.setSceneRect(QRectF(pixmap.rect()))
        self.zoom_level = 1.0
        self.main_preview.resetTransform()
        self.main_preview.centerOn(self.main_scene.sceneRect().center())
        self.update_green_box()

        # Optionally, you can also update any other parts of the UI after stretching the image
        print(f"AutoStretch {'applied to' if self.auto_stretch_button.text() == 'Turn Off AutoStretch' else 'removed from'} the image.")


    def zoom_to_object(self, item):
        """Zoom to the object in the main preview."""
        ra = float(item.text(0))  # Assuming RA is in the first column
        dec = float(item.text(1))  # Assuming Dec is in the second column
        self.main_preview.zoom_to_coordinates(ra, dec)
        

    def copy_object_information(self, item):
        """Copy object information to the clipboard."""
        info = f"RA: {item.text(0)}, Dec: {item.text(1)}, Name: {item.text(2)}, Diameter: {item.text(3)}, Type: {item.text(4)}"
        clipboard = QApplication.clipboard()
        clipboard.setText(info)

    def set_tool(self, tool_name):
        """Sets the current tool and updates button states."""
        self.current_tool = tool_name

        # Reset button styles and highlight the selected button
        for button in self.annotation_buttons:
            if button.tool_name == tool_name:
                button.setStyleSheet("background-color: lightblue;")  # Highlight selected button
            else:
                button.setStyleSheet("")  # Reset other buttons


    def select_color(self):
        """Opens a color dialog to choose annotation color."""
        color = QColorDialog.getColor(self.selected_color, self, "Select Annotation Color")
        if color.isValid():
            self.selected_color = color

    def select_font(self):
        """Opens a font dialog to choose text annotation font."""
        font, ok = QFontDialog.getFont(self.selected_font, self, "Select Annotation Font")
        if ok:
            self.selected_font = font                

    def toggle_annotation_tools(self):
        """Toggle the visibility of the annotation tools section."""
        is_visible = self.annotation_tools_section.isVisible()
        self.annotation_tools_section.setVisible(not is_visible)
        self.show_annotations_button.setText("Hide Annotation Tools" if not is_visible else "Show Annotation Tools")

    def save_plate_solved_fits(self):
        bit_depth, ok = QInputDialog.getItem(
            self, "Select Bit Depth", "Choose the bit depth for the FITS file:",
            ["8-bit", "16-bit", "32-bit"], 0, False
        )
        if not ok: return
        out_path, _ = QFileDialog.getSaveFileName(self, "Save Plate Solved FITS", "", "FITS Files (*.fits *.fit)")
        if not out_path: return
        if self.wcs is None:
            QMessageBox.warning(self, "WCS Data Missing", "WCS header data is not available.")
            return

        img = self.image_data
        # Prepare image data, preserving color if present.
        img = img.astype(np.float32)
        is_color = False
        if img.ndim == 2:
            src = img
        else:
            # If image has >=3 channels, keep the first 3 as RGB
            if img.shape[2] >= 3:
                src = img[..., :3]
                is_color = True
            else:
                # fallback to first plane
                src = img[..., 0]

        # If data seems to be in 0-255 range, scale to 0-1
        try:
            maxv = float(np.nanmax(src))
        except Exception:
            maxv = 0.0
        if maxv > 1.1:
            src = src / 255.0

        # Clip to [0,1]
        src = np.clip(src, 0.0, 1.0)

        # Convert to desired bit depth and arrange axes for multichannel data
        if bit_depth == "8-bit":
            if is_color:
                arr = (src * 255.0).astype(np.uint8)
                # FITS expects axes: (plane, y, x)
                arr = np.transpose(arr, (2, 0, 1))
            else:
                arr = (src * 255.0).astype(np.uint8)
        elif bit_depth == "16-bit":
            if is_color:
                arr = (src * 65535.0).astype(np.uint16)
                arr = np.transpose(arr, (2, 0, 1))
            else:
                arr = (src * 65535.0).astype(np.uint16)
        else:  # 32-bit float
            if is_color:
                arr = src.astype(np.float32)
                arr = np.transpose(arr, (2, 0, 1))
            else:
                arr = src.astype(np.float32)

        hdr = (self.original_header.copy() if getattr(self, 'original_header', None) else fits.Header())
        try:
            hdr.update(self.wcs.to_header(relax=True))
        except Exception:
            pass  # still save whatever we have

        # Let astropy infer NAXIS/BITPIX from data; don’t set them manually
        try:
            fits.writeto(out_path, arr, header=hdr, overwrite=True)
            QMessageBox.information(self, "File Saved", f"FITS file saved as {out_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save FITS file: {e}")




    def save_annotated_image(self):
        """Save the annotated image as a full or cropped view, excluding the search circle."""
        # Create a custom message box
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Save Annotated Image")
        msg_box.setText("Do you want to save the Full Image or Cropped Only?")
        
        # Add custom buttons
        full_image_button = msg_box.addButton("Save Full", QMessageBox.ButtonRole.AcceptRole)
        cropped_image_button = msg_box.addButton("Save Cropped", QMessageBox.ButtonRole.DestructiveRole)
        msg_box.addButton(QMessageBox.StandardButton.Cancel)

        # Show the message box and get the user's response
        msg_box.exec()

        # Determine the save type based on the selected button
        if msg_box.clickedButton() == full_image_button:
            save_full_image = True
        elif msg_box.clickedButton() == cropped_image_button:
            save_full_image = False
        else:
            return  # User cancelled

        # Open a file dialog to select the file name and format
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Annotated Image",
            "",
            "JPEG (*.jpg *.jpeg);;PNG (*.png);;TIFF (*.tiff *.tif)"
        )
        
        if not file_path:
            return  # User cancelled the save dialog

        # Temporarily disable the search circle in the custom graphics view
        original_circle_center = self.main_preview.circle_center
        original_circle_radius = self.main_preview.circle_radius
        self.main_preview.circle_center = None  # Hide the circle temporarily
        self.main_preview.circle_radius = 0

        # Redraw annotations without the search circle
        self.main_preview.draw_query_results()

        # Create a QPixmap to render the annotations
        if save_full_image:
            # Save the entire main image with annotations
            pixmap = QPixmap(self.main_image.size())
            pixmap.fill(Qt.GlobalColor.transparent)
            painter = QPainter(pixmap)
            self.main_scene.render(painter)  # Render the entire scene without the search circle
        else:
            # Save only the currently visible area (cropped view)
            rect = self.main_preview.viewport().rect()
            scene_rect = self.main_preview.mapToScene(rect).boundingRect()
            pixmap = QPixmap(int(scene_rect.width()), int(scene_rect.height()))
            pixmap.fill(Qt.GlobalColor.transparent)
            painter = QPainter(pixmap)
            self.main_scene.render(painter, QRectF(0, 0, pixmap.width(), pixmap.height()), scene_rect)

        painter.end()  # End QPainter to finalize drawing

        # Restore the search circle in the custom graphics view
        self.main_preview.circle_center = original_circle_center
        self.main_preview.circle_radius = original_circle_radius
        self.main_preview.draw_query_results()  # Redraw the scene with the circle

        # Save the QPixmap as an image file in the selected format
        try:
            if pixmap.save(file_path, file_path.split('.')[-1].upper()):
                QMessageBox.information(self, "Save Successful", f"Annotated image saved as {file_path}")
            else:
                raise Exception("Failed to save image due to format or file path issues.")
        except Exception as e:
            QMessageBox.critical(self, "Save Failed", f"An error occurred while saving the image: {str(e)}")


    def save_collage_of_objects(self):
        """Save a collage of 128x128 pixel patches centered around each object, with dynamically spaced text below."""
        # Options for display
        options = ["Name", "RA", "Dec", "Short Type", "Long Type", "Redshift", "Comoving Distance"]

        # Create a custom dialog to select information to display
        dialog = QDialog(self)
        dialog.setWindowTitle("Select Information to Display")
        layout = QVBoxLayout(dialog)
        
        # Add checkboxes for each option
        checkboxes = {}
        for option in options:
            checkbox = QCheckBox(option)
            checkbox.setChecked(True)  # Default to checked
            layout.addWidget(checkbox)
            checkboxes[option] = checkbox

        # Add OK and Cancel buttons
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        layout.addWidget(button_box)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)

        # Show the dialog and get the user's response
        if dialog.exec() == QDialog.DialogCode.Rejected:
            return  # User cancelled

        # Determine which fields to display based on user selection
        selected_fields = [key for key, checkbox in checkboxes.items() if checkbox.isChecked()]

        # Calculate required vertical space for text based on number of selected fields
        text_row_height = 15
        text_block_height = len(selected_fields) * text_row_height
        patch_size = 128
        space_between_patches = max(64, text_block_height + 20)  # Ensure enough space for text between patches

        # Set parameters for collage layout
        number_of_objects = len(self.results)

        if number_of_objects == 0:
            QMessageBox.warning(self, "No Objects", "No objects available to create a collage.")
            return

        # Determine grid size for the collage
        grid_size = math.ceil(math.sqrt(number_of_objects))
        collage_width = patch_size * grid_size + space_between_patches * (grid_size - 1) + 128
        collage_height = patch_size * grid_size + space_between_patches * (grid_size - 1) + 128

        # Create an empty black RGB image for the collage
        collage_image = Image.new("RGB", (collage_width, collage_height), (0, 0, 0))
        draw = ImageDraw.Draw(collage_image)   # <— add this

        # Temporarily disable annotations
        original_show_names = self.show_names
        original_circle_center = self.main_preview.circle_center
        original_circle_radius = self.main_preview.circle_radius
        self.show_names = False
        self.main_preview.circle_center = None
        self.main_preview.circle_radius = 0

        try:
            for i, obj in enumerate(self.results):
                # Calculate position in the grid
                row = i // grid_size
                col = i % grid_size
                offset_x = 64 + col * (patch_size + space_between_patches)
                offset_y = 64 + row * (patch_size + space_between_patches)

                # Calculate pixel coordinates around the object
                ra, dec = obj["ra"], obj["dec"]
                x, y = self.calculate_pixel_from_ra_dec(ra, dec)


                # Crop the relevant area for the object
                rect = QRectF(x - patch_size // 2, y - patch_size // 2, patch_size, patch_size)
                cropped = self.main_image.copy(rect.toRect()).toImage().convertToFormat(QImage.Format.Format_RGB888)

                buf = cropped.bits()
                buf.setsize(cropped.width() * cropped.height() * 3)
                pil_patch = Image.frombytes("RGB", (cropped.width(), cropped.height()), bytes(buf))
                from PIL import Image as _PILImage
                pil_patch = pil_patch.resize((patch_size, patch_size), _PILImage.Resampling.BILINEAR)
                collage_image.paste(pil_patch, (offset_x, offset_y))

                # Font fallback
                try:
                    font = ImageFont.truetype("arial.ttf", 12)
                except Exception:
                    font = ImageFont.load_default()
                text_y = offset_y + patch_size + 5

                for field in selected_fields:
                    # Retrieve data and only display if not "N/A"
                    if field == "Name" and obj.get("name") != "N/A":
                        text = obj["name"]
                    elif field == "RA" and obj.get("ra") is not None:
                        text = f"RA: {obj['ra']:.6f}"
                    elif field == "Dec" and obj.get("dec") is not None:
                        text = f"Dec: {obj['dec']:.6f}"
                    elif field == "Short Type" and obj.get("short_type") != "N/A":
                        text = f"Type: {obj['short_type']}"
                    elif field == "Long Type" and obj.get("long_type") != "N/A":
                        text = f"{obj['long_type']}"
                    elif field == "Redshift" and obj.get("redshift") != "N/A":
                        text = f"Redshift: {float(obj['redshift']):.5f}"  # Limit redshift to 5 decimal places
                    elif field == "Comoving Distance" and obj.get("comoving_distance") != "N/A":
                        text = f"Distance: {obj['comoving_distance']} GLy"
                    else:
                        continue  # Skip if field is not available or set to "N/A"

                    # Draw the text and increment the Y position
                    draw.text((offset_x + 10, text_y), text, (255, 255, 255), font=font)
                    text_y += text_row_height  # Space between lines

        finally:
            # Restore the original annotation and search circle settings
            self.show_names = original_show_names
            self.main_preview.circle_center = original_circle_center
            self.main_preview.circle_radius = original_circle_radius

        # Save the collage
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Collage of Objects", "", "JPEG (*.jpg *.jpeg);;PNG (*.png);;TIFF (*.tiff *.tif)"
        )

        if file_path:
            collage_image.save(file_path)
            QMessageBox.information(self, "Save Successful", f"Collage saved as {file_path}")


        # Restore the search circle in the custom graphics view
        self.main_preview.circle_center = original_circle_center
        self.main_preview.circle_radius = original_circle_radius

        # Re-sync the fast marker layer without clearing the scene again
        self._ensure_marker_layer()
        self._set_marker_points_from_results()
        if self._marker_layer:
            self._marker_layer.update()

    def show_3d_model_view(self):
        # ─── 0) Engineering‐notation helper ────────────────────────────
        def eng_notation(x):
            if x == 0:
                return "0"
            exp = int(math.floor(math.log10(abs(x)) / 3) * 3)
            val = x / (10 ** exp)
            prefixes = {
                -12: "p", -9: "n", -6: "µ", -3: "m",
                0: "",  3: "k",  6: "M",  9: "G",
                12: "T", 15: "P"
            }
            return f"{val:.2f}{prefixes.get(exp, f'e{exp}')}"

        # ─── 1) Sanity checks ─────────────────────────────────────────
        if not self.results or self.image_data is None:
            QMessageBox.warning(self, "Data Error", "No image or results available.")
            return
        if self.wcs is None:
            QMessageBox.warning(self, "WCS Missing", "WCS data is required to generate the 3D plot.")
            return

        # ─── 2) Get user settings ────────────────────────────────────
        settings = ThreeDSettingsDialog(self).getSettings()
        if not settings:
            return
        (plane_style, max_res, z_option, z_min, z_max,
        z_scale, linear_max, obj_color, custom_col,
        z_height, show_lines, reverse_z) = (
            settings[k] for k in (
                "plane_style","resolution","z_option","z_min","z_max",
                "z_scale","linear_max","object_color","custom_color",
                "z_height","show_lines","reverse_z"
            )
        )

        # ─── 3) Normalize & downsample image ──────────────────────────
        img = self.image_data
        if img.ndim == 2:
            img = np.stack([img]*3, axis=-1)
        img_norm = np.clip((img - img.min())/(np.ptp(img)+1e-8), 0, 1)
        full_h, full_w = img.shape[:2]
        scale = min(max_res/full_h, max_res/full_w, 1.0)
        img_ds = np.stack([zoom(img_norm[..., i], scale, order=1) for i in range(3)], axis=-1)
        h_ds, w_ds, _ = img_ds.shape

        # ─── 4) Build plane coordinates ───────────────────────────────
        xs = np.linspace(0, full_w-1, w_ds)
        ys = np.linspace(0, full_h-1, h_ds)
        Xp, Yp = np.meshgrid(xs, ys)
        RA, DEC = self.wcs.pixel_to_world_values(Xp, Yp)
        ra_min, ra_max = float(RA.min()), float(RA.max())
        dec_min, dec_max = float(DEC.min()), float(DEC.max())

        # ─── 5) Gather objects into rows (keep fields together) ───────
        rows = []
        for obj in self.results:
            try:
                name = obj["name"]
                ra   = float(obj["ra"])
                dec  = float(obj["dec"])

                # distance in ly (robust parse; always positive; +10 to avoid log10(0))
                try:
                    d_gy = float(obj["comoving_distance"])  # e.g., GLy
                    d_ly = abs(d_gy) * 1e9 + 10
                except (ValueError, TypeError):
                    raw_cd = str(obj["comoving_distance"]).strip()
                    if raw_cd.endswith("GLy"):
                        d_ly = abs(float(raw_cd[:-3].strip())) * 1e9 + 10
                    elif raw_cd.endswith("Ly"):
                        d_ly = abs(float(raw_cd[:-2].strip())) + 10
                    else:
                        d_ly = abs(float(raw_cd)) + 10
                if d_ly <= 0:
                    continue

                zshift = float(obj.get("redshift", 0.0))

                px, py = self.wcs.world_to_pixel_values(ra, dec)
                if not (0 <= px < full_w and 0 <= py < full_h):
                    continue

                zval = math.log10(d_ly) if z_scale == "Logarithmic" else d_ly
                ra0, dec0 = self.wcs.pixel_to_world_values(px, py)

                label = (
                    f"<b>{name}</b><br>"
                    f"RA: {ra:.6f}<br>"
                    f"Dec: {dec:.6f}<br>"
                    f"Distance: {eng_notation(d_ly)} ly<br>"
                    f"Redshift: {zshift:.5f}"
                )
                enc = urllib.parse.quote(name)
                url = (
                    "https://simbad.cds.unistra.fr/simbad/sim-basic?"
                    f"Ident={enc}&submit=SIMBAD+search"
                )

                rows.append(dict(
                    name=name, x=ra0, y=dec0, z=zval, url=url, label=label,
                    px=px, py=py, legend_qcolor=obj.get("color", QColor(128, 128, 128))
                ))
            except Exception:
                continue

        if not rows:
            QMessageBox.warning(self, "No Objects", "No valid distance objects to plot.")
            return

        # ─── 6) Filter rows by finite z (and optional range) ──────────
        rows = [r for r in rows if math.isfinite(r["z"])]
        if not rows:
            QMessageBox.warning(self, "No Objects", "All distance values are invalid.")
            return

        if z_option == "Custom":
            rows = [r for r in rows if z_min <= r["z"] <= z_max]
            if not rows:
                QMessageBox.warning(self, "No Objects", "No objects fall within the custom Z range.")
                return
            plane_z = z_min
            z_range = (z_min, z_max)
        elif z_option == "Min-Max":
            z_vals = np.array([r["z"] for r in rows], dtype=float)
            plane_z = float(np.nanmin(z_vals))
            z_range = (float(np.nanmin(z_vals)), float(np.nanmax(z_vals)))
        else:  # Default
            plane_z = 0
            z_range = None

        if z_scale == "Linear" and z_option == "Default":
            z_range = (0, linear_max)

        # ─── 7) Build image‐plane layer ───────────────────────────────
        if "Grayscale" in plane_style:
            Zp = np.full_like(RA, plane_z)
            plane = go.Surface(
                x=RA, y=DEC, z=Zp,
                surfacecolor=np.mean(img_ds, axis=2),
                colorscale="gray", showscale=False, opacity=1.0,
                contours={"x": {"show": False}, "y": {"show": False}, "z": {"show": False}}
            )
        else:
            flat = (img_ds * 255).astype(int).reshape(-1, 3)
            cols = [f"rgb({r},{g},{b})" for r, g, b in flat]
            plane = go.Scatter3d(
                x=RA.flatten(), y=DEC.flatten(), z=[plane_z] * RA.size,
                mode="markers",
                marker=dict(symbol="square", size=2, line=dict(width=0), color=cols, opacity=1.0),
                hoverinfo="skip", showlegend=False
            )

        # ─── 8) Object colors (after filtering) ───────────────────────
        H, W = img_norm.shape[:2]
        if obj_color == "Image-Based":
            patch_r = 5
            def patch_color(r):
                cx, cy = int(r["px"]), int(r["py"])
                x0, x1 = max(0, cx - patch_r), min(W, cx + patch_r + 1)
                y0, y1 = max(0, cy - patch_r), min(H, cy + patch_r + 1)
                patch = img_norm[y0:y1, x0:x1]
                if patch.size:
                    mr, mg, mb = (patch.reshape(-1, 3).mean(axis=0) * 255).astype(int)
                else:
                    mr = mg = mb = 0
                return f"rgb({mr},{mg},{mb})"
            obj_cols = [patch_color(r) for r in rows]
        elif obj_color == "Legend Color":
            obj_cols = [f"rgb({c.red()},{c.green()},{c.blue()})" for c in (r["legend_qcolor"] for r in rows)]
        elif obj_color == "Solid (Custom)":
            c = custom_col
            obj_cols = [f"rgb({c.red()},{c.green()},{c.blue()})"] * len(rows)
        else:
            obj_cols = ["red"] * len(rows)

        # ─── 9) Build arrays (all same length) and optional lines ─────
        world_xs = [r["x"] for r in rows]
        world_ys = [r["y"] for r in rows]
        zs       = np.array([r["z"] for r in rows], dtype=float)
        labels   = [r["label"] for r in rows]
        urls     = [r["url"] for r in rows]
        names    = [r["name"] for r in rows]

        lines = []
        if show_lines:
            for r in rows:
                lines.append(go.Scatter3d(
                    x=[r["x"], r["x"]], y=[r["y"], r["y"]], z=[plane_z, r["z"]],
                    mode="lines", line=dict(color="gray", width=1),
                    hoverinfo="skip", showlegend=False
                ))

        # ─── 10) Scatter objects ───────────────────────────────────────
        scatter = go.Scatter3d(
            x=world_xs, y=world_ys, z=zs,
            mode="markers",
            marker=dict(size=4, color=obj_cols),
            hovertext=labels, hoverinfo="text",
            customdata=urls, name="Objects"
        )

        # ─── 11) Compose figure ───────────────────────────────────────
        fig = go.Figure(data=[plane] + lines + [scatter])
        scene = dict(
            xaxis_title="RA (deg)",
            xaxis=dict(range=[ra_min, ra_max], autorange=False),
            yaxis_title="Dec (deg)",
            yaxis=dict(range=[dec_max, dec_min], autorange=False),
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=z_height),
            zaxis=dict(
                title=("log10(Distance in ly)" if z_scale == "Logarithmic" else "Distance (ly)"),
                tickformat="~s",
                **({"range": list(z_range)} if z_range else {}),
                **({"autorange": "reversed"} if reverse_z else {})
            )
        )
        fig.update_layout(title="3D Distance Model", autosize=True, scene=scene,
                        margin=dict(l=0, r=0, b=0, t=40))

        # ─── 12) Build & inject HTML ───────────────────────────────────
        html = fig.to_html(include_plotlyjs="cdn", full_html=True)
        items = "".join(f'<li><a href="{u}" target="_blank">{n}</a></li>' for n, u in zip(names, urls))
        sidebar = (
            '<div style="padding:10px;font-family:sans-serif;'
            'margin-top:20px;border-top:1px solid #ccc;">'
            '<h3>Objects</h3><ul>' + items + '</ul></div>'
        )
        js = """
        <script>
        var gd = document.getElementsByClassName('plotly-graph-div')[0];
        gd.on('plotly_click', function(e){
            var url = e.points[0].customdata;
            if (url) window.open(url, '_blank');
        });
        </script>
        """
        html = html.replace("</body>", sidebar + js + "</body>")

        # Save & preview
        default = os.path.expanduser("~/3d_distance_model.html")
        fn, _ = QFileDialog.getSaveFileName(self, "Save 3D Plot As", default, "HTML Files (*.html)")
        if fn:
            if not fn.lower().endswith(".html"):
                fn += ".html"
            with open(fn, "w", encoding="utf-8") as f:
                f.write(html)

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode="w", encoding="utf-8")
        tmp.write(html); tmp.close()
        webbrowser.open("file://" + tmp.name)



    def show_hr_diagram(self):
        """H-R Diagram: B–V vs Abs V, with selectable color, background, and axis ranges."""
        # Pop up the settings dialog
        settings = HRSettingsDialog(self).getSettings()
        if settings is None:
            return

        use_realistic = settings["color_mode"].startswith("Realistic")
        use_image_bg = settings["bg_mode"].startswith("HR Diagram")
        solid_qcolor = settings["custom_color"]
        show_sun = settings["show_sun"]

        # Determine axis bounds
        if settings["range_mode"].startswith("Default"):
            x0, x1 = -0.3, 2.25
            y0, y1 = -9.0, 19.0
        else:
            x0 = settings["x_min"]
            x1 = settings["x_max"]
            y0 = settings["y_min"]
            y1 = settings["y_max"]

        # Sanity: ensure query_results exist
        if not getattr(self, 'query_results', None):
            QMessageBox.information(self, "No Data",
                "Run a SIMBAD query first to gather B, V, and distance data.")
            return

        # Collect data
        B, V, Mv, names = [], [], [], []
        for obj in self.query_results:
            try:
                b = float(obj['Bmag'])
                v = float(obj['Vmag'])
                m = float(obj['absolute_mag'])
            except (TypeError, ValueError, KeyError):
                continue
            B.append(b); V.append(v); Mv.append(m); names.append(obj['name'])
        if not B:
            QMessageBox.warning(self, "Insufficient Data",
                "No objects have valid B-mag, V-mag and absolute magnitude.")
            return

        # Compute B−V & T_eff & colors
        bv = [b - v for b, v in zip(B, V)]
        T_eff = [4600.0 * (1/(0.92*x + 1.7) + 1/(0.92*x + 0.62)) for x in bv]
        if use_realistic:
            colors = [kelvin_to_rgb(T) for T in T_eff]
        else:
            hex_color = solid_qcolor.name()
            colors = [hex_color] * len(bv)

        # Prepare hover & URLs
        hover_texts, urls = [], []
        for nm, b, v, m, T in zip(names, B, V, Mv, T_eff):
            hover_texts.append(
                f"<b>{nm}</b><br>"
                f"B: {b:.2f}  V: {v:.2f}<br>"
                f"Abs V: {m:.2f}<br>"
                f"Tₑff: {T:.0f} K"
            )
            enc = urllib.parse.quote(nm)
            urls.append(
                f"https://simbad.cds.unistra.fr/simbad/sim-basic?"
                f"Ident={enc}&submit=SIMBAD+search"
            )

        # Create scatter
        scatter = go.Scatter(
            x=bv, y=Mv, mode='markers',
            marker_color=colors, marker_size=20,
            hovertext=hover_texts, customdata=urls,
            name="Stars"
        )
        fig = go.Figure(scatter)


        # 1) Dense BV grid over x0→x1
        bv_grid = np.linspace(x0, x1, 300)
        # 2) Compute Teff at each BV
        T_grid = 4600.0 * (1.0/(0.92*bv_grid + 1.7) + 1.0/(0.92*bv_grid + 0.62))
        # 3) Solar Teff for normalization
        T_sun = 5772.0

        # 4) Radii in R_sun
        radii = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
        # 5) Corresponding gray shades for each contour
        gray_colors = [
            "#444444",  # darkest gray for R=0.01
            "#666666",  # R=0.1
            "#888888",  # R=1.0
            "#AAAAAA",  # R=10
            "#CCCCCC",  # R=100
            "#EEEEEE"   # R=1000 (lightest)
        ]

        for idx, R_rs in enumerate(radii):
            # compute L/L_sun
            L_over_Lsun = (R_rs**2) * (T_grid / T_sun)**4
            # convert to M_V
            MV_line = 4.83 - 2.5 * np.log10(L_over_Lsun)

            fig.add_trace(
                go.Scatter(
                    x=bv_grid,
                    y=MV_line,
                    mode='lines',
                    line=dict(
                        color=gray_colors[idx],
                        dash='dash'
                    ),
                    name=f"R = {R_rs:g} R⊙",   # plain text “R⊙”
                    hoverinfo='none'
                )
            )
        # ───────────────────────────────────────────────────────────────────

        # Build tick labels
        bv_ticks = [-0.3, 0.0, 0.5, 1.0, 1.5, 2.0, 2.25]
        t_ticks = [4600.0*(1/(0.92*x+1.7)+1/(0.92*x+0.62)) for x in bv_ticks]
        tick_labels = [f"{x:.2f}<br>{int(t):,} K" for x,t in zip(bv_ticks, t_ticks)]

        # If using image background, load via PIL
        if use_image_bg:
            pil_img = Image.open(hrdiagram_path)

        # Force the specified x & y axis ranges
        fig.update_xaxes(
            title_text="B−V color (mag) ↔ Tₑff",
            tickvals=bv_ticks,
            ticktext=tick_labels,
            tickfont_color='white',
            title_font=dict(color='white'),
            gridcolor='gray',
            zerolinecolor='gray',
            range=[x0, x1]
        )
        fig.update_yaxes(
            title_text="Absolute V magnitude (mag)",
            range=[y1, y0],         # reversed on purpose: y1 (–9) at top, y0 (19) at bottom
            autorange=False,
            tickfont_color='white',
            title_font=dict(color='white'),
            gridcolor='gray',
            zerolinecolor='gray',
        )

        # Add the image behind the plot if chosen
        if use_image_bg:
            fig.add_layout_image(
                dict(
                    source=pil_img,
                    xref="x", yref="y",
                    x=x0, y=y0,
                    sizex=(x1 - x0),
                    sizey=(y1 - y0),
                    xanchor="left", yanchor="top",
                    sizing="stretch",
                    opacity=1.0,
                    layer="below"
                )
            )

        # Add a special marker for the Sun (B−V=0.66, Mv=4.8)
        if show_sun:
            sun_scatter = go.Scatter(
                x=[0.66], y=[4.8],
                mode='markers+text',
                marker=dict(
                    color='gold',
                    size=30,
                    symbol='star'
                ),
                name="Sun"
            )
            fig.add_trace(sun_scatter)

        # Style axes & background
        if use_image_bg:
            fig.update_layout(
                title=dict(text="Hertzsprung–Russell Diagram", font_color='white'),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='black',
                margin=dict(l=40, r=20, t=60, b=60)
            )
        else:
            fig.update_layout(
                title=dict(text="Hertzsprung–Russell Diagram", font_color='white'),
                plot_bgcolor='black',
                paper_bgcolor='black',
                margin=dict(l=40, r=20, t=60, b=60)
            )

        # Sidebar & click behaviour
        items = "".join(
            f'<li><a href="{u}" style="color:cyan" target="_blank">{n}</a></li>'
            for n,u in zip(names, urls)
        )
        sidebar = (
            '<div style="padding:10px;font-family:sans-serif;'
            'margin-top:10px;border-top:1px solid #444; background:black; color:white;">'
            '<h3>Objects</h3><ul>' + items + '</ul></div>'
        )
        js = """
        <script>
        var gd = document.getElementsByClassName('plotly-graph-div')[0];
        gd.on('plotly_click', function(e){
            var url = e.points[0].customdata;
            if(url) window.open(url,'_blank');
        });
        </script>
        """

        html = fig.to_html(include_plotlyjs='cdn', full_html=True)
        html = html.replace("</body>", sidebar + js + "</body>")

        # Save & preview
        default = os.path.join(os.path.expanduser("~"), "hr_diagram.html")
        fn, _ = QFileDialog.getSaveFileName(self, "Save H-R Diagram As",
                                            default, "HTML Files (*.html)")
        if fn:
            if not fn.lower().endswith('.html'):
                fn += '.html'
            with open(fn, 'w', encoding='utf-8') as f:
                f.write(html)

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.html',
                                        mode='w', encoding='utf-8')
        tmp.write(html); tmp.close()
        webbrowser.open("file://" + tmp.name)

    
    def search_defined_region(self):
        """Perform a Simbad search for the defined region and filter by selected object types."""
        selected_types = self.get_selected_object_types()
        if not selected_types:
            QMessageBox.warning(self, "No Object Types Selected", "Please select at least one object type.")
            return

        # Calculate the radius in degrees for the defined region (circle radius)
        radius_deg = self.get_defined_radius()

        # Perform the Simbad search in the defined region with the calculated radius
        self.query_simbad(radius_deg)


    def search_entire_image(self):
        """Search the entire image using Simbad with selected object types."""
        selected_types = self.get_selected_object_types()  # Get selected types from the advanced search panel
        if not selected_types:
            QMessageBox.warning(self, "No Object Types Selected", "Please select at least one object type.")
            return

        # Calculate radius as the distance from the image center to a corner
        width, height = self.main_image.width(), self.main_image.height()
        center_x, center_y = width / 2, height / 2
        corner_x, corner_y = width, height  # Bottom-right corner
        # Calculate distance in pixels from center to corner
        radius_px = np.sqrt((corner_x - center_x) ** 2 + (corner_y - center_y) ** 2)
        # Convert radius from pixels to degrees
        radius_deg = float((radius_px * self.pixscale) / 3600.0)

        # Automatically set circle_center and circle_radius for the entire image
        self.circle_center = QPointF(center_x, center_y)  # Assuming QPointF is used
        self.circle_radius = radius_px  # Set this to allow the check in `query_simbad`

        # Perform the query with the calculated radius
        self.query_simbad(radius_deg, max_results=100000)


    def _iterate_leaf_items(self):
        """Yield every otype (child) under every category parent."""
        root = self.object_tree.invisibleRootItem()
        for i in range(root.childCount()):
            parent = root.child(i)
            for j in range(parent.childCount()):
                yield parent.child(j)


    def _update_parent_states(self):
        """Parents become checked if any child is checked, else unchecked."""
        root = self.object_tree.invisibleRootItem()
        for i in range(root.childCount()):
            parent = root.child(i)
            # if any child checked → parent checked, else unchecked
            any_checked = any(
                parent.child(j).checkState(0) == Qt.CheckState.Checked
                for j in range(parent.childCount())
            )
            parent.setCheckState(0, Qt.CheckState.Checked if any_checked else Qt.CheckState.Unchecked)


    def on_object_tree_item_changed(self, item, column):
        # parent toggled → mirror to all children
        if item.childCount() > 0:
            state = item.checkState(0)
            blocker = QSignalBlocker(self.object_tree)
            for i in range(item.childCount()):
                item.child(i).setCheckState(0, state)
            blocker.unblock()
        # child toggled → recompute only its parent
        else:
            parent = item.parent()
            if not parent:
                return
            # parent checked if any child is
            any_checked = any(
                parent.child(i).checkState(0) == Qt.CheckState.Checked
                for i in range(parent.childCount())
            )
            blocker = QSignalBlocker(self.object_tree)
            parent.setCheckState(0, Qt.CheckState.Checked if any_checked else Qt.CheckState.Unchecked)
            blocker.unblock()


    def toggle_all_items(self):
        """Check/uncheck all otype leaf items."""
        leaves      = list(self._iterate_leaf_items())
        all_checked = all(li.checkState(0) == Qt.CheckState.Checked for li in leaves)
        new_state   = Qt.CheckState.Unchecked if all_checked else Qt.CheckState.Checked

        blocker = QSignalBlocker(self.object_tree)
        for li in leaves:
            li.setCheckState(0, new_state)
        blocker.unblock()

        self._update_parent_states()


    def save_custom_list(self):
        """
        Serializes the currently checked otypes to a .json file.
        """
        types = self.get_selected_object_types()
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Custom Type List",
            "",
            "JSON Files (*.json)"
        )
        if not path:
            return
        if not path.lower().endswith(".json"):
            path += ".json"
        try:
            with open(path, "w") as f:
                json.dump(types, f, indent=2)
            self.status_label.setText(f"👍 Saved list to {os.path.basename(path)}")
        except Exception as e:
            QMessageBox.critical(self, "Save Failed", str(e))


    def load_custom_list(self):
        """
        Reads a .json of otype codes and re-checks only those.
        """
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Custom Type List",
            "",
            "JSON Files (*.json)"
        )
        if not path:
            return

        try:
            with open(path, "r") as f:
                types = set(json.load(f))
        except Exception as e:
            QMessageBox.critical(self, "Load Failed", str(e))
            return

        # block signals so we don't recurse
        blocker = QSignalBlocker(self.object_tree)
        for li in self._iterate_leaf_items():
            li.setCheckState(
                0,
                Qt.CheckState.Checked if li.text(0) in types else Qt.CheckState.Unchecked
            )
        blocker.unblock()

        # now update parents
        self._update_parent_states()
        self.status_label.setText(f"📂 Loaded list from {os.path.basename(path)}")

    def toggle_advanced_search(self):
        """Toggle the visibility of the advanced search panel."""
        is_visible = self.advanced_search_panel_widget.isVisible()
        self.advanced_search_panel_widget.setVisible(not is_visible)

    def save_results_as_csv(self):
        """Save the results from the TreeWidget as a CSV file."""
        path, _ = QFileDialog.getSaveFileName(self, "Save CSV", "", "CSV Files (*.csv)")
        if path:
            with open(path, mode='w', newline='') as file:
                writer = csv.writer(file)
                # Write header
                writer.writerow(["RA", "Dec", "Name", "Diameter", "Type", "Long Type", "Redshift", "Comoving Radial Distance (GLy)"])

                # Write data from TreeWidget
                for i in range(self.results_tree.topLevelItemCount()):
                    item = self.results_tree.topLevelItem(i)
                    row_data = [item.text(column) for column in range(self.results_tree.columnCount())]
                    writer.writerow(row_data)

            QMessageBox.information(self, "CSV Saved", f"Results successfully saved to {path}")        

    def filter_visible_objects(self):
        """Filter objects based on visibility threshold."""
        if not self.main_image:  # Ensure there's an image loaded
            QMessageBox.warning(self, "No Image", "Please load an image first.")
            return

        n = 0.2  # Threshold multiplier, adjust as needed
        median, std_dev = self.calculate_image_statistics(self.main_image)

        # Remove objects below threshold from results
        filtered_results = []
        for obj in self.results:
            if self.is_marker_visible(obj, median, std_dev, n):
                filtered_results.append(obj)

        # Update the results and redraw the markers
        self.results = filtered_results
        self.main_preview.results = filtered_results
        self.update_results_tree()
        self.main_preview.draw_query_results()

    def calculate_image_statistics(self, image):
        """Calculate median and standard deviation for a grayscale image efficiently using OpenCV."""
        
        # Convert QPixmap to QImage if necessary
        qimage = image.toImage()

        # Convert QImage to a format compatible with OpenCV
        width = qimage.width()
        height = qimage.height()
        ptr = qimage.bits()
        ptr.setsize(height * width * 4)  # 4 channels (RGBA)
        img_array = np.array(ptr).reshape(height, width, 4)  # Convert to RGBA array

        # Convert to grayscale for analysis
        gray_image = cv2.cvtColor(img_array, cv2.COLOR_RGBA2GRAY)

        # Calculate median and standard deviation
        median = np.median(gray_image)
        _, std_dev = cv2.meanStdDev(gray_image)

        return median, std_dev[0][0]  # std_dev returns a 2D array, so we extract the single value
    
    def is_marker_visible(self, marker, median, std_dev, n):
        """Check if the marker's brightness is above the threshold."""
        threshold = median + n * std_dev
        check_size = 8  # Define a 4x4 region around the marker

        # Convert QPixmap to QImage to access pixel colors
        image = self.main_image.toImage()

        # Get marker coordinates in pixel space
        ra, dec = marker.get('ra'), marker.get('dec')
        if ra is not None and dec is not None:
            x, y = self.calculate_pixel_from_ra_dec(ra, dec)
            if x is None or y is None:
                return False  # Skip marker if it can't be converted to pixels
        else:
            return False

        # Calculate brightness in a 4x4 region around marker coordinates
        brightness_values = []
        for dx in range(-check_size // 2, check_size // 2):
            for dy in range(-check_size // 2, check_size // 2):
                px = x + dx
                py = y + dy
                if 0 <= px < image.width() and 0 <= py < image.height():
                    color = image.pixelColor(px, py)  # Get color from QImage
                    brightness = color.value() if color.isValid() else 0  # Adjust for grayscale
                    brightness_values.append(brightness)

        if brightness_values:
            average_brightness = sum(brightness_values) / len(brightness_values)
            return average_brightness > threshold
        else:
            return False



    def update_results_tree(self):
        """Refresh the TreeWidget to reflect current results."""
        self.results_tree.clear()
        for obj in self.results:
            item = QTreeWidgetItem([
                str(obj['ra']),
                str(obj['dec']),
                obj['name'],
                str(obj['diameter']),
                obj['short_type'],
                obj['long_type'],
                str(obj['redshift']),
                str(obj['comoving_distance'])
            ])
            self.results_tree.addTopLevelItem(item)

    # Function to clear search results and remove markers
    def clear_search_results(self):
        """Clear the search results and remove all markers."""
        self.results_tree.clear()        # Clear the results from the tree
        self.results = []                # Clear the results list
        self.main_preview.results = []   # Clear results from the main preview
        self.main_preview.selected_object = None
        self.main_preview.draw_query_results()  # Redraw the main image without markers
        self.status_label.setText("Results cleared.")

    def on_tree_item_clicked(self, item):
        """Handle item click in the TreeWidget to highlight the associated object."""
        object_name = item.text(2)

        # Find the object in results
        selected_object = next(
            (obj for obj in self.results if obj.get("name") == object_name), None
        )

        if selected_object:
            # Set the selected object in MainWindow and update views
            self.selected_object = selected_object
            self.main_preview.select_object(selected_object)
            self.main_preview.draw_query_results()
            self.main_preview.update_mini_preview() 
            
            

    def on_tree_item_double_clicked(self, item):
        """Handle double-click event on a TreeWidget item to open SIMBAD or NED URL based on source."""
        object_name = item.text(2)  # Assuming 'Name' is in the third column

        # parse only if float() fails
        def parse_value(txt):
            try:
                return float(txt)
            except ValueError:
                parts = txt.strip().split()
                if len(parts) == 3:
                    a, b, c = parts
                    sign = -1 if a.startswith('-') else 1
                    return sign * (abs(float(a)) + float(b)/60 + float(c)/3600)
                raise

        ra  = parse_value(item.text(0).strip())
        dec = parse_value(item.text(1).strip())

        # lookup, falling back to string→float only on failure
        def get_parsed(result, key):
            try:
                return float(result[key])
            except ValueError:
                return parse_value(result[key])

        def _close(a, b, tol=1e-6):
            return abs(a - b) <= tol

        entry = next(
            (r for r in self.query_results
            if _close(get_parsed(r, 'ra'), ra) and _close(get_parsed(r, 'dec'), dec)),
            None
        )
        source = (entry.get('source') if entry else 'Simbad') or 'Simbad'
        print(f"[DEBUG] Matched source: {source!r}")  # ← debug print

        s = source.strip().lower()

        if s == "simbad" and object_name:
            encoded = quote(object_name)
            webbrowser.open(
                f"https://simbad.cds.unistra.fr/simbad/sim-basic?"
                f"Ident={encoded}&submit=SIMBAD+search"
            )

        elif "viz" in s:   # catches 'Vizier', etc.
            radius   = 5/60  # arcminutes
            dec_sign = "%2B" if dec >= 0 else "-"
            webbrowser.open(
                f"http://ned.ipac.caltech.edu/conesearch?"
                f"search_type=Near%20Position%20Search&"
                f"ra={ra:.6f}d&"
                f"dec={dec_sign}{abs(dec):.6f}d&"
                f"radius={radius:.3f}&"
                "in_csys=Equatorial&in_equinox=J2000.0"
            )

        elif s == "mast":
            webbrowser.open(
                f"https://mast.stsci.edu/portal/Mashup/Clients/Mast/Portal.html?"
                f"searchQuery={ra}%2C{dec}%2Cradius%3D0.0006"
            )
           

    def copy_ra_dec_to_clipboard(self):
        """Copy the currently displayed RA and Dec to the clipboard."""
        # Access the RA and Dec labels directly
        ra_text = self.ra_label.text()
        dec_text = self.dec_label.text()
        
        # Combine RA and Dec text for clipboard
        clipboard_text = f"{ra_text}, {dec_text}"
        
        clipboard = QApplication.instance().clipboard()
        clipboard.setText(clipboard_text)
        
        QMessageBox.information(self, "Copied", "Current RA/Dec copied to clipboard!")
    
    def _load_from_view(self, doc):
        """
        Accepts a pro.doc_manager.ImageDocument and loads:
        - image pixels
        - best-available astrometric solution:
            1) metadata['wcs_header'] (preferred if present)
            2) metadata['original_header'] if it contains WCS
            3) XISF metadata (image_meta/file_meta) converted into a FITS-like header
        """
        if doc is None or getattr(doc, "image", None) is None:
            QMessageBox.information(self, "Load from View", "That view has no image data.")
            return

        self._loaded_doc = doc 

        # 1) show pixels
        self._set_image_from_array(doc.image)

        # 2) pick a header
        meta = dict(getattr(doc, "metadata", {}) or {})
        header = None

        # prefer an explicit WCS header saved by a solver
        if "wcs_header" in meta and meta["wcs_header"]:
            header = meta["wcs_header"]

        # otherwise try the original FITS header (if present and looks like WCS)
        orig_hdr = meta.get("original_header")
        if header is None and self.check_astrometry_data(orig_hdr):
            header = orig_hdr

        # otherwise try to synthesize from XISF metadata we may have stored
        if header is None:
            xim = meta.get("image_meta") or meta.get("XISF") or {}
            try:
                candidate = self.construct_fits_header_from_xisf(xim) if xim else None
            except Exception:
                candidate = None
            if candidate and self.check_astrometry_data(candidate):
                header = candidate

        if header is None:
            QMessageBox.information(self, "Blind Solve", "No WCS found.\nPerforming blind solve…")
            if not self._ensure_wcs_on_doc(doc):
                self.wcs = None
                self.status_label.setText("Status: Loaded view (no WCS) — auto-solve failed.")
                return
            # refresh from doc after solving
            meta = dict(getattr(doc, "metadata", {}) or {})
            header = meta.get("wcs_header") or meta.get("original_header")

        # 3) initialize WCS if we found/created a header
        if header is not None:
            try:
                header = self._sanitize_wcs_header(header)
                self.initialize_wcs_from_header(header)
                self.status_label.setText("Status: Loaded view with astrometric solution.")
            except Exception as e:
                self.wcs = None
                self.status_label.setText(f"Status: Loaded view (no valid WCS) — {e}")
        else:
            self.wcs = None
            self.status_label.setText("Status: Loaded view (no astrometric solution found).")

    def _set_image_from_array(self, arr: np.ndarray):
        self.image_data = arr
        img = arr
        if img is None:
            return
        # float32 [0..1] → uint8
        if img.dtype != np.uint8:
            img = np.clip(img, 0.0, 1.0)
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)
            img = (img * 255.0).astype(np.uint8)
        elif img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)

        h, w = img.shape[:2]
        bytes_per_line = 3 * w
        qimg = QImage(img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pm = QPixmap.fromImage(qimg)

        self.main_image = pm
        if self.main_scene is None:
            self.main_scene = QGraphicsScene(self)
            self.main_preview.setScene(self.main_scene)
        self.main_scene.clear()
        self.main_scene.addPixmap(pm)
        self.main_preview.setSceneRect(QRectF(pm.rect()))
        self.main_preview.resetTransform()
        self.main_preview.centerOn(self.main_scene.sceneRect().center())


        self._ensure_marker_layer()
        # If you already have results from a prior query, show them over the new image:
        self._set_marker_points_from_results()
        # keep your mini-preview sync code if you have it
        if hasattr(self, "mini_preview") and self.mini_preview is not None:
            scaled = pm.scaled(self.mini_preview.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.mini_preview.setPixmap(scaled)


    @pyqtSlot()
    def open_image(self):
        """Slot for the “Load...” button — always pops up the file dialog."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", 
            "Images (*.png *.jpg *.jpeg *.tif *.tiff *.fit *.fits *.xisf)"
        )
        if path:
            self._load_image(path)

    def load_image_path(self, path: str):
        """Call this when you already know the filename."""
        if path:
            self._load_image(path)

    def _sanitize_wcs_header(self, header):
        """
        Coerce WCS/SIP keywords to proper numeric types and fix common SIP issues.
        If SIP remains inconsistent, drop SIP keywords so astropy WCS can still load.
        """
        from copy import deepcopy
        import numpy as np

        hdr = deepcopy(header)

        def _to_int(key):
            if key in hdr:
                try:
                    v = hdr[key]
                    if isinstance(v, (int, np.integer)):
                        return
                    # handle strings like '2' or '2.0'
                    hdr[key] = int(float(str(v).strip().strip("'\"")))
                except Exception:
                    # if it's garbage, remove it so we can fall back cleanly
                    del hdr[key]

        def _to_float(key):
            if key in hdr:
                try:
                    v = hdr[key]
                    if isinstance(v, (float, int, np.floating, np.integer)):
                        hdr[key] = float(v)
                    else:
                        hdr[key] = float(str(v).strip().strip("'\""))
                except Exception:
                    # if it's truly bad, just leave it; astropy may ignore it
                    pass

        # 1) Ensure core integer fields
        for k in ("NAXIS", "NAXIS1", "NAXIS2", "A_ORDER", "B_ORDER", "AP_ORDER", "BP_ORDER"):
            _to_int(k)

        # 2) Mirror missing A/B order if only one is present
        a_order = hdr.get("A_ORDER")
        b_order = hdr.get("B_ORDER")
        if (a_order is None) ^ (b_order is None):
            try:
                val = int(a_order if a_order is not None else b_order)
                hdr["A_ORDER"] = val
                hdr["B_ORDER"] = val
            except Exception:
                # will be handled by SIP drop step below
                pass

        # 3) Coerce SIP coefficients to float
        for key in list(hdr.keys()):
            if key in ("A_ORDER", "B_ORDER", "AP_ORDER", "BP_ORDER"):
                continue
            if key.startswith(("A_", "B_", "AP_", "BP_")):
                try:
                    _to_float(key)
                except Exception:
                    # if a specific coeff is junk, remove it
                    try:
                        del hdr[key]
                    except Exception:
                        pass

        # 4) Coerce standard WCS numeric keywords to float
        for key in list(hdr.keys()):
            if key.startswith(("CD", "PC", "CDELT", "CRVAL", "CRPIX", "CROTA")) or key in ("EQUINOX", "EPOCH", "LONPOLE", "LATPOLE"):
                _to_float(key)

        # 5) If SIP is still inconsistent, drop SIP so WCS loads without distortion
        a_order = hdr.get("A_ORDER")
        b_order = hdr.get("B_ORDER")
        sip_ok = (a_order is not None and b_order is not None)
        if not sip_ok:
            for key in list(hdr.keys()):
                if key in ("A_ORDER", "B_ORDER", "AP_ORDER", "BP_ORDER") or key.startswith(("A_", "B_", "AP_", "BP_")):
                    del hdr[key]
            # Also remove "-SIP" tag from CTYPE if present to avoid implying SIP
            for c in ("CTYPE1", "CTYPE2"):
                if c in hdr and isinstance(hdr[c], str) and "SIP" in hdr[c]:
                    hdr[c] = hdr[c].replace("-SIP", "")

        return hdr

    def _load_image(self, path: str):
        """
        Load pixels and WCS like _load_from_view:
        1) use embedded WCS if present and valid
        2) try to synthesize from XISF metadata
        3) otherwise auto-solve (ASTAP → Astrometry.net)
        """
        img_array, original_header, bit_depth, is_mono = load_image(path)
        self.image_path = path
        if img_array is None:
            QMessageBox.warning(self, "Open Image", "Could not load the selected file.")
            return

        # persist basics
        self.image_data = img_array
        self.original_header = original_header
        self.bit_depth = bit_depth
        self.is_mono = is_mono

        # 1) show pixels (same renderer used by _load_from_view)
        self._set_image_from_array(img_array)

        # 2) choose a header using the same order as _load_from_view
        header = None

        # prefer an explicit/embedded WCS header (from FITS or what load_image provided)
        if self.check_astrometry_data(original_header):
            header = original_header

        # otherwise try to synthesize from XISF metadata
        if header is None and path.lower().endswith(".xisf"):
            xisf_meta = self.extract_xisf_metadata(path)
            try:
                candidate = self.construct_fits_header_from_xisf(xisf_meta) if xisf_meta else None
            except Exception:
                candidate = None
            if candidate and self.check_astrometry_data(candidate):
                header = candidate

        # 3) if we found/created a header → sanitize and init WCS
        if header is not None:
            try:
                header = self._sanitize_wcs_header(header)
                self.initialize_wcs_from_header(header)
                self.status_label.setText("Status: Loaded image with astrometric solution.")
                return
            except Exception as e:
                # embedded header present but unusable → fall through to auto-solve
                self.wcs = None
                self.status_label.setText(f"Status: Embedded WCS invalid — {e}. Attempting auto-solve…")
                QApplication.processEvents()

        # 4) no valid WCS → follow the same auto-solve path as _load_from_view
        #    (ASTAP first; if that fails, Astrometry.net)
        self.status_label.setText("Status: No WCS found. Performing auto-solve…")
        QMessageBox.information(self, "Auto-Solve", "No astrometric solution found.\nPerforming auto-solve…")
        QApplication.processEvents()


        doc = self._doc_for_solver()  # returns the view doc if present, else adapter around image_data
        if doc is None:
            QMessageBox.warning(self, "Auto-Solve", "No image loaded.")
            return

        mw = _find_main_window(self) or self.parent()
        settings = getattr(mw, "settings", QSettings())

        prev_mode = _get_seed_mode(settings)
        try:
            # ensure it’s blind for this first solve
            _set_seed_mode(settings, "none")
            ok, res = plate_solve_doc_inplace(mw, doc, settings)
        finally:
            _set_seed_mode(settings, prev_mode)

        if ok:
            meta = getattr(doc, "metadata", {}) or {}
            hdr = meta.get("wcs_header") or meta.get("original_header")
            try:
                if hdr is None:
                    raise RuntimeError("Solver returned no header.")
                self.initialize_wcs_from_header(hdr)
                self.status_label.setText("Status: Blind solve succeeded.")
            except Exception as e:
                self.status_label.setText(f"Status: WCS init failed — {e}")
                QMessageBox.warning(self, "Apply WCS", f"Solve succeeded but WCS init failed: {e}")
        else:
            self.status_label.setText(f"Status: Blind solve failed — {res}")
            QMessageBox.critical(self, "Auto-Solve Failed", str(res))

    def extract_xisf_metadata(self, xisf_path):
        """
        Extract metadata from a .xisf file, focusing on WCS and essential image properties.
        """
        try:
            # Load the XISF file
            xisf = XISF(xisf_path)
            
            # Extract file and image metadata
            self.file_meta = xisf.get_file_metadata()
            self.image_meta = xisf.get_images_metadata()[0]  # Get metadata for the first image
            return self.image_meta
        except Exception as e:
            print(f"Error reading XISF metadata: {e}")
            return None

    def initialize_wcs_from_header(self, header):
        """ Initialize WCS data from a FITS header or constructed XISF header """
        from astropy.io import fits

        # normalize header → always keep a real fits.Header on self.header
        if isinstance(header, fits.Header):
            self.header = header.copy()
        else:
            # assume dict-like
            h = fits.Header()
            for k, v in dict(header).items():
                h[k] = v
            self.header = h

        try:
            # Use only the first two dimensions for WCS
            self.wcs = WCS(self.header, naxis=2, relax=True)
            
            # Calculate and set pixel scale
            pixel_scale_matrix = self.wcs.pixel_scale_matrix
            self.pixscale = np.sqrt(pixel_scale_matrix[0, 0]**2 + pixel_scale_matrix[1, 0]**2) * 3600  # arcsec/pixel
            self.center_ra, self.center_dec = self.wcs.wcs.crval
            self.wcs_header = self.wcs.to_header(relax=True)  # Store the full WCS header, including non-standard keywords
            self.print_corner_coordinates()

            # --- 🔍 Debugging Output ---
            print(f"Header CROTA2 Value: {header.get('CROTA2', 'Not Found')}")

            # Display WCS information
            from astropy.wcs.utils import proj_plane_pixel_scales

            # pixel scale (arcsec/px) – average the two axes (or keep both if you want)
            scales = proj_plane_pixel_scales(self.wcs) * 3600.0  # arcsec/pixel
            self.pixscale = float(np.mean(scales))

            # orientation
            def _cd_orientation(hdr):
                c11 = hdr.get("CD1_1"); c12 = hdr.get("CD1_2"); c21 = hdr.get("CD2_1"); c22 = hdr.get("CD2_2")
                pc11 = hdr.get("PC1_1"); pc12 = hdr.get("PC1_2"); pc21 = hdr.get("PC2_1"); pc22 = hdr.get("PC2_2")
                if None not in (c11, c12, c21, c22):
                    return np.degrees(np.arctan2(-c12, c11))
                if None not in (pc11, pc12, pc21, pc22):
                    # If PC given, use CDELT to scale
                    cdelt1 = float(hdr.get("CDELT1", 1.0))
                    return np.degrees(np.arctan2(-pc12 * cdelt1, pc11 * cdelt1))
                return None

            if 'CROTA2' in self.header:
                try:
                    self.orientation = float(self.header['CROTA2'])
                except Exception:
                    self.orientation = _cd_orientation(self.header)
            else:
                self.orientation = _cd_orientation(self.header)
            self.orientation_label.setText(f"Orientation: {self.orientation:.2f}°" if self.orientation is not None else "Orientation: N/A")

            # --- ✅ Ensure `self.orientation` is a float before using it ---
            if self.orientation is not None:
                try:
                    self.orientation = float(self.orientation)  # Final conversion check
                    print(f"Orientation: {self.orientation:.2f}°")
                    self.orientation_label.setText(f"Orientation: {self.orientation:.2f}°")
                except (ValueError, TypeError):
                    print("Final conversion failed. Orientation is not a float.")
                    self.orientation_label.setText("Orientation: N/A")
            else:
                print("Orientation is None.")
                self.orientation_label.setText("Orientation: N/A")

            print(f"WCS data loaded: RA={self.center_ra}, Dec={self.center_dec}, Pixel Scale={self.pixscale} arcsec/px")

        except ValueError as e:
            raise ValueError(f"WCS initialization error: {e}")

    def construct_fits_header_from_xisf(self, xisf_meta):
        """ Convert XISF metadata to a FITS header compatible with WCS """
        header = fits.Header()

        # numeric‐only keys (everything except CTYPE1/2)
        numeric_keys = {
            "CRPIX1", "CRPIX2", "CRVAL1", "CRVAL2",
            "CDELT1", "CDELT2", "A_ORDER", "B_ORDER",
            "AP_ORDER", "BP_ORDER"
        }

        for keyword, entries in xisf_meta.get('FITSKeywords', {}).items():
            for entry in entries:
                if 'value' not in entry:
                    continue
                val = entry['value']
                if keyword in ("CTYPE1", "CTYPE2"):
                    # always a string
                    header[keyword] = val
                elif keyword in numeric_keys:
                    # try integer, then float
                    try:
                        header[keyword] = int(val)
                    except (ValueError, TypeError):
                        header[keyword] = float(val)
                else:
                    # anything else just store raw
                    header[keyword] = val

        # ensure CTYPEs exist
        header.setdefault('CTYPE1', 'RA---TAN')
        header.setdefault('CTYPE2', 'DEC--TAN')

        # Add SIP distortion suffix if SIP coefficients are present
        if any(key in header for key in ["A_ORDER", "B_ORDER", "AP_ORDER", "BP_ORDER"]):
            header['CTYPE1'] = 'RA---TAN-SIP'
            header['CTYPE2'] = 'DEC--TAN-SIP'

        # Set default reference pixel to the center of the image
        header.setdefault('CRPIX1', self.image_data.shape[1] / 2)
        header.setdefault('CRPIX2', self.image_data.shape[0] / 2)

        # Retrieve RA and DEC values if available
        if 'RA' in xisf_meta['FITSKeywords']:
            header['CRVAL1'] = float(xisf_meta['FITSKeywords']['RA'][0]['value'])  # Reference RA
        if 'DEC' in xisf_meta['FITSKeywords']:
            header['CRVAL2'] = float(xisf_meta['FITSKeywords']['DEC'][0]['value'])  # Reference DEC

        # Calculate pixel scale if focal length and pixel size are available
        if 'FOCALLEN' in xisf_meta['FITSKeywords'] and 'XPIXSZ' in xisf_meta['FITSKeywords']:
            focal_length = float(xisf_meta['FITSKeywords']['FOCALLEN'][0]['value'])  # in mm
            pixel_size = float(xisf_meta['FITSKeywords']['XPIXSZ'][0]['value'])  # in μm
            pixel_scale = (pixel_size * 206.265) / focal_length  # arcsec/pixel
            header['CDELT1'] = -pixel_scale / 3600.0
            header['CDELT2'] = pixel_scale / 3600.0
        else:
            header['CDELT1'] = -2.77778e-4  # ~1 arcsecond/pixel
            header['CDELT2'] = 2.77778e-4

        # Populate CD matrix using the XISF LinearTransformationMatrix if available
        if 'XISFProperties' in xisf_meta and 'PCL:AstrometricSolution:LinearTransformationMatrix' in xisf_meta['XISFProperties']:
            linear_transform = xisf_meta['XISFProperties']['PCL:AstrometricSolution:LinearTransformationMatrix']['value']
            header['CD1_1'] = linear_transform[0][0]
            header['CD1_2'] = linear_transform[0][1]
            header['CD2_1'] = linear_transform[1][0]
            header['CD2_2'] = linear_transform[1][1]
        else:
            # Use pixel scale for CD matrix if no linear transformation is defined
            header['CD1_1'] = header['CDELT1']
            header['CD1_2'] = 0.0
            header['CD2_1'] = 0.0
            header['CD2_2'] = header['CDELT2']

        # Ensure numeric types for SIP distortion keywords if present
        sip_keywords = ["A_ORDER", "B_ORDER", "AP_ORDER", "BP_ORDER"]
        for sip_key in sip_keywords:
            if sip_key in xisf_meta['XISFProperties']:
                try:
                    value = xisf_meta['XISFProperties'][sip_key]['value']
                    header[sip_key] = int(value) if isinstance(value, str) and value.isdigit() else float(value)
                except ValueError:
                    pass  # Ignore any invalid conversion

        return header

    def print_corner_coordinates(self):
        """Print the RA/Dec coordinates of the four corners of the image for debugging purposes."""
        if not hasattr(self, 'wcs'):
            print("WCS data is incomplete, cannot calculate corner coordinates.")
            return

        width = self.main_image.width()
        height = self.main_image.height()

        # Define the corner coordinates
        corners = {
            "Top-Left": (0, 0),
            "Top-Right": (width, 0),
            "Bottom-Left": (0, height),
            "Bottom-Right": (width, height)
        }

        print("Corner RA/Dec coordinates:")
        for corner_name, (x, y) in corners.items():
            ra, dec = self.calculate_ra_dec_from_pixel(x, y)
            ra_hms = self.convert_ra_to_hms(ra)
            dec_dms = self.convert_dec_to_dms(dec)
            print(f"{corner_name}: RA={ra_hms}, Dec={dec_dms}")

    def calculate_ra_dec_from_pixel(self, x, y):
        if not hasattr(self, 'wcs') or self.wcs is None:
            return None, None
        ra, dec = self.wcs.pixel_to_world_values(x, y)
        return float(ra), float(dec)
                        


    def update_ra_dec_from_mouse(self, event):
        """Update RA and Dec based on mouse position over the main preview."""
        if self.main_image and self.wcs:
            pos = self.main_preview.mapToScene(event.pos())
            x, y = int(pos.x()), int(pos.y())

            if 0 <= x < self.main_image.width() and 0 <= y < self.main_image.height():
                ra, dec = self.calculate_ra_dec_from_pixel(x, y)
                ra_hms = self.convert_ra_to_hms(ra)
                dec_dms = self.convert_dec_to_dms(dec)

                # Update RA/Dec labels
                self.ra_label.setText(f"RA: {ra_hms}")
                self.dec_label.setText(f"Dec: {dec_dms}")

                # --- 🔍 Debugging Output ---
                #print(f"Current Orientation Type: {type(self.orientation)}, Value: {self.orientation}")

                # ✅ Ensure `self.orientation` is a float before formatting
                if self.orientation is not None:
                    try:
                        self.orientation = float(self.orientation)  # Final safeguard conversion
                        self.orientation_label.setText(f"Orientation: {self.orientation:.2f}°")
                    except (ValueError, TypeError):
                        print(f"Failed to format orientation: {self.orientation}")
                        self.orientation_label.setText("Orientation: N/A")
                else:
                    self.orientation_label.setText("Orientation: N/A")
        else:
            self.ra_label.setText("RA: N/A")
            self.dec_label.setText("Dec: N/A")
            self.orientation_label.setText("Orientation: N/A")



    def convert_ra_to_hms(self, ra_deg):
        """Convert Right Ascension in degrees to Hours:Minutes:Seconds format."""
        ra_hours = ra_deg / 15.0  # Convert degrees to hours
        hours = int(ra_hours)
        minutes = int((ra_hours - hours) * 60)
        seconds = (ra_hours - hours - minutes / 60.0) * 3600
        return f"{hours:02d}h{minutes:02d}m{seconds:05.2f}s"

    def convert_dec_to_dms(self, dec_deg):
        """Convert Declination in degrees to Degrees:Minutes:Seconds format."""
        sign = "-" if dec_deg < 0 else "+"
        dec_deg = abs(dec_deg)
        degrees = int(dec_deg)
        minutes = int((dec_deg - degrees) * 60)
        seconds = (dec_deg - degrees - minutes / 60.0) * 3600
        degree_symbol = "\u00B0"
        return f"{sign}{degrees:02d}{degree_symbol}{minutes:02d}m{seconds:05.2f}s"                 

    def check_astrometry_data(self, header_like) -> bool:
        if header_like is None:
            return False
        try:
            h = header_like  # fits.Header is Mapping-like, dict is fine
            has_ctypes = ("CTYPE1" in h) and ("CTYPE2" in h)
            has_crval  = ("CRVAL1" in h) and ("CRVAL2" in h)
            has_scale  = ("CD1_1" in h) or ("CDELT1" in h) or ("PC1_1" in h)
            return bool(has_ctypes or (has_crval and has_scale))
        except Exception:
            return False


    def _settings(self) -> QSettings:
        return getattr(self, "settings", None) or QSettings()

    # ─────────────────────────────────────────
    # Minor-planet DB path via QSettings
    # ─────────────────────────────────────────
    def _minor_settings(self) -> QSettings:
        return getattr(self, "settings", None) or QSettings()

    # ─────────────────────────────────────────
    # Minor-planet DB path via QSettings
    # ─────────────────────────────────────────
    def _load_minor_db_path(self) -> str:
        """Load cached DB path from QSettings and update the label."""
        from pathlib import Path
        from pro import minorbodycatalog as mbc

        path = self.settings.value("wimi/minorbody_db_path", "", type=str)
        self.minor_db_path = path or ""

        if not hasattr(self, "minor_db_label"):
            return self.minor_db_path

        if not path:
            self.minor_db_label.setText("Database: not downloaded")
            return self.minor_db_path

        p = Path(path)
        if not p.is_file():
            self.minor_db_label.setText("Database: not downloaded")
            return self.minor_db_path

        # Try to load local manifest to show version + counts
        manifest_path = p.parent / mbc.DEFAULT_MANIFEST_BASENAME
        manifest = mbc.load_local_manifest(manifest_path)

        if manifest is not None:
            ast = f"{manifest.counts_asteroids:,}"
            com = f"{manifest.counts_comets:,}"
            self.minor_db_label.setText(
                f"Database: v{manifest.version} — {ast} asteroids, {com} comets"
            )
        else:
            self.minor_db_label.setText(f"Database: {p.name}")

        return self.minor_db_path

    def _save_minor_db_path(self, path: str) -> None:
        """Save DB path to QSettings and update label."""
        self.settings.setValue("wimi/minorbody_db_path", path)
        self.settings.sync()
        self.minor_db_path = path
        self._load_minor_db_path()   # reuse logic to update label

    def download_minor_body_catalog(self, force: bool = False) -> None:
        """
        Download or update the minor-body SQLite DB using pro.minorbodycatalog,
        showing a progress UI while it runs.

        Shift-clicking the button sets force=True (handled in the click slot).
        """
        from PyQt6.QtCore import QEventLoop
        from pro import minorbodycatalog as mbc

        data_dir = self._minorbody_data_dir()

        # Disable the button while downloading
        if hasattr(self, "btn_minor_download"):
            self.btn_minor_download.setEnabled(False)

        self.status_label.setText("Status: Downloading minor-body catalog…")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

        dlg = QProgressDialog(
            "Downloading minor-body catalog…\n"
            "This may take a minute on first use.",
            "",
            0,
            0,
            self,
        )
        dlg.setWindowTitle("Minor-Body Catalog")
        dlg.setWindowModality(Qt.WindowModality.ApplicationModal)
        dlg.setAutoClose(False)
        dlg.setAutoReset(False)
        dlg.setCancelButton(None)
        dlg.show()

        worker = MinorBodyDownloadWorker(
            data_dir=data_dir,
            force_refresh=force,
            parent=self,
        )

        result = {
            "ok": False,
            "db_path": "",
            "version": "",
            "ast": 0,
            "com": 0,
            "error": "",
        }

        loop = QEventLoop()

        def _on_ok(db_path: str, version: str, asteroids: int, comets: int):
            result["ok"] = True
            result["db_path"] = db_path
            result["version"] = version
            result["ast"] = asteroids
            result["com"] = comets
            loop.quit()

        def _on_fail(msg: str):
            result["ok"] = False
            result["error"] = msg
            loop.quit()

        worker.finished_ok.connect(_on_ok)
        worker.failed.connect(_on_fail)
        worker.finished.connect(loop.quit)

        worker.start()
        loop.exec()

        dlg.close()
        QApplication.restoreOverrideCursor()

        if hasattr(self, "btn_minor_download"):
            self.btn_minor_download.setEnabled(True)

        if not result["ok"]:
            self.status_label.setText("Status: Minor-body catalog download failed.")
            msg = result["error"] or "Unknown error."
            QMessageBox.critical(
                self,
                "Minor-Body Catalog Error",
                f"Failed to download minor-body catalog:\n{msg}",
            )
            return

        # Success
        self._save_minor_db_path(result["db_path"])

        ast = f"{result['ast']:,}"
        com = f"{result['com']:,}"

        self.status_label.setText("Status: Minor-body catalog ready.")
        QMessageBox.information(
            self,
            "Minor-Body Catalog",
            f"Minor-body catalog downloaded/updated successfully.\n"
            f"Version: {result['version']}\n"
            f"Asteroids: {ast}\n"
            f"Comets: {com}",
        )

    def _ensure_minor_planet_db(self) -> str | None:
        """
        Ensure the minor-body SQLite DB exists.

        Returns:
            Path string to the DB, or None if the user cancels or download fails.
        """
        from pathlib import Path

        # 1) If we already know the path and it exists, just reuse it.
        p_str = getattr(self, "minor_db_path", "") or self.settings.value(
            "wimi/minorbody_db_path", "", type=str
        )
        if p_str:
            p = Path(p_str)
            if p.is_file():
                self.minor_db_path = p_str
                return p_str

        # 2) Ask user to download / update
        reply = QMessageBox.question(
            self,
            "Minor-Body Catalog",
            "The minor-body catalog is not available.\n"
            "Do you want to download or update it now?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return None

        # This will check manifest and only re-download if needed
        self.download_minor_body_catalog(force=False)

        p_str = getattr(self, "minor_db_path", "")
        if p_str and Path(p_str).is_file():
            return p_str

        QMessageBox.critical(
            self,
            "Minor-Body Catalog",
            "Failed to download or locate the minor-body database.",
        )
        return None


    # ------------------------------------------------------------------
    # Minor-body helpers
    # ------------------------------------------------------------------
    def _minorbody_data_dir(self) -> Path:
        """
        Return the per-user data directory for the minor-body catalog.

        Uses QStandardPaths.AppDataLocation so it automatically maps to:
          - Windows: %LOCALAPPDATA%/SetiAstroSuitePro (or similar)
          - macOS:   ~/Library/Application Support/SetiAstroSuitePro
          - Linux:   ~/.local/share/SetiAstroSuitePro
        and then a 'minor_bodies' subfolder.
        """
        base = QStandardPaths.writableLocation(
            QStandardPaths.StandardLocation.AppDataLocation
        )
        base_path = Path(base) if base else Path.home() / ".saspro"
        data_dir = base_path / "minor_bodies"
        data_dir.mkdir(parents=True, exist_ok=True)
        return data_dir

    def _get_full_fov_center_and_radius_deg(self):
        """
        Compute (RA, Dec) at the image center and a radius (deg)
        that roughly covers the whole image (center to a corner).

        Returns (ra_center, dec_center, radius_deg) or (None, None, None)
        on failure.
        """
        if self.main_image is None or self.wcs is None or self.pixscale is None:
            return None, None, None

        try:
            # self.main_image is a QImage here (like in search_entire_image)
            width  = self.main_image.width()
            height = self.main_image.height()

            cx = width  / 2.0
            cy = height / 2.0

            ra_center, dec_center = self.calculate_ra_dec_from_pixel(cx, cy)
            if ra_center is None or dec_center is None:
                return None, None, None

            # pixel radius from center to bottom-right corner (same idea as SIMBAD)
            corner_x, corner_y = width, height
            dx = corner_x - cx
            dy = corner_y - cy
            radius_px = float((dx * dx + dy * dy) ** 0.5)

            # convert pixel radius to degrees using pixscale (arcsec/pixel)
            radius_deg = float((radius_px * self.pixscale) / 3600.0)

            return ra_center, dec_center, radius_deg
        except Exception:
            return None, None, None

    # ─────────────────────────────────────────
    # Observation datetime helpers
    # ─────────────────────────────────────────
    def _get_observation_datetime_from_header(self):
        """
        Try to extract observation datetime from:

          1) self.original_header   (full FITS header, if present)
          2) self.wcs.wcs.dateobs   (DATE-OBS stored in the WCS header)
          3) MJD-OBS in the WCS header

        Returns a Python datetime (UTC) or None.
        """

        def _extract_from_header(hdr: fits.Header):
            """Best-effort extraction from a FITS Header."""
            # 1) DATE-OBS (+ optional TIME-OBS)
            try:
                if "DATE-OBS" in hdr:
                    val = str(hdr["DATE-OBS"]).strip()

                    # Already full ISO timestamp
                    if "T" in val:
                        t = Time(val, format="isot", scale="utc")
                        return t.to_datetime()

                    # DATE-OBS + TIME-OBS split
                    if "TIME-OBS" in hdr:
                        val2 = str(hdr["TIME-OBS"]).strip()
                        t = Time(f"{val}T{val2}", format="isot", scale="utc")
                        return t.to_datetime()
            except Exception as e:
                print(f"[WIMI] DATE-OBS parse failed from header: {e!r}")

            # 2) MJD-OBS
            try:
                if "MJD-OBS" in hdr:
                    t = Time(float(hdr["MJD-OBS"]), format="mjd", scale="utc")
                    return t.to_datetime()
            except Exception as e:
                print(f"[WIMI] MJD-OBS parse failed from header: {e!r}")

            # 3) JD
            try:
                if "JD" in hdr:
                    t = Time(float(hdr["JD"]), format="jd", scale="utc")
                    return t.to_datetime()
            except Exception as e:
                print(f"[WIMI] JD parse failed from header: {e!r}")

            return None

        # --- 1) Try original_header if we have it ---
        hdr = getattr(self, "original_header", None)
        if isinstance(hdr, fits.Header):
            dt = _extract_from_header(hdr)
            if dt is not None:
                print(f"[WIMI] Observation time from original_header: {dt.isoformat()}")
                return dt

        # --- 2) Fall back to WCS header (this matches your Metadata.WCS.* block) ---
        w = getattr(self, "wcs", None)
        if w is not None:
            # 2a) native dateobs field
            dateobs = None
            try:
                dateobs = getattr(w.wcs, "dateobs", None)
            except Exception:
                dateobs = None

            if dateobs:
                try:
                    # Handles 'YYYY-MM-DDTHH:MM:SS...' (your case)
                    fmt = "isot" if "T" in dateobs else "iso"
                    t = Time(dateobs, format=fmt, scale="utc")
                    dt = t.to_datetime()
                    print(f"[WIMI] Observation time from WCS.dateobs: {dt.isoformat()}")
                    return dt
                except Exception as e:
                    print(f"[WIMI] failed to parse WCS.dateobs={dateobs!r}: {e!r}")

            # 2b) If that fails, try MJD-OBS in the WCS header
            try:
                w_hdr = w.to_header(relax=True)
                if "MJD-OBS" in w_hdr:
                    t = Time(float(w_hdr["MJD-OBS"]), format="mjd", scale="utc")
                    dt = t.to_datetime()
                    print(f"[WIMI] Observation time from WCS MJD-OBS: {dt.isoformat()}")
                    return dt
            except Exception as e:
                print(f"[WIMI] failed to parse MJD-OBS from WCS header: {e!r}")

        # --- 3) Give up; caller will prompt ---
        return None


    def _prompt_for_observation_datetime(self):
        """
        Show a small dialog asking for observation date/time if header doesn't have it.
        Returns a Python datetime or None if user cancels.
        """
        dialog = QDialog(self)
        dialog.setWindowTitle("Observation Date & Time")

        layout = QFormLayout(dialog)
        lbl = QLabel(
            "The FITS header does not contain an observation date/time.\n"
            "Please enter the midpoint of the exposure (UTC):"
        )
        lbl.setWordWrap(True)
        layout.addRow(lbl)

        dt_edit = QDateTimeEdit(dialog)
        dt_edit.setCalendarPopup(True)
        dt_edit.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
        dt_edit.setDateTime(QDateTime.currentDateTimeUtc())
        layout.addRow("Date & Time (UTC):", dt_edit)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        layout.addWidget(buttons)

        def _accept():
            dialog.accept()

        def _reject():
            dialog.reject()

        buttons.accepted.connect(_accept)
        buttons.rejected.connect(_reject)

        dialog.setLayout(layout)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            qdt = dt_edit.dateTime().toUTC()
            py_dt = qdt.toPyDateTime()
            return py_dt
        return None

    def _get_or_prompt_observation_datetime(self):
        """
        Returns a Python datetime for the observation. Uses:
        1) cached self.observation_datetime
        2) header (DATE-OBS / TIME-OBS / MJD-OBS / JD)
        3) prompt the user
        """
        # Cached?
        if getattr(self, "observation_datetime", None) is not None:
            return self.observation_datetime

        # Try header
        dt = self._get_observation_datetime_from_header()
        if dt is None:
            dt = self._prompt_for_observation_datetime()
        self.observation_datetime = dt
        return dt

    def perform_minor_body_search(self, mode: str = "circle"):
        """
        Search local minor-planet/comet database in either:

          - the current WCS-defined search circle (mode="circle"), or
          - the entire image FOV (mode="full").

        Uses the observation date/time from header or prompt.
        """
        # Scope from combo overrides the default 'mode' argument
        if hasattr(self, "minor_scope_combo"):
            scope = self.minor_scope_combo.currentText()
            if "Entire" in scope:
                mode = "full"
            else:
                mode = "circle"

        # Persist current minor-body settings so they survive restarts
        self.settings.setValue("wimi/minor/asteroid_H_max", self.minor_ast_H_spin.value())
        self.settings.setValue("wimi/minor/asteroid_max", self.minor_ast_max_spin.value())
        self.settings.setValue("wimi/minor/comet_H_max", self.minor_com_H_spin.value())
        self.settings.setValue("wimi/minor/comet_max", self.minor_com_max_spin.value())

        # Optional specific target from UI (designation or name)
        target_filter = ""
        if hasattr(self, "minor_target_edit") and self.minor_target_edit is not None:
            target_filter = self.minor_target_edit.text().strip()

        # Need WCS + pixscale for either mode
        if self.wcs is None or self.pixscale is None:
            QMessageBox.warning(
                self,
                "No WCS",
                "No valid WCS/pixel scale is available. Please solve the image first."
            )
            return

        # 1) Determine center + radius
        if mode == "full":
            ra_center, dec_center, radius_deg = self._get_full_fov_center_and_radius_deg()
            if ra_center is None or radius_deg is None:
                QMessageBox.warning(
                    self,
                    "Minor-Body Search",
                    "Could not determine the field of view for this image."
                )
                return
        else:
            # Default: use the defined search circle
            if not self.circle_center or self.circle_radius <= 0:
                QMessageBox.warning(
                    self,
                    "No Search Area",
                    "Please define a search circle by Shift-clicking and dragging."
                )
                return

            ra_center, dec_center = self.calculate_ra_dec_from_pixel(
                self.circle_center.x(), self.circle_center.y()
            )
            if ra_center is None or dec_center is None:
                QMessageBox.warning(
                    self,
                    "Invalid Coordinates",
                    "Could not determine the RA/Dec of the circle center."
                )
                return

            # radius in degrees from circle radius (pixels) and pixscale (arcsec/pixel)
            radius_deg = float((self.circle_radius * self.pixscale) / 3600.0)

        # 2) Observation datetime
        obs_dt = self._get_or_prompt_observation_datetime()
        if obs_dt is None:
            QMessageBox.information(
                self,
                "Minor-Body Search",
                "No observation date/time specified; search cancelled."
            )
            return

        # 3) DB path
        db_path = self._ensure_minor_planet_db()
        if not db_path:
            return

        # Optional: target name (e.g. "Semiramis")
        target_filter = None
        if hasattr(self, "minor_target_edit"):
            t = self.minor_target_edit.text().strip()
            if t:
                target_filter = t

        # 4) Query DB
        try:
            results = self._query_minor_planets_from_db(
                db_path=db_path,
                ra_center=ra_center,
                dec_center=dec_center,
                radius_deg=radius_deg,
                obs_datetime=obs_dt,
                target_filter=target_filter,
            )
        except Exception as e:
            QMessageBox.critical(self, "Minor-Body Search", f"Query failed:\n{e}")
            return

        if not results:
            where = "the specified region" if mode == "circle" else "the image field of view"
            extra = ""
            if target_filter:
                extra = f"\n(Target '{target_filter}' not found in the catalog subset.)"
            QMessageBox.information(
                self,
                "Minor-Body Search",
                f"No minor planets or comets found in {where} at the given time.{extra}"
            )
            return

        # 5) Populate tree & overlay (unchanged)
        self.results_tree.clear()
        query_results = []

        for r in results:
            ra  = float(r["ra"])
            dec = float(r["dec"])
            name  = r.get("name", r.get("designation", "N/A"))
            mag   = r.get("mag", "N/A")
            qtype = r.get("type", "Minor body")
            dist  = r.get("distance", "N/A")   # AU or whatever you decide
            src   = r.get("source", "MinorDB")

            item = QTreeWidgetItem([
                f"{ra:.6f}",
                f"{dec:.6f}",
                name,
                str(mag),
                "MP",
                qtype,
                str(dist),
                src,
            ])
            self.results_tree.addTopLevelItem(item)

            query_results.append({
                "ra": ra,
                "dec": dec,
                "name": name,
                "diameter": mag,
                "short_type": "MP",
                "long_type": qtype,
                "redshift": dist,
                "comoving_distance": dist,
                "source": src,
            })

        self.main_preview.set_query_results(query_results)
        self.query_results = query_results
        self.update_object_count()

    def _query_minor_planets_from_db(
        self,
        db_path,
        ra_center,
        dec_center,
        radius_deg,
        obs_datetime,
        target_filter: str | None = None,
    ):
        """
        Query the local minor-body DB using MinorBodyCatalog + Skyfield.

        Returns list of dicts:
            ra, dec, name, mag, type, distance, source, designation
        """
        from pathlib import Path
        from astropy.time import Time
        import pandas as pd
        from pro import minorbodycatalog as mbc

        db_path = Path(db_path)
        if not db_path.is_file():
            raise FileNotFoundError(f"Minor-body DB not found: {db_path}")

        # 1) Convert observation datetime to JD
        jd = Time(obs_datetime).jd

        print(
            f"[MinorBodies] Search: RA={ra_center:.6f} deg  "
            f"Dec={dec_center:.6f} deg  radius={radius_deg:.4f} deg  JD={jd:.6f}"
        )
        print(f"[MinorBodies] DB path: {db_path}")

        # Read user limits from the UI, with safe fallbacks
        try:
            H_ast = float(self.minor_ast_H_spin.value())
        except Exception:
            H_ast = 20.0
        try:
            N_ast = int(self.minor_ast_max_spin.value())
        except Exception:
            N_ast = 50000
        try:
            H_com = float(self.minor_com_H_spin.value())
        except Exception:
            H_com = 15.0
        try:
            N_com = int(self.minor_com_max_spin.value())
        except Exception:
            N_com = 5000

        # If searching for a specific target, override limits so we don't miss it
        if target_filter:
            H_ast = 40.0
            H_com = 40.0
            N_ast = 2_000_000
            N_com = 100_000
            print(
                "[MinorBodies] target search – overriding limits to near-full catalog: "
                f"Asteroids H<={H_ast}, max={N_ast}; "
                f"Comets H<={H_com}, max={N_com}"
            )

        print(
            f"[MinorBodies] user limits: "
            f"Asteroids H<={H_ast}, max={N_ast}; "
            f"Comets H<={H_com}, max={N_com}"
        )
        if target_filter:
            print(f"[MinorBodies] target filter: {target_filter!r}")

        # 2) Open catalog
        cat = mbc.MinorBodyCatalog(db_path)

        # 3) Get a manageable subset of bright objects
        ast_df: pd.DataFrame | None = None
        com_df: pd.DataFrame | None = None

        try:
            ast_df = cat.get_bright_asteroids(H_max=H_ast, limit=N_ast)
        except Exception as e:
            print(f"[MinorBodies] get_bright_asteroids FAILED: {e!r}")
            ast_df = None

        try:
            com_df = cat.get_bright_comets(H_max=H_com, limit=N_com)
        except Exception as e:
            print(f"[MinorBodies] get_bright_comets FAILED: {e!r}")
            com_df = None

        n_ast = len(ast_df) if ast_df is not None else 0
        n_com = len(com_df) if com_df is not None else 0
        print(f"[MinorBodies] bright subset (before target filter): asteroids={n_ast}, comets={n_com}")

        asteroid_designations: set[str] = set()
        comet_designations: set[str] = set()
        asteroid_mag: dict[str, float] = {}
        comet_mag: dict[str, float] = {}

        # --- Build mag lookup + designation sets ---
        if ast_df is not None and len(ast_df):
            mag_keys_ast = ("magnitude_H", "H", "absolute_magnitude")
            for _, row in ast_df.iterrows():
                desig = row.get("designation", "") or row.get("name", "")
                if not desig:
                    continue
                asteroid_designations.add(str(desig))

                mag_val = None
                for mk in mag_keys_ast:
                    if mk in row and row[mk] is not None:
                        try:
                            mag_val = float(row[mk])
                        except Exception:
                            mag_val = None
                        break
                if mag_val is not None:
                    asteroid_mag[str(desig)] = mag_val

        if com_df is not None and len(com_df):
            mag_keys_com = ("absolute_magnitude", "magnitude_H", "H")
            for _, row in com_df.iterrows():
                desig = row.get("designation", "") or row.get("name", "")
                if not desig:
                    continue
                comet_designations.add(str(desig))

                mag_val = None
                for mk in mag_keys_com:
                    if mk in row and row[mk] is not None:
                        try:
                            mag_val = float(row[mk])
                        except Exception:
                            mag_val = None
                        break
                if mag_val is not None:
                    comet_mag[str(desig)] = mag_val

        # --- Combine into a single DataFrame for Skyfield ---
        frames: list[pd.DataFrame] = []
        if ast_df is not None and len(ast_df):
            frames.append(ast_df)
        if com_df is not None and len(com_df):
            frames.append(com_df)

        if not frames:
            print("[MinorBodies] NO rows to compute positions for, returning [].")
            return []

        rows_df = pd.concat(frames, ignore_index=True)
        total_rows = len(rows_df)
        print(f"[MinorBodies] total candidate rows sent to Skyfield: {total_rows}")
        print(f"[MinorBodies] columns in candidate DataFrame: {list(rows_df.columns)}")

        # --- Robust target-name filtering across ALL text columns -----
        if target_filter:
            tf = target_filter.strip().lower()
            if tf:
                mask = pd.Series(False, index=rows_df.index)
                text_cols = [
                    c for c in rows_df.columns
                    if rows_df[c].dtype == object
                ]
                for col in text_cols:
                    s = rows_df[col].astype(str).str.lower()
                    col_mask = s.str.contains(tf, na=False)
                    count = int(col_mask.sum())
                    print(f"[MinorBodies] target filter: column '{col}' matches={count}")
                    mask |= col_mask

                rows_df = rows_df[mask]
                total_rows = len(rows_df)
                print(f"[MinorBodies] target filter: total rows after filter: {total_rows}")

                if total_rows == 0:
                    print("[MinorBodies] target filter removed all rows; returning [].")
                    return []
        # ----------------------------------------------------------------

        # --------------------------------------------------------------
        # Progress dialog for the Skyfield loop
        # --------------------------------------------------------------
        prog = QProgressDialog(
            "Computing minor-body ephemerides...\n"
            "This may take a while on first run.",
            "Cancel",
            0,
            total_rows,
            self,
        )
        prog.setWindowTitle("Minor-Body Search")
        prog.setWindowModality(Qt.WindowModality.ApplicationModal)
        prog.setAutoClose(True)
        prog.setAutoReset(True)
        prog.setMinimumDuration(500)
        prog.show()

        cancelled = {"flag": False}

        def progress_cb(done: int, total: int) -> bool:
            if cancelled["flag"]:
                return False
            prog.setMaximum(total)
            prog.setValue(done)
            QApplication.processEvents()
            if prog.wasCanceled():
                cancelled["flag"] = True
                return False
            return True

        # 4) Compute positions with Skyfield at this JD
        print(
            f"[MinorBodies] compute_positions_skyfield: sending {total_rows} rows, jd={jd}"
        )
        try:
            positions = cat.compute_positions_skyfield(
                asteroid_rows=rows_df,
                jd=jd,
                ephemeris_path=None,
                topocentric=None,
                progress_cb=progress_cb,
                debug=True,
            )
        finally:
            prog.close()

        if cancelled["flag"]:
            print("[MinorBodies] search cancelled by user.")
            return []

        print(f"[MinorBodies] positions returned from Skyfield: {len(positions)}")

        # 5) Filter to cone
        results: list[dict[str, object]] = []
        kept = 0

        for pos in positions:
            ra = float(pos["ra_deg"])
            dec = float(pos["dec_deg"])
            ang = self.calculate_angular_distance(ra_center, dec_center, ra, dec)
            if ang > radius_deg:
                continue

            kept += 1
            desig = pos.get("designation", "") or "Unknown"
            desig_str = str(desig)

            if desig_str in asteroid_designations:
                kind = "Asteroid"
                mag = asteroid_mag.get(desig_str, "N/A")
            elif desig_str in comet_designations:
                kind = "Comet"
                mag = comet_mag.get(desig_str, "N/A")
            else:
                kind = "Minor body"
                mag = "N/A"

            results.append(
                {
                    "ra": ra,
                    "dec": dec,
                    "name": desig_str,
                    "designation": desig_str,
                    "mag": mag,
                    "type": kind,
                    "distance": pos.get("distance_au", "N/A"),
                    "source": "MinorDB",
                }
            )

        print(f"[MinorBodies] objects inside cone: {kept}")
        return results
    

    def _get_astap_exe(self) -> str:
        s = self._settings()
        # preferred key (what SettingsDialog writes)
        p = s.value("paths/astap", "", type=str)
        if p:
            return p
        # migrate legacy key if present
        legacy = s.value("astap/exe_path", "", type=str)
        if legacy:
            s.setValue("paths/astap", legacy)
            s.remove("astap/exe_path")
            s.sync()
            return legacy
        return ""

    def _set_astap_exe(self, path: str) -> None:
        s = self._settings()
        s.setValue("paths/astap", path)
        s.sync()

    def plate_solve_image(self):
        """
        Attempts to plate-solve the loaded image using ASTAP,
        first trying a seeded solve (RA, SPD, scale, binning),
        then falling back to a blind solve if anything is missing.
        On success, updates self.header and self.wcs.
        """
        if not hasattr(self, 'image_path') or not self.image_path:
            return

        # 1) Ensure ASTAP path
        astap_exe = self._get_astap_exe()
        if not astap_exe or not os.path.exists(astap_exe):
            # last-resort browse if nothing in settings (keeps existing behavior)
            filt = "Executables (*.exe);;All Files (*)" if sys.platform.startswith("win") else "Executables (*)"
            new_path, _ = QFileDialog.getOpenFileName(self, "Select ASTAP Executable", "", filt)
            if not new_path:
                return
            astap_exe = new_path
            self._set_astap_exe(astap_exe)

        # 2) Write out the normalized FITS for ASTAP
        normalized = self.stretch_image(self.image_data.astype(np.float32))
        try:
            tmp_path = self.save_temp_fits_image(normalized, self.image_path)
        except Exception as e:
            QMessageBox.critical(self, "Plate Solve", f"Error saving temp FITS: {e}")
            return

        # 3) Seed arguments from header
        raw_hdr = None
        if isinstance(self.original_header, fits.Header):
            raw_hdr = self.original_header
        elif self.image_path.lower().endswith(('.fits','.fit')):
            with fits.open(self.image_path, memmap=False) as hdul:
                raw_hdr = hdul[0].header

        seed_args = []
        if isinstance(raw_hdr, fits.Header):
            # debug-dump
            print("🔍 Raw header contents:")
            for k,v in raw_hdr.items():
                print(f"    {k} = {v}")

            try:
                # RA→hours, SPD
                ra_deg = float(raw_hdr["CRVAL1"])
                dec_deg= float(raw_hdr["CRVAL2"])
                ra_h    = ra_deg / 15.0
                spd     = dec_deg + 90.0

                # plate scale from CD matrix (°/px→″/px)
                cd1 = float(raw_hdr.get("CD1_1", raw_hdr.get("CDELT1",0)))
                cd2 = float(raw_hdr.get("CD2_1", raw_hdr.get("CDELT2",0)))
                scale = np.hypot(cd1, cd2) * 3600.0

                # apply XBINNING/YBINNING
                bx = int(raw_hdr.get("XBINNING", 1))
                by = int(raw_hdr.get("YBINNING", bx))
                if bx != by:
                    print(f"⚠️ Unequal binning: {bx}×{by}, averaging.")
                binf = (bx+by)/2.0
                scale *= binf

                seed_args = [
                    "-ra",    f"{ra_h:.6f}",
                    "-spd",   f"{spd:.6f}",
                    "-scale", f"{scale:.3f}"
                ]
                print(f"🔸 Seeding ASTAP: RA={ra_h:.6f}h, SPD={spd:.6f}°, scale={scale:.3f}\"/px (×{binf} bin)")
            except Exception as e:
                print("⚠️ Failed to build seed args, will do blind solve:", e)

        # 4) Build ASTAP args
        if seed_args:
            args = ["-f", tmp_path] + seed_args + ["-wcs", "-sip"]
        else:
            args = ["-f", tmp_path, "-r", "179", "-fov", "0", "-z", "0", "-wcs", "-sip"]

        print("▶️ Running ASTAP with arguments:", args)

        # create and launch the process
        process = QProcess(self)
        process.start(astap_exe, args)
        if not process.waitForStarted(5000):
            #QMessageBox.critical(self, "Plate Solve", "Failed to start ASTAP process.")
            os.remove(tmp_path)
            
            return None
        if not process.waitForFinished(300000):
            #QMessageBox.critical(self, "Plate Solve", "ASTAP process timed out.")
            os.remove(tmp_path)
            return None

        exit_code = process.exitCode()
        stdout = process.readAllStandardOutput().data().decode()
        stderr = process.readAllStandardError().data().decode()
        print("ASTAP exit code:", exit_code)
        print("ASTAP STDOUT:\n", stdout)
        print("ASTAP STDERR:\n", stderr)
        
        if exit_code != 0:
            os.remove(tmp_path)
            #QMessageBox.warning(self, "Plate Solve", "ASTAP failed. Falling back to blind solve.")
            
            return None

        # --- Retrieve the initial solved header from the temporary FITS file ---
        try:
            with fits.open(tmp_path, memmap=False) as hdul:
                solved_header = dict(hdul[0].header)
            for key in ["COMMENT", "HISTORY", "END"]:
                solved_header.pop(key, None)
            print("Initial solved header retrieved from temporary FITS file:")
            for key, value in solved_header.items():
                print(f"{key} = {value}")
        except Exception as e:
            QMessageBox.critical(self, "Plate Solve", f"Error reading solved header: {e}")
            os.remove(tmp_path)
            
            return None

        # --- Check for a .wcs file and merge its header if present ---
        wcs_path = os.path.splitext(tmp_path)[0] + ".wcs"
        if os.path.exists(wcs_path):
            try:
                wcs_header = {}
                with open(wcs_path, "r") as f:
                    text = f.read()
                    # Matches a FITS header keyword and its value (with an optional comment).
                    pattern = r"(\w+)\s*=\s*('?[^/']*'?)[\s/]"
                    for match in re.finditer(pattern, text):
                        key = match.group(1).strip().upper()
                        val = match.group(2).strip()
                        if val.startswith("'") and val.endswith("'"):
                            val = val[1:-1].strip()
                        wcs_header[key] = val
                wcs_header.pop("END", None)
                print("WCS header retrieved from .wcs file:")
                for key, value in wcs_header.items():
                    print(f"{key} = {value}")
                # Merge the parsed WCS header into the solved header.
                solved_header.update(wcs_header)
            except Exception as e:
                print("Error reading .wcs file:", e)
        else:
            print("No .wcs file found; using header from temporary FITS.")

        # --- If loaded from a slot, merge the original file path from slot metadata ---
        if getattr(self, "_from_slot", False) and hasattr(self, "_slot_meta"):
            if "file_path" not in solved_header and "file_path" in self._slot_meta:
                solved_header["file_path"] = self._slot_meta["file_path"]
                print("Merged file_path from slot metadata into solved header.")

        # --- Add any missing required WCS keywords ---
        required_keys = {
            "CTYPE1": "RA---TAN",
            "CTYPE2": "DEC--TAN",
            "RADECSYS": "ICRS",
            "WCSAXES": 2,
            # CRVAL1, CRVAL2, CRPIX1, CRPIX2 are ideally provided by ASTAP.
        }
        for key, default in required_keys.items():
            if key not in solved_header:
                solved_header[key] = default
                print(f"Added missing key {key} with default value {default}.")

        # --- Convert keys that are expected to be numeric from strings to numbers ---
        expected_numeric_keys = {
            "CRPIX1", "CRPIX2", "CRVAL1", "CRVAL2", "CROTA1", "CROTA2",
            "CDELT1", "CDELT2", "CD1_1", "CD1_2", "CD2_1", "CD2_2", "WCSAXES"
        }
        for key in expected_numeric_keys:
            if key in solved_header:
                try:
                    # For keys that should be integers, you can use int(float(...)) if necessary.
                    solved_header[key] = float(solved_header[key])
                except ValueError:
                    print(f"Warning: Could not convert {key} value '{solved_header[key]}' to float.")

        # --- Ensure integer keywords are stored as integers ---
        for key in ["WCSAXES", "NAXIS", "NAXIS1", "NAXIS2", "NAXIS3"]:
            if key in solved_header:
                try:
                    solved_header[key] = int(float(solved_header[key]))
                except ValueError:
                    print(f"Warning: Could not convert {key} value '{solved_header[key]}' to int.")


        os.remove(tmp_path)
        print("ASTAP plate solving successful. Final solved header:")
        for key, value in solved_header.items():
            print(f"{key} = {value}")

        # --------------------------------------------------------------------
        # 1) Make sure A_ORDER/B_ORDER exist in pairs:
        if "B_ORDER" in solved_header and "A_ORDER" not in solved_header:
            solved_header["A_ORDER"] = solved_header["B_ORDER"]
        if "A_ORDER" in solved_header and "B_ORDER" not in solved_header:
            solved_header["B_ORDER"] = solved_header["A_ORDER"]

        # 2) Convert SIP‐order keywords to ints:
        for key in ("A_ORDER","B_ORDER","AP_ORDER","BP_ORDER"):
            if key in solved_header:
                solved_header[key] = int(float(solved_header[key]))

        # 3) Convert every SIP coefficient to float:
        for k in list(solved_header):
            if re.match(r"^(?:A|B|AP|BP)_[0-9]+_[0-9]+$", k):
                solved_header[k] = float(solved_header[k])

        # --------------------------------------------------------------------
        # 4) Now rebuild your FITS header from the dict, preserving ordering:
        new_hdr = fits.Header()
        for key, val in solved_header.items():
            # skip any stray non‑FITS metadata
            if key == "file_path":
                continue
            new_hdr[key] = val

        # 5) Finally swap in the new header and re-init WCS (with SIP!)
        self.header = new_hdr
        try:
            self.apply_wcs_header(self.header)
            self.status_label.setText("Status: ASTAP solve succeeded.")
        except Exception as e:
            QMessageBox.critical(self, "Plate Solve", f"Error initializing WCS from solved header:\n{e}")
            return

        return solved_header


    def save_temp_fits_image(self, normalized_image, image_path: str):
        """
        Save the normalized_image as a FITS file to a temporary file.
        
        If the original image is FITS, this method retrieves the stored metadata
        from the ImageManager and passes it directly to save_image().
        If not, it generates a minimal header.
        
        Returns the path to the temporary FITS file.
        """
        # Always save as FITS.
        selected_format = "fits"
        bit_depth = "32-bit floating point"
        is_mono = (normalized_image.ndim == 2 or 
                   (normalized_image.ndim == 3 and normalized_image.shape[2] == 1))
        
        # If the original image is FITS, try to get its stored metadata.
        original_header = None
        if image_path.lower().endswith((".fits", ".fit")):
            if self.parent() and hasattr(self.parent(), "image_manager"):
                # Use the metadata from the current slot.
                _, meta = self.parent().image_manager.get_current_image_and_metadata()
                # Assume that meta already contains a proper 'original_header'
                # (or the entire meta is the header).
                original_header = meta.get("original_header", None)
            # If nothing is stored, fall back to creating a minimal header.
            if original_header is None:
                print("No stored FITS header found; creating a minimal header.")
                original_header = self.create_minimal_fits_header(normalized_image, is_mono)
        else:
            # For non-FITS images, generate a minimal header.
            original_header = self.create_minimal_fits_header(normalized_image, is_mono)
        
        # Create a temporary filename.
        tmp_file = tempfile.NamedTemporaryFile(suffix=".fits", delete=False)
        tmp_path = tmp_file.name
        tmp_file.close()
        
        try:
            # Call your global save_image() exactly as in AstroEditingSuite.
            save_image(
                img_array=normalized_image,
                filename=tmp_path,
                original_format=selected_format,
                bit_depth=bit_depth,
                original_header=original_header,
                is_mono=is_mono
                # (image_meta and file_meta can be omitted if not needed)
            )
            print(f"Temporary normalized FITS saved to: {tmp_path}")
        except Exception as e:
            print("Error saving temporary FITS file using save_image():", e)
            raise e
        return tmp_path

    def create_minimal_fits_header(self, img_array, is_mono=False):
        """
        Creates a minimal FITS header when the original header is missing.
        """

        header = Header()
        header['SIMPLE'] = (True, 'Standard FITS file')
        header['BITPIX'] = -32  # 32-bit floating-point data
        header['NAXIS'] = 2 if is_mono else 3
        header['NAXIS1'] = img_array.shape[2] if img_array.ndim == 3 and not is_mono else img_array.shape[1]  # Image width
        header['NAXIS2'] = img_array.shape[1] if img_array.ndim == 3 and not is_mono else img_array.shape[0]  # Image height
        if not is_mono:
            header['NAXIS3'] = img_array.shape[0] if img_array.ndim == 3 else 1  # Number of color channels
        header['BZERO'] = 0.0  # No offset
        header['BSCALE'] = 1.0  # No scaling
        header.add_comment("Minimal FITS header generated by AstroEditingSuite.")

        return header

    def stretch_image(self, image):
        """
        Perform an unlinked linear stretch on the image.
        Each channel is stretched independently by subtracting its own minimum,
        recording its own median, and applying the stretch formula.
        Returns the stretched image in [0,1].
        """
        was_single_channel = False  # Flag to check if image was single-channel

        # If the image is 2D or has one channel, convert to 3-channel
        if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
            was_single_channel = True
            image = np.stack([image] * 3, axis=-1)

        image = np.asarray(image, dtype=np.float32)
        stretched_image = image.copy()  # Need copy for in-place modification
        self.stretch_original_mins = []
        self.stretch_original_medians = []
        target_median = 0.02

        for c in range(3):
            channel_min = np.min(stretched_image[..., c])
            self.stretch_original_mins.append(channel_min)
            stretched_image[..., c] -= channel_min
            channel_median = np.median(stretched_image[..., c])
            self.stretch_original_medians.append(channel_median)
            if channel_median != 0:
                numerator = (channel_median - 1) * target_median * stretched_image[..., c]
                denominator = (
                    channel_median * (target_median + stretched_image[..., c] - 1)
                    - target_median * stretched_image[..., c]
                )
                denominator = np.where(denominator == 0, 1e-6, denominator)
                stretched_image[..., c] = numerator / denominator
            else:
                print(f"Channel {c} - Median is zero. Skipping stretch.")

        stretched_image = np.clip(stretched_image, 0.0, 1.0)
        self.was_single_channel = was_single_channel
        return stretched_image

    def unstretch_image(self, image):
        """
        Undo the unlinked linear stretch using stored parameters.
        Returns the unstretched image.
        """
        original_mins = self.stretch_original_mins
        original_medians = self.stretch_original_medians
        was_single_channel = self.was_single_channel

        image = np.asarray(image, dtype=np.float32)

        if image.ndim == 2:
            channel_median = np.median(image)
            original_median = original_medians[0]
            original_min = original_mins[0]
            if channel_median != 0 and original_median != 0:
                numerator = (channel_median - 1) * original_median * image
                denominator = channel_median * (original_median + image - 1) - original_median * image
                denominator = np.where(denominator == 0, 1e-6, denominator)
                image = numerator / denominator
            else:
                print("Channel median or original median is zero. Skipping unstretch.")
            image += original_min
            image = np.clip(image, 0, 1)
            return image

        for c in range(3):
            channel_median = np.median(image[..., c])
            original_median = original_medians[c]
            original_min = original_mins[c]
            if channel_median != 0 and original_median != 0:
                numerator = (channel_median - 1) * original_median * image[..., c]
                denominator = (
                    channel_median * (original_median + image[..., c] - 1)
                    - original_median * image[..., c]
                )
                denominator = np.where(denominator == 0, 1e-6, denominator)
                image[..., c] = numerator / denominator
            else:
                print(f"Channel {c} - Median or original median is zero. Skipping unstretch.")
            image[..., c] += original_min

        image = np.clip(image, 0, 1)
        if was_single_channel and image.ndim == 3:
            image = np.mean(image, axis=2, keepdims=True)
        return image

    def retrieve_and_apply_wcs(self, job_id):
        """Download the wcs.fits file from Astrometry.net, extract WCS header data, and apply it."""
        try:
            wcs_url = f"https://nova.astrometry.net/wcs_file/{job_id}"
            wcs_filepath = "wcs.fits"
            max_retries = 10
            delay = 10  # seconds
            
            for attempt in range(max_retries):
                response = requests.get(wcs_url, stream=True)
                response.raise_for_status()

                with open(wcs_filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                try:
                    with fits.open(wcs_filepath, ignore_missing_simple=True, ignore_missing_end=True) as hdul:
                        wcs_header = hdul[0].header
                        print("WCS header successfully retrieved.")
                        # 🔁 use common path
                        self.apply_wcs_header(wcs_header)
                        return wcs_header
                except Exception as e:
                    print(f"Attempt {attempt + 1}: Failed to process WCS file - possibly HTML instead of FITS. Retrying in {delay} seconds...")
                    print(f"Error: {e}")
                    time.sleep(delay)
            
            print("Failed to download a valid WCS FITS file after multiple attempts.")
            return None

        except requests.exceptions.RequestException as e:
            print(f"Error downloading WCS file: {e}")
        except Exception as e:
            print(f"Error processing WCS file: {e}")
            
        return None



    def apply_wcs_header(self, wcs_header):
        """
        Apply a solved WCS header.  Sets self.wcs, self.pixscale (arcsec/pix),
        self.orientation, and updates the orientation label.
        """
        # 1) Initialize the WCS object
        self.wcs = WCS(wcs_header, naxis=2, relax=True)

        # 2) Derive pixel scale (arcsec/pixel)
        if 'CDELT1' in wcs_header:
            # CDELT1 is degrees/pixel
            self.pixscale = abs(float(wcs_header['CDELT1'])) * 3600.0
        elif 'CD1_1' in wcs_header and 'CD2_2' in wcs_header:
            # approximate from CD matrix determinant
            det = (wcs_header['CD1_1'] * wcs_header['CD2_2']
                - wcs_header['CD1_2'] * wcs_header['CD2_1'])
            pixscale_deg = math.sqrt(abs(det))
            self.pixscale = pixscale_deg * 3600.0
        else:
            self.pixscale = None
            print("Warning: could not derive pixscale from header.")

        # 3) Extract orientation (CROTA2 if present)
        if 'CROTA2' in wcs_header:
            self.orientation = float(wcs_header['CROTA2'])
        else:
            # fallback to your custom function
            self.orientation = calculate_orientation(wcs_header)

        # 4) Update the GUI label
        if self.orientation is not None:
            self.orientation_label.setText(f"Orientation: {self.orientation:.2f}°")
        else:
            self.orientation_label.setText("Orientation: N/A")

        print(f" -> pixscale = {self.pixscale} arcsec/pixel")
        print(f" -> orientation = {self.orientation}°")
        try:
            cr1 = wcs_header.get('CRVAL1')
            cr2 = wcs_header.get('CRVAL2')
            if cr1 is not None and cr2 is not None:
                self.center_ra  = float(cr1)
                self.center_dec = float(cr2)
                print(f" -> center RA/Dec = {self.center_ra:.6f}, {self.center_dec:.6f}")
        except Exception:
            print("Warning: could not extract CRVAL1/CRVAL2")        



    def calculate_pixel_from_ra_dec(self, ra, dec):
        """Convert RA/Dec to pixel coordinates using the WCS data."""
        if not hasattr(self, "wcs") or self.wcs is None:
            print("WCS not initialized.")
            return None, None

        # Convert RA and Dec to pixel coordinates using the WCS object
        try:
            sky_coord = SkyCoord(ra, dec, unit=(u.deg, u.deg), frame="icrs")
            x, y = self.wcs.world_to_pixel(sky_coord)
        except Exception as e:
            print(f"world_to_pixel failed for RA={ra}, Dec={dec}: {e}")
            return None, None

        # world_to_pixel can return scalars or numpy arrays – normalize to scalars
        x_arr = np.asarray(x)
        y_arr = np.asarray(y)

        if x_arr.size == 0 or y_arr.size == 0:
            print(f"WCS returned empty pixel coords for RA={ra}, Dec={dec}")
            return None, None

        x_val = float(x_arr.ravel()[0])
        y_val = float(y_arr.ravel()[0])

        # Guard against NaNs / infinities
        if not np.isfinite(x_val) or not np.isfinite(y_val):
            print(f"WCS returned invalid pixel coords for RA={ra}, Dec={dec}: x={x_val}, y={y_val}")
            return None, None

        return int(round(x_val)), int(round(y_val))

    def login_to_astrometry(self, api_key):
        try:
            response = requests.post(
                ASTROMETRY_API_URL + "login",
                data={'request-json': json.dumps({"apikey": api_key})}
            )
            response_data = response.json()
            if response_data.get("status") == "success":
                return response_data["session"]
            else:
                raise ValueError("Login failed: " + response_data.get("error", "Unknown error"))
        except Exception as e:
            raise Exception("Login to Astrometry.net failed: " + str(e))


    def upload_image_to_astrometry(self, image_path, session_key):
        try:
            # Check if the file is XISF format
            file_extension = os.path.splitext(image_path)[-1].lower()
            if file_extension == ".xisf":
                # Load the XISF image
                xisf = XISF(image_path)
                im_data = xisf.read_image(0)
                
                # Convert to a temporary TIFF file for upload
                temp_image_path = os.path.splitext(image_path)[0] + "_converted.tif"
                if im_data.dtype == np.float32 or im_data.dtype == np.float64:
                    im_data = np.clip(im_data, 0, 1) * 65535
                im_data = im_data.astype(np.uint16)

                # Save as TIFF
                if im_data.shape[-1] == 1:  # Grayscale
                    tiff.imwrite(temp_image_path, np.squeeze(im_data, axis=-1))
                else:  # RGB
                    tiff.imwrite(temp_image_path, im_data)

                print(f"Converted XISF file to TIFF at {temp_image_path} for upload.")
                image_path = temp_image_path  # Use the converted file for upload

            # Upload the image file
            with open(image_path, 'rb') as image_file:
                files = {'file': image_file}
                data = {
                    'request-json': json.dumps({
                        "publicly_visible": "y",
                        "allow_modifications": "d",
                        "session": session_key,
                        "allow_commercial_use": "d"
                    })
                }
                response = requests.post(ASTROMETRY_API_URL + "upload", files=files, data=data)
                response_data = response.json()
                if response_data.get("status") == "success":
                    return response_data["subid"]
                else:
                    raise ValueError("Image upload failed: " + response_data.get("error", "Unknown error"))

        except Exception as e:
            raise Exception("Image upload to Astrometry.net failed: " + str(e))

        finally:
            # Clean up temporary file if created
            if file_extension == ".xisf" and os.path.exists(temp_image_path):
                os.remove(temp_image_path)
                print(f"Temporary TIFF file {temp_image_path} deleted after upload.")



    def poll_submission_status(self, subid):
        """Poll Astrometry.net to retrieve the job ID once the submission is processed."""
        max_retries = 90  # Adjust as necessary
        retries = 0
        while retries < max_retries:
            try:
                response = requests.get(ASTROMETRY_API_URL + f"submissions/{subid}")
                response_data = response.json()
                jobs = response_data.get("jobs", [])
                if jobs and jobs[0] is not None:
                    return jobs[0]
                else:
                    print(f"Polling attempt {retries + 1}: Job not ready yet.")
            except Exception as e:
                print(f"Error while polling submission status: {e}")
            
            retries += 1
            time.sleep(10)  # Wait 10 seconds between retries
        
        return None

    def poll_calibration_data(self, job_id):
        """Poll Astrometry.net to retrieve the calibration data once it's available."""
        max_retries = 90  # Retry for up to 15 minutes (90 * 10 seconds)
        retries = 0
        while retries < max_retries:
            try:
                response = requests.get(ASTROMETRY_API_URL + f"jobs/{job_id}/calibration/")
                response_data = response.json()
                if response_data and 'ra' in response_data and 'dec' in response_data:
                    print("Calibration data retrieved:", response_data)
                    return response_data  # Calibration data is complete
                else:
                    print(f"Calibration data not available yet (Attempt {retries + 1})")
            except Exception as e:
                print(f"Error retrieving calibration data: {e}")

            retries += 1
            time.sleep(10)  # Wait 10 seconds between retries

        return None


    #If originally a fits file update the header
    def update_fits_with_wcs(self, filepath, calibration_data):
        if not filepath.lower().endswith(('.fits', '.fit')):
            print("File is not a FITS file. Skipping WCS header update.")
            return

        print("Updating image with calibration data:", calibration_data)
        with fits.open(filepath, mode='update') as hdul:
            header = hdul[0].header
            header['CTYPE1'] = 'RA---TAN'
            header['CTYPE2'] = 'DEC--TAN'
            header['CRVAL1'] = calibration_data['ra']
            header['CRVAL2'] = calibration_data['dec']
            header['CRPIX1'] = hdul[0].data.shape[1] / 2
            header['CRPIX2'] = hdul[0].data.shape[0] / 2
            scale = calibration_data['pixscale'] / 3600
            orientation = np.radians(calibration_data['orientation'])
            header['CD1_1'] = -scale * np.cos(orientation)
            header['CD1_2'] = scale * np.sin(orientation)
            header['CD2_1'] = -scale * np.sin(orientation)
            header['CD2_2'] = -scale * np.cos(orientation)
            header['RADECSYS'] = 'ICRS'

    def on_mini_preview_press(self, event):
        # Set dragging flag and scroll the main preview to the position in the mini preview.
        self.dragging = True
        self.scroll_main_preview_to_mini_position(event)

    def on_mini_preview_drag(self, event):
        # Scroll to the new position while dragging in the mini preview.
        if self.dragging:
            self.scroll_main_preview_to_mini_position(event)

    def on_mini_preview_release(self, event):
        # Stop dragging
        self.dragging = False

    def scroll_main_preview_to_mini_position(self, event):
        """Scrolls the main preview to the corresponding position based on the mini preview click."""
        if self.main_image:
            # Get the click position in the mini preview
            click_x = event.pos().x()
            click_y = event.pos().y()
            
            # Calculate scale factors based on the difference in dimensions between main image and mini preview
            scale_factor_x = self.main_scene.sceneRect().width() / self.mini_preview.width()
            scale_factor_y = self.main_scene.sceneRect().height() / self.mini_preview.height()
            
            # Scale the click position to the main preview coordinates
            scaled_x = click_x * scale_factor_x
            scaled_y = click_y * scale_factor_y
            
            # Center the main preview on the calculated position
            self.main_preview.centerOn(scaled_x, scaled_y)
            
            # Update the green box after scrolling
            self.main_preview.update_mini_preview()

    def update_green_box(self):
        if self.main_image:
            factor_x = self.mini_preview.width() / self.main_image.width()
            factor_y = self.mini_preview.height() / self.main_image.height()
            
            # Get the current view rectangle in the main preview (in scene coordinates)
            view_rect = self.main_preview.mapToScene(self.main_preview.viewport().rect()).boundingRect()
            
            # Calculate the green box rectangle, shifted upward by half its height to center it
            green_box_rect = QRectF(
                view_rect.x() * factor_x,
                view_rect.y() * factor_y,
                view_rect.width() * factor_x,
                view_rect.height() * factor_y
            )
            
            # Scale the main image for the mini preview and draw the green box on it
            pixmap = self.main_image.scaled(self.mini_preview.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            painter = QPainter(pixmap)
            pen = QPen(QColor(0, 255, 0), 2)
            painter.setPen(pen)
            painter.drawRect(green_box_rect)
            painter.end()
            self.mini_preview.setPixmap(pixmap)

    @staticmethod
    def calculate_angular_distance(ra1, dec1, ra2, dec2):
        # Convert degrees to radians
        ra1, dec1, ra2, dec2 = map(math.radians, [ra1, dec1, ra2, dec2])

        # Haversine formula for angular distance
        delta_ra = ra2 - ra1
        delta_dec = dec2 - dec1
        a = (math.sin(delta_dec / 2) ** 2 +
            math.cos(dec1) * math.cos(dec2) * math.sin(delta_ra / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        angular_distance = math.degrees(c)
        return angular_distance
    
    @staticmethod
    def format_distance_as_dms(angle):
        degrees = int(angle)
        minutes = int((angle - degrees) * 60)
        seconds = (angle - degrees - minutes / 60) * 3600
        return f"{degrees}° {minutes}' {seconds:.2f}\""


    def wheel_zoom(self, event):
        if event.angleDelta().y() > 0:
            self.zoom_in()
        else:
            self.zoom_out()


    def zoom_in(self):
        self.zoom_level *= 1.2
        self.main_preview.setTransform(QTransform().scale(self.zoom_level, self.zoom_level))
        self.update_green_box()
        

    def zoom_out(self):
        self.zoom_level /= 1.2
        self.main_preview.setTransform(QTransform().scale(self.zoom_level, self.zoom_level))
        self.update_green_box()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_green_box()


    def compute_pixscale(self):
        """
        Computes the pixel scale (arcsec/pixel) from WCS if possible,
        otherwise falls back to header CD keywords.
        """
        from astropy.wcs.utils import proj_plane_pixel_scales

        # Prefer WCS-derived scale
        if getattr(self, "wcs", None) is not None:
            try:
                scales = proj_plane_pixel_scales(self.wcs) * 3600.0  # arcsec/pixel
                pixscale = float(np.mean(scales))
                print("Calculated pixscale from WCS:", pixscale)
                return pixscale
            except Exception as e:
                print("Error calculating pixscale from WCS:", e)

        # Fallback: CD matrix
        try:
            cd1_1 = float(self.header.get('CD1_1', 0))
            cd1_2 = float(self.header.get('CD1_2', 0))
            pixscale = math.sqrt(cd1_1**2 + cd1_2**2) * 3600.0
            print("Calculated pixscale from header:", pixscale)
            return pixscale
        except Exception as e:
            print("Error calculating pixscale from header:", e)
            return None


    def update_circle_data(self):
        """Updates the status based on the circle's center and radius."""
        if self.circle_center and self.circle_radius > 0:
            # Make sure we have a valid pixscale.
            if self.pixscale is None:
                self.pixscale = self.compute_pixscale()
                if self.pixscale is None:
                    self.status_label.setText("No pixscale available for radius calculation.")
                    print("Warning: Pixscale is None. Cannot calculate radius in arcminutes.")
                    return

            # Convert circle center to RA/Dec and radius to arcminutes.
            ra, dec = self.calculate_ra_dec_from_pixel(self.circle_center.x(), self.circle_center.y())
            radius_arcmin = self.circle_radius * self.pixscale / 60.0  # from arcsec to arcmin
            self.status_label.setText(
                f"Circle set at center RA={ra:.6f}, Dec={dec:.6f}, radius={radius_arcmin:.2f} arcmin"
            )
        else:
            self.status_label.setText("No search area defined.")



    def get_defined_radius(self):
        """Calculate radius in degrees for the defined region (circle radius)."""
        if self.circle_radius <= 0:
            return 0
        return float((self.circle_radius * self.pixscale) / 3600.0)


    def query_simbad(self, radius_deg, max_results=None):
        """Two-step SIMBAD lookup with debug prints for flux/plx/sp data."""
        max_results = max_results if max_results is not None else self.max_results

        # ——— 1) Validate inputs ———
        if not self.circle_center or self.circle_radius <= 0:
            QMessageBox.warning(self, "No Search Area",
                                "Please define a search circle by Shift-clicking and dragging.")
            return

        ra_center, dec_center = self.calculate_ra_dec_from_pixel(
            self.circle_center.x(), self.circle_center.y()
        )
        if ra_center is None or dec_center is None:
            QMessageBox.warning(self, "Invalid Coordinates",
                                "Could not determine the RA/Dec of the circle center.")
            return

        selected_types = self.get_selected_object_types()
        if not selected_types:
            QMessageBox.warning(self, "No Object Types Selected",
                                "Please select at least one object type.")
            return

        # ——— 2) TAP query on BASIC ———
        query = f"""
            SELECT TOP {max_results}
                ra, dec, main_id,
                rvz_redshift, otype, galdim_majaxis
            FROM basic
            WHERE CONTAINS(
                POINT('ICRS', basic.ra, basic.dec),
                CIRCLE('ICRS', {ra_center}, {dec_center}, {radius_deg})
            ) = 1
        """
        for attempt in range(5):
            try:
                result = Simbad.query_tap(query)
                break
            except Exception as e:
                if attempt < 4:
                    time.sleep(1)
                else:
                    QMessageBox.critical(
                        self,
                        "Query Failed",
                        f"Try again later:\n{e}"
            )

        if result is None or len(result) == 0:
            QMessageBox.information(self, "No Results",
                                    "No objects found in the specified area.")
            return

        # ——— 3a) list of all “star” & binary/variable OTYPE codes ———
        star_codes = [
            "*","V*","Pe*","HB*","Y*O","Ae*","Em*","Be*","BS*","RG*","AB*",
            "C*","S*","sg*","s*r","s*y","s*b","HS*","pA*","WD*","LM*","BD*",
            "N*","OH*","TT*","WR*","PM*","HV*","C?*","Pec?","Y*?","TT?","C*?",
            "S*?","OH?","WR?","Be?","Ae?","HB?","RB?","sg?","s?r","s?y","s?b",
            "pA?","BS?","HS?","WD?",
            "**","EB*","Ce*","Ce?","cC*","**?",
            "EB?","Sy?","CV?","No?","XB?","LX?","HX?","RR?","WV?","LP?","Mi?"
        ]

        # 3b) build the two sub-criteria, un-encoded:
        ra_str  = f"{ra_center:.8f}"
        dec_str = f"{dec_center:+.8f}"   # keep the +/– sign
        rad_str = f"{radius_deg:.8f}d"

        region_crit = f"region(CIRCLE,{ra_str} {dec_str},{rad_str})"
        codes_list  = ",".join(f"'{c}'" for c in star_codes)
        otype_crit  = f"otypes in ({codes_list})"

        # combine with a literal '&' (not "AND")
        criteria = f"{region_crit}&{otype_crit}"

        # ——— 3c) fetch _only_ those via sim-sam ———
        sam_url = "https://simbad.cds.unistra.fr/simbad/sim-sam"
        params = {
            "Criteria":      criteria,
            "OutputMode":    "LIST",
            "maxObject":     str(max_results),
            "output.format": "votable",
            "output.params": ",".join([
                "MAIN_ID","RA","DEC",
                "FLUX(B)","FLUX(V)",
                "PLX_VALUE","RVZ_REDSHIFT",
                "OTYPE","SP_TYPE"
            ])
        }

        try:
            resp = requests.get(sam_url, params=params, timeout=300)
            resp.raise_for_status()

            vot = parse_single_table(BytesIO(resp.content))
            tbl = vot.to_table(use_names_over_ids=True)
            extras = { row["MAIN_ID"]: row for row in tbl }

        except Exception as e:
            print(f"DEBUG: sim-sam failed: {e}")
            QMessageBox.warning(
                self,
                "Star-only Extras Failed",
                "Could not fetch star flux/parallax—continuing without them."
            )
            extras = {}

        # ——— 4) Merge & populate results_tree exactly as before ———
        self.results_tree.clear()
        query_results = []

        for row in result:
            name       = row["main_id"]
            short_type = row["otype"]
            if short_type not in selected_types:
                continue

            # basics
            ra, dec = float(row["ra"]), float(row["dec"])
            diam     = row.get("galdim_majaxis", "N/A")

            rz_raw   = row["rvz_redshift"]
            red_z    = _mask_safe_float(rz_raw)  # ← no warning, None if masked

            # pull extras only if it’s a star
            extra = extras.get(name, {})

            Bmag  = _mask_safe_float(extra.get("FLUX_B"))
            Vmag  = _mask_safe_float(extra.get("FLUX_V"))
            plx   = _mask_safe_float(extra.get("PLX_VALUE"))
            spec  = extra.get("SP_TYPE")

            if plx is not None and plx > 0.0:
                pv = plx  # already positive; absolute if you want
                dist_pc  = 1000.0 / pv
                dist_ly  = dist_pc * 3.261563777
                # store comoving distance in Gyr for consistency with your zs_raw pipeline:
                distance = round(dist_ly / 1e9, 9)
                red_val  = pv  # mas, goes in the "redshift" column but clearly not a z
            else:
                # no usable parallax → fall back to redshift if present
                red_val  = red_z if red_z is not None else "--"
                distance = (calculate_comoving_distance(red_val)
                            if isinstance(red_val, (int, float)) else "N/A")

            # absolute V magnitude if we have Vmag & plx
            absV = None
            if Vmag is not None and plx is not None and plx > 0:
                absV = Vmag - (5 * np.log10(1000.0 / plx) - 5)

            long_type = otype_long_name_lookup.get(short_type, short_type)

            # add to tree
            item = QTreeWidgetItem([
                f"{ra:.6f}", f"{dec:.6f}", name,
                str(diam), short_type, long_type,
                f"{red_val:.6f}" if isinstance(red_val, (int, float)) else str(red_val),
                f"{distance:.6f}" if isinstance(distance, float) else str(distance)
            ])
            self.results_tree.addTopLevelItem(item)

            query_results.append({
                'ra': ra, 'dec': dec, 'name': name,
                'diameter': diam,
                'short_type': short_type,
                'long_type': long_type,
                'redshift': red_val,
                'comoving_distance': distance,
                'source': "Simbad",
                'Bmag': Bmag, 'Vmag': Vmag,
                'parallax_mas': plx,
                'spectral_type': spec,
                'absolute_mag': absV
            })

        # ——— 5) Finally hand off to your preview/plotter ———
        self.main_preview.set_query_results(query_results)
        self.query_results = query_results
        self.update_object_count()




    def perform_deep_vizier_search(self):
        CATALOG_NAMES = {
            "J/ApJS/199/26":    "2MRS",
            "VII/259/6dfgs":    "6dF Galaxy Survey",
            "V/147/sdss12":     "SDSS DR12",
            "VII/250/2dfgrs":   "2dFGRS",
            "J/MNRAS/474/3875": "GAMA DR3",
            "VII/291/gladep":   "GLADE+",
            "VII/237":          "HyperLEDA",
            "VII/221/psc":      "IRAS PSCz",
            "II/246":           "2MASS PSC",
            "I/350/gaiaedr3":   "Gaia EDR3",
            "I/322A":           "UCAC4",
            "V/154":            "Pan-STARRS 1",
        }        
        """Perform a Vizier catalog search and parse results, querying redshift surveys first."""
        if not self.circle_center or self.circle_radius <= 0:
            QMessageBox.warning(self, "No Search Area",
                                "Please define a search circle by Shift-clicking and dragging.")
            return

        # Convert center to RA/Dec
        ra_center, dec_center = self.calculate_ra_dec_from_pixel(
            self.circle_center.x(), self.circle_center.y()
        )
        if ra_center is None or dec_center is None:
            QMessageBox.warning(self, "Invalid Coordinates",
                                "Could not determine the RA/Dec of the circle center.")
            return

        radius_arcmin = float((self.circle_radius * self.pixscale) / 60.0)

        # Query true-redshift surveys first
        catalog_ids = [
            # 1) Major spectroscopic redshift surveys
            "J/ApJS/199/26",     # 2MASS Redshift Survey (2MRS)
            "VII/259/6dfgs",     # 6dF Galaxy Survey
            "V/147/sdss12",      # SDSS DR12 spectroscopic
            "VII/250/2dfgrs",    # 2dF Galaxy Redshift Survey (2dFGRS)
            "J/MNRAS/474/3875",  # GAMA DR3

            # 2) Meta-catalogs & composites
            "VII/291/gladep",    # GLADE+
            "VII/237",           # HyperLEDA
            "VII/221/psc",       # IRAS PSCz

            # 3) Photometric & astrometric surveys
            "II/246",            # 2MASS PSC
            "I/350/gaiaedr3",    # Gaia EDR3
            "I/322A",            # UCAC4
            "V/154"              # Pan-STARRS 1
        ]


        coord = SkyCoord(ra_center, dec_center, unit="deg")
        unique_entries = {}

        try:
            for catalog_id in catalog_ids:
                result = Vizier.query_region(
                    coord, radius=radius_arcmin * u.arcmin, catalog=catalog_id
                )
                if not result:
                    continue

                for row in result[0]:
                    # RA / Dec
                    ra = row.get("RAJ2000", row.get("RA_ICRS", None))
                    dec = row.get("DEJ2000", row.get("DE_ICRS", None))
                    if ra is None or dec is None:
                        continue
                    ra_str, dec_str = str(ra), str(dec)
                    key = (ra_str, dec_str)

                    # Name & types
                    name       = str(row.get("_2MASS", "") or row.get("Source", "") or row.get("SDSS12", ""))
                    type_short = CATALOG_NAMES.get(catalog_id, catalog_id)
                    long_type  = str(row.get("SpType", "N/A"))
                    diameter   = catalog_id

                    # ——— robust redshift/parallax parsing ———
                    if "cz" in row.colnames:
                        raw = row["cz"]
                        try:
                            cz = float(raw)
                            zval = cz / 299792.458
                            redshift = f"{zval:.6f}"
                            comoving_distance = f"{calculate_comoving_distance(zval):.5f} GLy"
                        except Exception:
                            redshift = str(raw)
                            comoving_distance = "N/A"

                    elif "z" in row.colnames:
                        raw = row["z"]
                        try:
                            zval = float(raw)
                            if np.isnan(zval):
                                raise ValueError
                            redshift = f"{zval:.6f}"
                            comoving_distance = f"{calculate_comoving_distance(zval):.5f} GLy"
                        except Exception:
                            redshift = str(raw)
                            comoving_distance = "N/A"

                    elif "zhelio" in row.colnames:
                        raw = row["zhelio"]
                        try:
                            zval = float(raw)
                            if np.isnan(zval):
                                raise ValueError
                            redshift = f"{zval:.6f}"
                            comoving_distance = f"{calculate_comoving_distance(zval):.5f} GLy"
                        except Exception:
                            redshift = str(raw)
                            comoving_distance = "N/A"

                    elif "zcmb" in row.colnames:
                        raw = row["zcmb"]
                        try:
                            zval = float(raw)
                            if np.isnan(zval):
                                raise ValueError
                            redshift = f"{zval:.6f}"
                            comoving_distance = f"{calculate_comoving_distance(zval):.5f} GLy"
                        except Exception:
                            redshift = str(raw)
                            comoving_distance = "N/A"

                    elif "zph" in row.colnames:
                        raw = row["zph"]
                        try:
                            zval = float(raw)
                            if np.isnan(zval):
                                raise ValueError
                            redshift = f"{zval:.6f}"
                            comoving_distance = f"{calculate_comoving_distance(zval):.5f} GLy"
                        except Exception:
                            redshift = str(raw)
                            comoving_distance = "N/A"

                    elif "Plx" in row.colnames:
                        raw = row["Plx"]
                        try:
                            pv = abs(float(raw))
                            redshift = f"{pv:.3f} (Parallax mas)"
                            comoving_distance = f"{1000/pv * 3.2615637769:.5f} Ly"
                        except Exception:
                            redshift = str(raw)
                            comoving_distance = "N/A"

                    else:
                        redshift = "N/A"
                        comoving_distance = "N/A"
                    # ——— end parsing block ———

                    # Duplicate handling: first entry wins, SDSS overrides Pan-STARRS
                    if key not in unique_entries:
                        unique_entries[key] = {
                            "ra": ra_str,
                            "dec": dec_str,
                            "name": name,
                            "diameter": diameter,
                            "short_type": type_short,
                            "long_type": long_type,
                            "redshift": redshift,
                            "comoving_distance": comoving_distance,
                            "source": "Vizier"
                        }
                    else:
                        existing = unique_entries[key]["diameter"]
                        if existing == "V/154" and diameter == "V/147/sdss12":
                            unique_entries[key].update({
                                "name": name,
                                "diameter": diameter,
                                "short_type": type_short,
                                "long_type": long_type,
                                "redshift": redshift,
                                "comoving_distance": comoving_distance,
                                "source": "Vizier"
                            })

            # Populate tree & preview
            all_results = []
            for e in unique_entries.values():
                item = QTreeWidgetItem([
                    e["ra"], e["dec"], e["name"], e["diameter"],
                    e["short_type"], e["long_type"],
                    e["redshift"], e["comoving_distance"]
                ])
                self.results_tree.addTopLevelItem(item)
                all_results.append(e)

            self.main_preview.set_query_results(all_results)
            self.query_results = all_results
            self.update_object_count()

        except Exception as err:
            QMessageBox.critical(self, "Vizier Search Failed", f"Failed to query Vizier: {err}")



    def perform_mast_search(self):
        """Perform a MAST cone search in the user-defined region using astroquery."""
        if not self.circle_center or self.circle_radius <= 0:
            QMessageBox.warning(self, "No Search Area", "Please define a search circle by Shift-clicking and dragging.")
            return

        # Calculate RA and Dec for the center point
        ra_center, dec_center = self.calculate_ra_dec_from_pixel(self.circle_center.x(), self.circle_center.y())
        if ra_center is None or dec_center is None:
            QMessageBox.warning(self, "Invalid Coordinates", "Could not determine the RA/Dec of the circle center.")
            return

        # Convert radius from arcseconds to degrees (MAST uses degrees)
        search_radius_deg = float((self.circle_radius * self.pixscale) / 3600.0)  # Convert to degrees
        ra_center = float(ra_center)  # Ensure it's a regular float
        dec_center = float(dec_center)  # Ensure it's a regular float

        try:
            # Perform the MAST cone search using Mast.mast_query for the 'Mast.Caom.Cone' service
            observations = Mast.mast_query(
                'Mast.Caom.Cone',
                ra=ra_center,
                dec=dec_center,
                radius=search_radius_deg
            )

            # Limit the results to the first 100 rows
            limited_observations = observations[:100]

            if len(observations) == 0:
                QMessageBox.information(self, "No Results", "No objects found in the specified area on MAST.")
                return

            # Clear previous results
            self.results_tree.clear()
            query_results = []

            # Process each observation in the results
            for obj in limited_observations:

                def safe_get(value):
                    return "N/A" if np.ma.is_masked(value) else str(value)


                ra = safe_get(obj.get("s_ra", "N/A"))
                dec = safe_get(obj.get("s_dec", "N/A"))
                target_name = safe_get(obj.get("target_name", "N/A"))
                instrument = safe_get(obj.get("instrument_name", "N/A"))
                jpeg_url = safe_get(obj.get("dataURL", "N/A"))  # Adjust URL field as needed

                # Add to TreeWidget
                item = QTreeWidgetItem([
                    ra,
                    dec,
                    target_name,
                    instrument,
                    "N/A",  # Placeholder for observation date if needed
                    "N/A",  # Other placeholder
                    jpeg_url,  # URL in place of long type
                    "MAST"  # Source
                ])
                self.results_tree.addTopLevelItem(item)

                # Append full details as a dictionary to query_results
                query_results.append({
                    'ra': ra,
                    'dec': dec,
                    'name': target_name,
                    'diameter': instrument,
                    'short_type': "N/A",
                    'long_type': jpeg_url,
                    'redshift': "N/A",
                    'comoving_distance': "N/A",
                    'source': "Mast"
                })

            # Set query results in the CustomGraphicsView for display
            self.main_preview.set_query_results(query_results)
            self.query_results = query_results  # Keep a reference to results in MainWindow
            self.update_object_count()

        except Exception as e:
            QMessageBox.critical(self, "MAST Query Failed", f"Failed to query MAST: {str(e)}")

    def toggle_show_names(self, state):
        """Toggle showing/hiding names on the main image."""
        self.show_names = state == Qt.CheckState.Checked
        self.main_preview.draw_query_results()  # Redraw with or without names

    def clear_results(self):
        """Clear the search results and remove markers from the main image."""
        self.results_tree.clear()
        self.main_preview.clear_query_results()
        self.status_label.setText("Results cleared.")

    def open_settings_dialog(self):
        """Open settings dialog to adjust max results and marker type."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Settings")
        
        layout = QFormLayout(dialog)
        

        # Max Results setting using CustomSpinBox
        max_results_spinbox = CustomSpinBox(minimum=1, maximum=100000, initial=self.max_results, step=1)
        layout.addRow("Max Results:", max_results_spinbox)

        
        # Marker Style selection
        marker_style_combo = QComboBox()
        marker_style_combo.addItems(["Circle", "Crosshair"])
        marker_style_combo.setCurrentText(self.marker_style)
        layout.addRow("Marker Style:", marker_style_combo)

        # Force Blind Solve button
        force_blind_solve_button = QPushButton("Force Blind Solve")
        force_blind_solve_button.clicked.connect(self.force_blind_solve)
        layout.addWidget(force_blind_solve_button)
        
        # OK and Cancel buttons
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(lambda: self.update_settings(max_results_spinbox.value, marker_style_combo.currentText(), dialog))
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        
        dialog.setLayout(layout)
        dialog.exec()

    def update_settings(self, max_results, marker_style, dialog):
        """Update settings based on dialog input."""
        self.max_results = max_results
        self.marker_style = marker_style  # Store the selected marker style
        self.main_preview.draw_query_results()
        dialog.accept()

    def force_blind_solve(self):
        doc = self._doc_for_solver()
        if doc is None:
            QMessageBox.information(self, "Force Blind Solve", "No image is loaded.")
            return

        mw = _find_main_window(self) or self.parent()
        settings = getattr(mw, "settings", QSettings())

        prev_mode = _get_seed_mode(settings)
        try:
            _set_seed_mode(settings, "none")  # blind for this one call
            ok, res = plate_solve_doc_inplace(mw, doc, settings)
        finally:
            _set_seed_mode(settings, prev_mode)  # restore user preference

        if ok:
            # Pull the header the solver just wrote and reflect it into WIMI
            meta = getattr(doc, "metadata", {}) or {}
            hdr = meta.get("wcs_header") or meta.get("original_header")
            try:
                if hdr is not None:
                    self.initialize_wcs_from_header(hdr)
                    self.status_label.setText("Status: Blind solve succeeded.")
                    QMessageBox.information(self, "Blind Solve", "Astrometric solution applied successfully.")
                else:
                    raise RuntimeError("Solver returned no header.")
            except Exception as e:
                QMessageBox.warning(self, "Apply WCS", f"Solve succeeded but WCS init failed: {e}")
                self.status_label.setText(f"Status: WCS init failed — {e}")
        else:
            QMessageBox.critical(self, "Blind Solve Failed", str(res))
            self.status_label.setText(f"Status: Blind solve failed — {res}")


def extract_wcs_data(file_path):
    try:
        # Open the FITS file with minimal validation to ignore potential errors in non-essential parts
        with fits.open(file_path, ignore_missing_simple=True, ignore_missing_end=True) as hdul:
            header = hdul[0].header

            # Extract essential WCS parameters
            wcs_params = {}
            keys_to_extract = [
                'WCSAXES', 'CTYPE1', 'CTYPE2', 'EQUINOX', 'LONPOLE', 'LATPOLE',
                'CRVAL1', 'CRVAL2', 'CRPIX1', 'CRPIX2', 'CUNIT1', 'CUNIT2',
                'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2', 'A_ORDER', 'A_0_0', 'A_0_1', 
                'A_0_2', 'A_1_0', 'A_1_1', 'A_2_0', 'B_ORDER', 'B_0_0', 'B_0_1', 
                'B_0_2', 'B_1_0', 'B_1_1', 'B_2_0', 'AP_ORDER', 'AP_0_0', 'AP_0_1', 
                'AP_0_2', 'AP_1_0', 'AP_1_1', 'AP_2_0', 'BP_ORDER', 'BP_0_0', 
                'BP_0_1', 'BP_0_2', 'BP_1_0', 'BP_1_1', 'BP_2_0'
            ]
            for key in keys_to_extract:
                if key in header:
                    wcs_params[key] = header[key]

            # Manually create a minimal header with WCS information
            wcs_header = fits.Header()
            for key, value in wcs_params.items():
                wcs_header[key] = value

            # Initialize WCS with this custom header
            wcs = WCS(wcs_header)
            print("WCS successfully initialized with minimal header.")
            return wcs

    except Exception as e:
        print(f"Error processing WCS file: {e}")
        return None

# Function to calculate comoving radial distance (in Gly)
def calculate_comoving_distance(z):
    z = abs(z)
    # Initialize variables
    WR = 4.165E-5 / ((H0 / 100) ** 2)  # Omega radiation
    WK = 1 - WM - WV - WR  # Omega curvature
    az = 1.0 / (1 + z)
    n = 1000  # number of points in integration

    # Comoving radial distance
    DCMR = 0.0
    for i in range(n):
        a = az + (1 - az) * (i + 0.5) / n
        adot = sqrt(WK + (WM / a) + (WR / (a ** 2)) + (WV * a ** 2))
        DCMR += 1 / (a * adot)
    
    DCMR = (1 - az) * DCMR / n
    DCMR_Gly = (c / H0) * DCMR * Mpc_to_Gly

    return round(DCMR_Gly, 6)  # Round to three decimal places for display

def calculate_orientation(header):
    """Calculate orientation from CD or PC matrix."""
    cd1_1 = header.get('CD1_1')
    cd1_2 = header.get('CD1_2')
    cd2_1 = header.get('CD2_1')
    cd2_2 = header.get('CD2_2')

    if all(v is not None for v in [cd1_1, cd1_2, cd2_1, cd2_2]):
        orientation = (np.degrees(np.arctan2(cd1_2, cd1_1)) + 180) % 360
        return orientation

    # Try PC matrix fallback
    pc1_1 = header.get('PC1_1')
    pc1_2 = header.get('PC1_2')
    cdelt1 = header.get('CDELT1')
    cdelt2 = header.get('CDELT2')

    if pc1_1 is not None and pc1_2 is not None and cdelt1 is not None and cdelt2 is not None:
        cd1_1 = pc1_1 * cdelt1
        cd1_2 = pc1_2 * cdelt1
        orientation = (np.degrees(np.arctan2(cd1_2, cd1_1)) + 180) % 360
        return orientation

    print("CD or PC matrix not found in header.")
    return None

