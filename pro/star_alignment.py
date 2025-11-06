#pro.star_alignment.py
from __future__ import annotations

import os, math, random, sys
import os as _os, threading as _threading, ctypes as _ctypes
import multiprocessing
N = str(max(1, min( (os.cpu_count() or 8), 32 )))
os.environ.setdefault("OMP_NUM_THREADS", N)
os.environ.setdefault("OPENBLAS_NUM_THREADS", N)
os.environ.setdefault("MKL_NUM_THREADS", N)
os.environ.setdefault("NUMEXPR_NUM_THREADS", N)
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", N)  # macOS Accelerate
try:
    import cv2
    cv2.setNumThreads(int(N))   # let OpenCV parallelize internally
except Exception:
    pass

from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
from typing import Callable, Iterable, Tuple
import tempfile
import traceback
import requests
import numpy as np

import astroalign
import sep
import re, warnings
import json
import time
from scipy.spatial import KDTree, Delaunay
from astropy.stats import sigma_clipped_stats
from astropy.io.fits import Header
from photutils.detection import DAOStarFinder
from astropy.table import vstack
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, get_body, get_sun
from astropy.wcs import FITSFixedWarning
import astropy.units as u
from astropy.wcs import WCS
from astropy.wcs.utils import skycoord_to_pixel
from reproject import reproject_interp
from pro.blink_comparator_pro import CustomDoubleSpinBox, CustomSpinBox
import numpy.ma as ma
from PyQt6.QtCore import Qt, QThread, QRunnable, QThreadPool, pyqtSignal, QObject, QTimer, QProcess, QPoint, QEvent, QSettings
from PyQt6.QtGui import QImage, QPixmap, QIcon, QMovie
from PyQt6.QtWidgets import (
    QDialog, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QGroupBox, QAbstractItemView, QListWidget, QInputDialog, QApplication, QProgressBar, QProgressDialog, 
    QRadioButton, QFileDialog, QComboBox, QMessageBox, QTextEdit, QDialogButtonBox, QTreeWidget,QCheckBox, QFormLayout, QListWidgetItem, QScrollArea, QTreeWidgetItem, QSpinBox, QDoubleSpinBox
)

from skimage.transform import warp, PolynomialTransform 

# I/O & stretch (same stack we used for Plate Solver)
from legacy.image_manager import load_image, save_image
try:
    from imageops.stretch import stretch_mono_image, stretch_color_image
except Exception:
    stretch_mono_image = None
    stretch_color_image = None

# Optional fast star detector; fall back gracefully if not present
try:
    from legacy.numba_utils import fast_star_detect  # your optimized detector, if available
except Exception:
    fast_star_detect = None

from legacy.numba_utils import *
from pro.abe import _generate_sample_points as abe_generate_sample_points

# ---------------------------------------------------------------------
# Small helpers to work with the *active view/document* (no slots)
# ---------------------------------------------------------------------

def _apply_affine_to_pts(A_2x3: np.ndarray, pts_xy: np.ndarray) -> np.ndarray:
    ones = np.ones((pts_xy.shape[0], 1), dtype=np.float32)
    P = np.hstack([pts_xy.astype(np.float32), ones])
    return (A_2x3.astype(np.float32) @ P.T).T  # (N,2)


def _align_prefs(settings: QSettings | None = None) -> dict:
    """
    Read alignment prefs with sane defaults, supporting:
      • primary keys:  stacking/align/*
      • legacy keys:   align/*          (back-compat)
    Also migrates 'tps' → 'poly3'.
    """
    if settings is None:
        settings = QSettings()

    def _get(name: str, default, cast):
        # Prefer new path, fall back to legacy
        val = settings.value(f"stacking/align/{name}", None)
        if val is None:
            val = settings.value(f"align/{name}", None)
        if val is None:
            return default
        try:
            if cast is bool:
                s = str(val).strip().lower()
                return s in ("1", "true", "yes", "on")
            return cast(val)
        except Exception:
            return default

    # Model with back-compat for 'tps'
    model = (_get("model", "affine", str) or "affine").lower()
    if model == "tps":
        model = "poly3"
        settings.setValue("stacking/align/model", model)  # migrate to new key

    prefs = {
        "model":       model,                 # "affine" | "homography" | "poly3" | "poly4"
        "max_cp":      _get("max_cp", 250, int),
        "downsample":  _get("downsample", 2, int),
        "h_reproj":    _get("h_reproj", 3.0, float),

        # New knobs (also read legacy align/* if present)
        "det_sigma":   _get("det_sigma", 10.0, float),   # try 8–12 in dense fields
        "limit_stars": _get("limit_stars", 500, int),    # 500–1500 typical
        "minarea":     _get("minarea", 10, int),
    }

    return prefs

# ---------- Shortcut / Headless integration ----------

STAR_ALIGN_CID = "star_alignment"

# Put this near the top (after imports is fine) — called once per run.
_NATIVE_THREAD_CAP_DONE = False
_AA_LOCK = _threading.Lock()
_CAP_DONE = False

def _cap_native_threads_once():
    global _CAP_DONE
    if _CAP_DONE:
        return
    # Env must be set before libs spin up their pools
    _os.environ.setdefault("OMP_NUM_THREADS", "1")
    _os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    _os.environ.setdefault("MKL_NUM_THREADS", "1")
    _os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    _os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    _os.environ.setdefault("OPENCV_OPENMP_DISABLE", "1")
    try:
        import numpy as _np
        # numpy does not expose thread cap directly; env is enough for BLAS backends
    except Exception:
        pass
    try:
        import cv2 as _cv2
        _cv2.setNumThreads(1)
    except Exception:
        pass
    _CAP_DONE = True

def _find_main_window_from_child(w):
    p = w
    while p is not None and not (hasattr(p, "doc_manager") or hasattr(p, "docman")):
        p = getattr(p, "parent", lambda: None)()
    return p

def _resolve_doc_and_sw_by_ptr(mw, doc_ptr: int):
    # Prefer helper if app exposes one
    if hasattr(mw, "_find_doc_by_id"):
        try:
            d, sw = mw._find_doc_by_id(int(doc_ptr))
            if d is not None:
                return d, sw
        except Exception:
            pass
    # Fallback: scan MDI
    try:
        for sw in mw.mdi.subWindowList():
            vw = sw.widget()
            d = getattr(vw, "document", None)
            if d is not None and id(d) == int(doc_ptr):
                return d, sw
    except Exception:
        pass
    return None, None

def _doc_from_sw(sw):
    try:
        return getattr(sw.widget(), "document", None)
    except Exception:
        return None

def _gray2d(a):
    return np.mean(a, axis=2) if a.ndim == 3 else a


def aa_find_transform_with_backoff(tgt_gray: np.ndarray, src_gray: np.ndarray):
    """
    Retry astroalign.find_transform() with progressively stricter detection,
    serializing SEP usage via _AA_LOCK; returns (transform_obj, (src_pts, tgt_pts)).
    """
    tgt32 = np.ascontiguousarray(tgt_gray.astype(np.float32))
    src32 = np.ascontiguousarray(src_gray.astype(np.float32))
    try:
        curr = sep.get_extract_pixstack()
        if curr < 1_500_000:
            sep.set_extract_pixstack(1_500_000)
    except Exception:
        pass

    tries = [
        dict(detection_sigma=15,  min_area=7,  max_control_points=75),
        dict(detection_sigma=25, min_area=9,  max_control_points=75),
        dict(detection_sigma=50, min_area=9,  max_control_points=75),
        dict(detection_sigma=80, min_area=11, max_control_points=75),
        dict(detection_sigma=120, min_area=11, max_control_points=75),
    ]
    last_exc = None
    for kw in tries:
        try:
            global _AA_LOCK
            with _AA_LOCK:
                return astroalign.find_transform(tgt32, src32, **kw)
        except Exception as e:
            last_exc = e
            if "internal pixel buffer full" in str(e).lower():
                try:
                    sep.set_extract_pixstack(int(sep.get_extract_pixstack() * 5))
                except Exception:
                    pass
            continue
    raise last_exc


def _warp_like_ref(target_img: np.ndarray, M_2x3: np.ndarray, ref_shape_hw: tuple[int,int]) -> np.ndarray:
    H, W = ref_shape_hw
    if target_img.ndim == 2:
        return cv2.warpAffine(target_img, M_2x3, (W, H),
                               flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    chs = [cv2.warpAffine(target_img[..., i], M_2x3, (W, H),
                           flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
           for i in range(target_img.shape[2])]
    return np.stack(chs, axis=2)

def run_star_alignment_headless(mw, target_sw, preset: dict) -> bool:
    """
    Headless align the TARGET view (target_sw) to a chosen REFERENCE, as per preset.
    Preset schema (all optional except a reference):
      {
        "reference": {"type":"view_ptr","doc_ptr":123456}
                     | {"type":"view_name","name":"Some View"}
                     | {"type":"active"}
                     | {"type":"file","path":"/abs/path/file.fit"}
        "overwrite": false,          # False => create new view
        "downsample": 2,             # 1,2,3,... speed vs precision
        "title_suffix": "Aligned",   # appended to new view title (if creating)
      }
    """
    try:
        # ---------- resolve target doc & image ----------
        target_doc = _doc_from_sw(target_sw) if target_sw else None
        if target_doc is None or getattr(target_doc, "image", None) is None:
            return False
        tgt_img = np.asarray(target_doc.image, dtype=np.float32)

        # ---------- resolve reference image ----------
        ref_spec = (preset or {}).get("reference", {"type": "active"})
        ref_type = (ref_spec or {}).get("type", "active")
        ref_img = None
        ref_name = "Reference"

        if ref_type == "view_ptr":
            doc_ptr = int(ref_spec.get("doc_ptr", 0))
            ref_doc, _ = _resolve_doc_and_sw_by_ptr(mw, doc_ptr)
            if ref_doc is None or getattr(ref_doc, "image", None) is None:
                raise RuntimeError("reference view_ptr not found or has no image")
            ref_img = np.asarray(ref_doc.image, dtype=np.float32)
            # nice name
            try:
                ref_name = ref_doc.display_name() if callable(getattr(ref_doc, "display_name", None)) else (getattr(ref_doc, "title", None) or "Reference")
            except Exception:
                pass

        elif ref_type == "view_name":
            wanted = str(ref_spec.get("name", "")).strip().lower()
            if not wanted:
                raise RuntimeError("reference view_name missing 'name'")
            ref_doc = None
            if hasattr(mw, "mdi"):
                for sw in mw.mdi.subWindowList():
                    d = getattr(sw.widget(), "document", None)
                    if d is None: continue
                    t = ""
                    try:
                        t = d.display_name() if callable(getattr(d, "display_name", None)) else getattr(d, "title", "") or ""
                    except Exception:
                        pass
                    if str(t).strip().lower() == wanted:
                        ref_doc = d; break
            if ref_doc is None or getattr(ref_doc, "image", None) is None:
                raise RuntimeError(f"reference view_name '{wanted}' not found")
            ref_img = np.asarray(ref_doc.image, dtype=np.float32)
            ref_name = wanted

        elif ref_type == "file":
            p = ref_spec.get("path")
            if not p or not os.path.exists(p):
                raise RuntimeError("reference file does not exist")
            ref_img, _, _, _ = load_image(p)
            if ref_img is None:
                raise RuntimeError("failed to load reference file")
            ref_name = os.path.basename(p)

        else:  # "active"
            # fall back to the app’s active view
            act_doc = target_doc
            if act_doc is None:
                return False
            ref_img = np.asarray(act_doc.image, dtype=np.float32)
            try:
                ref_name = act_doc.display_name() if callable(getattr(act_doc, "display_name", None)) else (getattr(act_doc, "title", None) or "Reference")
            except Exception:
                pass

        # ---------- downsample (optional) ----------
        #ds = int(max(1, (preset or {}).get("downsample", 2)))
        ref_gray = _gray2d(ref_img)
        tgt_gray = _gray2d(tgt_img)
        #if ds > 1:
        #    new_hw_ref = (max(1, ref_gray.shape[1] // ds), max(1, ref_gray.shape[0] // ds))
        #    new_hw_tgt = (max(1, tgt_gray.shape[1] // ds), max(1, tgt_gray.shape[0] // ds))
        #    ref_small = cv2.resize(ref_gray, new_hw_ref, interpolation=cv2.INTER_AREA)
        #    tgt_small = cv2.resize(tgt_gray, new_hw_tgt, interpolation=cv2.INTER_AREA)
        #else:
        #    ref_small, tgt_small = ref_gray, tgt_gray
        ref_small, tgt_small = ref_gray, tgt_gray
        # ---------- find transform ----------
        transform_obj, _pts = aa_find_transform_with_backoff(tgt_small, ref_small)
        M2 = np.array(transform_obj.params[0:2, :], dtype=np.float64)  # keep full precision
        #if ds > 1:
        #    M2 = M2.copy()
        #    M2[0, 2] *= ds
        #    M2[1, 2] *= ds

        # ---------- warp target like reference size ----------
        ref_h, ref_w = ref_gray.shape[:2]
        aligned = _warp_like_ref(tgt_img, M2, (ref_h, ref_w)).astype(np.float32, copy=False)

        # ---------- overwrite or create new ----------
        overwrite = bool((preset or {}).get("overwrite", False))
        if overwrite:
            # push pixels into target doc
            if hasattr(target_doc, "set_image"):
                target_doc.set_image(aligned, step_name=f"Star Alignment → {ref_name}")
            elif hasattr(target_doc, "apply_numpy"):
                target_doc.apply_numpy(aligned, step_name=f"Star Alignment → {ref_name}")
            else:
                target_doc.image = aligned
            # nudge UI
            try:
                if hasattr(target_doc, "changed"):
                    target_doc.changed.emit()
            except Exception:
                pass
        else:
            dm = getattr(mw, "docman", None) or getattr(mw, "doc_manager", None)
            if dm is None:
                raise RuntimeError("document manager not available to create a new view")
            base_title = getattr(target_doc, "display_name", None)
            base = base_title() if callable(base_title) else (base_title or "Image")
            suffix = str((preset or {}).get("title_suffix", "Aligned"))
            title = f"{base} [{suffix} → {ref_name}]"
            meta = {
                "step_name": "Star Alignment",
                "description": f"Aligned to {ref_name}",
                "is_mono": bool(aligned.ndim == 2 or (aligned.ndim == 3 and aligned.shape[2] == 1)),
            }
            newdoc = dm.open_array(aligned, metadata=meta, title=title)
            if hasattr(mw, "_spawn_subwindow_for"):
                mw._spawn_subwindow_for(newdoc)

        return True

    except Exception as e:
        # You can log here if you like
        print(f"[StarAlign headless] error: {e}")
        return False

def compute_pairs_astroalign(source_img: np.ndarray, reference_img: np.ndarray):
    """
    Lock astroalign, return (transform_obj, src_pts(float32), tgt_pts(float32)).
    """
    global _AA_LOCK
    with _AA_LOCK:
        transform_obj, (src_pts, tgt_pts) = astroalign.find_transform(source_img, reference_img)
    return transform_obj, np.asarray(src_pts, np.float32), np.asarray(tgt_pts, np.float32)


def handle_shortcut(payload: dict, mw, target_sw) -> bool:
    """
    Entry point for MainWindow._handle_command_drop.
    Returns True if this module handled the payload.
    """
    try:
        cmd = (payload or {}).get("command_id", "")
        if cmd != STAR_ALIGN_CID:
            return False
        preset = (payload or {}).get("preset", {}) or {}
        return run_star_alignment_headless(mw, target_sw, preset)
    except Exception as e:
        print(f"[StarAlign shortcut] {e}")
        return False

def _fmt_doc_title(doc, widget=None) -> str:
    """
    Best-effort human title for a document/view.
    - calls display_name()/displayName() if callable
    - falls back to widget.windowTitle(), doc.title/name, basename(path)
    """
    # 1) callable attributes
    for attr in ("display_name", "displayName", "title", "name"):
        val = getattr(doc, attr, None)
        if callable(val):
            try:
                s = val()
                if s: return str(s)
            except Exception:
                pass
        elif isinstance(val, (str, bytes)):
            s = val.decode() if isinstance(val, bytes) else val
            if s: return s

    # 2) widget/window title
    if widget is not None and hasattr(widget, "windowTitle"):
        try:
            s = widget.windowTitle()
            if s: return str(s)
        except Exception:
            pass

    # 3) path-ish
    for attr in ("path", "file_path", "filepath", "filename"):
        p = getattr(doc, attr, None)
        if isinstance(p, str) and p:
            return os.path.basename(p)

    return "Untitled"


def _list_open_docs_fallback(parent) -> list[tuple[str, object]]:
    """Fallback when parent._list_open_docs isn't available."""
    items = []
    mdi = getattr(parent, "mdi", None)
    if mdi and hasattr(mdi, "subWindowList"):
        for sub in mdi.subWindowList():
            try:
                w = sub.widget()
                doc = getattr(w, "document", None) or getattr(w, "doc", None)
                if doc is None:
                    continue
                title = _fmt_doc_title(doc, widget=w)
                items.append((title, doc))
            except Exception:
                pass
    return items

def _doc_image(doc):
    """Best-effort to fetch numpy image from a doc."""
    if doc is None:
        return None
    img = getattr(doc, "image", None)
    if img is None and hasattr(doc, "get_image"):
        try: img = doc.get_image()
        except Exception: img = None
    return img


def _active_doc_from_parent(parent) -> object | None:
    """Try your helpers to get the active document (same pattern as Plate Solver)."""
    if hasattr(parent, "_active_doc"):
        try:
            return parent._active_doc()
        except Exception:
            pass
    sw = getattr(parent, "mdi", None)
    if sw and hasattr(sw, "activeSubWindow"):
        asw = sw.activeSubWindow()
        if asw:
            w = asw.widget()
            return getattr(w, "document", None)
    return None

def _get_image_from_active_view(parent) -> tuple[np.ndarray | None, dict | None, bool]:
    """
    Return (image_array, metadata_dict, is_mono) from the active view.
    is_mono is True if the image is 2-D.
    """
    doc = _active_doc_from_parent(parent)
    if not doc:
        return None, None, False
    img = getattr(doc, "image", None)
    meta = getattr(doc, "metadata", None)
    if img is None:
        return None, meta, False
    return img, (meta if isinstance(meta, dict) else {}), (img.ndim == 2)

def _push_image_to_active_view(parent, new_image: np.ndarray, metadata_update: dict | None = None):
    """
    Overwrite the active view's pixels + (optionally) metadata.
    Emits doc.changed if available so views refresh immediately.
    """
    doc = _active_doc_from_parent(parent)
    if not doc:
        raise RuntimeError("No active view/document to push result into.")

    # Replace pixels
    setattr(doc, "image", new_image)

    # Merge metadata
    md = getattr(doc, "metadata", None)
    if not isinstance(md, dict):
        md = {}
        setattr(doc, "metadata", md)
    if metadata_update:
        md.update(metadata_update)

    # Notify UI
    if hasattr(doc, "changed"):
        try:
            doc.changed.emit()
        except Exception:
            pass

    # Give the main window a chance to refresh any side panels
    if hasattr(parent, "_refresh_header_viewer"):
        try:
            parent._refresh_header_viewer(doc)
        except Exception:
            pass
    if hasattr(parent, "currentDocumentChanged"):
        try:
            parent.currentDocumentChanged.emit(doc)
        except Exception:
            pass

ASTROMETRY_API_URL = "http://nova.astrometry.net/api/"

def _cap_points(src_pts: np.ndarray, tgt_pts: np.ndarray, max_cp: int) -> tuple[np.ndarray,np.ndarray]:
    if src_pts.shape[0] <= max_cp:
        return src_pts, tgt_pts
    idx = np.linspace(0, src_pts.shape[0]-1, max_cp, dtype=int)
    return src_pts[idx], tgt_pts[idx]


# ---------------------------------------------------------------------
# Stellar Alignment (Dialog) — uses Active View or File (no slots)
# ---------------------------------------------------------------------
class StellarAlignmentDialog(QDialog):
    def __init__(self, parent, settings, doc_manager=None, list_open_docs_fn=None):

        super().__init__(parent)
        self.setWindowTitle("Stellar Alignment")
        self.setModal(True)

        self.settings = settings
        self.parent_window = parent
        self._docman = doc_manager or getattr(parent, "doc_manager", None)

        # allow caller to pass same helper used by RGBCombinationDialogPro
        if list_open_docs_fn is None:
            cand = getattr(parent, "_list_open_docs", None)
            self._list_open_docs_fn = cand if callable(cand) else None
        else:
            self._list_open_docs_fn = list_open_docs_fn

        self.stellar_source = None
        self.stellar_target = None
        self.aligned_image = None
        self.stretched_image = None
        self.autostretch_enabled = False
        self.source_was_mono = False
        self.target_was_mono = False

        self.source_file_path = None
        self.target_file_path = None
        self._align_progress_in_slot = False

        self.initUI()

    def initUI(self):
        L = QHBoxLayout(self)

        # left controls
        controls = QVBoxLayout()

        # SOURCE
        src_box = QGroupBox("Source Image (Reference)")
        src_l = QVBoxLayout(src_box)

        src_radios = QHBoxLayout()
        self.source_from_file_radio = QRadioButton("From File")
        self.source_from_view_radio = QRadioButton("From View")
        self.source_from_view_radio.setChecked(True)
        src_radios.addWidget(self.source_from_file_radio)
        src_radios.addWidget(self.source_from_view_radio)
        src_l.addLayout(src_radios)

        src_file_row = QHBoxLayout()
        self.btn_source_file = QPushButton("Select File")
        self.lbl_source_file = QLabel("No file selected")
        self.btn_source_file.clicked.connect(self.select_source_file)
        src_file_row.addWidget(self.btn_source_file)
        src_file_row.addWidget(self.lbl_source_file)
        src_l.addLayout(src_file_row)

        # view picker (any view – not only active)
        self.source_view_combo = QComboBox()
        src_view_row = QHBoxLayout()
        src_view_row.addWidget(self.source_view_combo, 1)
        btn_src_refresh = QPushButton("Refresh")
        btn_src_refresh.clicked.connect(self._populate_view_combos)
        src_view_row.addWidget(btn_src_refresh)
        src_l.addLayout(src_view_row)

        controls.addWidget(src_box)

        # TARGET
        tgt_box = QGroupBox("Target Image (To be Aligned)")
        tgt_l = QVBoxLayout(tgt_box)

        tgt_radios = QHBoxLayout()
        self.target_from_file_radio = QRadioButton("From File")
        self.target_from_view_radio = QRadioButton("From View")
        self.target_from_view_radio.setChecked(True)
        tgt_radios.addWidget(self.target_from_file_radio)
        tgt_radios.addWidget(self.target_from_view_radio)
        tgt_l.addLayout(tgt_radios)

        tgt_file_row = QHBoxLayout()
        self.btn_target_file = QPushButton("Select File")
        self.lbl_target_file = QLabel("No file selected")
        self.btn_target_file.clicked.connect(self.select_target_file)
        tgt_file_row.addWidget(self.btn_target_file)
        tgt_file_row.addWidget(self.lbl_target_file)
        tgt_l.addLayout(tgt_file_row)

        self.target_view_combo = QComboBox()
        tgt_view_row = QHBoxLayout()
        tgt_view_row.addWidget(self.target_view_combo, 1)
        btn_tgt_refresh = QPushButton("Refresh")
        btn_tgt_refresh.clicked.connect(self._populate_view_combos)
        tgt_view_row.addWidget(btn_tgt_refresh)
        tgt_l.addLayout(tgt_view_row)

        controls.addWidget(tgt_box)

        xform_box = QGroupBox("Transform / Distortion")
        xf = QFormLayout(xform_box)

        self.xf_model = QComboBox()
        self.xf_model.addItems([
            "Affine (fast)",
            "Homography (projective)",
            "Polynomial (order 3)",
            "Polynomial (order 4)",
        ])
        # map saved value to index
        prefs = _align_prefs(self.settings)
        _model = prefs["model"]
        idx = 0 if _model=="affine" else 1 if _model=="homography" else 2 if _model=="poly3" else 3
        self.xf_model.setCurrentIndex(idx)
        xf.addRow("Model:", self.xf_model)

        self.xf_maxcp = QSpinBox(); self.xf_maxcp.setRange(20, 2000); self.xf_maxcp.setValue(prefs["max_cp"])
        xf.addRow("Max control points:", self.xf_maxcp)

        self.xf_downsample = QSpinBox(); self.xf_downsample.setRange(1, 8); self.xf_downsample.setValue(prefs["downsample"])
        xf.addRow("Solve downsample:", self.xf_downsample)

        self.xf_h_reproj = QDoubleSpinBox(); self.xf_h_reproj.setRange(0.1, 10.0); self.xf_h_reproj.setDecimals(2)
        self.xf_h_reproj.setValue(prefs["h_reproj"])
        xf.addRow("Homog. RANSAC reproj (px):", self.xf_h_reproj)

        def _toggle_rows():
            is_h = (self.xf_model.currentIndex() == 1)  # 1 = Homography
            # Enable/disable only the control…
            self.xf_h_reproj.setEnabled(is_h)
            # …and its label in the form layout (if present)
            lab = xf.labelForField(self.xf_h_reproj)
            if lab is not None:
                lab.setEnabled(is_h)
        _toggle_rows()
        self.xf_model.currentIndexChanged.connect(lambda _ : _toggle_rows())

        controls.addWidget(xform_box)

        # run + status
        self.btn_run = QPushButton("Run Alignment")
        self.btn_run.clicked.connect(self.run_alignment)
        controls.addWidget(self.btn_run)

        self.status_label = QLabel("Status: Ready")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        controls.addWidget(self.status_label)

        L.addLayout(controls)

        # right: preview + actions
        right = QVBoxLayout()
        grp = QGroupBox("Aligned Image")
        rg = QVBoxLayout(grp)

        self.result_preview_label = QLabel()
        self.result_preview_label.setFixedSize(400, 400)
        self.result_preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        rg.addWidget(self.result_preview_label)

        self.btn_autostretch = QPushButton("AutoStretch: OFF")
        self.btn_autostretch.clicked.connect(self.toggle_autostretch)
        rg.addWidget(self.btn_autostretch)

        # actions
        actions_row = QHBoxLayout()
        self.btn_apply_active = QPushButton("Apply to Active View")
        self.btn_create_view = QPushButton("Create New View")
        self.btn_apply_active.clicked.connect(self.apply_to_active_view)
        self.btn_create_view.clicked.connect(self.create_new_view)
        actions_row.addWidget(self.btn_apply_active)
        actions_row.addWidget(self.btn_create_view)
        rg.addLayout(actions_row)

        right.addWidget(grp)
        L.addLayout(right)

        # populate combos initially
        self._populate_view_combos()

    def _persist_xform_from_dialog(self):
        idx = self.xf_model.currentIndex()
        model = "affine" if idx==0 else ("homography" if idx==1 else ("poly3" if idx==2 else "poly4"))
        s = self.settings
        s.setValue("stacking/align/model", model)
        s.setValue("stacking/align/max_cp", int(self.xf_maxcp.value()))
        s.setValue("stacking/align/downsample", int(self.xf_downsample.value()))
        s.setValue("stacking/align/h_reproj", float(self.xf_h_reproj.value()))

    # ------------------------
    # Source/Target loaders (File / Active View)
    # ------------------------
    # inside StellarAlignmentDialog
    def _iter_docs(self) -> list[tuple[str, object]]:
        if self._list_open_docs_fn:
            try:
                return [(str(t), d) for (t, d) in self._list_open_docs_fn()]
            except Exception:
                pass
        return _list_open_docs_fallback(self.parent_window)

    def _populate_view_combos(self):
        items = self._iter_docs()

        def fill(cmb: QComboBox):
            cmb.blockSignals(True)
            cmb.clear()
            for title, doc in items:
                cmb.addItem(title, userData=doc)
            cmb.blockSignals(False)

        fill(self.source_view_combo)
        fill(self.target_view_combo)


    def load_source_from_view(self):
        doc = self.source_view_combo.currentData()
        img = _doc_image(doc)
        if img is None:
            QMessageBox.warning(self, "Error", "Selected source view has no image.")
            return
        self.source_was_mono = (img.ndim == 2)
        if self.source_was_mono:
            img = np.stack([img]*3, axis=-1)
        self.stellar_source = img
        self.lbl_source_file.setText(self.source_view_combo.currentText())

    def load_target_from_view(self):
        doc = self.target_view_combo.currentData()
        img = _doc_image(doc)
        if img is None:
            QMessageBox.warning(self, "Error", "Selected target view has no image.")
            return
        self.target_was_mono = (img.ndim == 2)
        if self.target_was_mono:
            img = np.stack([img]*3, axis=-1)
        self.stellar_target = img
        self.lbl_target_file.setText(self.target_view_combo.currentText())

    def select_source_file(self):
        default_dir = self.settings.value("working_directory", "")
        path, _ = QFileDialog.getOpenFileName(self, "Select Source Image", default_dir,
                    "Images (*.fits *.fit *.xisf *.tif *.tiff *.png *.jpg);;All Files (*)")
        if not path:
            return
        image, header, bit_depth, is_mono = load_image(path)
        if image is None:
            QMessageBox.warning(self, "Error", "Failed to load source image.")
            return
        self.source_was_mono = bool(is_mono or image.ndim == 2)
        if image.ndim == 2:
            image = np.stack([image]*3, axis=-1)
        self.stellar_source = image
        self.lbl_source_file.setText(os.path.basename(path))

    def select_target_file(self):
        default_dir = self.settings.value("working_directory", "")
        path, _ = QFileDialog.getOpenFileName(self, "Select Target Image", default_dir,
                    "Images (*.fits *.fit *.xisf *.tif *.tiff *.png *.jpg);;All Files (*)")
        if not path:
            return
        image, header, bit_depth, is_mono = load_image(path)
        if image is None:
            QMessageBox.warning(self, "Error", "Failed to load target image.")
            return
        self.target_was_mono = bool(is_mono or image.ndim == 2)
        if image.ndim == 2:
            image = np.stack([image]*3, axis=-1)
        self.stellar_target = image
        self.lbl_target_file.setText(os.path.basename(path))

    # ------------------------
    # Preview + stretch
    # ------------------------
    def toggle_autostretch(self):
        if self.aligned_image is None:
            QMessageBox.warning(self, "Stellar Alignment", "No aligned image available.")
            return
        self.autostretch_enabled = not self.autostretch_enabled
        self.btn_autostretch.setText(f"AutoStretch: {'ON' if self.autostretch_enabled else 'OFF'}")
        # recompute stretched version when turning on
        if self.autostretch_enabled:
            self.apply_autostretch()
        # draw
        img = self.stretched_image if self.autostretch_enabled and self.stretched_image is not None else self.aligned_image
        self.update_preview(self.result_preview_label, img)

    def apply_autostretch(self):
        if self.aligned_image is None:
            return
        a = self.aligned_image.astype(np.float32, copy=False)
        m = np.nanmax(a) if a.size else 1.0
        if not np.isfinite(m) or m <= 0:
            m = 1.0
        self.stretched_image = np.clip(a / m, 0, 1)

    def update_preview(self, label, image):
        if image is None:
            return
        disp = image
        if disp.dtype != np.uint8:
            # simple preview scale to 8-bit
            m = float(np.nanmax(disp)) if disp.size else 1.0
            m = m if np.isfinite(m) and m > 0 else 1.0
            disp = np.clip(disp / m * 255.0, 0, 255).astype(np.uint8, copy=False)

        if disp.ndim == 3 and disp.shape[2] == 3:
            h, w, _ = disp.shape
            qimg = QImage(disp.data, w, h, 3*w, QImage.Format.Format_RGB888)
        else:
            h, w = disp.shape[:2]
            qimg = QImage(disp.data, w, h, w, QImage.Format.Format_Grayscale8)
        scaled = qimg.scaled(label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        label.setPixmap(QPixmap.fromImage(scaled))

    # -----------------------------
    # Triangle helpers (kept from your version)
    # -----------------------------
    @staticmethod
    def compute_triangle_invariant(tri_points):
        d1 = np.linalg.norm(tri_points[0] - tri_points[1])
        d2 = np.linalg.norm(tri_points[1] - tri_points[2])
        d3 = np.linalg.norm(tri_points[2] - tri_points[0])
        sides = sorted([d1, d2, d3])
        if sides[0] == 0:
            return None
        return (sides[1] / sides[0], sides[2] / sides[0])

    @staticmethod
    def build_triangle_dict(coords):
        tri = Delaunay(coords)
        tri_dict = {}
        for simplex in tri.simplices:
            pts = coords[simplex]
            inv = StellarAlignmentDialog.compute_triangle_invariant(pts)
            if inv is None:
                continue
            inv_key = (round(inv[0], 2), round(inv[1], 2))
            tri_dict.setdefault(inv_key, []).append(simplex)
        return tri_dict

    @staticmethod
    def match_triangles(src_dict, tgt_dict, tol=0.1):
        matches = []
        for inv_src, src_tris in src_dict.items():
            for inv_tgt, tgt_tris in tgt_dict.items():
                if abs(inv_src[0] - inv_tgt[0]) < tol and abs(inv_src[1] - inv_tgt[1]) < tol:
                    for s in src_tris:
                        for t in tgt_tris:
                            matches.append((s, t))
        return matches

    @staticmethod
    def ransac_affine(src_coords, tgt_coords, matches, ransac_iter=500, inlier_thresh=3.0, update_callback=None):
        best_inliers = 0
        best_transform = None
        tgt_tree = KDTree(tgt_coords)
        total = ransac_iter
        for i in range(ransac_iter):
            src_tri, tgt_tri = random.choice(matches)
            pts_src = np.float32([src_coords[j] for j in src_tri])
            pts_tgt = np.float32([tgt_coords[j] for j in tgt_tri])
            transform, _ = cv2.estimateAffine2D(pts_src.reshape(-1, 1, 2),
                                                pts_tgt.reshape(-1, 1, 2),
                                                method=cv2.LMEDS)
            if transform is None:
                continue
            src_aug = np.hstack([src_coords, np.ones((src_coords.shape[0], 1))])
            transformed = (transform @ src_aug.T).T
            inliers = 0
            for pt in transformed:
                dist, _ = tgt_tree.query(pt)
                if dist < inlier_thresh:
                    inliers += 1
            if inliers > best_inliers:
                best_inliers = inliers
                best_transform = np.eye(3, dtype=np.float32)
                best_transform[:2] = transform

            if update_callback is not None and (i % 10 == 0 or i == total - 1):
                progress = int(100 * i / total)
                update_callback(f"RANSAC progress: {progress}% (Best inliers: {best_inliers})")
        return best_transform, best_inliers

    def estimate_transform_ransac(self, source_stars, target_stars):
        src_coords = np.array([[s[0], s[1]] for s in source_stars])
        tgt_coords = np.array([[s[0], s[1]] for s in target_stars])
        self.status_label.setText("Computing Delaunay triangulation...")
        src_tri_dict = self.build_triangle_dict(src_coords)
        tgt_tri_dict = self.build_triangle_dict(tgt_coords)
        self.status_label.setText("Matching triangles...")
        matches = self.match_triangles(src_tri_dict, tgt_tri_dict, tol=0.1)
        if len(matches) == 0:
            self.status_label.setText("No triangle matches found!")
            return None, 0
        self.status_label.setText(f"Found {len(matches)} candidate triangle matches. Running RANSAC...")
        update_callback = lambda msg: self.status_label.setText(msg)
        best_transform, best_inliers = self.ransac_affine(
            src_coords, tgt_coords, matches, ransac_iter=1000, inlier_thresh=3.0, update_callback=update_callback
        )
        return best_transform, best_inliers

    # -----------------------------
    # Astroalign (with backoff) + warp
    # -----------------------------
    def aa_find_transform_with_backoff(self, tgt_gray: np.ndarray, src_gray: np.ndarray):
        """
        Retry astroalign.find_transform() with progressively stricter detection,
        serializing SEP usage via _AA_LOCK; returns (transform_obj, (src_pts, tgt_pts)).
        """
        tgt32 = np.ascontiguousarray(tgt_gray.astype(np.float32))
        src32 = np.ascontiguousarray(src_gray.astype(np.float32))
        try:
            curr = sep.get_extract_pixstack()
            if curr < 1_500_000:
                sep.set_extract_pixstack(1_500_000)
        except Exception:
            pass

        tries = [
            dict(detection_sigma=5,  min_area=7,  max_control_points=75),
            dict(detection_sigma=12, min_area=9,  max_control_points=75),
            dict(detection_sigma=20, min_area=9,  max_control_points=75),
            dict(detection_sigma=30, min_area=11, max_control_points=75),
            dict(detection_sigma=50, min_area=11, max_control_points=75),
        ]
        last_exc = None
        for kw in tries:
            try:
                global _AA_LOCK
                with _AA_LOCK:
                    return astroalign.find_transform(tgt32, src32, **kw)
            except Exception as e:
                last_exc = e
                if "internal pixel buffer full" in str(e).lower():
                    try:
                        sep.set_extract_pixstack(int(sep.get_extract_pixstack() * 2))
                    except Exception:
                        pass
                continue
        raise last_exc

    def run_alignment(self):
        # Persist dialog choices back to QSettings (safe if the method isn’t present)
        self.status_label.setText("Starting Alignment…")
        QApplication.processEvents()
        try:
            self._persist_xform_from_dialog()
        except Exception:
            pass

        # Ensure sources are loaded
        if self.source_from_view_radio.isChecked() and self.stellar_source is None:
            self.load_source_from_view()
        if self.target_from_view_radio.isChecked() and self.stellar_target is None:
            self.load_target_from_view()

        if self.stellar_source is None:
            QMessageBox.warning(self, "Error", "Please choose a Source (file or view).")
            return
        if self.stellar_target is None:
            QMessageBox.warning(self, "Error", "Please choose a Target (file or view).")
            return

        # Local helpers (self-contained)
        def _cap_points(src_pts: np.ndarray, tgt_pts: np.ndarray, max_cp: int):
            if src_pts.shape[0] <= max_cp:
                return src_pts, tgt_pts
            idx = np.linspace(0, src_pts.shape[0] - 1, max_cp, dtype=int)
            return src_pts[idx], tgt_pts[idx]

        def _estimate_transform_from_pairs(model: str,
                                        src_xy: np.ndarray,
                                        tgt_xy: np.ndarray,
                                        h_reproj: float) -> tuple[str, object]:
            """
            Returns (kind, transform):
            kind="affine"      -> 2x3 float32
            kind="homography"  -> 3x3 float32
            kind="poly3|poly4" -> callable(img, out_hw)->img  (base warp + polynomial residual)
            """
            model = (model or "affine").lower()

            # Base model first (affine or homography)
            if model == "homography":
                H, _ = cv2.findHomography(src_xy, tgt_xy, method=cv2.RANSAC,
                                        ransacReprojThreshold=float(h_reproj))
                if H is None:
                    raise RuntimeError("Homography estimation failed.")
                base_kind, base_X = "homography", np.array(H, dtype=np.float64)
            else:
                A, _ = cv2.estimateAffine2D(src_xy, tgt_xy, method=cv2.RANSAC,
                                            ransacReprojThreshold=float(h_reproj))
                if A is None:
                    raise RuntimeError("Affine estimation failed.")
                base_kind, base_X = "affine", np.array(A, dtype=np.float64)

            if model not in ("poly3", "poly4"):
                return base_kind, base_X

            # Predict with base model
            if base_kind == "affine":
                ones = np.ones((src_xy.shape[0], 1), dtype=np.float32)
                P = np.hstack([src_xy, ones])
                pred_on_ref = (base_X @ P.T).T
            else:
                ones = np.ones((src_xy.shape[0], 1), dtype=np.float32)
                P = np.hstack([src_xy, ones]).T
                Q = (base_X @ P)
                pred_on_ref = (Q[:2, :] / Q[2:3, :]).T

            # Inlier selection for stable poly residual fit
            resid = np.linalg.norm(pred_on_ref - tgt_xy, axis=1)
            r_thresh = max(2.0, h_reproj * 1.5)
            inliers = resid < r_thresh
            if inliers.sum() < 20:
                return base_kind, base_X

            P_ref  = tgt_xy[inliers].astype(np.float32)
            P_pred = pred_on_ref[inliers].astype(np.float32)

            # Normalize to [0,1] domain for conditioning
            Hh, Ww = self.stellar_source.shape[:2]  # or use reference size if you prefer
            scale = np.array([Ww, Hh], dtype=np.float32)
            P_ref_n  = P_ref / scale
            P_pred_n = P_pred / scale

            order = 3 if model == "poly3" else 4
            t_poly = PolynomialTransform()
            ok = t_poly.estimate(P_ref_n, P_pred_n, order=order)  # ref_n -> basewarped_n
            if not ok:
                return base_kind, base_X

            def _warp_poly_residual(img: np.ndarray, out_hw: tuple[int,int]) -> np.ndarray:
                Hout, Wout = out_hw

                # Pass A: base warp to reference grid
                if base_kind == "affine":
                    if img.ndim == 2:
                        base_img = cv2.warpAffine(img, base_X, (Wout, Hout),
                                                flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                    else:
                        base_img = np.stack([cv2.warpAffine(img[..., c], base_X, (Wout, Hout),
                                                            flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                                            for c in range(img.shape[2])], axis=2)
                else:
                    if img.ndim == 2:
                        base_img = cv2.warpPerspective(img, base_X, (Wout, Hout),
                                                    flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                    else:
                        base_img = np.stack([cv2.warpPerspective(img[..., c], base_X, (Wout, Hout),
                                                                flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                                            for c in range(img.shape[2])], axis=2)

                # Pass B: polynomial residual via inverse_map (ref->basewarped), with normalization
                class _InvMap:
                    def __call__(self, coords):
                        coords_n = coords.astype(np.float32) / scale
                        mapped_n = t_poly(coords_n)
                        return mapped_n * scale

                inv = _InvMap()
                try:
                    out = warp(base_img.astype(np.float32, copy=False),
                            inverse_map=inv,
                            output_shape=(Hout, Wout),
                            preserve_range=True,
                            channel_axis=(-1 if base_img.ndim == 3 else None))
                except TypeError:
                    # older skimage: per-channel
                    if base_img.ndim == 2:
                        out = warp(base_img.astype(np.float32), inverse_map=inv,
                                output_shape=(Hout, Wout), preserve_range=True)
                    else:
                        out = np.stack([warp(base_img[..., c].astype(np.float32), inverse_map=inv,
                                            output_shape=(Hout, Wout), preserve_range=True)
                                        for c in range(base_img.shape[2])], axis=2)
                return out.astype(np.float32, copy=False)

            return f"poly{order}", _warp_poly_residual


        # Prepare grayscale for detection
        src = self.stellar_source
        tgt = self.stellar_target
        src_gray = np.mean(src, axis=2) if src.ndim == 3 else src
        tgt_gray = np.mean(tgt, axis=2) if tgt.ndim == 3 else tgt

        # Read dialog prefs
        model = ["affine", "homography", "poly3", "poly4"][self.xf_model.currentIndex()]
        max_cp = int(self.xf_maxcp.value())
        #ds = int(self.xf_downsample.value())
        h_reproj = float(self.xf_h_reproj.value())

        # Downsample for faster star matching (solve stage only)
        #if ds > 1:
        ##    new_ref = (max(1, src_gray.shape[1] // ds), max(1, src_gray.shape[0] // ds))
        #    new_tgt = (max(1, tgt_gray.shape[1] // ds), max(1, tgt_gray.shape[0] // ds))
        #    src_small = cv2.resize(src_gray, new_ref, interpolation=cv2.INTER_AREA)
        #    tgt_small = cv2.resize(tgt_gray, new_tgt, interpolation=cv2.INTER_AREA)
        #else:
        #    src_small, tgt_small = src_gray, tgt_gray
        src_small, tgt_small = src_gray, tgt_gray

        self.status_label.setText("Computing alignment with astroalign…")
        QApplication.processEvents()
        try:
            # NOTE: astroalign returns matched points as (src_pts, tgt_pts)
            #       but we called it with (tgt_small, src_small), so:
            #       src_pts are in tgt_small coords, tgt_pts in src_small coords
            transform_obj, (src_pts_s, tgt_pts_s) = self.aa_find_transform_with_backoff(tgt_small, src_small)
        except Exception as e:
            QMessageBox.warning(self, "Alignment Error", f"Astroalign failed: {e}")
            return

        # Convert to float32 arrays
        src_xy = np.asarray(src_pts_s, dtype=np.float32)
        tgt_xy = np.asarray(tgt_pts_s, dtype=np.float32)

        # Cap control points
        src_xy, tgt_xy = _cap_points(src_xy, tgt_xy, max_cp)

        # If we solved on a downsampled pair, re-fit transform at full resolution for accuracy
        #if ds > 1:
        #    src_xy *= ds
        #    tgt_xy *= ds

        # Estimate chosen transform on (possibly rescaled) full-res pairs
        try:
            kind, X = _estimate_transform_from_pairs(model, src_xy, tgt_xy, h_reproj)
        except Exception as e:
            QMessageBox.warning(self, "Alignment Error", f"Transform estimation failed: {e}")
            return

        self.status_label.setText("Warping target image…")
        QApplication.processEvents()
        H, W = src.shape[:2]

        # Apply the transform
        if kind == "affine":
            if tgt.ndim == 2:
                warped_target = cv2.warpAffine(
                    tgt, X, (W, H),
                    flags=cv2.INTER_LANCZOS4,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0
                )
            else:
                warped_target = np.stack(
                    [cv2.warpAffine(tgt[..., i], X, (W, H),
                                    flags=cv2.INTER_LANCZOS4,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                    for i in range(tgt.shape[2])],
                    axis=2
                )
            transform_3x3 = np.eye(3, dtype=np.float32); transform_3x3[:2] = X
            self.show_transform_info(transform_3x3)

        elif kind == "homography":
            if tgt.ndim == 2:
                warped_target = cv2.warpPerspective(
                    tgt, X, (W, H),
                    flags=cv2.INTER_LANCZOS4,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0
                )
            else:
                warped_target = np.stack(
                    [cv2.warpPerspective(tgt[..., i], X, (W, H),
                                        flags=cv2.INTER_LANCZOS4,
                                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                    for i in range(tgt.shape[2])],
                    axis=2
                )
            # Optional: show homography info as well
            try:
                self.show_transform_info(np.array(X, dtype=np.float64, copy=False))
            except Exception:
                pass

        else:  # polynomial residual callable
            try:
                warped_target = X(tgt, (H, W))
            except Exception as e:
                QMessageBox.warning(self, "Alignment Error", f"Polynomial warp failed ({e}); falling back to affine.")


        # Store + preview (with optional AutoStretch)
        self.aligned_image = warped_target.astype(np.float32, copy=False)
        self.stretched_image = None
        if self.autostretch_enabled:
            self.apply_autostretch()

        disp = self.stretched_image if (self.autostretch_enabled and self.stretched_image is not None) else self.aligned_image
        self.update_preview(self.result_preview_label, disp)
        self.status_label.setText(f"Alignment complete ({model}).")
        QApplication.processEvents()
        QMessageBox.information(self, "Alignment Complete", f"Alignment completed using {model}.")



    def show_transform_info(self, matrix):
        a, b, tx = matrix[0]
        c, d, ty = matrix[1]
        translation = (tx, ty)
        scale_x = np.sqrt(a * a + c * c)
        rotation_rad = np.arctan2(c, a)
        rotation_deg = np.degrees(rotation_rad)
        shear = (a * b + c * d) / (a * a + c * c) if (a * a + c * c) != 0 else 0.0
        det = a * d - b * c
        scale_y = det / scale_x if scale_x != 0 else 0.0

        info_text = (
            f"Transformation Matrix:\n\n"
            f"[{a:.3f}  {b:.3f}  {tx:.3f}]\n"
            f"[{c:.3f}  {d:.3f}  {ty:.3f}]\n"
            f"[0.000  0.000  1.000]\n\n"
            f"Translation: (tx, ty) = ({tx:.3f}, {ty:.3f})\n"
            f"Scaling: scale_x = {scale_x:.3f}, scale_y = {scale_y:.3f}\n"
            f"Rotation: {rotation_deg:.2f}°\n"
            f"Skew (shear): {shear:.3f}\n"
        )

        info_dialog = QDialog(self)
        info_dialog.setWindowTitle("Transformation Matrix Details")
        layout = QVBoxLayout(info_dialog)

        text_edit = QTextEdit(info_dialog)
        text_edit.setReadOnly(True)
        text_edit.setText(info_text)
        layout.addWidget(text_edit)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok, info_dialog)
        button_box.accepted.connect(info_dialog.accept)
        layout.addWidget(button_box)

        info_dialog.show()

    def _output_image(self) -> np.ndarray | None:
        if self.aligned_image is None:
            QMessageBox.warning(self, "Stellar Alignment", "No aligned image. Run alignment first.")
            return None
        img = self.aligned_image
        # if original target was mono and we produced a 3-channel array, collapse back
        if self.target_was_mono and img.ndim == 3 and img.shape[2] >= 1:
            img = img[..., 0]
        if img.dtype != np.float32:
            img = img.astype(np.float32, copy=False)
        return img

    def apply_to_active_view(self):
        img = self._output_image()
        if img is None:
            return
        if not self._docman:
            QMessageBox.warning(self, "Stellar Alignment", "No document manager available.")
            return
        self._docman.update_active_document(
            img, metadata={"description": "Stellar aligned image"},
            step_name="Stellar Alignment"
        )
        QMessageBox.information(self, "Stellar Alignment", "Applied to active view.")
        self.accept()

    def _proposed_title(self) -> str:
        # try to build a friendly title from the source selection
        if self.source_from_view_radio.isChecked():
            doc = self.source_view_combo.currentData()
            try:
                name = doc.display_name() if callable(getattr(doc, "display_name", None)) else getattr(doc, "display_name", None)
                if not isinstance(name, str):
                    name = _fmt_doc_title(doc)
            except Exception:
                name = "Source"
            return f"Aligned_to_{name}"
        if self.source_file_path:
            return f"Aligned_to_{os.path.splitext(os.path.basename(self.source_file_path))[0]}"
        return "Aligned"

    def create_new_view(self):
        img = self._output_image()
        if img is None:
            return
        if not self._docman:
            QMessageBox.warning(self, "Stellar Alignment", "No document manager available.")
            return
        meta = {"step_name": "Stellar Alignment", "description": "Stellar aligned image",
                "is_mono": bool(self.target_was_mono)}
        newdoc = self._docman.open_array(img, metadata=meta, title=self._proposed_title())
        if hasattr(self.parent_window, "_spawn_subwindow_for"):
            self.parent_window._spawn_subwindow_for(newdoc)
        QMessageBox.information(self, "Stellar Alignment", "Created a new view.")
        self.accept()


# ---------------------------------------------------------------------
# Registration (batch) — unchanged behavior; no slots, only files or view as ref
# ---------------------------------------------------------------------
class RegistrationWorkerSignals(QObject):
    progress = pyqtSignal(str)
    result = pyqtSignal(str)
    error = pyqtSignal(str)
    result_transform = pyqtSignal(str, object)  # (orig_file_path, transform_matrix)


# Identity transform (2x3)
IDENTITY_2x3 = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64)

def compute_affine_transform_astroalign_cropped(source_img, reference_img,
                                                scale: float = 1.20,
                                                limit_stars: int | None = None):
    """
    Center-crop the reference to ~scale×source, solve with astroalign on the crop,
    then lift the transform back to full reference coordinates.
    Returns 2x3 affine in full-reference coords, or None.
    """
    import numpy as np
    import astroalign
    Hs, Ws = source_img.shape[:2]
    Hr, Wr = reference_img.shape[:2]

    # crop box
    h = min(int(round(Hs * scale)), Hr)
    w = min(int(round(Ws * scale)), Wr)
    y0 = max(0, (Hr - h) // 2)
    x0 = max(0, (Wr - w) // 2)
    ref_crop = reference_img[y0:y0+h, x0:x0+w]

    # AA solve (pass cap if this AA version supports it)
    try:
        if limit_stars is not None:
            tform, _ = astroalign.find_transform(
                np.ascontiguousarray(source_img.astype(np.float32)),
                np.ascontiguousarray(ref_crop.astype(np.float32)),
                max_control_points=int(limit_stars)
            )
        else:
            tform, _ = astroalign.find_transform(
                np.ascontiguousarray(source_img.astype(np.float32)),
                np.ascontiguousarray(ref_crop.astype(np.float32))
            )
    except TypeError:
        # older astroalign with no max_control_points kwarg
        tform, _ = astroalign.find_transform(
            np.ascontiguousarray(source_img.astype(np.float32)),
            np.ascontiguousarray(ref_crop.astype(np.float32))
        )

    P = np.asarray(tform.params, dtype=np.float64)
    if P.shape == (3, 3):
        T = np.array([[1,0,x0],[0,1,y0],[0,0,1]], dtype=np.float64)
        H_full = T @ P
        return H_full[0:2, :]
    elif P.shape == (2, 3):
        A3 = np.vstack([P, [0,0,1]])
        T  = np.array([[1,0,x0],[0,1,y0],[0,0,1]], dtype=np.float64)
        A_full = (T @ A3)[0:2, :]
        return A_full
    return None


def _solve_delta_job(args):
    try:
        (orig_path, current_transform_2x3, ref_small, Wref, Href,
         resample_flag, det_sigma, limit_stars, minarea) = args

        try:
            cv2.setNumThreads(1)
        except Exception:
            pass

        # 1) read → gray
        with fits.open(orig_path, memmap=True) as hdul:
            arr = hdul[0].data
            if arr is None:
                return (orig_path, None, f"Could not load {os.path.basename(orig_path)}")
            gray = arr if arr.ndim == 2 else np.mean(arr, axis=2)
            gray = np.nan_to_num(gray, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

        # 2) pre-warp to ref size
        T_prev = np.asarray(current_transform_2x3, np.float32).reshape(2, 3)
        src_for_match = cv2.warpAffine(gray, T_prev, (Wref, Href),
                                       flags=resample_flag, borderMode=cv2.BORDER_REFLECT_101)

        # 3) delta solve (NO import — call the top-level helper)
        src_for_match = _suppress_tiny_islands(src_for_match, det_sigma=det_sigma, minarea=minarea)
        ref_small     = _suppress_tiny_islands(ref_small,     det_sigma=det_sigma, minarea=minarea)        
        tform = compute_affine_transform_astroalign_cropped(
            src_for_match, ref_small, limit_stars=limit_stars
        )
        if tform is None:
            return (orig_path, None,
                    f"Astroalign failed for {os.path.basename(orig_path)} – skipping (no transform returned)")

        T_new = np.asarray(tform, np.float64).reshape(2, 3)
        return (orig_path, T_new, None)

    except Exception as e:
        return (orig_path, None, f"Astroalign failed for {os.path.basename(orig_path)}: {e}")

def _suppress_tiny_islands(img32: np.ndarray, det_sigma: float, minarea: int) -> np.ndarray:
    """
    Zero out connected components smaller than `minarea`, using
    threshold = det_sigma * global RMS from SEP background.
    """
    import numpy as np
    import sep, cv2

    try:
        img32 = cv2.medianBlur(img32, 3)   # tame single-pixel spikes
    except Exception:
        pass

    bkg = sep.Background(img32, bw=64, bh=64)
    thresh = float(det_sigma) * float(bkg.globalrms)

    mask = (img32 > (bkg.back() + thresh)).astype(np.uint8)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return img32

    keep = np.zeros(num, dtype=np.uint8)
    keep[stats[:, cv2.CC_STAT_AREA] >= int(minarea)] = 1
    keep[0] = 0
    pruned = keep[labels]

    out = img32.copy()
    out[(mask == 1) & (pruned == 0)] = bkg.back()
    return out


class StarRegistrationWorker(QRunnable):
    def __init__(self, file_path, original_file, current_transform,
                ref_stars, ref_triangles, output_directory,
                use_triangle=False, use_astroalign=False, reference_image=None,
                downsample_factor: int = 2, model_name: str = "affine"):

        super().__init__()
        self.file_path = file_path
        self.original_file = original_file
        self.current_transform = current_transform if current_transform is not None else IDENTITY_2x3
        self.ref_stars = ref_stars
        self.ref_triangles = ref_triangles
        self.output_directory = output_directory
        self.use_triangle = use_triangle
        self.use_astroalign = use_astroalign
        self.reference_image = reference_image  # 2D reference image
        self.downsample_factor = downsample_factor
        self.signals = RegistrationWorkerSignals()
        self.model_name = str(model_name).lower()

    def run(self):
        """
        Incremental stellar registration:
        - load ORIGINAL frame
        - build preview
        - APPLY current accumulated transform to that preview
        - astroalign the already-aligned preview to the reference preview
        - emit *incremental* delta, keyed by ORIGINAL path
        """
        try:
            _cap_native_threads_once()
            try:
                curr = sep.get_extract_pixstack()
                if curr < 1_500_000:
                    sep.set_extract_pixstack(1_500_000)
            except Exception:
                pass

            # 1) load the ORIGINAL frame (we still read from original_file)
            with fits.open(self.original_file, memmap=True) as hdul:
                arr = hdul[0].data
                if arr is None:
                    self.signals.error.emit(f"Could not load {self.original_file}")
                    return
                if arr.ndim == 2:
                    gray = arr
                else:
                    gray = np.mean(arr, axis=2)
                gray = np.nan_to_num(gray, nan=0.0, posinf=0.0, neginf=0.0)

            gray_small = np.ascontiguousarray(gray.astype(np.float32, copy=False))

            ref_small = self.reference_image
            if ref_small is None:
                self.signals.error.emit("Worker missing reference preview.")
                return
            Href, Wref = ref_small.shape[:2]

            # 2) apply CURRENT transform to the preview so we align "from last pass" not "from raw"
            T_prev = np.array(self.current_transform, dtype=np.float32).reshape(2, 3)
            use_warp = not np.allclose(
                T_prev,
                np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32),
                rtol=1e-5,
                atol=1e-5,
            )

            if use_warp and cv2 is not None:
                # warp to REF size so astroalign compares apples to apples
                src_for_match = cv2.warpAffine(
                    gray_small,
                    T_prev,
                    (Wref, Href),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT_101,
                )
            else:
                # no warp or no cv2: at least make the size match
                if gray_small.shape != ref_small.shape and cv2 is not None:
                    src_for_match = cv2.resize(gray_small, (Wref, Href), interpolation=cv2.INTER_LINEAR)
                else:
                    src_for_match = gray_small

            # 3) NOW do astroalign on the already-aligned frame → this gives us the *incremental* delta
            try:
                transform = self.compute_affine_transform_astroalign(
                    src_for_match, ref_small, limit_stars=getattr(self, "limit_stars", None)
                )
            except Exception as e:
                msg = str(e)
                base = os.path.basename(self.original_file)
                if "of matching triangles exhausted" in msg.lower():
                    self.signals.error.emit(
                        f"Astroalign failed for {base}: List of matching triangles exhausted"
                    )
                else:
                    self.signals.error.emit(
                        f"Astroalign failed for {base}: {msg}"
                    )
                return

            if transform is None:
                # this is the other path where AA didn't throw, just couldn't solve
                base = os.path.basename(self.original_file)
                self.signals.error.emit(
                    f"Astroalign failed for {base} – skipping (no transform returned)"
                )
                return

            transform = np.array(transform, dtype=np.float64).reshape(2, 3)

            key = os.path.normpath(self.original_file)
            self.signals.result_transform.emit(key, transform)
            self.signals.progress.emit(
                f"Astroalign delta for {os.path.basename(self.original_file)} "
                f"(model={self.model_name}): dx={transform[0, 2]:.2f}, dy={transform[1, 2]:.2f}"
            )
            self.signals.result.emit(self.original_file)

        except Exception as e:
            self.signals.error.emit(f"Error processing {self.original_file}: {e}")


    @staticmethod
    def compute_affine_transform_astroalign(source_img, reference_img, scale=1.20, limit_stars: int | None = None):
        """
        Fast local match with center-crop, then AA on the crop.
        limit_stars caps the number of control points AA will use (if supported).
        """
        global _AA_LOCK
        try:
            Hs, Ws = source_img.shape[:2]
            Hr, Wr = reference_img.shape[:2]

            # 1) center-crop reference ≈ source*scale
            h = min(int(round(Hs * scale)), Hr)
            w = min(int(round(Ws * scale)), Wr)
            y0 = max(0, (Hr - h) // 2)
            x0 = max(0, (Wr - w) // 2)
            ref_crop = reference_img[y0:y0+h, x0:x0+w]

            # 2) find transform on the small pair
            with _AA_LOCK:
                # NEW: pass limit_stars when available
                try:
                    if limit_stars is not None:
                        tform, _ = astroalign.find_transform(
                            np.ascontiguousarray(source_img.astype(np.float32)),
                            np.ascontiguousarray(ref_crop.astype(np.float32)),
                            max_control_points=int(limit_stars)
                        )
                    else:
                        tform, _ = astroalign.find_transform(
                            np.ascontiguousarray(source_img.astype(np.float32)),
                            np.ascontiguousarray(ref_crop.astype(np.float32))
                        )
                except TypeError:
                    # Older astroalign without max_control_points kwarg
                    tform, _ = astroalign.find_transform(
                        np.ascontiguousarray(source_img.astype(np.float32)),
                        np.ascontiguousarray(ref_crop.astype(np.float32))
                    )

            # 3) lift crop→full coords (unchanged)
            P = np.asarray(tform.params, dtype=np.float64)
            if P.shape == (3, 3):
                T = np.array([[1,0,x0],[0,1,y0],[0,0,1]], dtype=np.float64)
                H_full = T @ P
                return H_full[0:2, :]
            elif P.shape == (2, 3):
                A3 = np.vstack([P, [0,0,1]])
                T  = np.array([[1,0,x0],[0,1,y0],[0,0,1]], dtype=np.float64)
                A_full = (T @ A3)[0:2, :]
                return A_full
            return None
        except Exception as e:
            print(f"[StarRegistrationWorker] astroalign (cropped) failed: {e}")
            return None



class StarRegistrationThread(QThread):
    progress_update = pyqtSignal(str)
    registration_complete = pyqtSignal(bool, str)
    progress_step = pyqtSignal(int, int)  # (done, total)

    def __init__(self, reference_image_path_or_view, files_to_align, output_directory,
                 max_refinement_passes=3, shift_tolerance=0.2, parent_window=None, align_prefs: dict | None = None):
        """
        reference_image_path_or_view: path string OR "__ACTIVE_VIEW__"
        If "__ACTIVE_VIEW__", we'll read the current active view as the reference frame.
        """
        super().__init__()
        self.reference = reference_image_path_or_view
        self.parent_window = parent_window
        self.original_files = [os.path.normpath(f) for f in files_to_align]
        self.files_to_align = self.original_files.copy()
        self.output_directory = os.path.normpath(output_directory)
        self.max_refinement_passes = max_refinement_passes
        self.shift_tolerance = shift_tolerance

        self.file_key_to_current_path = {f: f for f in self.original_files}
        self.alignment_matrices = {}
        self.transform_deltas = []
        self._done = 0
        self._total = len(self.original_files) * self.max_refinement_passes
        self.align_prefs = align_prefs or _align_prefs(QSettings())
        self.align_model = str(self.align_prefs.get("model", "affine")).lower()
        self.h_reproj   = float(self.align_prefs.get("h_reproj", 3.0))
        self.det_sigma   = float(self.align_prefs.get("det_sigma", 12.0))
        self.limit_stars = int(self.align_prefs.get("limit_stars", 500))
        self.minarea     = int(self.align_prefs.get("minarea", 10))
        self.downsample = int(self.align_prefs.get("downsample", 2))
        self.drizzle_xforms = {}  # {orig_norm_path: (kind, matrix)}

    def _estimate_model_transform(self, src_gray_full: np.ndarray) -> tuple[str, object]:
        """
        Fast, robust final transform: crop reference to ~1.2× source size (centered),
        solve on the crop, lift correspondences to FULL reference coords, then
        re-estimate the requested model (affine/homography or poly3/4). Returns (kind, X).
        """
        ref2d = self.reference_image_2d
        src = np.ascontiguousarray(src_gray_full.astype(np.float32))
        ref = np.ascontiguousarray(ref2d.astype(np.float32))
        Hs, Ws = src.shape[:2]
        Hr, Wr = ref.shape[:2]

        # ---- 1) center crop the reference to ~1.2× source ----
        scale = 1.20
        h = min(int(round(Hs * scale)), Hr)
        w = min(int(round(Ws * scale)), Wr)
        y0 = max(0, (Hr - h) // 2)
        x0 = max(0, (Wr - w) // 2)
        ref_crop = ref[y0:y0+h, x0:x0+w]

        # ---- 2) find_transform on the small pair; lift matches to full coords ----
        with _AA_LOCK:
            tform, (src_pts_s, tgt_pts_s) = astroalign.find_transform(src, ref_crop)

        src_xy = np.asarray(src_pts_s, dtype=np.float32)
        tgt_xy = np.asarray(tgt_pts_s, dtype=np.float32)
        tgt_xy[:, 0] += x0   # lift crop -> full
        tgt_xy[:, 1] += y0

        # Build a base full-ref transform from tform.params + crop translation
        P = np.asarray(tform.params, dtype=np.float64)
        if P.shape == (3,3):
            base_kind0 = "homography"
            T  = np.array([[1,0,x0],[0,1,y0],[0,0,1]], dtype=np.float64)
            base_X0 = T @ P
        else:
            base_kind0 = "affine"
            A3 = np.vstack([P[0:2,:], [0,0,1]])
            T  = np.array([[1,0,x0],[0,1,y0],[0,0,1]], dtype=np.float64)
            base_X0 = (T @ A3)[0:2, :]

        # ---- 3) re-estimate requested model using full-coord pairs ----
        model = (self.align_model or "affine").lower()
        h_reproj = float(self.h_reproj)

        if model == "homography":
            H, _ = cv2.findHomography(src_xy, tgt_xy, method=cv2.RANSAC, ransacReprojThreshold=h_reproj)
            if H is None:
                base_kind, base_X = base_kind0, base_X0
            else:
                base_kind, base_X = "homography", np.array(H, dtype=np.float64)
        elif model == "affine":
            A, _ = cv2.estimateAffine2D(src_xy, tgt_xy, method=cv2.RANSAC, ransacReprojThreshold=h_reproj)
            if A is None:
                base_kind, base_X = base_kind0, base_X0
            else:
                base_kind, base_X = "affine", np.array(A, dtype=np.float64)
        else:
            base_kind, base_X = base_kind0, base_X0  # for poly we refine from base

        # ---- 4) if not poly, we’re done ----
        if model not in ("poly3", "poly4"):
            return base_kind, base_X

        # ---- 5) poly residual refinement (unchanged logic, but with our pairs) ----
        if base_kind == "affine":
            pred_on_ref = _apply_affine_to_pts(base_X, src_xy)
        else:
            ones = np.ones((src_xy.shape[0], 1), dtype=np.float32)
            P3 = np.hstack([src_xy.astype(np.float32), ones]).T
            Q  = (np.asarray(base_X, np.float32) @ P3)
            pred_on_ref = (Q[:2, :] / Q[2:3, :]).T

        resid = np.linalg.norm(pred_on_ref - tgt_xy, axis=1)
        r_thresh = max(2.0, h_reproj * 1.5)
        inliers = resid < r_thresh
        if inliers.sum() < 20:
            return base_kind, base_X

        P_ref   = tgt_xy[inliers].astype(np.float32)
        P_pred  = pred_on_ref[inliers].astype(np.float32)

        Href, Wref = ref2d.shape[:2]
        scale_vec = np.array([Wref, Href], dtype=np.float32)
        P_ref_n  = P_ref  / scale_vec
        P_pred_n = P_pred / scale_vec

        order = 3 if model == "poly3" else 4
        t_poly = PolynomialTransform()
        ok = t_poly.estimate(P_ref_n, P_pred_n, order=order)  # ref_n -> basewarped_n
        if not ok:
            return base_kind, base_X

        def _warp_poly_residual(img: np.ndarray, out_hw: tuple[int,int]) -> np.ndarray:
            Hh, Ww = out_hw
            # Pass A: base warp
            if base_kind == "affine":
                if img.ndim == 2:
                    img_base = cv2.warpAffine(img, base_X, (Ww, Hh),
                                            flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                else:
                    img_base = np.stack([cv2.warpAffine(img[..., c], base_X, (Ww, Hh),
                                                        flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                                        for c in range(img.shape[2])], axis=2)
            else:
                if img.ndim == 2:
                    img_base = cv2.warpPerspective(img, base_X, (Ww, Hh),
                                                flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                else:
                    img_base = np.stack([cv2.warpPerspective(img[..., c], base_X, (Ww, Hh),
                                                            flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                                        for c in range(img.shape[2])], axis=2)

            class _InvMap:
                def __call__(self, coords):
                    coords_n = coords.astype(np.float32) / scale_vec
                    mapped_n = t_poly(coords_n)
                    return mapped_n * scale_vec

            try:
                out = warp(img_base.astype(np.float32, copy=False),
                        inverse_map=_InvMap(),
                        output_shape=(Hh, Ww),
                        preserve_range=True,
                        channel_axis=(-1 if img_base.ndim == 3 else None))
            except TypeError:
                if img_base.ndim == 2:
                    out = warp(img_base.astype(np.float32), inverse_map=_InvMap(),
                            output_shape=(Hh, Ww), preserve_range=True)
                else:
                    chs = [warp(img_base[..., c].astype(np.float32), inverse_map=_InvMap(),
                                output_shape=(Hh, Ww), preserve_range=True)
                        for c in range(img_base.shape[2])]
                    out = np.stack(chs, axis=2)
            return out.astype(np.float32, copy=False)

        return f"poly{order}", _warp_poly_residual



    def _warp_with_kind(self, img: np.ndarray, kind: str, X: object, out_hw: tuple[int,int]) -> np.ndarray:
        Hh, Ww = out_hw
        if kind == "affine":
            A = np.asarray(X, np.float64)
            if img.ndim == 2:
                return cv2.warpAffine(img, A, (Ww, Hh), flags=cv2.INTER_LANCZOS4,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            return np.stack([cv2.warpAffine(img[..., c], A, (Ww, Hh),
                                            flags=cv2.INTER_LANCZOS4,
                                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                            for c in range(img.shape[2])], axis=2)

        if kind.startswith("poly"):
            return X(img, (Hh, Ww))

        if kind == "homography":
            H = np.asarray(X, np.float64)
            if img.ndim == 2:
                return cv2.warpPerspective(img, H, (Ww, Hh), flags=cv2.INTER_LANCZOS4,
                                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            return np.stack([cv2.warpPerspective(img[..., c], H, (Ww, Hh),
                                                flags=cv2.INTER_LANCZOS4,
                                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                            for c in range(img.shape[2])], axis=2)

        # TPS: X is a callable(img, out_hw) -> img
        return X(img, (Hh, Ww))


    def run(self):
        self.progress_update.emit(f"Alignment model = {self.align_model}")
        try:
            _cap_native_threads_once()

            # Resolve reference → 2D float32
            if isinstance(self.reference, str) and self.reference == "__ACTIVE_VIEW__":
                ref_img, _, _ = _get_image_from_active_view(self.parent_window)
                if ref_img is None:
                    self.registration_complete.emit(False, "Active view not available for reference.")
                    return
                ref2d = np.mean(ref_img, axis=2) if ref_img.ndim == 3 else ref_img
            else:
                ref_img, _, _, _ = load_image(self.reference)
                if ref_img is None:
                    self.registration_complete.emit(False, "Reference image failed to load!")
                    return
                ref2d = np.mean(ref_img, axis=2) if ref_img.ndim == 3 else ref_img

            ref2d = np.nan_to_num(ref2d, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
            self.reference_image_2d = ref2d

            # ✂️ No DAO/RANSAC: astroalign handles detection internally.

            # Single shared downsampled ref for workers
            #ds = max(1, int(self.align_prefs.get("downsample", 2)))
            #if ds > 1:
            #    new_hw = (max(1, ref2d.shape[1] // ds), max(1, ref2d.shape[0] // ds))  # (W, H)
            #    ref_small = cv2.resize(ref2d, new_hw, interpolation=cv2.INTER_AREA)
            #else:
            #    ref_small = ref2d
            #self.ref_small = np.ascontiguousarray(ref_small.astype(np.float32))
            self.ref_small = np.ascontiguousarray(ref2d.astype(np.float32))

            # Initialize transforms to identity for EVERY original frame
            self.alignment_matrices = {os.path.normpath(f): IDENTITY_2x3.copy() for f in self.original_files}
            self.delta_transforms = {}

            # Progress totals (units = number of worker completions across passes)
            self._done = 0
            self._total = len(self.original_files) * max(1, int(self.max_refinement_passes))

            # Registration passes (compute deltas only)
            for pass_idx in range(self.max_refinement_passes):
                self.progress_update.emit(f"⏳ Refinement Pass {pass_idx + 1}/{self.max_refinement_passes}…")
                success, msg = self.run_one_registration_pass(None, None, pass_idx)
                if not success:
                    any_aligned = any(x is not None for x in self.alignment_matrices.values())
                    if not any_aligned:
                        self.registration_complete.emit(False, "No frames could be aligned. Aborting.")
                        return
                    self.progress_update.emit("Partial success: some frames permanently failed.")
                    break

                # Convergence check on this pass’ deltas
                if self.transform_deltas and max(self.transform_deltas[-1]) < self.shift_tolerance:
                    self.progress_update.emit("✅ Convergence reached! Stopping refinement.")
                    break

            # Finalize: single full-res read → warp → write per frame
            self._finalize_writes()

            # Summary based on our known corpus
            total_count = len(self.original_files)
            aligned_count = sum(1 for f in self.original_files if os.path.exists(self.file_key_to_current_path.get(f, "")))
            summary = f"Registration complete. Valid frames: {aligned_count}/{total_count}."
            self.registration_complete.emit(True, summary)

        except Exception as e:
            self.registration_complete.emit(False, f"Error: {e}")


    def _increment_progress(self):
        self._done += 1
        self.progress_step.emit(self._done, self._total)

    # ─────────────────────────────────────────────────────────────
    # Drop-in replacement for: StarRegistrationThread.run_one_registration_pass
    # ─────────────────────────────────────────────────────────────
    def run_one_registration_pass(self, _ref_stars_unused, _ref_triangles_unused, pass_index):
        _cap_native_threads_once()

        # Decide a light resampler for iterative solves (keep LANCZOS for final write)
        resample_flag = cv2.INTER_AREA if pass_index == 0 else cv2.INTER_LINEAR

        # Build the worklist: only frames that still need refinement
        if pass_index == 0:
            work_list = list(self.original_files)
        else:
            work_list = []
            for orig in self.original_files:
                k = os.path.normpath(orig)
                last_delta = self.delta_transforms.get(k, float("inf"))
                if not (last_delta < self.shift_tolerance):
                    work_list.append(orig)

        skipped = len(self.original_files) - len(work_list)
        if skipped > 0:
            self.progress_update.emit(
                f"Skipping {skipped} frame(s) already within {self.shift_tolerance:.2f}px."
            )
            # Advance progress for skipped items so the bar moves
            for _ in range(skipped):
                self._increment_progress()

        # If nothing to do in this pass, we’re already converged
        if not work_list:
            self.transform_deltas.append([
                self.delta_transforms.get(os.path.normpath(f), 0.0)
                for f in self.original_files
            ])
            return True, "Pass complete (nothing to refine)."

        # Prepare shared ref preview & geometry for processes
        ref_small = np.ascontiguousarray(self.ref_small.astype(np.float32, copy=False))
        Href, Wref = ref_small.shape[:2]

        # Max processes; be generous but avoid thrashing
        procs = max(2, min((os.cpu_count() or 8), 32))
        self.progress_update.emit(f"Using {procs} processes for stellar alignment (HW={os.cpu_count() or 8}).")

        # Build jobs (orig path, current 2x3, ref_small, Wref, Href, resample_flag)
        jobs = []
        for original_file in work_list:
            orig_key = os.path.normpath(original_file)
            current_transform = self.alignment_matrices.get(orig_key, IDENTITY_2x3)
            jobs.append((
                original_file, current_transform, ref_small, Wref, Href,
                resample_flag,
                float(self.det_sigma), int(self.limit_stars), int(self.minarea)
            ))

        # Run the delta solves out-of-process (true multi-core)
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=procs) as ex:
            futs = [ex.submit(_solve_delta_job, j) for j in jobs]
            for fut in as_completed(futs):
                try:
                    orig_path, T_new, err = fut.result()
                except Exception as e:
                    orig_path, T_new, err = ("<unknown>", None, f"Worker crashed: {e}")

                if err:
                    self.on_worker_error(err)
                    self._increment_progress()
                    continue

                # Collate the *delta* and accumulate into the running 2x3,
                # mirroring your on_worker_result_transform logic:
                k = os.path.normpath(orig_path)
                T_new = np.array(T_new, dtype=np.float64).reshape(2, 3)

                # delta magnitude for convergence reporting
                self.delta_transforms[k] = float(np.hypot(T_new[0, 2], T_new[1, 2]))

                T_prev = np.array(self.alignment_matrices.get(k, IDENTITY_2x3), dtype=np.float64).reshape(2, 3)
                prev_3 = np.vstack([T_prev, [0, 0, 1]])
                new_3  = np.vstack([T_new,  [0, 0, 1]])
                combined = new_3 @ prev_3
                self.alignment_matrices[k] = combined[0:2, :]

                # progress line like the original worker
                self.on_worker_progress(
                    f"Astroalign delta for {os.path.basename(orig_path)} "
                    f"(model={self.align_model}): dx={T_new[0, 2]:.2f}, dy={T_new[1, 2]:.2f}"
                )
                self._increment_progress()

        # Collate deltas deterministically in original order
        pass_deltas = []
        aligned_count = 0
        for orig in self.original_files:
            k = os.path.normpath(orig)
            d = self.delta_transforms.get(k, 0.0)
            pass_deltas.append(d)
            if d <= self.shift_tolerance:
                aligned_count += 1

        self.transform_deltas.append(pass_deltas)

        preview = ", ".join([f"{d:.2f}" for d in pass_deltas[:10]])
        if len(pass_deltas) > 10:
            preview += f" … ({len(pass_deltas)} total)"
        self.progress_update.emit(f"Pass {pass_index + 1} delta shifts: [{preview}]")
        if aligned_count:
            self.progress_update.emit(f"Skipped (delta < {self.shift_tolerance:.2f}px): {aligned_count} frame(s)")

        return True, "Pass complete."



    def on_worker_result_transform(self, persistent_key, new_transform):
        k = os.path.normpath(persistent_key)
        T_new = np.array(new_transform, dtype=np.float64).reshape(2, 3)

        self.delta_transforms[k] = float(np.hypot(T_new[0, 2], T_new[1, 2]))

        T_prev = np.array(self.alignment_matrices.get(k, IDENTITY_2x3), dtype=np.float64).reshape(2, 3)
        prev_3 = np.vstack([T_prev, [0, 0, 1]])
        new_3  = np.vstack([T_new,  [0, 0, 1]])
        combined = new_3 @ prev_3
        self.alignment_matrices[k] = combined[0:2, :]

    def on_worker_progress(self, msg):
        self.progress_update.emit(msg)

    def on_worker_error(self, msg):
        self.progress_update.emit("Error: " + msg)

    def on_worker_result(self, out):
        print("Saved: " + out)

    # ----- Star detection (reference) -----
    def detect_stars(self, image):
        self.progress_update.emit("✨ Detecting stars in reference frame")
        if image.ndim == 3:
            image = np.mean(image, axis=2)

        mean, median, std = sigma_clipped_stats(image)
        fwhm_list = [2.5, 3, 3.5, 4, 5, 6, 7]

        all_sources = []
        for fwhm in fwhm_list:
            daofind = DAOStarFinder(fwhm=fwhm, threshold=4 * std)
            sources = daofind(image - median)
            if sources is not None and len(sources) > 0:
                all_sources.append(sources)

        if not all_sources:
            return np.empty((0, 2), dtype=np.float32)

        combined_sources = vstack(all_sources)
        x_rounded = np.round(combined_sources['xcentroid'], 1)
        y_rounded = np.round(combined_sources['ycentroid'], 1)
        xy_rounded = np.array([x_rounded, y_rounded]).T

        seen = {}
        unique_rows = []
        for i, (rx, ry) in enumerate(xy_rounded):
            key = (rx, ry)
            if key not in seen:
                seen[key] = True
                unique_rows.append(i)

        final_sources = combined_sources[unique_rows]
        star_coords = np.vstack([final_sources['xcentroid'], final_sources['ycentroid']]).T
        return star_coords.astype(np.float32)

    # ----- Triangle dict helpers -----
    def build_triangle_dict(self, coords):
        tri = Delaunay(coords)
        tri_dict = {}
        for simplex in tri.simplices:
            pts = coords[simplex]
            inv = self.compute_triangle_invariant(pts)
            if inv is None:
                continue
            inv_key = (round(inv[0], 2), round(inv[1], 2))
            tri_dict.setdefault(inv_key, []).append(simplex)
        return tri_dict

    @staticmethod
    def compute_triangle_invariant(tri_points):
        d1 = np.linalg.norm(tri_points[0] - tri_points[1])
        d2 = np.linalg.norm(tri_points[1] - tri_points[2])
        d3 = np.linalg.norm(tri_points[2] - tri_points[0])
        sides = sorted([d1, d2, d3])
        if sides[0] == 0:
            return None
        return (round(sides[1] / sides[0], 4), round(sides[2] / sides[0], 4))

    # ----- Validity + warp -----
    @staticmethod
    def is_valid_transform_static(matrix):
        a, b, tx = matrix[0]
        c, d, ty = matrix[1]
        scale_x = np.sqrt(a ** 2 + c ** 2)
        scale_y = np.sqrt(b ** 2 + d ** 2)
        return (0.9 <= scale_x <= 1.1) and (0.9 <= scale_y <= 1.1)

    @staticmethod
    def apply_affine_transform_static(image, transform_matrix):
        T = np.array(transform_matrix, dtype=np.float32).reshape(2, 3)
        h, w = image.shape[:2]
        if image.ndim == 2:
            aligned = cv2.warpAffine(image, T, (w, h), flags=cv2.INTER_LANCZOS4,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        else:
            chans = []
            for i in range(image.shape[2]):
                chans.append(cv2.warpAffine(image[:, :, i], T, (w, h), flags=cv2.INTER_LANCZOS4,
                                            borderMode=cv2.BORDER_CONSTANT, borderValue=0))
            aligned = np.stack(chans, axis=2)
        return aligned

    # ─────────────────────────────────────────────────────────────
    # NEW METHOD: StarRegistrationThread._finalize_writes
    # ─────────────────────────────────────────────────────────────
    def _finalize_writes(self):
        """
        The only heavy IO:
        • full-res read ONCE
        • apply final 2×3 transform
        • write *_n_r.fit ONCE
        """
        
        from concurrent.futures import ThreadPoolExecutor, as_completed
        self.drizzle_xforms = {}  # { orig_norm_path : (kind, matrix_or_None_for_TPS) }

        io_workers = max(1, min(4, (os.cpu_count() or 8) // 4))  # be gentle to disk

        def _final_write(orig_path):
            try:
                k = os.path.normpath(orig_path)
                T_final = np.array(self.alignment_matrices.get(k, IDENTITY_2x3), dtype=np.float32)
                img, hdr, fmt, bit_depth = load_image(orig_path)
                if img is None:
                    return f"⚠️ Failed to read {os.path.basename(orig_path)}", False


                # Full-res, model-aware warp
                src_gray_full = np.mean(img, axis=2) if img.ndim == 3 else img
                src_gray_full = np.nan_to_num(src_gray_full, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
                Href, Wref = self.reference_image_2d.shape[:2]

                if img.ndim == 3:
                    Hs, Ws = img.shape[:2]
                else:
                    Hs, Ws = img.shape
                if abs(Hs - Href) > 0.05*Href or abs(Ws - Wref) > 0.05*Wref:
                    self.progress_update.emit(
                        f"ℹ️ {os.path.basename(orig_path)} shape {Ws}×{Hs} differs from ref {Wref}×{Href}; warping to match."
                    )

                try:
                    if self.align_model == "affine":
                        # reuse the accumulated 2x3 (no second astroalign)
                        kind = "affine"
                        X = np.asarray(self.alignment_matrices.get(k, IDENTITY_2x3), dtype=np.float64)
                    else:
                        # need a fresh full-res solve for homography/poly
                        kind, X = self._estimate_model_transform(src_gray_full)
                except Exception as e:
                    self.progress_update.emit(
                        f"⚠️ {os.path.basename(orig_path)} model solve failed ({self.align_model}): {e}. Falling back to affine."
                    )
                    kind, X = "affine", np.asarray(self.alignment_matrices.get(k, IDENTITY_2x3), dtype=np.float64)


                self.progress_update.emit(f"🌀 Distortion Correction on {os.path.basename(orig_path)}: warp={kind}")
                # Record drizzle transform (don’t np.asarray() a callable)

                aligned = self._warp_with_kind(img, kind, X, (Href, Wref))
                if aligned is None:
                    return f"⚠️ Warp failed {os.path.basename(orig_path)}", False

                # ✅ keep the exact model-aware transform for drizzle (per ORIGINAL key)
                try:
                    k_norm = os.path.normpath(orig_path)
                    if kind == "affine":
                        A = np.asarray(X, np.float64).reshape(2, 3)
                        self.drizzle_xforms[k_norm] = ("affine", A)
                    elif kind == "homography":
                        H = np.asarray(X, np.float64).reshape(3, 3)
                        self.drizzle_xforms[k_norm] = ("homography", H)
                    elif kind.startswith("poly"):
                        self.drizzle_xforms[k_norm] = (kind, None)
                    else:
                        # fall back to affine
                        A = np.asarray(X, np.float64).reshape(2, 3)
                        self.drizzle_xforms[k_norm] = ("affine", A)
                except Exception:
                    pass


                if np.isnan(aligned).any() or np.isinf(aligned).any():
                    aligned = np.nan_to_num(aligned, nan=0.0, posinf=0.0, neginf=0.0)

                base = os.path.basename(orig_path)
                name, _ = os.path.splitext(base)
                if name.endswith("_n"):
                    name = name[:-2]
                if not name.endswith("_n_r"):
                    name += "_n_r"
                out_path = os.path.join(self.output_directory, f"{name}.fit")

                save_image(
                    img_array=aligned,
                    filename=out_path,
                    original_format="fit",
                    bit_depth=bit_depth,
                    original_header=hdr,
                    is_mono=(aligned.ndim == 2)
                )
                # update downstream mapping to the NEW path
                self.file_key_to_current_path[k] = out_path
                return f"💾 Wrote {os.path.basename(out_path)} [{kind}]", True
            except Exception as e:
                return f"⚠️ Finalize error {os.path.basename(orig_path)}: {e}", False

        self.progress_update.emit("📝 Finalizing aligned outputs (single write per frame)…")
        ok = 0
        Href, Wref = self.reference_image_2d.shape[:2]
        self._ref_shape_for_sasd = (Href, Wref)        
        with ThreadPoolExecutor(max_workers=io_workers) as ex:
            futs = {ex.submit(_final_write, f): f for f in self.original_files}
            for fut in as_completed(futs):
                msg, success = fut.result()
                if success:
                    ok += 1
                self.progress_update.emit(msg)

        try:
            sasd_path = os.path.join(self.output_directory, "alignment_transforms.sasd")
            self._save_alignment_transforms_sasd(sasd_path)
            self.progress_update.emit(f"✅ Transform file saved as alignment_transforms.sasd")
        except Exception as e:
            self.progress_update.emit(f"⚠️ Failed to save alignment_transforms.sasd: {e}")

    def _save_alignment_transforms_sasd(self, out_path: str):
        """
        SASD v2.1 format:

            REF_SHAPE: <H>, <W>
            REF_PATH: <reference path or __ACTIVE_VIEW__>
            MODEL: mixed               # informative; real model is per file

            FILE: <abs path to *_n.fit>
            KIND: affine|homography|tps
            MATRIX:
            a, b, tx
            c, d, ty

            FILE: <next>
            KIND: homography
            MATRIX:
            h00, h01, h02
            h10, h11, h12
            h20, h21, h22

        Blank line between blocks.
        """
        # reference geometry + path
        try:
            Href, Wref = self.reference_image_2d.shape[:2]
        except Exception:
            Href, Wref = 0, 0
        ref_path = self.reference if isinstance(self.reference, str) else "__ACTIVE_VIEW__"

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"REF_SHAPE: {int(Href)}, {int(Wref)}\n")
            f.write(f"REF_PATH: {ref_path}\n")
            f.write("MODEL: mixed\n\n")

            # prefer model-aware drizzle_xforms; fall back to affine stack if missing
            for orig_key in self.original_files:
                k = os.path.normpath(orig_key)
                kind = None
                M = None

                if isinstance(getattr(self, "drizzle_xforms", None), dict) and k in self.drizzle_xforms:
                    kind, M = self.drizzle_xforms[k]
                else:
                    # fallback: affine-only (2x3) from accumulated alignment_matrices
                    M_aff = np.asarray(self.alignment_matrices.get(k, IDENTITY_2x3), dtype=np.float32)
                    kind, M = "affine", M_aff

                f.write(f"FILE: {k}\n")
                f.write(f"KIND: {kind}\n")
                f.write("MATRIX:\n")

                _fmt = lambda x: f"{float(x):.16g}"

                # then for writing:
                if kind == "homography":
                    H = np.asarray(M, np.float64).reshape(3, 3)
                    f.write(f"{_fmt(H[0,0])}, {_fmt(H[0,1])}, {_fmt(H[0,2])}\n")
                    f.write(f"{_fmt(H[1,0])}, {_fmt(H[1,1])}, {_fmt(H[1,2])}\n")
                    f.write(f"{_fmt(H[2,0])}, {_fmt(H[2,1])}, {_fmt(H[2,2])}\n\n")
                elif kind == "affine":
                    A = np.asarray(M, np.float64).reshape(2, 3)
                    f.write(f"{_fmt(A[0,0])}, {_fmt(A[0,1])}, {_fmt(A[0,2])}\n")
                    f.write(f"{_fmt(A[1,0])}, {_fmt(A[1,1])}, {_fmt(A[1,2])}\n\n")
                else:
                    f.write("MATRIX: \nUNSUPPORTED\n\n")

def _center_crop_params(Href, Wref, Hsrc, Wsrc, scale=1.10):
    # crop box centered in reference, sized ~ source*scale, but clamped to ref
    h = min(int(round(Hsrc * scale)), Href)
    w = min(int(round(Wsrc * scale)), Wref)
    y0 = max(0, (Href - h) // 2)
    x0 = max(0, (Wref - w) // 2)
    return y0, x0, h, w

def _crop(img, y0, x0, h, w):
    return img[y0:y0+h, x0:x0+w]

def _compose_with_ref_translation_affine(A_2x3, x0, y0):
    # A_full = T @ A  (homog), then take 2x3 back
    A = np.asarray(A_2x3, dtype=np.float64).reshape(2,3)
    A3 = np.vstack([A, [0,0,1]])
    T  = np.array([[1,0,x0],[0,1,y0],[0,0,1]], dtype=np.float64)
    return (T @ A3)[0:2,:]

def _compose_with_ref_translation_homography(H_3x3, x0, y0):
    H = np.asarray(H_3x3, dtype=np.float64).reshape(3,3)
    T = np.array([[1,0,x0],[0,1,y0],[0,0,1]], dtype=np.float64)
    return T @ H


# ---------------------------------------------------------------------
# Optional simple batch UI (file-based) — no slots
# ---------------------------------------------------------------------
class StarRegistrationWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.reference_image = None
        self.reference_image_path = None  # or "__ACTIVE_VIEW__"
        self.files_to_align = []
        self.output_directory = None
        self.thread = None
        self.parent_window = parent
        self.setWindowFlag(Qt.WindowType.Window, True)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)        
        self.initUI()

        self._prog_timer = QTimer(self)
        self._prog_timer.setInterval(100)      # 10 Hz flush
        self._prog_timer.setSingleShot(True)
        self._prog_timer.timeout.connect(self._flush_progress)
        self._pending_prog = None
        self._in_progress_slot = False

    def initUI(self):
        self.setWindowTitle("Star Registration")
        self.setGeometry(200, 200, 600, 450)
        main_layout = QVBoxLayout(self)

        # Reference selection
        ref_layout = QHBoxLayout()
        self.ref_label = QLabel("Reference Image:")
        self.ref_path_label = QLabel("No reference selected")
        self.ref_path_label.setWordWrap(True)

        self.select_ref_active_button = QPushButton("From Active View")
        self.select_ref_active_button.clicked.connect(self.select_reference_from_active_view)

        self.select_ref_file_button = QPushButton("From File")
        self.select_ref_file_button.clicked.connect(self.select_reference_from_file)

        ref_layout.addWidget(self.ref_label)
        ref_layout.addWidget(self.ref_path_label)
        ref_layout.addWidget(self.select_ref_active_button)
        ref_layout.addWidget(self.select_ref_file_button)

        # File selection
        file_selection_layout = QHBoxLayout()
        self.add_files_button = QPushButton("Select Files")
        self.add_files_button.clicked.connect(self.select_files_to_align)
        self.add_directory_button = QPushButton("Select Directory")
        self.add_directory_button.clicked.connect(self.select_directory_to_align)
        file_selection_layout.addWidget(self.add_files_button)
        file_selection_layout.addWidget(self.add_directory_button)

        # Tree of files
        self.tree_widget = QTreeWidget()
        self.tree_widget.setColumnCount(1)
        self.tree_widget.setHeaderLabels(["Files to Align"])
        self.tree_widget.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)

        tree_buttons_layout = QHBoxLayout()
        self.remove_selected_button = QPushButton("Remove Selected")
        self.remove_selected_button.clicked.connect(self.remove_selected_files)
        self.clear_tree_button = QPushButton("Clear All")
        self.clear_tree_button.clicked.connect(self.clear_tree)
        tree_buttons_layout.addWidget(self.remove_selected_button)
        tree_buttons_layout.addWidget(self.clear_tree_button)

        # Output directory
        output_layout = QHBoxLayout()
        self.output_label = QLabel("Output Directory:")
        self.output_path_label = QLabel("No directory selected")
        self.output_path_label.setWordWrap(True)
        self.select_output_button = QPushButton("Select Output Folder")
        self.select_output_button.clicked.connect(self.select_output_directory)
        output_layout.addWidget(self.output_label)
        output_layout.addWidget(self.output_path_label)
        output_layout.addWidget(self.select_output_button)

        # Progress + start
        self.progress_label = QLabel("Status: Waiting…")
        self.progress_label.setStyleSheet("color: blue; font-weight: bold;")

        self.start_button = QPushButton("Start Registration")
        self.start_button.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.start_button.clicked.connect(self.start_registration)

        # Add to layout
        main_layout.addLayout(ref_layout)
        main_layout.addLayout(file_selection_layout)
        main_layout.addWidget(self.tree_widget)
        main_layout.addLayout(tree_buttons_layout)
        main_layout.addLayout(output_layout)
        main_layout.addWidget(self.progress_label)
        main_layout.addWidget(self.start_button)

    def _enqueue_progress(self, message: str) -> None:
        # Save only the latest message; start the coalescing timer
        self._pending_prog = str(message) if message is not None else ""
        if not self._prog_timer.isActive():
            self._prog_timer.start()

    def _flush_progress(self) -> None:
        if self._pending_prog is None:
            return
        # Hard non-reentrancy guard — don’t let update trigger itself
        if self._in_progress_slot:
            return
        self._in_progress_slot = True
        try:
            self.progress_label.setText(f"Status: {self._pending_prog}")
        finally:
            self._in_progress_slot = False
            self._pending_prog = None


    # Reference selection (no slots)
    def select_reference_from_active_view(self):
        self.reference_image_path = "__ACTIVE_VIEW__"
        self.ref_path_label.setText("(Active View)")

    def select_reference_from_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Reference Image", "",
            "Images (*.png *.jpg *.jpeg *.tif *.tiff *.fits *.fit *.xisf);;All Files (*)"
        )
        if file_path:
            self.reference_image_path = file_path
            self.ref_path_label.setText(os.path.basename(file_path))

    # File selection
    def select_files_to_align(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Files to Align", "",
            "Images (*.png *.jpg *.jpeg *.tif *.tiff *.fits *.fit *.xisf);;All Files (*)"
        )
        if files:
            for file in files:
                if file not in self.files_to_align:
                    self.files_to_align.append(file)
                    self.tree_widget.addTopLevelItem(QTreeWidgetItem([os.path.basename(file)]))

    def select_directory_to_align(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Directory", "")
        if directory:
            exts = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.fits', '.fit', '.xisf')
            new_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(exts)]
            for file in new_files:
                if file not in self.files_to_align:
                    self.files_to_align.append(file)
                    self.tree_widget.addTopLevelItem(QTreeWidgetItem([os.path.basename(file)]))

    # Manage tree
    def remove_selected_files(self):
        selected_items = self.tree_widget.selectedItems()
        for item in selected_items:
            file_name = item.text(0)
            for file_path in list(self.files_to_align):
                if os.path.basename(file_path) == file_name:
                    self.files_to_align.remove(file_path)
                    break
            index = self.tree_widget.indexOfTopLevelItem(item)
            self.tree_widget.takeTopLevelItem(index)

    def clear_tree(self):
        self.tree_widget.clear()
        self.files_to_align.clear()

    def select_output_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory", "")
        if directory:
            self.output_directory = directory
            self.output_path_label.setText(directory)

    # Start/finish
    def start_registration(self):
        if not self.reference_image_path:
            QMessageBox.warning(self, "Missing Reference", "Please select a reference image (file or active view).")
            return
        if not self.files_to_align:
            QMessageBox.warning(self, "No Files", "Please add files to align before starting.")
            return
        if not self.output_directory:
            QMessageBox.warning(self, "No Output Directory", "Please select an output directory before starting.")
            return

        self.progress_label.setText("Status: Running…")
        self.progress_label.setStyleSheet("color: green; font-weight: bold;")

        self.thread = StarRegistrationThread(
            self.reference_image_path,
            self.files_to_align,
            self.output_directory,
            parent_window=self.parent_window
        )
        self.thread.progress_update.connect(self._enqueue_progress)
        self.thread.registration_complete.connect(self.registration_finished)
        self.thread.start()

    def update_progress(self, message):
        self.progress_label.setText(f"Status: {message}")

    def registration_finished(self, success, message):
        color = "green" if success else "red"
        self.progress_label.setText(f"Status: {message}")
        self.progress_label.setStyleSheet(f"color: {color}; font-weight: bold;")
        if success:
            QMessageBox.information(self, "Registration Complete", message)
        else:
            QMessageBox.warning(self, "Registration Error", message)


# --------------------------------------------------
# MosaicMasterDialog with blending/normalization integrated
# --------------------------------------------------
_NUM_FLOAT = {
    "CRPIX1","CRPIX2","CRVAL1","CRVAL2","CDELT1","CDELT2",
    "CD1_1","CD1_2","CD2_1","CD2_2","CROTA1","CROTA2","EQUINOX"
}
_3RD_AXIS_PREFIXES = ("NAXIS3","CTYPE3","CUNIT3","CRVAL3","CRPIX3","CDELT3","CD3_","PC3_","PC_3")

def _coerce_num(val, tp=float):
    if isinstance(val, (int, float)): return tp(val)
    s = str(val).strip().strip("'").strip()
    m = re.match(r"^[+-]?\d+(\.\d*)?([eE][+-]?\d+)?", s)
    if m:
        return tp(float(m.group(0)))
    raise ValueError

def sanitize_wcs_header(hdr_in):
    """Return a cleaned astropy Header suitable for WCS(relax=True) with SIP kept."""
    if not hdr_in:
        return None
    hdr = Header(hdr_in) if not isinstance(hdr_in, Header) else hdr_in.copy()

    # Drop any lingering 3rd-axis WCS bits
    for k in list(hdr.keys()):
        if any(k.startswith(pref) for pref in _3RD_AXIS_PREFIXES):
            try: del hdr[k]
            except Exception: pass

    # Minimal, sane defaults
    if not hdr.get("CTYPE1"): hdr["CTYPE1"] = "RA---TAN"
    if not hdr.get("CTYPE2"): hdr["CTYPE2"] = "DEC--TAN"

    # RADECSYS -> RADESYS (modern key)
    if "RADESYS" not in hdr and "RADECSYS" in hdr:
        hdr["RADESYS"] = str(hdr["RADECSYS"]).strip()
        try: del hdr["RADECSYS"]
        except Exception: pass

    # Coerce common numeric keys to the right types
    for k in _NUM_FLOAT:
        if k in hdr:
            try: hdr[k] = _coerce_num(hdr[k], float)
            except Exception: pass

    # SIP orders: ensure ints + pair up A/B and AP/BP if one is missing
    for k in ("A_ORDER","B_ORDER","AP_ORDER","BP_ORDER"):
        if k in hdr:
            try: hdr[k] = _coerce_num(hdr[k], int)
            except Exception: del hdr[k]
    if "A_ORDER" in hdr and "B_ORDER" not in hdr: hdr["B_ORDER"] = hdr["A_ORDER"]
    if "B_ORDER" in hdr and "A_ORDER" not in hdr: hdr["A_ORDER"] = hdr["B_ORDER"]
    if "AP_ORDER" in hdr and "BP_ORDER" not in hdr: hdr["BP_ORDER"] = hdr["AP_ORDER"]
    if "BP_ORDER" in hdr and "AP_ORDER" not in hdr: hdr["AP_ORDER"] = hdr["BP_ORDER"]

    # Keep axes sane
    if "WCSAXES" in hdr:
        try: hdr["WCSAXES"] = _coerce_num(hdr["WCSAXES"], int)
        except Exception: del hdr["WCSAXES"]
    if "NAXIS" in hdr:
        try: hdr["NAXIS"] = _coerce_num(hdr["NAXIS"], int)
        except Exception: del hdr["NAXIS"]

    return hdr

def get_wcs_from_header(header):
    """Build a WCS while keeping SIP terms; suppress fix warnings; force 2D if needed."""
    if not header:
        return None
    hdr = sanitize_wcs_header(header)
    if hdr is None:
        return None

    naxis = 2 if hdr.get("NAXIS", 2) > 2 else None
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FITSFixedWarning)
        try:
            w = WCS(hdr, naxis=naxis, relax=True)  # relax=True keeps SIP/AP/BP
            return w if w.is_celestial else None
        except Exception:
            try:
                w = WCS(hdr, naxis=2, relax=True)
                return w if w.is_celestial else None
            except Exception:
                return None
    
def robust_api_request(method, url, data=None, files=None, prompt_on_failure=False):
    """
    Sends an API request without automatic retries. If the request fails (network error or invalid JSON response),
    prompts the user if they want to start completely over. If the user chooses to try again,
    the function calls itself recursively.
    """
    try:
        if method == "GET":
            response = requests.get(url, timeout=600)
        elif method == "POST":
            response = requests.post(url, data=data, files=files, timeout=600)
        else:
            raise ValueError("Unsupported request method: " + method)

        response.raise_for_status()  # Raise HTTP errors (e.g., 500, 404)

        try:
            return response.json()  # Attempt to parse JSON
        except json.JSONDecodeError:
            error_message = f"Invalid JSON response from {url}."
            print(error_message)
            if prompt_on_failure:
                user_choice = QMessageBox.question(
                    None,
                    "Invalid Response",
                    f"{error_message}\nDo you want to start over?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                if user_choice == QMessageBox.StandardButton.Yes:
                    return robust_api_request(method, url, data, files, prompt_on_failure=prompt_on_failure)
                else:
                    return None
            else:
                return None

    except requests.exceptions.RequestException as e:
        error_message = f"Network error when contacting {url}: {e}."
        print(error_message)
        if prompt_on_failure:
            user_choice = QMessageBox.question(
                None,
                "Network Error",
                f"{error_message}\nDo you want to start over?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if user_choice == QMessageBox.StandardButton.Yes:
                return robust_api_request(method, url, data, files, prompt_on_failure=prompt_on_failure)
            else:
                return None
        else:
            return None


def scale_image_for_display(image):
    """
    Scales a floating point image (0-1) to 8-bit (0-255) for display.
    """
    if np.max(image) == np.min(image):
        return np.zeros_like(image, dtype=np.uint8)  # Prevent division by zero
    scaled = (255 * (image - np.min(image)) / (np.max(image) - np.min(image))).astype(np.uint8)
    return scaled

def generate_minimal_fits_header(image):
    header = Header()
    header['SIMPLE'] = True
    # Set BITPIX according to the image’s data type.
    if np.issubdtype(image.dtype, np.integer):
        header['BITPIX'] = 16  # For 16-bit integer data.
    elif np.issubdtype(image.dtype, np.floating):
        header['BITPIX'] = -32  # For 32-bit float data.
    else:
        raise ValueError("Unsupported image data type for FITS header generation.")
    header['NAXIS'] = 2
    header['NAXIS1'] = image.shape[1]  # width
    header['NAXIS2'] = image.shape[0]  # height
    header['COMMENT'] = "Minimal header generated for blind solve"
    return header


class MosaicPreviewWindow(QDialog):
    def __init__(self, image_array, title="", parent=None, push_cb=None):
        super().__init__(parent)
        self.setWindowTitle(title if title else "Preview")
        self._push_cb = push_cb

        # Keep the original array around for re-stretch or reset
        self.original_array = image_array.copy()
        # Current displayed array (8-bit or whatever you want)
        self.image_array = image_array.copy()

        # Zoom state
        self.zoom_factor = 1.0

        # Variables for panning (dragging)
        self.dragging = False
        self.last_mouse_pos = QPoint()
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout(self)

        # 1) QScrollArea to enable scrollbars for large images
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        layout.addWidget(self.scroll_area)

        # 2) Label inside the scroll area
        self.preview_label = QLabel("No image yet.")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scroll_area.setWidget(self.preview_label)
        self.scroll_area.viewport().installEventFilter(self)

        # 3) Auto-Stretch Toggle
        self.stretch_toggle = QCheckBox("Auto-Stretch for Display")
        self.stretch_toggle.setChecked(True)
        self.stretch_toggle.stateChanged.connect(self.update_display)
        layout.addWidget(self.stretch_toggle)

        # 4) Button row (Zoom, Fit, etc.)
        button_layout = QHBoxLayout()

        self.zoom_in_button = QPushButton("Zoom In")
        self.zoom_in_button.clicked.connect(self.zoom_in)
        button_layout.addWidget(self.zoom_in_button)

        self.zoom_out_button = QPushButton("Zoom Out")
        self.zoom_out_button.clicked.connect(self.zoom_out)
        button_layout.addWidget(self.zoom_out_button)

        self.fit_button = QPushButton("Fit to Preview")
        self.fit_button.clicked.connect(self.fit_to_preview)
        button_layout.addWidget(self.fit_button)

        self.autostretch_button = QPushButton("Reapply Stretch")
        self.autostretch_button.clicked.connect(self.autostretch_image)
        button_layout.addWidget(self.autostretch_button)

        self.push_btn = QPushButton("Push to New View")
        self.push_btn.clicked.connect(lambda: self._push_cb() if callable(self._push_cb) else None)
        button_layout.addWidget(self.push_btn)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        button_layout.addWidget(close_btn)

        layout.addLayout(button_layout)

        self.setLayout(layout)
        # Finally, display the initial image
        self.display_image(self.image_array)

    def display_image(self, arr):
        """
        Convert array to QPixmap and display in preview_label.
        We'll respect self.zoom_factor to scale the pixmap.
        """
        if arr is None or arr.size == 0:
            print("WARNING: Trying to display an empty image.")
            return

        # Possibly apply auto-stretch
        if self.stretch_toggle.isChecked():
            arr_display = self.stretch_for_display(arr)
        else:
            # If it's already 8-bit or float, just convert to 8-bit safely
            arr_display = self.to_8bit(arr)
        
        # Convert single-channel => 3 channels if needed
        if arr_display.ndim == 2:
            arr_3ch = np.stack([arr_display]*3, axis=-1)
        elif arr_display.ndim == 3 and arr_display.shape[2] == 1:
            arr_3ch = np.concatenate([arr_display, arr_display, arr_display], axis=2)
        else:
            arr_3ch = arr_display

        # Make QImage => QPixmap
        h, w, c = arr_3ch.shape
        bytes_per_line = w * c
        qimg = QImage(arr_3ch.tobytes(), w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        # Apply zoom factor
        new_w = int(w * self.zoom_factor)
        new_h = int(h * self.zoom_factor)
        if new_w < 1: new_w = 1
        if new_h < 1: new_h = 1

        scaled_pixmap = pixmap.scaled(
            new_w, new_h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

        # Set the label to the scaled pixmap
        self.preview_label.setPixmap(scaled_pixmap)
        # Important: set the label size so the scroll area can scroll if it's bigger
        self.preview_label.resize(scaled_pixmap.size())

    def stretch_for_display(self, arr):
        """
        Applies an auto-stretch to improve visualization:
          1) Compute 0.5 and 99.5 percentiles
          2) Rescale to [0..255]
        """
        arr = arr.astype(np.float32, copy=False)
        mn, mx = np.percentile(arr, (0.5, 99.5))
        if mx > mn:
            arr = (arr - mn) / (mx - mn)
        else:
            arr = np.zeros_like(arr)
        arr = (arr * 255).clip(0, 255).astype(np.uint8)
        return arr

    def eventFilter(self, source, event):
        """
        Capture mouse events on the scroll_area.viewport():
          - Left-button press => start dragging
          - Mouse move => if dragging, pan
          - Left-button release => stop dragging
          - Wheel => zoom in/out
        """
        if source == self.scroll_area.viewport():
            if event.type() == QEvent.Type.MouseButtonPress:
                if event.button() == Qt.MouseButton.LeftButton:
                    self.dragging = True
                    self.last_mouse_pos = event.pos()
                    return True  # We handled it
            elif event.type() == QEvent.Type.MouseMove:
                if self.dragging:
                    # Compute how far we moved
                    delta = event.pos() - self.last_mouse_pos
                    self.last_mouse_pos = event.pos()

                    # Adjust scrollbars
                    h_bar = self.scroll_area.horizontalScrollBar()
                    v_bar = self.scroll_area.verticalScrollBar()
                    h_bar.setValue(h_bar.value() - delta.x())
                    v_bar.setValue(v_bar.value() - delta.y())

                    return True
            elif event.type() == QEvent.Type.MouseButtonRelease:
                if event.button() == Qt.MouseButton.LeftButton:
                    self.dragging = False
                    return True
            elif event.type() == QEvent.Type.Wheel:
                # Zoom in or out
                if event.angleDelta().y() > 0:
                    self.zoom_in()
                else:
                    self.zoom_out()
                event.accept()
                return True
        return super().eventFilter(source, event)

    def to_8bit(self, arr):
        """
        Simple fallback if not using auto-stretch:
        - If float in [0..1], multiply by 255
        - If already 8-bit, do nothing
        """
        if arr.dtype == np.uint8:
            return arr
        # else assume float [0..1]
        return (arr*255).clip(0,255).astype(np.uint8)

    def update_display(self):
        """
        Called when toggling the stretch checkbox. Re-display the image.
        """
        self.display_image(self.image_array)

    def autostretch_image(self):
        """
        Auto-stretch the original image and update preview.
        If the image has multiple channels, we do the same approach as stretch_for_display
        but typically you'd do something more advanced for color.
        """
        arr = self.original_array.copy()
        # e.g., if color, you might do a channel-by-channel approach
        # For simplicity, let's do a grayscale approach using the mean:
        if arr.ndim == 3:
            # we create a single channel for stretch
            g = np.mean(arr, axis=-1)
            arr_stretched = self.stretch_for_display(g)
            # Then broadcast back to 3 channels
            arr_stretched = np.stack([arr_stretched]*3, axis=-1)
        else:
            arr_stretched = self.stretch_for_display(arr)
        self.image_array = arr_stretched
        self.display_image(self.image_array)

    # ---------------------
    # ZOOM Methods
    # ---------------------
    
    def zoom_in(self):
        self.zoom_factor *= 1.2
        self.display_image(self.image_array)

    
    def zoom_out(self):
        self.zoom_factor /= 1.2
        if self.zoom_factor < 0.05:
            self.zoom_factor = 0.05
        self.display_image(self.image_array)

    def fit_to_preview(self):
        """
        Scale the image so it fits inside the scroll_area's viewport.
        We'll measure the image's actual size, compare to the viewport,
        and adjust zoom_factor accordingly.
        """
        if self.image_array is None or self.image_array.size == 0:
            return

        # We'll figure out the image's *unzoomed* dimensions
        arr_display = self.image_array
        if self.stretch_toggle.isChecked():
            arr_display = self.stretch_for_display(arr_display)
        else:
            arr_display = self.to_8bit(arr_display)

        # Convert single-channel => 3 channels if needed, to find w,h
        if arr_display.ndim == 2:
            arr_3ch = np.stack([arr_display]*3, axis=-1)
        elif arr_display.ndim == 3 and arr_display.shape[2] == 1:
            arr_3ch = np.concatenate([arr_display, arr_display, arr_display], axis=2)
        else:
            arr_3ch = arr_display

        h, w, c = arr_3ch.shape

        # The scroll area viewport size
        viewport_size = self.scroll_area.viewport().size()
        vw, vh = viewport_size.width(), viewport_size.height()

        # Compute the scale factor to fit image inside viewport
        scale_w = vw / w if w else 1.0
        scale_h = vh / h if h else 1.0
        new_zoom = min(scale_w, scale_h)
        if new_zoom <= 0:
            new_zoom = 0.01

        self.zoom_factor = new_zoom
        self.display_image(self.image_array)

    def resizeEvent(self, event):
        """
        Refresh displayed pixmap when window is resized,
        only if we want the displayed image to keep fitting automatically.
        But typically, we won't auto-fit on window resize if user is controlling zoom.
        """
        super().resizeEvent(event)
        # Optionally do:
        # self.fit_to_preview()
        # or if you want to keep the user-chosen zoom, just re-display:
        self.display_image(self.image_array)

class MosaicSettingsDialog(QDialog):
    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.setWindowTitle("Mosaic Master Settings")
        self.initUI()

    def initUI(self):
        layout = QFormLayout(self)

        # Number of Stars to Attempt to Use
        self.starCountSpin = CustomSpinBox(minimum=1, maximum=1000,
                                        initial=self.settings.value("mosaic/num_stars", 150, type=int),
                                        step=1)
        layout.addRow("Number of Stars:", self.starCountSpin)

        # Translation Max Tolerance
        self.transTolSpin = CustomDoubleSpinBox(minimum=0.0, maximum=10.0,
                                                initial=self.settings.value("mosaic/translation_max_tolerance", 3.0, type=float),
                                                step=0.1)
        layout.addRow("Translation Max Tolerance:", self.transTolSpin)

        # Scale Min Tolerance
        self.scaleMinSpin = CustomDoubleSpinBox(minimum=0.0, maximum=10.0,
                                                initial=self.settings.value("mosaic/scale_min_tolerance", 0.8, type=float),
                                                step=0.1)
        layout.addRow("Scale Min Tolerance:", self.scaleMinSpin)

        # Scale Max Tolerance
        self.scaleMaxSpin = CustomDoubleSpinBox(minimum=0.0, maximum=10.0,
                                                initial=self.settings.value("mosaic/scale_max_tolerance", 1.25, type=float),
                                                step=0.1)
        layout.addRow("Scale Max Tolerance:", self.scaleMaxSpin)

        # Rotation Max Tolerance
        self.rotationMaxSpin = CustomDoubleSpinBox(minimum=0.0, maximum=180.0,
                                                initial=self.settings.value("mosaic/rotation_max_tolerance", 45.0, type=float),
                                                step=0.1)
        # Force two decimals in display
        self.rotationMaxSpin.lineEdit.setText(f"{self.rotationMaxSpin.value():.2f}")
        layout.addRow("Rotation Max Tolerance (°):", self.rotationMaxSpin)

        # Skew Max Tolerance
        self.skewMaxSpin = CustomDoubleSpinBox(minimum=0.0, maximum=1.0,
                                            initial=self.settings.value("mosaic/skew_max_tolerance", 0.1, type=float),
                                            step=0.01)
        layout.addRow("Skew Max Tolerance:", self.skewMaxSpin)

        # FWHM for Star Detection
        self.fwhmSpin = CustomDoubleSpinBox(minimum=0.0, maximum=20.0,
                                            initial=self.settings.value("mosaic/star_fwhm", 3.0, type=float),
                                            step=0.1)
        self.fwhmSpin.lineEdit.setText(f"{self.fwhmSpin.value():.2f}")
        layout.addRow("FWHM for Star Detection:", self.fwhmSpin)

        # Sigma for Star Detection
        self.sigmaSpin = CustomDoubleSpinBox(minimum=0.0, maximum=10.0,
                                            initial=self.settings.value("mosaic/star_sigma", 3.0, type=float),
                                            step=0.1)
        self.sigmaSpin.lineEdit.setText(f"{self.sigmaSpin.value():.2f}")
        layout.addRow("Sigma for Star Detection:", self.sigmaSpin)

        # Polynomial Degree
        self.polyDegreeSpin = CustomSpinBox(minimum=1, maximum=6,
                                            initial=self.settings.value("mosaic/poly_degree", 3, type=int),
                                            step=1)
        layout.addRow("Polynomial Degree:", self.polyDegreeSpin)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def accept(self):
        # Save the values to QSettings
        self.settings.setValue("mosaic/num_stars", self.starCountSpin.value)
        self.settings.setValue("mosaic/translation_max_tolerance", self.transTolSpin.value())
        self.settings.setValue("mosaic/scale_min_tolerance", self.scaleMinSpin.value())
        self.settings.setValue("mosaic/scale_max_tolerance", self.scaleMaxSpin.value())
        self.settings.setValue("mosaic/rotation_max_tolerance", self.rotationMaxSpin.value())
        self.settings.setValue("mosaic/skew_max_tolerance", self.skewMaxSpin.value())
        self.settings.setValue("mosaic/star_fwhm", self.fwhmSpin.value())
        self.settings.setValue("mosaic/star_sigma", self.sigmaSpin.value())
        self.settings.setValue("mosaic/poly_degree", self.polyDegreeSpin.value)
        super().accept()

class PolyGradientRemoval:
    """
    A headless class that replicates the polynomial background removal
    logic from GradientRemovalDialog, minus the RBF step and UI code.

    Flow:
      1) Stretch the image (unlinked linear stretch).
      2) Downsample.
      3) Build an exclusion mask that:
         - Skips zero-valued pixels in any channel.
         - Optionally skip user-specified mask areas if desired (can pass mask to process()).
      4) Generate sample points from corners, borders, quartiles, do gradient_descent_to_dim_spot, skip bright areas.
      5) Fit a polynomial background and subtract it.
      6) Re-normalize median, clip to [0..1].
      7) Unstretch the final image back to the original domain.
    """

    def __init__(
        self,
        image: np.ndarray,
        poly_degree: int = 2,
        downsample_scale: int = 5,
        num_sample_points: int = 100
    ):
        """
        Args:
            image (np.ndarray): Input image in [0..1], shape (H,W) or (H,W,3), float32 recommended.
            poly_degree (int): Polynomial degree (1=linear,2=quadratic).
            downsample_scale (int): Factor for area downsampling.
            num_sample_points (int): Number of sample points to generate.
        """
        self.image = image.copy()
        self.poly_degree = poly_degree
        self.downsample_scale = downsample_scale
        self.num_sample_points = num_sample_points

        # For the stretch/unstretch logic
        self.stretch_original_mins = []
        self.stretch_original_medians = []
        self.was_single_channel = False

    def process(self, user_exclusion_mask: np.ndarray = None) -> np.ndarray:
        # 1) Stretch
        stretched = self.pixel_math_stretch(self.image)

        # 2) Downsample
        small_stretched = self.downsample_image(stretched, self.downsample_scale)
        h_s, w_s = small_stretched.shape[:2]

        # --- NEW: downsample user mask (≥0.5 keeps) ---
        mask_small = None
        if user_exclusion_mask is not None:
            m = user_exclusion_mask.astype(np.float32)
            mask_small = cv2.resize(m, (w_s, h_s), interpolation=cv2.INTER_AREA) >= 0.5

        # 4) Generate sample points using ABE’s sampler (fallback if missing)
        sample_points = self._gen_sample_points(
            small_stretched,
            num_points=self.num_sample_points,
            exclusion_mask=mask_small,
            patch_size=15,  # or make this configurable
        )

        # 5) Fit polynomial on the downsampled image
        poly_background_small = self.fit_polynomial_gradient(
            small_stretched, sample_points, degree=self.poly_degree
        )

        # Upscale background to full size
        poly_background = self.upscale_background(
            poly_background_small, stretched.shape[:2]
        )

        # Subtract
        after_poly = stretched - poly_background

        # Re-normalize median to original
        original_median = float(np.median(stretched))
        after_poly = self.normalize_image(after_poly, original_median)

        # Clip
        after_poly = np.clip(after_poly, 0, 1)

        # 6) Unstretch
        corrected = self.unstretch_image(after_poly)
        return corrected

    # --- NEW helper: delegate to ABE sampler, with a robust fallback ---
    def _gen_sample_points(
        self,
        small_image: np.ndarray,
        num_points: int,
        exclusion_mask: np.ndarray | None,
        patch_size: int = 15,
    ) -> np.ndarray:
        if abe_generate_sample_points is not None:
            return abe_generate_sample_points(
                small_image, num_points=num_points,
                exclusion_mask=exclusion_mask, patch_size=patch_size
            )

        # Fallback: simple grid (still respects exclusion_mask)
        H, W = small_image.shape[:2]
        grid = max(3, int(np.sqrt(max(9, num_points))))
        xs = np.linspace(10, max(11, W - 11), grid, dtype=int)
        ys = np.linspace(10, max(11, H - 11), grid, dtype=int)
        pts = []
        for y in ys:
            for x in xs:
                if exclusion_mask is not None and not exclusion_mask[y, x]:
                    continue
                pts.append((x, y))
        if not pts:
            pts = [(W // 2, H // 2)]
        return np.asarray(pts, dtype=np.int32)

    # ---------------------------------------------------------------
    # Helper: Stretch / Unstretch
    # ---------------------------------------------------------------
    def pixel_math_stretch(self, image: np.ndarray) -> np.ndarray:
        """
        Unlinked linear stretch using your existing Numba functions.

        Steps:
        1) If single-channel, replicate to 3-ch so we can store stats & do consistent logic.
        2) For each channel c: subtract the channel's min => data is >= 0.
        3) Compute the median after min subtraction for that channel.
        4) Call the appropriate Numba function:
            - If single-channel (was originally 1-ch), call numba_mono_final_formula
            on the 1-ch array.
            - If 3-ch color, call numba_color_final_formula_unlinked.
        5) Clip to [0,1].
        6) Store self.stretch_original_mins / medians so we can unstretch later.
        """
        target_median = 0.25

        # 1) Handle single-channel => replicate to 3 channels
        if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
            self.was_single_channel = True
            image_3ch = np.stack([image.squeeze()] * 3, axis=-1)
        else:
            self.was_single_channel = False
            image_3ch = image

        image_3ch = image_3ch.astype(np.float32, copy=True)

        H, W, C = image_3ch.shape
        # We assume C=3 now.

        self.stretch_original_mins = []
        self.stretch_original_medians = []

        # 2) Subtract min per channel
        for c in range(C):
            cmin = image_3ch[..., c].min()
            image_3ch[..., c] -= cmin
            self.stretch_original_mins.append(float(cmin))

        # 3) Compute median after min subtraction
        medians_after_sub = []
        for c in range(C):
            cmed = float(np.median(image_3ch[..., c]))
            medians_after_sub.append(cmed)
        self.stretch_original_medians = medians_after_sub

        # 4) Apply the final formula with your Numba functions
        if self.was_single_channel:
            # If originally single-channel, let's do a single pass with numba_mono_final_formula
            # on the single channel. We can do that by extracting one channel from image_3ch.
            # Then replicate the result to 3 channels, or keep it as 1-ch?
            # Typically we keep it as 1-ch in the end, so let's do that.

            # We'll just pick channel 0, run the mono formula, store it back in a 2D array.
            mono_array = image_3ch[..., 0]  # shape (H,W)
            cmed = medians_after_sub[0]     # The median for that channel
            # We call the numba function
            stretched_mono = numba_mono_final_formula(mono_array, cmed, target_median)

            # Now place it back into image_3ch for consistency
            for c in range(3):
                image_3ch[..., c] = stretched_mono
        else:
            # 3-channel unlinked
            medians_rescaled = np.array(medians_after_sub, dtype=np.float32)
            # 'image_3ch' is our 'rescaled'
            stretched_3ch = numba_color_final_formula_unlinked(
                image_3ch, medians_rescaled, target_median
            )
            image_3ch = stretched_3ch

        # 5) Clip to [0..1]
        np.clip(image_3ch, 0.0, 1.0, out=image_3ch)
        image = image_3ch
        return image


    def unstretch_image(self, image: np.ndarray) -> np.ndarray:
        """
        Calls the Numba-optimized unstretch function.
        """
        image = image.astype(np.float32, copy=True)

        # Convert lists to NumPy arrays for efficient Numba processing
        stretch_original_medians = np.array(self.stretch_original_medians, dtype=np.float32)
        stretch_original_mins = np.array(self.stretch_original_mins, dtype=np.float32)

        # Call the Numba function
        unstretched = numba_unstretch(image, stretch_original_medians, stretch_original_mins)

        if self.was_single_channel:
            # Convert back to grayscale
            unstretched = np.mean(unstretched, axis=2, keepdims=True)

        return unstretched


    # ---------------------------------------------------------------
    # Helper: Downsample
    # ---------------------------------------------------------------
    def downsample_image(self, image: np.ndarray, scale: int=6) -> np.ndarray:
        """
        Downsamples with area interpolation.
        """
        h, w = image.shape[:2]
        new_w = max(1, w//scale)
        new_h = max(1, h//scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)



    # ---------------------------------------------------------------
    # 5) Fit Polynomial
    # ---------------------------------------------------------------
    def fit_polynomial_gradient(self, image: np.ndarray, sample_points: np.ndarray, degree: int = 2, patch_size: int = 15) -> np.ndarray:
        """
        Optimized polynomial background fitting.
        - Extracts sample points using vectorized NumPy median calculations.
        - Solves for polynomial coefficients in parallel.
        - Precomputes polynomial basis terms for efficiency.
        """

        H, W = image.shape[:2]
        half_patch = patch_size // 2
        num_samples = len(sample_points)

        # Convert sample points to NumPy arrays
        sample_points = np.array(sample_points, dtype=np.int32)
        x_coords, y_coords = sample_points[:, 0], sample_points[:, 1]

        # Precompute polynomial design matrix
        A = build_poly_terms(x_coords, y_coords, degree)

        # Extract sample values efficiently
        if image.ndim == 3 and image.shape[2] == 3:
            # Color image
            background = np.zeros_like(image, dtype=np.float32)
            for c in range(3):
                # Extract patches and compute medians using vectorized NumPy operations
                z_vals = np.array([
                    np.median(image[max(0, y-half_patch):min(H, y+half_patch+1),
                                    max(0, x-half_patch):min(W, x+half_patch+1), c])
                    for x, y in zip(x_coords, y_coords)
                ], dtype=np.float32)

                # Solve for polynomial coefficients
                coeffs = np.linalg.lstsq(A, z_vals, rcond=None)[0]

                # Generate full polynomial background
                background[..., c] = evaluate_polynomial(H, W, coeffs, degree)

        else:
            # Grayscale image
            background = np.zeros((H, W), dtype=np.float32)

            z_vals = np.array([
                np.median(image[max(0, y-half_patch):min(H, y+half_patch+1),
                                max(0, x-half_patch):min(W, x+half_patch+1)])
                for x, y in zip(x_coords, y_coords)
            ], dtype=np.float32)

            # Solve for polynomial coefficients
            coeffs = np.linalg.lstsq(A, z_vals, rcond=None)[0]

            # Generate full polynomial background
            background = evaluate_polynomial(H, W, coeffs, degree)

        return background
    # ---------------------------------------------------------------
    # 6) Upscale
    # ---------------------------------------------------------------
    def upscale_background(self, background: np.ndarray, out_shape: tuple) -> np.ndarray:
        """
        Resizes 'background' to out_shape=(H,W) using OpenCV interpolation.
        """
        oh, ow = out_shape

        if background.ndim == 3 and background.shape[2] == 3:
            # Resizing each channel efficiently without looping in Python
            return np.stack([cv2.resize(background[..., c], (ow, oh), interpolation=cv2.INTER_LANCZOS4)
                            for c in range(3)], axis=-1)
        else:
            return cv2.resize(background, (ow, oh), interpolation=cv2.INTER_LANCZOS4).astype(np.float32)
    # ---------------------------------------------------------------
    # 7) Normalize
    # ---------------------------------------------------------------
    def normalize_image(self, image: np.ndarray, target_median: float) -> np.ndarray:
        """
        Shift image so its median matches target_median.
        """
        cmed = np.median(image)
        diff = target_median - cmed
        return image + diff

settings = QSettings("SetiAstro", "Seti Astro Suite Pro")

def save_api_key(api_key):
    settings.setValue("astrometry_api_key", api_key)  # Save to QSettings
    print("API key saved.")

def load_api_key():
    api_key = settings.value("astrometry_api_key", "")  # Load from QSettings
    if api_key:
        print("API key loaded.")
    return api_key




class MosaicMasterDialog(QDialog):
    def __init__(self, settings: QSettings, parent=None, image_manager=None,
                 doc_manager=None, wrench_path=None, spinner_path=None,
                 list_open_docs_fn=None):                                 # ← add param
        super().__init__(parent)
        self.settings = settings
        self.image_manager = image_manager
        self._docman = doc_manager or getattr(parent, "doc_manager", None)
        # same pattern as StellarAlignmentDialog
        if list_open_docs_fn is None:
            cand = getattr(parent, "_list_open_docs", None)
            self._list_open_docs_fn = cand if callable(cand) else None
        else:
            self._list_open_docs_fn = list_open_docs_fn

        self.setWindowTitle("Mosaic Master")
        self.wrench_path = wrench_path
        self.spinner_path = spinner_path

        self.resize(600, 400)
        self.loaded_images = []  
        self.final_mosaic = None
        self.weight_mosaic = None
        self.wcs_metadata = None  # To store mosaic WCS header
        self.astap_exe = self.settings.value("paths/astap", "", type=str)
        # Variables to store stretching parameters:
        self.stretch_original_mins = []
        self.stretch_original_medians = []
        self.was_single_channel = False
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout(self)

        instructions = QLabel(
            "Mosaic Master:\n"
            "1) Add images - Highly Recommend Images be Linear FITS\n"
            "2) Choose Transformation Type:\n"
            "....Partial Affine - Great for Images with Translation, Rotation, and Scaling Needs\n"
            "....Affine - Great for Images that also have skew distortions\n"
            "....Homography - Great for Images that also have lens or perspective distortion\n"
            "....Polynomial Warp - Useful in large mosaics to bend the images together\n"
            "3) Align & Create Mosaic\n"
            "4) Save to Image Manager"
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        btn_layout = QHBoxLayout()
        # Button to add image from disk
        add_btn = QPushButton("Add Image")
        add_btn.clicked.connect(self.add_image)
        btn_layout.addWidget(add_btn)

        # New button to add an image from one of the ImageManager slots
        add_from_view_btn = QPushButton("Add from View")
        add_from_view_btn.setToolTip("Add an image from any open View")
        add_from_view_btn.clicked.connect(self.add_image_from_view)
        btn_layout.addWidget(add_from_view_btn)

        remove_btn = QPushButton("Remove Selected")
        remove_btn.clicked.connect(self.remove_selected)
        btn_layout.addWidget(remove_btn)

        preview_btn = QPushButton("Preview Selected")
        preview_btn.clicked.connect(self.preview_selected)
        btn_layout.addWidget(preview_btn)

        align_btn = QPushButton("Align and Create Mosaic")
        align_btn.clicked.connect(self.align_images)
        btn_layout.addWidget(align_btn)

        save_btn = QPushButton("Save to New View")
        save_btn.clicked.connect(self.save_mosaic_to_new_view)
        btn_layout.addWidget(save_btn)

        # Add the wrench button for settings.
        wrench_btn = QPushButton()
        wrench_btn.setIcon(QIcon(self.wrench_path))
        wrench_btn.setToolTip("Mosaic Settings")
        wrench_btn.clicked.connect(self.openSettings)
        btn_layout.addWidget(wrench_btn)

        layout.addLayout(btn_layout)

        # Horizontal sizer for checkboxes.
        checkbox_layout = QHBoxLayout()
        self.forceBlindCheckBox = QCheckBox("Force Blind Solve (ignore existing WCS)")
        checkbox_layout.addWidget(self.forceBlindCheckBox)
        # New Seestar Mode checkbox:
        self.seestarCheckBox = QCheckBox("Seestar Mode")
        self.seestarCheckBox.setToolTip("Wwen enabled, images are aligned iteratively using astroalign without plate solving.")
        checkbox_layout.addWidget(self.seestarCheckBox)
        layout.addLayout(checkbox_layout)

        self.transform_combo = QComboBox()
        self.transform_combo.addItems([
            "Partial Affine Transform",
            "Affine Transform",
            "Homography Transform",
            "Polynomial Warp Based Transform"
        ])
        # Set the default selection to "Affine Transform" (index 1)
        self.transform_combo.setCurrentIndex(1)
        layout.addWidget(QLabel("Select Transformation Method:"))
        layout.addWidget(self.transform_combo)

        self.images_list = QListWidget()
        self.images_list.setSelectionMode(self.images_list.SelectionMode.SingleSelection)
        layout.addWidget(self.images_list)

        self.status_label = QLabel("Status: no images")
        layout.addWidget(self.status_label)

        self.spinnerLabel = QLabel(self)
        self.spinnerLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.spinnerMovie = QMovie(self.spinner_path)
        self.spinnerLabel.setMovie(self.spinnerMovie)
        self.spinnerLabel.hide() 
        layout.addWidget(self.spinnerLabel)

        self.setLayout(layout)


    def _get_astrometry_api_key(self) -> str:
        # Prefer QSettings; fall back to your legacy loader if present.
        key = self.settings.value("api/astrometry_key", "", type=str)
        if key:
            return key
        try:
            return load_api_key() or ""
        except Exception:
            return ""

    def _is_view_path(self, p) -> bool:
        return isinstance(p, str) and p.startswith("view://")

    def _push_mosaic_to_new_doc(self):
        if self.final_mosaic is None:
            QMessageBox.information(self, "Mosaic Master", "No mosaic available to push.")
            return
        img = self.final_mosaic.astype(np.float32, copy=False)
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        meta = dict(self.wcs_metadata or {})

        dm = self._docman
        if dm is not None:
            newdoc = dm.open_array(img, metadata=meta, title="Mosaic")
            if hasattr(self.parent(), "_spawn_subwindow_for"):
                self.parent()._spawn_subwindow_for(newdoc)
            QMessageBox.information(self, "Mosaic Master", "Pushed to new view.")
        else:
            # last-resort fallback if only legacy image_manager exists
            if self.image_manager and hasattr(self.image_manager, "create_document"):
                self.image_manager.create_document(image=img, metadata=meta)
                QMessageBox.information(self, "Mosaic Master", "Pushed via image manager.")

    # ---- view helpers (same spirit as StellarAlignmentDialog) ----
    def _safe_call(self, maybe_callable):
        try:
            return maybe_callable() if callable(maybe_callable) else maybe_callable
        except Exception:
            return None

    def _title_for_doc(self, doc, preferred=None):
        # If we were given a candidate title and it's usable, keep it.
        if isinstance(preferred, str) and preferred.strip():
            return preferred.strip()

        # Try common “nice” sources (call methods when present).
        for attr in ("display_name", "windowTitle", "objectName", "title", "name"):
            val = self._safe_call(getattr(doc, attr, None))
            if isinstance(val, str) and val.strip():
                return val.strip()

        # Try metadata fallbacks.
        meta = getattr(doc, "metadata", None)
        if isinstance(meta, dict):
            for k in ("display_name", "title", "name"):
                val = meta.get(k)
                if isinstance(val, str) and val.strip():
                    return val.strip()

        # Last resort.
        return f"Untitled View {id(doc)}"

    def _iter_docs(self):
        """
        Returns a list of (title:str, doc:any) pairs.
        Accepts helpers that yield either bare docs or (title, doc) tuples.
        """
        # Prefer host helper
        items = []
        if self._list_open_docs_fn:
            try:
                raw = list(self._list_open_docs_fn()) or []
            except Exception as e:
                print("[Mosaic] _list_open_docs_fn failed:", e)
                raw = []
            for it in raw:
                if isinstance(it, (tuple, list)) and len(it) >= 2:
                    t, d = it[0], it[1]
                    items.append((self._title_for_doc(d, preferred=t), d))
                else:
                    d = it
                    items.append((self._title_for_doc(d), d))

        if items:
            return items

        # Fallbacks from the doc manager
        docs = []
        try:
            if self._docman and hasattr(self._docman, "iter_documents") and callable(self._docman.iter_documents):
                docs = list(self._docman.iter_documents())
            elif self._docman and hasattr(self._docman, "documents"):
                docs = list(self._docman.documents)
        except Exception:
            docs = []

        return [(self._title_for_doc(d), d) for d in docs]


    # --- title helpers -------------------------------------------------
    def _callmaybe(self, obj, attr):
        """Return obj.attr() if callable, else obj.attr; else None."""
        try:
            v = getattr(obj, attr, None)
            return v() if callable(v) else v
        except Exception:
            return None

    def _best_title_from_obj(self, o):
        """Try common title/name providers on a QWidget/ImageDocument."""
        if o is None:
            return None
        # 1) display_name (method or attr)
        t = self._callmaybe(o, "display_name")
        if isinstance(t, str) and t.strip():
            return t
        # 2) windowTitle / objectName (Qt)
        for a in ("windowTitle", "objectName", "title", "name"):
            t = self._callmaybe(o, a)
            if isinstance(t, str) and t.strip():
                return t
        # 3) metadata.display_name
        try:
            md = getattr(o, "metadata", {}) or {}
            t = md.get("display_name")
            if isinstance(t, str) and t.strip():
                return t
        except Exception:
            pass
        return None

    def _resolve_view_title(self, view, doc, title_hint=None):
        """Prefer the host-provided title, then view, then doc, then a safe default."""
        # host-provided title from _list_open_docs()
        if isinstance(title_hint, str) and title_hint.strip():
            return title_hint.strip()
        # view wins over doc
        t = self._best_title_from_obj(view) or self._best_title_from_obj(doc)
        return t if t else "Untitled View"


    def _pick_view_dialog(self):
        items = self._iter_docs()
        if not items:
            QMessageBox.information(self, "Add from View", "No open views found.")
            return None

        dlg = QDialog(self)
        dlg.setWindowTitle("Select View")
        v = QVBoxLayout(dlg)
        v.addWidget(QLabel("Choose a view to add:"))

        combo = QComboBox()
        for (_, doc) in items:
            title = self._fmt_doc_title(doc)
            # append dims if available
            try:
                img = _doc_image(doc)  # same utility used by Stellar Alignment
                if img is not None:
                    h, w = img.shape[:2]
                    title = f"{title} — {w}×{h}"
            except Exception:
                pass
            combo.addItem(title, userData=doc)
        v.addWidget(combo)

        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                            QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        v.addWidget(bb)

        return combo.currentData() if dlg.exec() else None


    def openSettings(self):
        dlg = MosaicSettingsDialog(self.settings, self)
        if dlg.exec():
            self.status_label.setText("Mosaic settings updated.")

    # ---------- Add / Remove ----------
    def add_image(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Add Image(s)",
            "",
            "Images (*.png *.jpg *.jpeg *.tif *.tiff *.fits *.fit *.fz *.fz *.xisf *.cr2 *.nef *.arw *.dng *.raf *.orf *.rw2 *.pef)"
        )
        if paths:
            for path in paths:
                arr, header, bitdepth, ismono = load_image(path)
                header = sanitize_wcs_header(header) if header else None
                wcs_obj = get_wcs_from_header(header) if header else None
                d = {
                    "path": path,
                    "image": arr,
                    "header": header,
                    "wcs": wcs_obj,
                    "bit_depth": bitdepth,
                    "is_mono": ismono,
                    "transform": None
                }
                self.loaded_images.append(d)

                text = os.path.basename(path)
                if wcs_obj is not None:
                    text += " [WCS]"
                item = QListWidgetItem(text)
                item.setToolTip(path)
                self.images_list.addItem(item)
            self.update_status()

    def add_image_from_view(self):
        items = self._iter_docs()
        if not items:
            QMessageBox.information(self, "Add from View", "No open views found.")
            return

        candidates = []
        for title, doc in items:
            # image
            img = None
            try:
                if hasattr(doc, "get_image") and callable(doc.get_image):
                    img = doc.get_image()
                else:
                    img = getattr(doc, "image", None)
            except Exception:
                img = None
            if not isinstance(img, np.ndarray):
                continue

            # metadata/header
            meta = {}
            try:
                if hasattr(doc, "get_metadata") and callable(doc.get_metadata):
                    meta = doc.get_metadata() or {}
                else:
                    meta = getattr(doc, "metadata", {}) or {}
            except Exception:
                meta = {}

            header = (meta.get("original_header")
                    or meta.get("header")
                    or meta.get("fits_header"))
            header = sanitize_wcs_header(header) if header else None
            wcs_obj = get_wcs_from_header(header) if header else None

            h, w = img.shape[:2]
            label = f"{title} — {w}×{h}"
            key = f"view://{id(doc)}"  # stable pseudo-path for removal

            candidates.append((label, img, meta, key, header, wcs_obj))

        if not candidates:
            QMessageBox.information(self, "Add from View", "No views with image data are open.")
            return

        labels = [c[0] for c in candidates]
        choice, ok = QInputDialog.getItem(self, "Select View", "Choose a view to add:", labels, 0, False)
        if not ok or not choice:
            return

        label, image, metadata, path_key, header, wcs_obj = candidates[labels.index(choice)]

        d = {
            "path": path_key,
            "image": image,
            "header": header,
            "wcs": wcs_obj,
            "bit_depth": (metadata.get("bit_depth") if isinstance(metadata, dict) else None),
            "is_mono": (metadata.get("is_mono", image.ndim == 2) if isinstance(metadata, dict) else (image.ndim == 2)),
            "transform": None,
        }
        self.loaded_images.append(d)

        txt = label + (" [WCS]" if wcs_obj is not None else "")
        item = QListWidgetItem(txt)
        item.setToolTip(path_key)
        self.images_list.addItem(item)
        self.update_status()


    def remove_selected(self):
        s = self.images_list.selectedItems()
        if not s:
            QMessageBox.information(self, "Remove", "No item selected.")
            return
        for itm in s:
            row = self.images_list.row(itm)
            self.images_list.takeItem(row)
            p = itm.toolTip()
            self.loaded_images = [x for x in self.loaded_images if x["path"] != p]
        self.update_status()

    def update_status(self):
        c = len(self.loaded_images)
        self.status_label.setText(f"{c} images loaded.")

    # ---------- Preview ----------
    def preview_selected(self):
        s = self.images_list.selectedItems()
        if not s:
            QMessageBox.information(self, "Preview", "No item selected.")
            return
        path = s[0].toolTip()
        for d in self.loaded_images:
            if d["path"] == path:
                preview_image = d["image"]
                disp = self.stretch_for_display(preview_image)
                MosaicPreviewWindow(disp, title="Preview Selected", parent=self,
                                    push_cb=self._push_mosaic_to_new_doc).show()
                break

    def _resolve_view_title(self, *objs) -> str:
        """
        Try hard to get a meaningful title from any of the passed objects
        (view, subwindow, doc, etc.). Returns a non-empty string or 'Untitled View'.
        """
        def first_str(x):
            return str(x).strip() if x is not None and str(x).strip() else None

        for o in objs:
            if o is None:
                continue
            # Methods that return a title
            for meth in ("windowTitle", "title", "displayTitle", "text"):
                fn = getattr(o, meth, None)
                if callable(fn):
                    s = first_str(fn())
                    if s: return s
            # Properties/attributes that might hold a title
            for attr in ("title", "display_title", "name", "objectName", "display_name"):
                s = first_str(getattr(o, attr, None))
                if s: return s
            # From metadata
            meta = getattr(o, "metadata", None)
            if isinstance(meta, dict):
                s = first_str(meta.get("display_name") or meta.get("title") or meta.get("name"))
                if s: return s
            # From file path-ish things
            for attr in ("file_path", "filepath", "path", "filename"):
                p = getattr(o, attr, None)
                if p:
                    base = os.path.basename(str(p))
                    s = first_str(base)
                    if s: return s

        return "Untitled View"

    # --- WCS/SIP helpers ---------------------------------------------------------
    def _ensure_image_naxis(self, hdr: dict | fits.Header, shape):
        try:
            h, w = (shape[0], shape[1]) if len(shape) >= 2 else (None, None)
            if h is not None and w is not None:
                hdr["NAXIS"]  = 2
                hdr["NAXIS1"] = int(w)
                hdr["NAXIS2"] = int(h)
        except Exception:
            pass

    def _coerce_float(self, hdr: dict, key: str):
        if key in hdr:
            try:
                hdr[key] = float(hdr[key])
            except Exception:
                try:
                    hdr[key] = float(str(hdr[key]).strip().strip("'"))
                except Exception:
                    del hdr[key]

    def _normalize_wcs_header(self, hdr: dict, img_shape):
        # Required CTYPE + RADESYS
        hdr.setdefault("CTYPE1", "RA---TAN")
        hdr.setdefault("CTYPE2", "DEC--TAN")
        # RADECSYS is deprecated → RADESYS
        if "RADECSYS" in hdr and "RADESYS" not in hdr:
            hdr["RADESYS"] = hdr.pop("RADECSYS")
        hdr.setdefault("RADESYS", "ICRS")
        hdr.setdefault("WCSAXES", 2)

        # Coerce numeric offenders occasionally written as strings
        for k in ("CRPIX1","CRPIX2","CRVAL1","CRVAL2","CD1_1","CD1_2","CD2_1","CD2_2","CDELT1","CDELT2","EQUINOX","CROTA1","CROTA2"):
            self._coerce_float(hdr, k)

        # SIP orders should be ints if present
        for k in ("A_ORDER","B_ORDER","AP_ORDER","BP_ORDER"):
            if k in hdr:
                try:
                    hdr[k] = int(hdr[k])
                except Exception:
                    del hdr[k]

        # Fill missing A/B vs AP/BP if only one is present
        if "A_ORDER" in hdr and "B_ORDER" not in hdr: hdr["B_ORDER"] = hdr["A_ORDER"]
        if "B_ORDER" in hdr and "A_ORDER" not in hdr: hdr["A_ORDER"] = hdr["B_ORDER"]
        if "AP_ORDER" in hdr and "BP_ORDER" not in hdr: hdr["BP_ORDER"] = hdr["AP_ORDER"]
        if "BP_ORDER" in hdr and "AP_ORDER" not in hdr: hdr["AP_ORDER"] = hdr["BP_ORDER"]

        # Make sure dimensions are present (avoids “more axes than image” warning)
        self._ensure_image_naxis(hdr, img_shape)

        return hdr

    def _build_wcs(self, hdr: dict, img_shape):
        # normalize then pass relax=True so SIP and non-standard keys are parsed
        nh = self._normalize_wcs_header(dict(hdr), img_shape)
        return WCS(nh, relax=True)


    # ---------- Align (Entry Point) ----------
    def align_images(self):
        if self.seestarCheckBox.isChecked():
            self.align_images_seestar_mode()
            return

        if len(self.loaded_images) == 0:
            QMessageBox.warning(self, "Align", "No images to align.")
            return

        # Show spinner and start animation.
        self.spinnerLabel.show()
        self.spinnerMovie.start()
        QApplication.processEvents()

        # Step 1: Force blind solve if requested.
        force_blind = self.forceBlindCheckBox.isChecked()
        images_to_process = (self.loaded_images if force_blind
                            else [item for item in self.loaded_images if item.get("wcs") is None])

        # Process each image for plate solving.
        for item in images_to_process:
            # Check if ASTAP is set.
            if not self.astap_exe or not os.path.exists(self.astap_exe):
                executable_filter = "Executables (*.exe);;All Files (*)" if sys.platform.startswith("win") else "Executables (*)"
                new_path, _ = QFileDialog.getOpenFileName(self, "Select ASTAP Executable", "", executable_filter)
                if new_path:
                    self._save_astap_exe_to_settings(new_path)
                    QMessageBox.information(self, "Mosaic Master", "ASTAP path updated successfully.")
                else:
                    QMessageBox.warning(self, "Mosaic Master", "ASTAP path not provided. Falling back to blind solve.")
                    solved_header = self.perform_blind_solve(item)
                    if solved_header:
                        # normalize + build WCS with relax=True so SIP is retained
                        solved_header = self._normalize_wcs_header(solved_header, item["image"].shape)
                        item["wcs"] = self._build_wcs(solved_header, item["image"].shape)
                    continue

            # Attempt ASTAP solve.
            self.status_label.setText(f"Attempting ASTAP solve for {item['path']}...")
            QApplication.processEvents()
            solved_header = self.attempt_astap_solve(item)

            if solved_header is None:
                self.status_label.setText(f"ASTAP failed for {item['path']}. Falling back to blind solve...")
                QApplication.processEvents()
                solved_header = self.perform_blind_solve(item)
            else:
                self.status_label.setText(f"Plate solve successful using ASTAP for {item['path']}.")

            if solved_header:
                # Single, centralized sanitize → keeps SIP and fixes types
                solved_header = self._normalize_wcs_header(solved_header, item["image"].shape)
                item["wcs"] = self._build_wcs(solved_header, item["image"].shape)
            else:
                print(f"Plate solving failed for {item['path']}.")

        # After processing, get all images with valid WCS.
        wcs_items = [x for x in self.loaded_images if x.get("wcs") is not None]
        if not wcs_items:
            print("No images have WCS, skipping WCS alignment.")
            self.spinnerMovie.stop()
            self.spinnerLabel.hide()
            return

        # Use the first image's WCS as reference and compute the mosaic bounding box.
        # (Rebuild with relax=True just in case, then deepcopy.)
        reference_wcs = self._build_wcs(wcs_items[0]["wcs"].to_header(relax=True), wcs_items[0]["image"].shape).deepcopy()
        min_x, min_y, max_x, max_y = self.compute_mosaic_bounding_box(wcs_items, reference_wcs)
        mosaic_width = int(max_x - min_x)
        mosaic_height = int(max_y - min_y)

        if mosaic_width < 1 or mosaic_height < 1:
            print("ERROR: Computed mosaic size is invalid. Check WCS or inputs.")
            self.spinnerMovie.stop()
            self.spinnerLabel.hide()
            return

        # Adjust the reference WCS so that (min_x, min_y) becomes (0,0).
        mosaic_wcs = reference_wcs.deepcopy()
        mosaic_wcs.wcs.crpix[0] -= min_x
        mosaic_wcs.wcs.crpix[1] -= min_y
        # keep SIP in the stored header
        self.wcs_metadata = mosaic_wcs.to_header(relax=True)

        # Set up accumulators.
        is_color = any(not item["is_mono"] for item in wcs_items)
        if is_color:
            self.final_mosaic = np.zeros((mosaic_height, mosaic_width, 3), dtype=np.float32)
        else:
            self.final_mosaic = np.zeros((mosaic_height, mosaic_width), dtype=np.float32)
        self.weight_mosaic = np.zeros((mosaic_height, mosaic_width), dtype=np.float32)

        first_image = True
        for idx, itm in enumerate(wcs_items):
            arr = itm["image"]
            self.status_label.setText(f"Projecting {itm['path']} onto the celestial sphere...")
            QApplication.processEvents()

            # Pre-stretch the image.
            stretched_arr = self.stretch_image(arr)
            # Use the first channel for alignment.
            if not itm["is_mono"]:
                red_stretched = stretched_arr[..., 0]
            else:
                red_stretched = stretched_arr[..., 0] if stretched_arr.ndim == 3 else stretched_arr

            # Reproject the image.
            if not itm["is_mono"]:
                channels = []
                for c in range(3):
                    channel = stretched_arr[..., c]
                    reproj, _ = reproject_interp((channel, itm["wcs"]), mosaic_wcs, shape_out=(mosaic_height, mosaic_width))
                    reproj = np.nan_to_num(reproj, nan=0.0).astype(np.float32)
                    channels.append(reproj)
                reprojected = np.stack(channels, axis=-1)
                reproj_red = reprojected[..., 0]
            else:
                reproj_red, _ = reproject_interp((red_stretched, itm["wcs"]), mosaic_wcs, shape_out=(mosaic_height, mosaic_width))
                reproj_red = np.nan_to_num(reproj_red, nan=0.0).astype(np.float32)
                reprojected = np.stack([reproj_red, reproj_red, reproj_red], axis=-1)

            self.status_label.setText(f"WCS Reproject: {itm['path']} processed.")
            QApplication.processEvents()

            # --- Stellar Alignment ---
            if not first_image:
                transform_method = self.transform_combo.currentText()
                mosaic_gray = (self.final_mosaic if self.final_mosaic.ndim == 2
                            else np.mean(self.final_mosaic, axis=-1))
                try:
                    self.status_label.setText("Computing affine transform with astroalign...")
                    QApplication.processEvents()
                    # Use the backoff wrapper to avoid SEP buffer overflows
                    transform_obj, (src_pts, dst_pts) = self._aa_find_transform_with_backoff(reproj_red, mosaic_gray)
                    transform_matrix = transform_obj.params[0:2, :].astype(np.float32)
                    self.status_label.setText("Astroalign computed transform successfully.")
                except Exception as e:
                    self.status_label.setText(f"Astroalign failed: {e}. Using identity transform.")
                    transform_matrix = np.eye(2, 3, dtype=np.float32)

                A = transform_matrix[:, :2]
                scale1 = np.linalg.norm(A[:, 0])
                scale2 = np.linalg.norm(A[:, 1])
                print(f"Computed affine scales: {scale1:.6f}, {scale2:.6f}")

                self.status_label.setText("Affine alignment computed. Warping image...")
                QApplication.processEvents()
                affine_aligned = cv2.warpAffine(reprojected, transform_matrix, (mosaic_width, mosaic_height),
                                                flags=cv2.INTER_LANCZOS4)
                aligned = affine_aligned

                if transform_method in ["Homography Transform", "Polynomial Warp Based Transform"]:
                    self.status_label.setText(f"Starting refined alignment using {transform_method}...")
                    QApplication.processEvents()
                    refined_result = self.refined_alignment(affine_aligned, mosaic_gray, method=transform_method)
                    if refined_result is not None:
                        aligned, best_inliers2 = refined_result
                        self.status_label.setText(f"Refined alignment succeeded with {best_inliers2} inliers.")
                    else:
                        self.status_label.setText("Refined alignment failed; falling back to affine alignment.")
            else:
                aligned = reprojected
                first_image = False

            gray_aligned = aligned[..., 0] if aligned.ndim == 3 else aligned

            # Compute weight mask
            binary_mask = (gray_aligned > 0).astype(np.uint8)
            smooth_mask = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
            if np.max(smooth_mask) > 0:
                smooth_mask = smooth_mask / np.max(smooth_mask)
            else:
                smooth_mask = binary_mask.astype(np.float32)
            smooth_mask = cv2.GaussianBlur(smooth_mask, (15, 15), 0)

            # Accumulate
            if is_color:
                self.final_mosaic += aligned * smooth_mask[..., np.newaxis]
            else:
                self.final_mosaic += aligned[..., 0] * smooth_mask
            self.weight_mosaic += smooth_mask

            self.status_label.setText(f"Processed: {itm['path']}")
            QApplication.processEvents()

        # Final blending.
        nonzero_mask = (self.weight_mosaic > 0)
        if is_color:
            self.final_mosaic = np.where(self.weight_mosaic[..., None] > 0,
                                        self.final_mosaic / self.weight_mosaic[..., None],
                                        self.final_mosaic)
        else:
            self.final_mosaic[nonzero_mask] = self.final_mosaic[nonzero_mask] / self.weight_mosaic[nonzero_mask]

        print("WCS + Star Alignment Complete.")
        self.status_label.setText("WCS + Star Alignment Complete. De-Normalizing Mosaic...")
        self.final_mosaic = self.unstretch_image(self.final_mosaic)
        self.status_label.setText("Final Mosaic Ready.")
        QApplication.processEvents()

        display_image = (np.stack([self.final_mosaic]*3, axis=-1)
                        if self.final_mosaic.ndim == 2 else self.stretch_for_display(self.final_mosaic))
        MosaicPreviewWindow(display_image, title="Final Mosaic", parent=self,
                            push_cb=self._push_mosaic_to_new_doc).show()

        self.spinnerMovie.stop()
        self.spinnerLabel.hide()
        QApplication.processEvents()
      

    def debayer_image(self, image, file_path, header):
        if file_path.lower().endswith(('.cr2', '.cr3', '.nef', '.arw', '.dng', '.raf', '.orf', '.rw2', '.pef')):
            print(f"Debayering RAW image: {file_path}")
            return debayer_raw_fast(image)
        elif file_path.lower().endswith(('.fits', '.fit', '.fz', '.fz')):
            bayer_pattern = header.get('BAYERPAT')
            if bayer_pattern:
                print(f"Debayering FITS image: {file_path} with Bayer pattern {bayer_pattern}")
                return debayer_fits_fast(image, bayer_pattern)
        return image

    def refine_via_overlap(self, new_gray, mosaic_gray, rough_matrix):
        """
        1) Warp new_gray into mosaic coords using rough_matrix.
        2) Compute overlap region by checking where both warped_new_gray and mosaic_gray > 0.
        3) Restrict astroalign.find_transform() to that overlap only.
        4) Combine the refinement transform with rough_matrix so you have a single final 2×3 matrix
        mapping the original new_gray -> mosaic_gray coordinates.

        Returns the final 2×3 transform matrix for cv2.warpAffine, or None if no overlap or alignment fails.
        """

        # ------------------------------------------
        # A) Warp new_gray with rough_matrix
        # ------------------------------------------
        h_m, w_m = mosaic_gray.shape
        warped_new_gray = cv2.warpAffine(
            new_gray,
            rough_matrix,
            (w_m, h_m),   # (width, height)
            flags=cv2.INTER_LANCZOS4
        )

        # ------------------------------------------
        # B) Determine overlap region (both > 0)
        # ------------------------------------------
        mask_warped = (warped_new_gray > 0)
        mask_mosaic = (mosaic_gray > 0)
        overlap_mask = mask_warped & mask_mosaic

        overlap_pixels = np.count_nonzero(overlap_mask)
        print(f"[DEBUG] Overlap region has {overlap_pixels} pixels.")

        # If there’s not enough overlap, bail out
        if overlap_pixels < 50:
            # Not enough area for star matching
            return None

        # ------------------------------------------
        # C) Mask out everything except the overlap
        # ------------------------------------------
        # We'll use numpy.ma arrays so astroalign sees "valid" data only in overlap region
        warped_new_ma = ma.array(warped_new_gray, mask=~overlap_mask)
        mosaic_ma = ma.array(mosaic_gray, mask=~overlap_mask)

        # ------------------------------------------
        # D) Attempt astroalign on that overlap
        #    Note: the transform we get is from warped_new_gray -> mosaic_gray,
        #    i.e., in "mosaic coordinate space." So it should be *almost* identity
        #    if the rough transform was close, but with some tweak for rotation/translation.
        # ------------------------------------------
        try:
            refine_obj, (src_pts, dst_pts) = astroalign.find_transform(
                warped_new_ma, mosaic_ma,
                max_control_points=50,
                detection_sigma=2
            )
            print(f"[DEBUG] Overlap-limited astroalign success with {len(src_pts)} stars.")
        except Exception as e:
            print(f"[DEBUG] Overlap-limited astroalign failed: {e}")
            return None

        # ------------------------------------------
        # E) Combine refine_obj with the original rough_matrix
        # ------------------------------------------
        # refine_obj.params is typically a 3×3 (similarity).
        # rough_matrix is 2×3 for warpAffine.
        # We'll promote rough_matrix to 3×3, multiply, then convert back.
        refine_mat_3x3 = refine_obj.params  # e.g. a 3×3
        rough_mat_3x3 = np.eye(3, dtype=np.float32)
        rough_mat_3x3[:2, :] = rough_matrix

        # The final transform is refine_mat_3x3 * rough_mat_3x3
        #   (in linear-algebra order, the right-hand transform applies first)
        combined_3x3 = refine_mat_3x3 @ rough_mat_3x3

        # Convert that back to 2×3 for cv2
        final_transform = combined_3x3[:2, :].astype(np.float32)
        return final_transform


    def align_images_seestar_mode(self):
        """ 
        Align images in Seestar Mode by first selecting the center-most image (based on WCS centers)
        as the initial mosaic. Then, sort and add images by increasing distance from the mosaic center.
        If WCS information is present in the header, a rough pre-alignment is computed before refining
        with astroalign.find_transform. After processing all images, failed images are reattempted.

        Final image is produced by summing all warped images and dividing by the pixel counts 
        (sum-then-divide approach).
        """

        if len(self.loaded_images) == 0:
            QMessageBox.warning(self, "Align", "No images to align.")
            return

        self.status_label.setText("🔄 Image Registration Started...")
        self.spinnerLabel.show()
        self.spinnerMovie.start()
        QApplication.processEvents()

        total_files = len(self.loaded_images)  # how many images in total
        if total_files == 0:
            QMessageBox.warning(self, "Align", "No images to align.")
            return

        # Create a QProgressDialog
        progress = QProgressDialog("Aligning images...", "Cancel", 0, total_files, self)
        progress.setWindowTitle("Seestar Alignment")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        progress.setMinimumDuration(0)
        progress.show()

        # We define how many pixels to zero out on each edge AFTER warp
        POST_WARP_BORDER = 10
        THRESHOLD_RATIO = 0.9  # if new pixel < 0.5 * mosaic pixel, skip it

        # -----------------------------------------------------
        # Helper: Extract the world-coordinate center from an image header
        # -----------------------------------------------------
        def get_wcs_center(item):
            header = item.get("header", None)
            if header is None:
                print(f"[DEBUG] No header for {item.get('path','')}")
                return None
            try:
                wcs = WCS(header)
                naxis1 = int(header.get("NAXIS1", 0))
                naxis2 = int(header.get("NAXIS2", 0))
                center_pix = np.array([naxis1 / 2.0, naxis2 / 2.0])
                center_world = wcs.all_pix2world(center_pix[None, :], 1)[0]
                print(f"[DEBUG] {item.get('path','')} center_world: {center_world}")
                return center_world
            except Exception as e:
                print(f"[DEBUG] Failed to get WCS center for {item.get('path','')}: {e}")
                return None

        # -----------------------------------------------------
        # Helper: Count how many stars appear in an image (2D or 3D)
        # -----------------------------------------------------
        def count_stars_in_image(image):
            """
            Use DAOStarFinder to return number of detected stars.
            Expects a 2D image; if 3D, we average over channels.
            You might need to tweak fwhm/threshold for your data.
            """
            if image.ndim == 3:
                image = np.mean(image, axis=2)

            mean_val, median_val, std_val = sigma_clipped_stats(image, sigma=3.0)
            daofind = DAOStarFinder(fwhm=3.0, threshold=5.0 * std_val)
            sources = daofind(image - median_val)

            if sources is None:
                return 0
            else:
                return len(sources)

        # -----------------------------------------------------
        # 1) Gather all images with valid WCS and compute average center
        # -----------------------------------------------------
        centers = []
        valid_items = []
        for item in self.loaded_images:
            center = get_wcs_center(item)
            if center is not None:
                centers.append(center)
                valid_items.append(item)

        if not centers:
            QMessageBox.warning(self, "Align", "No images with valid WCS found.")
            return

        centers = np.array(centers)
        avg_center = np.mean(centers, axis=0)  # or np.median if you prefer

        # -----------------------------------------------------
        # Sort valid items by distance from average center
        # -----------------------------------------------------
        def distance_from_avg(item):
            center = get_wcs_center(item)
            dist = np.linalg.norm(center - avg_center) if center is not None else np.inf
            print(f"[DEBUG] {item['path']} distance from avg: {dist}")
            return dist

        valid_items.sort(key=distance_from_avg)
        for item in valid_items:
            print(f"[DEBUG] Sorted valid item: {item['path']}")

        # -----------------------------------------------------
        # Select from the top few “closest to center,” pick the one with the most stars as the base.
        # -----------------------------------------------------
        top_n = 5
        candidate_subset = valid_items[:top_n]

        best_item = None
        best_star_count = 0
        for candidate in candidate_subset:
            star_count = count_stars_in_image(candidate["image"])
            print(f"[DEBUG] {candidate['path']} star count: {star_count}")
            if star_count > best_star_count:
                best_star_count = star_count
                best_item = candidate

        if best_item is None:
            best_item = valid_items[0]

        base_item = best_item
        print(f"[DEBUG] Selected base image: {base_item['path']} with star count = {best_star_count}")

        # -----------------------------------------------------
        # Reorder loaded_images so that valid (WCS) items come first
        # -----------------------------------------------------
        remaining_items = []
        valid_paths = {v["path"] for v in valid_items}  # set of valid item paths

        for item in self.loaded_images:
            if item["path"] not in valid_paths:
                remaining_items.append(item)

        self.loaded_images = valid_items + remaining_items


        # -----------------------------------------------------
        # Helper: Crop a 5-pixel border from each edge
        # -----------------------------------------------------
        def crop_5px_border(img):
            if img.ndim == 3:
                h, w, c = img.shape
                if h <= 10 or w <= 10:
                    return np.empty((0, 0, c), dtype=img.dtype)
                return img[10:-10, 10:-10, :]
            else:
                h, w = img.shape
                if h <= 10 or w <= 10:
                    return np.empty((0, 0), dtype=img.dtype)
                return img[10:-10, 10:-10]

        # -----------------------------------------------------
        # Helper: Convert image to float32 and normalize if needed
        # -----------------------------------------------------
        def ensure_float32_in_01(img):
            img = img.astype(np.float32, copy=False)
            mx = np.max(img)
            if mx <= 1.0:
                return img
            elif mx <= 255.0:
                return img / 255.0
            elif mx <= 65535.0:
                return img / 65535.0
            else:
                return img / mx if mx > 0 else img

        # -----------------------------------------------------
        # Helper: Compute a rough transform (rotation+translation) using WCS
        # -----------------------------------------------------
        def compute_rough_transform(new_header, ref_header):
            """Compute a rough transformation (rotation and translation) using WCS."""
            try:
                new_wcs = WCS(new_header)
                ref_wcs = WCS(ref_header)
                ref_naxis1 = int(ref_header.get('NAXIS1', 0))
                ref_naxis2 = int(ref_header.get('NAXIS2', 0))
                ref_center = np.array([ref_naxis1 / 2.0, ref_naxis2 / 2.0])
                world_center = ref_wcs.all_pix2world(ref_center[None, :], 1)
                new_center = new_wcs.all_world2pix(world_center, 1)[0]
                ref_offset = ref_center + np.array([1, 0])
                world_offset = ref_wcs.all_pix2world(ref_offset[None, :], 1)[0]
                new_offset = new_wcs.all_world2pix(world_offset[None, :], 1)[0]
                vector = new_offset - new_center
                angle = np.arctan2(vector[1], vector[0])
                cos_a = np.cos(angle)
                sin_a = np.sin(angle)
                R = np.array([[cos_a, -sin_a],
                            [sin_a,  cos_a]])
                t = ref_center - R @ new_center
                rough = np.hstack([R, t.reshape(2, 1)]).astype(np.float32)
                return rough
            except Exception as e:
                print(f"Rough transform skipped: {e}")
                return None

        def compute_rough_transform_seestar(new_header, ref_header):
            """
            Builds a 2×3 transform matrix (rotation + translation) that
            roughly aligns an image with RA/DEC, pixel scale, and focal length
            to a reference image with the same kind of data.

            Required header fields (example):
            new_header["RA"]       (degrees)
            new_header["DEC"]      (degrees)
            new_header["XPIXSZ"]   (microns)
            new_header["YPIXSZ"]   (microns)
            new_header["FOCALLEN"] (mm)
            Similarly for ref_header.

            Returns:
            A 2×3 np.float32 array suitable for cv2.warpAffine,
            or None if missing data or something fails.
            """

            # 1) Parse required fields from new_header
            try:
                ra_new_deg = float(new_header["RA"])       # degrees
                dec_new_deg = float(new_header["DEC"])     # degrees
                xpixsz_new = float(new_header["XPIXSZ"])   # microns
                ypixsz_new = float(new_header["YPIXSZ"])   # microns
                flen_new   = float(new_header["FOCALLEN"]) # mm
            except KeyError as e:
                print(f"[DEBUG] new_header missing key {e}. No rough transform.")
                return None
            except Exception as e:
                print(f"[DEBUG] parse error in new_header => {e}")
                return None

            # 2) Parse required fields from ref_header
            try:
                ra_ref_deg = float(ref_header["RA"])
                dec_ref_deg = float(ref_header["DEC"])
                xpixsz_ref = float(ref_header["XPIXSZ"])
                ypixsz_ref = float(ref_header["YPIXSZ"])
                flen_ref   = float(ref_header["FOCALLEN"])
            except KeyError as e:
                print(f"[DEBUG] ref_header missing key {e}. No rough transform.")
                return None
            except Exception as e:
                print(f"[DEBUG] parse error in ref_header => {e}")
                return None

            # 3) Compute average arcsec/pixel for each image
            #    arcsec_per_pix = 206265 * (XPIXSZ [um]) / (FOCALLEN [mm])
            #    (assuming 1 mm = 1000 um)
            #    If you want to be more precise, you might average XPIXSZ and YPIXSZ,
            #    or handle them separately if the camera has rectangular pixels.
            def arcsec_per_pixel(xpixsz, flen):
                return 206265.0 * (xpixsz / 1000.0) / flen

            scale_new = arcsec_per_pixel(xpixsz_new, flen_new)
            scale_ref = arcsec_per_pixel(xpixsz_ref, flen_ref)

            # For simplicity, let's assume the "reference" image defines the final scale.
            # If the images have different scales, we can do a ratio of scales => ~some "zoom."
            # Here we assume they are close enough, or you can do scale = scale_ref / scale_new
            # to get a small difference in pixel scale if needed.
            scale_ratio = scale_ref / scale_new  # how many ref pixels = 1 new pixel

            # 4) Compute difference in RA/DEC in arcseconds
            #    RA/DEC difference in degrees => multiply by 3600 to get arcseconds
            #    Then convert to pixel shift in reference coordinates.
            d_ra_deg  = (ra_new_deg  - ra_ref_deg)
            d_dec_deg = (dec_new_deg - dec_ref_deg)

            # For small fields, we can do a simple approximation ignoring spherical geometry:
            d_ra_arcsec  = d_ra_deg  * 3600.0 * np.cos(np.radians(dec_ref_deg))  # cos(dec) if you want
            d_dec_arcsec = d_dec_deg * 3600.0

            # If you want a more precise approach, you'd do a spherical trig approach,
            # but for small fields, this is usually good enough.

            # 5) Convert arcseconds => pixel shift in "reference" image coords
            dx_pix = d_ra_arcsec  / scale_ref
            dy_pix = - d_dec_arcsec / scale_ref
            # Note: we do "dy_pix = -d_dec_arcsec" if we assume +DEC => "up" in the image.
            # You may need to invert or swap RA/DEC sign depending on your orientation.

            # 6) Combine the scale ratio (if needed) and shift into a 2×3 transform.
            #    If you want to guess rotation from the difference in rotation angles,
            #    you can do that here. For now, we do no rotation => angle=0 => cos=1, sin=0.
            angle_rad = 0.0
            cos_a = np.cos(angle_rad)
            sin_a = np.sin(angle_rad)

            # scale_ratio is for "zoom," if you want to handle that.
            # If you prefer no scale difference, set scale_ratio = 1.0
            # or if the difference is small, ignore it.
            # scale_ratio = 1.0

            M = np.array([
                [scale_ratio*cos_a, -scale_ratio*sin_a, dx_pix],
                [scale_ratio*sin_a,  scale_ratio*cos_a, dy_pix]
            ], dtype=np.float32)

            print(f"[DEBUG] Seestar rough transform => dx_pix={dx_pix:.2f}, dy_pix={dy_pix:.2f}, scale={scale_ratio:.3f}")
            return M

        # -----------------------------------------------------
        # 1) Initialize mosaic_sum/mosaic_count from the base image
        # -----------------------------------------------------
        base_img = ensure_float32_in_01(base_item["image"])
        if base_item["path"].lower().endswith(('.fits', '.fit', '.fz')):
            if (base_img.ndim == 2 
                and "header" in base_item 
                and base_item["header"] is not None 
                and base_item["header"].get('BAYERPAT')):
                self.status_label.setText(f"Debayering {base_item['path']}")
                QApplication.processEvents()
                base_img = self.debayer_image(base_img, base_item["path"], base_item["header"])
                print(f"[DEBUG] Finished debayering base image: {base_item['path']}")

        base_img = crop_5px_border(base_img)
        if base_img.size == 0:
            QMessageBox.warning(self, "Crop Error", "Initial image is too small after cropping.")
            return
    
        base_median = np.median(base_img)

        # mosaic_sum holds the sum of pixel intensities
        mosaic_sum = base_img.astype(np.float32, copy=True)
        # mosaic_count tracks how many images contributed at each pixel
        if mosaic_sum.ndim == 3:
            # For color: count “contributed” if any channel is >0
            contrib_mask = np.any(mosaic_sum > 0, axis=2)
            mosaic_count = np.zeros_like(mosaic_sum, dtype=np.float32)
            for c in range(mosaic_sum.shape[2]):
                mosaic_count[..., c][contrib_mask] = 1.0
        else:
            contrib_mask = (mosaic_sum > 0)
            mosaic_count = np.zeros_like(mosaic_sum, dtype=np.float32)
            mosaic_count[contrib_mask] = 1.0

        # We’ll use a helper that returns a mono “view” for alignment:
        def get_current_mosaic_gray():
            """Return current mosaic as a float32 array in [0..1], averaged if color."""
            with np.errstate(divide='ignore', invalid='ignore'):
                stacked = mosaic_sum / np.maximum(mosaic_count, 1e-6)
            # If color, reduce to mono for astroalign
            if stacked.ndim == 3:
                return np.mean(stacked, axis=2).astype(np.float32)
            else:
                return stacked

        self.status_label.setText("Starting alignment of subsequent images...")
        QApplication.processEvents()

        # --- Helper: Process alignment for one image, then accumulate into mosaic_sum/count ---
        def process_alignment(item):
            nonlocal mosaic_sum, mosaic_count

            print(f"[DEBUG] Aligning image: {item['path']}")

            try:
                new_img = item["image"].astype(np.float32)
                print(f"[DEBUG] new_img initial shape={new_img.shape}, dtype={new_img.dtype}, "
                    f"min={np.min(new_img):.3f}, max={np.max(new_img):.3f}")

                # Debayer if needed
                if item["path"].lower().endswith(('.fits', '.fit')):
                    if (new_img.ndim == 2 
                        and "header" in item 
                        and item["header"] is not None 
                        and item["header"].get('BAYERPAT')):
                        print(f"[DEBUG] Debayering image: {item['path']}")
                        new_img = self.debayer_image(new_img, item["path"], item["header"])
                        print(f"[DEBUG] debayered new_img shape={new_img.shape}, dtype={new_img.dtype}, "
                            f"min={np.min(new_img):.3f}, max={np.max(new_img):.3f}")

                new_img = crop_5px_border(new_img)
                if new_img.size == 0:
                    print(f"[DEBUG] new_img is empty after cropping => skip.")
                    self.status_label.setText(f"Skipping {item['path']} (too small after crop).")
                    QApplication.processEvents()
                    return False

                self.status_label.setText(f"Removing linear gradient from {item['path']}...")
                QApplication.processEvents()
                # Create an instance of PolyGradientRemoval with degree 1.
                poly_remover = PolyGradientRemoval(new_img, poly_degree=2, downsample_scale=5, num_sample_points=100)
                # Process the image to remove the gradient
                new_img = poly_remover.process()
                print("[DEBUG] Finished polynomial gradient removal on subframe.")

                # If mono:
                self.status_label.setText(f"Normalizing {item['path']}.")
                QApplication.processEvents()
                new_img = new_img-np.min(new_img)
                if new_img.ndim == 2:
                    new_img = stretch_mono_image(new_img, target_median=base_median, normalize=False, apply_curves=False, curves_boost=0.0)

                else:
                    # color approach - you might do a single ratio or call your stretch_color_image
                    new_img = stretch_color_image(new_img, target_median=base_median, linked=False, normalize=False, apply_curves=False, curves_boost=0.0)


                # If mosaic is color and new_img is mono, expand new_img
                # But remember, mosaic_sum is our "canvas"
                if mosaic_sum.ndim == 3 and new_img.ndim == 2:
                    print("[DEBUG] Expanding new_img from 2D => 3D to match mosaic channels.")
                    new_img = np.repeat(new_img[:, :, np.newaxis], mosaic_sum.shape[2], axis=2)
                elif mosaic_sum.ndim == 2 and new_img.ndim == 3:
                    print("[DEBUG] new_img is color but mosaic is mono => average new_img channels.")
                    new_img = np.mean(new_img, axis=2, dtype=np.float32)

                # Create the grayscale for astroalign from mosaic_sum/mosaic_count
                mosaic_gray = get_current_mosaic_gray()

                new_gray = new_img
                if new_gray.ndim == 3:
                    new_gray = np.mean(new_gray, axis=2)

                # Optional: mild stretch to help astroalign
                mosaic_gray = stretch_mono_image(mosaic_gray, target_median=0.1, normalize=False, apply_curves=False, curves_boost=0.0)
                new_gray = stretch_mono_image(new_gray, target_median=0.1, normalize=False, apply_curves=False, curves_boost=0.0)

                # Attempt rough transform via WCS
                rough_matrix = None
                if ("header" in item and item["header"] is not None
                    and "CTYPE1" in item["header"] and "CTYPE2" in item["header"]):
                    # Normal WCS approach
                    rough_matrix = compute_rough_transform(item["header"], base_item["header"])
                    if rough_matrix is not None:
                        print("[DEBUG] WCS-based rough transform:\n", rough_matrix)
                else:
                    # Fallback: check if we have RA,DEC,XPIXSZ,FOCALLEN, etc.
                    rough_matrix = compute_rough_transform_seestar(item["header"], base_item["header"])
                    if rough_matrix is not None:
                        print("[DEBUG] Seestar rough transform:\n", rough_matrix)

                self.status_label.setText(f"Astroaligning for {item['path']}...")
                QApplication.processEvents()

                # Find transform
                transform_matrix = None
                try:
                    # astroalign wants double precision
                    transform_obj, (src_pts, dst_pts) = astroalign.find_transform(
                        new_gray,
                        mosaic_gray,
                        max_control_points=50,
                        detection_sigma=4,
                        min_area=5
                    )
                    transform_matrix = transform_obj.params[0:2, :].astype(np.float32)
                    print(f"[DEBUG] Astroalign success with {len(src_pts)} stars. transform_matrix:\n{transform_matrix}")

                except Exception as e:
                    print(f"[DEBUG] astroalign.find_transform failed: {e}")
                    self.status_label.setText(f"Alignment failed: {e}")
                    QApplication.processEvents()

                    # Fallback if rough_matrix is available
                    if rough_matrix is not None:
                        print("[DEBUG] Attempting refine_via_overlap fallback with rough_matrix.")
                        transform_matrix = self.refine_via_overlap(new_gray, mosaic_gray, rough_matrix)
                        if transform_matrix is None:
                            print("[DEBUG] Overlap approach also failed => skip this image.")
                            return False
                        print("[DEBUG] Overlap-based refinement succeeded. transform_matrix:\n", transform_matrix)
                    else:
                        print("[DEBUG] No rough transform => skipping image.")
                        return False

                if transform_matrix is None:
                    self.status_label.setText("No transform found; skipping image.")
                    print("[DEBUG] transform_matrix is None => returning False")
                    return False

                self.status_label.setText(f"Astroalign success for {item['path']}")
                QApplication.processEvents()

                # Warp the new image onto mosaic_sum's coordinate system
                h_m, w_m = mosaic_sum.shape[:2]
                new_h, new_w = new_img.shape[:2]

                mosaic_corners = np.array([[0, 0], [w_m, 0], [0, h_m], [w_m, h_m]], dtype=np.float32)
                new_img_corners = np.array([[0, 0], [new_w, 0], [0, new_h], [new_w, new_h]], dtype=np.float32)
                ones = np.ones((4, 1), dtype=np.float32)
                new_img_corners_hom = np.hstack([new_img_corners, ones])

                warped_corners = (transform_matrix @ new_img_corners_hom.T).T
                all_corners = np.vstack([mosaic_corners, warped_corners])
                min_xy = np.min(all_corners, axis=0)
                max_xy = np.max(all_corners, axis=0)

                margin = 10
                new_canvas_width = int(np.ceil(max_xy[0] - min_xy[0])) + margin
                new_canvas_height = int(np.ceil(max_xy[1] - min_xy[1])) + margin
                shift = -min_xy + np.array([margin / 2, margin / 2])
                shift_int = np.round(shift).astype(int)
                y0, x0 = shift_int[1], shift_int[0]

                print(f"[DEBUG] new_canvas_width={new_canvas_width}, new_canvas_height={new_canvas_height}")
                print(f"[DEBUG] shift={shift}, shift_int={shift_int}, y0={y0}, x0={x0}")

                # Expand mosaic_sum/mosaic_count to fit new canvas if needed
                expanded_sum = None
                expanded_count = None

                if mosaic_sum.ndim == 3:
                    channels = mosaic_sum.shape[2]
                    expanded_sum = np.zeros((new_canvas_height, new_canvas_width, channels), dtype=np.float32)
                    expanded_count = np.zeros_like(expanded_sum, dtype=np.float32)
                    # Copy existing mosaic_sum/mosaic_count
                    expanded_sum[y0:y0+h_m, x0:x0+w_m, :] = mosaic_sum
                    expanded_count[y0:y0+h_m, x0:x0+w_m, :] = mosaic_count
                else:
                    expanded_sum = np.zeros((new_canvas_height, new_canvas_width), dtype=np.float32)
                    expanded_count = np.zeros_like(expanded_sum, dtype=np.float32)
                    expanded_sum[y0:y0+h_m, x0:x0+w_m] = mosaic_sum
                    expanded_count[y0:y0+h_m, x0:x0+w_m] = mosaic_count

                # Adjust transform for the shift
                new_transform = transform_matrix.copy()
                new_transform[0, 2] += shift[0]
                new_transform[1, 2] += shift[1]

                # Warp the new_img
                try:
                    warped_new = cv2.warpAffine(
                        new_img,
                        new_transform,
                        (new_canvas_width, new_canvas_height),
                        flags=cv2.INTER_LANCZOS4
                    )
                except cv2.error as cv2_err:
                    print(f"[OpenCV] warpAffine error => {cv2_err}")
                    return False

                # 1) Build the "border_mask" that excludes the outer POST_WARP_BORDER region
                h_w, w_w = warped_new.shape[:2]
                border_mask = np.ones((h_w, w_w), dtype=bool)
                b = POST_WARP_BORDER
                border_mask[:b, :] = False
                border_mask[-b:, :] = False
                border_mask[:, :b] = False
                border_mask[:, -b:] = False

                # 2) Build a "valid_mask" by warping an image of ones using INTER_NEAREST.
                ones_img = np.ones(new_img.shape[:2], dtype=np.uint8)
                warped_mask = cv2.warpAffine(
                    ones_img,
                    new_transform,
                    (new_canvas_width, new_canvas_height),
                    flags=cv2.INTER_NEAREST
                )
                valid_mask = (warped_mask == 1)

                # 2b) Compute a feathering weight from the valid_mask:
                # Compute the distance transform of the valid_mask.
                # This gives, for each pixel inside the valid region, its distance (in pixels)
                # from the nearest 0 (i.e. from the border of the valid region).
                FEATHER_RADIUS = 20  # adjust as needed; this is the maximum distance for feathering
                # Convert valid_mask to uint8 for distance transform.
                valid_uint8 = valid_mask.astype(np.uint8)
                dist = cv2.distanceTransform(valid_uint8, cv2.DIST_L2, 5)
                # Compute weights: linear ramp from 0 (at distance=0) to 1 (at distance>=FEATHER_RADIUS).
                feather_weights = np.clip(dist / FEATHER_RADIUS, 0, 1)

                # 3) Build "mosaic_gray" = the grayscale of (expanded_sum / expanded_count)
                mosaic_gray = np.zeros((new_canvas_height, new_canvas_width), dtype=np.float32)
                if expanded_sum.ndim == 2:
                    nonzero_mask = (expanded_count > 0)
                    mosaic_gray[nonzero_mask] = expanded_sum[nonzero_mask] / expanded_count[nonzero_mask]
                else:
                    sum_channels = np.sum(expanded_sum, axis=2)
                    sum_counts = np.sum(expanded_count, axis=2)
                    nonzero_mask = (sum_counts > 0)
                    mosaic_gray[nonzero_mask] = sum_channels[nonzero_mask] / sum_counts[nonzero_mask]

                # 4) Build new_gray for the warped image
                if warped_new.ndim == 2:
                    new_gray = warped_new
                else:
                    new_gray = np.mean(warped_new, axis=2)

                # 5) Build the brightness_mask: accept pixels where new_gray >= THRESHOLD_RATIO * mosaic_gray.
                brightness_mask = (new_gray >= THRESHOLD_RATIO * mosaic_gray)

                # 6) Combine masks.
                # Here we combine border_mask, brightness_mask, and valid_mask.
                # The final effective weight will be the feathering weight from within the valid_mask,
                # applied only where the other masks also pass.
                combined_mask = border_mask & valid_mask & brightness_mask

                # 7) Add only those pixels, using the feathering weights.
                if warped_new.ndim == 2:
                    # MONO: weight each pixel accordingly.
                    expanded_sum[combined_mask] += warped_new[combined_mask] * feather_weights[combined_mask]
                    expanded_count[combined_mask] += feather_weights[combined_mask]
                else:
                    # COLOR: do it per channel.
                    for c in range(warped_new.shape[2]):
                        expanded_sum[..., c][combined_mask] += warped_new[..., c][combined_mask] * feather_weights[combined_mask]
                        expanded_count[..., c][combined_mask] += feather_weights[combined_mask]

                # 8) Update mosaic_sum/mosaic_count.
                mosaic_sum = expanded_sum
                mosaic_count = expanded_count


                self.status_label.setText(f"Integrated image: {item['path']}")
                QApplication.processEvents()

                print("[DEBUG] process_alignment done successfully for this image.")
                return True

            except Exception as e:
                print(f"[DEBUG] process_alignment error => {e}")
                traceback.print_exc()
                return False

        # ---------------------------------------------------------
        # Process each subsequent image once
        # ---------------------------------------------------------
        failed_items = []
        # We'll do a normal for-loop with enumerate so we can pass index to progress
        for i, item in enumerate(self.loaded_images):
            # If user cancels
            if progress.wasCanceled():
                self.status_label.setText("Alignment canceled by user.")
                break

            # Update the progress label
            progress.setLabelText(f"Aligning image {i+1}/{total_files}: {item['path']}")
            progress.setValue(i)  # update the progress bar

            if item is base_item:
                continue

            success = process_alignment(item)
            if not success:
                failed_items.append(item)

            QApplication.processEvents()  # allow UI updates

        # Final step: mark progress done
        progress.setValue(total_files)

        # ---------------------------------------------------------
        # Reattempt failed images (max 1 additional attempt)
        # ---------------------------------------------------------
        max_retries = 1
        attempt = 1
        while failed_items and attempt <= max_retries:
            reattempt_count = len(failed_items)
            self.status_label.setText(f"Reattempting alignment for {reattempt_count} images (Attempt {attempt})")
            print(f"Reattempting alignment for {reattempt_count} images (Attempt {attempt})")
            QApplication.processEvents()

            # 2) Reset the existing progress bar to track the reattempt pass
            progress.setRange(0, reattempt_count)
            progress.setValue(0)
            progress.setLabelText(f"Reattempt pass {attempt}...")

            current_failures = []
            for i, item in enumerate(failed_items):
                if progress.wasCanceled():
                    self.status_label.setText("Reattempt canceled by user.")
                    break

                progress.setLabelText(f"Reattempting {item['path']} ({i+1}/{reattempt_count})")
                progress.setValue(i)
                QApplication.processEvents()

                success = process_alignment(item)
                if not success:
                    current_failures.append(item)

            progress.setValue(reattempt_count)  # done reattempt pass

            failed_items = current_failures
            attempt += 1

        # All passes complete
        progress.close()
        self.status_label.setText("All alignment attempts complete.")
        QApplication.processEvents()

        # ---------------------------------------------------------
        # Finally, build the average mosaic = mosaic_sum / mosaic_count
        # ---------------------------------------------------------
        with np.errstate(divide='ignore', invalid='ignore'):
            final_mosaic = np.zeros_like(mosaic_sum, dtype=np.float32)
            if final_mosaic.ndim == 2:
                nonzero = (mosaic_count > 0)
                final_mosaic[nonzero] = mosaic_sum[nonzero] / mosaic_count[nonzero]
            else:
                # color
                for c in range(final_mosaic.shape[2]):
                    chan_nonzero = (mosaic_count[..., c] > 0)
                    final_mosaic[..., c][chan_nonzero] = (
                        mosaic_sum[..., c][chan_nonzero] 
                        / mosaic_count[..., c][chan_nonzero]
                    )

        # Optional: you could do a final normalization
        max_val = np.max(final_mosaic)
        if max_val > 0:
            final_mosaic /= max_val

        self.final_mosaic = final_mosaic
        self.spinnerMovie.stop()
        self.spinnerLabel.hide()
        self.status_label.setText("Seestar Mode alignment (sum/divide) complete.")
        QApplication.processEvents()

        display_image = self.stretch_for_display(self.final_mosaic)
        mosaic_win = MosaicPreviewWindow(display_image, title="Robot Telescope Mosaic", parent=self,
                    push_cb=self._push_mosaic_to_new_doc).show()

    # ---------- Star alignment using triangle matching ----------

    def refined_alignment(self, affine_aligned, mosaic_img, method="Homography Transform"):
        """
        Refined alignment that assumes affine_aligned is the result of the affine alignment step.
        It re-detects stars in the candidate (overlap) region and computes a refined transform from
        affine_aligned to mosaic_img. Then it applies the refined transform to affine_aligned and
        returns the fully warped image.
        
        Returns:
        - For "Homography Transform": (warped_image, inlier_count)
        - For "Polynomial Warp Based Transform": (warped_image, inlier_count)
        - If refinement fails, returns None.
        """
        print("\n--- Starting Refined Alignment ---")
        poly_degree = self.settings.value("mosaic/poly_degree", 3, type=int)
        self.status_label.setText("Refinement: Converting images to grayscale...")
        QApplication.processEvents()
        
        # Convert images to grayscale
        if affine_aligned.ndim == 3:
            affine_aligned_gray = np.mean(affine_aligned, axis=-1)
        else:
            affine_aligned_gray = affine_aligned
        if mosaic_img.ndim == 3:
            mosaic_gray = np.mean(mosaic_img, axis=-1)
        else:
            mosaic_gray = mosaic_img

        print("Grayscale conversion done.")
        
        # Compute overlap mask
        self.status_label.setText("Refinement: Computing overlap mask...")
        QApplication.processEvents()
        overlap_mask = (mosaic_gray > 0) & (affine_aligned_gray > 0)

        # Detect stars
        self.status_label.setText("Refinement: Detecting stars in mosaic and affine-aligned images...")
        QApplication.processEvents()
        # Increase max_stars to 50 for debugging purposes.
        mosaic_stars_masked = self.detect_stars(np.where(overlap_mask, mosaic_gray, 0), max_stars=300)
        new_stars_aligned = self.detect_stars(np.where(overlap_mask, affine_aligned_gray, 0), max_stars=300)

        # Debug: Print out the star lists.
        print("Mosaic stars (refined alignment):")
        for s in mosaic_stars_masked:
            print(f"({s[0]:.2f}, {s[1]:.2f}) flux: {s[2]:.2f}")
        print("New stars (refined alignment):")
        for s in new_stars_aligned:
            print(f"({s[0]:.2f}, {s[1]:.2f}) flux: {s[2]:.2f}")

        self.status_label.setText(f"Refinement: Detected {len(mosaic_stars_masked)} mosaic stars and {len(new_stars_aligned)} new stars.")
        QApplication.processEvents()

        if len(mosaic_stars_masked) < 4 or len(new_stars_aligned) < 4:
            self.status_label.setText("Refinement: Not enough stars detected in candidate region.")
            return None

        # Match stars using position and flux.
        self.status_label.setText("Refinement: Matching stars...")
        QApplication.processEvents()
        # For debugging, you might try a higher threshold.
        matches = self.match_stars(new_stars_aligned, mosaic_stars_masked, distance_thresh=10.0, flux_thresh=1.0)
        print(f"Matched stars: {len(matches)}")
        self.status_label.setText(f"Refinement: {len(matches)} matches found.")
        if len(matches) < 4:
            self.status_label.setText("Refinement: Not enough matched stars for refined transform.")
            return None

        src_pts = np.float32([match[0][:2] for match in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([match[1][:2] for match in matches]).reshape(-1, 1, 2)

        if method == "Homography Transform":
            self.status_label.setText("Refinement: Computing homography transform...")
            QApplication.processEvents()
            H_refined, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            inliers = int(np.count_nonzero(mask)) if mask is not None else 0
            self.status_label.setText(f"Refinement: Homography computed with {inliers} inliers. Warping image...")
            QApplication.processEvents()
            if H_refined is None:
                self.status_label.setText("Refinement: Homography estimation failed.")
                return None
            warped_image = cv2.warpPerspective(affine_aligned, H_refined,
                                                (affine_aligned.shape[1], affine_aligned.shape[0]),
                                                flags=cv2.INTER_LANCZOS4)
            return (warped_image, inliers)
        elif method == "Polynomial Warp Based Transform":
            self.status_label.setText("Refinement: Computing Polynomial Warp based transform...")
            QApplication.processEvents()
            # Extract control points from matches.
            src_pts = np.float32([match[0][:2] for match in matches])
            dst_pts = np.float32([match[1][:2] for match in matches])
            # Call the static Polynomial Warp warp function.
            try:
                warped_image = MosaicMasterDialog.poly_warp(affine_aligned, src_pts, dst_pts, degree=poly_degree, status_label=self.status_label)
                inliers = len(matches)
                self.status_label.setText(f"Refinement: Polynomial Warp warp applied with {inliers} matched points.")
                return (warped_image, inliers)
            except Exception as e:
                self.status_label.setText(f"Refinement: Polynomial Warp warp failed: {e}")
                return None
        else:
            self.status_label.setText("Refinement: Unexpected transformation method.")
            return None

    @staticmethod
    def poly_warp(image, src_pts, dst_pts, degree=3, status_label=None):
        """
        Warp `image` using a polynomial transformation of the specified degree.
        
        The transformation is defined as:
        u = sum_{i+j<=degree} a_{ij} * x^i * y^j
        v = sum_{i+j<=degree} b_{ij} * x^i * y^j
        
        where the coefficients are solved via least squares using the control points.
        
        Parameters:
        image: Input image.
        src_pts: numpy array of shape (N,2) with control points in the source image.
        dst_pts: numpy array of shape (N,2) with corresponding control points in the destination.
        degree: Degree of the polynomial transformation (allowed 1 to 6, default=3).
        status_label: If provided, a Qt widget where progress messages are displayed.
        
        Returns:
        The warped image.
        """
        h, w = image.shape[:2]
        
        # Function to build the design matrix for points given a degree.
        def build_design_matrix(pts, degree):
            # pts: (N,2) array, where each row is (x, y)
            N = pts.shape[0]
            terms = []
            x = pts[:, 0]
            y = pts[:, 1]
            # Loop over exponents i and j with i+j <= degree.
            for i in range(degree + 1):
                for j in range(degree + 1 - i):
                    terms.append((x**i) * (y**j))
            X = np.vstack(terms).T  # shape: (N, number_of_terms)
            return X

        # Build the design matrix for the control points.
        if status_label is not None:
            status_label.setText("Polynomial warp: Building design matrix for control points...")
            QApplication.processEvents()
        X = build_design_matrix(src_pts, degree)
        
        # Destination coordinates.
        U = dst_pts[:, 0]
        V = dst_pts[:, 1]
        
        if status_label is not None:
            status_label.setText("Polynomial warp: Solving for polynomial coefficients...")
            QApplication.processEvents()
        
        # Solve the least-squares problem.
        coeffs_u, _, _, _ = np.linalg.lstsq(X, U, rcond=None)
        coeffs_v, _, _, _ = np.linalg.lstsq(X, V, rcond=None)
        
        if status_label is not None:
            status_label.setText("Polynomial warp: Computing full mapping for image...")
            QApplication.processEvents()
        
        # Build a full grid of coordinates.
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        flat_x = grid_x.ravel()
        flat_y = grid_y.ravel()
        pts_full = np.vstack([flat_x, flat_y]).T  # shape: (h*w, 2)
        
        # Build design matrix for the full grid.
        X_full = build_design_matrix(pts_full, degree)
        
        # Evaluate the polynomial mappings.
        map_u = np.dot(X_full, coeffs_u).reshape(h, w).astype(np.float32)
        map_v = np.dot(X_full, coeffs_v).reshape(h, w).astype(np.float32)
        
        if status_label is not None:
            status_label.setText("Polynomial warp: Full mapping computed. Remapping image...")
            QApplication.processEvents()
        
        # Remap the image.
        warped = cv2.remap(image, map_u, map_v, interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT)
        
        if status_label is not None:
            status_label.setText("Polynomial warp: Image warped.")
            QApplication.processEvents()
        
        return warped



    def stretch_for_display(self, arr):
        """
        Uses your global stretch_mono_image or stretch_color_image to produce
        a display-ready 8-bit image. For color images, uses unlinked stretching.
        """
        arr = arr.astype(np.float32)

        # Decide if it's mono or color based on shape
        if arr.ndim == 3 and arr.shape[2] == 3:
            # Color image => use stretch_color_image with unlinked stretching
            stretched = stretch_color_image(
                image=arr,
                target_median=0.25,   # Adjust if you prefer a different default
                linked=False,         # "unlinked" mode
                normalize=True,       # Ensures final values are in [0,1]
                apply_curves=False,   # Adjust if needed
                curves_boost=0.0
            )
        else:
            # Mono image => use stretch_mono_image
            stretched = stretch_mono_image(
                image=arr,
                target_median=0.25,
                normalize=True,
                apply_curves=False,
                curves_boost=0.0
            )

        # Convert [0,1] => [0,255]
        disp = (stretched * 255.0).clip(0, 255).astype(np.uint8)
        return disp


    def detect_stars(self, image2d, max_stars=50):
        # Retrieve user-defined values for sigma and fwhm.
        sigma_val = self.settings.value("mosaic/star_sigma", 3.0, type=float)
        fwhm_val = self.settings.value("mosaic/star_fwhm", 3.0, type=float)
        
        mean_val, median_val, std_val = sigma_clipped_stats(image2d, sigma=3.0)
        # Use the user-defined fwhm and scale the threshold by the standard deviation.
        daofind = DAOStarFinder(threshold=sigma_val * std_val, fwhm=fwhm_val)
        sources = daofind(image2d - median_val)
        if sources is None or len(sources) == 0:
            return []
        x_coords = sources['xcentroid'].data
        y_coords = sources['ycentroid'].data
        flux = sources['flux'].data
        # Sort stars by brightness (flux) and select the top ones.
        sorted_indices = np.argsort(-flux)
        top_indices = sorted_indices[:max_stars]
        stars = [(x_coords[i], y_coords[i], flux[i]) for i in top_indices]
        return stars


    def match_stars(self, new_stars, mosaic_stars, distance_thresh=10.0, flux_thresh=0.2):
        """
        Matches stars between two lists.
        new_stars and mosaic_stars should be lists of (x, y, flux).
        
        This version normalizes the flux values in each list by dividing by the median flux.
        
        distance_thresh: maximum distance (in pixels) allowed for a match.
        flux_thresh: allowed absolute difference in normalized flux.
                    For example, if set to 0.2, then the normalized flux difference must be less than 0.2.
                    
        Returns a list of matched pairs: [(new_star, mosaic_star), ...]
        """
        # If either list is empty, return an empty match list.
        if not new_stars or not mosaic_stars:
            return []
        
        # Compute median fluxes.
        new_fluxes = [s[2] for s in new_stars]
        mosaic_fluxes = [s[2] for s in mosaic_stars]
        new_median = np.median(new_fluxes) if new_fluxes else 1.0
        mosaic_median = np.median(mosaic_fluxes) if mosaic_fluxes else 1.0

        # Normalize the flux for each star.
        norm_new_stars = [(s[0], s[1], s[2] / new_median) for s in new_stars]
        norm_mosaic_stars = [(s[0], s[1], s[2] / mosaic_median) for s in mosaic_stars]

        matches = []
        for ns in norm_new_stars:
            best_match = None
            best_distance = float('inf')
            for ms in norm_mosaic_stars:
                dx = ns[0] - ms[0]
                dy = ns[1] - ms[1]
                dist = np.hypot(dx, dy)
                # Check spatial proximity.
                if dist < distance_thresh and dist < best_distance:
                    # Check if the normalized fluxes are similar.
                    # Here, flux_thresh is an absolute threshold on the difference.
                    if abs(ns[2] - ms[2]) < flux_thresh:
                        best_match = ms
                        best_distance = dist
            if best_match is not None:
                matches.append((ns, best_match))
        return matches

    def estimate_transform(self, source_stars, dest_stars):
        min_len = min(len(source_stars), len(dest_stars))
        if min_len < 3:
            return None
        src_pts = np.float32(source_stars[:min_len])
        dst_pts = np.float32(dest_stars[:min_len])
        matrix, inliers = cv2.estimateAffinePartial2D(
            src_pts, dst_pts,
            method=cv2.RANSAC, ransacReprojThreshold=3.0
        )
        if matrix is None:
            return None
        full_mat = np.eye(3, dtype=np.float32)
        full_mat[:2] = matrix
        return full_mat


    def compute_mosaic_bounding_box(self, wcs_items, reference_wcs):
        """
        Compute the mosaic bounding box in pixel coordinates relative to a shared WCS frame,
        properly accounting for rotation and orientation dynamically.
        """
        all_pixels = []

        for itm in wcs_items:
            wcs = itm["wcs"]
            H, W = itm["image"].shape[:2]  # Use only the height and width

            # Get image corner coordinates in world coordinates (RA/Dec)
            pixel_corners = np.array([
                [0, 0],         # Top-left
                [W - 1, 0],     # Top-right
                [0, H - 1],     # Bottom-left
                [W - 1, H - 1]  # Bottom-right
            ])

            # Convert pixel to world coordinates (RA, Dec)
            world_coords = np.column_stack(wcs.pixel_to_world_values(pixel_corners[:, 0], pixel_corners[:, 1]))

            # Convert RA/Dec to pixel coordinates in the reference WCS
            sky_coords = SkyCoord(ra=world_coords[:, 0] * u.deg, dec=world_coords[:, 1] * u.deg, frame='icrs')
            pixel_coords = skycoord_to_pixel(sky_coords, reference_wcs)

            # Ensure we're only using the first two values (x, y)
            all_pixels.append(np.column_stack(pixel_coords[:2]))

        # Stack all pixel coordinates and compute bounding box
        all_pixels = np.vstack(all_pixels)

        min_x, max_x = np.min(all_pixels[:, 0]), np.max(all_pixels[:, 0])
        min_y, max_y = np.min(all_pixels[:, 1]), np.max(all_pixels[:, 1])

        # **Determine whether the mosaic is wider or taller dynamically**
        width = max_x - min_x
        height = max_y - min_y

        print(f"Detected Bounding Box (X={min_x} to {max_x}, Y={min_y} to {max_y})")
        print(f"Calculated Mosaic Size: Width={width}, Height={height}")
        self.status_label.setText(f"Detected Bounding Box: X={min_x} to {max_x}, Y={min_y} to {max_y}")
        self.status_label.setText(f"Calculated Mosaic Size: Width={width}, Height={height}")
        QApplication.processEvents()

        return int(np.floor(min_x)), int(np.floor(min_y)), int(np.ceil(max_x)), int(np.ceil(max_y))

    def save_mosaic_to_new_view(self):
        """
        Finalize and create a brand-new document/view in SASpro.
        """
        finalized = self.finalize_mosaic()
        if finalized is None:
            QMessageBox.information(self, "Mosaic Master", "No mosaic to save.")
            return

        img, meta = finalized

        if not self._docman:
            QMessageBox.warning(self, "Mosaic Master", "No document manager available.")
            return

        title = self._proposed_mosaic_title()

        try:
            if hasattr(self._docman, "open_array") and callable(self._docman.open_array):
                newdoc = self._docman.open_array(img, metadata=meta, title=title)
            elif hasattr(self._docman, "open_numpy") and callable(self._docman.open_numpy):
                newdoc = self._docman.open_numpy(img, metadata=meta, title=title)
            else:
                # very old API
                newdoc = self._docman.create_document(image=img, metadata=meta, name=title)

            # show it in a subwindow if the host provides the helper
            parent = self.parent() if hasattr(self, "parent") else None
            if parent and hasattr(parent, "_spawn_subwindow_for"):
                parent._spawn_subwindow_for(newdoc)

            QMessageBox.information(self, "Mosaic Master", "Created a new view for the mosaic.")
            self.accept()

        except Exception as e:
            QMessageBox.critical(self, "Mosaic Master", f"Failed to create new view:\n{e}")

    def _proposed_mosaic_title(self) -> str:
        return "Mosaic"

    def finalize_mosaic(self):
        """
        Prepare the final mosaic (and its WCS metadata) and return (image, metadata).
        Does not push anywhere. Returns None on failure.
        """
        if self.final_mosaic is None:
            print("No mosaic to finalize.")
            return None

        # Build metadata header
        if not self.wcs_metadata or not any(self.wcs_metadata.values()):
            print("WCS metadata not available; creating minimal header.")
            is_mono = (self.final_mosaic.ndim == 2 or
                    (self.final_mosaic.ndim == 3 and self.final_mosaic.shape[2] == 1))
            minimal_header = self.create_minimal_fits_header(self.final_mosaic, is_mono)
            meta = dict(minimal_header)
        else:
            meta = dict(self.wcs_metadata)

        # Add helpful tags
        is_mono = (self.final_mosaic.ndim == 2 or
                (self.final_mosaic.ndim == 3 and self.final_mosaic.shape[2] == 1))
        meta["step_name"] = "Mosaic Master"
        meta["is_mono"] = bool(is_mono)

        # Keep the array as-is (2D mono or 3-channel)
        return self.final_mosaic, meta

    def stretch_image(self, image):
        """
        Perform an unlinked linear stretch on the image.
        Each channel is stretched independently by subtracting its own minimum,
        recording its own median, and applying the stretch formula.
        Returns the stretched image.
        """
        was_single_channel = False  # Flag to check if image was single-channel

        # Check if the image is single-channel
        if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
            was_single_channel = True
            image = np.stack([image] * 3, axis=-1)  # Convert to 3-channel by duplicating

        # Ensure the image is a float32 array for precise calculations and writable
        image = image.astype(np.float32).copy()

        # Initialize lists to store per-channel minima and medians
        self.stretch_original_mins = []
        self.stretch_original_medians = []

        # Initialize stretched_image as a copy of the input image
        stretched_image = image.copy()

        # Define the target median for stretching
        target_median = 0.08

        # Apply the stretch for each channel independently
        for c in range(3):
            # Record the minimum of the current channel
            channel_min = np.min(stretched_image[..., c])
            self.stretch_original_mins.append(channel_min)

            # Subtract the channel's minimum to shift the image
            stretched_image[..., c] -= channel_min

            # Record the median of the shifted channel
            channel_median = np.median(stretched_image[..., c])
            self.stretch_original_medians.append(channel_median)

            if channel_median != 0:
                numerator = (channel_median - 1) * target_median * stretched_image[..., c]
                denominator = (
                    channel_median * (target_median + stretched_image[..., c] - 1)
                    - target_median * stretched_image[..., c]
                )
                # To avoid division by zero
                denominator = np.where(denominator == 0, 1e-6, denominator)
                stretched_image[..., c] = numerator / denominator
            else:
                print(f"Channel {c} - Median is zero. Skipping stretch.")

        # Clip stretched image to [0, 1] range
        stretched_image = np.clip(stretched_image, 0.0, 1.0)

        # Store stretch parameters
        self.was_single_channel = was_single_channel

        return stretched_image


    def unstretch_image(self, image):
        """
        Undo the unlinked linear stretch to return the image to its original state.
        Each channel is unstretched independently by reverting the stretch formula
        using the stored medians and adding back the individual channel minima.
        Returns the unstretched image.
        """
        original_mins = self.stretch_original_mins
        original_medians = self.stretch_original_medians
        was_single_channel = self.was_single_channel

        # Ensure the image is a float32 array for precise calculations and writable
        image = image.astype(np.float32).copy()

        # If the image is 2D, treat it as a single channel.
        if image.ndim == 2:
            # Process as a single channel:
            channel_median = np.median(image)
            original_median = original_medians[0]
            original_min = original_mins[0]

            if channel_median != 0 and original_median != 0:
                numerator = (channel_median - 1) * original_median * image
                denominator = channel_median * (original_median + image - 1) - original_median * image
                # To avoid division by zero
                denominator = np.where(denominator == 0, 1e-6, denominator)
                image = numerator / denominator
            else:
                print("Channel median or original median is zero. Skipping unstretch.")

            # Add back the original minimum
            image += original_min

            # Clip to [0, 1]
            image = np.clip(image, 0, 1)
            # Optionally, if you want to keep it 2D (since it was originally mono), just return image.
            # If you want to convert to a 3-channel image for display later, you can do that later.
            return image

        # Otherwise, if the image is 3D, process each channel
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
                # To avoid division by zero
                denominator = np.where(denominator == 0, 1e-6, denominator)
                image[..., c] = numerator / denominator
            else:
                print(f"Channel {c} - Median or original median is zero. Skipping unstretch.")

            # Add back the channel's original minimum
            image[..., c] += original_min

        # Clip to [0, 1] range
        image = np.clip(image, 0, 1)

        # If the image was originally single-channel but has 3 dimensions now, convert it back.
        if was_single_channel and image.ndim == 3:
            image = np.mean(image, axis=2, keepdims=True)

        return image



    def _save_astap_exe_to_settings(self, path: str) -> None:
        path = os.path.normpath(os.path.expanduser(os.path.expandvars(path)))
        self.astap_exe = path
        self.settings.setValue("paths/astap", path)
        self.settings.sync()

    def _load_astrometry_api_key(self) -> str:
        # primary source: QSettings (matches SettingsDialog)
        key = self.settings.value("api/astrometry_key", "", type=str) or ""
        if key:
            return key
        # optional fallback to any app-level file helpers you already had
        try:
            k = load_api_key()
            return k or ""
        except Exception:
            return ""

    def _save_astrometry_api_key(self, key: str) -> None:
        key = (key or "").strip()
        self.settings.setValue("api/astrometry_key", key)
        self.settings.sync()
        # keep old helper in sync if you want
        try:
            save_api_key(key)
        except Exception:
            pass


    # ---------- Blind Solve via Astrometry.net ----------
    def perform_blind_solve(self, item):
        """
        Blind solve with Astrometry.net, construct WCS, keep SIP, and (optionally) update on-disk FITS.
        """
        while True:
            self.status_label.setText("Status: Logging in to Astrometry.net...")
            QApplication.processEvents()
            api_key = self._get_astrometry_api_key()
            if not api_key:
                api_key, ok = QInputDialog.getText(self, "Enter API Key", "Please enter your Astrometry.net API key:")
                if ok and api_key:
                    self.settings.setValue("api/astrometry_key", api_key)
                    self.settings.sync()
                else:
                    QMessageBox.warning(self, "API Key Required", "Blind solve canceled (no API key).")
                    return None

            session_key = self.login_to_astrometry(api_key)
            if session_key is None:
                if QMessageBox.question(self, "Login Failed",
                                        "Could not log in to Astrometry.net. Try again?",
                                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes:
                    continue
                else:
                    return None

            self.status_label.setText("Status: Uploading image to Astrometry.net...")
            QApplication.processEvents()

            # Determine the file extension of the original image
            ext = os.path.splitext(str(item.get("path","")))[1].lower()
            if ext not in ('.fits', '.fit'):
                tmp = tempfile.NamedTemporaryFile(suffix=".fit", delete=False)
                tmp.close()
                minimal_header = generate_minimal_fits_header(item["image"])
                save_image(
                    img_array=item["image"],
                    filename=tmp.name,
                    original_format="fit",
                    bit_depth="16-bit",
                    original_header=minimal_header,
                    is_mono=item.get("is_mono", False)
                )
                upload_path = tmp.name
            else:
                upload_path = item["path"]

            subid = self.upload_image_to_astrometry(upload_path, session_key)
            if not subid:
                if QMessageBox.question(self, "Upload Failed",
                                        "Image upload failed or no subid returned. Try again?",
                                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes:
                    continue
                else:
                    return None

            self.status_label.setText("Status: Waiting for job ID...")
            QApplication.processEvents()
            job_id = self.poll_submission_status(subid)
            if not job_id:
                if QMessageBox.question(self, "Blind Solve Failed",
                                        "Failed to retrieve job ID from Astrometry.net. Try again?",
                                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes:
                    continue
                else:
                    return None

            self.status_label.setText("Status: Retrieving calibration data...")
            QApplication.processEvents()
            calibration_data = self.poll_calibration_data(job_id)
            if not calibration_data:
                if QMessageBox.question(self, "Blind Solve Failed",
                                        "Calibration data did not arrive from Astrometry.net. Try again?",
                                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes:
                    continue
                else:
                    return None

            # Clean up tmp if we created one
            if ext not in ('.fits', '.fit', '.tif', '.tiff'):
                try:
                    os.remove(upload_path)
                except Exception:
                    pass

            break  # success

        # Build + normalize header, keep SIP, and build WCS with relax=True
        wcs_header = self.construct_wcs_header(calibration_data, item["image"].shape)
        wcs_header = self._normalize_wcs_header(wcs_header, item["image"].shape)
        item["wcs"] = self._build_wcs(wcs_header, item["image"].shape)

        dest = str(item.get("path", ""))
        if dest.lower().endswith((".fits", ".fit")) and not self._is_view_path(dest):
            self.update_fits_with_wcs(dest, calibration_data, wcs_header)
        else:
            print("Blind solve succeeded; not updating file on disk (view or non-FITS source).")

        return wcs_header


    def login_to_astrometry(self, api_key):
        url = ASTROMETRY_API_URL + "login"
        data = {'request-json': json.dumps({"apikey": api_key})}
        response = robust_api_request("POST", url, data=data, prompt_on_failure=True)
        if response and response.get("status") == "success":
            return response["session"]
        print("Login failed after multiple attempts.")
        QMessageBox.critical(self, "Login Failed", "Could not log in to Astrometry.net. Check your API key or internet connection.")
        return None

    def upload_image_to_astrometry(self, image_path, session_key):
        url = ASTROMETRY_API_URL + "upload"
        with open(image_path, 'rb') as f:
            files = {'file': f}
            data = {
                'request-json': json.dumps({
                    "publicly_visible": "y",
                    "allow_modifications": "d",
                    "session": session_key,
                    "allow_commercial_use": "d"
                })
            }
            response = robust_api_request("POST", url, data=data, files=files)
        if response and response.get("status") == "success":
            return response["subid"]
        QMessageBox.critical(self, "Upload Failed", "Image upload failed after multiple attempts.")
        return None

    def poll_submission_status(self, subid):
        url = ASTROMETRY_API_URL + f"submissions/{subid}"
        for attempt in range(90):  # up to ~15 minutes
            response = robust_api_request("GET", url)
            if response:
                jobs = response.get("jobs", [])
                if jobs and jobs[0] is not None:
                    return jobs[0]
            print(f"Polling attempt {attempt+1}: Job ID not ready yet.")
            time.sleep(10)
        QMessageBox.critical(self, "Blind Solve Failed", "Failed to retrieve job ID from Astrometry.net after multiple attempts.")
        return None

    def poll_calibration_data(self, job_id):
        url = ASTROMETRY_API_URL + f"jobs/{job_id}/calibration/"
        for attempt in range(90):
            response = robust_api_request("GET", url)
            if response and 'ra' in response and 'dec' in response:
                print("Calibration data retrieved:", response)
                return response
            print(f"Calibration data not available yet (attempt {attempt+1})")
            time.sleep(10)
        QMessageBox.critical(self, "Blind Solve Failed", "Calibration data did not complete in the expected timeframe.")
        return None

    def construct_wcs_header(self, calibration_data, image_shape):
        h = Header()
        h['CTYPE1'] = 'RA---TAN'
        h['CTYPE2'] = 'DEC--TAN'
        h['CRPIX1'] = image_shape[1] / 2
        h['CRPIX2'] = image_shape[0] / 2
        h['CRVAL1'] = calibration_data['ra']
        h['CRVAL2'] = calibration_data['dec']
        scale = calibration_data['pixscale'] / 3600.0  # degrees/pixel
        orientation = math.radians(calibration_data['orientation'])
        h['CD1_1'] = -scale * np.cos(orientation)
        h['CD1_2'] = scale * np.sin(orientation)
        h['CD2_1'] = -scale * np.sin(orientation)
        h['CD2_2'] = -scale * np.cos(orientation)
        h['RADECSYS'] = 'ICRS'
        h['WCSAXES'] = 2
        print("Generated WCS header from calibration data.")
        return h

    def update_fits_with_wcs(self, filepath, calibration_data, wcs_header):
        if not filepath.lower().endswith(('.fits','.fit')):
            print("Not a FITS, skipping WCS update.")
            return
        try:
            with fits.open(filepath, mode='update') as hdul:
                hdr = hdul[0].header
                if 'NAXIS3' in hdr:
                    del hdr['NAXIS3']
                hdr['NAXIS'] = 2
                hdr['CTYPE1'] = 'RA---TAN'
                hdr['CTYPE2'] = 'DEC--TAN'
                hdr['CRVAL1'] = calibration_data['ra']
                hdr['CRVAL2'] = calibration_data['dec']
                # Determine H and W based on the data's dimensionality.
                if hdul[0].data.ndim == 3:
                    # Assume data are stored as (channels, height, width)
                    _, H, W = hdul[0].data.shape
                else:
                    H, W = hdul[0].data.shape[:2]
                hdr['CRPIX1'] = W/2.0
                hdr['CRPIX2'] = H/2.0
                scale = calibration_data['pixscale']/3600.0
                orientation = math.radians(calibration_data.get('orientation', 0.0))
                hdr['CD1_1'] = -scale * np.cos(orientation)
                hdr['CD1_2'] = scale * np.sin(orientation)
                hdr['CD2_1'] = -scale * np.sin(orientation)
                hdr['CD2_2'] = -scale * np.cos(orientation)
                hdr['WCSAXES'] = 2
                hdr['RADECSYS'] = 'ICRS'
                hdul.flush()
                print("WCS updated in FITS.")
            # Re-open to verify changes:
            with fits.open(filepath) as hdul_verify:
                print("Updated header keys:", hdul_verify[0].header.keys())
        except Exception as e:
            print(f"Error updating FITS with WCS: {e}")

    # ---------- Blind Solve via ASTAP ----------
    def attempt_astap_solve(self, item):
        """
        Attempt to plate-solve the image using ASTAP.
        Returns a solved header (as a dict) on success or None on failure.
        """
        # 1) Normalize the image (using your stretch_image).
        normalized_image = self.stretch_image(item["image"])
        
        # 2) Save normalized image to a temporary FITS file for ASTAP.
        try:
            tmp_path = self.save_temp_fits_image(normalized_image, item["path"])
        except Exception as e:
            print("Failed to save temporary FITS file:", e)
            return None

        # 3) Run ASTAP on the temporary file.
        process = QProcess(self)
        args = ["-f", tmp_path, "-r", "179", "-fov", "0", "-z", "0", "-wcs", "-sip"]
        print("Running ASTAP with arguments:", args)
        process.start(self.astap_exe, args)
        if not process.waitForStarted(5000):
            print("Failed to start ASTAP process:", process.errorString())
            os.remove(tmp_path)
            return None
        if not process.waitForFinished(300000):  # wait up to 5 minutes
            print("ASTAP process timed out.")
            os.remove(tmp_path)
            return None

        exit_code = process.exitCode()
        stdout = process.readAllStandardOutput().data().decode()
        stderr = process.readAllStandardError().data().decode()
        print("ASTAP exit code:", exit_code)
        print("ASTAP STDOUT:\n", stdout)
        print("ASTAP STDERR:\n", stderr)

        if exit_code != 0:
            try:
                os.remove(tmp_path)
            except Exception as e:
                print("Error removing temporary file:", e)
            return None

        # 4) Retrieve updated header from the temporary file.
        try:
            with fits.open(tmp_path, memmap=False) as hdul:
                solved_header = dict(hdul[0].header)
            # Remove some extraneous keywords
            solved_header.pop("COMMENT", None)
            solved_header.pop("HISTORY", None)
        except Exception as e:
            print("Error reading solved header:", e)
            os.remove(tmp_path)
            return None

        # 5) Merge .wcs file (if ASTAP wrote one) into the solved_header.
        wcs_path = os.path.splitext(tmp_path)[0] + ".wcs"
        if os.path.exists(wcs_path):
            try:
                with open(wcs_path, "r") as f:
                    content = f.read()
                pattern = r"(\w+)\s*=\s*('?[^/']*'?)[\s/]"
                for match in re.finditer(pattern, content):
                    key = match.group(1).strip().upper()
                    val = match.group(2).strip()
                    if val.startswith("'") and val.endswith("'"):
                        val = val[1:-1].strip()
                    solved_header[key] = val
            except Exception as e:
                print("Error reading .wcs file:", e)
            finally:
                try:
                    os.remove(wcs_path)
                except Exception as e:
                    print("Error removing .wcs file:", e)

        # Remove the END keyword if present
        solved_header.pop("END", None)
        # Remove any unneeded keywords
        for keyword in ["RANGE_LOW", "RANGE_HIGH", "HISTORY"]:
            solved_header.pop(keyword, None)

        # --- Ensure required WCS keys are present ---
        if "CTYPE1" not in solved_header or not solved_header["CTYPE1"].strip():
            solved_header["CTYPE1"] = "RA---TAN"
        if "CTYPE2" not in solved_header or not solved_header["CTYPE2"].strip():
            solved_header["CTYPE2"] = "DEC--TAN"
        if "RADECSYS" not in solved_header:
            solved_header["RADECSYS"] = "ICRS"
        if "WCSAXES" not in solved_header:
            solved_header["WCSAXES"] = 2

        # Convert known WCS keys to float or int
        expected_float_keys = {"CRPIX1", "CRPIX2", "CRVAL1", "CRVAL2", "CDELT1", "CDELT2", "CD1_1", "CD1_2", "CD2_1", "CD2_2"}
        expected_int_keys = {"NAXIS", "WCSAXES"}

        for key in expected_float_keys:
            if key in solved_header:
                try:
                    solved_header[key] = float(solved_header[key])
                except ValueError:
                    print(f"Warning: Could not convert {key}='{solved_header[key]}' to float.")

        for key in expected_int_keys:
            if key in solved_header:
                try:
                    solved_header[key] = int(float(solved_header[key]))
                except ValueError:
                    print(f"Warning: Could not convert {key}='{solved_header[key]}' to int.")

        dest = str(item.get("path", ""))

        ok_to_overwrite = (
            bool(dest)
            and not self._is_view_path(dest)
            and dest.lower().endswith((".fits", ".fit"))
        )

        if ok_to_overwrite:
            try:
                save_image(
                    img_array=normalized_image,
                    filename=dest,
                    original_format="fit",
                    bit_depth="32",
                    original_header=solved_header,
                    is_mono=False
                )
                print(f"✅ Updated FITS header with full WCS solution for {dest}.")
            except Exception as e:
                print("Error saving updated FITS file using save_image():", e)
                # don't re-raise; we can still continue in-memory
        else:
            print("Skipping on-disk write (source is a view or not a FITS path).")

        # cleanup temp files as you already do...
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
        is_fits_path = isinstance(image_path, str) and image_path.lower().endswith((".fits", ".fit"))
        if is_fits_path and not self._is_view_path(image_path):
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

        image = image.astype(np.float32).copy()
        stretched_image = image.copy()
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

        image = image.astype(np.float32).copy()

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
