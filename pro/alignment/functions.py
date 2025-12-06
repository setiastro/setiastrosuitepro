from __future__ import annotations

import os
import math
import random
import sys
import gc
import os as _os
import threading as _threading
import ctypes as _ctypes
import multiprocessing

# ---------------------------------------------------------------------
# Thread control
N = str(max(1, min( (os.cpu_count() or 8), 32 )))
# We set these in core or here? Better safe than sorry.
os.environ.setdefault("OMP_NUM_THREADS", N)
os.environ.setdefault("OPENBLAS_NUM_THREADS", N)
os.environ.setdefault("MKL_NUM_THREADS", N)
os.environ.setdefault("NUMEXPR_NUM_THREADS", N)
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", N)  # macOS Accelerate
try:
    import cv2
    cv2.setNumThreads(int(N))
except Exception:
    pass

from concurrent.futures import ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED
from itertools import combinations
from typing import Callable, Iterable, Tuple
import tempfile
import traceback
import requests
import numpy as np

import astroalign
import sep
import re
import warnings
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

from pro.memory_utils import smart_zeros, should_use_memmap
from legacy.image_manager import load_image, save_image
try:
    from imageops.stretch import stretch_mono_image, stretch_color_image
except Exception:
    stretch_mono_image = None
    stretch_color_image = None

try:
    from legacy.numba_utils import fast_star_detect
except Exception:
    fast_star_detect = None

from legacy.numba_utils import (
    rescale_image_numba,
    flip_horizontal_numba,
    flip_vertical_numba,
    rotate_90_clockwise_numba,
    rotate_90_counterclockwise_numba,
    rotate_180_numba,
    invert_image_numba,
)
from pro.abe import _generate_sample_points as abe_generate_sample_points

# CORE IMPORTS
from pro.alignment.core import (
    _gray2d,
    aa_find_transform_with_backoff,
    compute_pairs_astroalign,
    _cap_points
)

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

        # Star detection / solve limits
        "det_sigma":   _get("det_sigma", 12.0, float),   # try 8–12 in dense fields
        "limit_stars": _get("limit_stars", 500, int),    # 500–1500 typical
        "minarea":     _get("minarea", 10, int),

        # NEW: per-job timeout (seconds)
        "timeout_per_job_sec": _get("timeout_per_job_sec", 300, int),
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

# _gray2d and aa_find_transform_with_backoff MOVED TO core.py


def _warp_like_ref(target_img: np.ndarray, M_2x3: np.ndarray, ref_shape_hw: tuple[int,int]) -> np.ndarray:
    H, W = ref_shape_hw
    if target_img.ndim == 2:
        if not target_img.flags['C_CONTIGUOUS']:
            target_img = np.ascontiguousarray(target_img)
        return cv2.warpAffine(target_img, M_2x3, (W, H),
                               flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    chs = []
    for i in range(target_img.shape[2]):
        ch = target_img[..., i]
        if not ch.flags['C_CONTIGUOUS']:
            ch = np.ascontiguousarray(ch)
        chs.append(cv2.warpAffine(ch, M_2x3, (W, H),
                           flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=0))
    return np.stack(chs, axis=2)

def run_star_alignment_headless(mw, target_sw, preset: dict) -> bool:
    """
    Headless align the TARGET view (target_sw) to a chosen REFERENCE, as per preset.
    """
    try:
        # ---------- resolve target doc & image ----------
        target_doc = _doc_from_sw(target_sw) if target_sw else None
        if target_doc is None or getattr(target_doc, "image", None) is None:
            return False
        tgt_img = np.ascontiguousarray(np.asarray(target_doc.image, dtype=np.float32))

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
            ref_img = np.ascontiguousarray(np.asarray(ref_doc.image, dtype=np.float32))
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
            ref_img = np.ascontiguousarray(np.asarray(ref_doc.image, dtype=np.float32))
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
            ref_img = np.ascontiguousarray(np.asarray(act_doc.image, dtype=np.float32))
            try:
                ref_name = act_doc.display_name() if callable(getattr(act_doc, "display_name", None)) else (getattr(act_doc, "title", None) or "Reference")
            except Exception:
                pass

        # ---------- downsample (optional) ----------
        ref_gray = _gray2d(ref_img)
        tgt_gray = _gray2d(tgt_img)
        ref_small, tgt_small = ref_gray, tgt_gray
        # ---------- find transform ----------
        transform_obj, _pts = aa_find_transform_with_backoff(tgt_small, ref_small)
        M2 = np.array(transform_obj.params[0:2, :], dtype=np.float64)  # keep full precision

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


