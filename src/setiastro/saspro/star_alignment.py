#saspro.star_alignment.py
from __future__ import annotations

import os
import math
import random
import sys
from PyQt6.QtCore import QByteArray

def _qs_raw(settings, key, default=None):
    try:
        return settings.value(key, default)
    except Exception:
        return default

def _qs_text(v):
    """Best-effort: convert QByteArray/bytes -> str, preserve strings, else None."""
    try:
        if isinstance(v, QByteArray):
            # PyQt6-safe
            v = bytes(v.data()).decode("utf-8", "ignore")
        elif isinstance(v, (bytes, bytearray)):
            v = bytes(v).decode("utf-8", "ignore")
        if isinstance(v, str):
            return v.strip()
    except Exception:
        pass
    return None

def _purge(settings, key):
    try:
        settings.remove(key)
    except Exception:
        pass

def qs_int(settings, key, default=0, *, purge_bad=True):
    v = _qs_raw(settings, key, default)
    try:
        if isinstance(v, bool):
            return int(v)
        if isinstance(v, int):
            return v
        if isinstance(v, float):
            return int(v)

        s = _qs_text(v)
        if s is not None:
            if s == "":
                return default
            s = s.replace(",", ".")   # locale fix
            return int(float(s))      # handles "3.0"

        # Last resort: try numeric conversion, but don't guess on random objects
        return int(v)
    except Exception:
        if purge_bad:
            _purge(settings, key)
        return default

def qs_float(settings, key, default=0.0, *, purge_bad=True):
    v = _qs_raw(settings, key, default)
    try:
        if isinstance(v, bool):
            return float(int(v))
        if isinstance(v, (int, float)):
            return float(v)

        s = _qs_text(v)
        if s is not None:
            if s == "":
                return default
            s = s.replace(",", ".")  # locale fix
            return float(s)

        return float(v)
    except Exception:
        if purge_bad:
            _purge(settings, key)
        return default

def qs_bool(settings, key, default=False, *, purge_bad=True):
    v = _qs_raw(settings, key, default)
    try:
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return bool(int(v))

        s = _qs_text(v)
        if s is not None:
            s = s.lower()
            if s in ("1", "true", "yes", "on"):
                return True
            if s in ("0", "false", "no", "off"):
                return False
            return default

        # Do NOT guess truthiness of random objects; that's how weirdness spreads
        return default
    except Exception:
        if purge_bad:
            _purge(settings, key)
        return default


# ---------------------------------------------------------------------
# Executor helper: avoid ProcessPool in frozen (PyInstaller) builds
# ---------------------------------------------------------------------
_IS_FROZEN = bool(getattr(sys, "frozen", False))

def _make_executor(max_workers: int):
    """
    Return an appropriate executor.

    - In frozen builds (PyInstaller), we MUST avoid ProcessPoolExecutor
      because each worker spawns a full copy of the EXE (extra SASpro windows).
    - In dev (non-frozen), you can still use processes if you want. For
      now we keep it simple and always use threads – safer everywhere.
    """
    # If you want to keep processes in dev, uncomment the if-block:
    # if not _IS_FROZEN:
    #     return ProcessPoolExecutor(max_workers=max_workers)
    return ThreadPoolExecutor(max_workers=max_workers)


import gc  # For explicit memory cleanup after heavy operations
import os as _os
import threading as _threading
import ctypes as _ctypes
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

from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
    wait,
    FIRST_COMPLETED,
)
from itertools import combinations
from typing import Callable, Iterable, Tuple
import tempfile
import traceback
import requests
import numpy as np

from setiastro.saspro import astroalign
import sep
sep.set_extract_pixstack(20000000)
import re
import warnings
import json
import time
from scipy.spatial import KDTree, Delaunay
from astropy.stats import sigma_clipped_stats
from astropy.io.fits import Header

from astropy.table import vstack
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, get_body, get_sun
from astropy.wcs import FITSFixedWarning
import astropy.units as u
from astropy.wcs import WCS
from astropy.wcs.utils import skycoord_to_pixel
from reproject import reproject_interp
from setiastro.saspro.blink_comparator_pro import CustomDoubleSpinBox, CustomSpinBox
import numpy.ma as ma
from PyQt6.QtCore import Qt, QThread, QRunnable, QThreadPool, pyqtSignal, QObject, QTimer, QProcess, QPoint, QEvent, QSettings
from PyQt6.QtGui import QImage, QPixmap, QIcon, QMovie
from PyQt6.QtWidgets import (
    QDialog, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QGroupBox, QAbstractItemView, QListWidget, QInputDialog, QApplication, QProgressBar, QProgressDialog, 
    QRadioButton, QFileDialog, QComboBox, QMessageBox, QTextEdit, QDialogButtonBox, QTreeWidget,QCheckBox, QFormLayout, QListWidgetItem, QScrollArea, QTreeWidgetItem, QSpinBox, QDoubleSpinBox
)

from skimage.transform import warp, PolynomialTransform 

# Memory management utilities
from setiastro.saspro.memory_utils import smart_zeros, should_use_memmap

# I/O & stretch (same stack we used for Plate Solver)
from setiastro.saspro.legacy.image_manager import load_image, save_image
try:
    from setiastro.saspro.imageops.stretch import stretch_mono_image, stretch_color_image
except Exception:
    stretch_mono_image = None
    stretch_color_image = None

# Optional fast star detector; fall back gracefully if not present
try:
    from setiastro.saspro.legacy.numba_utils import fast_star_detect  # your optimized detector, if available
except Exception:
    fast_star_detect = None

from setiastro.saspro.legacy.numba_utils import (
    rescale_image_numba,
    flip_horizontal_numba,
    flip_vertical_numba,
    rotate_90_clockwise_numba,
    rotate_90_counterclockwise_numba,
    rotate_180_numba,
    invert_image_numba,
)
from setiastro.saspro.abe import _generate_sample_points as abe_generate_sample_points
from setiastro.saspro.widgets.themed_buttons import themed_toolbtn

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
        "model":       model,
        "max_cp":      _get("max_cp", 250, int),
        "downsample":  _get("downsample", 3, int),
        "h_reproj":    _get("h_reproj", 3.0, float),
        "det_sigma":   _get("det_sigma", 12.0, float),
        "limit_stars": _get("limit_stars", 500, int),
        "minarea":     _get("minarea", 10, int),
        "timeout_per_job_sec": _get("timeout_per_job_sec", 300, int),
        # Hot pixel rejection
        "min_fwhm":        _get("min_fwhm", 1.2, float),
        "max_ellipticity": _get("max_ellipticity", 0.6, float),
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

# ---------------------------------------------------------------------
# WCS-based alignment (no stars): reproject TARGET onto the REFERENCE grid
# using each image's full WCS (SIP included), exactly like Mosaic Master's
# exact remap. Handles arbitrary scale / rotation / flip / FOV as long as
# both images carry a usable plate solve.
# ---------------------------------------------------------------------

def _wcs_from_header(header):
    """Sanitize + build a usable WCS (SIP kept) or None. The header parsing
    lives in mosaic_master; import it lazily so there's no import cycle
    (mosaic_master imports qs_*/ASTROMETRY_API_URL back from here)."""
    if header is None:
        return None
    try:
        from setiastro.saspro.mosaic_master import get_wcs_from_header
    except Exception:
        return None
    try:
        return get_wcs_from_header(header)
    except Exception:
        return None


def _wcs_align_footprint_bbox(tgt_shape, tgt_wcs, ref_wcs, out_shape, pad=16, n=96):
    """Bounding box (x0,y0,x1,y1) of the target inside the reference grid.
    Target-boundary pixels -> world (forward SIP) -> reference pixels."""
    H, W = int(out_shape[0]), int(out_shape[1])
    h, w = int(tgt_shape[0]), int(tgt_shape[1])
    xs = np.linspace(0, w - 1, n); ys = np.linspace(0, h - 1, n)
    px = np.concatenate([xs, xs, np.zeros(n), np.full(n, w - 1)])
    py = np.concatenate([np.zeros(n), np.full(n, h - 1), ys, ys])
    try:
        ra, dec = tgt_wcs.all_pix2world(px, py, 0)   # forward SIP: exact
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dx, dy = ref_wcs.all_world2pix(ra, dec, 0)
    except Exception:
        return (0, 0, W, H)
    good = np.isfinite(dx) & np.isfinite(dy)
    if not good.any():
        return None
    x0 = max(0, int(np.floor(dx[good].min())) - pad)
    y0 = max(0, int(np.floor(dy[good].min())) - pad)
    x1 = min(W, int(np.ceil(dx[good].max())) + pad + 1)
    y1 = min(H, int(np.ceil(dy[good].max())) + pad + 1)
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def wcs_reproject_align(tgt_img, tgt_wcs, ref_wcs, out_shape,
                        tile=512, progress_cb=None, log=None):
    """
    Resample tgt_img onto the reference pixel grid so output pixel (x,y) sees
    the same sky as reference pixel (x,y). Inverse mapping, tiled, footprint-
    limited, with a forward round-trip check to drop diverged SIP solves.

    tgt_img   : HxW or HxWxC float
    tgt_wcs   : WCS of tgt_img (SIP kept)
    ref_wcs   : WCS defining the OUTPUT grid (the reference image's own WCS)
    out_shape : (H, W) of the reference image
    Returns float32 (H, W[,C]).
    """
    _log = log or (lambda *_a, **_k: None)

    H, W = int(out_shape[0]), int(out_shape[1])
    is_color = (tgt_img.ndim == 3)
    nch = tgt_img.shape[2] if is_color else 1
    sh, sw = tgt_img.shape[:2]
    src = np.ascontiguousarray(tgt_img.astype(np.float32, copy=False))

    dst = (np.zeros((H, W, nch), np.float32) if is_color
           else np.zeros((H, W), np.float32))

    box = _wcs_align_footprint_bbox(tgt_img.shape, tgt_wcs, ref_wcs, (H, W))
    if box is None:
        _log("[WCS-align] target footprint falls outside the reference frame.")
        return dst
    bx0, by0, bx1, by1 = box
    frac = ((bx1 - bx0) * (by1 - by0)) / float(max(1, W * H))
    _log(f"[WCS-align] footprint x[{bx0}:{bx1}] y[{by0}:{by1}] "
         f"({frac * 100:.1f}% of reference)")

    try:
        from astropy.wcs.utils import proj_plane_pixel_scales
        scale_deg = float(np.mean(proj_plane_pixel_scales(tgt_wcs)))
    except Exception:
        scale_deg = 1.0 / 3600.0
    tol_deg = 0.25 * scale_deg

    rejected = 0
    n_tiles = (max(1, -(-(by1 - by0) // tile)) *
               max(1, -(-(bx1 - bx0) // tile)))
    done_tiles = 0

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*all_world2pix.*")
        warnings.filterwarnings("ignore", message=".*failed to converge.*")

        for y0 in range(by0, by1, tile):
            y1 = min(y0 + tile, by1); hh = y1 - y0
            ys = np.arange(y0, y1, dtype=np.float64)[:, None]
            for x0 in range(bx0, bx1, tile):
                x1 = min(x0 + tile, bx1); ww = x1 - x0
                xs = np.arange(x0, x1, dtype=np.float64)[None, :]
                Xd = np.broadcast_to(xs, (hh, ww))
                Yd = np.broadcast_to(ys, (hh, ww))

                # reference pixel -> world (forward SIP, exact)
                ra, dec = ref_wcs.all_pix2world(Xd.ravel(), Yd.ravel(), 0)
                # world -> target pixel (iterative SIP inverse)
                sx, sy = tgt_wcs.all_world2pix(ra, dec, 0)

                mapx = sx.reshape(hh, ww).astype(np.float32)
                mapy = sy.reshape(hh, ww).astype(np.float32)

                bad = ~(np.isfinite(mapx) & np.isfinite(mapy))
                bad |= (mapx < -1.0) | (mapx > sw) | (mapy < -1.0) | (mapy > sh)

                ok = ~bad
                if ok.any():
                    rx = np.where(ok, mapx, 0.0).astype(np.float64).ravel()
                    ry = np.where(ok, mapy, 0.0).astype(np.float64).ravel()
                    ra2, dec2 = tgt_wcs.all_pix2world(rx, ry, 0)
                    dra = ((ra2 - ra + 180.0) % 360.0 - 180.0) * np.cos(np.radians(dec))
                    err = np.hypot(dra, dec2 - dec).reshape(hh, ww)
                    bad |= ~np.isfinite(err)
                    bad |= (err > tol_deg)

                if bad.any():
                    rejected += int(bad.sum())
                    mapx = np.where(bad, np.float32(-1.0), mapx)
                    mapy = np.where(bad, np.float32(-1.0), mapy)

                if is_color:
                    for c in range(nch):
                        dst[y0:y1, x0:x1, c] = cv2.remap(
                            src[..., c], mapx, mapy,
                            interpolation=cv2.INTER_LANCZOS4,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
                else:
                    dst[y0:y1, x0:x1] = cv2.remap(
                        src, mapx, mapy,
                        interpolation=cv2.INTER_LANCZOS4,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)

                done_tiles += 1
                if progress_cb is not None and (done_tiles % 8) == 0:
                    progress_cb(done_tiles / max(1, n_tiles))

    if rejected:
        tot = (bx1 - bx0) * (by1 - by0)
        _log(f"[WCS-align] rejected {rejected}/{tot} mapped pixels "
             f"({100.0 * rejected / max(1, tot):.1f}% of footprint)")
    return dst

def _warp_like_ref(target_img: np.ndarray, M_2x3: np.ndarray, ref_shape_hw: tuple[int,int]) -> np.ndarray:
    H, W = ref_shape_hw
    if target_img.ndim == 2:
        if not target_img.flags['C_CONTIGUOUS']:
            target_img = np.ascontiguousarray(target_img)
        return cv2.warpAffine(target_img, M_2x3, (W, H),
                               flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    # Optimization: If standard RGB/BGR (3 channels) or 4 channels, OpenCV handles it natively.
    # Note: OpenCV warpAffine support n-channel images, but typically 1, 3, or 4.
    C = target_img.shape[2]
    if C <= 4:
         if not target_img.flags['C_CONTIGUOUS']:
             target_img = np.ascontiguousarray(target_img)
         return cv2.warpAffine(target_img, M_2x3, (W, H),
                               flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # Fallback for >4 channels (e.g. hyperspectral or special stacks)
    chs = []
    for i in range(C):
        ch = target_img[..., i]
        if not ch.flags['C_CONTIGUOUS']:
            ch = np.ascontiguousarray(ch)
        chs.append(cv2.warpAffine(ch, M_2x3, (W, H),
                           flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=0))
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
    # Ensure contiguous arrays for astroalign/sep
    source_img = np.ascontiguousarray(source_img)
    reference_img = np.ascontiguousarray(reference_img)

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
        self.setWindowFlag(Qt.WindowType.Window, True)
        import platform
        if platform.system() == "Darwin":
            self.setWindowFlag(Qt.WindowType.Tool, True)  
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setModal(False)
        try:
            self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        except Exception:
            pass  # older PyQt6 versions

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
        self.source_header = None
        self.target_header = None
        self.source_wcs = None
        self.target_wcs = None
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
        # Alignment method
        self.xf_method = QComboBox()
        self.xf_method.addItems([
            "Stellar (star matching)",
            "WCS / SIP (plate solve, no stars)",
            "WCS → Stellar refinement",
        ])
        self.xf_method.setToolTip(
            "Stellar: astroalign star matching (best when both frames are the same scope).\n"
            "WCS/SIP: reproject via each image's plate solve — handles arbitrary scale,\n"
            "  rotation, flip/mirror, FOV. Needs both images plate-solved.\n"
            "WCS → Stellar: WCS gets cross-instrument frames onto the same grid, then a\n"
            "  star pass refines the residual for sub-pixel accuracy. Best of both."
        )
        _saved_method = self.settings.value("stacking/align/method", "stellar", type=str)
        self.xf_method.setCurrentIndex(
            {"stellar": 0, "wcs": 1, "wcs_stellar": 2}.get(str(_saved_method).lower(), 0))
        xf.addRow("Method:", self.xf_method)

        # WCS plate-solve status row (used by the WCS / WCS→Stellar methods)
        wcs_status_row = QHBoxLayout()
        self.btn_check_wcs = QPushButton("Check WCS")
        self.btn_check_wcs.setFixedHeight(26)
        self.btn_check_wcs.setToolTip(
            "Report whether the current source and target carry a usable plate solve."
        )
        self.btn_check_wcs.clicked.connect(self._refresh_wcs_status)
        self._lbl_wcs_status = QLabel("")
        self._lbl_wcs_status.setStyleSheet("color:#888;font-size:10px;")
        self._lbl_wcs_status.setWordWrap(True)
        wcs_status_row.addWidget(self.btn_check_wcs)
        wcs_status_row.addWidget(self._lbl_wcs_status, 1)
        xf.addRow("", wcs_status_row)

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
        self.xf_det_sigma = QDoubleSpinBox()
        self.xf_det_sigma.setRange(1.0, 200.0)
        self.xf_det_sigma.setDecimals(1)
        self.xf_det_sigma.setSingleStep(1.0)
        self.xf_det_sigma.setValue(prefs["det_sigma"])
        self.xf_det_sigma.setToolTip(
            "Detection threshold in sigma above background.\n"
            "Lower = more stars detected (may include noise).\n"
            "Higher = only bright stars (more robust on crowded fields).\n"
            "Use Trial Detect to test before aligning."
        )
        xf.addRow("Detection threshold (σ):", self.xf_det_sigma)

        # Trial detect row
        trial_row = QHBoxLayout()
        self.btn_trial_detect = QPushButton("🔍 Trial Detect")
        self.btn_trial_detect.setFixedHeight(26)
        self.btn_trial_detect.setToolTip(
            "Run star detection on both source and target with the current\n"
            "threshold and report how many stars were found in each."
        )
        self.btn_trial_detect.clicked.connect(self._run_trial_detect)
        self._lbl_trial_result = QLabel("")
        self._lbl_trial_result.setStyleSheet("color:#888;font-size:10px;")
        self._lbl_trial_result.setWordWrap(True)
        trial_row.addWidget(self.btn_trial_detect)
        trial_row.addWidget(self._lbl_trial_result, 1)
        xf.addRow("", trial_row)
        sync_note = QLabel("ℹ  Detection settings are shared with Stacking Suite.")
        sync_note.setStyleSheet("color:#888;font-size:10px;font-style:italic;")
        xf.addRow("", sync_note)

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

        def _method_key():
            return ("stellar", "wcs", "wcs_stellar")[self.xf_method.currentIndex()]

        def _toggle_method(_=None):
            m = _method_key()
            uses_stars = (m != "wcs")           # star controls used by stellar + refinement
            uses_wcs   = (m != "stellar")       # WCS status relevant unless pure stellar
            for wdg in (self.xf_model, self.xf_maxcp, self.xf_downsample,
                        self.xf_det_sigma, self.btn_trial_detect):
                wdg.setEnabled(uses_stars)
            self.xf_h_reproj.setEnabled(uses_stars and self.xf_model.currentIndex() == 1)
            lab = xf.labelForField(self.xf_h_reproj)
            if lab is not None:
                lab.setEnabled(uses_stars and self.xf_model.currentIndex() == 1)
            self.btn_check_wcs.setEnabled(uses_wcs)
            if uses_stars:
                _toggle_rows()
        self._method_key = _method_key
        self.xf_method.currentIndexChanged.connect(_toggle_method)
        _toggle_method()

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

    def _doc_header(self, doc):
        try:
            meta = (doc.get_metadata() if hasattr(doc, "get_metadata") and callable(doc.get_metadata)
                    else getattr(doc, "metadata", {}) or {})
        except Exception:
            meta = {}
        if not isinstance(meta, dict):
            return None
        return meta.get("original_header") or meta.get("header") or meta.get("fits_header")

    def _run_trial_detect(self):
        import sep
        import numpy as np
        from PyQt6.QtWidgets import QApplication

        thresh = float(self.xf_det_sigma.value())

        # Load source and target images on demand from current selections
        src_img = None
        tgt_img = None

        try:
            if self.source_from_view_radio.isChecked():
                doc = self.source_view_combo.currentData()
                src_img = _doc_image(doc)
            elif self.stellar_source is not None:
                src_img = self.stellar_source

            if self.target_from_view_radio.isChecked():
                doc = self.target_view_combo.currentData()
                tgt_img = _doc_image(doc)
            elif self.stellar_target is not None:
                tgt_img = self.stellar_target
        except Exception as e:
            self._lbl_trial_result.setText(f"⚠  Could not load images: {e}")
            return

        if src_img is None and tgt_img is None:
            self._lbl_trial_result.setText("⚠  Select source and/or target first.")
            return

        self.btn_trial_detect.setEnabled(False)
        self._lbl_trial_result.setText("Detecting…")
        QApplication.processEvents()

        def _count_stars(img):
            if img is None:
                return None, None
            gray = np.mean(img, axis=2).astype(np.float32) if img.ndim == 3 else img.astype(np.float32)
            gray = np.ascontiguousarray(gray)
            try:
                sep.set_extract_pixstack(5000000)
                bkg = sep.Background(gray)
                objs = sep.extract(gray - bkg.back(), thresh * float(bkg.globalrms))
                return len(objs), None
            except Exception as e:
                err = str(e)
                if "pixel buffer full" in err or "pixstack" in err.lower():
                    return None, f"too many pixels at σ={thresh:.1f} — raise threshold"
                return None, err

        parts = []
        src_n, src_err = _count_stars(src_img)
        tgt_n, tgt_err = _count_stars(tgt_img)

        def _fmt(label, n, err):
            if err:
                return f"⚠ {label}: {err}"
            if n is None:
                return f"{label}: —"
            if n == 0:
                return f"⚠ {label}: 0 stars — lower threshold"
            if n > 50000:
                return f"⚠ {label}: {n:,} (very high — raise threshold)"
            if n > 10000:
                return f"⚠ {label}: {n:,} (high — consider raising threshold)"
            return f"✓ {label}: {n:,} stars"

        msg = "  |  ".join(filter(None, [
            _fmt("Source", src_n, src_err) if src_img is not None else None,
            _fmt("Target", tgt_n, tgt_err) if tgt_img is not None else None,
        ]))

        has_warning = "⚠" in msg
        self._lbl_trial_result.setStyleSheet(
            f"color:{'#ffc107' if has_warning else '#4caf50'};font-size:10px;"
        )
        self._lbl_trial_result.setText(msg)
        self.btn_trial_detect.setEnabled(True)

    def _persist_xform_from_dialog(self):
        idx = self.xf_model.currentIndex()
        model = "affine" if idx==0 else ("homography" if idx==1 else ("poly3" if idx==2 else "poly4"))
        s = self.settings
        s.setValue("stacking/align/model", model)
        s.setValue("stacking/align/max_cp", int(self.xf_maxcp.value()))
        s.setValue("stacking/align/downsample", int(self.xf_downsample.value()))
        s.setValue("stacking/align/h_reproj", float(self.xf_h_reproj.value()))
        s.setValue("stacking/align/det_sigma", float(self.xf_det_sigma.value()))
        s.setValue("stacking/align/method",
                   ("stellar", "wcs", "wcs_stellar")[self.xf_method.currentIndex()])

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
        self.source_header = self._doc_header(doc)
        self.source_wcs = _wcs_from_header(self.source_header)        
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
        self.target_header = self._doc_header(doc)
        self.target_wcs = _wcs_from_header(self.target_header)        
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
        self.source_header = header
        self.source_wcs = _wcs_from_header(header)        
        self.lbl_source_file.setText(os.path.basename(path))
        self.source_file_path = path

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
        self.target_header = header
        self.target_wcs = _wcs_from_header(header)        
        self.lbl_target_file.setText(os.path.basename(path))
        self.target_file_path = path

    def _refresh_wcs_status(self):
        """Report source/target plate-solve availability without popping warnings."""
        def _wcs_for(is_source):
            if is_source:
                from_view, combo, hdr = (self.source_from_view_radio.isChecked(),
                                         self.source_view_combo, self.source_header)
            else:
                from_view, combo, hdr = (self.target_from_view_radio.isChecked(),
                                         self.target_view_combo, self.target_header)
            if from_view:
                doc = combo.currentData()
                hdr = self._doc_header(doc) if doc is not None else None
            return _wcs_from_header(hdr)

        self.source_wcs = _wcs_for(True)
        self.target_wcs = _wcs_for(False)
        s_ok, t_ok = self.source_wcs is not None, self.target_wcs is not None
        both = s_ok and t_ok
        self._lbl_wcs_status.setStyleSheet(
            f"color:{'#4caf50' if both else '#ffc107'};font-size:10px;")
        self._lbl_wcs_status.setText(
            f"{'✓' if s_ok else '⚠'} Source WCS   |   {'✓' if t_ok else '⚠'} Target WCS")
        return both
    
    def _resolve_both_wcs(self) -> bool:
        """Make sure images + both WCS objects are populated from current
        selections. Returns True if both WCS are usable."""
        try:
            if self.source_from_view_radio.isChecked():
                self.load_source_from_view()
            if self.target_from_view_radio.isChecked():
                self.load_target_from_view()
        except Exception:
            pass
        if self.source_wcs is None and self.source_header is not None:
            self.source_wcs = _wcs_from_header(self.source_header)
        if self.target_wcs is None and self.target_header is not None:
            self.target_wcs = _wcs_from_header(self.target_header)
        return (self.source_wcs is not None) and (self.target_wcs is not None)

    def _wcs_prealign_target(self):
        """Reproject the target onto the source grid via WCS/SIP and return the
        buffer (source-sized, target's channel layout). None on failure."""
        # Load images + WCS from the current selections FIRST. "Check WCS" only
        # populates the WCS objects, not the pixel buffers, so the buffers can
        # still be None here even when both plate solves are good.
        if not self._resolve_both_wcs():
            missing = [n for n, w in (("source", self.source_wcs),
                                      ("target", self.target_wcs)) if w is None]
            QMessageBox.warning(
                self, "WCS Alignment",
                "No usable plate solve for the "
                f"{' and '.join(missing)} image — skipping the WCS stage.")
            return None

        if self.stellar_source is None or self.stellar_target is None:
            QMessageBox.warning(
                self, "WCS Alignment",
                "Source or target image could not be loaded.")
            return None

        H, W = self.stellar_source.shape[:2]
        self.status_label.setText("WCS stage: reprojecting target onto source…")
        QApplication.processEvents()
        try:
            return wcs_reproject_align(
                self.stellar_target, self.target_wcs, self.source_wcs, (H, W),
                tile=512,
                progress_cb=lambda f: (self.status_label.setText(f"WCS stage… {f*100:.0f}%"),
                                       QApplication.processEvents()),
                log=print,
            ).astype(np.float32, copy=False)
        except Exception as e:
            QMessageBox.warning(self, "WCS Alignment", f"WCS reprojection failed: {e}")
            return None
        
    def run_alignment_wcs(self):
        aligned = self._wcs_prealign_target()
        if aligned is None:
            # hard fail for WCS-only mode → offer stellar
            if QMessageBox.question(
                self, "WCS Alignment",
                "WCS alignment unavailable. Run stellar alignment instead?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            ) == QMessageBox.StandardButton.Yes:
                self.xf_method.setCurrentIndex(0)
                return self.run_alignment()
            return

        self.aligned_image = aligned
        self.stretched_image = None
        if self.autostretch_enabled:
            self.apply_autostretch()
        disp = self.stretched_image if (self.autostretch_enabled and self.stretched_image is not None) else self.aligned_image
        self.update_preview(self.result_preview_label, disp)
        self.status_label.setText("WCS alignment complete.")
        QApplication.processEvents()
        QMessageBox.information(self, "Alignment Complete", "Alignment completed using full WCS (SIP).")

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

    def _accept_wcs_fallback(self, wcs_prealigned, reason: str):
        """WCS→Stellar refinement failed but the WCS stage produced a good
        buffer — keep it instead of bailing."""
        self.aligned_image = np.asarray(wcs_prealigned, np.float32)
        self.stretched_image = None
        if self.autostretch_enabled:
            self.apply_autostretch()
        disp = (self.stretched_image if (self.autostretch_enabled and self.stretched_image is not None)
                else self.aligned_image)
        self.update_preview(self.result_preview_label, disp)
        self.status_label.setText("Star refinement failed — kept WCS alignment.")
        QApplication.processEvents()
        QMessageBox.information(
            self, "Alignment Complete",
            f"Star refinement failed ({reason}); kept the WCS (SIP) alignment.")

    def run_alignment(self):
        self.status_label.setText("Starting Alignment…")
        QApplication.processEvents()

        method = self._method_key()
        try:
            self._persist_xform_from_dialog()
        except Exception:
            pass

        if method == "wcs":
            return self.run_alignment_wcs()

        # For pure stellar, target is the raw target. For refinement, we first
        # WCS-reproject the target onto the source grid, then star-align that.
        wcs_prealigned = None
        if method == "wcs_stellar":
            wcs_prealigned = self._wcs_prealign_target()
            if wcs_prealigned is None:
                # couldn't WCS-prealign; _wcs_prealign_target already messaged.
                # fall through to plain stellar on the original target.
                self.status_label.setText("WCS prealign unavailable — running stellar only…")
                QApplication.processEvents()

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
                        if not img.flags['C_CONTIGUOUS']:
                            img = np.ascontiguousarray(img)
                        base_img = cv2.warpAffine(img, base_X, (Wout, Hout),
                                                flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                    else:
                        # Ensure contiguous channels
                        base_img = np.stack([cv2.warpAffine(np.ascontiguousarray(img[..., c]), base_X, (Wout, Hout),
                                                            flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                                            for c in range(img.shape[2])], axis=2)
                else:
                    if img.ndim == 2:
                        if not img.flags['C_CONTIGUOUS']:
                            img = np.ascontiguousarray(img)
                        base_img = cv2.warpPerspective(img, base_X, (Wout, Hout),
                                                    flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                    else:
                        # Ensure contiguous channels
                        base_img = np.stack([cv2.warpPerspective(np.ascontiguousarray(img[..., c]), base_X, (Wout, Hout),
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
                        out = np.stack([warp(np.ascontiguousarray(base_img[..., c].astype(np.float32)), inverse_map=inv,
                                            output_shape=(Hout, Wout), preserve_range=True)
                                        for c in range(base_img.shape[2])], axis=2)
                return out.astype(np.float32, copy=False)

            return f"poly{order}", _warp_poly_residual


        # Prepare grayscale for detection
        src = self.stellar_source
        tgt = wcs_prealigned if wcs_prealigned is not None else self.stellar_target
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
            if wcs_prealigned is not None:
                return self._accept_wcs_fallback(wcs_prealigned, f"astroalign: {e}")
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
            if wcs_prealigned is not None:
                return self._accept_wcs_fallback(wcs_prealigned, f"transform estimation: {e}")
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
        _label = f"WCS → {model}" if wcs_prealigned is not None else model
        self.status_label.setText(f"Alignment complete ({_label}).")
        QApplication.processEvents()
        QMessageBox.information(self, "Alignment Complete", f"Alignment completed using {_label}.")



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
        # Base the title on the TARGET (the image that was aligned),
        # with a suffix indicating what it was aligned to.
        target_name = ""
        source_name = ""

        # Get target name
        if self.target_from_view_radio.isChecked():
            try:
                doc = self.target_view_combo.currentData()
                target_name = doc.display_name() if callable(getattr(doc, "display_name", None)) else getattr(doc, "display_name", None)
                if not isinstance(target_name, str):
                    target_name = _fmt_doc_title(doc)
            except Exception:
                target_name = ""
        elif self.target_file_path:
            target_name = os.path.splitext(os.path.basename(self.target_file_path))[0]

        # Get source name (for the suffix)
        if self.source_from_view_radio.isChecked():
            try:
                doc = self.source_view_combo.currentData()
                source_name = doc.display_name() if callable(getattr(doc, "display_name", None)) else getattr(doc, "display_name", None)
                if not isinstance(source_name, str):
                    source_name = _fmt_doc_title(doc)
            except Exception:
                source_name = ""
        elif self.source_file_path:
            source_name = os.path.splitext(os.path.basename(self.source_file_path))[0]

        if target_name and source_name:
            return f"{target_name}_aligned_to_{source_name}"
        if target_name:
            return f"{target_name}_aligned"
        if source_name:
            return f"aligned_to_{source_name}"
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

def _to3x3_affine(A2x3: np.ndarray) -> np.ndarray:
    A = np.asarray(A2x3, np.float64).reshape(2,3)
    return np.vstack([A, [0,0,1]])

def _from3x3_affine(A3: np.ndarray) -> np.ndarray:
    return np.asarray(A3, np.float64)[:2,:]

def _S(ds: float) -> np.ndarray:
    ds = float(ds)
    return np.array([[1.0/ds, 0, 0],
                     [0, 1.0/ds, 0],
                     [0, 0, 1]], np.float64)

def lift_affine_2x3_from_ds(A_ds_2x3: np.ndarray, ds: float) -> np.ndarray:
    S = _S(ds); Si = np.linalg.inv(S)
    A3_full = Si @ _to3x3_affine(A_ds_2x3) @ S
    return _from3x3_affine(A3_full)

def downscale_affine_2x3_to_ds(A_full_2x3: np.ndarray, ds: float) -> np.ndarray:
    S = _S(ds); Si = np.linalg.inv(S)
    A3_ds = S @ _to3x3_affine(A_full_2x3) @ Si
    return _from3x3_affine(A3_ds)

def lift_homography_from_ds(H_ds: np.ndarray, ds: float) -> np.ndarray:
    S = _S(ds); Si = np.linalg.inv(S)
    return Si @ np.asarray(H_ds, np.float64) @ S


def compute_affine_transform_astroalign_cropped(source_img, reference_img,
                                                scale: float = 1.20,
                                                limit_stars: int | None = None,
                                                det_sigma: float = 12.0,
                                                minarea: int = 10,
                                                min_fwhm: float = 1.2,
                                                max_ellipticity: float = 0.6):
    """
    Solve affine on a ~1.2x center crop of reference and lift into full-ref coords.
    Returns a 2x3 affine matrix in float64, or None.
    """
    import numpy as np
    from setiastro.saspro import astroalign

    # Optional global AA lock (if present in your module)
    try:
        _lock = _AA_LOCK
    except NameError:
        from contextlib import nullcontext
        _lock = nullcontext()

    Hs, Ws = source_img.shape[:2]
    Hr, Wr = reference_img.shape[:2]

    h = min(int(round(Hs * scale)), Hr)
    w = min(int(round(Ws * scale)), Wr)
    y0 = max(0, (Hr - h) // 2)
    x0 = max(0, (Wr - w) // 2)
    ref_crop = reference_img[y0:y0+h, x0:x0+w]

    kwargs = {"detection_sigma": float(det_sigma), "min_area": int(minarea)}
    if limit_stars is not None:
        kwargs["max_control_points"] = int(limit_stars)

    with _lock:
        try:
            # ---- NEW: uniform control points ----
            src_pts = _detect_stars_uniform(source_img, det_sigma, minarea,
                                            grid=(4,4), max_per_cell=25,
                                            max_total=(limit_stars or 500),
                                            min_fwhm=min_fwhm,
                                            max_ellipticity=max_ellipticity)
            ref_pts = _detect_stars_uniform(ref_crop, det_sigma, minarea,
                                            grid=(4,4), max_per_cell=25,
                                            max_total=(limit_stars or 500),
                                            min_fwhm=min_fwhm,
                                            max_ellipticity=max_ellipticity)

            cov_src = _coverage_fraction(src_pts, Hs, Ws, grid=(4,4))
            cov_ref = _coverage_fraction(ref_pts, h,  w,  grid=(4,4))
            if cov_src < 0.5 or cov_ref < 0.5:
                # only log; don't fail
                # (use whatever logger/progress emitter you have)
                # print is fine in worker context
                print(f"[AA] low coverage src={cov_src:.2f}, ref={cov_ref:.2f} for crop solve")

            if src_pts.shape[0] >= 8 and ref_pts.shape[0] >= 8:
                # When passing points, some AA versions ignore detection kwargs;
                # keep max_control_points only.
                pt_kwargs = {}
                if "max_control_points" in kwargs:
                    pt_kwargs["max_control_points"] = kwargs["max_control_points"]

                tform, _ = astroalign.find_transform(src_pts, ref_pts, **pt_kwargs)
            else:
                raise RuntimeError("Too few uniform points, falling back to images.")

        except Exception:
            # ---- fallback: original image-based AA ----
            try:
                tform, _ = astroalign.find_transform(
                    np.ascontiguousarray(source_img.astype(np.float32)),
                    np.ascontiguousarray(ref_crop.astype(np.float32)),
                    **kwargs
                )
            except TypeError:
                legacy_kwargs = {}
                if "max_control_points" in kwargs:
                    legacy_kwargs["max_control_points"] = kwargs["max_control_points"]
                tform, _ = astroalign.find_transform(
                    np.ascontiguousarray(source_img.astype(np.float32)),
                    np.ascontiguousarray(ref_crop.astype(np.float32)),
                    **legacy_kwargs
                )

    P = np.asarray(tform.params, dtype=np.float64)
    T = np.array([[1, 0, x0], [0, 1, y0], [0, 0, 1]], dtype=np.float64)

    if P.shape == (3, 3):
        return (T @ P)[0:2, :]
    elif P.shape == (2, 3):
        A3 = np.vstack([P, [0, 0, 1]])
        return (T @ A3)[0:2, :]
    return None

def _coverage_fraction(pts, H, W, grid=(4,4)):
    gy, gx = grid
    if len(pts) == 0:
        return 0.0
    cell_w = W / gx; cell_h = H / gy
    occ = np.zeros((gy,gx), bool)
    for x,y in pts:
        cx = int(x / cell_w); cy = int(y / cell_h)
        if 0 <= cx < gx and 0 <= cy < gy:
            occ[cy,cx] = True
    return occ.mean()

def _points_spread_ok(tgt_xy: np.ndarray, Wref: int, Href: int,
                      frac_span: float = 0.35,
                      grid: int = 3,
                      min_cells: int = 6,
                      _dbg=None) -> bool:
    if tgt_xy is None or len(tgt_xy) < 8:
        if _dbg: _dbg("[spread] too few points")
        return False

    xy = np.asarray(tgt_xy, np.float32)
    x = xy[:, 0]; y = xy[:, 1]

    p05x, p95x = np.percentile(x, [5, 95])
    p05y, p95y = np.percentile(y, [5, 95])
    span_x = float(p95x - p05x)
    span_y = float(p95y - p05y)

    gx = np.clip((x / max(Wref,1) * grid).astype(int), 0, grid-1)
    gy = np.clip((y / max(Href,1) * grid).astype(int), 0, grid-1)
    cells = set(zip(gx.tolist(), gy.tolist()))

    if _dbg:
        _dbg(f"[spread] N={len(xy)} span_x={span_x:.1f} ({span_x/Wref:.2f}W) span_y={span_y:.1f} ({span_y/Href:.2f}H) cells={len(cells)}/{grid*grid}")

    if span_x < frac_span * Wref or span_y < frac_span * Href:
        if _dbg: _dbg("[spread] fail span")
        return False
    if len(cells) < min_cells:
        if _dbg: _dbg("[spread] fail grid occupancy")
        return False
    return True

def _fit_poly_xy(src_xy, tgt_xy, order=3):
    import numpy as np
    x, y = src_xy[:,0], src_xy[:,1]
    xp, yp = tgt_xy[:,0], tgt_xy[:,1]

    # build design matrix for 2D poly
    terms = []
    for i in range(order+1):
        for j in range(order+1-i):
            terms.append((x**i)*(y**j))
    A = np.vstack(terms).T  # (N, M)

    cx, *_ = np.linalg.lstsq(A, xp, rcond=None)
    cy, *_ = np.linalg.lstsq(A, yp, rcond=None)
    return cx, cy

def _poly_eval_grid(cx, cy, W, H, order=3):
    import numpy as np
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    terms = []
    for i in range(order+1):
        for j in range(order+1-i):
            terms.append((xx**i)*(yy**j))
    A = np.stack(terms, axis=0)  # (M, H, W)

    map_x = np.tensordot(cx, A, axes=(0,0)).astype(np.float32)
    map_y = np.tensordot(cy, A, axes=(0,0)).astype(np.float32)
    return map_x, map_y

def _subsample_points_spatially(src_xy: np.ndarray,
                                tgt_xy: np.ndarray,
                                max_points: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Deterministic farthest-point subsampling.
    Keeps spatial coverage much better than random choice.
    """
    import numpy as np

    src_xy = np.asarray(src_xy, np.float32)
    tgt_xy = np.asarray(tgt_xy, np.float32)

    n = len(src_xy)
    if n <= max_points or max_points <= 0:
        return src_xy, tgt_xy

    # Start from point farthest from centroid for stability
    centroid = src_xy.mean(axis=0)
    d0 = np.sum((src_xy - centroid) ** 2, axis=1)
    first = int(np.argmax(d0))

    selected = [first]
    selected_mask = np.zeros(n, dtype=bool)
    selected_mask[first] = True

    # Distance to nearest selected point
    min_dist2 = np.sum((src_xy - src_xy[first]) ** 2, axis=1)

    while len(selected) < max_points:
        # Never re-pick selected points
        min_dist2[selected_mask] = -1.0
        nxt = int(np.argmax(min_dist2))
        if nxt < 0 or selected_mask[nxt]:
            break

        selected.append(nxt)
        selected_mask[nxt] = True

        d2 = np.sum((src_xy - src_xy[nxt]) ** 2, axis=1)
        min_dist2 = np.minimum(min_dist2, d2)

    sel = np.array(selected, dtype=int)
    return src_xy[sel], tgt_xy[sel]


def _aa_find_pairs_multitile(src_gray: np.ndarray,
                             ref2d: np.ndarray,
                             scale: float = 1.20,
                             tile_positions=None,
                             tiles: int = 1,
                             det_sigma: float = 12.0,
                             minarea: int = 10,
                             max_control_points: int | None = 40,
                             *, _dbg=None):
    """
    Run astroalign on 1x1 (center), 5-tile (corners+center), or NxN tiles of the reference.
    Return merged (src_xy, tgt_xy_full, best_tform_params, best_offsets).

    tiles=1 => center crop only.
    tiles=5 => corners + center.
    tiles=3 => 3x3 grid (center+edges+corners).
    tile_positions => explicit [(x0,y0), ...] overrides tiles.

    max_control_points:
      - passed to astroalign (limits detections per crop)
      - ALSO used as a per-tile cap after detection to balance tiles
        (if you want only AA limiting, set tiles=1).
    """
    import numpy as np
    from setiastro.saspro import astroalign
    try:
        _lock = _AA_LOCK
    except NameError:
        from contextlib import nullcontext
        _lock = nullcontext()

    def dbg(msg):
        if _dbg:
            try: _dbg(msg)
            except Exception as e:
                import logging
                logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")

    src = np.ascontiguousarray(src_gray.astype(np.float32))
    ref = np.ascontiguousarray(ref2d.astype(np.float32))
    Hs, Ws = src.shape[:2]
    Hr, Wr = ref.shape[:2]

    # crop size same as today
    h = min(int(round(Hs * scale)), Hr)
    w = min(int(round(Ws * scale)), Wr)

    # if crop almost full ref, no need to tile
    if tiles <= 1 or h >= Hr * 0.95 or w >= Wr * 0.95:
        tiles = 1

    # sanitize max_control_points
    mcp = None
    try:
        if max_control_points is not None and int(max_control_points) > 0:
            mcp = int(max_control_points)
    except Exception:
        mcp = None

    kwargs = {"detection_sigma": float(det_sigma), "min_area": int(minarea)}
    if mcp is not None:
        kwargs["max_control_points"] = mcp

    # --- helper to clamp tile top-lefts ---
    def _clamp_xy0(x0, y0):
        x0 = int(np.clip(x0, 0, max(0, Wr - w)))
        y0 = int(np.clip(y0, 0, max(0, Hr - h)))
        return x0, y0

    # --- pick tile positions (top-lefts in full ref coords) ---
    positions = []

    if tile_positions is not None:
        # explicit override
        for (x0, y0) in tile_positions:
            positions.append(_clamp_xy0(x0, y0))

    elif tiles == 1:
        positions = [_clamp_xy0((Wr - w) // 2, (Hr - h) // 2)]

    elif tiles == 5:
        # corners + center
        positions = [
            _clamp_xy0((Wr - w) // 2, (Hr - h) // 2),  # center
            _clamp_xy0(0, 0),                          # TL
            _clamp_xy0(Wr - w, 0),                     # TR
            _clamp_xy0(0, Hr - h),                     # BL
            _clamp_xy0(Wr - w, Hr - h),                # BR
        ]

        # de-dupe in case w/h ~= Wr/Hr (clamp collapses positions)
        positions = list(dict.fromkeys(positions))

    else:
        # NxN grid (tiles x tiles)
        ys = np.linspace(0, max(0, Hr - h), tiles).astype(int).tolist()
        xs = np.linspace(0, max(0, Wr - w), tiles).astype(int).tolist()
        for y0 in ys:
            for x0 in xs:
                positions.append(_clamp_xy0(x0, y0))

        positions = list(dict.fromkeys(positions))

    all_src, all_tgt = [], []
    best_n = -1
    best_P, best_xy0 = None, (0, 0)

    tile_idx = 0
    for (x0, y0) in positions:
        tile_idx += 1
        ref_crop = ref[y0:y0+h, x0:x0+w]

        try:
            with _lock:
                tform, (src_pts_s, tgt_pts_s) = astroalign.find_transform(src, ref_crop, **kwargs)
        except TypeError:
            # legacy AA without det_sigma/min_area
            legacy = {}
            if "max_control_points" in kwargs:
                legacy["max_control_points"] = kwargs["max_control_points"]
            with _lock:
                tform, (src_pts_s, tgt_pts_s) = astroalign.find_transform(src, ref_crop, **legacy)
        except Exception as e:
            dbg(f"[AA tile {tile_idx}] fail x0={x0} y0={y0}: {e}")
            continue

        src_xy = np.asarray(src_pts_s, np.float32)
        tgt_xy = np.asarray(tgt_pts_s, np.float32)

        if len(src_xy) == 0:
            dbg(f"[AA tile {tile_idx}] 0 matches x0={x0} y0={y0}")
            continue

        # lift crop coords to full ref coords
        tgt_xy[:, 0] += x0
        tgt_xy[:, 1] += y0

        # ---- per-tile balancing cap ----
        if len(positions) > 1 and mcp is not None and len(src_xy) > mcp:
            src_xy, tgt_xy = _subsample_points_spatially(src_xy, tgt_xy, mcp)

        all_src.append(src_xy)
        all_tgt.append(tgt_xy)

        dbg(f"[AA tile {tile_idx}] matches={len(src_xy)} x0={x0} y0={y0}")

        if len(src_xy) > best_n:
            best_n = len(src_xy)
            best_P = np.asarray(tform.params, np.float64)
            best_xy0 = (x0, y0)

    if not all_src:
        return None, None, None, None

    src_all = np.vstack(all_src)
    tgt_all = np.vstack(all_tgt)

    return src_all, tgt_all, best_P, best_xy0

def compute_similarity_transform_astroalign_cropped(source_img, reference_img,
                                                   scale: float = 1.20,
                                                   limit_stars: int | None = None,
                                                   det_sigma: float = 12.0,
                                                   minarea: int = 10,
                                                   h_reproj: float = 3.0,
                                                   min_fwhm: float = 1.2,
                                                   max_ellipticity: float = 0.6):
    import numpy as np
    from setiastro.saspro import astroalign
    import cv2

    try:
        _lock = _AA_LOCK
    except NameError:
        from contextlib import nullcontext
        _lock = nullcontext()

    Hs, Ws = source_img.shape[:2]
    Hr, Wr = reference_img.shape[:2]
    h = min(int(round(Hs * scale)), Hr)
    w = min(int(round(Ws * scale)), Wr)
    y0 = max(0, (Hr - h) // 2)
    x0 = max(0, (Wr - w) // 2)
    ref_crop = reference_img[y0:y0+h, x0:x0+w]

    kwargs = {"detection_sigma": float(det_sigma), "min_area": int(minarea)}
    if limit_stars is not None:
        kwargs["max_control_points"] = int(limit_stars)

    with _lock:
        try:
            src_pts = _detect_stars_uniform(source_img, det_sigma, minarea,
                                            grid=(4,4), max_per_cell=25,
                                            max_total=(limit_stars or 500),
                                            min_fwhm=min_fwhm,
                                            max_ellipticity=max_ellipticity)
            ref_pts = _detect_stars_uniform(ref_crop, det_sigma, minarea,
                                            grid=(4,4), max_per_cell=25,
                                            max_total=(limit_stars or 500),
                                            min_fwhm=min_fwhm,
                                            max_ellipticity=max_ellipticity)

            cov_src = _coverage_fraction(src_pts, Hs, Ws, grid=(4,4))
            cov_ref = _coverage_fraction(ref_pts, h,  w,  grid=(4,4))
            if cov_src < 0.5 or cov_ref < 0.5:
                print(f"[AA] low coverage src={cov_src:.2f}, ref={cov_ref:.2f} for crop similarity")

            if src_pts.shape[0] >= 8 and ref_pts.shape[0] >= 8:
                pt_kwargs = {}
                if limit_stars is not None:
                    pt_kwargs["max_control_points"] = int(limit_stars)
                tform, (src_pts_s, tgt_pts_s) = astroalign.find_transform(
                    src_pts, ref_pts, **pt_kwargs
                )
            else:
                raise RuntimeError("Too few uniform points, falling back to images.")
        except Exception:
            tform, (src_pts_s, tgt_pts_s) = astroalign.find_transform(
                np.ascontiguousarray(source_img.astype(np.float32)),
                np.ascontiguousarray(ref_crop.astype(np.float32)),
                **kwargs
            )

    src_xy = np.asarray(src_pts_s, dtype=np.float32)
    tgt_xy = np.asarray(tgt_pts_s, dtype=np.float32)
    tgt_xy[:, 0] += x0
    tgt_xy[:, 1] += y0

    A, inl = cv2.estimateAffinePartial2D(
        src_xy, tgt_xy, method=cv2.RANSAC,
        ransacReprojThreshold=float(h_reproj)
    )
    if A is not None:
        return np.asarray(A, np.float64).reshape(2, 3)

    P = np.asarray(tform.params, dtype=np.float64)
    if P.shape == (3, 3):
        base = (np.array([[1,0,x0],[0,1,y0],[0,0,1]]) @ P)[0:2, :]
    else:
        A3 = np.vstack([P[0:2, :], [0,0,1]])
        base = (np.array([[1,0,x0],[0,1,y0],[0,0,1]]) @ A3)[0:2, :]
    return project_affine_to_similarity(base)

def project_affine_to_similarity(A2x3: np.ndarray) -> np.ndarray:
    """
    Take a 2x3 affine and remove shear by projecting to nearest similarity transform.
    Keeps translation, preserves best rotation+uniform scale.
    """
    import numpy as np
    A = np.asarray(A2x3, np.float64).reshape(2, 3)
    M = A[:, :2]
    t = A[:, 2]

    # polar/SVD
    U, S, Vt = np.linalg.svd(M)
    R = U @ Vt
    s = float(np.mean(S))  # uniform scale
    M_sim = s * R

    out = np.zeros((2,3), np.float64)
    out[:, :2] = M_sim
    out[:, 2] = t
    return out


def _solve_delta_job(args):
    """
    Worker: compute incremental affine/similarity delta for one frame against the ref preview.
    args =
        (orig_path, current_transform_2x3,
         ref_small_ds, Wref_ds, Href_ds,
         resample_flag, det_sigma, limit_stars, minarea,
         model, h_reproj, ds,
         min_fwhm, max_ellipticity) = args 
    """
    try:
        import os
        import numpy as np
        import cv2
        from astropy.io import fits

        (orig_path, current_transform_2x3,
         ref_small_ds, Wref_ds, Href_ds,
         resample_flag, det_sigma, limit_stars, minarea,
         model, h_reproj, ds,
         min_fwhm, max_ellipticity) = args 

        try:
            cv2.setNumThreads(1)
            try: cv2.ocl.setUseOpenCL(False)
            except Exception:
                pass
        except Exception:
            pass

        # 1) read → gray float32 (full)
        with fits.open(orig_path, memmap=True) as hdul:
            arr = hdul[0].data
            if arr is None:
                return (orig_path, None, f"Could not load {os.path.basename(orig_path)}")
            gray = arr if arr.ndim == 2 else np.mean(arr, axis=2)
            gray = np.nan_to_num(gray, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

        # NOTE: `ds` is already provided in args (matches self.solve_downsample,
        # which was also used to build ref_small_ds/Wref_ds/Href_ds). Previously
        # this was hardcoded/recomputed here, ignoring the passed-in value and
        # the user-configured downsample — that override has been removed;
        # `ds` from args is used as-is below.
        ds = max(1, int(ds))

        # 2) downsample source to DS space
        if ds > 1:
            Wds = max(1, int(gray.shape[1] // ds))
            Hds = max(1, int(gray.shape[0] // ds))
            gray_ds = cv2.resize(gray, (Wds, Hds), interpolation=cv2.INTER_AREA)
        else:
            gray_ds = gray

        # 3) pre-warp in DS space using downscaled transform
        T_prev_full = np.asarray(current_transform_2x3, np.float64).reshape(2, 3)
        T_prev_ds = downscale_affine_2x3_to_ds(T_prev_full, ds).astype(np.float32)

        # Warp DS source into DS ref geometry
        src_for_match_ds = cv2.warpAffine(
            gray_ds, T_prev_ds, (int(Wref_ds), int(Href_ds)),
            flags=resample_flag, borderMode=cv2.BORDER_REFLECT_101
        )

        # 4) denoise sparse islands in DS space (cheaper)
        # fast hot pixel suppression — 3x3 median replaces only extreme spikes,
        # no SEP background estimation needed
        #src_for_match_ds = _suppress_hotpx_fast(src_for_match_ds)
        ref_for_match_ds = np.asarray(ref_small_ds, np.float32, order="C") #_suppress_hotpx_fast(np.asarray(ref_small_ds, np.float32, order="C"))

        # 5) AA delta solve in DS space
        m = (model or "affine").lower()
        if m in ("no_distortion", "nodistortion"):
            m = "similarity"

        if m == "similarity":
            tform_ds = compute_similarity_transform_astroalign_cropped(
                src_for_match_ds, ref_for_match_ds,
                limit_stars=int(limit_stars) if limit_stars is not None else None,
                det_sigma=float(det_sigma),
                minarea=int(minarea),
                h_reproj=float(h_reproj),
                min_fwhm=float(min_fwhm),
                max_ellipticity=float(max_ellipticity),
            )
        else:
            tform_ds = compute_affine_transform_astroalign_cropped(
                src_for_match_ds, ref_for_match_ds,
                limit_stars=int(limit_stars) if limit_stars is not None else None,
                det_sigma=float(det_sigma),
                minarea=int(minarea),
                min_fwhm=float(min_fwhm),
                max_ellipticity=float(max_ellipticity),
            )

        if tform_ds is None:
            return (orig_path, None,
                    f"Astroalign failed for {os.path.basename(orig_path)} – skipping (no transform returned)")

        # 6) lift DS delta back to full-res coords
        T_new_full = lift_affine_2x3_from_ds(np.asarray(tform_ds, np.float64).reshape(2, 3), ds)

        return (orig_path, np.asarray(T_new_full, np.float64).reshape(2, 3), None)

    except Exception as e:
        try:
            base = os.path.basename(args[0]) if args else "<unknown>"
        except Exception:
            base = "<unknown>"
        return (args[0] if args else "<unknown>", None,
                f"Astroalign failed for {base}: {e}")



def _residual_job_worker(args):
    """
    Process-safe worker for non-affine residual measurement.
    args = (path, ref_npy, model, h_reproj, det_sigma, minarea, limit_stars)
    Returns: (path, rms_px, err_or_None)
    """
    (path, ref_npy, model, h_reproj, det_sigma, minarea, limit_stars) = args
    try:
        import numpy as np  # re-imports are OK in spawned workers
        from astropy.io import fits

        # Load source (gray, float32, finite)
        with fits.open(path, memmap=True) as hdul:
            arr = hdul[0].data
            if arr is None:
                return (path, float("inf"), "Could not load")
            g = arr if arr.ndim == 2 else np.mean(arr, axis=2)
            g = np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

        # Memmap the shared reference
        ref_small = np.load(ref_npy, mmap_mode="r").astype(np.float32, copy=False)

        # Use the staticmethod that’s importable by workers
        _, _, rms, _ = StarRegistrationThread._aa_model_and_residual(
            g, ref_small, str(model).lower(),
            float(h_reproj), float(det_sigma), int(minarea),
            int(limit_stars) if limit_stars is not None else None
        )
        return (path, float(rms), None)

    except Exception as e:
        return (path, float("inf"), str(e))

def _suppress_tiny_islands(img32: np.ndarray, det_sigma: float, minarea: int) -> np.ndarray:
    import sep
    import cv2

    img32 = np.asarray(img32, np.float32, order="C")

    # ── Pre-filter: suppress hot pixels before background estimation ──
    # 3x3 for single-pixel, then replace only pixels that spiked far above
    # their neighborhood — preserves star cores better than a blind median.
    try:
        med3 = cv2.medianBlur(img32, 3)
        # Only replace pixels where the original is >5σ above the local median
        # (i.e. genuine isolated spikes, not star wings)
        bkg_est = sep.Background(img32, bw=64, bh=64)
        spike_thresh = float(bkg_est.globalrms) * 8.0
        spike_mask = (img32 - med3) > spike_thresh
        img32 = np.where(spike_mask, med3, img32)
    except Exception:
        try:
            img32 = cv2.medianBlur(img32, 3)
        except Exception:
            pass
    # ─────────────────────────────────────────────────────────────────

    bkg = sep.Background(img32, bw=64, bh=64)
    back_img = bkg.back().astype(np.float32, copy=False)
    thresh = float(det_sigma) * float(bkg.globalrms)

    mask = (img32 > (back_img + thresh)).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return img32

    keep = np.zeros(num, dtype=np.uint8)
    keep[stats[:, cv2.CC_STAT_AREA] >= int(minarea)] = 1
    keep[0] = 0
    pruned = keep[labels]

    out = np.where(((mask == 1) & (pruned == 0)), back_img, img32)
    return out.astype(np.float32, copy=False)

# ─────────────────────────────────────────────────────────────
# Final warp+save worker (process-safe)
# ─────────────────────────────────────────────────────────────
def _satmask_sidecar_path_for_image(image_path: str) -> str:
    import os
    root, _ = os.path.splitext(image_path)
    return root + "_satmask.npy"


def _candidate_satmask_paths_for_image(image_path: str):
    """
    Return possible sidecar paths for an image path.

    Examples:
      Aligned/abc_c_n_r.fit -> Aligned/abc_c_n_r_satmask.npy
      Normalized/abc_c_n.fit -> Normalized/abc_c_n_satmask.npy, then
                                ../Calibrated/abc_c_satmask.npy
    """
    import os

    image_path = os.path.normpath(image_path)
    out = []

    # 1) exact sidecar next to the image path
    out.append(_satmask_sidecar_path_for_image(image_path))

    base = os.path.basename(image_path)
    dirname = os.path.dirname(image_path)
    parent = os.path.dirname(dirname)

    # 2) normalized -> calibrated fallback
    #    e.g. foo_c_n.fit -> Calibrated/foo_c_satmask.npy
    if base.lower().endswith("_n.fit"):
        denorm_base = base[:-6] + ".fit"   # strip "_n.fit", restore ".fit"
        cal_dir = os.path.join(parent, "Calibrated")
        out.append(_satmask_sidecar_path_for_image(os.path.join(cal_dir, denorm_base)))

    # de-duplicate while preserving order
    seen = set()
    uniq = []
    for p in out:
        pn = os.path.normcase(os.path.normpath(p))
        if pn not in seen:
            seen.add(pn)
            uniq.append(p)
    return uniq


def _load_satmask_sidecar(image_path: str):
    import os
    import numpy as np

    for p in _candidate_satmask_paths_for_image(image_path):
        if not os.path.exists(p):
            continue
        try:
            m = np.load(p, allow_pickle=False)
            m = np.asarray(m, dtype=bool)
            if m.ndim != 2:
                continue
            return m
        except Exception:
            continue
    return None


def _save_satmask_sidecar(image_path: str, mask2d):
    import numpy as np
    p = _satmask_sidecar_path_for_image(image_path)
    np.save(p, np.asarray(mask2d, dtype=np.uint8), allow_pickle=False)
    return p

def _warp_satmask_with_kind(mask2d, kind: str, X: object, out_hw: tuple[int, int]):
    """
    Warp a 2D boolean rejection mask with the same final registration model.
    Uses nearest-neighbor everywhere to preserve binary mask semantics.
    out_hw = (H, W) in reference/aligned space.
    """
    import numpy as np
    import cv2

    Hh, Ww = int(out_hw[0]), int(out_hw[1])
    src = np.asarray(mask2d, dtype=np.uint8)

    if kind in ("affine", "similarity"):
        A = np.asarray(X, np.float32).reshape(2, 3)
        out = cv2.warpAffine(
            src, A, (Ww, Hh),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        return out.astype(bool)

    if kind == "homography":
        Hm = np.asarray(X, np.float32).reshape(3, 3)
        out = cv2.warpPerspective(
            src, Hm, (Ww, Hh),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        return out.astype(bool)

    if kind in ("poly3", "poly4"):
        map_x, map_y = X
        out = cv2.remap(
            src,
            map_x.astype(np.float32),
            map_y.astype(np.float32),
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        return out.astype(bool)

    # Fallback: no warp
    hh = min(Hh, src.shape[0])
    ww = min(Ww, src.shape[1])
    out = np.zeros((Hh, Ww), dtype=bool)
    out[:hh, :ww] = src[:hh, :ww].astype(bool)
    return out

def _suppress_hotpx_fast(img32: np.ndarray) -> np.ndarray:
    """
    Fast hot pixel suppression via 3x3 median, no SEP needed.
    Only replaces pixels that are extreme outliers vs their local neighborhood.
    Hot pixels typically spike 10-100x above neighbors; real stars have
    smooth PSFs so their cores survive this filter.
    """
    import cv2
    img32 = np.asarray(img32, np.float32, order="C")
    med3  = cv2.medianBlur(img32, 3)

    # Replace only pixels > 8x their local median — genuine hot pixels
    # Real star cores are typically < 3x their immediate neighbors
    ratio = np.where(med3 > 1e-9, img32 / med3, 1.0)
    return np.where(ratio > 8.0, med3, img32)

def _finalize_write_job(args):
    """
    Process-safe worker: read full-res, choose model, warp, save.
    Also propagates satellite rejection sidecar mask if present.

    Returns (orig_path, out_path or "", msg, success, drizzle_tuple or None)
    drizzle_tuple = (kind, matrix_or_None)
    """
    try:
        (orig_path, align_model, ref_shape, ref_npy_path,
         affine_2x3, h_reproj, output_directory,
         det_sigma, minarea, limit_stars,
         min_fwhm, max_ellipticity, solve_downsample) = args
    except ValueError:
        try:
            # format without solve_downsample (default to 3, matching the
            # previous fixed-ds behavior)
            (orig_path, align_model, ref_shape, ref_npy_path,
             affine_2x3, h_reproj, output_directory,
             det_sigma, minarea, limit_stars,
             min_fwhm, max_ellipticity) = args
            solve_downsample = 3
        except ValueError:
            # oldest-format args without hot pixel params or solve_downsample
            (orig_path, align_model, ref_shape, ref_npy_path,
             affine_2x3, h_reproj, output_directory,
             det_sigma, minarea, limit_stars) = args
            min_fwhm, max_ellipticity = 1.2, 0.6
            solve_downsample = 3

    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    import numpy as np
    from astropy.io import fits
    import cv2

    try:
        cv2.setNumThreads(1)
        try:
            cv2.ocl.setUseOpenCL(False)
        except Exception:
            pass
    except Exception:
        pass

    debug_lines = []
    def dbg(s: str):
        debug_lines.append(str(s))

    def _A3(A2x3):
        A = np.asarray(A2x3, np.float64).reshape(2, 3)
        return np.vstack([A, [0, 0, 1]])

    try:
        # 1) load source (full-res)
        with fits.open(orig_path, memmap=True) as hdul:
            img = hdul[0].data
            hdr = hdul[0].header
        if img is None:
            return (orig_path, "", f"⚠️ Failed to read {os.path.basename(orig_path)}", False, None)

        # normalize ints
        if img.dtype == np.uint16:
            img = img.astype(np.float32) / 65535.0
        elif img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0

        is_mono = (img.ndim == 2)
        src_gray_full = img if is_mono else np.mean(img, axis=2)
        src_gray_full = np.nan_to_num(src_gray_full, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        img = np.ascontiguousarray(img)

        # ---- NEW: load original satellite mask sidecar if present ----
        satmask_src = _load_satmask_sidecar(orig_path)
        if satmask_src is None:
            dbg(f"[satmask] no source sidecar found for {os.path.basename(orig_path)}")
        else:
            dbg(f"[satmask] loaded source sidecar for {os.path.basename(orig_path)} ({int(np.count_nonzero(satmask_src))} px)")
        Href, Wref = ref_shape

        # 2) load reference (full-res) via memmap
        ref2d = np.load(ref_npy_path, mmap_mode="r").astype(np.float32, copy=False)
        if ref2d.shape[:2] != (Href, Wref):
            return (orig_path, "", f"⚠️ Ref shape mismatch for {os.path.basename(orig_path)}", False, None)

        base = os.path.basename(orig_path)

        model = (align_model or "affine").lower()
        if model in ("no_distortion", "nodistortion"):
            model = "similarity"

        # Base (accumulated) affine from refinement
        A_prev = np.asarray(affine_2x3, np.float64).reshape(2, 3)
        A_prev3 = _A3(A_prev)

        # Default finalize is just the affine refinement result
        kind = "affine"
        X = A_prev.copy()

        # ---- Non-affine finalize: DS solve + lift, but KEEP affine-as-start ----
        if model != "affine":
            dbg(f"[finalize] base={base} model={model} det_sigma={det_sigma} minarea={minarea} limit_stars={limit_stars}")

            # Use the same solve-grid downsample factor as the refinement
            # passes (user-configurable, default 3) so the finalize pass
            # stays consistent with the deltas it builds on.
            ds = max(1, int(solve_downsample))

            if ds > 1:
                ref_ds = cv2.resize(ref2d, (max(1, Wref // ds), max(1, Href // ds)), interpolation=cv2.INTER_AREA)
            else:
                ref_ds = np.ascontiguousarray(ref2d)

            ref_ds = np.ascontiguousarray(ref_ds.astype(np.float32, copy=False))
            Hds, Wds = ref_ds.shape[:2]

            if ds > 1:
                src_ds0 = cv2.resize(src_gray_full, (Wds, Hds), interpolation=cv2.INTER_AREA)
            else:
                src_ds0 = cv2.resize(src_gray_full, (Wds, Hds), interpolation=cv2.INTER_AREA) if (src_gray_full.shape[:2] != (Hds, Wds)) else src_gray_full

            src_ds0 = np.ascontiguousarray(src_ds0.astype(np.float32, copy=False))

            A_prev_ds = downscale_affine_2x3_to_ds(A_prev, ds).astype(np.float32)
            src_pre_ds = cv2.warpAffine(
                src_ds0, A_prev_ds, (Wds, Hds),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
            )

            #src_pre_ds = _suppress_hotpx_fast(src_pre_ds)
            #ref_ds     = _suppress_hotpx_fast(ref_ds)

            max_cp = None
            try:
                if limit_stars is not None and int(limit_stars) > 0:
                    max_cp = int(limit_stars)
            except Exception:
                max_cp = None

            AA_SCALE = 0.80

            src_xy, tgt_xy, best_P, best_xy0 = _aa_find_pairs_multitile(
                src_pre_ds, ref_ds,
                scale=AA_SCALE,
                tiles=1,
                det_sigma=float(det_sigma),
                minarea=int(minarea),
                max_control_points=max_cp,
                _dbg=dbg
            )
            if src_xy is None or len(src_xy) < 8:
                raise RuntimeError("astroalign produced too few matches (finalize)")

            hth = float(h_reproj)

            if model == "homography":
                H_delta_ds, inl = cv2.findHomography(src_xy, tgt_xy, cv2.RANSAC, ransacReprojThreshold=hth)
                ninl = int(inl.sum()) if inl is not None else 0
                dbg(f"[RANSAC] homography delta(DS) inliers={ninl}/{len(src_xy)} thr={hth}")

                if H_delta_ds is None:
                    kind, X = "affine", A_prev.copy()
                else:
                    H_delta_full = lift_homography_from_ds(H_delta_ds, ds)
                    H_final = np.asarray(H_delta_full, np.float64) @ A_prev3
                    kind, X = "homography", H_final

            elif model == "similarity":
                A_delta_ds, inl = cv2.estimateAffinePartial2D(
                    src_xy, tgt_xy, cv2.RANSAC, ransacReprojThreshold=hth
                )
                ninl = int(inl.sum()) if inl is not None else 0
                dbg(f"[RANSAC] similarity delta(DS) inliers={ninl}/{len(src_xy)} thr={hth}")

                if A_delta_ds is None:
                    kind, X = "similarity", _project_to_similarity(A_prev)
                else:
                    A_delta_full = lift_affine_2x3_from_ds(A_delta_ds, ds)
                    A_final3 = _A3(A_delta_full) @ A_prev3
                    A_final = A_final3[:2, :]
                    kind, X = "similarity", _project_to_similarity(A_final)

            elif model in ("poly3", "poly4"):
                order = 3 if model == "poly3" else 4
                src_full = (np.asarray(src_xy, np.float32) * float(ds)).astype(np.float32)
                tgt_full = (np.asarray(tgt_xy, np.float32) * float(ds)).astype(np.float32)

                cx, cy = _fit_poly_xy(src_full, tgt_full, order=order)
                map_x, map_y = _poly_eval_grid(cx, cy, Wref, Href, order=order)
                kind, X = model, (map_x, map_y)

            else:
                kind, X = "affine", A_prev.copy()

        # 4) warp full-res image
        Hh, Ww = Href, Wref

        if kind in ("affine", "similarity"):
            A = np.asarray(X, np.float64).reshape(2, 3)
            if is_mono:
                aligned = cv2.warpAffine(img, A, (Ww, Hh), flags=cv2.INTER_LANCZOS4,
                                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            else:
                aligned = np.stack([
                    cv2.warpAffine(img[..., c], A, (Ww, Hh), flags=cv2.INTER_LANCZOS4,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                    for c in range(img.shape[2])
                ], axis=2)

            drizzle_tuple = ("affine", A.astype(np.float64))
            warp_label = ("similarity" if kind == "similarity" else "affine")

        elif kind == "homography":
            Hm = np.asarray(X, np.float64).reshape(3, 3)
            if is_mono:
                aligned = cv2.warpPerspective(img, Hm, (Ww, Hh), flags=cv2.INTER_LANCZOS4,
                                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            else:
                aligned = np.stack([
                    cv2.warpPerspective(img[..., c], Hm, (Ww, Hh), flags=cv2.INTER_LANCZOS4,
                                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                    for c in range(img.shape[2])
                ], axis=2)

            drizzle_tuple = ("homography", Hm.astype(np.float64))
            warp_label = "homography"

        elif kind in ("poly3", "poly4"):
            map_x, map_y = X
            if is_mono:
                aligned = cv2.remap(img, map_x, map_y, cv2.INTER_LANCZOS4,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            else:
                aligned = np.stack([
                    cv2.remap(img[..., c], map_x, map_y, cv2.INTER_LANCZOS4,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                    for c in range(img.shape[2])
                ], axis=2)

            drizzle_tuple = (align_model, None)
            warp_label = align_model

        if np.isnan(aligned).any() or np.isinf(aligned).any():
            aligned = np.nan_to_num(aligned, nan=0.0, posinf=0.0, neginf=0.0)

        # ---- NEW: warp satellite mask with the SAME final transform ----
        satmask_aligned = None
        if satmask_src is not None:
            try:
                satmask_aligned = _warp_satmask_with_kind(
                    satmask_src, kind, X, (Hh, Ww)
                )
                dbg(f"[satmask] propagated {int(np.count_nonzero(satmask_src))} -> {int(np.count_nonzero(satmask_aligned))} px")
            except Exception as e:
                dbg(f"[satmask] warp failed: {e}")
                satmask_aligned = None

        # 5) save aligned image
        name, _ = os.path.splitext(base)
        if name.endswith("_n"):
            name = name[:-2]
        if not name.endswith("_n_r"):
            name += "_n_r"

        out_path = os.path.join(output_directory, f"{name}.fit")

        from setiastro.saspro.legacy.image_manager import save_image as _legacy_save
        _legacy_save(
            img_array=aligned,
            filename=out_path,
            original_format="fit",
            bit_depth=None,
            original_header=hdr,
            is_mono=is_mono
        )

        # ---- NEW: save aligned satellite sidecar mask if present ----
        if satmask_aligned is not None:
            try:
                saved_p = _save_satmask_sidecar(out_path, satmask_aligned)
                dbg(f"[satmask] saved aligned sidecar: {os.path.basename(saved_p)} ({int(np.count_nonzero(satmask_aligned))} px)")
            except Exception as e:
                dbg(f"[satmask] save failed: {e}")

        msg = (
            f"🌀 Distortion Correction on {base}: warp={warp_label}\n"
            f"💾 Wrote {os.path.basename(out_path)} [{warp_label}]"
        )
        if debug_lines:
            msg = "\n".join(debug_lines) + "\n" + msg
        return (orig_path, out_path, msg, True, drizzle_tuple)

    except Exception as e:
        if debug_lines:
            pre = "\n".join(debug_lines)
            return (orig_path, "", f"{pre}\n⚠️ Finalize error {os.path.basename(orig_path)}: {e}", False, None)
        return (orig_path, "", f"⚠️ Finalize error {os.path.basename(orig_path)}: {e}", False, None)

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
        Refinement worker ALWAYS computes incremental deltas in affine/similarity space,
        even if the FINAL requested model is homography/poly3/poly4.

        The final non-affine model (if any) is applied in _finalize_write_job only.
        """
        try:
            _cap_native_threads_once()
            try:
                curr = sep.get_extract_pixstack()
                if curr < 1_500_000:
                    sep.set_extract_pixstack(1_500_000)
            except Exception:
                pass

            # --- Load ORIGINAL frame → grayscale float32 ---
            with fits.open(self.original_file, memmap=True) as hdul:
                arr = hdul[0].data
                if arr is None:
                    self.signals.error.emit(f"Could not load {self.original_file}")
                    return
                gray = arr if arr.ndim == 2 else np.mean(arr, axis=2)
                gray_small = np.nan_to_num(gray, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

            ref_small = self.reference_image
            if ref_small is None:
                self.signals.error.emit("Worker missing reference preview.")
                return
            Href, Wref = ref_small.shape[:2]

            # ✅ Refinement solve model: always affine or similarity
            model_req = (self.model_name or "affine").lower()
            if model_req in ("no_distortion", "nodistortion", "similarity"):
                refine_model = "similarity"
            else:
                refine_model = "affine"  # includes when final requested is homography/poly*

            T_prev = np.array(self.current_transform, dtype=np.float32).reshape(2, 3)
            use_warp = not np.allclose(
                T_prev,
                np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32),
                rtol=1e-5, atol=1e-5
            )

            if use_warp and cv2 is not None:
                src_for_match = cv2.warpAffine(
                    gray_small, T_prev, (Wref, Href),
                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101,
                )
            else:
                if gray_small.shape != ref_small.shape and cv2 is not None:
                    src_for_match = cv2.resize(gray_small, (Wref, Href), interpolation=cv2.INTER_LINEAR)
                else:
                    src_for_match = gray_small

            try:
                if refine_model == "similarity":
                    transform = compute_similarity_transform_astroalign_cropped(
                        src_for_match, ref_small,
                        limit_stars=getattr(self, "limit_stars", None),
                        det_sigma=getattr(self, "det_sigma", 12.0),
                        minarea=getattr(self, "minarea", 10),
                        h_reproj=getattr(self, "h_reproj", 3.0),
                    )
                else:
                    transform = self.compute_affine_transform_astroalign(
                        src_for_match, ref_small,
                        limit_stars=getattr(self, "limit_stars", None),
                        det_sigma=getattr(self, "det_sigma", 12.0),
                        minarea=getattr(self, "minarea", 10),
                    )
            except Exception as e:
                msg = str(e)
                base = os.path.basename(self.original_file)
                if "of matching triangles exhausted" in msg.lower():
                    self.signals.error.emit(f"Astroalign failed for {base}: List of matching triangles exhausted")
                else:
                    self.signals.error.emit(f"Astroalign failed for {base}: {msg}")
                return

            if transform is None:
                base = os.path.basename(self.original_file)
                self.signals.error.emit(f"Astroalign failed for {base} – skipping (no transform returned)")
                return

            transform = np.array(transform, dtype=np.float64).reshape(2, 3)

            # Similarity projection safety (no shear)
            if refine_model == "similarity":
                transform = _project_to_similarity(transform)

            key = os.path.normpath(self.original_file)
            self.signals.result_transform.emit(key, transform)
            self.signals.progress.emit(
                f"Astroalign delta for {os.path.basename(self.original_file)} "
                f"(refine={refine_model}, final={self.model_name}): dx={transform[0,2]:.2f}, dy={transform[1,2]:.2f}"
            )
            self.signals.result.emit(self.original_file)

        except Exception as e:
            self.signals.error.emit(f"Error processing {self.original_file}: {e}")

    @staticmethod
    def compute_affine_transform_astroalign(source_img, reference_img,
                                            scale=1.20,
                                            limit_stars: int | None = None,
                                            det_sigma: float = 12.0,
                                            minarea: int = 10):
        """
        Solve affine on a ~1.2x center crop of reference and lift into full-ref coords.
        Uses bright, spatially distributed control points first; falls back to
        image-based Astroalign only if needed.
        """
        import numpy as np
        from setiastro.saspro import astroalign

        global _AA_LOCK

        Hs, Ws = source_img.shape[:2]
        Hr, Wr = reference_img.shape[:2]

        h = min(int(round(Hs * scale)), Hr)
        w = min(int(round(Ws * scale)), Wr)
        y0 = max(0, (Hr - h) // 2)
        x0 = max(0, (Wr - w) // 2)
        ref_crop = reference_img[y0:y0+h, x0:x0+w]

        kwargs = {"detection_sigma": float(det_sigma), "min_area": int(minarea)}
        if limit_stars is not None:
            kwargs["max_control_points"] = int(limit_stars)

        with _AA_LOCK:
            try:
                # Use bright + spatially distributed stars first
                src_pts = _detect_stars_uniform(
                    source_img,
                    det_sigma=float(det_sigma),
                    minarea=int(minarea),
                    grid=(4, 4),
                    max_per_cell=25,
                    max_total=(int(limit_stars) if limit_stars is not None else 500)
                )
                ref_pts = _detect_stars_uniform(
                    ref_crop,
                    det_sigma=float(det_sigma),
                    minarea=int(minarea),
                    grid=(4, 4),
                    max_per_cell=25,
                    max_total=(int(limit_stars) if limit_stars is not None else 500)
                )

                cov_src = _coverage_fraction(src_pts, Hs, Ws, grid=(4, 4))
                cov_ref = _coverage_fraction(ref_pts, h,  w,  grid=(4, 4))
                if cov_src < 0.5 or cov_ref < 0.5:
                    print(f"[AA] low coverage src={cov_src:.2f}, ref={cov_ref:.2f} for static affine crop solve")

                if src_pts.shape[0] >= 8 and ref_pts.shape[0] >= 8:
                    pt_kwargs = {}
                    if limit_stars is not None:
                        pt_kwargs["max_control_points"] = int(limit_stars)

                    tform, _ = astroalign.find_transform(src_pts, ref_pts, **pt_kwargs)
                else:
                    raise RuntimeError("Too few uniform points, falling back to image-based AA.")

            except Exception:
                # fallback to original image-based Astroalign
                try:
                    tform, _ = astroalign.find_transform(
                        np.ascontiguousarray(source_img.astype(np.float32)),
                        np.ascontiguousarray(ref_crop.astype(np.float32)),
                        **kwargs
                    )
                except TypeError:
                    legacy_kwargs = {}
                    if "max_control_points" in kwargs:
                        legacy_kwargs["max_control_points"] = kwargs["max_control_points"]
                    tform, _ = astroalign.find_transform(
                        np.ascontiguousarray(source_img.astype(np.float32)),
                        np.ascontiguousarray(ref_crop.astype(np.float32)),
                        **legacy_kwargs
                    )

        P = np.asarray(tform.params, dtype=np.float64)
        if P.shape == (3, 3):
            T = np.array([[1, 0, x0], [0, 1, y0], [0, 0, 1]], dtype=np.float64)
            return (T @ P)[0:2, :]
        elif P.shape == (2, 3):
            A3 = np.vstack([P, [0, 0, 1]])
            T  = np.array([[1, 0, x0], [0, 1, y0], [0, 0, 1]], dtype=np.float64)
            return (T @ A3)[0:2, :]
        return None

def _project_to_similarity(T2x3: np.ndarray) -> np.ndarray:
    T2x3 = np.asarray(T2x3, np.float64).reshape(2,3)
    R = T2x3[:, :2]
    t = T2x3[:, 2]
    U, S, Vt = np.linalg.svd(R)
    rot = U @ Vt
    if np.linalg.det(rot) < 0:
        U[:, -1] *= -1
        rot = U @ Vt
    s = float((S[0] + S[1]) * 0.5)  # uniform scale
    Rsim = rot * s
    out = np.zeros((2,3), np.float64)
    out[:, :2] = Rsim
    out[:, 2] = t
    return out

def _detect_stars_uniform(img32: np.ndarray,
                          det_sigma: float = 12.0,
                          minarea: int = 10,
                          grid=(4,4),
                          max_per_cell: int = 25,
                          max_total: int = 500,
                          min_fwhm: float = 1.2,
                          max_ellipticity: float = 0.6) -> np.ndarray:
    import numpy as np
    import sep

    img32 = np.asarray(img32, np.float32, order="C")
    H, W = img32.shape[:2]

    bkg = sep.Background(img32, bw=64, bh=64)
    thresh = float(det_sigma) * float(bkg.globalrms)

    # Request shape parameters from SEP
    objs = sep.extract(img32 - bkg.back(), thresh, minarea=int(minarea),
                       segmentation_map=False)
    if objs is None or len(objs) == 0:
        return np.empty((0,2), np.float32)

    # ── Hot pixel rejection ──────────────────────────────────────────
    # 1) FWHM proxy: SEP gives a2 (semi-major axis). Hot pixels have a2 ~ 0.5px
    #    Real stars have a2 >= ~1.0px even when undersampled.
    a2 = objs["a"].astype(np.float32)   # semi-major axis (px)
    b2 = objs["b"].astype(np.float32)   # semi-minor axis (px)

    fwhm_approx = 2.0 * np.sqrt(2.0 * np.log(2.0)) * a2  # Gaussian FWHM estimate

    # ellipticity = 1 - b/a (0=round, 1=infinitely elongated)
    ellipticity = np.where(a2 > 1e-6, 1.0 - (b2 / a2), 1.0)

    # npix = number of pixels above threshold — hot pixels are typically 1-4px
    npix = objs["npix"].astype(np.int32) if "npix" in objs.dtype.names else np.ones(len(objs), np.int32)

    valid = (
        (fwhm_approx >= float(min_fwhm)) &   # reject single-pixel spikes
        (ellipticity <= float(max_ellipticity)) &  # reject cosmic rays / streaks
        (npix >= int(minarea))               # belt-and-suspenders with npix
    )

    if valid.sum() == 0:
        # All rejected — fall back to size filter only (never return empty if detections exist)
        valid = npix >= max(2, int(minarea) // 2)

    objs = objs[valid]
    if len(objs) == 0:
        return np.empty((0,2), np.float32)
    # ────────────────────────────────────────────────────────────────

    # Sort by flux desc (brightest surviving real stars first)
    order = np.argsort(objs["flux"])[::-1]
    xs = objs["x"][order].astype(np.float32)
    ys = objs["y"][order].astype(np.float32)

    gy, gx = int(grid[0]), int(grid[1])
    cell_w = W / gx
    cell_h = H / gy

    keep_counts = np.zeros((gy, gx), dtype=np.int32)
    pts = []

    for x, y in zip(xs, ys):
        cx = int(x / cell_w)
        cy = int(y / cell_h)
        if cx < 0 or cy < 0 or cx >= gx or cy >= gy:
            continue
        if keep_counts[cy, cx] >= max_per_cell:
            continue
        keep_counts[cy, cx] += 1
        pts.append((x, y))
        if len(pts) >= max_total:
            break

    if not pts:
        return np.empty((0,2), np.float32)
    return np.asarray(pts, np.float32)

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
        self._cancel_check = None
        pw = parent_window
        if pw is not None and hasattr(pw, "_cancel_event"):
            # capture the shared event directly — thread-safe to read from here
            self._cancel_event = pw._cancel_event
        else:
            self._cancel_event = None        
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
        self.downsample = int(self.align_prefs.get("downsample", 3))
        self.drizzle_xforms = {}  # {orig_norm_path: (kind, matrix)}
        self.min_fwhm        = float(self.align_prefs.get("min_fwhm", 1.2))
        self.max_ellipticity = float(self.align_prefs.get("max_ellipticity", 0.6))

    @staticmethod
    def _aa_model_and_residual(src_gray: np.ndarray,
                            ref2d: np.ndarray,
                            model: str,
                            h_reproj: float,
                            det_sigma: float,
                            minarea: int,
                            max_control_points: int | None = None):
        """
        AA on a ~1.2× center crop; lift matches to full coords; re-estimate requested model.
        Returns: (kind, X, residual_rms_px, n_inliers)

        kind in {"affine","homography","similarity"} or base affine/homography if poly fails upstream.
        For poly3/4 we still return the base model here; finalize does the true residual warp.
        """
        import numpy as np
        from setiastro.saspro import astroalign
        import cv2

        src = np.ascontiguousarray(src_gray.astype(np.float32))
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

        kwargs = {"detection_sigma": float(det_sigma), "min_area": int(minarea)}
        if max_control_points is not None:
            kwargs["max_control_points"] = int(max_control_points)

        # ---- 2) astroalign correspondences (adaptive tiling) ----
        src_xy, tgt_xy, best_P, best_xy0 = _aa_find_pairs_multitile(
            src, ref,
            scale=1.20, tiles=1,
            det_sigma=det_sigma, minarea=minarea,
            max_control_points=max_control_points
        )

        if src_xy is None or len(src_xy) < 8:
            raise RuntimeError("astroalign produced too few matches")

        # ✅ your spread / covariance gate:
        if not _points_spread_ok(tgt_xy, Wr, Hr):
            src_xy2, tgt_xy2, best_P2, best_xy0_2 = _aa_find_pairs_multitile(
                src, ref,
                scale=1.20, tiles=3,           # 3x3 grid
                det_sigma=det_sigma, minarea=minarea,
                max_control_points=max_control_points
            )
            if src_xy2 is not None and len(src_xy2) > len(src_xy):
                src_xy, tgt_xy = src_xy2, tgt_xy2
                best_P, best_xy0 = best_P2, best_xy0_2

        # ---- 3) base full-ref transform from best_P + crop translation ----
        x0, y0 = best_xy0
        P = np.asarray(best_P, dtype=np.float64)
        if P.shape == (3, 3):
            T  = np.array([[1,0,x0],[0,1,y0],[0,0,1]], dtype=np.float64)
            base_kind = "homography"
            base_X    = T @ P
        else:
            A3 = np.vstack([P[0:2,:], [0,0,1]])
            T  = np.array([[1,0,x0],[0,1,y0],[0,0,1]], dtype=np.float64)
            base_kind = "affine"
            base_X    = (T @ A3)[0:2, :]

        # helper: force an affine 2x3 into nearest similarity (no shear)
        def _affine_to_similarity(A2x3: np.ndarray) -> np.ndarray:
            A2x3 = np.asarray(A2x3, np.float64).reshape(2, 3)
            R = A2x3[:, :2]
            t = A2x3[:, 2]
            # SVD to get closest rotation + uniform scale
            U, S, Vt = np.linalg.svd(R)
            rot = U @ Vt
            if np.linalg.det(rot) < 0:
                U[:, -1] *= -1
                rot = U @ Vt
            s = float((S[0] + S[1]) * 0.5)
            Rsim = rot * s
            out = np.zeros((2, 3), dtype=np.float64)
            out[:, :2] = Rsim
            out[:, 2] = t
            return out

        # ---- 4) re-estimate requested model (RANSAC) with lifted pairs ----
        hth = float(h_reproj)
        m = (model or "affine").lower()
        if m in ("no_distortion", "nodistortion"):
            m = "similarity"

        if m == "homography":
            H, inl = cv2.findHomography(src_xy, tgt_xy, cv2.RANSAC, ransacReprojThreshold=hth)
            if H is None:
                kind, X = base_kind, base_X
                inl_mask = None
            else:
                kind, X = "homography", np.asarray(H, np.float64)
                inl_mask = inl.ravel().astype(bool)

        elif m == "affine":
            A, inl = cv2.estimateAffine2D(src_xy, tgt_xy, method=cv2.RANSAC, ransacReprojThreshold=hth)
            if A is None:
                kind, X = base_kind, base_X
                inl_mask = None
            else:
                kind, X = "affine", np.asarray(A, np.float64)
                inl_mask = inl.ravel().astype(bool)

        elif m == "similarity":
            A, inl = cv2.estimateAffinePartial2D(src_xy, tgt_xy, method=cv2.RANSAC, ransacReprojThreshold=hth)
            if A is None:
                # fallback: project base to similarity so we NEVER shear
                if base_kind == "affine":
                    kind, X = "similarity", _affine_to_similarity(base_X)
                else:
                    kind, X = base_kind, base_X
                inl_mask = None
            else:
                kind, X = "similarity", np.asarray(A, np.float64)
                inl_mask = inl.ravel().astype(bool)

        else:
            # poly3/4: report residual versus base model; finalize applies poly residual warp
            kind, X  = base_kind, base_X
            inl_mask = None

        # ---- 5) residual RMS (px) using whichever model we returned ----
        if kind == "homography":
            ones = np.ones((src_xy.shape[0], 1), dtype=np.float32)
            P3   = np.hstack([src_xy.astype(np.float32), ones]).T
            Q    = (np.asarray(X, np.float32) @ P3)
            pred = (Q[:2, :] / Q[2:3, :]).T
        else:  # affine or similarity (2x3)
            A2 = np.asarray(X, np.float32).reshape(2, 3)
            pred = (src_xy @ A2[:, :2].T) + A2[:, 2]

        if inl_mask is not None and inl_mask.sum() >= 10:
            res = np.linalg.norm(pred[inl_mask] - tgt_xy[inl_mask], axis=1)
            nin = int(inl_mask.sum())
        else:
            res = np.linalg.norm(pred - tgt_xy, axis=1)
            nin = int(res.shape[0])

        residual_rms = float(np.sqrt(np.mean(res**2))) if res.size else float("inf")
        return kind, X, residual_rms, nin

    def _is_cancelled(self) -> bool:
        ev = getattr(self, "_cancel_event", None)
        return bool(ev is not None and ev.is_set())

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

            self.ref_small_full = np.ascontiguousarray(ref2d.astype(np.float32, copy=False))
            print(f"[SRT] run() started, files={len(self.original_files)}, ref={self.reference}")
            # Solve grid downsample factor, user-configurable in Stacking
            # Settings (default 3, stored in self.downsample via align_prefs).
            # Lower values give a finer solve grid; some high-resolution
            # sensors need ds=2 to avoid a bimodal registration lock on
            # meridian-flipped frames.
            ds = max(1, int(self.downsample))
            self.solve_downsample = ds

            if ds > 1 and cv2 is not None:
                new_hw = (max(1, ref2d.shape[1] // ds), max(1, ref2d.shape[0] // ds))  # (W, H)
                ref_ds = cv2.resize(self.ref_small_full, new_hw, interpolation=cv2.INTER_AREA)
            else:
                ref_ds = self.ref_small_full

            self.ref_small = self.ref_small_full               # keep existing attribute name (full)
            self.ref_small_ds = np.ascontiguousarray(ref_ds.astype(np.float32, copy=False))

            # Initialize transforms to identity for EVERY original frame
            self.alignment_matrices = {os.path.normpath(f): IDENTITY_2x3.copy() for f in self.original_files}
            self.delta_transforms = {}

            # Progress totals (units = number of worker completions across passes)
            # Progress totals:
            #   passes = N * passes
            #   finalize = N
            N = len(self.original_files)
            P = max(1, int(self.max_refinement_passes))

            self._done = 0
            self._total = (N * P) + N   # <-- IMPORTANT: include finalize
            self.progress_step.emit(self._done, self._total)  # optional but helps UI reset immediately

            # Registration passes (compute deltas only)
            for pass_idx in range(self.max_refinement_passes):
                if self._is_cancelled():
                    self.registration_complete.emit(False, "Cancelled by user.")
                    return
                self.progress_update.emit(f"⏳ Refinement Pass {pass_idx + 1}/{self.max_refinement_passes}…")
                print(f"[SRT] starting pass {pass_idx}, work_list={len(self.original_files)}")
                success, msg = self.run_one_registration_pass(None, None, pass_idx)
                print(f"[SRT] pass {pass_idx} completed: {success} ({msg})")
                if self._is_cancelled():
                    self.registration_complete.emit(False, "Cancelled by user.")
                    return
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
        import os
        import cv2
        import time

        # Requested final model (used ONLY in finalize)
        final_model = (self.align_model or "affine").lower()

        # ✅ Refinement model: affine or similarity only
        if final_model in ("no_distortion", "nodistortion", "similarity"):
            refine_model = "similarity"
        else:
            refine_model = "affine"

        ref_small_ds = np.ascontiguousarray(self.ref_small_ds.astype(np.float32, copy=False))
        Href_ds, Wref_ds = ref_small_ds.shape[:2]
        ds = max(1, int(getattr(self, "solve_downsample", 1)))

        # --- reverse map: current_path -> original_key
        rev_current_to_orig = {}
        for orig_k, curr_p in self.file_key_to_current_path.items():
            rev_current_to_orig[os.path.normpath(curr_p)] = os.path.normpath(orig_k)

        resample_flag = cv2.INTER_AREA if pass_index == 0 else cv2.INTER_LINEAR

        # Work list: pass 0 all; later passes skip within tolerance
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
            for _ in range(skipped):
                self._increment_progress()

        if not work_list:
            self.transform_deltas.append([
                self.delta_transforms.get(os.path.normpath(f), 0.0)
                for f in self.original_files
            ])
            return True, "Pass complete (nothing to refine)."

        procs = max(2, min((os.cpu_count() or 8), 32))
        self.progress_update.emit(f"Using {procs} processes for stellar alignment (refine={refine_model}).")

        timeout_sec = int(self.align_prefs.get("timeout_per_job_sec", 300))

        jobs = []
        for orig_key in work_list:
            ok = os.path.normpath(orig_key)

            # IMPORTANT: refinement reads ORIGINAL frame (no intermediate saves)
            current_path = ok

            current_transform = self.alignment_matrices.get(ok)
            if current_transform is None:
                current_transform = IDENTITY_2x3.copy()

            jobs.append((
                current_path,
                current_transform,
                ref_small_ds, int(Wref_ds), int(Href_ds),
                resample_flag, float(self.det_sigma),
                int(self.limit_stars) if self.limit_stars is not None else None,
                int(self.minarea),
                refine_model, float(self.h_reproj),
                int(ds),
                float(self.min_fwhm),          # ← new
                float(self.max_ellipticity),   # ← new                
            ))

        executor = _make_executor(procs)
        try:
            fut_info, pending = {}, set()
            for j in jobs:
                f = executor.submit(_solve_delta_job, j)
                fut_info[f] = j[0]
                pending.add(f)

            while pending:
                if self._is_cancelled():
                    self.progress_update.emit("⏹ Cancelling alignment — stopping at next frame boundary…")
                    for f in pending:
                        f.cancel()
                    break
                done, pending = wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)
                for fut in done:
                    returned_path = fut_info.pop(fut, "<unknown>")
                    try:
                        curr_path_r, T_new, err = fut.result()
                    except Exception as e:
                        curr_path_r, T_new, err = (returned_path or "<unknown>", None, f"Worker crashed: {e}")

                    curr_norm = os.path.normpath(curr_path_r)
                    k_orig = rev_current_to_orig.get(curr_norm, curr_norm)

                    if err:
                        self.on_worker_error(err)
                        k_orig_for_err = rev_current_to_orig.get(
                            os.path.normpath(returned_path),
                            os.path.normpath(returned_path)
                        )
                        if pass_index == 0:
                            self.alignment_matrices[k_orig_for_err] = None
                        self._increment_progress()
                        continue

                    T_new = np.array(T_new, dtype=np.float64).reshape(2, 3)

                    if refine_model == "similarity":
                        T_new = _project_to_similarity(T_new)

                    self.delta_transforms[k_orig] = float(np.hypot(T_new[0, 2], T_new[1, 2]))

                    T_prev_raw = self.alignment_matrices.get(k_orig, IDENTITY_2x3)
                    if T_prev_raw is None or np.asarray(T_prev_raw).size != 6:
                        T_prev_raw = IDENTITY_2x3
                    T_prev = np.array(T_prev_raw, dtype=np.float64).reshape(2, 3)
                    prev_3 = np.vstack([T_prev, [0, 0, 1]])
                    new_3  = np.vstack([T_new,  [0, 0, 1]])
                    self.alignment_matrices[k_orig] = (new_3 @ prev_3)[0:2, :]

                    self.on_worker_progress(
                        f"Astroalign delta for {os.path.basename(curr_path_r)} "
                        f"(refine={refine_model}, final={final_model}): dx={T_new[0,2]:.2f}, dy={T_new[1,2]:.2f}"
                    )
                    self._increment_progress()
            if self._is_cancelled():
                return False, "Cancelled by user."
            pass_deltas, aligned_count = [], 0
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
                self.progress_update.emit(f"Within tolerance (≤ {self.shift_tolerance:.2f}px): {aligned_count} frame(s)")
            return True, "Pass complete."
        finally:
            try:
                executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass

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
        from photutils.detection import DAOStarFinder
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
        import shutil
        print(f"[SRT] _finalize_writes called, {len(self.original_files)} files, output={self.output_directory}")

        self.drizzle_xforms = {}

        try:
            Href, Wref = self.reference_image_2d.shape[:2]
        except Exception:
            self.progress_update.emit("⚠️ No reference image available; aborting finalize.")
            return
        self._ref_shape_for_sasd = (Href, Wref)

        tmpdir = tempfile.mkdtemp(prefix="sas_align_")
        ref_npy = os.path.join(tmpdir, "ref2d.npy")
        try:
            np.save(ref_npy, np.asarray(self.reference_image_2d, dtype=np.float32))
        except Exception as e:
            self.progress_update.emit(f"⚠️ Failed to persist reference for workers: {e}")
            try: shutil.rmtree(tmpdir, ignore_errors=True)
            except Exception as e:
                import logging
                logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")
            return

        finalize_workers = int(self.align_prefs.get("finalize_workers", min(os.cpu_count() or 8, 8)))
        finalize_workers = max(2, finalize_workers)

        jobs = []
        for orig_path in self.original_files:
            k = os.path.normpath(orig_path)
            A = self.alignment_matrices.get(k, IDENTITY_2x3)

            # Skip frames that failed alignment entirely
            if A is None:
                self.progress_update.emit(
                    f"⏭️ Skipping {os.path.basename(orig_path)} — alignment failed, not writing output."
                )
                self._increment_progress()
                continue

            jobs.append((
                orig_path,
                self.align_model,
                (Href, Wref),
                ref_npy,
                np.asarray(A, np.float64),
                float(self.h_reproj),
                self.output_directory,
                float(self.det_sigma),
                int(self.minarea),
                int(self.limit_stars) if self.limit_stars is not None else None,
                float(getattr(self, "min_fwhm", 1.2)),
                float(getattr(self, "max_ellipticity", 0.6)),
                int(getattr(self, "solve_downsample", 3)),
            ))
        self.progress_update.emit(f"📝 Finalizing aligned outputs with {finalize_workers} processes…")

        ok = 0
        try:
            with _make_executor(finalize_workers) as ex:
                futs = [ex.submit(_finalize_write_job, j) for j in jobs]
                for fut in as_completed(futs):
                    try:
                        orig_path, out_path, msg, success, drizzle = fut.result()
                    except Exception as e:
                        self.progress_update.emit(f"⚠️ Finalize worker crashed: {e}")
                        self._increment_progress()
                        continue

                    if msg:
                        for line in (msg.splitlines() or [msg]):
                            self.progress_update.emit(line)

                    if success:
                        ok += 1
                        k = os.path.normpath(orig_path)
                        self.file_key_to_current_path[k] = out_path

                        if isinstance(drizzle, tuple) and len(drizzle) == 2:
                            kind, M = drizzle
                            try:
                                if kind == "affine" and M is not None:
                                    self.drizzle_xforms[k] = ("affine", np.asarray(M, np.float64).reshape(2, 3))
                                elif kind == "homography" and M is not None:
                                    self.drizzle_xforms[k] = ("homography", np.asarray(M, np.float64).reshape(3, 3))
                                elif M is None:
                                    # poly or other model — fall back to accumulated affine for drizzle
                                    A_fallback = self.alignment_matrices.get(k)
                                    if A_fallback is not None:
                                        self.drizzle_xforms[k] = ("affine", np.asarray(A_fallback, np.float64).reshape(2, 3))
                                else:
                                    self.drizzle_xforms[k] = (str(kind), None)
                            except Exception as ex:
                                self.progress_update.emit(f"⚠️ Could not store drizzle transform for {os.path.basename(orig_path)}: {ex}")
                    self._increment_progress()
        finally:
            try: shutil.rmtree(tmpdir, ignore_errors=True)
            except Exception as e:
                import logging
                logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")
            gc.collect()

        try:
            sasd_path = os.path.join(self.output_directory, "alignment_transforms.sasd")
            self._save_alignment_transforms_sasd(sasd_path)
            self.progress_update.emit("✅ Transform file saved as alignment_transforms.sasd")
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
                    M_raw = self.alignment_matrices.get(k)
                    if M_raw is None:
                        f.write(f"FILE: {k}\n")
                        f.write("KIND: affine\n")
                        f.write("MATRIX:\nUNSUPPORTED\n\n")
                        continue
                    M_aff = np.asarray(M_raw, dtype=np.float32)
                    kind, M = "affine", M_aff

                f.write(f"FILE: {k}\n")
                f.write(f"KIND: {kind}\n")
                f.write("MATRIX:\n")

                _fmt = lambda x: f"{float(x):.16g}"

                if kind == "homography" and M is not None:
                    try:
                        H = np.asarray(M, np.float64).reshape(3, 3)
                        f.write(f"{_fmt(H[0,0])}, {_fmt(H[0,1])}, {_fmt(H[0,2])}\n")
                        f.write(f"{_fmt(H[1,0])}, {_fmt(H[1,1])}, {_fmt(H[1,2])}\n")
                        f.write(f"{_fmt(H[2,0])}, {_fmt(H[2,1])}, {_fmt(H[2,2])}\n\n")
                    except Exception:
                        f.write("UNSUPPORTED\n\n")
                elif kind in ("affine", "similarity") and M is not None:
                    try:
                        A = np.asarray(M, np.float64).reshape(2, 3)
                        f.write(f"{_fmt(A[0,0])}, {_fmt(A[0,1])}, {_fmt(A[0,2])}\n")
                        f.write(f"{_fmt(A[1,0])}, {_fmt(A[1,1])}, {_fmt(A[1,2])}\n\n")
                    except Exception:
                        f.write("UNSUPPORTED\n\n")
                else:
                    # M is None (poly models) or unknown kind — write what we can from alignment_matrices
                    M_raw = self.alignment_matrices.get(k)
                    if M_raw is not None:
                        try:
                            A = np.asarray(M_raw, np.float64).reshape(2, 3)
                            f.write(f"# fallback affine from refinement\n")
                            f.write(f"{_fmt(A[0,0])}, {_fmt(A[0,1])}, {_fmt(A[0,2])}\n")
                            f.write(f"{_fmt(A[1,0])}, {_fmt(A[1,1])}, {_fmt(A[1,2])}\n\n")
                        except Exception:
                            f.write("UNSUPPORTED\n\n")
                    else:
                        f.write("UNSUPPORTED\n\n")

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
        import platform
        if platform.system() == "Darwin":
            self.setWindowFlag(Qt.WindowType.Tool, True)  
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        #self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)        
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


# ---------------------------------------------------------------------
# Backward-compat shim: Mosaic Master moved to mosaic_master.py
# ---------------------------------------------------------------------
# Deferred (module __getattr__) so there's no import cycle — mosaic_master
# imports qs_int/qs_float/qs_bool and ASTROMETRY_API_URL back from here.
# TODO: delete once every caller imports from mosaic_master directly.
_MOVED_TO_MOSAIC = {
    "MosaicMasterDialog",
    "MosaicPreviewWindow",
    "MosaicSettingsDialog",
    "PolyGradientRemoval",
    "coerce_to_header",
    "sanitize_wcs_header",
    "get_wcs_from_header",
    "robust_api_request",
    "scale_image_for_display",
    "generate_minimal_fits_header",
    "save_api_key",
    "load_api_key",
    "estimate_background_level",
    "solve_linear_match",
}


def __getattr__(name):
    if name in _MOVED_TO_MOSAIC:
        import importlib
        mod = importlib.import_module("setiastro.saspro.mosaic_master")
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")