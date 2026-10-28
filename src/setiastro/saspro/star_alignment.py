#saspro.star_alignment.py
from __future__ import annotations

import os
import math
import random
import sys
from PyQt6.QtCore import QByteArray
from setiastro.saspro.help_support import make_help_button


# ─────────────────────────────────────────────────────────────────────
# Hi-sharpness warp kernel (used by every warp site)
#
# Lanczos4 is objectively the sharpest OpenCV ships and also the ringiest
# — its negative lobes cause the dark-halo artifact on bright narrowband
# stars that bit us in 1.22.6. The anti-ring guard (`_warp_antiring`)
# contains most of the damage, but a user who still sees traces of
# ringing can swap to bicubic (a Mitchell-family cubic kernel with much
# shorter negative lobes) without a rebuild.
#
# Three ways to pick the kernel, in priority order:
#   1. QSettings  stacking/align/kernel  (written by the Star Alignment UI
#                                         dropdown and the Stacking Suite
#                                         settings dropdown — shared key)
#   2. Env var    SASPRO_WARP_KERNEL     (set once at launcher time)
#   3. Default    "lanczos"
#
# Allowed values (case-insensitive):
#   "lanczos" | "lanczos4"      → cv2.INTER_LANCZOS4 (sharpest, default)
#   "cubic" | "mitchell" | "bicubic"
#                               → cv2.INTER_CUBIC    (recommended if rings persist)
#   "linear" | "bilinear"       → cv2.INTER_LINEAR   (no sharpening at all)
#
# True Mitchell-Netravali (B=1/3, C=1/3) isn't in OpenCV; cv2.INTER_CUBIC
# is close enough (Catmull-Rom-ish) for a drop-in. Switching to CUBIC
# costs some perceived sharpness on well-aligned frames but is noticeably
# less ringy on bright stars than Lanczos4.
# ─────────────────────────────────────────────────────────────────────
# QSettings key shared with Stacking Suite's "Warp interpolation" dropdown.
# Both UIs read and write THIS key so the user's choice follows them around.
_WARP_KERNEL_SETTINGS_KEY = "stacking/align/kernel"

def _warp_kernel_name_to_cv2(name: str):
    """Map a string kernel name (case-insensitive) to a cv2 interpolation
    flag. Unknown names fall back to Lanczos4 so a typo in settings never
    crashes the warp — just silently uses the sharp default."""
    import cv2
    sel = str(name or "").strip().lower()
    if sel in ("cubic", "mitchell", "bicubic"):
        return cv2.INTER_CUBIC
    if sel in ("linear", "bilinear"):
        return cv2.INTER_LINEAR
    return cv2.INTER_LANCZOS4

def _warp_kernel_name() -> str:
    """Canonical name of the configured hi-sharpness kernel. Reads
    QSettings first, env var next, "lanczos" as a last-resort default.
    Returns one of {"lanczos", "cubic", "linear"}."""
    try:
        from PyQt6.QtCore import QSettings as _QS
        v = _QS().value(_WARP_KERNEL_SETTINGS_KEY, None)
        if v is not None:
            s = str(v).strip().lower()
            if s in ("lanczos", "lanczos4"):
                return "lanczos"
            if s in ("cubic", "mitchell", "bicubic"):
                return "cubic"
            if s in ("linear", "bilinear"):
                return "linear"
    except Exception:
        pass
    env = os.environ.get("SASPRO_WARP_KERNEL", "").strip().lower()
    if env in ("cubic", "mitchell", "bicubic"):
        return "cubic"
    if env in ("linear", "bilinear"):
        return "linear"
    return "lanczos"

def _warp_hi_interp():
    """
    Return the configured hi-sharpness interpolation flag.

    Resolved on every call (NOT cached) so that a UI change to the
    "Warp interpolation" dropdown takes effect on the next warp without a
    restart. The resolution is cheap (one QSettings.value + a string
    compare); the per-call cost is noise next to a Lanczos warp.
    """
    return _warp_kernel_name_to_cv2(_warp_kernel_name())
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
# Frozen builds re-exec the EXE per worker; only enable process pools there
# after a test build confirms multiprocessing.freeze_support() (in __main__)
# catches the children instead of launching extra SASpro windows.
_ALIGN_USE_PROCS_FROZEN = True   # frozen spawn children are caught by freeze_support() at the top of gui_entry.main() (no splash, no GUI)

def _make_executor(max_workers: int):
    """
    Return an appropriate executor.

    - In frozen builds (PyInstaller), we MUST avoid ProcessPoolExecutor
      because each worker spawns a full copy of the EXE (extra SASpro windows).
    - In dev (non-frozen), you can still use processes if you want. For
      now we keep it simple and always use threads – safer everywhere.
    """
    #sasprocesses give real multi-core; the thread pool serialized every solve
    # through the GIL + _AA_LOCK and pinned cv2 to one thread process-wide.
    # Frozen builds re-exec the EXE per worker, so they use processes only once
    # freeze_support() is wired in the entry AND _ALIGN_USE_PROCS_FROZEN is True.
    use_procs = (not _IS_FROZEN) or _ALIGN_USE_PROCS_FROZEN
    # A broken worker environment (NumPy/SciPy mismatch, or two mixed Python
    # envs on Windows) makes spawned workers die on import (BrokenProcessPool),
    # which would leave registration reporting 0 succeeded and aborting the
    # whole stack. Fall back to threads so registration still completes.
    try:
        from setiastro.saspro.worker_env import process_pools_ok
        if not process_pools_ok():
            use_procs = False
    except Exception:
        pass
    if use_procs:
        try:
            _ctx = multiprocessing.get_context("spawn")   # matches frozen + Qt-safe
            return ProcessPoolExecutor(max_workers=max_workers, mp_context=_ctx)
        except Exception:
            pass
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
import numpy as np

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
from setiastro.saspro.widgets.spinboxes import CustomDoubleSpinBox, CustomSpinBox
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

    # Warp kernel (shared with Stacking Suite settings). Canonicalised to
    # one of {"lanczos", "cubic", "linear"} so callers that pass it back
    # to `_warp_kernel_name_to_cv2()` always get a resolvable flag. The
    # key is `stacking/align/kernel` — same key the Star Alignment UI's
    # dropdown writes and the Stacking Suite settings page writes.
    _k_raw = (_get("kernel", "lanczos", str) or "lanczos").strip().lower()
    if _k_raw in ("lanczos", "lanczos4"):
        kernel = "lanczos"
    elif _k_raw in ("cubic", "mitchell", "bicubic"):
        kernel = "cubic"
    elif _k_raw in ("linear", "bilinear"):
        kernel = "linear"
    else:
        kernel = "lanczos"

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
        # Warp kernel used by every anti-ring warp site. See _warp_hi_interp.
        "kernel":          kernel,
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
    from setiastro.saspro import astroalign  # deferred: see tools/apply_import_fixes.py (astroalign)
    tgt32 = _masked(tgt_gray, _zero_ignore_mask(tgt_gray))
    src32 = _masked(src_gray, _zero_ignore_mask(src_gray))
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

def _coerce_to_fits_header(obj) -> "Header | None":
    """
    Normalise anything SASpro might store as "a header" into an
    astropy.io.fits.Header. Accepts a Header (passed through), a dict
    (built into cards), a FITS-serialized string (what shows up in
    metadata['wcs_header']), or None. Returns None when nothing usable
    can be built.

    Pulled out because both the WCS-building path (_wcs_from_header) and
    the view-spill writer (_write_view_to_fits) need the same tolerant
    conversion; keeping it in one place means a user whose live view only
    carries wcs_header gets the WCS preserved through both paths.
    """
    if obj is None:
        return None
    if isinstance(obj, Header):
        return obj
    if isinstance(obj, dict):
        hdr = Header()
        for k, v in obj.items():
            try:
                hdr[k] = v
            except Exception:
                pass
        return hdr if len(hdr) else None
    if isinstance(obj, (str, bytes)):
        s = obj.decode() if isinstance(obj, bytes) else obj
        s = s.strip()
        if not s:
            return None
        try:
            # Header.fromstring is strict about 80-column records; pad to
            # multiple of 80 when a stored string got trimmed on a
            # whitespace boundary.
            if "\n" not in s and len(s) % 80 != 0:
                s = s + " " * (80 - (len(s) % 80))
            return Header.fromstring(s)
        except Exception:
            try:
                # Fallback for human-formatted multi-line dumps.
                return Header.fromstring("\n".join(s.splitlines()), sep="\n")
            except Exception:
                return None
    return None


def _wcs_from_header(header):
    """Sanitize + build a usable WCS (SIP kept) or None. The header parsing
    lives in mosaic_master; import it lazily so there's no import cycle
    (mosaic_master imports qs_*/ASTROMETRY_API_URL back from here).

    Accepts an astropy.io.fits.Header, a dict of card name → value, or a
    raw FITS header string (what SASpro stores under meta['wcs_header']).
    All of those get normalised to Header first via _coerce_to_fits_header
    so get_wcs_from_header sees the shape it expects — the previous version
    silently returned None for the string form, which is why a view with a
    plate solve stored only under 'wcs_header' surfaced as "no WCS".
    """
    hdr = _coerce_to_fits_header(header)
    if hdr is None:
        return None
    try:
        from setiastro.saspro.mosaic_master import get_wcs_from_header
    except Exception:
        return None
    try:
        return get_wcs_from_header(hdr)
    except Exception:
        return None


def _doc_header_or_wcs(doc) -> "object | None":
    """
    Pull a usable FITS header off a SASpro document, checking every key
    the codebase has been seen to use. Returns None if nothing resembling
    a header is on the doc.

    Keys tried, in priority order:
      - metadata['original_header']   (the full FITS header, preferred)
      - metadata['header']            (older naming)
      - metadata['fits_header']       (even older naming)
      - metadata['wcs_header']        (WCS-only slice, what dumping shows
                                       on live SASpro views — valid for
                                       WCS-based tooling even though it
                                       drops the imaging cards)
      - doc.header / doc.fits_header  (attribute fallback — some doc
                                       implementations attach the header
                                       directly rather than through meta)

    This matches StellarAlignmentDialog._doc_header's intent but extends
    it with 'wcs_header' and the attribute fallbacks, which the WCS
    registration path needs — views carrying only wcs_header previously
    looked like "no plate solve".
    """
    if doc is None:
        return None
    try:
        meta = (doc.get_metadata() if hasattr(doc, "get_metadata")
                and callable(doc.get_metadata)
                else getattr(doc, "metadata", {}) or {})
    except Exception:
        meta = {}
    if isinstance(meta, dict):
        for k in ("original_header", "header", "fits_header", "wcs_header"):
            v = meta.get(k)
            if v:
                return v
    # Attribute-level fallback — some doc shapes carry the header directly.
    for attr in ("header", "fits_header", "original_header"):
        v = getattr(doc, attr, None)
        if v:
            return v
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
                        dst[y0:y1, x0:x1, c] = _warp_antiring(
                            lambda c=c: cv2.remap(
                                src[..., c], mapx, mapy,
                                interpolation=cv2.INTER_LANCZOS4,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0.0),
                            lambda c=c: cv2.remap(
                                src[..., c], mapx, mapy,
                                interpolation=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0.0))
                else:
                    dst[y0:y1, x0:x1] = _warp_antiring(
                        lambda: cv2.remap(
                            src, mapx, mapy,
                            interpolation=cv2.INTER_LANCZOS4,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0.0),
                        lambda: cv2.remap(
                            src, mapx, mapy,
                            interpolation=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0.0))

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
    def _wa(ch, flags):
        return cv2.warpAffine(ch, M_2x3, (W, H),
                              flags=flags, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    if target_img.ndim == 2:
        if not target_img.flags['C_CONTIGUOUS']:
            target_img = np.ascontiguousarray(target_img)
        return _warp_antiring(lambda: _wa(target_img, cv2.INTER_LANCZOS4),
                              lambda: _wa(target_img, cv2.INTER_LINEAR))

    # RGB/BGR (<=4 ch): warpAffine handles n-channel natively, and cv2 dilate/erode
    # inside _warp_antiring operate per-channel, so the envelope clamp is correct.
    C = target_img.shape[2]
    if C <= 4:
        if not target_img.flags['C_CONTIGUOUS']:
            target_img = np.ascontiguousarray(target_img)
        return _warp_antiring(lambda: _wa(target_img, cv2.INTER_LANCZOS4),
                              lambda: _wa(target_img, cv2.INTER_LINEAR))

    # Fallback for >4 channels (warpAffine can't do them natively) — per-
    # channel loop with a shared cross-channel star mask (see
    # _warp_hwc_antiring; replaces the old per-channel _warp_antiring stack
    # whose per-channel mask missed narrowband stars in the dim channels).
    return _warp_hwc_antiring(
        C,
        lambda c: _wa(np.ascontiguousarray(target_img[..., c]), cv2.INTER_LANCZOS4),
        lambda c: _wa(np.ascontiguousarray(target_img[..., c]), cv2.INTER_LINEAR),
    )

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

        # Meridian-flip pre-rotate (headless path).
        # If the solve reports a rotation in the 180° window OR a
        # translation large enough (|tx|>frac*W AND |ty|>frac*H) that it
        # can only be explained by a point reflection in the canvas,
        # physically rotate the target buffers with np.rot90(..., 2)
        # (bit-exact) and redo the solve. The downstream warp then only
        # resamples a sub-pixel residual instead of a full 180° rotation,
        # which is where the per-frame PSF drift on flipped stacks comes
        # from. src_shape of the TARGET (what we're warping) is what the
        # test needs; tgt_gray.shape = target source canvas.
        if _is_near_180(M2, src_shape=tgt_gray.shape[:2]):
            tgt_img = np.ascontiguousarray(tgt_img[::-1, ::-1]) if tgt_img.ndim == 2 \
                        else np.ascontiguousarray(tgt_img[::-1, ::-1, :])
            tgt_gray = np.ascontiguousarray(tgt_gray[::-1, ::-1])
            tgt_small = tgt_gray
            transform_obj, _pts = aa_find_transform_with_backoff(tgt_small, ref_small)
            M2 = np.array(transform_obj.params[0:2, :], dtype=np.float64)

        # ---------- warp target like reference size ----------
        ref_h, ref_w = ref_gray.shape[:2]
        aligned = _warp_like_ref(tgt_img, M2, (ref_h, ref_w)).astype(np.float32, copy=False)

        # ---------- overwrite or create new ----------
        overwrite = bool((preset or {}).get("overwrite", False))
        if overwrite:
            # push pixels into target doc
            if hasattr(target_doc, "set_image"):
                target_doc.set_image(aligned, step_name=f"Star Alignment → {ref_name}")
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
    from setiastro.saspro import astroalign  # deferred: see tools/apply_import_fixes.py (astroalign)
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


_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_filename_base(name: str) -> str:
    """
    Reduce a display name to a safe filesystem basename (no extension).
    Collapses runs of unusual characters to a single underscore, trims
    leading/trailing dots & underscores, and falls back to "view" if the
    result is empty. Used when spilling views to temp FITS so the output
    filename stays recognisably tied to the view name.
    """
    if not name:
        return "view"
    s = _SAFE_NAME_RE.sub("_", str(name)).strip("._")
    return s or "view"


def _read_aligned_fits(path: str) -> tuple[np.ndarray, "Header | None"]:
    """
    Read an aligned FITS the worker wrote so we can route it back into a
    SASpro document. Converts the data to float32 and preserves the FITS
    header (WCS, filter, etc.) so overwritten / newly-opened docs don't
    lose that metadata. Matches the integer-normalisation the rest of
    SASpro uses (uint16/255-scale).
    """
    with fits.open(path, memmap=False) as hdul:
        data = hdul[0].data
        hdr = hdul[0].header.copy() if hdul[0].header is not None else None
    if data is None:
        raise RuntimeError(f"FITS at {path} has no data")
    arr = np.asarray(data)
    if arr.dtype == np.uint16:
        arr = arr.astype(np.float32) / 65535.0
    elif arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    elif arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=False)
    return arr, hdr


def _apply_aligned_to_doc(doc, aligned_img: np.ndarray,
                          aligned_hdr: "Header | None") -> None:
    """
    Overwrite a doc's pixels with the aligned result and nudge its
    metadata so WCS/filter stay in sync. Prefers doc.set_image() (which
    gives SASpro a hook for undo and change notification); falls back to
    direct attribute assignment the way _push_image_to_active_view does.
    """
    if hasattr(doc, "set_image"):
        try:
            # Pass a step_name so the history tracks what happened.
            doc.set_image(aligned_img, step_name="Star Registration")
        except TypeError:
            doc.set_image(aligned_img)
    else:
        doc.image = aligned_img

    # Merge header into doc.metadata['original_header'] when the doc holds
    # a dict-shaped metadata (SASpro convention). Don't invent one — a doc
    # that doesn't carry metadata stays as-is.
    md = getattr(doc, "metadata", None)
    if isinstance(md, dict) and aligned_hdr is not None:
        md["original_header"] = aligned_hdr
        md["step_name"] = "Star Registration"

    # Nudge any observers watching the doc.
    try:
        if hasattr(doc, "changed"):
            doc.changed.emit()
    except Exception:
        pass


def _open_aligned_as_new_doc(parent_window, dm,
                             aligned_img: np.ndarray,
                             aligned_hdr: "Header | None",
                             source_title: str) -> None:
    """
    Create a new SASpro document for the aligned result, using the same
    pattern the stellar-alignment headless path uses: dm.open_array +
    parent._spawn_subwindow_for.
    """
    title = f"{source_title} [Registered]"
    meta = {
        "step_name": "Star Registration",
        "description": f"Aligned from view: {source_title}",
        "is_mono": bool(aligned_img.ndim == 2 or
                        (aligned_img.ndim == 3 and aligned_img.shape[2] == 1)),
    }
    if aligned_hdr is not None:
        meta["original_header"] = aligned_hdr
    newdoc = dm.open_array(aligned_img, metadata=meta, title=title)
    if parent_window is not None and hasattr(parent_window, "_spawn_subwindow_for"):
        try:
            parent_window._spawn_subwindow_for(newdoc)
        except Exception:
            pass


def _write_view_to_fits(doc, img, out_path: str) -> None:
    """
    Write a view's in-memory image to a FITS file for the registration
    workers to read. Preserves the original header when the doc carries
    one (so WCS, filter, exposure etc. survive into the aligned output)
    and normalises the pixel layout to what astroalign/the solve path
    expects (mono: HxW; RGB: HxWxC, native astropy order).

    This runs on the GUI thread briefly — intentional: the thread/worker
    processes only consume paths, so the spill has to happen before the
    thread starts. For typical frame sizes the write is well under a
    second; for very large stacks it's a few seconds and the progress
    label already says "Status: Running…" by then.
    """
    arr = np.asarray(img)
    if arr.dtype == np.uint16:
        arr = arr.astype(np.float32) / 65535.0
    elif arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    elif arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=False)

    # Preserve whatever header the doc carries so downstream WCS lookups
    # work on the aligned output. Uses the same multi-key resolver the
    # WCS registration path uses — original_header → header → fits_header
    # → wcs_header (as a string) → attribute fallback. Converts non-Header
    # shapes to Header via _coerce_to_fits_header. Previously only
    # original_header was tried; views whose plate solve was stored as the
    # 'wcs_header' string lost their WCS at spill time, making the
    # downstream WCS registration refuse the target.
    hdr = None
    try:
        raw_hdr = _doc_header_or_wcs(doc)
        hdr = _coerce_to_fits_header(raw_hdr)
    except Exception:
        hdr = None

    hdu = fits.PrimaryHDU(data=arr, header=hdr)
    # overwrite=True so a stale spill from a crashed previous run can't
    # block this write.
    hdu.writeto(out_path, overwrite=True)


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

# Flip to True to log NaN carry-through at the alignment checkpoints (per frame).
NAN_PROBE = False

def _cap_points(src_pts: np.ndarray, tgt_pts: np.ndarray, max_cp: int) -> tuple[np.ndarray,np.ndarray]:
    if src_pts.shape[0] <= max_cp:
        return src_pts, tgt_pts
    idx = np.linspace(0, src_pts.shape[0]-1, max_cp, dtype=int)
    return src_pts[idx], tgt_pts[idx]

def _zero_ignore_mask(gray2d: np.ndarray, edge_trim: int = 8) -> np.ndarray | None:
    """SEP 'ignore' mask (True = skip) for the border-fill left by WCS
    reprojection: exact-zero pixels, dilated inward a few px to also swallow
    the Lanczos ringing fringe at the footprint edge. None if nothing to mask."""
    m = (gray2d == 0.0) | ~np.isfinite(gray2d)   # trail no-data may be 0.0 or NaN
    if not m.any():
        return None
    if edge_trim > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*edge_trim+1, 2*edge_trim+1))
        m = cv2.dilate(m.astype(np.uint8), k).astype(bool)   # grow the ignore region
    return m

def _masked(gray2d: np.ndarray, mask: np.ndarray | None):
    g = np.ascontiguousarray(gray2d.astype(np.float32))
    if mask is None:
        return g
    # full 2-D bool mask avoids the ma.nomask scalar that _mask() chokes on
    return np.ma.MaskedArray(g, mask=np.asarray(mask, bool), fill_value=0.0)

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
        run_row = QHBoxLayout()
        self.btn_run = QPushButton("Run Alignment")
        self.btn_run.clicked.connect(self.run_alignment)
        run_row.addWidget(self.btn_run, 1)
        run_row.addWidget(make_help_button("star_alignment", self))
        controls.addLayout(run_row)

        # Drag-to-canvas grip (PI-style "new instance"). Deferred import
        # avoids an import cycle (shortcuts.py imports things back from this
        # file). Lives on its own row at the bottom of the controls column
        # so it's unmistakably a drag handle, not a run button. Dropping on
        # empty canvas creates a shortcut carrying the dialog's current
        # settings; dropping on an image view runs the tool headlessly
        # against that view, same wiring every other tool uses.
        try:
            from setiastro.saspro.shortcuts import PresetDragHandle
            # Icon comes from the shared resources module via staralign_path
            # — same pattern ABE, Stat Stretch, etc. use (abeicon_path /
            # statstretchicon_path). Wrapped in try/except because an older
            # SASpro install that predates the staralign asset shouldn't
            # crash the dialog; the grip just falls back to its default
            # glyph when the icon can't be loaded.
            try:
                from setiastro.saspro.resources import staralign_path
                _grip_icon = QIcon(staralign_path)
                if _grip_icon.isNull():
                    _grip_icon = QIcon()
            except Exception:
                _grip_icon = QIcon()

            drag_row = QHBoxLayout()
            drag_row.setContentsMargins(0, 0, 0, 0)
            self.preset_drag_handle = PresetDragHandle(
                "star_align",
                self._collect_preset_for_grip,
                icon=_grip_icon,
                tooltip=self.tr(
                    "Drag to the canvas to create a Star Alignment shortcut\n"
                    "with these exact settings.\n"
                    "Drop directly on an image view to apply them headlessly."
                ),
                parent=self,
            )
            drag_row.addWidget(self.preset_drag_handle)
            drag_row.addStretch(1)
            controls.addLayout(drag_row)
        except Exception:
            # If shortcuts.py isn't importable in this host (e.g. test
            # harness), silently skip the grip — the rest of the dialog
            # still works.
            pass

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

    def _collect_preset_for_grip(self) -> dict:
        """
        Snapshot the dialog's current UI into the preset dict the headless
        runner (star_alignment_preset.run_star_alignment_via_preset) expects.
        Called by the PresetDragHandle at drag-start time, so the shortcut /
        headless apply uses whatever the user has set up THIS SESSION — not
        a stale saved preset.

        Reference-side mapping (Stellar Alignment is a two-image dialog but
        the preset model is single-reference → dropped-target):
          source_from_file_radio checked → ref_mode="file",  ref_file=<path>
          source_from_view_radio checked → ref_mode="view_name", ref_name=<combo title>
        When From-View is checked but the combo is empty, fall back to
        "active" so the headless resolver uses whatever view is active at
        drop time (safer than emitting an unresolvable view name).
        """
        # Reference mapping -------------------------------------------------
        ref_mode = "active"
        ref_name = ""
        ref_file = ""
        try:
            if getattr(self, "source_from_file_radio", None) is not None and \
                    self.source_from_file_radio.isChecked():
                ref_mode = "file"
                ref_file = str(getattr(self, "source_file_path", "") or "")
            elif getattr(self, "source_from_view_radio", None) is not None and \
                    self.source_from_view_radio.isChecked():
                title = ""
                try:
                    title = str(self.source_view_combo.currentText() or "").strip()
                except Exception:
                    title = ""
                if title:
                    ref_mode = "view_name"
                    ref_name = title
                else:
                    ref_mode = "active"
        except Exception:
            pass

        # Advanced mapping — read directly off the xf_* widgets since those
        # are the live source of truth, and the two combos (method, model)
        # are indexed the same way the headless runner's _IDX_TO_* arrays
        # are.
        try:
            method = ("stellar", "wcs", "wcs_stellar")[self.xf_method.currentIndex()]
        except Exception:
            method = "stellar"
        try:
            model = ("affine", "homography", "poly3", "poly4")[self.xf_model.currentIndex()]
        except Exception:
            model = "affine"

        try:
            det_sigma = float(self.xf_det_sigma.value())
        except Exception:
            det_sigma = 12.0
        try:
            max_cp = int(self.xf_maxcp.value())
        except Exception:
            max_cp = 250
        try:
            downsample = int(self.xf_downsample.value())
        except Exception:
            downsample = 2
        try:
            h_reproj = float(self.xf_h_reproj.value())
        except Exception:
            h_reproj = 3.0

        # min_area, limit_stars, min_fwhm, max_ellipticity aren't exposed as
        # dialog widgets (they live only in the stacking/align/* QSettings
        # path). Pull them from the live settings so a shortcut carries the
        # user's current defaults, not hard-coded 10/500/1.2/0.6 placeholders.
        try:
            prefs = _align_prefs(self.settings) if hasattr(self, "settings") else _align_prefs(QSettings())
        except Exception:
            prefs = {}
        min_area        = int(prefs.get("minarea",         10))
        limit_stars     = int(prefs.get("limit_stars",     500))
        min_fwhm        = float(prefs.get("min_fwhm",        1.2))
        max_ellipticity = float(prefs.get("max_ellipticity", 0.6))

        return {
            # Reference + output
            "ref_mode":   ref_mode,
            "ref_name":   ref_name,
            "ref_file":   ref_file,
            # Default a drag-born preset to Overwrite = False (safer — the
            # user can toggle it by right-clicking the shortcut and editing
            # the preset). Matches other tools' grip-born defaults.
            "overwrite":  False,
            "downsample": downsample,
            # Advanced — full set the preset dialog can also edit later.
            "method":              method,
            "model":               model,
            "det_sigma":           det_sigma,
            "min_area":            min_area,
            "max_control_points":  max_cp,
            "h_reproj":            h_reproj,
            "limit_stars":         limit_stars,
            "min_fwhm":            min_fwhm,
            "max_ellipticity":     max_ellipticity,
        }

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
        from setiastro.saspro import astroalign  # deferred: see tools/apply_import_fixes.py (astroalign)
        tgt32 = _masked(tgt_gray, _zero_ignore_mask(tgt_gray))
        src32 = _masked(src_gray, _zero_ignore_mask(src_gray))
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

                # Pass A: base warp to reference grid (anti-ring clamped).
                # Color case batches the per-channel call into one HWC warp
                # so the star mask is computed max-across-channels (see
                # _warp_antiring docstring for the OSC regression this fixes).
                if base_kind == "affine":
                    def _wa(ch, flags):
                        return cv2.warpAffine(ch, base_X, (Wout, Hout),
                                              flags=flags, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                    if img.ndim == 2:
                        if not img.flags['C_CONTIGUOUS']:
                            img = np.ascontiguousarray(img)
                        base_img = _warp_antiring(lambda: _wa(img, cv2.INTER_LANCZOS4),
                                                  lambda: _wa(img, cv2.INTER_LINEAR))
                    elif img.shape[2] <= 4:
                        _img = img if img.flags['C_CONTIGUOUS'] else np.ascontiguousarray(img)
                        base_img = _warp_antiring(lambda: _wa(_img, cv2.INTER_LANCZOS4),
                                                  lambda: _wa(_img, cv2.INTER_LINEAR))
                    else:
                        base_img = _warp_hwc_antiring(
                            img.shape[2],
                            lambda c: _wa(np.ascontiguousarray(img[..., c]), cv2.INTER_LANCZOS4),
                            lambda c: _wa(np.ascontiguousarray(img[..., c]), cv2.INTER_LINEAR),
                        )
                else:
                    def _wp(ch, flags):
                        return cv2.warpPerspective(ch, base_X, (Wout, Hout),
                                                   flags=flags, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                    if img.ndim == 2:
                        if not img.flags['C_CONTIGUOUS']:
                            img = np.ascontiguousarray(img)
                        base_img = _warp_antiring(lambda: _wp(img, cv2.INTER_LANCZOS4),
                                                  lambda: _wp(img, cv2.INTER_LINEAR))
                    elif img.shape[2] <= 4:
                        _img = img if img.flags['C_CONTIGUOUS'] else np.ascontiguousarray(img)
                        base_img = _warp_antiring(lambda: _wp(_img, cv2.INTER_LANCZOS4),
                                                  lambda: _wp(_img, cv2.INTER_LINEAR))
                    else:
                        base_img = _warp_hwc_antiring(
                            img.shape[2],
                            lambda c: _wp(np.ascontiguousarray(img[..., c]), cv2.INTER_LANCZOS4),
                            lambda c: _wp(np.ascontiguousarray(img[..., c]), cv2.INTER_LINEAR),
                        )

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

        pairs = None
        if wcs_prealigned is not None:
            _minarea = int(_align_prefs(self.settings).get("minarea", 10))
            pairs = _proximity_match_pairs(
                tgt_small, src_small,
                det_sigma=float(self.xf_det_sigma.value()),
                minarea=_minarea, tol_px=5.0)

        if pairs is not None:
            src_xy, tgt_xy = pairs
            self.status_label.setText(
                f"Matched {len(src_xy)} stars by proximity (WCS pre-aligned).")
        else:
            try:
                transform_obj, (src_pts_s, tgt_pts_s) = \
                    self.aa_find_transform_with_backoff(tgt_small, src_small)
            except Exception as e:
                if wcs_prealigned is not None:
                    return self._accept_wcs_fallback(wcs_prealigned, f"astroalign: {e}")
                QMessageBox.warning(self, "Alignment Error", f"Astroalign failed: {e}")
                return
            src_xy = np.asarray(src_pts_s, dtype=np.float32)
            tgt_xy = np.asarray(tgt_pts_s, dtype=np.float32)

            # Meridian-flip pre-rotate.
            # Astroalign finds correct ~180° correspondences for a flipped
            # pair, but its similarity LS fit then carries a fractional-
            # degree rotation residual pulled by sub-pixel centroid noise.
            # The downstream Lanczos warp of that near-180° rotation is
            # what gives the "almost right" ghosting on stacked, flipped
            # halves. Fix: if the solve reports a rotation in the 180°
            # window, physically rotate the target buffers with
            # np.rot90(..., 2) (bit-exact, no interpolation) and redo the
            # solve. The downstream warp then only resamples a sub-pixel
            # residual.
            try:
                _P = np.asarray(transform_obj.params, dtype=np.float64)
                _A = _P[:2, :3] if _P.shape == (3, 3) else np.asarray(_P, np.float64).reshape(2, 3)
            except Exception:
                _A = None
            # Both signals active: rotation window OR huge translation. The
            # second signal is what catches the case where the solver fell
            # back to a translation-dominated solution and the rotation
            # reading alone would miss it.
            if _A is not None and _is_near_180(_A, src_shape=tgt_gray.shape[:2]):
                self.status_label.setText(
                    "Meridian flip detected — pre-rotating target 180° and re-solving…")
                QApplication.processEvents()
                tgt = np.ascontiguousarray(tgt[::-1, ::-1]) if tgt.ndim == 2 \
                        else np.ascontiguousarray(tgt[::-1, ::-1, :])
                tgt_gray = np.ascontiguousarray(tgt_gray[::-1, ::-1])
                tgt_small = tgt_gray  # downsample stage currently off in dialog
                try:
                    transform_obj, (src_pts_s, tgt_pts_s) = \
                        self.aa_find_transform_with_backoff(tgt_small, src_small)
                except Exception as e:
                    QMessageBox.warning(
                        self, "Alignment Error",
                        f"Re-solve after meridian pre-rotation failed: {e}")
                    return
                src_xy = np.asarray(src_pts_s, dtype=np.float32)
                tgt_xy = np.asarray(tgt_pts_s, dtype=np.float32)

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

        # Apply the transform.
        # Color case: ONE batched call so _warp_antiring builds the mask from
        # max-across-channels (fixes the OSC narrowband dark-ring regression).
        if kind == "affine":
            def _wa(ch, flags):
                return cv2.warpAffine(ch, X, (W, H), flags=flags,
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            if tgt.ndim == 2:
                warped_target = _warp_antiring(lambda: _wa(tgt, cv2.INTER_LANCZOS4),
                                               lambda: _wa(tgt, cv2.INTER_LINEAR))
            elif tgt.shape[2] <= 4:
                _tgt = tgt if tgt.flags['C_CONTIGUOUS'] else np.ascontiguousarray(tgt)
                warped_target = _warp_antiring(lambda: _wa(_tgt, cv2.INTER_LANCZOS4),
                                               lambda: _wa(_tgt, cv2.INTER_LINEAR))
            else:
                warped_target = _warp_hwc_antiring(
                    tgt.shape[2],
                    lambda i: _wa(np.ascontiguousarray(tgt[..., i]), cv2.INTER_LANCZOS4),
                    lambda i: _wa(np.ascontiguousarray(tgt[..., i]), cv2.INTER_LINEAR),
                )
            transform_3x3 = np.eye(3, dtype=np.float32); transform_3x3[:2] = X
            _note = ("NOTE: WCS/SIP reprojection was applied first; this matrix is\n"
                     "only the residual star-refinement on top of that alignment."
                     if wcs_prealigned is not None else None)
            self.show_transform_info(transform_3x3, stage_note=_note)

        elif kind == "homography":
            def _wp(ch, flags):
                return cv2.warpPerspective(ch, X, (W, H), flags=flags,
                                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            if tgt.ndim == 2:
                warped_target = _warp_antiring(lambda: _wp(tgt, cv2.INTER_LANCZOS4),
                                               lambda: _wp(tgt, cv2.INTER_LINEAR))
            elif tgt.shape[2] <= 4:
                _tgt = tgt if tgt.flags['C_CONTIGUOUS'] else np.ascontiguousarray(tgt)
                warped_target = _warp_antiring(lambda: _wp(_tgt, cv2.INTER_LANCZOS4),
                                               lambda: _wp(_tgt, cv2.INTER_LINEAR))
            else:
                warped_target = _warp_hwc_antiring(
                    tgt.shape[2],
                    lambda i: _wp(np.ascontiguousarray(tgt[..., i]), cv2.INTER_LANCZOS4),
                    lambda i: _wp(np.ascontiguousarray(tgt[..., i]), cv2.INTER_LINEAR),
                )
            # Optional: show homography info as well
            try:
                _note = ("NOTE: WCS/SIP reprojection was applied first; this matrix is\n"
                         "only the residual star-refinement on top of that alignment."
                         if wcs_prealigned is not None else None)
                self.show_transform_info(np.array(X, dtype=np.float64, copy=False), stage_note=_note)
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

        if wcs_prealigned is not None:
            _resid_txt = "a residual correction"   # poly: no single scalar
            try:
                if kind == "affine":
                    A = np.asarray(X, np.float64).reshape(2, 3)
                    _resid_txt = f"a {float(np.hypot(A[0, 2], A[1, 2])):.1f}px residual correction"
                elif kind == "homography":
                    Hm = np.asarray(X, np.float64).reshape(3, 3)
                    w = Hm[2, 2] if abs(Hm[2, 2]) > 1e-12 else 1.0
                    _resid_txt = f"a {float(np.hypot(Hm[0, 2] / w, Hm[1, 2] / w)):.1f}px residual correction"
            except Exception:
                pass
            _label = f"WCS/SIP reprojection + {model} refinement"
            _detail = (f"WCS/SIP reprojection did the primary alignment "
                       f"(scale, rotation, orientation, field of view); the {model} "
                       f"star pass then applied {_resid_txt}.")
        else:
            _label = model
            _detail = f"Aligned using {model} star matching."

        self.status_label.setText(f"Alignment complete ({_label}).")
        QApplication.processEvents()
        QMessageBox.information(self, "Alignment Complete", _detail)



    def show_transform_info(self, matrix, *, stage_note: str | None = None):
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
        if stage_note:
            info_text += f"\n{stage_note}\n"

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

def _S(ds: float, full_wh=None, ds_wh=None) -> np.ndarray:
    """
    Full-res -> DS pixel coordinates (3x3), matching cv2.resize's pixel-centre
    mapping:  x_full = sx * (x_ds + 0.5) - 0.5,  sx = W / W_ds.

    full_wh: (W, H) of the full-res image. ds_wh: (W_ds, H_ds) it was resized
    to; defaults to (W // ds, H // ds). Without full_wh the scale is taken as
    exactly ds. The true scale and the half-pixel term matter: dropping them
    shifts frames rotated ~180° vs the reference (meridian flip) by a few px.
    """
    ds = float(ds)
    if full_wh is None:
        sx = sy = ds
    else:
        W, H = float(full_wh[0]), float(full_wh[1])
        if ds_wh is None:
            ds_wh = (max(1, int(W // ds)), max(1, int(H // ds)))
        sx, sy = W / float(ds_wh[0]), H / float(ds_wh[1])
    return np.array([[1.0/sx, 0, 0.5/sx - 0.5],
                     [0, 1.0/sy, 0.5/sy - 0.5],
                     [0, 0, 1]], np.float64)

# The lift/downscale helpers take the DS transform as mapping the source DS grid
# onto the reference DS grid. S_ref / S_src are _S() matrices for each image;
# S_src defaults to S_ref (e.g. a delta solved between two reference-grid images).

def lift_affine_2x3_from_ds(A_ds_2x3: np.ndarray, ds: float,
                            S_ref: np.ndarray | None = None,
                            S_src: np.ndarray | None = None) -> np.ndarray:
    S_ref = _S(ds) if S_ref is None else S_ref
    S_src = S_ref if S_src is None else S_src
    A3_full = np.linalg.inv(S_ref) @ _to3x3_affine(A_ds_2x3) @ S_src
    return _from3x3_affine(A3_full)

def downscale_affine_2x3_to_ds(A_full_2x3: np.ndarray, ds: float,
                               S_ref: np.ndarray | None = None,
                               S_src: np.ndarray | None = None) -> np.ndarray:
    S_ref = _S(ds) if S_ref is None else S_ref
    S_src = S_ref if S_src is None else S_src
    A3_ds = S_ref @ _to3x3_affine(A_full_2x3) @ np.linalg.inv(S_src)
    return _from3x3_affine(A3_ds)

def lift_homography_from_ds(H_ds: np.ndarray, ds: float,
                            S_ref: np.ndarray | None = None,
                            S_src: np.ndarray | None = None) -> np.ndarray:
    S_ref = _S(ds) if S_ref is None else S_ref
    S_src = S_ref if S_src is None else S_src
    return np.linalg.inv(S_ref) @ np.asarray(H_ds, np.float64) @ S_src


# ─────────────────────────────────────────────────────────────────────
# Meridian-flip aware solve
#
# A GEM meridian flip is a pure 180° rotation (det=+1, no reflection).
# Astroalign's invariants + parity-aware RANSAC find the correct
# correspondences, but the final similarity LS fit is biased by
# sub-pixel centroid noise: for θ≈180° the sin(θ) component is tiny
# and gets pulled by noise, giving a fractional-degree rotation
# residual. Lanczos resampling of a near-180° rotation then imprints
# a per-frame PSF variation across the frame (bigger at the edges
# than at the center), which stacks into the "almost right" ghosting
# we see on flipped halves.
#
# Fix: if the solve reports a rotation in a 180° window, redo the
# solve on np.rot90(src, 2) — a bit-exact point reflection — and
# compose T_final = T_residual @ T_180. Astroalign then only has to
# recover a small residual, and the finalize warp performs a single
# sub-pixel resample instead of a full 180° rotation on top of the
# refinement.
# ─────────────────────────────────────────────────────────────────────

# Rotation window (deg) that triggers meridian-flip pre-rotate. Widened
# from the original [150, 210] to [135, 225] because the OSC narrowband
# dark-ring regression showed real-world flips landing at ~140° / ~220°
# when the imaging train isn't perfectly orthogonal — those fell outside
# the old window, so the solver's near-180° result ran through the final
# Lanczos warp (which is where the ring artifact actually gets imprinted).
# A false trigger costs one extra solve pass; the composed transform is
# numerically identical to the direct solve, so correctness is unaffected.
_MERIDIAN_LOW_DEG  = 135.0
_MERIDIAN_HIGH_DEG = 225.0

# Quarter-turn window (deg) that triggers 90°/270° bit-exact pre-rotate.
# Rare in GEM imaging — mostly a cover for users who rotated the imaging
# train 90° between sessions, or alt-az rigs where field rotation chewed
# through a quarter turn across a long run. Same half-width as the
# meridian window (±45°) so the three quarter-turn buckets (90, 180, 270)
# cover the circle without overlap and without leaving gaps.
_QUARTER_TURN_HALF_WIDTH_DEG = 45.0

# Fraction of (W, H) that |tx|, |ty| must EACH exceed for the translation
# test to flag a flip. A pure 180° point reflection in an H×W canvas
#sasproduces tx ≈ (W - 1) and ty ≈ (H - 1), so magnitudes close to 1.0*W/H.
# A normal frame-to-frame drift is a few hundred px — well under 0.5
# fractions of the canvas even on short-focal rigs. 0.5 is wide enough
# to catch borderline cases where the solver absorbed part of the flip
# into translation and the rotation reading is weak, and narrow enough
# that a real wide drift (dithering, re-centering between runs) will
# not false-trigger.
_MERIDIAN_TRANSLATION_FRAC = 0.5


def _rot_angle_deg_2x3(T2x3: np.ndarray) -> float:
    """
    Dominant rotation angle (deg, signed) of a 2x3 affine. Robust to
    uniform scale; for a mild shear the atan2 of the (sin, cos)
    components still returns a value close to the true orientation,
    which is all we need for the 180° window test.
    """
    A = np.asarray(T2x3, np.float64).reshape(2, 3)
    return float(np.degrees(np.arctan2(A[1, 0], A[0, 0])))


def _translation_magnitudes(T2x3: np.ndarray) -> tuple[float, float]:
    """Return (|tx|, |ty|) from a 2x3 affine."""
    A = np.asarray(T2x3, np.float64).reshape(2, 3)
    return float(abs(A[0, 2])), float(abs(A[1, 2]))


def _is_near_180(T2x3: np.ndarray,
                 src_shape: tuple | None = None,
                 translation_frac: float = _MERIDIAN_TRANSLATION_FRAC) -> bool:
    """
    True iff T encodes a near-180° rotation OR a translation large enough
    that it can only be explained by one.

    Two orthogonal signals, OR-ed together:

    1. Rotation window. |atan2(a10, a00)| in [150°, 210°] catches the
       straightforward case: solver recovered the ~180° rotation cleanly.
       The LS fit's sub-pixel residual on sin(θ)/cos(θ) does not move the
       dominant angle out of this window — a near-180° result reads near
       180° in atan2 regardless of the fractional-degree bias.

    2. Translation-magnitude test (only when src_shape is provided).
       A 2x3 that maps source pixel x -> ref pixel y via y = A·x + t,
       applied to the image center (W/2, H/2) with a point reflection
       (a00 ≈ -1, a11 ≈ -1), gives y_center ≈ -x_center + t, so the
       translation vector must be roughly (W, H) in magnitude to bring
       the reflected image back onto the ref canvas. A normal frame-pair
       alignment has |t| of order a few hundred px even for aggressive
       dithering — nowhere near 0.5*W or 0.5*H on real sensors.

       The translation-magnitude test is the backup signal for the case
       the user described: when the solver gets confused on a flipped
       pair and returns something where the rotation reading is weak or
       absorbed into translation. If |tx| > frac*W AND |ty| > frac*H
       (both axes), it's a meridian flip and nothing else — no legitimate
       frame-pair alignment produces that.

    When src_shape is None, only the rotation window applies (back-
    compatible with earlier call sites and with cases where we don't
    know the canvas size for ratio-ing).
    """
    if T2x3 is None:
        return False
    try:
        # Signal 1: rotation window
        if _MERIDIAN_LOW_DEG <= abs(_rot_angle_deg_2x3(T2x3)) <= _MERIDIAN_HIGH_DEG:
            return True
        # Signal 2: translation that can only be explained by a point
        # reflection in the canvas. Both axes must be well above the
        # threshold — a wide one-axis drift (e.g. a mosaic step) will
        # fail the AND and not false-trigger.
        if src_shape is not None:
            H = int(src_shape[0]); W = int(src_shape[1])
            tx, ty = _translation_magnitudes(T2x3)
            if tx > translation_frac * W and ty > translation_frac * H:
                return True
        return False
    except Exception:
        return False


def _rot180_affine_2x3(H: int, W: int) -> np.ndarray:
    """
    2x3 that reproduces np.rot90(img, 2) exactly for an H×W image:
    new_x = (W-1) - x, new_y = (H-1) - y.

    This is a bit-exact point reflection — no interpolation, no bias.
    Composing a solver's near-180° result with this inverse means the
    downstream Lanczos warp performs a single sub-pixel resample
    instead of a near-180° rotation that would carry per-frame PSF
    drift across the stack.
    """
    return np.array([[-1.0, 0.0, float(W - 1)],
                     [0.0, -1.0, float(H - 1)]], dtype=np.float64)


def _compose_affine_2x3(A2x3: np.ndarray, B2x3: np.ndarray) -> np.ndarray:
    """Return A ∘ B as 2x3 (apply B first, then A)."""
    A3 = np.vstack([np.asarray(A2x3, np.float64).reshape(2, 3), [0, 0, 1]])
    B3 = np.vstack([np.asarray(B2x3, np.float64).reshape(2, 3), [0, 0, 1]])
    return (A3 @ B3)[:2, :]


def _rotk_affine_2x3(k: int, H: int, W: int) -> np.ndarray:
    """
    2x3 that reproduces np.rot90(img, k) exactly for an H×W image.

    k is taken mod 4:
      0 : identity        (new_x = x,         new_y = y)
      1 : 90° CCW         (new_x = y,         new_y = (W-1) - x)   shape -> (W, H)
      2 : 180°            (new_x = (W-1) - x, new_y = (H-1) - y)
      3 : 270° CCW / 90° CW
                          (new_x = (H-1) - y, new_y = x)           shape -> (W, H)

    Bit-exact point reflections (no interpolation, no bias). Composing a
    solver's near-quarter-turn result with the inverse of the matching
    rot-k means the downstream Lanczos warp performs a single sub-pixel
    resample instead of a near-quarter-turn rotation that would carry
    per-frame PSF drift across the stack — same reasoning as _rot180_affine_2x3,
    extended to the full quarter-turn family.
    """
    k = int(k) & 3
    if k == 0:
        return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    if k == 1:
        return np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, float(W - 1)]], dtype=np.float64)
    if k == 2:
        return np.array([[-1.0, 0.0, float(W - 1)], [0.0, -1.0, float(H - 1)]], dtype=np.float64)
    # k == 3
    return np.array([[0.0, -1.0, float(H - 1)], [1.0, 0.0, 0.0]], dtype=np.float64)


def _nearest_quarter_turn(T2x3: np.ndarray,
                          src_shape: tuple | None = None,
                          half_width_deg: float = _QUARTER_TURN_HALF_WIDTH_DEG,
                          translation_frac: float = _MERIDIAN_TRANSLATION_FRAC) -> int:
    """
    Return the quarter-turn count k ∈ {0, 1, 2, 3} that best matches the
    rotation component of T, or 0 if T is close to identity (no pre-rotate
    needed). Only non-zero values trigger the bit-exact pre-rotate path.

    Rotation signal: wrap the signed atan2 angle to [-180, 180) and bucket
    it by nearest multiple of 90°. k is returned only when the angle is
    within ±half_width_deg of its bucket centre; otherwise 0.

    Translation backup (180° only, when src_shape is given): a solve that
    absorbed the flip into translation reads near-zero rotation but has
    |tx|≈W AND |ty|≈H. Covers the same case the previous _is_near_180
    guarded against so this function is a drop-in replacement.
    """
    if T2x3 is None:
        return 0
    try:
        ang = _rot_angle_deg_2x3(T2x3)
        # Nearest multiple of 90°, folded onto the circle.
        k = int(round(ang / 90.0)) & 3    # 0, 1, 2, or 3
        if k != 0 and abs(ang - 90.0 * (k if k <= 2 else k - 4)) <= half_width_deg:
            return k
        # Also check the k==2 (180°) bucket via the explicit meridian window
        # — tolerant to a 1-degree slop between round() and the half-width check.
        if _MERIDIAN_LOW_DEG <= abs(ang) <= _MERIDIAN_HIGH_DEG:
            return 2
        # Translation backup (180° only — the k=1/3 point reflections produce
        # swapped-axis translations that are hard to disambiguate from mosaic
        # steps without matching the solver's recovered shape).
        if src_shape is not None:
            H = int(src_shape[0]); W = int(src_shape[1])
            tx, ty = _translation_magnitudes(T2x3)
            if tx > translation_frac * W and ty > translation_frac * H:
                return 2
        return 0
    except Exception:
        return 0


def _solve_with_meridian_check(src_gray: np.ndarray,
                               ref_gray: np.ndarray,
                               solver,
                               *,
                               log=None) -> np.ndarray | None:
    """
    Run `solver(src, ref) -> 2x3 or None`. If the result is a near-
    quarter-turn rotation (90°, 180°, or 270°), redo the solve on the
    matching bit-exact np.rot90(src, k) and fold the pre-rotation back
    in so the returned 2x3 still maps ORIGINAL src coords -> ref coords.

    Rationale: the triangle+vote stage recovers the correct quarter-turn
    correspondences, but a least-squares similarity fit biased by sub-
    pixel centroid noise leaves a fractional-degree rotation residual.
    Lanczos resampling of a near-quarter-turn rotation then imprints a
    position-dependent PSF variation per frame that stacks into visible
    ringing on bright stars (the OSC dark-ring regression we chased
    through 1.22.6). Pre-rotating the source with a bit-exact np.rot90
    collapses the final warp to a sub-pixel residual correction, which
    the resampler handles cleanly and uniformly across the stack.

    Covers 90° / 270° in addition to the original 180° meridian flip:
    rare for GEM runs but exactly the case a user who rotated the
    imaging train between sessions, or an alt-az rig with heavy field
    rotation, falls into. For those angles a Lanczos warp rings just as
    badly as it does near 180°; the bit-exact rot90 fixes all three.

    Returns a 2x3 (same contract as `solver`) or None if both
    attempts fail.
    """
    Hs, Ws = src_gray.shape[:2]
    T = solver(src_gray, ref_gray)
    # Pass the SOURCE canvas size so the translation-magnitude backup
    # signal is active for the 180° case. The translation test catches
    # the case where the solver absorbed the flip into translation and
    # the rotation reading alone is weak — for example, a huge |tx|≈W,
    # |ty|≈H transform with a00, a11 that read closer to 0° than 180°
    # because of RANSAC bias.
    k = _nearest_quarter_turn(T, src_shape=(Hs, Ws)) if T is not None else 0
    if T is None or k == 0:
        return T

    # Bit-exact rotation: no kernel, no interpolation, no bias. For k=2
    # (meridian flip) this is the familiar [::-1, ::-1]; for k=1/3 the
    # output shape swaps H and W, which numpy and the solver both handle.
    src_rot = np.ascontiguousarray(np.rot90(src_gray, k))
    T_resid = solver(src_rot, ref_gray)
    if T_resid is None:
        # Pre-rotate path couldn't find a solve at all — keep the
        # original. We don't want to turn a solved frame into a
        # dropped one just because the fallback path failed.
        if log:
            try:
                _tag = {1: "quarter-turn 90°", 2: "meridian-flip",
                        3: "quarter-turn 270°"}.get(k, f"quarter-turn k={k}")
                log(f"[{_tag}] detected, but pre-rotated solve "
                    "failed; keeping original transform")
            except Exception:
                pass
        return T

    T_rotk = _rotk_affine_2x3(k, Hs, Ws)
    T_final = _compose_affine_2x3(T_resid, T_rotk)

    if log:
        try:
            orig_ang  = _rot_angle_deg_2x3(T)
            orig_tx, orig_ty = _translation_magnitudes(T)
            resid_ang = _rot_angle_deg_2x3(T_resid)
            _tag = {1: "quarter-turn 90°", 2: "meridian-flip",
                    3: "quarter-turn 270°"}.get(k, f"quarter-turn k={k}")
            log(f"[{_tag}] detected (rot={orig_ang:+.2f}°, "
                f"|t|=({orig_tx:.0f}, {orig_ty:.0f}) px on {Ws}x{Hs} canvas); "
                f"residual after pre-rot = {resid_ang:+.3f}°, "
                f"translation = ({T_resid[0,2]:+.2f}, {T_resid[1,2]:+.2f}) px")
        except Exception:
            pass
    return T_final


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
         ref_ds_npy, Wref_ds, Href_ds,
         resample_flag, det_sigma, limit_stars, minarea,
         model, h_reproj, ds,
         min_fwhm, max_ellipticity, ref_full_wh) = args
    ref_full_wh is the full-res reference (W, H); optional (older 14-tuples).
    """
    try:
        import os
        import numpy as np
        import cv2
        from astropy.io import fits

        (orig_path, current_transform_2x3,
         ref_ds_npy, Wref_ds, Href_ds,
         resample_flag, det_sigma, limit_stars, minarea,
         model, h_reproj, ds,
         min_fwhm, max_ellipticity) = args[:14]
        ref_full_wh = tuple(args[14]) if len(args) > 14 and args[14] is not None else None

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
        if ds > 1:
            S_src = _S(ds, (gray.shape[1], gray.shape[0]), (gray_ds.shape[1], gray_ds.shape[0]))
            S_ref = _S(ds, ref_full_wh, (int(Wref_ds), int(Href_ds))) if ref_full_wh else _S(ds)
        else:
            S_src = S_ref = np.eye(3)
        T_prev_ds = downscale_affine_2x3_to_ds(T_prev_full, ds, S_ref, S_src).astype(np.float32)

        # Warp DS source into DS ref geometry
        src_for_match_ds = cv2.warpAffine(
            gray_ds, T_prev_ds, (int(Wref_ds), int(Href_ds)),
            flags=resample_flag, borderMode=cv2.BORDER_REFLECT_101
        )

        # 4) denoise sparse islands in DS space (cheaper)
        # fast hot pixel suppression — 3x3 median replaces only extreme spikes,
        # no SEP background estimation needed
        #src_for_match_ds = _suppress_hotpx_fast(src_for_match_ds)
        # memmap the DS reference (persisted once in run()); the OS page-cache
        # is shared across workers and only the crop is materialized downstream.
        ref_for_match_ds = np.load(ref_ds_npy, mmap_mode="r")

        # 5) AA delta solve in DS space (with meridian-flip pre-rotate).
        #    A GEM meridian flip stacks the "near-180° residual bias" on top
        #    of whatever this refinement pass would otherwise contribute, so
        #    we wrap the solve call: if the solver reports a rotation in the
        #    180° window, we redo the solve on np.rot90(src, 2) and compose
        #    the result with a bit-exact 180° matrix. The returned 2x3 still
        #    maps original src -> ref, so the lift-to-full-res code below
        #    needs no change. See _solve_with_meridian_check.
        m = (model or "affine").lower()
        if m in ("no_distortion", "nodistortion"):
            m = "similarity"

        _limit = int(limit_stars) if limit_stars is not None else None

        if m == "similarity":
            def _solver(_src, _ref):
                return compute_similarity_transform_astroalign_cropped(
                    _src, _ref,
                    limit_stars=_limit,
                    det_sigma=float(det_sigma),
                    minarea=int(minarea),
                    h_reproj=float(h_reproj),
                    min_fwhm=float(min_fwhm),
                    max_ellipticity=float(max_ellipticity),
                )
        else:
            def _solver(_src, _ref):
                return compute_affine_transform_astroalign_cropped(
                    _src, _ref,
                    limit_stars=_limit,
                    det_sigma=float(det_sigma),
                    minarea=int(minarea),
                    min_fwhm=float(min_fwhm),
                    max_ellipticity=float(max_ellipticity),
                )

        # Silent meridian-flip handling. _solve_with_meridian_check performs
        # the solve, inspects the rotation, and if it's near-180° redoes it
        # on a bit-exact np.rot90 of the DS source then returns the composed
        # transform. No logging callback — the orchestrator's existing per-
        # frame progress line (dx/dy of the delta) is what the stacking UI
        # displays; the flip is baked into the returned 2x3 and lifted to
        # full-res below, so the stored transform IS the complete
        # source->ref mapping. Drizzle and the .sasd writer read this
        # composed matrix downstream, never the pre-flip residual.
        tform_ds = _solve_with_meridian_check(
            src_for_match_ds, ref_for_match_ds, _solver,
        )

        if tform_ds is None:
            return (orig_path, None,
                    f"Astroalign failed for {os.path.basename(orig_path)} – skipping (no transform returned)")

        # 6) lift DS delta back to full-res coords. S_ref uses the true
        # resize scale and pixel-centre offset, so a composed near-180°
        # (meridian-flip) transform lifts exactly; a pure ds scale would
        # leave it a few px off.
        T_new_full = lift_affine_2x3_from_ds(np.asarray(tform_ds, np.float64).reshape(2, 3), ds, S_ref)

        return (orig_path, np.asarray(T_new_full, np.float64).reshape(2, 3), None)

    except Exception as e:
        try:
            base = os.path.basename(args[0]) if args else "<unknown>"
        except Exception:
            base = "<unknown>"
        return (args[0] if args else "<unknown>", None,
                f"Astroalign failed for {base}: {e}")



# ─────────────────────────────────────────────────────────────
# Final warp+save worker (process-safe)
# ─────────────────────────────────────────────────────────────
def _warp_antiring(warp_hi, warp_lo, star_sigma: float = 3.0,
                   grow_px: int = 6, feather_px: float = 2.0,
                   envelope: bool = True,
                   envelope_pad_px: int = 1,
                   ref_mask: np.ndarray | None = None):
    """
    Selective anti-ring resampling for HDR point sources (saturated stars).

    Three layers of defence against Lanczos ringing, applied in this order:

      1. ENVELOPE CLAMP (always on by default). For every pixel, compute the
         per-channel min/max of the bilinear warp over a (2*envelope_pad_px+1)
         neighbourhood and clip the Lanczos result into that envelope. This
         kills every undershoot/overshoot the kernel's negative lobes can
         produce, EVERYWHERE in the frame — not just near flagged stars.
         Cheap: two cv2 min/max filters on an array we already have.

      2. STAR-DISK BLEND (on when stars are present). Lanczos everywhere
         EXCEPT a grown, feathered disk around bright stars, where the
         ring-free bilinear warp is blended in. Catches the residual
         rosette/"petal" facets the envelope clamp can leave around a
         bright PSF when the Lanczos lobes approach the envelope wall.

      3. FINAL ZERO FLOOR. Any pixel still below zero (very rare after
         envelope + blend) is clamped to zero so stacking can't amplify
         a negative artefact.

    PER-CHANNEL CALL BUG (the one that caused the dark-ring regression on
    OSC narrowband data): when `_warp_antiring` is called inside a per-
    channel loop, the `lo` array is 2D and the star mask is computed from
    only that single channel's intensity. A star bright in Hα (R) but dim
    in G/B would mask correctly in R and MISS in G/B, letting Lanczos
    rings through in two of three channels — exactly the greenish dark
    halos the user saw. Two fixes available:
      a) Call the warp with the full HWC image (cv2.warpAffine/
         warpPerspective handle up to 4 channels natively), so `lo.ndim==3`
         and `lo.max(axis=2)` runs. Preferred — single-call per frame.
      b) When a per-channel loop is unavoidable (>4 channels or a
         resampler that doesn't handle multi-channel), pass a cross-
         channel `ref_mask` computed once up front. See
         `_warp_hwc_antiring` below for the batched helper.

    star_sigma      : robust threshold in sigmas above background (median + k·MAD)
    grow_px         : dilate the star mask this many px to cover the ring zone
    feather_px      : Gaussian feather so the Lanczos↔bilinear seam is invisible
    envelope        : enable the per-pixel bilinear min/max clamp (recommended on)
    envelope_pad_px : neighbourhood radius for the envelope filter (1 = 3x3)
    ref_mask        : if given, use this 2D float32 mask [0..1] instead of
                      computing one from `lo`. Lets per-channel callers inject
                      a mask built from the max-across-channels luminance.
    """
    import cv2
    import numpy as np

    hi = warp_hi()   # Lanczos (or configured hi-sharpness kernel): sharp, rings
    lo = warp_lo()   # bilinear: soft, no ring

    # ─── 1. ENVELOPE CLAMP ────────────────────────────────────────────────
    # Clip the Lanczos result to the local min/max of the bilinear warp.
    # cv2.erode = per-pixel minimum, cv2.dilate = per-pixel maximum over the
    # structuring element. Works on 2D or HWC (cv2 broadcasts per channel).
    if envelope and envelope_pad_px > 0:
        try:
            ksz = int(2 * envelope_pad_px + 1)
            kern = cv2.getStructuringElement(cv2.MORPH_RECT, (ksz, ksz))
            lo_min = cv2.erode(lo,  kern)
            lo_max = cv2.dilate(lo, kern)
            # np.clip + explicit float32 preserves dtype on HWC inputs.
            hi = np.minimum(np.maximum(hi, lo_min), lo_max).astype(np.float32,
                                                                   copy=False)
        except Exception:
            # Envelope clamp is best-effort — on failure fall through to the
            # blend-only path (never worse than the pre-envelope behaviour).
            pass

    # ─── 2. STAR-DISK BLEND ───────────────────────────────────────────────
    if ref_mask is not None:
        # Caller handed us the mask (per-channel loop computed it against
        # the full-channel luminance). Just reuse it.
        ref = None
        mask = np.ascontiguousarray(ref_mask, dtype=np.float32)
    else:
        # Compute the mask from `lo`. For HWC this is max-over-channels so a
        # star bright in only one channel still masks correctly in all three.
        ref = lo if lo.ndim == 2 else lo.max(axis=2)
        finite = np.isfinite(ref)
        if finite.any():
            vals = ref[finite]
            med = float(np.median(vals))
            mad = float(np.median(np.abs(vals - med)))
        else:
            med = mad = 0.0
        thr = med + float(star_sigma) * 1.4826 * mad   # 1.4826 = MAD→sigma
        mask = (ref > thr).astype(np.float32)

    if mask.any():
        if ref_mask is None:
            # Only grow+feather an INTERNALLY-computed mask. A caller-supplied
            # mask is assumed to be already grown/feathered the way they want.
            if grow_px > 0:
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                              (2 * grow_px + 1, 2 * grow_px + 1))
                mask = cv2.dilate(mask, k)
            if feather_px and feather_px > 0:
                ksz = int(2 * round(feather_px) + 1)
                mask = cv2.GaussianBlur(mask, (ksz, ksz), float(feather_px))
        if lo.ndim == 3 and mask.ndim == 2:
            mask = mask[..., None]
        out = hi * (1.0 - mask) + lo * mask
    else:
        out = hi

    # ─── 3. FINAL ZERO FLOOR ──────────────────────────────────────────────
    out = np.asarray(out, np.float32)
    np.maximum(out, 0.0, out=out)
    return out


def _build_antiring_mask(ref2d: np.ndarray,
                         star_sigma: float = 3.0,
                         grow_px: int = 6,
                         feather_px: float = 2.0) -> np.ndarray:
    """
    Build the grown + feathered star mask from a 2D luminance reference.

    Factored out of `_warp_antiring` so per-channel callers can compute ONE
    mask against the max-across-channels luminance, then feed it to each
    per-channel `_warp_antiring` call via the `ref_mask` parameter. That
    closes the OSC narrowband hole where a star bright only in Hα would be
    masked in R but miss in G/B, letting Lanczos rings through in two of
    three channels (the dark-ring symptom).
    """
    import cv2
    import numpy as np

    finite = np.isfinite(ref2d)
    if finite.any():
        vals = ref2d[finite]
        med = float(np.median(vals))
        mad = float(np.median(np.abs(vals - med)))
    else:
        med = mad = 0.0
    thr = med + float(star_sigma) * 1.4826 * mad
    mask = (ref2d > thr).astype(np.float32)
    if not mask.any():
        return mask
    if grow_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (2 * grow_px + 1, 2 * grow_px + 1))
        mask = cv2.dilate(mask, k)
    if feather_px and feather_px > 0:
        ksz = int(2 * round(feather_px) + 1)
        mask = cv2.GaussianBlur(mask, (ksz, ksz), float(feather_px))
    return mask


def _warp_hwc_antiring(nch: int,
                      warp_hi_ch,     # callable(c) -> 2D Lanczos warp of channel c
                      warp_lo_ch,     # callable(c) -> 2D bilinear warp of channel c
                      *,
                      star_sigma: float = 3.0,
                      grow_px: int = 6,
                      feather_px: float = 2.0,
                      envelope: bool = True,
                      envelope_pad_px: int = 1) -> np.ndarray:
    """
    Per-channel anti-ring warp with a CROSS-CHANNEL star mask.

    Use this when the per-channel loop is unavoidable (>4 channels, or a
    custom resampler that cannot be fed HWC in one shot). It:
      1. Runs the bilinear warp for every channel first.
      2. Builds the star mask ONCE from the max-across-channels luminance.
      3. Runs the Lanczos warp per channel and blends it with the bilinear
         result through the shared mask + envelope clamp.

    For <=4 channel data the simpler path is to call `_warp_antiring`
    directly with the full HWC image — cv2.warpAffine/warpPerspective
    handle up to 4 channels natively and `_warp_antiring` already computes
    the max-across-channels mask internally. This helper exists for the
    cases where the single-call path isn't available.
    """
    import numpy as np

    # Pass 1: bilinear warps for every channel (needed for the mask AND the
    # final blend). These are cheap compared to Lanczos.
    lo_planes = [np.ascontiguousarray(warp_lo_ch(c)) for c in range(nch)]
    lo_hwc = np.stack(lo_planes, axis=2)
    # Max-across-channels luminance for the mask. A star bright in one
    # channel (narrowband Hα on an OSC) shows up here and masks correctly
    # in ALL channels in the per-channel blend below.
    ref2d = lo_hwc.max(axis=2)
    mask = _build_antiring_mask(ref2d, star_sigma=star_sigma,
                                grow_px=grow_px, feather_px=feather_px)

    # Pass 2: Lanczos warps + per-channel blend using the shared mask.
    out_planes = []
    for c in range(nch):
        hi = warp_hi_ch(c)
        lo = lo_planes[c]
        out = _warp_antiring(
            lambda hi=hi: hi,
            lambda lo=lo: lo,
            star_sigma=star_sigma,
            grow_px=grow_px,
            feather_px=feather_px,
            envelope=envelope,
            envelope_pad_px=envelope_pad_px,
            ref_mask=mask,
        )
        out_planes.append(out)
    return np.stack(out_planes, axis=2)

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
        _ft = int(os.environ.get("SASPRO_FINALIZE_CV2_THREADS", "1"))
        cv2.setNumThreads(max(1, _ft))   # fill cores on the warp; parent bounds this to cores/workers
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
        # Keep satellite-trail NaN (no-data) in the frame we warp + save; scrub
        # only +/-inf. Detection uses the filled src_gray_full above, so star
        # matching is unaffected. NaN blooms through the Lanczos warp (covering
        # the ring halo) and is dropped by isfinite() at integration.
        img = np.nan_to_num(img, nan=np.nan, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        img = np.ascontiguousarray(img)

        # Satellite sidecar masks retired: trail pixels are baked to 0.0 at
        # calibration and treated as no-data through integration, so there is no
        # sidecar to load, warp, or save here.
        satmask_src = None
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

            if ds > 1:
                S_ref = _S(ds, (Wref, Href), (Wds, Hds))
                S_src = _S(ds, (src_gray_full.shape[1], src_gray_full.shape[0]), (Wds, Hds))
            else:
                S_ref = S_src = np.eye(3)
            A_prev_ds = downscale_affine_2x3_to_ds(A_prev, ds, S_ref, S_src).astype(np.float32)
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

            # Meridian-flip guard. By the time finalize runs, A_prev
            # usually already carries any 180° composition that pass 0
            # applied (via _solve_with_meridian_check in _solve_delta_job),
            # so src_pre_ds is near-identity vs ref_ds. The guard still
            # matters when refinement is disabled (max_refinement_passes=0)
            # or if pass 0's meridian catch missed, which would leave
            # the full 180° in src_pre_ds. We do one correspondence pass,
            # inspect the implied rotation, and if it falls in the 180°
            # window we pre-rotate src_pre_ds in-place with np.rot90 and
            # retry — bit-exactly, no interpolation. The downstream RANSAC
            # fit will then see a near-identity residual, which the Lanczos
            # pass can resample without the position-dependent PSF drift
            # that makes 180° warps stack poorly.
            src_xy, tgt_xy, best_P, best_xy0 = _aa_find_pairs_multitile(
                src_pre_ds, ref_ds,
                scale=AA_SCALE,
                tiles=1,
                det_sigma=float(det_sigma),
                minarea=int(minarea),
                max_control_points=max_cp,
                _dbg=dbg
            )

            # If the initial correspondence set implies a ~180° rotation
            # OR a translation too big to be anything else, pre-rotate the
            # pre-warped source and solve again. Then rewrite A_prev /
            # A_prev3 so the composition logic below applies the 180°
            # exactly once, as part of the delta. src_pre_ds is in ref DS
            # canvas coords, so its shape is what the translation test
            # ratios against.
            _flip_applied = False
            if (src_xy is not None and len(src_xy) >= 8
                    and best_P is not None
                    and _is_near_180(np.asarray(best_P)[:2, :],
                                     src_shape=src_pre_ds.shape[:2])):
                dbg("[finalize] near-180° detected in pre-warped src -> ref; "
                    "applying bit-exact np.rot90 and re-solving")
                src_pre_ds = np.ascontiguousarray(src_pre_ds[::-1, ::-1])
                # Compose A_prev with the full-res 180° so the finalize path
                # downstream (which composes A_prev with the recovered delta)
                # bakes the point reflection into the stored transform.
                Hsrc_full, Wsrc_full = src_gray_full.shape[:2]
                T_180_full = _rot180_affine_2x3(Hsrc_full, Wsrc_full)
                A_prev = _compose_affine_2x3(A_prev, T_180_full)
                A_prev3 = _A3(A_prev)
                src_xy, tgt_xy, best_P, best_xy0 = _aa_find_pairs_multitile(
                    src_pre_ds, ref_ds,
                    scale=AA_SCALE,
                    tiles=1,
                    det_sigma=float(det_sigma),
                    minarea=int(minarea),
                    max_control_points=max_cp,
                    _dbg=dbg
                )
                _flip_applied = True
                dbg("[finalize] re-solve after np.rot90 complete")

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
                    H_delta_full = lift_homography_from_ds(H_delta_ds, ds, S_ref)
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
                    A_delta_full = lift_affine_2x3_from_ds(A_delta_ds, ds, S_ref)
                    A_final3 = _A3(A_delta_full) @ A_prev3
                    A_final = A_final3[:2, :]
                    kind, X = "similarity", _project_to_similarity(A_final)

            elif model in ("poly3", "poly4"):
                order = 3 if model == "poly3" else 4
                # DS -> full-res with the same pixel-centre mapping as the lifts
                Si = np.linalg.inv(S_ref)
                src_full = (np.asarray(src_xy, np.float64).reshape(-1, 2) * np.diag(Si)[:2] + Si[:2, 2]).astype(np.float32)
                tgt_full = (np.asarray(tgt_xy, np.float64).reshape(-1, 2) * np.diag(Si)[:2] + Si[:2, 2]).astype(np.float32)

                cx, cy = _fit_poly_xy(src_full, tgt_full, order=order)
                map_x, map_y = _poly_eval_grid(cx, cy, Wref, Href, order=order)
                kind, X = model, (map_x, map_y)

            else:
                kind, X = "affine", A_prev.copy()

        # 4) warp full-res image  (anti-ring: clamp Lanczos to bilinear envelope)
        Hh, Ww = Href, Wref
        # Hi-sharpness kernel is resolved from SASPRO_WARP_KERNEL at import —
        # defaults to Lanczos4, user can set the env var to "cubic" to swap
        # in bicubic (Mitchell-family, much shorter negative lobes) when
        # faint-residual-ring artifacts persist past the anti-ring guard.
        _LANCZOS = _warp_hi_interp()
        _LINEAR  = cv2.INTER_LINEAR
        _BMODE, _BVAL = cv2.BORDER_CONSTANT, 0

        # ─── OSC anti-ring note ─────────────────────────────────────────
        # The <=4-channel color branch deliberately calls `_warp_antiring`
        # ONCE with the full HWC image rather than looping per channel. The
        # per-channel loop (removed) built the star mask from only that
        # single channel's intensity: a narrowband star bright in Hα (R)
        # but dim in G/B would mask correctly in R and MISS in G/B, letting
        # Lanczos undershoot through in two of three channels — the
        # greenish dark halos seen in the 1.22.6 regression. One batched
        # call lets `_warp_antiring` compute the mask against max-across-
        # channels luminance so a star bright in ANY channel masks all of
        # them. cv2.warpAffine/warpPerspective/remap handle up to 4
        # channels natively. For >4 channels we fall back to
        # `_warp_hwc_antiring`, which still shares one cross-channel mask.
        if kind in ("affine", "similarity"):
            A = np.asarray(X, np.float64).reshape(2, 3)
            def _wa(ch, flags):
                return cv2.warpAffine(ch, A, (Ww, Hh), flags=flags,
                                      borderMode=_BMODE, borderValue=_BVAL)
            if is_mono:
                aligned = _warp_antiring(lambda: _wa(img, _LANCZOS),
                                         lambda: _wa(img, _LINEAR))
            elif img.shape[2] <= 4:
                _img = img if img.flags['C_CONTIGUOUS'] else np.ascontiguousarray(img)
                aligned = _warp_antiring(lambda: _wa(_img, _LANCZOS),
                                         lambda: _wa(_img, _LINEAR))
            else:
                aligned = _warp_hwc_antiring(
                    img.shape[2],
                    lambda c: _wa(np.ascontiguousarray(img[..., c]), _LANCZOS),
                    lambda c: _wa(np.ascontiguousarray(img[..., c]), _LINEAR),
                )

            drizzle_tuple = ("affine", A.astype(np.float64))
            warp_label = ("similarity" if kind == "similarity" else "affine")

        elif kind == "homography":
            Hm = np.asarray(X, np.float64).reshape(3, 3)
            def _wp(ch, flags):
                return cv2.warpPerspective(ch, Hm, (Ww, Hh), flags=flags,
                                           borderMode=_BMODE, borderValue=_BVAL)
            if is_mono:
                aligned = _warp_antiring(lambda: _wp(img, _LANCZOS),
                                         lambda: _wp(img, _LINEAR))
            elif img.shape[2] <= 4:
                _img = img if img.flags['C_CONTIGUOUS'] else np.ascontiguousarray(img)
                aligned = _warp_antiring(lambda: _wp(_img, _LANCZOS),
                                         lambda: _wp(_img, _LINEAR))
            else:
                aligned = _warp_hwc_antiring(
                    img.shape[2],
                    lambda c: _wp(np.ascontiguousarray(img[..., c]), _LANCZOS),
                    lambda c: _wp(np.ascontiguousarray(img[..., c]), _LINEAR),
                )

            drizzle_tuple = ("homography", Hm.astype(np.float64))
            warp_label = "homography"

        elif kind in ("poly3", "poly4"):
            map_x, map_y = X
            def _wr(ch, flags):
                return cv2.remap(ch, map_x, map_y, flags,
                                 borderMode=_BMODE, borderValue=_BVAL)
            if is_mono:
                aligned = _warp_antiring(lambda: _wr(img, _LANCZOS),
                                         lambda: _wr(img, _LINEAR))
            elif img.shape[2] <= 4:
                _img = img if img.flags['C_CONTIGUOUS'] else np.ascontiguousarray(img)
                aligned = _warp_antiring(lambda: _wr(_img, _LANCZOS),
                                         lambda: _wr(_img, _LINEAR))
            else:
                aligned = _warp_hwc_antiring(
                    img.shape[2],
                    lambda c: _wr(np.ascontiguousarray(img[..., c]), _LANCZOS),
                    lambda c: _wr(np.ascontiguousarray(img[..., c]), _LINEAR),
                )

            drizzle_tuple = (align_model, None)
            warp_label = align_model

        # Keep satellite-trail NaN as no-data through to integration; scrub
        # only +/-inf (which would poison the reducer sums). Kernels drop NaN
        # lanes via isfinite().
        if np.isinf(aligned).any():
            aligned = np.nan_to_num(aligned, nan=np.nan, posinf=0.0, neginf=0.0)

        # NaN carry-through probe (debug): confirm the trail no-data survived the
        # warp into the aligned frame integration will read. Enable via the
        # module-level NAN_PROBE flag; routed through dbg() so it lands in the log.
        if NAN_PROBE:
            try:
                dbg(f"🔎 NaN-probe [after-align {base}] "
                    f"NaN={int(np.isnan(aligned).sum())} inf={int(np.isinf(aligned).sum())} "
                    f"shape={tuple(aligned.shape)}")
            except Exception:
                pass

        # 5) save aligned image
        name, _ = os.path.splitext(base)
        if name.endswith("_n"):
            name = name[:-2]
        if not name.endswith("_n_r"):
            name += "_n_r"

        out_path = os.path.join(output_directory, f"{name}.fit")

        # ── Drizzle self-description stamp ───────────────────────────────────
        # Persist everything drizzle needs to reconstruct this frame's entry
        # WITHOUT the .sasd: the source original, the reference grid, and the
        # transform kind + matrix. This lets "Skip Registration and Integrate"
        # drizzle from already-aligned frames (any session, even after the .sasd
        # was overwritten by a later run). affine/homography carry their matrix;
        # poly*/tps are dense coordinate maps with no small matrix, so we record
        # only kind + ref grid and let the deposit treat them as already-warped
        # (identity canvas), which is exactly how the drizzle loop handles them.
        try:
            _dz_kind, _dz_mat = (drizzle_tuple if isinstance(drizzle_tuple, tuple)
                                 and len(drizzle_tuple) == 2 else (None, None))
            if _dz_kind:
                hdr["SASDZ"]    = (True, "SASpro drizzle stamp present")
                hdr["SASDZKND"] = (str(_dz_kind), "Drizzle transform kind")
                hdr["SASDZRFH"] = (int(Href), "Drizzle reference height (px)")
                hdr["SASDZRFW"] = (int(Wref), "Drizzle reference width (px)")
                # Original source path (drizzle deposits from these pixels for
                # affine/homography; for poly*/tps the aligned pixels are used).
                try:
                    hdr["SASDZORG"] = (os.path.basename(orig_path), "Drizzle source original (basename)")
                    hdr.add_comment(f"SASDZORGPATH={os.path.normpath(orig_path)}")
                except Exception:
                    pass
                # Matrix rows, only for the small-matrix kinds.
                if _dz_mat is not None and str(_dz_kind) in ("affine", "similarity", "homography"):
                    _M = np.asarray(_dz_mat, np.float64)
                    if str(_dz_kind) == "homography":
                        _M = _M.reshape(3, 3)
                        _rows = 3
                    else:
                        _M = _M.reshape(2, 3)
                        _rows = 2
                    hdr["SASDZNR"] = (int(_rows), "Drizzle matrix row count")
                    hdr["SASDZNC"] = (int(_M.shape[1]), "Drizzle matrix col count")
                    # Flatten row-major into SASDZ00.. cards (FITS key <= 8 chars).
                    _flat = _M.reshape(-1)
                    for _i, _v in enumerate(_flat):
                        hdr[f"SASDZ{_i:02d}"] = (float(_v), f"Drizzle matrix elem {_i}")
        except Exception:
            # Stamping is best-effort; never block the aligned write on it.
            pass

        from setiastro.saspro.legacy.image_manager import save_image as _legacy_save
        _legacy_save(
            img_array=aligned,
            filename=out_path,
            original_format="fit",
            bit_depth=None,
            original_header=hdr,
            is_mono=is_mono
        )

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

def _proximity_match_pairs(tgt_gray, ref_gray, *, det_sigma=12.0, minarea=10,
                           tol_px=5.0, max_total=2000,
                           min_fwhm=1.2, max_ellipticity=0.6):
    """ICP-style match for two already-near-aligned frames (post-WCS-reproject).
    Nearest-neighbour in pixel space, like the Gaia plate solver. Returns
    (src_xy, tgt_xy) in target/reference coords, or None if too few matches."""
    P_t = _detect_stars_uniform(tgt_gray, det_sigma, minarea, grid=(6, 6),
                                max_per_cell=60, max_total=max_total,
                                min_fwhm=min_fwhm, max_ellipticity=max_ellipticity,
                                mask=_zero_ignore_mask(tgt_gray))
    P_r = _detect_stars_uniform(ref_gray, det_sigma, minarea, grid=(6, 6),
                                max_per_cell=60, max_total=max_total,
                                min_fwhm=min_fwhm, max_ellipticity=max_ellipticity,
                                mask=_zero_ignore_mask(ref_gray))
    if len(P_t) < 8 or len(P_r) < 8:
        return None

    dist, idx = KDTree(P_r).query(P_t, k=1, workers=-1)
    keep = dist < float(tol_px)
    if keep.sum() < 8:
        return None

    # collapse many target stars snapping to one reference star → keep the closest
    order = np.argsort(dist[keep])
    t_sel = np.flatnonzero(keep)[order]
    r_sel = idx[t_sel]
    seen, s_t, s_r = set(), [], []
    for ti, ri in zip(t_sel, r_sel):
        ri = int(ri)
        if ri in seen:
            continue
        seen.add(ri); s_t.append(ti); s_r.append(ri)
    if len(s_t) < 8:
        return None
    return P_t[np.array(s_t)].astype(np.float32), P_r[np.array(s_r)].astype(np.float32)

def _detect_stars_uniform(img32: np.ndarray,
                          det_sigma: float = 12.0,
                          minarea: int = 10,
                          grid=(4,4),
                          max_per_cell: int = 25,
                          max_total: int = 500,
                          min_fwhm: float = 1.2,
                          max_ellipticity: float = 0.6, mask=None) -> np.ndarray:
    import numpy as np
    import sep

    img32 = np.asarray(img32, np.float32, order="C")
    H, W = img32.shape[:2]

    # Satellite-trail no-data reaches detection as NaN; SEP cannot background or
    # extract across NaN, so fold non-finite pixels into the ignore mask and
    # fill them for the stats. Detection copy only -- the saved frame keeps NaN.
    if not np.isfinite(img32).all():
        _nf = ~np.isfinite(img32)
        mask = _nf if mask is None else (np.asarray(mask, bool) | _nf)
        img32 = np.nan_to_num(img32, nan=0.0, posinf=0.0, neginf=0.0)

    bkg = sep.Background(img32, bw=64, bh=64, mask=mask)                    # <- mask
    thresh = float(det_sigma) * float(bkg.globalrms)
    objs = sep.extract(img32 - bkg.back(), thresh, minarea=int(minarea),
                       mask=mask, segmentation_map=False)                   # <- mask
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
        from setiastro.saspro import astroalign  # deferred: see tools/apply_import_fixes.py (astroalign)
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
            # Pass A: base warp (anti-ring clamped). Color case batched into
            # ONE HWC warp so the star mask sees max-across-channels (fixes
            # the OSC narrowband dark-ring regression — see _warp_antiring).
            if base_kind == "affine":
                def _wa(ch, flags):
                    return cv2.warpAffine(ch, base_X, (Ww, Hh),
                                          flags=flags, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                if img.ndim == 2:
                    img_base = _warp_antiring(lambda: _wa(img, cv2.INTER_LANCZOS4),
                                              lambda: _wa(img, cv2.INTER_LINEAR))
                elif img.shape[2] <= 4:
                    _img = img if img.flags['C_CONTIGUOUS'] else np.ascontiguousarray(img)
                    img_base = _warp_antiring(lambda: _wa(_img, cv2.INTER_LANCZOS4),
                                              lambda: _wa(_img, cv2.INTER_LINEAR))
                else:
                    img_base = _warp_hwc_antiring(
                        img.shape[2],
                        lambda c: _wa(np.ascontiguousarray(img[..., c]), cv2.INTER_LANCZOS4),
                        lambda c: _wa(np.ascontiguousarray(img[..., c]), cv2.INTER_LINEAR),
                    )
            else:
                def _wp(ch, flags):
                    return cv2.warpPerspective(ch, base_X, (Ww, Hh),
                                               flags=flags, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                if img.ndim == 2:
                    img_base = _warp_antiring(lambda: _wp(img, cv2.INTER_LANCZOS4),
                                              lambda: _wp(img, cv2.INTER_LINEAR))
                elif img.shape[2] <= 4:
                    _img = img if img.flags['C_CONTIGUOUS'] else np.ascontiguousarray(img)
                    img_base = _warp_antiring(lambda: _wp(_img, cv2.INTER_LANCZOS4),
                                              lambda: _wp(_img, cv2.INTER_LINEAR))
                else:
                    img_base = _warp_hwc_antiring(
                        img.shape[2],
                        lambda c: _wp(np.ascontiguousarray(img[..., c]), cv2.INTER_LANCZOS4),
                        lambda c: _wp(np.ascontiguousarray(img[..., c]), cv2.INTER_LINEAR),
                    )

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
        # Color case: batched HWC call so _warp_antiring builds its star
        # mask from max-across-channels (fixes OSC narrowband ring).
        Hh, Ww = out_hw
        if kind == "affine":
            A = np.asarray(X, np.float64)
            def _wa(ch, flags):
                return cv2.warpAffine(ch, A, (Ww, Hh), flags=flags,
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            if img.ndim == 2:
                return _warp_antiring(lambda: _wa(img, cv2.INTER_LANCZOS4),
                                      lambda: _wa(img, cv2.INTER_LINEAR))
            if img.shape[2] <= 4:
                _img = img if img.flags['C_CONTIGUOUS'] else np.ascontiguousarray(img)
                return _warp_antiring(lambda: _wa(_img, cv2.INTER_LANCZOS4),
                                      lambda: _wa(_img, cv2.INTER_LINEAR))
            return _warp_hwc_antiring(
                img.shape[2],
                lambda c: _wa(np.ascontiguousarray(img[..., c]), cv2.INTER_LANCZOS4),
                lambda c: _wa(np.ascontiguousarray(img[..., c]), cv2.INTER_LINEAR),
            )

        if kind.startswith("poly"):
            return X(img, (Hh, Ww))

        if kind == "homography":
            H = np.asarray(X, np.float64)
            def _wp(ch, flags):
                return cv2.warpPerspective(ch, H, (Ww, Hh), flags=flags,
                                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            if img.ndim == 2:
                return _warp_antiring(lambda: _wp(img, cv2.INTER_LANCZOS4),
                                      lambda: _wp(img, cv2.INTER_LINEAR))
            if img.shape[2] <= 4:
                _img = img if img.flags['C_CONTIGUOUS'] else np.ascontiguousarray(img)
                return _warp_antiring(lambda: _wp(_img, cv2.INTER_LANCZOS4),
                                      lambda: _wp(_img, cv2.INTER_LINEAR))
            return _warp_hwc_antiring(
                img.shape[2],
                lambda c: _wp(np.ascontiguousarray(img[..., c]), cv2.INTER_LANCZOS4),
                lambda c: _wp(np.ascontiguousarray(img[..., c]), cv2.INTER_LINEAR),
            )

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
            # Lower values give a finer solve grid but are slower. (The
            # double stars on meridian-flipped frames once blamed on a coarse
            # grid were the DS->full lift error fixed in _S().)
            ds = max(1, int(self.downsample))
            self.solve_downsample = ds

            if ds > 1 and cv2 is not None:
                new_hw = (max(1, ref2d.shape[1] // ds), max(1, ref2d.shape[0] // ds))  # (W, H)
                ref_ds = cv2.resize(self.ref_small_full, new_hw, interpolation=cv2.INTER_AREA)
            else:
                ref_ds = self.ref_small_full

            self.ref_small = self.ref_small_full               # keep existing attribute name (full)
            self.ref_small_ds = np.ascontiguousarray(ref_ds.astype(np.float32, copy=False))

            # Persist the DS reference ONCE so solve workers memmap it instead of
            # receiving a multi-MB pickled copy in every job (N x passes copies over
            # the spawn pipe). Cleaned up in the finally at the end of run().
            self._refds_dir  = tempfile.mkdtemp(prefix="sas_refds_")
            self._ref_ds_npy = os.path.join(self._refds_dir, "ref_ds.npy")
            np.save(self._ref_ds_npy, self.ref_small_ds)

            # Initialize transforms to identity for EVERY original frame
            self.alignment_matrices = {os.path.normpath(f): IDENTITY_2x3.copy() for f in self.original_files}
            self.delta_transforms = {}

            #sasprogress totals (units = number of worker completions across passes)
            #sasprogress totals:
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
            # A frame counts only once finalize wrote its aligned output; until
            # then its entry still points at its own input.
            aligned_count = 0
            for f in self.original_files:
                k = os.path.normpath(f)
                cur = self.file_key_to_current_path.get(k, k)
                if os.path.normpath(cur) != k and os.path.exists(cur):
                    aligned_count += 1
            summary = f"Registration complete. Valid frames: {aligned_count}/{total_count}."
            self.registration_complete.emit(True, summary)

        except Exception as e:
            self.registration_complete.emit(False, f"Error: {e}")
        finally:
            try:
                import shutil
                _d = getattr(self, "_refds_dir", None)
                if _d:
                    shutil.rmtree(_d, ignore_errors=True)
            except Exception:
                pass


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

        Href_ds, Wref_ds = self.ref_small_ds.shape[:2]
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
                self._ref_ds_npy, int(Wref_ds), int(Href_ds),
                resample_flag, float(self.det_sigma),
                int(self.limit_stars) if self.limit_stars is not None else None,
                int(self.minarea),
                refine_model, float(self.h_reproj),
                int(ds),
                float(self.min_fwhm),          # ← new
                float(self.max_ellipticity),   # ← new
                (int(self.ref_small_full.shape[1]), int(self.ref_small_full.shape[0])),  # full-res ref (W, H)
            ))

        executor = _make_executor(procs)
        try:
            fut_info, pending = {}, set()
            for j in jobs:
                f = executor.submit(_solve_delta_job, j)
                fut_info[f] = j[0]
                pending.add(f)

            last_completion = time.monotonic()
            while pending:
                if self._is_cancelled():
                    self.progress_update.emit("⏹ Cancelling alignment — stopping at next frame boundary…")
                    for f in pending:
                        f.cancel()
                    break
                done, pending = wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)

                if not done:
                    # timeout_sec was read but never enforced, so a wedged solve
                    # job span forever too. Treat "nothing completed at all" as
                    # the stall it is.
                    if timeout_sec > 0 and (time.monotonic() - last_completion) > timeout_sec:
                        stuck = [os.path.basename(str(fut_info.get(f, "?"))) for f in list(pending)[:5]]
                        more = f" (+{len(pending) - 5} more)" if len(pending) > 5 else ""
                        self.progress_update.emit(
                            f"⏱️ Alignment stalled: nothing completed in {timeout_sec}s. "
                            f"Abandoning {len(pending)} frame(s): {', '.join(stuck)}{more}"
                        )
                        for f in pending:
                            f.cancel()
                            k_stuck = rev_current_to_orig.get(
                                os.path.normpath(str(fut_info.get(f, ""))),
                                os.path.normpath(str(fut_info.get(f, "")))
                            )
                            if pass_index == 0:
                                self.alignment_matrices[k_stuck] = None
                            self._increment_progress()
                        pending = set()
                        break
                    continue

                last_completion = time.monotonic()
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
        def _wa(ch, flags):
            return cv2.warpAffine(ch, T, (w, h), flags=flags,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        if image.ndim == 2:
            aligned = _warp_antiring(lambda: _wa(image, cv2.INTER_LANCZOS4),
                                     lambda: _wa(image, cv2.INTER_LINEAR))
        else:
            aligned = np.stack([
                _warp_antiring(lambda i=i: _wa(np.ascontiguousarray(image[:, :, i]), cv2.INTER_LANCZOS4),
                               lambda i=i: _wa(np.ascontiguousarray(image[:, :, i]), cv2.INTER_LINEAR))
                for i in range(image.shape[2])], axis=2)
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

        # Finalize is capped on worker COUNT (each job holds a full-res frame +
        # anti-ring warp buffers). To fill the remaining cores we instead let each
        # worker's OpenCV use cores/workers threads on the compute-heavy warp — no
        # extra frames in RAM. Self-balances if finalize_workers is raised.
        _fin_cv2_threads = max(1, (os.cpu_count() or 8) // finalize_workers)
        os.environ["SASPRO_FINALIZE_CV2_THREADS"] = str(_fin_cv2_threads)
        self.progress_update.emit(
            self.tr(f"📝 Finalize: {finalize_workers} workers x {_fin_cv2_threads} cv2 threads")
        )

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
        # Watchdog: a single wedged worker used to hang this stage forever at
        # N-1 of N with the Stop button dead, because as_completed() has no
        # timeout and `with ... as ex` then blocked again in shutdown(wait=True).
        # Mirror the solve pass: poll with a timeout, honour cancellation, and
        # give up on frames that stop making progress.
        import time as _time
        stall_timeout = int(self.align_prefs.get("timeout_per_job_sec", 300))
        ex = _make_executor(finalize_workers)
        try:
            fut_info, pending = {}, set()
            for j in jobs:
                f = ex.submit(_finalize_write_job, j)
                fut_info[f] = j[0]
                pending.add(f)

            last_completion = _time.monotonic()
            while pending:
                if self._is_cancelled():
                    self.progress_update.emit(
                        "⏹ Cancelling finalize — stopping at next frame boundary…"
                    )
                    for f in pending:
                        f.cancel()
                    break

                done, pending = wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)

                if not done:
                    if stall_timeout > 0 and (_time.monotonic() - last_completion) > stall_timeout:
                        self.progress_update.emit(
                            f"⏱️ Finalize stalled: nothing completed in {stall_timeout}s. "
                            f"Treating {len(pending)} unfinished frame(s) as alignment failures."
                        )
                        for f in pending:
                            f.cancel()
                            stuck_path = str(fut_info.get(f, "") or "")
                            # Mark it failed exactly as a frame that never
                            # solved: no output was written, so it must not be
                            # counted as valid, and the .sasd must not hand out
                            # a transform for a frame that has no aligned file.
                            if stuck_path:
                                self.alignment_matrices[os.path.normpath(stuck_path)] = None
                                self.progress_update.emit(
                                    f"⏭️ Skipping {os.path.basename(stuck_path)} — "
                                    f"finalize did not complete, not writing output."
                                )
                            self._increment_progress()   # keep the bar able to reach total
                        pending = set()
                        break
                    continue

                last_completion = _time.monotonic()
                for fut in done:
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
                            except Exception as _dz_err:
                                self.progress_update.emit(f"⚠️ Could not store drizzle transform for {os.path.basename(orig_path)}: {_dz_err}")
                    self._increment_progress()
        finally:
            # Never wait here: a wedged worker would block shutdown just as it
            # blocked as_completed(), re-freezing the UI at the same point.
            try:
                ex.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
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
# WCS-based registration thread (plate-solve based, optional stellar refine)
# ---------------------------------------------------------------------
# Lives alongside StarRegistrationThread (which is astroalign-only). The dialog
# dispatches to this thread when the Advanced panel's Method combo is set to
# "WCS" or "WCS → Stellar refine". The literal implementation is what the user
# asked for: iterate targets, call the SAME wcs_reproject_align helper that
# the StellarAlignmentDialog's single-shot WCS button already uses. No new
# alignment code — just a loop over frames with signalling + file I/O to match
# StarRegistrationThread's interface so the dialog can swap one for the other
# transparently (same signals, same file_key_to_current_path mapping, so view
# routing and the completion dialog work unchanged).
#
# "WCS → Stellar" additionally runs a single astroalign.find_transform step on
# the already-reprojected result against the reference, then a cv2.warpAffine
# with the correction. That's the same two-pass pattern Stellar Alignment's
# "wcs_stellar" method uses, done per-frame here.
class WCSRegistrationThread(QThread):
    progress_update = pyqtSignal(str)
    registration_complete = pyqtSignal(bool, str)
    progress_step = pyqtSignal(int, int)

    def __init__(self, reference_image_path_or_view, files_to_align,
                 output_directory, method: str = "wcs",
                 parent_window=None, align_prefs: dict | None = None):
        """
        reference_image_path_or_view: path string OR "__ACTIVE_VIEW__".
        method: "wcs" (reproject only) or "wcs_stellar" (reproject + astroalign refine).
        """
        super().__init__()
        self.reference = reference_image_path_or_view
        self.parent_window = parent_window
        self._cancel_event = (getattr(parent_window, "_cancel_event", None)
                              if parent_window is not None else None)
        self.original_files = [os.path.normpath(f) for f in files_to_align]
        self.files_to_align = self.original_files.copy()
        self.output_directory = os.path.normpath(output_directory)
        self.method = (method or "wcs").strip().lower()
        if self.method not in ("wcs", "wcs_stellar"):
            self.method = "wcs"
        self.align_prefs = align_prefs or _align_prefs(QSettings())
        # Same bookkeeping StarRegistrationThread exposes so the dialog's
        # _route_view_outputs_to_docs() can match aligned outputs back to
        # view-sourced inputs without caring which thread class produced them.
        self.file_key_to_current_path = {f: f for f in self.original_files}
        self.drizzle_xforms = {}
        self._done = 0
        self._total = len(self.original_files)

    def _is_cancelled(self) -> bool:
        try:
            return bool(self._cancel_event is not None and self._cancel_event.is_set())
        except Exception:
            return False

    def _load_ref(self) -> tuple[np.ndarray | None, "object | None"]:
        """
        Return (ref_image, ref_header) for the chosen reference. For a
        view-sourced reference the header comes from the live doc via
        _doc_header_or_wcs — which tries original_header FIRST but also
        falls back to the 'wcs_header' key SASpro commonly stores for a
        plate-solved view. Views previously surfaced as "no plate solve"
        here when the full FITS header wasn't in the metadata dict but a
        parsed WCS string was.
        """
        if isinstance(self.reference, str) and self.reference == "__ACTIVE_VIEW__":
            doc = _active_doc_from_parent(self.parent_window)
            img = _doc_image(doc)
            hdr = _doc_header_or_wcs(doc)
            return img, hdr
        try:
            img, hdr, _, _ = load_image(self.reference)
            return img, hdr
        except Exception:
            return None, None

    def _out_path_for(self, orig_path: str) -> str:
        """
        Mirror _finalize_write_job's output-naming convention (<base>_n_r.fit)
        so a view spill coming out of this thread lands at the same place the
        stellar-path output would, and the dialog's view-routing lookup keys
        on paths identically.
        """
        base = os.path.basename(orig_path)
        name, _ = os.path.splitext(base)
        if name.endswith("_n"):
            name = name[:-2]
        if not name.endswith("_n_r"):
            name += "_n_r"
        return os.path.join(self.output_directory, f"{name}.fit")

    def _stellar_refine(self, aligned: np.ndarray, ref2d: np.ndarray) -> np.ndarray:
        """
        One-shot astroalign correction on an already-reprojected result.
        Reuses the same library (and tuning knobs) the stellar path uses —
        no new algorithm invented. Returns the refined image, or the input
        unchanged if the correction can't be found (we'd rather keep the
        WCS result than drop the frame).
        """
        try:
            from setiastro.saspro import astroalign as aa
            src_gray = (np.mean(aligned, axis=2) if aligned.ndim == 3
                        else aligned).astype(np.float32)
            src_gray = np.ascontiguousarray(src_gray)
            ref_gray = np.ascontiguousarray(ref2d.astype(np.float32))
            det_sigma = float(self.align_prefs.get("det_sigma", 12.0))
            minarea   = int(self.align_prefs.get("minarea", 10))
            tform, _ = aa.find_transform(
                src_gray, ref_gray,
                detection_sigma=det_sigma, min_area=minarea,
            )
            M = np.asarray(tform.params, dtype=np.float64)[:2, :]
            H, W = ref2d.shape[:2]
            _LANCZOS = _warp_hi_interp()
            _LINEAR  = cv2.INTER_LINEAR
            # Route the WCS→Stellar refine warp through _warp_antiring too
            # — raw Lanczos here was a second, overlooked ring-exposure path
            # that bypassed the anti-ring guard entirely. One batched HWC
            # call so the star mask is max-across-channels.
            def _wa(ch, flags):
                return cv2.warpAffine(
                    ch, M, (W, H), flags=flags,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
                )
            if aligned.ndim == 2:
                return _warp_antiring(lambda: _wa(aligned, _LANCZOS),
                                      lambda: _wa(aligned, _LINEAR))
            if aligned.shape[2] <= 4:
                _aligned = (aligned if aligned.flags['C_CONTIGUOUS']
                            else np.ascontiguousarray(aligned))
                return _warp_antiring(lambda: _wa(_aligned, _LANCZOS),
                                      lambda: _wa(_aligned, _LINEAR))
            return _warp_hwc_antiring(
                aligned.shape[2],
                lambda c: _wa(np.ascontiguousarray(aligned[..., c]), _LANCZOS),
                lambda c: _wa(np.ascontiguousarray(aligned[..., c]), _LINEAR),
            )
        except Exception as e:
            self.progress_update.emit(
                f"⚠ Stellar refine skipped (keeping WCS result): {e}"
            )
            return aligned

    def run(self):
        try:
            ref_img, ref_hdr = self._load_ref()
            if ref_img is None:
                self.registration_complete.emit(
                    False, "WCS registration: reference image failed to load.")
                return
            ref_wcs = _wcs_from_header(ref_hdr)
            if ref_wcs is None:
                self.registration_complete.emit(
                    False, "WCS registration: reference image has no usable "
                           "plate solve. Plate-solve it first.")
                return

            H, W = ref_img.shape[:2]
            ref2d = (np.mean(ref_img, axis=2) if ref_img.ndim == 3
                     else ref_img).astype(np.float32, copy=False)

            self._total = len(self.original_files)
            self._done = 0
            self.progress_step.emit(self._done, self._total)
            self.progress_update.emit(
                f"WCS registration started — method={self.method}, "
                f"{self._total} frames."
            )

            os.makedirs(self.output_directory, exist_ok=True)

            n_ok = 0
            n_fail = 0
            n_no_wcs = 0

            for orig in self.original_files:
                if self._is_cancelled():
                    self.registration_complete.emit(False, "Cancelled by user.")
                    return
                k = os.path.normpath(orig)
                base = os.path.basename(orig)
                try:
                    img, hdr, _, _ = load_image(orig)
                    if img is None:
                        self.progress_update.emit(f"⚠ {base}: failed to load; skipping.")
                        n_fail += 1
                        self._done += 1
                        self.progress_step.emit(self._done, self._total)
                        continue

                    tgt_wcs = _wcs_from_header(hdr)
                    if tgt_wcs is None:
                        self.progress_update.emit(
                            f"⚠ {base}: no usable WCS; skipping (plate-solve it first)."
                        )
                        n_no_wcs += 1
                        self._done += 1
                        self.progress_step.emit(self._done, self._total)
                        continue

                    self.progress_update.emit(f"WCS reproject: {base}…")
                    aligned = wcs_reproject_align(
                        img, tgt_wcs, ref_wcs, (H, W),
                        tile=512,
                        progress_cb=None,
                        log=lambda s: None,
                    )

                    if self.method == "wcs_stellar":
                        self.progress_update.emit(f"Stellar refine: {base}…")
                        aligned = self._stellar_refine(aligned, ref2d)

                    # Write out matching _finalize_write_job's naming scheme
                    # so downstream (view routing, drizzle sidecars) sees a
                    # path shaped like the stellar path would have produced.
                    out_path = self._out_path_for(orig)
                    out_hdr = hdr.copy() if isinstance(hdr, Header) else Header()
                    # Record which method produced this frame for later audit.
                    try:
                        out_hdr["SASREGMD"] = (self.method.upper(),
                                               "SASpro registration method")
                    except Exception:
                        pass
                    fits.PrimaryHDU(
                        data=np.asarray(aligned, dtype=np.float32),
                        header=out_hdr,
                    ).writeto(out_path, overwrite=True)

                    self.file_key_to_current_path[k] = out_path
                    n_ok += 1
                except Exception as e:
                    import traceback
                    self.progress_update.emit(f"⚠ {base}: {e}")
                    try:
                        print(traceback.format_exc())
                    except Exception:
                        pass
                    n_fail += 1

                self._done += 1
                self.progress_step.emit(self._done, self._total)

            parts = [f"WCS registration complete."]
            parts.append(f"{n_ok}/{self._total} frames written")
            if n_no_wcs:
                parts.append(f"{n_no_wcs} skipped (no WCS)")
            if n_fail:
                parts.append(f"{n_fail} failed")
            self.registration_complete.emit(True, " · ".join(parts) + ".")

        except Exception as e:
            self.registration_complete.emit(False, f"Error: {e}")


# ---------------------------------------------------------------------
# Optional simple batch UI (file-based) — no slots
# ---------------------------------------------------------------------
class StarRegistrationWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.reference_image = None
        # One of:
        #   None                              — nothing selected yet
        #   "<path>"                          — a FITS/image file on disk
        #   "__VIEW__"                        — a chosen SASpro view
        #                                       (actual doc in self._reference_doc)
        #   "__ACTIVE_VIEW__" (deprecated)    — the live active view;
        #                                       kept as a sentinel the thread
        #                                       layer still understands for
        #                                       back-compat with callers that
        #                                       set it directly.
        self.reference_image_path = None
        # Set only when reference_image_path == "__VIEW__". The dialog spills
        # this doc's pixels to a temp FITS at start_registration time (so WCS
        # headers survive and the thread sees a plain file path), rather than
        # relying on "active view" at run-time, which would race with the user
        # clicking other views between selection and Start.
        self._reference_doc = None
        # Mixed-target list. Each entry is either a path (str) OR a dict
        # {"kind": "view", "doc": <doc>, "title": <display name>}. The dialog
        # keeps this shape; start_registration spills every view entry to a
        # temp FITS file (feeding the multiprocess workers the same way
        # file-path entries do) and the thread only ever sees paths.
        self.files_to_align: list = []
        # Scratch directory holding view-spill FITS files for one run. Set
        # in start_registration, torn down in registration_finished so a
        # cancelled or crashed run doesn't leak gigabytes of spill images.
        self._view_spill_dir: str | None = None
        # Per-run bookkeeping: {spill_input_path_normcased: {"doc": <doc>,
        # "title": <str>, "spill_path": <str>}}. Lets registration_finished
        # match each aligned output file back to the source doc so the
        # pixels can be pushed into it (or opened as a new doc) and the
        # on-disk output file deleted — no "final files" leak into user
        # folders from view-sourced targets.
        self._view_spill_by_input: dict[str, dict] = {}
        # True while the current run's output_directory is the view-spill
        # scratch dir (no file targets at all). Lets registration_finished
        # drop the whole dir without worrying about user files.
        self._output_dir_is_scratch: bool = False
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

        self.select_ref_view_button = QPushButton("From View…")
        self.select_ref_view_button.setToolTip(
            "Pick any open view as the reference image. Replaced the former\n"
            "'From Active View' shortcut so you can choose a specific view "
            "without having to activate it first — matches the target-side\n"
            "'From Views…' chooser."
        )
        self.select_ref_view_button.clicked.connect(self.select_reference_from_view)

        self.select_ref_file_button = QPushButton("From File")
        self.select_ref_file_button.clicked.connect(self.select_reference_from_file)

        ref_layout.addWidget(self.ref_label)
        ref_layout.addWidget(self.ref_path_label)
        ref_layout.addWidget(self.select_ref_view_button)
        ref_layout.addWidget(self.select_ref_file_button)

        # File selection — plus "From Views" so the user can register images
        # that are open as documents in SASpro but have never been saved to
        # disk (a freshly stacked combination, a work-in-progress). Views
        # are spilled to temp FITS at start-registration time so the
        # downstream multiprocess workers can read them exactly like any
        # file-path target.
        file_selection_layout = QHBoxLayout()
        self.add_files_button = QPushButton("Select Files")
        self.add_files_button.clicked.connect(self.select_files_to_align)
        self.add_directory_button = QPushButton("Select Directory")
        self.add_directory_button.clicked.connect(self.select_directory_to_align)
        self.add_views_button = QPushButton("From Views…")
        self.add_views_button.setToolTip(
            "Add one or more open views as alignment targets.\n"
            "The view's current pixels are used (not whatever is on disk).\n"
            "Each selected view is spilled to a temp FITS at run time and "
            "aligned like any other target. The aligned result is routed "
            "back into SASpro (overwrite the source view OR open as a new "
            "view, see 'View output' below) — nothing is left behind on disk."
        )
        self.add_views_button.clicked.connect(self.select_views_to_align)
        file_selection_layout.addWidget(self.add_files_button)
        file_selection_layout.addWidget(self.add_directory_button)
        file_selection_layout.addWidget(self.add_views_button)

        # Tree of files + views to align. Column 0 is the display label,
        # column 1 is a tiny "Source" badge ("File" / "View") so the user
        # can tell at a glance which entries are open-doc targets.
        self.tree_widget = QTreeWidget()
        self.tree_widget.setColumnCount(2)
        self.tree_widget.setHeaderLabels(["Targets to Align", "Source"])
        self.tree_widget.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        try:
            self.tree_widget.setColumnWidth(1, 60)
        except Exception:
            pass

        tree_buttons_layout = QHBoxLayout()
        self.remove_selected_button = QPushButton("Remove Selected")
        self.remove_selected_button.clicked.connect(self.remove_selected_files)
        self.clear_tree_button = QPushButton("Clear All")
        self.clear_tree_button.clicked.connect(self.clear_tree)
        tree_buttons_layout.addWidget(self.remove_selected_button)
        tree_buttons_layout.addWidget(self.clear_tree_button)

        # Output directory. Required when there are FILE targets (the
        # aligned files land here). When every target is a view it becomes
        # optional — a per-run scratch dir takes its place and nothing is
        # written to a user folder.
        output_layout = QHBoxLayout()
        self.output_label = QLabel("Output Directory:")
        self.output_path_label = QLabel("No directory selected")
        self.output_path_label.setWordWrap(True)
        self.select_output_button = QPushButton("Select Output Folder")
        self.select_output_button.clicked.connect(self.select_output_directory)
        output_layout.addWidget(self.output_label)
        output_layout.addWidget(self.output_path_label)
        output_layout.addWidget(self.select_output_button)

        # ---- Advanced alignment settings --------------------------------
        # Mirror the knobs the sibling Stellar Alignment dialog exposes, so
        # a Registration user can tune per-run without opening Stacking
        # Suite's prefs panel. Values are seeded from _align_prefs() (same
        # loader every other tool uses — stacking/align/* with legacy
        # align/* fallback), and get WRITTEN BACK to stacking/align/* on
        # Start so changes propagate to the rest of the suite — matching
        # Stellar Alignment's save-on-run behaviour. A small italic note
        # calls that out so a user editing here doesn't think it's dialog-
        # local.
        _prefs = _align_prefs(QSettings())

        self.advanced_group = QGroupBox("Advanced — Alignment Settings")
        self.advanced_group.setCheckable(True)
        self.advanced_group.setChecked(False)   # collapsed by default
        _adv = QFormLayout(self.advanced_group)
        _adv.setContentsMargins(10, 6, 10, 6)
        _adv.setVerticalSpacing(4)

        # Method — same three choices as Stellar Alignment, same persistence
        # key (stacking/align/method). Dispatches which thread runs:
        #   stellar       → StarRegistrationThread (astroalign, N-pass)
        #   wcs           → WCSRegistrationThread (reproject only)
        #   wcs_stellar   → WCSRegistrationThread (reproject + astroalign refine)
        # The two WCS methods require every target AND the reference to have
        # a valid plate solve in their FITS header — the thread refuses the
        # frame when WCS isn't there, and reports the skipped count in the
        # completion message.
        self.cmb_adv_method = QComboBox()
        self.cmb_adv_method.addItems([
            "Stellar (astroalign)",
            "WCS (plate-solve reproject)",
            "WCS → Stellar refine",
        ])
        _saved_method = str(QSettings().value("stacking/align/method", "stellar")).lower()
        self.cmb_adv_method.setCurrentIndex(
            {"stellar": 0, "wcs": 1, "wcs_stellar": 2}.get(_saved_method, 0)
        )
        self.cmb_adv_method.setToolTip(
            "How frames get aligned to the reference:\n"
            "  • Stellar: fast, uses detected stars (astroalign). No WCS needed.\n"
            "  • WCS: reprojects each frame via its own plate solve (SIP\n"
            "    preserved) onto the reference pixel grid. Requires every\n"
            "    frame + reference to be plate-solved.\n"
            "  • WCS → Stellar: WCS first for the coarse alignment, then one\n"
            "    astroalign correction pass on top. Best accuracy when WCS\n"
            "    is available but imperfect."
        )
        _adv.addRow("Method:", self.cmb_adv_method)

        # Transform model — same wording and index order as Stellar
        # Alignment so a user who learned the names there isn't surprised.
        self.cmb_adv_model = QComboBox()
        self.cmb_adv_model.addItems([
            "Affine (fast)",
            "Homography (projective)",
            "Polynomial (order 3)",
            "Polynomial (order 4)",
        ])
        _MODEL_TO_IDX = {"affine": 0, "homography": 1, "poly3": 2, "poly4": 3}
        self.cmb_adv_model.setCurrentIndex(_MODEL_TO_IDX.get(_prefs["model"], 0))
        self.cmb_adv_model.setToolTip(
            "Transform model used for final warp.\n"
            " • Affine: fast; good for well-matched optics.\n"
            " • Homography: handles perspective shear (e.g. meridian flips)\n"
            " • Polynomial 3/4: fits radial / mild field distortion."
        )
        _adv.addRow("Model:", self.cmb_adv_model)

        self.sp_adv_downsample = QSpinBox()
        self.sp_adv_downsample.setRange(1, 8)
        self.sp_adv_downsample.setValue(int(_prefs["downsample"]))
        self.sp_adv_downsample.setToolTip(
            "Downsample factor used ONLY for the solve step. Lower = finer\n"
            "grid, slower; higher = faster but less precise."
        )
        _adv.addRow("Solve downsample:", self.sp_adv_downsample)

        self.sp_adv_detsigma = QDoubleSpinBox()
        self.sp_adv_detsigma.setRange(1.0, 200.0)
        self.sp_adv_detsigma.setDecimals(1)
        self.sp_adv_detsigma.setSingleStep(1.0)
        self.sp_adv_detsigma.setValue(float(_prefs["det_sigma"]))
        self.sp_adv_detsigma.setToolTip(
            "Star detection threshold (σ above background). Lower = more\n"
            "stars (may include noise). Higher = only bright stars."
        )
        _adv.addRow("Detection threshold (σ):", self.sp_adv_detsigma)

        self.sp_adv_limitstars = QSpinBox()
        self.sp_adv_limitstars.setRange(50, 5000)
        self.sp_adv_limitstars.setSingleStep(50)
        self.sp_adv_limitstars.setValue(int(_prefs["limit_stars"]))
        self.sp_adv_limitstars.setToolTip(
            "Hard cap on stars used for matching per frame. 500 is enough\n"
            "for most fields; crowded fields may benefit from more."
        )
        _adv.addRow("Max stars (limit):", self.sp_adv_limitstars)

        self.sp_adv_minarea = QSpinBox()
        self.sp_adv_minarea.setRange(1, 100)
        self.sp_adv_minarea.setValue(int(_prefs["minarea"]))
        self.sp_adv_minarea.setToolTip(
            "Minimum connected-pixel area for a star detection. Smaller\n"
            "numbers include faint/undersampled stars; larger numbers\n"
            "reject noise and hot pixels."
        )
        _adv.addRow("Min area (px):", self.sp_adv_minarea)

        self.sp_adv_minfwhm = QDoubleSpinBox()
        self.sp_adv_minfwhm.setRange(0.1, 10.0)
        self.sp_adv_minfwhm.setDecimals(2)
        self.sp_adv_minfwhm.setSingleStep(0.1)
        self.sp_adv_minfwhm.setValue(float(_prefs["min_fwhm"]))
        self.sp_adv_minfwhm.setToolTip(
            "Reject detections narrower than this FWHM — filters out\n"
            "hot pixels and cosmic ray hits that imitate point sources."
        )
        _adv.addRow("Min FWHM (px):", self.sp_adv_minfwhm)

        self.sp_adv_maxellip = QDoubleSpinBox()
        self.sp_adv_maxellip.setRange(0.0, 1.0)
        self.sp_adv_maxellip.setDecimals(2)
        self.sp_adv_maxellip.setSingleStep(0.05)
        self.sp_adv_maxellip.setValue(float(_prefs["max_ellipticity"]))
        self.sp_adv_maxellip.setToolTip(
            "Reject detections more elliptical than this (1 - b/a).\n"
            "Lower = strict (rejects slightly trailed stars); 0.6 is a\n"
            "forgiving default that keeps most real stars."
        )
        _adv.addRow("Max ellipticity:", self.sp_adv_maxellip)

        self.sp_adv_hreproj = QDoubleSpinBox()
        self.sp_adv_hreproj.setRange(0.1, 10.0)
        self.sp_adv_hreproj.setDecimals(2)
        self.sp_adv_hreproj.setSingleStep(0.1)
        self.sp_adv_hreproj.setValue(float(_prefs["h_reproj"]))
        self.sp_adv_hreproj.setToolTip(
            "RANSAC inlier threshold (px) used ONLY when Model = Homography."
        )
        _adv.addRow("Homog. RANSAC reproj (px):", self.sp_adv_hreproj)

        self.sp_adv_timeout = QSpinBox()
        self.sp_adv_timeout.setRange(30, 3600)
        self.sp_adv_timeout.setSingleStep(30)
        self.sp_adv_timeout.setValue(int(_prefs["timeout_per_job_sec"]))
        self.sp_adv_timeout.setSuffix(" s")
        self.sp_adv_timeout.setToolTip(
            "Stall watchdog. If no frame finishes within this time, the\n"
            "remaining frames are abandoned and marked as alignment\n"
            "failures so the run can finish. This is not a per-frame limit:\n"
            "a slow frame is fine as long as others keep completing. Raise\n"
            "for very large frames (6000+ px wide) or crowded fields."
        )
        _adv.addRow("Per-frame timeout:", self.sp_adv_timeout)

        # Warp interpolation kernel — the final full-res resample pass.
        # Shared QSettings key `stacking/align/kernel` is read/written by
        # BOTH this dropdown and the Stacking Suite settings dropdown, so a
        # user's choice follows them around. See `_warp_hi_interp()` for
        # the resolution order (QSettings → env var → "lanczos").
        self.cmb_adv_kernel = QComboBox()
        self.cmb_adv_kernel.addItems([
            "Lanczos4 (sharpest; rings on bright stars)",
            "Bicubic / Mitchell (less ringy; recommended)",
            "Bilinear (softest; no ringing)",
        ])
        _KERNEL_TO_IDX = {"lanczos": 0, "cubic": 1, "linear": 2}
        self.cmb_adv_kernel.setCurrentIndex(
            _KERNEL_TO_IDX.get(str(_prefs.get("kernel", "lanczos")).lower(), 0)
        )
        self.cmb_adv_kernel.setToolTip(
            "Resampling kernel used for the final full-resolution warp.\n"
            " • Lanczos4: sharpest; its negative lobes cause dark halos\n"
            "   around bright narrowband stars (what the anti-ring guard\n"
            "   is there to catch).\n"
            " • Bicubic / Mitchell: shorter negative lobes — noticeably\n"
            "   less ringy, slightly softer on well-aligned frames.\n"
            "   Recommended if dark rings persist despite the anti-ring\n"
            "   guard (OSC narrowband data in particular).\n"
            " • Bilinear: no sharpening and no ringing at all; the softest\n"
            "   result. Keep for ring-sensitive diagnostics only."
        )
        _adv.addRow("Warp interpolation:", self.cmb_adv_kernel)

        # Trial Detect — same spirit as the Stellar Alignment dialog's but
        # adapted to Registration's N-target model: runs sep detection with
        # the CURRENT Detection σ against the REFERENCE and the FIRST target
        # (file or view), so the user can dial σ in before kicking off a
        # long batch run. If either count is way off (0, or tens of
        # thousands), the colour-coded label makes it obvious — same taxonomy
        # of warnings Stellar Alignment uses so the UX is consistent.
        _trial_row = QHBoxLayout()
        self.btn_trial_detect = QPushButton("🔍 Trial Detect")
        self.btn_trial_detect.setFixedHeight(26)
        self.btn_trial_detect.setToolTip(
            "Run star detection on both the reference and the FIRST target "
            "with the current Detection σ, and report how many stars were "
            "found in each. A quick way to sanity-check σ before starting "
            "a long batch run."
        )
        self.btn_trial_detect.clicked.connect(self._run_trial_detect)
        self._lbl_trial_result = QLabel("")
        self._lbl_trial_result.setStyleSheet("color:#888;font-size:10px;")
        self._lbl_trial_result.setWordWrap(True)
        _trial_row.addWidget(self.btn_trial_detect)
        _trial_row.addWidget(self._lbl_trial_result, 1)
        _adv.addRow("", _trial_row)

        # Reset-to-defaults + stacking-suite share note
        _adv_btn_row = QHBoxLayout()
        self.btn_adv_reset = QPushButton("Reset to Stacking Suite Defaults")
        self.btn_adv_reset.setToolTip(
            "Re-read the Stacking Suite's current alignment settings and\n"
            "load them into these fields. Does NOT save — Start Registration\n"
            "is what persists any further edits back to Stacking Suite."
        )
        self.btn_adv_reset.clicked.connect(self._adv_reload_defaults)
        _adv_btn_row.addWidget(self.btn_adv_reset)
        _adv_btn_row.addStretch(1)
        _adv.addRow("", _adv_btn_row)

        _adv_note = QLabel(
            "ℹ  These settings are shared with Stacking Suite and the Stellar "
            "Alignment dialog. Starting Registration also saves changes back."
        )
        _adv_note.setStyleSheet("color:#888;font-size:10px;font-style:italic;")
        _adv_note.setWordWrap(True)
        _adv.addRow("", _adv_note)

        # Enable/disable H.-RANSAC-reproj by Model = Homography. Matches
        # Stellar Alignment's behaviour so the UI feels the same.
        def _adv_toggle_hreproj():
            is_h = (self.cmb_adv_model.currentIndex() == 1)
            self.sp_adv_hreproj.setEnabled(is_h)
            try:
                lab = _adv.labelForField(self.sp_adv_hreproj)
                if lab is not None:
                    lab.setEnabled(is_h)
            except Exception:
                pass
        _adv_toggle_hreproj()
        self.cmb_adv_model.currentIndexChanged.connect(lambda _ : _adv_toggle_hreproj())

        # Enable/disable the stellar-only knobs based on Method. For a pure
        # WCS run none of the astroalign tuning applies, so grey those out
        # instead of letting the user tweak values that will be ignored.
        # For "WCS → Stellar refine", det_sigma + minarea DO apply (the refine
        # step uses them), so those stay enabled.
        _STELLAR_ONLY_WIDGETS = (
            self.cmb_adv_model, self.sp_adv_downsample,
            self.sp_adv_limitstars, self.sp_adv_minfwhm,
            self.sp_adv_maxellip, self.sp_adv_hreproj,
            self.sp_adv_timeout,
        )
        _STELLAR_REFINE_WIDGETS = (self.sp_adv_detsigma, self.sp_adv_minarea)

        def _adv_toggle_method():
            idx = self.cmb_adv_method.currentIndex()
            uses_stellar_tuning = (idx == 0)  # Stellar only
            uses_refine_tuning  = (idx == 2)  # WCS → Stellar
            for w in _STELLAR_ONLY_WIDGETS:
                w.setEnabled(uses_stellar_tuning)
                try:
                    lab = _adv.labelForField(w)
                    if lab is not None:
                        lab.setEnabled(uses_stellar_tuning)
                except Exception:
                    pass
            # det_sigma + minarea stay active for Stellar OR WCS→Stellar.
            for w in _STELLAR_REFINE_WIDGETS:
                enabled = uses_stellar_tuning or uses_refine_tuning
                w.setEnabled(enabled)
                try:
                    lab = _adv.labelForField(w)
                    if lab is not None:
                        lab.setEnabled(enabled)
                except Exception:
                    pass
            # Re-apply the H-RANSAC rule within the Stellar case so the
            # Homography-only field stays disabled unless both conditions hold.
            if uses_stellar_tuning:
                _adv_toggle_hreproj()
        _adv_toggle_method()
        self.cmb_adv_method.currentIndexChanged.connect(lambda _ : _adv_toggle_method())

        # Collapse the group when unchecked so it visually gets out of the
        # way. Qt's checkable QGroupBox already disables child widgets on
        # uncheck; we hide them too so the dialog shrinks.
        def _adv_toggle_visible(checked: bool):
            for i in range(_adv.rowCount()):
                lab_item = _adv.itemAt(i, QFormLayout.ItemRole.LabelRole)
                fld_item = _adv.itemAt(i, QFormLayout.ItemRole.FieldRole)
                for it in (lab_item, fld_item):
                    if it is None:
                        continue
                    w = it.widget()
                    lay = it.layout()
                    if w is not None:
                        w.setVisible(checked)
                    elif lay is not None:
                        for j in range(lay.count()):
                            sw = lay.itemAt(j).widget()
                            if sw is not None:
                                sw.setVisible(checked)
        _adv_toggle_visible(False)
        self.advanced_group.toggled.connect(_adv_toggle_visible)

        # View-output mode — only meaningful when at least one view target is
        # on the list. Overwrite replaces the source view's pixels in place
        # (fast, no new document clutter); New view opens the aligned result
        # as a fresh document (keeps the original intact for comparison). The
        # group is shown or hidden by _refresh_view_output_group as the user
        # adds/removes view targets, so it never appears for pure file runs.
        self.view_output_group = QGroupBox("View output (how aligned views are returned to SASpro)")
        _vo_lay = QHBoxLayout(self.view_output_group)
        self.radio_view_overwrite = QRadioButton("Overwrite source view")
        self.radio_view_newdoc    = QRadioButton("Open as new view")
        self.radio_view_overwrite.setChecked(True)
        self.radio_view_overwrite.setToolTip(
            "Replace the source view's pixels with the aligned result.\n"
            "Fast and in-place — the original view is modified. Use when\n"
            "you don't need the pre-aligned pixels any more."
        )
        self.radio_view_newdoc.setToolTip(
            "Open the aligned result as a new view, leaving the original\n"
            "untouched. Use when you want to compare before/after or keep\n"
            "the pre-aligned version around."
        )
        _vo_lay.addWidget(self.radio_view_overwrite)
        _vo_lay.addWidget(self.radio_view_newdoc)
        _vo_lay.addStretch(1)
        # Hidden until a view target is added.
        self.view_output_group.setVisible(False)

        #sasprogress + start
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
        main_layout.addWidget(self.advanced_group)
        main_layout.addWidget(self.view_output_group)
        start_row = QHBoxLayout()
        start_row.addWidget(self.start_button, 1)
        start_row.addWidget(make_help_button("star_registration", self))
        main_layout.addWidget(self.progress_label)
        main_layout.addLayout(start_row)

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
    def select_reference_from_view(self):
        """
        Pop a single-select chooser listing every open view and let the
        user pick which one is the reference. Replaces the former "From
        Active View" shortcut: more explicit, and avoids the subtle bug
        where the user clicks the button, switches to a different view to
        inspect a target, then presses Start — "Active View" would resolve
        to whatever happened to be active at Start, not what they intended.

        The chosen doc is remembered on self._reference_doc so a run-time
        active-view switch can't change what gets registered against.
        """
        docs = self._enumerate_open_docs()
        docs = [(t, d) for (t, d) in docs if _doc_image(d) is not None]
        if not docs:
            QMessageBox.information(
                self, "No Views",
                "No open views have image data available to use as a reference.\n\n"
                "Open one or more images first, then try again."
            )
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Pick Reference View")
        dlg.setMinimumWidth(360)
        lay = QVBoxLayout(dlg)
        lay.addWidget(QLabel("Select the reference view (every target is aligned to this one):"))
        lst = QListWidget(dlg)
        lst.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        # Pre-select the currently-active view when possible so a quick OK
        # with no clicking reproduces the old "From Active View" behaviour.
        active = _active_doc_from_parent(self.parent_window)
        pre_row = 0
        for i, (title, doc) in enumerate(docs):
            item = QListWidgetItem(title + ("  (active)" if doc is active else ""))
            item.setData(Qt.ItemDataRole.UserRole, doc)
            lst.addItem(item)
            if doc is active:
                pre_row = i
        lst.setCurrentRow(pre_row)
        lay.addWidget(lst)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=dlg,
        )
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        lay.addWidget(btns)
        # Double-click = pick + OK, matches typical QListWidget UX.
        lst.itemDoubleClicked.connect(lambda _: dlg.accept())

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        it = lst.currentItem()
        if it is None:
            return
        doc = it.data(Qt.ItemDataRole.UserRole)
        if doc is None:
            return

        title = _fmt_doc_title(doc)
        self.reference_image_path = "__VIEW__"
        self._reference_doc = doc
        self.ref_path_label.setText(f"View: {title}")

    def select_reference_from_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Reference Image", "",
            "Images (*.png *.jpg *.jpeg *.tif *.tiff *.fits *.fit *.xisf);;All Files (*)"
        )
        if file_path:
            self.reference_image_path = file_path
            self._reference_doc = None  # file supersedes any prior view pick
            self.ref_path_label.setText(os.path.basename(file_path))

    # File selection
    def _add_file_entry(self, file_path: str) -> bool:
        """Append a file-path target to the tree if not already present.
        Returns True if a new row was added."""
        # Dedup against the string paths currently in the list; view entries
        # are dicts and never match a file-path lookup.
        for existing in self.files_to_align:
            if isinstance(existing, str) and existing == file_path:
                return False
        self.files_to_align.append(file_path)
        item = QTreeWidgetItem([os.path.basename(file_path), "File"])
        item.setToolTip(0, file_path)
        self.tree_widget.addTopLevelItem(item)
        self._refresh_view_output_group()
        return True

    def _has_view_targets(self) -> bool:
        return any(isinstance(e, dict) and e.get("kind") == "view"
                   for e in self.files_to_align)

    def _has_file_targets(self) -> bool:
        return any(isinstance(e, str) for e in self.files_to_align)

    def _refresh_view_output_group(self):
        """
        Show the View-output radios only when at least one view target is
        on the list — pure file runs don't need the extra UI and seeing it
        with no bearing on the run would be confusing.
        """
        try:
            self.view_output_group.setVisible(self._has_view_targets())
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Advanced alignment settings
    # ------------------------------------------------------------------
    _ADV_IDX_TO_MODEL = ("affine", "homography", "poly3", "poly4")
    _ADV_IDX_TO_KERNEL = ("lanczos", "cubic", "linear")

    def _collect_align_prefs(self) -> dict:
        """
        Snapshot the Advanced panel into the dict shape that
        StarRegistrationThread (and _align_prefs) expects. Values returned
        here are also what gets written back to QSettings on Start, so this
        is the one place that defines "what the UI is currently set to."
        """
        return {
            "model":       self._ADV_IDX_TO_MODEL[self.cmb_adv_model.currentIndex()],
            "max_cp":      250,  # unused by registration; keep key for schema parity
            "downsample":  int(self.sp_adv_downsample.value()),
            "h_reproj":    float(self.sp_adv_hreproj.value()),
            "det_sigma":   float(self.sp_adv_detsigma.value()),
            "limit_stars": int(self.sp_adv_limitstars.value()),
            "minarea":     int(self.sp_adv_minarea.value()),
            "timeout_per_job_sec": int(self.sp_adv_timeout.value()),
            "min_fwhm":        float(self.sp_adv_minfwhm.value()),
            "max_ellipticity": float(self.sp_adv_maxellip.value()),
            # Warp interpolation kernel — written to the shared QSettings key
            # `stacking/align/kernel` so the Stacking Suite settings dropdown
            # sees the same value.
            "kernel": self._ADV_IDX_TO_KERNEL[
                max(0, min(self.cmb_adv_kernel.currentIndex(),
                           len(self._ADV_IDX_TO_KERNEL) - 1))
            ],
        }

    def _adv_reload_defaults(self):
        """
        Repopulate the Advanced fields from whatever _align_prefs() reads
        off QSettings right now. Used by the Reset button and when the
        dialog opens. Also re-reads the Method key (stacking/align/method)
        and refreshes the enabled-fields gating accordingly.
        """
        p = _align_prefs(QSettings())
        _MODEL_TO_IDX = {"affine": 0, "homography": 1, "poly3": 2, "poly4": 3}
        self.cmb_adv_model.setCurrentIndex(_MODEL_TO_IDX.get(p["model"], 0))
        self.sp_adv_downsample.setValue(int(p["downsample"]))
        self.sp_adv_hreproj.setValue(float(p["h_reproj"]))
        self.sp_adv_detsigma.setValue(float(p["det_sigma"]))
        self.sp_adv_limitstars.setValue(int(p["limit_stars"]))
        self.sp_adv_minarea.setValue(int(p["minarea"]))
        self.sp_adv_timeout.setValue(int(p["timeout_per_job_sec"]))
        self.sp_adv_minfwhm.setValue(float(p["min_fwhm"]))
        self.sp_adv_maxellip.setValue(float(p["max_ellipticity"]))
        _saved_method = str(QSettings().value("stacking/align/method", "stellar")).lower()
        self.cmb_adv_method.setCurrentIndex(
            {"stellar": 0, "wcs": 1, "wcs_stellar": 2}.get(_saved_method, 0)
        )
        _KERNEL_TO_IDX = {"lanczos": 0, "cubic": 1, "linear": 2}
        self.cmb_adv_kernel.setCurrentIndex(
            _KERNEL_TO_IDX.get(str(p.get("kernel", "lanczos")).lower(), 0)
        )

    _ADV_IDX_TO_METHOD = ("stellar", "wcs", "wcs_stellar")

    def _current_method(self) -> str:
        idx = self.cmb_adv_method.currentIndex()
        return self._ADV_IDX_TO_METHOD[idx] if 0 <= idx < 3 else "stellar"

    def _save_align_prefs(self, prefs: dict) -> None:
        """
        Write the Advanced panel back to QSettings under stacking/align/*
        — same keys the Stellar Alignment dialog and Stacking Suite use.
        Called on Start so Registration edits propagate to the rest of the
        suite (matches Stellar Alignment's save-on-run).

        `stacking/align/kernel` is the one Stacking Suite's settings page
        also writes; `_warp_hi_interp()` reads it live on every warp so a
        change here takes effect without a restart.
        """
        s = QSettings()
        s.setValue("stacking/align/method",      self._current_method())
        s.setValue("stacking/align/model",       str(prefs["model"]))
        s.setValue("stacking/align/downsample",  int(prefs["downsample"]))
        s.setValue("stacking/align/h_reproj",    float(prefs["h_reproj"]))
        s.setValue("stacking/align/det_sigma",   float(prefs["det_sigma"]))
        s.setValue("stacking/align/limit_stars", int(prefs["limit_stars"]))
        s.setValue("stacking/align/minarea",     int(prefs["minarea"]))
        s.setValue("stacking/align/timeout_per_job_sec",
                   int(prefs["timeout_per_job_sec"]))
        s.setValue("stacking/align/min_fwhm",        float(prefs["min_fwhm"]))
        s.setValue("stacking/align/max_ellipticity", float(prefs["max_ellipticity"]))
        s.setValue(_WARP_KERNEL_SETTINGS_KEY,         str(prefs.get("kernel", "lanczos")))

    # ------------------------------------------------------------------
    # Trial detect
    # ------------------------------------------------------------------
    def _load_trial_image_from_reference(self) -> tuple[np.ndarray | None, str]:
        """
        Return (ref_image_2D_or_3D, label) for whatever reference the user
        currently has selected. Three branches, matching how
        start_registration resolves the reference:
          - "__VIEW__"        : chosen doc (self._reference_doc, pixels read live)
          - "__ACTIVE_VIEW__" : deprecated — resolves to the currently active doc
          - "<path>"          : a FITS/image file on disk
        """
        rp = self.reference_image_path
        if rp == "__VIEW__" and self._reference_doc is not None:
            img = _doc_image(self._reference_doc)
            title = _fmt_doc_title(self._reference_doc)
            return img, f"Reference view ({title})"
        if rp == "__ACTIVE_VIEW__":
            img, _, _ = _get_image_from_active_view(self.parent_window)
            return img, "Reference (active view)"
        if isinstance(rp, str) and rp:
            try:
                img, _, _, _ = load_image(rp)
                return img, f"Reference ({os.path.basename(rp)})"
            except Exception:
                return None, "Reference"
        return None, "Reference"

    def _load_trial_image_from_first_target(self) -> tuple[np.ndarray | None, str]:
        """
        Return (target_image, label) for the FIRST target on the list.
        Views don't need spilling here — detection just needs numpy pixels
        and _doc_image gives us that directly. File entries get loaded via
        load_image, same as the thread would.
        """
        if not self.files_to_align:
            return None, "First target"

        first = self.files_to_align[0]
        if isinstance(first, dict) and first.get("kind") == "view":
            doc = first.get("doc")
            img = _doc_image(doc)
            title = first.get("title") or "view"
            return img, f"First target ({title})"

        if isinstance(first, str):
            try:
                img, _, _, _ = load_image(first)
                return img, f"First target ({os.path.basename(first)})"
            except Exception:
                return None, f"First target ({os.path.basename(first)})"

        return None, "First target"

    def _run_trial_detect(self):
        """
        Run sep-based star detection on the reference and the first target
        with the CURRENT Advanced-panel σ, and show counts in the little
        label next to the button. The warning taxonomy (0 stars, very high
        counts, pixstack exhaustion) is copied from Stellar Alignment's
        trial detect so the two tools report problems the same way.

        Pure-WCS runs don't use star detection, so Trial Detect would be
        meaningless for them; return a short info note instead of running.
        """
        if self._current_method() == "wcs":
            self._lbl_trial_result.setStyleSheet("color:#888;font-size:10px;")
            self._lbl_trial_result.setText(
                "Trial detect doesn't apply to pure WCS alignment "
                "(no stars are detected — reprojection uses the plate solve)."
            )
            return

        import sep
        thresh = float(self.sp_adv_detsigma.value())

        self.btn_trial_detect.setEnabled(False)
        self._lbl_trial_result.setText("Detecting…")
        QApplication.processEvents()

        try:
            ref_img, ref_label = self._load_trial_image_from_reference()
            tgt_img, tgt_label = self._load_trial_image_from_first_target()
        except Exception as e:
            self._lbl_trial_result.setText(f"⚠ Could not load images: {e}")
            self.btn_trial_detect.setEnabled(True)
            return

        if ref_img is None and tgt_img is None:
            self._lbl_trial_result.setText(
                "⚠ Select a reference and at least one target first."
            )
            self.btn_trial_detect.setEnabled(True)
            return

        def _count_stars(img):
            if img is None:
                return None, None
            gray = (np.mean(img, axis=2).astype(np.float32)
                    if img.ndim == 3 else img.astype(np.float32))
            gray = np.ascontiguousarray(gray)
            try:
                sep.set_extract_pixstack(5000000)
                bkg = sep.Background(gray)
                objs = sep.extract(gray - bkg.back(),
                                   thresh * float(bkg.globalrms))
                return len(objs), None
            except Exception as e:
                err = str(e)
                if "pixel buffer full" in err or "pixstack" in err.lower():
                    return None, f"too many pixels at σ={thresh:.1f} — raise threshold"
                return None, err

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

        ref_n, ref_err = _count_stars(ref_img)
        tgt_n, tgt_err = _count_stars(tgt_img)

        parts = []
        if ref_img is not None:
            parts.append(_fmt(ref_label, ref_n, ref_err))
        if tgt_img is not None:
            parts.append(_fmt(tgt_label, tgt_n, tgt_err))

        msg = "  |  ".join(parts) if parts else "(no images to test)"
        has_warning = "⚠" in msg
        self._lbl_trial_result.setStyleSheet(
            f"color:{'#ffc107' if has_warning else '#4caf50'};font-size:10px;"
        )
        self._lbl_trial_result.setText(msg)
        self.btn_trial_detect.setEnabled(True)

    def select_files_to_align(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Files to Align", "",
            "Images (*.png *.jpg *.jpeg *.tif *.tiff *.fits *.fit *.xisf);;All Files (*)"
        )
        if files:
            for file in files:
                self._add_file_entry(file)

    def select_directory_to_align(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Directory", "")
        if directory:
            exts = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.fits', '.fit', '.xisf')
            new_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(exts)]
            for file in new_files:
                self._add_file_entry(file)

    def select_views_to_align(self):
        """
        Pop a chooser listing every open document and add the user's picks
        as view-targets. The view's CURRENT pixels are what gets aligned —
        this reads the live doc, not whatever file the doc may have been
        loaded from, so a stacked combination that hasn't been saved to
        disk aligns correctly.

        Uses parent._list_open_docs when available, falling back to the
        same MDI-scraping helper the Stellar Alignment dialog already uses
        (_list_open_docs_fallback), so the same set of views appears here
        as in the sibling tools.
        """
        docs = self._enumerate_open_docs()
        # Filter to docs that actually have image data to align — a doc
        # with doc.image is None is useless as a target and showing it in
        # the chooser just invites confusion.
        docs = [(t, d) for (t, d) in docs if _doc_image(d) is not None]
        if not docs:
            QMessageBox.information(
                self, "No Views",
                "No open views have image data available to align.\n\n"
                "Open one or more images first, then use 'From Views…' again."
            )
            return

        # Dedup against views already on the list — matching by doc identity,
        # not by title (two docs can share a title; the user still means two
        # distinct inputs).
        already = {id(e["doc"]) for e in self.files_to_align
                   if isinstance(e, dict) and e.get("kind") == "view"}

        # Build the chooser dialog. QListWidget with multi-select is enough
        # here; a tree would be overkill for a flat list of docs.
        dlg = QDialog(self)
        dlg.setWindowTitle("Add Views as Alignment Targets")
        dlg.setMinimumWidth(360)
        lay = QVBoxLayout(dlg)
        lay.addWidget(QLabel("Select one or more open views to add as targets:"))
        lst = QListWidget(dlg)
        lst.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        for title, doc in docs:
            item = QListWidgetItem(title + (" (already added)" if id(doc) in already else ""))
            # Stash the doc on the item so we don't have to re-match by title.
            item.setData(Qt.ItemDataRole.UserRole, doc)
            if id(doc) in already:
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsSelectable
                              & ~Qt.ItemFlag.ItemIsEnabled)
            lst.addItem(item)
        lay.addWidget(lst)
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=dlg,
        )
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        lay.addWidget(btns)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        for item in lst.selectedItems():
            doc = item.data(Qt.ItemDataRole.UserRole)
            if doc is None or id(doc) in already:
                continue
            title = _fmt_doc_title(doc)
            self.files_to_align.append({
                "kind": "view",
                "doc": doc,
                "title": title,
            })
            # Row label shows the view name, column 1 is a "View" badge so
            # mixed lists (files + views) stay unambiguous at a glance.
            tree_item = QTreeWidgetItem([title, "View"])
            tree_item.setToolTip(0, f"Open view: {title} "
                                    f"(pixels taken from the live doc at run time)")
            self.tree_widget.addTopLevelItem(tree_item)
        self._refresh_view_output_group()

    def _enumerate_open_docs(self) -> list[tuple[str, object]]:
        """
        Return [(title, doc), ...] for every open view the host exposes.
        Prefers parent._list_open_docs (used by other SASpro tools) and
        falls back to walking the MDI area — same two-tier pattern as
        _list_open_docs_fallback.
        """
        pw = self.parent_window
        for attr in ("_list_open_docs", "list_open_docs"):
            fn = getattr(pw, attr, None)
            if callable(fn):
                try:
                    out = fn()
                    if out:
                        return [(str(t), d) for (t, d) in out]
                except Exception:
                    pass
        return _list_open_docs_fallback(pw)

    # Manage tree
    def remove_selected_files(self):
        """
        Remove highlighted rows. Rows are matched to backing entries by the
        tree's visual index (not by text) so files and views with similar
        display names, or two files with the same basename in different
        folders, remove the correct entry.
        """
        selected_items = self.tree_widget.selectedItems()
        # Collect indices first so popping doesn't shift later indices
        indices = sorted(
            {self.tree_widget.indexOfTopLevelItem(it) for it in selected_items},
            reverse=True,
        )
        for idx in indices:
            if 0 <= idx < len(self.files_to_align):
                del self.files_to_align[idx]
            self.tree_widget.takeTopLevelItem(idx)
        self._refresh_view_output_group()

    def clear_tree(self):
        self.tree_widget.clear()
        self.files_to_align.clear()
        self._refresh_view_output_group()

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
            QMessageBox.warning(self, "No Targets", "Please add files or views to align before starting.")
            return

        # Output directory is only REQUIRED when there are file targets —
        # those have to land somewhere the user can find them. A view-only
        # run routes everything back into SASpro documents, so the user
        # shouldn't have to pick a folder just to see nothing land in it.
        if self._has_file_targets() and not self.output_directory:
            QMessageBox.warning(self, "No Output Directory",
                                "Please select an output directory for the file targets.")
            return

        # Resolve mixed file + view targets into a flat list of file paths
        # for the thread (which only speaks paths — see spill rationale on
        # self._view_spill_dir). Any view that fails to spill is skipped
        # with a user-visible message rather than silently dropped.
        try:
            resolved_paths = self._spill_views_and_resolve_paths()
        except Exception as e:
            QMessageBox.critical(
                self, "View Spill Failed",
                f"Could not prepare open views for registration:\n{e}"
            )
            self._cleanup_view_spill()
            return

        if not resolved_paths:
            QMessageBox.warning(
                self, "No Targets",
                "After resolving views, no targets remained to align."
            )
            self._cleanup_view_spill()
            return

        # Resolve a "__VIEW__" reference by spilling the chosen doc's pixels
        # to a FITS in the same per-run scratch dir the targets use. The
        # thread then sees a plain file path and loads WCS via load_image /
        # _wcs_from_header exactly as it would for a file reference. The
        # scratch dir is created on-demand if no target views exist yet.
        effective_reference = self.reference_image_path
        if effective_reference == "__VIEW__":
            if self._reference_doc is None:
                QMessageBox.warning(
                    self, "Missing Reference",
                    "Please pick a reference view again — the previous "
                    "selection is no longer available."
                )
                self._cleanup_view_spill()
                return
            try:
                effective_reference = self._spill_reference_view()
            except Exception as e:
                QMessageBox.critical(
                    self, "Reference Spill Failed",
                    f"Could not prepare the reference view for registration:\n{e}"
                )
                self._cleanup_view_spill()
                return

        # Pick the output directory the thread will use. For a view-only
        # run we hand it the scratch dir — the aligned FITS land there,
        # registration_finished routes them into docs, then the whole
        # scratch dir is nuked. No files leak into user folders.
        if self.output_directory:
            effective_output_dir = self.output_directory
            self._output_dir_is_scratch = False
        else:
            # View-only run; scratch dir doubles as output dir.
            effective_output_dir = self._view_spill_dir
            self._output_dir_is_scratch = True

        # Snapshot the Advanced panel into an align_prefs dict and
        # persist it back to QSettings so other tools see the user's
        # choices (matches Stellar Alignment's save-on-run). The thread
        # gets the same dict directly, so a change here affects THIS run
        # even if QSettings somehow races.
        active_prefs = self._collect_align_prefs()
        try:
            self._save_align_prefs(active_prefs)
        except Exception as e:
            # Non-fatal — the run still has active_prefs in hand.
            self._enqueue_progress(f"Note: couldn't persist alignment settings: {e}")

        self.progress_label.setText("Status: Running…")
        self.progress_label.setStyleSheet("color: green; font-weight: bold;")

        # Dispatch to the right thread for the chosen Method. Both threads
        # expose the same signals (progress_update / registration_complete /
        #sasprogress_step) and populate self.thread.file_key_to_current_path
        # with the same convention, so _route_view_outputs_to_docs,
        # _enqueue_progress and registration_finished are oblivious to
        # which one actually ran.
        method = self._current_method()
        if method in ("wcs", "wcs_stellar"):
            self.thread = WCSRegistrationThread(
                effective_reference,
                resolved_paths,
                effective_output_dir,
                method=method,
                parent_window=self.parent_window,
                align_prefs=active_prefs,
            )
        else:
            self.thread = StarRegistrationThread(
                effective_reference,
                resolved_paths,
                effective_output_dir,
                parent_window=self.parent_window,
                align_prefs=active_prefs,
            )
        self.thread.progress_update.connect(self._enqueue_progress)
        self.thread.registration_complete.connect(self.registration_finished)
        self.thread.start()

    def update_progress(self, message):
        self.progress_label.setText(f"Status: {message}")

    def registration_finished(self, success, message):
        # On success, route any view-sourced aligned outputs back into
        # SASpro documents BEFORE cleaning the spill scratch. The user
        # explicitly asked for no "final files" to be left behind for
        # view-sourced targets; this is where that promise is kept.
        view_report = ""
        if success and self._view_spill_by_input:
            try:
                view_report = self._route_view_outputs_to_docs()
            except Exception as e:
                # Don't swallow — tell the user something went wrong,
                # but still clean up spills so disk isn't leaked.
                view_report = (
                    f"\n\nAligned view routing FAILED: {e}\n"
                    f"The aligned files remain in the output directory "
                    f"you can import them manually."
                )

        # Clean up spill scratch + output-dir-is-scratch. If output_directory
        # was the scratch dir (view-only run) the aligned files live there
        # and we're about to delete them — but _route_view_outputs_to_docs
        # above has already consumed them into docs, so this is correct.
        self._cleanup_view_spill()

        full_msg = message + view_report
        color = "green" if success else "red"
        self.progress_label.setText(f"Status: {message}")
        self.progress_label.setStyleSheet(f"color: {color}; font-weight: bold;")
        if success:
            QMessageBox.information(self, "Registration Complete", full_msg)
        else:
            QMessageBox.warning(self, "Registration Error", full_msg)

    # ------------------------------------------------------------------
    # View-target spill helpers
    # ------------------------------------------------------------------
    def _spill_views_and_resolve_paths(self) -> list[str]:
        """
        Walk self.files_to_align and return a flat list of file paths for
        the thread. File-path entries pass through unchanged. View entries
        are written to temp FITS files in a per-run scratch directory so
        the downstream multiprocess workers can read them like any other
        FITS input.

        Also populates self._view_spill_by_input, mapping the normalised
        spill-input path to the originating doc so registration_finished
        can route the aligned output back into SASpro (overwrite the view
        or open as new) and then delete the on-disk aligned output. No
        "final files" are left on the user's disk for view-sourced targets.
        """
        resolved: list[str] = []
        self._view_spill_by_input = {}
        view_entries = [e for e in self.files_to_align
                        if isinstance(e, dict) and e.get("kind") == "view"]

        if view_entries:
            # One scratch dir per run so cleanup is a single rmtree.
            self._view_spill_dir = tempfile.mkdtemp(prefix="sas_reg_views_")

        for entry in self.files_to_align:
            if isinstance(entry, str):
                resolved.append(entry)
                continue
            if not (isinstance(entry, dict) and entry.get("kind") == "view"):
                # Unknown shape — skip and warn instead of crashing the run.
                self._enqueue_progress(f"Skipping unknown target entry: {entry!r}")
                continue

            doc = entry.get("doc")
            img = _doc_image(doc)
            if img is None:
                self._enqueue_progress(
                    f"View '{entry.get('title', '?')}' has no image data; skipping."
                )
                continue

            # Sanitise the display name into a filesystem-safe basename.
            # Multiple views sharing a title get a numeric suffix so writes
            # don't trample each other.
            safe = _safe_filename_base(entry.get("title") or "view")
            idx = 0
            while True:
                candidate = os.path.join(
                    self._view_spill_dir,
                    f"{safe}.fits" if idx == 0 else f"{safe}_{idx}.fits",
                )
                if not os.path.exists(candidate):
                    break
                idx += 1

            try:
                _write_view_to_fits(doc, img, candidate)
            except Exception as e:
                self._enqueue_progress(
                    f"Failed to spill view '{entry.get('title', '?')}' to temp FITS: {e}; skipping."
                )
                continue

            # Record mapping so routing can find the doc for this aligned
            # output after the thread finishes. Key is normcased path (as
            # the thread also normalises its keys).
            key = os.path.normcase(os.path.normpath(candidate))
            self._view_spill_by_input[key] = {
                "doc": doc,
                "title": entry.get("title") or "view",
                "spill_path": candidate,
            }
            resolved.append(candidate)

        return resolved

    def _route_view_outputs_to_docs(self) -> str:
        """
        For every view-sourced target, read the aligned output written by
        the worker, push it back into the source doc (overwrite) or open
        it as a new doc, then delete the on-disk aligned file so nothing
        leaks into the user's output folder. Returns a short summary
        string to append to the completion dialog.
        """
        if not self._view_spill_by_input:
            return ""

        overwrite = bool(self.radio_view_overwrite.isChecked())
        mw = self.parent_window
        dm = (getattr(mw, "docman", None) or getattr(mw, "doc_manager", None)) if mw else None

        if not overwrite and dm is None:
            # Can't open a new doc — fall back to overwrite with a notice.
            self._enqueue_progress(
                "Document manager unavailable — overwriting source views instead "
                "of opening new ones."
            )
            overwrite = True

        # Build reverse lookup from the thread's file_key_to_current_path.
        # file_key_to_current_path maps "normpath(original_input)" ->
        # "aligned_output_path" (or the original if alignment failed).
        key_to_out = {}
        try:
            key_to_out = dict(self.thread.file_key_to_current_path)
        except Exception:
            pass

        pushed = 0
        failed = 0
        skipped = 0

        for norm_input, info in self._view_spill_by_input.items():
            doc = info["doc"]
            title = info["title"]
            spill_path = info["spill_path"]

            # The thread's keys are built with os.path.normpath without
            # normcase; match either form.
            out_path = key_to_out.get(os.path.normpath(spill_path))
            if out_path is None:
                # Case-insensitive fallback
                for k, v in key_to_out.items():
                    if os.path.normcase(k) == norm_input:
                        out_path = v
                        break

            # If the worker never wrote an output (alignment failed), the
            # file_key_to_current_path still points at the SPILL input.
            # Skip those — the source doc stays untouched.
            if (not out_path) or os.path.normcase(os.path.normpath(out_path)) == norm_input:
                self._enqueue_progress(
                    f"View '{title}': alignment produced no output, source view unchanged."
                )
                skipped += 1
                continue

            if not os.path.exists(out_path):
                self._enqueue_progress(
                    f"View '{title}': aligned output missing at {out_path}; skipping."
                )
                skipped += 1
                continue

            try:
                aligned_img, aligned_hdr = _read_aligned_fits(out_path)
            except Exception as e:
                self._enqueue_progress(
                    f"View '{title}': failed to read aligned output: {e}"
                )
                failed += 1
                continue

            try:
                if overwrite:
                    _apply_aligned_to_doc(doc, aligned_img, aligned_hdr)
                else:
                    _open_aligned_as_new_doc(mw, dm, aligned_img, aligned_hdr,
                                             source_title=title)
                pushed += 1
            except Exception as e:
                self._enqueue_progress(
                    f"View '{title}': failed to route aligned result into SASpro: {e}"
                )
                failed += 1
                continue

            # Delete the on-disk aligned file now that it's safely in a doc.
            # No "final files" linger for view-sourced targets. For a scratch
            # output_directory the whole dir gets rmtree'd shortly anyway;
            # the per-file unlink here matters for the mixed case where the
            # output_directory is a real user folder shared with file targets.
            try:
                os.remove(out_path)
            except Exception:
                pass

        bits = []
        if pushed:
            bits.append(f"{pushed} view(s) {'overwritten' if overwrite else 'opened as new doc'}")
        if failed:
            bits.append(f"{failed} view routing failure(s)")
        if skipped:
            bits.append(f"{skipped} view(s) skipped (no aligned output)")
        if not bits:
            return ""
        return "\n\n" + " · ".join(bits) + "."

    def _spill_reference_view(self) -> str:
        """
        Write the chosen reference doc's pixels to a FITS in the per-run
        view-spill scratch dir and return the path. Reuses
        _write_view_to_fits so WCS stored under 'wcs_header' / an
        attribute / etc. is preserved into the FITS header via
        _doc_header_or_wcs — the same resolver the WCS path uses to find
        plate solves on live views.

        If no scratch dir exists yet (view-only runs may spill the
        reference before any target views), create one on demand. The
        shared cleanup in _cleanup_view_spill drops this file along with
        the target spills.
        """
        doc = self._reference_doc
        if doc is None:
            raise RuntimeError("No reference view selected.")
        img = _doc_image(doc)
        if img is None:
            raise RuntimeError("Reference view has no image data.")

        if self._view_spill_dir is None:
            self._view_spill_dir = tempfile.mkdtemp(prefix="sas_reg_views_")

        safe = _safe_filename_base(_fmt_doc_title(doc) or "reference")
        # Prefix "_ref_" so it's trivially distinguishable from target spills
        # in the scratch dir during a crash-dump debug session, and so it
        # never collides with a target view that happens to share the title.
        idx = 0
        while True:
            candidate = os.path.join(
                self._view_spill_dir,
                f"_ref_{safe}.fits" if idx == 0 else f"_ref_{safe}_{idx}.fits",
            )
            if not os.path.exists(candidate):
                break
            idx += 1

        _write_view_to_fits(doc, img, candidate)
        return candidate

    def _cleanup_view_spill(self):
        """
        Remove the per-run view-spill scratch directory and reset trackers.
        Safe to call multiple times and when no spill occurred.
        """
        d = self._view_spill_dir
        self._view_spill_dir = None
        self._view_spill_by_input = {}
        self._output_dir_is_scratch = False
        if not d:
            return
        try:
            import shutil
            shutil.rmtree(d, ignore_errors=True)
        except Exception:
            pass


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