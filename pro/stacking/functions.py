#pro.stacking_suite.py
from __future__ import annotations
import os
import glob
import shutil
import tempfile
import datetime as _dt
import sys
import platform
import gc  # For explicit memory cleanup after heavy operations
from time import perf_counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import numpy.ma as ma
import hashlib
from numpy.lib.format import open_memmap 
import tzlocal
import weakref
import re
import unicodedata
import math            # used in compute_safe_chunk
import psutil          # used in bytes_available / compute_safe_chunk
from typing import List

# Memory management utilities
from pro.memory_utils import (
    smart_zeros, smart_empty, get_buffer_pool, 
    should_use_memmap, cleanup_memmap, get_thumbnail_cache,
    LRUDict,  # LRU-bounded dict for caches
)

from PyQt6.QtCore import Qt, QTimer, QSettings, pyqtSignal, QObject, pyqtSlot, QThread, QEvent, QPoint, QSize, QEventLoop, QCoreApplication, QRectF, QPointF, QMetaObject
from PyQt6.QtGui import QIcon, QImage, QPixmap, QAction, QIntValidator, QDoubleValidator, QFontMetrics, QTextCursor, QPalette, QPainter, QPen, QTransform, QColor, QBrush, QCursor
from PyQt6.QtWidgets import (QDialog, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QLineEdit, QTreeWidget, QHeaderView, QTreeWidgetItem, QProgressBar, QProgressDialog,
                             QFormLayout, QDialogButtonBox, QToolBar, QToolButton, QFileDialog, QTabWidget, QAbstractItemView, QSpinBox, QDoubleSpinBox, QGroupBox,QRadioButton,
                             QSizePolicy, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QApplication, QScrollArea, QTextEdit, QMenu, QPlainTextEdit, QGraphicsEllipseItem,
                             QMessageBox, QSlider, QCheckBox, QInputDialog, QComboBox)





                            
from datetime import datetime, time, timedelta, timezone
# keep: import time
from datetime import datetime as dt_datetime, time as dt_time, timedelta as dt_timedelta, timezone as dt_timezone

import pyqtgraph as pg
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from PIL import Image                 # used by _get_image_size (and elsewhere)
import tifffile as tiff               # _get_image_size -> tiff.imread(...)
import cv2                            # _get_image_size -> cv2.imread(...)
cv2.setNumThreads(0)
CursorShape = Qt.CursorShape 
import rawpy
import time
import exifread
import contextlib
NO_GRAD = contextlib.nullcontext  # fallback when torch isn’t present
from typing import List, Tuple, Optional
import sep
from pathlib import Path
from legacy.xisf import XISF  # ← add this
from PyQt6 import sip
# your helpers/utilities
from imageops.stretch import stretch_mono_image, stretch_color_image, siril_style_autostretch

# Fix for missing star count preview function
from legacy.numba_utils import compute_star_count_fast_preview

# Import shared utilities
from pro.widgets.image_utils import nearest_resize_2d as _nearest_resize_2d_shared
from legacy.numba_utils import (
    windsorized_sigma_clip_weighted,
    kappa_sigma_clip_weighted,
    apply_flat_division_numba,
    subtract_dark_with_pedestal,
    debayer_raw_fast,
    drizzle_deposit_numba_kernel_mono,
    drizzle_deposit_color_kernel,
    finalize_drizzle_2d,
    finalize_drizzle_3d,
)
from numba_utils import (
    bulk_cosmetic_correction_numba,
    drizzle_deposit_numba_naive,
    drizzle_deposit_color_naive,
    bulk_cosmetic_correction_bayer
)
from legacy.image_manager import load_image, save_image, get_valid_header
from pro.star_alignment import StarRegistrationWorker, StarRegistrationThread, IDENTITY_2x3
from pro.log_bus import LogBus
from pro import comet_stacking as CS
#from pro.remove_stars import starnet_starless_from_array, darkstar_starless_from_array
from pro.mfdeconv import MultiFrameDeconvWorker
from pro.mfdeconvcudnn import MultiFrameDeconvWorkercuDNN
from pro.mfdeconvsport import MultiFrameDeconvWorkerSport
from pro.accel_installer import current_backend
from pro.accel_workers import AccelInstallWorker
from pro.runtime_torch import add_runtime_to_sys_path
from pro.free_torch_memory import _free_torch_memory
from pro.torch_rejection import (
    torch_available as _torch_ok,
    gpu_algo_supported as _gpu_algo_supported,
    torch_reduce_tile as _torch_reduce_tile,
)

import inspect
try:
    _ASARRAY_HAS_COPY = 'copy' in inspect.signature(np.asarray).parameters
except Exception:
    _ASARRAY_HAS_COPY = False

def _asarray(x, dtype=None, copy=False):
    if _ASARRAY_HAS_COPY:
        return np.asarray(x, dtype=dtype, copy=copy)
    a = np.asarray(x, dtype=dtype)
    return a.copy() if copy else a

try:
    # 3.9+: stdlib time zones
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None

try:
    # very robust parsing
    from dateutil import parser as date_parser
except Exception:
    date_parser = None

from pro.mfdeconv import (
    THRESHOLD_SIGMA as _SM_DEF_THRESH,
    GROW_PX         as _SM_DEF_GROW,
    SOFT_SIGMA      as _SM_DEF_SOFT,
    MAX_STAR_RADIUS as _SM_DEF_RMAX,
    STAR_MASK_MAXOBJS as _SM_DEF_MAXOBJS,
    KEEP_FLOOR      as _SM_DEF_KEEPF,
    ELLIPSE_SCALE   as _SM_DEF_ES,
    VARMAP_SAMPLE_STRIDE as _VM_DEF_STRIDE,
)

_WINDOWS_RESERVED = {
    "CON","PRN","AUX","NUL",
    "COM1","COM2","COM3","COM4","COM5","COM6","COM7","COM8","COM9",
    "LPT1","LPT2","LPT3","LPT4","LPT5","LPT6","LPT7","LPT8","LPT9",
}


_FITS_EXTS = ('.fits', '.fit', '.fts', '.fits.gz', '.fit.gz', '.fts.gz', '.fz')

def get_valid_header(path: str):
    """
    Return a robust FITS header for both normal and compressed FITS.

    - Opens the HDU list and picks the first image-like HDU (ndim >= 2).
    - Forces NAXIS, NAXIS1, NAXIS2 from the actual data.shape if possible.
    - Falls back to ZNAXIS1/2 for tile-compressed images.
    """
    try:
        from astropy.io import fits

        with fits.open(path, memmap=False) as hdul:
            science_hdu = None

            # Prefer the first HDU that actually has 2D+ image data
            for hdu in hdul:
                data = getattr(hdu, "data", None)
                if data is None:
                    continue
                if getattr(data, "ndim", 0) >= 2:
                    science_hdu = hdu
                    break

            if science_hdu is None:
                # Fallback: just use primary
                science_hdu = hdul[0]

            hdr = science_hdu.header.copy()
            data = science_hdu.data

            # --- Ensure NAXIS / NAXIS1 / NAXIS2 are real numbers ---
            try:
                if data is not None and getattr(data, "ndim", 0) >= 2:
                    shape = data.shape
                    # FITS: final axes are X, Y
                    ny, nx = shape[-2], shape[-1]
                    hdr["NAXIS"] = int(data.ndim)
                    hdr["NAXIS1"] = int(nx)
                    hdr["NAXIS2"] = int(ny)
            except Exception:
                pass

            # --- Extra fallback from ZNAXISn (tile-compressed FITS) ---
            for ax in (1, 2):
                key = f"NAXIS{ax}"
                zkey = f"ZNAXIS{ax}"
                val = hdr.get(key, None)
                if (val is None or (isinstance(val, str) and not val.strip())) and zkey in hdr:
                    try:
                        hdr[key] = int(hdr[zkey])
                    except Exception:
                        pass

        return hdr, True

    except Exception:
        return None, False




def _read_tile_stack(file_list, y0, y1, x0, x1, channels, out_buf):
    """
    Fill `out_buf` with the tile stack for (y0:y1, x0:x1).
    out_buf: (N, th, tw, C) float32, C-order (preallocated by caller).
    Returns (th, tw) = actual tile extents.
    """
    import os
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor, as_completed

    th = int(y1 - y0)
    tw = int(x1 - x0)
    N  = len(file_list)

    ts = out_buf[:N, :th, :tw, :channels]
    ts[...] = 0.0

    def _coerce_to_f32_hwc(a: np.ndarray, want_c: int) -> np.ndarray:
        if a is None:
            return None

        if a.ndim == 2:
            a = a[:, :, None]
        elif a.ndim == 3:
            # CHW -> HWC heuristic
            if a.shape[0] in (1, 3) and a.shape[1] >= 8 and a.shape[2] >= 8 and a.shape[-1] not in (1, 3):
                a = a.transpose(1, 2, 0)
        else:
            return None

        if a.shape[0] != th or a.shape[1] != tw:
            a = a[:th, :tw, ...]

        Csrc = a.shape[2]
        if Csrc == want_c:
            pass
        elif Csrc == 1 and want_c == 3:
            a = np.repeat(a, 3, axis=2)
        elif Csrc == 3 and want_c == 1:
            a = a.mean(axis=2, keepdims=True)
        else:
            if Csrc > want_c:
                a = a[:, :, :want_c]
            else:
                a = np.repeat(a, want_c, axis=2)

        a = np.asarray(a, dtype=np.float32, order="C")
        if not np.isfinite(a).all():
            np.nan_to_num(a, copy=False, posinf=0.0, neginf=0.0)
        return a

    max_workers = min(N, max(1, (os.cpu_count() or 4)))
    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        fut2i = {
            exe.submit(load_fits_tile, fpath, y0, y1, x0, x1): i
            for i, fpath in enumerate(file_list)
        }
        for fut in as_completed(fut2i):
            i = fut2i[fut]
            try:
                sub = fut.result()
            except Exception:
                continue

            sub = _coerce_to_f32_hwc(sub, channels)
            if sub is None:
                continue

            ts[i, :, :, :] = sub

    return th, tw



def _tile_grid(height, width, chunk_h, chunk_w):
    tiles = []
    for y0 in range(0, height, chunk_h):
        y1 = min(y0 + chunk_h, height)
        for x0 in range(0, width, chunk_w):
            x1 = min(x0 + chunk_w, width)
            tiles.append((y0, y1, x0, x1))
    return tiles


def _find_first_image_hdu(hdul):
    """
    Pick the first HDU that has a 2D or 3D image array.
    Very lightweight vs. your get_valid_header().
    """
    for idx, h in enumerate(hdul):
        try:
            d = h.data
            if d is None: 
                continue
            if d.ndim in (2, 3):
                return idx, h.header
        except Exception:
            continue
    # Fallback to primary header if nothing else
    return 0, hdul[0].header

def _to_float32_unit(img, hdr):
    """
    Convert integer types to [0..1] float32 the same way your full loader does,
    and pass-through float32/64 (cast to 32).
    """
    dt = img.dtype
    if dt == np.uint8:
        return (img.astype(np.float32, copy=False) / 255.0)
    if dt == np.uint16:
        return (img.astype(np.float32, copy=False) / 65535.0)
    if dt == np.uint32:
        return (img.astype(np.float32, copy=False) / 4294967295.0)
    if dt == np.int32:
        # honor BSCALE/BZERO like your loader
        bzero  = hdr.get('BZERO', 0.0)
        bscale = hdr.get('BSCALE', 1.0)
        return (img.astype(np.float32, copy=False) * float(bscale) + float(bzero))
    if dt in (np.float32, np.float64):
        return img.astype(np.float32, copy=False)
    # last resort
    return img.astype(np.float32, copy=False)

def load_image_fast_norm(path: str) -> tuple[np.ndarray | None, fits.Header | None]:
    """
    Super-light loader for normalization stage.
    - FITS only; eager read (memmap=False)
    - No WCS/SIP
    - Manual BLANK/BSCALE/BZERO like the preview path
    """
    lp = path.lower()
    if not lp.endswith(_FITS_EXTS):   # ensure _FITS_EXTS includes (".fit", ".fits", ".fts", ".fit.gz", ".fits.gz", ".fz")
        img, hdr, *_ = load_image(path)   # fallback for non-FITS
        return img, (hdr or fits.Header())

    try:
        img, hdr, _ = _fits_read_any_hdu_noscale(path, memmap=False)
        if img is None:
            return None, None

        # If 3D with trailing channel=1, squeeze
        if img.ndim == 3 and img.shape[-1] == 1:
            img = np.squeeze(img, axis=-1)

        # Make sure we return float32 contiguous
        img = np.ascontiguousarray(img.astype(np.float32, copy=False))
        return img, (hdr or fits.Header())
    except Exception:
        return None, None

    
def _safe_torch_inference_ctx():
    """
    Return a context manager suitable for inference:
    - torch.inference_mode() if available and enterable
    - else torch.no_grad()
    - else a no-op context if torch isn't importable
    """
    try:
        import torch
    except Exception:
        return contextlib.nullcontext  # returns the *type*; call as _ctx()

    cm = getattr(torch, "inference_mode", None)
    if cm is not None:
        # Probe once — older/alt backends may have the symbol but not the C++ hook
        try:
            with cm():
                pass
            return cm
        except Exception:
            return getattr(torch, "no_grad", contextlib.nullcontext)
    return getattr(torch, "no_grad", contextlib.nullcontext)

# ---------- Lightweight FITS preview + header/bin cache + link helper ----------

import shutil
from contextlib import contextmanager

# LRU-limited caches (500 entries each prevents unbounded growth during long sessions)
_HDR_CACHE = LRUDict(500)     # path -> fits.Header
_BIN_CACHE = LRUDict(500)     # path -> (xb, yb)

def _to_native_endian(a: np.ndarray) -> np.ndarray:
    """Return a view of `a` in native-endian dtype (no copy)."""
    bo = a.dtype.byteorder
    if bo in ('=', '|'):         # already native or not applicable
        return a
    # NumPy 2.0 replacement for .newbyteorder(): swap bytes, then view with native-endian dtype
    return a.byteswap().view(a.dtype.newbyteorder('='))

def _get_header_fast(fp):
    h = _HDR_CACHE.get(fp)
    if h is not None:
        return h
    try:
        h = fits.getheader(fp, ext=0)
    except Exception:
        h = fits.Header()
    _HDR_CACHE[fp] = h
    return h

def _bin_from_header_fast(fp: str) -> tuple[int, int]:
    b = _BIN_CACHE.get(fp)
    if b is not None:
        return b
    try:
        # Try to read the image HDU header only, fast
        with fits.open(fp, memmap=True, do_not_scale_image_data=True, ignore_missing_end=True) as hdul:
            for h in hdul:
                d = getattr(h, "data", None)
                if isinstance(d, np.ndarray) and d.ndim in (2,3) and d.size > 0:
                    xb = int(h.header.get("XBINNING", h.header.get("XBIN", 1)))
                    yb = int(h.header.get("YBINNING", h.header.get("YBIN", 1)))
                    xb = xb if xb > 0 else 1
                    yb = yb if yb > 0 else 1
                    _BIN_CACHE[fp] = (xb, yb)
                    return xb, yb
    except Exception:
        pass
    # fallback to primary header
    hdr = _get_header_fast(fp)
    xb, yb = _parse_binning_from_header(hdr)
    _BIN_CACHE[fp] = (xb, yb)
    return xb, yb

def _superpixel2x2_fast(x: np.ndarray) -> np.ndarray:
    h, w = x.shape[:2]
    h2, w2 = h - (h % 2), w - (w % 2)
    if h2 <= 0 or w2 <= 0:
        return x.astype(np.float32, copy=False)
    x = x[:h2, :w2].astype(np.float32, copy=False)
    if x.ndim == 2:
        return (x[0:h2:2, 0:w2:2] + x[0:h2:2, 1:w2:2] +
                x[1:h2:2, 0:w2:2] + x[1:h2:2, 1:w2:2]) * 0.25
    else:
        r = x[..., 0]; g = x[..., 1]; b = x[..., 2]
        L = 0.2126*r + 0.7152*g + 0.0722*b
        return (L[0:h2:2, 0:w2:2] + L[0:h2:2, 1:w2:2] +
                L[1:h2:2, 0:w2:2] + L[1:h2:2, 1:w2:2]) * 0.25

def _resize_to_scale_fast(im: np.ndarray, sx: float, sy: float) -> np.ndarray:
    if abs(sx - 1.0) < 1e-6 and abs(sy - 1.0) < 1e-6:
        return im
    H, W = im.shape[:2]
    outW = max(1, int(round(W * (1.0 / sx))))
    outH = max(1, int(round(H * (1.0 / sy))))
    return cv2.resize(im, (outW, outH), interpolation=cv2.INTER_AREA)

def _quick_preview_from_fits(fp: str, target_xbin: int, target_ybin: int) -> np.ndarray | None:
    # Robust read (no autoscale; we scale manually)
    arr, hdr, _ = _fits_read_any_hdu_noscale(fp, memmap=True)
    if arr is None:
        return None

    # Luma/collapse for 2D preview
    if arr.ndim == 3 and arr.shape[-1] == 3:
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        prev2d = 0.2126*r + 0.7152*g + 0.0722*b
    elif arr.ndim == 3 and arr.shape[0] == 3:
        r, g, b = arr[0], arr[1], arr[2]
        prev2d = 0.2126*r + 0.7152*g + 0.0722*b
    elif arr.ndim == 3:
        prev2d = np.mean(arr, axis=-1)
    else:
        prev2d = arr

    prev2d = np.asarray(prev2d, dtype=np.float32, order="C")
    prev2d = np.nan_to_num(prev2d, nan=0.0, posinf=0.0, neginf=0.0)

    # Superpixel 2×2
    prev = _superpixel2x2_fast(prev2d)

    # Use per-frame binning from this HDU’s header if present, else fallback
    xb = int(hdr.get("XBINNING", hdr.get("XBIN", 1))) if hdr else 1
    yb = int(hdr.get("YBINNING", hdr.get("YBIN", 1))) if hdr else 1
    if xb <= 0 or yb <= 0:
        xb, yb = _bin_from_header_fast(fp)

    sx = float(xb) / float(max(1, target_xbin))
    sy = float(yb) / float(max(1, target_ybin))
    if (abs(sx - 1.0) > 1e-6) or (abs(sy - 1.0) > 1e-6):
        prev = _resize_to_scale_fast(prev, sx, sy)

    return np.ascontiguousarray(prev, dtype=np.float32)


def _quick_preview_from_fits_cached(fp: str, target_xbin: int, target_ybin: int) -> np.ndarray | None:
    """
    Cached version of _quick_preview_from_fits.
    Uses thumbnail cache for faster repeated access.
    """
    cache = get_thumbnail_cache()
    target_size = (target_xbin, target_ybin)
    
    # Check cache first
    cached = cache.get(fp, target_size)
    if cached is not None:
        return cached
    
    # Generate preview
    preview = _quick_preview_from_fits(fp, target_xbin, target_ybin)
    
    # Cache it
    if preview is not None:
        cache.put(fp, target_size, preview)
    
    return preview


def _safe_hard_or_soft_link(src: str, dst: str) -> bool:
    """
    Try hardlink first (fast, space-free), fallback to symlink, then copy.
    Returns True if dst now exists (by link or copy).
    """
    try:
        if os.path.exists(dst):
            os.remove(dst)
    except Exception:
        pass
    # Hardlink (same filesystem)
    try:
        os.link(src, dst)
        return True
    except Exception:
        pass
    # Symlink (requires privileges on Windows)
    try:
        os.symlink(src, dst)
        return True
    except Exception:
        pass
    # Fallback copy (last resort)
    try:
        shutil.copyfile(src, dst)
        return True
    except Exception:
        return False


def _is_deleted(obj) -> bool:
    try:
        return sip.isdeleted(obj)
    except Exception:
        # Fallback heuristic
        try:
            _ = obj.thread()
            return False
        except Exception:
            return True

def _as_C(a: np.ndarray) -> np.ndarray:
    """Contiguous float32, no copy if possible."""
    if a.dtype != np.float32:
        a = a.astype(np.float32, copy=False)
    return a if a.flags.c_contiguous else np.ascontiguousarray(a)

def _ensure_chw(img: np.ndarray, is_mono: bool) -> np.ndarray:
    """
    Return HxW (mono) or CHW (color) float32 contiguous.
    Accepts HxW, HxWx3, CHW, or HxWx1.
    """
    if is_mono or img.ndim == 2:
        return _as_C(img)
    # 3D
    if img.shape[-1] == 1:                      # HWC with 1
        return _as_C(np.squeeze(img, axis=-1))
    if img.shape[-1] == 3:                      # HWC -> CHW
        return _as_C(img.transpose(2, 0, 1))
    if img.shape[0] in (1, 3):                  # already CHW
        return _as_C(img)
    return _as_C(img)  # fallback

def _dark_to_ch_or_hw(dark: np.ndarray) -> tuple[bool, np.ndarray]:
    """
    Normalize a dark to either (HxW) or (C,H,W) float32 contiguous.
    Returns (is_per_channel, array).
    - If HxW: (False, HxW)
    - If HWC or CHW: (True, CHW)
    """
    d = dark
    if d.ndim == 2:
        return False, _as_C(d)
    if d.ndim == 3 and d.shape[-1] in (1, 3):   # HWC -> CHW
        return True, _as_C(d.transpose(2, 0, 1))
    if d.ndim == 3 and d.shape[0] in (1, 3):    # CHW
        return True, _as_C(d)
    # Unknown: treat as mono plane by squeezing if possible
    if d.ndim == 3 and 1 in d.shape:
        d2 = np.squeeze(d)
        if d2.ndim == 2:
            return False, _as_C(d2)
    return False, _as_C(d)

def _apply_master_dark_light(light: np.ndarray, dark: np.ndarray, is_mono: bool, pedestal: float):
    """
    Subtract dark + pedestal for a *single light frame*.
    - light: HxW (mono) or CHW (color), float32 contiguous
    - dark:  HxW or CHW/HWC; automatically matched
    Uses your numba kernel for exact behavior.
    """
    from legacy.numba_utils import subtract_dark_with_pedestal as _sub

    if is_mono or light.ndim == 2:
        # mono path
        _, D = _dark_to_ch_or_hw(dark)
        if D.ndim == 3:              # CHW given → choose any channel
            D = D[0]
        return _sub(light[None, ...], D, float(pedestal))[0]

    # color path (CHW)
    is_per_ch, D = _dark_to_ch_or_hw(dark)
    out = np.empty_like(light, dtype=np.float32)
    if not is_per_ch:
        # mono dark for all channels
        for c in range(light.shape[0]):
            out[c] = _sub(light[c][None, ...], D, float(pedestal))[0]
    else:
        # per-channel (handles 1 or 3)
        base = D[0]
        for c in range(light.shape[0]):
            Dc = D[c] if c < D.shape[0] else base
            out[c] = _sub(light[c][None, ...], Dc, float(pedestal))[0]
    return out

def _subtract_dark_stack_inplace_hwc(stack_FHWC: np.ndarray, dark_HWC_or_HW: np.ndarray, pedestal: float = 0.0):
    """
    Fast vectorized subtraction for a *stack of tiles* used in master-flat build.
    Shapes:
      - stack_FHWC: (F, H, W, C) float32 contiguous
      - dark_HWC_or_HW: (H, W, C) or (H, W) float32
    Subtracts in-place: stack -= dark + pedestal.
    """
    D = dark_HWC_or_HW
    if D.ndim == 2:
        D = D[..., None]                      # (H,W) -> (H,W,1)
        if stack_FHWC.shape[-1] == 3:
            D = np.repeat(D, 3, axis=2)       # (H,W,3)
    # ensure contig
    D = _as_C(D)
    np.subtract(stack_FHWC, D[None, ...], out=stack_FHWC)
    if pedestal:
        stack_FHWC -= float(pedestal)



def _Luma(img: np.ndarray) -> np.ndarray:
    """
    Return a float32 mono view: grayscale image → itself, RGB → Luma, (H,W,1) → squeeze.
    """
    a = np.asarray(img)
    if a.ndim == 2:
        return a.astype(np.float32, copy=False)
    if a.ndim == 3 and a.shape[-1] == 3:
        a = a.astype(np.float32, copy=False)
        r, g, b = a[..., 0], a[..., 1], a[..., 2]
        return 0.2126 * r + 0.7152 * g + 0.0722 * b
    return np.squeeze(a, axis=-1).astype(np.float32, copy=False)

def normalize_images(stack: np.ndarray,
                     target_median: float,
                     use_Luma: bool = True) -> np.ndarray:
    """
    Min-offset + median-match normalization with NO clipping.
    For each frame f:
      - L = Luma(f) if RGB else f
      - f0 = f - min(L)
      - gain = target_median / median(Luma(f0))
      - out = f0 * gain

    Returns float32, C-contiguous array with same shape as input.
    """
    assert stack.ndim in (3, 4), "stack must be (F,H,W) or (F,H,W,3)"
    F = stack.shape[0]
    out = np.empty_like(stack, dtype=np.float32)
    eps = 1e-12

    def _L(a: np.ndarray) -> np.ndarray:
        if not use_Luma:
            return a.astype(np.float32, copy=False)
        if a.ndim == 2:
            return a.astype(np.float32, copy=False)
        if a.ndim == 3 and a.shape[-1] == 3:
            a = a.astype(np.float32, copy=False)
            r, g, b = a[..., 0], a[..., 1], a[..., 2]
            return 0.2126 * r + 0.7152 * g + 0.0722 * b
        return np.squeeze(a, axis=-1).astype(np.float32, copy=False)

    for i in range(F):
        print(f"Normalizing {i}")
        f = stack[i].astype(np.float32, copy=False)
        L = _L(f)
        fmin = float(np.nanmin(L))
        f0 = f - fmin
        L0 = _L(f0)
        fmed = float(np.nanmedian(L0))
        gain = (target_median / max(fmed, eps)) if target_median > 0 else 1.0
        out[i] = f0 * gain

    return np.ascontiguousarray(out, dtype=np.float32)

# ----- small helpers -----

def _force_shape_hw(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """
    Return img with exactly (target_h, target_w) spatial shape.
    If bigger → center-crop; if smaller → reflect-pad. Preserves channels/layout.
    """
    import numpy as np
    import cv2
    a = np.asarray(img)
    if a.ndim < 2:
        return a

    if a.ndim == 2:
        H, W = a.shape
        Caxis = None
    elif a.ndim == 3:
        # supports HWC and CHW; detect which one by comparing small channel count
        if a.shape[-1] in (1, 3):      # HWC
            H, W = a.shape[:2]; Caxis = -1
        elif a.shape[0] in (1, 3):     # CHW
            H, W = a.shape[1:]; Caxis = 0
        else:
            # assume HWC
            H, W = a.shape[:2]; Caxis = -1
    else:
        return a

    th, tw = int(target_h), int(target_w)
    if (H, W) == (th, tw): 
        return a

    # --- center-crop if larger ---
    y0 = max(0, (H - th) // 2)
    x0 = max(0, (W - tw) // 2)
    y1 = min(H, y0 + th)
    x1 = min(W, x0 + tw)

    if a.ndim == 2:
        cropped = a[y0:y1, x0:x1]
    elif Caxis == -1:  # HWC
        cropped = a[y0:y1, x0:x1, ...]
    else:               # CHW
        cropped = a[:, y0:y1, x0:x1]

    ch, cw = (cropped.shape[:2] if (cropped.ndim == 2 or Caxis == -1) else cropped.shape[1:3])

    # --- reflect-pad if smaller ---
    pad_t = max(0, (th - ch) // 2)
    pad_b = max(0, th - ch - pad_t)
    pad_l = max(0, (tw - cw) // 2)
    pad_r = max(0, tw - cw - pad_l)

    if pad_t or pad_b or pad_l or pad_r:
        if cropped.ndim == 2:
            cropped = cv2.copyMakeBorder(cropped, pad_t, pad_b, pad_l, pad_r,
                                         borderType=cv2.BORDER_REFLECT_101)
        elif Caxis == -1:
            # HWC: pad spatial, keep channels intact
            cropped = cv2.copyMakeBorder(cropped, pad_t, pad_b, pad_l, pad_r,
                                         borderType=cv2.BORDER_REFLECT_101)
        else:
            # CHW: pad each channel slice
            chans = []
            for c in range(cropped.shape[0]):
                chans.append(cv2.copyMakeBorder(cropped[c], pad_t, pad_b, pad_l, pad_r,
                                                borderType=cv2.BORDER_REFLECT_101))
            cropped = np.stack(chans, axis=0)

    return cropped


def _downsample_area(img: np.ndarray, scale: int) -> np.ndarray:
    """
    Robust area downsample by an integer scale (>=1).
    - Prefers exact block-mean pooling (no OpenCV) when full blocks fit.
    - Falls back to cv2.resize with a clamped, non-zero dsize.
    - Final fallback uses stride slicing (nearest-like) so this never throws.
    """
    if img is None:
        return None

    # Normalize inputs
    scale = int(max(1, scale))
    a = np.asarray(img, dtype=np.float32)
    if a.ndim < 2 or a.size == 0:
        return a

    H, W = int(a.shape[0]), int(a.shape[1])
    if H <= 0 or W <= 0 or scale == 1:
        return a

    # ---- Prefer exact block mean when we have whole blocks ----
    Hs = (H // scale) * scale
    Ws = (W // scale) * scale
    if Hs >= scale and Ws >= scale:
        a_c = np.ascontiguousarray(a[:Hs, :Ws, ...])  # ensure contiguous for reshape
        if a.ndim == 2:
            out = a_c.reshape(Hs // scale, scale, Ws // scale, scale).mean(axis=(1, 3))
            return out.astype(np.float32, copy=False)
        else:
            C = a.shape[2]
            out = a_c.reshape(Hs // scale, scale, Ws // scale, scale, C).mean(axis=(1, 3))
            return out.astype(np.float32, copy=False)

    # ---- Fallback to OpenCV with explicit, clamped dsize ----
    tw = max(1, W // scale)
    th = max(1, H // scale)
    if tw == W and th == H:
        return a  # nothing to do

    try:
        import cv2
        a_c = np.ascontiguousarray(a)  # OpenCV likes contiguous
        out = cv2.resize(a_c, (int(tw), int(th)), interpolation=cv2.INTER_AREA)
        return out.astype(np.float32, copy=False)
    except Exception:
        # Last resort: stride slicing (nearest-ish), always returns something
        if a.ndim == 2:
            return a[::scale, ::scale].astype(np.float32, copy=False)
        else:
            return a[::scale, ::scale, :].astype(np.float32, copy=False)



def _upscale_bg(bg_small: np.ndarray, oh: int, ow: int) -> np.ndarray:
    """
    Robust upscale of the background model to (oh, ow). Never passes 0 sizes to cv2.
    """
    oh = int(max(1, oh)); ow = int(max(1, ow))
    if bg_small is None:
        return np.zeros((oh, ow), dtype=np.float32)

    b = np.asarray(bg_small, dtype=np.float32)
    if b.ndim < 2 or b.shape[0] == 0 or b.shape[1] == 0:
        return np.zeros((oh, ow), dtype=np.float32)

    try:
        import cv2
        return cv2.resize(b, (ow, oh), interpolation=cv2.INTER_LANCZOS4).astype(np.float32, copy=False)
    except Exception:
        # Safe pure-numpy nearest-neighbor fallback
        y_idx = (np.linspace(0, b.shape[0]-1, oh)).astype(np.int32)
        x_idx = (np.linspace(0, b.shape[1]-1, ow)).astype(np.int32)
        return b[y_idx][:, x_idx].astype(np.float32, copy=False)


def _to_Luma(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img.astype(np.float32, copy=False)
    # HWC RGB
    r, g, b = img[..., 0].astype(np.float32), img[..., 1].astype(np.float32), img[..., 2].astype(np.float32)
    return 0.2989 * r + 0.5870 * g + 0.1140 * b

def _build_poly_terms_deg2(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    # [1, x, y, x^2, x*y, y^2]
    return np.stack([np.ones_like(x), x, y, x*x, x*y, y*y], axis=1).astype(np.float32, copy=False)

# ----- ABE-like sampling (corners, borders, quartiles with bright-avoid & descent) -----

def _exclude_bright_regions(gray_small: np.ndarray, exclusion_fraction: float = 0.5) -> np.ndarray:
    """
    Returns a boolean mask selecting the dimmer ~exclusion_fraction of pixels.
    Robust to empty arrays and NaNs — never throws.
    """
    a = np.asarray(gray_small, dtype=np.float32)
    if a.size == 0:
        return np.zeros_like(a, dtype=bool)

    # Use nanpercentile; if everything is NaN, allow all (avoid empty elig later)
    q = max(0.0, min(100.0, 100.0 * (1.0 - float(exclusion_fraction))))
    try:
        thresh = np.nanpercentile(a, q)
    except Exception:
        # Fallback: if something went wrong, just allow all
        return np.ones_like(a, dtype=bool)

    mask = a < thresh
    # If mask ends up all False (flat image, or numeric quirks), allow all
    if not np.any(mask):
        mask = np.ones_like(a, dtype=bool)
    return mask


def _gradient_descent_to_dim_spot(gray_small: np.ndarray, x: int, y: int, patch: int) -> tuple[int, int]:
    H, W = gray_small.shape[:2]
    half = patch // 2
    def patch_median(px, py):
        x0, x1 = max(0, px - half), min(W, px + half + 1)
        y0, y1 = max(0, py - half), min(H, py + half + 1)
        return float(np.median(gray_small[y0:y1, x0:x1]))
    cx, cy = int(np.clip(x, 0, W-1)), int(np.clip(y, 0, H-1))
    for _ in range(60):
        cur = patch_median(cx, cy)
        best = (cx, cy); best_val = cur
        for nx in (cx-1, cx, cx+1):
            for ny in (cy-1, cy, cy+1):
                if nx == cx and ny == cy: continue
                if nx < 0 or ny < 0 or nx >= W or ny >= H: continue
                val = patch_median(nx, ny)
                if val < best_val:
                    best_val = val; best = (nx, ny)
        if best == (cx, cy):
            break
        cx, cy = best
    return cx, cy

def _generate_sample_points_small(
    img_small: np.ndarray,
    num_samples: int,
    patch_size: int,
    exclusion_mask_small: np.ndarray | None
) -> np.ndarray:
    H, W = img_small.shape[:2]
    gray = _to_Luma(img_small) if img_small.ndim == 3 else img_small
    pts: list[tuple[int, int]] = []

    # If the downsampled image is extremely small, fallback to a minimal grid
    if H < max(3, patch_size*2+1) or W < max(3, patch_size*2+1):
        g = max(2, min(4, int(np.sqrt(max(1, num_samples)))))
        xs = np.linspace(0, max(W-1, 0), g, dtype=int)
        ys = np.linspace(0, max(H-1, 0), g, dtype=int)
        for y in ys:
            for x in xs:
                pts.append((int(x), int(y)))
        return np.asarray(pts, dtype=np.int32)

    border = max(6, patch_size)

    def allowed(x, y):
        if exclusion_mask_small is None: 
            return True
        yy = int(np.clip(y, 0, H-1)); xx = int(np.clip(x, 0, W-1))
        return bool(exclusion_mask_small[yy, xx])

    # corners (guard each)
    for (x, y) in [(border, border), (W-border-1, border), (border, H-border-1), (W-border-1, H-border-1)]:
        if 0 <= x < W and 0 <= y < H and allowed(x, y):
            nx, ny = _gradient_descent_to_dim_spot(gray, x, y, patch_size)
            if allowed(nx, ny):
                pts.append((nx, ny))

    # borders (guard each)
    xs = np.linspace(border, max(W-border-1, border), 5, dtype=int)
    ys = np.linspace(border, max(H-border-1, border), 5, dtype=int)
    xs = np.unique(xs); ys = np.unique(ys)

    for x in xs:
        if 0 <= x < W and 0 <= border < H and allowed(x, border):
            nx, ny = _gradient_descent_to_dim_spot(gray, x, border, patch_size)
            if allowed(nx, ny): pts.append((nx, ny))
        if 0 <= x < W and 0 <= (H-border-1) < H and allowed(x, H-border-1):
            nx, ny = _gradient_descent_to_dim_spot(gray, x, H-border-1, patch_size)
            if allowed(nx, ny): pts.append((nx, ny))
    for y in ys:
        if 0 <= border < W and 0 <= y < H and allowed(border, y):
            nx, ny = _gradient_descent_to_dim_spot(gray, border, y, patch_size)
            if allowed(nx, ny): pts.append((nx, ny))
        if 0 <= (W-border-1) < W and 0 <= y < H and allowed(W-border-1, y):
            nx, ny = _gradient_descent_to_dim_spot(gray, W-border-1, y, patch_size)
            if allowed(nx, ny): pts.append((nx, ny))

    # quartiles: guard empty slices and skip those that collapse
    hh, ww = H // 2, W // 2
    quads = [
        (slice(0, hh),    slice(0, ww),    (0, 0)),
        (slice(0, hh),    slice(ww, W),    (ww, 0)),
        (slice(hh, H),    slice(0, ww),    (0, hh)),
        (slice(hh, H),    slice(ww, W),    (ww, hh)),
    ]
    per_quad = max(1, num_samples // 4)

    for ysl, xsl, (x0, y0) in quads:
        sub = gray[ysl, xsl]
        if sub.size == 0:
            continue
        mask_sub = _exclude_bright_regions(sub, exclusion_fraction=0.5)
        if exclusion_mask_small is not None:
            em = exclusion_mask_small[ysl, xsl]
            if em.size == mask_sub.size:
                mask_sub = mask_sub & em
        elig = np.argwhere(mask_sub)
        if elig.size == 0:
            continue
        k = min(len(elig), per_quad)
        sel = elig[np.random.choice(len(elig), k, replace=False)]
        for (yy, xx) in sel:
            gx, gy = x0 + int(xx), y0 + int(yy)
            if 0 <= gx < W and 0 <= gy < H and allowed(gx, gy):
                nx, ny = _gradient_descent_to_dim_spot(gray, gx, gy, patch_size)
                if allowed(nx, ny):
                    pts.append((nx, ny))

    # Absolute fallback if pts stayed empty (degenerate case): small grid
    if not pts:
        g = max(3, int(np.sqrt(max(16, num_samples))))
        xs = np.linspace(border, W-border-1, g, dtype=int)
        ys = np.linspace(border, H-border-1, g, dtype=int)
        for y in ys:
            for x in xs:
                if 0 <= x < W and 0 <= y < H and allowed(x, y):
                    pts.append((int(x), int(y)))

    return np.asarray(pts, dtype=np.int32)


# ----- fit/eval on small image -----

def _fit_poly2_on_small(img_small: np.ndarray, pts_small: np.ndarray, patch_size: int) -> np.ndarray:
    """Fit degree-2 polynomial on Luma of small image using patch medians at pts."""
    gray = _to_Luma(img_small) if img_small.ndim == 3 else img_small
    Hs, Ws = gray.shape[:2]
    half = patch_size // 2

    xs = np.clip(pts_small[:, 0], 0, Ws-1).astype(np.int32)
    ys = np.clip(pts_small[:, 1], 0, Hs-1).astype(np.int32)

    z = np.empty(xs.shape[0], dtype=np.float32)
    for i, (x, y) in enumerate(zip(xs, ys)):
        x0, x1 = max(0, x - half), min(Ws, x + half + 1)
        y0, y1 = max(0, y - half), min(Hs, y + half + 1)
        z[i] = float(np.median(gray[y0:y1, x0:x1]))

    A = _build_poly_terms_deg2(xs.astype(np.float32), ys.astype(np.float32))
    coef, *_ = np.linalg.lstsq(A, z, rcond=None)

    # evaluate on full small grid
    yy, xx = np.meshgrid(np.arange(Hs, dtype=np.float32),
                         np.arange(Ws, dtype=np.float32), indexing='ij')
    bg_small = (coef[0] + coef[1]*xx + coef[2]*yy + coef[3]*xx*xx + coef[4]*xx*yy + coef[5]*yy*yy).astype(np.float32)
    return bg_small

# ----- public API -----

def remove_poly2_gradient_abe(
    image: np.ndarray,
    *,
    mode: str = "subtract",
    num_samples: int = 120,
    downsample: int = 6,
    patch_size: int = 15,
    min_strength: float = 0.01,
    gain_clip: tuple[float,float] = (0.2, 5.0),
    exclusion_mask: np.ndarray | None = None,
    log_fn=None
) -> np.ndarray:
    if image is None:
        return image

    img = _asarray(image, dtype=np.float32)

    # ---- Detect original layout
    is_2d  = (img.ndim == 2)
    is_hwc = (img.ndim == 3 and img.shape[-1] in (1, 3))
    is_chw = (img.ndim == 3 and img.shape[0]  in (1, 3) and not is_hwc)

    # ---- Convert to HWC "work" view (internal processing)
    if is_2d:
        work = img
    elif is_hwc:
        work = img
    elif is_chw:
        work = np.moveaxis(img, 0, -1)  # CHW -> HWC
    else:
        # Unexpected layout; treat as 2D luma via mean over last axis
        work = img.mean(axis=-1).astype(np.float32, copy=False)

    H, W = work.shape[:2]

    # --- Downsample image & optional mask
    img_small = _downsample_area(work, max(1, int(downsample)))
    mask_small = None
    if exclusion_mask is not None:
        em  = _asarray(exclusion_mask, dtype=np.float32)
        mask_small = _downsample_area(em, max(1, int(downsample))) >= 0.5

    # --- Sample & fit
    pts_small = _generate_sample_points_small(
        img_small, num_samples=int(num_samples),
        patch_size=int(patch_size),
        exclusion_mask_small=mask_small
    )
    bg_small = _fit_poly2_on_small(img_small, pts_small, patch_size=int(patch_size))
    bg = _upscale_bg(bg_small, H, W)

    # --- Strength check
    bg_med = float(np.nanmedian(bg)) or 1e-6
    p5, p95 = np.nanpercentile(bg, 5), np.nanpercentile(bg, 95)
    rel_amp = float((p95 - p5) / max(bg_med, 1e-6))
    if log_fn:
        log_fn(f"ABE poly2: samples={num_samples}, ds={downsample}, patch={patch_size} | "
               f"bg_med={bg_med:.6f}, rel_amp={rel_amp*100:.2f}%")
    QApplication.processEvents()    
    if rel_amp < float(min_strength):
        # Return original image in original layout
        return img

    # --- Apply (luma-only fit, channel-consistent apply)
    def _apply_sub(ch):  # re-center to preserve median
        med0 = float(np.nanmedian(ch)) or 1e-6
        out = ch - bg
        med1 = float(np.nanmedian(out)) or 1e-6
        out += (med0 - med1)
        return out

    def _apply_div(ch):
        med0 = float(np.nanmedian(ch)) or 1e-6
        norm_bg = np.clip(bg / bg_med, gain_clip[0], gain_clip[1])
        out = ch / norm_bg
        med1 = float(np.nanmedian(out)) or 1e-6
        out *= (med0 / med1)
        return out

    if work.ndim == 2:
        ch = work
        out_work = _apply_sub(ch) if mode.lower() == "subtract" else _apply_div(ch)
    else:
        # HWC
        if mode.lower() == "subtract":
            r = _apply_sub(work[..., 0])
            g = _apply_sub(work[..., 1]) if work.shape[-1] > 1 else r
            b = _apply_sub(work[..., 2]) if work.shape[-1] > 2 else r
        else:
            r = _apply_div(work[..., 0])
            g = _apply_div(work[..., 1]) if work.shape[-1] > 1 else r
            b = _apply_div(work[..., 2]) if work.shape[-1] > 2 else r
        out_work = np.stack([r, g, b], axis=-1) if work.shape[-1] == 3 else r[..., None]

    out_work = out_work.astype(np.float32, copy=False)

    # ---- Convert back to original layout
    if is_2d:
        return out_work
    if is_hwc:
        return out_work
    if is_chw:
        return np.moveaxis(out_work, -1, 0).astype(np.float32, copy=False)
    # Fallback: shape-preserving best effort
    return out_work


def _ensure_like(x: np.ndarray, like: np.ndarray) -> np.ndarray:
    """Return x with the same layout/shape as like (2D, HWC, or CHW)."""
    if x.shape == like.shape:
        return x

    # 2D cases
    if like.ndim == 2:
        if x.ndim == 3 and x.shape[-1] in (1,3):  # HWC -> 2D (luma)
            return (0.2126*x[...,0] + 0.7152*x[...,1] + 0.0722*x[...,2]).astype(np.float32, copy=False) if x.shape[-1]==3 else x[...,0]
        if x.ndim == 3 and x.shape[0] in (1,3):   # CHW -> 2D (first/mean)
            return x[0]
        return x

    # HWC target
    if like.ndim == 3 and like.shape[-1] in (1,3):
        if x.ndim == 3 and x.shape[0] in (1,3):   # CHW -> HWC
            return np.moveaxis(x, 0, -1).astype(np.float32, copy=False)
        if x.ndim == 2 and like.shape[-1] == 3:   # 2D -> HWC repeat
            return np.repeat(x[..., None], 3, axis=-1).astype(np.float32, copy=False)
        if x.ndim == 2 and like.shape[-1] == 1:
            return x[..., None].astype(np.float32, copy=False)
        return x

    # CHW target
    if like.ndim == 3 and like.shape[0] in (1,3):
        if x.ndim == 3 and x.shape[-1] in (1,3):  # HWC -> CHW
            return np.moveaxis(x, -1, 0).astype(np.float32, copy=False)
        if x.ndim == 2 and like.shape[0] == 3:    # 2D -> CHW repeat
            return np.repeat(x[None, ...], 3, axis=0).astype(np.float32, copy=False)
        if x.ndim == 2 and like.shape[0] == 1:
            return x[None, ...].astype(np.float32, copy=False)
        return x

    return x

def remove_gradient_stack_abe(stack, target_hw: tuple[int,int] | None = None, **kw):
    """
    stack: (N,H,W) or (N,H,W,C) or (N,C,H,W)
    Returns the same layout as 'stack'. If target_hw=(H,W) is provided,
    every output is forced to exactly (H,W) via center-crop / reflect-pad.
    """

    # ---- local helper: force exact (H,W) via center-crop or reflect-pad ----
    def _force_shape_hw(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        import numpy as np
        import cv2
        a = np.asarray(img)
        if a.ndim < 2:
            return a

        # detect layout
        if a.ndim == 2:
            H, W = a.shape; Caxis = None
        elif a.ndim == 3:
            if a.shape[-1] in (1, 3):      # HWC
                H, W = a.shape[:2]; Caxis = -1
            elif a.shape[0] in (1, 3):     # CHW
                H, W = a.shape[1:]; Caxis = 0
            else:
                H, W = a.shape[:2]; Caxis = -1
        else:
            return a

        th, tw = int(target_h), int(target_w)
        if (H, W) == (th, tw):
            return a

        # center-crop if larger
        y0 = max(0, (H - th) // 2); x0 = max(0, (W - tw) // 2)
        y1 = min(H, y0 + th);       x1 = min(W, x0 + tw)

        if a.ndim == 2:
            cropped = a[y0:y1, x0:x1]
        elif Caxis == -1:
            cropped = a[y0:y1, x0:x1, ...]
        else:
            cropped = a[:, y0:y1, x0:x1]

        ch, cw = (cropped.shape[:2] if (cropped.ndim == 2 or Caxis == -1) else cropped.shape[1:3])

        # reflect-pad if smaller
        pad_t = max(0, (th - ch) // 2)
        pad_b = max(0, th - ch - pad_t)
        pad_l = max(0, (tw - cw) // 2)
        pad_r = max(0, tw - cw - pad_l)

        if pad_t or pad_b or pad_l or pad_r:
            if cropped.ndim == 2:
                cropped = cv2.copyMakeBorder(cropped, pad_t, pad_b, pad_l, pad_r,
                                             borderType=cv2.BORDER_REFLECT_101)
            elif Caxis == -1:
                cropped = cv2.copyMakeBorder(cropped, pad_t, pad_b, pad_l, pad_r,
                                             borderType=cv2.BORDER_REFLECT_101)
            else:
                chans = []
                for c in range(cropped.shape[0]):
                    chans.append(cv2.copyMakeBorder(cropped[c], pad_t, pad_b, pad_l, pad_r,
                                                    borderType=cv2.BORDER_REFLECT_101))
                cropped = np.stack(chans, axis=0)

        return cropped

    arr = _asarray(stack, dtype=np.float32)
    N = arr.shape[0]
    out = np.empty_like(arr)

    # Use first frame as layout reference (channels/order), not necessarily size
    ref = arr[0]

    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as ex:
        futs = {}
        for i in range(N):
            img = arr[i]
            futs[ex.submit(remove_poly2_gradient_abe, img, **kw)] = i

        for fut in as_completed(futs):
            i_ = futs[fut]
            res = fut.result()
            res = _ensure_like(res, ref)   # normalize layout (2D/HWC/CHW)

            # 🔒 size lock: prefer caller-provided canonical size
            if target_hw is not None:
                th, tw = int(target_hw[0]), int(target_hw[1])
                res = _force_shape_hw(res, th, tw)
            else:
                # legacy behavior: conform to ref's current size
                if res.shape != ref.shape:
                    if ref.ndim == 2:
                        res = _force_shape_hw(res, ref.shape[0], ref.shape[1])
                    elif ref.ndim == 3 and ref.shape[-1] in (1, 3):   # HWC
                        res_h, res_w = res.shape[:2]
                        res = _force_shape_hw(res, ref.shape[0], ref.shape[1])
                    elif ref.ndim == 3 and ref.shape[0] in (1, 3):     # CHW
                        res_h, res_w = res.shape[1:3]
                        res = _force_shape_hw(res, ref.shape[1], ref.shape[2])
                    # else: leave as-is (best effort)

            out[i_] = res.astype(out.dtype, copy=False)

    return out

def load_fits_tile(filepath, y_start, y_end, x_start, x_end):
    """
    Loads a sub-region from a FITS file, including:
      - normal FITS
      - .fits.gz / .fit.gz
      - fpack / Rice tile-compressed FITS stored as plain .fits (CompImageHDU in ext 1)

    Uses get_valid_header() to choose the correct HDU and copy any BAYERPAT, etc.

    Returns:
      - (th, tw) mono tile
      - (th, tw, 3) color tile (or CHW if that’s how it lives on disk)
    """
    import numpy as np
    import gzip
    from astropy.io import fits
    from io import BytesIO

    # Find the "intended" HDU index (may be wrong for this reopen; we re-check below)
    hdr, ext_index = get_valid_header(filepath)

    # Open appropriately for gzip-compressed files
    if filepath.lower().endswith((".fits.gz", ".fit.gz")):
        with gzip.open(filepath, "rb") as f:
            file_content = f.read()
        hdul = fits.open(BytesIO(file_content), memmap=False)
    else:
        # memmap=False is IMPORTANT for CompImageHDU / tile-compressed
        hdul = fits.open(filepath, memmap=False)

    with hdul as H:
        # ---- pick a safe HDU index in THIS open ----
        ei = ext_index if isinstance(ext_index, int) else None
        data = None

        if ei is not None and 0 <= ei < len(H):
            try:
                data = H[ei].data
            except Exception:
                data = None

        # Fallback: rescan this HDUList for first image HDU
        if data is None:
            ei = None
            for i, hdu in enumerate(H):
                try:
                    if hdu.data is not None:
                        data = hdu.data
                        ei = i
                        break
                except Exception:
                    continue

        if data is None:
            return None

        # Ensure native byte order
        if data.dtype.byteorder not in ("=", "|"):
            data = data.astype(data.dtype.newbyteorder("="), copy=False)

        orig_dtype = data.dtype

        # Prefer scaling keywords from the actual image HDU, fall back to composite hdr
        try:
            img_hdr = H[ei].header if ei is not None else hdr
        except Exception:
            img_hdr = hdr
        bzero  = float(img_hdr.get("BZERO", hdr.get("BZERO", 0.0)))
        bscale = float(img_hdr.get("BSCALE", hdr.get("BSCALE", 1.0)))

        a = np.asarray(data)
        a = np.squeeze(a)  # match load_image behavior

        shape = a.shape
        ndim  = a.ndim

        # ---------------------------
        # Slice spatial region
        # ---------------------------
        if ndim == 2:
            dim0, dim1 = shape
            if (y_end <= dim0) and (x_end <= dim1):
                tile_data = a[y_start:y_end, x_start:x_end]
            else:
                tile_data = a[x_start:x_end, y_start:y_end]

        elif ndim == 3:
            dim0, dim1, dim2 = shape

            def do_slice_spatial(data3d, spat0, spat1):
                slicer = [slice(None)] * 3
                slicer[spat0] = slice(y_start, y_end)
                slicer[spat1] = slice(x_start, x_end)
                return data3d[tuple(slicer)]

            # Identify color axis (size 3); others are spatial
            color_axis = None
            spat_axes = []
            for idx, d in enumerate((dim0, dim1, dim2)):
                if d == 3 and color_axis is None:
                    color_axis = idx
                else:
                    spat_axes.append(idx)

            if color_axis is None or len(spat_axes) != 2:
                tile_data = a[y_start:y_end, x_start:x_end]
            else:
                spat0, spat1 = spat_axes
                d0 = shape[spat0]
                d1 = shape[spat1]
                if (y_end <= d0) and (x_end <= d1):
                    tile_data = do_slice_spatial(a, spat0, spat1)
                else:
                    tile_data = do_slice_spatial(a, spat1, spat0)

        else:
            return None

        # ---------------------------
        # Scale/normalize like load_image
        # ---------------------------
        if orig_dtype == np.uint8:
            tile_data = tile_data.astype(np.float32) / 255.0
        elif orig_dtype == np.uint16:
            tile_data = tile_data.astype(np.float32) / 65535.0
        elif orig_dtype == np.int32:
            tile_data = tile_data.astype(np.float32) * bscale + bzero
        elif orig_dtype == np.uint32:
            tile_data = tile_data.astype(np.float32) * bscale + bzero
        elif orig_dtype == np.float32:
            tile_data = np.asarray(tile_data, dtype=np.float32, order="C")
        else:
            tile_data = tile_data.astype(np.float32)

        return tile_data



def _get_log_dock():
    app = QApplication.instance()
    return getattr(app, "_sasd_status_console", None)  # set by main window

class _MMFits:
    """
    Keeps one FITS open (memmap=True) for the whole group; slices tiles without
    re-opening the file. Handles spatial/color axis detection once.
    """
    def __init__(self, path: str):
        self.path = path
        # Keep file open for the whole integration of the group
        self.hdul = fits.open(path, memmap=True)
        self.data = self.hdul[0].data
        if self.data is None:
            raise ValueError(f"Empty FITS: {path}")

        self.shape = self.data.shape
        self.ndim  = self.data.ndim
        self.orig_dtype = self.data.dtype

        # detect color axis (size==3) and spatial axes once
        if self.ndim == 2:
            self.color_axis = None
            self.spat_axes = (0, 1)
        elif self.ndim == 3:
            dims = self.shape
            self.color_axis = next((i for i, d in enumerate(dims) if d == 3), None)
            if self.color_axis is None:
                self.spat_axes = (0, 1)
            else:
                self.spat_axes = tuple(i for i in range(3) if i != self.color_axis)
        else:
            raise ValueError(f"Unsupported ndim={self.ndim} for {path}")

        # scalar normalization chosen once
        if   self.orig_dtype == np.uint8:  self._scale = 1.0/255.0
        elif self.orig_dtype == np.uint16: self._scale = 1.0/65535.0
        else:                              self._scale = None

    def read_tile(self, y0, y1, x0, x1) -> np.ndarray:
        d = self.data
        if self.ndim == 2:
            tile = d[y0:y1, x0:x1]
        else:
            sl = [slice(None)]*3
            sl[self.spat_axes[0]] = slice(y0, y1)
            sl[self.spat_axes[1]] = slice(x0, x1)
            tile = d[tuple(sl)]
            # move color to last if present
            if self.color_axis is not None and self.color_axis != 2:
                tile = np.moveaxis(tile, self.color_axis, -1)

        # late normalize to float32
        if self._scale is None:
            tile = tile.astype(np.float32, copy=False)
        else:
            tile = tile.astype(np.float32, copy=False) * self._scale

        # ensure (h,w,3) or (h,w)
        if tile.ndim == 3 and tile.shape[-1] not in (1,3):
            # uncommon mono-3D (e.g. (3,h,w)): move to (h,w,3)
            if tile.shape[0] == 3 and tile.shape[-1] != 3:
                tile = np.moveaxis(tile, 0, -1)
        return tile

    def read_full(self) -> np.ndarray:
        """
        Return the whole frame as float32 in (H,W) or (H,W,3) with color last,
        normalized the same way as read_tile().
        """
        d = self.data
        if self.ndim == 2:
            full = d
        else:
            # move color to last if present
            full = d
            if self.color_axis is not None and self.color_axis != 2:
                full = np.moveaxis(full, self.color_axis, -1)

        # late normalize to float32
        if self._scale is None:
            full = full.astype(np.float32, copy=False)
        else:
            full = full.astype(np.float32, copy=False) * self._scale

        # ensure (H,W,3) or (H,W)
        if full.ndim == 3 and full.shape[-1] not in (1, 3):
            if full.shape[0] == 3 and full.shape[-1] != 3:
                full = np.moveaxis(full, 0, -1)
        return full

    def close(self):
        try:
            self.hdul.close()
        except Exception:
            pass

# --------------------------------------------------
# Stacking Suite
# --------------------------------------------------
