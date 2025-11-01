#pro.stacking_suite.py
from __future__ import annotations
import os, glob, shutil, tempfile, datetime as _dt
import sys, platform
from time import perf_counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import numpy.ma as ma
import hashlib
from numpy.lib.format import open_memmap 
import tzlocal
import weakref
import re, unicodedata
import math            # used in compute_safe_chunk
import psutil          # used in bytes_available / compute_safe_chunk
from typing import List 
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
from legacy.numba_utils import *   
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

def get_valid_header(path):
    try:
        from astropy.io import fits
        hdr = fits.getheader(path, ext=0)
        return hdr, True
    except Exception:
        return None, False


def _read_tile_stack(file_list, y0, y1, x0, x1, channels, out_buf):
    """
    Fill `out_buf` with the tile stack for (y0:y1, x0:x1).
    out_buf: (N, th, tw, C) float32, C-order (preallocated by caller).
    Returns (th, tw) = actual tile extents.
    """
    th = int(y1 - y0)
    tw = int(x1 - x0)
    N  = len(file_list)

    # Slice to actual extents (edge tiles); zero-fill defensively
    ts = out_buf[:N, :th, :tw, :channels]
    ts[...] = 0.0

    # Helper: coerce any incoming array to float32 HWC with requested channel count
    def _coerce_to_f32_hwc(a: np.ndarray, want_c: int) -> np.ndarray:
        if a is None:
            return None

        # ---- layout normalize to HWC ----
        if a.ndim == 2:
            a = a[:, :, None]  # (H,W) -> (H,W,1)
        elif a.ndim == 3:
            # Heuristics: treat (3,H,W) as CHW and transpose; keep (H,W,3) as is
            if a.shape[0] in (1, 3) and a.shape[1] >= 8 and a.shape[2] >= 8 and a.shape[-1] not in (1, 3):
                # looks like CHW -> HWC
                a = a.transpose(1, 2, 0)
        else:
            # unknown layout; bail to None so caller zeros
            return None

        # ---- crop/guard (should already match) ----
        if a.shape[0] != th or a.shape[1] != tw:
            a = a[:th, :tw, ...]

        # ---- channel reconcile ----
        Csrc = a.shape[2]
        if Csrc == want_c:
            pass
        elif Csrc == 1 and want_c == 3:
            a = np.repeat(a, 3, axis=2)
        elif Csrc == 3 and want_c == 1:
            # For calibration masters, deterministic luminance/mean is fine.
            # (Mean avoids color bias if channels differ a bit.)
            a = a.mean(axis=2, keepdims=True)
        else:
            # Fallback: slice/expand as best-effort
            if Csrc > want_c:
                a = a[:, :, :want_c]
            else:
                a = np.repeat(a, want_c, axis=2)

        # ---- dtype/finite ----
        a = np.asarray(a, dtype=np.float32, order="C")
        if not np.isfinite(a).all():
            np.nan_to_num(a, copy=False, posinf=0.0, neginf=0.0)

        return a

    # Load each frame's tile in parallel (avoid oversubscribing)
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
                # leave zeros for this frame/tile
                continue

            sub = _coerce_to_f32_hwc(sub, channels)
            if sub is None:
                continue

            # Commit (shapes now match by construction)
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

_HDR_CACHE = {}     # path -> fits.Header
_BIN_CACHE = {}     # path -> (xb, yb)

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
        return np.asarray(out, dtype=np.float32, copy=False)
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

    img = np.asarray(image, dtype=np.float32, copy=False)

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
        em = np.asarray(exclusion_mask, dtype=np.float32, copy=False)
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

def remove_gradient_stack_abe(stack, **kw):
    """
    stack: (N,H,W) or (N,H,W,C) or (N,C,H,W)
    Returns the same shape/layout as 'stack'.
    """
    arr = np.asarray(stack, dtype=np.float32, copy=False)
    N = arr.shape[0]
    out = np.empty_like(arr)

    # Use first frame as layout reference
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
            res = _ensure_like(res, ref)   # <<< normalize layout
            # Also guard size mismatch by center-cropping if necessary
            if res.shape != ref.shape:
                def _cc(a, tgt):
                    if a.ndim == 2:
                        h,w = a.shape; ht,wt = tgt.shape
                        y0 = max(0,(h-ht)//2); x0 = max(0,(w-wt)//2)
                        return a[y0:y0+ht, x0:x0+wt]
                    if a.ndim == 3:
                        if tgt.shape[-1] in (1,3) and a.shape[-1] in (1,3):  # HWC
                            h,w,c = a.shape; ht,wt,ct = tgt.shape
                            y0 = max(0,(h-ht)//2); x0 = max(0,(w-wt)//2)
                            a2 = a[y0:y0+ht, x0:x0+wt, :]
                            if c != ct:
                                if ct == 1: a2 = a2[..., :1]
                                else:       a2 = np.repeat(a2[..., :1], ct, axis=-1)
                            return a2
                        if tgt.shape[0] in (1,3) and a.shape[0] in (1,3):   # CHW
                            c,h,w = a.shape; ct,ht,wt = tgt.shape
                            y0 = max(0,(h-ht)//2); x0 = max(0,(w-wt)//2)
                            a2 = a[:, y0:y0+ht, x0:x0+wt]
                            if c != ct:
                                if ct == 1: a2 = a2[:1]
                                else:       a2 = np.repeat(a2[:1], ct, axis=0)
                            return a2
                    return a
                res = _cc(res, ref)

            out[i_] = res.astype(out.dtype, copy=False)

    return out




def load_fits_tile(filepath, y_start, y_end, x_start, x_end):
    """
    Loads a sub-region from a FITS file, detecting which axes are spatial vs. color.
    
    * If the data is 2D, it might be (height, width) or (width, height).
    * If the data is 3D, it might be:
        - (height, width, 3)
        - (3, height, width)
        - (width, height, 3)
        - (3, width, height)
      We only slice the two spatial dimensions; the color axis remains intact.
    
    The returned tile will always have the shape:
      - (tile_height, tile_width) for mono
      - (tile_height, tile_width, 3) for color
    (though the color dimension may still be first if it was first in the file).
    It's up to the caller to reorder if needed.
    """
    with fits.open(filepath, memmap=False) as hdul:
        data = hdul[0].data
        if data is None:
            return None

        # Save the original data type for normalization later.
        orig_dtype = data.dtype

        shape = data.shape
        ndim = data.ndim

        if ndim == 2:
            # Data is 2D; shape could be (height, width) or (width, height)
            dim0, dim1 = shape
            if (y_end <= dim0) and (x_end <= dim1):
                tile_data = data[y_start:y_end, x_start:x_end]
            else:
                tile_data = data[x_start:x_end, y_start:y_end]
        elif ndim == 3:
            # Data is 3D; could be (height, width, 3) or (3, height, width), etc.
            dim0, dim1, dim2 = shape

            def do_slice_spatial(data3d, spat0, spat1, color_axis):
                slicer = [slice(None)] * 3
                slicer[spat0] = slice(y_start, y_end)
                slicer[spat1] = slice(x_start, x_end)
                tile = data3d[tuple(slicer)]
                return tile

            # Identify the color axis (assumed to have size 3)
            color_axis = None
            spat_axes = []
            for idx, d in enumerate((dim0, dim1, dim2)):
                if d == 3:
                    color_axis = idx
                else:
                    spat_axes.append(idx)

            if color_axis is None:
                # No axis with size 3; assume the image is mono and use the first two dims.
                tile_data = data[y_start:y_end, x_start:x_end]
            else:
                # Ensure we have two spatial axes.
                if len(spat_axes) != 2:
                    spat_axes = [0, 1]
                spat0, spat1 = spat_axes
                d0 = shape[spat0]
                d1 = shape[spat1]
                if (y_end <= d0) and (x_end <= d1):
                    tile_data = do_slice_spatial(data, spat0, spat1, color_axis)
                else:
                    tile_data = do_slice_spatial(data, spat1, spat0, color_axis)
        else:
            return None

        # Normalize based on the original data type.
        if orig_dtype == np.uint8:
            tile_data = tile_data.astype(np.float32) / 255.0
        elif orig_dtype == np.uint16:
            tile_data = tile_data.astype(np.float32) / 65535.0
        elif orig_dtype == np.uint32:
            # 32-bit data: convert to float32 but leave values as is.
            tile_data = tile_data.astype(np.float32)
        elif orig_dtype == np.float32:
            # Already 32-bit float; assume it's in the desired range.
            tile_data = tile_data
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
class BatchSettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Set Type, Exposure, and Filter for All Files")

        layout = QVBoxLayout(self)

        # 1) IMAGETYP Combo
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Image Type (IMAGETYP):"))
        self.type_combo = QComboBox()
        self.type_combo.addItems(["LIGHT", "DARK", "FLAT", "BIAS", "UNKNOWN"])
        type_layout.addWidget(self.type_combo)
        layout.addLayout(type_layout)

        # 2) Exposure Time
        exp_layout = QHBoxLayout()
        exp_layout.addWidget(QLabel("Exposure Time (seconds):"))
        self.exptime_edit = QLineEdit()
        self.exptime_edit.setText("Unknown")  # default
        exp_layout.addWidget(self.exptime_edit)
        layout.addLayout(exp_layout)

        # 3) Filter
        filt_layout = QHBoxLayout()
        filt_layout.addWidget(QLabel("Filter:"))
        self.filter_edit = QLineEdit()
        self.filter_edit.setText("None")  # default
        filt_layout.addWidget(self.filter_edit)
        layout.addLayout(filt_layout)

        # Buttons
        btn_layout = QHBoxLayout()
        ok_btn = QPushButton("OK")
        cancel_btn = QPushButton("Cancel")
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

        ok_btn.clicked.connect(self.accept)
        cancel_btn.clicked.connect(self.reject)

        # Final layout
        self.setLayout(layout)

    def get_values(self):
        """
        Returns (imagetyp, exptime_str, filter_str)
        after the dialog is accepted.
        """
        return (
            self.type_combo.currentText(),
            self.exptime_edit.text(),
            self.filter_edit.text()
        )

class ReferenceFrameReviewDialog(QDialog):
    def __init__(self, ref_frame_path, stats, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Reference Frame Review")
        self.ref_frame_path = ref_frame_path
        self.stats = stats  # e.g., {"star_count": 250, "eccentricity": 0.12, "mean": 0.45}
        self.autostretch_enabled = False
        self.original_image = None  # Will store the loaded image array
        self.target_median = self.stats.get("mean", 0.25)
        self.user_choice = None  # Will be set to 'use' or 'select_other'
        self.zoom_factor = 1.0
        self.current_preview_image = None  # Store the image array currently shown in preview

        # For panning functionality
        self._panning = False
        self._last_mouse_pos = QPoint()

        self.initUI()
        self.loadImageArray()  # Load the image into self.original_image
        if self.original_image is not None:
            self.updatePreview(self.original_image)  # Ensure the first image is shown
        if self.original_image is not None:
            QTimer.singleShot(0, self.zoomIn)            


    def initUI(self):
        main_layout = QVBoxLayout(self)
        
        # Create a scroll area for the preview image
        self.scrollArea = QScrollArea(self)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setMinimumSize(QSize(600, 400))
        self.previewLabel = QLabel("Reference Preview", self)
        self.previewLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scrollArea.setWidget(self.previewLabel)
        main_layout.addWidget(self.scrollArea)
        self.scrollArea.viewport().installEventFilter(self)
        
        # Zoom control buttons
        zoom_layout = QHBoxLayout()
        self.zoomInButton = QPushButton("Zoom In", self)
        self.zoomInButton.clicked.connect(self.zoomIn)
        zoom_layout.addWidget(self.zoomInButton)
        self.zoomOutButton = QPushButton("Zoom Out", self)
        self.zoomOutButton.clicked.connect(self.zoomOut)
        zoom_layout.addWidget(self.zoomOutButton)
        main_layout.addLayout(zoom_layout)
        
        # Stats display
        stats_text = (
            f"Star Count: {self.stats.get('star_count', 'N/A')}\n"
            f"Eccentricity: {self.stats.get('eccentricity', 'N/A'):.4f}\n"
            f"Mean: {self.stats.get('mean', 'N/A'):.4f}"
        )
        self.statsLabel = QLabel(stats_text, self)
        main_layout.addWidget(self.statsLabel)
        
        # Buttons layout for reference selection and autostretch toggle
        button_layout = QHBoxLayout()
        self.toggleAutoStretchButton = QPushButton("Enable Autostretch", self)
        self.toggleAutoStretchButton.clicked.connect(self.toggleAutostretch)
        button_layout.addWidget(self.toggleAutoStretchButton)
        
        # New button to let the user select a new reference frame file
        self.selectNewRefButton = QPushButton("Select New Reference Frame", self)
        self.selectNewRefButton.clicked.connect(self.selectNewReferenceFrame)
        button_layout.addWidget(self.selectNewRefButton)
        
        self.useRefButton = QPushButton("Use This Reference Frame", self)
        self.useRefButton.clicked.connect(self.useReference)
        button_layout.addWidget(self.useRefButton)
        
        self.selectOtherButton = QPushButton("Cancel", self)
        self.selectOtherButton.clicked.connect(self.reject)
        button_layout.addWidget(self.selectOtherButton)
        
        main_layout.addLayout(button_layout)
        self.setLayout(main_layout)
        self.zoomIn()
    
    def fitToPreview(self):
        """Calculate and set the zoom factor so that the image fills the preview area."""
        if self.original_image is None:
            return
        # Get the available size from the scroll area's viewport.
        available_size = self.scrollArea.viewport().size()
        # Determine the original image dimensions.
        if self.original_image.ndim == 2:
            orig_height, orig_width = self.original_image.shape
        elif self.original_image.ndim == 3:
            orig_height, orig_width = self.original_image.shape[:2]
        else:
            return
        # Calculate the zoom factor that will allow the image to fit.
        factor = min(available_size.width() / orig_width,
                    available_size.height() / orig_height)
        self.zoom_factor = factor
        # Choose the current preview image if available, otherwise use the original image.
        if self.current_preview_image is not None:
            image = self.current_preview_image
        else:
            image = self.original_image
        self.updatePreview(image)



    def loadImageArray(self):
        """Load image for preview; ensure float32 in [0,1] and HxWx{1,3}."""
        image_data, header, _, _ = load_image(self.ref_frame_path)
        if image_data is None:
            QMessageBox.critical(self, "Error", "Failed to load the reference image.")
            return

        # If CHW (3,H,W), convert to HWC for preview
        if image_data.ndim == 3 and image_data.shape[0] == 3 and image_data.shape[-1] != 3:
            image_data = np.transpose(image_data, (1, 2, 0))  # CHW -> HWC

        # Squeeze last singleton channel
        if image_data.ndim == 3 and image_data.shape[-1] == 1:
            image_data = np.squeeze(image_data, axis=-1)

        img = image_data.astype(np.float32, copy=False)

        # Preview-normalize: if not already ~[0,1], bring it into [0,1]
        mn = float(np.nanmin(img)); mx = float(np.nanmax(img))
        if not np.isfinite(mn) or not np.isfinite(mx):
            img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
            mn = float(img.min()); mx = float(img.max())

        if mx > 1.0 or mn < 0.0:
            ptp = mx - mn
            img = (img - mn) / ptp if ptp > 0.0 else np.zeros_like(img, dtype=np.float32)

        self.original_image = np.clip(img, 0.0, 1.0)

    
    def updatePreview(self, image):
        """
        Convert a given image array to a QPixmap and update the preview label.
        """
        self.current_preview_image = image
        pixmap = self.convertArrayToPixmap(image)
        if pixmap is None or pixmap.isNull():
            self.previewLabel.setText("Unable to load preview.")
        else:
            available_size = self.scrollArea.viewport().size()
            new_size = QSize(int(available_size.width() * self.zoom_factor),
                             int(available_size.height() * self.zoom_factor))
            scaled_pixmap = pixmap.scaled(new_size, Qt.AspectRatioMode.KeepAspectRatio,
                                          Qt.TransformationMode.SmoothTransformation)
            self.previewLabel.setPixmap(scaled_pixmap)
    
    def _preview_boost(self, img: np.ndarray) -> np.ndarray:
        """Robust, very gentle stretch for display when image would quantize to black."""
        # Use your implemented siril_style_autostretch
        try:
            out = siril_style_autostretch(img, sigma=3.0).astype(np.float32, copy=False)
            mx = float(out.max())
            if mx > 0: out /= mx  # keep in [0,1]
            return np.clip(out, 0.0, 1.0)
        except Exception:
            return np.clip(img, 0.0, 1.0)

    def convertArrayToPixmap(self, image):
        if image is None:
            return None

        img = image.astype(np.float32, copy=False)

        # If image is so dim or flat that 8-bit will zero-out, boost for preview
        ptp = float(img.max() - img.min())
        needs_boost = (float(img.max()) <= (1.0 / 255.0)) or (ptp < 1e-6) or (not np.isfinite(img).all())
        if needs_boost:
            img = self._preview_boost(np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0))

        # Convert to 8-bit for QImage
        display_image = (img * 255.0).clip(0, 255).astype(np.uint8)

        if display_image.ndim == 2:
            h, w = display_image.shape
            q_image = QImage(display_image.data, w, h, w, QImage.Format.Format_Grayscale8)
        elif display_image.ndim == 3 and display_image.shape[2] == 3:
            h, w, _ = display_image.shape
            q_image = QImage(display_image.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        else:
            return None
        return QPixmap.fromImage(q_image)
    
    def toggleAutostretch(self):
        if self.original_image is None:
            QMessageBox.warning(self, "Error", "Reference image not loaded.")
            return
        self.autostretch_enabled = not self.autostretch_enabled
        if self.autostretch_enabled:
            if self.original_image.ndim == 2:
                new_image = stretch_mono_image(self.original_image, target_median=0.3,
                                               normalize=True, apply_curves=False)
            elif self.original_image.ndim == 3 and self.original_image.shape[2] == 3:
                new_image = stretch_color_image(self.original_image, target_median=0.3,
                                                linked=False, normalize=True, apply_curves=False)
            else:
                new_image = self.original_image
            self.toggleAutoStretchButton.setText("Disable Autostretch")
        else:
            new_image = self.original_image
            self.toggleAutoStretchButton.setText("Enable Autostretch")
        self.updatePreview(new_image)
    
    def zoomIn(self):
        self.zoom_factor *= 1.2
        if self.current_preview_image is not None:
            self.updatePreview(self.current_preview_image)
    
    def zoomOut(self):
        self.zoom_factor /= 1.2
        if self.current_preview_image is not None:
            self.updatePreview(self.current_preview_image)
    
    def eventFilter(self, source, event):
        if source is self.scrollArea.viewport():
            if event.type() == QEvent.Type.Wheel:
                if event.angleDelta().y() > 0:
                    self.zoomIn()
                else:
                    self.zoomOut()
                return True
            if event.type() == QEvent.Type.MouseButtonPress:
                if event.button() == Qt.MouseButton.LeftButton:
                    self._panning = True
                    self._last_mouse_pos = event.pos()
                    self.scrollArea.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
                    return True
            if event.type() == QEvent.Type.MouseMove:
                if self._panning:
                    delta = event.pos() - self._last_mouse_pos
                    self._last_mouse_pos = event.pos()
                    h_bar = self.scrollArea.horizontalScrollBar()
                    v_bar = self.scrollArea.verticalScrollBar()
                    h_bar.setValue(h_bar.value() - delta.x())
                    v_bar.setValue(v_bar.value() - delta.y())
                    return True
            if event.type() == QEvent.Type.MouseButtonRelease:
                if event.button() == Qt.MouseButton.LeftButton:
                    self._panning = False
                    self.scrollArea.viewport().setCursor(Qt.CursorShape.ArrowCursor)
                    return True
        return super().eventFilter(source, event)
    
    def resizeEvent(self, event):
        if self.current_preview_image is not None:
            self.updatePreview(self.current_preview_image)
        super().resizeEvent(event)
    
    def selectNewReferenceFrame(self):
        """Open a file dialog to select a new reference frame, update preview accordingly."""
        new_file, _ = QFileDialog.getOpenFileName(
            self,
            "Select New Reference Frame",
            "",
            "FITS Files (*.fits *.fit);;All Files (*)"
        )
        if new_file:
            self.ref_frame_path = new_file
            self.loadImageArray()          # Reload the new image
            self.updatePreview(self.original_image)  # Update the preview
            # Optionally, you could also update stats if needed.
    
    def useReference(self):
        self.user_choice = "use"
        self.accept()
    
    def selectOtherReference(self):
        self.user_choice = "select_other"
        self.reject()
    
    def getUserChoice(self):
        return self.user_choice

def bytes_available():
    vm = psutil.virtual_memory()
    # Keep a safety margin (e.g. leave 10% free)
    return int(vm.available * 0.9)


def compute_safe_chunk(height, width, N, channels, dtype, pref_h, pref_w):
    vm    = psutil.virtual_memory()
    avail = vm.free * 0.9
    bpe64 = np.dtype(dtype).itemsize      # 8 bytes
    workers = os.cpu_count() or 1

    # budget *all* float64 copies (master + per-thread)
    bytes_per_pixel = (N + workers) * channels * bpe64 / 2
    max_pixels      = int(avail // bytes_per_pixel)
    if max_pixels < 1:
        raise MemoryError("Not enough RAM for even a 1×1 tile")

    raw_side = int(math.sqrt(max_pixels))
    # **shrink by √workers to be super-safe**
    fudge    = int(math.sqrt(workers)) or 1
    safe_side = max(1, raw_side // fudge)

    # clamp to user prefs and image dims
    ch = min(pref_h, height, safe_side)
    cw = min(pref_w, width,  safe_side)

    # final area clamp
    if ch * cw > max_pixels // fudge**2:
        # extra safety: adjust cw so area ≤ max_pixels/fudge²
        cw = max(1, (max_pixels // (fudge**2)) // ch)

    if ch < 1 or cw < 1:
        raise MemoryError(f"Chunk too small after fudge: {ch}×{cw}")

    print(f"[DEBUG] raw_side={raw_side}, workers={workers} ⇒ safe_side={safe_side}")
    print(f"[DEBUG] final chunk: {ch}×{cw}")
    return ch, cw

_DIM_RE = re.compile(r"\s*\(\d+\s*x\s*\d+\)\s*")

class _Responder(QObject):
    finished = pyqtSignal(object)   # emits the edited dict or None

class AfterAlignWorker(QObject):
    progress = pyqtSignal(str)                 # emits status lines
    finished = pyqtSignal(bool, str)           # (success, message)
    need_comet_review = pyqtSignal(list, dict, object)  # (files, initial_xy, responder)

    def __init__(self, dialog, *,
                 light_files,
                 frame_weights,
                 transforms_dict,
                 drizzle_dict,
                 autocrop_enabled,
                 autocrop_pct, ui_owner=None):
        super().__init__()
        self.dialog = dialog                    # we will call pure methods on it
        self.light_files = light_files
        self.frame_weights = frame_weights
        self.transforms_dict = transforms_dict
        self.drizzle_dict = drizzle_dict
        self.autocrop_enabled = autocrop_enabled
        self.autocrop_pct = autocrop_pct
        self.ui_owner         = ui_owner  

    @pyqtSlot()
    def run(self):
        dlg = self.dialog  # the StackingSuiteDialog you passed in

        try:
            result = dlg.stack_images_mixed_drizzle(
                grouped_files=self.light_files,
                frame_weights=self.frame_weights,
                transforms_dict=self.transforms_dict,
                drizzle_dict=self.drizzle_dict,
                autocrop_enabled=self.autocrop_enabled,
                autocrop_pct=self.autocrop_pct,
                status_cb=self.progress.emit,   # stream status back to UI
            )
            summary = "\n".join(result["summary_lines"])
            self.finished.emit(True, f"Post-alignment complete.\n\n{summary}")
        except Exception as e:
            self.finished.emit(False, f"Post-alignment failed: {e}")

class StatusLogWindow(QDialog):
    MAX_BLOCKS = 2000

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Stacking Suite Log")

        # ── key flags ─────────────────────────────────────────────
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)   # hide, don't delete
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setWindowFlag(Qt.WindowType.Tool, True)                    # tool window (no taskbar)
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, False)   # ❗ not global topmost
        self.setWindowModality(Qt.WindowModality.NonModal)              # don't block UI
        self._was_visible_on_deactivate = False   
        # ─────────────────────────────────────────────────────────
        # follow app activation/deactivation
        QApplication.instance().applicationStateChanged.connect(self._on_app_state_changed)

        # watch the parent to keep the log above it while the app is active
        #if parent is not None:
        #    parent.installEventFilter(self)

        self.resize(800, 250)

        lay = QVBoxLayout(self)
        self.view = QPlainTextEdit(self)
        self.view.setReadOnly(True)
        self.view.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.view.setStyleSheet(
            "background-color: black; color: white; font-family: Monospace; padding: 6px;"
        )
        lay.addWidget(self.view)

        row = QHBoxLayout()
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.view.clear)
        row.addWidget(self.clear_btn)
        row.addStretch(1)
        lay.addLayout(row)


    def _apply_topmost(self, enable: bool):
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, enable)
        # Re-apply the native flags and stacking
        if enable:
            # When re-activating, make sure it’s shown even if the OS hid it
            self.show()               # ← **always** show on enable
            self.raise_()
        else:
            # When deactivating, keep whatever visible state it had
            # (don’t force-hide here—let the OS do whatever it wants)
            self.show()  # reapply flags without changing visibility

    @pyqtSlot(Qt.ApplicationState)
    def _on_app_state_changed(self, state):
        if state == Qt.ApplicationState.ApplicationActive:
            # If it was visible when we lost focus, ensure it’s back
            if self._was_visible_on_deactivate:
                self.show()
                self.raise_()
            self._apply_topmost(True)
        else:
            # Remember whether we should bring it back later
            self._was_visible_on_deactivate = self.isVisible()
            self._apply_topmost(False)

    def eventFilter(self, obj, event):
        if obj is self.parent() and event.type() in (
            QEvent.Type.WindowActivate,
            QEvent.Type.ZOrderChange,
            QEvent.Type.ActivationChange,
        ):
            if (QApplication.instance().applicationState() == Qt.ApplicationState.ApplicationActive
                and self.isVisible()):
                self.raise_()
        return super().eventFilter(obj, event)

    def show_raise(self):
        self.show()
        self.raise_()

    @pyqtSlot(str)
    def append_line(self, message: str):
        doc = self.view.document()

        if message.startswith("🔄 Normalizing") and doc.blockCount() > 0:
            last = doc.findBlockByNumber(doc.blockCount() - 1)
            if last.isValid() and last.text().startswith("🔄 Normalizing"):
                cur = self.view.textCursor()
                cur.movePosition(QTextCursor.MoveOperation.End)
                cur.movePosition(QTextCursor.MoveOperation.StartOfBlock,
                                 QTextCursor.MoveMode.KeepAnchor)
                cur.removeSelectedText()
                cur.insertText(message)
                self.view.setTextCursor(cur)
            else:
                self.view.appendPlainText(message)
        else:
            self.view.appendPlainText(message)

        # trim
        if doc.blockCount() > self.MAX_BLOCKS:
            extra = doc.blockCount() - self.MAX_BLOCKS
            cur = self.view.textCursor()
            cur.movePosition(QTextCursor.MoveOperation.Start)
            cur.movePosition(QTextCursor.MoveOperation.Down,
                             QTextCursor.MoveMode.KeepAnchor, extra)
            cur.removeSelectedText()
            self.view.setTextCursor(self.view.textCursor())

        # autoscroll
        sb = self.view.verticalScrollBar()
        sb.setValue(sb.maximum())

    @pyqtSlot(str)
    def _on_post_status(self, msg: str):
        # update small “last status” indicator in the dialog (GUI thread slot)
        self._update_status_gui(msg)
        # append to the shared log window
        self.status_signal.emit(msg)
        # reflect in progress dialog label if it exists
        try:
            if getattr(self, "post_progress", None):
                self.post_progress.setLabelText(msg)
                QApplication.processEvents()
        except Exception:
            pass

def _save_master_with_rejection_layers(
    img_array: np.ndarray,
    hdr: "fits.Header",
    out_path: str,
    *,
    rej_any: "np.ndarray | None" = None,     # 2D bool
    rej_frac: "np.ndarray | None" = None,    # 2D float32 [0..1]
):
    """
    Writes a MEF (multi-extension FITS) file:
      - Primary HDU: the master image (2D or 3D) as float32
        * Mono: (H, W)
        * Color: (3, H, W)  <-- channels-first for FITS
      - Optional EXTNAME=REJ_COMB: uint8 (0/1) combined rejection mask
      - Optional EXTNAME=REJ_FRAC: float32 fraction-of-frames rejected per pixel
    """
    # --- sanitize/shape primary data ---
    data = np.asarray(img_array, dtype=np.float32, order="C")

    # If channels-last, move to channels-first for FITS
    if data.ndim == 3:
        # squeeze accidental singleton channels
        if data.shape[-1] == 1:
            data = np.squeeze(data, axis=-1)  # becomes (H, W)
        elif data.shape[-1] in (3, 4):       # RGB or RGBA
            data = np.transpose(data, (2, 0, 1))  # (C, H, W)
        # If already (C, H, W) leave it as-is.

    # After squeeze/transpose, re-evaluate dims
    if data.ndim not in (2, 3):
        raise ValueError(f"Unsupported master image shape for FITS: {data.shape}")

    # --- clone + annotate header, and align NAXIS* with 'data' ---
    H = (hdr.copy() if hdr is not None else fits.Header())
    # purge prior NAXIS keys to avoid conflicts after transpose/squeeze
    for k in ("NAXIS", "NAXIS1", "NAXIS2", "NAXIS3", "NAXIS4"):
        if k in H:
            del H[k]

    H["IMAGETYP"] = "MASTER STACK"
    H["BITPIX"]   = -32
    H["STACKED"]  = (True, "Stacked with rejection; channels-first in FITS if color")
    H["CREATOR"]  = "SetiAstroSuite"
    H["DATE-OBS"] = datetime.utcnow().isoformat()

    # Fill NAXIS* to match data (optional; Astropy will infer if omitted)
    if data.ndim == 2:
        H["NAXIS"]  = 2
        H["NAXIS1"] = int(data.shape[1])  # width
        H["NAXIS2"] = int(data.shape[0])  # height
    else:
        # data.shape == (C, H, W)
        C, Hh, Ww = data.shape
        H["NAXIS"]  = 3
        H["NAXIS1"] = int(Ww)  # width
        H["NAXIS2"] = int(Hh)  # height
        H["NAXIS3"] = int(C)   # channels/planes

    # --- build HDU list ---
    prim = fits.PrimaryHDU(data=data, header=H)
    hdul = [prim]

    # Optional layers: must be 2D (H, W). Convert types safely.
    if rej_any is not None:
        rej_any_2d = np.asarray(rej_any, dtype=bool)
        if rej_any_2d.ndim != 2:
            raise ValueError(f"REJ_COMB must be 2D, got {rej_any_2d.shape}")
        h = fits.Header()
        h["EXTNAME"] = "REJ_COMB"
        h["COMMENT"] = "Combined rejection mask (any algorithm / any frame)"
        hdul.append(fits.ImageHDU(data=rej_any_2d.astype(np.uint8, copy=False), header=h))

    if rej_frac is not None:
        rej_frac_2d = np.asarray(rej_frac, dtype=np.float32)
        if rej_frac_2d.ndim != 2:
            raise ValueError(f"REJ_FRAC must be 2D, got {rej_frac_2d.shape}")
        h = fits.Header()
        h["EXTNAME"] = "REJ_FRAC"
        h["COMMENT"] = "Per-pixel fraction of frames rejected [0..1]"
        hdul.append(fits.ImageHDU(data=rej_frac_2d, header=h))

    fits.HDUList(hdul).writeto(out_path, overwrite=True)

class _SimplePickDialog(QDialog):
    def __init__(self, np_image, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Click the comet center")
        self._orig = np.clip(np.asarray(np_image, dtype=np.float32), 0.0, 1.0)
        if self._orig.ndim == 3 and self._orig.shape[-1] == 1:
            self._orig = np.squeeze(self._orig, axis=-1)

        self._autostretch = False
        self._zoom = 1.0
        self._marker_xy = None  # image coords (float)

        v = QVBoxLayout(self)

        # ---- Graphics View scaffold ----
        self.scene = QGraphicsScene(self)
        self.view = _ZoomableGraphicsView(self.scene, self)
        self.view.setRenderHints(self.view.renderHints() | QPainter.RenderHint.SmoothPixmapTransform)
        self.view.setDragMode(QGraphicsView.DragMode.NoDrag)  # keep arrow; pan via wheel/scrollbars or your own handler
        self.view.setCursor(QCursor(CursorShape.ArrowCursor))
        self.view.viewport().setCursor(QCursor(CursorShape.ArrowCursor))
        v.addWidget(self.view)

        # pixmap item that holds the image
        self.pix_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pix_item)

        # marker (crosshair as small ellipse; we’ll draw lines into it)
        self.marker = QGraphicsEllipseItem(-6, -6, 12, 12)

        pen = QPen(QColor(0, 255, 0))   # bright green outline
        pen.setWidth(2)
        pen.setCosmetic(True)           # stays 2px regardless of zoom
        self.marker.setPen(pen)

        # optional: translucent green fill; or use Qt.NoBrush for hollow circle
        #self.marker.setBrush(QBrush(QColor(0, 255, 0, 60)))

        self.marker.setZValue(1_000_000)  # ensure it draws on top
        self.marker.setVisible(False)
        self.scene.addItem(self.marker)

        # ---- Controls ----
        row = QHBoxLayout()
        self.btn_fit = QPushButton("Fit")
        self.btn_fit.clicked.connect(self.fitToView)
        row.addWidget(self.btn_fit)

        self.btn_1x = QPushButton("1:1")
        self.btn_1x.clicked.connect(self.zoom1x)
        row.addWidget(self.btn_1x)

        self.btn_zi = QPushButton("Zoom In")
        self.btn_zi.clicked.connect(lambda: self.zoomBy(1.2))
        row.addWidget(self.btn_zi)

        self.btn_zo = QPushButton("Zoom Out")
        self.btn_zo.clicked.connect(lambda: self.zoomBy(1/1.2))
        row.addWidget(self.btn_zo)

        self.btn_st = QPushButton("Enable Autostretch")
        self.btn_st.setCheckable(True)
        self.btn_st.toggled.connect(self._toggle_autostretch)
        row.addWidget(self.btn_st)

        row.addStretch(1)
        v.addLayout(row)

        # ---- OK/Cancel ----
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        v.addWidget(btns)

        # image render + first fit
        self._update_pixmap()
        QTimer.singleShot(0, self.fitToView)

        # click to place marker
        self.view.mousePressOnScene = self._on_scene_click
        self.view.viewport().installEventFilter(self)

    def eventFilter(self, obj, ev):
        if obj is self.view.viewport():
            if ev.type() == QEvent.Type.CursorChange:
                obj.setCursor(QCursor(CursorShape.ArrowCursor))
                return True
        return super().eventFilter(obj, ev)

    # ---------- Display pipeline ----------
    def _make_display(self):
        if self._autostretch:
            if self._orig.ndim == 2:
                from imageops.stretch import stretch_mono_image
                disp = stretch_mono_image(self._orig, target_median=0.30, normalize=True, apply_curves=False)
            elif self._orig.ndim == 3 and self._orig.shape[2] == 3:
                from imageops.stretch import stretch_color_image
                disp = stretch_color_image(self._orig, target_median=0.30, linked=False, normalize=True, apply_curves=False)
            else:
                disp = self._orig
        else:
            disp = self._orig
        return (np.clip(disp, 0, 1) * 255).astype(np.uint8)

    def _update_pixmap(self):
        disp = self._make_display()
        if disp.ndim == 2:
            h, w = disp.shape; bpl = w
            qimg = QImage(disp.tobytes(), w, h, bpl, QImage.Format.Format_Grayscale8)
        else:
            h, w, _ = disp.shape; bpl = 3*w
            qimg = QImage(disp.tobytes(), w, h, bpl, QImage.Format.Format_RGB888)
        pm = QPixmap.fromImage(qimg)
        self.pix_item.setPixmap(pm)
        # reset scene rect to image bounds so fit works
        self.scene.setSceneRect(QRectF(0, 0, pm.width(), pm.height()))

        # keep marker visible at same image coordinate
        if self._marker_xy is not None:
            self._place_marker_at(self._marker_xy[0], self._marker_xy[1])

    # ---------- Zoom / Fit ----------
    def fitToView(self):
        if self.pix_item.pixmap().isNull():
            return
        self.view.fitInView(self.pix_item, Qt.AspectRatioMode.KeepAspectRatio)
        # record current zoom (approx) from view transform
        m = self.view.transform().m11()
        self._zoom = m

    def zoom1x(self):
        self.view.setTransform(QTransform())  # identity
        self._zoom = 1.0

    def zoomBy(self, factor):
        self._zoom *= factor
        self.view.scale(factor, factor)

    def _toggle_autostretch(self, checked):
        self._autostretch = bool(checked)
        self.btn_st.setText("Disable Autostretch" if checked else "Enable Autostretch")
        self._update_pixmap()

    # ---------- Picking ----------
    def _on_scene_click(self, scene_pos: QPointF, button: Qt.MouseButton):
        if button != Qt.MouseButton.LeftButton:
            return
        # clamp to image rect
        img_rect = self.scene.sceneRect()
        x = float(max(img_rect.left(), min(img_rect.right() - 1.0, scene_pos.x())))
        y = float(max(img_rect.top(),  min(img_rect.bottom() - 1.0, scene_pos.y())))
        self._marker_xy = (x, y)
        self._place_marker_at(x, y)

    def _place_marker_at(self, x, y):
        self.marker.setPos(QPointF(x, y))
        self.marker.setVisible(True)

    def point(self):
        """Return (x, y) in native image coordinates (float), or (0.0, 0.0) if none."""
        return self._marker_xy or (0.0, 0.0)

class _ZoomableGraphicsView(QGraphicsView):
    """
    Small helper view: Ctrl+wheel to zoom centered on cursor.
    Right or middle mouse drag pans (ScrollHandDrag is enabled).
    We call back to owner for clicks in scene coords.
    """
    def __init__(self, scene, owner):
        super().__init__(scene)
        self.owner = owner
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.SmartViewportUpdate)
        self.mousePressOnScene = None  # set by owner
        self.setMouseTracking(True)

    def wheelEvent(self, e):
        if e.modifiers() & Qt.KeyboardModifier.ControlModifier:
            factor = 1.15 if e.angleDelta().y() > 0 else 1/1.15
            # zoom around cursor
            pos = e.position()
            old_pos = self.mapToScene(int(pos.x()), int(pos.y()))
            self.scale(factor, factor)
            new_pos = self.mapToScene(int(pos.x()), int(pos.y()))
            delta = new_pos - old_pos
            self.translate(delta.x(), delta.y())
            e.accept()
        else:
            super().wheelEvent(e)

    def mousePressEvent(self, e):
        if self.mousePressOnScene and e.button() in (Qt.MouseButton.LeftButton,):
            sp = self.mapToScene(e.pos())
            self.mousePressOnScene(sp, e.button())
        super().mousePressEvent(e)

def _canonize_img(img: np.ndarray) -> np.ndarray:
    """Return image as float32, shape (H,W,C) where C is 1 or 3."""
    x = np.asarray(img, dtype=np.float32)
    if x.ndim == 2:
        return x[..., None]                # (H,W) -> (H,W,1)
    if x.ndim == 3 and x.shape[2] in (1, 3):
        return x
    # unexpected (e.g., more channels)
    raise ValueError(f"Unexpected image shape {x.shape}")

def _canonize_mask(mask: np.ndarray, channels: int) -> np.ndarray:
    """Return mask in [0,1], shape (H,W,channels)."""
    m = np.asarray(mask, dtype=np.float32)
    if m.ndim == 2:
        m = m[..., None]                   # (H,W) -> (H,W,1)
    elif m.ndim == 3 and m.shape[2] == 1:
        pass
    else:
        raise ValueError(f"Unexpected mask shape {m.shape}")
    # repeat to RGB if needed
    if channels == 3 and m.shape[2] == 1:
        m = np.repeat(m, 3, axis=2)
    return np.clip(m, 0.0, 1.0)

def feather_mask(mask_hw_or_hw1: np.ndarray, feather_px: int, channels: int) -> np.ndarray:
    """Feather with Gaussian blur; returns (H,W,channels) in [0,1]."""
    m = _canonize_mask(mask_hw_or_hw1, 1)[..., 0]   # work in 2-D
    if feather_px and feather_px > 0:
        # Convert feather_px (approx radius) to sigma and odd kernel size
        sigma = max(0.1, float(feather_px) / 2.0)
        k = int(2 * int(3 * sigma) + 1)            # ~3σ radius, odd
        m = cv2.GaussianBlur(m, (k, k), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REPLICATE)
        m = np.clip(m, 0.0, 1.0)
    # expand to channels
    if channels == 3:
        m = np.repeat(m[..., None], 3, axis=2)
    else:
        m = m[..., None]
    return m

def blend_stars_comet(stars_img, comet_img, comet_mask, feather_px=16, mix=1.0):
    """
    stars_img: (H,W) or (H,W,1/3)
    comet_img: (H,W) or (H,W,1/3)
    comet_mask: (H,W) or (H,W,1)
    mix in [0..1]: 0 = only stars, 1 = only comet in masked areas
    """
    S = _canonize_img(stars_img)
    C = _canonize_img(comet_img)

    # If one is mono and the other RGB, upcast mono to RGB for a consistent result
    if S.shape[2] != C.shape[2]:
        if S.shape[2] == 1 and C.shape[2] == 3:
            S = np.repeat(S, 3, axis=2)
        elif S.shape[2] == 3 and C.shape[2] == 1:
            C = np.repeat(C, 3, axis=2)
        else:
            raise ValueError(f"Unsupported channel combination: {S.shape} vs {C.shape}")

    A = feather_mask(comet_mask, feather_px, S.shape[2]) * float(np.clip(mix, 0.0, 1.0))
    out = (1.0 - A) * S + A * C
    return np.clip(out, 0.0, 1.0)

def _match_channels(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (A', B') with matching channel counts (1 or 3), float32."""
    A = np.asarray(A, dtype=np.float32)
    B = np.asarray(B, dtype=np.float32)
    def _c(x):
        if x.ndim == 2:    return 1
        if x.ndim == 3:    return x.shape[2]
        raise ValueError(f"Unexpected image shape {x.shape}")
    ca, cb = _c(A), _c(B)
    if ca == cb:
        return A, B
    if ca == 1 and cb == 3:
        A = np.repeat(A[..., None] if A.ndim == 2 else A, 3, axis=2)
    elif ca == 3 and cb == 1:
        B = np.repeat(B[..., None] if B.ndim == 2 else B, 3, axis=2)
    else:
        # handle (H,W,1) vs (H,W): squeeze to 2D then upcast consistently
        if A.ndim == 3 and A.shape[2] == 1: A = A[..., 0]
        if B.ndim == 3 and B.shape[2] == 1: B = B[..., 0]
        A = A[..., None] if A.ndim == 2 else A
        B = B[..., None] if B.ndim == 2 else B
    return A, B

# --- comet-friendly reducers ---
def _lower_trimmed_mean(ts: np.ndarray, trim_hi_frac: float = 0.30) -> np.ndarray:
    """
    Per-pixel lower-trimmed mean: drop the brightest t% (star trails) then mean.
    ts: (N, th, tw, C) float32
    """
    n = ts.shape[0]
    k = int(np.floor(n * (1.0 - trim_hi_frac)))
    if k <= 0:
        return np.median(ts, axis=0)
    part = np.partition(ts, k-1, axis=0)[:k, ...]  # keep lowest k across N
    return np.mean(part, axis=0)

def _percentile40(ts: np.ndarray) -> np.ndarray:
    """Low-percentile combiner; good at suppressing bright streaks."""
    return np.percentile(ts, 40.0, axis=0)

def _high_clip_percentile(ts: np.ndarray, k: float = 2.5, p: float = 40.0) -> np.ndarray:
    """
    Robust high-side winsorize per pixel, then take a low percentile.
    ts: (N, th, tw, C) float32
    k:  MAD multiplier for high tail clamp
    p:  percentile to return (e.g. 35..45)
    """
    med = np.median(ts, axis=0)
    mad = np.median(np.abs(ts - med), axis=0) + 1e-6
    hi = med + (k * 1.4826 * mad)
    clipped = np.minimum(ts, hi, dtype=ts.dtype)
    return np.percentile(clipped, p, axis=0)

def _high_clip_percentile_fast(ts: np.ndarray, k: float = 2.5, p: float = 40.0,
                               _work: dict = {}) -> np.ndarray:
    """
    Same math as _high_clip_percentile, but ~2–4× faster on CPU:
      - median / MAD via np.partition (no full sort)
      - percentile via partition index (nearest-rank)
      - reuse scratch buffers between calls (pass via _work dict)
    ts: (N, th, tw, C) float32
    """
    N = ts.shape[0]
    assert N >= 2, "need at least two frames"

    # allocate / reuse scratch buffers
    tmp  = _work.get("tmp");   # for med
    tmp2 = _work.get("tmp2");  # for mad & hi
    tmp3 = _work.get("tmp3");  # for percentile
    if (tmp is None) or (tmp.shape != ts.shape):
        tmp  = np.empty_like(ts)
        tmp2 = np.empty_like(ts)
        tmp3 = np.empty_like(ts)
        _work["tmp"], _work["tmp2"], _work["tmp3"] = tmp, tmp2, tmp3

    # ---- median along axis 0 (copy -> partition -> take middle slice)
    np.copyto(tmp, ts)
    np.partition(tmp, N // 2, axis=0)
    med = tmp[N // 2]  # shape: (th, tw, C)

    # ---- MAD: median(|ts - med|)
    np.subtract(ts, med, out=tmp2)
    np.abs(tmp2, out=tmp2)
    np.copyto(tmp, tmp2)
    np.partition(tmp, N // 2, axis=0)
    mad = tmp[N // 2] + 1e-6

    # ---- high-side clip threshold: hi = med + k * 1.4826 * MAD
    np.multiply(mad, (k * 1.4826), out=tmp)   # tmp = k*1.4826*MAD
    np.add(med, tmp, out=tmp)                 # tmp = hi

    # ---- clip high side into tmp2 (reuse): clipped = min(ts, hi)
    np.minimum(ts, tmp, out=tmp2)

    # ---- percentile via nearest-rank partition
    # index of p-th percentile along axis=0
    p = float(p)
    idx = int(np.clip(round((p / 100.0) * (N - 1)), 0, N - 1))
    np.copyto(tmp3, tmp2)
    np.partition(tmp3, idx, axis=0)
    return tmp3[idx]

def _parse_binning_from_header(hdr) -> tuple[int, int]:
    """
    Return (xbin, ybin), defaulting to (1,1). Handles common FITS keys and string forms.
    Accepts: XBINNING/YBINNING, CCDXBIN/CCDYBIN, XBIN/YBIN, BINNING="2 2" | "2x2" | "2,2"
    """
    def _coerce(v):
        try:
            return int(float(v))
        except Exception:
            return None

    if hdr is None:
        return 1, 1

    # direct numeric keys
    for kx, ky in (("XBINNING", "YBINNING"),
                   ("CCDXBIN", "CCDYBIN"),
                   ("XBIN", "YBIN")):
        if kx in hdr and ky in hdr:
            bx = _coerce(hdr.get(kx)); by = _coerce(hdr.get(ky))
            if bx and by:
                return max(1, bx), max(1, by)

    # combined string key
    if "BINNING" in hdr:
        s = str(hdr.get("BINNING", "")).lower().replace("x", " ").replace(",", " ")
        parts = [p for p in s.split() if p.strip()]
        if len(parts) >= 2:
            bx = _coerce(parts[0]); by = _coerce(parts[1])
            if bx and by:
                return max(1, bx), max(1, by)

    return 1, 1


def _resize_to_scale(img: np.ndarray, scale_x: float, scale_y: float) -> np.ndarray:
    """
    Resample image to apply per-axis scale. Works for 2D or HxWx3.
    Uses OpenCV if present (fast, high quality). Fallbacks to SciPy; last resort = NumPy kron for integer upscales.
    """
    if scale_x == 1.0 and scale_y == 1.0:
        return img

    h, w = img.shape[:2]
    new_w = max(1, int(round(w * (scale_x))))
    new_h = max(1, int(round(h * (scale_y))))

    # Prefer OpenCV
    try:
        import cv2
        interp = cv2.INTER_CUBIC if (scale_x > 1.0 or scale_y > 1.0) else cv2.INTER_AREA
        if img.ndim == 2:
            return cv2.resize(img.astype(np.float32, copy=False), (new_w, new_h), interpolation=interp)
        else:
            return cv2.resize(img.astype(np.float32, copy=False), (new_w, new_h), interpolation=interp)
    except Exception:
        pass

    # SciPy fallback
    try:
        from scipy.ndimage import zoom
        if img.ndim == 2:
            return zoom(img.astype(np.float32, copy=False), (scale_y, scale_x), order=3)
        else:
            # zoom each channel; zoom expects zoom per axis
            return zoom(img.astype(np.float32, copy=False), (scale_y, scale_x, 1.0), order=3)
    except Exception:
        pass

    # Last resort (integer upscale only): kron
    sx_i = int(round(scale_x)); sy_i = int(round(scale_y))
    if abs(scale_x - sx_i) < 1e-6 and abs(scale_y - sy_i) < 1e-6 and sx_i >= 1 and sy_i >= 1:
        if img.ndim == 2:
            return np.kron(img, np.ones((sy_i, sx_i), dtype=np.float32))
        else:
            up2d = np.kron(img[..., 0], np.ones((sy_i, sx_i), dtype=np.float32))
            out = np.empty((up2d.shape[0], up2d.shape[1], img.shape[2]), dtype=np.float32)
            for c in range(img.shape[2]):
                out[..., c] = np.kron(img[..., c], np.ones((sy_i, sx_i), dtype=np.float32))
            return out

    # If all else fails, return original
    return img

def _center_crop_2d(a: np.ndarray, Ht: int, Wt: int) -> np.ndarray:
    H, W = a.shape[:2]
    if H == Ht and W == Wt:
        return a
    y0 = max(0, (H - Ht) // 2)
    x0 = max(0, (W - Wt) // 2)
    return a[y0:y0+Ht, x0:x0+Wt]

def _median_fast_sample(img: np.ndarray, stride: int = 8) -> float:
    """
    Fast robust median on a small strided sample.
    Works on mono or RGB (uses luma).
    Assumes 'img' is float32 and non-NaN/-Inf already.
    """
    if img.ndim == 3 and img.shape[-1] == 3:
        r, g, b = img[..., 0], img[..., 1], img[..., 2]
        L = 0.2126 * r + 0.7152 * g + 0.0722 * b
        sample = L[::stride, ::stride]
    else:
        sample = img[::stride, ::stride]
    sample = sample - float(np.nanmin(sample))
    return float(np.median(sample))

def _luma_view(img: np.ndarray) -> np.ndarray:
    """Return a float32 2D luma view (no copy if mono)."""
    if img.ndim == 2:
        return img.astype(np.float32, copy=False)
    if img.ndim == 3 and img.shape[-1] == 3:
        r, g, b = img[..., 0], img[..., 1], img[..., 2]
        # compute in float32, no allocation explosion
        return (0.2126 * r + 0.7152 * g + 0.0722 * b).astype(np.float32, copy=False)
    if img.ndim == 3 and img.shape[-1] == 1:
        return np.squeeze(img, axis=-1).astype(np.float32, copy=False)
    return img.astype(np.float32, copy=False)


def _median_fast_sample(img: np.ndarray, stride: int = 8) -> float:
    """
    Very fast robust median from a decimated grid.
    Works on mono or color (uses luma).
    """
    v = _luma_view(img)
    v = v[::stride, ::stride]
    # subtract a tiny pedestal so transparency/G gain differences dominate
    vmin = float(np.nanmin(v))
    return float(np.median(v - vmin)) if v.size else 0.0


def _compute_scale(ref_target_median: float, preview_median: float, img: np.ndarray,
                   refine_stride: int = 8, refine_if_rel_err: float = 0.10) -> float:
    """
    Start from preview-based scale, optionally refine once on a tiny decimated sample.
    """
    eps = 1e-6
    s0 = ref_target_median / max(preview_median, eps)  # first guess (from preview)
    # quick refinement only if we're likely off by >10%
    m_post = _median_fast_sample(img, stride=refine_stride)
    if m_post > eps:
        s1 = ref_target_median / m_post
        # if preview and refined differ a lot, trust refined; otherwise keep s0 (avoids jitter)
        if abs(s1 - s0) / max(s0, eps) >= refine_if_rel_err:
            return float(s1)
    return float(s0)


def _apply_scale_inplace(img: np.ndarray, s: float) -> np.ndarray:
    """
    Scale in-place when safe (saves alloc), else returns a new float32 array.
    """
    # ensure float32 for consistent stack later; try in-place if possible
    if img.dtype != np.float32:
        img = img.astype(np.float32, copy=False)
    # numpy multiply is already vectorized/SIMD
    img *= np.float32(s)
    return img

def _fits_first_image_hdu(hdul):
    """
    Return (hdu_index, hdu) for the first HDU that actually contains an image
    (PrimaryHDU or ImageHDU or CompImageHDU) with a numeric ndarray.
    """
    for i, h in enumerate(hdul):
        try:
            data = h.data  # astropy will auto-scale with BSCALE/BZERO
        except Exception:
            continue
        if isinstance(data, np.ndarray) and data.size > 0:
            # Restrict to 2D or 3D images only
            if data.ndim in (2, 3):
                return i, h
    return None, None

def _fits_read_any_hdu(path, prefer_float32=True, memmap=True):
    """
    Open FITS and return (img, header, hdu_index). Finds the *first* HDU that has
    real image data. Handles PrimaryHDU, ImageHDU, CompImageHDU.
    Converts to float32 unless prefer_float32=False.
    Squeezes trailing singleton channel dim.
    """
    with fits.open(path, memmap=memmap) as hdul:
        idx, h = _fits_first_image_hdu(hdul)
        if h is None:
            return None, None, None
        img = h.data
        hdr = h.header
        # standardize dtype
        if prefer_float32 and img is not None:
            img = np.asarray(img, dtype=np.float32, order="C")
        # squeeze trailing channel=1
        if img is not None and img.ndim == 3 and img.shape[-1] == 1:
            img = np.squeeze(img, axis=-1)
        return img, hdr, idx

def _apply_blank_bscale_bzero(arr: np.ndarray, hdr: fits.Header) -> np.ndarray:
    """Manual FITS image scaling: apply BLANK (→ 0), then BSCALE/BZERO. Return float32."""
    if ma.isMaskedArray(arr):
        arr = arr.filled(0)
    # Replace BLANK sentinel for integer images
    if hdr is not None and "BLANK" in hdr and np.issubdtype(arr.dtype, np.integer):
        blank = int(hdr["BLANK"])
        arr = np.where(arr == blank, 0, arr)

    # Manual scaling
    bscale = float(hdr.get("BSCALE", 1.0)) if hdr is not None else 1.0
    bzero  = float(hdr.get("BZERO", 0.0)) if hdr is not None else 0.0
    arr = arr.astype(np.float32, copy=False)
    if (bscale != 1.0) or (bzero != 0.0):
        arr = arr * np.float32(bscale) + np.float32(bzero)
    return arr

def _fits_read_any_hdu_noscale(path: str, memmap: bool) -> tuple[np.ndarray | None, fits.Header | None, int | None]:
    """
    Open FITS with do_not_scale_image_data=True and return (img, header, hdu_index)
    for the first HDU that has 2D/3D numeric image data. Applies BLANK/BSCALE/BZERO manually.
    """
    with fits.open(path,
                   memmap=memmap,
                   do_not_scale_image_data=True,
                   ignore_missing_end=True,
                   uint=False) as hdul:
        for i, h in enumerate(hdul):
            try:
                d = h.data
            except Exception:
                continue
            if not isinstance(d, np.ndarray):
                continue
            if d.ndim not in (2, 3) or d.size == 0:
                continue
            # Ensure native byte order (avoid later copies)
            d = _to_native_endian(d)
            # Squeeze trivial leading/ending dims
            d = np.squeeze(d)
            # Manual scaling to float32
            d = _apply_blank_bscale_bzero(d, h.header)

            # If 3D and trailing channel is 1, squeeze it
            if d.ndim == 3 and d.shape[-1] == 1:
                d = np.squeeze(d, axis=-1)

            # Clean numerics
            d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
            return d, h.header, i
    return None, None, None

class _Progress:
    def __init__(self, owner, title: str, maximum: int):
        self._owner = owner
        self._pd = QProgressDialog(title, "Cancel", 0, max(1, int(maximum)), owner)
        self._pd.setWindowModality(Qt.WindowModality.ApplicationModal)
        self._pd.setMinimumDuration(0)
        self._pd.setAutoClose(False)
        self._pd.setAutoReset(False)
        self._pd.setValue(0)
        self._pd.setWindowTitle(title)
        self._pd.setMinimumWidth(520)
        self._cancelled = False
        self._value = 0

        def _on_cancel():
            self._cancelled = True
        self._pd.canceled.connect(_on_cancel, Qt.ConnectionType.QueuedConnection)
        self._pd.show()
        QApplication.processEvents()

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def set_label(self, text: str):
        try:
            self._pd.setLabelText(text)
        except Exception:
            pass

    def set_value(self, v: int):
        self._value = max(0, min(int(v), self._pd.maximum()))
        try:
            self._pd.setValue(self._value)
        except Exception:
            pass
        QApplication.processEvents()

    def step(self, n: int = 1, label: str | None = None):
        if label:
            self.set_label(label)
        self.set_value(self._value + n)

    def close(self):
        try:
            self._pd.reset()
            self._pd.deleteLater()
        except Exception:
            pass

def _count_tiles(h: int, w: int, ch: int, cw: int) -> int:
    ty = (h + ch - 1) // ch
    tx = (w + cw - 1) // cw
    return ty * tx



def _nearest_index(src_len: int, dst_len: int) -> np.ndarray:
    """Nearest-neighbor index map from dst→src (no rounding drift)."""
    if dst_len <= 0:
        return np.zeros((0,), dtype=np.int32)
    scale = src_len / float(dst_len)
    idx = np.floor((np.arange(dst_len, dtype=np.float32) + 0.5) * scale).astype(np.int32)
    idx[idx < 0] = 0
    idx[idx >= src_len] = src_len - 1
    return idx

def _nearest_resize_2d(m: np.ndarray, H: int, W: int) -> np.ndarray:
    """Resize 2-D array to (H,W) using nearest-neighbor, cv2 if available else pure NumPy."""
    m = np.asarray(m, dtype=np.float32)
    if cv2 is not None:
        return cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST).astype(np.float32, copy=False)
    yi = _nearest_index(m.shape[0], H)
    xi = _nearest_index(m.shape[1], W)
    return m[yi[:, None], xi[None, :]].astype(np.float32, copy=False)

def _expand_mask_for(image_like: np.ndarray, mask_like: np.ndarray) -> np.ndarray:
    """
    Return a float32 mask in [0..1] shaped for image_like:
      - If image is HxW  (mono): mask -> HxW
      - If image is HxWxC: mask -> HxWxC (repeated per channel)
    Accepts mask as HxW, HxWx1, or HxWxC (we'll reduce to 2-D then repeat).
    """
    im = np.asarray(image_like)
    m  = np.asarray(mask_like, dtype=np.float32)

    # squeeze trivial last dim
    if m.ndim == 3 and m.shape[2] == 1:
        m = m[..., 0]

    # normalize to [0..1] if mask is 0..255 or arbitrary
    mmin, mmax = float(m.min(initial=0.0)), float(m.max(initial=1.0))
    if mmax > 1.0 or mmin < 0.0:
        if mmax > 0:
            m = m / mmax
        m = np.clip(m, 0.0, 1.0)

    if im.ndim == 2:
        # want 2-D
        if m.ndim == 3:
            m = m[..., 0]
        if m.shape != im.shape:
            m = _nearest_resize_2d(m, im.shape[0], im.shape[1])
        return m.astype(np.float32, copy=False)

    # RGB path
    H, W, C = im.shape[:3]
    if m.ndim == 3 and m.shape[:2] == (H, W):
        # reduce multi-channel mask to 2-D via max then repeat
        m = np.max(m, axis=2)
    if m.ndim == 2:
        if m.shape != (H, W):
            m = _nearest_resize_2d(m, H, W)
        m = np.repeat(m[..., None], C, axis=2)
    return np.clip(m, 0.0, 1.0).astype(np.float32, copy=False)

def _debug_dump_mask_blend(prefix: str, img: np.ndarray, mask: np.ndarray, out_dir: str):
    try:
        os.makedirs(out_dir, exist_ok=True)
        m = _expand_mask_for(img, mask)
        # save per-channel PNGs to spot one-channel issues instantly
        def _to8(u): 
            u = np.clip(u.astype(np.float32), 0, 1) * 255.0
            return u.astype(np.uint8)
        if img.ndim == 3:
            for i, ch in enumerate(["R","G","B"]):
                cv2.imwrite(os.path.join(out_dir, f"{prefix}_img_{ch}.png"), _to8(img[..., i]))
                cv2.imwrite(os.path.join(out_dir, f"{prefix}_mask_{ch}.png"), _to8(m[..., i]))
        else:
            cv2.imwrite(os.path.join(out_dir, f"{prefix}_img.png"), _to8(img))
            cv2.imwrite(os.path.join(out_dir, f"{prefix}_mask.png"), _to8(m if m.ndim==2 else m[...,0]))
    except Exception:
        pass

def _float01_to_u16(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=np.float32), 0.0, 1.0)
    return (x * 65535.0 + 0.5).astype(np.uint16, copy=False)

def _u16_to_float01(x: np.ndarray) -> np.ndarray:
    if x.dtype != np.uint16:
        # be forgiving: if someone returns 8-bit, still normalize
        if x.dtype == np.uint8:
            return (x.astype(np.float32) / 255.0).astype(np.float32)
        return np.clip(x.astype(np.float32), 0.0, 1.0)
    return (x.astype(np.float32) / 65535.0).astype(np.float32)

def _write_poly_or_unknown(fh, file_key: str, kind_name: str):
    """Helper: write a block with no numeric matrix (loader maps to (kind, None))."""
    fh.write(f"FILE: {file_key}\n")
    fh.write(f"KIND: {kind_name}\n")
    fh.write("MATRIX:\nUNSUPPORTED\n\n")

_SESSION_PATTERNS = [
    # "Night1", "night_02", "Session-3", "sess7"
    (re.compile(r"(session|sess|night|noche|nuit)[ _-]?(\d{1,2})", re.I),
    lambda m: f"Session-{int(m.group(2)):02d}"),

    # ISO-ish dates in folder names: 2024-10-09, 2024_10_09, 20241009
    (re.compile(r"\b(20\d{2})[-_]?([01]\d)[-_]?([0-3]\d)\b"),
    lambda m: f"{m.group(1)}-{m.group(2)}-{m.group(3)}"),
]

def _to_writable_f32(arr):
    import numpy as np
    a = np.asarray(arr)
    if a.dtype != np.float32:
        a = a.astype(np.float32, copy=False)
    # Make sure we can write into it (mmap/XISF often returns read-only views)
    if (not a.flags.writeable) or (not a.flags.c_contiguous):
        a = np.ascontiguousarray(a.copy())
    return a

def _is_fits(path: str) -> bool:
    p = path.lower()
    return p.endswith(".fit") or p.endswith(".fits") or p.endswith(".fz")

def _is_xisf(path: str) -> bool:
    return path.lower().endswith(".xisf")

def _xisf_first_kw(meta_dict: dict, key: str, cast=str):
    """
    XISF images_meta['FITSKeywords'] → { KEY: [ {value, comment}, ...], ... }.
    Return first value if present, casted.
    """
    try:
        v = meta_dict.get("FITSKeywords", {}).get(key, [])
        if not v: return None
        val = v[0]["value"]
        return cast(val) if cast and val is not None else val
    except Exception:
        return None

def _get_header_fast(path: str):
    """
    FITS: std fast header. XISF: synthesize a dict with common keys used upstream.
    """
    if _is_fits(path):
        try:
            return fits.getheader(path, ext=0)
        except Exception:
            return {}
    if _is_xisf(path):
        try:
            x = XISF(path)
            im = x.get_images_metadata()[0]  # first image block
            # Build a tiny FITS-like dict
            hdr = {}
            # Common bits we use elsewhere
            filt = _xisf_first_kw(im, "FILTER", str)
            if filt is not None:
                hdr["FILTER"] = filt
            # Exposure: EXPTIME/EXPOSURE
            for k in ("EXPOSURE", "EXPTIME"):
                v = _xisf_first_kw(im, k, float)
                if v is not None:
                    hdr[k] = v
                    break
            # Bayer pattern (PI often writes CFA pattern as a keyword too)
            bp = _xisf_first_kw(im, "BAYERPAT", str)
            if bp:
                hdr["BAYERPAT"] = bp
            # Some PI exports use XISF properties instead of FITS keywords.
            props = im.get("XISFProperties", {})
            # Try a few common property ids for convenience
            for cand in ("Instrument:FILTER", "FILTER", "Filter:Name", "FilterName"):
                if cand in props and "value" in props[cand]:
                    hdr.setdefault("FILTER", str(props[cand]["value"]))
                    break
            for cand in ("Exposure:Time", "EXPTIME"):
                if cand in props and "value" in props[cand]:
                    try:
                        hdr.setdefault("EXPTIME", float(props[cand]["value"]))
                    except Exception:
                        pass
                    break
            return hdr
        except Exception:
            return {}
    return {}

def _quick_preview_from_path(path: str, *, target_xbin=1, target_ybin=1) -> np.ndarray | None:
    """
    Debayer-aware, tiny, 2D float32 preview for FITS or XISF.
    Mirrors your _quick_preview_from_fits behavior but supports XISF.
    Returned image is small-ish, writeable, contiguous.
    """
    def _superpixel2x2(x: np.ndarray) -> np.ndarray:
        h, w = x.shape[:2]
        h2, w2 = h - (h % 2), w - (w % 2)
        if h2 <= 0 or w2 <= 0:
            return x.astype(np.float32, copy=False)
        x = x[:h2, :w2].astype(np.float32, copy=False)
        if x.ndim == 2:
            return (x[0:h2:2, 0:w2:2] + x[0:h2:2, 1:w2:2] +
                    x[1:h2:2, 0:w2:2] + x[1:h2:2, 1:w2:2]) * 0.25
        # RGB → luma then superpixel
        r, g, b = x[..., 0], x[..., 1], x[..., 2]
        L = 0.2126 * r + 0.7152 * g + 0.0722 * b
        return (L[0:h2:2, 0:w2:2] + L[0:h2:2, 1:w2:2] +
                L[1:h2:2, 0:w2:2] + L[1:h2:2, 1:w2:2]) * 0.25

    try:
        if _is_fits(path):
            # Reuse your existing routine if you like; otherwise a tiny inline path:
            from astropy.io import fits
            with fits.open(path, memmap=True) as hdul:
                # Prefer primary image HDU; fall back to first image-like HDU
                hdu = None
                for cand in (0,):
                    try:
                        h = hdul[cand]
                        if getattr(h, "data", None) is not None:
                            hdu = h; break
                    except Exception:
                        pass
                if hdu is None:
                    for h in hdul:
                        if getattr(h, "data", None) is not None:
                            hdu = h; break
                if hdu is None or hdu.data is None:
                    return None
                data = np.asanyarray(hdu.data)
                # Make small, 2D, float32
                if data.ndim == 3 and data.shape[-1] == 1:
                    data = np.squeeze(data, axis=-1)
                data = data.astype(np.float32, copy=False)
                prev = _superpixel2x2(data)
                return np.ascontiguousarray(prev, dtype=np.float32)
        elif _is_xisf(path):
            x = XISF(path)
            im = x.read_image(0)  # channels_last
            if im is None:
                return None
            if im.ndim == 3 and im.shape[-1] == 1:
                im = np.squeeze(im, axis=-1)
            im = im.astype(np.float32, copy=False)
            prev = _superpixel2x2(im)
            return np.ascontiguousarray(prev, dtype=np.float32)
        else:
            # Non-FITS raster: you can fall back to PIL if desired
            return None
    except Exception:
        return None

def _synth_header_from_xisf_meta(im_meta: dict) -> fits.Header:
    """Make a minimal FITS-like Header from XISF image metadata."""
    h = fits.Header()
    try:
        # FITSKeywords shape: { KEY: [ {value, comment}, ...] }
        kwords = im_meta.get("FITSKeywords", {}) or {}
        def _first(key):
            v = kwords.get(key)
            return None if not v else v[0].get("value")
        # Filter / Exposure
        flt = _first("FILTER")
        if flt is not None: h["FILTER"] = str(flt)
        for k in ("EXPOSURE", "EXPTIME"):
            v = _first(k)
            if v is not None:
                try: h[k] = float(v)
                except Exception: h[k] = v
                break
        bp = _first("BAYERPAT")
        if bp: h["BAYERPAT"] = str(bp)
        # A couple of safe hints
        h["ORIGIN"] = "XISF-import"
    except Exception:
        pass
    return h

def _load_image_for_stack(path: str):
    """
    Return (img_float32_contig, header, is_mono) for either FITS or XISF.
    Never returns a read-only view; always writeable contiguous float32.
    """
    p = path.lower()
    if p.endswith((".fit", ".fits", ".fz")):
        with fits.open(path, memmap=True) as hdul:
            # choose first image-like HDU
            hdu = None
            for h in hdul:
                if getattr(h, "data", None) is not None:
                    hdu = h; break
            if hdu is None or hdu.data is None:
                raise IOError(f"No image data in FITS: {path}")
            arr = np.asanyarray(hdu.data)
            # Make a safe float32 copy (writeable, native-endian, contiguous)
            img = np.array(arr, dtype=np.float32, copy=True, order="C")
            if img.ndim == 3 and img.shape[-1] == 1:
                img = np.squeeze(img, axis=-1)
            hdr = hdu.header or fits.Header()
            is_mono = bool(img.ndim == 2 or (img.ndim == 3 and img.shape[-1] == 1))
            return img, hdr, is_mono

    if p.endswith(".xisf"):
        x = XISF(path)
        im_meta = x.get_images_metadata()[0]
        arr = x.read_image(0)  # channels_last
        if arr is None:
            raise IOError(f"No image data in XISF: {path}")
        img = np.array(arr, dtype=np.float32, copy=True, order="C")
        if img.ndim == 3 and img.shape[-1] == 1:
            img = np.squeeze(img, axis=-1)
        hdr = _synth_header_from_xisf_meta(im_meta)
        is_mono = bool(img.ndim == 2 or (img.ndim == 3 and img.shape[-1] == 1))
        return img, hdr, is_mono

    # Fallback: try FITS loader so we get a useful error
    with fits.open(path, memmap=True) as hdul:
        hdu = hdul[0]
        arr = np.asanyarray(hdu.data)
        img = np.array(arr, dtype=np.float32, copy=True, order="C")
        hdr = hdu.header or fits.Header()
        is_mono = bool(img.ndim == 2 or (img.ndim == 3 and img.shape[-1] == 1))
        return img, hdr, is_mono

class _MMImage:
    """
    Unified memory-friendly reader for FITS (memmap) and XISF.
    Exposes: .shape, .ndim, .read_tile(y0,y1,x0,x1), .read_full(), .close()
    Always returns float32 arrays (writeable) with color last if present.
    """

    def __init__(self, path: str):
        self.path = path
        self._kind = None          # "fits" | "xisf"
        self._scale = None         # integer -> [0..1] scale
        self._fits_hdul = None
        self._xisf = None
        self._xisf_memmap = None   # np.memmap when possible
        self._xisf_arr = None      # decompressed ndarray when needed
        self._xisf_color_axis = None
        self._xisf_spat_axes = (0, 1)
        self._xisf_dtype = None

        p = path.lower()
        if p.endswith((".fit", ".fits", ".fz")):
            self._open_fits(path)
            self._kind = "fits"
        elif p.endswith(".xisf"):
            if XISF is None:
                raise RuntimeError("XISF support not available (import failed).")
            self._open_xisf(path)
            self._kind = "xisf"
        else:
            # let FITS try anyway so you get a useful error text
            self._open_fits(path)
            self._kind = "fits"

    # ---------------- FITS ----------------
    def _open_fits(self, path: str):
        self._fits_hdul = fits.open(path, memmap=True)
        # choose first image-like HDU (don’t assume PRIMARY only)
        hdu = None
        for h in self._fits_hdul:
            if getattr(h, "data", None) is not None:
                hdu = h; break
        if hdu is None or hdu.data is None:
            raise ValueError(f"Empty FITS: {path}")

        d = hdu.data
        self._fits_data = d
        self.shape = d.shape
        self.ndim  = d.ndim
        self._orig_dtype = d.dtype

        # Detect color axis (size==3) and spatial axes once
        if self.ndim == 2:
            self._color_axis = None
            self._spat_axes  = (0, 1)
        elif self.ndim == 3:
            dims = self.shape
            self._color_axis = next((i for i, s in enumerate(dims) if s == 3), None)
            self._spat_axes  = (0, 1) if self._color_axis is None else tuple(i for i in range(3) if i != self._color_axis)
        else:
            raise ValueError(f"Unsupported ndim={self.ndim} for {path}")

        # late normalization scale for integer data
        if   self._orig_dtype == np.uint8:  self._scale = 1.0/255.0
        elif self._orig_dtype == np.uint16: self._scale = 1.0/65535.0
        else:                                self._scale = None

    # ---------------- XISF ----------------
    def _open_xisf(self, path: str):
        x = XISF(path)
        ims = x.get_images_metadata()
        if not ims:
            raise ValueError(f"Empty XISF: {path}")
        m0 = ims[0]
        self._xisf = x
        self._xisf_dtype = m0["dtype"]              # numpy dtype
        w, h, chc = m0["geometry"]                  # (width, height, channels)
        self.shape = (h, w) if chc == 1 else (h, w, chc)
        self.ndim  = 2 if chc == 1 else 3

        # color and spatial axes (image data is planar CHW on disk, we expose HWC)
        self._xisf_color_axis = None if chc == 1 else 2
        self._xisf_spat_axes  = (0, 1)

        # choose integer scale (same convention as FITS branch)
        if   self._xisf_dtype == np.dtype("uint8"):  self._scale = 1.0/255.0
        elif self._xisf_dtype == np.dtype("uint16"): self._scale = 1.0/65535.0
        else:                                        self._scale = None

        # Try zero-copy memmap only when the image block is an uncompressed attachment.
        loc = m0.get("location", None)
        comp = m0.get("compression", None)
        if isinstance(loc, tuple) and loc[0] == "attachment" and not comp:
            # location = ("attachment", pos, size)
            pos = int(loc[1])
            # on-disk order for XISF planar is (C,H,W). We memmap that and rearrange at slice time.
            chc = 1 if self.ndim == 2 else self.shape[2]
            shp_on_disk = (chc, self.shape[0], self.shape[1])
            # Align dtype endianness to native when mapping
            dt = self._xisf_dtype.newbyteorder("<") if self._xisf_dtype.byteorder == ">" else self._xisf_dtype
            self._xisf_memmap = np.memmap(path, mode="r", dtype=dt, offset=pos, shape=shp_on_disk)
            self._xisf_arr = None
        else:
            # Compressed / inline / embedded → must decompress whole image once
            arr = x.read_image(0)  # HWC float/uint per metadata
            # Ensure we own a writeable, contiguous float32 buffer
            self._xisf_arr = np.array(arr, dtype=np.float32, copy=True, order="C")
            self._xisf_memmap = None

    # ---------------- common API ----------------
    def read_tile(self, y0, y1, x0, x1) -> np.ndarray:
        if self._kind == "fits":
            d = self._fits_data
            if self.ndim == 2:
                tile = d[y0:y1, x0:x1]
            else:
                sl = [slice(None)]*3
                sl[self._spat_axes[0]] = slice(y0, y1)
                sl[self._spat_axes[1]] = slice(x0, x1)
                tile = d[tuple(sl)]
                if (self._color_axis is not None) and (self._color_axis != 2):
                    tile = np.moveaxis(tile, self._color_axis, -1)
        else:
            if self._xisf_memmap is not None:
                # memmapped (C,H,W) → slice, then move to (H,W,C)
                C = 1 if self.ndim == 2 else self.shape[2]
                if C == 1:
                    tile = self._xisf_memmap[0, y0:y1, x0:x1]
                else:
                    tile = np.moveaxis(self._xisf_memmap[:, y0:y1, x0:x1], 0, -1)
            else:
                # decompressed full array (H,W[,C]) → slice
                tile = self._xisf_arr[y0:y1, x0:x1]  # already HWC or HW

        # late normalize → float32 writeable, contiguous
        if self._scale is None:
            out = np.array(tile, dtype=np.float32, copy=True, order="C")
        else:
            out = np.array(tile, dtype=np.float32, copy=True, order="C")
            out *= self._scale

        # ensure (h,w,3) or (h,w)
        if out.ndim == 3 and out.shape[-1] not in (1, 3):
            if out.shape[0] == 3 and out.shape[-1] != 3:
                out = np.moveaxis(out, 0, -1)
        if out.ndim == 3 and out.shape[-1] == 1:
            out = np.squeeze(out, axis=-1)
        return out

    def read_full(self) -> np.ndarray:
        if self._kind == "fits":
            d = self._fits_data
            if self.ndim == 2:
                full = d
            else:
                full = d
                if (self._color_axis is not None) and (self._color_axis != 2):
                    full = np.moveaxis(full, self._color_axis, -1)
        else:
            if self._xisf_memmap is not None:
                C = 1 if self.ndim == 2 else self.shape[2]
                full = self._xisf_memmap[0] if C == 1 else np.moveaxis(self._xisf_memmap, 0, -1)
            else:
                full = self._xisf_arr

        # late normalize → float32 writeable, contiguous
        if self._scale is None:
            out = np.array(full, dtype=np.float32, copy=True, order="C")
        else:
            out = np.array(full, dtype=np.float32, copy=True, order="C")
            out *= self._scale

        if out.ndim == 3 and out.shape[-1] not in (1, 3):
            if out.shape[0] == 3 and out.shape[-1] != 3:
                out = np.moveaxis(out, 0, -1)
        if out.ndim == 3 and out.shape[-1] == 1:
            out = np.squeeze(out, axis=-1)
        return out

    def close(self):
        try:
            if self._fits_hdul is not None:
                self._fits_hdul.close()
        except Exception:
            pass

def _open_sources_for_mfdeconv(paths, log):
    srcs = []
    try:
        for p in paths:
            srcs.append(_MMImage(p))   # <-- handles FITS or XISF
        return srcs
    except Exception as e:
        # close anything we already opened
        for s in srcs:
            try: s.close()
            except Exception: pass
        raise RuntimeError(f"{e}")

# --- RAW helpers ------------------------------------------------------------

def _rawpy_pattern_to_token(rp) -> Optional[str]:
    """
    Turn rawpy's 2x2 raw_pattern + color_desc into 'RGGB' | 'BGGR' | 'GRBG' | 'GBRG'.
    Returns None if not Bayer.
    """
    try:
        pat = np.array(getattr(rp, "raw_pattern"))
        desc = getattr(rp, "color_desc")  # e.g. b'RGBG' or similar
        if pat.shape != (2, 2) or desc is None:
            return None
        # Map indices in raw_pattern (0..3) to letters in color_desc
        letters = []
        for y in range(2):
            for x in range(2):
                idx = int(pat[y, x])
                c = chr(desc[idx]).upper() if isinstance(desc, (bytes, bytearray)) else str(desc[idx]).upper()
                # Normalize to R/G/B only
                if c.startswith('R'): c = 'R'
                elif c.startswith('G'): c = 'G'
                elif c.startswith('B'): c = 'B'
                else: return None
                letters.append(c)
        token = ''.join(letters)
        return token if token in {"RGGB","BGGR","GRBG","GBRG"} else None
    except Exception:
        return None


def _is_xtrans_from_path(path: str) -> Optional[bool]:
    """
    Returns True if X-Trans, False if definitely Bayer, None if unknown.
    """
    try:
        import rawpy
        with rawpy.imread(path) as rp:
            if getattr(rp, "xtrans_pattern", None) is not None:
                return True
            if getattr(rp, "raw_pattern", None) is not None:
                return False
    except Exception:
        pass
    return None

def _probably_fuji_xtrans(file_path: str, header) -> bool:
    p = str(file_path).lower()
    if p.endswith(".raf"):
        return True  # RAF is Fujifilm; X-Series uses X-Trans (GFX uses Bayer but extension still RAF)
    # Look at header if present
    try:
        make  = str(header.get("MAKE", "") or header.get("CAMERAM", "")).upper()
        model = str(header.get("MODEL", "")).upper()
    except Exception:
        make = model = ""
    if "FUJIFILM" in make and any(t in model for t in (
        "X-T", "X-E", "X-PRO", "X-H", "X100", "X-S", "X30", "X70"
    )):
        return True
    return False


def _rawpy_is_xtrans_or_bayer(rp) -> str:
    """
    Returns 'XTRANS', 'BAYER', or 'UNKNOWN' from a rawpy handle without trusting a single flag.
    """
    try:
        # Prefer explicit X-Trans structure if available
        if getattr(rp, "xtrans_pattern", None) is not None:
            return "XTRANS"
    except Exception:
        pass
    try:
        pat = getattr(rp, "raw_pattern", None)
        if pat is not None and np.asarray(pat).shape == (2, 2):
            return "BAYER"
    except Exception:
        pass
    # LibRaw builds vary; check metadata as a heuristic
    try:
        md = getattr(rp, "metadata", None)
        make  = (md.make or "").upper() if md else ""
        model = (md.model or "").upper() if md else ""
        if "FUJIFILM" in make and ("GFX" not in model):  # GFX medium format is Bayer
            return "XTRANS"
    except Exception:
        pass
    return "UNKNOWN"

def _fmt2(x):
    return "NA" if x is None else f"{float(x):.2f}"


class StackingSuiteDialog(QDialog):
    requestRelaunch = pyqtSignal(str, str)  # old_dir, new_dir
    status_signal = pyqtSignal(str)

    def __init__(self, parent=None, wrench_path=None, spinner_path=None, **_ignored):
        super().__init__(parent)
        self.settings = QSettings()
        self._wrench_path = wrench_path
        self._spinner_path = spinner_path
        self._post_progress_label = None


        self.setWindowTitle("Stacking Suite")
        self.setGeometry(300, 200, 800, 600)

        self.per_group_drizzle = {}
        self.manual_dark_overrides = {}
        self.manual_flat_overrides = {}
        self.conversion_output_directory = None
        self.reg_files = {}
        self.session_tags = {}           # file_path => session_tag
        self.flat_dark_override = {}  # {(group_key:str): path | None | "__NO_DARK__"}
        self.deleted_calibrated_files = []
        self._norm_map = {}
        if not hasattr(self, "_mismatch_policy"):
            self._mismatch_policy = {}
        # Remember GUI thread for fast-path updates
        self._gui_thread = QThread.currentThread()

        # Status bus (singleton across app)
        app = QApplication.instance()
        if not hasattr(app, "_sasd_log_bus"):
            app._sasd_log_bus = LogBus()
        self._log_bus = app._sasd_log_bus

        # Connect status signal ONCE, queued to GUI thread
        try:
            self.status_signal.disconnect()
        except Exception:
            pass
        self.status_signal.connect(self._update_status_gui, Qt.ConnectionType.QueuedConnection)
        self.status_signal.connect(self._log_bus.posted.emit, Qt.ConnectionType.QueuedConnection)

        self._cfa_for_this_run = None  # None = follow checkbox; True/False = override for this run

        # Debounced progress (alignment)
        self._align_prog_timer = QTimer(self)
        self._align_prog_timer.setSingleShot(True)
        self._align_prog_timer.timeout.connect(self._flush_align_progress)
        self._align_prog_pending = None      # tuple[int, int] (done, total)
        self._align_prog_in_slot = False
        self._align_prog_last = None

        self.reference_frame = None
        self._comet_seed = None             # {'path': <original file>, 'xy': (x,y)}
        self._orig2norm = {}                # original path -> normalized *_n.fit
        self._comet_ref_xy = None           # comet coordinate in reference frame

        self.manual_light_files = []
        self._reg_excluded_files = set()

        # Show docking log window (if main window created it)
        self._ensure_log_visible_once()

        # Settings
        self.auto_rot180 = self.settings.value("stacking/auto_rot180", True, type=bool)
        self.auto_rot180_tol_deg = self.settings.value("stacking/auto_rot180_tol_deg", 89.0, type=float)

        dtype_str = self.settings.value("stacking/internal_dtype", "float64", type=str)
        self.internal_dtype = np.float64 if dtype_str == "float64" else np.float32
        self.star_trail_mode = self.settings.value("stacking/star_trail_mode", False, type=bool)

        self.align_refinement_passes = self.settings.value("stacking/refinement_passes", 3, type=int)
        self.align_shift_tolerance = self.settings.value("stacking/shift_tolerance_px", 0.2, type=float)

        # Load or default these
        self.stacking_directory = self.settings.value("stacking/dir", "", type=str)
        self.sigma_high = self.settings.value("stacking/sigma_high", 3.0, type=float)
        self.sigma_low = self.settings.value("stacking/sigma_low", 3.0, type=float)
        self.rejection_algorithm = self.settings.value(
            "stacking/rejection_algorithm", "Weighted Windsorized Sigma Clipping", type=str
        )
        self.kappa = self.settings.value("stacking/kappa", 2.5, type=float)
        self.iterations = self.settings.value("stacking/iterations", 3, type=int)
        self.esd_threshold = self.settings.value("stacking/esd_threshold", 3.0, type=float)
        self.biweight_constant = self.settings.value("stacking/biweight_constant", 6.0, type=float)
        self.trim_fraction = self.settings.value("stacking/trim_fraction", 0.1, type=float)
        self.modz_threshold = self.settings.value("stacking/modz_threshold", 3.5, type=float)
        self.chunk_height = self.settings.value("stacking/chunk_height", 2048, type=int)
        self.chunk_width = self.settings.value("stacking/chunk_width", 2048, type=int)

        # Dictionaries to store file paths
        self.conversion_files = {}
        self.dark_files = {}
        self.flat_files = {}
        self.light_files = {}
        self.master_files = {}
        self.master_sizes = {}

        # Layout & tabs
        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        self.dir_path_edit = QLineEdit(self.stacking_directory)

        # Create tabs
        self.conversion_tab = self.create_conversion_tab()
        self.dark_tab = self.create_dark_tab()
        self.flat_tab = self.create_flat_tab()
        self.light_tab = self.create_light_tab()
        add_runtime_to_sys_path(status_cb=lambda *_: None)
        self.image_integration_tab = self.create_image_registration_tab()

        # Add tabs
        self.tabs.addTab(self.conversion_tab, "Convert Non-FITS Formats")
        self.tabs.addTab(self.dark_tab, "Darks")
        self.tabs.addTab(self.flat_tab, "Flats")
        self.tabs.addTab(self.light_tab, "Lights")
        self.tabs.addTab(self.image_integration_tab, "Image Integration")
        self.tabs.setCurrentIndex(1)

        # Header row
        self.wrench_button = QPushButton()
        self.wrench_button.setIcon(QIcon(self._wrench_path))
        self.wrench_button.setToolTip("Set Stacking Directory & Sigma Clipping")
        self.wrench_button.clicked.connect(self.open_stacking_settings)
        self.wrench_button.setStyleSheet("""
            QPushButton {
                background-color: #FF4500;
                color: white;
                font-size: 16px;
                padding: 8px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #FF6347;
            }
        """)

        header_row = QHBoxLayout()
        header_row.addWidget(self.wrench_button)

        self.stacking_path_display = QLineEdit(self.stacking_directory or "")
        self.stacking_path_display.setReadOnly(True)
        self.stacking_path_display.setPlaceholderText("No stacking folder selected")
        self.stacking_path_display.setFrame(False)
        self.stacking_path_display.setToolTip(self.stacking_directory or "No stacking folder selected")
        header_row.addWidget(self.stacking_path_display, 1)

        layout.addLayout(header_row)

        self.log_btn = QToolButton(self)
        self.log_btn.setText("Open Log")
        self.log_btn.setToolTip("Show the Stacking Suite log window")
        self.log_btn.clicked.connect(self._show_log_window)
        header_row.addWidget(self.log_btn)

        self.tabs.currentChanged.connect(self.on_tab_changed)
        self.restore_saved_master_calibrations()
        self.update_override_dark_combo()  
        self._update_stacking_path_display()

        if self.settings.value("stacking/mfdeconv/after_mf_run_integration", None) is None:
            self.settings.setValue("stacking/mfdeconv/after_mf_run_integration", False)

        # Drizzle UI wiring
        self.drizzle_checkbox.toggled.connect(
            lambda v: (
                self.drizzle_scale_combo.setEnabled(v),
                self.drizzle_drop_shrink_spin.setEnabled(v),
                self.settings.setValue("stacking/drizzle_enabled", bool(v))
            )
        )
        drizzle_on = self.settings.value("stacking/drizzle_enabled", False, type=bool)
        self.drizzle_scale_combo.setEnabled(drizzle_on)
        self.drizzle_drop_shrink_spin.setEnabled(drizzle_on)

        # Comet wiring
        self.comet_cb.toggled.connect(self._on_comet_toggled_public)
        self._on_comet_toggled_public(self.comet_cb.isChecked())
        self.comet_blend_cb.toggled.connect(lambda v: self.settings.setValue("stacking/comet/blend", bool(v)))
        self.comet_mix.valueChanged.connect(lambda v: self.settings.setValue("stacking/comet/mix", float(v)))

        # Multi-frame wiring
        self.mf_enabled_cb.toggled.connect(self._on_mf_toggled_public)
        self._on_mf_toggled_public(self.mf_enabled_cb.isChecked())

        # Mutually exclusive modes
        self.drizzle_checkbox.toggled.connect(lambda v: v and self._apply_mode_enforcement(self.drizzle_checkbox))
        self.comet_cb.toggled.connect(        lambda v: v and self._apply_mode_enforcement(self.comet_cb))
        self.mf_enabled_cb.toggled.connect(   lambda v: v and self._apply_mode_enforcement(self.mf_enabled_cb))
        self.trail_cb.toggled.connect(        lambda v: v and self._apply_mode_enforcement(self.trail_cb))

        for cb in (self.trail_cb, self.comet_cb, self.mf_enabled_cb, self.drizzle_checkbox):
            if cb.isChecked():
                self._apply_mode_enforcement(cb)
                break

        self.use_gpu_integration = self.settings.value("stacking/use_hardware_accel", True, type=bool)
        self._migrate_drizzle_keys_once()

    def ingest_paths_from_blink(self, paths: list[str], target: str):
        """
        Called by the main window when Blink wants to send files over.
        target = "lights"       → Light tab
        target = "integration"  → Image Registration / Integration tab
        """
        if not paths:
            return

        if target == "lights":
            self._ingest_paths_with_progress(
                paths=paths,
                tree=self.light_tree,
                expected_type="LIGHT",
                title="Adding Blink files to Light tab…"
            )
            # same behavior as normal add
            self.assign_best_master_files()
            self.tabs.setCurrentWidget(self.light_tab)
            return

        if target == "integration":
            # treat them as calibrated lights you want to integrate
            self._ingest_paths_with_progress(
                paths=paths,
                tree=self.reg_tree,
                expected_type="LIGHT",
                title="Adding Blink files to Image Integration…"
            )
            self._refresh_reg_tree_summaries()
            self.tabs.setCurrentWidget(self.image_integration_tab)
            return


    def _migrate_drizzle_keys_once(self):
        s = self.settings
        if s.value("stacking/drizzle_pixfrac", None) is None:
            # take whichever existed, prefer Integration tab value
            v = s.value("stacking/drizzle_drop", None, type=float)
            if v is None:
                v = s.value("stacking/drop_shrink", 0.65, type=float)
            self._set_drizzle_pixfrac(v)


    def _on_align_progress(self, done: int, total: int):
        """
        Queued, debounced progress sink. DO NOT emit any signals from here.
        Only update local state and schedule a single UI update via QTimer.
        """
        # Optional: drop exact duplicates to reduce churn
        tup = (int(done), int(total))
        if tup == self._align_prog_last:
            return
        self._align_prog_last = tup

        self._align_prog_pending = tup

        # If you created a QProgressDialog earlier, make sure its range matches total
        if getattr(self, "align_progress", None) and total > 0:
            if self.align_progress.maximum() != total:
                self.align_progress.setRange(0, total)
                self.align_progress.setAutoClose(True)
                self.align_progress.setAutoReset(True)

        # Coalesce bursts to a single UI update on the event loop
        if not self._align_prog_timer.isActive():
            self._align_prog_timer.start(0)

    def _flush_align_progress(self):
        """
        Coalesced progress update for the align QProgressDialog.
        Called by self._align_prog_timer (singleShot).
        """
        if self._align_prog_in_slot or self._align_prog_pending is None:
            return
        self._align_prog_in_slot = True
        try:
            dlg = getattr(self, "align_progress", None)
            if dlg is not None:
                done, total = self._align_prog_pending
                if total > 0:
                    done = max(0, min(int(done), int(total)))
                    dlg.setRange(0, int(total))
                    dlg.setValue(int(done))
                    dlg.setLabelText(f"Aligning stars… ({int(done)}/{int(total)})")
                else:
                    # unknown total: keep it as a pulsing dialog
                    dlg.setRange(0, 0)
                    dlg.setLabelText("Aligning stars…")
        finally:
            self._align_prog_in_slot = False
            self._align_prog_pending = None



    def _hw_accel_enabled(self) -> bool:
        try:
            return bool(self.settings.value("stacking/use_hardware_accel", True, type=bool))
        except Exception:
            return bool(getattr(self, "use_gpu_integration", True))

    def _set_check_safely(self, cb, on: bool):
        """Set a checkbox without firing its signals (prevents recursion)."""
        old = cb.blockSignals(True)
        try:
            cb.setChecked(on)
        finally:
            cb.blockSignals(old)

    def _apply_mode_enforcement(self, who):
        """
        When one of the 'mode' boxes turns ON, uncheck the others,
        persist settings, and re-apply per-mode enable/disable.
        """
        # 1) Uncheck other modes
        mode_boxes = (self.drizzle_checkbox, self.comet_cb, self.mf_enabled_cb, self.trail_cb)
        for cb in mode_boxes:
            if cb is not who:
                self._set_check_safely(cb, False)

        # 2) Persist current state
        self.settings.setValue("stacking/drizzle_enabled",      self.drizzle_checkbox.isChecked())
        self.settings.setValue("stacking/comet/enabled",        self.comet_cb.isChecked())
        self.settings.setValue("stacking/mfdeconv/enabled",     self.mf_enabled_cb.isChecked())
        self.settings.setValue("stacking/star_trail_enabled",   self.trail_cb.isChecked())

        # 3) Re-apply UI gating so widgets match the new state
        self._on_drizzle_checkbox_toggled(self.drizzle_checkbox.isChecked())
        self._on_star_trail_toggled(self.trail_cb.isChecked())
        self._on_comet_toggled_public(self.comet_cb.isChecked())
        self._on_mf_toggled_public(self.mf_enabled_cb.isChecked())

    def _on_comet_toggled_public(self, v: bool):
        """
        Public comet toggle (replaces the inline closure). Also handles the
        'Stars+Comet blend' interlock + persists setting.
        """
        self.comet_pick_btn.setEnabled(v)
        self.comet_blend_cb.setEnabled(v)
        self.comet_mix.setEnabled(v)
        self.settings.setValue("stacking/comet/enabled", bool(v))

    def _on_mf_toggled_public(self, v: bool):
        widgets = [
            getattr(self, "mf_iters_spin", None),
            getattr(self, "mf_min_iters_spin", None),
            getattr(self, "mf_kappa_spin", None),
            getattr(self, "mf_color_combo", None),
            getattr(self, "mf_rho_combo", None),
            getattr(self, "mf_Huber_spin", None),
            getattr(self, "mf_Huber_hint", None),
            getattr(self, "mf_use_star_mask_cb", None),
            getattr(self, "mf_use_noise_map_cb", None),
            getattr(self, "mf_save_intermediate_cb", None),
            getattr(self, "mf_sr_cb", None),  # NEW
        ]
        for w in widgets:
            if w is not None:
                w.setEnabled(v)
        self.settings.setValue("stacking/mfdeconv/enabled", bool(v))

    def _elide_chars(self, s: str, max_chars: int) -> str:
        if not s:
            return ""
        s = s.replace("\n", " ")
        if len(s) <= max_chars:
            return s
        # reserve one char for the ellipsis
        return s[:max_chars - 1] + "…"

    def _dtype(self):
        return self.internal_dtype

    def _dtype_name(self):
        return np.dtype(self.internal_dtype).name

    def _set_last_status(self, message: str):
        disp = self._elide_chars(message, getattr(self, "_last_status_max_chars", 50))
        self._last_status_label.setText(disp)
        # keep full message available on hover
        self._last_status_label.setToolTip(message)


    def _ensure_log_visible_once(self):
        dock = _get_log_dock()
        if dock:
            dock.setVisible(True)
            dock.raise_()

    def _show_log_window(self):
        dock = _get_log_dock()
        if dock:
            dock.setVisible(True)
            dock.raise_()
        else:
            QMessageBox.information(
                self, "Stacking Log",
                "Open the main window to see the Stacking Log dock."
            )

    def _label_with_dims(self, label: str, width: int, height: int) -> str:
        """Replace or append (WxH) in a human label."""
        clean = _DIM_RE.sub("", label).rstrip()
        return f"{clean} ({width}x{height})"

    def _update_stacking_path_display(self):
        txt = self.stacking_directory or ""
        self.stacking_path_display.setText(txt)
        self.stacking_path_display.setToolTip(txt or "No stacking folder selected")

    def restore_saved_master_calibrations(self):
        saved_darks = self.settings.value("stacking/master_darks", [], type=list)
        saved_flats = self.settings.value("stacking/master_flats", [], type=list)

        if saved_darks:
            self.add_master_files(self.master_dark_tree, "DARK", saved_darks)

        if saved_flats:
            self.add_master_files(self.master_flat_tree, "FLAT", saved_flats)

    def create_conversion_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.addWidget(QLabel("Batch Convert Files to Debayered FITS (.fit)"))

        # 1) Create the tree
        self.conversion_tree = QTreeWidget()
        self.conversion_tree.setColumnCount(2)
        self.conversion_tree.setHeaderLabels(["File", "Status"])

        # 2) Make columns user-resizable (Interactive)
        header = self.conversion_tree.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Interactive)

        # 3) After populating the tree, do an initial auto-resize
        self.conversion_tree.resizeColumnToContents(0)
        self.conversion_tree.resizeColumnToContents(1)
        layout.addWidget(self.conversion_tree)

        # Buttons for adding files, adding a directory,
        # selecting an output directory, and clearing the list.
        btn_layout = QHBoxLayout()
        self.add_conversion_files_btn = QPushButton("Add Conversion Files")
        self.add_conversion_files_btn.clicked.connect(self.add_conversion_files)
        self.add_conversion_dir_btn = QPushButton("Add Conversion Directory")
        self.add_conversion_dir_btn.clicked.connect(self.add_conversion_directory)
        self.select_conversion_output_btn = QPushButton("Select Output Directory")
        self.select_conversion_output_btn.clicked.connect(self.select_conversion_output_dir)
        self.clear_conversion_btn = QPushButton("Clear List")
        self.clear_conversion_btn.clicked.connect(self.clear_conversion_list)
        btn_layout.addWidget(self.add_conversion_files_btn)
        btn_layout.addWidget(self.add_conversion_dir_btn)
        btn_layout.addWidget(self.select_conversion_output_btn)
        btn_layout.addWidget(self.clear_conversion_btn)
        layout.addLayout(btn_layout)

        # Convert All button (converts all files in the tree).
        self.convert_btn = QPushButton("Convert All Files to FITS")
        self.convert_btn.clicked.connect(self.convert_all_files)
        layout.addWidget(self.convert_btn)

        return tab

    def add_conversion_files(self):
        last_dir = self.settings.value("last_opened_folder", "", type=str)
        files, _ = QFileDialog.getOpenFileNames(self, "Select Files for Conversion", last_dir,
                                                "Supported Files (*.fits *.fit *.fz *.fz *.fits.gz *.fit.gz *.tiff *.tif *.png *.jpg *.jpeg *.cr2 *.cr3 *.nef *.arw *.dng *.raf *.orf *.rw2 *.pef *.xisf)")
        if files:
            self.settings.setValue("last_opened_folder", os.path.dirname(files[0]))
            for file in files:
                item = QTreeWidgetItem([os.path.basename(file), "Pending"])
                item.setData(0, 1000, file)  # store full path in role 1000
                self.conversion_tree.addTopLevelItem(item)

    def add_conversion_directory(self):
        last_dir = self.settings.value("last_opened_folder", "", type=str)
        directory = QFileDialog.getExistingDirectory(self, "Select Directory for Conversion", last_dir)
        if directory:
            self.settings.setValue("last_opened_folder", directory)
            for file in os.listdir(directory):
                if file.lower().endswith((".fits", ".fit", ".fz", ".fz", ".fit.gz", ".fits.gz", ".tiff", ".tif", ".png", ".jpg", ".jpeg", 
                                           ".cr2", ".cr3", ".nef", ".arw", ".dng", ".raf", ".orf", ".rw2", ".pef", ".xisf")):
                    full_path = os.path.join(directory, file)
                    item = QTreeWidgetItem([file, "Pending"])
                    item.setData(0, 1000, full_path)
                    self.conversion_tree.addTopLevelItem(item)

    def select_conversion_output_dir(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Conversion Output Directory")
        if directory:
            self.conversion_output_directory = directory
            self.update_status(f"Conversion output directory set to: {directory}")

    def clear_conversion_list(self):
        self.conversion_tree.clear()
        self.update_status("Conversion list cleared.")

    def convert_all_files(self):
        # If no output directory is set, ask the user if they want to set it now.
        if not self.conversion_output_directory:
            reply = QMessageBox.question(
                self,
                "No Output Directory",
                "No output directory is set. Do you want to select one now?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.select_conversion_output_dir()  # Let them pick a folder
            else:
                # They chose 'No' → just stop
                return

            # If it's still empty after that, bail out
            if not self.conversion_output_directory:
                QMessageBox.warning(self, "No Output Directory", "Please select a conversion output directory first.")
                return

        count = self.conversion_tree.topLevelItemCount()
        if count == 0:
            QMessageBox.information(self, "No Files", "There are no files to convert.")
            return

        # 1) Show the batch settings dialog
        dialog = BatchSettingsDialog(self)
        result = dialog.exec()
        if result == int(QDialog.DialogCode.Rejected):
            # user canceled
            return
        # user pressed OK => get the values
        imagetyp_user, exptime_user, filter_user = dialog.get_values()

        for i in range(count):
            item = self.conversion_tree.topLevelItem(i)
            file_path = item.data(0, 1000)
            result = load_image(file_path)
            if result[0] is None:
                item.setText(1, "Failed to load")
                self.update_status(f"Failed to load {os.path.basename(file_path)}")
                continue

            image, header, bit_depth, is_mono = result

            if image is None:
                item.setText(1, "Failed to load")
                self.update_status(f"Failed to load {os.path.basename(file_path)}")
                continue

            # 🔹 If the file has no header (TIFF, PNG, JPG, etc.), create a minimal one
            if header is None:
                header = fits.Header()
                header["SIMPLE"]   = True
                header["BITPIX"]   = 16  # Or 16, depending on your preference
                header["CREATOR"]  = "SetiAstroSuite"
                header["IMAGETYP"] = "UNKNOWN"  # We'll set it properly below
                header["EXPTIME"]  = "Unknown"  # Just a placeholder
                # You can add more default keywords as needed

            # Debayer if needed:
            image = self.debayer_image(image, file_path, header)
            if image.ndim == 3:
                is_mono = False

            if image.ndim == 3:
                header['COLOR'] = True
                header['NAXIS'] = 3
                header['NAXIS1'] = image.shape[1]
                header['NAXIS2'] = image.shape[0]
                header['NAXIS3'] = image.shape[2]
            else:
                header['COLOR'] = False
                header['NAXIS'] = 2
                header['NAXIS1'] = image.shape[1]
                header['NAXIS2'] = image.shape[0]

            # If it's a RAW format, definitely treat as color
            if file_path.lower().endswith(('.cr2', '.cr3', '.nef', '.arw', '.dng', '.raf', '.orf', '.rw2', '.pef')):
                is_mono = False

                # Try extracting EXIF metadata
                try:
                    with open(file_path, 'rb') as f:
                        tags = exifread.process_file(f, details=False)

                    exptime_tag = tags.get("EXIF ExposureTime")  # e.g. "1/125"
                    iso_tag = tags.get("EXIF ISOSpeedRatings")
                    date_obs_tag = tags.get("EXIF DateTimeOriginal")

                    # Create or replace with a fresh header, but keep some existing fields if desired
                    new_header = fits.Header()
                    new_header['SIMPLE'] = True
                    new_header['BITPIX'] = 16
                    new_header['IMAGETYP'] = header.get('IMAGETYP', "UNKNOWN")

                    # Attempt to parse exptime. If fraction or numeric fails, store 'Unknown'.
                    if exptime_tag:
                        exptime_str = str(exptime_tag.values)  # or exptime_tag.printable
                        # Attempt fraction or float
                        try:
                            if '/' in exptime_str:  
                                # e.g. "1/125"
                                top, bot = exptime_str.split('/', 1)
                                fexp = float(top) / float(bot)
                                new_header['EXPTIME'] = (fexp, "Exposure Time in seconds")
                            else:
                                # e.g. "0.008" or "8"
                                fexp = float(exptime_str)
                                new_header['EXPTIME'] = (fexp, "Exposure Time in seconds")
                        except (ValueError, ZeroDivisionError):
                            new_header['EXPTIME'] = 'Unknown'
                    # If no exptime_tag, set Unknown
                    else:
                        new_header['EXPTIME'] = 'Unknown'

                    if iso_tag:
                        new_header['ISO'] = str(iso_tag.values)
                    if date_obs_tag:
                        new_header['DATE-OBS'] = str(date_obs_tag.values)

                    # Replace old header with new
                    header = new_header

                except Exception as e:
                    # If exif extraction fails for any reason, we just keep the existing header
                    # but ensure we set EXPTIME if missing
                    self.update_status(f"Warning: Failed to extract RAW header from {os.path.basename(file_path)}: {e}")

            header['IMAGETYP'] = imagetyp_user
            header['FILTER'] = filter_user

            # For exptime_user, try to parse float or fraction
            try:
                if '/' in exptime_user:
                    top, bot = exptime_user.split('/', 1)
                    exptime_val = float(top) / float(bot)
                    header['EXPTIME'] = (exptime_val, "User-specified exposure (s)")
                else:
                    exptime_val = float(exptime_user)
                    header['EXPTIME'] = (exptime_val, "User-specified exposure (s)")
            except (ValueError, ZeroDivisionError):
                # If user typed "Unknown" or something non-numeric
                header['EXPTIME'] = exptime_user

            # Remove any existing NAXIS keywords
            for key in ["NAXIS", "NAXIS1", "NAXIS2", "NAXIS3"]:
                header.pop(key, None)

            if image.ndim == 2:
                header['NAXIS'] = 2
                header['NAXIS1'] = image.shape[1]
                header['NAXIS2'] = image.shape[0]
            elif image.ndim == 3:
                header['NAXIS'] = 3
                header['NAXIS1'] = image.shape[1]
                header['NAXIS2'] = image.shape[0]
                header['NAXIS3'] = image.shape[2]

            # -- Ensure EXPTIME is defined --
            if 'EXPTIME' not in header:
                # If the camera or exif didn't provide it, we set it to 'Unknown'
                header['EXPTIME'] = 'Unknown'

            # Build output filename and save
            base = os.path.basename(file_path)
            name, _ = os.path.splitext(base)
            output_filename = os.path.join(self.conversion_output_directory, f"{name}.fit")
            maxv = float(np.max(image))
            if maxv > 0:
                image = image / maxv

            try:
                save_image(
                    img_array=image,
                    filename=output_filename,
                    original_format="fit",
                    bit_depth="16-bit",
                    original_header=header,
                    is_mono=is_mono
                )
                item.setText(1, "Converted")
                self.update_status(
                    f"Converted {os.path.basename(file_path)} to FITS with "
                    f"IMAGETYP={header['IMAGETYP']}, EXPTIME={header['EXPTIME']}."
                )
            except Exception as e:
                item.setText(1, f"Error: {e}")
                self.update_status(f"Error converting {os.path.basename(file_path)}: {e}")

            QApplication.processEvents()

        self.update_status("Conversion complete.")



    def debayer_image(self, image, file_path, header):
        """
        Returns RGB if debayered, otherwise returns input unchanged.
        Also stamps header with BAYERPAT or XTRANS where possible.
        """
        # per-run override if set, else honor the checkbox
        if self._cfa_for_this_run is None:
            cfa = bool(getattr(self, "cfa_drizzle_cb", None) and self.cfa_drizzle_cb.isChecked())
        else:
            cfa = bool(self._cfa_for_this_run)
        print(f"[DEBUG] Debayering with CFA drizzle = {cfa}")

        ext = file_path.lower()
        is_raw = ext.endswith(('.cr2','.cr3','.nef','.arw','.dng','.raf','.orf','.rw2','.pef'))

        # --- RAW files ---
        if is_raw:
            try:
                import rawpy
                with rawpy.imread(file_path) as rp:
                    fam = _rawpy_is_xtrans_or_bayer(rp)
                    if fam == "BAYER":
                        # Try to get a concrete 2x2 token
                        token = _rawpy_pattern_to_token(rp)
                        if token:
                            try: header['BAYERPAT'] = token
                            except Exception: pass
                            try:
                                from legacy.numba_utils import debayer_raw_fast
                                return debayer_raw_fast(image, bayer_pattern=token, cfa_drizzle=cfa, method="edge")
                            except Exception:
                                return debayer_fits_fast(image, token, cfa_drizzle=cfa)

                    # X-Trans or ambiguous → treat Fuji RAF/X-Series as X-Trans
                    if fam == "XTRANS" or _probably_fuji_xtrans(file_path, header):
                        print("[INFO] Demosaicing via rawpy (X-Trans path).")
                        rgb16 = rp.postprocess(
                            demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,  # or DHT
                            no_auto_bright=True,
                            gamma=(1.0, 1.0),
                            output_bps=16,
                            use_camera_wb=True,
                            fbdd_noise_reduction=rawpy.FBDDNoiseReductionMode.Off,
                            four_color_rgb=False,
                            half_size=False,
                            bright=1.0,
                            highlight_mode=rawpy.HighlightMode.Clip
                        )
                        rgb = (rgb16.astype(np.float32) / 65535.0)
                        try: header['XTRANS'] = True
                        except Exception: pass
                        return rgb

                    # If still unknown, last attempt: if we do have a 2x2 pattern, use it
                    token = _rawpy_pattern_to_token(rp)
                    if token:
                        try: header['BAYERPAT'] = token
                        except Exception: pass
                        try:
                            from legacy.numba_utils import debayer_raw_fast
                            return debayer_raw_fast(image, bayer_pattern=token, cfa_drizzle=cfa, method="edge")
                        except Exception:
                            return debayer_fits_fast(image, token, cfa_drizzle=cfa)

                    print("[WARN] RAW family unknown; leaving as-is.")
            except Exception as e:
                print(f"[WARN] rawpy probe failed for {file_path}: {e}")
                # fall through

        # --- FITS (already mosaic) ---
        if ext.endswith(('.fits','.fit','.fz')):
            bp = (str(header.get('BAYERPAT') or header.get('BAYERPATN') or header.get('CFA_PATTERN') or "")).upper()
            if bp in {"RGGB","BGGR","GRBG","GBRG"}:
                try:
                    from legacy.numba_utils import debayer_raw_fast
                    return debayer_raw_fast(image, bayer_pattern=bp, cfa_drizzle=cfa, method="edge")
                except Exception:
                    return debayer_fits_fast(image, bp, cfa_drizzle=cfa)

            # If FITS came from Fuji RAW (RAF) and no BAYERPAT, it was likely X-Trans;
            # without the RAW we cannot demosaic now, so keep mosaic but mark it.
            if _probably_fuji_xtrans(file_path, header):
                try: header['XTRANS'] = True
                except Exception: pass
                print("[INFO] FITS likely from X-Trans; cannot demosaic without RAW. Keeping mosaic.")
                return image

        return image



    def setup_status_bar(self, layout):
        """ Sets up a scrollable status log at the bottom of the UI. """
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        self.status_text.setStyleSheet(
            "background-color: black; color: white; font-family: Monospace; padding: 4px;"
        )

        self.status_scroll = QScrollArea()
        self.status_scroll.setWidgetResizable(True)
        self.status_scroll.setWidget(self.status_text)
        # Make the scroll area respect a fixed height
        self.status_scroll.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed
        )
        layout.addWidget(self.status_scroll)

        # show ~10 lines
        self.set_status_visible_lines(6)

    def set_status_visible_lines(self, n_lines: int):
        fm = QFontMetrics(self.status_text.font())
        line_h = fm.lineSpacing()

        # Add margins/frames (a small fudge keeps things from clipping)
        frame = self.status_text.frameWidth()
        docm  = int(self.status_text.document().documentMargin())
        extra = 2 * frame + 2 * docm + 8

        self.status_scroll.setFixedHeight(int(n_lines * line_h + extra))

    @pyqtSlot(str)
    def _update_status_gui(self, message: str):
        # Update in-dialog status label if you have one
        if hasattr(self, "_last_status_label") and self._last_status_label:
            self._last_status_label.setText(message)

        # Optional: your own helper for status breadcrumb/history
        if hasattr(self, "_set_last_status"):
            try:
                self._set_last_status(message)
            except Exception:
                pass


    def update_status(self, message: str):
        """
        Thread-safe status update:
        • If already on GUI thread -> update immediately + push to log bus directly
        • If from worker thread    -> emit once; slots are queued to GUI + log
        """
        if QThread.currentThread() is self._gui_thread:
            # Immediate UI update
            self._update_status_gui(message)
            # Send to log window directly (don’t signal back into ourselves)
            try:
                self._log_bus.posted.emit(message)
            except Exception:
                pass
        else:
            # One queued signal fan-out to GUI + log
            self.status_signal.emit(message)


    @pyqtSlot(str)
    def _on_post_status(self, msg: str):
        # 1) your central logger
        self.update_status(msg)
        # 2) also reflect in the progress dialog label if it exists
        try:
            if getattr(self, "post_progress", None):
                self.post_progress.setLabelText(msg)
                QApplication.processEvents()
        except Exception:
            pass


    def _norm_dir(self, p: str) -> str:
        if not p:
            return ""
        p = os.path.expanduser(os.path.expandvars(p))
        p = os.path.abspath(p)
        p = os.path.normpath(p)
        if os.name == "nt":
            p = p.lower()
        return p

    def _choose_dir_into(self, line_edit: QLineEdit):
        d = QFileDialog.getExistingDirectory(self, "Select Stacking Directory",
                                            line_edit.text() or self.stacking_directory or "")
        if d:
            line_edit.setText(d)

    def open_stacking_settings(self):
        """Opens a 2-column Stacking Settings dialog."""
        from PyQt6.QtWidgets import (
            QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox, QFormLayout,
            QLabel, QLineEdit, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
            QCheckBox, QDialogButtonBox, QScrollArea, QWidget
        )
        dialog = QDialog(self)
        dialog.setWindowTitle("Stacking Settings")

        # Top-level layout with a scroll area (nice for small screens)
        root = QVBoxLayout(dialog)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        root.addWidget(scroll)
        body = QWidget()
        scroll.setWidget(body)

        cols = QHBoxLayout(body)
        left_col = QVBoxLayout()
        right_col = QVBoxLayout()
        cols.addLayout(left_col, 1)
        cols.addSpacing(12)
        cols.addLayout(right_col, 1)

        # ========== LEFT COLUMN ==========
        # --- General ---
        gb_general = QGroupBox("General")
        fl_general = QFormLayout(gb_general)

        # Stacking directory
        dir_row = QHBoxLayout()
        dir_edit = QLineEdit(self.stacking_directory or "")
        dialog._dir_edit = dir_edit
        btn_browse = QPushButton("Browse")
        btn_browse.clicked.connect(lambda: self._choose_dir_into(dir_edit))
        dir_row.addWidget(dir_edit, 1)
        dir_row.addWidget(btn_browse)
        fl_general.addRow(QLabel("Stacking Directory:"), QWidget())
        fl_general.addRow(dir_row)

        # Precision
        self.precision_combo = QComboBox()
        self.precision_combo.addItems(["32-bit float", "64-bit float"])
        self.precision_combo.setCurrentIndex(1 if self.internal_dtype is np.float64 else 0)
        self.precision_combo.setToolTip("64-bit uses ~2× RAM; 32-bit is faster/lighter.")
        fl_general.addRow("Internal Precision:", self.precision_combo)

        # Chunk sizes
        self.chunkHeightSpinBox = QSpinBox()
        self.chunkHeightSpinBox.setRange(128, 8192)
        self.chunkHeightSpinBox.setValue(self.settings.value("stacking/chunk_height", 2048, type=int))
        self.chunkWidthSpinBox = QSpinBox()
        self.chunkWidthSpinBox.setRange(128, 8192)
        self.chunkWidthSpinBox.setValue(self.settings.value("stacking/chunk_width", 2048, type=int))
        hw_row = QHBoxLayout()
        hw_row.addWidget(QLabel("H:")); hw_row.addWidget(self.chunkHeightSpinBox)
        hw_row.addSpacing(8)
        hw_row.addWidget(QLabel("W:")); hw_row.addWidget(self.chunkWidthSpinBox)
        w_hw = QWidget(); w_hw.setLayout(hw_row)
        fl_general.addRow("Chunk Size:", w_hw)

        left_col.addWidget(gb_general)

        # --- Distortion / Transform model ---
        # --- Distortion / Transform model ---
        disto_box  = QGroupBox("Distortion / Transform")
        disto_form = QFormLayout(disto_box)

        self.align_model_combo = QComboBox()
        # Order matters for index mapping below
        self.align_model_combo.addItems([
            "Affine (fast)",
            "Homography (projective)",
            "Polynomial 3rd-order",
            "Polynomial 4th-order",
        ])

        # Map saved string -> index
        _saved_model = (self.settings.value("stacking/align/model", "affine") or "affine").lower()
        _model_to_idx = {
            "affine": 0,
            "homography": 1,
            "poly3": 2,
            "poly4": 3,
        }
        self.align_model_combo.setCurrentIndex(_model_to_idx.get(_saved_model, 0))
        disto_form.addRow("Model:", self.align_model_combo)

        # Shared
        self.align_max_cp = QSpinBox()
        self.align_max_cp.setRange(20, 2000)
        self.align_max_cp.setValue(self.settings.value("stacking/align/max_cp", 250, type=int))
        disto_form.addRow("Max control points:", self.align_max_cp)

        self.align_downsample = QSpinBox()
        self.align_downsample.setRange(1, 8)
        self.align_downsample.setValue(self.settings.value("stacking/align/downsample", 2, type=int))
        disto_form.addRow("Solve downsample:", self.align_downsample)

        # Homography-specific
        self.h_ransac_reproj = QDoubleSpinBox()
        self.h_ransac_reproj.setRange(0.1, 10.0)
        self.h_ransac_reproj.setDecimals(2)
        self.h_ransac_reproj.setSingleStep(0.1)
        self.h_ransac_reproj.setValue(self.settings.value("stacking/align/h_reproj", 3.0, type=float))
        self._h_label = QLabel("Homog. RANSAC reproj (px):")
        disto_form.addRow(self._h_label, self.h_ransac_reproj)

        # (Removed TPS controls)

        def _toggle_disto_rows():
            m = self.align_model_combo.currentIndex()
            is_h = (m == 1)             # homography
            # enable only homography controls when selected
            self._h_label.setEnabled(is_h)
            self.h_ransac_reproj.setEnabled(is_h)

        _toggle_disto_rows()
        self.align_model_combo.currentIndexChanged.connect(lambda _: _toggle_disto_rows())

        left_col.addWidget(disto_box)


        # --- Alignment ---
        gb_align = QGroupBox("Alignment")
        fl_align = QFormLayout(gb_align)

        self.align_passes_combo = QComboBox()
        self.align_passes_combo.addItems(["Fast (1 pass)", "Accurate (3 passes)"])
        curr_passes = self.settings.value("stacking/refinement_passes", 3, type=int)
        self.align_passes_combo.setCurrentIndex(0 if curr_passes <= 1 else 1)
        self.align_passes_combo.setToolTip("Fast = single pass; Accurate = 3-pass refinement.")
        fl_align.addRow("Refinement:", self.align_passes_combo)

        self.shift_tol_spin = QDoubleSpinBox()
        self.shift_tol_spin.setRange(0.05, 5.0)
        self.shift_tol_spin.setDecimals(2)
        self.shift_tol_spin.setSingleStep(0.05)
        self.shift_tol_spin.setValue(self.settings.value("stacking/shift_tolerance", 0.2, type=float))
        fl_align.addRow("Accept tolerance (px):", self.shift_tol_spin)

        # Sigma high/low
        self.sigma_high_spinbox = QDoubleSpinBox()
        self.sigma_high_spinbox.setRange(0.1, 10.0)
        self.sigma_high_spinbox.setDecimals(2)
        self.sigma_high_spinbox.setValue(self.sigma_high)
        self.sigma_low_spinbox = QDoubleSpinBox()
        self.sigma_low_spinbox.setRange(0.1, 10.0)
        self.sigma_low_spinbox.setDecimals(2)
        self.sigma_low_spinbox.setValue(self.sigma_low)
        hs_row = QHBoxLayout()
        hs_row.addWidget(QLabel("High:")); hs_row.addWidget(self.sigma_high_spinbox)
        hs_row.addSpacing(8)
        hs_row.addWidget(QLabel("Low:")); hs_row.addWidget(self.sigma_low_spinbox)
        w_hs = QWidget(); w_hs.setLayout(hs_row)
        fl_align.addRow("Sigma Clipping:", w_hs)

        left_col.addWidget(gb_align)
        

        # --- Performance ---
        gb_perf = QGroupBox("Performance")
        fl_perf = QFormLayout(gb_perf)

        self.hw_accel_cb = QCheckBox("Use hardware acceleration if available")
        self.hw_accel_cb.setToolTip("Enable GPU/MPS via PyTorch when supported; falls back to CPU automatically.")
        self.hw_accel_cb.setChecked(self.settings.value("stacking/use_hardware_accel", True, type=bool))
        fl_perf.addRow(self.hw_accel_cb)

        # NEW: MFDeconv engine choice (radio buttons)
        eng_box = QGroupBox("MFDeconv Engine")
        eng_row = QHBoxLayout(eng_box)
        self.mf_eng_normal_rb = QRadioButton("Normal")
        self.mf_eng_cudnn_rb  = QRadioButton("Normal (cuDNN-free)")
        self.mf_eng_sport_rb  = QRadioButton("High-Octane (Let ’er rip)")

        # restore from settings (default "normal")
        _saved_eng = (self.settings.value("stacking/mfdeconv/engine", "normal", type=str) or "normal").lower()
        if   _saved_eng == "cudnn": self.mf_eng_cudnn_rb.setChecked(True)
        elif _saved_eng == "sport":      self.mf_eng_sport_rb.setChecked(True)
        else:                            self.mf_eng_normal_rb.setChecked(True)

        eng_row.addWidget(self.mf_eng_normal_rb)
        eng_row.addWidget(self.mf_eng_cudnn_rb)
        eng_row.addWidget(self.mf_eng_sport_rb)
        fl_perf.addRow(eng_box)

        # (Optional) show detected backend for user feedback
        try:
            backend_str = current_backend() or "CPU only"
        except Exception:
            backend_str = "CPU only"
        fl_perf.addRow("Detected backend:", QLabel(backend_str))

        left_col.addWidget(gb_perf)
        left_col.addStretch(1)



        # ========== RIGHT COLUMN ==========
        # --- Normalization & Gradient (ABE poly2) ---
        gb_normgrad = QGroupBox("Normalization & Gradient (ABE Poly²)")
        fl_ng = QFormLayout(gb_normgrad)

        # master enable
        self.chk_poly2 = QCheckBox("Remove background gradient (ABE Poly²)")
        self.chk_poly2.setChecked(self.settings.value("stacking/grad_poly2/enabled", False, type=bool))
        fl_ng.addRow(self.chk_poly2)

        # mode (subtract vs divide)
        self.grad_mode_combo = QComboBox()
        self.grad_mode_combo.addItems(["Subtract (additive)", "Divide (flat-like)"])
        _saved_mode = self.settings.value("stacking/grad_poly2/mode", "subtract")
        self.grad_mode_combo.setCurrentIndex(0 if _saved_mode.lower() != "divide" else 1)
        fl_ng.addRow("Mode:", self.grad_mode_combo)

        # ABE-style controls
        self.grad_samples_spin = QSpinBox()
        self.grad_samples_spin.setRange(20, 600)
        self.grad_samples_spin.setValue(self.settings.value("stacking/grad_poly2/samples", 120, type=int))
        fl_ng.addRow("Sample points:", self.grad_samples_spin)

        self.grad_downsample_spin = QSpinBox()
        self.grad_downsample_spin.setRange(1, 16)
        self.grad_downsample_spin.setValue(self.settings.value("stacking/grad_poly2/downsample", 6, type=int))
        fl_ng.addRow("Downsample (AREA):", self.grad_downsample_spin)

        self.grad_patch_spin = QSpinBox()
        self.grad_patch_spin.setRange(5, 51)
        self.grad_patch_spin.setSingleStep(2)
        self.grad_patch_spin.setValue(self.settings.value("stacking/grad_poly2/patch_size", 15, type=int))
        fl_ng.addRow("Patch size (small):", self.grad_patch_spin)

        self.grad_min_strength = QDoubleSpinBox()
        self.grad_min_strength.setRange(0.0, 0.20)
        self.grad_min_strength.setDecimals(3)
        self.grad_min_strength.setSingleStep(0.005)
        self.grad_min_strength.setValue(self.settings.value("stacking/grad_poly2/min_strength", 0.01, type=float))
        fl_ng.addRow("Skip if strength <", self.grad_min_strength)

        # division-only gain clip
        self.grad_gain_lo = QDoubleSpinBox()
        self.grad_gain_lo.setRange(0.01, 1.00); self.grad_gain_lo.setDecimals(2); self.grad_gain_lo.setSingleStep(0.01)
        self.grad_gain_lo.setValue(self.settings.value("stacking/grad_poly2/gain_lo", 0.20, type=float))
        self.grad_gain_hi = QDoubleSpinBox()
        self.grad_gain_hi.setRange(1.0, 25.0); self.grad_gain_hi.setDecimals(1); self.grad_gain_hi.setSingleStep(0.5)
        self.grad_gain_hi.setValue(self.settings.value("stacking/grad_poly2/gain_hi", 5.0, type=float))

        row_gain = QWidget()
        row_gain_h = QHBoxLayout(row_gain); row_gain_h.setContentsMargins(0,0,0,0)
        row_gain_h.addWidget(QLabel("Clip (lo/hi):"))
        row_gain_h.addWidget(self.grad_gain_lo)
        row_gain_h.addWidget(QLabel(" / "))
        row_gain_h.addWidget(self.grad_gain_hi)
        fl_ng.addRow("Divide gain limits:", row_gain)

        # enable/disable
        def _toggle_grad_enabled(on: bool):
            for w in (self.grad_mode_combo, self.grad_samples_spin, self.grad_downsample_spin,
                    self.grad_patch_spin, self.grad_min_strength, row_gain):
                w.setEnabled(on)

        def _toggle_gain_row():
            is_div = (self.grad_mode_combo.currentIndex() == 1)
            row_gain.setVisible(is_div)
            row_gain.setEnabled(self.chk_poly2.isChecked() and is_div)

        self.chk_poly2.toggled.connect(_toggle_grad_enabled)
        self.grad_mode_combo.currentIndexChanged.connect(lambda _: _toggle_gain_row())

        _toggle_grad_enabled(self.chk_poly2.isChecked())
        _toggle_gain_row()

        left_col.addWidget(gb_normgrad)

        gb_drizzle = QGroupBox("Drizzle")
        fl_dz = QFormLayout(gb_drizzle)

        self.drizzle_kernel_combo = QComboBox()
        self.drizzle_kernel_combo.addItems(["Square (pixfrac)", "Circular (disk)", "Gaussian"])
        # restore
        _saved_k = self.settings.value("stacking/drizzle_kernel", "square").lower()
        if _saved_k.startswith("gauss"): self.drizzle_kernel_combo.setCurrentIndex(2)
        elif _saved_k.startswith("circ"): self.drizzle_kernel_combo.setCurrentIndex(1)
        else: self.drizzle_kernel_combo.setCurrentIndex(0)
        fl_dz.addRow("Kernel:", self.drizzle_kernel_combo)

        self.drop_shrink_spin = QDoubleSpinBox()
        self.drop_shrink_spin.setRange(0.0, 1.0)  # make this the same concept: pixfrac
        self.drop_shrink_spin.setDecimals(3)
        self.drop_shrink_spin.setValue(self._get_drizzle_pixfrac())
        self.drop_shrink_spin.valueChanged.connect(lambda v: self._set_drizzle_pixfrac(v))
        fl_dz.addRow("Kernel width:", self.drop_shrink_spin)

        # Optional: a separate σ for Gaussian (if you want it distinct)
        self.gauss_sigma_spin = QDoubleSpinBox()
        self.gauss_sigma_spin.setRange(0.05, 3.0)
        self.gauss_sigma_spin.setDecimals(3)
        self.gauss_sigma_spin.setSingleStep(0.05)
        self.gauss_sigma_spin.setValue(self.settings.value("stacking/drizzle_gauss_sigma",
                                                        self.drop_shrink_spin.value()*0.5, type=float))
        fl_dz.addRow("Gaussian σ (px):", self.gauss_sigma_spin)

        def _toggle_gauss_sigma():
            self.gauss_sigma_spin.setEnabled(self.drizzle_kernel_combo.currentIndex()==2)
        _toggle_gauss_sigma()
        self.drizzle_kernel_combo.currentIndexChanged.connect(lambda _ : _toggle_gauss_sigma())

        right_col.addWidget(gb_drizzle)

        # --- MF Deconvolution  ---
        gb_mf = QGroupBox("Multi-frame Deconvolution")
        fl_mf = QFormLayout(gb_mf)
        def _row(lbl, w):
            c = QWidget(); h = QHBoxLayout(c); h.setContentsMargins(0,0,0,0); h.addWidget(w, 1); return (lbl, c)
        
        self.mf_seed_combo = QComboBox()
        self.mf_seed_combo.addItems([
            "Robust μ–σ (live stack)",
            "Median (Sukhdeep et al.)"
        ])
        # Persisted value → UI
        seed_mode_saved = str(self.settings.value("stacking/mfdeconv/seed_mode", "robust"))
        seed_idx = 0 if seed_mode_saved.lower() != "median" else 1
        self.mf_seed_combo.setCurrentIndex(seed_idx)
        self.mf_seed_combo.setToolTip(
            "Choose the initial seed image for MFDeconv:\n"
            "• Robust μ–σ: running mean with sigma clipping (RAM-friendly, default)\n"
            "• Median: tiled median stack (more outlier-resistant; heavier I/O, esp. for XISF)"
        )
        fl_mf.addRow(*_row("Seed image:", self.mf_seed_combo))

        self.sm_thresh = QDoubleSpinBox(); self.sm_thresh.setRange(0.1, 20.0); self.sm_thresh.setDecimals(2)
        self.sm_thresh.setValue(
            self.settings.value("stacking/mfdeconv/star_mask/thresh_sigma", _SM_DEF_THRESH, type=float)
        )
        fl_mf.addRow(*_row("Star detect σ:", self.sm_thresh))

        self.sm_grow = QSpinBox(); self.sm_grow.setRange(0, 128)
        self.sm_grow.setValue(
            self.settings.value("stacking/mfdeconv/star_mask/grow_px", _SM_DEF_GROW, type=int)
        )
        fl_mf.addRow(*_row("Dilate (+px):", self.sm_grow))

        self.sm_soft = QDoubleSpinBox(); self.sm_soft.setRange(0.0, 10.0); self.sm_soft.setDecimals(2)
        self.sm_soft.setValue(
            self.settings.value("stacking/mfdeconv/star_mask/soft_sigma", _SM_DEF_SOFT, type=float)
        )
        fl_mf.addRow(*_row("Feather σ (px):", self.sm_soft))

        self.sm_rmax = QSpinBox(); self.sm_rmax.setRange(2, 256)
        self.sm_rmax.setValue(
            self.settings.value("stacking/mfdeconv/star_mask/max_radius_px", _SM_DEF_RMAX, type=int)
        )
        fl_mf.addRow(*_row("Max star radius (px):", self.sm_rmax))

        self.sm_maxobjs = QSpinBox(); self.sm_maxobjs.setRange(10, 50000)
        self.sm_maxobjs.setValue(
            self.settings.value("stacking/mfdeconv/star_mask/max_objs", _SM_DEF_MAXOBJS, type=int)
        )
        fl_mf.addRow(*_row("Max stars kept:", self.sm_maxobjs))

        self.sm_keepfloor = QDoubleSpinBox(); self.sm_keepfloor.setRange(0.0, 0.95); self.sm_keepfloor.setDecimals(3)
        self.sm_keepfloor.setValue(
            self.settings.value("stacking/mfdeconv/star_mask/keep_floor", _SM_DEF_KEEPF, type=float)
        )
        self.sm_keepfloor.setToolTip("Lower = stronger masking near stars; 0 = hard mask, 0.2 = gentle.")
        fl_mf.addRow(*_row("Keep-floor:", self.sm_keepfloor))

        # (optional) expose ellipse scale if you like:
        self.sm_es = QDoubleSpinBox(); self.sm_es.setRange(0.5, 3.0); self.sm_es.setDecimals(2)
        self.sm_es.setValue(
            self.settings.value("stacking/mfdeconv/star_mask/ellipse_scale", _SM_DEF_ES, type=float)
        )
        fl_mf.addRow(*_row("Ellipse scale:", self.sm_es))

        # --- Variance map tuning ---
        self.vm_stride = QSpinBox(); self.vm_stride.setRange(1, 64)
        self.vm_stride.setValue(
            self.settings.value("stacking/mfdeconv/varmap/sample_stride", _VM_DEF_STRIDE, type=int)
        )
        fl_mf.addRow(*_row("VarMap sample stride:", self.vm_stride))

        self.vm_sigma = QDoubleSpinBox(); self.vm_sigma.setRange(0.0, 5.0); self.vm_sigma.setDecimals(2)
        self.vm_sigma.setValue(self.settings.value("stacking/mfdeconv/varmap/smooth_sigma", 1.0, type=float))
        fl_mf.addRow(*_row("VarMap smooth σ:", self.vm_sigma))

        self.vm_floor_log = QDoubleSpinBox()
        self.vm_floor_log.setRange(-12.0, -2.0)
        self.vm_floor_log.setDecimals(2)
        self.vm_floor_log.setSingleStep(0.5)
        self.vm_floor_log.setValue(math.log10(
            self.settings.value("stacking/mfdeconv/varmap/floor", 1e-8, type=float)
        ))
        self.vm_floor_log.setToolTip("log10 of variance floor (DN²). -8 ≡ 1e-8.")
        fl_mf.addRow(*_row("VarMap floor (log10):", self.vm_floor_log))

        btn_mf_reset = QPushButton("Reset MFDeconv to Recommended")
        btn_mf_reset.setToolTip(
            "Restore MFDeconv star mask + variance map tuning to the recommended defaults."
        )

        def _reset_mfdeconv_defaults():
            # Star mask tuning
            self.mf_seed_combo.setCurrentIndex(0)
            self.sm_thresh.setValue(4.5)     # Star detect σ
            self.sm_grow.setValue(6)         # Dilate (+px)
            self.sm_soft.setValue(3.0)       # Feather σ (px)
            self.sm_rmax.setValue(36)        # Max star radius (px)
            self.sm_maxobjs.setValue(5000)   # Max stars kept
            self.sm_keepfloor.setValue(0.015)# Keep-floor
            self.sm_es.setValue(1.12)        # Ellipse scale

            # Variance map tuning
            self.vm_stride.setValue(8)       # VarMap sample stride
            self.vm_sigma.setValue(3.0)      # VarMap smooth σ
            self.vm_floor_log.setValue(-10)  # VarMap floor (log10)

            # (Optional) preload QSettings so Cancel still reverts if user wants.
            # If you prefer to save only on OK, you can omit this block.
            s = self.settings
            s.setValue("stacking/mfdeconv/seed_mode", "robust")
            s.setValue("stacking/mfdeconv/star_mask/thresh_sigma", 4.5)
            s.setValue("stacking/mfdeconv/star_mask/grow_px", 6)
            s.setValue("stacking/mfdeconv/star_mask/soft_sigma", 3.0)
            s.setValue("stacking/mfdeconv/star_mask/max_radius_px", 36)
            s.setValue("stacking/mfdeconv/star_mask/max_objs", 5000)
            s.setValue("stacking/mfdeconv/star_mask/keep_floor", 0.015)
            s.setValue("stacking/mfdeconv/star_mask/ellipse_scale", 1.12)
            s.setValue("stacking/mfdeconv/varmap/sample_stride", 8)
            s.setValue("stacking/mfdeconv/varmap/smooth_sigma", 3.0)
            s.setValue("stacking/mfdeconv/varmap/floor", 10 ** (-10))  # store linear value if you persist floor linearly

        btn_mf_reset.clicked.connect(_reset_mfdeconv_defaults)
        fl_mf.addRow(btn_mf_reset)

        right_col.addWidget(gb_mf)

        # --- Comet Star Removal (Optional) ---
        gb_csr = QGroupBox("Comet Star Removal (Optional)")
        fl_csr = QFormLayout(gb_csr)

        self.csr_enable = QCheckBox("Remove stars on comet-aligned frames")
        self.csr_enable.setChecked(self.settings.value("stacking/comet_starrem/enabled", False, type=bool))
        fl_csr.addRow(self.csr_enable)
        self.csr_enable.toggled.connect(lambda v: self.settings.setValue("stacking/comet_starrem/enabled", bool(v)))

        self.csr_tool = QComboBox()
        self.csr_tool.addItems(["StarNet", "CosmicClarityDarkStar"])
        curr_tool = self.settings.value("stacking/comet_starrem/tool", "StarNet", type=str)
        self.csr_tool.setCurrentText(curr_tool if curr_tool in ("StarNet","CosmicClarityDarkStar") else "StarNet")
        fl_csr.addRow("Tool:", self.csr_tool)

        self.csr_core_r = QDoubleSpinBox(); self.csr_core_r.setRange(2.0, 200.0); self.csr_core_r.setDecimals(1)
        self.csr_core_r.setValue(self.settings.value("stacking/comet_starrem/core_r", 22.0, type=float))
        fl_csr.addRow("Protect core radius (px):", self.csr_core_r)

        self.csr_core_soft = QDoubleSpinBox(); self.csr_core_soft.setRange(0.0, 100.0); self.csr_core_soft.setDecimals(1)
        self.csr_core_soft.setValue(self.settings.value("stacking/comet_starrem/core_soft", 6.0, type=float))
        fl_csr.addRow("Core mask feather (px):", self.csr_core_soft)

        def _toggle_csr(on: bool):
            for w in (self.csr_tool, self.csr_core_r, self.csr_core_soft):
                w.setEnabled(on)
        _toggle_csr(self.csr_enable.isChecked())
        self.csr_enable.toggled.connect(_toggle_csr)

        left_col.addWidget(gb_csr)


        # --- Rejection ---
        gb_rej = QGroupBox("Rejection")
        rej_layout = QVBoxLayout(gb_rej)

        # Algorithm choice
        algo_row = QHBoxLayout()
        algo_label = QLabel("Algorithm:")
        self.rejection_algo_combo = QComboBox()
        self.rejection_algo_combo.addItems([
            "Weighted Windsorized Sigma Clipping",
            "Kappa-Sigma Clipping",
            "Simple Average (No Rejection)",
            "Simple Median (No Rejection)",
            "Trimmed Mean",
            "Extreme Studentized Deviate (ESD)",
            "Biweight Estimator",
            "Modified Z-Score Clipping",
            "Max Value"
        ])
        saved_algo = self.settings.value("stacking/rejection_algorithm", "Weighted Windsorized Sigma Clipping")
        idx = self.rejection_algo_combo.findText(saved_algo)
        if idx >= 0:
            self.rejection_algo_combo.setCurrentIndex(idx)
        algo_row.addWidget(algo_label); algo_row.addWidget(self.rejection_algo_combo, 1)
        rej_layout.addLayout(algo_row)

        # Param rows as small containers we can show/hide
        def _mini_row(label_text, widget, help_text=None):
            row = QWidget()
            h = QHBoxLayout(row); h.setContentsMargins(0,0,0,0)
            h.addWidget(QLabel(label_text))
            h.addWidget(widget, 1)
            if help_text:
                btn = QPushButton("?"); btn.setFixedSize(20,20)
                btn.clicked.connect(lambda: QMessageBox.information(self, label_text, help_text))
                h.addWidget(btn)
            return row

        self.kappa_spinbox = QDoubleSpinBox()
        self.kappa_spinbox.setRange(0.1, 10.0); self.kappa_spinbox.setDecimals(2)
        self.kappa_spinbox.setValue(self.settings.value("stacking/kappa", 2.5, type=float))
        row_kappa = _mini_row("Kappa:", self.kappa_spinbox, "Std-devs from median to reject; higher = more lenient.")

        self.iterations_spinbox = QSpinBox()
        self.iterations_spinbox.setRange(1, 10)
        self.iterations_spinbox.setValue(self.settings.value("stacking/iterations", 3, type=int))
        row_iters = _mini_row("Iterations:", self.iterations_spinbox, "Number of kappa-sigma iterations.")

        self.esd_spinbox = QDoubleSpinBox()
        self.esd_spinbox.setRange(0.1, 10.0); self.esd_spinbox.setDecimals(2)
        self.esd_spinbox.setValue(self.settings.value("stacking/esd_threshold", 3.0, type=float))
        row_esd = _mini_row("ESD threshold:", self.esd_spinbox, "Lower = more aggressive outlier rejection.")

        self.biweight_spinbox = QDoubleSpinBox()
        self.biweight_spinbox.setRange(1.0, 10.0); self.biweight_spinbox.setDecimals(2)
        self.biweight_spinbox.setValue(self.settings.value("stacking/biweight_constant", 6.0, type=float))
        row_bi = _mini_row("Biweight constant:", self.biweight_spinbox, "Controls down-weighting strength.")

        self.trim_spinbox = QDoubleSpinBox()
        self.trim_spinbox.setRange(0.0, 0.5); self.trim_spinbox.setDecimals(2)
        self.trim_spinbox.setValue(self.settings.value("stacking/trim_fraction", 0.1, type=float))
        row_trim = _mini_row("Trim fraction:", self.trim_spinbox, "Fraction trimmed on each end before averaging.")

        self.modz_spinbox = QDoubleSpinBox()
        self.modz_spinbox.setRange(0.1, 10.0); self.modz_spinbox.setDecimals(2)
        self.modz_spinbox.setValue(self.settings.value("stacking/modz_threshold", 3.5, type=float))
        row_modz = _mini_row("Modified Z threshold:", self.modz_spinbox, "Lower = more aggressive (MAD-based).")

        # add all; visibility managed below
        for w in (row_kappa, row_iters, row_esd, row_bi, row_trim, row_modz):
            rej_layout.addWidget(w)

        # show/hide param rows based on algorithm
        def _update_algo_params():
            algo = self.rejection_algo_combo.currentText()
            # default all hidden
            rows = {
                "kappa": row_kappa, "iters": row_iters, "esd": row_esd,
                "bi": row_bi, "trim": row_trim, "modz": row_modz
            }
            for w in rows.values(): w.hide()

            if "Kappa-Sigma" in algo:
                row_kappa.show(); row_iters.show()
            elif "ESD" in algo:
                row_esd.show()
            elif "Biweight" in algo:
                row_bi.show()
            elif "Trimmed Mean" in algo:
                row_trim.show()
            elif "Modified Z-Score" in algo:
                row_modz.show()
            # others (simple average/median, weighted winsorized, max value) need no extra params

        self.rejection_algo_combo.currentTextChanged.connect(_update_algo_params)
        _update_algo_params()

        right_col.addWidget(gb_rej)


        # --- Cosmetic Correction (Advanced) ---
        gb_cosm = QGroupBox("Cosmetic Correction (Advanced)")
        fl_cosm = QFormLayout(gb_cosm)

        # Enable/disable advanced controls (purely for UI clarity)
        self.cosm_enable_cb = QCheckBox("Enable advanced cosmetic tuning")
        self.cosm_enable_cb.setChecked(
            self.settings.value("stacking/cosmetic/custom_enable", False, type=bool)
        )
        fl_cosm.addRow(self.cosm_enable_cb)

        def _mk_fspin(minv, maxv, step, decimals, key, default):
            sb = QDoubleSpinBox()
            sb.setRange(minv, maxv)
            sb.setDecimals(decimals)
            sb.setSingleStep(step)
            sb.setValue(self.settings.value(key, default, type=float))
            return sb

        # σ thresholds
        self.cosm_hot_sigma = _mk_fspin(0.1, 20.0, 0.1, 2,
            "stacking/cosmetic/hot_sigma", 5.0)
        self.cosm_cold_sigma = _mk_fspin(0.1, 20.0, 0.1, 2,
            "stacking/cosmetic/cold_sigma", 5.0)

        row_sig = QWidget(); row_sig_h = QHBoxLayout(row_sig); row_sig_h.setContentsMargins(0,0,0,0)
        row_sig_h.addWidget(QLabel("Hot σ:")); row_sig_h.addWidget(self.cosm_hot_sigma)
        row_sig_h.addSpacing(8)
        row_sig_h.addWidget(QLabel("Cold σ:")); row_sig_h.addWidget(self.cosm_cold_sigma)
        fl_cosm.addRow("Sigma thresholds:", row_sig)

        # Star guards (skip replacements if neighbors look like a PSF)
        self.cosm_star_mean_ratio = _mk_fspin(0.05, 0.60, 0.01, 3,
            "stacking/cosmetic/star_mean_ratio", 0.22)
        self.cosm_star_max_ratio  = _mk_fspin(0.10, 0.95, 0.01, 3,
            "stacking/cosmetic/star_max_ratio", 0.55)
        row_star = QWidget(); row_star_h = QHBoxLayout(row_star); row_star_h.setContentsMargins(0,0,0,0)
        row_star_h.addWidget(QLabel("Mean ratio:")); row_star_h.addWidget(self.cosm_star_mean_ratio)
        row_star_h.addSpacing(8)
        row_star_h.addWidget(QLabel("Max ratio:"));  row_star_h.addWidget(self.cosm_star_max_ratio)
        fl_cosm.addRow("Star guards:", row_star)

        # Saturation guard quantile
        self.cosm_sat_quantile = _mk_fspin(0.90, 0.9999, 0.0005, 4,
            "stacking/cosmetic/sat_quantile", 0.9995)
        self.cosm_sat_quantile.setToolTip("Pixels above this image quantile are treated as saturated and never replaced.")
        fl_cosm.addRow("Saturation quantile:", self.cosm_sat_quantile)

        # Small helper to enable/disable rows by master checkbox
        def _toggle_cosm_enabled(on: bool):
            for w in (row_sig, row_star, self.cosm_sat_quantile):
                w.setEnabled(on)

        # Defaults button
        btn_defaults = QPushButton("Restore Recommended")
        def _restore_defaults():
            self.cosm_hot_sigma.setValue(5.0)
            self.cosm_cold_sigma.setValue(5.0)
            self.cosm_star_mean_ratio.setValue(0.22)
            self.cosm_star_max_ratio.setValue(0.55)
            self.cosm_sat_quantile.setValue(0.9995)
        btn_defaults.clicked.connect(_restore_defaults)
        fl_cosm.addRow(btn_defaults)

        # wire
        self.cosm_enable_cb.toggled.connect(_toggle_cosm_enabled)
        _toggle_cosm_enabled(self.cosm_enable_cb.isChecked())

        right_col.addWidget(gb_cosm)

        # --- Comet (tuning only; not an algorithm picker) ---
        gb_comet = QGroupBox("Comet (High-Clip Percentile tuning)")
        fl_comet = QFormLayout(gb_comet)

        # load saved values (with defaults)
        def _getf(key, default):
            return self.settings.value(key, default, type=float)

        self.comet_hclip_k = QDoubleSpinBox()
        self.comet_hclip_k.setRange(0.1, 10.0)
        self.comet_hclip_k.setDecimals(2)
        self.comet_hclip_k.setSingleStep(0.05)
        self.comet_hclip_k.setValue(_getf("stacking/comet_hclip_k", 1.30))

        self.comet_hclip_p = QDoubleSpinBox()
        self.comet_hclip_p.setRange(1.0, 99.0)
        self.comet_hclip_p.setDecimals(1)
        self.comet_hclip_p.setSingleStep(1.0)
        self.comet_hclip_p.setValue(_getf("stacking/comet_hclip_p", 25.0))

        row_hclip = QWidget()
        row_hclip_h = QHBoxLayout(row_hclip); row_hclip_h.setContentsMargins(0,0,0,0)
        row_hclip_h.addWidget(QLabel("High-clip k / Percentile p:"))
        row_hclip_h.addWidget(self.comet_hclip_k)
        row_hclip_h.addWidget(QLabel(" / "))
        row_hclip_h.addWidget(self.comet_hclip_p)

        fl_comet.addRow(row_hclip)
        right_col.addWidget(gb_comet)
        right_col.addStretch(1)

        # --- Buttons ---
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(lambda: self.save_stacking_settings(dialog))
        btns.rejected.connect(dialog.reject)
        root.addWidget(btns)

        dialog.resize(900, 640)
        dialog.exec()

        try:
            self._refresh_comet_starless_enable()
        except Exception:
            pass        

    # --- Drizzle config: single source of truth ---
    def _get_drizzle_pixfrac(self) -> float:
        s = self.settings
        # new canonical -> old aliases -> default
        v = s.value("stacking/drizzle_pixfrac", None, type=float)
        if v is None:
            v = s.value("stacking/drizzle_drop", None, type=float)
        if v is None:
            v = s.value("stacking/drop_shrink", 0.65, type=float)
        # clamp to [0, 1]
        return float(max(0.0, min(1.0, v)))

    def _set_drizzle_pixfrac(self, v: float) -> None:
        v = float(max(0.0, min(1.0, v)))
        s = self.settings
        # write to canonical + legacy keys (back-compat)
        s.setValue("stacking/drizzle_pixfrac", v)
        s.setValue("stacking/drizzle_drop", v)
        s.setValue("stacking/drop_shrink", v)

        # reflect in any live widgets without feedback loops
        for wname in ("drizzle_drop_shrink_spin", "drop_shrink_spin"):
            w = getattr(self, wname, None)
            if w is not None and abs(float(w.value()) - v) > 1e-9:
                w.blockSignals(True); w.setValue(v); w.blockSignals(False)

    def _get_drizzle_scale(self) -> float:
        # Accepts "1x/2x/3x" or numeric
        val = self.settings.value("stacking/drizzle_scale", "2x", type=str)
        if isinstance(val, str) and val.endswith("x"):
            try: return float(val[:-1])
            except: return 2.0
        return float(val)

    def _set_drizzle_scale(self, r: float | str) -> None:
        if isinstance(r, str):
            try: r = float(r.rstrip("xX"))
            except: r = 2.0
        r = float(max(1.0, min(3.0, r)))
        # store as “Nx” so the combo’s string stays in sync
        self.settings.setValue("stacking/drizzle_scale", f"{int(r)}x")
        if hasattr(self, "drizzle_scale_combo"):
            txt = f"{int(r)}x"
            if self.drizzle_scale_combo.currentText() != txt:
                self.drizzle_scale_combo.blockSignals(True)
                self.drizzle_scale_combo.setCurrentText(txt)
                self.drizzle_scale_combo.blockSignals(False)


    def closeEvent(self, e):
        # Graceful shutdown for any running workers
        try:
            if hasattr(self, "alignment_thread") and self.alignment_thread and self.alignment_thread.isRunning():
                self.alignment_thread.requestInterruption()
                self.alignment_thread.wait(1500)
        except Exception:
            pass
        super().closeEvent(e)

    def _mf_worker_class_from_settings(self):
        """Return (WorkerClass, engine_name) from settings."""
        # local import avoids import-time cost if user never runs MFDeconv
        from pro.mfdeconv import MultiFrameDeconvWorker
        from pro.mfdeconvcudnn import MultiFrameDeconvWorkercuDNN
        from pro.mfdeconvsport import MultiFrameDeconvWorkerSport

        eng = str(self.settings.value("stacking/mfdeconv/engine", "normal", type=str) or "normal").lower()
        if eng == "cudnn":
            return (MultiFrameDeconvWorkercuDNN, "Normal (cuDNN-free)")
        if eng == "sport":
            return (MultiFrameDeconvWorkerSport, "High-Octane")
        return (MultiFrameDeconvWorker, "Normal")


    def save_stacking_settings(self, dialog):
        """
        Save settings and restart the Stacking Suite if the directory OR internal dtype changed.
        Uses dialog-scoped dir_edit and normalized path comparison.
        """
        # --- capture previous state BEFORE we change anything ---
        prev_dir_raw   = self.stacking_directory or ""
        prev_dir       = self._norm_dir(prev_dir_raw)
        prev_dtype_str = "float64" if (getattr(self, "internal_dtype", np.float64) is np.float64) else "float32"

        # --- read dialog widgets ---
        dir_edit   = getattr(dialog, "_dir_edit", None)
        new_dir_raw = (dir_edit.text() if dir_edit else prev_dir_raw)
        new_dir     = self._norm_dir(new_dir_raw)

        # Persist the rest
        self.sigma_high       = self.sigma_high_spinbox.value()
        self.sigma_low        = self.sigma_low_spinbox.value()
        self.rejection_algorithm = self.rejection_algo_combo.currentText()
        self.kappa           = self.kappa_spinbox.value()
        self.iterations      = self.iterations_spinbox.value()
        self.esd_threshold   = self.esd_spinbox.value()
        self.biweight_constant = self.biweight_spinbox.value()
        self.trim_fraction   = self.trim_spinbox.value()
        self.modz_threshold  = self.modz_spinbox.value()
        self.chunk_height    = self.chunkHeightSpinBox.value()
        self.chunk_width     = self.chunkWidthSpinBox.value()

        # Update instance + QSettings (write RAW path; use normalized only for comparison)
        self.stacking_directory = new_dir_raw
        self.settings.setValue("stacking/dir", new_dir_raw)
        self.settings.setValue("stacking/sigma_high", self.sigma_high)
        self.settings.setValue("stacking/sigma_low", self.sigma_low)
        self.settings.setValue("stacking/rejection_algorithm", self.rejection_algorithm)
        self.settings.setValue("stacking/kappa", self.kappa)
        self.settings.setValue("stacking/iterations", self.iterations)
        self.settings.setValue("stacking/esd_threshold", self.esd_threshold)
        self.settings.setValue("stacking/biweight_constant", self.biweight_constant)
        self.settings.setValue("stacking/trim_fraction", self.trim_fraction)
        self.settings.setValue("stacking/modz_threshold", self.modz_threshold)
        self.settings.setValue("stacking/chunk_height", self.chunk_height)
        self.settings.setValue("stacking/chunk_width", self.chunk_width)
        self.settings.setValue("stacking/autocrop_enabled", self.autocrop_cb.isChecked())
        self.settings.setValue("stacking/autocrop_pct", float(self.autocrop_pct.value()))

        # ----- alignment model (affine | homography | poly3 | poly4) -----
        model_idx = self.align_model_combo.currentIndex()
        if   model_idx == 0: model_name = "affine"
        elif model_idx == 1: model_name = "homography"
        elif model_idx == 2: model_name = "poly3"
        else:                model_name = "poly4"
        self.settings.setValue("stacking/align/model", model_name)
        self.settings.setValue("stacking/align/max_cp",      int(self.align_max_cp.value()))
        self.settings.setValue("stacking/align/downsample",  int(self.align_downsample.value()))
        self.settings.setValue("stacking/align/h_reproj",    float(self.h_ransac_reproj.value()))

        # Seed mode (persist as stable tokens: 'robust' | 'median')
        seed_idx = int(self.mf_seed_combo.currentIndex())
        seed_mode_val = "median" if seed_idx == 1 else "robust"
        self.settings.setValue("stacking/mfdeconv/seed_mode", seed_mode_val)

        # Star mask params
        self.settings.setValue("stacking/mfdeconv/star_mask/thresh_sigma",  float(self.sm_thresh.value()))
        self.settings.setValue("stacking/mfdeconv/star_mask/grow_px",       int(self.sm_grow.value()))
        self.settings.setValue("stacking/mfdeconv/star_mask/soft_sigma",    float(self.sm_soft.value()))
        self.settings.setValue("stacking/mfdeconv/star_mask/max_radius_px", int(self.sm_rmax.value()))
        self.settings.setValue("stacking/mfdeconv/star_mask/max_objs",      int(self.sm_maxobjs.value()))
        self.settings.setValue("stacking/mfdeconv/star_mask/keep_floor",    float(self.sm_keepfloor.value()))
        self.settings.setValue("stacking/mfdeconv/star_mask/ellipse_scale", float(self.sm_es.value()))

        # Variance map params
        self.settings.setValue("stacking/mfdeconv/varmap/sample_stride", int(self.vm_stride.value()))
        self.settings.setValue("stacking/mfdeconv/varmap/smooth_sigma",  float(self.vm_sigma.value()))
        vm_floor = 10.0 ** float(self.vm_floor_log.value())
        self.settings.setValue("stacking/mfdeconv/varmap/floor", vm_floor)

        # MFDeconv engine selection
        if   self.mf_eng_cudnn_rb.isChecked(): mf_engine = "cudnn"
        elif self.mf_eng_sport_rb.isChecked(): mf_engine = "sport"
        else:                                   mf_engine = "normal"
        self.settings.setValue("stacking/mfdeconv/engine", mf_engine)

        # (compat: drop the legacy boolean; keep it synced for older configs if you like)
        self.settings.setValue("stacking/high_octane", mf_engine == "sport")
        # Gradient settings
        self.settings.setValue("stacking/grad_poly2/enabled",   self.chk_poly2.isChecked())
        self.settings.setValue("stacking/grad_poly2/mode",       "divide" if self.grad_mode_combo.currentIndex() == 1 else "subtract")
        self.settings.setValue("stacking/grad_poly2/samples",    self.grad_samples_spin.value())
        self.settings.setValue("stacking/grad_poly2/downsample", self.grad_downsample_spin.value())
        self.settings.setValue("stacking/grad_poly2/patch_size", self.grad_patch_spin.value())
        self.settings.setValue("stacking/grad_poly2/min_strength", float(self.grad_min_strength.value()))
        self.settings.setValue("stacking/grad_poly2/gain_lo",    float(self.grad_gain_lo.value()))
        self.settings.setValue("stacking/grad_poly2/gain_hi",    float(self.grad_gain_hi.value()))

        # Cosmetic (Advanced)
        self.settings.setValue("stacking/cosmetic/custom_enable", self.cosm_enable_cb.isChecked())
        self.settings.setValue("stacking/cosmetic/hot_sigma",      float(self.cosm_hot_sigma.value()))
        self.settings.setValue("stacking/cosmetic/cold_sigma",     float(self.cosm_cold_sigma.value()))
        self.settings.setValue("stacking/cosmetic/star_mean_ratio",float(self.cosm_star_mean_ratio.value()))
        self.settings.setValue("stacking/cosmetic/star_max_ratio", float(self.cosm_star_max_ratio.value()))
        self.settings.setValue("stacking/cosmetic/sat_quantile",   float(self.cosm_sat_quantile.value()))

        self.settings.setValue("stacking/use_hardware_accel", self.hw_accel_cb.isChecked())
        self.use_gpu_integration = bool(self.hw_accel_cb.isChecked())

        self.settings.setValue("stacking/comet_starrem/enabled", self.csr_enable.isChecked())
        self.settings.setValue("stacking/comet_starrem/tool", self.csr_tool.currentText())
        self.settings.setValue("stacking/comet_starrem/core_r", float(self.csr_core_r.value()))
        self.settings.setValue("stacking/comet_starrem/core_soft", float(self.csr_core_soft.value()))

        passes = 1 if self.align_passes_combo.currentIndex() == 0 else 3
        self.settings.setValue("stacking/refinement_passes", passes)
        self.settings.setValue("stacking/shift_tolerance", self.shift_tol_spin.value())

        self.settings.setValue("stacking/drop_shrink", float(self.drop_shrink_spin.value()))

        kidx = self.drizzle_kernel_combo.currentIndex()
        kname = "square" if kidx==0 else ("circular" if kidx==1 else "gaussian")
        self.settings.setValue("stacking/drizzle_kernel", kname)

        self.settings.setValue("stacking/drizzle_gauss_sigma", float(self.gauss_sigma_spin.value()))
        self.settings.setValue("stacking/comet_hclip_k", float(self.comet_hclip_k.value()))
        self.settings.setValue("stacking/comet_hclip_p", float(self.comet_hclip_p.value()))
        # --- precision (internal dtype) ---
        chosen = self.precision_combo.currentText()  # "32-bit float" or "64-bit float"
        new_dtype_str = "float64" if "64" in chosen else "float32"
        dtype_changed = (new_dtype_str != prev_dtype_str)

        self.internal_dtype = np.float64 if new_dtype_str == "float64" else np.float32
        self.settings.setValue("stacking/internal_dtype", new_dtype_str)

        # Make sure everything is flushed
        self.settings.sync()

        # Logging
        self.update_status("✅ Saved stacking settings.")
        self.update_status(f"• Internal precision: {new_dtype_str}")
        self.update_status(f"• Hardware acceleration: {'ON' if self.use_gpu_integration else 'OFF'}")
        self._update_stacking_path_display()

        # --- restart if needed ---
        dir_changed = (new_dir != prev_dir)
        if dir_changed or dtype_changed:
            reasons = []
            if dir_changed:
                reasons.append("folder change")
            if dtype_changed:
                reasons.append(f"precision → {new_dtype_str}")
            self.update_status(f"🔁 Restarting Stacking Suite to apply {', '.join(reasons)}…")
            dialog.accept()
            self._restart_self()
            return

        dialog.accept()


    def _restart_self(self):
        geom = self.saveGeometry()
        try:
            cur_tab = self.tabs.currentIndex()
        except Exception:
            cur_tab = None

        parent = self.parent()  # may be None

        app = QApplication.instance()
        # Keep a global strong ref so GC can't collect the new dialog
        if not hasattr(app, "_stacking_suite_ref"):
            app._stacking_suite_ref = None

        def spawn():
            new = StackingSuiteDialog(parent=parent)
            if geom:
                new.restoreGeometry(geom)
            if cur_tab is not None:
                try:
                    new.tabs.setCurrentIndex(cur_tab)
                except Exception:
                    pass
            new.show()
            app._stacking_suite_ref = new  # <<< strong ref lives for app lifetime

        QTimer.singleShot(0, spawn)
        self.close()

    def _on_stacking_directory_changed(self, old_dir: str, new_dir: str):
        # Stop any running worker safely
        if hasattr(self, "alignment_thread") and self.alignment_thread:
            try:
                if self.alignment_thread.isRunning():
                    self.alignment_thread.requestInterruption()
                    self.alignment_thread.wait(1500)
            except Exception:
                pass

        self._ensure_stacking_subdirs(new_dir)
        self._clear_integration_state()

        # 🔁 RESCAN + REPOPULATE (the key bit you’re missing)
        self._reload_lists_for_new_dir()

        # If your tabs populate on change, poke the active one:
        if hasattr(self, "on_tab_changed"):
            self.on_tab_changed(self.tabs.currentIndex())

        # Update any path labels
        self._update_stacking_path_display()

        # Reload any persisted master selections
        try:
            self.restore_saved_master_calibrations()
        except Exception:
            pass

        self.update_status(f"📂 Stacking directory changed:\n    {old_dir or '(none)'} → {new_dir}")

    def _reload_lists_for_new_dir(self):
        """
        Re-scan the new stacking directory and repopulate internal dicts AND UI.
        """
        base = self.stacking_directory or ""
        self.conversion_output_directory = os.path.join(base, "Converted_Images")

        # Rebuild dictionaries from disk
        self.dark_files  = self._discover_grouped(os.path.join(base, "Calibrated_Darks"))
        self.flat_files  = self._discover_grouped(os.path.join(base, "Calibrated_Flats"))
        self.light_files = self._discover_grouped(os.path.join(base, "Calibrated_Lights"))

        # If you store master lists/sizes by path, clear/reseed minimally
        self.master_files.clear()
        self.master_sizes.clear()

        # 🔄 Update the tab UIs if you have builders; try common method names safely
        # Darks
        if hasattr(self, "rebuild_dark_tree"):
            self.rebuild_dark_tree(self.dark_files)
        elif hasattr(self, "populate_dark_tab"):
            self.populate_dark_tab()

        # Flats
        if hasattr(self, "rebuild_flat_tree"):
            self.rebuild_flat_tree(self.flat_files)
        elif hasattr(self, "populate_flat_tab"):
            self.populate_flat_tab()

        # Lights
        if hasattr(self, "rebuild_light_tree"):
            self.rebuild_light_tree(self.light_files)
        elif hasattr(self, "populate_light_tab"):
            self.populate_light_tab()

        # Image Integration (registration) tab often shows counts/paths
        if hasattr(self, "refresh_integration_tab"):
            self.refresh_integration_tab()

        self.update_status(f"🔄 Re-scanned calibrated sets in: {base}")

    def _discover_grouped(self, root_dir: str) -> dict:
        """
        Walk 'root_dir' and return {group_name: [file_paths,...]}.
        Group = immediate subfolder name; if files are directly in root, group 'Ungrouped'.
        """
        groups = {}
        if not root_dir or not os.path.isdir(root_dir):
            return groups

        valid_ext = (".fit", ".fits", ".xisf", ".tif", ".tiff")
        root_dir = os.path.normpath(root_dir)

        for dirpath, _, files in os.walk(root_dir):
            for fn in files:
                if not fn.lower().endswith(valid_ext):
                    continue
                fpath = os.path.normpath(os.path.join(dirpath, fn))
                parent = os.path.basename(os.path.dirname(fpath))
                group  = parent if os.path.dirname(fpath) != root_dir else "Ungrouped"
                groups.setdefault(group, []).append(fpath)

        # Stable ordering helps
        for g in groups:
            groups[g].sort()
        return groups

    def _refresh_all_tabs_once(self):
        current = self.tabs.currentIndex()
        if hasattr(self, "on_tab_changed"):
            for idx in range(self.tabs.count()):
                self.on_tab_changed(idx)
        self.tabs.setCurrentIndex(current)

    def _ensure_stacking_subdirs(self, base_dir: str):
        try:
            os.makedirs(base_dir, exist_ok=True)
            for sub in (
                "Aligned_Images",
                "Normalized_Images",
                "Calibrated_Darks",
                "Calibrated_Flats",
                "Calibrated_Lights",
                "Converted_Images",
                "Masters",
            ):
                os.makedirs(os.path.join(base_dir, sub), exist_ok=True)
        except Exception as e:
            self.update_status(f"⚠️ Could not ensure subfolders in '{base_dir}': {e}")

    def _clear_integration_state(self):
        # wipe per-run state so we don't “blend” two directories
        self.per_group_drizzle.clear()
        self.manual_dark_overrides.clear()
        self.manual_flat_overrides.clear()
        self.reg_files.clear()
        self.session_tags.clear()
        self.deleted_calibrated_files.clear()
        self._norm_map.clear()
        setattr(self, "valid_transforms", {})
        setattr(self, "frame_weights", {})
        setattr(self, "_global_autocrop_rect", None)

    def _rebuild_tabs_after_dir_change(self):
        # Rebuild the tab widgets so any path assumptions inside them reset to the new dir
        current = self.tabs.currentIndex()

        # Remove all tabs & delete widgets
        while self.tabs.count():
            w = self.tabs.widget(0)
            self.tabs.removeTab(0)
            try:
                w.deleteLater()
            except Exception:
                pass

        # Recreate against the new base path
        self.conversion_tab = self.create_conversion_tab()
        self.dark_tab       = self.create_dark_tab()
        self.flat_tab       = self.create_flat_tab()
        self.light_tab      = self.create_light_tab()
        self.image_integration_tab = self.create_image_registration_tab()

        self.tabs.addTab(self.conversion_tab, "Convert Non-FITS Formats")
        self.tabs.addTab(self.dark_tab,       "Darks")
        self.tabs.addTab(self.flat_tab,       "Flats")
        self.tabs.addTab(self.light_tab,      "Lights")
        self.tabs.addTab(self.image_integration_tab, "Image Integration")

        # Restore previously active tab if possible
        if 0 <= current < self.tabs.count():
            self.tabs.setCurrentIndex(current)
        else:
            self.tabs.setCurrentIndex(1)  # Darks by default

    def select_stacking_directory(self):
        """ Opens a dialog to choose a stacking directory. """
        directory = QFileDialog.getExistingDirectory(self, "Select Stacking Directory")
        if directory:
            self.stacking_directory = directory
            self.dir_path_edit.setText(directory)  # No more AttributeError
            self.settings.setValue("stacking/dir", directory)  # Save the new directory
            self._update_stacking_path_display()

    def _bind_shared_setting_checkbox(self, key: str, checkbox: QCheckBox, default: bool = True):
        """Bind a QCheckBox to a shared QSettings key and keep all bound boxes in sync."""
        # registry of sets of checkboxes per setting key
        if not hasattr(self, "_shared_checkboxes"):
            self._shared_checkboxes = {}

        # initialize from settings
        val = self.settings.value(key, default, type=bool)
        checkbox.blockSignals(True)
        checkbox.setChecked(bool(val))
        checkbox.blockSignals(False)

        # track this checkbox (weak so GC is fine when tabs close)
        boxset = self._shared_checkboxes.setdefault(key, weakref.WeakSet())
        boxset.add(checkbox)

        # when this one toggles, update settings and all siblings
        def _on_toggled(v: bool):
            self.settings.setValue(key, bool(v))
            # sync siblings without re-emitting signals
            for cb in list(self._shared_checkboxes.get(key, [])):
                if cb is checkbox:
                    continue
                try:
                    cb.blockSignals(True)
                    cb.setChecked(bool(v))
                    cb.blockSignals(False)
                except RuntimeError:
                    # widget was likely deleted; ignore
                    pass

        checkbox.toggled.connect(_on_toggled, Qt.ConnectionType.QueuedConnection)

    def create_dark_tab(self):
        tab = QWidget()
        main_layout = QVBoxLayout(tab)  # Vertical layout to separate sections

        # --- DARK FRAMES TREEBOX (TOP) ---
        darks_layout = QHBoxLayout()  # Left = Dark Tree, Right = Controls

        # Left Side - Dark Frames
        dark_frames_layout = QVBoxLayout()
        dark_frames_layout.addWidget(QLabel("Dark Frames"))
        # 1) Create the tree
        self.dark_tree = QTreeWidget()
        self.dark_tree.setColumnCount(2)
        self.dark_tree.setHeaderLabels(["Exposure Time", "Metadata"])
        self.dark_tree.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)

        # 2) Make columns user-resizable
        header = self.dark_tree.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Interactive)

        # 3) After you fill the tree with items, auto-resize
        self.dark_tree.resizeColumnToContents(0)
        self.dark_tree.resizeColumnToContents(1)

        # Then add it to the layout
        dark_frames_layout.addWidget(self.dark_tree)

        # Buttons to Add Dark Files & Directories
        btn_layout = QHBoxLayout()
        self.add_dark_files_btn = QPushButton("Add Dark Files")
        self.add_dark_files_btn.clicked.connect(self.add_dark_files)
        self.add_dark_dir_btn = QPushButton("Add Dark Directory")
        self.add_dark_dir_btn.clicked.connect(self.add_dark_directory)
        btn_layout.addWidget(self.add_dark_files_btn)
        btn_layout.addWidget(self.add_dark_dir_btn)
        dark_frames_layout.addLayout(btn_layout)

        self.clear_dark_selection_btn = QPushButton("Clear Selection")
        self.clear_dark_selection_btn.clicked.connect(lambda: self.clear_tree_selection(self.dark_tree, self.dark_files))
        dark_frames_layout.addWidget(self.clear_dark_selection_btn)

        darks_layout.addLayout(dark_frames_layout, 2)  # Dark Frames Tree takes more space


        # --- RIGHT SIDE: Exposure Tolerance & Master Darks Button ---
        right_controls_layout = QVBoxLayout()

        # Exposure Tolerance
        exposure_tolerance_layout = QHBoxLayout()
        exposure_tolerance_label = QLabel("Exposure Tolerance (seconds):")
        self.exposure_tolerance_spinbox = QSpinBox()
        self.exposure_tolerance_spinbox.setRange(0, 30)  # Acceptable range
        self.exposure_tolerance_spinbox.setValue(5)  # Default: ±5 sec
        exposure_tolerance_layout.addWidget(exposure_tolerance_label)
        exposure_tolerance_layout.addWidget(self.exposure_tolerance_spinbox)
        right_controls_layout.addLayout(exposure_tolerance_layout)

        # --- "Turn Those Darks Into Master Darks" Button ---
        self.create_master_dark_btn = QPushButton("Turn Those Darks Into Master Darks")
        self.create_master_dark_btn.clicked.connect(self.create_master_dark)

        # Apply a bold font, padding, and a highlighted effect
        self.create_master_dark_btn.setStyleSheet("""
            QPushButton {
                background-color: #333;  /* Dark gray */
                color: white;
                font-size: 14px;
                padding: 8px;
                border-radius: 5px;
                border: 2px solid yellow;  /* Subtle yellow border */
            }
            QPushButton:hover {
                border: 2px solid #FFD700;  /* Brighter yellow on hover */
            }
            QPushButton:pressed {
                background-color: #222;  /* Darker gray on press */
                border: 2px solid #FFA500;  /* Orange border when pressed */
            }
        """)

        right_controls_layout.addWidget(self.create_master_dark_btn)


        darks_layout.addLayout(right_controls_layout, 1)  # Right side takes less space

        main_layout.addLayout(darks_layout)

        # --- MASTER DARKS TREEBOX (BOTTOM) ---
        main_layout.addWidget(QLabel("Master Darks"))
        self.master_dark_tree = QTreeWidget()
        self.master_dark_tree.setColumnCount(2)
        self.master_dark_tree.setHeaderLabels(["Exposure Time", "Master File"])
        self.master_dark_tree.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        main_layout.addWidget(self.master_dark_tree)

        # Master Dark Selection Button
        self.master_dark_btn = QPushButton("Load Master Dark")
        self.master_dark_btn.clicked.connect(self.load_master_dark)
        main_layout.addWidget(self.master_dark_btn)

        # Add "Clear Selection" button for Master Darks
        self.clear_master_dark_selection_btn = QPushButton("Clear Selection")
        self.clear_master_dark_selection_btn.clicked.connect(
            lambda: self.clear_tree_selection(self.master_dark_tree, self.master_files)
        )
        self.clear_master_dark_selection_btn.clicked.connect(
            lambda: (self.clear_tree_selection(self.master_dark_tree, self.master_files),
                    self.save_master_paths_to_settings())
        )        
        main_layout.addWidget(self.clear_master_dark_selection_btn)

        return tab

    def _tree_for_type(self, t: str):
        t = (t or "").upper()
        if t == "LIGHT": return getattr(self, "light_tree", None)
        if t == "FLAT":  return getattr(self, "flat_tree", None)
        if t == "DARK":  return getattr(self, "dark_tree", None)
        return None

    def _guess_type_from_imagetyp(self, imagetyp: str) -> str | None:
        s = (imagetyp or "").strip().lower()
        # common patterns
        if "flat" in s: return "FLAT"
        if "dark" in s: return "DARK"
        if "bias" in s or "offset" in s: return "DARK"  # treat bias as darks bucket if you like
        if "light" in s or "object" in s: return "LIGHT"
        return None

    def _parse_exposure_from_groupkey(self, group_key: str) -> float | None:
        # expects "... - 300s (WxH)" → 300

        m = re.search(r"\s-\s([0-9.]+)s\s*\(", group_key)
        if not m: return None
        try:
            return float(m.group(1))
        except Exception:
            return None


    def _report_group_summary(self, expected_type: str):
        """
        Print #files and total integration per group (and session when applicable).
        For LIGHT/FLAT we key by (group_key, session_tag). For DARK we key by group_key.
        """
        t = (expected_type or "").upper()
        if t == "LIGHT":
            store = getattr(self, "light_files", {})
            # store keys: (group_key, session_tag)
            self.update_status("📈 Light groups summary:")
            seen = set()
            for (gkey, sess), paths in store.items():
                if not paths: continue
                exp = self._parse_exposure_from_groupkey(gkey) or 0.0
                tot = exp * len(paths)
                label = f"• {gkey}  |  session: {sess}  →  {len(paths)} files, {self._fmt_hms(tot)}"
                if (gkey, sess) not in seen:
                    self.update_status(label)
                    seen.add((gkey, sess))
        elif t == "FLAT":
            store = getattr(self, "flat_files", {})
            self.update_status("📈 Flat groups summary:")
            seen = set()
            for (gkey, sess), paths in store.items():
                if not paths: continue
                exp = self._parse_exposure_from_groupkey(gkey) or 0.0
                tot = exp * len(paths)
                label = f"• {gkey}  |  session: {sess}  →  {len(paths)} files, {self._fmt_hms(tot)}"
                if (gkey, sess) not in seen:
                    self.update_status(label)
                    seen.add((gkey, sess))
        elif t == "DARK":
            store = getattr(self, "dark_files", {})
            self.update_status("📈 Dark groups summary:")
            for gkey, paths in store.items():
                if not paths: continue
                exp = self._parse_exposure_from_groupkey(gkey) or 0.0
                tot = exp * len(paths)
                self.update_status(f"• {gkey}  →  {len(paths)} files, {self._fmt_hms(tot)}")


    def create_flat_tab(self):
        tab = QWidget()
        main_layout = QVBoxLayout(tab)  # Main layout to organize sections

        # --- FLAT FRAMES TREEBOX (TOP) ---
        flats_layout = QHBoxLayout()  # Left = Flat Tree, Right = Controls

        # Left Side - Flat Frames
        flat_frames_layout = QVBoxLayout()
        flat_frames_layout.addWidget(QLabel("Flat Frames"))

        self.flat_tree = QTreeWidget()
        self.flat_tree.setColumnCount(3)  # Added 3rd column for Master Dark Used
        self.flat_tree.setHeaderLabels(["Filter & Exposure", "Metadata", "Master Dark Used"])
        self.flat_tree.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.flat_tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.flat_tree.customContextMenuRequested.connect(self.flat_tree_context_menu)
        flat_frames_layout.addWidget(self.flat_tree)

        # Buttons to Add Flat Files & Directories
        btn_layout = QHBoxLayout()
        self.add_flat_files_btn = QPushButton("Add Flat Files")
        self.add_flat_files_btn.clicked.connect(self.add_flat_files)
        self.add_flat_dir_btn = QPushButton("Add Flat Directory")
        self.add_flat_dir_btn.clicked.connect(self.add_flat_directory)
        btn_layout.addWidget(self.add_flat_files_btn)
        btn_layout.addWidget(self.add_flat_dir_btn)
        flat_frames_layout.addLayout(btn_layout)
        # under your existing buttons:
        opts_row = QHBoxLayout()

        self.flat_recurse_cb = QCheckBox("Recurse subfolders")
        self._bind_shared_setting_checkbox("stacking/recurse_dirs", self.flat_recurse_cb, default=True)

        self.flat_auto_session_cb = QCheckBox("Auto-detect session")
        self._bind_shared_setting_checkbox("stacking/auto_session", self.flat_auto_session_cb, default=True)

        opts_row.addWidget(self.flat_recurse_cb)
        opts_row.addWidget(self.flat_auto_session_cb)
        flat_frames_layout.addLayout(opts_row)      
        # 🔧 Session Tag Hint
        session_hint_label = QLabel("Right Click to Assign Session Keys if desired")
        session_hint_label.setStyleSheet("color: #888; font-style: italic; font-size: 11px; margin-left: 4px;")
        flat_frames_layout.addWidget(session_hint_label)

        # Add "Clear Selection" button for Flat Frames
        self.clear_flat_selection_btn = QPushButton("Clear Selection")
        self.clear_flat_selection_btn.clicked.connect(lambda: self.clear_tree_selection_flat(self.flat_tree, self.flat_files))
        flat_frames_layout.addWidget(self.clear_flat_selection_btn)

        flats_layout.addLayout(flat_frames_layout, 2)  # Left side takes more space

        # --- RIGHT SIDE: Exposure Tolerance & Master Dark Selection ---
        right_controls_layout = QVBoxLayout()

        # Exposure Tolerance
        exposure_tolerance_layout = QHBoxLayout()
        exposure_tolerance_label = QLabel("Exposure Tolerance (seconds):")
        self.flat_exposure_tolerance_spinbox = QSpinBox()
        self.flat_exposure_tolerance_spinbox.setRange(0, 30)  # Allow ±0 to 30 seconds
        self.flat_exposure_tolerance_spinbox.setValue(5)  # Default: ±5 sec
        exposure_tolerance_layout.addWidget(exposure_tolerance_label)
        exposure_tolerance_layout.addWidget(self.flat_exposure_tolerance_spinbox)
        right_controls_layout.addLayout(exposure_tolerance_layout)
        self.flat_exposure_tolerance_spinbox.valueChanged.connect(self.rebuild_flat_tree)


        # Auto-Select Master Dark
        self.auto_select_dark_checkbox = QCheckBox("Auto-Select Closest Master Dark")
        self.auto_select_dark_checkbox.setChecked(True)  # Default enabled
        right_controls_layout.addWidget(self.auto_select_dark_checkbox)

        # Manual Override: Select a Master Dark
        self.override_dark_combo = QComboBox()
        self.override_dark_combo.addItem("None (Use Auto-Select)")
        self.override_dark_combo.currentIndexChanged.connect(self.override_selected_master_dark_for_flats)
        right_controls_layout.addWidget(QLabel("Override Master Dark Selection"))
        right_controls_layout.addWidget(self.override_dark_combo)

        self.create_master_flat_btn = QPushButton("Turn Those Flats Into Master Flats")
        self.create_master_flat_btn.clicked.connect(self.create_master_flat)

        # Apply a bold font, padding, and a glowing effect
        self.create_master_flat_btn.setStyleSheet("""
            QPushButton {
                background-color: #333;  /* Dark gray */
                color: white;
                font-size: 14px;
                padding: 8px;
                border-radius: 5px;
                border: 2px solid yellow;  /* Subtle yellow border */
            }
            QPushButton:hover {
                border: 2px solid #FFD700;  /* Brighter yellow on hover */
            }
            QPushButton:pressed {
                background-color: #222;  /* Darker gray on press */
                border: 2px solid #FFA500;  /* Orange border when pressed */
            }
        """)


        right_controls_layout.addWidget(self.create_master_flat_btn)

        flats_layout.addLayout(right_controls_layout, 1)  # Right side takes less space

        main_layout.addLayout(flats_layout)

        # --- MASTER FLATS TREEBOX (BOTTOM) ---
        main_layout.addWidget(QLabel("Master Flats"))
        self.master_flat_tree = QTreeWidget()
        self.master_flat_tree.setColumnCount(2)
        self.master_flat_tree.setHeaderLabels(["Filter", "Master File"])
        self.master_flat_tree.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        
        main_layout.addWidget(self.master_flat_tree)

        # Master Flat Selection Button
        self.master_flat_btn = QPushButton("Load Master Flat")
        self.master_flat_btn.clicked.connect(self.load_master_flat)
        main_layout.addWidget(self.master_flat_btn)

        self.clear_master_flat_selection_btn = QPushButton("Clear Selection")
        self.clear_master_flat_selection_btn.clicked.connect(
            lambda: (self.clear_tree_selection(self.master_flat_tree, self.master_files),
                    self.save_master_paths_to_settings())
        )
        main_layout.addWidget(self.clear_master_flat_selection_btn)
        self.override_dark_combo.currentIndexChanged[int].connect(self.override_selected_master_dark_for_flats)
        self.update_override_dark_combo()
        self.rebuild_flat_tree()
    
        return tab

    def flat_tree_context_menu(self, position):
        item = self.flat_tree.itemAt(position)
        if item:
            menu = QMenu()
            set_session_action = menu.addAction("Set Session Tag")
            action = menu.exec(self.flat_tree.viewport().mapToGlobal(position))
            if action == set_session_action:
                self.prompt_set_session(item, "flat")

    def _refresh_light_tree_summaries(self):
        """
        Fill the Metadata column (col=1) with:
        • for each exposure node:  "<N> files · <HHh MMm SSs>"
        • for each filter node:    "<N_total> files · <HHh MMm SSs> (sum of all children)"
        """
        tree = getattr(self, "light_tree", None)
        if tree is None:
            return

        total_filters = tree.topLevelItemCount()
        for i in range(total_filters):
            filt_item = tree.topLevelItem(i)
            if filt_item is None:
                continue

            filt_total_files = 0
            filt_total_secs  = 0.0

            # children are exposure-size groups like "20s (1080x1920)"
            for j in range(filt_item.childCount()):
                exp_item = filt_item.child(j)
                if exp_item is None:
                    continue

                exp_seconds = self._exposure_from_label(exp_item.text(0)) or 0.0
                n_files     = exp_item.childCount()
                n_secs      = exp_seconds * n_files

                # set exposure-row metadata
                if n_files == 1:
                    exp_item.setText(1, f"1 file · {self._fmt_hms(n_secs)}")
                else:
                    exp_item.setText(1, f"{n_files} files · {self._fmt_hms(n_secs)}")

                filt_total_files += n_files
                filt_total_secs  += n_secs

            # set filter-row metadata (sum of children)
            if filt_total_files == 1:
                filt_item.setText(1, f"1 file · {self._fmt_hms(filt_total_secs)}")
            else:
                filt_item.setText(1, f"{filt_total_files} files · {self._fmt_hms(filt_total_secs)}")


    def create_light_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        if not hasattr(self, "manual_flat_overrides"):
            self.manual_flat_overrides = {}
        if not hasattr(self, "manual_dark_overrides"):
            self.manual_dark_overrides = {}

        # Tree widget for light frames
        self.light_tree = QTreeWidget()
        self.light_tree.setColumnCount(5)  # Added columns for Master Dark and Flat
        self.light_tree.setHeaderLabels(["Filter & Exposure", "Metadata", "Master Dark", "Master Flat", "Corrections"])
        self.light_tree.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)

        layout.addWidget(QLabel("Light Frames"))
        layout.addWidget(self.light_tree)

        # Buttons for adding files and directories
        btn_layout = QHBoxLayout()
        self.add_light_files_btn = QPushButton("Add Light Files")
        self.add_light_files_btn.clicked.connect(self.add_light_files)
        self.add_light_dir_btn = QPushButton("Add Light Directory")
        self.add_light_dir_btn.clicked.connect(self.add_light_directory)
        btn_layout.addWidget(self.add_light_files_btn)
        btn_layout.addWidget(self.add_light_dir_btn)
        layout.addLayout(btn_layout)
        # under your existing buttons:
        opts_row = QHBoxLayout()

        self.light_recurse_cb = QCheckBox("Recurse subfolders")
        self._bind_shared_setting_checkbox("stacking/recurse_dirs", self.light_recurse_cb, default=True)

        self.light_auto_session_cb = QCheckBox("Auto-detect session")
        self._bind_shared_setting_checkbox("stacking/auto_session", self.light_auto_session_cb, default=True)

        # keep this one — it's independent of the shared pair
        self.auto_register_after_calibration_cb = QCheckBox("Auto-register & integrate after calibration")
        self.auto_register_after_calibration_cb.setToolTip(
            "When checked, once calibration finishes the app will switch to Image Registration and run "
            "‘Register and Integrate Images’ automatically."
        )
        self.auto_register_after_calibration_cb.setChecked(
            self.settings.value("stacking/auto_register_after_cal", False, type=bool)
        )
        self.auto_register_after_calibration_cb.toggled.connect(
            lambda v: self.settings.setValue("stacking/auto_register_after_cal", bool(v))
        )

        opts_row.addWidget(self.light_recurse_cb)
        opts_row.addWidget(self.light_auto_session_cb)
        opts_row.addWidget(self.auto_register_after_calibration_cb)
        layout.addLayout(opts_row)       
        session_hint_label = QLabel("Right Click to Assign Session Keys if desired")
        session_hint_label.setStyleSheet("color: #888; font-style: italic; font-size: 11px; margin-left: 4px;")
        layout.addWidget(session_hint_label)

        clear_selection_btn = QPushButton("Remove Selected")
        clear_selection_btn.clicked.connect(lambda: self.clear_tree_selection_light(self.light_tree))
        layout.addWidget(clear_selection_btn)

        # Cosmetic Correction & Pedestal Controls
        correction_layout = QHBoxLayout()

        self.cosmetic_checkbox = QCheckBox("Enable Cosmetic Correction")
        # default = True, but keep it sticky via QSettings
        self.cosmetic_checkbox.setChecked(
            self.settings.value("stacking/cosmetic_enabled", True, type=bool)
        )
        # 🔧 NEW: persist initial value immediately so background code sees it
        self.settings.setValue("stacking/cosmetic_enabled", bool(self.cosmetic_checkbox.isChecked()))        
        self.cosmetic_checkbox.toggled.connect(
            lambda v: self.settings.setValue("stacking/cosmetic_enabled", bool(v))
        )

        self.pedestal_checkbox = QCheckBox("Apply Pedestal")
        self.pedestal_checkbox.setChecked(
            self.settings.value("stacking/pedestal_enabled", False, type=bool)
        )
        self.pedestal_checkbox.toggled.connect(
            lambda v: self.settings.setValue("stacking/pedestal_enabled", bool(v))
        )

        self.bias_checkbox = QCheckBox("Apply Bias Subtraction (For CCD Users)")
        self.bias_checkbox.setChecked(
            self.settings.value("stacking/bias_enabled", False, type=bool)
        )
        self.bias_checkbox.toggled.connect(
            lambda v: self.settings.setValue("stacking/bias_enabled", bool(v))
        )

        correction_layout.addWidget(self.cosmetic_checkbox)
        correction_layout.addWidget(self.pedestal_checkbox)
        correction_layout.addWidget(self.bias_checkbox)

        # Pedestal Value (0-1000, converted to 0-1)
        pedestal_layout = QHBoxLayout()
        self.pedestal_label = QLabel("Pedestal (0-1000):")
        self.pedestal_spinbox = QSpinBox()
        self.pedestal_spinbox.setRange(0, 1000)
        self.pedestal_spinbox.setValue(self.settings.value("stacking/pedestal_value", 50, type=int))
        self.pedestal_spinbox.valueChanged.connect(
            lambda v: self.settings.setValue("stacking/pedestal_value", int(v))
        )

        pedestal_layout.addWidget(self.pedestal_label)
        pedestal_layout.addWidget(self.pedestal_spinbox)
        pedestal_layout.addStretch(1)
        layout.addLayout(pedestal_layout)

        # 👇 tie enabled state to the checkbox (initial + live updates)
        def _sync_pedestal_enabled(checked: bool):
            self.pedestal_label.setEnabled(checked)
            self.pedestal_spinbox.setEnabled(checked)

        _sync_pedestal_enabled(self.pedestal_checkbox.isChecked())
        self.pedestal_checkbox.toggled.connect(_sync_pedestal_enabled)

        # Tooltips (unchanged)
        self.bias_checkbox.setToolTip(
            "CMOS users: Bias Subtraction is not needed.\n"
            "Modern CMOS cameras use Correlated Double Sampling (CDS),\n"
            "meaning bias is already subtracted at the sensor level."
        )

        # Connect to your existing correction updater
        self.cosmetic_checkbox.stateChanged.connect(self.update_light_corrections)
        self.pedestal_checkbox.stateChanged.connect(self.update_light_corrections)
        self.bias_checkbox.stateChanged.connect(self.update_light_corrections)

        # Add checkboxes to layout
        correction_layout.addWidget(self.cosmetic_checkbox)
        correction_layout.addWidget(self.pedestal_checkbox)
        correction_layout.addWidget(self.bias_checkbox)

        layout.addLayout(correction_layout)        

        # --- RIGHT SIDE CONTROLS: Override Dark & Flat ---
        override_layout = QHBoxLayout()

        self.override_dark_btn = QPushButton("Override Dark Frame")
        self.override_dark_btn.clicked.connect(self.override_selected_master_dark)
        override_layout.addWidget(self.override_dark_btn)

        self.override_flat_btn = QPushButton("Override Flat Frame")
        self.override_flat_btn.clicked.connect(self.override_selected_master_flat)
        override_layout.addWidget(self.override_flat_btn)

        layout.addLayout(override_layout)

        # Calibrate Lights Button
        self.calibrate_lights_btn = QPushButton("🚀 Calibrate Light Frames 🚀")
        self.calibrate_lights_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF4500;
                color: white;
                font-size: 16px;
                padding: 8px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #FF6347;
            }
        """)
        self.calibrate_lights_btn.clicked.connect(self.calibrate_lights)
        layout.addWidget(self.calibrate_lights_btn)

        # Enable Context Menu
        self.light_tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.light_tree.customContextMenuRequested.connect(self.light_tree_context_menu)

        return tab



    def prompt_set_session(self, item, frame_type):
        text, ok = QInputDialog.getText(self, "Set Session Tag", "Enter session name:")
        if not (ok and text.strip()):
            return

        session_name = text.strip()
        is_flat = frame_type.upper() == "FLAT"
        tree = self.flat_tree if is_flat else self.light_tree
        target_dict = self.flat_files if is_flat else self.light_files

        selected_items = tree.selectedItems()

        def update_file_session(filename, widget_item):
            for key in list(target_dict.keys()):
                if isinstance(key, tuple) and len(key) == 2:
                    group_key, old_session = key
                else:
                    continue  # Skip malformed keys

                files = target_dict.get(key, [])
                for f in list(files):
                    if os.path.basename(f) == filename:
                        if old_session != session_name:
                            new_key = (group_key, session_name)
                            if new_key not in target_dict:
                                target_dict[new_key] = []
                            target_dict[new_key].append(f)
                            target_dict[key].remove(f)
                            if not target_dict[key]:
                                del target_dict[key]

                        # Update internal session tag
                        self.session_tags[f] = session_name

                        # Update leaf's metadata column
                        old_meta = widget_item.text(1)
                        if "Session:" in old_meta:
                            new_meta = re.sub(r"Session: [^|]*", f"Session: {session_name}", old_meta)
                        else:
                            new_meta = f"{old_meta} | Session: {session_name}"
                        widget_item.setText(1, new_meta)
                        return

        def recurse_all_leaf_items(parent_item):
            for i in range(parent_item.childCount()):
                child = parent_item.child(i)
                if child.childCount() == 0:
                    update_file_session(child.text(0), child)
                else:
                    recurse_all_leaf_items(child)

        # Case 1: Multi-leaf selection (e.g. Shift/Ctrl-click)
        if selected_items and any(i.childCount() == 0 for i in selected_items):
            for leaf in selected_items:
                if leaf.childCount() == 0:
                    update_file_session(leaf.text(0), leaf)

        # Case 2: Right-clicked on a group (e.g. filter+exposure node)
        elif item and item.childCount() > 0:
            recurse_all_leaf_items(item)

        # ✅ Reassign matching master flats/darks per leaf
        self.assign_best_master_files()

    def _quad_coverage_add(self, cov: np.ndarray, quad: np.ndarray):
        """
        Rasterize a convex quad (4x2 float array of (x,y) in aligned coords) into 'cov' by +1 filling.
        Bounds/clipping are handled. Small, robust scanline fill.
        """
        H, W = cov.shape
        pts = quad.astype(np.float32)

        ymin = max(int(np.floor(np.min(pts[:,1]))), 0)
        ymax = min(int(np.ceil (np.max(pts[:,1]))), H-1)
        if ymin > ymax: return

        # Edges (x0,y0)->(x1,y1), 4 of them
        edges = []
        for i in range(4):
            x0, y0 = pts[i]
            x1, y1 = pts[(i+1) % 4]
            edges.append((x0, y0, x1, y1))

        for y in range(ymin, ymax+1):
            xs = []
            yf = float(y) + 0.5  # sample at pixel center
            for (x0, y0, x1, y1) in edges:
                # Skip horizontal edges
                if (y0 <= yf < y1) or (y1 <= yf < y0):
                    # Linear interpolate X at scanline yf
                    t = (yf - y0) / (y1 - y0)
                    xs.append(x0 + t * (x1 - x0))

            if len(xs) < 2:
                continue
            xs.sort()
            # Fill between pairs
            for i in range(0, len(xs), 2):
                xL = int(np.floor(min(xs[i], xs[i+1])))
                xR = int(np.ceil (max(xs[i], xs[i+1])))
                if xR < 0 or xL > W-1: 
                    continue
                xL = max(xL, 0); xR = min(xR, W)
                if xR > xL:
                    cov[y, xL:xR] += 1


    def _max_rectangle_in_binary(self, mask: np.ndarray):
        """
        Largest axis-aligned rectangle of 1s in a binary mask (H×W, dtype=bool).
        Returns (x0, y0, x1, y1) where x1,y1 are exclusive, or None if empty.
        O(H*W) using 'largest rectangle in histogram' per row.
        """
        H, W = mask.shape
        heights = np.zeros(W, dtype=np.int32)
        best = (0, 0, 0, 0, 0)  # (area, x0, y0, x1, y1)

        for y in range(H):
            row = mask[y]
            heights[row] += 1
            heights[~row] = 0

            # Largest rectangle in histogram 'heights'
            stack = []
            i = 0
            while i <= W:
                h = heights[i] if i < W else 0
                if not stack or h >= heights[stack[-1]]:
                    stack.append(i); i += 1
                else:
                    top = stack.pop()
                    height = heights[top]
                    left = stack[-1] + 1 if stack else 0
                    right = i
                    area = height * (right - left)
                    if area > best[0]:
                        # rectangle spans rows [y-height+1 .. y], columns [left .. right-1]
                        y0 = y - height + 1
                        y1 = y + 1
                        best = (area, left, y0, right, y1)

        if best[0] == 0:
            return None
        _, x0, y0, x1, y1 = best
        return (x0, y0, x1, y1)


    def _rect_from_transforms_fast(
        self,
        transforms: dict[str, object],
        src_hw: tuple[int, int],
        coverage_pct: float = 95.0,
        *,
        allow_homography: bool = True,
        min_side: int = 16,
    ) -> tuple[int, int, int, int] | None:
        """
        Robust fast autocrop from a bunch of transforms.

        Accepts:
        - plain 2x3 affine np.array / list-of-lists
        - plain 3x3 homography
        - tuples like (M, meta) or (M, bbox, ...)
        - dicts like {"M": M} or {"matrix": M} or {"H": M}

        Returns (x0, y0, x1, y1) in reference coords, or None.
        """
        H, W = map(int, src_hw)
        if not transforms:
            return None

        # corners in source-image coords
        corners = np.array(
            [
                [0, 0, 1],
                [W, 0, 1],
                [W, H, 1],
                [0, H, 1],
            ],
            dtype=np.float64,
        )

        def _extract_matrix(raw):
            """
            Try hard to get a 2x3 or 3x3 ndarray out of whatever `raw` is.
            Returns np.ndarray or None.
            """
            if raw is None:
                return None

            # common case: (M, extra...) → pick the first thing that looks like a matrix
            if isinstance(raw, (tuple, list)):
                for item in raw:
                    m = _extract_matrix(item)
                    if m is not None:
                        return m
                return None

            # dict style: {"M": ..., "bbox": ...} or {"matrix": ...}
            if isinstance(raw, dict):
                for key in ("M", "matrix", "affine", "H", "homography"):
                    if key in raw:
                        return _extract_matrix(raw[key])
                return None

            # already ndarray-ish
            try:
                arr = np.asarray(raw, dtype=np.float64)
            except Exception:
                return None

            if arr.shape == (2, 3) or arr.shape == (3, 3):
                return arr

            # anything else (like 1D or wrong shape) → ignore
            return None

        lefts, rights, tops, bottoms = [], [], [], []

        for path, raw_M in transforms.items():
            M = _extract_matrix(raw_M)
            if M is None:
                # e.g. this was (M, bbox) but we couldn't unwrap → skip
                continue

            # normalize to 3x3
            if M.shape == (2, 3):
                A = np.eye(3, dtype=np.float64)
                A[:2, :3] = M
                M = A
            elif M.shape == (3, 3):
                if not allow_homography:
                    continue
            else:
                # should not happen due to _extract_matrix, but be safe
                continue

            pts = M @ corners.T  # 3x4
            w = pts[2, :]
            # avoid div0
            w = np.where(np.abs(w) < 1e-12, 1.0, w)
            xs = pts[0, :] / w
            ys = pts[1, :] / w

            lefts.append(xs.min())
            rights.append(xs.max())
            tops.append(ys.min())
            bottoms.append(ys.max())

        if not lefts:
            # all entries were weird / non-matrix
            return None

        lefts = np.asarray(lefts, dtype=np.float64)
        rights = np.asarray(rights, dtype=np.float64)
        tops = np.asarray(tops, dtype=np.float64)
        bottoms = np.asarray(bottoms, dtype=np.float64)

        p = float(np.clip(coverage_pct, 0.0, 100.0)) / 100.0

        # quantile logic: keep the region covered by ≥ p of frames
        x0 = float(np.quantile(lefts, p))
        y0 = float(np.quantile(tops, p))
        x1 = float(np.quantile(rights, 1.0 - p))
        y1 = float(np.quantile(bottoms, 1.0 - p))

        if not np.isfinite((x0, y0, x1, y1)).all():
            return None

        # round inwards
        xi0 = int(np.ceil(x0))
        yi0 = int(np.ceil(y0))
        xi1 = int(np.floor(x1))
        yi1 = int(np.floor(y1))

        if xi1 - xi0 < min_side or yi1 - yi0 < min_side:
            return None

        return (xi0, yi0, xi1, yi1)


    def _compute_common_autocrop_rect(self, grouped_files: dict, coverage_pct: float, status_cb=None):
        log = status_cb or self.update_status
        transforms_path = os.path.join(self.stacking_directory, "alignment_transforms.sasd")
        common_mask = None
        for group_key, file_list in grouped_files.items():
            if not file_list:
                continue
            mask = self._compute_coverage_mask(file_list, transforms_path, coverage_pct)
            if mask is None:
                log(f"✂️ Global crop: no mask for '{group_key}' → disabling global crop.")
                return None
            if common_mask is None:
                common_mask = mask.astype(bool, copy=True)
            else:
                if mask.shape != common_mask.shape:
                    log("✂️ Global crop: mask shapes differ across groups.")
                    return None
                np.logical_and(common_mask, mask, out=common_mask)

        if common_mask is None or not common_mask.any():
            return None

        rect = self._max_rectangle_in_binary(common_mask)
        # Optional safety guard so we never get pencil-thin rectangles:
        if rect:
            x0, y0, x1, y1 = rect
            if (x1 - x0) < 16 or (y1 - y0) < 16:
                log("✂️ Global crop: rect too small; disabling global crop.")
                return None
            log(f"✂️ Global crop rect={rect} → size {x1-x0}×{y1-y0}")
        return rect

    def _first_non_none(self, *vals):
        for v in vals:
            if v is not None:
                return v
        return None

    def _compute_coverage_mask(self, file_list: List[str], transforms_path: str, coverage_pct: float):
        """
        Build a coverage-count image on the aligned canvas for 'file_list'.
        Threshold at coverage_pct, but use the number of frames we ACTUALLY rasterized (N_eff).
        Returns a bool mask (H×W) or None if nothing rasterized.
        """
        if not file_list:
            return None

        # Canvas from first aligned image
        ref_img, _, _, _ = load_image(file_list[0])
        if ref_img is None:
            self.update_status("✂️ Auto-crop: could not load first aligned ref.")
            return None
        H, W = (ref_img.shape if ref_img.ndim == 2 else ref_img.shape[:2])

        if not os.path.exists(transforms_path):
            self.update_status(f"✂️ Auto-crop: no transforms file at {transforms_path}")
            return None
        self.update_status(f"✂️ Auto-crop: Loading transforms...")
        QApplication.processEvents()
        transforms = self.load_alignment_matrices_custom(transforms_path)

        # --- Robust transform lookup: key by normalized full path AND by basename ---
        def _normcase(p):  # windows-insensitive
            p = os.path.normpath(os.path.abspath(p))
            return p.lower() if os.name == "nt" else p

        xforms_by_full = { _normcase(k): v for k, v in transforms.items() }
        xforms_by_name = {}
        for k, v in transforms.items():
            xforms_by_name.setdefault(os.path.basename(k), v)

        cov = np.zeros((H, W), dtype=np.uint16)
        used = 0

        for aligned_path in file_list:
            base = os.path.basename(aligned_path)
            if base.endswith("_n_r.fit"):
                raw_base = base.replace("_n_r.fit", "_n.fit")
            elif base.endswith("_r.fit"):
                raw_base = base.replace("_r.fit", ".fit")
            else:
                raw_base = base

            # try normalized-Images location first
            raw_path_guess = os.path.join(self.stacking_directory, "Normalized_Images", raw_base)

            # find transform
            M = self._first_non_none(
                xforms_by_full.get(_normcase(raw_path_guess)),
                xforms_by_full.get(_normcase(aligned_path)),
                transforms.get(raw_path_guess),
                transforms.get(os.path.normpath(aligned_path)),
                xforms_by_name.get(raw_base),
            )

            if M is None:
                # Can't rasterize this frame
                continue

            # raw size
            h_raw = w_raw = None
            if os.path.exists(raw_path_guess):
                raw_img, _, _, _ = load_image(raw_path_guess)
                if raw_img is not None:
                    h_raw, w_raw = (raw_img.shape if raw_img.ndim == 2 else raw_img.shape[:2])

            if h_raw is None or w_raw is None:
                # fallback to aligned canvas size (still okay; affine provides placement)
                h_raw, w_raw = H, W

            corners = np.array([[0,0],[w_raw-1,0],[w_raw-1,h_raw-1],[0,h_raw-1]], dtype=np.float32)
            A = M[:, :2]; t = M[:, 2]
            quad = (corners @ A.T) + t

            self._quad_coverage_add(cov, quad)
            used += 1

        if used == 0:
            self.update_status("✂️ Auto-crop: 0/{} frames had usable transforms; skipping.".format(len(file_list)))
            return None

        need = int(np.ceil((coverage_pct / 100.0) * used))
        mask = (cov >= need)
        self.update_status(f"✂️ Auto-crop: rasterized {used}/{len(file_list)} frames; need {need} per-pixel.")
        QApplication.processEvents()
        if not mask.any():
            self.update_status("✂️ Auto-crop: threshold produced empty mask.")
            return None
        return mask



    def _compute_autocrop_rect(self, file_list: List[str], transforms_path: str, coverage_pct: float):
        """
        Build a coverage-count image (aligned canvas), threshold at pct, and extract largest rectangle.e
        Returns (x0, y0, x1, y1) or None.
        """
        if not file_list:
            return None

        # Load aligned reference to get canvas size
        ref_img, ref_hdr, _, _ = load_image(file_list[0])
        if ref_img is None:
            return None
        if ref_img.ndim == 2:
            H, W = ref_img.shape
        else:
            H, W = ref_img.shape[:2]

        # Load transforms (raw _n path -> 2x3 matrix mapping raw->aligned)
        if not os.path.exists(transforms_path):
            return None
        transforms = self.load_alignment_matrices_custom(transforms_path)

        # We need the raw (normalized) image size for each file to transform its corners
        # From aligned name "..._n_r.fit" get raw name "..._n.fit" (like in your drizzle code)
        cov = np.zeros((H, W), dtype=np.uint16)
        for aligned_path in file_list:
            base = os.path.basename(aligned_path)
            if base.endswith("_n_r.fit"):
                raw_base = base.replace("_n_r.fit", "_n.fit")
            elif base.endswith("_r.fit"):
                raw_base = base.replace("_r.fit", ".fit")  # fallback
            else:
                raw_base = base  # fallback

            raw_path = os.path.join(self.stacking_directory, "Normalized_Images", raw_base)
            # Fallback if normalized folder differs:
            raw_key = os.path.normpath(raw_path)
            M = transforms.get(raw_key, None)
            if M is None:
                # Try direct key (some pipelines use normalized path equal to aligned key)
                M = transforms.get(os.path.normpath(aligned_path), None)
            if M is None:
                continue

            # Determine raw size
            raw_img, _, _, _ = load_image(raw_key) if os.path.exists(raw_key) else (None, None, None, None)
            if raw_img is None:
                # last resort: assume same canvas; still yields a conservative crop
                h_raw, w_raw = H, W
            else:
                if raw_img.ndim == 2:
                    h_raw, w_raw = raw_img.shape
                else:
                    h_raw, w_raw = raw_img.shape[:2]

            # Transform raw rectangle corners into aligned coords
            corners = np.array([
                [0,       0      ],
                [w_raw-1, 0      ],
                [w_raw-1, h_raw-1],
                [0,       h_raw-1]
            ], dtype=np.float32)

            # Apply affine: [x' y']^T = A*[x y]^T + t
            A = M[:, :2]; t = M[:, 2]
            quad = (corners @ A.T) + t  # shape (4,2)

            # Rasterize into coverage
            self._quad_coverage_add(cov, quad)

        # Threshold at requested coverage
        N = len(file_list)
        need = int(np.ceil((coverage_pct / 100.0) * N))
        mask = (cov >= need)

        # Largest rectangle of 1s
        rect = self._max_rectangle_in_binary(mask)
        return rect

    def _refresh_reg_tree_summaries(self):
        """
        Image Integration (Registration) tree:
        • For a top-level group like "LP - 20s (1080x1920)":  "<N> files · <HHh MMm SSs>"
        • For a parent filter row (if present):               sum over its children.
        """
        tree = getattr(self, "reg_tree", None)
        if tree is None:
            return

        def _summarize_item(item) -> tuple[int, float]:
            """Return (n_files_total, seconds_total) for this node (recurses children)."""
            if item is None:
                return (0, 0.0)

            # If this node has leaf children (files), assume label carries exposure (e.g., "… 20s (…)").
            n_children = item.childCount()
            if n_children == 0:
                # leaf (file) → contribute 1 file, but exposure is taken from parent; return (1, 0) so parent can add its exp
                return (1, 0.0)

            # group / parent
            label_exp = self._exposure_from_label(item.text(0))  # may be None for pure filter rows
            total_files = 0
            total_secs  = 0.0

            for i in range(n_children):
                ch = item.child(i)
                ch_files, ch_secs = _summarize_item(ch)
                total_files += ch_files
                total_secs  += ch_secs

            # If this node itself encodes an exposure, multiply its exposure by its direct leaf count
            # (i.e., files directly under this group).
            if label_exp is not None:
                direct_files = sum(1 for i in range(n_children) if item.child(i).childCount() == 0)
                total_secs += (label_exp * direct_files)

                # Also set this row's Metadata to its own group’s numbers:
                if direct_files > 0:
                    item.setText(1, (f"{direct_files} file · {self._fmt_hms(label_exp * direct_files)}"
                                    if direct_files == 1 else
                                    f"{direct_files} files · {self._fmt_hms(label_exp * direct_files)}"))
                else:
                    # clear if empty
                    item.setText(1, "")

            # For a pure parent (filter) row with no exposure in its label, show the sum across children
            if label_exp is None:
                if total_files == 1:
                    item.setText(1, f"1 file · {self._fmt_hms(total_secs)}")
                else:
                    item.setText(1, f"{total_files} files · {self._fmt_hms(total_secs)}")

            return (total_files, total_secs)

        for i in range(tree.topLevelItemCount()):
            _summarize_item(tree.topLevelItem(i))


    def create_image_registration_tab(self):
        """
        Image Registration tab with:
        - tree of calibrated lights
        - tolerance row (now includes Auto-crop)
        - add/remove buttons
        - global drizzle controls
        - reference selection + auto-accept toggle
        - MFDeconv (no 'beta')
        - Comet + Star-Trail checkboxes in one row BELOW MFDeconv
        - Backend (acceleration) row moved BELOW the comet+trail row
        """
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # ─────────────────────────────────────────
        # 1) QTreeWidget
        # ─────────────────────────────────────────
        self.reg_tree = QTreeWidget()
        self.reg_tree.setColumnCount(3)
        self.reg_tree.setHeaderLabels([
            "Filter - Exposure - Size",
            "Metadata",
            "Drizzle"
        ])
        self.reg_tree.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        header = self.reg_tree.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)

        layout.addWidget(QLabel("Calibrated Light Frames"))
        layout.addWidget(self.reg_tree)

        model = self.reg_tree.model()
        model.rowsInserted.connect(lambda *_: QTimer.singleShot(0, self._refresh_reg_tree_summaries))
        model.rowsRemoved.connect(lambda *_: QTimer.singleShot(0, self._refresh_reg_tree_summaries))

        # ─────────────────────────────────────────
        # 2) Exposure tolerance + Auto-crop + Split dual-band (same row)
        # ─────────────────────────────────────────
        tol_layout = QHBoxLayout()
        tol_layout.addWidget(QLabel("Exposure Tolerance (sec):"))

        self.exposure_tolerance_spin = QSpinBox()
        self.exposure_tolerance_spin.setRange(0, 900)
        self.exposure_tolerance_spin.setValue(0)
        self.exposure_tolerance_spin.setSingleStep(5)
        tol_layout.addWidget(self.exposure_tolerance_spin)
        tol_layout.addStretch()

        # Auto-crop moved here
        self.autocrop_cb = QCheckBox("Auto-crop output")
        self.autocrop_cb.setToolTip("Crop final image to pixels covered by ≥ Coverage % of frames")
        tol_layout.addWidget(self.autocrop_cb)

        tol_layout.addWidget(QLabel("Coverage:"))
        self.autocrop_pct = QDoubleSpinBox()
        self.autocrop_pct.setRange(50.0, 100.0)
        self.autocrop_pct.setSingleStep(1.0)
        self.autocrop_pct.setSuffix(" %")
        self.autocrop_pct.setValue(self.settings.value("stacking/autocrop_pct", 95.0, type=float))
        self.autocrop_cb.setChecked(self.settings.value("stacking/autocrop_enabled", True, type=bool))
        tol_layout.addWidget(self.autocrop_pct)

        tol_layout.addStretch()

        self.split_dualband_cb = QCheckBox("Split dual-band OSC before integration")
        self.split_dualband_cb.setToolTip("For OSC dual-band data: SII/OIII → R=SII, G=OIII; Ha/OIII → R=Ha, G=OIII")
        tol_layout.addWidget(self.split_dualband_cb)

        layout.addLayout(tol_layout)
        self.exposure_tolerance_spin.valueChanged.connect(
                lambda _ : (self.populate_calibrated_lights(), self._refresh_reg_tree_summaries())
            )

        # ─────────────────────────────────────────
        # 3) Buttons for Managing Files
        # ─────────────────────────────────────────
        btn_layout = QHBoxLayout()
        self.add_reg_files_btn = QPushButton("Add Light Files")
        self.add_reg_files_btn.clicked.connect(self.add_light_files_to_registration)
        btn_layout.addWidget(self.add_reg_files_btn)

        self.clear_selection_btn = QPushButton("Remove Selected")
        self.clear_selection_btn.clicked.connect(lambda: self.clear_tree_selection_registration(self.reg_tree))
        btn_layout.addWidget(self.clear_selection_btn)

        layout.addLayout(btn_layout)

        # ─────────────────────────────────────────
        # 4) Global Drizzle Controls
        # ─────────────────────────────────────────
        drizzle_layout = QHBoxLayout()

        self.drizzle_checkbox = QCheckBox("Enable Drizzle")
        self.drizzle_checkbox.toggled.connect(self._on_drizzle_checkbox_toggled)
        drizzle_layout.addWidget(self.drizzle_checkbox)

        drizzle_layout.addWidget(QLabel("Scale:"))
        self.drizzle_scale_combo = QComboBox()
        self.drizzle_scale_combo.addItems(["1x", "2x", "3x"])
        self.drizzle_scale_combo.currentIndexChanged.connect(self._on_drizzle_param_changed)
        drizzle_layout.addWidget(self.drizzle_scale_combo)

        drizzle_layout.addWidget(QLabel("Drop Shrink:"))
        self.drizzle_drop_shrink_spin = QDoubleSpinBox()
        self.drizzle_drop_shrink_spin.setRange(0.0, 1.0)  # pixfrac is [0..1]
        self.drizzle_drop_shrink_spin.setDecimals(3)
        self.drizzle_drop_shrink_spin.setValue(self._get_drizzle_pixfrac())
        self.drizzle_drop_shrink_spin.valueChanged.connect(lambda v: self._set_drizzle_pixfrac(v))
        drizzle_layout.addWidget(self.drizzle_drop_shrink_spin)

        self.cfa_drizzle_cb = QCheckBox("CFA Drizzle")
        self.cfa_drizzle_cb.setChecked(self.settings.value("stacking/cfa_drizzle", False, type=bool))
        self.cfa_drizzle_cb.toggled.connect(self._on_cfa_drizzle_toggled)
        self.cfa_drizzle_cb.setToolTip("Requires 'Enable Drizzle'. Maps R/G/B CFA samples directly into channels (no interpolation).")
        drizzle_layout.addWidget(self.cfa_drizzle_cb)

        layout.addLayout(drizzle_layout)

        # ─────────────────────────────────────────
        # 5) Reference Frame Selection
        # ─────────────────────────────────────────
        self.ref_frame_label = QLabel("Select Reference Frame:")
        self.ref_frame_path = QLabel("No file selected")
        self.ref_frame_path.setWordWrap(True)
        self.select_ref_frame_btn = QPushButton("Select Reference Frame")
        self.select_ref_frame_btn.clicked.connect(self.select_reference_frame)

        self.auto_accept_ref_cb = QCheckBox("Auto-accept measured reference")
        self.auto_accept_ref_cb.setChecked(self.settings.value("stacking/auto_accept_ref", False, type=bool))
        self.auto_accept_ref_cb.setToolTip("If checked, the best measured frame is accepted automatically.")
        self.auto_accept_ref_cb.toggled.connect(
            lambda v: self.settings.setValue("stacking/auto_accept_ref", bool(v))
        )

        ref_layout = QHBoxLayout()
        ref_layout.addWidget(self.ref_frame_label)
        ref_layout.addWidget(self.ref_frame_path, 1)
        ref_layout.addWidget(self.select_ref_frame_btn)
        ref_layout.addWidget(self.auto_accept_ref_cb)
        ref_layout.addStretch()
        layout.addLayout(ref_layout)

        # Disable Select button when auto-accept is on
        self.auto_accept_ref_cb.toggled.connect(self.select_ref_frame_btn.setDisabled)
        self.select_ref_frame_btn.setDisabled(self.auto_accept_ref_cb.isChecked())

        # ─────────────────────────────────────────
        # 6) MFDeconv (title cleaned; no “beta”)
        # ─────────────────────────────────────────
        mf_box = QGroupBox("MFDeconv — Multi-Frame Deconvolution (ImageMM)")
        mf_v = QVBoxLayout(mf_box)

        def _get(key, default, t):
            return self.settings.value(key, default, type=t)

        # row 1: enable
        mf_row1 = QHBoxLayout()
        self.mf_enabled_cb = QCheckBox("Enable MFDeconv during integration")
        self.mf_enabled_cb.setChecked(_get("stacking/mfdeconv/enabled", False, bool))
        self.mf_enabled_cb.setToolTip("Runs multi-frame deconvolution during integration. Turn off if testing.")
        mf_row1.addWidget(self.mf_enabled_cb)

        # NEW: Super-Resolution checkbox goes here (between Enable and Save-Intermediate)
        self.mf_sr_cb = QCheckBox("Super Resolution")
        self.mf_sr_cb.setToolTip(
            "Reconstruct on an r× super-res grid using SR PSFs.\n"
            "Compute cost grows roughly ~ r^4. Drizzle usually provides better results."
        )
        self.mf_sr_cb.setChecked(self.settings.value("stacking/mfdeconv/sr_enabled", False, type=bool))
        mf_row1.addWidget(self.mf_sr_cb)

        # Integer box to the RIGHT of the checkbox (2..4× typical; tweak if you want bigger)
        mf_row1.addSpacing(6)
        self.mf_sr_factor_spin = QSpinBox()
        self.mf_sr_factor_spin.setRange(2, 4)         # set 2..8 if you dare; r^4 cost!
        self.mf_sr_factor_spin.setSingleStep(1)
        self.mf_sr_factor_spin.setSuffix("×")
        self.mf_sr_factor_spin.setToolTip("Super-resolution scale factor r (integer ≥2).")
        self.mf_sr_factor_spin.setValue(self.settings.value("stacking/mfdeconv/sr_factor", 2, type=int))
        self.mf_sr_factor_spin.setEnabled(self.mf_sr_cb.isChecked())
        mf_row1.addWidget(self.mf_sr_factor_spin)

        mf_row1.addSpacing(16)

        self.mf_save_intermediate_cb = QCheckBox("Save intermediate iterative images")
        self.mf_save_intermediate_cb.setToolTip("If enabled, saves the seed and every iteration image into a subfolder next to the final output.")
        self.mf_save_intermediate_cb.setChecked(self.settings.value("stacking/mfdeconv/save_intermediate", False, type=bool))
        mf_row1.addWidget(self.mf_save_intermediate_cb)

        mf_row1.addStretch(1)
        mf_v.addLayout(mf_row1)

        # row 2: iterations, min iters, kappa
        mf_row2 = QHBoxLayout()
        mf_row2.addWidget(QLabel("Iterations (max):"))
        self.mf_iters_spin = QSpinBox(); self.mf_iters_spin.setRange(1, 500)
        self.mf_iters_spin.setValue(_get("stacking/mfdeconv/iters", 20, int))
        mf_row2.addWidget(self.mf_iters_spin)

        mf_row2.addSpacing(12)
        mf_row2.addWidget(QLabel("Min iters:"))
        self.mf_min_iters_spin = QSpinBox(); self.mf_min_iters_spin.setRange(1, 500)
        self.mf_min_iters_spin.setValue(_get("stacking/mfdeconv/min_iters", 3, int))
        mf_row2.addWidget(self.mf_min_iters_spin)

        mf_row2.addSpacing(16)
        mf_row2.addWidget(QLabel("Update clip (κ):"))
        self.mf_kappa_spin = QDoubleSpinBox(); self.mf_kappa_spin.setRange(0.0, 10.0)
        self.mf_kappa_spin.setDecimals(3); self.mf_kappa_spin.setSingleStep(0.1)
        self.mf_kappa_spin.setValue(_get("stacking/mfdeconv/kappa", 2.0, float))
        # NEW: κ tooltip (layman’s terms)
        self.mf_kappa_spin.setToolTip(
            "κ (kappa) limits how big each multiplicative update can be per iteration.\n"
            "• κ = 1.0 → no change (1×); larger κ allows bigger step sizes.\n"
            "• Lower values = gentler, safer updates; higher values = faster but riskier.\n"
            "Typical: 1.05–1.5 for conservative, ~2 for punchier updates."
        )
        mf_row2.addWidget(self.mf_kappa_spin)
        mf_row2.addStretch(1)
        mf_v.addLayout(mf_row2)

        # row 3: color / rho / huber / toggles
        mf_row3 = QHBoxLayout()
        mf_row3.addWidget(QLabel("Color mode:"))
        self.mf_color_combo = QComboBox(); self.mf_color_combo.addItems(["PerChannel", "Luma"])
        _cm = _get("stacking/mfdeconv/color_mode", "PerChannel", str)
        if _cm not in ("PerChannel", "Luma"): _cm = "PerChannel"
        self.mf_color_combo.setCurrentText(_cm)
        self.mf_color_combo.setToolTip("‘Luma’ deconvolves luminance only; ‘PerChannel’ runs on RGB independently.")
        mf_row3.addWidget(self.mf_color_combo)

        mf_row3.addSpacing(16)
        mf_row3.addWidget(QLabel("ρ (loss):"))
        self.mf_rho_combo = QComboBox(); self.mf_rho_combo.addItems(["Huber", "L2"])
        self.mf_rho_combo.setCurrentText(self.settings.value("stacking/mfdeconv/rho", "Huber", type=str))
        self.mf_rho_combo.currentTextChanged.connect(lambda s: self.settings.setValue("stacking/mfdeconv/rho", s))
        mf_row3.addWidget(self.mf_rho_combo)

        mf_row3.addSpacing(16)
        mf_row3.addWidget(QLabel("Huber δ:"))
        self.mf_Huber_spin = QDoubleSpinBox()
        self.mf_Huber_spin.setRange(-1000.0, 1000.0); self.mf_Huber_spin.setDecimals(4); self.mf_Huber_spin.setSingleStep(0.1)
        self.mf_Huber_spin.setValue(_get("stacking/mfdeconv/Huber_delta", -2.0, float))
        # NEW: Huber tooltip (layman’s terms)
        self.mf_Huber_spin.setToolTip(
            "Huber δ sets the cutoff between ‘quadratic’ (treat as normal) and ‘linear’ (treat as outlier) behavior.\n"
            "• |residual| ≤ δ → quadratic (more aggressive corrections)\n"
            "• |residual| > δ → linear (gentler, more robust)\n"
            "Negative values mean ‘scale by RMS’: e.g., δ = -2 uses 2×RMS.\n"
            "Smaller |δ| (closer to 0) → more pixels counted as outliers → more conservative.\n"
            "Examples: κ=1.1 & δ=-0.7 = gentle; κ=2 & δ=-2 = more aggressive."
        )
        mf_row3.addWidget(self.mf_Huber_spin)

        self.mf_Huber_hint = QLabel("(<0 = scale×RMS, >0 = absolute Δ)")
        self.mf_Huber_hint.setStyleSheet("color:#888;")
        mf_row3.addWidget(self.mf_Huber_hint)

        mf_row3.addSpacing(16)
        self.mf_use_star_mask_cb = QCheckBox("Auto Star Mask")
        self.mf_use_noise_map_cb = QCheckBox("Auto Noise Map")
        self.mf_use_star_mask_cb.setChecked(self.settings.value("stacking/mfdeconv/use_star_masks", False, type=bool))
        self.mf_use_noise_map_cb.setChecked(self.settings.value("stacking/mfdeconv/use_noise_maps", False, type=bool))
        mf_row3.addWidget(self.mf_use_star_mask_cb)
        mf_row3.addWidget(self.mf_use_noise_map_cb)
        mf_row3.addStretch(1)
        mf_v.addLayout(mf_row3)

        # persist
        self.mf_enabled_cb.toggled.connect(lambda v: self.settings.setValue("stacking/mfdeconv/enabled", bool(v)))
        self.mf_iters_spin.valueChanged.connect(lambda v: self.settings.setValue("stacking/mfdeconv/iters", int(v)))
        self.mf_min_iters_spin.valueChanged.connect(lambda v: self.settings.setValue("stacking/mfdeconv/min_iters", int(v)))
        self.mf_kappa_spin.valueChanged.connect(lambda v: self.settings.setValue("stacking/mfdeconv/kappa", float(v)))
        self.mf_color_combo.currentTextChanged.connect(lambda s: self.settings.setValue("stacking/mfdeconv/color_mode", s))
        self.mf_Huber_spin.valueChanged.connect(lambda v: self.settings.setValue("stacking/mfdeconv/Huber_delta", float(v)))
        self.mf_use_star_mask_cb.toggled.connect(lambda v: self.settings.setValue("stacking/mfdeconv/use_star_masks", bool(v)))
        self.mf_use_noise_map_cb.toggled.connect(lambda v: self.settings.setValue("stacking/mfdeconv/use_noise_maps", bool(v)))
        self.mf_sr_cb.toggled.connect(lambda v: self.settings.setValue("stacking/mfdeconv/sr_enabled", bool(v)))
        self.mf_save_intermediate_cb.toggled.connect(lambda v: self.settings.setValue("stacking/mfdeconv/save_intermediate", bool(v)))

        layout.addWidget(mf_box)

        # ─────────────────────────────────────────
        # 7) Comet + Star-Trail checkboxes (same row) — now BELOW MFDeconv
        # ─────────────────────────────────────────
        comet_trail_row = QHBoxLayout()
        self.comet_cb = QCheckBox("🌠 Create comet stack (comet-aligned)")
        self.comet_cb.setChecked(self.settings.value("stacking/comet/enabled", False, type=bool))
        comet_trail_row.addWidget(self.comet_cb)

        comet_trail_row.addSpacing(12)
        self.trail_cb = QCheckBox("★★ Star-Trail Mode ★★ (Max-Value Stack)")
        self.trail_cb.setChecked(self.star_trail_mode)
        self.trail_cb.setToolTip("Skip registration/alignment and use Maximum-Intensity projection for star trails")
        self.trail_cb.stateChanged.connect(self._on_star_trail_toggled)
        comet_trail_row.addWidget(self.trail_cb)
        comet_trail_row.addStretch(1)
        layout.addLayout(comet_trail_row)

        # keep comet options in a compact row just beneath
        comet_opts = QHBoxLayout()
        self.comet_pick_btn = QPushButton("Pick comet center…")
        self.comet_pick_btn.setEnabled(self.comet_cb.isChecked())
        self.comet_pick_btn.clicked.connect(self._pick_comet_center)
        self.comet_cb.toggled.connect(self.comet_pick_btn.setEnabled)
        comet_opts.addWidget(self.comet_pick_btn)

        self.comet_blend_cb = QCheckBox("Also output Stars+Comet blend")
        self.comet_blend_cb.setChecked(self.settings.value("stacking/comet/blend", True, type=bool))
        comet_opts.addWidget(self.comet_blend_cb)

        comet_opts.addWidget(QLabel("Mix:"))
        self.comet_mix = QDoubleSpinBox(); self.comet_mix.setRange(0.0, 1.0); self.comet_mix.setSingleStep(0.05)
        self.comet_mix.setValue(self.settings.value("stacking/comet/mix", 1.0, type=float))
        comet_opts.addWidget(self.comet_mix)
        self.comet_save_starless_cb = QCheckBox("Save all starless comet-aligned frames")
        self.comet_save_starless_cb.setChecked(self.settings.value("stacking/comet/save_starless", False, type=bool))
        comet_opts.addWidget(self.comet_save_starless_cb)        
        comet_opts.addStretch(1)

        layout.addLayout(comet_opts)

        # persist comet settings
        self.settings.setValue("stacking/comet/enabled", self.comet_cb.isChecked())
        self.settings.setValue("stacking/comet/blend", self.comet_blend_cb.isChecked())
        self.settings.setValue("stacking/comet/mix", self.comet_mix.value())
        self.comet_save_starless_cb.toggled.connect(
            lambda v: self.settings.setValue("stacking/comet/save_starless", bool(v))
        )
        # ─────────────────────────────────────────
        # 8) Backend / Install GPU Acceleration — MOVED BELOW comet+trail row
        # ─────────────────────────────────────────
        accel_row = QHBoxLayout()
        self.backend_label = QLabel(f"Backend: {current_backend()}")
        accel_row.addWidget(self.backend_label)

        self.install_accel_btn = QPushButton("Install/Update GPU Acceleration…")
        self.install_accel_btn.setToolTip("Downloads PyTorch with the right backend (CUDA/MPS/CPU). One-time per machine.")
        accel_row.addWidget(self.install_accel_btn)

        gpu_help_btn = QToolButton()
        gpu_help_btn.setText("?")
        gpu_help_btn.setToolTip("If GPU still not being used — click for fix steps")
        gpu_help_btn.clicked.connect(self._show_gpu_accel_fix_help)
        accel_row.addWidget(gpu_help_btn)

        accel_row.addStretch(1)
        layout.addLayout(accel_row)

        # same installer wiring as before
        def _install_accel():
            v = sys.version_info
            if not (v.major == 3 and v.minor in (10, 11, 12)):
                why = (f"This app is running on Python {v.major}.{v.minor}. "
                    "GPU acceleration requires Python 3.10, 3.11, or 3.12.")
                tip = ""
                sysname = platform.system()
                if sysname == "Darwin":
                    tip = ("\n\nmacOS tip (Apple Silicon):\n"
                        " • Install Python 3.12:  brew install python@3.12\n"
                        " • Then relaunch the app so it can create its runtime with 3.12.")
                elif sysname == "Windows":
                    tip = ("\n\nWindows tip:\n"
                        " • Install Python 3.12/3.11/3.10 (x64) from python.org\n"
                        " • Then relaunch the app.")
                else:
                    tip = ("\n\nLinux tip:\n"
                        " • Install python3.12 or 3.11 via your package manager\n"
                        " • Then relaunch the app.")

                QMessageBox.warning(self, "Unsupported Python Version", why + tip)
                # reflect the abort in UI/status and leave button enabled
                try:
                    self.backend_label.setText("Backend: CPU (Python version not supported for GPU install)")
                    self.status_signal.emit("❌ GPU Acceleration install aborted: unsupported Python version.")
                except Exception:
                    pass
                return
            self.install_accel_btn.setEnabled(False)
            self.backend_label.setText("Backend: installing…")
            self._accel_pd = QProgressDialog("Preparing runtime…", "Cancel", 0, 0, self)
            self._accel_pd.setWindowTitle("Installing GPU Acceleration")
            self._accel_pd.setWindowModality(Qt.WindowModality.ApplicationModal)
            self._accel_pd.setAutoClose(True)
            self._accel_pd.setMinimumDuration(0)
            self._accel_pd.show()

            self._accel_thread = QThread(self)
            self._accel_worker = AccelInstallWorker(prefer_gpu=True)
            self._accel_worker.moveToThread(self._accel_thread)
            self._accel_thread.started.connect(self._accel_worker.run, Qt.ConnectionType.QueuedConnection)
            self._accel_worker.progress.connect(self._accel_pd.setLabelText, Qt.ConnectionType.QueuedConnection)
            self._accel_worker.progress.connect(lambda s: self.status_signal.emit(s), Qt.ConnectionType.QueuedConnection)

            def _cancel():
                if self._accel_thread.isRunning():
                    self._accel_thread.requestInterruption()
            self._accel_pd.canceled.connect(_cancel, Qt.ConnectionType.QueuedConnection)

            def _done(ok: bool, msg: str):
                if getattr(self, "_accel_pd", None):
                    self._accel_pd.reset(); self._accel_pd.deleteLater(); self._accel_pd = None
                self._accel_thread.quit(); self._accel_thread.wait()
                self.install_accel_btn.setEnabled(True)
                from pro.accel_installer import current_backend
                self.backend_label.setText(f"Backend: {current_backend()}")
                self.status_signal.emit(("✅ " if ok else "❌ ") + msg)
                if ok: QMessageBox.information(self, "Acceleration", f"✅ {msg}")
                else:  QMessageBox.warning(self, "Acceleration", f"❌ {msg}")

            self._accel_worker.finished.connect(_done, Qt.ConnectionType.QueuedConnection)
            self._accel_thread.finished.connect(self._accel_worker.deleteLater, Qt.ConnectionType.QueuedConnection)
            self._accel_thread.finished.connect(self._accel_thread.deleteLater, Qt.ConnectionType.QueuedConnection)
            self._accel_thread.start()

        self.install_accel_btn.clicked.connect(_install_accel)

        # ─────────────────────────────────────────
        # 9) Action Buttons
        # ─────────────────────────────────────────
        self.register_images_btn = QPushButton("🔥🚀Register and Integrate Images🔥🚀")
        self.register_images_btn.clicked.connect(self.register_images)
        self.register_images_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF4500;
                color: white;
                font-size: 16px;
                padding: 8px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #FF6347; }
        """)
        layout.addWidget(self.register_images_btn)

        self._registration_busy = False

        self.integrate_registered_btn = QPushButton("Integrate Previously Registered Images")
        self.integrate_registered_btn.clicked.connect(self.integrate_registered_images)
        self.integrate_registered_btn.setStyleSheet("""
            QPushButton {
                background-color: #333;
                color: white;
                font-size: 14px;
                padding: 8px;
                border-radius: 5px;
                border: 2px solid yellow;
            }
            QPushButton:hover  { border: 2px solid #FFD700; }
            QPushButton:pressed{ background-color: #222; border: 2px solid #FFA500; }
        """)
        layout.addWidget(self.integrate_registered_btn)

        # ─────────────────────────────────────────
        # 10) Init + persist bits
        # ─────────────────────────────────────────
        self.populate_calibrated_lights()
        self._refresh_reg_tree_summaries()
        tab.setLayout(layout)

        self.drizzle_checkbox.setChecked(self.settings.value("stacking/drizzle_enabled", False, type=bool))
        self.drizzle_scale_combo.setCurrentText(self.settings.value("stacking/drizzle_scale", "2x", type=str))
        self.drizzle_drop_shrink_spin.setValue(self.settings.value("stacking/drizzle_drop", 0.65, type=float))

        drizzle_on = self.settings.value("stacking/drizzle_enabled", False, type=bool)
        self.cfa_drizzle_cb.setEnabled(drizzle_on)
        if not drizzle_on and self.cfa_drizzle_cb.isChecked():
            self.cfa_drizzle_cb.blockSignals(True)
            self.cfa_drizzle_cb.setChecked(False)
            self.cfa_drizzle_cb.blockSignals(False)
            self.settings.setValue("stacking/cfa_drizzle", False)
        self._update_drizzle_summary_columns()

        # persist SR on/off and factor
        self.mf_sr_cb.toggled.connect(lambda v: self.settings.setValue("stacking/mfdeconv/sr_enabled", bool(v)))
        self.mf_sr_cb.toggled.connect(self.mf_sr_factor_spin.setEnabled)
        self.mf_sr_factor_spin.valueChanged.connect(lambda v: self.settings.setValue("stacking/mfdeconv/sr_factor", int(v)))

        # persist auto-crop settings
        self.autocrop_cb.toggled.connect(
            lambda v: (self.settings.setValue("stacking/autocrop_enabled", bool(v)),
                    self.settings.sync())
        )
        self.autocrop_pct.valueChanged.connect(
            lambda v: (self.settings.setValue("stacking/autocrop_pct", float(v)),
                    self.settings.sync())
        )


        # If comet star-removal is globally disabled, gray this out with a helpful tooltip
        csr_enabled_globally = self.settings.value("stacking/comet_starrem/enabled", False, type=bool)

        def _refresh_comet_starless_enable():
            csr_on   = bool(self.settings.value("stacking/comet_starrem/enabled", False, type=bool))
            comet_on = bool(self.comet_cb.isChecked())
            ok = comet_on and csr_on

            self.comet_save_starless_cb.setEnabled(ok)

            tip = ("Save all comet-aligned starless frames.\n"
                "Requires: ‘Create comet stack’ AND ‘Remove stars on comet-aligned frames’.")
            if not csr_on:
                tip += "\n\n(Comet Star Removal is currently OFF in Settings.)"
            self.comet_save_starless_cb.setToolTip(tip)

            if not ok and self.comet_save_starless_cb.isChecked():
                self.comet_save_starless_cb.blockSignals(True)
                self.comet_save_starless_cb.setChecked(False)
                self.comet_save_starless_cb.blockSignals(False)
                self.settings.setValue("stacking/comet/save_starless", False)

        self._refresh_comet_starless_enable = _refresh_comet_starless_enable  # expose to self

        # wire it
        self.comet_cb.toggled.connect(lambda _v: self._refresh_comet_starless_enable())
        self.comet_save_starless_cb.toggled.connect(lambda _v: self._refresh_comet_starless_enable())  # optional belt/suspenders

        # run once AFTER you restore all initial states from settings
        self._refresh_comet_starless_enable()


        return tab


    def _show_gpu_accel_fix_help(self):
        from PyQt6.QtWidgets import QMessageBox, QApplication
        msg = QMessageBox(self)
        msg.setWindowTitle("GPU still not being used?")
        msg.setIcon(QMessageBox.Icon.Information)
        msg.setText(
            "Open Command Prompt and run the following.\n\n"
            "Step 1: uninstall PyTorch\n"
            "Step 2: install the correct build for your GPU"
        )

        # Exact commands (kept as Windows-friendly with %LOCALAPPDATA%)
        cmds = r'''
    "%LOCALAPPDATA%\SASpro\runtime\py312\venv\Scripts\python.exe" -m pip uninstall -y torch

    -> Then install ONE of the following:

    -> AMD / Intel GPUs:
    "%LOCALAPPDATA%\SASpro\runtime\py312\venv\Scripts\python.exe" -m pip install torch-directml

    -> NVIDIA GPUs (CUDA 12.9):
    "%LOCALAPPDATA%\SASpro\runtime\py312\venv\Scripts\python.exe" -m pip install torch --extra-index-url https://download.pytorch.org/whl/cu129
    '''.strip()

        # Show commands in the expandable details area
        msg.setDetailedText(cmds)

        # Add a one-click copy button
        copy_btn = msg.addButton("Copy commands", QMessageBox.ButtonRole.ActionRole)
        msg.addButton(QMessageBox.StandardButton.Close)

        msg.exec()
        if msg.clickedButton() is copy_btn:
            QApplication.clipboard().setText(cmds)


    def _on_drizzle_checkbox_toggled(self, checked: bool):
        """Drizzle master switch."""
        self.drizzle_scale_combo.setEnabled(checked)
        self.drizzle_drop_shrink_spin.setEnabled(checked)

        # NEW: make CFA Drizzle depend on Enable Drizzle
        self.cfa_drizzle_cb.setEnabled(checked)
        if not checked and self.cfa_drizzle_cb.isChecked():
            # force false without re-entering handlers
            self.cfa_drizzle_cb.blockSignals(True)
            self.cfa_drizzle_cb.setChecked(False)
            self.cfa_drizzle_cb.blockSignals(False)
            self.settings.setValue("stacking/cfa_drizzle", False)

        self.settings.setValue("stacking/drizzle_enabled", bool(checked))
        self._update_drizzle_summary_columns()

    def _on_drizzle_param_changed(self, *_):
        # Persist drizzle params whenever changed
        self.settings.setValue("stacking/drizzle_scale", self.drizzle_scale_combo.currentText())
        self.settings.setValue("stacking/drizzle_drop", float(self.drizzle_drop_shrink_spin.value()))
        # If you reflect params to tree rows, update here:
        # self._refresh_reg_tree_drizzle_column()

    def _on_star_trail_toggled(self, enabled: bool):
        """
        When Star-Trail mode is ON, we skip registration/alignment and use max-value stack.
        Disable other registration-dependent features (drizzle/comet/MFDeconv) to avoid confusion.
        """
        # Controls to gate
        drizzle_widgets = (self.drizzle_checkbox, self.drizzle_scale_combo, self.drizzle_drop_shrink_spin, self.cfa_drizzle_cb)
        comet_widgets = (self.comet_cb, self.comet_pick_btn, self.comet_blend_cb, self.comet_mix)
        mf_widgets = (self.mf_enabled_cb, self.mf_iters_spin, self.mf_kappa_spin, self.mf_color_combo, self.mf_Huber_spin)

        for w in drizzle_widgets + comet_widgets + mf_widgets:
            w.setEnabled(not enabled)

        if enabled:
            self.status_signal.emit("⭐ Star-Trail Mode enabled: Drizzle, Comet stack, and MFDeconv disabled.")
        else:
            self.status_signal.emit("⭐ Star-Trail Mode disabled: other options re-enabled.")


    def _pick_comet_center(self):
        """
        Let the user click a point on ANY light frame. We store (file_path, x, y)
        and defer mapping into the reference frame until after alignment.
        """
        # choose a source file
        src_path = None

        # 1) try current selection in reg_tree
        it = self._first_selected_leaf(self.reg_tree) if hasattr(self, "_first_selected_leaf") else None
        if it and it.parent() is not None:
            group = it.parent().text(0)
            fname = it.text(0)
            # reconstruct full path from our dicts
            lst = self.light_files.get(group) or []
            for p in lst:
                if os.path.basename(p) == fname or os.path.splitext(os.path.basename(p))[0] in fname:
                    src_path = p; break

        # 2) else, fall back to “first light”, or prompt
        if not src_path:
            all_files = [f for lst in self.light_files.values() for f in lst]
            if all_files:
                src_path = all_files[0]
            else:
                fp, _ = QFileDialog.getOpenFileName(
                    self, "Pick a frame to mark the comet center", self.stacking_directory or "",
                    "Images (*.fit *.fits *.tif *.tiff *.png *.jpg *.jpeg)"
                )
                if not fp:
                    QMessageBox.information(self, "Comet Center", "No file chosen.")
                    return
                src_path = fp

        # load and show a simple click-to-pick dialog
        try:
            img, hdr, _, _ = load_image(src_path)
            if img is None:
                raise RuntimeError("Failed to load image.")
        except Exception as e:
            QMessageBox.critical(self, "Comet Center", f"Could not load:\n{src_path}\n\n{e}")
            return

        dlg = _SimplePickDialog(img, parent=self)  # small helper below
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        x, y = dlg.point()

        # store the seed in ORIGINAL file space (or the path we used)
        self._comet_seed = {"path": os.path.normpath(src_path), "xy": (float(x), float(y))}
        self._comet_ref_xy = None  # will be resolved post-align
        self.update_status(f"🌠 Comet seed set on {os.path.basename(src_path)} at ({x:.1f}, {y:.1f}).")


    def _on_cfa_drizzle_toggled(self, checked: bool):
        # If Drizzle is OFF, CFA Drizzle must remain False
        if not self.drizzle_checkbox.isChecked():
            if checked:
                self.cfa_drizzle_cb.blockSignals(True)
                self.cfa_drizzle_cb.setChecked(False)
                self.cfa_drizzle_cb.blockSignals(False)
            checked = False

        self.settings.setValue("stacking/cfa_drizzle", bool(checked))
        self._update_drizzle_summary_columns()


    def _on_drizzle_param_changed(self, *_):
        # persist
        self.settings.setValue("stacking/drizzle_scale", self.drizzle_scale_combo.currentText())
        self.settings.setValue("stacking/drizzle_drop", float(self.drizzle_drop_shrink_spin.value()))
        self._update_drizzle_summary_columns()

    def _update_drizzle_summary_columns(self):
        desc = "OFF"
        if self.drizzle_checkbox.isChecked():
            scale = self.drizzle_scale_combo.currentText()
            drop  = self.drizzle_drop_shrink_spin.value()
            desc = f"ON, Scale {scale}, Drop {drop:.2f}"
            if self.cfa_drizzle_cb.isChecked():
                desc += " + CFA"

        root = self.reg_tree.invisibleRootItem()
        for i in range(root.childCount()):
            root.child(i).setText(2, f"Drizzle: {desc}")

    def _on_star_trail_toggled(self, state):
        self.star_trail_mode = bool(state)
        self.settings.setValue("stacking/star_trail_mode", self.star_trail_mode)
        # if they turn it on, immediately override the rejection combo:
        if self.star_trail_mode:
            self.rejection_algorithm = "Maximum Value"
        else:
            # reload whatever the user picked
            self.rejection_algorithm = self.settings.value("stacking/rejection_algorithm",
                                                          self.rejection_algorithm,
                                                          type=str)

    def select_reference_frame(self):
        """ Opens a file dialog to select the reference frame. """
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Reference Frame", "", 
                                                "FITS Images (*.fits *.fit);;All Files (*)")
        if file_path:
            self.reference_frame = file_path
            self.ref_frame_path.setText(os.path.basename(file_path))

    def save_master_paths_to_settings(self):
        """Save current master dark and flat paths to QSettings using their actual trees."""

        # Master Darks
        dark_paths = []
        for i in range(self.master_dark_tree.topLevelItemCount()):
            group = self.master_dark_tree.topLevelItem(i)
            for j in range(group.childCount()):
                fname = group.child(j).text(0)
                for path in self.master_files.values():
                    if os.path.basename(path) == fname:
                        dark_paths.append(path)

        # Master Flats
        flat_paths = []
        for i in range(self.master_flat_tree.topLevelItemCount()):
            group = self.master_flat_tree.topLevelItem(i)
            for j in range(group.childCount()):
                fname = group.child(j).text(0)
                for path in self.master_files.values():
                    if os.path.basename(path) == fname:
                        flat_paths.append(path)

        self.settings.setValue("stacking/master_darks", dark_paths)
        self.settings.setValue("stacking/master_flats", flat_paths)

    def clear_tree_selection(self, tree, file_dict):
        """Clears selected items from a simple (non-tuple-keyed) tree like Master Darks or Darks tab."""
        selected_items = tree.selectedItems()
        if not selected_items:
            return

        for item in selected_items:
            parent = item.parent()
            if parent is None:
                # Top-level group item
                key = item.text(0)
                if key in file_dict:
                    del file_dict[key]
                tree.takeTopLevelItem(tree.indexOfTopLevelItem(item))
            else:
                # Child item
                key = parent.text(0)
                filename = item.text(0)
                if key in file_dict:
                    file_dict[key] = [f for f in file_dict[key] if os.path.basename(f) != filename]
                    if not file_dict[key]:
                        del file_dict[key]
                parent.removeChild(item)


    def clear_tree_selection_light(self, tree):
        """Clears the selection in the light tree and updates self.light_files accordingly."""
        selected_items = tree.selectedItems()
        if not selected_items:
            return

        for item in selected_items:
            parent = item.parent()
            if parent is None:
                # Top-level filter node selected
                filter_name = item.text(0)
                # Remove all composite keys whose group_key starts with filter_name
                keys_to_remove = [key for key in list(self.light_files.keys())
                                if isinstance(key, tuple) and key[0].startswith(f"{filter_name} - ")]
                for key in keys_to_remove:
                    del self.light_files[key]
                tree.takeTopLevelItem(tree.indexOfTopLevelItem(item))
            else:
                if parent.parent() is None:
                    # Exposure node selected (child)
                    filter_name = parent.text(0)
                    exposure_text = item.text(0)
                    group_key = f"{filter_name} - {exposure_text}"
                    keys_to_remove = [key for key in list(self.light_files.keys())
                                    if isinstance(key, tuple) and key[0] == group_key]
                    for key in keys_to_remove:
                        del self.light_files[key]
                    parent.removeChild(item)
                else:
                    # Grandchild file node selected
                    filter_name = parent.parent().text(0)
                    exposure_text = parent.text(0)
                    group_key = f"{filter_name} - {exposure_text}"
                    filename = item.text(0)

                    keys_to_check = [key for key in list(self.light_files.keys())
                                    if isinstance(key, tuple) and key[0] == group_key]

                    for key in keys_to_check:
                        self.light_files[key] = [
                            f for f in self.light_files[key] if os.path.basename(f) != filename
                        ]
                        if not self.light_files[key]:
                            del self.light_files[key]
                    parent.removeChild(item)

        self._refresh_light_tree_summaries()            

    def clear_tree_selection_flat(self, tree, file_dict):
        """Clears the selection in the given tree widget and removes items from the corresponding dictionary."""
        selected_items = tree.selectedItems()
        if not selected_items:
            return

        for item in selected_items:
            parent = item.parent()

            if parent:
                # Grandchild level (actual file)
                if parent.parent() is not None:
                    filter_name = parent.parent().text(0)
                    exposure_text = parent.text(0)
                    group_key = f"{filter_name} - {exposure_text}"
                else:
                    # Exposure level
                    filter_name = parent.text(0)
                    exposure_text = item.text(0)
                    group_key = f"{filter_name} - {exposure_text}"

                filename = item.text(0)

                # Remove from all matching (group_key, session) tuples
                keys_to_check = [key for key in list(file_dict.keys())
                                if isinstance(key, tuple) and key[0] == group_key]

                for key in keys_to_check:
                    file_dict[key] = [f for f in file_dict[key] if os.path.basename(f) != filename]
                    if not file_dict[key]:
                        del file_dict[key]

                parent.removeChild(item)
            else:
                # Top-level (filter group) selected
                filter_name = item.text(0)
                keys_to_remove = [key for key in list(file_dict.keys())
                                if isinstance(key, tuple) and key[0].startswith(f"{filter_name} - ")]
                for key in keys_to_remove:
                    del file_dict[key]
                tree.takeTopLevelItem(tree.indexOfTopLevelItem(item))

    def _sync_group_userrole(self, top_item: QTreeWidgetItem):
        paths = []
        for i in range(top_item.childCount()):
            child = top_item.child(i)
            fp = child.data(0, Qt.ItemDataRole.UserRole)
            if fp:
                paths.append(fp)
        top_item.setData(0, Qt.ItemDataRole.UserRole, paths)

    def clear_tree_selection_registration(self, tree):
        """
        Remove selected rows from the Registration tree and *persist* those removals,
        so refreshes / 'Add Light Files' won't resurrect them.
        """
        selected_items = tree.selectedItems()
        if not selected_items:
            return

        # ensure attrs exist
        if not hasattr(self, "_reg_excluded_files"):
            self._reg_excluded_files = set()
        if not hasattr(self, "deleted_calibrated_files"):
            self.deleted_calibrated_files = []

        removed_paths = []

        for item in selected_items:
            parent = item.parent()

            if parent is None:
                # Top-level group
                group_key = item.text(0)

                # paths are stored on the group's UserRole
                full_paths = item.data(0, Qt.ItemDataRole.UserRole) or []
                for p in full_paths:
                    if isinstance(p, str):
                        removed_paths.append(p)

                # Keep internal dict in sync
                self.reg_files.pop(group_key, None)

                # Remove group row
                idx = tree.indexOfTopLevelItem(item)
                if idx >= 0:
                    tree.takeTopLevelItem(idx)

            else:
                # Leaf (single file)
                group_key = parent.text(0)
                fp = item.data(0, Qt.ItemDataRole.UserRole)

                # Track the absolute path if available, else fall back to name
                if isinstance(fp, str):
                    removed_paths.append(fp)
                else:
                    # fallback to name-based match (kept for backward compat)
                    filename = item.text(0)
                    removed_paths.append(filename)

                # Update reg_files
                if group_key in self.reg_files:
                    self.reg_files[group_key] = [
                        f for f in self.reg_files[group_key]
                        if f != fp and os.path.basename(f) != item.text(0)
                    ]
                    if not self.reg_files[group_key]:
                        del self.reg_files[group_key]

                # Remove leaf row
                parent.removeChild(item)

                # Keep parent's stored list in sync (your helper)
                self._sync_group_userrole(parent)

        # Persist the exclusions so they won't reappear on refresh
        self._reg_excluded_files.update(p for p in removed_paths if isinstance(p, str))

        # Maintain your legacy list too (if you still use it elsewhere)
        for p in removed_paths:
            if p not in self.deleted_calibrated_files:
                self.deleted_calibrated_files.append(p)

        # Also prune manual list so it doesn't re-inject removed files
        if hasattr(self, "manual_light_files") and self.manual_light_files:
            self.manual_light_files = [p for p in self.manual_light_files if p not in self._reg_excluded_files]

        # Optional but helpful: rebuild so empty groups disappear cleanly
        self.populate_calibrated_lights()
        self._refresh_reg_tree_summaries()

    def rebuild_flat_tree(self):
        """Regroup flat frames in the flat_tree based on the exposure tolerance."""
        self.flat_tree.clear()
        if not self.flat_files:
            return

        tol = float(self.flat_exposure_tolerance_spinbox.value())

        # Flatten
        all_flats = []
        for (filter_exp_size, session_tag), files in self.flat_files.items():
            for f in files:
                all_flats.append((filter_exp_size, session_tag, f))

        # Group
        grouped = {}  # (filter_name, min_exp, max_exp, image_size) -> [(path, exposure, session)]
        for (filter_exp_size, session_tag, file_path) in all_flats:
            try:
                hdr = fits.getheader(file_path, ext=0)
                filter_name = self._sanitize_name(hdr.get("FILTER", "Unknown"))
                exposure    = float(hdr.get("EXPOSURE", hdr.get("EXPTIME", 0.0)) or 0.0)
                W, H        = hdr.get("NAXIS1", 0), hdr.get("NAXIS2", 0)
                image_size  = f"{W}x{H}" if (W and H) else "Unknown"

                # find compatible bucket (by filter+size+exp within tol)
                found = None
                for (filt, mn, mx, sz) in grouped.keys():
                    if filt == filter_name and sz == image_size and (mn - tol) <= exposure <= (mx + tol):
                        found = (filt, mn, mx, sz); break

                if found:
                    grouped[found].append((file_path, exposure, session_tag))
                else:
                    key = (filter_name, exposure, exposure, image_size)
                    grouped.setdefault(key, []).append((file_path, exposure, session_tag))

            except Exception as e:
                print(f"⚠️ Failed reading {file_path}: {e}")

        # Build tree
        for (filter_name, min_exp, max_exp, image_size), files in grouped.items():
            # normalize the bin so it’s stable
            e_min = min(p[1] for p in files)
            e_max = max(p[1] for p in files)
            # A readable label range (you can tweak formatting)
            expmin = np.floor(e_min)
            exposure_str = f"{expmin:.1f}s–{(expmin + tol):.1f}s" if len(files) > 1 else f"{e_min:.1f}s"

            top_item = QTreeWidgetItem()
            top_item.setText(0, f"{filter_name} - {exposure_str} ({image_size})")
            top_item.setText(1, f"{len(files)} files")

            # ---- canonical group_key (used everywhere) ----
            group_key = f"{filter_name}|{image_size}|{expmin:.3f}|{(expmin+tol):.3f}"
            top_item.setData(0, Qt.ItemDataRole.UserRole, group_key)

            # column 3 shows what will be used
            ud = self.flat_dark_override.get(group_key, None)  # None→Auto, "__NO_DARK__", or path
            if ud is None:
                col2_txt = "Auto" if self.auto_select_dark_checkbox.isChecked() else "None"
            elif ud == "__NO_DARK__":
                col2_txt = "No Calibration"
            else:
                col2_txt = os.path.basename(ud)
            top_item.setText(2, col2_txt)
            top_item.setData(2, Qt.ItemDataRole.UserRole, ud)

            self.flat_tree.addTopLevelItem(top_item)

            # leaves
            for file_path, _, session_tag in files:
                leaf_item = QTreeWidgetItem([
                    os.path.basename(file_path),
                    f"Size: {image_size} | Session: {session_tag}"
                ])
                top_item.addChild(leaf_item)



    def exposures_within_tolerance(self, exp1, exp2, tolerance):
        try:
            return abs(float(exp1) - float(exp2)) <= tolerance
            
        except Exception:
            return False

    def parse_group_key(self, group_key):
        """
        Parses a group key string like 'Luminance - 90s (3000x2000)'
        into filter_name, exposure (float), and image_size (str).
        """
        try:
            parts = group_key.split(' - ')
            filter_name = parts[0]
            exp_size_part = parts[1] if len(parts) > 1 else ""

            # Separate exposure and size correctly
            if '(' in exp_size_part and ')' in exp_size_part:
                exposure_str, size_part = exp_size_part.split('(', 1)
                exposure = exposure_str.replace('s', '').strip()
                size = size_part.strip(') ').strip()
            else:
                exposure = exp_size_part.replace('s', '').strip()
                size = "Unknown"

            
            return filter_name, float(exposure), size

        except Exception as e:
            
            return "Unknown", 0.0, "Unknown"

    def _get_image_size(self, fp):
        ext = os.path.splitext(fp)[1].lower()
        # first try FITS
        if ext in (".fits", ".fit"):
            hdr0 = fits.getheader(fp, ext=0)
            data0 = fits.getdata(fp, ext=0)
            h, w = data0.shape[-2:]
        else:
            # try Pillow
            try:
                with Image.open(fp) as img:
                    w, h = img.size
            except Exception:
                # Pillow failed on TIFF or exotic format → try tifffile
                try:
                    arr = tiff.imread(fp)
                    h, w = arr.shape[:2]
                except Exception:
                    # last resort: OpenCV
                    arr = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
                    if arr is None:
                        raise IOError(f"Cannot read image size for {fp}")
                    h, w = arr.shape[:2]
        return w, h

    def _probe_fits_meta(self, fp: str):
        """
        Return (filter:str, exposure:float, size_str:'WxH') from FITS PRIMARY HDU.
        Robust to missing keywords.
        """
        try:
            hdr0 = fits.getheader(fp, ext=0)
            filt = self._sanitize_name(hdr0.get("FILTER", "Unknown"))
            # exposure can be EXPTIME or EXPOSURE, sometimes a string
            exp_raw = hdr0.get("EXPTIME", hdr0.get("EXPOSURE", 0.0))
            try:
                exp = float(exp_raw)
            except Exception:
                exp = 0.0
            # size
            data0 = fits.getdata(fp, ext=0)
            h, w = (int(data0.shape[-2]), int(data0.shape[-1])) if hasattr(data0, "shape") else (0, 0)
            size = f"{w}x{h}" if (w and h) else "Unknown"
            return filt or "Unknown", float(exp), size
        except Exception as e:
            print(f"⚠️ Could not read FITS {fp}: {e}; treating as generic image")
            return "Unknown", 0.0, "Unknown"


    def _probe_xisf_meta(self, fp: str):
        """
        Return (filter:str, exposure:float, size_str:'WxH') from XISF header.
        Uses first <Image> block; never loads full pixel data.
        """
        try:
            x = XISF(fp)
            ims = x.get_images_metadata()
            if not ims:
                return "Unknown", 0.0, "Unknown"
            m0 = ims[0]  # first image block
            # size from geometry tuple (w,h,channels)
            try:
                w, h, _ = m0.get("geometry", (0, 0, 0))
                size = f"{int(w)}x{int(h)}" if (w and h) else "Unknown"
            except Exception:
                size = "Unknown"

            # FITS-like keywords are stored in dict of lists: {"FILTER":[{"value":..., "comment":...}], ...}
            def _kw(name, default=None):
                lst = m0.get("FITSKeywords", {}).get(name)
                if lst and isinstance(lst, list) and lst[0] and "value" in lst[0]:
                    return lst[0]["value"]
                return default

            filt = self._sanitize_name(_kw("FILTER", "Unknown"))

            exp_raw = _kw("EXPTIME", None)
            if exp_raw is None:
                exp_raw = _kw("EXPOSURE", 0.0)
            try:
                exp = float(exp_raw)
            except Exception:
                exp = 0.0

            # bonus: sometimes exposure is in XISF properties (project dependent). Try a few common ids.
            if exp == 0.0:
                props = m0.get("XISFProperties", {}) or {}
                for pid in ("EXPTIME", "ExposureTime", "XISF:ExposureTime"):
                    v = props.get(pid, {}).get("value")
                    try:
                        exp = float(v)
                        break
                    except Exception:
                        pass

            return filt or "Unknown", float(exp), size
        except Exception as e:
            print(f"⚠️ Could not read XISF {fp}: {e}; treating as generic image")
            return "Unknown", 0.0, "Unknown"


    def populate_calibrated_lights(self):
        from PIL import Image

        def _fmt(enabled, scale, drop):
            return (f"Drizzle: True, Scale: {scale:g}x, Drop: {drop:.2f}" if enabled else "Drizzle: False")

        self.reg_tree.clear()
        self.reg_tree.setColumnCount(3)
        self.reg_tree.setHeaderLabels(["Filter - Exposure - Size", "Metadata", "Drizzle"])
        hdr = self.reg_tree.header()
        for col in (0, 1, 2):
            hdr.setSectionResizeMode(col, QHeaderView.ResizeMode.Interactive)

        # gather files
        calibrated_folder = os.path.join(self.stacking_directory or "", "Calibrated")
        files = []
        if os.path.isdir(calibrated_folder):
            for fn in os.listdir(calibrated_folder):
                files.append(os.path.join(calibrated_folder, fn))

        # include manual files
        files += self.manual_light_files

        # filter exclusions + dedupe
        if self._reg_excluded_files:
            files = [f for f in files if f not in self._reg_excluded_files]
        files = list(dict.fromkeys(files))
        if not files:
            self.light_files = {}
            return

        # group by (filter, ~exposure, size) within tolerance
        grouped = {}  # key -> list of dicts: {"path", "exp", "size"}
        tol = self.exposure_tolerance_spin.value()

        for fp in files:
            ext = os.path.splitext(fp)[1].lower()
            filt = "Unknown"
            exp = 0.0
            size = "Unknown"

            if ext in (".fits", ".fit", ".fz"):
                filt, exp, size = self._probe_fits_meta(fp)
            elif ext == ".xisf":
                filt, exp, size = self._probe_xisf_meta(fp)
            else:
                # generic image (TIFF/PNG/JPEG, etc.)
                try:
                    w, h = self._get_image_size(fp)
                    size = f"{w}x{h}"
                except Exception as e:
                    print(f"⚠️ Cannot read image size for {fp}: {e}")
                    continue

            # find existing group with same filter+size and exposure within tolerance
            match_key = None
            for key in grouped:
                f2, e2, s2 = self.parse_group_key(key)
                if filt == f2 and s2 == size and abs(exp - e2) <= tol:
                    match_key = key
                    break

            key = match_key or f"{filt} - {exp:.1f}s ({size})"
            grouped.setdefault(key, []).append({"path": fp, "exp": exp, "size": size})

        # populate tree & self.light_files
        self.light_files = {}

        # current global drizzle defaults
        global_enabled = self.drizzle_checkbox.isChecked()
        try:
            global_scale = float(self.drizzle_scale_combo.currentText().replace("x", "", 1))
        except Exception:
            global_scale = 1.0
        global_drop = self.drizzle_drop_shrink_spin.value()

        for key, entries in grouped.items():
            paths = [d["path"] for d in entries]
            exps  = [d["exp"]  for d in entries]

            top = QTreeWidgetItem()
            top.setText(0, key)
            if len(exps) > 1:
                mn, mx = min(exps), max(exps)
                top.setText(1, f"{len(paths)} files, {mn:.0f}s–{mx:.0f}s")
            else:
                top.setText(1, f"{len(paths)} file")

            # per-group drizzle state (persisted), default to global
            state = self.per_group_drizzle.get(key)
            if state is None:
                state = {"enabled": bool(global_enabled), "scale": float(global_scale), "drop": float(global_drop)}
                self.per_group_drizzle[key] = state

            try:
                top.setText(2, self._format_drizzle_text(state["enabled"], state["scale"], state["drop"]))
            except AttributeError:
                top.setText(2, _fmt(state["enabled"], state["scale"], state["drop"]))

            top.setData(0, Qt.ItemDataRole.UserRole, paths)
            self.reg_tree.addTopLevelItem(top)

            # leaf rows: show basename + per-file size
            for d in entries:
                fp = d["path"]
                leaf = QTreeWidgetItem([os.path.basename(fp), f"Size: {d['size']}"])
                leaf.setData(0, Qt.ItemDataRole.UserRole, fp)
                top.addChild(leaf)

            top.setExpanded(True)
            self.light_files[key] = paths


    def _iter_group_items(self):
        for i in range(self.reg_tree.topLevelItemCount()):
            yield self.reg_tree.topLevelItem(i)

    def _format_drizzle_text(self, enabled: bool, scale: float, drop: float) -> str:
        return (f"Drizzle: True, Scale: {scale:g}x, Drop: {drop:.2f}"
                if enabled else "Drizzle: False")

    def _set_drizzle_on_items(self, items, enabled: bool, scale: float, drop: float):
        txt_on  = self._format_drizzle_text(True,  scale, drop)
        txt_off = self._format_drizzle_text(False, scale, drop)
        for it in items:
            # dedupe child selection → parent group
            if it.parent() is not None:
                it = it.parent()
            group_key = it.text(0)
            it.setText(2, txt_on if enabled else txt_off)
            self.per_group_drizzle[group_key] = {
                "enabled": bool(enabled),
                "scale": float(scale),
                "drop":  float(drop),
            }

    def update_drizzle_settings(self):
        """
        Called whenever the user toggles the 'Enable Drizzle' checkbox,
        changes the scale combo, or changes the drop shrink spinbox.
        Applies to all *selected* top-level items in the reg_tree.
        """
        # Current states from global controls
        drizzle_enabled = self.drizzle_checkbox.isChecked()
        scale_str = self.drizzle_scale_combo.currentText()  # e.g. "1x","2x","3x"
        drop_val = self.drizzle_drop_shrink_spin.value()    # e.g. 0.65

        # Gather selected items
        selected_items = self.reg_tree.selectedItems()
        if not selected_items:
            return

        for item in selected_items:
            # If the user selected a child row, go up to its parent group
            if item.parent() is not None:
                item = item.parent()

            group_key = item.text(0)

            if drizzle_enabled:
                # Show scale + drop shrink
                drizzle_text = (f"Drizzle: True, "
                                f"Scale: {scale_str}, "
                                f"Drop: {drop_val:.2f}")
            else:
                # Just show "Drizzle: False"
                drizzle_text = "Drizzle: False"

            # Update column 2 with the new text
            item.setText(2, drizzle_text)

            # If you also store it in a dictionary:
            self.per_group_drizzle[group_key] = {
                "enabled": drizzle_enabled,
                "scale": float(scale_str.replace("x","", 1)),
                "drop": drop_val
            }

    def _on_drizzle_param_changed(self, *_):
        enabled = self.drizzle_checkbox.isChecked()
        scale   = float(self.drizzle_scale_combo.currentText().replace("x","",1))
        drop    = self.drizzle_drop_shrink_spin.value()

        sel = self.reg_tree.selectedItems()
        if sel:
            # update selected groups
            seen, targets = set(), []
            for it in sel:
                top = it if it.parent() is None else it.parent()
                key = top.text(0)
                if key not in seen:
                    seen.add(key); targets.append(top)
        else:
            # no selection → update ALL groups (keeps UI intuitive)
            targets = list(self._iter_group_items())

        self._set_drizzle_on_items(targets, enabled, scale, drop)

    def gather_drizzle_settings_from_tree(self):
        """Return per-group drizzle settings based on the global controls."""
        enabled = bool(self.drizzle_checkbox.isChecked())
        scale_txt = self.drizzle_scale_combo.currentText()
        try:
            scale_factor = float(scale_txt.replace("x", "").strip())
        except Exception:
            scale_factor = 1.0
        drop_shrink = float(self.drizzle_drop_shrink_spin.value())

        out = {}
        root = self.reg_tree.invisibleRootItem()
        for i in range(root.childCount()):
            group_key = root.child(i).text(0)   # e.g. "L Ultimate - 300.0s (4144x2822)"
            out[group_key] = {
                "drizzle_enabled": enabled,
                "scale_factor": scale_factor,
                "drop_shrink": drop_shrink,
            }
        # Optional: debug once to verify
        self.update_status(f"🧪 drizzle_dict: {out}")
        return out



    def add_light_files_to_registration(self):
        last_dir = self.settings.value("last_opened_folder", "", type=str)
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Light Frames", last_dir,
            "FITS Files (*.fits *.fit *.fz *.xisf *.tif *.tiff *.png *.jpg *.jpeg)"
        )
        if not files:
            return

        self.settings.setValue("last_opened_folder", os.path.dirname(files[0]))

        # Exclude files the user has removed previously
        new_files = [f for f in files if f not in self._reg_excluded_files]

        # Deduplicate while preserving order
        merged = list(dict.fromkeys(self.manual_light_files + new_files))
        self.manual_light_files = merged

        self.populate_calibrated_lights()
        self._refresh_reg_tree_summaries()


    def on_tab_changed(self, idx):
        try:
            if self.tabs.tabText(idx) == "Flats":
                self.update_override_dark_combo()
        except Exception:
            pass



    def add_dark_files(self):
        self.add_files(self.dark_tree, "Select Dark Files", "DARK")
    
    def add_dark_directory(self):
        self.add_directory(self.dark_tree, "Select Dark Directory", "DARK")

    def add_flat_files(self):
        self.prompt_session_before_adding("FLAT")


    def add_flat_directory(self):
        auto = self.settings.value("stacking/auto_session", True, type=bool)
        if auto:
            # No prompt — auto session tagging happens inside add_directory()
            self.add_directory(self.flat_tree, "Select Flat Directory", "FLAT")
            self.assign_best_master_dark()
            self.rebuild_flat_tree()
        else:
            # Manual path (will prompt once for a session name)
            self.prompt_session_before_adding("FLAT", directory_mode=True)


    
    def add_light_files(self):
        auto = self.settings.value("stacking/auto_session", True, type=bool)
        if auto:
            self.add_files(self.light_tree, "Select Light Files", "LIGHT")
            self.assign_best_master_files()
        else:
            self.prompt_session_before_adding("LIGHT", directory_mode=False)

    
    def add_light_directory(self):
        auto = self.settings.value("stacking/auto_session", True, type=bool)
        if auto:
            self.add_directory(self.light_tree, "Select Light Directory", "LIGHT")
            self.assign_best_master_files()
        else:
            self.prompt_session_before_adding("LIGHT", directory_mode=True)


    def prompt_session_before_adding(self, frame_type, directory_mode=False):
        # Respect auto-detect; do nothing here if auto is ON
        if self.settings.value("stacking/auto_session", True, type=bool):
            # Defer to the non-prompt paths
            if frame_type.upper() == "FLAT":
                if directory_mode:
                    self.add_directory(self.flat_tree, "Select Flat Directory", "FLAT")
                    self.assign_best_master_dark()
                    self.rebuild_flat_tree()
                else:
                    self.add_files(self.flat_tree, "Select Flat Files", "FLAT")
                    self.assign_best_master_dark()
                    self.rebuild_flat_tree()
            else:
                if directory_mode:
                    self.add_directory(self.light_tree, "Select Light Directory", "LIGHT")
                else:
                    self.add_files(self.light_tree, "Select Light Files", "LIGHT")
                self.assign_best_master_files()
            return

        # Manual session flow (auto OFF): ask once
        text, ok = QInputDialog.getText(self, "Set Session Tag", "Enter session name:", text="Default")
        if not (ok and text.strip()):
            return
        self.current_session_tag = text.strip()

        if frame_type.upper() == "FLAT":
            if directory_mode:
                self.add_directory(self.flat_tree, "Select Flat Directory", "FLAT")
            else:
                self.add_files(self.flat_tree, "Select Flat Files", "FLAT")
            self.assign_best_master_dark()
            self.rebuild_flat_tree()
        else:
            if directory_mode:
                self.add_directory(self.light_tree, "Select Light Directory", "LIGHT")
            else:
                self.add_files(self.light_tree, "Select Light Files", "LIGHT")
            self.assign_best_master_files()


    def load_master_dark(self):
        """ Loads a Master Dark and updates the UI. """
        last_dir = self.settings.value("last_opened_folder", "", type=str)  # Get last folder
        files, _ = QFileDialog.getOpenFileNames(self, "Select Master Dark", last_dir, "FITS Files (*.fits *.fit)")
        
        if files:
            self.settings.setValue("last_opened_folder", os.path.dirname(files[0]))  # Save last used folder
            self.add_master_files(self.master_dark_tree, "DARK", files)
            self.save_master_paths_to_settings() 

        self.update_override_dark_combo()
        self.assign_best_master_dark()
        self.assign_best_master_files()
        print("DEBUG: Loaded Master Darks and updated assignments.")


    def load_master_flat(self):
        last_dir = self.settings.value("last_opened_folder", "", type=str)
        files, _ = QFileDialog.getOpenFileNames(self, "Select Master Flat", last_dir, "FITS Files (*.fits *.fit)")

        if files:
            self.settings.setValue("last_opened_folder", os.path.dirname(files[0]))
            self.add_master_files(self.master_flat_tree, "FLAT", files)
            self.save_master_paths_to_settings() 


    def add_files(self, tree, title, expected_type):
        """ Adds FITS files and assigns best master files if needed. """
        last_dir = self.settings.value("last_opened_folder", "", type=str)
        files, _ = QFileDialog.getOpenFileNames(
            self, title, last_dir,
            "FITS Files (*.fits *.fit *.fts *.fits.gz *.fit.gz *.fz)"
        )
        if not files:
            return

        self.settings.setValue("last_opened_folder", os.path.dirname(files[0]))

        # Show a standalone progress dialog while ingesting
        self._ingest_paths_with_progress(
            paths=files,
            tree=tree,
            expected_type=expected_type,
            title=f"Adding {expected_type.title()} Files…"
        )

        # Auto-assign after ingest (LIGHT only, same behavior you had)
        if expected_type.upper() == "LIGHT":
            busy = self._busy_progress("Assigning best Master Dark/Flat…")
            try:
                self.assign_best_master_files()
            finally:
                busy.close()

    def on_auto_session_toggled(self, checked: bool):
        self.settings.setValue("stacking/auto_session", bool(checked))
        self.sessionNameEdit.setEnabled(not checked)      # text box
        self.sessionNameButton.setEnabled(not checked)    # any “set session” button


    def add_directory(self, tree, title, expected_type):
        last_dir = self.settings.value("last_opened_folder", "", type=str)
        directory = QFileDialog.getExistingDirectory(self, title, last_dir)
        if not directory:
            return
        self.settings.setValue("last_opened_folder", directory)

        recursive = self.settings.value("stacking/recurse_dirs", True, type=bool)
        paths = self._collect_fits_paths(directory, recursive=recursive)
        if not paths:
            return

        auto_session = self.settings.value("stacking/auto_session", True, type=bool)

        # ✅ Only build auto session map if auto is ON
        session_by_path = self._auto_tag_sessions_for_paths(paths) if auto_session else {}

        # ✅ Manual session name ONLY if auto is OFF
        manual_session_name = None
        if not auto_session:
            from PyQt6.QtWidgets import QInputDialog
            session_name, ok = QInputDialog.getText(self, "Session name",
                                                    "Enter a session name (e.g. 2024-10-09):")
            if ok and session_name.strip():
                manual_session_name = session_name.strip()

        # Ingest (tell it the manual_session_name – it may be None)
        self._ingest_paths_with_progress(
            paths=paths,
            tree=tree,
            expected_type=expected_type,
            title=f"Adding {expected_type.title()} from Directory…",
            manual_session_name=manual_session_name,   # NEW
        )

        # ✅ Apply auto tags to the UI only when auto is ON
        if auto_session:
            target_dict = self.flat_files if expected_type.upper() == "FLAT" else self.light_files
            self._apply_session_tags_to_tree(tree=tree, target_dict=target_dict, session_by_path=session_by_path)

        # As before…
        if expected_type.upper() == "LIGHT":
            busy = self._busy_progress("Assigning best Master Dark/Flat…")
            try:
                self.assign_best_master_files()
            finally:
                busy.close()
        elif expected_type.upper() == "FLAT":
            self.assign_best_master_dark()
            self.rebuild_flat_tree()



    # --- Directory walking ---------------------------------------------------------
    def _collect_fits_paths(self, root: str, recursive: bool = True) -> list[str]:
        exts = (".fits", ".fit", ".fts", ".fits.gz", ".fit.gz", ".fz")
        paths = []
        if recursive:
            for d, _subdirs, files in os.walk(root):
                for f in files:
                    if f.lower().endswith(exts):
                        paths.append(os.path.join(d, f))
        else:
            for f in os.listdir(root):
                if f.lower().endswith(exts):
                    paths.append(os.path.join(root, f))
        # stable order (helps reproducibility + nice UX)
        paths.sort(key=lambda p: (os.path.dirname(p).lower(), os.path.basename(p).lower()))
        return paths


    # --- Session autodetect --------------------------------------------------------
    _SESSION_PATTERNS = [
        # "Night1", "night_02", "Session-3", "sess7"
        (re.compile(r"(session|sess|night|noche|nuit)[ _-]?(\d{1,2})", re.I),
        lambda m: f"Session-{int(m.group(2)):02d}"),

        # ISO-ish dates in folder names: 2024-10-09, 2024_10_09, 20241009
        (re.compile(r"\b(20\d{2})[-_]?([01]\d)[-_]?([0-3]\d)\b"),
        lambda m: f"{m.group(1)}-{m.group(2)}-{m.group(3)}"),
    ]

    def _auto_session_from_path(self, path: str, hdr=None) -> str:
        # 1) Prefer DATE-OBS → local night bucket
        if hdr:
            sess = self._session_from_dateobs_local_night(hdr)
            if sess:
                return sess

        # 2) Otherwise fall back to folder patterns (unchanged)
        parts = [p.lower() for p in Path(path).parts][:-1]
        for name in reversed(parts):
            for rx, fmt in _SESSION_PATTERNS:
                m = rx.search(name)
                if m:
                    return fmt(m)

        # 3) Parent folder name
        try:
            parent = Path(path).parent.name.strip()
            if parent:
                return re.sub(r"\s+", "_", parent)
        except Exception:
            pass

        return "Default"

    def _get_site_timezone(self, hdr=None):
        """
        Resolve a tz for 'local night' bucketing.

        Priority:
        1) QSettings key 'stacking/site_timezone' (e.g., 'America/Los_Angeles')
        2) FITS header TZ-like hints (TIMEZONE, TZ)
        3) System local timezone; finally, a best-effort local offset or UTC
        """
        # 1) App setting
        tz_name = self.settings.value("stacking/site_timezone", "", type=str) or ""
        try:
            from zoneinfo import ZoneInfo
        except Exception:
            ZoneInfo = None

        if tz_name and ZoneInfo:
            try:
                return ZoneInfo(tz_name)
            except Exception:
                pass

        # 2) From header
        if hdr:
            for key in ("TIMEZONE", "TZ"):
                if key in hdr and ZoneInfo:
                    try:
                        return ZoneInfo(str(hdr[key]).strip())
                    except Exception:
                        pass

        # 3) System local timezone
        if ZoneInfo:
            try:

                return ZoneInfo(str(tzlocal.get_localzone()))
            except Exception:
                pass

        # 3b) Fallback to the current local offset (aware tzinfo) or UTC
        try:
            now = dt_datetime.now().astimezone()
            return now.tzinfo or dt_timezone.utc
        except Exception:
            return dt_timezone.utc


    def _parse_date_obs(self, s: str):
        """
        Parse DATE-OBS robustly → aware UTC datetime if possible.
        Accepts ISO8601 with or without 'Z', fractional seconds, or date-only.
        """
        if not s:
            return None
        s = str(s).strip()

        # Prefer dateutil if present
        try:
            from dateutil import parser as date_parser
        except Exception:
            date_parser = None

        if date_parser:
            try:
                dt = date_parser.isoparse(s)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=dt_timezone.utc)
                return dt.astimezone(dt_timezone.utc)
            except Exception:
                pass

        # Manual fallbacks
        for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ",
                    "%Y-%m-%dT%H:%M:%SZ",
                    "%Y-%m-%dT%H:%M:%S",
                    "%Y-%m-%d"):
            try:
                dt = dt_datetime.strptime(s, fmt)
                if fmt.endswith("Z") or "T" in fmt:
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=dt_timezone.utc)
                else:
                    # date-only; assume midday UTC to avoid accidental date flips
                    dt = dt.replace(tzinfo=dt_timezone.utc, hour=12)
                return dt.astimezone(dt_timezone.utc)
            except Exception:
                continue
        return None


    def _session_from_dateobs_local_night(self, hdr, cutoff_hour: int = None) -> str | None:
        """
        Return 'YYYY-MM-DD' session key by snapping DATE-OBS to local 'night'.

        Rule: if local_time < cutoff_hour → use previous calendar date.
        Default cutoff = self.settings('stacking/night_cutoff_hour', 12)  # noon
        """
        date_obs = hdr.get("DATE-OBS") if hdr else None
        dt_utc = self._parse_date_obs(date_obs) if date_obs else None
        if dt_utc is None:
            return None

        if cutoff_hour is None:
            cutoff_hour = int(self.settings.value("stacking/night_cutoff_hour", 12, type=int))

        tz = self._get_site_timezone(hdr)
        dt_local = dt_utc.astimezone(tz)

        # ✅ Use datetime.time alias, not the 'time' module
        cutoff = dt_time(hour=max(0, min(23, int(cutoff_hour))), minute=0)

        local_date = dt_local.date()
        if dt_local.time() < cutoff:
            local_date = (dt_local - dt_timedelta(days=1)).date()

        return local_date.isoformat()


    def _auto_tag_sessions_for_paths(self, paths: list[str]) -> dict[str, str]:
        session_map = {}
        for p in paths:
            try:
                hdr, _ = get_valid_header(p)
            except Exception:
                hdr = None
            session_map[p] = self._auto_session_from_path(p, hdr)           # ✅ self.
        return session_map


    # --- Apply session tags to your internal dicts and tree -----------------------
    def _apply_session_tags_to_tree(self, *, tree: QTreeWidget, target_dict: dict, session_by_path: dict[str,str]):
        """
        After ingest, set self.session_tags[path] and update the UI 'Metadata' column
        to include 'Session: <tag>'. Also move items across the (group_key, session) buckets.
        """
        # Build reverse index: basename -> full path we just tagged
        by_base = {os.path.basename(p): p for p in session_by_path.keys()}

        def _update_leaf(leaf: QTreeWidgetItem):
            base = leaf.text(0)
            full = by_base.get(base)
            if not full:
                return
            sess = session_by_path.get(full, "Default")
            self.session_tags[full] = sess

            # Update the metadata column
            old_meta = leaf.text(1) or ""
            if "Session:" in old_meta:
                new_meta = re.sub(r"Session:\s*[^|]*", f"Session: {sess}", old_meta)
            else:
                new_meta = (old_meta + (" | " if old_meta else "") + f"Session: {sess}")
            leaf.setText(1, new_meta)

            # Move file into (group_key, session) bucket in target_dict
            # group_key currently your "Filter & Exposure" string for that leaf's top-level parent
            # Ascend to find the top-level "Filter & Exposure" node:
            parent = leaf.parent()
            while parent and parent.parent():
                parent = parent.parent()
            group_key = parent.text(0) if parent else "Unknown"

            # locate and move inside target_dict
            for key in list(target_dict.keys()):
                files = target_dict.get(key, [])
                if full in files:
                    old_session = key[1] if (isinstance(key, tuple) and len(key) == 2) else "Default"
                    if old_session != sess:
                        # remove from old bucket
                        files.remove(full)
                        if not files:
                            target_dict.pop(key, None)
                        new_key = (key[0] if isinstance(key, tuple) else group_key, sess)
                        target_dict.setdefault(new_key, []).append(full)
                    break

        # Walk all leaves
        root_count = tree.topLevelItemCount()
        for i in range(root_count):
            g = tree.topLevelItem(i)
            for j in range(g.childCount()):
                e = g.child(j)
                for k in range(e.childCount()):
                    leaf = e.child(k)
                    if leaf.childCount() == 0:
                        _update_leaf(leaf)

    def _get_file_dt_cache(self):
        # lazy init
        cache = getattr(self, "_file_dt_cache", None)
        if cache is None:
            cache = {}
            self._file_dt_cache = cache
        return cache

    def _file_local_night_date(self, path: str, hdr=None):
        """
        Return a date() representing the local 'night' for this file,
        using DATE-OBS and the app/site timezone & cutoff logic you already have.
        """
        cache = self._get_file_dt_cache()
        if path in cache:
            return cache[path]

        try:
            if hdr is None:
                with fits.open(path, memmap=True) as hdul:
                    hdr = hdul[0].header
        except Exception:
            hdr = None

        sess = self._session_from_dateobs_local_night(hdr) if hdr else None
        # sess is 'YYYY-MM-DD' or None
        import datetime as _dt
        d = _dt.date.fromisoformat(sess) if sess else None
        cache[path] = d
        return d

    def _closest_flat_for(self, *, filter_name: str, image_size: str, light_path: str):
        """
        Choose a flat master path by:
        1) exact key 'Filter (WxH) [Session]' matching light's session (if present)
        2) else among 'Filter (WxH) ...', pick by nearest local-night date to light.
        Returns path or None.
        """
        # normalize filter token the same way you build master keys
        ftoken = self._sanitize_name(filter_name)
        # light's date
        light_d = self._file_local_night_date(light_path)

        # Collect candidate flats (same filter+size)
        candidates = []
        for key, path in self.master_files.items():
            # flats often keyed like "Filter (WxH) [Session]" or "Filter (WxH)"
            if (ftoken in key) and (f"({image_size})" in key):
                candidates.append((key, path))

        if not candidates:
            return None

        # 1) exact session match if we can infer the light's session name
        light_hdr = None
        try:
            with fits.open(light_path, memmap=True) as hdul:
                light_hdr = hdul[0].header
        except Exception:
            pass
        light_session = self._session_from_dateobs_local_night(light_hdr) if light_hdr else None

        if light_session:
            exact_key = f"{ftoken} ({image_size}) [{light_session}]"
            for key, path in candidates:
                if key == exact_key:
                    return path

        # 2) otherwise pick by nearest date (local night)
        if light_d is None:
            # No DATE-OBS for light → fall back to first candidate
            return candidates[0][1]

        best = (None, 10**9)  # (path, |Δdays|)
        for _key, fpath in candidates:
            fd = self._file_local_night_date(fpath)
            if fd is None:
                continue
            delta = abs((fd - light_d).days)
            if delta < best[1]:
                best = (fpath, delta)

        return best[0] if best[0] else candidates[0][1]


    def _ingest_paths_with_progress(self, paths, tree, expected_type, title, manual_session_name=None):
        """
        Show a small standalone progress dialog while ingesting headers,
        with cancel support. Keeps UI responsive via processEvents().
        """
        total = len(paths)
        dlg = QProgressDialog(title, "Cancel", 0, total, self)
        dlg.setWindowTitle("Please wait")
        dlg.setWindowModality(Qt.WindowModality.ApplicationModal)
        dlg.setMinimumDuration(0)
        dlg.setAutoReset(True)
        dlg.setAutoClose(True)
        dlg.setValue(0)

        added = 0
        for i, path in enumerate(paths, start=1):
            if dlg.wasCanceled():
                break

            try:
                base = os.path.basename(path)
                dlg.setLabelText(f"{base}  ({i}/{total})")
                # Process events so the dialog repaints & remains responsive
                QCoreApplication.processEvents()
                self.process_fits_header(path, tree, expected_type, manual_session_name=manual_session_name)
                added += 1
            except Exception as e:
                # Optional: log or show a brief error — keep going
                # print(f"Failed to add {path}: {e}")
                pass

            dlg.setValue(i)
            QCoreApplication.processEvents()

        # Make sure it closes
        dlg.setValue(total)
        QCoreApplication.processEvents()

        try:
            self._report_group_summary(expected_type)
        except Exception:
            pass

        try:
            if (expected_type or "").upper() == "LIGHT":
                self._refresh_light_tree_summaries()
        except Exception:
            pass

        # Optional: brief status line (non-intrusive)
        try:
            if expected_type.upper() == "LIGHT":
                self.statusBar().showMessage(f"Added {added}/{total} Light frames", 3000)
        except Exception:
            pass

    def _fmt_hms(self, seconds: float) -> str:
        s = int(round(max(0.0, float(seconds))))
        h, r = divmod(s, 3600)
        m, s = divmod(r, 60)
        if h: return f"{h}h {m}m {s}s"
        if m: return f"{m}m {s}s"
        return f"{s}s"

    def _exposure_from_label(self, label: str) -> float | None:
        # Parses "20s (1080x1920)" → 20.0

        m = re.search(r"([0-9]*\.?[0-9]+)\s*s\b", (label or ""))
        if not m: return None
        try:
            return float(m.group(1))
        except Exception:
            return None


    def _busy_progress(self, text):
        """
        Returns a modal, indeterminate QProgressDialog you can open during
        short post-steps (e.g., assigning masters). Caller must .close().
        """
        dlg = QProgressDialog(text, None, 0, 0, self)
        dlg.setWindowTitle("Please wait")
        dlg.setWindowModality(Qt.WindowModality.ApplicationModal)
        dlg.setMinimumDuration(0)
        dlg.setCancelButton(None)
        dlg.show()
        QCoreApplication.processEvents()
        return dlg


    def _sanitize_name(self, name: str) -> str:
        """
        Replace any character that isn’t a letter, digit, space, dash or underscore
        with an underscore so it’s safe to use in filenames, dict-keys, tree labels, etc.
        """
        return re.sub(r"[^\w\s\-]", "_", name)
    
    def process_fits_header(self, path, tree, expected_type, manual_session_name=None):
        try:
            # --- Read header only (fast) ---
            header, _ = get_valid_header(path)   # FIX: use 'path', not 'file_path'

            # --- Basic image size ---
            try:
                width = int(header.get("NAXIS1"))
                height = int(header.get("NAXIS2"))
                image_size = f"{width}x{height}"
            except Exception as e:
                self.update_status(f"Warning: Could not read dimensions for {os.path.basename(path)}: {e}")
                width = height = None
                image_size = "Unknown"

            # --- Image type & exposure ---
            imagetyp = str(header.get("IMAGETYP", "UNKNOWN")).lower()

            exp_val = header.get("EXPOSURE")
            if exp_val is None:
                exp_val = header.get("EXPTIME")
            if exp_val is None:
                exposure_text = "Unknown"
            else:
                try:
                    # canonical like "300s" or "0.5s"
                    fexp = float(exp_val)
                    # use :g for tidy formatting (no trailing .0 unless needed)
                    exposure_text = f"{fexp:g}s"
                except Exception:
                    exposure_text = str(exp_val)

            # --- Mismatch prompt (respects "Yes to all / No to all") ---
            # --- Mismatch prompt (redirect/keep/skip with 'apply to all') ---
            if expected_type.upper() == "DARK":
                forbidden = ["light", "flat"]
            elif expected_type.upper() == "FLAT":
                forbidden = ["dark", "light"]
            elif expected_type.upper() == "LIGHT":
                forbidden = ["dark", "flat"]
            else:
                forbidden = []

            actual_type = self._guess_type_from_imagetyp(header.get("IMAGETYP"))
            decision_key = (expected_type.upper(), actual_type or "UNKNOWN")

            # Respect prior "Yes to all/No to all" style cache first
            # (compat with your existing decision_attr if you want)
            decision_attr = f"auto_confirm_{expected_type.lower()}"
            if hasattr(self, decision_attr):
                decision = getattr(self, decision_attr)
                if decision is False:
                    return
                # if True, keep going as-is (legacy behavior)

            # New logic: if actual looks forbidden, propose options
            if (actual_type is not None) and (actual_type.lower() in forbidden):
                # has the user already decided during this ingest burst?
                cached = self._mismatch_policy.get(decision_key)
                if cached == "skip":
                    return
                elif cached == "redirect":
                    # Re-route to the proper tab
                    dst_tree = self._tree_for_type(actual_type)
                    if dst_tree is not None:
                        # Recursively ingest into the correct type & tree
                        self.process_fits_header(path, dst_tree, actual_type, manual_session_name=manual_session_name)
                        return
                    # if no tree (shouldn't happen), fall through to prompt

                elif cached == "keep":
                    pass  # keep going in current tab

                else:
                    # Prompt the user
                    msg = QMessageBox(self)
                    msg.setWindowTitle("Mismatched Image Type")
                    pretty_actual = actual_type or "Unknown"
                    msg.setText(
                        f"Found '{os.path.basename(path)}' with IMAGETYP = {header.get('IMAGETYP')}\n"
                        f"which looks like **{pretty_actual}**, not {expected_type}.\n\n"
                        f"What would you like to do?"
                    )
                    btn_redirect = msg.addButton(f"Send to {pretty_actual.title()} tab", QMessageBox.ButtonRole.YesRole)
                    btn_keep     = msg.addButton(f"Add to {expected_type.title()} tab", QMessageBox.ButtonRole.YesRole)
                    btn_cancel   = msg.addButton("Skip file", QMessageBox.ButtonRole.RejectRole)

                    # “Apply to all” for the rest of this add session
                    from PyQt6.QtWidgets import QCheckBox
                    apply_all_cb = QCheckBox("Apply this choice to all remaining mismatches of this kind")
                    msg.setCheckBox(apply_all_cb)

                    msg.exec()
                    clicked = msg.clickedButton()
                    apply_all = bool(apply_all_cb.isChecked())

                    if clicked is btn_cancel:
                        choice = "skip"
                    elif clicked is btn_redirect:
                        choice = "redirect"
                    else:
                        choice = "keep"

                    if apply_all:
                        self._mismatch_policy[decision_key] = choice

                    if choice == "skip":
                        return
                    if choice == "redirect":
                        dst_tree = self._tree_for_type(actual_type)
                        if dst_tree is not None:
                            self.process_fits_header(path, dst_tree, actual_type, manual_session_name=manual_session_name)
                            return

            # --- Resolve session tag (auto vs manual vs legacy current_session_tag) ---
            auto_session = self.settings.value("stacking/auto_session", True, type=bool)
            if manual_session_name:                           # only passed when auto is OFF and user typed one
                session_tag = manual_session_name.strip()
            elif auto_session:
                session_tag = self._auto_session_from_path(path, header) or "Default"
            else:
                session_tag = getattr(self, "current_session_tag", "Default")

            # --- Common helpers ---
            filter_name_raw = header.get("FILTER", "Unknown")
            filter_name     = self._sanitize_name(filter_name_raw)

            # === DARKs ===
            if expected_type.upper() == "DARK":
                key = f"{exposure_text} ({image_size})"
                self.dark_files.setdefault(key, []).append(path)

                # Tree: top-level = key; child = file
                items = tree.findItems(key, Qt.MatchFlag.MatchExactly, 0)
                exposure_item = items[0] if items else QTreeWidgetItem([key])
                if not items:
                    tree.addTopLevelItem(exposure_item)

                metadata = f"Size: {image_size} | Session: {session_tag}"
                leaf = QTreeWidgetItem([os.path.basename(path), metadata])
                leaf.setData(0, Qt.ItemDataRole.UserRole, path)  # store full path
                exposure_item.addChild(leaf)

            # === FLATs ===
            elif expected_type.upper() == "FLAT":
                # internal dict keying (group by filter+exp+size, partition by session)
                flat_key = f"{filter_name} - {exposure_text} ({image_size})"
                composite_key = (flat_key, session_tag)
                self.flat_files.setdefault(composite_key, []).append(path)
                self.session_tags[path] = session_tag

                # Tree: top-level = filter; second = "exp (WxH)"; leaf = file
                filter_items = tree.findItems(filter_name, Qt.MatchFlag.MatchExactly, 0)
                filter_item = filter_items[0] if filter_items else QTreeWidgetItem([filter_name])
                if not filter_items:
                    tree.addTopLevelItem(filter_item)

                want_label = f"{exposure_text} ({image_size})"
                exposure_item = None
                for i in range(filter_item.childCount()):
                    if filter_item.child(i).text(0) == want_label:
                        exposure_item = filter_item.child(i); break
                if exposure_item is None:
                    exposure_item = QTreeWidgetItem([want_label])
                    filter_item.addChild(exposure_item)

                metadata = f"Size: {image_size} | Session: {session_tag}"
                leaf = QTreeWidgetItem([os.path.basename(path), metadata])
                leaf.setData(0, Qt.ItemDataRole.UserRole, path)  # store full path
                exposure_item.addChild(leaf)

            # === LIGHTs ===
            elif expected_type.upper() == "LIGHT":
                light_key = f"{filter_name} - {exposure_text} ({image_size})"
                composite_key = (light_key, session_tag)
                self.light_files.setdefault(composite_key, []).append(path)
                self.session_tags[path] = session_tag

                # Tree: top-level = filter; second = "exp (WxH)"; leaf = file
                filter_items = tree.findItems(filter_name, Qt.MatchFlag.MatchExactly, 0)
                filter_item = filter_items[0] if filter_items else QTreeWidgetItem([filter_name])
                if not filter_items:
                    tree.addTopLevelItem(filter_item)

                want_label = f"{exposure_text} ({image_size})"
                exposure_item = None
                for i in range(filter_item.childCount()):
                    if filter_item.child(i).text(0) == want_label:
                        exposure_item = filter_item.child(i); break
                if exposure_item is None:
                    exposure_item = QTreeWidgetItem([want_label])
                    filter_item.addChild(exposure_item)

                metadata = f"Size: {image_size} | Session: {session_tag}"
                leaf = QTreeWidgetItem([os.path.basename(path), metadata])
                leaf.setData(0, Qt.ItemDataRole.UserRole, path)  # ✅ needed for date-aware flat fallback
                exposure_item.addChild(leaf)

            # --- Done ---
            self.update_status(f"✅ Added {os.path.basename(path)} as {expected_type}")
            QApplication.processEvents()

        except Exception as e:
            self.update_status(f"❌ ERROR: Could not read FITS header for {os.path.basename(path)} - {e}")
            QApplication.processEvents()



    def add_master_files(self, tree, file_type, files):
        """ 
        Adds multiple master calibration files to the correct treebox with metadata including image dimensions.
        This version only reads the FITS header to extract image dimensions, making it much faster.
        """
        for file_path in files:
            try:
                # Read only the FITS header (fast)
                header = fits.getheader(file_path)
                
                # Check for both EXPOSURE and EXPTIME
                exposure = header.get("EXPOSURE", header.get("EXPTIME", "Unknown"))
                filter_name = header.get("FILTER", "Unknown")
                filter_name     = self._sanitize_name(filter_name)
                # Extract image dimensions from header keywords NAXIS1 and NAXIS2
                width = header.get("NAXIS1")
                height = header.get("NAXIS2")
                if width is not None and height is not None:
                    image_size = f"{width}x{height}"
                else:
                    image_size = "Unknown"
                
                # Construct key based on file type
                if file_type.upper() == "DARK":
                    key = f"{exposure}s ({image_size})"
                    self.master_files[key] = file_path  # Store master dark
                    self.master_sizes[file_path] = image_size  # Store size
                elif file_type.upper() == "FLAT":
                    # Attempt to extract session name from filename
                    session_name = "Default"
                    filename = os.path.basename(file_path)
                    if filename.lower().startswith("masterflat_"):
                        parts = filename.split("_")
                        if len(parts) > 1:
                            session_name = parts[1]

                    key = f"{filter_name} ({image_size}) [{session_name}]"
                    self.master_files[key] = file_path
                    self.master_sizes[file_path] = image_size

                # Extract additional metadata from header.
                sensor_temp = header.get("CCD-TEMP", "N/A")
                date_obs = header.get("DATE-OBS", "Unknown")
                metadata = f"Size: {image_size}, Temp: {sensor_temp}°C, Date: {date_obs}"

                # Check if category item already exists in the tree.
                items = tree.findItems(key, Qt.MatchFlag.MatchExactly, 0)
                if not items:
                    item = QTreeWidgetItem([key])
                    tree.addTopLevelItem(item)
                else:
                    item = items[0]

                # Add the master file as a child node with metadata.
                item.addChild(QTreeWidgetItem([os.path.basename(file_path), metadata]))

                print(f"✅ DEBUG: Added Master {file_type} -> {file_path} under {key} with metadata: {metadata}")
                self.update_status(f"✅ Added Master {file_type} -> {file_path} under {key} with metadata: {metadata}")
                print(f"📂 DEBUG: Master Files Stored: {self.master_files}")
                self.update_status(f"📂 DEBUG: Master Files Stored: {self.master_files}")
                QApplication.processEvents()
                self.assign_best_master_files()

            except Exception as e:
                print(f"❌ ERROR: Failed to load master file {file_path} - {e}")
                self.update_status(f"❌ ERROR: Failed to load master file {file_path} - {e}")
                QApplication.processEvents()

    def create_master_dark(self):
        """Creates master darks with minimal RAM usage by loading frames in small tiles (GPU-accelerated if available),
        with adaptive reducers."""
        if not self.stacking_directory:
            self.select_stacking_directory()
            if not self.stacking_directory:
                QMessageBox.warning(self, "Error", "Output directory is not set.")
                return

        # Keep both paths available; we'll override algo selection per group.
        ui_algo = getattr(self, "calib_rejection_algorithm", "Windsorized Sigma Clipping")
        if ui_algo == "Weighted Windsorized Sigma Clipping":
            ui_algo = "Windsorized Sigma Clipping"

        exposure_tolerance = self.exposure_tolerance_spinbox.value()
        dark_files_by_group = {}

        # group darks by exposure + size
        for exposure_key, file_list in self.dark_files.items():
            exposure_time_str, image_size = exposure_key.split(" (")
            image_size = image_size.rstrip(")")
            exposure_time = float(exposure_time_str.replace("s", "")) if "Unknown" not in exposure_time_str else 0

            matched_group = None
            for (existing_exposure, existing_size) in dark_files_by_group.keys():
                if abs(existing_exposure - exposure_time) <= exposure_tolerance and existing_size == image_size:
                    matched_group = (existing_exposure, existing_size)
                    break
            if matched_group is None:
                matched_group = (exposure_time, image_size)
                dark_files_by_group[matched_group] = []
            dark_files_by_group[matched_group].extend(file_list)

        master_dir = os.path.join(self.stacking_directory, "Master_Calibration_Files")
        os.makedirs(master_dir, exist_ok=True)

        # Pre-count tiles
        chunk_height = self.chunk_height
        chunk_width  = self.chunk_width
        total_tiles = 0
        group_shapes = {}
        for (exposure_time, image_size), file_list in dark_files_by_group.items():
            if len(file_list) < 2:
                continue
            ref_data, _, _, _ = load_image(file_list[0])
            if ref_data is None:
                continue
            H, W = ref_data.shape[:2]
            C = 1 if (ref_data.ndim == 2) else 3
            group_shapes[(exposure_time, image_size)] = (H, W, C)
            total_tiles += _count_tiles(H, W, chunk_height, chunk_width)

        if total_tiles == 0:
            self.update_status("⚠️ No eligible dark groups found to stack.")
            return

        # ------- small helpers (local) -------------------------------------------
        def _select_reducer(kind: str, N: int):
            if kind == "dark":
                if N < 16:
                    return ("Kappa-Sigma Clipping", {"kappa": 3.0, "iterations": 1}, "kappa1")
                elif N <= 200:
                    return ("Simple Median (No Rejection)", {}, "median")
                else:
                    return ("Trimmed Mean", {"trim_fraction": 0.05}, "trimmed")
            else:
                raise ValueError("wrong kind")

        def _cpu_tile_median(ts4: np.ndarray) -> np.ndarray:
            return np.median(ts4, axis=0).astype(np.float32, copy=False)

        def _cpu_tile_trimmed_mean(ts4: np.ndarray, frac: float) -> np.ndarray:
            if frac <= 0.0:
                return ts4.mean(axis=0, dtype=np.float32)
            F = ts4.shape[0]
            k = int(max(1, round(frac * F)))
            if 2 * k >= F:
                return _cpu_tile_median(ts4)
            s = np.sort(ts4, axis=0)
            core = s[k:F - k]
            return core.mean(axis=0, dtype=np.float32)

        def _cpu_tile_kappa_sigma_1iter(ts4: np.ndarray, kappa: float = 3.0) -> np.ndarray:
            med = np.median(ts4, axis=0)
            std = ts4.std(axis=0, dtype=np.float32)
            lo = med - kappa * std
            hi = med + kappa * std
            mask = (ts4 >= lo) & (ts4 <= hi)
            num = (ts4 * mask).sum(axis=0, dtype=np.float32)
            cnt = mask.sum(axis=0).astype(np.float32)
            out = np.where(cnt > 0, num / np.maximum(cnt, 1.0), med)
            return out.astype(np.float32, copy=False)

        pd = _Progress(self, "Create Master Darks", total_tiles)
        try:
            for (exposure_time, image_size), file_list in dark_files_by_group.items():
                if len(file_list) < 2:
                    self.update_status(f"⚠️ Skipping {exposure_time}s ({image_size}) - Not enough frames to stack.")
                    QApplication.processEvents()
                    continue
                if pd.cancelled:
                    self.update_status("⛔ Master Dark creation cancelled.")
                    break

                self.update_status(f"🟢 Processing {len(file_list)} darks for {exposure_time}s ({image_size}) exposure…")
                QApplication.processEvents()

                # reference shape
                if (exposure_time, image_size) in group_shapes:
                    height, width, channels = group_shapes[(exposure_time, image_size)]
                else:
                    ref_data, _, _, _ = load_image(file_list[0])
                    if ref_data is None:
                        self.update_status(f"❌ Failed to load reference {os.path.basename(file_list[0])}")
                        continue
                    height, width = ref_data.shape[:2]
                    channels = 1 if (ref_data.ndim == 2) else 3

                # --- choose reducer adaptively ---
                N = len(file_list)
                algo_name, params, cpu_label = _select_reducer("dark", N)
                use_gpu = bool(self._hw_accel_enabled()) and _torch_ok() and _gpu_algo_supported(algo_name)
                self.update_status(f"⚙️ {'GPU' if use_gpu else 'CPU'} reducer for darks — {algo_name} ({'k=%.1f' % params.get('kappa', 0) if cpu_label=='kappa1' else 'trim=%.0f%%' % (params.get('trim_fraction', 0)*100) if cpu_label=='trimmed' else 'median'})")
                QApplication.processEvents()

                memmap_path = os.path.join(master_dir, f"temp_dark_{exposure_time}_{image_size}.dat")
                final_stacked = np.memmap(memmap_path, dtype=np.float32, mode='w+', shape=(height, width, channels))

                tiles = _tile_grid(height, width, chunk_height, chunk_width)
                total_tiles_group = len(tiles)

                # double-buffer
                buf0 = np.empty((N, min(chunk_height, height), min(chunk_width, width), channels),
                                dtype=np.float32, order="C")
                buf1 = np.empty_like(buf0)

                from concurrent.futures import ThreadPoolExecutor
                tp = ThreadPoolExecutor(max_workers=1)

                # prime first read
                (y0, y1, x0, x1) = tiles[0]
                fut = tp.submit(_read_tile_stack, file_list, y0, y1, x0, x1, channels, buf0)
                use0 = True

                for t_idx, (y0, y1, x0, x1) in enumerate(tiles, start=1):
                    if pd.cancelled:
                        break

                    th, tw = fut.result()
                    ts_np = (buf0 if use0 else buf1)[:N, :th, :tw, :channels]

                    # prefetch next
                    if t_idx < total_tiles_group:
                        ny0, ny1, nx0, nx1 = tiles[t_idx]
                        fut = tp.submit(_read_tile_stack, file_list, ny0, ny1, nx0, nx1, channels,
                                        (buf1 if use0 else buf0))

                    pd.set_label(f"{int(exposure_time)}s ({image_size}) — y:{y0}-{y1} x:{x0}-{x1}")

                    # ---- reduction (GPU or CPU) ----
                    if use_gpu:
                        tile_result, _ = _torch_reduce_tile(
                            ts_np,
                            np.ones((N,), dtype=np.float32),
                            algo_name=algo_name,
                            kappa=float(params.get("kappa", getattr(self, "kappa", 3.0))),
                            iterations=int(params.get("iterations", getattr(self, "iterations", 1))),
                            sigma_low=float(getattr(self, "sigma_low", 2.5)),
                            sigma_high=float(getattr(self, "sigma_high", 2.5)),
                            trim_fraction=float(params.get("trim_fraction", getattr(self, "trim_fraction", 0.05))),
                            esd_threshold=float(getattr(self, "esd_threshold", 3.0)),
                            biweight_constant=float(getattr(self, "biweight_constant", 6.0)),
                            modz_threshold=float(getattr(self, "modz_threshold", 3.5)),
                            comet_hclip_k=float(self.settings.value("stacking/comet_hclip_k", 1.30, type=float)),
                            comet_hclip_p=float(self.settings.value("stacking/comet_hclip_p", 25.0, type=float)),
                        )
                    else:
                        if cpu_label == "median":
                            tile_result = _cpu_tile_median(ts_np)
                        elif cpu_label == "trimmed":
                            tile_result = _cpu_tile_trimmed_mean(ts_np, float(params.get("trim_fraction", 0.05)))
                        else:  # 'kappa1'
                            tile_result = _cpu_tile_kappa_sigma_1iter(ts_np, float(params.get("kappa", 3.0)))

                    # commit
                    final_stacked[y0:y1, x0:x1, :] = tile_result
                    pd.step()
                    use0 = not use0

                tp.shutdown(wait=True)

                if pd.cancelled:
                    try: del final_stacked
                    except Exception: pass
                    try: os.remove(memmap_path)
                    except Exception: pass
                    break

                master_dark_data = np.asarray(final_stacked, dtype=np.float32)
                del final_stacked

                master_dark_stem = f"MasterDark_{int(exposure_time)}s_{image_size}"
                master_dark_path = self._build_out(master_dir, master_dark_stem, "fit")

                master_header = fits.Header()
                master_header["IMAGETYP"] = "DARK"
                master_header["EXPTIME"]  = (exposure_time, "User-specified or from grouping")
                master_header["NAXIS"]    = 3 if channels==3 else 2
                master_header["NAXIS1"]   = master_dark_data.shape[1]
                master_header["NAXIS2"]   = master_dark_data.shape[0]
                if channels==3: master_header["NAXIS3"] = 3

                save_image(master_dark_data, master_dark_path, "fit", "32-bit floating point", master_header, is_mono=(channels==1))
                self.add_master_dark_to_tree(f"{exposure_time}s ({image_size})", master_dark_path)
                self.update_status(f"✅ Master Dark saved: {master_dark_path}")
                self.assign_best_master_files()
                self.save_master_paths_to_settings()

            # wrap-up
            self.assign_best_master_dark()
            self.update_override_dark_combo()
            self.assign_best_master_files()
        finally:
            try: _free_torch_memory()
            except Exception: pass
            pd.close()

            
    def add_master_dark_to_tree(self, exposure_label: str, master_dark_path: str):
        """
        Adds the newly created Master Dark to the Master Dark TreeBox and updates the dropdown.

        Parameters
        ----------
        exposure_label : str
            A human-friendly label like "30s (4128x2806)". This is used as the
            tree's top-level item text and as the dictionary key.
        master_dark_path : str
            Full path to the master dark FITS.
        """
        # Build/confirm size string from the file (for master_sizes)
        try:
            with fits.open(master_dark_path, memmap=False) as hdul:
                data = hdul[0].data
                if data is None:
                    size = "Unknown"
                else:
                    if data.ndim == 2:
                        h, w = data.shape
                    elif data.ndim == 3:
                        h, w = data.shape[:2]
                    else:
                        h = w = 0
                    size = f"{w}x{h}" if (h and w) else "Unknown"
        except Exception:
            size = "Unknown"

        exposure_key = str(exposure_label).strip()  # e.g. "30s (4128x2806)"

        # ✅ Record paths/sizes
        if not hasattr(self, "master_files"):
            self.master_files = {}
        if not hasattr(self, "master_sizes"):
            self.master_sizes = {}

        self.master_files[exposure_key] = master_dark_path
        self.master_sizes[master_dark_path] = size

        # ✅ Update UI Tree
        existing_items = self.master_dark_tree.findItems(
            exposure_key, Qt.MatchFlag.MatchExactly, 0
        )
        if existing_items:
            exposure_item = existing_items[0]
        else:
            exposure_item = QTreeWidgetItem([exposure_key])
            self.master_dark_tree.addTopLevelItem(exposure_item)

        master_item = QTreeWidgetItem([os.path.basename(master_dark_path)])
        exposure_item.addChild(master_item)

        # ✅ Refresh UI bits that depend on master darks
        self.update_override_dark_combo()
        self.assign_best_master_dark()  # auto-select
        self.update_status(f"✅ Master Dark saved and added to UI: {master_dark_path}")
        print(f"📝 DEBUG: Stored Master Dark -> {exposure_key}: {master_dark_path}")




    def assign_best_master_dark(self):
        """ Assigns the closest matching master dark based on exposure & image size. """
        print("\n🔍 DEBUG: Assigning best master darks to flats...\n")

        if not self.master_files:
            print("⚠️ WARNING: No Master Darks available.")
            self.update_status("⚠️ WARNING: No Master Darks available.")
            return  # Exit early if there are no master darks

        print(f"📂 Loaded Master Darks ({len(self.master_files)} total):")
        for key, value in self.master_files.items():
            print(f"   📌 {key} -> {value}")

        # Iterate through all flat filters
        for i in range(self.flat_tree.topLevelItemCount()):
            filter_item = self.flat_tree.topLevelItem(i)

            for j in range(filter_item.childCount()):
                exposure_item = filter_item.child(j)
                exposure_text = exposure_item.text(0)  # Example: "0.0007s (8288x5644)"

                # Extract exposure time
                match = re.match(r"([\d.]+)s?", exposure_text)
                if not match:
                    print(f"⚠️ WARNING: Could not parse exposure time from {exposure_text}")
                    continue  # Skip if exposure is invalid

                exposure_time = float(match.group(1))  # Extracted number
                print(f"🟢 Checking Flat Group: {exposure_text} (Parsed: {exposure_time}s)")

                # Extract image size from metadata
                if exposure_item.childCount() > 0:
                    metadata_text = exposure_item.child(0).text(1)  # Metadata column
                    size_match = re.search(r"Size: (\d+x\d+)", metadata_text)
                    image_size = size_match.group(1) if size_match else "Unknown"
                else:
                    image_size = "Unknown"

                print(f"✅ Parsed Flat Size: {image_size}")

                # Find the best matching master dark
                best_match = None
                best_diff = float("inf")

                for master_dark_exposure, master_dark_path in self.master_files.items():
                    master_dark_exposure_match = re.match(r"([\d.]+)s?", master_dark_exposure)
                    if not master_dark_exposure_match:
                        continue  # Skip if master dark exposure is invalid

                    master_dark_exposure_time = float(master_dark_exposure_match.group(1))
                    master_dark_size = self.master_sizes.get(master_dark_path, "Unknown")
                    if master_dark_size == "Unknown":
                        with fits.open(master_dark_path) as hdul:
                            master_dark_size = f"{hdul[0].data.shape[1]}x{hdul[0].data.shape[0]}"
                            self.master_sizes[master_dark_path] = master_dark_size  # ✅ Store it

                    print(f"🔎 Comparing with Master Dark: {master_dark_exposure_time}s ({master_dark_size})")

                    # Match both image size and exposure time
                    if image_size == master_dark_size:
                        diff = abs(master_dark_exposure_time - exposure_time)
                        if diff < best_diff:
                            best_match = master_dark_path
                            best_diff = diff

                # Assign best match in column 3
                if best_match:
                    exposure_item.setText(2, os.path.basename(best_match))
                    print(f"🔵 Assigned Master Dark: {os.path.basename(best_match)}")
                else:
                    exposure_item.setText(2, "None")
                    print(f"⚠️ No matching Master Dark found for {exposure_text}")

        # 🔥 Force UI update to reflect changes
        self.flat_tree.viewport().update()

        print("\n✅ DEBUG: Finished assigning best matching Master Darks to Flats.\n")



    def update_override_dark_combo(self):
        """Populate the dropdown with available Master Darks (from self.master_files)."""
        if not hasattr(self, "override_dark_combo"):
            return

        self.override_dark_combo.blockSignals(True)
        try:
            self.override_dark_combo.clear()
            self.override_dark_combo.addItem("None (Use Auto-Select)", userData=None)
            self.override_dark_combo.addItem("None (Use no Dark to Calibrate)", userData="__NO_DARK__")

            seen = set()
            for key, path in (self.master_files or {}).items():
                fn = os.path.basename(path or "")
                if not fn:
                    continue
                # Keep only dark-like files (bias-as-dark is fine)
                if ("masterdark" in fn.lower()) or fn.lower().startswith(("masterbias", "master_bias")):
                    if path and os.path.exists(path) and fn not in seen:
                        self.override_dark_combo.addItem(fn, userData=path)
                        seen.add(fn)
            print(f"✅ DEBUG: override_dark_combo items = {self.override_dark_combo.count()}")
        finally:
            self.override_dark_combo.blockSignals(False)



    def override_selected_master_dark_for_flats(self, idx: int):
        """Apply combo choice to selected flat groups; stores path/token in row + dict."""
        items = self.flat_tree.selectedItems()
        if not items:
            return

        # read combo selection (userData carries the path or sentinel)
        ud = self.override_dark_combo.itemData(idx)   # None | "__NO_DARK__" | "/path/to/MasterDark_..."
        txt = self.override_dark_combo.currentText()

        disp = ("Auto" if ud is None else
                "No Calibration" if ud == "__NO_DARK__" else
                os.path.basename(str(ud)))

        for it in items:
            # ensure we’re on a group row (no parent)
            if it.parent():
                continue
            gk = it.data(0, Qt.ItemDataRole.UserRole)
            if not gk:
                continue
            it.setText(2, disp)
            it.setData(2, Qt.ItemDataRole.UserRole, ud)
            self.flat_dark_override[gk] = ud

        print(f"✅ Override Master Dark applied → {disp}")



    def create_master_flat(self):
        """Creates master flats using per-frame dark subtraction before stacking (GPU-accelerated if available),
        with adaptive reducers and fast per-frame normalization."""
        if not self.stacking_directory:
            QMessageBox.warning(self, "Error", "Please set the stacking directory first using the wrench button.")
            return

        # Keep both paths available; we'll override algo selection per group.
        ui_algo = getattr(self, "calib_rejection_algorithm", "Windsorized Sigma Clipping")
        if ui_algo == "Weighted Windsorized Sigma Clipping":
            ui_algo = "Windsorized Sigma Clipping"

        exposure_tolerance = self.flat_exposure_tolerance_spinbox.value()
        flat_files_by_group = {}  # (Exposure, Size, Filter, Session) -> list

        # --- group flats exactly as before ---
        for (filter_exposure, session), file_list in self.flat_files.items():
            try:
                filter_name, exposure_size = filter_exposure.split(" - ")
                exposure_time_str, image_size = exposure_size.split(" (")
                image_size = image_size.rstrip(")")
            except ValueError:
                self.update_status(f"⚠️ ERROR: Could not parse {filter_exposure}")
                continue

            match = re.match(r"([\d.]+)s?", exposure_time_str)
            exposure_time = float(match.group(1)) if match else -10.0

            matched_group = None
            for key in flat_files_by_group:
                existing_exposure, existing_size, existing_filter, existing_session = key
                if (
                    abs(existing_exposure - exposure_time) <= exposure_tolerance
                    and existing_size == image_size
                    and existing_filter == filter_name
                    and existing_session == session
                ):
                    matched_group = key
                    break

            if matched_group is None:
                matched_group = (exposure_time, image_size, filter_name, session)
                flat_files_by_group[matched_group] = []
            flat_files_by_group[matched_group].extend(file_list)

        # Output folder
        master_dir = os.path.join(self.stacking_directory, "Master_Calibration_Files")
        os.makedirs(master_dir, exist_ok=True)

        # Pre-count tiles
        total_tiles = 0
        group_shapes = {}
        for (exposure_time, image_size, filter_name, session), file_list in flat_files_by_group.items():
            if len(file_list) < 2:
                continue
            ref_data, _, _, _ = load_image(file_list[0])
            if ref_data is None:
                continue
            H, W = ref_data.shape[:2]
            C = 1 if ref_data.ndim == 2 else 3
            group_shapes[(exposure_time, image_size, filter_name, session)] = (H, W, C)
            total_tiles += _count_tiles(H, W, self.chunk_height, self.chunk_width)

        if total_tiles == 0:
            self.update_status("⚠️ No eligible flat groups found to stack.")
            return

        # ------- helpers (local to this function) --------------------------------
        def _select_reducer(kind: str, N: int):
            """
            kind: 'flat' or 'dark'; return (algo_name, params_dict, cpu_label)
            algo_name is a GPU-string if GPU is used; CPU gets cpu_label switch.
            """
            if kind == "flat":
                # <16: robust median; 17–200: trimmed mean 5%; >200: trimmed mean 2% (nearly mean)
                if N < 16:
                    return ("Simple Median (No Rejection)", {}, "median")
                elif N <= 200:
                    return ("Trimmed Mean", {"trim_fraction": 0.05}, "trimmed")
                else:
                    return ("Trimmed Mean", {"trim_fraction": 0.02}, "trimmed")
            else:  # darks
                # <16: Kappa-Sigma 1-iter; 17–200: simple median; >200: trimmed mean 5% to reduce noise
                if N < 16:
                    return ("Kappa-Sigma Clipping", {"kappa": 3.0, "iterations": 1}, "kappa1")
                elif N <= 200:
                    return ("Simple Median (No Rejection)", {}, "median")
                else:
                    return ("Trimmed Mean", {"trim_fraction": 0.05}, "trimmed")

        def _cpu_tile_median(ts4: np.ndarray) -> np.ndarray:
            # ts4: (F, th, tw, C)
            return np.median(ts4, axis=0).astype(np.float32, copy=False)

        def _cpu_tile_trimmed_mean(ts4: np.ndarray, frac: float) -> np.ndarray:
            if frac <= 0.0:
                return ts4.mean(axis=0, dtype=np.float32)
            F = ts4.shape[0]
            k = int(max(1, round(frac * F)))
            if 2 * k >= F:  # if too aggressive, fall back to median
                return _cpu_tile_median(ts4)
            # sort along frame axis and average middle slice
            s = np.sort(ts4, axis=0)
            core = s[k:F - k]
            return core.mean(axis=0, dtype=np.float32)

        def _cpu_tile_kappa_sigma_1iter(ts4: np.ndarray, kappa: float = 3.0) -> np.ndarray:
            med = np.median(ts4, axis=0)
            std = ts4.std(axis=0, dtype=np.float32)
            lo = med - kappa * std
            hi = med + kappa * std
            mask = (ts4 >= lo) & (ts4 <= hi)
            # avoid division by zero
            num = (ts4 * mask).sum(axis=0, dtype=np.float32)
            cnt = mask.sum(axis=0).astype(np.float32)
            out = np.where(cnt > 0, num / np.maximum(cnt, 1.0), med)
            return out.astype(np.float32, copy=False)

        def _estimate_flat_scales(file_list: list[str], H: int, W: int, C: int, dark_data: np.ndarray | None):
            """
            Read one central patch (min(512, H/W)) from each frame, subtract dark (if present),
            compute per-frame median, and normalize scales to overall median.
            """
            # central patch
            th = min(512, H); tw = min(512, W)
            y0 = (H - th) // 2; y1 = y0 + th
            x0 = (W - tw) // 2; x1 = x0 + tw

            N = len(file_list)
            meds = np.empty((N,), dtype=np.float64)

            # small parallel read
            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_workers=min(os.cpu_count() or 4, 8)) as exe:
                fut2i = {exe.submit(load_fits_tile, fp, y0, y1, x0, x1): i for i, fp in enumerate(file_list)}
                for fut in as_completed(fut2i):
                    i = fut2i[fut]
                    sub = fut.result()
                    if sub is None:
                        meds[i] = 1.0
                        continue
                    # to HWC f32
                    if sub.ndim == 2:
                        sub = sub[:, :, None]
                    elif sub.ndim == 3 and sub.shape[0] in (1, 3):
                        sub = sub.transpose(1, 2, 0)
                    sub = sub.astype(np.float32, copy=False)

                    if dark_data is not None:
                        dd = dark_data
                        if dd.ndim == 3 and dd.shape[0] in (1, 3):
                            dd = dd.transpose(1, 2, 0)
                        d_tile = dd[y0:y1, x0:x1].astype(np.float32, copy=False)
                        if d_tile.ndim == 2 and sub.shape[2] == 3:
                            d_tile = np.repeat(d_tile[..., None], 3, axis=2)
                        sub = sub - d_tile

                    meds[i] = np.median(sub, axis=(0, 1, 2))
            # normalize to global median
            gmed = np.median(meds) if np.all(np.isfinite(meds)) else 1.0
            gmed = 1.0 if gmed == 0.0 else gmed
            scales = meds / gmed
            # clamp to sane range
            scales = np.clip(scales, 1e-3, 1e3).astype(np.float32)
            return scales

        pd = _Progress(self, "Create Master Flats", total_tiles)
        try:
            for (exposure_time, image_size, filter_name, session), file_list in flat_files_by_group.items():
                if len(file_list) < 2:
                    self.update_status(f"⚠️ Skipping {exposure_time}s ({image_size}) [{filter_name}] [{session}] - Not enough frames to stack.")
                    continue
                if pd.cancelled:
                    self.update_status("⛔ Master Flat creation cancelled.")
                    break

                self.update_status(f"🟢 Processing {len(file_list)} flats for {exposure_time}s ({image_size}) [{filter_name}] in session '{session}'…")
                QApplication.processEvents()

                # Select matching master dark (optional)
                # --- resolve override / auto-select per group ---
                # rebuild the SAME key we stored on the tree
                expmin = math.floor(float(exposure_time))
                tol    = float(self.flat_exposure_tolerance_spinbox.value())
                group_key = f"{filter_name}|{image_size}|{expmin:.3f}|{(expmin+tol):.3f}"

                override_val = self.flat_dark_override.get(group_key, None)  # None | "__NO_DARK__" | path
                selected_master_dark = None

                if override_val == "__NO_DARK__":
                    self.update_status("ℹ️ This flat group: override = No Calibration.")
                elif isinstance(override_val, str):
                    if os.path.exists(override_val):
                        selected_master_dark = override_val
                        self.update_status(f"🌓 This flat group: using OVERRIDE dark → {os.path.basename(selected_master_dark)}")
                    else:
                        self.update_status("⚠️ Override dark missing on disk; falling back to Auto/None.")
                        override_val = None  # fall through

                if (override_val is None) and self.auto_select_dark_checkbox.isChecked():
                    # Auto-select closest exposure, same size
                    best_diff = float("inf")
                    for key, path in (getattr(self, "master_files", {}) or {}).items():
                        # keep only MasterDark files
                        if "MasterDark" not in os.path.basename(path):
                            continue
                        dark_size = self.master_sizes.get(path, "Unknown")
                        m = re.match(r"([\d.]+)s", key or "")
                        dark_exp = float(m.group(1)) if m else float("inf")
                        if dark_size == image_size:
                            diff = abs(dark_exp - float(exposure_time))
                            if diff < best_diff:
                                best_diff = diff
                                selected_master_dark = path

                    if selected_master_dark:
                        self.update_status(f"🌓 This flat group: using AUTO dark → {os.path.basename(selected_master_dark)}")
                    else:
                        self.update_status("ℹ️ This flat group: no matching Master Dark (size) — proceeding without subtraction.")
                elif (override_val is None) and (not self.auto_select_dark_checkbox.isChecked()):
                    # explicit: no auto and no override → no dark
                    self.update_status("ℹ️ This flat group: Auto-Select is OFF and no override set → No Calibration.")

                # Load the chosen dark if any
                if selected_master_dark:
                    dark_data, _, _, _ = load_image(selected_master_dark)
                else:
                    dark_data = None

                # reference shape
                if (exposure_time, image_size, filter_name, session) in group_shapes:
                    height, width, channels = group_shapes[(exposure_time, image_size, filter_name, session)]
                else:
                    ref_data, _, _, _ = load_image(file_list[0])
                    if ref_data is None:
                        self.update_status(f"❌ Failed to load reference {os.path.basename(file_list[0])}")
                        continue
                    height, width = ref_data.shape[:2]
                    channels = 1 if ref_data.ndim == 2 else 3

                # --- choose reducer based on N (adaptive) ---
                N = len(file_list)
                algo_name, params, cpu_label = _select_reducer("flat", N)
                use_gpu = bool(self._hw_accel_enabled()) and _torch_ok() and _gpu_algo_supported(algo_name)
                self.update_status(f"⚙️ Normalizing {N} flats by per-frame medians (central patch).")
                QApplication.processEvents()
                # --- precompute normalization scales (fast, one central patch) ---
                scales = _estimate_flat_scales(file_list, height, width, channels, dark_data)

                self.update_status(f"⚙️ {'GPU' if use_gpu else 'CPU'} reducer for flats — {algo_name} ({'k=%.1f' % params.get('kappa', 0) if cpu_label=='kappa1' else 'trim=%.0f%%' % (params.get('trim_fraction', 0)*100) if cpu_label=='trimmed' else 'median'})")
                QApplication.processEvents()

                memmap_path = os.path.join(master_dir, f"temp_flat_{session}_{exposure_time}_{image_size}_{filter_name}.dat")
                final_stacked = np.memmap(memmap_path, dtype=np.float32, mode="w+", shape=(height, width, channels))

                tiles = _tile_grid(height, width, self.chunk_height, self.chunk_width)
                total_tiles_group = len(tiles)

                # allocate max-chunk buffers (C-order, float32)
                buf0 = np.empty((N, min(self.chunk_height, height), min(self.chunk_width, width), channels),
                                dtype=np.float32, order="C")
                buf1 = np.empty_like(buf0)

                from concurrent.futures import ThreadPoolExecutor
                tp = ThreadPoolExecutor(max_workers=1)

                # prime first read
                (y0, y1, x0, x1) = tiles[0]
                fut = tp.submit(_read_tile_stack, file_list, y0, y1, x0, x1, channels, buf0)
                use0 = True

                for t_idx, (y0, y1, x0, x1) in enumerate(tiles, start=1):
                    if pd.cancelled:
                        break

                    th, tw = fut.result()
                    ts_np = (buf0 if use0 else buf1)[:N, :th, :tw, :channels]  # (F, th, tw, C)

                    # prefetch next
                    if t_idx < total_tiles_group:
                        ny0, ny1, nx0, nx1 = tiles[t_idx]
                        fut = tp.submit(_read_tile_stack, file_list, ny0, ny1, nx0, nx1, channels,
                                        (buf1 if use0 else buf0))

                    pd.set_label(f"[{filter_name}] {session} — y:{y0}-{y1} x:{x0}-{x1}")

                    # ---- per-tile dark subtraction (HWC), BEFORE normalization ----
                    if dark_data is not None:
                        dsub = dark_data
                        if dsub.ndim == 3 and dsub.shape[0] in (1, 3):  # CHW → HWC
                            dsub = dsub.transpose(1, 2, 0)
                        d_tile = dsub[y0:y1, x0:x1].astype(np.float32, copy=False)
                        if d_tile.ndim == 2 and channels == 3:
                            d_tile = np.repeat(d_tile[..., None], 3, axis=2)
                        _subtract_dark_stack_inplace_hwc(ts_np, d_tile, pedestal=0.0)

                    # ---- fast per-frame normalization (divide by precomputed scale) ----
                    ts_np /= scales.reshape(N, 1, 1, 1)

                    # ---- reduction (GPU or CPU) ----
                    if use_gpu:
                        # pass parameters through the GPU reducer
                        tile_result, _ = _torch_reduce_tile(
                            ts_np,
                            np.ones((N,), dtype=np.float32),
                            algo_name=algo_name,
                            kappa=float(params.get("kappa", getattr(self, "kappa", 3.0))),
                            iterations=int(params.get("iterations", getattr(self, "iterations", 1))),
                            sigma_low=float(getattr(self, "sigma_low", 2.5)),
                            sigma_high=float(getattr(self, "sigma_high", 2.5)),
                            trim_fraction=float(params.get("trim_fraction", getattr(self, "trim_fraction", 0.05))),
                            esd_threshold=float(getattr(self, "esd_threshold", 3.0)),
                            biweight_constant=float(getattr(self, "biweight_constant", 6.0)),
                            modz_threshold=float(getattr(self, "modz_threshold", 3.5)),
                            comet_hclip_k=float(self.settings.value("stacking/comet_hclip_k", 1.30, type=float)),
                            comet_hclip_p=float(self.settings.value("stacking/comet_hclip_p", 25.0, type=float)),
                        )
                    else:
                        if cpu_label == "median":
                            tile_result = _cpu_tile_median(ts_np)
                        elif cpu_label == "trimmed":
                            tile_result = _cpu_tile_trimmed_mean(ts_np, float(params.get("trim_fraction", 0.05)))
                        else:  # 'kappa1'
                            tile_result = _cpu_tile_kappa_sigma_1iter(ts_np, float(params.get("kappa", 3.0)))

                    # commit + progress
                    final_stacked[y0:y1, x0:x1, :] = tile_result
                    pd.step()
                    use0 = not use0

                tp.shutdown(wait=True)

                if pd.cancelled:
                    try: del final_stacked
                    except Exception: pass
                    try: os.remove(memmap_path)
                    except Exception: pass
                    break

                master_flat_data = np.asarray(final_stacked, dtype=np.float32)
                del final_stacked

                master_flat_stem = f"MasterFlat_{session}_{int(exposure_time)}s_{image_size}_{filter_name}"
                master_flat_path = self._build_out(master_dir, master_flat_stem, "fit")

                header = fits.Header()
                header["IMAGETYP"] = "FLAT"
                header["EXPTIME"]  = (exposure_time, "grouped exposure")
                header["FILTER"]   = filter_name
                header["NAXIS"]    = 3 if channels == 3 else 2
                header["NAXIS1"]   = width
                header["NAXIS2"]   = height
                if channels == 3: header["NAXIS3"] = 3

                save_image(master_flat_data, master_flat_path, "fit", "32-bit floating point", header, is_mono=(channels == 1))
                key = f"{filter_name} ({image_size}) [{session}]"
                self.master_files[key] = master_flat_path
                self.master_sizes[master_flat_path] = image_size
                self.add_master_flat_to_tree(filter_name, master_flat_path)
                self.update_status(f"✅ Master Flat saved: {master_flat_path}")
                self.save_master_paths_to_settings()

            self.assign_best_master_dark()
            self.assign_best_master_files()
        finally:
            try: _free_torch_memory()
            except Exception: pass
            pd.close()


    def add_master_flat_to_tree(self, filter_name, master_flat_path):
        """ Adds the newly created Master Flat to the Master Flat TreeBox and stores it. """

        key = f"{filter_name} ({self.master_sizes[master_flat_path]})"
        self.master_files[key] = master_flat_path  # ✅ Store the flat file for future use
        print(f"📝 DEBUG: Stored Master Flat -> {key}: {master_flat_path}")

        existing_items = self.master_flat_tree.findItems(filter_name, Qt.MatchFlag.MatchExactly, 0)

        if existing_items:
            filter_item = existing_items[0]
        else:
            filter_item = QTreeWidgetItem([filter_name])
            self.master_flat_tree.addTopLevelItem(filter_item)

        master_item = QTreeWidgetItem([os.path.basename(master_flat_path)])
        filter_item.addChild(master_item)

    def assign_best_master_files(self, fill_only: bool = True):
        """
        Assign best matching Master Dark and Flat to each Light leaf.
        - Honors manual overrides (never ignored).
        - If fill_only is True, do NOT overwrite non-empty cells.
        """
        print("\n🔍 DEBUG: Assigning best Master Darks & Flats to Lights...\n")

        if not getattr(self, "master_files", None):
            print("⚠️ WARNING: No Master Calibration Files available.")
            self.update_status("⚠️ WARNING: No Master Calibration Files available.")
            return

        # Ensure override dicts exist
        dark_over = getattr(self, "manual_dark_overrides", {}) or {}
        flat_over = getattr(self, "manual_flat_overrides", {}) or {}
        master_sizes = getattr(self, "master_sizes", {})
        self.master_sizes = master_sizes  # keep cache alive

        for i in range(self.light_tree.topLevelItemCount()):
            filter_item = self.light_tree.topLevelItem(i)
            filter_name_raw = filter_item.text(0)
            filter_name = self._sanitize_name(filter_name_raw)

            for j in range(filter_item.childCount()):
                exposure_item = filter_item.child(j)
                exposure_text = exposure_item.text(0)  # e.g. "300.0s (4144x2822)"

                # Parse exposure seconds (for dark matching)
                mexp = re.match(r"([\d.]+)s?", exposure_text or "")
                exposure_time = float(mexp.group(1)) if mexp else 0.0

                for k in range(exposure_item.childCount()):
                    leaf_item = exposure_item.child(k)
                    meta_text = leaf_item.text(1) or ""

                    # Parse size & session from metadata column
                    msize = re.search(r"Size:\s*(\d+x\d+)", meta_text)
                    image_size = msize.group(1) if msize else "Unknown"
                    msess = re.search(r"Session:\s*([^|]+)", meta_text)
                    session_name = (msess.group(1).strip() if msess else "Default")

                    # Current cells (so we can skip if fill_only)
                    curr_dark = (leaf_item.text(2) or "").strip()
                    curr_flat = (leaf_item.text(3) or "").strip()

                    # ---------- DARK RESOLUTION ----------
                    # 1) Manual overrides: prefer "Filter - exposure" then bare exposure
                    dark_key_full  = f"{filter_name_raw} - {exposure_text}"
                    dark_key_short = exposure_text
                    dark_override  = dark_over.get(dark_key_full) or dark_over.get(dark_key_short)

                    if dark_override:
                        dark_choice = os.path.basename(dark_override)
                    else:
                        # 2) If fill_only and cell already nonempty & not "None", keep it
                        if fill_only and curr_dark and curr_dark.lower() != "none":
                            dark_choice = curr_dark
                        else:
                            # 3) Auto-pick by size+closest exposure
                            best_dark_match = None
                            best_dark_diff = float("inf")
                            for master_key, master_path in self.master_files.items():
                                dmatch = re.match(r"^([\d.]+)s\b", master_key)  # darks start with "<exp>s"
                                if not dmatch:
                                    continue
                                master_dark_exposure_time = float(dmatch.group(1))

                                # Ensure size known/cached
                                md_size = master_sizes.get(master_path)
                                if not md_size:
                                    try:
                                        with fits.open(master_path) as hdul:
                                            md_size = f"{hdul[0].data.shape[1]}x{hdul[0].data.shape[0]}"
                                    except Exception:
                                        md_size = "Unknown"
                                    master_sizes[master_path] = md_size

                                if md_size == image_size:
                                    diff = abs(master_dark_exposure_time - exposure_time)
                                    if diff < best_dark_diff:
                                        best_dark_diff = diff
                                        best_dark_match = master_path

                            dark_choice = os.path.basename(best_dark_match) if best_dark_match else ("None" if not curr_dark else curr_dark)

                    # ---------- FLAT RESOLUTION ----------
                    flat_key_full  = f"{filter_name_raw} - {exposure_text}"
                    flat_key_short = exposure_text
                    flat_override  = flat_over.get(flat_key_full) or flat_over.get(flat_key_short)

                    if flat_override:
                        flat_choice = os.path.basename(flat_override)
                    else:
                        if fill_only and curr_flat and curr_flat.lower() != "none":
                            flat_choice = curr_flat
                        else:
                            # Get the full path of the light leaf (we stored it during ingest)
                            light_path = leaf_item.data(0, Qt.ItemDataRole.UserRole)
                            # Prefer exact session match; otherwise nearest-night fallback
                            best_flat_path = None

                            # Fast exact-key path
                            exact_key = f"{filter_name} ({image_size}) [{session_name}]"
                            if exact_key in self.master_files:
                                best_flat_path = self.master_files[exact_key]
                            else:
                                # Date-aware fallback across same filter+size
                                best_flat_path = self._closest_flat_for(
                                    filter_name=filter_name,
                                    image_size=image_size,
                                    light_path=light_path
                                )

                            flat_choice = os.path.basename(best_flat_path) if best_flat_path else ("None" if not curr_flat else curr_flat)

                    # ---------- WRITE CELLS ----------
                    leaf_item.setText(2, dark_choice)
                    leaf_item.setText(3, flat_choice)

                    print(f"📌 Assigned to {leaf_item.text(0)} -> Dark: {leaf_item.text(2)}, Flat: {leaf_item.text(3)}")

        self.light_tree.viewport().update()
        print("\n✅ DEBUG: Finished assigning Master Files per leaf.\n")


    def update_light_corrections(self):
        """ Updates the light frame corrections when checkboxes change. """
        corrections = []
        if self.cosmetic_checkbox.isChecked():
            corrections.append("Cosmetic: True")
        else:
            corrections.append("Cosmetic: False")

        if self.pedestal_checkbox.isChecked():
            corrections.append("Pedestal: True")
        else:
            corrections.append("Pedestal: False")

        if self.bias_checkbox.isChecked():
            # Show file dialog to select a Master Bias
            bias_file, _ = QFileDialog.getOpenFileName(self, "Select Master Bias Frame", "", "FITS Files (*.fits *.fit)")
            if bias_file:
                self.master_files["Bias"] = bias_file  # ✅ Store bias path
                corrections.append(f"Bias: {os.path.basename(bias_file)}")
            else:
                self.bias_checkbox.setChecked(False)  # If no file selected, uncheck
                return

        # Update all rows
        for i in range(self.light_tree.topLevelItemCount()):
            filter_item = self.light_tree.topLevelItem(i)
            for j in range(filter_item.childCount()):
                exposure_item = filter_item.child(j)
                exposure_item.setText(4, ", ".join(corrections))

    def light_tree_context_menu(self, pos):
        item = self.light_tree.itemAt(pos)
        if not item:
            return

        menu = QMenu(self.light_tree)
        override_dark_action = menu.addAction("Override Dark Frame")
        override_flat_action = menu.addAction("Override Flat Frame")
        set_session_action = menu.addAction("Set Session Tag...")

        action = menu.exec(self.light_tree.viewport().mapToGlobal(pos))

        if action == override_dark_action:
            self.override_selected_master_dark()
        elif action == override_flat_action:
            self.override_selected_master_flat()
        elif action == set_session_action:
            self.prompt_set_session(item, "LIGHT")


    def set_session_tag_for_group(self, item):
        """
        Prompt the user to assign a session tag to all frames in this group.
        """
        session_name, ok = QInputDialog.getText(self, "Set Session Tag", "Enter session label (e.g., Night1, RedFilterSet2):")
        if not ok or not session_name.strip():
            return

        session_name = session_name.strip()
        filter_name = item.text(0)

        for i in range(item.childCount()):
            exposure_item = item.child(i)
            exposure_label = exposure_item.text(0)

            # Update metadata text
            if exposure_item.childCount() > 0:
                metadata_item = exposure_item.child(0)
                metadata_text = metadata_item.text(1)
                metadata_text = re.sub(r"Session: [^|]+", f"Session: {session_name}", metadata_text)
                if "Session:" not in metadata_text:
                    metadata_text += f" | Session: {session_name}"
                metadata_item.setText(1, metadata_text)

            # Update internal session tag mapping
            composite_key = (f"{filter_name} - {exposure_label}", session_name)
            original_key = f"{filter_name} - {exposure_label}"

            if original_key in self.light_files:
                self.light_files[composite_key] = self.light_files.pop(original_key)

                for path in self.light_files[composite_key]:
                    self.session_tags[path] = session_name

        self.update_status(f"🟢 Assigned session '{session_name}' to group '{filter_name}'")


    def override_selected_master_dark(self):
        """ Override Dark for selected Light exposure group or individual files. """
        selected_items = self.light_tree.selectedItems()
        if not selected_items:
            print("⚠️ No light item selected for dark frame override.")
            return

        file_path, _ = QFileDialog.getOpenFileName(self, "Select Master Dark", "", "FITS Files (*.fits *.fit)")
        if not file_path:
            return

        for item in selected_items:
            # If the user clicked a group (exposure row), push override to all leaves:
            if item.parent() and item.childCount() > 0:
                # exposure row under a filter
                filter_name = item.parent().text(0)
                exposure_text = item.text(0)
                # store override under BOTH keys
                self.manual_dark_overrides[f"{filter_name} - {exposure_text}"] = file_path
                self.manual_dark_overrides[exposure_text] = file_path

                for i in range(item.childCount()):
                    leaf = item.child(i)
                    leaf.setText(2, os.path.basename(file_path))
            # If the user clicked a leaf, just set that leaf and still store under both keys
            elif item.parent() and item.parent().parent():
                exposure_item = item.parent()
                filter_name = exposure_item.parent().text(0)
                exposure_text = exposure_item.text(0)
                self.manual_dark_overrides[f"{filter_name} - {exposure_text}"] = file_path
                self.manual_dark_overrides[exposure_text] = file_path
                item.setText(2, os.path.basename(file_path))

        print("✅ DEBUG: Light Dark override applied.")

    def _auto_pick_master_dark(self, image_size: str, exposure_time: float):
        best_path, best_diff = None, float("inf")
        for key, path in self.master_files.items():
            m = re.match(r"^\s*([\d.]+)s\b", str(key))
            if not m:
                continue
            try:
                dark_exp = float(m.group(1))
            except Exception:
                continue
            size = self.master_sizes.get(path)
            if size is None:
                try:
                    with fits.open(path) as hdul:
                        size = f"{hdul[0].data.shape[1]}x{hdul[0].data.shape[0]}"
                    self.master_sizes[path] = size
                except Exception:
                    continue
            if size == image_size:
                diff = abs(dark_exp - exposure_time)
                if diff < best_diff:
                    best_diff, best_path = diff, path
        return best_path

    def _auto_pick_master_flat(self, filter_name: str, image_size: str, session_name: str):
        # Prefer session-specific, then session-agnostic
        key_pref = f"{filter_name} ({image_size}) [{session_name}]"
        if key_pref in self.master_files:
            return self.master_files[key_pref]
        fallback_key = f"{filter_name} ({image_size})"
        return self.master_files.get(fallback_key)

    def _lookup_flat_override(self, filter_name: str, exposure_text: str) -> str | None:
        """Prefer 'Filter - exposure' override, else bare exposure."""
        if not hasattr(self, "manual_flat_overrides"):
            self.manual_flat_overrides = {}
        key_full  = f"{filter_name} - {exposure_text}"
        key_short = exposure_text
        return (self.manual_flat_overrides.get(key_full)
                or self.manual_flat_overrides.get(key_short))

    def _lookup_dark_override(self, filter_name: str, exposure_text: str) -> str | None:
        if not hasattr(self, "manual_dark_overrides"):
            self.manual_dark_overrides = {}
        key_full  = f"{filter_name} - {exposure_text}"
        key_short = exposure_text
        return (self.manual_dark_overrides.get(key_full)
                or self.manual_dark_overrides.get(key_short))



    def override_selected_master_flat(self):
        """
        Override Master Flat for selected Light items.
        - Accepts selection at filter row, exposure row, or leaf row.
        - Updates the leaf's column 3 with the chosen master flat's basename.
        - Stores overrides under BOTH keys:
            * "Filter - <exposure_text>"
            * "<exposure_text>"
        so calibrate_lights() can resolve it reliably.
        """
        selected_items = self.light_tree.selectedItems()
        if not selected_items:
            print("⚠️ No Light items selected for flat override.")
            return

        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Master Flat", "", "FITS Files (*.fits *.fit)"
        )
        if not file_path:
            return  # user cancelled

        base = os.path.basename(file_path)
        overrides_set = []

        def _apply_to_exposure_row(exp_item, filter_name):
            exposure_text = exp_item.text(0)  # e.g. "300.0s (4144x2822)"
            # Store override under both key shapes
            self.manual_flat_overrides[f"{filter_name} - {exposure_text}"] = file_path
            self.manual_flat_overrides[exposure_text] = file_path
            overrides_set.append(f"{filter_name} - {exposure_text}")
            overrides_set.append(exposure_text)
            # Push to all leaves
            for i in range(exp_item.childCount()):
                leaf = exp_item.child(i)
                leaf.setText(3, base)

        for item in selected_items:
            # Case A: leaf row (filename row)
            if item.parent() and item.parent().parent():
                exposure_item = item.parent()
                filter_item = exposure_item.parent()
                filter_name = filter_item.text(0)
                _apply_to_exposure_row(exposure_item, filter_name)
                item.setText(3, base)

            # Case B: exposure row (under a filter, has children)
            elif item.parent() and item.childCount() > 0:
                filter_name = item.parent().text(0)
                _apply_to_exposure_row(item, filter_name)

            # Case C: top-level filter row (apply to all its exposure groups)
            elif item.parent() is None and item.childCount() > 0:
                filter_name = item.text(0)
                for j in range(item.childCount()):
                    exposure_item = item.child(j)
                    _apply_to_exposure_row(exposure_item, filter_name)

        if overrides_set:
            print(f"✅ DEBUG: Overrode Master Flat with {base} for keys: {sorted(set(overrides_set))}")
        else:
            print("ℹ️ No applicable rows found to apply flat override.")



    def toggle_group_correction(self, group_item, which):
        """
        group_item: a top-level item in the light_tree
        which: either "cosmetic" or "pedestal"
        """
        old_text = group_item.text(4)  # e.g. "Cosmetic: True, Pedestal: False"
        # If there's nothing, default them to False
        if not old_text:
            old_text = "Cosmetic: False, Pedestal: False"

        # Parse
        # old_text might be "Cosmetic: True, Pedestal: False"
        # split by comma
        # part[0] => "Cosmetic: True"
        # part[1] => " Pedestal: False"
        parts = old_text.split(",")
        cosmetic_str = "False"
        pedestal_str = "False"
        if len(parts) == 2:
            # parse cosmetic
            cos_part = parts[0].split(":")[-1].strip()  # "True" or "False"
            cosmetic_str = cos_part
            # parse pedestal
            ped_part = parts[1].split(":")[-1].strip()
            pedestal_str = ped_part

        # Convert to bool
        cosmetic_bool = (cosmetic_str.lower() == "true")
        pedestal_bool = (pedestal_str.lower() == "true")

        # Toggle whichever was requested
        if which == "cosmetic":
            cosmetic_bool = not cosmetic_bool
        elif which == "pedestal":
            pedestal_bool = not pedestal_bool

        # Rebuild the new text
        new_text = f"Cosmetic: {str(cosmetic_bool)}, Pedestal: {str(pedestal_bool)}"
        group_item.setText(4, new_text)

    def _resolve_corrections_for_exposure(self, exposure_item):
        """
        Decide whether to apply cosmetic correction & pedestal for a given exposure row.
        Priority:
        1) Live UI checkboxes (if present)
        2) Corrections column text on the row
        3) QSettings defaults
        """
        # 1) Live UI
        try:
            cosmetic_ui  = bool(self.cosmetic_checkbox.isChecked())
        except Exception:
            cosmetic_ui = None
        try:
            pedestal_ui  = bool(self.pedestal_checkbox.isChecked())
        except Exception:
            pedestal_ui = None

        # 2) Row text (Corrections column)
        apply_cosmetic_col = apply_pedestal_col = None
        try:
            correction_text = exposure_item.text(4) or ""
            if correction_text:
                parts = [p.strip().lower() for p in correction_text.split(",")]
                # Expect "Cosmetic: True, Pedestal: False"
                for p in parts:
                    if p.startswith("cosmetic:"):
                        apply_cosmetic_col = (p.split(":")[-1].strip() == "true")
                    elif p.startswith("pedestal:"):
                        apply_pedestal_col = (p.split(":")[-1].strip() == "true")
        except Exception:
            pass

        # 3) Settings default
        cosmetic_cfg = self.settings.value("stacking/cosmetic_enabled", True, type=bool)
        pedestal_cfg = self.settings.value("stacking/pedestal_enabled", False, type=bool)

        apply_cosmetic = (
            cosmetic_ui if cosmetic_ui is not None
            else (apply_cosmetic_col if apply_cosmetic_col is not None else cosmetic_cfg)
        )
        apply_pedestal = (
            pedestal_ui if pedestal_ui is not None
            else (apply_pedestal_col if apply_pedestal_col is not None else pedestal_cfg)
        )
        return bool(apply_cosmetic), bool(apply_pedestal)


    def calibrate_lights(self):
        """Performs calibration on selected light frames using Master Darks and Flats, considering overrides."""
        # Make sure columns 2/3 have something where possible
        self.assign_best_master_files(fill_only=True)

        if not self.stacking_directory:
            QMessageBox.warning(self, "Error", "Please set the stacking directory first.")
            return

        calibrated_dir = os.path.join(self.stacking_directory, "Calibrated")
        os.makedirs(calibrated_dir, exist_ok=True)

        total_files = sum(len(files) for files in self.light_files.values())
        processed_files = 0

        # ---------- LOAD MASTER BIAS ONCE (optional) ----------
        master_bias = None
        bias_path = self.master_files.get("Bias")
        if bias_path:
            try:
                with fits.open(bias_path) as bias_hdul:
                    master_bias = bias_hdul[0].data.astype(np.float32)
                self.update_status(f"Using Master Bias: {os.path.basename(bias_path)}")
            except Exception as e:
                self.update_status(f"⚠️ Could not load Master Bias: {e}")
                master_bias = None

        for i in range(self.light_tree.topLevelItemCount()):
            filter_item = self.light_tree.topLevelItem(i)
            filter_name = filter_item.text(0)

            for j in range(filter_item.childCount()):
                exposure_item = filter_item.child(j)
                exposure_text = exposure_item.text(0)

                # Get default corrections
                apply_cosmetic, apply_pedestal = self._resolve_corrections_for_exposure(exposure_item)
                pedestal_value = (self.pedestal_spinbox.value() / 65535.0) if apply_pedestal else 0.0

                # (optional) keep the Corrections column in sync for visibility/debugging
                try:
                    exposure_item.setText(
                        4,
                        f"Cosmetic: {'True' if apply_cosmetic else 'False'}, "
                        f"Pedestal: {'True' if apply_pedestal else 'False'}"
                    )
                except Exception:
                    pass

                for k in range(exposure_item.childCount()):
                    leaf = exposure_item.child(k)
                    filename = leaf.text(0)
                    meta = leaf.text(1)

                    # Get session from metadata
                    session_name = "Default"
                    m = re.search(r"Session: ([^|]+)", meta)
                    if m:
                        session_name = m.group(1).strip()

                    # Look up the light file from session-specific group
                    composite_key = (f"{filter_name} - {exposure_text}", session_name)
                    light_file_list = self.light_files.get(composite_key, [])
                    light_file = next((f for f in light_file_list if os.path.basename(f) == filename), None)
                    if not light_file:
                        continue

                    # Determine size from header
                    header, _ = get_valid_header(light_file)
                    width = int(header.get("NAXIS1", 0))
                    height = int(header.get("NAXIS2", 0))
                    image_size = f"{width}x{height}"

                    # ---------- RESOLVE MASTER DARK ----------
                    manual_dark_key_full  = f"{filter_name} - {exposure_text}"
                    manual_dark_key_short = exposure_text
                    master_dark_path = (
                        self.manual_dark_overrides.get(manual_dark_key_full)
                        or self.manual_dark_overrides.get(manual_dark_key_short)
                    )
                    if master_dark_path is None:
                        # If the leaf shows a basename, map it back to a stored path
                        name_in_leaf = (leaf.text(2) or "").strip()
                        if name_in_leaf:
                            for _, path in self.master_files.items():
                                if os.path.basename(path) == name_in_leaf:
                                    master_dark_path = path
                                    break
                    if master_dark_path is None:
                        # Last resort: auto-pick by size+exposure
                        mm = re.match(r"([\d.]+)s", exposure_text or "")
                        exp_time = float(mm.group(1)) if mm else 0.0
                        master_dark_path = self._auto_pick_master_dark(image_size, exp_time)
                    print(f"master_dark_path is {master_dark_path}")

                    # ---------- RESOLVE MASTER FLAT ----------
                    manual_flat_key_full = f"{filter_name} - {exposure_text}"
                    master_flat_path = self.manual_flat_overrides.get(manual_flat_key_full)
                    if master_flat_path is None:
                        name_in_leaf = (leaf.text(3) or "").strip()
                        if name_in_leaf:
                            for _, path in self.master_files.items():
                                if os.path.basename(path) == name_in_leaf:
                                    master_flat_path = path
                                    break
                    if master_flat_path is None:
                        # Prefer session-matched flat, else fall back to any size+filter flat
                        master_flat_path = self._auto_pick_master_flat(filter_name, image_size, session_name)
                    print(f"master_flat_path is {master_flat_path}")

                    self.update_status(f"Processing: {os.path.basename(light_file)}")
                    QApplication.processEvents()

                    # ---------- LOAD LIGHT ----------
                    light_data, hdr, bit_depth, is_mono = load_image(light_file)
                    if light_data is None or hdr is None:
                        self.update_status(f"❌ ERROR: Failed to load {os.path.basename(light_file)}")
                        continue

                    # Work in CHW for color; leave mono as H,W
                    if not is_mono and light_data.ndim == 3 and light_data.shape[-1] == 3:
                        light_data = light_data.transpose(2, 0, 1)  # HWC -> CHW

                    # ---------- APPLY BIAS (optional) ----------
                    if master_bias is not None:
                        if is_mono:
                            light_data -= master_bias
                        else:
                            light_data -= master_bias[np.newaxis, :, :]
                        self.update_status("Bias Subtracted")
                        QApplication.processEvents()

                    # ---------- APPLY DARK (if resolved) ----------
                    if master_dark_path:
                        dark_data, _, _, dark_is_mono = load_image(master_dark_path)
                        if dark_data is not None:
                            if not dark_is_mono and dark_data.ndim == 3 and dark_data.shape[-1] == 3:
                                dark_data = dark_data.transpose(2, 0, 1)  # HWC -> CHW
                            # shape-safe subtract with pedestal (expects stack,F dimension)
                            if light_data.ndim == 2:  # mono
                                tmp = subtract_dark_with_pedestal(light_data[np.newaxis, :, :], dark_data, pedestal_value)
                                light_data = tmp[0]
                            else:                      # CHW
                                tmp = subtract_dark_with_pedestal(light_data[np.newaxis, :, :], dark_data, pedestal_value)
                                light_data = tmp[0]
                            self.update_status(f"Dark Subtracted: {os.path.basename(master_dark_path)}")
                            QApplication.processEvents()

                    # ---------- APPLY FLAT (if resolved) ----------
                    if master_flat_path:
                        flat_data, _, _, flat_is_mono = load_image(master_flat_path)
                        if flat_data is not None:
                            if not flat_is_mono and flat_data.ndim == 3 and flat_data.shape[-1] == 3:
                                flat_data = flat_data.transpose(2, 0, 1)  # HWC -> CHW

                            # Ensure float32 and normalize flat to mean 1.0 to preserve flux scaling
                            flat_data = flat_data.astype(np.float32, copy=False)
                            # Avoid zero/NaN in flats
                            flat_data = np.nan_to_num(flat_data, nan=1.0, posinf=1.0, neginf=1.0)
                            if flat_data.ndim == 2:
                                denom = np.median(flat_data[flat_data > 0])
                                if not np.isfinite(denom) or denom <= 0: denom = 1.0
                                flat_data /= denom
                            else:  # CHW: normalize per-channel
                                for c in range(flat_data.shape[0]):
                                    band = flat_data[c]
                                    denom = np.median(band[band > 0])
                                    if not np.isfinite(denom) or denom <= 0: denom = 1.0
                                    flat_data[c] = band / denom

                            # Safety: forbid exact zeros in denominator
                            flat_data[flat_data == 0] = 1.0

                            light_data = apply_flat_division_numba(light_data, flat_data)
                            self.update_status(f"Flat Applied: {os.path.basename(master_flat_path)}")
                            QApplication.processEvents()

                    # ---------- COSMETIC (optional) ----------
                    if apply_cosmetic:
                        # Pull configured values (fallbacks preserve current behavior)
                        hot_sigma      = self.settings.value("stacking/cosmetic/hot_sigma", 5.0, type=float)
                        cold_sigma     = self.settings.value("stacking/cosmetic/cold_sigma", 5.0, type=float)
                        star_mean_ratio= self.settings.value("stacking/cosmetic/star_mean_ratio", 0.22, type=float)
                        star_max_ratio = self.settings.value("stacking/cosmetic/star_max_ratio", 0.55, type=float)
                        sat_quantile   = self.settings.value("stacking/cosmetic/sat_quantile", 0.9995, type=float)

                        if hdr.get("BAYERPAT"):
                            # Use the FITS header pattern if present
                            pattern = str(hdr.get("BAYERPAT")).strip().upper()
                            light_data = bulk_cosmetic_correction_bayer(
                                light_data,
                                hot_sigma=hot_sigma,
                                cold_sigma=cold_sigma,
                                star_mean_ratio=star_mean_ratio,
                                star_max_ratio=star_max_ratio,
                                sat_quantile=sat_quantile,
                                pattern=pattern if pattern in ("RGGB","BGGR","GRBG","GBRG") else "RGGB"
                            )
                            self.update_status("Cosmetic Correction Applied for Bayer Pattern")
                        else:
                            light_data = bulk_cosmetic_correction_numba(
                                light_data,
                                hot_sigma=hot_sigma,
                                cold_sigma=cold_sigma,
                                star_mean_ratio=star_mean_ratio,
                                star_max_ratio=star_max_ratio,
                                sat_quantile=sat_quantile
                            )
                            self.update_status("Cosmetic Correction Applied")
                        QApplication.processEvents()

                    # Back to HWC for saving if color
                    # Back to HWC for saving if color
                    if not is_mono and light_data.ndim == 3 and light_data.shape[0] == 3:
                        light_data = light_data.transpose(1, 2, 0)  # CHW -> HWC

                    # Sanitize numerics but DO NOT rescale
                    light_data = np.nan_to_num(light_data.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)

                    min_val = float(np.min(light_data))
                    max_val = float(np.max(light_data))
                    self.update_status(f"Before saving: min = {min_val:.4f}, max = {max_val:.4f}")
                    print(f"Before saving: min = {min_val:.4f}, max = {max_val:.4f}")
                    QApplication.processEvents()

                    # Annotate header
                    try:
                        hdr['HISTORY'] = 'Calibrated: bias/dark sub, flat division'
                        hdr['CALMIN']  = (min_val, 'Min pixel before save (float)')
                        hdr['CALMAX']  = (max_val, 'Max pixel before save (float)')
                    except Exception:
                        pass

                    calibrated_filename = os.path.join(
                        calibrated_dir, os.path.basename(light_file).replace(".fit", "_c.fit")
                    )

                    # Force float32 FITS regardless of camera bit depth
                    save_image(
                        img_array=light_data,
                        filename=calibrated_filename,
                        original_format="fit",
                        bit_depth="32-bit floating point",          # << force float32 to avoid clipping negatives
                        original_header=hdr,
                        is_mono=is_mono
                    )

                    processed_files += 1
                    self.update_status(f"Saved: {os.path.basename(calibrated_filename)} ({processed_files}/{total_files})")
                    QApplication.processEvents()

        self.update_status("✅ Calibration Complete!")
        QApplication.processEvents()
        self.populate_calibrated_lights()


        # ── NEW: optionally roll straight into registration+integration ──
        try:
            if self.settings.value("stacking/auto_register_after_cal", False, type=bool):
                # Switch UI to the Image Registration tab, if we can find it
                if hasattr(self, "tabs") and self.tabs is not None:
                    try:
                        for i in range(self.tabs.count()):
                            # match by tab label if available
                            if self.tabs.tabText(i).lower().startswith("image registration"):
                                self.tabs.setCurrentIndex(i)
                                break
                    except Exception:
                        pass  # harmless if tab text lookup fails

                self.update_status("⚙️ Auto: starting registration & integration…")
                QApplication.processEvents()

                # Prefer button .click() (preserves any guard/flags)
                if hasattr(self, "register_images_btn") and self.register_images_btn is not None:
                    # Guard against re-entrancy if a run is already in progress
                    if not getattr(self, "_registration_busy", False):
                        self.register_images_btn.click()
                    else:
                        self.update_status("ℹ️ Registration already in progress; auto-run skipped.")
                # Fallback: call the method directly
                elif hasattr(self, "register_images"):
                    if not getattr(self, "_registration_busy", False):
                        self.register_images()
                    else:
                        self.update_status("ℹ️ Registration already in progress; auto-run skipped.")
        except Exception as e:
            self.update_status(f"⚠️ Auto register/integrate failed: {e}")

    def extract_light_files_from_tree(self):
        """
        Walks self.reg_tree and rebuilds self.light_files as
        { group_key: [abs_path1, abs_path2, ...], ... }
        """
        new = {}
        for i in range(self.reg_tree.topLevelItemCount()):
            group = self.reg_tree.topLevelItem(i)
            key   = group.text(0)
            files = []

            # dive into exposure → leaf or direct leaf
            for j in range(group.childCount()):
                sub = group.child(j)
                leaves = []
                if sub.childCount()>0:
                    for k in range(sub.childCount()):
                        leaves.append(sub.child(k))
                else:
                    leaves.append(sub)

                for leaf in leaves:
                    fp = leaf.data(0, Qt.ItemDataRole.UserRole)
                    if fp and os.path.exists(fp):
                        files.append(fp)
                    else:
                        self.update_status(f"⚠️ WARNING: File not found: {fp}")
            if files:
                new[key] = files

        self.light_files = new
        total = sum(len(v) for v in new.values())
        self.update_status(f"✅ Extracted Light Files: {total} total")


    def select_reference_frame_robust(self, frame_weights, sigma_threshold=1.0):
        """
        Instead of sigma filtering, pick the frame at the 75th percentile of frame weights.
        This assumes that higher weights are better and that the 75th percentile represents
        a good-quality frame.
        
        Parameters
        ----------
        frame_weights : dict
            Mapping { file_path: weight_value } for each frame.
        
        Returns
        -------
        best_frame : str or None
            The file path of the chosen reference frame, or None if no frames are available.
        """
        items = list(frame_weights.items())  # List of (file_path, weight) pairs
        if not items:
            return None

        # Sort frames by weight in ascending order.
        items.sort(key=lambda x: x[1])
        n = len(items)
        # Get the index corresponding to the 75th percentile.
        index = int(0.75 * (n - 1))
        best_frame = items[index][0]
        return best_frame

    def prompt_for_reference_frame(self):
        new_ref, _ = QFileDialog.getOpenFileName(
            self,
            "Select New Reference Frame",
            "",  # default directory
            "FITS Files (*.fit *.fits);;All Files (*)"
        )
        return new_ref if new_ref else None

    def extract_light_files_from_tree(self, *, debug: bool = False):
        """
        Rebuild self.light_files from what's *currently shown* in reg_tree.
        - Only uses leaf items (childCount()==0)
        - Repairs missing leaf UserRole by matching basename against parent's cached list
        - Filters non-existent paths
        """
        light_files: dict[str, list[str]] = {}
        total_leafs = 0
        total_paths = 0

        for i in range(self.reg_tree.topLevelItemCount()):
            top = self.reg_tree.topLevelItem(i)
            group_key = top.text(0)
            repaired_from_parent = 0

            # Parent's cached list (may be stale but useful for repairing)
            parent_cached = top.data(0, Qt.ItemDataRole.UserRole) or []

            paths: list[str] = []
            for j in range(top.childCount()):
                leaf = top.child(j)
                # Only accept real leaf rows (no grandchildren expected in this tree)
                if leaf.childCount() != 0:
                    continue

                total_leafs += 1

                fp = leaf.data(0, Qt.ItemDataRole.UserRole)
                if not fp:
                    # Try to repair by basename match against parent's cached list
                    name = leaf.text(0).lstrip("⚠️ ").strip()
                    match = next((p for p in parent_cached if os.path.basename(p) == name), None)
                    if match:
                        leaf.setData(0, Qt.ItemDataRole.UserRole, match)
                        fp = match
                        repaired_from_parent += 1

                if fp and isinstance(fp, str) and os.path.exists(fp):
                    paths.append(fp)

            if paths:
                light_files[group_key] = paths
                # keep the parent cache in sync for future repairs
                top.setData(0, Qt.ItemDataRole.UserRole, paths)
                total_paths += len(paths)

            if debug:
                self.update_status(
                    f"⤴ {group_key}: {len(paths)} files"
                    + (f" (repaired {repaired_from_parent})" if repaired_from_parent else "")
                )

        self.light_files = light_files
        if debug:
            self.update_status(f"🧭 Tree snapshot → groups: {len(light_files)}, leaves seen: {total_leafs}, paths kept: {total_paths}")
        return light_files

    def _norm_filter_key(self, s: str) -> str:
        s = (s or "").lower()
        # map greek letters to ascii
        s = s.replace("α", "a").replace("β", "b")
        return re.sub(r"[^a-z0-9]+", "", s)

    def _classify_filter(self, filt_str: str) -> str:
        """
        Return one of:
        'DUAL_HA_OIII', 'DUAL_SII_OIII', 'DUAL_SII_HB',
        'MONO_HA', 'MONO_SII', 'MONO_OIII', 'MONO_HB',
        'UNKNOWN'
        """
        k = self._norm_filter_key(filt_str)
        comps = set()

        # explicit component tokens
        if "ha"    in k or "halpha" in k: comps.add("ha")
        if "sii"   in k or "s2"     in k: comps.add("sii")
        if "oiii"  in k or "o3"     in k: comps.add("oiii")
        if "hb"    in k or "hbeta"  in k: comps.add("hb")

        # common vendor aliases → Ha/OIII
        vendor_aliases = (
            "lextreme", "lenhance", "lultimate",
            "nbz", "nbzu", "alpt", "alp",
            "duo-band", "duoband", "dual band", "dual-band", "dualband"
        )
        if any(alias in k for alias in vendor_aliases):
            comps.update({"ha", "oiii"})

        # generic dual/duo/bicolor markers → assume Ha/OIII (most OSC duals)
        dual_markers = (
            "dual", "duo", "2band", "2-band", "two band",
            "bicolor", "bi-color", "bicolour", "bi-colour",
            "dualnb", "dual-nb", "duo-nb", "duonb",
            "duo narrow", "dual narrow"
        )
        if any(m in k for m in dual_markers):
            comps.update({"ha", "oiii"})

        # decide
        if {"ha","oiii"}.issubset(comps):  return "DUAL_HA_OIII"
        if {"sii","oiii"}.issubset(comps): return "DUAL_SII_OIII"
        if {"sii","hb"}.issubset(comps):   return "DUAL_SII_HB"

        if comps == {"ha"}:   return "MONO_HA"
        if comps == {"sii"}:  return "MONO_SII"
        if comps == {"oiii"}: return "MONO_OIII"
        if comps == {"hb"}:   return "MONO_HB"

        # NEW: if user explicitly asked to split dual-band, default to Ha/OIII
        try:
            if hasattr(self, "split_dualband_cb") and self.split_dualband_cb.isChecked():
                return "DUAL_HA_OIII"
        except Exception:
            pass

        return "UNKNOWN"

    def _get_filter_name(self, path: str) -> str:
        # Prefer FITS header 'FILTER'; fall back to filename tokens
        try:
            hdr = fits.getheader(path, ext=0)
            for key in ("FILTER", "FILTER1", "HIERARCH INDI FILTER", "HIERARCH ESO INS FILT1 NAME"):
                if key in hdr and str(hdr[key]).strip():
                    return str(hdr[key]).strip()
        except Exception:
            pass
        return os.path.basename(path)

    def _current_global_drizzle(self):
        # read from the “global” controls (used as a template)
        return {
            "enabled": self.drizzle_checkbox.isChecked(),
            "scale": float(self.drizzle_scale_combo.currentText().replace("x","", 1)),
            "drop": float(self.drizzle_drop_shrink_spin.value())
        }

    def _split_dual_band_osc(self, selected_groups=None):
        """
        Create mono Ha/SII/OIII frames from dual-band OSC files and
        update self.light_files so integration sees separate channels.
        """
        selected_groups = selected_groups or set()
        out_dir = os.path.join(self.stacking_directory, "DualBand_Split")
        os.makedirs(out_dir, exist_ok=True)

        ha_files, sii_files, oiii_files, hb_files = [], [], [], []
        inherit_map = {}                      # gk -> set(parent_group names)   # <<< NEW
        parent_of = {}                        # path -> parent_group            # <<< NEW

        # Walk all groups/files you already collected
        old_groups = list(self.light_files.items())
        old_drizzle = dict(self.per_group_drizzle)
        for group, files in old_groups:
            for fp in files:
                try:
                    img, hdr, _, _ = load_image(fp)
                    if img is None:
                        self.update_status(f"⚠️ Cannot load {fp}; skipping.")
                        continue

                    if hdr and hdr.get("BAYERPAT"):
                        img = self.debayer_image(img, fp, hdr)

                    # 3-channel split; otherwise treat mono via classifier
                    if img.ndim != 3 or img.shape[-1] < 2:
                        filt = self._get_filter_name(fp)
                        cls  = self._classify_filter(filt)
                        if cls == "MONO_HA":
                            ha_files.append(fp);   parent_of[fp] = group        # <<< NEW
                        elif cls == "MONO_SII":
                            sii_files.append(fp);  parent_of[fp] = group        # <<< NEW
                        elif cls == "MONO_OIII":
                            oiii_files.append(fp); parent_of[fp] = group        # <<< NEW
                        elif cls == "MONO_HB":   hb_files.append(fp);  parent_of[fp] = group        # <<< NEW
                        # else: leave in original groups
                        continue

                    filt = self._get_filter_name(fp)
                    cls  = self._classify_filter(filt)

                    R = img[..., 0]; G = img[..., 1]
                    base = os.path.splitext(os.path.basename(fp))[0]

                    if cls == "DUAL_HA_OIII":
                        ha_path   = os.path.join(out_dir, f"{base}_Ha.fit")
                        oiii_path = os.path.join(out_dir, f"{base}_OIII.fit")
                        self._write_band_fit(ha_path,  R, hdr, "Ha",  src_filter=filt)
                        self._write_band_fit(oiii_path, G, hdr, "OIII", src_filter=filt)
                        ha_files.append(ha_path);     parent_of[ha_path]   = group   # <<< NEW
                        oiii_files.append(oiii_path); parent_of[oiii_path] = group   # <<< NEW

                    elif cls == "DUAL_SII_OIII":
                        sii_path  = os.path.join(out_dir, f"{base}_SII.fit")
                        oiii_path = os.path.join(out_dir, f"{base}_OIII.fit")
                        self._write_band_fit(sii_path, R, hdr, "SII",  src_filter=filt)
                        self._write_band_fit(oiii_path, G, hdr, "OIII", src_filter=filt)
                        sii_files.append(sii_path);    parent_of[sii_path]  = group  # <<< NEW
                        oiii_files.append(oiii_path);  parent_of[oiii_path] = group  # <<< NEW

                    elif cls == "DUAL_SII_HB":  # NEW → R=SII, G=Hb  (G works well; we can add G+B later if you want)
                        sii_path = os.path.join(out_dir, f"{base}_SII.fit")
                        hb_path  = os.path.join(out_dir, f"{base}_Hb.fit")
                        self._write_band_fit(sii_path, R, hdr, "SII", src_filter=filt)
                        self._write_band_fit(hb_path,  G, hdr, "Hb",  src_filter=filt)
                        sii_files.append(sii_path); parent_of[sii_path] = group
                        hb_files.append(hb_path);   parent_of[hb_path]  = group

                    else:
                        pass

                except Exception as e:
                    self.update_status(f"⚠️ Split error on {os.path.basename(fp)}: {e}")

        # Group the new files
        def _group_key(band: str, path: str) -> str:
            try:
                h = fits.getheader(path, ext=0)
                exp = h.get("EXPTIME") or h.get("EXPOSURE") or ""
                w   = h.get("NAXIS1","?"); hgt = h.get("NAXIS2","?")
                exp_str = f"{float(exp):.1f}s" if isinstance(exp, (int,float)) else str(exp)
                return f"{band} - {exp_str} - {w}x{hgt}"
            except Exception:
                return f"{band} - ? - ?x?"

        new_groups = {}
        for band, flist in (("Ha", ha_files), ("SII", sii_files), ("OIII", oiii_files), ("Hb", hb_files)):  # NEW Hb
            for p in flist:
                gk = _group_key(band, p)
                new_groups.setdefault(gk, []).append(p)
                parent = parent_of.get(p)
                if parent:
                    inherit_map.setdefault(gk, set()).add(parent)

        if new_groups:
            self.light_files = new_groups

            # Seed drizzle for the new groups based on parents
            seeded = 0
            global_template = self._current_global_drizzle()   # make sure this helper exists
            self.per_group_drizzle = {}  # rebuild for the new groups

            for gk, parents in inherit_map.items():
                parent_cfgs = [old_drizzle.get(pg) for pg in parents if old_drizzle.get(pg)]
                chosen = None
                for cfg in parent_cfgs:
                    if cfg.get("enabled"):
                        chosen = cfg
                        break
                if not chosen and parent_cfgs:
                    chosen = parent_cfgs[0]

                if not chosen and (parents & selected_groups) and global_template.get("enabled"):
                    chosen = global_template

                if chosen:
                    self.per_group_drizzle[gk] = dict(chosen)
                    seeded += 1


            self.update_status(
                f"✅ Dual-band split complete: Ha={len(ha_files)}, SII={len(sii_files)}, "
                f"OIII={len(oiii_files)}, Hb={len(hb_files)} (drizzle seeded on {seeded} new group(s))"
            )
        else:
            self.update_status("ℹ️ No dual-band frames detected or split.")

    def _write_band_fit(self, out_path: str, data: np.ndarray, src_header: Optional[fits.Header],
                        band: str, src_filter: str):

        arr = np.ascontiguousarray(data.astype(np.float32))

        hdr = (src_header.copy() if isinstance(src_header, fits.Header) else fits.Header())

        # --- strip CFA/Bayer-related cards so we never try to debayer these ---
        cfa_like = (
            "BAYERPAT", "BAYER_PATTERN", "DEBAYER", "DEBAYERING", "DEMAT", "DEMOSAIC",
            "XBAYROFF", "YBAYROFF", "COLORTYP", "COLORSPACE", "HIERARCH CFA", "HIERARCH OSC",
            "HIERARCH ASI BAYERPATTERN", "HIERARCH DNG CFA", "HIERARCH ZWO CFA"
        )
        for k in list(hdr.keys()):
            kk = str(k).upper()
            if any(token in kk for token in ("BAYER", "CFA", "DEMOSA")) or kk in cfa_like:
                try:
                    del hdr[k]
                except Exception:
                    pass

        # Mark these as mono split files & set the band as the filter
        hdr["FILTER"] = (band, "Channel from dual-band split")
        hdr["SPLITDB"] = (True, "This frame was generated by dual-band splitting")
        hdr.add_history(f"Dual-band split: {band} from {src_filter}")

        fits.PrimaryHDU(data=arr, header=hdr).writeto(out_path, overwrite=True)

    def _drizzle_text_for_group(self, group_key: str) -> str:
        d = self.per_group_drizzle.get(group_key)
        if not d:
            return ""
        return f"Drizzle: {d.get('enabled', False)}, Scale: {d.get('scale','1x')}, Drop:{d.get('drop',0.65)}"

    def _refresh_reg_tree_from_light_files(self):
        self.reg_tree.clear()
        for group, files in self.light_files.items():
            top = QTreeWidgetItem([group, f"{len(files)} file(s)", self._drizzle_text_for_group(group)])
            self.reg_tree.addTopLevelItem(top)
            for fp in files:
                # Optional: show some header metadata
                meta = ""
                try:
                    hdr = fits.getheader(fp, ext=0)
                    filt = hdr.get("FILTER", "")
                    exp  = hdr.get("EXPTIME") or hdr.get("EXPOSURE") or ""
                    if isinstance(exp, (int, float)): exp = f"{exp:.1f}s"
                    meta = f"Filter={filt}  Exp={exp}"
                except Exception:
                    pass
                child = QTreeWidgetItem([os.path.basename(fp), meta, ""])
                top.addChild(child)
        self.reg_tree.expandAll()

    def _norm_ang(self, a):
        a = a % 360.0
        return a + 360.0 if a < 0 else a

    def _angdiff(self, a, b):
        # smallest absolute difference in degrees
        return abs((self._norm_ang(a) - self._norm_ang(b) + 180.0) % 360.0 - 180.0)

    def _extract_pa_deg(self, hdr):
        """
        Try common FITS/PI keys for camera/sky position angle (degrees).
        Works with both astropy FITS Header and XISF image-metadata dicts.

        Heuristics:
        1) Direct keywords (first match wins): POSANGLE, ANGLE, ROTANGLE, ROTSKYPA,
            ROTATOR, PA, ORIENTAT, CROTA2, CROTA1.
            • Values may be numbers or strings like '123.4 deg' — we parse the first float.
        2) WCS fallback:
            • CD matrix:  pa = atan2(-CD1_2, CD2_2)
            • PC matrix (+ optional CDELT scaling): pa = atan2(-PC1_2, PC2_2)
        Returns float in degrees, or None if unknown.
        """


        if hdr is None:
            return None

        # helper: unified header getter (FITS Header or XISF dict)
        def _get(k, default=None):
            try:
                return self._hdr_get(hdr, k, default)
            except Exception:
                # fall back to direct mapping-like access if needed
                try:
                    return hdr.get(k, default)
                except Exception:
                    return default

        # parse possibly-string value → float
        def _as_float_deg(v):
            if v is None:
                return None
            if isinstance(v, (int, float, np.integer, np.floating)):
                try:
                    return float(v)
                except Exception:
                    return None
            # strings like "123.4", "123.4 deg", "123,4", etc.
            try:
                s = str(v)
                m = re.search(r"[-+]?\d+(?:[.,]\d+)?", s)
                if not m:
                    return None
                val = float(m.group(0).replace(",", "."))
                return val
            except Exception:
                return None

        # 1) direct keyword attempts (first hit wins)
        direct_keys = (
            "POSANGLE", "ANGLE", "ROTANGLE", "ROTSKYPA", "ROTATOR",
            "PA", "ORIENTAT", "CROTA2", "CROTA1"
        )
        for k in direct_keys:
            v = _get(k, None)
            ang = _as_float_deg(v)
            if ang is not None:
                return float(ang)

        # 2) WCS fallback using CD or PC matrices
        # CD first
        cd11 = _get('CD1_1'); cd12 = _get('CD1_2'); cd22 = _get('CD2_2')
        if cd11 is not None and cd12 is not None and cd22 is not None:
            try:
                pa = np.degrees(np.arctan2(-float(cd12), float(cd22)))
                return float(pa)
            except Exception:
                pass

        # PC (optionally combined with CDELT)
        pc11 = _get('PC1_1'); pc12 = _get('PC1_2'); pc22 = _get('PC2_2')
        if pc11 is not None and pc12 is not None and pc22 is not None:
            try:
                # If CDELT present, rotation is still given by PC (scale cancels out for angle)
                pa = np.degrees(np.arctan2(-float(pc12), float(pc22)))
                return float(pa)
            except Exception:
                pass

        return None


    def _maybe_rot180(self, img, pa_cur, pa_ref, tol_deg):
        """
        If |(pa_cur - pa_ref)| ≈ 180° (within tol), rotate image 180°.
        Works for (H,W) or (H,W,3).
        Returns (img_out, rotated_bool).
        """
        
        if pa_cur is None or pa_ref is None:
            return img, False
        d = self._angdiff(pa_cur, pa_ref)
        if abs(d - 180.0) <= tol_deg:
            # 180° is just two 90° rotations; cheap & exact
            self.update_status(f"Flipping Image")
            QApplication.processEvents()
            return np.rot90(img, 2).copy(), True
        return img, False

    def _ui_log(self, msg: str):
        self.update_status(msg)  # your existing status logger
        # let Qt process pending paint/input signals so the UI updates
        QCoreApplication.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 25)

    def _maybe_warn_cfa_low_frames(self):
        if not (getattr(self, "cfa_drizzle_cb", None) and self.cfa_drizzle_cb.isChecked()):
            self._cfa_for_this_run = None  # follow checkbox (OFF)
            return

        # Count frames per group (use the *current* reg tree groups)
        per_group_counts = {g: len(v) for g, v in (self.light_files or {}).items() if v}
        if not per_group_counts:
            self._cfa_for_this_run = None
            return

        worst = min(per_group_counts.values())

        # Scale-aware cutoff (you can expose this in QSettings if you like)
        try:
            scale_txt = self.drizzle_scale_combo.currentText()
            scale = float(scale_txt.replace("x", "").strip())
        except Exception:
            scale = 1.0
        cutoff = {1.0: 32, 2.0: 64, 3.0: 96}.get(scale, 64)

        if worst >= cutoff:
            self._cfa_for_this_run = True   # keep raw CFA mapping
            return

        # Ask the user
        msg = (f"CFA Drizzle is enabled but at least one group has only {worst} frames.\n\n"
            f"CFA Drizzle typically needs ≥{cutoff} frames (scale {scale:.0f}×) for good coverage.\n"
            "Switch to Edge-Aware Interpolation for this run?")
        ret = QMessageBox.question(
            self, "CFA Drizzle: Low Sample Count",
            msg, QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes
        )

        if ret == QMessageBox.StandardButton.Yes:
            # Disable raw CFA just for this run
            self._cfa_for_this_run = False
            self.update_status("⚠️ CFA Drizzle: low-count fallback → using Edge-Aware Interpolation for this run.")
        else:
            self._cfa_for_this_run = True
            self.update_status("ℹ️ CFA Drizzle kept despite low frame count (you chose No).")

    def _ensure_comet_seed_now(self) -> bool:
        """If no comet seed exists, open the picker. Return True IFF we have a seed after."""
        if getattr(self, "_comet_seed", None):
            return True
        # Open the same picker you already have
        self._pick_comet_center()
        return bool(getattr(self, "_comet_seed", None))

    # small helper to toggle UI while registration is running
    def _set_registration_busy(self, busy: bool):
        self._registration_busy = bool(busy)
        self.register_images_btn.setEnabled(not busy)
        self.integrate_registered_btn.setEnabled(not busy)
        # optional visual hint
        if busy:
            self.register_images_btn.setText("⏳ Registering…")
            self.register_images_btn.setToolTip("Registration in progress…")
        else:
            self.register_images_btn.setText("🔥🚀Register and Integrate Images🔥🚀")
            self.register_images_btn.setToolTip("")

        # prevent accidental double-queue from keyboard/space
        self.register_images_btn.blockSignals(busy)

    def _cosmetic_enabled(self) -> bool:
        try:
            if hasattr(self, "cosmetic_checkbox") and self.cosmetic_checkbox is not None:
                return bool(self.cosmetic_checkbox.isChecked())
        except Exception:
            pass
        return bool(self.settings.value("stacking/cosmetic_enabled", True, type=bool))

    # ——— generic header access ———
    def _hdr_get(self, hdr, key, default=None):
        """Uniform header getter for FITS (astropy Header) or XISF (dict with FITSKeywords)."""
        if hdr is None:
            return default
        # FITS Header
        try:
            if hasattr(hdr, "get"):
                return hdr.get(key, default)
        except Exception:
            pass
        # XISF-style dict from XISF.get_images_metadata()[i]
        try:
            # FITSKeywords: { KEY: [ {value, comment}, ... ] }
            fkw = hdr.get("FITSKeywords", {})
            lst = fkw.get(key)
            if lst and isinstance(lst, list) and "value" in lst[0]:
                return lst[0]["value"]
        except Exception:
            pass
        # fall back to plain dict
        try:
            return hdr.get(key, default)
        except Exception:
            return default


    # ——— binning for either format ———
    def _bin_from_header_fast_any(self, fp: str) -> tuple[int,int]:
        ext = os.path.splitext(fp)[1].lower()
        # FITS quick path
        if ext in (".fits", ".fit", ".fz"):
            try:
                h = fits.getheader(fp, ext=0)
                # common variants
                xb = int(h.get("XBINNING", h.get("XBIN", 1)))
                yb = int(h.get("YBINNING", h.get("YBIN", 1)))
                return max(1, xb), max(1, yb)
            except Exception:
                return (1,1)
        # XISF path
        if ext == ".xisf":
            try:
                from legacy.xisf import XISF
                x = XISF(fp)
                ims = x.get_images_metadata()
                if not ims:
                    return (1,1)
                m0 = ims[0]
                fkw = m0.get("FITSKeywords", {})
                def _kw(name, default=None):
                    vals = fkw.get(name)
                    return vals[0]["value"] if vals and "value" in vals[0] else default
                xb = int(_kw("XBINNING", _kw("XBIN", 1)) or 1)
                yb = int(_kw("YBINNING", _kw("YBIN", 1)) or 1)
                return max(1, xb), max(1, yb)
            except Exception:
                return (1,1)
        return (1,1)


    # ——— fast preview for either format (2D float32 at target bin scale) ———
    def _quick_preview_any(self, fp: str, target_xbin: int, target_ybin: int) -> np.ndarray | None:
        ext = os.path.splitext(fp)[1].lower()
        # 1) Load a small-ish *image* block without full decode work
        img = None; hdr = None

        def _superpixel2x2(x: np.ndarray) -> np.ndarray:
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

        try:
            if ext in (".fits", ".fit", ".fz"):
                # primary data only for speed
                data = fits.getdata(fp, ext=0)
                hdr  = fits.getheader(fp, ext=0)
                img = np.asanyarray(data)
            elif ext == ".xisf":
                from legacy.xisf import XISF
                x = XISF(fp)
                ims = x.get_images_metadata()
                if not ims:
                    return None
                hdr = ims[0]  # carry XISF image metadata dict as "header"
                img = x.read_image(0)  # channels-last
            else:
                # generic formats: use your existing thumb path
                w, h = self._get_image_size(fp)
                # cheap preview load (PIL) — grayscale
                from PIL import Image
                im = Image.open(fp).convert("L")
                img = np.asarray(im, dtype=np.float32) / 255.0
                hdr = {}

            a = np.asarray(img)
            if a.ndim == 3 and a.shape[-1] == 1:
                a = a[...,0]
            # if it’s color, make a luma-like 2×2 superpixel preview; if mono/CFA, same superpixel trick
            if a.ndim == 3 and a.shape[-1] == 3:
                prev2d = _superpixel2x2(a)
            else:
                prev2d = _superpixel2x2(a)

            # resample preview to the target bin scale
            xb, yb = self._bin_from_header_fast_any(fp)
            sx = float(xb) / float(target_xbin)
            sy = float(yb) / float(target_ybin)
            if (abs(sx - 1.0) > 1e-6) or (abs(sy - 1.0) > 1e-6):
                prev2d = _resize_to_scale(prev2d, sx, sy)

            return np.ascontiguousarray(prev2d.astype(np.float32, copy=False))
        except Exception:
            return None


    # ——— on-demand full load (float32, header-like), for normalization stage ———
    def _load_image_any(self, fp: str):
        """
        Return (img, hdr_like) for FITS/XISF/other.
        img is float32, channels-last if color, shape (H,W) if mono.
        hdr_like: FITS Header for FITS; XISF image metadata dict for XISF; {} otherwise.
        """
        ext = os.path.splitext(fp)[1].lower()
        try:
            if ext in (".fits", ".fit", ".fz"):
                from legacy.image_manager import load_image as legacy_load_image
                img, hdr, _, _ = legacy_load_image(fp)
                return img, (hdr or fits.Header())
            if ext == ".xisf":
                from legacy.xisf import XISF
                x = XISF(fp)
                ims = x.get_images_metadata()
                if not ims:
                    return None, {}
                img = x.read_image(0)  # channels-last
                # normalize integer types to [0..1] just like FITS path
                a = np.asarray(img)
                if a.dtype.kind in "ui":
                    a = a.astype(np.float32) / np.float32(np.iinfo(a.dtype).max)
                elif a.dtype.kind == "f":
                    a = a.astype(np.float32, copy=False)
                else:
                    a = a.astype(np.float32, copy=False)
                hdr_like = ims[0]  # carry XISF image metadata dict
                return a, hdr_like
            # generic images (TIFF/PNG/JPEG)
            from legacy.image_manager import load_image as legacy_load_image
            img, hdr, _, _ = legacy_load_image(fp)
            return img, (hdr or {})
        except Exception:
            return None, {}


    def register_images(self):

        if getattr(self, "_registration_busy", False):
            self.update_status("⏸ Registration already running; ignoring extra click.")
            return

        self._set_registration_busy(True)

        try:
            if self.star_trail_mode:
                self.update_status("🌠 Star-Trail Mode enabled: skipping registration & using max-value stack")
                QApplication.processEvents()
                return self._make_star_trail()

            self.update_status("🔄 Image Registration Started...")
            self.extract_light_files_from_tree(debug=True)

            comet_mode = bool(getattr(self, "comet_cb", None) and self.comet_cb.isChecked())
            if comet_mode:
                self.update_status("🌠 Comet mode: please click the comet center to continue…")
                QApplication.processEvents()
                ok = self._ensure_comet_seed_now()
                if not ok:
                    QMessageBox.information(self, "Comet Mode",
                        "No comet center was selected. Registration has been cancelled so you can try again.")
                    self.update_status("❌ Registration cancelled (no comet seed).")
                    return
                else:
                    self._comet_ref_xy = None
                    self.update_status("✅ Comet seed set. Proceeding with registration…")
                    QApplication.processEvents()

            if not self.light_files:
                self.update_status("⚠️ No light files to register!")
                return

            # dual-band split unchanged...
            selected_groups = set()
            for it in self.reg_tree.selectedItems():
                top = it if it.parent() is None else it.parent()
                selected_groups.add(top.text(0))
            if self.split_dualband_cb.isChecked():
                self.update_status("🌈 Splitting dual-band OSC frames into Ha / SII / OIII...")
                self._split_dual_band_osc(selected_groups=selected_groups)
                self._refresh_reg_tree_from_light_files()

            self._maybe_warn_cfa_low_frames()

            all_files = [f for lst in self.light_files.values() for f in lst]
            self.update_status(f"📊 Found {len(all_files)} total frames. Now measuring in parallel batches...")

            # ── binning (FITS/XISF aware) ─────────────────────────────
            bin_map = {}
            min_xbin, min_ybin = None, None
            for fp in all_files:
                xb, yb = self._bin_from_header_fast_any(fp)
                bin_map[fp] = (xb, yb)
                if min_xbin is None or xb < min_xbin:
                    min_xbin = xb
                if min_ybin is None or yb < min_ybin:
                    min_ybin = yb

            target_xbin, target_ybin = (min_xbin or 1), (min_ybin or 1)
            self.update_status(
                f"🧮 Binning summary → target={target_xbin}×{target_ybin} "
                f"(range observed: x=[{min(b[0] for b in bin_map.values())}..{max(b[0] for b in bin_map.values())}], "
                f"y=[{min(b[1] for b in bin_map.values())}..{max(b[1] for b in bin_map.values())}])"
            )

            # ─────────────────────────────────────────────────────────────────────
            # Helpers
            # ─────────────────────────────────────────────────────────────────────
            def mono_preview_for_stats(img: np.ndarray, hdr, fp: str) -> np.ndarray:
                """
                Returns a 2D float32 preview that is:
                • debayer-aware (uses luma if RGB; superpixel if mono/CFA)
                • resampled to the target bin (so previews from mixed binning match scale)
                • superpixel-averaged 2×2 for speed
                """
                if img is None:
                    return None

                def _superpixel2x2(x: np.ndarray) -> np.ndarray:
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
                        Luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
                        return (Luma[0:h2:2, 0:w2:2] + Luma[0:h2:2, 1:w2:2] +
                                Luma[1:h2:2, 0:w2:2] + Luma[1:h2:2, 1:w2:2]) * 0.25

                # 1) quick 2D preview
                if img.ndim == 3 and img.shape[-1] == 3:
                    prev2d = _superpixel2x2(img)
                elif hdr and hdr.get('BAYERPAT') and img.ndim == 2:
                    prev2d = _superpixel2x2(img)          # CFA → superpixel Luma-ish
                elif img.ndim == 2:
                    prev2d = _superpixel2x2(img)          # mono
                else:
                    prev2d = img.astype(np.float32, copy=False)
                    if prev2d.ndim == 3:
                        prev2d = _superpixel2x2(prev2d)

                # 2) resample preview to the target bin
                xb, yb = bin_map.get(fp, (1, 1))
                sx = float(xb) / float(target_xbin)
                sy = float(yb) / float(target_ybin)
                if (abs(sx - 1.0) > 1e-6) or (abs(sy - 1.0) > 1e-6):
                    prev2d = _resize_to_scale(prev2d, sx, sy)

                return np.ascontiguousarray(prev2d.astype(np.float32, copy=False))

            def chunk_list(lst, size):
                for i in range(0, len(lst), size):
                    yield lst[i:i+size]

            # ─────────────────────────────────────────────────────────────────────
            # PHASE 1: measure (NO demosaic here)
            # ─────────────────────────────────────────────────────────────────────
            self.frame_weights = {}
            mean_values = {}
            star_counts = {}
            measured_frames = []
            preview_medians = {}

            max_workers = os.cpu_count() or 4
            chunk_size = max_workers
            chunks = list(chunk_list(all_files, chunk_size))
            total_chunks = len(chunks)

            from concurrent.futures import ThreadPoolExecutor, as_completed

            for idx, chunk in enumerate(chunks, 1):
                self.update_status(f"📦 Measuring chunk {idx}/{total_chunks} ({len(chunk)} frames)")
                chunk_images = []
                chunk_valid_files = []

                self.update_status(f"🌍 Loading {len(chunk)} previews in parallel (up to {max_workers} threads)...")
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futs = {executor.submit(self._quick_preview_any, fp, target_xbin, target_ybin): fp
                            for fp in chunk}
                    for fut in as_completed(futs):
                        fp = futs[fut]
                        try:
                            preview = fut.result()
                            if preview is None:
                                continue
                            chunk_images.append(preview)
                            chunk_valid_files.append(fp)
                        except Exception as e:
                            self.update_status(f"⚠️ Error previewing {fp}: {e}")
                        QApplication.processEvents()

                if not chunk_images:
                    self.update_status("⚠️ No valid previews in this chunk (couldn’t find image data in any HDU).")
                    continue

                # size align (crop) before stats
                min_h = min(im.shape[0] for im in chunk_images)
                min_w = min(im.shape[1] for im in chunk_images)
                if any((im.shape[0] != min_h or im.shape[1] != min_w) for im in chunk_images):
                    chunk_images = [_center_crop_2d(im, min_h, min_w) for im in chunk_images]

                self.update_status("🌍 Measuring global means in parallel...")
                QApplication.processEvents()
                means = np.array([float(np.mean(ci)) for ci in chunk_images], dtype=np.float32)
                mean_values.update({fp: float(means[i]) for i, fp in enumerate(chunk_valid_files)})

                def _star_job(i_fp):
                    i, fp = i_fp
                    p = chunk_images[i]
                    pmin = float(np.nanmin(p))
                    med = float(np.median(p - pmin))
                    c, ecc = compute_star_count_fast_preview(p)
                    return fp, med, c, ecc

                star_workers = min(max_workers, 8)
                with ThreadPoolExecutor(max_workers=star_workers) as ex:
                    for fp, med, c, ecc in ex.map(_star_job, enumerate(chunk_valid_files)):
                        preview_medians[fp] = med
                        star_counts[fp] = {"count": c, "eccentricity": ecc}
                        measured_frames.append(fp)

                del chunk_images

            if not measured_frames:
                self.update_status("⚠️ No frames could be measured!")
                return

            self.update_status(f"✅ All chunks complete! Measured {len(measured_frames)} frames total.")
            # ─────────────────────────────────────────────────────────────────────
            # FAST reference selection: score = starcount / (median * ecc)
            # uses stats we ALREADY measured → good for 100s of frames
            # ─────────────────────────────────────────────────────────────────────
            self.update_status("🧠 Selecting reference optimized for AstroAlign (starcount/(median*ecc))…")
            QApplication.processEvents()

            def _dominant_pa_cluster_simple(fps, get_pa, tol=12.0):
                """
                Cluster PAs in [0,180) and return the biggest cluster.
                This keeps us from picking a frame that's on the other side
                of a meridian flip when most frames agree.
                """
                vals = []
                for fp in fps:
                    pa = get_pa(fp)
                    if pa is None:
                        continue
                    # normalize PA to [0,180)
                    v = ((pa % 180.0) + 180.0) % 180.0
                    vals.append((fp, v))

                # if we couldn't read any PA, just return everything
                if not vals:
                    return fps

                vals.sort(key=lambda t: t[1])
                best_group = []
                best_size = -1

                for i in range(len(vals)):
                    center = vals[i][1]
                    inliers = []
                    for fp, v in vals:
                        d = abs(v - center)
                        d = min(d, 180.0 - d)  # wrap
                        if d <= tol:
                            inliers.append(fp)
                    if len(inliers) > best_size:
                        best_group, best_size = inliers, len(inliers)

                return best_group if best_group else fps

            def _fast_ref_score(fp: str) -> float:
                """
                score = star_count / (median * ecc)

                - star_count: more is better
                - median: darker backgrounds (lower median) get rewarded
                - ecc: rounder stars (ecc ~ 1) get rewarded; we clamp ecc
                  so tiny values don't explode the score
                """
                info = star_counts.get(fp, {"count": 0, "eccentricity": 1.0})
                star_count = float(info.get("count", 0.0))
                ecc = float(info.get("eccentricity", 1.0))
                med = float(preview_medians.get(fp, 0.0))

                # guards / clamps
                med = max(med, 1e-3)          # avoid divide-by-zero on very dark frames
                ecc = max(0.25, min(ecc, 3.0))  # keep in sane range

                return star_count / (med * ecc)

            # 1) Respect manual reference if provided
            user_ref = getattr(self, "reference_frame", None)
            if user_ref:
                self.update_status(f"📌 Using user-specified reference: {user_ref}")
                self.reference_frame = user_ref
            else:
                # 2) Optionally restrict to dominant PA cluster (helps big sets)
                def _pa_of(fp):
                    try:
                        hdr0 = fits.getheader(fp, ext=0)
                        return self._extract_pa_deg(hdr0)
                    except Exception:
                        return None

                # if we have a lot of frames, PA-cluster first
                if len(measured_frames) > 60:
                    candidates = _dominant_pa_cluster_simple(measured_frames, _pa_of, tol=12.0)
                    if not candidates:
                        candidates = measured_frames[:]
                else:
                    candidates = measured_frames[:]

                # 3) Pick the best by the fast score
                best_fp = None
                best_score = -1.0
                for fp in candidates:
                    s = _fast_ref_score(fp)
                    if s > best_score:
                        best_fp = fp
                        best_score = s

                # 4) Fallback if something went sideways
                if best_fp is None and measured_frames:
                    # fall back to previous weighting you computed
                    best_fp = max(measured_frames, key=lambda f: self.frame_weights.get(f, 0.0))
                    best_score = float(self.frame_weights.get(best_fp, 0.0))

                self.reference_frame = best_fp
                if self.reference_frame:
                    self.update_status(
                        f"📌 Auto-selected reference: {os.path.basename(self.reference_frame)} "
                        f"(score={best_score:.4f})"
                    )
                else:
                    # last resort — this should basically never happen
                    self.reference_frame = measured_frames[0]
                    self.update_status(
                        f"📌 Auto-selected reference (fallback): {os.path.basename(self.reference_frame)}"
                    )
                QApplication.processEvents()

            # expose these so the later code can show them / log them
            ref_stats_meas = star_counts.get(self.reference_frame, {"count": 0, "eccentricity": 0.0})
            ref_count = ref_stats_meas["count"]
            ref_ecc   = ref_stats_meas["eccentricity"]

            # ─────────────────────────────────────────────────────────────────────
            # Debayer the reference ONCE and compute ref_median from debayered ref
            # ─────────────────────────────────────────────────────────────────────
            ref_img_raw, ref_hdr = self._load_image_any(self.reference_frame)
            if ref_img_raw is None:
                self.update_status(f"🚨 Could not load reference {self.reference_frame}. Aborting.")
                return

            bayer = self._hdr_get(ref_hdr, 'BAYERPAT')
            splitdb = bool(self._hdr_get(ref_hdr, 'SPLITDB', False))
            if bayer and not splitdb and (ref_img_raw.ndim == 2 or (ref_img_raw.ndim == 3 and ref_img_raw.shape[-1] == 1)):
                self.update_status("📦 Debayering reference frame…")
                ref_img = self.debayer_image(ref_img_raw, self.reference_frame, ref_hdr)
            else:
                ref_img = ref_img_raw
                if ref_img.ndim == 3 and ref_img.shape[-1] == 1:
                    ref_img = np.squeeze(ref_img, axis=-1)

            # Use Luma median if color, else direct median
            if ref_img.ndim == 3 and ref_img.shape[-1] == 3:
                r, g, b = ref_img[..., 0], ref_img[..., 1], ref_img[..., 2]
                ref_Luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
                ref_median = float(np.median(ref_Luma))
            else:
                ref_median = float(np.median(ref_img))

            self.update_status(f"📊 Reference (debayered) median: {ref_median:.4f}")
            QApplication.processEvents()
            ref_stats_meas = star_counts.get(self.reference_frame, {"count": 0, "eccentricity": 0.0})
            ref_count = ref_stats_meas["count"]
            ref_ecc   = ref_stats_meas["eccentricity"]    
            # Modeless ref review (unchanged)
            stats_payload = {"star_count": ref_count, "eccentricity": ref_ecc, "mean": ref_median}

            if self.auto_accept_ref_cb.isChecked():
                self.update_status("✅ Auto-accept measured reference is enabled; using the measured best frame.")
                try:
                    self.ref_frame_path.setText(self.reference_frame or "No file selected")
                except Exception:
                    pass
            else:
                dialog = ReferenceFrameReviewDialog(self.reference_frame, stats_payload, parent=self)
                dialog.setModal(False)
                dialog.setWindowModality(Qt.WindowModality.NonModal)
                dialog.show(); dialog.raise_(); dialog.activateWindow()
                _loop = QEventLoop(self); dialog.finished.connect(_loop.quit); _loop.exec()
                result = dialog.result(); user_choice = dialog.getUserChoice()
                if result != QDialog.DialogCode.Accepted and user_choice == "select_other":
                    new_ref = self.prompt_for_reference_frame()
                    if new_ref:
                        self.reference_frame = new_ref
                        self.update_status(f"User selected a new reference frame: {new_ref}")
                        ref_img_raw, ref_hdr, _, _ = load_image(self.reference_frame)
                        if ref_img_raw is None:
                            self.update_status(f"🚨 Could not load reference {self.reference_frame}. Aborting.")
                            return
                        if ref_hdr and ref_hdr.get('BAYERPAT') and not ref_hdr.get('SPLITDB', False) and (ref_img_raw.ndim == 2 or (ref_img_raw.ndim == 3 and ref_img_raw.shape[-1] == 1)):
                            self.update_status("📦 Debayering reference frame…")
                            ref_img = self.debayer_image(ref_img_raw, self.reference_frame, ref_hdr)
                        else:
                            ref_img = ref_img_raw
                            if ref_img.ndim == 3 and ref_img.shape[-1] == 1:
                                ref_img = np.squeeze(ref_img, axis=-1)
                        if ref_img.ndim == 3 and ref_img.shape[-1] == 3:
                            r, g, b = ref_img[..., 0], ref_img[..., 1], ref_img[..., 2]
                            ref_Luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
                            ref_median = float(np.median(ref_Luma))
                        else:
                            ref_median = float(np.median(ref_img))
                        self.update_status(f"📊 (New) reference median: {ref_median:.4f}")
                    
                    else:
                        self.update_status("No new reference selected; using previous reference.")

            ref_L = _Luma(ref_img)
            ref_min = float(np.nanmin(ref_L))
            ref_target_median = float(np.nanmedian(ref_L - ref_min))
            self.update_status(f"📊 Reference min={ref_min:.6f}, normalized-median={ref_target_median:.6f}")
            QApplication.processEvents()

            # Initial per-file scale factors from preview medians
            eps = 1e-6
            scale_guess = {}
            missing = []
            for fp in measured_frames:
                m = preview_medians.get(fp, 0.0)
                if m <= eps:
                    missing.append(os.path.basename(fp))
                    m = 1.0
                scale_guess[fp] = ref_target_median / max(m, eps)
            if missing:
                self.update_status(f"ℹ️ {len(missing)} frame(s) had zero/NaN preview medians; using neutral scale for those.")

            # ─────────────────────────────────────────────────────────────────────
            # PHASE 1b: Meridian flips
            # ─────────────────────────────────────────────────────────────────────
            ref_pa = self._extract_pa_deg(ref_hdr)
            self.update_status(f"🧭 Reference PA: {ref_pa:.2f}°" if ref_pa is not None else "🧭 Reference PA: (unknown)")
            QApplication.processEvents()

            # ─────────────────────────────────────────────────────────────────────
            # NEW: build arcsec/px map & choose target = reference scale
            # ─────────────────────────────────────────────────────────────────────
            self.arcsec_per_px = {}  # fp -> (sx, sy)

            # --- helpers placed just above the normalize loop (once) --------------------
            def _scale_from_header_raw(hdr):
                """
                Return (sx, sy) in arcsec/px as the file is stored on disk (no bin correction).
                Priority: CD matrix -> CDELT(+PC) -> pixel size + focal length.
                Pixel-size fallback avoids double-counting bin: PIXSIZE* are unbinned; XPIXSZ/YPIXSZ are typically effective.
                """
                try:
                    # 1) CD matrix (deg/pix)
                    if all(k in hdr for k in ("CD1_1","CD1_2","CD2_1","CD2_2")):
                        cd11 = float(hdr["CD1_1"]); cd12 = float(hdr["CD1_2"])
                        cd21 = float(hdr["CD2_1"]); cd22 = float(hdr["CD2_2"])
                        sx_deg = (cd11**2 + cd12**2) ** 0.5
                        sy_deg = (cd21**2 + cd22**2) ** 0.5
                        return abs(sx_deg) * 3600.0, abs(sy_deg) * 3600.0

                    # 2) CDELT (+ optional PC)
                    if ("CDELT1" in hdr) and ("CDELT2" in hdr):
                        cdelt1 = float(hdr["CDELT1"]); cdelt2 = float(hdr["CDELT2"])
                        pc11 = float(hdr.get("PC1_1", 1.0)); pc12 = float(hdr.get("PC1_2", 0.0))
                        pc21 = float(hdr.get("PC2_1", 0.0)); pc22 = float(hdr.get("PC2_2", 1.0))
                        m11 = cdelt1 * pc11; m12 = cdelt1 * pc12
                        m21 = cdelt2 * pc21; m22 = cdelt2 * pc22
                        sx_deg = (m11**2 + m12**2) ** 0.5
                        sy_deg = (m21**2 + m22**2) ** 0.5
                        return abs(sx_deg) * 3600.0, abs(sy_deg) * 3600.0

                    # 3) Pixel-size + focal length (arcsec/px)
                    fl_mm = float(hdr.get("FOCALLEN", 0.0) or 0.0)
                    if fl_mm <= 0:
                        return (None, None)

                    # Prefer unbinned physical sizes (PIXSIZE*), else use effective (XPIXSZ/YPIXSZ)
                    pxsize1 = hdr.get("PIXSIZE1"); pxsize2 = hdr.get("PIXSIZE2")
                    if (pxsize1 is not None) or (pxsize2 is not None):
                        px_um = float(pxsize1 or pxsize2 or 0.0)        # unbinned
                        py_um = float(pxsize2 or pxsize1 or px_um)
                        xb = int(hdr.get("XBINNING", hdr.get("BINX", 1)) or 1)
                        yb = int(hdr.get("YBINNING", hdr.get("BINY", 1)) or 1)
                        sx = 206.265 * (px_um * xb) / fl_mm            # include bin here
                        sy = 206.265 * (py_um * yb) / fl_mm
                        return sx, sy

                    xpixsz = hdr.get("XPIXSZ"); ypixsz = hdr.get("YPIXSZ")
                    if (xpixsz is not None) or (ypixsz is not None):
                        # Typically already effective (post-binning) pitch
                        px_um = float(xpixsz or ypixsz or 0.0)
                        py_um = float(ypixsz or xpixsz or px_um)
                        sx = 206.265 * px_um / fl_mm                   # do NOT multiply by bin again
                        sy = 206.265 * py_um / fl_mm
                        return sx, sy
                except Exception:
                    pass
                return (None, None)


            def _effective_scale_for_target_bin(raw_sx, raw_sy, xb, yb, target_xbin, target_ybin):
                """
                Given the scale as stored (raw_sx, raw_sy) and the fact that we then resampled
                geometry by (xb/target_xbin, yb/target_ybin) to bin-normalize, return the
                new (effective) scales after that step. Resampling by k makes arcsec/px divide by k.
                """
                if (raw_sx is None) or (raw_sy is None):
                    return (None, None)
                kx = float(xb) / float(target_xbin)
                ky = float(yb) / float(target_ybin)
                eff_sx = float(raw_sx) / max(1e-12, kx)
                eff_sy = float(raw_sy) / max(1e-12, ky)
                return eff_sx, eff_sy


            # Fill map for all measured frames (cheap header read)
            for fp in measured_frames:
                try:
                    hdr0 = fits.getheader(fp, ext=0)
                except Exception:
                    hdr0 = {}
                self.arcsec_per_px[fp] = _scale_from_header_raw(hdr0)

            # Reference (true) pixel scale is our target
            target_sx, target_sy = _scale_from_header_raw(ref_hdr)
            if target_sx and target_sy:
                self.update_status(f"🎯 Target pixel scale = reference: {target_sx:.3f}\"/px × {target_sy:.3f}\"/px")
                QApplication.processEvents()
            else:
                self.update_status("🎯 Target pixel scale unknown (no WCS/pixel size). Will skip scale normalization.")
                target_sx = target_sy = None

            # ─────────────────────────────────────────────────────────────────────
            # PHASE 2: normalize (DEBAYER everything once here)
            # ─────────────────────────────────────────────────────────────────────
            norm_dir = os.path.join(self.stacking_directory, "Normalized_Images")
            os.makedirs(norm_dir, exist_ok=True)

            ocv_prev = None
            try:
                ocv_prev = cv2.getNumThreads()
                cv2.setNumThreads(max(2, min(8, (os.cpu_count() or 8)//2)))
            except Exception:
                pass

            normalized_files = []
            chunks = list(chunk_list(measured_frames, chunk_size))
            total_chunks = len(chunks)

            ncpu = os.cpu_count() or 8
            io_workers = min(8, max(2, ncpu // 2))

            for idx, chunk in enumerate(chunks, 1):
                self.update_status(f"🌀 Normalizing chunk {idx}/{total_chunks} ({len(chunk)} frames)…")
                QApplication.processEvents()

                abe_enabled = bool(self.settings.value("stacking/grad_poly2/enabled", False, type=bool))
                if abe_enabled:
                    mode         = "divide" if self.settings.value("stacking/grad_poly2/mode", "subtract") == "divide" else "subtract"
                    samples      = int(self.settings.value("stacking/grad_poly2/samples", 120, type=int))
                    downsample   = int(self.settings.value("stacking/grad_poly2/downsample", 6, type=int))
                    patch_size   = int(self.settings.value("stacking/grad_poly2/patch_size", 15, type=int))
                    min_strength = float(self.settings.value("stacking/grad_poly2/min_strength", 0.01, type=float))
                    gain_lo      = float(self.settings.value("stacking/grad_poly2/gain_lo", 0.20, type=float))
                    gain_hi      = float(self.settings.value("stacking/grad_poly2/gain_hi", 5.0, type=float))

                scaled_images = []; scaled_paths = []; scaled_hdrs = []

                from concurrent.futures import ThreadPoolExecutor, as_completed
                self.update_status(f"🌍 Loading {len(chunk)} images in parallel for normalization (up to {io_workers} threads)…")
                with ThreadPoolExecutor(max_workers=io_workers) as ex:
                    futs = {ex.submit(self._load_image_any, fp): fp for fp in chunk}
                    for fut in as_completed(futs):
                        fp = futs[fut]
                        try:
                            img, hdr = fut.result()
                            if img is None:
                                self.update_status(f"⚠️ No data for {fp}")
                                continue

                            # make writable float32
                            img = _to_writable_f32(img)

                            bayer = self._hdr_get(hdr, 'BAYERPAT')
                            splitdb = bool(self._hdr_get(hdr, 'SPLITDB', False))
                            if bayer and not splitdb and (img.ndim == 2 or (img.ndim == 3 and img.shape[-1] == 1)):
                                img = self.debayer_image(img, fp, hdr)  # HxWx3
                            else:
                                if img.ndim == 3 and img.shape[-1] == 1:
                                    img = np.squeeze(img, axis=-1)

                            # meridian flip assist
                            if self.auto_rot180 and ref_pa is not None:
                                pa = self._extract_pa_deg(hdr)
                                img, did = self._maybe_rot180(img, pa, ref_pa, self.auto_rot180_tol_deg)
                                if did:
                                    self.update_status(f"↻ 180° rotate (PA Δ≈180°): {os.path.basename(fp)}")
                                    try:
                                        if hasattr(hdr, "__setitem__"):
                                            hdr['ROT180'] = (True, 'Rotated 180° pre-align by SAS')
                                    except Exception:
                                        pass

                            # 1) resample to target bin
                            xb, yb = bin_map.get(fp, (1, 1))
                            sx = float(xb) / float(target_xbin)
                            sy = float(yb) / float(target_ybin)
                            if (abs(sx - 1.0) > 1e-6) or (abs(sy - 1.0) > 1e-6):
                                before = img.shape[:2]
                                img = _resize_to_scale(img, sx, sy)
                                after = img.shape[:2]
                                self.update_status(
                                    f"🔧 Resampled for binning {xb}×{yb} → {target_xbin}×{target_ybin} "
                                    f"size {before[1]}×{before[0]} → {after[1]}×{after[0]}"
                                )

                            # 2) Pixel-scale normalize (arcsec/px), corrected for binning already applied
                            # 2) Pixel-scale normalize (arcsec/px), corrected for the bin step we just did
                            # Read raw scales (as on disk)
                            raw_psx, raw_psy = _scale_from_header_raw(hdr)
                            raw_ref_psx, raw_ref_psy = _scale_from_header_raw(ref_hdr)

                            # Compute effective scales after step (1) bin-normalization resample
                            xb, yb = bin_map.get(fp, (1, 1))
                            ref_xb, ref_yb = bin_map.get(self.reference_frame, (1, 1))
                            eff_psx,     eff_psy     = _effective_scale_for_target_bin(raw_psx,     raw_psy,     xb,     yb,     target_xbin, target_ybin)
                            eff_ref_psx, eff_ref_psy = _effective_scale_for_target_bin(raw_ref_psx, raw_ref_psy, ref_xb, ref_yb, target_xbin, target_ybin)

                            rx = ry = 1.0
                            if (eff_psx and eff_psy and eff_ref_psx and eff_ref_psy):
                                rx = float(eff_psx) / float(eff_ref_psx)
                                ry = float(eff_psy) / float(eff_ref_psy)

                                # Only warp if materially different
                                if (abs(rx - 1.0) > 1e-3) or (abs(ry - 1.0) > 1e-3):
                                    before_hw = img.shape[:2]
                                    img = _resize_to_scale(img, rx, ry)
                                    after_hw = img.shape[:2]
                                    self.update_status(
                                        f"📏 Pixel-scale normalize "
                                        f"{_fmt2(eff_psx)}\"/{_fmt2(eff_psy)}\" → {_fmt2(eff_ref_psx)}\"/{_fmt2(eff_ref_psy)}\" "
                                        f"| size {before_hw[1]}×{before_hw[0]} → {after_hw[1]}×{after_hw[0]}"
                                    )

                                    try:
                                        self.update_status(
                                            "⚖️ Scales (arcsec/px): "
                                            f"raw(frame)=({_fmt2(raw_psx)},{_fmt2(raw_psy)}), "
                                            f"raw(ref)=({_fmt2(raw_ref_psx)},{_fmt2(raw_ref_psy)}) → "
                                            f"eff(frame→tbin)=({_fmt2(eff_psx)},{_fmt2(eff_psy)}), "
                                            f"eff(ref→tbin)=({_fmt2(eff_ref_psx)},{_fmt2(eff_ref_psy)}) ; "
                                            f"resize=(rx={_fmt2(rx)}, ry={_fmt2(ry)})"
                                        )
                                    except Exception:
                                        pass

                            else:
                                self.update_status("⚠️ Pixel-scale normalize skipped (insufficient header scale info).")

                            # normalization scale (unchanged)
                            pm = float(preview_medians.get(fp, 0.0))
                            s = _compute_scale(ref_target_median, pm if pm > 0 else 1.0,
                                            img, refine_stride=8, refine_if_rel_err=0.10)
                            img = _apply_scale_inplace(img, s)

                            if abe_enabled:
                                scaled_images.append(img.astype(np.float32, copy=False))
                                scaled_paths.append(fp)
                                scaled_hdrs.append(hdr)
                            else:
                                # write out normalized FITS
                                base = os.path.basename(fp)
                                if base.endswith("_n.fit"):
                                    base = base.replace("_n.fit", ".fit")
                                if base.lower().endswith(".fits"):
                                    out_name = base[:-5] + "_n.fit"
                                elif base.lower().endswith(".fit"):
                                    out_name = base[:-4] + "_n.fit"
                                else:
                                    out_name = base + "_n.fit"
                                out_path = os.path.join(norm_dir, out_name)

                                try:
                                    if os.path.splitext(fp)[1].lower() in (".fits", ".fit", ".fz"):
                                        orig_header = fits.getheader(fp, ext=0)
                                    else:
                                        orig_header = fits.Header()
                                except Exception:
                                    orig_header = fits.Header()

                                if isinstance(img, np.ndarray) and img.ndim == 3 and img.shape[-1] == 3:
                                    orig_header["DEBAYERED"] = (True, "Color debayered normalized")
                                else:
                                    orig_header["DEBAYERED"] = (False, "Mono normalized")

                                self._orig2norm[os.path.normpath(fp)] = os.path.normpath(out_path)
                                fits.PrimaryHDU(data=img.astype(np.float32), header=orig_header).writeto(out_path, overwrite=True)
                                normalized_files.append(out_path)

                        except Exception as e:
                            self.update_status(f"⚠️ Error normalizing {fp}: {e}")
                        finally:
                            QApplication.processEvents()

                # 2) ABE (unchanged)
                if abe_enabled and scaled_images:
                    self.update_status(
                        f"Gradient removal (ABE Poly²): mode={mode}, samples={samples}, "
                        f"downsample={downsample}, patch={patch_size}, min_strength={min_strength*100:.2f}%, "
                        f"gain_clip=[{gain_lo},{gain_hi}]"
                    )
                    QApplication.processEvents()

                    abe_stack = np.ascontiguousarray(np.stack(scaled_images, axis=0, dtype=np.float32))
                    abe_stack = remove_gradient_stack_abe(
                        abe_stack,
                        mode=mode,
                        num_samples=samples,
                        downsample=downsample,
                        patch_size=patch_size,
                        min_strength=min_strength,
                        gain_clip=(gain_lo, gain_hi),
                        log_fn=(self._ui_log if hasattr(self, "_ui_log") else self.update_status),
                    )

                    for i, fp in enumerate(scaled_paths):
                        img_out = abe_stack[i]; hdr = scaled_hdrs[i]
                        base = os.path.basename(fp)
                        if base.endswith("_n.fit"):
                            base = base.replace("_n.fit", ".fit")
                        if base.lower().endswith(".fits"):
                            out_name = base[:-5] + "_n.fit"
                        elif base.lower().endswith(".fit"):
                            out_name = base[:-4] + "_n.fit"
                        else:
                            out_name = base + "_n.fit"
                        out_path = os.path.join(norm_dir, out_name)

                        try:
                            orig_header = fits.getheader(fp, ext=0)
                        except Exception:
                            orig_header = fits.Header()

                        if isinstance(img_out, np.ndarray) and img_out.ndim == 3 and img_out.shape[-1] == 3:
                            orig_header["DEBAYERED"] = (True, "Color debayered normalized")
                        else:
                            orig_header["DEBAYERED"] = (False, "Mono normalized")

                        self._orig2norm[os.path.normpath(fp)] = os.path.normpath(out_path)
                        fits.PrimaryHDU(data=img_out.astype(np.float32), header=orig_header).writeto(out_path, overwrite=True)
                        normalized_files.append(out_path)

            # restore OpenCV threads
            try:
                if ocv_prev is not None:
                    cv2.setNumThreads(ocv_prev)
            except Exception:
                pass

            # Update self.light_files to *_n.fit
            for group, file_list in self.light_files.items():
                new_list = []
                for old_path in file_list:
                    base = os.path.basename(old_path)
                    if base.endswith("_n.fit"):
                        new_list.append(os.path.join(norm_dir, base))
                    else:
                        if base.lower().endswith(".fits"):
                            n_name = base[:-5] + "_n.fit"
                        elif base.lower().endswith(".fit"):
                            n_name = base[:-4] + "_n.fit"
                        else:
                            n_name = base + "_n.fit"
                        new_list.append(os.path.join(norm_dir, n_name))
                self.light_files[group] = new_list

            self.update_status("✅ Updated self.light_files to use debayered, normalized *_n.fit frames.")

            # Pick normalized reference path to align against
            ref_base = os.path.basename(self.reference_frame)
            if ref_base.endswith("_n.fit"):
                norm_ref_path = os.path.join(norm_dir, ref_base)
            else:
                if ref_base.lower().endswith(".fits"):
                    norm_ref_base = ref_base[:-5] + "_n.fit"
                elif ref_base.lower().endswith(".fit"):
                    norm_ref_base = ref_base[:-4] + "_n.fit"
                else:
                    norm_ref_base = ref_base + "_n.fit"
                norm_ref_path = os.path.join(norm_dir, norm_ref_base)

            # ─────────────────────────────────────────────────────────────────────
            # Start alignment on the normalized files
            # ─────────────────────────────────────────────────────────────────────
            align_dir = os.path.join(self.stacking_directory, "Aligned_Images")
            os.makedirs(align_dir, exist_ok=True)

            passes = self.settings.value("stacking/refinement_passes", 3, type=int)
            shift_tol = self.settings.value("stacking/shift_tolerance", 0.2, type=float)

            self.alignment_thread = StarRegistrationThread(
                norm_ref_path,
                normalized_files,
                align_dir,
                max_refinement_passes=passes,
                shift_tolerance=shift_tol,
                parent_window=self
            )
            try:
                self.alignment_thread.progress_update.disconnect(self.update_status)
            except TypeError:
                pass
            self.alignment_thread.progress_update.connect(
                self.update_status, Qt.ConnectionType.QueuedConnection
            )

            self.alignment_thread.registration_complete.connect(self.on_registration_complete)

            self.align_progress = QProgressDialog("Aligning stars…", None, 0, 0, self)
            self.align_progress.setWindowModality(Qt.WindowModality.WindowModal)
            self.align_progress.setMinimumDuration(0)
            self.align_progress.setCancelButton(None)
            self.align_progress.setWindowTitle("Stellar Alignment")
            self.align_progress.setValue(0)
            self.align_progress.show()

            try:
                self.alignment_thread.progress_step.disconnect(self._on_align_progress)
            except TypeError:
                pass
            self.alignment_thread.progress_step.connect(
                self._on_align_progress, Qt.ConnectionType.QueuedConnection
            )
            try:
                self.alignment_thread.registration_complete.disconnect(self._on_align_done)
            except TypeError:
                pass
            self.alignment_thread.registration_complete.connect(
                self._on_align_done, Qt.ConnectionType.QueuedConnection
            )
            self.alignment_thread.start()

        except Exception as e:
            self._set_registration_busy(False)
            raise

        

    @pyqtSlot(bool, str)
    def _on_align_done(self, success: bool, message: str):
        # Stop any coalesced progress updates
        try:
            if hasattr(self, "_align_prog_timer") and self._align_prog_timer.isActive():
                self._align_prog_timer.stop()
            # Clear pending state (if you added the debounce members)
            if hasattr(self, "_align_prog_in_slot"):
                self._align_prog_in_slot = False
            if hasattr(self, "_align_prog_pending"):
                self._align_prog_pending = None
        except Exception:
            pass

        # Close/reset the progress dialog without triggering more signals
        dlg = getattr(self, "align_progress", None)
        if dlg is not None:
            try:
                # If total is known this forces the bar to 100% (purely cosmetic)
                if dlg.maximum() > 0:
                    dlg.setValue(dlg.maximum())
            except Exception:
                pass
            try:
                # If AutoClose/AutoReset is set, this both resets and closes
                dlg.reset()
            except Exception:
                try:
                    dlg.close()
                except Exception:
                    pass
            # Remove the attribute to avoid stale references
            try:
                del self.align_progress
            except Exception:
                try:
                    self.align_progress = None
                except Exception:
                    pass

        # Re-enable UI / busy state
        try:
            self._set_registration_busy(False)
        except Exception:
            pass

        # Update any local label directly (avoid update_status to prevent feedback)
        try:
            if hasattr(self, "progress_label"):
                color = "green" if success else "red"
                self.progress_label.setText(f"Status: {message}")
                self.progress_label.setStyleSheet(f"color: {color}; font-weight: bold;")
        except Exception:
            pass

    def save_alignment_matrices_sasd(self, transforms_dict):
        out_path = os.path.join(self.stacking_directory, "alignment_transforms.sasd")
        try:
            with open(out_path, "w") as f:
                for norm_path, matrix in transforms_dict.items():
                    # Use the original normalized input path (e.g., *_n.fit)
                    orig_path = os.path.normpath(norm_path)

                    a, b, tx = matrix[0]
                    c, d, ty = matrix[1]

                    f.write(f"FILE: {orig_path}\n")
                    f.write("MATRIX:\n")
                    f.write(f"{a:.4f}, {b:.4f}, {tx:.4f}\n")
                    f.write(f"{c:.4f}, {d:.4f}, {ty:.4f}\n")
                    f.write("\n")  # blank line
            self.update_status(f"✅ Transform file saved as {os.path.basename(out_path)}")
        except Exception as e:
            self.update_status(f"⚠️ Failed to save transform file: {e}")



    def load_alignment_matrices_custom(self, file_path):
        transforms = {}
        with open(file_path, "r") as f:
            content = f.read()

        blocks = re.split(r"\n\s*\n", content.strip())
        for block in blocks:
            lines = [ln.strip() for ln in block.splitlines() if ln.strip() and not ln.strip().startswith("#")]
            if not lines or not lines[0].startswith("FILE:"):
                continue
            curr_file = os.path.normpath(lines[0].replace("FILE:", "").strip())
            # find "MATRIX:" line
            try:
                m_idx = next(i for i,l in enumerate(lines) if l.startswith("MATRIX:"))
            except StopIteration:
                continue

            # Try parse 2×3 first, else 3×3
            try:
                r0 = [float(x) for x in lines[m_idx+1].split(",")]
                r1 = [float(x) for x in lines[m_idx+2].split(",")]
                if len(r0) == 3 and len(r1) == 3:
                    transforms[curr_file] = np.array([[r0[0], r0[1], r0[2]],
                                                    [r1[0], r1[1], r1[2]]], dtype=np.float32)
                    continue
                # 3×3
                r2 = [float(x) for x in lines[m_idx+3].split(",")]
                if len(r0) == len(r1) == len(r2) == 3:
                    H = np.array([r0, r1, r2], dtype=np.float32)
                    transforms[curr_file] = H
            except Exception:
                # skip malformed block
                continue
        return transforms


    def _make_star_trail(self):
        # 1) collect all your calibrated light frames
        all_files = [f for flist in self.light_files.values() for f in flist]
        n_frames = len(all_files)
        if not all_files:
            self.update_status("⚠️ No calibrated lights available for star trails.")
            return

        # 2) load every frame (once), compute its median, and remember its header
        frames: list[tuple[np.ndarray, fits.Header]] = []
        medians: list[float] = []

        for fn in all_files:
            img, hdr, _, _ = load_image(fn)
            if img is None:
                self.update_status(f"⚠️ Failed to load {os.path.basename(fn)}; skipping")
                QApplication.processEvents()
                continue

            arr = img.astype(np.float32)
            medians.append(float(np.median(arr)))
            frames.append((arr, hdr))

        if not frames:
            self.update_status("⚠️ No valid frames to compute reference median; aborting star-trail.")
            return

        # reference median is the median of per-frame medians
        ref_median = float(np.median(medians))

        # grab the header from the first valid frame, strip out extra NAXIS keywords
        first_hdr = frames[0][1]
        if first_hdr is not None:
            hdr_to_use = first_hdr.copy()
            for key in list(hdr_to_use):
                if key.startswith("NAXIS") and key not in ("NAXIS", "NAXIS1", "NAXIS2"):
                    hdr_to_use.pop(key, None)
        else:
            hdr_to_use = None

        # 3) normalize each frame and write to a temp dir
        with tempfile.TemporaryDirectory(prefix="startrail_norm_") as norm_dir:
            normalized_paths = []
            for idx, (arr, hdr) in enumerate(frames, start=1):
                self.update_status(f"🔄 Normalizing frame {idx}/{len(frames)}")
                QApplication.processEvents()

                # guard against divide-by-zero
                m = float(np.median(arr))
                scale = ref_median / (m + 1e-12)
                img_norm = arr * scale

                stem = Path(all_files[idx-1]).stem
                out_path = os.path.join(norm_dir, f"{stem}_st.fit")
                fits.PrimaryHDU(data=img_norm, header=hdr).writeto(out_path, overwrite=True)
                normalized_paths.append(out_path)

            # 4) stack and do max-value projection
            self.update_status(f"📊 Stacking {len(normalized_paths)} frames")
            QApplication.processEvents()
            stack = np.stack([fits.getdata(p).astype(np.float32) for p in normalized_paths], axis=0)
            trail_img, _ = max_value_stack(stack)

            # 5) stretch final image and prompt user for save location & format
            trail_img = trail_img.astype(np.float32)
            # normalize to [0–1] for our save helper
            trail_norm = trail_img / (trail_img.max() + 1e-12)

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_name = self._safe_component(f"StarTrail_{n_frames:03d}frames_{ts}")
            filters = "TIFF (*.tif);;PNG (*.png);;JPEG (*.jpg *.jpeg);;FITS (*.fits);;XISF (*.xisf)"
            path, chosen_filter = QFileDialog.getSaveFileName(
                self, "Save Star-Trail Image",
                os.path.join(self.stacking_directory, default_name),
                "TIFF (*.tif);;PNG (*.png);;JPEG (*.jpg *.jpeg);;FITS (*.fits);;XISF (*.xisf)"
            )
            if not path:
                self.update_status("✖ Star-trail save cancelled.")
                return

            # figure out extension
            ext = os.path.splitext(path)[1].lower().lstrip('.')
            if not ext:
                ext = chosen_filter.split('(')[1].split(')')[0].lstrip('*.').lower()
                path += f".{ext}"

            # if user picked FITS, supply the first frame’s header; else None
            use_hdr = hdr_to_use if ext in ('fits', 'fit') else None

            # 16-bit everywhere
            save_image(
                img_array=trail_norm,
                filename=path,
                original_format=ext,
                bit_depth="16-bit",
                original_header=use_hdr,
                is_mono=False
            )

        # once we exit the with-block, all the _st.fit files are deleted
        self.update_status(f"✅ Star‐Trail image written to {path}")
        return


    def _apply_autocrop(self, arr, file_list, header, scale=1.0, rect_override=None):
        """
        If rect_override is provided, use it; else compute per-file_list.
        """
        try:
            enabled = self.autocrop_cb.isChecked()
            pct = float(self.autocrop_pct.value())
        except Exception:
            enabled = self.settings.value("stacking/autocrop_enabled", False, type=bool)
            pct = float(self.settings.value("stacking/autocrop_pct", 95.0, type=float))

        if not enabled or not file_list:
            return arr, header

        rect = rect_override
        if rect is None:
            transforms_path = os.path.join(self.stacking_directory, "alignment_transforms.sasd")
            rect = self._compute_autocrop_rect(file_list, transforms_path, pct)

        if not rect:
            self.update_status("✂️ Auto-crop: no common area found; skipping.")
            return arr, header

        x0, y0, x1, y1 = rect
        if scale != 1.0:
            # scale rect to drizzle resolution
            x0 = int(math.floor(x0 * scale))
            y0 = int(math.floor(y0 * scale))
            x1 = int(math.ceil (x1 * scale))
            y1 = int(math.ceil (y1 * scale))

        # Clamp to image bounds
        H, W = arr.shape[:2]
        x0 = max(0, min(W, x0)); x1 = max(x0, min(W, x1))
        y0 = max(0, min(H, y0)); y1 = max(y0, min(H, y1))

        # --- Crop while preserving channels ---
        if arr.ndim == 2:
            arr = arr[y0:y1, x0:x1]
        else:
            arr = arr[y0:y1, x0:x1, :]
            # If this is actually mono stored as (H,W,1), squeeze back to (H,W)
            if arr.shape[-1] == 1:
                arr = arr[..., 0]

        # Update header dims (+ shift CRPIX if present)
        if header is None:
            header = fits.Header()

        # NAXIS / sizes consistent with the new array
        if arr.ndim == 2:
            header["NAXIS"]  = 2
            header["NAXIS1"] = arr.shape[1]
            header["NAXIS2"] = arr.shape[0]
            # Remove any stale NAXIS3
            if "NAXIS3" in header:
                del header["NAXIS3"]
        else:
            header["NAXIS"]  = 3
            header["NAXIS1"] = arr.shape[1]
            header["NAXIS2"] = arr.shape[0]
            header["NAXIS3"] = arr.shape[2]

        if "CRPIX1" in header:
            header["CRPIX1"] = float(header["CRPIX1"]) - x0
        if "CRPIX2" in header:
            header["CRPIX2"] = float(header["CRPIX2"]) - y0

        self.update_status(f"✂️ Auto-cropped to [{x0}:{x1}]×[{y0}:{y1}] (scale {scale}×)")
        return arr, header

    def _dither_phase_fill(self, matrices: dict[str, np.ndarray], bins=8) -> float:
        hist = np.zeros((bins, bins), dtype=np.int32)
        for M in matrices.values():
            M = np.asarray(M)
            if M.shape == (2,3):
                tx, ty = float(M[0,2]), float(M[1,2])
            elif M.shape == (3,3):
                # translation at origin for homography
                tx, ty = float(M[0,2]), float(M[1,2])
            else:
                continue
            fx = (tx - math.floor(tx)) % 1.0
            fy = (ty - math.floor(ty)) % 1.0
            ix = min(int(fx * bins), bins - 1)
            iy = min(int(fy * bins), bins - 1)
            hist[iy, ix] += 1
        return float(np.count_nonzero(hist)) / float(hist.size)

    def on_registration_complete(self, success, msg):
       
        self.update_status(msg)
        if not success:
            self._set_registration_busy(False)
            return

        alignment_thread = self.alignment_thread
        if alignment_thread is None:
            self.update_status("⚠️ Error: No alignment data available.")
            self._set_registration_busy(False) 
            return

        # ----------------------------
        # Gather results from the thread
        # ----------------------------
        all_transforms = dict(alignment_thread.alignment_matrices)  # {orig_norm_path -> 2x3 or None}
        keys = list(all_transforms.keys())

        # Build a per-file shift map (last pass), defaulting to 0 when missing
        shift_map = {}
        if alignment_thread.transform_deltas and alignment_thread.transform_deltas[-1]:
            last = alignment_thread.transform_deltas[-1]
            for i, k in enumerate(keys):
                if i < len(last):
                    shift_map[k] = float(last[i])
                else:
                    shift_map[k] = 0.0
        else:
            shift_map = {k: 0.0 for k in keys}

        # Fast mode if only 1 pass was requested
        fast_mode = (getattr(alignment_thread, "max_refinement_passes", 3) <= 1)
        # Threshold is only used in normal mode
        accept_thresh = float(self.settings.value("stacking/accept_shift_px", 2.0, type=float))

        def _accept(k: str) -> bool:
            """Accept criteria for a frame."""
            if all_transforms.get(k) is None:
                # real failure (e.g., astroalign couldn't find a transform)
                return False
            if fast_mode:
                # In fast mode we keep everything that didn't fail
                return True
            # Normal (multi-pass) behavior: keep small last-pass shifts
            return shift_map.get(k, 0.0) <= accept_thresh

        accepted = [k for k in keys if _accept(k)]
        rejected = [k for k in keys if not _accept(k)]

        # ----------------------------
        # Persist numeric transforms we accepted (for drizzle, etc.)
        # ----------------------------
        valid_matrices = {k: all_transforms[k] for k in accepted}
        self.valid_matrices = {
            os.path.normpath(k): np.asarray(v, dtype=np.float32)
            for k, v in valid_matrices.items() if v is not None
        }

        # ✅ Write SASD v2 using model-aware transforms captured by the thread
        try:
            sasd_out = os.path.join(self.stacking_directory, "alignment_transforms.sasd")
            # pull over the per-file model-aware xforms and reference info
            self.drizzle_xforms = dict(getattr(alignment_thread, "drizzle_xforms", {}))
            self.ref_shape_for_drizzle = tuple(getattr(alignment_thread, "reference_image_2d", np.zeros((1,1), np.float32)).shape[:2])
            self.ref_path_for_drizzle  = alignment_thread.reference if isinstance(alignment_thread.reference, str) else "__ACTIVE_VIEW__"
            self._save_alignment_transforms_sasd_v2(
                out_path=sasd_out,
                ref_shape=self.ref_shape_for_drizzle,
                ref_path=self.ref_path_for_drizzle,
                drizzle_xforms=self.drizzle_xforms,
                fallback_affine=self.valid_matrices,  # in case a few files missed model-aware
            )
            self.update_status("✅ Transform file saved as alignment_transforms.sasd (v2)")
        except Exception as e:
            self.update_status(f"⚠️ Failed to write SASD v2 ({e}); writing affine-only fallback.")
            self.save_alignment_matrices_sasd(valid_matrices)  # old writer as last resort


        try:
            if self._comet_seed and self.reference_frame and getattr(self, "valid_matrices", None):
                seed_orig = os.path.normpath(self._comet_seed["path"])
                seed_xy   = self._comet_seed["xy"]

                # find the normalized counterpart of the original seed frame
                seed_norm = self._orig2norm.get(seed_orig)
                if not seed_norm:
                    # if the seed was picked on an already-normalized path, try that as-is
                    if seed_orig in self.valid_matrices:
                        seed_norm = seed_orig

                M = self.valid_matrices.get(os.path.normpath(seed_norm)) if seed_norm else None
                if M is not None:
                    x, y = seed_xy
                    X = float(M[0,0]*x + M[0,1]*y + M[0,2])
                    Y = float(M[1,0]*x + M[1,1]*y + M[1,2])
                    self._comet_ref_xy = (X, Y)
                    self.update_status(f"🌠 Comet anchor in reference frame: ({X:.1f}, {Y:.1f})")
                else:
                    self.update_status("ℹ️ Could not resolve comet seed to reference (no matrix for that frame).")
        except Exception as e:
            self.update_status(f"⚠️ Comet seed resolve failed: {e}")

        # ----------------------------
        # Build mapping from normalized -> aligned paths
        # Use the *actual* final paths produced by the thread.
        # ----------------------------
        final_map = alignment_thread.file_key_to_current_path  # {orig_norm_path -> final_aligned_path}
        self.valid_transforms = {
            os.path.normpath(k): os.path.normpath(final_map[k])
            for k in accepted
            if k in final_map and os.path.exists(final_map[k])
        }

        self.matrix_by_aligned = {}
        self.orig_by_aligned   = {}
        for norm_path, aligned_path in self.valid_transforms.items():
            M = self.valid_matrices.get(norm_path)
            if M is not None:
                self.matrix_by_aligned[os.path.normpath(aligned_path)] = M
            self.orig_by_aligned[os.path.normpath(aligned_path)] = os.path.normpath(norm_path)
        # For drizzle: keep model-aware xforms and reference geometry
        self.drizzle_xforms = dict(getattr(alignment_thread, "drizzle_xforms", {}))
        self.ref_shape_for_drizzle = tuple(getattr(alignment_thread, "reference_image_2d", np.zeros((1,1), np.float32)).shape[:2])
        self.ref_path_for_drizzle  = alignment_thread.reference if isinstance(alignment_thread.reference, str) else "__ACTIVE_VIEW__"

        # finalize alignment phase
        self.alignment_thread = None

        # Status
        prefix = "⚡ Fast mode: " if fast_mode else ""
        self.update_status(f"{prefix}Alignment summary: {len(accepted)} succeeded, {len(rejected)} rejected.")
        QApplication.processEvents()
        if (not fast_mode) and rejected:
            self.update_status(f"🚨 Rejected {len(rejected)} frame(s) due to shift > {accept_thresh}px.")
            for rf in rejected:
                self.update_status(f"  ❌ {os.path.basename(rf)}")

        if not self.valid_transforms:
            self.update_status("⚠️ No frames to stack; aborting.")
            self._set_registration_busy(False)
            return

        # ----------------------------
        # Build aligned file groups (unchanged)
        # ----------------------------
        filtered_light_files = {}
        for group, file_list in self.light_files.items():
            filtered = [f for f in file_list if os.path.normpath(f) in self.valid_transforms]
            filtered_light_files[group] = filtered
            self.update_status(f"Group '{group}' has {len(filtered)} file(s) after filtering.")
            QApplication.processEvents()

        aligned_light_files = {}
        for group, file_list in filtered_light_files.items():
            new_list = []
            for f in file_list:
                normed = os.path.normpath(f)
                aligned = self.valid_transforms.get(normed)
                if aligned and os.path.exists(aligned):
                    new_list.append(aligned)
                else:
                    self.update_status(f"DEBUG: File '{aligned}' does not exist on disk.")
            aligned_light_files[group] = new_list

        def _start_after_align_worker(aligned_light_files: dict[str, list[str]]):
            # ----------------------------
            # Snapshot UI-dependent settings (your existing code)
            # ----------------------------
            drizzle_dict = self.gather_drizzle_settings_from_tree()
            try:
                autocrop_enabled = self.autocrop_cb.isChecked()
                autocrop_pct = float(self.autocrop_pct.value())
            except Exception:
                autocrop_enabled = self.settings.value("stacking/autocrop_enabled", False, type=bool)
                autocrop_pct = float(self.settings.value("stacking/autocrop_pct", 95.0, type=float))

            cfa_effective = bool(
                self._cfa_for_this_run
                if getattr(self, "_cfa_for_this_run", None) is not None
                else (getattr(self, "cfa_drizzle_cb", None) and self.cfa_drizzle_cb.isChecked())
            )
            if cfa_effective and getattr(self, "valid_matrices", None):
                fill = self._dither_phase_fill(self.valid_matrices, bins=8)
                self.update_status(f"🔎 CFA drizzle sub-pixel phase fill (8×8): {fill*100:.1f}%")
                if fill < 0.65:
                    self.update_status("💡 For best results with CFA drizzle, aim for >65% fill.")
                    self.update_status("   With <~40–55% fill, expect visible patching even with many frames)")
            QApplication.processEvents()

            # ----------------------------
            # Kick off post-align worker (unchanged body)
            # ----------------------------
            self.post_thread = QThread(self)
            self.post_worker = AfterAlignWorker(
                self,
                light_files=aligned_light_files,
                frame_weights=dict(self.frame_weights),
                transforms_dict=dict(self.valid_transforms),
                drizzle_dict=drizzle_dict,
                autocrop_enabled=autocrop_enabled,
                autocrop_pct=autocrop_pct,
                ui_owner=self
            )
            self.post_worker.ui_owner = self
            self.post_worker.need_comet_review.connect(self.on_need_comet_review)
            self.post_worker.progress.connect(self._on_post_status)
            self.post_worker.finished.connect(self._on_post_pipeline_finished)
            self.post_worker.moveToThread(self.post_thread)
            self.post_thread.started.connect(self.post_worker.run)
            self.post_thread.start()

            self.post_progress = QProgressDialog("Stacking & drizzle (if enabled)…", None, 0, 0, self)
            self.post_progress.setWindowModality(Qt.WindowModality.WindowModal)
            self.post_progress.setCancelButton(None)
            self.post_progress.setMinimumDuration(0)
            self.post_progress.setWindowTitle("Post-Alignment")
            self.post_progress.show()

            self._set_registration_busy(False)

        try:
            autocrop_enabled_ui = self.autocrop_cb.isChecked()
            autocrop_pct_ui = float(self.autocrop_pct.value())
        except Exception:
            autocrop_enabled_ui = self.settings.value("stacking/autocrop_enabled", False, type=bool)
            autocrop_pct_ui = float(self.settings.value("stacking/autocrop_pct", 95.0, type=float))

        # Build a single global rect from all aligned frames (registered paths)
        _mf_global_rect = None
        if autocrop_enabled_ui:
            pd = QProgressDialog("Calculating autocrop bounding box…", None, 0, 0, self)
            pd.setWindowTitle("Auto Crop")
            pd.setWindowModality(Qt.WindowModality.WindowModal)
            pd.setCancelButton(None)
            pd.setMinimumDuration(0)
            pd.setRange(0, 0)
            pd.show()
            QApplication.processEvents()

            try:
                self.update_status("✂️ (MF) Auto Crop: using transform footprints…")
                QApplication.processEvents()

                # Prefer model-aware (drizzle) transforms, else affine fallback
                xforms = (dict(getattr(self, "drizzle_xforms", {}))
                        if getattr(self, "drizzle_xforms", None)
                        else dict(getattr(self, "valid_matrices", {})))
                ref_hw = tuple(getattr(self, "ref_shape_for_drizzle", ()))  # (H,W)

                if xforms and len(ref_hw) == 2:
                    _mf_global_rect = self._rect_from_transforms_fast(
                        xforms,
                        src_hw=ref_hw,
                        coverage_pct=float(autocrop_pct_ui),
                        allow_homography=True,
                        min_side=16
                    )
                    if _mf_global_rect:
                        x0,y0,x1,y1 = _mf_global_rect
                        self.update_status(
                            f"✂️ (MF) Transform crop → [{x0}:{x1}]×[{y0}:{y1}] "
                            f"({x1-x0}×{y1-y0})"
                        )
                    else:
                        self.update_status("✂️ (MF) Transform crop yielded no valid rect; falling back to mask-based method…")
                else:
                    self.update_status("✂️ (MF) No transforms/geometry available; falling back to mask-based method…")

                # Fallback to existing (mask-based) method if needed
                if _mf_global_rect is None:
                    _mf_global_rect = self._compute_common_autocrop_rect(
                        aligned_light_files,
                        autocrop_pct_ui,
                        status_cb=self.update_status
                    )
                    if _mf_global_rect:
                        x0,y0,x1,y1 = _mf_global_rect
                        self.update_status(
                            f"✂️ (MF) Mask-based crop → [{x0}:{x1}]×[{y0}:{y1}] "
                            f"({x1-x0}×{y1-y0})"
                        )
                    else:
                        self.update_status("✂️ (MF) Auto-crop disabled (no common region).")

                QApplication.processEvents()
            except Exception as e:
                self.update_status(f"⚠️ (MF) Global crop failed: {e}")
                _mf_global_rect = None
            finally:
                try:
                    pd.reset()
                    pd.deleteLater()
                except Exception:
                    pass

        self._mf_autocrop_rect   = _mf_global_rect
        self._mf_autocrop_enabled = bool(autocrop_enabled_ui)
        self._mf_autocrop_pct     = float(autocrop_pct_ui)


        mf_enabled = self.settings.value("stacking/mfdeconv/enabled", False, type=bool)

        if mf_enabled:
            self.update_status("🧪 Multi-frame PSF-aware deconvolution path enabled.")

            mf_groups = [(g, lst) for g, lst in aligned_light_files.items() if lst]
            if not mf_groups:
                self.update_status("⚠️ No aligned frames available for MF deconvolution.")
            else:
                self._mf_pd = QProgressDialog("Multi-frame deconvolving…", "Cancel", 0, len(mf_groups), self)
                # self._mf_pd.setWindowModality(Qt.WindowModality.ApplicationModal)
                self._mf_pd.setMinimumDuration(0)
                self._mf_pd.setWindowTitle("MF Deconvolution")
                self._mf_pd.setValue(0)
                self._mf_pd.show()

                if getattr(self, "_mf_pd", None):
                    self._mf_pd.setLabelText("Preparing MF deconvolution…")
                    self._mf_pd.setMinimumWidth(520)

                self._mf_total_groups = len(mf_groups)
                self._mf_groups_done = 0
                # progress range = groups * 1000 (each group gets 0..1000)
                self._mf_pd.setRange(0, self._mf_total_groups * 1000)
                self._mf_pd.setValue(0)
                self._mf_queue = list(mf_groups)
                self._mf_results = {}
                self._mf_cancelled = False
                self._mf_thread = None
                self._mf_worker = None

                def _cancel_all():
                    self._mf_cancelled = True
                self._mf_pd.canceled.connect(_cancel_all, Qt.ConnectionType.QueuedConnection)

                def _finish_mf_phase_and_exit():
                    """Tear down MF UI/threads and either continue or exit."""
                    if getattr(self, "_mf_pd", None):
                        try:
                            self._mf_pd.reset()
                            self._mf_pd.deleteLater()
                        except Exception:
                            pass
                        self._mf_pd = None
                    try:
                        if self._mf_thread:
                            self._mf_thread.quit()
                            self._mf_thread.wait()
                    except Exception:
                        pass
                    self._mf_thread = None
                    self._mf_worker = None

                    run_after = self.settings.value("stacking/mfdeconv/after_mf_run_integration", False, type=bool)
                    if run_after:
                        _start_after_align_worker(aligned_light_files)
                    else:
                        self.update_status("✅ MFDeconv complete for all groups. Skipping normal integration as requested.")
                        self._set_registration_busy(False)

                def _start_next_mf_job():
                    # end of queue or canceled → finish
                    if self._mf_cancelled or not self._mf_queue:
                        _finish_mf_phase_and_exit()
                        return

                    group_key, frames = self._mf_queue.pop(0)

                    out_dir = os.path.join(self.stacking_directory, "Masters")
                    os.makedirs(out_dir, exist_ok=True)
                    out_path = os.path.join(out_dir, f"MasterLight_{group_key}_MFDeconv.fit")

                    # ── read config ─────────────────────────────────────────
                    iters = self.settings.value("stacking/mfdeconv/iters", 20, type=int)
                    min_iters = self.settings.value("stacking/mfdeconv/min_iters", 3, type=int)
                    kappa = self.settings.value("stacking/mfdeconv/kappa", 2.0, type=float)
                    mode  = self.mf_color_combo.currentText()
                    Huber = self.settings.value("stacking/mfdeconv/Huber_delta", 0.0, type=float)
                    seed_mode_cfg = str(self.settings.value("stacking/mfdeconv/seed_mode", "robust"))
                    use_star_masks    = self.mf_use_star_mask_cb.isChecked()
                    use_variance_maps = self.mf_use_noise_map_cb.isChecked()
                    rho               = self.mf_rho_combo.currentText()
                    save_intermediate = self.mf_save_intermediate_cb.isChecked()

                    sr_enabled_ui = self.mf_sr_cb.isChecked()
                    sr_factor_ui  = getattr(self, "mf_sr_factor_spin", None)
                    sr_factor_val = sr_factor_ui.value() if sr_factor_ui is not None else self.settings.value("stacking/mfdeconv/sr_factor", 2, type=int)
                    super_res_factor = int(sr_factor_val) if sr_enabled_ui else 1

                    star_mask_cfg = {
                        "thresh_sigma":  self.settings.value("stacking/mfdeconv/star_mask/thresh_sigma",  _SM_DEF_THRESH, type=float),
                        "grow_px":       self.settings.value("stacking/mfdeconv/star_mask/grow_px",       _SM_DEF_GROW, type=int),
                        "soft_sigma":    self.settings.value("stacking/mfdeconv/star_mask/soft_sigma",    _SM_DEF_SOFT, type=float),
                        "max_radius_px": self.settings.value("stacking/mfdeconv/star_mask/max_radius_px", _SM_DEF_RMAX, type=int),
                        "max_objs":      self.settings.value("stacking/mfdeconv/star_mask/max_objs",      _SM_DEF_MAXOBJS, type=int),
                        "keep_floor":    self.settings.value("stacking/mfdeconv/star_mask/keep_floor",    _SM_DEF_KEEPF, type=float),
                        "ellipse_scale": self.settings.value("stacking/mfdeconv/star_mask/ellipse_scale", _SM_DEF_ES, type=float),
                    }
                    varmap_cfg = {
                        "sample_stride": self.settings.value("stacking/mfdeconv/varmap/sample_stride", _VM_DEF_STRIDE, type=int),
                        "smooth_sigma":  self.settings.value("stacking/mfdeconv/varmap/smooth_sigma", 1.0, type=float),
                        "floor":         self.settings.value("stacking/mfdeconv/varmap/floor",        1e-8, type=float),
                    }

                    self._mf_thread = QThread(self)
                    star_mask_ref = self.reference_frame if use_star_masks else None

                    # ── choose engine plainly (Normal / cuDNN-free / High Octane) ─────────────
                    # Expect a setting saved by your radio buttons: "normal" | "cudnn" | "sport"
                    engine = str(self.settings.value("stacking/mfdeconv/engine", "normal")).lower()

                    try:
                        if engine == "cudnn":
                            from pro.mfdeconvcudnn import MultiFrameDeconvWorkercuDNN as MFCls
                            eng_name = "Normal (cuDNN-free)"
                        elif engine == "sport":  # High Octane let 'er rip
                            from pro.mfdeconvsport import MultiFrameDeconvWorkerSport as MFCls
                            eng_name = "High Octane"
                        else:
                            from pro.mfdeconv import MultiFrameDeconvWorker as MFCls
                            eng_name = "Normal"
                    except Exception as e:
                        # if an import fails, fall back to the safe Normal path
                        self.update_status(f"⚠️ MFDeconv engine import failed ({e}); falling back to Normal.")
                        from pro.mfdeconv import MultiFrameDeconvWorker as MFCls
                        eng_name = "Normal (fallback)"

                    self.update_status(f"⚙️ MFDeconv engine: {eng_name}")

                    # ── build worker exactly the same in all modes ────────────────────────────
                    self._mf_worker = MFCls(
                        parent=None,
                        aligned_paths=frames,
                        output_path=out_path,
                        iters=iters,
                        kappa=kappa,
                        color_mode=mode,
                        huber_delta=Huber,
                        min_iters=min_iters,
                        use_star_masks=use_star_masks,
                        use_variance_maps=use_variance_maps,
                        rho=rho,
                        star_mask_cfg=star_mask_cfg,
                        varmap_cfg=varmap_cfg,
                        save_intermediate=save_intermediate,
                        super_res_factor=super_res_factor,
                        star_mask_ref_path=star_mask_ref,
                        seed_mode=seed_mode_cfg,
                    )

                    # ── standard Qt wiring (no gymnastics) ───────────────────────────────────
                    self._mf_worker.moveToThread(self._mf_thread)
                    self._mf_worker.progress.connect(self._on_mf_progress, Qt.ConnectionType.QueuedConnection)

                    # when the worker says finished, log & cleanup; the *thread* quitting will trigger starting the next job
                    def _job_finished(ok: bool, message: str, out: str):
                        if getattr(self, "_mf_pd", None):
                            self._mf_pd.setLabelText(f"{'✅' if ok else '❌'} {group_key}: {message}")
                        if ok and out:
                            self._mf_results[group_key] = out
                            if getattr(self, "_mf_autocrop_enabled", False) and getattr(self, "_mf_autocrop_rect", None):
                                try:
                                    from astropy.io import fits
                                    with fits.open(out, memmap=False) as hdul:
                                        img = hdul[0].data; hdr = hdul[0].header
                                    rect = tuple(map(int, self._mf_autocrop_rect))
                                    sr_enabled_ui = self.mf_sr_cb.isChecked()
                                    sr_factor_ui  = getattr(self, "mf_sr_factor_spin", None)
                                    sr_factor_val = sr_factor_ui.value() if sr_factor_ui is not None else self.settings.value("stacking/mfdeconv/sr_factor", 2, type=int)
                                    sr_factor = int(sr_factor_val) if sr_enabled_ui else 1
                                    if sr_factor > 1:
                                        x1,y1,x2,y2 = rect
                                        rect = (x1*sr_factor, y1*sr_factor, x2*sr_factor, y2*sr_factor)
                                    x1,y1,x2,y2 = rect
                                    if img.ndim == 2:
                                        crop = img[y1:y2, x1:x2]
                                    elif img.ndim == 3:
                                        crop = (img[:, y1:y2, x1:x2] if img.shape[0] in (1,3)
                                                else img[y1:y2, x1:x2, :])
                                    out_crop = out.replace(".fit", "_autocrop.fit").replace(".fits", "_autocrop.fits")
                                    fits.PrimaryHDU(data=crop.astype(np.float32, copy=False), header=hdr).writeto(out_crop, overwrite=True)
                                    self.update_status(f"✂️ (MF) Saved auto-cropped copy → {out_crop}")
                                except Exception as e:
                                    self.update_status(f"⚠️ (MF) Auto-crop of output failed: {e}")

                        # advance progress segment
                        if getattr(self, "_mf_pd", None):
                            self._mf_groups_done = min(self._mf_groups_done + 1, self._mf_total_groups)
                            self._mf_pd.setValue(self._mf_groups_done * 1000)

                    self._mf_worker.finished.connect(_job_finished, Qt.ConnectionType.QueuedConnection)

                    # thread start/stop
                    self._mf_thread.started.connect(self._mf_worker.run, Qt.ConnectionType.QueuedConnection)
                    self._mf_worker.finished.connect(self._mf_thread.quit, Qt.ConnectionType.QueuedConnection)
                    self._mf_thread.finished.connect(self._mf_worker.deleteLater, Qt.ConnectionType.QueuedConnection)
                    self._mf_thread.finished.connect(self._mf_thread.deleteLater, Qt.ConnectionType.QueuedConnection)

                    # when the thread is fully down, kick the next job
                    def _next_after_thread():
                        try:
                            self._mf_thread.finished.disconnect(_next_after_thread)
                        except Exception:
                            pass
                        QTimer.singleShot(0, _start_next_mf_job)

                    self._mf_thread.finished.connect(_next_after_thread, Qt.ConnectionType.QueuedConnection)

                    # go
                    self._mf_thread.start()
                    if getattr(self, "_mf_pd", None):
                        self._mf_pd.setLabelText(f"Deconvolving '{group_key}' ({len(frames)} frames)…")


                # Kick off the first job (queue-driven)
                QTimer.singleShot(0, _start_next_mf_job)

                # Defer rest of pipeline; MF block will decide at MF completion.
                self._set_registration_busy(False)
                return



        # ----------------------------
        # Snapshot UI-dependent settings
        # ----------------------------
        drizzle_dict = self.gather_drizzle_settings_from_tree()
        try:
            autocrop_enabled = self.autocrop_cb.isChecked()
            autocrop_pct = float(self.autocrop_pct.value())
        except Exception:
            autocrop_enabled = self.settings.value("stacking/autocrop_enabled", False, type=bool)
            autocrop_pct = float(self.settings.value("stacking/autocrop_pct", 95.0, type=float))

        # Only report fill % if CFA mapping is actually in use for this run
        cfa_effective = bool(
            self._cfa_for_this_run
            if getattr(self, "_cfa_for_this_run", None) is not None
            else (getattr(self, "cfa_drizzle_cb", None) and self.cfa_drizzle_cb.isChecked())
        )
        print("CFA effective for this run:", cfa_effective)

        if cfa_effective and getattr(self, "valid_matrices", None):
            fill = self._dither_phase_fill(self.valid_matrices, bins=8)
            self.update_status(f"🔎 CFA drizzle sub-pixel phase fill (8×8): {fill*100:.1f}%")
            if fill < 0.65:
                self.update_status("💡 For best results with CFA drizzle, aim for >65% fill.")
                self.update_status("   With <~40–55% fill, expect visible patching even with many frames.")
        QApplication.processEvents()

        # ----------------------------
        # Kick off post-align worker (unchanged)
        # ----------------------------
        self.post_thread = QThread(self)
        self.post_worker = AfterAlignWorker(
            self,                                   # parent QObject
            light_files=aligned_light_files,
            frame_weights=dict(self.frame_weights),
            transforms_dict=dict(self.valid_transforms),
            drizzle_dict=drizzle_dict,
            autocrop_enabled=autocrop_enabled,
            autocrop_pct=autocrop_pct,
            ui_owner=self                           # 👈 PASS THE OWNER HERE
        )
        self.post_worker.ui_owner = self
        self.post_worker.need_comet_review.connect(self.on_need_comet_review) 

        self.post_worker.progress.connect(self._on_post_status)
        self.post_worker.finished.connect(self._on_post_pipeline_finished)

        self.post_worker.moveToThread(self.post_thread)
        self.post_thread.started.connect(self.post_worker.run)
        self.post_thread.start()

        self.post_progress = QProgressDialog("Stacking & drizzle (if enabled)…", None, 0, 0, self)
        self.post_progress.setWindowModality(Qt.WindowModality.WindowModal)
        self.post_progress.setCancelButton(None)
        self.post_progress.setMinimumDuration(0)
        self.post_progress.setWindowTitle("Post-Alignment")
        self.post_progress.show()

        self._set_registration_busy(False)

    def _on_mf_progress(self, s: str):
        # Mirror non-token messages
        if not s.startswith("__PROGRESS__"):
            self._on_post_status(s)
            if getattr(self, "_mf_pd", None):
                self._mf_pd.setLabelText(s)
            return

        # "__PROGRESS__ <float> [message]"
        parts = s.split(maxsplit=2)
        try:
            pct = float(parts[1])
        except Exception:
            return

        if len(parts) >= 3 and getattr(self, "_mf_pd", None):
            self._mf_pd.setLabelText(parts[2])

        if getattr(self, "_mf_pd", None):
            groups_done = getattr(self, "_mf_groups_done", 0)
            total_groups = max(1, getattr(self, "_mf_total_groups", 1))
            base = groups_done * 1000
            val = base + int(round(max(0.0, min(1.0, pct)) * 1000))
            self._mf_pd.setRange(0, total_groups * 1000)
            self._mf_pd.setValue(min(val, total_groups * 1000))

    @pyqtSlot(bool, str)
    def _on_post_pipeline_finished(self, ok: bool, message: str):
        try:
            if getattr(self, "post_progress", None):
                self.post_progress.close()
                self.post_progress = None
        except Exception:
            pass

        try:
            self.post_thread.quit()
            self.post_thread.wait()
        except Exception:
            pass
        try:
            self.post_worker.deleteLater()
            self.post_thread.deleteLater()
        except Exception:
            pass

        self.update_status(message)
        self._cfa_for_this_run = None
        QApplication.processEvents()


    def save_rejection_map_sasr(self, rejection_map, out_file):
        """
        Writes the per-file rejection map to a custom text file.
        Format:
            FILE: path/to/file1
            x1, y1
            x2, y2

            FILE: path/to/file2
            ...
        """
        with open(out_file, "w") as f:
            for fpath, coords_list in rejection_map.items():
                f.write(f"FILE: {fpath}\n")
                for (x, y) in coords_list:
                    # Convert to Python int in case they're NumPy int64
                    f.write(f"{int(x)}, {int(y)}\n")
                f.write("\n")  # blank line to separate blocks

    def load_rejection_map_sasr(self, in_file):
        """
        Reads a .sasr text file and rebuilds the rejection map dictionary.
        Returns a dict { fpath: [(x, y), (x, y), ...], ... }
        """
        rejections = {}
        with open(in_file, "r") as f:
            content = f.read().strip()

        # Split on blank lines
        blocks = re.split(r"\n\s*\n", content)
        for block in blocks:
            lines = [line.strip() for line in block.splitlines() if line.strip()]
            if not lines:
                continue

            # First line should be 'FILE: <path>'
            if lines[0].startswith("FILE:"):
                raw_path = lines[0].replace("FILE:", "").strip()
                coords = []
                for line in lines[1:]:
                    # Each subsequent line is "x, y"
                    parts = line.split(",")
                    if len(parts) == 2:
                        x_str, y_str = parts
                        x = int(x_str.strip())
                        y = int(y_str.strip())
                        coords.append((x, y))
                rejections[raw_path] = coords
        return rejections


    @pyqtSlot(list, dict, result=object)   # (files: list[str], initial_xy: dict[str, (x,y)]) -> dict|None
    def show_comet_preview(self, files, initial_xy):
        dlg = CS.CometCentroidPreview(files, initial_xy=initial_xy, parent=self)
        if dlg.exec() == int(QDialog.DialogCode.Accepted):
            return dlg.get_seeds()
        return None

    @pyqtSlot(list, dict, object)
    def on_need_comet_review(self, files, initial_xy, responder):
        # This runs on the GUI thread.
        dlg = CS.CometCentroidPreview(files, initial_xy=initial_xy, parent=self)
        if dlg.exec() == int(QDialog.DialogCode.Accepted):
            result = dlg.get_seeds()
        else:
            result = None
        responder.finished.emit(result)

    def stack_images_mixed_drizzle(
        self,
        grouped_files,           # { group_key: [aligned _n_r.fit paths] }
        frame_weights,           # { file_path: weight }
        transforms_dict,         # { normalized_path -> aligned_path } (kept for compatibility)
        drizzle_dict,            # { group_key: {drizzle_enabled, scale_factor, drop_shrink} }
        *,
        autocrop_enabled: bool,
        autocrop_pct: float,
        status_cb=None
    ):
        """
        Runs normal integration (to get rejection coords), saves masters,
        optionally runs comet-mode stacks and drizzle. Designed to run in a worker thread.

        Returns:
            {
            "summary_lines": [str, ...],
            "autocrop_outputs": [(group_key, out_path_crop), ...]
            }
        """
        log = status_cb or (lambda *_: None)
        comet_mode = bool(getattr(self, "comet_cb", None) and self.comet_cb.isChecked())
        if not hasattr(self, "orig_by_aligned") or not isinstance(getattr(self, "orig_by_aligned"), dict):
            self.orig_by_aligned = {}
        # Comet-mode defaults (surface later if desired)
        COMET_ALGO = "Comet High-Clip Percentile"          # or "Comet Lower-Trim (30%)"
        STARS_ALGO = "Comet High-Clip Percentile"          # star-aligned stack: median best suppresses moving comet

        n_groups = len(grouped_files)
        n_frames = sum(len(v) for v in grouped_files.values())
        log(f"📁 Post-align: {n_groups} group(s), {n_frames} aligned frame(s).")
        QApplication.processEvents()

        # Precompute a single global crop rect if enabled (pure computation, no UI).
        global_rect = None
        if autocrop_enabled:
            log("✂️ Auto Crop Enabled. Calculating bounding box…")

            # --- FAST PATH: use transforms (no image I/O) ---------------------------
            try:
                # Prefer model-aware (drizzle) xforms if present; else affine fallback
                xforms = dict(getattr(self, "drizzle_xforms", {}) or {})
                if not xforms:
                    xforms = dict(getattr(self, "valid_matrices", {}) or {})

                # Geometry of the reference (H, W) captured at alignment time
                ref_hw = tuple(getattr(self, "ref_shape_for_drizzle", ()))
                if xforms and len(ref_hw) == 2:
                    global_rect = self._rect_from_transforms_fast(
                        xforms,
                        src_hw=ref_hw,
                        coverage_pct=float(autocrop_pct),
                        allow_homography=True,   # supports 3x3 too
                        min_side=16
                    )
                    if global_rect:
                        x0, y0, x1, y1 = map(int, global_rect)
                        log(f"✂️ Transform crop (global) → [{x0}:{x1}]×[{y0}:{y1}] ({x1-x0}×{y1-y0})")
                    else:
                        log("✂️ Transform crop produced no stable global rect; falling back to mask-based.")
                else:
                    log("✂️ No transforms/geometry available for fast global crop; falling back to mask-based.")
            except Exception as e:
                log(f"⚠️ Transform-based global crop failed ({e}); falling back to mask-based.")
                global_rect = None

            # --- SLOW FALLBACK: your existing mask-based method ---------------------
            if global_rect is None:
                try:
                    global_rect = self._compute_common_autocrop_rect(
                        grouped_files, autocrop_pct, status_cb=log
                    )
                    if global_rect:
                        x0, y0, x1, y1 = map(int, global_rect)
                        log(f"✂️ Mask-based crop (global) → [{x0}:{x1}]×[{y0}:{y1}] ({x1-x0}×{y1-y0})")
                    else:
                        log("✂️ Global crop disabled; will fall back to per-group.")
                except Exception as e:
                    global_rect = None
                    log(f"⚠️ Global crop (mask-based) failed: {e}")
        QApplication.processEvents()

        group_integration_data = {}
        summary_lines = []
        autocrop_outputs = []

        for gi, (group_key, file_list) in enumerate(grouped_files.items(), 1):
            t_g = perf_counter()
            log(f"🔹 [{gi}/{n_groups}] Integrating '{group_key}' with {len(file_list)} file(s)…")
            QApplication.processEvents()

            # ---- STARS (reference-aligned) integration ----
            # Force a comet-safe reducer for the star-aligned stack only when comet_mode is on.
            integrated_image, rejection_map, ref_header = self.normal_integration_with_rejection(
                group_key, file_list, frame_weights,
                status_cb=log,
                algo_override=(STARS_ALGO if comet_mode else None)   # << correct: stars use STARS_ALGO in comet mode
            )
            log(f"   ↳ Integration done in {perf_counter() - t_g:.1f}s.")
            QApplication.processEvents()
            if integrated_image is None:
                continue

            if ref_header is None:
                ref_header = fits.Header()

            # --- Save the non-cropped STAR master (MEF w/ rejection layers if present) ---
            hdr_orig = ref_header.copy()
            hdr_orig["IMAGETYP"] = "MASTER STACK"
            hdr_orig["BITPIX"]   = -32
            hdr_orig["STACKED"]  = (True, "Stacked using normal_integration_with_rejection")
            hdr_orig["CREATOR"]  = "SetiAstroSuite"
            hdr_orig["DATE-OBS"] = datetime.utcnow().isoformat()

            is_mono_orig = (integrated_image.ndim == 2)
            if is_mono_orig:
                hdr_orig["NAXIS"]  = 2
                hdr_orig["NAXIS1"] = integrated_image.shape[1]
                hdr_orig["NAXIS2"] = integrated_image.shape[0]
                if "NAXIS3" in hdr_orig:
                    del hdr_orig["NAXIS3"]
            else:
                hdr_orig["NAXIS"]  = 3
                hdr_orig["NAXIS1"] = integrated_image.shape[1]
                hdr_orig["NAXIS2"] = integrated_image.shape[0]
                hdr_orig["NAXIS3"] = integrated_image.shape[2]

            n_frames_group = len(file_list)
            H, W = integrated_image.shape[:2]
            display_group = self._label_with_dims(group_key, W, H)
            base = f"MasterLight_{display_group}_{n_frames_group}stacked"
            base = self._normalize_master_stem(base)
            out_path_orig = self._build_out(self.stacking_directory, base, "fit")

            # Try to attach rejection maps that were accumulated during integration
            maps = getattr(self, "_rej_maps", {}).get(group_key)
            save_layers = self.settings.value("stacking/save_rejection_layers", True, type=bool)

            if maps and save_layers:
                try:
                    _save_master_with_rejection_layers(
                        integrated_image,
                        hdr_orig,
                        out_path_orig,
                        rej_any = maps.get("any"),
                        rej_frac= maps.get("frac"),
                    )
                    log(f"✅ Saved integrated image (with rejection layers) for '{group_key}': {out_path_orig}")
                except Exception as e:
                    log(f"⚠️ MEF save failed ({e}); falling back to single-HDU save.")
                    save_image(
                        img_array=integrated_image,
                        filename=out_path_orig,
                        original_format="fit",
                        bit_depth="32-bit floating point",
                        original_header=hdr_orig,
                        is_mono=is_mono_orig
                    )
                    log(f"✅ Saved integrated image (single-HDU) for '{group_key}': {out_path_orig}")
            else:
                # No maps available or feature disabled → single-HDU save
                save_image(
                    img_array=integrated_image,
                    filename=out_path_orig,
                    original_format="fit",
                    bit_depth="32-bit floating point",
                    original_header=hdr_orig,
                    is_mono=is_mono_orig
                )
                log(f"✅ Saved integrated image (original) for '{group_key}': {out_path_orig}")

            # ---- Decide the group’s fixed crop rect (used for ALL outputs in this group) ----
            group_rect = None
            if autocrop_enabled:
                if global_rect is not None:
                    group_rect = tuple(global_rect)
                else:
                    # FAST per-group rect from transforms (no image I/O)
                    try:
                        # Build subset of transforms for just this group's aligned paths
                        xforms_g = {}
                        # orig_by_aligned maps registered path -> original/normalized path
                        oba = getattr(self, "orig_by_aligned", {}) or {}
                        has_model_xforms = bool(getattr(self, "drizzle_xforms", None))

                        for ap in file_list:
                            apn = os.path.normpath(ap)
                            M = None
                            # Prefer model-aware transform keyed by original path
                            if has_model_xforms:
                                op = oba.get(apn)
                                if op:
                                    M = getattr(self, "drizzle_xforms", {}).get(os.path.normpath(op))
                            # Fallback: affine keyed by aligned path
                            if M is None:
                                M = getattr(self, "matrix_by_aligned", {}).get(apn)
                            if M is not None:
                                xforms_g[apn] = np.asarray(M)

                        ref_hw = tuple(getattr(self, "ref_shape_for_drizzle", ()))
                        if xforms_g and len(ref_hw) == 2:
                            group_rect = self._rect_from_transforms_fast(
                                xforms_g,
                                src_hw=ref_hw,
                                coverage_pct=float(autocrop_pct),
                                allow_homography=True,
                                min_side=16
                            )

                        if group_rect is None:
                            # Fallback to existing mask-based per-group method
                            group_rect = self._compute_common_autocrop_rect(
                                {group_key: file_list}, autocrop_pct, status_cb=log
                            )
                    except Exception as e:
                        log(f"⚠️ Per-group transform crop failed for '{group_key}': {e}")
                        group_rect = None

                if group_rect:
                    x1, y1, x2, y2 = map(int, group_rect)
                    log(f"✂️ Using fixed crop rect for '{group_key}': ({x1},{y1})–({x2},{y2})")
                else:
                    log("✂️ No stable rect found for this group; per-image fallback will be used.")

            # --- Optional: auto-cropped STAR copy (uses group_rect if available, else global/per-image logic) ---
            if autocrop_enabled:
                cropped_img, hdr_crop = self._apply_autocrop(
                    integrated_image,
                    file_list,
                    ref_header.copy(),
                    scale=1.0,
                    rect_override=group_rect if group_rect is not None else global_rect
                )
                is_mono_crop = (cropped_img.ndim == 2)
                Hc, Wc = (cropped_img.shape[:2] if cropped_img.ndim >= 2 else (H, W))
                display_group_crop = self._label_with_dims(group_key, Wc, Hc)
                base_crop = f"MasterLight_{display_group_crop}_{n_frames_group}stacked_autocrop"
                base_crop = self._normalize_master_stem(base_crop)
                out_path_crop = self._build_out(self.stacking_directory, base_crop, "fit")

                save_image(
                    img_array=cropped_img,
                    filename=out_path_crop,
                    original_format="fit",
                    bit_depth="32-bit floating point",
                    original_header=hdr_crop,
                    is_mono=is_mono_crop
                )
                log(f"✂️ Saved auto-cropped image for '{group_key}': {out_path_crop}")
                autocrop_outputs.append((group_key, out_path_crop))

            # ---- Optional: COMET mode ----
            if comet_mode:
                log("🌠 Comet mode enabled for this group")

                # registered, time-sorted
                sorted_files = sorted(file_list, key=CS.time_key)

                # Build seeds in the *registered* space
                seeds = {}
                reg_path = None  # ensure defined for logging checks below
                if hasattr(self, "_comet_seed") and self._comet_seed:
                    seed_src_path = os.path.normpath(self._comet_seed.get("path", ""))
                    seed_xy       = tuple(self._comet_seed.get("xy", (0.0, 0.0)))

                    # 1) try exact mapping: original/normalized -> registered path
                    if transforms_dict:
                        reg_path = transforms_dict.get(seed_src_path)

                    # 2) fuzzy fallback by basename prefix (handles _n -> _n_r)
                    if not reg_path:
                        bn = os.path.basename(os.path.splitext(seed_src_path)[0])
                        for p in sorted_files:
                            if os.path.basename(p).startswith(bn):
                                reg_path = p
                                break

                    # 3) transform XY into registered frame
                    if reg_path:
                        M = self.matrix_by_aligned.get(os.path.normpath(reg_path))
                        if M is not None and np.asarray(M).shape == (2,3):
                            x, y = seed_xy
                            a,b,tx = M[0]; c,d,ty = M[1]
                            seeds[os.path.normpath(reg_path)] = (
                                float(a*x + b*y + tx),
                                float(c*x + d*y + ty)
                            )
                            log(f"  ◦ using user seed on {os.path.basename(reg_path)}")
                        else:
                            log("  ⚠️ user seed: no affine for that registered file")

                # 4) Last resort: if no seed mapped to any of the files, drop the reference-frame seed
                if not any(fp in seeds for fp in sorted_files):
                    if getattr(self, "_comet_ref_xy", None):
                        seeds[sorted_files[0]] = tuple(map(float, self._comet_ref_xy))
                        log("  ◦ seeding first registered frame with _comet_ref_xy")

                # Sanity log if we actually have a reg_path and seed
                if reg_path and (os.path.normpath(reg_path) in seeds):
                    sx, sy = seeds[os.path.normpath(reg_path)]
                    log(f"  ◦ seed xy={sx:.1f},{sy:.1f} within {W}×{H}? "
                        f"{'OK' if (0<=sx<W and 0<=sy<H) else 'OUT-OF-BOUNDS'}")

                # 1) Measure comet centers (auto baseline)
                log("🟢 Measuring comet centers (template match)…")
                comet_xy = CS.measure_comet_positions(sorted_files, seeds=seeds, status_cb=log)

                # 2) Offer preview (GUI) via worker signal
                ui_target = None
                try:
                    ui_target = self._find_ui_target() if hasattr(self, "_find_ui_target") else None
                    if ui_target is None:
                        # inline helper (same as your earlier version)
                        def _find_ui_target() -> QWidget | None:
                            ow = getattr(self, "ui_owner", None)
                            if isinstance(ow, QWidget) and hasattr(ow, "show_comet_preview"):
                                return ow
                            par = self.parent()
                            if isinstance(par, QWidget) and hasattr(par, "show_comet_preview"):
                                return par
                            aw = QApplication.activeWindow()
                            if isinstance(aw, QWidget) and hasattr(aw, "show_comet_preview"):
                                return aw
                            for w in QApplication.topLevelWidgets():
                                if hasattr(w, "show_comet_preview"):
                                    return w
                            return None
                        ui_target = _find_ui_target()
                except Exception:
                    ui_target = None

                if ui_target is not None:
                    try:
                        responder = _Responder()
                        loop = QEventLoop()
                        result_box = {"res": None}

                        def _store_and_quit(res):
                            result_box["res"] = res
                            loop.quit()

                        responder.finished.connect(_store_and_quit)

                        emitter = getattr(self, "post_worker", None)
                        if emitter is None:
                            log("  ⚠️ comet preview skipped: no worker emitter present")
                        else:
                            emitter.need_comet_review.emit(sorted_files, comet_xy, responder)
                            loop.exec()  # block this worker thread until GUI responds

                            edited = result_box["res"]
                            if isinstance(edited, dict) and edited:
                                comet_xy = edited
                                log(f"  ◦ user confirmed/edited {len(comet_xy)} centroids")
                            else:
                                log("  ◦ user cancelled or no edits — using auto centroids")
                    except Exception as e:
                        log(f"  ⚠️ comet preview skipped: {e!r}")
                else:
                    log("  ⚠️ comet preview unavailable (no UI target)")

                # 3) Comet-aligned integration
                usable = [fp for fp in sorted_files if fp in comet_xy]
                if len(usable) < 2:
                    log("⚠️ Not enough frames with valid comet centroids; skipping comet stack.")
                else:
                    log("🟠 Comet-aligned integration…")
                    comet_only, comet_rej_map, ref_header_c = self.integrate_comet_aligned(
                        group_key=f"{group_key}",
                        file_list=usable,
                        comet_xy=comet_xy,
                        frame_weights=frame_weights,
                        status_cb=log,
                        algo_override=COMET_ALGO  # << comet-friendly reducer
                    )

                    # Save CometOnly
                    Hc, Wc = comet_only.shape[:2]
                    display_group_c = self._label_with_dims(group_key, Wc, Hc)
                    comet_path = self._build_out(
                        self.stacking_directory,
                        f"MasterCometOnly_{display_group_c}_{len(usable)}stacked",
                        "fit"
                    )
                    save_image(
                        comet_only, comet_path, "fit", "32-bit floating point",
                        original_header=(ref_header_c or ref_header),
                        is_mono=(comet_only.ndim==2)
                    )
                    log(f"✅ Saved CometOnly → {comet_path}")

                    # --- Crop CometOnly identically (if requested) ---
                    if autocrop_enabled and (group_rect is not None or global_rect is not None):
                        comet_only_crop, hdr_c_crop = self._apply_autocrop(
                            comet_only,
                            file_list,  # ok to reuse; rect is forced
                            (ref_header_c or ref_header).copy(),
                            scale=1.0,
                            rect_override=group_rect if group_rect is not None else global_rect
                        )
                        Hcc, Wcc = comet_only_crop.shape[:2]
                        display_group_cc = self._label_with_dims(group_key, Wcc, Hcc)
                        comet_path_crop = self._build_out(
                            self.stacking_directory,
                            f"MasterCometOnly_{display_group_cc}_{len(usable)}stacked_autocrop",
                            "fit"
                        )
                        save_image(
                            comet_only_crop, comet_path_crop, "fit", "32-bit floating point",
                            original_header=hdr_c_crop,
                            is_mono=(comet_only_crop.ndim==2)
                        )
                        log(f"✂️ Saved CometOnly (auto-cropped) → {comet_path_crop}")

                    # Optional blend
                    if getattr(self, "comet_blend_cb", None) and self.comet_blend_cb.isChecked():
                        mix = float(self.comet_mix.value())

                        log(f"🟡 Blending Stars+Comet (screen after 5% stretch; mix={mix:.2f})…")
                        stars_img, comet_img = _match_channels(integrated_image, comet_only)

                        # Screen blend after identical display-stretch on both images
                        blend = CS.blend_screen_stretched(
                            comet_only=comet_img,
                            stars_only=stars_img,
                            stretch_pct=0.05,
                            mix=mix
                        )

                        is_mono_blend = (blend.ndim == 2) or (blend.ndim == 3 and blend.shape[2] == 1)
                        blend_path = self._build_out(
                            self.stacking_directory,
                            f"MasterCometBlend_{display_group_c}_{len(usable)}stacked",
                            "fit"
                        )
                        save_image(blend, blend_path, "fit", "32-bit floating point",
                                ref_header, is_mono=is_mono_blend)
                        log(f"✅ Saved CometBlend → {blend_path}")

                        # --- Crop CometBlend identically (if requested) ---
                        if autocrop_enabled and (group_rect is not None or global_rect is not None):
                            blend_crop, hdr_b_crop = self._apply_autocrop(
                                blend,
                                file_list,
                                ref_header.copy(),
                                scale=1.0,
                                rect_override=group_rect if group_rect is not None else global_rect
                            )
                            Hb, Wb = blend_crop.shape[:2]
                            display_group_bc = self._label_with_dims(group_key, Wb, Hb)
                            blend_path_crop = self._build_out(
                                self.stacking_directory,
                                f"MasterCometBlend_{display_group_bc}_{len(usable)}stacked_autocrop",
                                "fit"
                            )
                            save_image(
                                blend_crop, blend_path_crop, "fit", "32-bit floating point",
                                original_header=hdr_b_crop,
                                is_mono=(blend_crop.ndim == 2 or (blend_crop.ndim == 3 and blend_crop.shape[2] == 1))
                            )
                            log(f"✂️ Saved CometBlend (auto-cropped) → {blend_path_crop}")

            # ---- Drizzle bookkeeping for this group ----
            dconf = drizzle_dict.get(group_key, {})
            if dconf.get("drizzle_enabled", False):
                sasr_path = os.path.join(self.stacking_directory, f"{group_key}_rejections.sasr")
                self.save_rejection_map_sasr(rejection_map, sasr_path)
                log(f"✅ Saved rejection map to {sasr_path}")
                group_integration_data[group_key] = {
                    "integrated_image": integrated_image,
                    "rejection_map": rejection_map,
                    "n_frames": n_frames_group,
                    "drizzled": True
                }
            else:
                group_integration_data[group_key] = {
                    "integrated_image": integrated_image,
                    "rejection_map": None,
                    "n_frames": n_frames_group,
                    "drizzled": False
                }
                log(f"ℹ️ Skipping rejection map save for '{group_key}' (drizzle disabled).")

        QApplication.processEvents()
        # Build ORIGINALS list for each group (needed for true drizzle)
        originals_by_group = {}
        oba = getattr(self, "orig_by_aligned", {}) or {}
        for group, reg_list in grouped_files.items():
            orig_list = []
            for rp in reg_list:
                op = oba.get(os.path.normpath(rp))
                if op:
                    orig_list.append(op)
            originals_by_group[group] = orig_list
        # ---- Drizzle pass (only for groups with drizzle enabled) ----
        for group_key, file_list in grouped_files.items():
            dconf = drizzle_dict.get(group_key)
            if not (dconf and dconf.get("drizzle_enabled", False)):
                log(f"✅ Group '{group_key}' not set for drizzle. Integrated image already saved.")
                continue

            scale_factor = self._get_drizzle_scale()
            drop_shrink  = self._get_drizzle_pixfrac()

            # Optional: also read kernel for logging/branching
            kernel = (self.settings.value("stacking/drizzle_kernel", "square", type=str) or "square").lower()
            status_cb(f"Drizzle cfg → scale={scale_factor}×, pixfrac={drop_shrink:.3f}, kernel={kernel}")
            rejections_for_group = group_integration_data[group_key]["rejection_map"]
            n_frames_group = group_integration_data[group_key]["n_frames"]

            log(f"📐 Drizzle for '{group_key}' at {scale_factor}× (drop={drop_shrink}) using {n_frames_group} frame(s).")

            self.drizzle_stack_one_group(
                group_key=group_key,
                file_list=file_list,                          # registered (for headers/labels)
                original_list=originals_by_group.get(group_key, []),  # <-- NEW
                transforms_dict=transforms_dict,
                frame_weights=frame_weights,
                scale_factor=scale_factor,
                drop_shrink=drop_shrink,
                rejection_map=rejections_for_group,
                autocrop_enabled=autocrop_enabled,
                rect_override=global_rect,
                status_cb=log
            )

        # Build summary lines
        for group_key, info in group_integration_data.items():
            n_frames_group = info["n_frames"]
            drizzled = info["drizzled"]
            summary_lines.append(f"• {group_key}: {n_frames_group} stacked{' + drizzle' if drizzled else ''}")

        if autocrop_outputs:
            summary_lines.append("")
            summary_lines.append("Auto-cropped files saved:")
            for g, p in autocrop_outputs:
                summary_lines.append(f"  • {g} → {p}")

        return {
            "summary_lines": summary_lines,
            "autocrop_outputs": autocrop_outputs
        }



    def integrate_comet_aligned(
        self,
        group_key: str,
        file_list: list[str],
        comet_xy: dict[str, tuple[float,float]],
        frame_weights: dict[str, float],
        status_cb=None,
        *,
        algo_override: str | None = None
    ):
        
        debug_starrem = bool(self.settings.value("stacking/comet_starrem/debug_dump", False, type=bool))
        debug_dir = os.path.join(self.stacking_directory, "debug_comet_starrem")
        os.makedirs(debug_dir, exist_ok=True)        
        """
        Translate each frame so its comet centroid lands on a single reference pixel
        (from file_list[0]). Optional comet star-removal runs AFTER this alignment,
        with a single fixed core mask in comet space. No NaNs; reduction uses the
        selected rejection algorithm.
        """
        log = status_cb or (lambda *_: None)
        if not file_list:
            return None, {}, None

        # --- Reference frame / canvas shape ---
        ref_file = file_list[0]
        ref_img, ref_header, _, _ = load_image(ref_file)
        if ref_img is None:
            log(f"⚠️ Could not load reference '{ref_file}' for comet stack.")
            return None, {}, None

        is_color = (ref_img.ndim == 3 and ref_img.shape[2] == 3)
        H, W = ref_img.shape[:2]
        C = 3 if is_color else 1

        # The single pixel we align to (in ref frame):
        ref_xy = comet_xy[ref_file]
        log(f"📌 Comet reference pixel @ {ref_file} → ({ref_xy[0]:.2f},{ref_xy[1]:.2f})")

        # --- Open sources (mem-mapped readers) ---
        sources = []
        try:
            for p in file_list:
                sources.append(_MMImage(p))   # << was _MMFits
        except Exception as e:
            for s in sources:
                try: s.close()
                except Exception: pass
            log(f"⚠️ Failed to open images (memmap): {e}")
            return None, {}, None

        DTYPE = self._dtype()
        integrated_image = np.zeros((H, W, C), dtype=DTYPE)
        per_file_rejections = {p: [] for p in file_list}

        # --- Chunking (same policy as normal integration) ---
        pref_h, pref_w = self.chunk_height, self.chunk_width
        try:
            chunk_h, chunk_w = compute_safe_chunk(H, W, len(file_list), C, DTYPE, pref_h, pref_w)
            log(f"🔧 Comet stack chunk {chunk_h}×{chunk_w}")
        except MemoryError as e:
            for s in sources:
                try: s.close()
                except Exception: pass
            log(f"⚠️ {e}")
            return None, {}, None

        # Reusable tile buffer
        ts_buf = np.empty((len(file_list), chunk_h, chunk_w, C), dtype=np.float32, order='F')
        weights_array = np.array([frame_weights.get(p, 1.0) for p in file_list], dtype=np.float32)

        # Rejection maps (for MEF layers)
        rej_any   = np.zeros((H, W), dtype=np.bool_)
        rej_count = np.zeros((H, W), dtype=np.uint16)

        # --- Per-frame pure-translation affines (into comet space) ---
        affines = {}
        for p in file_list:
            cx, cy = comet_xy[p]
            dx = ref_xy[0] - cx
            dy = ref_xy[1] - cy
            affines[p] = np.array([[1.0, 0.0, dx],
                                [0.0, 1.0, dy]], dtype=np.float32)

        # ---------- OPTIONAL comet star removal (pre-process per frame) ----------
        csr_enabled = self.settings.value("stacking/comet_starrem/enabled", False, type=bool)
        csr_tool    = self.settings.value("stacking/comet_starrem/tool", "StarNet", type=str)
        core_r      = float(self.settings.value("stacking/comet_starrem/core_r", 22.0, type=float))
        core_soft   = float(self.settings.value("stacking/comet_starrem/core_soft", 6.0, type=float))

        csr_outputs_are_aligned = False   # tells the tile loop whether to warp again
        tmp_root = None
        starless_temp_paths: list[str] | None = None

        if csr_enabled:
            log("✨ Comet star removal enabled — pre-processing frames…")

            # Build a single core-protection mask in comet-aligned coords (center = ref_xy)
            core_mask = CS._protect_core_mask(H, W, ref_xy[0], ref_xy[1], core_r, core_soft).astype(np.float32)

            starless_temp_paths = []
            starless_map = {}  # ← add this
            tmp_root = tempfile.mkdtemp(prefix="sas_comet_starless_")
            try:
                for i, p in enumerate(file_list, 1):
                    try:
                        src = sources[i-1].read_full()  # float32 (H,W) or (H,W,3)
                        # Ensure 3ch for the external tools
                        if src.ndim == 2:
                            src = src[..., None]
                        if src.shape[2] == 1:
                            src = np.repeat(src, 3, axis=2)

                        # Warp into comet space once (so the same mask applies to all frames)
                        M = affines[p]
                        warped = cv2.warpAffine(
                            src, M, (W, H),
                            flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT
                        ).astype(np.float32, copy=False)

                        # Run chosen remover in comet space
                        if csr_tool == "CosmicClarityDarkStar":
                            log("  ◦ DarkStar comet star removal…")
                            starless = CS.darkstar_starless_from_array(warped, self.settings)
                            orig_for_blend = warped

                            m3 = _expand_mask_for(warped, core_mask)
                            protected = np.clip(starless * (1.0 - m3) + orig_for_blend * m3, 0.0, 1.0).astype(np.float32)                            
                        else:
                            log("  ◦ StarNet comet star removal…")
                            # Frames are linear at this stage
                            protected, _ = CS.starnet_starless_pair_from_array(
                                warped, self.settings, is_linear=True,
                                debug_save_dir=debug_dir, debug_tag=f"{i:04d}_{os.path.splitext(os.path.basename(p))[0]}", core_mask=core_mask
                            )
                            protected = np.clip(protected, 0.0, 1.0).astype(np.float32)


                        # Persist as temp FITS (comet-aligned)
                        outp = os.path.join(tmp_root, f"starless_{i:04d}.fit")
                        save_image(
                            img_array=protected,
                            filename=outp,
                            original_format="fit",
                            bit_depth="32-bit floating point",
                            original_header=ref_header,  # simple header OK
                            is_mono=False
                        )
                        if self.settings.value("stacking/comet/save_starless", False, type=bool):
                            save_dir = os.path.join(self.stacking_directory, "starless_comet_aligned")
                            os.makedirs(save_dir, exist_ok=True)
                            base_name = os.path.splitext(os.path.basename(p))[0]
                            out_user = os.path.join(save_dir, f"{base_name}_starless.fit")
                            save_image(
                                img_array=protected,
                                filename=out_user,
                                original_format="fit",
                                bit_depth="32-bit floating point",
                                original_header=ref_header,
                                is_mono=False
                            )                        
                        starless_temp_paths.append(outp)
                        starless_map[p] = outp    
                        log(f"    ✓ [{i}/{len(file_list)}] starless saved")
                    except Exception as e:
                        log(f"  ⚠️ star removal failed on {os.path.basename(p)}: {e}")
                        # Fallback: use the warped original (still comet-aligned)
                        outp = os.path.join(tmp_root, f"starless_{i:04d}.fit")
                        save_image(
                            img_array=warped.astype(np.float32, copy=False),
                            filename=outp,
                            original_format="fit",
                            bit_depth="32-bit floating point",
                            original_header=ref_header,
                            is_mono=False
                        )
                        starless_temp_paths.append(outp)

                # Swap readers to the comet-aligned starless temp files
                for s in sources:
                    try: s.close()
                    except Exception: pass
                sources = [_MMFits(p) for p in starless_temp_paths]
                starless_readers_paths = list(starless_temp_paths)  

                # These temp frames are already comet-aligned ⇒ no further warp in tile loop
                for p in file_list:
                    affines[p] = np.array([[1.0, 0.0, 0.0],
                                        [0.0, 1.0, 0.0]], dtype=np.float32)
                csr_outputs_are_aligned = True
                self._last_comet_used_starless = True                    # ← record for UI/summary
                log(f"✨ Using comet-aligned STARLESS frames for stack ({len(starless_temp_paths)} files).")
            except Exception as e:
                log(f"⚠️ Comet star removal pre-process aborted: {e}")
                csr_outputs_are_aligned = False
                self._last_comet_used_starless = False

        # --- Tile loop ---
        t_idx = 0
        for y0 in range(0, H, chunk_h):
            y1 = min(y0 + chunk_h, H); th = y1 - y0
            for x0 in range(0, W, chunk_w):
                x1 = min(x0 + chunk_w, W); tw = x1 - x0
                t_idx += 1
                log(f"Integrating comet tile {t_idx}…")
                if csr_outputs_are_aligned:
                    log("   • Tile source: STARLESS (pre-aligned)")

                ts = ts_buf[:, :th, :tw, :C]

                for i, src in enumerate(sources):
                    full = src.read_full()  # (H,W) or (H,W,3) float32

                    # --- sanity: ensure this reader corresponds to the original file index
                    if csr_outputs_are_aligned:
                        # Optional soft sanity check (index-based)
                        expected = os.path.normpath(starless_readers_paths[i])
                        actual   = os.path.normpath(getattr(src, "path", expected))
                        if actual != expected:
                            log(f"   ⚠️ Starless reader path mismatch at i={i}; "
                                f"got {os.path.basename(actual)}, expected {os.path.basename(expected)}. Using index order.")

                    if csr_outputs_are_aligned:
                        # Already comet-aligned; just slice the tile
                        if C == 1:
                            if full.ndim == 3:
                                full = full[..., 0]  # collapse RGB→mono (same as stars stack behavior)
                            tile = full[y0:y1, x0:x1]
                            ts[i, :, :, 0] = tile
                        else:
                            if full.ndim == 2:
                                full = full[..., None].repeat(3, axis=2)
                            ts[i, :, :, :] = full[y0:y1, x0:x1, :]
                    else:
                        # Warp into comet space on the fly
                        M = affines[file_list[i]]
                        if C == 1:
                            full2d = full[..., 0] if full.ndim == 3 else full
                            warped2d = cv2.warpAffine(full2d, M, (W, H),
                                                    flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)
                            ts[i, :, :, 0] = warped2d[y0:y1, x0:x1]
                        else:
                            if full.ndim == 2:
                                full = full[..., None].repeat(3, axis=2)
                            warped = cv2.warpAffine(full, M, (W, H),
                                                    flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)
                            ts[i, :, :, :] = warped[y0:y1, x0:x1, :]

                # --- Apply selected rejection algorithm ---
                algo = (algo_override or self.rejection_algorithm)
                log(f"  ◦ applying rejection algorithm: {algo}")

                if algo in ("Comet Median", "Simple Median (No Rejection)"):
                    tile_result  = np.median(ts, axis=0)
                    tile_rej_map = np.zeros((len(file_list), th, tw), dtype=bool)

                elif algo == "Comet High-Clip Percentile":
                    k = self.settings.value("stacking/comet_hclip_k", 1.30, type=float)
                    p = self.settings.value("stacking/comet_hclip_p", 25.0, type=float)
                    # keep a small dict across tiles to reuse scratch buffers

                    tile_result = _high_clip_percentile(ts, k=float(k), p=float(p))
                    tile_rej_map = np.zeros((len(file_list), th, tw), dtype=bool)

                elif algo == "Comet Lower-Trim (30%)":
                    tile_result  = _lower_trimmed_mean(ts, trim_hi_frac=0.30)
                    tile_rej_map = np.zeros((len(file_list), th, tw), dtype=bool)

                elif algo == "Comet Percentile (40th)":
                    tile_result  = _percentile40(ts)
                    tile_rej_map = np.zeros((len(file_list), th, tw), dtype=bool)

                elif algo == "Simple Average (No Rejection)":
                    tile_result  = np.average(ts, axis=0, weights=weights_array)
                    tile_rej_map = np.zeros((len(file_list), th, tw), dtype=bool)

                elif algo == "Weighted Windsorized Sigma Clipping":
                    tile_result, tile_rej_map = windsorized_sigma_clip_weighted(
                        ts, weights_array, lower=self.sigma_low, upper=self.sigma_high
                    )

                elif algo == "Kappa-Sigma Clipping":
                    tile_result, tile_rej_map = kappa_sigma_clip_weighted(
                        ts, weights_array, kappa=self.kappa, iterations=self.iterations
                    )

                elif algo == "Trimmed Mean":
                    tile_result, tile_rej_map = trimmed_mean_weighted(
                        ts, weights_array, trim_fraction=self.trim_fraction
                    )

                elif algo == "Extreme Studentized Deviate (ESD)":
                    tile_result, tile_rej_map = esd_clip_weighted(
                        ts, weights_array, threshold=self.esd_threshold
                    )

                elif algo == "Biweight Estimator":
                    tile_result, tile_rej_map = biweight_location_weighted(
                        ts, weights_array, tuning_constant=self.biweight_constant
                    )

                elif algo == "Modified Z-Score Clipping":
                    tile_result, tile_rej_map = modified_zscore_clip_weighted(
                        ts, weights_array, threshold=self.modz_threshold
                    )

                elif algo == "Max Value":
                    tile_result, tile_rej_map = max_value_stack(ts, weights_array)

                else:
                    # default to comet-safe median
                    tile_result  = np.median(ts, axis=0)
                    tile_rej_map = np.zeros((len(file_list), th, tw), dtype=bool)

                integrated_image[y0:y1, x0:x1, :] = tile_result

                # Accumulate rejection bookkeeping
                trm = tile_rej_map
                if trm.ndim == 4:
                    trm = np.any(trm, axis=-1)  # (N, th, tw)
                rej_any[y0:y1, x0:x1]  |= np.any(trm, axis=0)
                rej_count[y0:y1, x0:x1] += trm.sum(axis=0).astype(np.uint16)

                for i, fpath in enumerate(file_list):
                    ys, xs = np.where(trm[i])
                    if ys.size:
                        per_file_rejections[fpath].extend(zip(x0 + xs, y0 + ys))

        # Close readers and clean temp
        for s in sources:
            try: s.close()
            except Exception: pass

        if tmp_root is not None:
            try: shutil.rmtree(tmp_root, ignore_errors=True)
            except Exception: pass

        if C == 1:
            integrated_image = integrated_image[..., 0]

        # Store MEF rejection maps for this comet stack
        if not hasattr(self, "_rej_maps"):
            self._rej_maps = {}
        rej_frac = (rej_count.astype(np.float32) / float(max(1, len(file_list))))
        self._rej_maps[group_key + " (COMET)"] = {
            "any":   rej_any,
            "frac":  rej_frac,
            "count": rej_count,
            "n":     len(file_list),
        }

        return integrated_image, per_file_rejections, ref_header


    def save_registered_images(self, success, msg, frame_weights):
        if not success:
            self.update_status(f"⚠️ Image registration failed: {msg}")
            return

        self.update_status("✅ All frames registered successfully!")
        QApplication.processEvents()
        
        # Use the grouped files already stored from the tree view.
        if not self.light_files:
            self.update_status("⚠️ No light frames available for stacking!")
            return
        
        self.update_status(f"📂 Preparing to stack {sum(len(v) for v in self.light_files.values())} frames in {len(self.light_files)} groups.")
        QApplication.processEvents()
        
        # Pass the dictionary (grouped by filter, exposure, dimensions) to the stacking function.
        self.stack_registered_images(self.light_files, frame_weights)

    def _mmcache_dir(self) -> str:
        d = os.path.join(self.stacking_directory, "_mmcache")
        os.makedirs(d, exist_ok=True)
        return d

    def _memmap_key(self, file_path: str) -> str:
        """Stable key bound to path + size + mtime (regen if source changes)."""
        st = os.stat(file_path)
        sig = f"{os.path.abspath(file_path)}|{st.st_size}|{st.st_mtime_ns}"
        return hashlib.sha1(sig.encode("utf-8")).hexdigest()[:16]

    def _ensure_float32_memmap(self, file_path: str) -> tuple[str, tuple[int,int,int]]:
        """
        Ensure a (H,W,C float32) .npy exists for file_path. Returns (npy_path, shape).
        We keep C=1 for mono, C=3 for color. Values in [0..1].
        """
        key = self._memmap_key(file_path)
        npy_path = os.path.join(self._mmcache_dir(), f"{key}.npy")
        if os.path.exists(npy_path):
            # Shape header is embedded in the .npy; we’ll read when opening.
            return npy_path, None

        img, hdr, _, _ = load_image(file_path)
        if img is None:
            raise RuntimeError(f"Could not load {file_path} to create memmap cache.")

        # Normalize → float32 [0..1], ensure channels-last (H,W,C).
        if img.ndim == 2:
            arr = img.astype(np.float32, copy=False)
            if arr.dtype == np.uint16: arr = arr / 65535.0
            elif arr.dtype == np.uint8: arr = arr / 255.0
            else: arr = np.clip(arr, 0.0, 1.0)
            arr = arr[..., None]  # (H,W,1)
        elif img.ndim == 3:
            if img.shape[0] == 3 and img.shape[2] != 3:
                img = np.transpose(img, (1,2,0))  # (H,W,3)
            arr = img.astype(np.float32, copy=False)
            if arr.dtype == np.uint16: arr = arr / 65535.0
            elif arr.dtype == np.uint8: arr = arr / 255.0
            else: arr = np.clip(arr, 0.0, 1.0)
        else:
            raise ValueError(f"Unsupported image ndim={img.ndim} for {file_path}")

        H, W, C = arr.shape
        mm = open_memmap(npy_path, mode="w+", dtype=np.float32, shape=(H, W, C))
        mm[:] = arr  # single write
        del mm
        return npy_path, (H, W, C)

    def _open_memmaps_readonly(self, paths: list[str]) -> dict[str, np.memmap]:
        """Open all cached arrays in read-only mmap mode."""
        views = {}
        for p in paths:
            npy, _ = self._ensure_float32_memmap(p)
            views[p] = np.load(npy, mmap_mode="r")  # returns numpy.memmap
        return views


    def stack_registered_images_chunked(
        self,
        grouped_files,
        frame_weights,
        chunk_height=2048,
        chunk_width=2048
    ):
        self.update_status(f"✅ Chunked stacking {len(grouped_files)} group(s)...")
        QApplication.processEvents()

        all_rejection_coords = []

        for group_key, file_list in grouped_files.items():
            num_files = len(file_list)
            self.update_status(f"📊 Group '{group_key}' has {num_files} aligned file(s).")
            QApplication.processEvents()
            if num_files < 2:
                self.update_status(f"⚠️ Group '{group_key}' does not have enough frames to stack.")
                continue

            # Reference shape/header (unchanged)
            ref_file = file_list[0]
            if not os.path.exists(ref_file):
                self.update_status(f"⚠️ Reference file '{ref_file}' not found, skipping group.")
                continue

            ref_data, ref_header, _, _ = load_image(ref_file)
            if ref_data is None:
                self.update_status(f"⚠️ Could not load reference '{ref_file}', skipping group.")
                continue

            is_color = (ref_data.ndim == 3 and ref_data.shape[2] == 3)
            height, width = ref_data.shape[:2]
            channels = 3 if is_color else 1

            # Final output memmap (unchanged)
            memmap_path = self._build_out(self.stacking_directory, f"chunked_{group_key}", "dat")
            final_stacked = np.memmap(memmap_path, dtype=np.float32, mode='w+', shape=(height, width, channels))

            # Valid files + weights
            aligned_paths, weights_list = [], []
            for fpath in file_list:
                if os.path.exists(fpath):
                    aligned_paths.append(fpath)
                    weights_list.append(frame_weights.get(fpath, 1.0))
                else:
                    self.update_status(f"⚠️ File not found: {fpath}, skipping.")
            if len(aligned_paths) < 2:
                self.update_status(f"⚠️ Not enough valid frames in group '{group_key}' to stack.")
                continue

            weights_list = np.array(weights_list, dtype=np.float32)

            # ⬇️ NEW: open read-only memmaps for all aligned frames (float32 [0..1], HxWxC)
            mm_views = self._open_memmaps_readonly(aligned_paths)

            self.update_status(f"📊 Stacking group '{group_key}' with {self.rejection_algorithm}")
            QApplication.processEvents()

            rejection_coords = []
            N = len(aligned_paths)
            DTYPE  = self._dtype()
            pref_h = self.chunk_height
            pref_w = self.chunk_width

            try:
                chunk_h, chunk_w = compute_safe_chunk(height, width, N, channels, DTYPE, pref_h, pref_w)
                self.update_status(f"🔧 Using chunk size {chunk_h}×{chunk_w} for {self._dtype()}")
            except MemoryError as e:
                self.update_status(f"⚠️ {e}")
                return None, {}, None

            # Tile loop (same structure, but tile loading reads from memmaps)
            from concurrent.futures import ThreadPoolExecutor, as_completed
            LOADER_WORKERS = min(max(2, (os.cpu_count() or 4) // 2), 8)  # tuned for memory bw

            for y_start in range(0, height, chunk_h):
                y_end  = min(y_start + chunk_h, height)
                tile_h = y_end - y_start

                for x_start in range(0, width, chunk_w):
                    x_end  = min(x_start + chunk_w, width)
                    tile_w = x_end - x_start

                    # Preallocate tile stack
                    tile_stack = np.empty((N, tile_h, tile_w, channels), dtype=np.float32)

                    # ⬇️ NEW: fill tile_stack from the memmaps (parallel copy)
                    def _copy_one(i, path):
                        v = mm_views[path][y_start:y_end, x_start:x_end]  # view on disk
                        if v.ndim == 2:
                            # mono memmap stored as (H,W,1); but if legacy mono npy exists as (H,W),
                            # make it (H,W,1) here:
                            vv = v[..., None]
                        else:
                            vv = v
                        if vv.shape[2] == 1 and channels == 3:
                            vv = np.repeat(vv, 3, axis=2)
                        tile_stack[i] = vv

                    with ThreadPoolExecutor(max_workers=LOADER_WORKERS) as exe:
                        futs = {exe.submit(_copy_one, i, p): i for i, p in enumerate(aligned_paths)}
                        for _ in as_completed(futs):
                            pass

                    # Rejection (unchanged – uses your Numba kernels)
                    algo = self.rejection_algorithm
                    if algo == "Simple Median (No Rejection)":
                        tile_result  = np.median(tile_stack, axis=0)
                        tile_rej_map = np.zeros(tile_stack.shape[1:3], dtype=np.bool_)
                    elif algo == "Simple Average (No Rejection)":
                        tile_result  = np.average(tile_stack, axis=0, weights=weights_list)
                        tile_rej_map = np.zeros(tile_stack.shape[1:3], dtype=np.bool_)
                    elif algo == "Weighted Windsorized Sigma Clipping":
                        tile_result, tile_rej_map = windsorized_sigma_clip_weighted(
                            tile_stack, weights_list, lower=self.sigma_low, upper=self.sigma_high
                        )
                    elif algo == "Kappa-Sigma Clipping":
                        tile_result, tile_rej_map = kappa_sigma_clip_weighted(
                            tile_stack, weights_list, kappa=self.kappa, iterations=self.iterations
                        )
                    elif algo == "Trimmed Mean":
                        tile_result, tile_rej_map = trimmed_mean_weighted(
                            tile_stack, weights_list, trim_fraction=self.trim_fraction
                        )
                    elif algo == "Extreme Studentized Deviate (ESD)":
                        tile_result, tile_rej_map = esd_clip_weighted(
                            tile_stack, weights_list, threshold=self.esd_threshold
                        )
                    elif algo == "Biweight Estimator":
                        tile_result, tile_rej_map = biweight_location_weighted(
                            tile_stack, weights_list, tuning_constant=self.biweight_constant
                        )
                    elif algo == "Modified Z-Score Clipping":
                        tile_result, tile_rej_map = modified_zscore_clip_weighted(
                            tile_stack, weights_list, threshold=self.modz_threshold
                        )
                    elif algo == "Max Value":
                        tile_result, tile_rej_map = max_value_stack(
                            tile_stack, weights_list
                        )
                    else:
                        tile_result, tile_rej_map = windsorized_sigma_clip_weighted(
                            tile_stack, weights_list, lower=self.sigma_low, upper=self.sigma_high
                        )

                    # Commit tile
                    final_stacked[y_start:y_end, x_start:x_end, :] = tile_result

                    # Collect per-tile rejection coords (unchanged logic)
                    if tile_rej_map.ndim == 3:          # (N, tile_h, tile_w)
                        combined_rej = np.any(tile_rej_map, axis=0)
                    elif tile_rej_map.ndim == 4:        # (N, tile_h, tile_w, C)
                        combined_rej = np.any(tile_rej_map, axis=0)
                        combined_rej = np.any(combined_rej, axis=-1)
                    else:
                        combined_rej = np.zeros((tile_h, tile_w), dtype=np.bool_)

                    ys_tile, xs_tile = np.where(combined_rej)
                    for dy, dx in zip(ys_tile, xs_tile):
                        rejection_coords.append((x_start + dx, y_start + dy))

            # Finish/save (unchanged from your version) …
            final_array = np.array(final_stacked)
            del final_stacked

            flat = final_array.ravel()
            nz = np.where(flat > 0)[0]
            if nz.size > 0:
                final_array -= flat[nz[0]]

            new_max = final_array.max()
            if new_max > 1.0:
                new_min = final_array.min()
                rng = new_max - new_min
                final_array = (final_array - new_min) / rng if rng != 0 else np.zeros_like(final_array, np.float32)

            if final_array.ndim == 3 and final_array.shape[-1] == 1:
                final_array = final_array[..., 0]
            is_mono = (final_array.ndim == 2)

            if ref_header is None:
                ref_header = fits.Header()
            ref_header["IMAGETYP"] = "MASTER STACK"
            ref_header["BITPIX"] = -32
            ref_header["STACKED"] = (True, "Stacked using chunked approach")
            ref_header["CREATOR"] = "SetiAstroSuite"
            ref_header["DATE-OBS"] = datetime.utcnow().isoformat()
            if is_mono:
                ref_header["NAXIS"]  = 2
                ref_header["NAXIS1"] = final_array.shape[1]
                ref_header["NAXIS2"] = final_array.shape[0]
                if "NAXIS3" in ref_header: del ref_header["NAXIS3"]
            else:
                ref_header["NAXIS"]  = 3
                ref_header["NAXIS1"] = final_array.shape[1]
                ref_header["NAXIS2"] = final_array.shape[0]
                ref_header["NAXIS3"] = 3

            output_stem = f"MasterLight_{group_key}_{len(aligned_paths)}stacked"
            output_path  = self._build_out(self.stacking_directory, output_stem, "fit")

            save_image(
                img_array=final_array,
                filename=output_path,
                original_format="fit",
                bit_depth="32-bit floating point",
                original_header=ref_header,
                is_mono=is_mono
            )

            self.update_status(f"✅ Group '{group_key}' stacked {len(aligned_paths)} frame(s)! Saved: {output_path}")

            print(f"✅ Master Light saved for group '{group_key}': {output_path}")

            # Optionally, you might want to store or log 'rejection_coords' (here appended to all_rejection_coords)
            all_rejection_coords.extend(rejection_coords)

            # Clean up memmap file
            try:
                os.remove(memmap_path)
            except OSError:
                pass

        QMessageBox.information(
            self,
            "Stacking Complete",
            f"All stacking finished successfully.\n"
            f"Frames per group:\n" +
            "\n".join([f"{group_key}: {len(files)} frame(s)" for group_key, files in grouped_files.items()])
        )

        # Optionally, you could return the global rejection coordinate list.
        return all_rejection_coords        

    def _start_after_align_worker(self, aligned_light_files: dict[str, list[str]]):
        # Snapshot UI settings
        if getattr(self, "_suppress_normal_integration_once", False):
            self._suppress_normal_integration_once = False
            self.update_status("⏭️ Normal integration suppressed (MFDeconv-only run).")
            self._set_registration_busy(False)
            return        
        drizzle_dict = self.gather_drizzle_settings_from_tree()
        try:
            autocrop_enabled = self.autocrop_cb.isChecked()
            autocrop_pct = float(self.autocrop_pct.value())
        except Exception:
            autocrop_enabled = self.settings.value("stacking/autocrop_enabled", False, type=bool)
            autocrop_pct = float(self.settings.value("stacking/autocrop_pct", 95.0, type=float))

        # CFA fill log (optional)
        if getattr(self, "valid_matrices", None):
            try:
                cfa_effective = bool(
                    self._cfa_for_this_run
                    if getattr(self, "_cfa_for_this_run", None) is not None
                    else (getattr(self, "cfa_drizzle_cb", None) and self.cfa_drizzle_cb.isChecked())
                )
                if cfa_effective:
                    fill = self._dither_phase_fill(self.valid_matrices, bins=8)
                    self.update_status(f"🔎 CFA drizzle sub-pixel phase fill (8×8): {fill*100:.1f}%")
            except Exception:
                pass

        # Launch the normal post-align worker
        self.post_thread = QThread(self)
        self.post_worker = AfterAlignWorker(
            self,
            light_files=aligned_light_files,
            frame_weights=dict(self.frame_weights),
            transforms_dict=dict(self.valid_transforms),
            drizzle_dict=drizzle_dict,
            autocrop_enabled=autocrop_enabled,
            autocrop_pct=autocrop_pct,
            ui_owner=self
        )
        self.post_worker.ui_owner = self
        self.post_worker.need_comet_review.connect(self.on_need_comet_review)
        self.post_worker.progress.connect(self._on_post_status)
        self.post_worker.finished.connect(self._on_post_pipeline_finished)
        self.post_worker.moveToThread(self.post_thread)
        self.post_thread.started.connect(self.post_worker.run)
        self.post_thread.start()

        self.post_progress = QProgressDialog("Stacking & drizzle (if enabled)…", None, 0, 0, self)
        self.post_progress.setWindowModality(Qt.WindowModality.WindowModal)
        self.post_progress.setCancelButton(None)
        self.post_progress.setMinimumDuration(0)
        self.post_progress.setWindowTitle("Post-Alignment")
        self.post_progress.show()

        # Important for button state
        self._set_registration_busy(False)


    def _pd_alive(self):
        pd = getattr(self, "_mf_pd", None)
        if pd is None:
            return None
        # If Qt already destroyed it, skip
        if sip.isdeleted(pd):
            return None
        return pd



    def _run_mfdeconv_then_continue(self, aligned_light_files: dict[str, list[str]]):
        """Queue MFDeconv per group if enabled, then continue into AfterAlignWorker for all groups."""
        mf_enabled = self.settings.value("stacking/mfdeconv/enabled", False, type=bool)
        if not mf_enabled:
            self._start_after_align_worker(aligned_light_files)
            return

        # Build list of non-empty groups
        mf_groups = [(g, lst) for g, lst in aligned_light_files.items() if lst]
        if not mf_groups:
            self.update_status("⚠️ No aligned frames available for MF deconvolution.")
            self._start_after_align_worker(aligned_light_files)
            return

        # Progress UI for the entire MF phase
        self._mf_total_groups = len(mf_groups)
        self._mf_groups_done = 0
        self._mf_pd = QProgressDialog("Multi-frame deconvolving…", "Cancel", 0, self._mf_total_groups * 1000, self)
        self._mf_pd.setValue(0)
        # self._mf_pd.setWindowModality(Qt.WindowModality.ApplicationModal)
        self._mf_pd.setMinimumDuration(0)
        self._mf_pd.setWindowTitle("MF Deconvolution")
        self._mf_pd.setRange(0, self._mf_total_groups * 1000)
        self._mf_pd.setValue(0)
        self._mf_pd.show()

        self._mf_queue = list(mf_groups)
        self._mf_results = {}
        self._mf_cancelled = False
        self._mf_thread = None
        self._mf_worker = None

        def _cancel_all():
            self._mf_cancelled = True
        self._mf_pd.canceled.connect(_cancel_all, Qt.ConnectionType.QueuedConnection)

        def _start_next():
            # End of queue or canceled → finish gracefully
            if self._mf_cancelled or not self._mf_queue:
                if getattr(self, "_mf_pd", None):
                    pd = self._pd_alive()
                    if pd:
                        pd.reset()
                        pd.deleteLater()
                    self._mf_pd = None
                try:
                    if self._mf_thread:
                        self._mf_thread.quit()
                        self._mf_thread.wait()
                except Exception:
                    pass
                self._mf_thread = None
                self._mf_worker = None

                # Continue the normal pipeline for ALL groups
                self._suppress_normal_integration_once = True
                self.update_status("✅ MFDeconv complete for all groups. Skipping normal integration.")
                self._set_registration_busy(False)
                return

            group_key, frames = self._mf_queue.pop(0)
            out_dir = os.path.join(self.stacking_directory, "Masters")
            os.makedirs(out_dir, exist_ok=True)

            # Settings snapshot
            iters = self.settings.value("stacking/mfdeconv/iters", 20, type=int)
            min_iters = self.settings.value("stacking/mfdeconv/min_iters", 3, type=int)
            kappa = self.settings.value("stacking/mfdeconv/kappa", 2.0, type=float)
            mode = self.mf_color_combo.currentText()
            Huber = self.settings.value("stacking/mfdeconv/Huber_delta", 0.0, type=float)
            save_intermediate = self.mf_save_intermediate_cb.isChecked()
            seed_mode_cfg = str(self.settings.value("stacking/mfdeconv/seed_mode", "robust"))
            use_star_masks = self.mf_use_star_mask_cb.isChecked()
            use_variance_maps = self.mf_use_noise_map_cb.isChecked()
            rho = self.mf_rho_combo.currentText()

            star_mask_cfg = {
                "thresh_sigma":  self.settings.value("stacking/mfdeconv/star_mask/thresh_sigma",  _SM_DEF_THRESH, type=float),
                "grow_px":       self.settings.value("stacking/mfdeconv/star_mask/grow_px",       _SM_DEF_GROW, type=int),
                "soft_sigma":    self.settings.value("stacking/mfdeconv/star_mask/soft_sigma",    _SM_DEF_SOFT, type=float),
                "max_radius_px": self.settings.value("stacking/mfdeconv/star_mask/max_radius_px", _SM_DEF_RMAX, type=int),
                "max_objs":      self.settings.value("stacking/mfdeconv/star_mask/max_objs",      _SM_DEF_MAXOBJS, type=int),
                "keep_floor":    self.settings.value("stacking/mfdeconv/star_mask/keep_floor",    _SM_DEF_KEEPF, type=float),
                "ellipse_scale": self.settings.value("stacking/mfdeconv/star_mask/ellipse_scale", _SM_DEF_ES, type=float),
            }
            varmap_cfg = {
                "sample_stride": self.settings.value("stacking/mfdeconv/varmap/sample_stride", _VM_DEF_STRIDE, type=int),
                "smooth_sigma":  self.settings.value("stacking/mfdeconv/varmap/smooth_sigma", 1.0, type=float),
                "floor":         self.settings.value("stacking/mfdeconv/varmap/floor",        1e-8, type=float),
            }

            sr_enabled_ui = self.mf_sr_cb.isChecked()
            sr_factor_ui = getattr(self, "mf_sr_factor_spin", None)
            sr_factor_val = sr_factor_ui.value() if sr_factor_ui is not None else self.settings.value("stacking/mfdeconv/sr_factor", 2, type=int)
            super_res_factor = int(sr_factor_val) if sr_enabled_ui else 1

            # Unique, safe filename
            safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", str(group_key)) or "Group"
            out_path = os.path.join(out_dir, f"MasterLight_{safe_name}_{len(frames)}f_MFDeconv_{mode}_{iters}it_k{int(round(kappa*100))}.fit")

            # Thread + worker
            self._mf_thread = QThread(self)
            star_mask_ref = self.reference_frame if use_star_masks else None

            # ── choose engine plainly (Normal / cuDNN-free / High Octane) ─────────────
            # Expect a setting saved by your radio buttons: "normal" | "cudnn" | "sport"
            engine = str(self.settings.value("stacking/mfdeconv/engine", "normal")).lower()

            try:
                if engine == "cudnn":
                    from pro.mfdeconvcudnn import MultiFrameDeconvWorkercuDNN as MFCls
                    eng_name = "Normal (cuDNN-free)"
                elif engine == "sport":  # High Octane let 'er rip
                    from pro.mfdeconvsport import MultiFrameDeconvWorkerSport as MFCls
                    eng_name = "High Octane"
                else:
                    from pro.mfdeconv import MultiFrameDeconvWorker as MFCls
                    eng_name = "Normal"
            except Exception as e:
                # if an import fails, fall back to the safe Normal path
                self.update_status(f"⚠️ MFDeconv engine import failed ({e}); falling back to Normal.")
                from pro.mfdeconv import MultiFrameDeconvWorker as MFCls
                eng_name = "Normal (fallback)"

            self.update_status(f"⚙️ MFDeconv engine: {eng_name}")

            # ── build worker exactly the same in all modes ────────────────────────────
            self._mf_worker = MFCls(
                parent=None,
                aligned_paths=frames,
                output_path=out_path,
                iters=iters,
                kappa=kappa,
                color_mode=mode,
                huber_delta=Huber,
                min_iters=min_iters,
                use_star_masks=use_star_masks,
                use_variance_maps=use_variance_maps,
                rho=rho,
                star_mask_cfg=star_mask_cfg,
                varmap_cfg=varmap_cfg,
                save_intermediate=save_intermediate,
                super_res_factor=super_res_factor,
                star_mask_ref_path=star_mask_ref,
                seed_mode=seed_mode_cfg,
            )

            # Wiring
            self._mf_worker.moveToThread(self._mf_thread)
            self._mf_worker.progress.connect(self._on_mf_progress, Qt.ConnectionType.QueuedConnection)
            self._mf_thread.started.connect(self._mf_worker.run, Qt.ConnectionType.QueuedConnection)
            self._mf_worker.finished.connect(self._mf_thread.quit, Qt.ConnectionType.QueuedConnection)
            self._mf_thread.finished.connect(self._mf_worker.deleteLater)    # free worker on thread end
            self._mf_thread.finished.connect(self._mf_thread.deleteLater)    # free thread object

            def _done(ok: bool, message: str, out: str):
                pd = self._pd_alive()
                if pd:
                    try:
                        # if you keep the 0..groups*1000 range, snap to segment boundary:
                        val = min(pd.value() + 1000, pd.maximum())
                        pd.setValue(val)
                        pd.setLabelText(f"{'✅' if ok else '❌'} {group_key}: {message}")
                    except Exception:
                        pass

                if ok and out:
                    self._mf_results[group_key] = out
                else:
                    self.update_status(f"❌ MFDeconv failed for '{group_key}': {message}")

                try:
                    self._mf_thread.quit()
                    self._mf_thread.wait()
                except Exception:
                    pass
                self._mf_thread = None
                self._mf_worker = None

                QTimer.singleShot(0, _start_next)

            self._mf_worker.finished.connect(_done, Qt.ConnectionType.QueuedConnection)

            self._mf_thread.start()

            if getattr(self, "_mf_pd", None):
                pd = self._pd_alive()
                if pd:
                    pd.setLabelText(f"Deconvolving '{group_key}' ({len(frames)} frames)…")

        QTimer.singleShot(0, _start_next)



    def integrate_registered_images(self):
        """
        Integrate frames that are already aligned (and typically normalized).
        We only do fast measurements for weights; no re-normalization, no re-alignment.
        """
        if getattr(self, "_registration_busy", False):
            self.update_status("⏸ Another job is running; ignoring extra click.")
            return
        self._set_registration_busy(True)

        try:
            self.update_status("🔄 Integrating Previously Registered Images…")

            # 1) Pull files from the tree
            self.extract_light_files_from_tree()
            if not self.light_files:
                self.update_status("⚠️ No registered images found!")
                self._set_registration_busy(False)
                return

            # Flatten
            all_files = [p for lst in self.light_files.values() for p in lst]
            if not all_files:
                self.update_status("⚠️ No frames found in the registration tree!")
                self._set_registration_busy(False)
                return

            # Prefer already-normalized/registered frames
            def _looks_norm_reg(p: str) -> bool:
                bn = os.path.basename(p).lower()
                if (
                    bn.endswith("_n.fit") or bn.endswith("_n.fits")
                    or bn.endswith("_n_r.fit") or bn.endswith("_n_r.fits")
                    or ("aligned_images" in os.path.normpath(p).lower())
                    or bn.endswith(".xisf")  # ← treat XISF as already prepared by PI
                ):
                    return True
                # header fallback
                try:
                    hdr = _get_header_fast(p)  # now supports XISF
                    if hdr and (hdr.get("DEBAYERED") is not None or hdr.get("SAS_RSMP") or hdr.get("SAS_NORM") or hdr.get("NORMALIZ")):
                        return True
                except Exception:
                    pass
                return False

            cand = [p for p in all_files if _looks_norm_reg(p)]
            if not cand:
                # fall back to everything, but we still won't normalize here
                cand = all_files[:]

            self.update_status(f"📊 Found {len(cand)} aligned/normalized frames. Measuring in parallel previews…")

            # 2) Chunked preview measurement (mean + star count/ecc)
            self.frame_weights = {}
            mean_values = {}
            star_counts = {}
            measured_frames = []

            max_workers = os.cpu_count() or 4
            chunk_size = max_workers

            def chunk_list(lst, size):
                for i in range(0, len(lst), size):
                    yield lst[i:i+size]

            chunks = list(chunk_list(cand, chunk_size))
            total_chunks = len(chunks)

            # For already registered images, we don’t need to rescale to a target bin.
            # We can just make small previews directly from the FITS (debayer-aware superpixel).
            from concurrent.futures import ThreadPoolExecutor, as_completed

            for idx, chunk in enumerate(chunks, 1):
                self.update_status(f"📦 Measuring chunk {idx}/{total_chunks} ({len(chunk)} frames)")
                QApplication.processEvents()

                # Load tiny previews in parallel
                previews = []
                paths_ok = []

                def _preview_job(fp: str):
                    return _quick_preview_from_path(fp, target_xbin=1, target_ybin=1)

                with ThreadPoolExecutor(max_workers=max_workers) as ex:
                    futs = {ex.submit(_preview_job, fp): fp for fp in chunk}
                    for fut in as_completed(futs):
                        fp = futs[fut]
                        try:
                            p = fut.result()
                            if p is None:
                                continue
                            previews.append(p)
                            paths_ok.append(fp)
                        except Exception as e:
                            self.update_status(f"⚠️ Preview error for {fp}: {e}")
                        QApplication.processEvents()

                if not previews:
                    self.update_status("⚠️ No valid previews in this chunk.")
                    continue

                # Crop all previews in this chunk to a common min size (cheap)
                min_h = min(im.shape[0] for im in previews)
                min_w = min(im.shape[1] for im in previews)
                if any((im.shape[0] != min_h or im.shape[1] != min_w) for im in previews):
                    previews = [_center_crop_2d(im, min_h, min_w) for im in previews]

                # Means (vectorized)
                means = np.array([float(np.mean(ci)) for ci in previews], dtype=np.float32)

                # Star count + ecc on small, further-downsampled previews
                def _star_job(i_fp):
                    i, fp = i_fp
                    p = previews[i]
                    # normalize preview to [0..] by subtracting local min for robustness
                    pmin = float(np.nanmin(p))
                    # fast count on tiny image
                    c, ecc = compute_star_count_fast_preview(p - pmin)
                    med = float(np.median(p - pmin))
                    return fp, float(means[i]), med, c, ecc

                star_workers = min(max_workers, 8)
                with ThreadPoolExecutor(max_workers=star_workers) as ex:
                    for fp, mean_v, med, c, ecc in ex.map(_star_job, enumerate(paths_ok)):
                        mean_values[fp] = mean_v
                        star_counts[fp] = {"count": int(c), "eccentricity": float(ecc)}
                        measured_frames.append(fp)

                del previews

            if not measured_frames:
                self.update_status("⚠️ No frames could be measured!")
                return

            self.update_status(f"✅ All chunks complete! Measured {len(measured_frames)} frames total.")
            QApplication.processEvents()

            # 3) Weights — keep your current logic (fast & good)
            self.update_status("⚖️ Computing frame weights…")
            dbg = ["\n📊 **Frame Weights Debug Log:**"]
            max_w = 0.0
            for fp in measured_frames:
                c   = star_counts[fp]["count"]
                ecc = star_counts[fp]["eccentricity"]
                m   = mean_values[fp]
                # same weighting you had during registration measurement
                c = max(c, 1)
                m = max(m, 1e-6)
                raw_w = (c * min(1.0, max(1.0 - ecc, 0.0))) / m
                self.frame_weights[fp] = raw_w
                max_w = max(max_w, raw_w)
                dbg.append(f"📂 {os.path.basename(fp)} → StarCount={c}, Ecc={ecc:.4f}, Mean={m:.4f}, Weight={raw_w:.4f}")

            if max_w > 0:
                for k in self.frame_weights:
                    self.frame_weights[k] /= max_w

            self.update_status("\n".join(dbg))
            self.update_status("✅ Frame weights computed!")
            QApplication.processEvents()

            # 4) Choose reference (optional for visual/log purposes)
            if getattr(self, "reference_frame", None):
                self.update_status(f"📌 Using user-specified reference: {self.reference_frame}")
            else:
                self.reference_frame = max(self.frame_weights, key=self.frame_weights.get)
                self.update_status(f"📌 Auto-selected reference: {self.reference_frame}")

            # 5) Clear transforms; not needed for already aligned frames
            self.valid_transforms = {}

            # 6) Hand off to your unified pipeline (this will stack without re-alignment)
            aligned_light_files = {g: lst for g, lst in self.light_files.items() if lst}
            self._run_mfdeconv_then_continue(aligned_light_files)
            return

        except Exception:
            self._set_registration_busy(False)
            raise


    @staticmethod
    def invert_affine_transform(matrix):
        """
        Inverts a 2x3 affine transformation matrix.
        Given matrix = [[a, b, tx],
                        [c, d, ty]],
        returns the inverse matrix.
        """
        A = matrix[:, :2]
        t = matrix[:, 2]
        A_inv = np.linalg.inv(A)
        t_inv = -A_inv @ t
        inv = np.hstack([A_inv, t_inv.reshape(2, 1)])
        return inv

    @staticmethod
    def apply_affine_transform_point(matrix, x, y):
        """
        Applies a 2x3 affine transformation to a point (x, y).
        Returns the transformed (x, y) coordinates.
        """
        point = np.array([x, y])
        result = matrix[:, :2] @ point + matrix[:, 2]
        return result[0], result[1]

    def _save_alignment_transforms_sasd_v2(
        self,
        *,
        out_path: str,
        ref_shape: tuple[int, int],
        ref_path: str,
        drizzle_xforms: dict,
        fallback_affine: dict,
    ):
        """
        Write SASD v2 with per-file KIND and proper matrix sizes.

        drizzle_xforms: {orig_path: (kind, matrix_or_None)}
            kind ∈ {"affine","homography","poly3","poly4","tps","thin_plate_spline"} (case-insensitive)
            For poly*/tps, matrix must be None (we'll write a sentinel).

        fallback_affine: {orig_path: 2x3}
            Used only if we truly lack model info for a file, or the numeric
            matrix for an affine/homography entry is malformed.
        """

        # --- header dimensions ---
        try:
            Href, Wref = (int(ref_shape[0]), int(ref_shape[1]))
        except Exception:
            Href, Wref = 0, 0

        # --- normalize keys once ---
        def _normdict(d):
            return {os.path.normpath(k): v for k, v in (d or {}).items()}

        dx = _normdict(drizzle_xforms or {})
        fa = _normdict(fallback_affine or {})

        # Deterministic union of originals we know about
        originals = sorted(set(dx.keys()) | set(fa.keys()))

        # Helper to write non-matrix kinds (poly*/tps/unknown) with UNSUPPORTED sentinel
        def _write_poly_or_unknown(_f, _k, _kind_text: str):
            _f.write(f"FILE: {_k}\n")
            _f.write(f"KIND: {_kind_text}\n")
            _f.write("MATRIX:\n")
            _f.write("UNSUPPORTED\n\n")

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"REF_SHAPE: {Href}, {Wref}\n")
            f.write(f"REF_PATH: {ref_path}\n")
            f.write("MODEL: mixed\n\n")

            for k in originals:
                kind, M = (None, None)
                if k in dx:
                    kind, M = dx[k]

                kind = (kind or "").strip().lower()

                # --- Decide whether to use fallback affine ---
                # We ONLY use fallback when:
                #   a) there is no kind at all, or
                #   b) kind is affine/homography but matrix is missing/malformed.
                use_fallback_affine = False
                if not kind:
                    use_fallback_affine = True
                elif kind in ("affine", "homography"):
                    # We'll validate below; mark for possible fallback
                    use_fallback_affine = (M is None)

                if use_fallback_affine and (k in fa):
                    kind = "affine"
                    M = np.asarray(fa[k], np.float32).reshape(2, 3)

                # --- Write by kind ---
                if kind == "homography":
                    # Expect 3x3; if bad, try fallback or write UNSUPPORTED homography
                    try:
                        H = np.asarray(M, np.float64).reshape(3, 3)
                    except Exception:
                        if k in fa:
                            # Fallback to affine numbers if available
                            A = np.asarray(fa[k], np.float64).reshape(2, 3)
                            f.write(f"FILE: {k}\n")
                            f.write("KIND: affine\n")
                            f.write("MATRIX:\n")
                            f.write(f"{A[0,0]:.9f}, {A[0,1]:.9f}, {A[0,2]:.9f}\n")
                            f.write(f"{A[1,0]:.9f}, {A[1,1]:.9f}, {A[1,2]:.9f}\n\n")
                        else:
                            _write_poly_or_unknown(f, k, "homography")
                        continue

                    f.write(f"FILE: {k}\n")
                    f.write("KIND: homography\n")
                    f.write("MATRIX:\n")
                    f.write(f"{H[0,0]:.9f}, {H[0,1]:.9f}, {H[0,2]:.9f}\n")
                    f.write(f"{H[1,0]:.9f}, {H[1,1]:.9f}, {H[1,2]:.9f}\n")
                    f.write(f"{H[2,0]:.9f}, {H[2,1]:.9f}, {H[2,2]:.9f}\n\n")
                    continue

                if kind == "affine":
                    # Expect 2x3; if bad, try fallback or write UNSUPPORTED affine
                    try:
                        A = np.asarray(M, np.float64).reshape(2, 3)
                    except Exception:
                        if k in fa:
                            A = np.asarray(fa[k], np.float64).reshape(2, 3)
                        else:
                            _write_poly_or_unknown(f, k, "affine")
                            continue

                    f.write(f"FILE: {k}\n")
                    f.write("KIND: affine\n")
                    f.write("MATRIX:\n")
                    f.write(f"{A[0,0]:.9f}, {A[0,1]:.9f}, {A[0,2]:.9f}\n")
                    f.write(f"{A[1,0]:.9f}, {A[1,1]:.9f}, {A[1,2]:.9f}\n\n")
                    continue

                # Non-matrix kinds we want to preserve (identity deposit on _n_r):
                if kind in ("poly3", "poly4", "tps", "thin_plate_spline"):
                    _write_poly_or_unknown(f, k, kind)
                    continue

                # Unknown but present -> keep KIND so loader can choose strategy; mark matrix unsupported
                if kind:
                    _write_poly_or_unknown(f, k, kind)
                    continue

                # Truly nothing for this file: skip writing a broken block
                # (drizzle will simply not see this frame)
                continue



    def drizzle_stack_one_group(
        self,
        *,
        group_key,
        file_list,           # registered _n_r.fit (keep for headers/metadata)
        original_list,       # NEW: originals (normalized), used as pixel sources
        transforms_dict,
        frame_weights,
        scale_factor,
        drop_shrink,
        rejection_map,
        autocrop_enabled,
        rect_override,
        status_cb
    ):
        # Load per-frame transforms (SASD v2)
        log = status_cb or (lambda *_: None)        
        sasd_path = os.path.join(self.stacking_directory, "alignment_transforms.sasd")
        ref_H, ref_W, xforms = self._load_sasd_v2(sasd_path)
        if not (ref_H and ref_W):
            # Fallback to in-memory shape captured at alignment time
            rs = getattr(self, "ref_shape_for_drizzle", None)
            if isinstance(rs, tuple) and len(rs) == 2 and all(int(v) > 0 for v in rs):
                ref_H, ref_W = int(rs[0]), int(rs[1])
                status_cb(f"ℹ️ Using in-memory REF_SHAPE fallback: {ref_H}×{ref_W}")
            else:
                status_cb("⚠️ Missing REF_SHAPE in SASD; cannot drizzle.")
                return

        log(f"✅ SASD v2: loaded {len(xforms)} transform(s).")
        # Debug (first few):
        try:
            sample_need = [os.path.basename(p) for p in original_list[:5]]
            sample_have = [os.path.basename(p) for p in list(xforms.keys())[:5]]
            log(f"   originals needed (sample): {sample_need}")
            log(f"   sasd FILEs (sample):       {sample_have}")
        except Exception:
            pass

        canvas_H, canvas_W = int(ref_H * scale_factor), int(ref_W * scale_factor)        


        # --- kernel config from settings ---
        kernel_name = self.settings.value("stacking/drizzle_kernel", "square", type=str).lower()
        gauss_sigma = self.settings.value(
            "stacking/drizzle_gauss_sigma", float(drop_shrink) * 0.5, type=float
        )
        if kernel_name.startswith("gauss"):
            _kcode = 2
        elif kernel_name.startswith("circ"):
            _kcode = 1
        else:
            _kcode = 0  # square

        total_rej = sum(len(v) for v in (rejection_map or {}).values())
        log(f"🔭 Drizzle stacking for group '{group_key}' with {total_rej} total rejected pixels.")

        if len(file_list) < 2:
            log(f"⚠️ Group '{group_key}' does not have enough frames to drizzle.")
            return

        # --- establish geometry + is_mono before choosing depositor ---
        first_file = file_list[0]
        first_img, hdr, _, _ = load_image(first_file)
        if first_img is None:
            log(f"⚠️ Could not load {first_file} to determine drizzle shape!")
            return

        if first_img.ndim == 2:
            is_mono = True
            h, w = first_img.shape
            c = 1
        else:
            is_mono = False
            h, w, c = first_img.shape

        # --- choose depositor ONCE (and log it) ---
        if _kcode == 0 and drop_shrink >= 0.99:
            # square + pixfrac≈1 → naive “one-to-one” deposit
            deposit_func = drizzle_deposit_numba_naive if is_mono else drizzle_deposit_color_naive
            kinf = "naive (square, pixfrac≈1)"
        else:
            # Any other case → kernelized path (square/circular/gaussian)
            deposit_func = drizzle_deposit_numba_kernel_mono if is_mono else drizzle_deposit_color_kernel
            kinf = ["square", "circular", "gaussian"][_kcode]
        log(f"Using {kinf} kernel drizzle ({'mono' if is_mono else 'color'}).")

        # --- allocate buffers ---
        out_h = int(canvas_H)
        out_w = int(canvas_W)
        drizzle_buffer  = np.zeros((out_h, out_w) if is_mono else (out_h, out_w, c), dtype=self._dtype())
        coverage_buffer = np.zeros_like(drizzle_buffer, dtype=self._dtype())
        finalize_func   = finalize_drizzle_2d if is_mono else finalize_drizzle_3d

        def _invert_2x3(A23: np.ndarray) -> np.ndarray:
            A23 = np.asarray(A23, np.float32).reshape(2, 3)
            A33 = np.eye(3, dtype=np.float32); A33[:2] = A23
            return np.linalg.inv(A33)  # 3x3

        def _apply_H_point(H: np.ndarray, x: float, y: float) -> tuple[int, int]:
            v = H @ np.array([x, y, 1.0], dtype=np.float32)
            if abs(v[2]) < 1e-8:
                return (int(round(v[0])), int(round(v[1])))
            return (int(round(v[0] / v[2])), int(round(v[1] / v[2])))


        # --- main loop ---
        # Map original (normalized) → aligned path (for weights & rejection map lookups)
        orig_to_aligned = {}
        for op in original_list:
            ap = transforms_dict.get(os.path.normpath(op))
            if ap:
                orig_to_aligned[os.path.normpath(op)] = os.path.normpath(ap)

        for orig_file in original_list:
            orig_key = os.path.normpath(orig_file)
            aligned_file = orig_to_aligned.get(orig_key)

            weight = frame_weights.get(aligned_file, frame_weights.get(orig_key, 1.0))

            kind, X = xforms.get(orig_key, (None, None))
            log(f"🧭 Drizzle uses {kind or '-'} for {os.path.basename(orig_key)}")
            if kind is None:
                log(f"⚠️ No usable transform for {os.path.basename(orig_file)} – skipping")
                continue

            # --- choose pixel source + mapping ---
            pixels_are_registered = False
            img_data = None

            if isinstance(kind, str) and (kind.startswith("poly") or kind in ("tps","thin_plate_spline")):
                # Already warped to reference during registration
                pixel_path = aligned_file
                if not pixel_path:
                    log(f"⚠️ {kind} frame has no aligned counterpart – skipping {os.path.basename(orig_file)}")
                    continue
                H_canvas = np.eye(3, dtype=np.float32)
                pixels_are_registered = True

            elif kind == "affine" and X is not None:
                pixel_path = orig_file
                H_canvas = np.eye(3, dtype=np.float32)
                H_canvas[:2] = np.asarray(X, np.float32).reshape(2, 3)

            elif kind == "homography" and X is not None:
                # Pre-warp originals -> reference, then identity deposit
                pixel_path = orig_file
                raw_img, _, _, _ = load_image(pixel_path)
                if raw_img is None:
                    log(f"⚠️ Failed to read {os.path.basename(pixel_path)} – skipping")
                    continue
                H = np.asarray(X, np.float32).reshape(3, 3)
                if raw_img.ndim == 2:
                    img_data = cv2.warpPerspective(
                        raw_img, H, (ref_W, ref_H),
                        flags=cv2.INTER_LANCZOS4,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0
                    )
                else:
                    img_data = np.stack([
                        cv2.warpPerspective(
                            raw_img[..., ch], H, (ref_W, ref_H),
                            flags=cv2.INTER_LANCZOS4,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0
                        ) for ch in range(raw_img.shape[2])
                    ], axis=2)
                H_canvas = np.eye(3, dtype=np.float32)
                pixels_are_registered = True

            else:
                log(f"⚠️ Unsupported transform '{kind}' – skipping {os.path.basename(orig_file)}")
                continue

            # read pixels if not produced above
            if img_data is None:
                img_data, _, _, _ = load_image(pixel_path)
                if img_data is None:
                    log(f"⚠️ Failed to read {os.path.basename(pixel_path)} – skipping")
                    continue

            # --- debug bbox once ---
            if orig_file is original_list[0]:
                x0, y0 = 0, 0; x1, y1 = ref_W-1, ref_H-1
                p0 = H_canvas @ np.array([x0, y0, 1], np.float32); p0 /= max(p0[2], 1e-8)
                p1 = H_canvas @ np.array([x1, y1, 1], np.float32); p1 /= max(p1[2], 1e-8)
                log(f"   bbox(ref)→reg: ({p0[0]:.1f},{p0[1]:.1f}) to ({p1[0]:.1f},{p1[1]:.1f}); "
                    f"canvas {int(ref_W*scale_factor)}×{int(ref_H*scale_factor)} @ {scale_factor}×")

            # --- apply per-file rejections ---
            if rejection_map and aligned_file in rejection_map:
                coords_for_this_file = rejection_map.get(aligned_file, [])
                if coords_for_this_file:
                    dilate_px = int(self.settings.value("stacking/reject_dilate_px", 0, type=int))
                    dilate_shape = self.settings.value("stacking/reject_dilate_shape", "square", type=str).lower()
                    offsets = [(0, 0)]
                    if dilate_px > 0:
                        r = dilate_px
                        offsets = [(dx, dy) for dx in range(-r, r+1) for dy in range(-r, r+1)]
                        if dilate_shape.startswith("dia"):
                            offsets = [(dx, dy) for (dx,dy) in offsets if (abs(dx)+abs(dy) <= r)]

                    Hraw, Wraw = img_data.shape[0], img_data.shape[1]

                    if pixels_are_registered:
                        # Directly zero registered pixels
                        for (x_r, y_r) in coords_for_this_file:
                            for (ox, oy) in offsets:
                                xr, yr = x_r + ox, y_r + oy
                                if 0 <= xr < Wraw and 0 <= yr < Hraw:
                                    img_data[yr, xr] = 0.0
                    else:
                        # Back-project via inverse affine
                        Hinv = np.linalg.inv(H_canvas)
                        for (x_r, y_r) in coords_for_this_file:
                            for (ox, oy) in offsets:
                                xr, yr = x_r + ox, y_r + oy
                                v = Hinv @ np.array([xr, yr, 1.0], np.float32)
                                x_raw = int(round(v[0] / max(v[2], 1e-8)))
                                y_raw = int(round(v[1] / max(v[2], 1e-8)))
                                if 0 <= x_raw < Wraw and 0 <= y_raw < Hraw:
                                    img_data[y_raw, x_raw] = 0.0

            # --- deposit (identity for registered pixels) ---
            if deposit_func is drizzle_deposit_numba_naive:
                drizzle_buffer, coverage_buffer = deposit_func(
                    img_data, H_canvas[:2], drizzle_buffer, coverage_buffer, scale_factor, weight
                )
            elif deposit_func is drizzle_deposit_color_naive:
                drizzle_buffer, coverage_buffer = deposit_func(
                    img_data, H_canvas[:2], drizzle_buffer, coverage_buffer, scale_factor, drop_shrink, weight
                )
            else:
                A23 = H_canvas[:2, :]
                drizzle_buffer, coverage_buffer = deposit_func(
                    img_data, A23, drizzle_buffer, coverage_buffer,
                    scale_factor, drop_shrink, weight, _kcode, float(gauss_sigma)
                )


        # --- finalize, save, optional autocrop ---
        final_drizzle = np.zeros_like(drizzle_buffer, dtype=np.float32)
        final_drizzle = finalize_func(drizzle_buffer, coverage_buffer, final_drizzle)

        # Save original drizzle (single-HDU; no rejection layers here)
        Hd, Wd = final_drizzle.shape[:2] if final_drizzle.ndim >= 2 else (0, 0)
        display_group_driz = self._label_with_dims(group_key, Wd, Hd)
        base_stem = f"MasterLight_{display_group_driz}_{len(file_list)}stacked_drizzle"
        base_stem = self._normalize_master_stem(base_stem) 
        out_path_orig = self._build_out(self.stacking_directory, base_stem, "fit")

        hdr_orig = hdr.copy() if hdr is not None else fits.Header()
        hdr_orig["IMAGETYP"]   = "MASTER STACK - DRIZZLE"
        hdr_orig["DRIZFACTOR"] = (float(scale_factor), "Drizzle scale factor")
        hdr_orig["DROPFRAC"]   = (float(drop_shrink),  "Drizzle drop shrink/pixfrac")
        hdr_orig["CREATOR"]    = "SetiAstroSuite"
        hdr_orig["DATE-OBS"]   = datetime.utcnow().isoformat()

        if final_drizzle.ndim == 2:
            hdr_orig["NAXIS"]  = 2
            hdr_orig["NAXIS1"] = final_drizzle.shape[1]
            hdr_orig["NAXIS2"] = final_drizzle.shape[0]
            if "NAXIS3" in hdr_orig:
                del hdr_orig["NAXIS3"]
        else:
            hdr_orig["NAXIS"]  = 3
            hdr_orig["NAXIS1"] = final_drizzle.shape[1]
            hdr_orig["NAXIS2"] = final_drizzle.shape[0]
            hdr_orig["NAXIS3"] = final_drizzle.shape[2]

        is_mono_driz = (final_drizzle.ndim == 2)

        save_image(
            img_array=final_drizzle,
            filename=out_path_orig,
            original_format="fit",
            bit_depth="32-bit floating point",
            original_header=hdr_orig,
            is_mono=is_mono_driz
        )
        log(f"✅ Drizzle (original) saved: {out_path_orig}")

        # Optional auto-crop (respects global rect if provided)
        if autocrop_enabled:
            cropped_drizzle, hdr_crop = self._apply_autocrop(
                final_drizzle,
                file_list,
                hdr.copy() if hdr is not None else fits.Header(),
                scale=float(scale_factor),
                rect_override=rect_override
            )
            is_mono_crop = (cropped_drizzle.ndim == 2)
            display_group_driz_crop = self._label_with_dims(group_key, cropped_drizzle.shape[1], cropped_drizzle.shape[0])
            base_crop = f"MasterLight_{display_group_driz_crop}_{len(file_list)}stacked_drizzle_autocrop"
            base_crop = self._normalize_master_stem(base_crop) 
            out_path_crop = self._build_out(self.stacking_directory, base_crop, "fit")

            save_image(
                img_array=cropped_drizzle,
                filename=out_path_crop,
                original_format="fit",
                bit_depth="32-bit floating point",
                original_header=hdr_crop,
                is_mono=is_mono_crop
            )
            if not hasattr(self, "_autocrop_outputs"):
                self._autocrop_outputs = []
            self._autocrop_outputs.append((group_key, out_path_crop))
            log(f"✂️ Drizzle (auto-cropped) saved: {out_path_crop}")

    def _load_sasd_v2(self, path: str):
        """
        Returns (ref_H, ref_W, xforms) where xforms maps
        normalized original paths → (kind, matrix)
        Supports:
        • HEADER: REF_SHAPE, REF_PATH, MODEL
        • Per-file: FILE, KIND, MATRIX with either 2 rows (affine) or 3 rows (homography)
        """
        if not os.path.exists(path):
            return 0, 0, {}

        ref_H = ref_W = 0
        xforms = {}
        cur_file = None
        cur_kind = None
        reading_matrix = False
        mat_rows = []
        matrix_unsupported = False   # <— NEW

        def _commit_block():
            nonlocal cur_file, cur_kind, mat_rows, matrix_unsupported
            if cur_file and cur_kind:
                k = os.path.normpath(cur_file)
                ck = (cur_kind or "").lower()

                if ck == "homography" and (mat_rows and len(mat_rows) == 3 and len(mat_rows[0]) == 3):
                    M = np.array(mat_rows, dtype=np.float32).reshape(3, 3)
                    xforms[k] = ("homography", M)

                elif ck == "affine" and (mat_rows and len(mat_rows) == 2 and len(mat_rows[0]) == 3):
                    M = np.array(mat_rows, dtype=np.float32).reshape(2, 3)
                    xforms[k] = ("affine", M)

                elif ck.startswith("poly"):
                    # poly3 / poly4 are callable in-memory; SASD stores no numeric matrix
                    xforms[k] = (ck, None)

                elif matrix_unsupported:
                    # Future kinds that explicitly say UNSUPPORTED: keep kind, no matrix
                    xforms[k] = (ck, None)

            cur_file = None
            cur_kind = None
            mat_rows = []
            matrix_unsupported = False  # <— reset

        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    if reading_matrix:
                        reading_matrix = False
                    _commit_block()
                    continue

                if reading_matrix:
                    # Handle explicit sentinel
                    if line.upper().startswith("UNSUPPORTED"):
                        matrix_unsupported = True          # <— flag it
                        reading_matrix = False
                        continue
                    # Try numeric row
                    try:
                        row = [float(x.strip()) for x in line.split(",")]
                        mat_rows.append(row)
                    except Exception:
                        # Non-numeric inside MATRIX: treat as unsupported for safety
                        matrix_unsupported = True
                        reading_matrix = False
                    continue

                if line.startswith("REF_SHAPE:"):
                    ...
                    continue

                if line.startswith("FILE:"):
                    _commit_block()
                    cur_file = line.split(":", 1)[1].strip()
                    continue

                if line.startswith("KIND:"):
                    cur_kind = line.split(":", 1)[1].strip().lower()
                    continue

                if line.startswith("MATRIX:"):
                    reading_matrix = True
                    mat_rows = []
                    matrix_unsupported = False
                    continue

                if line.startswith("MODEL:"):
                    continue

        if reading_matrix or cur_file:
            _commit_block()

        return int(ref_H), int(ref_W), xforms



    def normal_integration_with_rejection(
        self,
        group_key,
        file_list,
        frame_weights,
        status_cb=None,
        *,
        algo_override: str | None = None
    ):
        log = status_cb or (lambda *_: None)
        log(f"Starting integration for group '{group_key}' with {len(file_list)} files.")
        if not file_list:
            return None, {}, None

        # --- reference frame (unchanged) ---
        ref_file = file_list[0]
        ref_data, ref_header, _, _ = load_image(ref_file)
        if ref_data is None:
            log(f"⚠️ Could not load reference '{ref_file}' for group '{group_key}'.")
            return None, {}, None
        if ref_header is None:
            ref_header = fits.Header()

        is_color = (ref_data.ndim == 3 and ref_data.shape[2] == 3)
        height, width = ref_data.shape[:2]
        channels = 3 if is_color else 1
        N = len(file_list)

        algo = (algo_override or self.rejection_algorithm)
        use_gpu = bool(self._hw_accel_enabled()) and _torch_ok() and _gpu_algo_supported(algo)

        log(f"📊 Stacking group '{group_key}' with {algo}{' [GPU]' if use_gpu else ''}")

        # --- keep all FITSes open (memmap) once for the whole group ---
        sources = []
        try:
            for p in file_list:
                sources.append(_MMImage(p))   # << was _MMFits
        except Exception as e:
            for s in sources:
                try: s.close()
                except Exception: pass
            log(f"⚠️ Failed to open images (memmap): {e}")
            return None, {}, None

        DTYPE = self._dtype()
        integrated_image = np.zeros((height, width, channels), dtype=DTYPE)
        per_file_rejections = {f: [] for f in file_list}

        # --- chunk size ---
        pref_h = self.chunk_height
        pref_w = self.chunk_width
        try:
            chunk_h, chunk_w = compute_safe_chunk(height, width, N, channels, DTYPE, pref_h, pref_w)
            log(f"🔧 Using chunk size {chunk_h}×{chunk_w} for {DTYPE}")
        except MemoryError as e:
            for s in sources: s.close()
            log(f"⚠️ {e}")
            return None, {}, None

        # --- reusable C-order tile buffers (avoid copies before GPU) ---
        # Use pinned memory only if we’ll actually ship tiles to GPU.
        def _mk_buf():
            buf = np.empty((N, chunk_h, chunk_w, channels), dtype=np.float32, order='C')
            if use_gpu:
                # Mark as pinned (page-locked) so torch can H->D quickly if _torch_reduce_tile uses it.
                try:
                    import torch
                    # torch can wrap numpy pinned via from_numpy(...).pin_memory() ONLY on tensors.
                    # We'll keep numpy here; _torch_reduce_tile will move to pinned tensors internally.
                    # So no-op here to avoid extra copies.
                except Exception:
                    pass
            return buf

        buf0 = _mk_buf()
        buf1 = _mk_buf()

        # --- weights once per group ---
        weights_array = np.array([frame_weights.get(p, 1.0) for p in file_list], dtype=np.float32)

        n_rows  = math.ceil(height / chunk_h)
        n_cols  = math.ceil(width  / chunk_w)
        total_tiles = n_rows * n_cols

        # --- group-level rejection maps (RAM-light) ---
        rej_any   = np.zeros((height, width), dtype=np.bool_)
        rej_count = np.zeros((height, width), dtype=np.uint16)

        # --------- helper: read a tile into a provided buffer (blocking) ----------
        def _read_tile_into(buf, y0, y1, x0, x1):
            th = y1 - y0
            tw = x1 - x0
            # slice view (C-order)
            ts = buf[:N, :th, :tw, :channels]
            # sequential, low-overhead sliced reads (OS prefetch + memmap)
            for i, src in enumerate(sources):
                sub = src.read_tile(y0, y1, x0, x1)  # float32, (th,tw) or (th,tw,3)
                if sub.ndim == 2:
                    if channels == 3:
                        sub = sub[:, :, None].repeat(3, axis=2)
                    else:
                        sub = sub[:, :, None]
                ts[i, :, :, :] = sub
            return th, tw  # actual extents for edge tiles

        # Prefetcher (single background worker is enough; IO is the bottleneck)
        from concurrent.futures import ThreadPoolExecutor
        tp = ThreadPoolExecutor(max_workers=1)

        # Precompute tile grid
        tiles = []
        for y0 in range(0, height, chunk_h):
            y1 = min(y0 + chunk_h, height)
            for x0 in range(0, width, chunk_w):
                x1 = min(x0 + chunk_w, width)
                tiles.append((y0, y1, x0, x1))

        # Prime first read
        tile_idx = 0
        y0, y1, x0, x1 = tiles[0]
        fut = tp.submit(_read_tile_into, buf0, y0, y1, x0, x1)
        use_buf0 = True

        # Torch inference guard (if available)
        _ctx = _safe_torch_inference_ctx() if use_gpu else contextlib.nullcontext
        with _ctx():
            for tile_idx, (y0, y1, x0, x1) in enumerate(tiles, start=1):
                t0 = time.perf_counter()

                # Wait for current tile to be ready
                th, tw = fut.result()
                ts = (buf0 if use_buf0 else buf1)[:N, :th, :tw, :channels]

                # Kick off prefetch for the NEXT tile (if any) into the other buffer
                if tile_idx < total_tiles:
                    ny0, ny1, nx0, nx1 = tiles[tile_idx]
                    fut = tp.submit(_read_tile_into, (buf1 if use_buf0 else buf0), ny0, ny1, nx0, nx1)

                # --- rejection/integration for this tile ---
                log(f"Integrating tile {tile_idx}/{total_tiles} "
                    f"[y:{y0}:{y1} x:{x0}:{x1} size={th}×{tw}] "
                    f"mode={'GPU' if use_gpu else 'CPU'}…")

                if use_gpu:
                    tile_result, tile_rej_map = _torch_reduce_tile(
                        ts,                         # NumPy view, C-contiguous
                        weights_array,              # (N,)
                        algo_name=algo,
                        kappa=float(self.kappa),
                        iterations=int(self.iterations),
                        sigma_low=float(self.sigma_low),
                        sigma_high=float(self.sigma_high),
                        trim_fraction=float(self.trim_fraction),
                        esd_threshold=float(self.esd_threshold),
                        biweight_constant=float(self.biweight_constant),
                        modz_threshold=float(self.modz_threshold),
                        comet_hclip_k=float(self.settings.value("stacking/comet_hclip_k", 1.30, type=float)),
                        comet_hclip_p=float(self.settings.value("stacking/comet_hclip_p", 25.0, type=float)),
                    )
                    # _torch_reduce_tile should already return NumPy; if it returns tensors, convert here.
                    if hasattr(tile_result, "detach"):
                        tile_result = tile_result.detach().cpu().numpy()
                    if hasattr(tile_rej_map, "detach"):
                        tile_rej_map = tile_rej_map.detach().cpu().numpy()
                else:
                    # CPU path (NumPy/Numba)
                    if algo in ("Comet Median", "Simple Median (No Rejection)"):
                        tile_result  = np.median(ts, axis=0)
                        tile_rej_map = np.zeros((N, th, tw), dtype=bool)

                    elif algo == "Comet High-Clip Percentile":
                        k = self.settings.value("stacking/comet_hclip_k", 1.30, type=float)
                        p = self.settings.value("stacking/comet_hclip_p", 25.0, type=float)
                        tile_result  = _high_clip_percentile(ts, k=float(k), p=float(p))
                        tile_rej_map = np.zeros((N, th, tw), dtype=bool)

                    elif algo == "Comet Lower-Trim (30%)":
                        tile_result  = _lower_trimmed_mean(ts, trim_hi_frac=0.30)
                        tile_rej_map = np.zeros((N, th, tw), dtype=bool)

                    elif algo == "Comet Percentile (40th)":
                        tile_result  = _percentile40(ts)
                        tile_rej_map = np.zeros((N, th, tw), dtype=bool)

                    elif algo == "Simple Average (No Rejection)":
                        tile_result  = np.average(ts, axis=0, weights=weights_array)
                        tile_rej_map = np.zeros((N, th, tw), dtype=bool)

                    elif algo == "Weighted Windsorized Sigma Clipping":
                        tile_result, tile_rej_map = windsorized_sigma_clip_weighted(
                            ts, weights_array, lower=self.sigma_low, upper=self.sigma_high
                        )
                    elif algo == "Kappa-Sigma Clipping":
                        tile_result, tile_rej_map = kappa_sigma_clip_weighted(
                            ts, weights_array, kappa=self.kappa, iterations=self.iterations
                        )
                    elif algo == "Trimmed Mean":
                        tile_result, tile_rej_map = trimmed_mean_weighted(
                            ts, weights_array, trim_fraction=self.trim_fraction
                        )
                    elif algo == "Extreme Studentized Deviate (ESD)":
                        tile_result, tile_rej_map = esd_clip_weighted(
                            ts, weights_array, threshold=self.esd_threshold
                        )
                    elif algo == "Biweight Estimator":
                        tile_result, tile_rej_map = biweight_location_weighted(
                            ts, weights_array, tuning_constant=self.biweight_constant
                        )
                    elif algo == "Modified Z-Score Clipping":
                        tile_result, tile_rej_map = modified_zscore_clip_weighted(
                            ts, weights_array, threshold=self.modz_threshold
                        )
                    elif algo == "Max Value":
                        tile_result, tile_rej_map = max_value_stack(ts, weights_array)
                    else:
                        tile_result, tile_rej_map = windsorized_sigma_clip_weighted(
                            ts, weights_array, lower=self.sigma_low, upper=self.sigma_high
                        )

                # write back
                integrated_image[y0:y1, x0:x1, :] = tile_result

                # --- rejection bookkeeping ---
                trm = tile_rej_map
                if trm.ndim == 4:
                    trm = np.any(trm, axis=-1)  # collapse color → (N, th, tw)

                rej_any[y0:y1, x0:x1]  |= np.any(trm, axis=0)
                rej_count[y0:y1, x0:x1] += trm.sum(axis=0).astype(np.uint16)

                # per-file coords (existing behavior)
                for i, fpath in enumerate(file_list):
                    ys, xs = np.where(trm[i])
                    if ys.size:
                        per_file_rejections[fpath].extend(zip(x0 + xs, y0 + ys))

                # perf log
                dt = time.perf_counter() - t0
                # simple “work” metric: pixels processed (× frames × channels)
                work_px = th * tw * N * channels
                mpx_s = (work_px / 1e6) / dt if dt > 0 else float("inf")
                log(f"  ↳ tile {tile_idx} done in {dt:.3f}s  (~{mpx_s:.1f} MPx/s)")

                # flip buffer
                use_buf0 = not use_buf0

        # close mmapped FITSes and prefetch pool
        tp.shutdown(wait=True)
        for s in sources:
            s.close()

        if channels == 1:
            integrated_image = integrated_image[..., 0]

        # stash group-level maps
        if not hasattr(self, "_rej_maps"):
            self._rej_maps = {}
        rej_frac = (rej_count.astype(np.float32) / float(max(1, N)))  # [0..1]
        self._rej_maps[group_key] = {"any": rej_any, "frac": rej_frac, "count": rej_count, "n": N}

        log(f"Integration complete for group '{group_key}'.")
        try:
            _free_torch_memory()
        except:
            pass    
        return integrated_image, per_file_rejections, ref_header


    def _safe_component(self, s: str, *, replacement:str="_", maxlen:int=180) -> str:
        """
        Sanitize a *single* path component for cross-platform safety.
        - normalizes unicode (NFKC)
        - replaces path separators and illegal chars
        - collapses whitespace to `_`
        - strips leading/trailing dots/spaces (Windows rule)
        - avoids reserved device names (Windows)
        - truncates to maxlen, keeping extension if present
        """
        s = unicodedata.normalize("NFKC", str(s))

        # nuke path separators
        other_sep = "/" if os.sep == "\\" else "\\"
        s = s.replace(os.sep, replacement).replace(other_sep, replacement)

        # replace illegal Windows chars + control chars
        s = re.sub(r'[<>:"/\\|?*\x00-\x1F]', replacement, s)

        # collapse whitespace → _
        s = re.sub(r"\s+", replacement, s)

        # allow only [A-Za-z0-9._-()] plus 'x' (for 1234x5678) — tweak if you want
        s = re.sub(r"[^A-Za-z0-9._\-()x]", replacement, s)

        # collapse multiple replacements
        rep = re.escape(replacement)
        s = re.sub(rep + r"+", replacement, s)

        # trim leading/trailing spaces/dots/dashes/underscores
        s = s.strip(" .-_")

        if not s:
            s = "untitled"

        # avoid reserved basenames on Windows (compare stem only)
        stem, ext = os.path.splitext(s)
        if stem.upper() in _WINDOWS_RESERVED:
            s = "_" + s  # prefix underscore

        # enforce maxlen, preserving extension if present
        if len(s) > maxlen:
            stem, ext = os.path.splitext(s)
            keep = max(1, maxlen - len(ext))
            s = stem[:keep].rstrip(" .-_") + ext

        return s

    def _normalize_master_stem(self, stem: str) -> str:
        """
        Clean up common artifacts in master filenames:
        - collapse _-_ / -_ / _- into a single _
        - turn 40.0s → 40s (strip trailing .0…)
        - keep non-integer exposures filename-safe (e.g., 2.5s → 2p5s)
        """
        # 1) collapse weird joiners like "_-_" or "-_" or "_-"
        stem = re.sub(r'(?:_-+|-+_)+', '_', stem)

        # 2) normalize exposures: <number>s
        def _fix_exp(m):
            txt = m.group(1)  # the numeric part
            try:
                val = float(txt)
            except ValueError:
                return m.group(0)  # leave it as-is

            # If it's an integer (e.g., 40.0) → 40s
            if abs(val - round(val)) < 1e-6:
                return f"{int(round(val))}s"
            # Otherwise make it filename-friendly by replacing '.' with '_' → 2_5s
            return txt.replace('.', '_') + 's'

        stem = re.sub(r'(\d+(?:\.\d+)?)s', _fix_exp, stem)

        stem = re.sub(r'\((\d+)x(\d+)\)', r'\1x\2', stem)
        return stem

    def _build_out(self, directory: str, stem: str, ext: str) -> str:
        """
        Join directory + sanitized stem + sanitized extension.
        Ensures parent dir exists.
        """
        ext = (ext or "").lstrip(".").lower() or "fit"
        safe_stem = self._safe_component(stem)
        safe_dir  = os.path.abspath(directory or ".")
        os.makedirs(safe_dir, exist_ok=True)
        return os.path.join(safe_dir, f"{safe_stem}.{ext}")