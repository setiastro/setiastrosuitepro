#pro.stacking_suite.py
from __future__ import annotations
import os, glob, shutil, tempfile, datetime as _dt
from time import perf_counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import hashlib
from numpy.lib.format import open_memmap 
import re, unicodedata
import math            # used in compute_safe_chunk
import psutil          # used in bytes_available / compute_safe_chunk
from typing import List 
from PyQt6.QtCore import Qt, QTimer, QSettings, pyqtSignal, QObject, pyqtSlot, QThread, QEvent, QPoint, QSize, QEventLoop, QCoreApplication, QRectF, QPointF, QMetaObject
from PyQt6.QtGui import QIcon, QImage, QPixmap, QAction, QIntValidator, QDoubleValidator, QFontMetrics, QTextCursor, QPalette, QPainter, QPen, QTransform, QColor, QBrush, QCursor
from PyQt6.QtWidgets import (QDialog, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QLineEdit, QTreeWidget, QHeaderView, QTreeWidgetItem, QProgressBar, QProgressDialog,
                             QFormLayout, QDialogButtonBox, QToolBar, QToolButton, QFileDialog, QTabWidget, QAbstractItemView, QSpinBox, QDoubleSpinBox, QGroupBox,
                             QSizePolicy, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QApplication, QScrollArea, QTextEdit, QMenu, QPlainTextEdit, QGraphicsEllipseItem,
                             QMessageBox, QSlider, QCheckBox, QInputDialog, QComboBox)
from datetime import datetime, timezone
import pyqtgraph as pg
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from PIL import Image                 # used by _get_image_size (and elsewhere)
import tifffile as tiff               # _get_image_size -> tiff.imread(...)
import cv2                            # _get_image_size -> cv2.imread(...)
cv2.setNumThreads(0)
CursorShape = Qt.CursorShape 
import rawpy

import exifread

from typing import List, Tuple, Optional
import sep
from pathlib import Path

from PyQt6 import sip
# your helpers/utilities
from imageops.stretch import stretch_mono_image, stretch_color_image
from legacy.numba_utils import *   
from legacy.image_manager import load_image, save_image, get_valid_header
from pro.star_alignment import StarRegistrationWorker, StarRegistrationThread, IDENTITY_2x3
from pro.log_bus import LogBus
from pro import comet_stacking as CS
#from pro.remove_stars import starnet_starless_from_array, darkstar_starless_from_array
from pro.mfdeconv import MultiFrameDeconvWorker
from pro.accel_installer import current_backend
from pro.accel_workers import AccelInstallWorker
from pro.runtime_torch import add_runtime_to_sys_path

_WINDOWS_RESERVED = {
    "CON","PRN","AUX","NUL",
    "COM1","COM2","COM3","COM4","COM5","COM6","COM7","COM8","COM9",
    "LPT1","LPT2","LPT3","LPT4","LPT5","LPT6","LPT7","LPT8","LPT9",
}

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



def _luma(img: np.ndarray) -> np.ndarray:
    """
    Return a float32 mono view: grayscale image → itself, RGB → luma, (H,W,1) → squeeze.
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
                     use_luma: bool = True) -> np.ndarray:
    """
    Min-offset + median-match normalization with NO clipping.
    For each frame f:
      - L = luma(f) if RGB else f
      - f0 = f - min(L)
      - gain = target_median / median(luma(f0))
      - out = f0 * gain

    Returns float32, C-contiguous array with same shape as input.
    """
    assert stack.ndim in (3, 4), "stack must be (F,H,W) or (F,H,W,3)"
    F = stack.shape[0]
    out = np.empty_like(stack, dtype=np.float32)
    eps = 1e-12

    def _L(a: np.ndarray) -> np.ndarray:
        if not use_luma:
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
    if scale <= 1:
        return img.astype(np.float32, copy=False)
    h, w = img.shape[:2]
    return cv2.resize(img, (max(1, w // scale), max(1, h // scale)),
                      interpolation=cv2.INTER_AREA).astype(np.float32, copy=False)

def _upscale_bg(bg_small: np.ndarray, oh: int, ow: int) -> np.ndarray:
    return cv2.resize(bg_small, (ow, oh), interpolation=cv2.INTER_LANCZOS4).astype(np.float32, copy=False)

def _to_luma(img: np.ndarray) -> np.ndarray:
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
    flat = gray_small.ravel()
    thresh = np.percentile(flat, 100 * (1 - exclusion_fraction))
    return (gray_small < thresh)

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
    gray = _to_luma(img_small) if img_small.ndim == 3 else img_small
    pts: list[tuple[int, int]] = []
    border = max(6, patch_size)  # keep patch fully inside

    def allowed(x, y):
        if exclusion_mask_small is None: return True
        return bool(exclusion_mask_small[min(max(0,y), H-1), min(max(0,x), W-1)])

    # corners
    for (x, y) in [(border, border), (W-border-1, border), (border, H-border-1), (W-border-1, H-border-1)]:
        if not allowed(x, y): continue
        nx, ny = _gradient_descent_to_dim_spot(gray, x, y, patch_size)
        if allowed(nx, ny): pts.append((nx, ny))

    # borders (5 along each side)
    xs = np.linspace(border, W-border-1, 5, dtype=int)
    ys = np.linspace(border, H-border-1, 5, dtype=int)
    for x in xs:
        if allowed(x, border):
            nx, ny = _gradient_descent_to_dim_spot(gray, x, border, patch_size); 
            if allowed(nx, ny): pts.append((nx, ny))
        if allowed(x, H-border-1):
            nx, ny = _gradient_descent_to_dim_spot(gray, x, H-border-1, patch_size); 
            if allowed(nx, ny): pts.append((nx, ny))
    for y in ys:
        if allowed(border, y):
            nx, ny = _gradient_descent_to_dim_spot(gray, border, y, patch_size); 
            if allowed(nx, ny): pts.append((nx, ny))
        if allowed(W-border-1, y):
            nx, ny = _gradient_descent_to_dim_spot(gray, W-border-1, y, patch_size); 
            if allowed(nx, ny): pts.append((nx, ny))

    # quartiles: choose dim locations avoiding bright half
    hh, ww = H // 2, W // 2
    quads = [
        (slice(0, hh),    slice(0, ww),    (0, 0)),
        (slice(0, hh),    slice(ww, W),    (ww, 0)),
        (slice(hh, H),    slice(0, ww),    (0, hh)),
        (slice(hh, H),    slice(ww, W),    (ww, hh)),
    ]
    for ysl, xsl, (x0, y0) in quads:
        sub = gray[ysl, xsl]
        mask_sub = _exclude_bright_regions(sub, exclusion_fraction=0.5)
        if exclusion_mask_small is not None:
            mask_sub &= exclusion_mask_small[ysl, xsl]
        elig = np.argwhere(mask_sub)
        if elig.size == 0:
            continue
        k = min(len(elig), max(1, num_samples // 4))
        sel = elig[np.random.choice(len(elig), k, replace=False)]
        for (yy, xx) in sel:
            gx, gy = x0 + int(xx), y0 + int(yy)
            nx, ny = _gradient_descent_to_dim_spot(gray, gx, gy, patch_size)
            if allowed(nx, ny): pts.append((nx, ny))

    if len(pts) == 0:
        # fall back to simple grid
        g = max(3, int(np.sqrt(max(16, num_samples))))
        xs = np.linspace(border, W-border-1, g, dtype=int)
        ys = np.linspace(border, H-border-1, g, dtype=int)
        pts = [(x, y) for y in ys for x in xs if allowed(x, y)]

    return np.asarray(pts, dtype=np.int32)

# ----- fit/eval on small image -----

def _fit_poly2_on_small(img_small: np.ndarray, pts_small: np.ndarray, patch_size: int) -> np.ndarray:
    """Fit degree-2 polynomial on luma of small image using patch medians at pts."""
    gray = _to_luma(img_small) if img_small.ndim == 3 else img_small
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
    mode: str = "subtract",         # "subtract" or "divide"
    num_samples: int = 120,          # like ABE
    downsample: int = 6,             # like ABE
    patch_size: int = 15,            # area median window on small image
    min_strength: float = 0.01,      # skip if gradient <1% of bg median (5–95% spread)
    gain_clip: tuple[float,float] = (0.2, 5.0),  # for divide mode
    exclusion_mask: np.ndarray | None = None,    # optional (H,W) bool
    log_fn=None
) -> np.ndarray:
    """
    ABE-like degree-2 background model:
      1) Downsample (AREA) for speed.
      2) Sample corners/borders/quartiles, avoid bright half, walk to dim spots.
      3) Fit deg-2 poly on small via patch medians at points.
      4) Upscale background and apply (subtract or divide) + median re-center.
      5) Luma-only fit; apply to all channels to avoid color shifts.
    """
    if image is None:
        return image
    img = np.asarray(image).astype(np.float32, copy=False)
    mono = (img.ndim == 2)

    H, W = img.shape[:2]
    t0 = perf_counter()

    # downsample image and (optional) mask
    img_small = _downsample_area(img, max(1, int(downsample)))
    mask_small = None
    if exclusion_mask is not None:
        mask_small = _downsample_area(exclusion_mask.astype(np.float32), max(1, int(downsample))) >= 0.5

    # generate points + fit
    pts_small = _generate_sample_points_small(img_small, num_samples=int(num_samples),
                                              patch_size=int(patch_size),
                                              exclusion_mask_small=mask_small)
    bg_small = _fit_poly2_on_small(img_small, pts_small, patch_size=int(patch_size))
    bg = _upscale_bg(bg_small, H, W)
    t1 = perf_counter()

    # quick strength check (like your guard)
    bg_med = float(np.nanmedian(bg)) or 1e-6
    p5, p95 = np.nanpercentile(bg, 5), np.nanpercentile(bg, 95)
    rel_amp = float((p95 - p5) / max(bg_med, 1e-6))
    if log_fn:
        log_fn(f"ABE poly2: samples={num_samples}, ds={downsample}, patch={patch_size} | "
               f"bg_med={bg_med:.6f}, rel_amp={rel_amp*100:.2f}% | {t1-t0:.3f}s")

    if rel_amp < float(min_strength):
        if log_fn: log_fn(f"ABE poly2: gradient weak ({rel_amp*100:.2f}% < {min_strength*100:.2f}%), skipping.")
        return img

    # apply on channels with *same* bg to avoid color shifts
    def _apply_sub(ch: np.ndarray) -> np.ndarray:
        med0 = float(np.nanmedian(ch)) or 1e-6
        out = ch - bg
        med1 = float(np.nanmedian(out)) or 1e-6
        out += (med0 - med1)           # ABE: re-center by adding a constant, no scaling
        return out

    def _apply_div(ch: np.ndarray) -> np.ndarray:
        med0 = float(np.nanmedian(ch)) or 1e-6
        norm_bg = bg / bg_med
        lo, hi = gain_clip
        norm_bg = np.clip(norm_bg, lo, hi)
        out = ch / norm_bg
        med1 = float(np.nanmedian(out)) or 1e-6
        out *= (med0 / med1)
        return out

    if mono:
        ch = img
        out = _apply_sub(ch) if mode.lower() == "subtract" else _apply_div(ch)
        return out.astype(np.float32, copy=False)

    # color (HWC)
    r = _apply_sub(img[...,0]) if mode.lower() == "subtract" else _apply_div(img[...,0])
    g = _apply_sub(img[...,1]) if mode.lower() == "subtract" else _apply_div(img[...,1])
    b = _apply_sub(img[...,2]) if mode.lower() == "subtract" else _apply_div(img[...,2])
    return np.stack([r,g,b], axis=-1).astype(np.float32, copy=False)

def remove_gradient_stack_abe(
    stack: np.ndarray,
    *,
    mode: str = "subtract",
    num_samples: int = 120,
    downsample: int = 6,
    patch_size: int = 15,
    min_strength: float = 0.01,
    gain_clip: tuple[float,float] = (0.2, 5.0),
    log_fn=None
) -> np.ndarray:
    F = int(stack.shape[0])
    out = np.empty_like(stack, dtype=np.float32)
    if log_fn:
        log_fn(f"🌀 ABE poly2 on {F} frame(s): mode={mode}, samples={num_samples}, ds={downsample}, "
               f"patch={patch_size}, min_strength={min_strength*100:.2f}%, gain_clip={gain_clip}")
    for i in range(F):
        out[i] = remove_poly2_gradient_abe(
            stack[i],
            mode=mode, num_samples=num_samples, downsample=downsample,
            patch_size=patch_size, min_strength=min_strength,
            gain_clip=gain_clip,
            log_fn=(lambda s, idx=i: log_fn(f"[{idx+1}/{F}] {s}")) if log_fn else None
        )
        # let UI breathe if you pass update_status
        try:
            from PyQt6.QtWidgets import QApplication
            QApplication.processEvents()
        except Exception:
            pass
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
        """
        Load the image from the reference frame file using the global load_image function.
        """
        image_data, header, _, _ = load_image(self.ref_frame_path)
        if image_data is not None:
            if image_data.ndim == 3 and image_data.shape[-1] == 1:
                image_data = np.squeeze(image_data, axis=-1)
            self.original_image = image_data
        else:
            QMessageBox.critical(self, "Error", "Failed to load the reference image.")
    
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
    
    def convertArrayToPixmap(self, image):
        if image is None:
            return None
        display_image = (image * 255).clip(0, 255).astype(np.uint8)
        if display_image.ndim == 2:
            h, w = display_image.shape
            bytes_per_line = w
            q_image = QImage(display_image.tobytes(), w, h, bytes_per_line, QImage.Format.Format_Grayscale8)
        elif display_image.ndim == 3 and display_image.shape[2] == 3:
            h, w, _ = display_image.shape
            bytes_per_line = 3 * w
            q_image = QImage(display_image.tobytes(), w, h, bytes_per_line, QImage.Format.Format_RGB888)
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



class StackingSuiteDialog(QDialog):
    requestRelaunch = pyqtSignal(str, str)  # old_dir, new_dir
    status_signal = pyqtSignal(str)

    def __init__(self, parent=None, wrench_path=None, spinner_path=None, **_ignored):
        super().__init__(parent)
        self.settings = QSettings()         
        self._wrench_path = wrench_path
        self._spinner_path = spinner_path
        self._post_progress_label = None
        # ...
        self.wrench_button = QPushButton()
        self.wrench_button.setIcon(QIcon(self._wrench_path))
        self.setWindowTitle("Stacking Suite")
        self.setGeometry(300, 200, 800, 600)
        self.per_group_drizzle = {}
        self.manual_dark_overrides = {}  
        self.manual_flat_overrides = {}
        self.conversion_output_directory = None
        self.reg_files = {}
        self.session_tags = {}  # 🔑 file_path => session_tag (e.g., "Session1", "Blue Flats", etc.)
        self.deleted_calibrated_files = []
        self._norm_map = {}
        self._gui_thread = QThread.currentThread()        # remember GUI thread
        self.status_signal.connect(self._update_status_gui)  # queued to GUI
        self._cfa_for_this_run = None  # None = follow checkbox; True/False = override for this run
 
        self.reference_frame = None     # make it explicit from the start
        self._comet_seed = None         # {'path': <original file>, 'xy': (x,y)}
        self._orig2norm = {}            # map original path -> normalized *_n.fit
        self._comet_ref_xy = None       # comet coordinate in reference frame (filled post-align)


        # --- singleton status console (no parent, recreate if gone) ---
        app = QApplication.instance()
        if not hasattr(app, "_sasd_log_bus"):
            app._sasd_log_bus = LogBus()
        self._log_bus = app._sasd_log_bus
        self.status_signal.connect(self._log_bus.posted.emit, Qt.ConnectionType.QueuedConnection)

        # show the dock (if the main window created it)
        self._ensure_log_visible_once()

        self.auto_rot180 = self.settings.value("stacking/auto_rot180", True, type=bool)
        self.auto_rot180_tol_deg = self.settings.value("stacking/auto_rot180_tol_deg", 89.0, type=float)             

        # QSettings for your app

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
            "stacking/rejection_algorithm",
            "Weighted Windsorized Sigma Clipping",
            type=str
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

        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        self.dir_path_edit = QLineEdit(self.stacking_directory)  # Add this here
        # Create the new Conversion tab.
        self.conversion_tab = self.create_conversion_tab()
        # Existing tabs...
        self.dark_tab = self.create_dark_tab()
        self.flat_tab = self.create_flat_tab()
        self.light_tab = self.create_light_tab()
        add_runtime_to_sys_path(status_cb=lambda *_: None)
        self.image_integration_tab = self.create_image_registration_tab()

        # Add the tabs in desired order. (Conversion first)
        self.tabs.addTab(self.conversion_tab, "Convert Non-FITS Formats")
        self.tabs.addTab(self.dark_tab, "Darks")
        self.tabs.addTab(self.flat_tab, "Flats")
        self.tabs.addTab(self.light_tab, "Lights")
        self.tabs.addTab(self.image_integration_tab, "Image Integration")
        self.tabs.setCurrentIndex(1)  # Default to Darks tab

        # Wrench button, status bar, etc.
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
        self.stacking_path_display.setFrame(False)  # nicer, label-like look
        self.stacking_path_display.setToolTip(self.stacking_directory or "No stacking folder selected")
        header_row.addWidget(self.stacking_path_display, 1)  # stretch

        layout.addLayout(header_row)
        self.log_btn = QToolButton(self)
        self.log_btn.setText("Open Log")
        self.log_btn.setToolTip("Show the Stacking Suite log window")
        self.log_btn.clicked.connect(self._show_log_window)
        header_row.addWidget(self.log_btn)
        self.tabs.currentChanged.connect(self.on_tab_changed)
        self.restore_saved_master_calibrations()
        self._update_stacking_path_display()


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
                                                "Supported Files (*.fits *.fit *.fz *.fz *.fits.gz *.fit.gz *.tiff *.tif *.png *.jpg *.jpeg *.cr2 *.cr3 *.nef *.arw *.dng *.orf *.rw2 *.pef *.xisf)")
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
                                           ".cr2", ".cr3", ".nef", ".arw", ".dng", ".orf", ".rw2", ".pef", ".xisf")):
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

            # If it's a RAW format, definitely treat as color
            if file_path.lower().endswith(('.cr2', '.cr3', '.nef', '.arw', '.dng', '.orf', '.rw2', '.pef')):
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
            image=image/np.max(image)

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
        # per-run override if set, else honor the checkbox
        if self._cfa_for_this_run is None:
            cfa = bool(getattr(self, "cfa_drizzle_cb", None) and self.cfa_drizzle_cb.isChecked())
        else:
            cfa = bool(self._cfa_for_this_run)
        print(f"[DEBUG] Debayering with CFA drizzle = {cfa}")
        ext = file_path.lower()
        if ext.endswith(('.cr2','.cr3','.nef','.arw','.dng','.orf','.rw2','.pef')):
            return debayer_raw_fast(image, cfa_drizzle=cfa)
        elif ext.endswith(('.fits','.fit','.fz')):
            bp = (header.get('BAYERPAT') or header.get('BAYERPATN') or "").upper()
            if bp:
                return debayer_fits_fast(image, bp, cfa_drizzle=cfa)
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
        # tiny ‘now’ indicator in the dialog header
        if hasattr(self, "_last_status_label"):
            self._last_status_label.setText(message)
            self._set_last_status(message)


    def update_status(self, message: str):
        if QThread.currentThread() is self._gui_thread:
            self._update_status_gui(message)
            # ALSO emit so the log window gets the line if we’re on the GUI thread
            self.status_signal.emit(message)
        else:
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
        self.drop_shrink_spin.setRange(0.05, 2.0)
        self.drop_shrink_spin.setDecimals(3)
        self.drop_shrink_spin.setSingleStep(0.05)
        self.drop_shrink_spin.setValue(self.settings.value("stacking/drop_shrink", 0.65, type=float))
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

        # --- Comet Star Removal (Optional) ---
        gb_csr = QGroupBox("Comet Star Removal (Optional)")
        fl_csr = QFormLayout(gb_csr)

        self.csr_enable = QCheckBox("Remove stars on comet-aligned frames")
        self.csr_enable.setChecked(self.settings.value("stacking/comet_starrem/enabled", False, type=bool))
        fl_csr.addRow(self.csr_enable)

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

        right_col.addWidget(gb_csr)


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
        right_col.addStretch(1)

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

        # --- Buttons ---
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(lambda: self.save_stacking_settings(dialog))
        btns.rejected.connect(dialog.reject)
        root.addWidget(btns)

        dialog.resize(900, 640)
        dialog.exec()


    def closeEvent(self, e):
        # Graceful shutdown for any running workers
        try:
            if hasattr(self, "alignment_thread") and self.alignment_thread and self.alignment_thread.isRunning():
                self.alignment_thread.requestInterruption()
                self.alignment_thread.wait(1500)
        except Exception:
            pass
        super().closeEvent(e)

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
        return tab

    def flat_tree_context_menu(self, position):
        item = self.flat_tree.itemAt(position)
        if item:
            menu = QMenu()
            set_session_action = menu.addAction("Set Session Tag")
            action = menu.exec(self.flat_tree.viewport().mapToGlobal(position))
            if action == set_session_action:
                self.prompt_set_session(item, "flat")

    def create_light_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

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


    def create_image_registration_tab(self):
        """
        Creates an Image Registration tab that mimics how the Light tab handles
        cosmetic corrections—i.e., we have global Drizzle controls (checkbox, combo, spin),
        and we update a text column in the QTreeWidget to show each group's drizzle state.
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
            "Drizzle"  # We'll display "Drizzle: True, Scale: 2x, Drop:0.65" here
        ])
        self.reg_tree.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)

        # Optional: make columns resize nicely
        header = self.reg_tree.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)

        layout.addWidget(QLabel("Calibrated Light Frames"))
        layout.addWidget(self.reg_tree)

        tol_layout = QHBoxLayout()
        tol_layout.addWidget(QLabel("Exposure Tolerance (sec):"))
        self.exposure_tolerance_spin = QSpinBox()
        self.exposure_tolerance_spin.setRange(0, 900)
        self.exposure_tolerance_spin.setValue(0)
        self.exposure_tolerance_spin.setSingleStep(5)
        tol_layout.addWidget(self.exposure_tolerance_spin)
        tol_layout.addStretch()
        self.split_dualband_cb = QCheckBox("Split dual-band OSC before integration")
        self.split_dualband_cb.setToolTip("For OSC dual-band data: SII/OIII → R=SII, G=OIII; Ha/OIII → R=Ha, G=OIII")
        tol_layout.addWidget(self.split_dualband_cb)
        layout.addLayout(tol_layout)

        self.exposure_tolerance_spin.valueChanged.connect(lambda _: self.populate_calibrated_lights())



        # ─────────────────────────────────────────
        # 2) Buttons for Managing Files
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
        # 3) Global Drizzle Controls
        # ─────────────────────────────────────────
        drizzle_layout = QHBoxLayout()

        self.drizzle_checkbox = QCheckBox("Enable Drizzle")
        self.drizzle_checkbox.toggled.connect(self._on_drizzle_checkbox_toggled) # <─ connect signal
        drizzle_layout.addWidget(self.drizzle_checkbox)

        drizzle_layout.addWidget(QLabel("Scale:"))
        self.drizzle_scale_combo = QComboBox()
        self.drizzle_scale_combo.addItems(["1x", "2x", "3x"])
        self.drizzle_scale_combo.currentIndexChanged.connect(self._on_drizzle_param_changed)  # <─ connect
        drizzle_layout.addWidget(self.drizzle_scale_combo)

        drizzle_layout.addWidget(QLabel("Drop Shrink:"))
        self.drizzle_drop_shrink_spin = QDoubleSpinBox()
        self.drizzle_drop_shrink_spin.setRange(0.0, 1.0)
        self.drizzle_drop_shrink_spin.setSingleStep(0.05)
        self.drizzle_drop_shrink_spin.setValue(0.65)
        self.drizzle_drop_shrink_spin.valueChanged.connect(self._on_drizzle_param_changed)  # <─ connect
        drizzle_layout.addWidget(self.drizzle_drop_shrink_spin)

        self.cfa_drizzle_cb = QCheckBox("CFA Drizzle")
        self.cfa_drizzle_cb.setChecked(self.settings.value("stacking/cfa_drizzle", False, type=bool))
        self.cfa_drizzle_cb.toggled.connect(self._on_cfa_drizzle_toggled)
        self.cfa_drizzle_cb.setToolTip("Map R/G/B CFA samples directly into channels and skip interpolation.")
        drizzle_layout.addWidget(self.cfa_drizzle_cb)

        layout.addLayout(drizzle_layout)

        # ─────────────────────────────────────────
        # 4) Reference Frame Selection
        # ─────────────────────────────────────────
        self.ref_frame_label = QLabel("Select Reference Frame:")
        self.ref_frame_path = QLabel("No file selected")
        self.ref_frame_path.setWordWrap(True)
        self.select_ref_frame_btn = QPushButton("Select Reference Frame")
        self.select_ref_frame_btn.clicked.connect(self.select_reference_frame)

        # NEW: Auto-accept toggle
        self.auto_accept_ref_cb = QCheckBox("Auto-accept measured reference")
        self.auto_accept_ref_cb.setToolTip(
            "When checked, the best measured frame is accepted automatically and the review dialog is skipped."
        )
        # restore from settings
        self.auto_accept_ref_cb.setChecked(self.settings.value("stacking/auto_accept_ref", False, type=bool))
        self.auto_accept_ref_cb.toggled.connect(
            lambda v: self.settings.setValue("stacking/auto_accept_ref", bool(v))
        )

        ref_layout = QHBoxLayout()
        ref_layout.addWidget(self.ref_frame_label)
        ref_layout.addWidget(self.ref_frame_path, 1)
        ref_layout.addWidget(self.select_ref_frame_btn)
        ref_layout.addWidget(self.auto_accept_ref_cb)  # ← add to the row
        layout.addLayout(ref_layout)

        crop_row = QHBoxLayout()
        self.autocrop_cb = QCheckBox("Auto-crop output")
        self.autocrop_cb.setToolTip("Crop the final image to pixels covered by ≥ Coverage % of frames")
        self.autocrop_pct = QDoubleSpinBox()
        self.autocrop_pct.setRange(50.0, 100.0)
        self.autocrop_pct.setSingleStep(1.0)
        self.autocrop_pct.setSuffix(" %")
        self.autocrop_pct.setValue(self.settings.value("stacking/autocrop_pct", 95.0, type=float))
        self.autocrop_cb.setChecked(self.settings.value("stacking/autocrop_enabled", True, type=bool))
        crop_row.addWidget(self.autocrop_cb)
        crop_row.addWidget(QLabel("Coverage:"))
        crop_row.addWidget(self.autocrop_pct)
        crop_row.addStretch(1)
        layout.addLayout(crop_row)

        # In create_image_registration_tab(), after the crop_row:
        comet_row = QHBoxLayout()

        self.comet_cb = QCheckBox("🌠🌠Create comet stack (comet-aligned)🌠🌠")
        self.comet_cb.setChecked(self.settings.value("stacking/comet/enabled", False, type=bool))
        comet_row.addWidget(self.comet_cb)

        self.comet_pick_btn = QPushButton("Pick comet center…")
        self.comet_pick_btn.setEnabled(self.comet_cb.isChecked())
        self.comet_pick_btn.clicked.connect(self._pick_comet_center)
        self.comet_cb.toggled.connect(self.comet_pick_btn.setEnabled)
        comet_row.addWidget(self.comet_pick_btn)

        self.comet_blend_cb = QCheckBox("Also output Stars+Comet blend")
        self.comet_blend_cb.setChecked(self.settings.value("stacking/comet/blend", True, type=bool))
        comet_row.addWidget(self.comet_blend_cb)


        comet_row.addWidget(QLabel("Mix:"))
        self.comet_mix = QDoubleSpinBox()
        self.comet_mix.setRange(0.0, 1.0); self.comet_mix.setSingleStep(0.05)
        self.comet_mix.setValue(self.settings.value("stacking/comet/mix", 1.0, type=float))
        comet_row.addWidget(self.comet_mix)


        comet_row.addStretch(1)
        layout.addLayout(comet_row)

        mf_box = QGroupBox("MFDeconv (beta) — Multi-Frame Deconvolution (ImageMM)")
        mf_v = QVBoxLayout(mf_box)

        # sensible defaults if missing
        def _get(key, default, t):
            return self.settings.value(key, default, type=t)

        # row 1: enable
        mf_row1 = QHBoxLayout()
        self.mf_enabled_cb = QCheckBox("Enable MFDeconv during integration")
        self.mf_enabled_cb.setChecked(_get("stacking/mfdeconv/enabled", False, bool))
        self.mf_enabled_cb.setToolTip("Runs multi-frame deconvolution during integration. Turn off if testing.")
        mf_row1.addWidget(self.mf_enabled_cb)
        mf_row1.addStretch(1)
        mf_v.addLayout(mf_row1)

        # row 2: iterations + kappa
        mf_row2 = QHBoxLayout()
        mf_row2.addWidget(QLabel("Iterations:"))
        self.mf_iters_spin = QSpinBox()
        self.mf_iters_spin.setRange(1, 500)
        self.mf_iters_spin.setValue(_get("stacking/mfdeconv/iters", 20, int))
        self.mf_iters_spin.setToolTip("Number of multiplicative update steps (e.g., 10–60).")
        mf_row2.addWidget(self.mf_iters_spin)

        mf_row2.addSpacing(16)
        mf_row2.addWidget(QLabel("Update clip (κ):"))
        self.mf_kappa_spin = QDoubleSpinBox()
        self.mf_kappa_spin.setRange(0.0, 10.0)
        self.mf_kappa_spin.setDecimals(3)
        self.mf_kappa_spin.setSingleStep(0.1)
        self.mf_kappa_spin.setValue(_get("stacking/mfdeconv/kappa", 2.0, float))
        self.mf_kappa_spin.setToolTip("Kappa clipping for multiplicative updates (typ. ~2.0).")
        mf_row2.addWidget(self.mf_kappa_spin)
        mf_row2.addStretch(1)
        mf_v.addLayout(mf_row2)

        # row 3: color mode + huber_delta
        mf_row3 = QHBoxLayout()
        mf_row3.addWidget(QLabel("Color mode:"))
        self.mf_color_combo = QComboBox()
        self.mf_color_combo.addItems(["luma", "perchannel"])
        # ensure current text is valid
        _cm = _get("stacking/mfdeconv/color_mode", "perchannel", str)
        if _cm not in ("luma", "perchannel"):
            _cm = "perchannel"
        self.mf_color_combo.setCurrentText(_cm)
        self.mf_color_combo.setToolTip("‘luma’ deconvolves the luminance only; ‘perchannel’ runs on RGB independently.")
        mf_row3.addWidget(self.mf_color_combo)

        mf_row3.addSpacing(16)
        mf_row3.addWidget(QLabel("Huber δ:"))
        self.mf_huber_spin = QDoubleSpinBox()
        self.mf_huber_spin.setRange(-1000.0, 1000.0)   # negative allowed
        self.mf_huber_spin.setDecimals(4)
        self.mf_huber_spin.setSingleStep(0.1)
        self.mf_huber_spin.setValue(_get("stacking/mfdeconv/huber_delta", 2.0, float))
        self.mf_huber_spin.setToolTip(
            "Robust Huber threshold in σ units. ~2–3× background RMS is typical.\n"
            "Negative values are scale*RMS, positive values are hardcoded deltas."
        )
        mf_row3.addWidget(self.mf_huber_spin)
        
        self.mf_huber_hint = QLabel("(<0 = scale×RMS, >0 = absolute Δ)")
        self.mf_huber_hint.setToolTip("Negative = factor times background RMS (auto). Positive = fixed absolute threshold.")
        self.mf_huber_hint.setStyleSheet("color: #888;")  # subtle hint color
        mf_row3.addWidget(self.mf_huber_hint)        
        mf_row3.addStretch(1)
        mf_v.addLayout(mf_row3)

        # write-through bindings
        self.mf_enabled_cb.toggled.connect(
            lambda v: self.settings.setValue("stacking/mfdeconv/enabled", bool(v))
        )
        self.mf_iters_spin.valueChanged.connect(
            lambda v: self.settings.setValue("stacking/mfdeconv/iters", int(v))
        )
        self.mf_kappa_spin.valueChanged.connect(
            lambda v: self.settings.setValue("stacking/mfdeconv/kappa", float(v))
        )
        self.mf_color_combo.currentTextChanged.connect(
            lambda s: self.settings.setValue("stacking/mfdeconv/color_mode", s)
        )
        self.mf_huber_spin.valueChanged.connect(
            lambda v: self.settings.setValue("stacking/mfdeconv/huber_delta", float(v))
        )

        accel_row = QHBoxLayout()
        self.backend_label = QLabel(f"Backend: {current_backend()}")
        accel_row.addWidget(self.backend_label)

        self.install_accel_btn = QPushButton("Install/Update GPU Acceleration…")
        self.install_accel_btn.setToolTip("Downloads PyTorch with the right backend (CUDA/MPS/CPU). One-time per machine.")
        accel_row.addWidget(self.install_accel_btn)
        accel_row.addStretch(1)
        mf_v.addLayout(accel_row)

        def _install_accel():
            # --- UI: all created on the GUI thread ---
            self.install_accel_btn.setEnabled(False)
            self.backend_label.setText("Backend: installing…")

            # Indeterminate progress dialog on GUI thread
            self._accel_pd = QProgressDialog("Preparing runtime…", "Cancel", 0, 0, self)
            self._accel_pd.setWindowTitle("Installing GPU Acceleration")
            self._accel_pd.setWindowModality(Qt.WindowModality.ApplicationModal)
            self._accel_pd.setAutoClose(True)
            self._accel_pd.setMinimumDuration(0)
            self._accel_pd.show()

            # Worker thread
            self._accel_thread = QThread(self)
            self._accel_worker = AccelInstallWorker(prefer_gpu=True)
            self._accel_worker.moveToThread(self._accel_thread)

            # Start worker when thread starts (queued so it runs in worker thread)
            self._accel_thread.started.connect(self._accel_worker.run, Qt.ConnectionType.QueuedConnection)

            # Pipe worker progress to both the progress dialog and your log bus (GUI thread)
            self._accel_worker.progress.connect(self._accel_pd.setLabelText, Qt.ConnectionType.QueuedConnection)
            self._accel_worker.progress.connect(lambda s: self.status_signal.emit(s), Qt.ConnectionType.QueuedConnection)

            # Allow cancel to request interruption
            def _cancel():
                if self._accel_thread.isRunning():
                    self._accel_thread.requestInterruption()
            self._accel_pd.canceled.connect(_cancel, Qt.ConnectionType.QueuedConnection)

            # Finish handler (runs on GUI thread)
            def _done(ok: bool, msg: str):
                # Close progress dialog safely
                if getattr(self, "_accel_pd", None):
                    self._accel_pd.reset()
                    self._accel_pd.deleteLater()
                    self._accel_pd = None

                # Stop thread cleanly
                self._accel_thread.quit()
                self._accel_thread.wait()

                # Re-enable UI and refresh backend label
                self.install_accel_btn.setEnabled(True)
                from pro.accel_installer import current_backend
                self.backend_label.setText(f"Backend: {current_backend()}")

                # User feedback + log
                self.status_signal.emit(("✅ " if ok else "❌ ") + msg)
                if ok:
                    QMessageBox.information(self, "Acceleration", f"✅ {msg}")
                else:
                    QMessageBox.warning(self, "Acceleration", f"❌ {msg}")

            self._accel_worker.finished.connect(_done, Qt.ConnectionType.QueuedConnection)

            # Cleanup objects when thread finishes
            self._accel_thread.finished.connect(self._accel_worker.deleteLater, Qt.ConnectionType.QueuedConnection)
            self._accel_thread.finished.connect(self._accel_thread.deleteLater, Qt.ConnectionType.QueuedConnection)

            self._accel_thread.start()

        self.install_accel_btn.clicked.connect(_install_accel)

        layout.addWidget(mf_box)

        # ★★ Star-Trail Mode ★★
        trail_layout = QHBoxLayout()
        self.trail_cb = QCheckBox("★★ Star-Trail Mode ★★ (Max-Value Stack)")
        self.trail_cb.setChecked(self.star_trail_mode)
        self.trail_cb.setToolTip(
            "Skip registration/alignment and use Maximum-Intensity projection for star trails"
        )
        self.trail_cb.stateChanged.connect(self._on_star_trail_toggled)
        trail_layout.addWidget(self.trail_cb)
        layout.addLayout(trail_layout)
        # ─────────────────────────────────────────
        # 5) Register & Integrate Buttons
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
            QPushButton:hover {
                background-color: #FF6347;
            }
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
            QPushButton:hover {
                border: 2px solid #FFD700;
            }
            QPushButton:pressed {
                background-color: #222;
                border: 2px solid #FFA500;
            }
        """)
        layout.addWidget(self.integrate_registered_btn)

        # Populate the tree from your calibrated folder
        self.populate_calibrated_lights()
        tab.setLayout(layout)

        self.drizzle_checkbox.setChecked(self.settings.value("stacking/drizzle_enabled", False, type=bool))
        self.drizzle_scale_combo.setCurrentText(self.settings.value("stacking/drizzle_scale", "2x", type=str))
        self.drizzle_drop_shrink_spin.setValue(self.settings.value("stacking/drizzle_drop", 0.65, type=float))

        self.settings.setValue("stacking/comet/enabled", self.comet_cb.isChecked())
        self.settings.setValue("stacking/comet/blend", self.comet_blend_cb.isChecked())

        self.settings.setValue("stacking/comet/mix", self.comet_mix.value())


        self.auto_accept_ref_cb.toggled.connect(self.select_ref_frame_btn.setDisabled)
        self.select_ref_frame_btn.setDisabled(self.auto_accept_ref_cb.isChecked())


        return tab

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
        selected_items = tree.selectedItems()
        if not selected_items:
            return

        for item in selected_items:
            parent = item.parent()

            if parent is None:
                # Remove entire group
                group_key = item.text(0)
                # Track deleted files (optional)
                full_paths = item.data(0, Qt.ItemDataRole.UserRole) or []
                self.deleted_calibrated_files.extend(p for p in full_paths
                                                    if p not in self.deleted_calibrated_files)
                # Remove from dict + tree
                self.reg_files.pop(group_key, None)
                tree.takeTopLevelItem(tree.indexOfTopLevelItem(item))

            else:
                # Remove a single child
                group_key = parent.text(0)
                filename = item.text(0)

                if group_key in self.reg_files:
                    self.reg_files[group_key] = [
                        f for f in self.reg_files[group_key]
                        if os.path.basename(f) != filename
                    ]
                    if not self.reg_files[group_key]:
                        del self.reg_files[group_key]

                # Track deleted path (optional)
                fp = item.data(0, Qt.ItemDataRole.UserRole)
                if fp and fp not in self.deleted_calibrated_files:
                    self.deleted_calibrated_files.append(fp)

                # Remove child from tree
                parent.removeChild(item)

                # 🔑 keep parent’s stored list in sync
                self._sync_group_userrole(parent)

    def rebuild_flat_tree(self):
        """Regroup flat frames in the flat_tree based on the exposure tolerance."""
        self.flat_tree.clear()

        if not self.flat_files:
            return

        tolerance = self.flat_exposure_tolerance_spinbox.value()

        # Flatten all flats into a list
        all_flats = []
        for (filter_exp_size, session_tag), files in self.flat_files.items():
            for file in files:
                all_flats.append((filter_exp_size, session_tag, file))

        # Group the flats
        grouped = {}

        for (filter_exp_size, session_tag, file_path) in all_flats:
            try:
                header = fits.getheader(file_path, ext=0)
                filter_name = header.get("FILTER", "Unknown")
                filter_name     = self._sanitize_name(filter_name)
                exposure = header.get("EXPOSURE", header.get("EXPTIME", "Unknown"))
                width = header.get("NAXIS1", 0)
                height = header.get("NAXIS2", 0)
                image_size = f"{width}x{height}" if width and height else "Unknown"
                exposure = float(exposure)

                found_group = None
                for group_key in grouped.keys():
                    g_filter, g_min_exp, g_max_exp, g_size = group_key
                    if (
                        filter_name == g_filter and
                        image_size == g_size and
                        g_min_exp - tolerance <= exposure <= g_max_exp + tolerance
                    ):
                        found_group = group_key
                        break

                if found_group:
                    grouped[found_group].append((file_path, exposure))
                else:
                    new_key = (filter_name, exposure, exposure, image_size)
                    grouped[new_key] = [(file_path, exposure)]

            except Exception as e:
                print(f"⚠️ Failed reading {file_path}: {e}")

        # Now create the tree
        for (filter_name, min_exp, max_exp, image_size), files in grouped.items():
            top_item = QTreeWidgetItem()
            expmin = np.floor(min_exp)
            tolerance = self.flat_exposure_tolerance_spinbox.value()

            if len(files) > 1:
                exposure_str = f"{expmin:.1f}s–{(expmin + tolerance):.1f}s"
            else:
                exposure_str = f"{min_exp:.1f}s"

            top_item.setText(0, f"{filter_name} - {exposure_str} ({image_size})")
            top_item.setText(1, f"{len(files)} files")
            top_item.setText(2, "Auto-Selected Dark" if self.auto_select_dark_checkbox.isChecked() else "None")

            self.flat_tree.addTopLevelItem(top_item)

            for file_path, _ in files:
                session_tag = self.session_tags.get(file_path, "Default")
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


    def populate_calibrated_lights(self):
        """
        Reads both the Calibrated folder and any manually-added files,
        groups them by FILTER, EXPOSURE±tol, SIZE, and fills self.reg_tree.
        Also sets each group's Drizzle column from saved per-group state,
        or from the current global controls if none exists yet.
        """
        from PIL import Image

        # Fallback in case helper wasn't added
        def _fmt(enabled, scale, drop):
            return (f"Drizzle: True, Scale: {scale:g}x, Drop: {drop:.2f}"
                    if enabled else "Drizzle: False")

        # 1) clear tree
        self.reg_tree.clear()
        self.reg_tree.setColumnCount(3)
        self.reg_tree.setHeaderLabels(["Filter - Exposure - Size", "Metadata", "Drizzle"])
        hdr = self.reg_tree.header()
        for col in (0, 1, 2):
            hdr.setSectionResizeMode(col, QHeaderView.ResizeMode.Interactive)

        # 2) gather files
        calibrated_folder = os.path.join(self.stacking_directory or "", "Calibrated")
        files = []
        if os.path.isdir(calibrated_folder):
            for fn in os.listdir(calibrated_folder):
                files.append(os.path.join(calibrated_folder, fn))
        files += getattr(self, "manual_light_files", [])

        if not files:
            return

        # 3) group by header (or defaults)
        grouped = {}  # key -> list of dicts: {"path", "exp", "size"}
        tol = self.exposure_tolerance_spin.value()

        for fp in files:
            ext = os.path.splitext(fp)[1].lower()
            filt = "Unknown"
            exp = 0.0
            size = "Unknown"

            if ext in (".fits", ".fit"):
                try:
                    hdr0 = fits.getheader(fp, ext=0)
                    filt = self._sanitize_name(hdr0.get("FILTER", "Unknown"))
                    exp_raw = hdr0.get("EXPOSURE", hdr0.get("EXPTIME", None))
                    try:
                        exp = float(exp_raw)
                    except (TypeError, ValueError):
                        print(f"⚠️ Exposure invalid in {fp}, defaulting to 0.0s")
                        exp = 0.0
                    data0 = fits.getdata(fp, ext=0)
                    h, w = data0.shape[-2:]
                    size = f"{w}x{h}"
                except Exception as e:
                    print(f"⚠️ Could not read FITS {fp}: {e}; treating as generic image")

            if filt == "Unknown" and ext not in (".fits", ".fit"):
                # generic image via PIL/utility
                try:
                    w, h = self._get_image_size(fp)
                    size = f"{w}x{h}"
                except Exception as e:
                    print(f"⚠️ Cannot read image size for {fp}: {e}")
                    continue

            # find existing group
            match_key = None
            for key in grouped:
                f2, e2, s2 = self.parse_group_key(key)
                if filt == f2 and s2 == size and abs(exp - e2) <= tol:
                    match_key = key
                    break

            key = match_key or f"{filt} - {exp:.1f}s ({size})"
            grouped.setdefault(key, []).append({"path": fp, "exp": exp, "size": size})

        # 4) populate tree & self.light_files
        self.light_files = {}

        # read current global drizzle controls (used as default)
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

            # Use saved per-group drizzle state if present; else default to global controls
            state = self.per_group_drizzle.get(key)
            if state is None:
                state = {
                    "enabled": bool(global_enabled),
                    "scale":   float(global_scale),
                    "drop":    float(global_drop),
                }
                self.per_group_drizzle[key] = state  # persist default for this group

            # Show in column 2
            try:
                top.setText(2, self._format_drizzle_text(state["enabled"], state["scale"], state["drop"]))
            except AttributeError:
                top.setText(2, _fmt(state["enabled"], state["scale"], state["drop"]))

            top.setData(0, Qt.ItemDataRole.UserRole, paths)
            self.reg_tree.addTopLevelItem(top)

            # leaf rows: show basename + *per-file* size (fixes the old "same size for all leaves" issue)
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

    def _on_drizzle_checkbox_toggled(self, checked: bool):
        self.settings.setValue("stacking/drizzle_enabled", bool(checked))
        self._update_drizzle_summary_columns()

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
        """
        Let the user pick some new LIGHT frames, then
        immediately re-populate the tree so they show up
        in the same Filter–Exposure–Size groups as everything else.
        """
        last_dir = self.settings.value("last_opened_folder", "", type=str)
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Light Frames",
            last_dir,
            "FITS Files (*.fits *.fit *.fz *.fz *.xisf *.tif *.tiff *.png *.jpg *.jpeg)"
        )
        if not files:
            return

        # remember for next time
        self.settings.setValue("last_opened_folder", os.path.dirname(files[0]))

        # store these in a manual list, then rebuild the tree
        if not hasattr(self, "manual_light_files"):
            self.manual_light_files = []
        self.manual_light_files.extend(files)

        # rebuild the registration tree (it reads manual_light_files + calibrated folder)
        self.populate_calibrated_lights()





    def on_tab_changed(self, index):
        """ Detects when user switches to the Flats tab and triggers auto-assign. """
        if self.tabs.tabText(index) == "Flats":
            print("🔄 Auto-checking best Master Darks for Flats...")
            self.assign_best_master_dark()


    def add_dark_files(self):
        self.add_files(self.dark_tree, "Select Dark Files", "DARK")
    
    def add_dark_directory(self):
        self.add_directory(self.dark_tree, "Select Dark Directory", "DARK")

    def add_flat_files(self):
        self.prompt_session_before_adding("FLAT")


    def add_flat_directory(self):
        self.prompt_session_before_adding("FLAT", directory_mode=True)


    
    def add_light_files(self):
        self.prompt_session_before_adding("LIGHT")

    
    def add_light_directory(self):
        self.prompt_session_before_adding("LIGHT", directory_mode=True)


    def prompt_session_before_adding(self, frame_type, directory_mode=False):
        # 🔥 Prompt user first
        text, ok = QInputDialog.getText(self, "Set Session Tag", "Enter session name:", text="Default")
        if not (ok and text.strip()):
            return

        session_name = text.strip()

        # 🔥 Set it globally before adding
        self.current_session_tag = session_name

        # 🔥 Then add files or directory
        if frame_type.upper() == "FLAT":
            if directory_mode:
                self.add_directory(self.flat_tree, "Select Flat Directory", "FLAT")
            else:
                self.add_files(self.flat_tree, "Select Flat Files", "FLAT")
            self.assign_best_master_dark()
            self.rebuild_flat_tree()

        elif frame_type.upper() == "LIGHT":
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



    def add_directory(self, tree, title, expected_type):
        """ Adds all FITS files from a directory and assigns best master files if needed. """
        last_dir = self.settings.value("last_opened_folder", "", type=str)
        directory = QFileDialog.getExistingDirectory(self, title, last_dir)
        if not directory:
            return

        self.settings.setValue("last_opened_folder", directory)

        # Collect files first so we know the total for the progress range
        exts = (".fits", ".fit", ".fts", ".fits.gz", ".fit.gz", ".fz")
        paths = [
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if f.lower().endswith(exts)
        ]
        if not paths:
            return

        # Standalone progress while ingesting
        self._ingest_paths_with_progress(
            paths=paths,
            tree=tree,
            expected_type=expected_type,
            title=f"Adding {expected_type.title()} from Directory…"
        )

        # Auto-assign after ingest (LIGHT only, same behavior you had)
        if expected_type.upper() == "LIGHT":
            busy = self._busy_progress("Assigning best Master Dark/Flat…")
            try:
                self.assign_best_master_files()
            finally:
                busy.close()

    def _ingest_paths_with_progress(self, paths, tree, expected_type, title):
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
                self.process_fits_header(path, tree, expected_type)
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

        # Optional: brief status line (non-intrusive)
        try:
            if expected_type.upper() == "LIGHT":
                self.statusBar().showMessage(f"Added {added}/{total} Light frames", 3000)
        except Exception:
            pass

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
    
    def process_fits_header(self, file_path, tree, expected_type):
        try:
            # Read only the FITS header (fast)
            header, _ = get_valid_header(file_path)

            try:
                width = int(header.get("NAXIS1"))
                height = int(header.get("NAXIS2"))
            except Exception as e:
                self.update_status(f"Warning: Could not convert dimensions to int for {file_path}: {e}")
                width, height = None, None

            if width is not None and height is not None:
                image_size = f"{width}x{height}"
            else:
                image_size = "Unknown"

            # Retrieve IMAGETYP (default to "UNKNOWN" if not present)
            imagetyp = header.get("IMAGETYP", "UNKNOWN").lower()

            # Retrieve exposure from either EXPOSURE or EXPTIME
            exposure_val = header.get("EXPOSURE")
            if not exposure_val:
                exposure_val = header.get("EXPTIME")
            if not exposure_val:
                exposure_val = "Unknown"  # fallback if neither keyword is present

            # Define forbidden keywords per expected type.
            if expected_type.upper() == "DARK":
                forbidden = ["light", "flat"]
            elif expected_type.upper() == "FLAT":
                forbidden = ["dark", "light"]
            elif expected_type.upper() == "LIGHT":
                forbidden = ["dark", "flat"]
            else:
                forbidden = []

            # Determine attribute name for auto-confirm decision (per expected type)
            decision_attr = f"auto_confirm_{expected_type.lower()}"
            # If a decision has already been made, use it.
            if hasattr(self, decision_attr):
                decision = getattr(self, decision_attr)
                if decision is False:
                    # Skip this file automatically.
                    return
                # If decision is True, then add without prompting.
            elif any(word in imagetyp for word in forbidden):
                # Prompt the user with Yes, Yes to All, No, and No to All options.
                msgBox = QMessageBox(self)
                msgBox.setWindowTitle("Mismatched Image Type")
                msgBox.setText(
                    f"The file:\n{os.path.basename(file_path)}\n"
                    f"has IMAGETYP = {header.get('IMAGETYP')} "
                    f"which does not match the expected type ({expected_type}).\n\n"
                    f"Do you want to add it anyway?"
                )
                yesButton = msgBox.addButton("Yes", QMessageBox.ButtonRole.YesRole)
                yesToAllButton = msgBox.addButton("Yes to All", QMessageBox.ButtonRole.YesRole)
                noButton = msgBox.addButton("No", QMessageBox.ButtonRole.NoRole)
                noToAllButton = msgBox.addButton("No to All", QMessageBox.ButtonRole.NoRole)
                msgBox.exec()
                clicked = msgBox.clickedButton()
                if clicked == yesToAllButton:
                    setattr(self, decision_attr, True)
                elif clicked == noToAllButton:
                    setattr(self, decision_attr, False)
                    return
                elif clicked == noButton:
                    return

            # Now handle each expected type
            if expected_type.upper() == "DARK":
                key = f"{exposure_val} ({image_size})"
                if key not in self.dark_files:
                    self.dark_files[key] = []
                self.dark_files[key].append(file_path)

                items = tree.findItems(key, Qt.MatchFlag.MatchExactly, 0)
                if not items:
                    exposure_item = QTreeWidgetItem([key])
                    tree.addTopLevelItem(exposure_item)
                else:
                    exposure_item = items[0]
                metadata = f"Size: {image_size}"
                exposure_item.addChild(QTreeWidgetItem([os.path.basename(file_path), metadata]))

            elif expected_type.upper() == "FLAT":
                filter_name = header.get("FILTER", "Unknown")
                filter_name = self._sanitize_name(filter_name)
                flat_key = f"{filter_name} - {exposure_val} ({image_size})"
                session_tag = getattr(self, "current_session_tag", "Default")
                composite_key = (flat_key, session_tag)

                if composite_key not in self.flat_files:
                    self.flat_files[composite_key] = []
                self.flat_files[composite_key].append(file_path)

                # ✅ Also store session tag internally
                self.session_tags[file_path] = session_tag

                # Tree UI update
                filter_items = tree.findItems(filter_name, Qt.MatchFlag.MatchExactly, 0)
                if not filter_items:
                    filter_item = QTreeWidgetItem([filter_name])
                    tree.addTopLevelItem(filter_item)
                else:
                    filter_item = filter_items[0]

                exposure_items = [filter_item.child(i) for i in range(filter_item.childCount())]
                exposure_item = next((item for item in exposure_items
                                    if item.text(0) == f"{exposure_val} ({image_size})"), None)
                if not exposure_item:
                    exposure_item = QTreeWidgetItem([f"{exposure_val} ({image_size})"])
                    filter_item.addChild(exposure_item)

                metadata = f"Size: {image_size} | Session: {session_tag}"
                exposure_item.addChild(QTreeWidgetItem([os.path.basename(file_path), metadata]))


            elif expected_type.upper() == "LIGHT":
                filter_name = header.get("FILTER", "Unknown")
                filter_name = self._sanitize_name(filter_name)
                session_tag = getattr(self, "current_session_tag", "Default")  # ⭐️ Step 1: Get session label

                light_key = f"{filter_name} - {exposure_val} ({image_size})"
                composite_key = (light_key, session_tag)

                if composite_key not in self.light_files:
                    self.light_files[composite_key] = []
                self.light_files[composite_key].append(file_path)

                # Update Tree UI
                filter_items = tree.findItems(filter_name, Qt.MatchFlag.MatchExactly, 0)
                if not filter_items:
                    filter_item = QTreeWidgetItem([filter_name])
                    tree.addTopLevelItem(filter_item)
                else:
                    filter_item = filter_items[0]

                exposure_items = [filter_item.child(i) for i in range(filter_item.childCount())]
                exposure_item = next((item for item in exposure_items
                                    if item.text(0) == f"{exposure_val} ({image_size})"), None)
                if not exposure_item:
                    exposure_item = QTreeWidgetItem([f"{exposure_val} ({image_size})"])
                    filter_item.addChild(exposure_item)

                leaf_item = QTreeWidgetItem([os.path.basename(file_path), f"Size: {image_size} | Session: {session_tag}"])
                exposure_item.addChild(leaf_item)
                self.session_tags[file_path] = session_tag  # ✅ Store per-file session tag here


            self.update_status(f"✅ Added {os.path.basename(file_path)} as {expected_type}")
            QApplication.processEvents()

        except Exception as e:
            self.update_status(f"❌ ERROR: Could not read FITS header for {file_path} - {e}")
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
        """ Creates master darks with minimal RAM usage by loading frames in small tiles. """

        if not self.stacking_directory:
            self.select_stacking_directory()
            if not self.stacking_directory:
                QMessageBox.warning(self, "Error", "Output directory is not set.")
                return

        exposure_tolerance = self.exposure_tolerance_spinbox.value()
        dark_files_by_group = {}

        # 1) Group dark files by exposure time & image size within tolerance
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

        # 2) Create Master Calibration Directory
        master_dir = os.path.join(self.stacking_directory, "Master_Calibration_Files")
        os.makedirs(master_dir, exist_ok=True)

        # 3) Stack Each Group in a Chunked Manner
        chunk_height = self.chunk_height
        chunk_width  = self.chunk_width

        for (exposure_time, image_size), file_list in dark_files_by_group.items():
            if len(file_list) < 2:
                self.update_status(f"⚠️ Skipping {exposure_time}s ({image_size}) - Not enough frames to stack.")
                QApplication.processEvents()
                continue

            self.update_status(f"🟢 Processing {len(file_list)} darks for {exposure_time}s ({image_size}) exposure...")
            QApplication.processEvents()

            # (A) Identify reference shape from the first file
            ref_file = file_list[0]
            ref_data, ref_header, bit_depth, is_mono = load_image(ref_file)
            if ref_data is None:
                self.update_status(f"❌ Failed to load reference {os.path.basename(ref_file)}")
                continue

            height, width = ref_data.shape[:2]
            channels = 1 if (ref_data.ndim == 2) else 3

            # (B) Create a memmap for the final stacked result
            # shape=(height, width, channels)
            memmap_path = os.path.join(master_dir, f"temp_dark_{exposure_time}_{image_size}.dat")
            final_stacked = np.memmap(
                memmap_path,
                dtype=self._dtype(),
                mode='w+',
                shape=(height, width, channels)
            )

            # (C) For each tile, load that tile from all frames, do outlier rejection, store in final_stacked
            num_frames = len(file_list)

            for y_start in range(0, height, chunk_height):
                y_end = min(y_start + chunk_height, height)
                tile_h = y_end - y_start

                for x_start in range(0, width, chunk_width):
                    x_end = min(x_start + chunk_width, width)
                    tile_w = x_end - x_start

                    # tile_stack shape => (num_frames, tile_h, tile_w, channels)
                    tile_stack = np.zeros((num_frames, tile_h, tile_w, channels), dtype=np.float32)


                    num_cores = os.cpu_count() or 4
                    with ThreadPoolExecutor(max_workers=num_cores) as executor:
                        future_to_index = {}
                        # 1) Submit each file’s tile load in parallel
                        for i, fpath in enumerate(file_list):
                            future = executor.submit(load_fits_tile, fpath, y_start, y_end, x_start, x_end)
                            future_to_index[future] = i

                        # 2) Collect results as they complete
                        for future in as_completed(future_to_index):
                            i = future_to_index[future]
                            sub_img = future.result()
                            if sub_img is None:
                                continue

                            # --- shape handling (same as before) ---
                            # If sub_img is (H,W) & channels=3 => expand
                            if sub_img.ndim == 2 and channels == 3:
                                sub_img = np.repeat(sub_img[:, :, np.newaxis], 3, axis=2)
                            elif sub_img.ndim == 2 and channels == 1:
                                sub_img = sub_img[:, :, np.newaxis]

                            # If sub_img is (3,H,W) but we want (H,W,3), transpose
                            if sub_img.ndim == 3 and sub_img.shape[0] == 3 and channels == 3:
                                sub_img = sub_img.transpose(1, 2, 0)

                            sub_img = sub_img.astype(np.float32, copy=False)
                            tile_stack[i] = sub_img

                    # (D) Outlier rejection => tile_result
                    # Use your existing 3D or 4D Windsorized Sigma Clip depending on channels
                    if channels == 3:
                        # tile_stack => shape (F,H,W,3)
                        tile_result = windsorized_sigma_clip_4d(
                            tile_stack,
                            lower=self.sigma_low,
                            upper=self.sigma_high
                        )
                        # If the function returns a tuple, extract the first element.
                        if isinstance(tile_result, tuple):
                            tile_result = tile_result[0]
                    else:
                        # tile_stack => shape (F,H,W,1) or (F,H,W)
                        # If shape=(F,H,W,1), we can squeeze or just call 3D version
                        tile_stack_3d = tile_stack[..., 0] if tile_stack.ndim == 4 else tile_stack
                        tile_result_3d = windsorized_sigma_clip_3d(tile_stack_3d, lower=self.sigma_low, upper=self.sigma_high)
                        # If the function returns a tuple, extract the first element.
                        if isinstance(tile_result_3d, tuple):
                            tile_result_3d = tile_result_3d[0]
                        # Now, ensure the result has shape (H, W, 1)
                        tile_result = tile_result_3d[..., np.newaxis]

                    # (E) Store tile_result in final_stacked
                    final_stacked[y_start:y_end, x_start:x_end, :] = tile_result

            # Convert final_stacked to a normal array
            master_dark_data = np.array(final_stacked)
            del final_stacked

            # (F) Save Master Dark
            
            master_dark_stem = f"MasterDark_{int(exposure_time)}s_{image_size}"
            master_dark_path = self._build_out(master_dir, master_dark_stem, "fit")

            # Build a minimal header
            # Possibly store EXPTIME, IMAGETYP="DARK", etc.
            master_header = fits.Header()
            master_header["IMAGETYP"] = "DARK"
            master_header["EXPTIME"]  = (exposure_time, "User-specified or from grouping")
            # plus any other fields you want

            # Remove NAXIS from the old ref_header if you want
            # or define them fresh
            master_header["NAXIS"] = 3 if channels==3 else 2
            master_header["NAXIS1"] = master_dark_data.shape[1]
            master_header["NAXIS2"] = master_dark_data.shape[0]
            if channels==3:
                master_header["NAXIS3"] = 3

            save_image(
                img_array=master_dark_data,
                filename=master_dark_path,
                original_format="fit",
                bit_depth="32-bit floating point",
                original_header=master_header,
                is_mono=(channels==1)
            )

            # (G) Add to tree, status, etc.
            self.add_master_dark_to_tree(f"{exposure_time}s ({image_size})", master_dark_path)
            self.update_status(f"✅ Master Dark saved: {master_dark_path}")
            self.assign_best_master_files()
            self.save_master_paths_to_settings()

        # Finally, assign best master dark, etc.
        self.assign_best_master_dark()
        self.update_override_dark_combo()
        self.assign_best_master_files()



    def save_master_dark(self, master_dark, output_path, exposure_time, is_mono):
        """Saves the master dark as 32-bit floating point FITS while maintaining OSC structure."""
        if is_mono:
            # Mono => shape (H, W)
            h, w = master_dark.shape
            # Wrap in an HDU
            hdu_data = master_dark.astype(np.float32)
            hdu = fits.PrimaryHDU(hdu_data)
            image_size = f"{w}x{h}"  # Width x Height
        else:
            # Color => shape (H, W, C)
            h, w, c = master_dark.shape
            # Transpose to (C, H, W)
            hdu_data = master_dark.transpose(2, 0, 1).astype(np.float32)
            hdu = fits.PrimaryHDU(hdu_data)
            image_size = f"{w}x{h}"

        # Now 'hdu' is a fits.PrimaryHDU in both branches
        hdr = hdu.header
        hdr["SIMPLE"]   = True
        hdr["BITPIX"]   = -32
        hdr["NAXIS"]    = 3 if not is_mono else 2
        hdr["NAXIS1"]   = w  # Width
        hdr["NAXIS2"]   = h  # Height
        if not is_mono:
            hdr["NAXIS3"] = c
        hdr["BSCALE"]   = 1.0
        hdr["BZERO"]    = 0.0
        hdr["IMAGETYP"] = "MASTER DARK"
        hdr["EXPOSURE"] = exposure_time
        hdr["DATE-OBS"] = datetime.utcnow().isoformat()
        hdr["CREATOR"]  = "SetiAstroSuite"

        # Write the FITS file
        hdu.writeto(output_path, overwrite=True)

        # Store Master Dark path with correct key
        key = f"{exposure_time}s ({image_size})"
        self.master_files[key] = output_path
        self.master_sizes[output_path] = image_size

        print(f"✅ Master Dark FITS saved: {output_path}")
        self.update_status(f"✅ Stored Master Dark -> {key}: {output_path}")



            
    def add_master_dark_to_tree(self, exposure_time, master_dark_path):
        """ Adds the newly created Master Dark to the Master Dark TreeBox and updates the dropdown. """

        exposure_key = f"{exposure_time}s"

        # ✅ Store in the dictionary
        self.master_files[exposure_key] = master_dark_path  # Store master dark
        print(f"📝 DEBUG: Stored Master Dark -> {exposure_key}: {master_dark_path}")

        # ✅ Update UI Tree
        existing_items = self.master_dark_tree.findItems(exposure_key, Qt.MatchFlag.MatchExactly, 0)

        if existing_items:
            exposure_item = existing_items[0]
        else:
            exposure_item = QTreeWidgetItem([exposure_key])
            self.master_dark_tree.addTopLevelItem(exposure_item)

        master_item = QTreeWidgetItem([os.path.basename(master_dark_path)])
        exposure_item.addChild(master_item)

        # ✅ Refresh the override dropdown
        self.update_override_dark_combo()
        self.assign_best_master_dark()  # 🔥 Ensure auto-selection works

        self.update_status(f"✅ Master Dark saved and added to UI: {master_dark_path}")



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
        """ Populates the dropdown with available Master Darks and prevents duplicate entries. """
        self.override_dark_combo.clear()
        self.override_dark_combo.addItem("None (Use Auto-Select)")
        self.override_dark_combo.addItem("None (Use no Dark to Calibrate)")

        seen_files = set()
        for exposure, path in self.master_files.items():
            file_name = os.path.basename(path)
            if file_name not in seen_files:
                self.override_dark_combo.addItem(f"{file_name} ({exposure})")
                seen_files.add(file_name)

        print("✅ DEBUG: Updated Override Master Dark dropdown with unique entries.")


    def override_selected_master_dark_for_flats(self):
        """ Overrides the selected master dark for the currently highlighted flat group. """
        selected_items = self.flat_tree.selectedItems()
        if not selected_items:
            return

        new_dark = self.override_dark_combo.currentText()

        # ✅ Handle "None (Use no Dark to Calibrate)" explicitly
        if new_dark == "None (Use no Dark to Calibrate)":
            new_dark = "No Calibration"  # Show "No Calibration" in the UI
        elif new_dark == "None (Use Auto-Select)":
            new_dark = None  # Auto-select behavior

        for item in selected_items:
            if item.parent():  # Ensure it's an exposure group, not the top filter name
                item.setText(2, new_dark if new_dark else "Auto")

        print(f"✅ DEBUG: Override Master Dark set to: {new_dark}")

    def create_master_flat(self):
        """ Creates master flats using per-frame dark subtraction before stacking. """

        if not self.stacking_directory:
            QMessageBox.warning(self, "Error", "Please set the stacking directory first using the wrench button.")
            return

        exposure_tolerance = self.flat_exposure_tolerance_spinbox.value()
        flat_files_by_group = {}  # Group by (Exposure, Image Size, Filter, Session)

        # Group Flats by Filter, Exposure & Size within Tolerance
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

        # Create output folder
        master_dir = os.path.join(self.stacking_directory, "Master_Calibration_Files")
        os.makedirs(master_dir, exist_ok=True)

        # Stack each grouped flat set
        for (exposure_time, image_size, filter_name, session), file_list in flat_files_by_group.items():
            if len(file_list) < 2:
                self.update_status(f"⚠️ Skipping {exposure_time}s ({image_size}) [{filter_name}] [{session}] - Not enough frames to stack.")
                continue

            self.update_status(f"🟢 Processing {len(file_list)} flats for {exposure_time}s ({image_size}) [{filter_name}] in session '{session}'...")
            QApplication.processEvents()

            # Load master dark
            best_diff = float("inf")
            selected_master_dark = None
            for key, path in self.master_files.items():
                match = re.match(r"([\d.]+)s", key)
                if not match:
                    continue
                dark_exposure = float(match.group(1))
                dark_size = self.master_sizes.get(path, "Unknown")
                if dark_size == image_size:
                    diff = abs(dark_exposure - exposure_time)
                    if diff < best_diff:
                        best_diff = diff
                        selected_master_dark = path

            if selected_master_dark:
                dark_data, _, _, _ = load_image(selected_master_dark)
            else:
                dark_data = None
                self.update_status("DEBUG: No matching Master Dark found.")

            # Load reference image
            ref_data, _, _, _ = load_image(file_list[0])
            if ref_data is None:
                self.update_status(f"❌ Failed to load reference {os.path.basename(file_list[0])}")
                continue

            height, width = ref_data.shape[:2]
            channels = 1 if ref_data.ndim == 2 else 3
            memmap_path = os.path.join(master_dir, f"temp_flat_{session}_{exposure_time}_{image_size}_{filter_name}.dat")
            final_stacked = np.memmap(memmap_path, dtype=self._dtype(), mode="w+", shape=(height, width, channels))
            num_frames = len(file_list)

            for y_start in range(0, height, self.chunk_height):
                y_end = min(y_start + self.chunk_height, height)
                tile_h = y_end - y_start
                for x_start in range(0, width, self.chunk_width):
                    x_end = min(x_start + self.chunk_width, width)
                    tile_w = x_end - x_start
                    tile_stack = np.zeros((num_frames, tile_h, tile_w, channels), dtype=np.float32)

                    with ThreadPoolExecutor() as executor:
                        
                        futures = {
                            executor.submit(load_fits_tile, f, y_start, y_end, x_start, x_end): idx
                            for idx, f in enumerate(file_list)
                        }

                        for future in as_completed(futures):
                            i = futures[future]
                            sub_img = future.result()
                            if sub_img is None:
                                self.update_status(f"⚠️ Skipping tile {i} due to load failure.")
                                continue

                            # Ensure correct shape
                            if sub_img.ndim == 2:
                                sub_img = sub_img[:, :, np.newaxis]
                                if channels == 3:
                                    sub_img = np.repeat(sub_img, 3, axis=2)
                            elif sub_img.shape[0] == 3:
                                sub_img = sub_img.transpose(1, 2, 0)

                            tile_stack[i] = sub_img


                    if dark_data is not None:
                        # prepare a matching HWC dark tile (float32, contiguous)
                        dsub = dark_data
                        if dsub.ndim == 3 and dsub.shape[0] in (1,3):   # CHW -> HWC
                            dsub = dsub.transpose(1, 2, 0)
                        d_tile = dsub[y_start:y_end, x_start:x_end]
                        d_tile = _as_C(d_tile.astype(np.float32, copy=False))
                        if d_tile.ndim == 2 and channels == 3:
                            d_tile = np.repeat(d_tile[..., None], 3, axis=2)

                        # in-place subtraction for entire stack (F,H,W,C)
                        _subtract_dark_stack_inplace_hwc(tile_stack, d_tile, pedestal=0.0)

                    if channels == 3:
                        tile_result = windsorized_sigma_clip_4d(tile_stack, lower=self.sigma_low, upper=self.sigma_high)[0]
                    else:
                        stack_3d = tile_stack[..., 0]
                        tile_result = windsorized_sigma_clip_3d(stack_3d, lower=self.sigma_low, upper=self.sigma_high)[0]
                        tile_result = tile_result[..., np.newaxis]

                    final_stacked[y_start:y_end, x_start:x_end, :] = tile_result

            master_flat_data = np.array(final_stacked)
            del final_stacked

            master_flat_stem = f"MasterFlat_{session}_{int(exposure_time)}s_{image_size}_{filter_name}"
            master_flat_path = self._build_out(master_dir, master_flat_stem, "fit")

            header = fits.Header()
            header["IMAGETYP"] = "FLAT"
            header["EXPTIME"] = (exposure_time, "grouped exposure")
            header["FILTER"] = filter_name
            header["NAXIS"] = 3 if channels == 3 else 2
            header["NAXIS1"] = width
            header["NAXIS2"] = height
            if channels == 3:
                header["NAXIS3"] = 3

            save_image(
                img_array=master_flat_data,
                filename=master_flat_path,
                original_format="fit",
                bit_depth="32-bit floating point",
                original_header=header,
                is_mono=(channels == 1)
            )

            key = f"{filter_name} ({image_size}) [{session}]"
            self.master_files[key] = master_flat_path
            self.master_sizes[master_flat_path] = image_size
            self.add_master_flat_to_tree(filter_name, master_flat_path)
            self.update_status(f"✅ Master Flat saved: {master_flat_path}")
            self.save_master_paths_to_settings()

        self.assign_best_master_dark()
        self.assign_best_master_files()



    def save_master_flat(self, master_flat, output_path, exposure_time, filter_name):
        """ Saves master flat as both a 32-bit floating point FITS and TIFF while ensuring no unintended normalization. """

        # ✅ Retrieve FITS header from a sample flat (to check if it's mono or color)
        original_header = None
        is_mono = True  # Default to mono

        if self.flat_files:
            sample_flat = next(iter(self.flat_files.values()))[0]  # Get the first flat file
            try:
                with fits.open(sample_flat) as hdul:
                    original_header = hdul[0].header

                    # **🔍 Detect if the flat is color by checking NAXIS3**
                    if original_header.get("NAXIS", 2) == 3 and original_header.get("NAXIS3", 1) == 3:
                        is_mono = False  # ✅ It's a color flat

            except Exception as e:
                print(f"⚠️ Warning: Could not retrieve FITS header from {sample_flat}: {e}")

        # ✅ Explicitly ensure we are saving raw values (NO normalization)
        fits_header = original_header if original_header else fits.Header()
        fits_header["BSCALE"] = 1.0  # 🔹 Prevent rescaling
        fits_header["BZERO"] = 0.0   # 🔹 Prevent offset

        # ✅ Save as FITS
        save_image(
            img_array=master_flat,
            filename=output_path,
            original_format="fit",
            bit_depth="32-bit floating point",
            original_header=fits_header,
            is_mono=is_mono
        )

        print(f"✅ Master Flat FITS saved: {output_path}")




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

    def assign_best_master_files(self):
        """ Assign best matching Master Dark and Flat to each Light Frame (per leaf). """
        print("\n🔍 DEBUG: Assigning best Master Darks & Flats to Lights...\n")

        if not self.master_files:
            print("⚠️ WARNING: No Master Calibration Files available.")
            self.update_status("⚠️ WARNING: No Master Calibration Files available.")
            return

        for i in range(self.light_tree.topLevelItemCount()):
            filter_item = self.light_tree.topLevelItem(i)
            filter_name = filter_item.text(0)
            filter_name     = self._sanitize_name(filter_name)

            for j in range(filter_item.childCount()):
                exposure_item = filter_item.child(j)
                exposure_text = exposure_item.text(0)

                match = re.match(r"([\d.]+)s?", exposure_text)
                if not match:
                    print(f"⚠️ WARNING: Could not parse exposure time from {exposure_text}")
                    continue

                exposure_time = float(match.group(1))

                for k in range(exposure_item.childCount()):
                    leaf_item = exposure_item.child(k)
                    meta_text = leaf_item.text(1)

                    size_match = re.search(r"Size: (\d+x\d+)", meta_text)
                    session_match = re.search(r"Session: ([^|]+)", meta_text)
                    image_size = size_match.group(1) if size_match else "Unknown"
                    session_name = session_match.group(1).strip() if session_match else "Default"

                    print(f"🧠 Leaf: {leaf_item.text(0)} | Size: {image_size} | Session: {session_name}")

                    # 🔍 Match Dark
                    best_dark_match = None
                    best_dark_diff = float("inf")

                    for master_key, master_path in self.master_files.items():
                        # ✅ Only consider keys that start with an exposure (i.e. darks)
                        dark_match = re.match(r"^([\d.]+)s\b", master_key)
                        if not dark_match:
                            continue
                        master_dark_exposure_time = float(dark_match.group(1))

                        # Ensure we know the dark’s size
                        master_dark_size = self.master_sizes.get(master_path, "Unknown")
                        if master_dark_size == "Unknown":
                            with fits.open(master_path) as hdul:
                                master_dark_size = f"{hdul[0].data.shape[1]}x{hdul[0].data.shape[0]}"
                                self.master_sizes[master_path] = master_dark_size

                        # Only compare if sizes match
                        if master_dark_size == image_size:
                            diff = abs(master_dark_exposure_time - exposure_time)
                            if diff < best_dark_diff:
                                best_dark_match = master_path
                                best_dark_diff = diff

                    # 🔍 Match Flat
                    best_flat_match = None
                    for flat_key, flat_path in self.master_files.items():
                        if filter_name not in flat_key or f"({image_size})" not in flat_key:
                            continue
                        if session_name in flat_key:
                            best_flat_match = flat_path
                            break
                    if not best_flat_match:
                        fallback_key = f"{filter_name} ({image_size})"
                        best_flat_match = self.master_files.get(fallback_key)

                    # 🔄 Assign to leaf
                    leaf_item.setText(2, os.path.basename(best_dark_match) if best_dark_match else "None")
                    leaf_item.setText(3, os.path.basename(best_flat_match) if best_flat_match else "None")

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
        self.assign_best_master_files()

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
                            flat_data = flat_data.astype(np.float32, copy=False)
                            flat_data[flat_data == 0] = 1.0  # safety
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
                    if not is_mono and light_data.ndim == 3 and light_data.shape[0] == 3:
                        light_data = light_data.transpose(1, 2, 0)  # CHW -> HWC

                    min_val = float(light_data.min())
                    max_val = float(light_data.max())
                    self.update_status(f"Before saving: min = {min_val:.4f}, max = {max_val:.4f}")
                    print(f"Before saving: min = {min_val:.4f}, max = {max_val:.4f}")
                    QApplication.processEvents()

                    calibrated_filename = os.path.join(
                        calibrated_dir, os.path.basename(light_file).replace(".fit", "_c.fit")
                    )

                    save_image(
                        img_array=light_data,
                        filename=calibrated_filename,
                        original_format="fit",
                        bit_depth=bit_depth,
                        original_header=hdr,
                        is_mono=is_mono
                    )

                    processed_files += 1
                    self.update_status(f"Saved: {os.path.basename(calibrated_filename)} ({processed_files}/{total_files})")
                    QApplication.processEvents()

        self.update_status("✅ Calibration Complete!")
        QApplication.processEvents()
        self.populate_calibrated_lights()


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
        Try common FITS keys for camera/sky position angle.
        Fallback: estimate from WCS CD/PC matrix (CROTA2-ish).
        Returns float degrees or None.
        """
        if hdr is None:
            return None
        keys = ("POSANGLE","ANGLE","ROTANGLE","ROTSKYPA","ROTATOR",
                "PA","CROTA2","CROTA1")
        for k in keys:
            if k in hdr:
                try:
                    return float(hdr[k])
                except Exception:
                    pass
        # crude WCS fallback (angle of +Y axis on detector)
        try:
            cd11 = float(hdr.get('CD1_1', hdr.get('PC1_1')))
            cd12 = float(hdr.get('CD1_2', hdr.get('PC1_2')))
            cd22 = float(hdr.get('CD2_2', hdr.get('PC2_2')))
            # common CROTA2-style estimate
            pa = np.degrees(np.arctan2(-cd12, cd22))
            return float(pa)
        except Exception:
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

    def register_images(self):

        if getattr(self, "_registration_busy", False):
            self.update_status("⏸ Registration already running; ignoring extra click.")
            return

        self._set_registration_busy(True)

        try:
            """Measure → choose reference → DEBAYER ref → DEBAYER+normalize all → align."""

            if self.star_trail_mode:
                self.update_status("🌠 Star-Trail Mode enabled: skipping registration & using max-value stack")
                QApplication.processEvents()
                return self._make_star_trail()

            self.update_status("🔄 Image Registration Started...")
            self.extract_light_files_from_tree(debug=True)

            # Determine comet mode from checkbox (safe if widget missing)
            comet_mode = bool(getattr(self, "comet_cb", None) and self.comet_cb.isChecked())

            if comet_mode:
                # Force a seed BEFORE any loops or registration compute
                self.update_status("🌠 Comet mode: please click the comet center to continue…")
                QApplication.processEvents()

                ok = self._ensure_comet_seed_now()
                if not ok:
                    # Hard-stop: we don't want the trash fallback
                    QMessageBox.information(
                        self, "Comet Mode",
                        "No comet center was selected. Registration has been cancelled so you can try again."
                    )
                    self.update_status("❌ Registration cancelled (no comet seed).")
                    return
                else:
                    # Clear any stale mapped ref coords from a previous run; we’ll recompute post-align
                    self._comet_ref_xy = None
                    self.update_status("✅ Comet seed set. Proceeding with registration…")
                    QApplication.processEvents()

                

            if not self.light_files:
                self.update_status("⚠️ No light files to register!")
                return

            # Which groups are selected? (used for optional dual-band split)
            selected_groups = set()
            for it in self.reg_tree.selectedItems():
                top = it if it.parent() is None else it.parent()
                selected_groups.add(top.text(0))

            if self.split_dualband_cb.isChecked():
                self.update_status("🌈 Splitting dual-band OSC frames into Ha / SII / OIII...")
                self._split_dual_band_osc(selected_groups=selected_groups)
                self._refresh_reg_tree_from_light_files()

            self._maybe_warn_cfa_low_frames()

            # Flatten to get all files
            all_files = [f for lst in self.light_files.values() for f in lst]
            self.update_status(f"📊 Found {len(all_files)} total frames. Now measuring in parallel batches...")

            # ─────────────────────────────────────────────────────────────────────
            # Helpers
            # ─────────────────────────────────────────────────────────────────────
            def mono_preview_for_stats(img: np.ndarray, hdr) -> np.ndarray:
                """
                Returns a 2D float32 preview for measurement:
                • RGB → luma → 2×2 superpixel average
                • CFA (2D with BAYERPAT) → 2×2 superpixel average
                • Mono → 2×2 superpixel average
                Always returns float32 and trims odd edges to keep even dims.
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
                        # 2D mono/CFA
                        return (x[0:h2:2, 0:w2:2] + x[0:h2:2, 1:w2:2] +
                                x[1:h2:2, 0:w2:2] + x[1:h2:2, 1:w2:2]) * 0.25
                    else:
                        # 3D: bin on luma, not per-channel
                        r = x[..., 0]; g = x[..., 1]; b = x[..., 2]
                        luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
                        return (luma[0:h2:2, 0:w2:2] + luma[0:h2:2, 1:w2:2] +
                                luma[1:h2:2, 0:w2:2] + luma[1:h2:2, 1:w2:2]) * 0.25

                # RGB
                if img.ndim == 3 and img.shape[-1] == 3:
                    return _superpixel2x2(img)

                # CFA hinted by header (keep simple: any BAYERPAT means mosaic)
                if hdr and hdr.get('BAYERPAT') and img.ndim == 2:
                    return _superpixel2x2(img)

                # Mono (or already debayered to 2D)
                if img.ndim == 2:
                    return _superpixel2x2(img)

                # Fallback
                return img.astype(np.float32, copy=False)

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

            max_workers = os.cpu_count() or 4
            chunk_size = max_workers
            chunks = list(chunk_list(all_files, chunk_size))
            total_chunks = len(chunks)

            from concurrent.futures import ThreadPoolExecutor, as_completed

            for idx, chunk in enumerate(chunks, 1):
                self.update_status(f"📦 Measuring chunk {idx}/{total_chunks} ({len(chunk)} frames)")
                chunk_images = []
                chunk_valid_files = []

                self.update_status(f"🌍 Loading {len(chunk)} images in parallel (up to {max_workers} threads)...")
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_file = {executor.submit(load_image, fp): fp for fp in chunk}
                    for fut in as_completed(future_to_file):
                        fp = future_to_file[fut]
                        try:
                            img, hdr, _, _ = fut.result()
                            if img is None:
                                continue
                            preview = mono_preview_for_stats(img, hdr)
                            if preview is None:
                                continue
                            chunk_images.append(preview)
                            chunk_valid_files.append(fp)
                            self.update_status(f"  Loaded {fp}")
                        except Exception as e:
                            self.update_status(f"⚠️ Error loading {fp}: {e}")
                        QApplication.processEvents()

                if not chunk_images:
                    self.update_status("⚠️ No valid images in this chunk.")
                    continue

                self.update_status("🌍 Measuring global means in parallel...")
                means = parallel_measure_frames(chunk_images)  # expects list/stack of 2D previews

                # star counts per preview
                for i, fp in enumerate(chunk_valid_files):
                    mv = float(means[i])
                    mean_values[fp] = mv
                    c, ecc = compute_star_count(chunk_images[i])  # 2D preview
                    star_counts[fp] = {"count": c, "eccentricity": ecc}
                    measured_frames.append(fp)

                del chunk_images

            if not measured_frames:
                self.update_status("⚠️ No frames could be measured!")
                return

            self.update_status(f"✅ All chunks complete! Measured {len(measured_frames)} frames total.")

            # ─────────────────────────────────────────────────────────────────────
            # Pick reference & compute weights
            # ─────────────────────────────────────────────────────────────────────
            self.update_status("⚖️ Computing frame weights...")
            debug_log = "\n📊 **Frame Weights Debug Log:**\n"
            for fp in measured_frames:
                c = star_counts[fp]["count"]
                ecc = star_counts[fp]["eccentricity"]
                m = mean_values[fp]
                c = max(c, 1)
                m = max(m, 1e-6)
                raw_w = (c * min(1.0, max(1.0 - ecc, 0.0))) / m
                self.frame_weights[fp] = raw_w
                debug_log += f"📂 {os.path.basename(fp)} → StarCount={c}, Ecc={ecc:.4f}, Mean={m:.4f}, Weight={raw_w:.4f}\n"
            self.update_status(debug_log)
            QApplication.processEvents()

            max_w = max(self.frame_weights.values()) if self.frame_weights else 0.0
            if max_w > 0:
                for k in self.frame_weights:
                    self.frame_weights[k] /= max_w

            # Choose reference (path)
            if getattr(self, "reference_frame", None):
                self.update_status(f"📌 Using user-specified reference: {self.reference_frame}")
            else:
                self.reference_frame = self.select_reference_frame_robust(self.frame_weights, sigma_threshold=2.0)
                self.update_status(f"📌 Auto-selected robust reference frame: {self.reference_frame}")

            # Stats for the chosen reference from the measurement pass
            ref_stats_meas = star_counts.get(self.reference_frame, {"count": 0, "eccentricity": 0.0})
            ref_count = ref_stats_meas["count"]
            ref_ecc   = ref_stats_meas["eccentricity"]

            # ─────────────────────────────────────────────────────────────────────
            # Debayer the reference ONCE and compute ref_median from debayered ref
            # ─────────────────────────────────────────────────────────────────────
            ref_img_raw, ref_hdr, _, _ = load_image(self.reference_frame)
            if ref_img_raw is None:
                self.update_status(f"🚨 Could not load reference {self.reference_frame}. Aborting.")
                return

            # If CFA, debayer; if already color, keep; if mono but 3D with last=1, squeeze.
            if ref_hdr and ref_hdr.get('BAYERPAT') and not ref_hdr.get('SPLITDB', False) and (ref_img_raw.ndim == 2 or (ref_img_raw.ndim == 3 and ref_img_raw.shape[-1] == 1)):
                self.update_status("📦 Debayering reference frame…")
                ref_img = self.debayer_image(ref_img_raw, self.reference_frame, ref_hdr)  # HxWx3
            else:
                ref_img = ref_img_raw
                if ref_img.ndim == 3 and ref_img.shape[-1] == 1:
                    ref_img = np.squeeze(ref_img, axis=-1)

            # Use luma median if color, else direct median
            if ref_img.ndim == 3 and ref_img.shape[-1] == 3:
                r, g, b = ref_img[..., 0], ref_img[..., 1], ref_img[..., 2]
                ref_luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
                ref_median = float(np.median(ref_luma))
            else:
                ref_median = float(np.median(ref_img))

            self.update_status(f"📊 Reference (debayered) median: {ref_median:.4f}")

            # Show review dialog; if user changes reference, redo debayer+median
            stats_payload = {"star_count": ref_count, "eccentricity": ref_ecc, "mean": ref_median}

            if self.auto_accept_ref_cb.isChecked():
                # ✅ Auto-accept path: no dialog
                self.update_status("✅ Auto-accept measured reference is enabled; using the measured best frame.")
                # If you show the chosen path somewhere in the UI, update it:
                try:
                    self.ref_frame_path.setText(self.reference_frame or "No file selected")
                except Exception:
                    pass
            else:
                # Show review dialog (existing behavior)
                dialog = ReferenceFrameReviewDialog(self.reference_frame, stats_payload, parent=self)

                # Make it non-modal but raised/focused
                dialog.setModal(False)
                dialog.setWindowModality(Qt.WindowModality.NonModal)
                dialog.show()
                dialog.raise_()
                dialog.activateWindow()

                # Wait here without freezing UI (modeless pseudo-modal)
                _loop = QEventLoop(self)
                dialog.finished.connect(_loop.quit)   # finished(int) -> quit loop
                _loop.exec()

                # After the user closes the dialog, proceed exactly as before:
                result = dialog.result()
                user_choice = dialog.getUserChoice()   # "use", "select_other", or None

                if result == QDialog.DialogCode.Accepted:
                    self.update_status("User accepted the reference frame.")
                elif user_choice == "select_other":
                    new_ref = self.prompt_for_reference_frame()
                    if new_ref:
                        self.reference_frame = new_ref
                        self.update_status(f"User selected a new reference frame: {new_ref}")
                        # re-load and debayer/median the new reference (same logic as above)
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
                            ref_luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
                            ref_median = float(np.median(ref_luma))
                        else:
                            ref_median = float(np.median(ref_img))
                        self.update_status(f"📊 (New) reference median: {ref_median:.4f}")
                    else:
                        self.update_status("No new reference selected; using previous reference.")
                else:
                    self.update_status("Dialog closed without explicit choice; using selected reference.")

            ref_L = _luma(ref_img)
            ref_min = float(np.nanmin(ref_L))
            ref_target_median = float(np.nanmedian(ref_L - ref_min))
            self.update_status(f"📊 Reference min={ref_min:.6f}, normalized-median={ref_target_median:.6f}")

            # ─────────────────────────────────────────────────────────────────────
            # PHASE 1b: Meridian flips
            # ─────────────────────────────────────────────────────────────────────
            ref_pa = self._extract_pa_deg(ref_hdr)
            self.update_status(f"🧭 Reference PA: {ref_pa:.2f}°" if ref_pa is not None else "🧭 Reference PA: (unknown)")

            # ─────────────────────────────────────────────────────────────────────
            # PHASE 2: normalize (DEBAYER everything once here)
            # ─────────────────────────────────────────────────────────────────────
            norm_dir = os.path.join(self.stacking_directory, "Normalized_Images")
            os.makedirs(norm_dir, exist_ok=True)

            normalized_files = []
            chunks = list(chunk_list(measured_frames, chunk_size))
            total_chunks = len(chunks)

            for idx, chunk in enumerate(chunks, 1):
                self.update_status(f"🌀 Normalizing chunk {idx}/{total_chunks} ({len(chunk)} frames)…")
                QApplication.processEvents()

                loaded_images = []
                valid_paths = []

                self.update_status(f"🌍 Loading {len(chunk)} images in parallel for normalization (up to {max_workers} threads)…")
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_file = {executor.submit(load_image, fp): fp for fp in chunk}
                    for fut in as_completed(future_to_file):
                        fp = future_to_file[fut]
                        try:
                            img, hdr, _, _ = fut.result()
                            if img is None:
                                self.update_status(f"⚠️ No data for {fp}")
                                continue

                            # Debayer ONCE here if CFA
                            if hdr and hdr.get('BAYERPAT') and not hdr.get('SPLITDB', False) and (img.ndim == 2 or (img.ndim == 3 and img.shape[-1] == 1)):
                                img = self.debayer_image(img, fp, hdr)  # → HxWx3
                            else:
                                if img.ndim == 3 and img.shape[-1] == 1:
                                    img = np.squeeze(img, axis=-1)

                            # --- Meridian flip assist: auto 180° rotate by header PA ---
                            if self.auto_rot180 and ref_pa is not None:
                                pa = self._extract_pa_deg(hdr)
                                img, did = self._maybe_rot180(img, pa, ref_pa, self.auto_rot180_tol_deg)
                                if did:
                                    self.update_status(f"↻ 180° rotate (PA Δ≈180°): {os.path.basename(fp)}")
                                    # Optional: mark header so we remember we pre-rotated
                                    QApplication.processEvents()
                                    try:
                                        hdr['ROT180'] = (True, 'Rotated 180° pre-align by SAS')
                                    except Exception:
                                        pass

                            loaded_images.append(img)
                            valid_paths.append(fp)
                        except Exception as e:
                            self.update_status(f"⚠️ Error loading {fp} for normalization: {e}")
                        QApplication.processEvents()

                if not loaded_images:
                    continue

                stack = np.ascontiguousarray(np.array(loaded_images, dtype=np.float32))
                normalized_stack = normalize_images(stack, target_median=ref_target_median)

                # Optional ABE-style degree-2 background cleanup
                if self.settings.value("stacking/grad_poly2/enabled", False, type=bool):
                    mode         = "divide" if self.settings.value("stacking/grad_poly2/mode", "subtract") == "divide" else "subtract"
                    samples      = int(self.settings.value("stacking/grad_poly2/samples", 120, type=int))
                    downsample   = int(self.settings.value("stacking/grad_poly2/downsample", 6, type=int))
                    patch_size   = int(self.settings.value("stacking/grad_poly2/patch_size", 15, type=int))
                    min_strength = float(self.settings.value("stacking/grad_poly2/min_strength", 0.01, type=float))
                    gain_lo      = float(self.settings.value("stacking/grad_poly2/gain_lo", 0.20, type=float))
                    gain_hi      = float(self.settings.value("stacking/grad_poly2/gain_hi", 5.0, type=float))

                    # Log the plan
                    self.update_status(
                        f"Gradient removal (ABE Poly²): mode={mode}, samples={samples}, "
                        f"downsample={downsample}, patch={patch_size}, min_strength={min_strength*100:.2f}%, "
                        f"gain_clip=[{gain_lo},{gain_hi}]"
                    )
                    QApplication.processEvents()

                    normalized_stack = remove_gradient_stack_abe(
                        normalized_stack,
                        mode=mode,
                        num_samples=samples,
                        downsample=downsample,
                        patch_size=patch_size,
                        min_strength=min_strength,
                        gain_clip=(gain_lo, gain_hi),
                        log_fn=(self._ui_log if hasattr(self, "_ui_log") else self.update_status),
                    )
                    QApplication.processEvents()


                # Save each with "_n.fit"
                for i, orig_file in enumerate(valid_paths):
                    base = os.path.basename(orig_file)
                    # normalize suffix handling for .fit/.fits
                    if base.endswith("_n.fit"):
                        base = base.replace("_n.fit", ".fit")
                    if base.lower().endswith(".fits"):
                        out_name = base[:-5] + "_n.fit"
                    elif base.lower().endswith(".fit"):
                        out_name = base[:-4] + "_n.fit"
                    else:
                        out_name = base + "_n.fit"

                    out_path = os.path.join(norm_dir, out_name)
                    frame_data = normalized_stack[i]

                    try:
                        orig_header = fits.getheader(orig_file, ext=0)
                    except Exception:
                        orig_header = fits.Header()

                    # Mark representation to avoid re-debayering later
                    if isinstance(frame_data, np.ndarray) and frame_data.ndim == 3 and frame_data.shape[-1] == 3:
                        orig_header["DEBAYERED"] = (True, "Color debayered normalized")
                    else:
                        orig_header["DEBAYERED"] = (False, "Mono normalized")
                    self._orig2norm[os.path.normpath(orig_file)] = os.path.normpath(out_path)
                    hdu = fits.PrimaryHDU(data=frame_data.astype(np.float32), header=orig_header)
                    hdu.writeto(out_path, overwrite=True)
                    normalized_files.append(out_path)

                del loaded_images, stack, normalized_stack

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
            self.alignment_thread.progress_update.connect(self.update_status)
            self.alignment_thread.registration_complete.connect(self.on_registration_complete)

            self.align_progress = QProgressDialog("Aligning stars…", None, 0, 0, self)
            self.align_progress.setWindowModality(Qt.WindowModality.WindowModal)
            self.align_progress.setMinimumDuration(0)
            self.align_progress.setCancelButton(None)
            self.align_progress.setWindowTitle("Stellar Alignment")
            self.align_progress.setValue(0)
            self.align_progress.show()

            self.alignment_thread.progress_step.connect(self._on_align_progress)
            self.alignment_thread.registration_complete.connect(self._on_align_done)
            self.alignment_thread.start()
        except Exception as e:
            # on unexpected error, re-enable the UI
            self._set_registration_busy(False)
            raise
        
    @pyqtSlot(int, int)
    def _on_align_progress(self, done, total):
        self.align_progress.setLabelText(f"Aligning stars… ({done}/{total})")
        self.align_progress.setMaximum(total)
        self.align_progress.setValue(done)
        QApplication.processEvents()

    @pyqtSlot(bool, str)
    def _on_align_done(self, success, message):
        if hasattr(self, "align_progress"):
            self.align_progress.close()
            del self.align_progress
        self.update_status(message)

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
            lines = [line.strip() for line in block.splitlines() if line.strip()]
            if not lines:
                continue
            if lines[0].startswith("FILE:"):
                raw_file_path = lines[0].replace("FILE:", "").strip()
                # *** KEY FIX: normalize here
                curr_file = os.path.normpath(raw_file_path)
            else:
                continue
            
            if len(lines) < 4 or not lines[1].startswith("MATRIX:"):
                continue

            row0 = lines[2].split(",")
            row1 = lines[3].split(",")
            a, b, tx = [float(x) for x in row0]
            c, d, ty = [float(x) for x in row1]

            transforms[curr_file] = np.array([[a, b, tx],
                                            [c, d, ty]], dtype=np.float32)
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
            filters = "TIFF (*.tif);;PNG (*.png);;JPEG (*.jpg *.jpeg);;FITS (*.fits);;XISF (*.xisf')"
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
        """Return fraction of occupied (dx,dy) phase bins in [0,1)×[0,1)."""
        hist = np.zeros((bins, bins), dtype=np.int32)
        for M in matrices.values():
            # fractional translation at origin
            tx, ty = float(M[0,2]), float(M[1,2])
            fx = (tx - math.floor(tx)) % 1.0
            fy = (ty - math.floor(ty)) % 1.0
            ix = min(int(fx * bins), bins-1)
            iy = min(int(fy * bins), bins-1)
            hist[iy, ix] += 1
        return float(np.count_nonzero(hist)) / float(hist.size)

    def on_registration_complete(self, success, msg):
       
        self.update_status(msg)
        if not success:
            return

        alignment_thread = self.alignment_thread
        if alignment_thread is None:
            self.update_status("⚠️ Error: No alignment data available.")
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
        self.save_alignment_matrices_sasd(valid_matrices)

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

        mf_enabled = self.settings.value("stacking/mfdeconv/enabled", False, type=bool)
        if mf_enabled:
            self.update_status("🧪 Beta: Multi-frame PSF-aware deconvolution path enabled.")
            # choose which group to run; simplest: run on the first non-empty group
            target_group = next((g for g, lst in aligned_light_files.items() if lst), None)
            if target_group is None:
                self.update_status("⚠️ No aligned frames available for MF deconvolution.")
            else:
                frames = aligned_light_files[target_group]
                out_dir = os.path.join(self.stacking_directory, "Masters")
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, f"MasterLight_MFDeconv.fit")
                iters  = self.settings.value("stacking/mfdeconv/iters", 20, type=int)
                kappa  = self.settings.value("stacking/mfdeconv/kappa", 2.0, type=float)
                mode   = self.settings.value("stacking/mfdeconv/color_mode", "luma", type=str)
                huber  = self.settings.value("stacking/mfdeconv/huber_delta", 0.0, type=float)

                # Run synchronously on the post thread (we are in GUI thread here; kick to a worker)
                self.post_thread = QThread(self)
                self.post_worker = MultiFrameDeconvWorker(
                    parent=self,
                    aligned_paths=frames,
                    output_path=out_path,
                    iters=iters,
                    kappa=kappa,
                    color_mode=mode,
                    huber_delta=huber
                )
                self.post_worker.progress.connect(self._on_post_status)
                self.post_worker.finished.connect(self._on_post_pipeline_finished)
                self.post_worker.moveToThread(self.post_thread)
                self.post_thread.started.connect(self.post_worker.run)
                self.post_thread.start()

                # Progress dialog (reuse yours)
                self.post_progress = QProgressDialog("Multi-frame deconvolving…", None, 0, 0, self)
                self.post_progress.setWindowModality(Qt.WindowModality.WindowModal)
                self.post_progress.setCancelButton(None)
                self.post_progress.setMinimumDuration(0)
                self.post_progress.setWindowTitle("MF Deconvolution (Beta)")
                self.post_progress.show()
                return  # ⬅️ short-circuit the normal AfterAlignWorker for this beta path

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
            try:
                global_rect = self._compute_common_autocrop_rect(grouped_files, autocrop_pct, status_cb=log)
            except Exception as e:
                global_rect = None
                log(f"⚠️ Global crop failed: {e}")
            if global_rect is None:
                log("✂️ Global crop disabled; will fall back to per-group.")
            else:
                log("✂️ Auto Crop Bounding Box Calculated")
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
                    # derive a per-group rect once from this group's aligned images
                    try:
                        group_rect = self._compute_common_autocrop_rect(
                            {group_key: file_list},  # single-group dict
                            autocrop_pct,
                            status_cb=log
                        )
                    except Exception as e:
                        group_rect = None
                        log(f"⚠️ Per-group crop failed for '{group_key}': {e}")
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
                        M = self.valid_matrices.get(os.path.normpath(reg_path))
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
                CS.debug_save_marks(sorted_files, comet_xy,
                                    os.path.join(self.stacking_directory, "debug_comet_xy"))

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
                        # If you want a user knob for blackpoint, expose a spinbox and read it here.
                        blackpoint_pct = 50.0  # median; change to e.g. float(self.comet_blackpoint_pct.value())

                        log(f"🟡 Blending Stars+Comet (additive; bp={blackpoint_pct:.1f}%, mix={mix:.2f})…")
                        stars_img, comet_img = _match_channels(integrated_image, comet_only)

                        # New additive blend
                        blend = CS.blend_additive_comet(
                            comet_only=comet_img,
                            stars_only=stars_img,
                            blackpoint_percentile=blackpoint_pct,
                            per_channel=True,
                            mix=mix,
                            preserve_dtype=False
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

        # ---- Drizzle pass (only for groups with drizzle enabled) ----
        for group_key, file_list in grouped_files.items():
            dconf = drizzle_dict.get(group_key)
            if not (dconf and dconf.get("drizzle_enabled", False)):
                log(f"✅ Group '{group_key}' not set for drizzle. Integrated image already saved.")
                continue

            scale_factor = float(dconf["scale_factor"])
            drop_shrink  = float(dconf["drop_shrink"])
            rejections_for_group = group_integration_data[group_key]["rejection_map"]
            n_frames_group = group_integration_data[group_key]["n_frames"]

            log(f"📐 Drizzle for '{group_key}' at {scale_factor}× (drop={drop_shrink}) using {n_frames_group} frame(s).")

            self.drizzle_stack_one_group(
                group_key=group_key,
                file_list=file_list,
                transforms_dict=transforms_dict,   # kept for compatibility; method reloads from disk
                frame_weights=frame_weights,
                scale_factor=scale_factor,
                drop_shrink=drop_shrink,
                rejection_map=rejections_for_group,
                autocrop_enabled=autocrop_enabled,
                rect_override=global_rect,         # drizzle path already handles rect internally
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
                sources.append(_MMFits(p))
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
                        else:
                            log("  ◦ StarNet comet star removal…")
                            # Frames are linear at this stage
                            starless = CS.starnet_starless_from_array(
                                    warped, self.settings, is_linear=True
                                )

                        # Protect nucleus: blend original back under soft core mask
                        m3 = np.repeat(core_mask[..., None], 3, axis=2)
                        protected = np.clip(starless * (1.0 - m3) + warped * m3, 0.0, 1.0).astype(np.float32)

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



    def integrate_registered_images(self):
        """ 
        Integrates previously registered images (already aligned) without re-aligning them,
        but uses a chunked measurement approach so we don't load all frames at once.
        """
        self.update_status("🔄 Integrating Previously Registered Images...")

        # 1) Extract files from the registration tree
        self.extract_light_files_from_tree()
        if not self.light_files:
            self.update_status("⚠️ No registered images found!")
            return

        # Flatten the dictionary to get all registered files
        all_files = [f for file_list in self.light_files.values() for f in file_list]
        if not all_files:
            self.update_status("⚠️ No frames found in the registration tree!")
            return

        # 2) We'll measure means + star counts in chunks, so we don't load everything at once
        self.update_status(f"📊 Found {len(all_files)} total aligned frames. Measuring in parallel batches...")

        self.frame_weights = {}
        mean_values = {}
        star_counts = {}
        measured_frames = []

        # Decide how many images to load at once. Typically # of CPU cores:
        max_workers = os.cpu_count() or 4
        chunk_size = max_workers  # or a custom formula if you prefer

        def chunk_list(lst, size):
            for i in range(0, len(lst), size):
                yield lst[i : i + size]

        chunked_files = list(chunk_list(all_files, chunk_size))
        total_chunks = len(chunked_files)

        # 3) Process each chunk
        chunk_index = 0
        for chunk in chunked_files:
            chunk_index += 1
            self.update_status(f"📦 Loading and measuring chunk {chunk_index}/{total_chunks} with {len(chunk)} frames...")
            QApplication.processEvents()

            # Load this chunk of images
            images = []
            valid_files_for_this_chunk = []
            for file in chunk:
                image_data, _, _, _ = load_image(file)
                if image_data is not None:
                    images.append(image_data)
                    valid_files_for_this_chunk.append(file)
                else:
                    self.update_status(f"⚠️ Could not load {file}, skipping.")

            if not images:
                self.update_status("⚠️ No valid images in this chunk.")
                continue

            # Parallel measure the mean pixel value
            self.update_status("🌍 Measuring global statistics (mean) in parallel...")
            QApplication.processEvents()
            means = parallel_measure_frames(images)

            # Now measure star counts
            for i, file in enumerate(valid_files_for_this_chunk):
                mean_signal = means[i]
                mean_values[file] = mean_signal
                measured_frames.append(file)

                self.update_status(f"⭐ Measuring star stats for {file}...")
                QApplication.processEvents()
                count, ecc = compute_star_count(images[i])
                star_counts[file] = {"count": count, "eccentricity": ecc}

            # Clear the images from memory before moving on
            del images

        # If we never measured any frames at all
        if not measured_frames:
            self.update_status("⚠️ No frames could be measured!")
            return

        self.update_status(f"✅ All chunks complete! Measured {len(measured_frames)} frames total.")
        QApplication.processEvents()

        # 4) Compute Weights
        self.update_status("⚖️ Computing frame weights...")

        debug_weight_log = "\n📊 **Frame Weights Debug Log:**\n"
        QApplication.processEvents()
        for file in measured_frames:
            c = star_counts[file]["count"]
            ecc = star_counts[file]["eccentricity"]
            mean_val = mean_values[file]

            star_weight = max(c, 1e-6)
            mean_weight = max(mean_val, 1e-6)

            # Basic ratio-based weight: star_count / mean
            raw_weight = star_weight / mean_weight
            self.frame_weights[file] = raw_weight

            debug_weight_log += (
                f"📂 {os.path.basename(file)} → "
                f"Star Count: {c}, Mean: {mean_val:.4f}, Final Weight: {raw_weight:.4f}\n"
            )
            QApplication.processEvents()

        self.update_status(debug_weight_log)
        self.update_status("✅ Frame weights computed!")
        QApplication.processEvents()

        # 5) Pick the best reference frame if not user-specified
        if hasattr(self, "reference_frame") and self.reference_frame:
            self.update_status(f"📌 Using user-specified reference frame: {self.reference_frame}")
            QApplication.processEvents()
        else:
            self.reference_frame = max(self.frame_weights, key=self.frame_weights.get)
            self.update_status(f"📌 Auto-selected reference frame: {self.reference_frame} (Best Weight)")
            
        chunk_h = self.chunk_height  # or self.settings.value("stacking/chunk_height", 1024, type=int)
        chunk_w = self.chunk_width   # or self.settings.value("stacking/chunk_width", 1024, type=int)

        # 6) Finally, call the chunked stacking method using the already registered images
        self.stack_registered_images_chunked(self.light_files, self.frame_weights, chunk_height=chunk_h, chunk_width=chunk_w)

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

    def drizzle_stack_one_group(
        self,
        group_key,
        file_list,
        transforms_dict,
        frame_weights,
        scale_factor=2.0,
        drop_shrink=0.65,
        rejection_map=None,
        *,
        autocrop_enabled: bool = False,
        rect_override=None,
        status_cb=None
    ):
        log = status_cb or (lambda *_: None)

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

        transforms_path = os.path.join(self.stacking_directory, "alignment_transforms.sasd")
        if not os.path.exists(transforms_path):
            log(f"⚠️ No alignment_transforms.sasd found at {transforms_path}!")
            return

        new_transforms_dict = self.load_alignment_matrices_custom(transforms_path)
        log(f"✅ Loaded {len(new_transforms_dict)} transforms from disk for drizzle.")

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
        out_h = int(h * scale_factor)
        out_w = int(w * scale_factor)
        drizzle_buffer  = np.zeros((out_h, out_w) if is_mono else (out_h, out_w, c), dtype=self._dtype())
        coverage_buffer = np.zeros_like(drizzle_buffer, dtype=self._dtype())
        finalize_func   = finalize_drizzle_2d if is_mono else finalize_drizzle_3d

        # --- main loop ---
        for aligned_file in file_list:
            aligned_base = os.path.basename(aligned_file)
            raw_base = aligned_base.replace("_n_r.fit", "_n.fit") if aligned_base.endswith("_n_r.fit") else aligned_base
            raw_file = os.path.join(self.stacking_directory, "Normalized_Images", raw_base)

            raw_img_data, _, _, _ = load_image(raw_file)
            if raw_img_data is None:
                log(f"⚠️ Could not load raw file '{raw_file}' for drizzle!")
                continue

            raw_key = os.path.normpath(raw_file)
            transform = new_transforms_dict.get(raw_key, None)
            if transform is None:
                log(f"⚠️ No transform found for raw '{raw_base}'! Skipping drizzle.")
                continue

            log(f"🧩 Drizzling (raw): {raw_base}")
            log(f"    Matrix: [[{transform[0,0]:.4f}, {transform[0,1]:.4f}, {transform[0,2]:.4f}], "
                f"[{transform[1,0]:.4f}, {transform[1,1]:.4f}, {transform[1,2]:.4f}]]")

            weight = frame_weights.get(aligned_file, 1.0)
            if transform.dtype != np.float32:
                transform = transform.astype(np.float32)

            # dilation settings (square or diamond), defaults safe to 0 (disabled)
            dilate_px = int(self.settings.value("stacking/reject_dilate_px", 0, type=int))
            dilate_shape = self.settings.value("stacking/reject_dilate_shape", "square", type=str).lower()

            # precompute integer offsets for dilation (in ALIGNED space)
            _offsets = [(0, 0)]
            if dilate_px > 0:
                r = dilate_px
                if dilate_shape.startswith("dia"):  # diamond = L1 radius
                    _offsets = [(dx, dy) for dx in range(-r, r+1)
                                    for dy in range(-r, r+1)
                                    if (abs(dx) + abs(dy) <= r)]
                else:  # square = Chebyshev radius
                    _offsets = [(dx, dy) for dx in range(-r, r+1)
                                    for dy in range(-r, r+1)]

            # zero out rejected raw pixels (sparse, coordinate-based)
            coords_for_this_file = rejection_map.get(aligned_file, []) if rejection_map else []
            if coords_for_this_file:
                inv_transform = self.invert_affine_transform(transform)
                Hraw, Wraw = raw_img_data.shape[:2]
                for (x_r, y_r) in coords_for_this_file:
                    for (ox, oy) in _offsets:
                        xr, yr = x_r + ox, y_r + oy
                        x_raw, y_raw = self.apply_affine_transform_point(inv_transform, xr, yr)
                        x_raw = int(round(x_raw)); y_raw = int(round(y_raw))
                        if 0 <= x_raw < Wraw and 0 <= y_raw < Hraw:
                            raw_img_data[y_raw, x_raw] = 0.0

            # deposit
            if deposit_func is drizzle_deposit_numba_naive:
                drizzle_buffer, coverage_buffer = deposit_func(
                    raw_img_data, transform, drizzle_buffer, coverage_buffer,
                    scale_factor, weight
                )
            elif deposit_func is drizzle_deposit_color_naive:
                drizzle_buffer, coverage_buffer = deposit_func(
                    raw_img_data, transform, drizzle_buffer, coverage_buffer,
                    scale_factor, drop_shrink, weight
                )
            else:
                # kernelized (square/circular/gaussian)
                drizzle_buffer, coverage_buffer = deposit_func(
                    raw_img_data, transform, drizzle_buffer, coverage_buffer,
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


    def normal_integration_with_rejection(
        self,
        group_key,
        file_list,
        frame_weights,
        status_cb=None,
        *,
        algo_override: str | None = None  # <--- NEW
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

        log(f"📊 Stacking group '{group_key}' with {self.rejection_algorithm}")

        # --- keep all FITSes open (memmap) once for the whole group ---
        sources = []
        try:
            for p in file_list:
                sources.append(_MMFits(p))
        except Exception as e:
            for s in sources:
                s.close()
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
            for s in sources:
                s.close()
            log(f"⚠️ {e}")
            return None, {}, None

        # --- reusable Fortran-order tile buffer (frames contiguous) ---
        tile_stack_buf = np.empty((N, chunk_h, chunk_w, channels), dtype=np.float32, order='F')

        # --- weights once per group ---
        weights_array = np.array([frame_weights.get(p, 1.0) for p in file_list], dtype=np.float32)

        n_rows  = math.ceil(height / chunk_h)
        n_cols  = math.ceil(width  / chunk_w)
        total_tiles = n_rows * n_cols
        tile_idx = 0

        # --- group-level rejection maps (RAM-light) ---
        rej_any   = np.zeros((height, width), dtype=np.bool_)
        rej_count = np.zeros((height, width), dtype=np.uint16)

        for y0 in range(0, height, chunk_h):
            y1  = min(y0 + chunk_h, height)
            th  = y1 - y0
            for x0 in range(0, width, chunk_w):
                x1  = min(x0 + chunk_w, width)
                tw  = x1 - x0
                tile_idx += 1
                log(f"Integrating tile {tile_idx}/{total_tiles}…")

                # view into the reusable buffer with *actual* edge sizes
                ts = tile_stack_buf[:, :th, :tw, :channels]

                # sequential, low-overhead sliced reads (OS prefetch + memmap)
                for i, src in enumerate(sources):
                    sub = src.read_tile(y0, y1, x0, x1)  # float32, (th,tw) or (th,tw,3)
                    if sub.ndim == 2:
                        if channels == 3:
                            sub = sub[:, :, None].repeat(3, axis=2)
                        else:
                            sub = sub[:, :, None]  # unify to 3D for assignment
                    ts[i, :, :, :] = sub

                # --- rejection/integration ---
                algo = (algo_override or self.rejection_algorithm)
                print(f"  ◦ applying rejection algorithm: {algo}")

                if algo in ("Comet Median", "Simple Median (No Rejection)"):
                    tile_result  = np.median(ts, axis=0)
                    tile_rej_map = np.zeros((N, th, tw), dtype=bool)

                elif algo == "Comet High-Clip Percentile":
                    k = self.settings.value("stacking/comet_hclip_k", 1.30, type=float)
                    p = self.settings.value("stacking/comet_hclip_p", 25.0, type=float)
                    # keep a small dict across tiles to reuse scratch buffers

                    tile_result = _high_clip_percentile(ts, k=float(k), p=float(p))
                    tile_rej_map = np.zeros((len(file_list), th, tw), dtype=bool)

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
                    tile_result, tile_rej_map = max_value_stack(
                        ts, weights_array
                    )
                else:
                    tile_result, tile_rej_map = windsorized_sigma_clip_weighted(
                        ts, weights_array, lower=self.sigma_low, upper=self.sigma_high
                    )

                integrated_image[y0:y1, x0:x1, :] = tile_result

                # --- rejection bookkeeping ---
                trm = tile_rej_map
                if trm.ndim == 4:
                    trm = np.any(trm, axis=-1)  # collapse color → (N, th, tw)

                # accumulate group maps
                rej_any[y0:y1, x0:x1]  |= np.any(trm, axis=0)
                rej_count[y0:y1, x0:x1] += trm.sum(axis=0).astype(np.uint16)

                # per-file coords (existing behavior)
                for i, fpath in enumerate(file_list):
                    ys, xs = np.where(trm[i])
                    if ys.size:
                        per_file_rejections[fpath].extend(zip(x0 + xs, y0 + ys))

        # close mmapped FITSes
        for s in sources:
            s.close()

        if channels == 1:
            integrated_image = integrated_image[..., 0]

        # stash group-level maps so the *master* saver can MEF-embed them
        if not hasattr(self, "_rej_maps"):
            self._rej_maps = {}
        rej_frac = (rej_count.astype(np.float32) / float(max(1, N)))  # [0..1]
        self._rej_maps[group_key] = {"any": rej_any, "frac": rej_frac, "count": rej_count, "n": N}

        log(f"Integration complete for group '{group_key}'.")
        return integrated_image, per_file_rejections, ref_header



    def outlier_rejection_with_mask(self, tile_stack, weights_array):
        """
        Example outlier rejection routine that computes the weighted median of the tile stack
        and returns both the integrated tile and a rejection mask.
        
        Parameters:
        tile_stack: numpy array of shape (N, H, W, C)
        weights_array: numpy array of shape (N,)
        
        Returns:
        tile_result: numpy array of shape (H, W, C)
        rejection_mask: boolean numpy array of shape (H, W) where True indicates a rejected pixel.
        
        This is a simple example. Replace this logic with your actual rejection algorithm.
        """
        # Compute the weighted median along axis 0.
        # For simplicity, we'll use the unweighted median here.
        tile_result = np.median(tile_stack, axis=0)
        
        # Compute the absolute deviation for each frame and take the median deviation.
        # Then mark as rejected any pixel in any frame that deviates by more than a threshold.
        # Here we define a threshold factor (this value may need tuning).
        threshold_factor = 1.5
        abs_deviation = np.abs(tile_stack - tile_result)
        # Compute the median deviation per pixel over the frames.
        median_deviation = np.median(abs_deviation, axis=0)
        # Define a rejection mask: True if the median deviation exceeds a threshold.
        # (For demonstration, assume threshold = threshold_factor * some constant; here we choose 0.05.)
        rejection_mask = median_deviation[..., 0] > (threshold_factor * 0.05)
        # If color, you might combine channels or process each channel separately.
        
        return tile_result, rejection_mask

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
