#saspro/crop_dialog_pro.py
from __future__ import annotations

import math
import os
import sys
import weakref
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cv2
from typing import Optional
import platform
from PyQt6.QtCore import Qt, QEvent, QPointF, QRectF, pyqtSignal, QPoint, QTimer, QSettings, QByteArray, QSignalBlocker, QThread
from PyQt6.QtGui import QPixmap, QImage, QPen, QBrush, QColor, QPainterPath, QPainter, QCursor, QIcon
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox, QToolButton,
    QMessageBox, QGraphicsScene, QGraphicsView, QGraphicsRectItem, QGraphicsEllipseItem,
    QGraphicsItem, QGraphicsPixmapItem, QSpinBox, QDoubleSpinBox, QGraphicsPathItem,
    QGroupBox, QCheckBox, QScrollArea, QWidget, QSizePolicy, QFrame,
)

from setiastro.saspro.wcs_update import update_wcs_after_crop
from setiastro.saspro.widgets.themed_buttons import themed_toolbtn
from setiastro.saspro.help_support import make_help_button
from setiastro.saspro.shortcuts import PresetDragHandle
from setiastro.saspro.resources import get_icon_path
from setiastro.saspro.thread_keeper import start_owned_thread
from setiastro.saspro.widgets.factory_reset import (
    make_reset_button, snapshot_controls, restore_controls)

_ROTATION_CURSOR = None


def shared_axis_crop(src: np.ndarray, bounds: tuple[int, int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Crop ``src`` to the primary pixel rectangle, padding when ``src`` is smaller.

    Every frame in a batch then has the same width and height. The translation
    uses the primary origin, so the plate solution of each frame describes the
    same crop window.
    """
    x0, x1, y0, y1 = (int(v) for v in bounds)
    out_w = x1 - x0
    out_h = y1 - y0
    if out_w < 1 or out_h < 1:
        raise ValueError("empty crop")
    src = np.asarray(src)
    height, width = src.shape[:2]
    ix0 = max(0, min(width, x0))
    iy0 = max(0, min(height, y0))
    ix1 = max(ix0, min(width, x1))
    iy1 = max(iy0, min(height, y1))
    if src.ndim == 2:
        dest = np.zeros((out_h, out_w), dtype=np.float32)
    else:
        dest = np.zeros((out_h, out_w) + tuple(src.shape[2:]), dtype=np.float32)
    if ix1 > ix0 and iy1 > iy0:
        patch = np.asarray(src[iy0:iy1, ix0:ix1], dtype=np.float32)
        dy = iy0 - y0
        dx = ix0 - x0
        dest[dy:dy + patch.shape[0], dx:dx + patch.shape[1]] = patch
    matrix = np.array([[1.0, 0.0, -float(x0)],
                       [0.0, 1.0, -float(y0)],
                       [0.0, 0.0,  1.0]], dtype=np.float32)
    return dest, matrix

def _rotation_cursor() -> QCursor:
    """A curved-arrow 'rotate' cursor, built once and cached."""
    global _ROTATION_CURSOR
    if _ROTATION_CURSOR is not None:
        return _ROTATION_CURSOR
    size = 30
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    cx = cy = size / 2.0
    R = size * 0.30
    start_deg, span_deg = 30.0, 300.0          # ~300° arc with a gap
    a = math.radians(start_deg + span_deg)
    tip = QPointF(cx + R * math.cos(a), cy - R * math.sin(a))
    tang = math.atan2(-math.cos(a), -math.sin(a))   # tangent at the open end
    ah = size * 0.24
    # dark halo first (so it shows on light images), then white core
    for col, w in ((QColor(0, 0, 0, 230), 4.2), (QColor(255, 255, 255, 255), 2.0)):
        pen = QPen(col, w)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawArc(QRectF(cx - R, cy - R, 2 * R, 2 * R), int(start_deg * 16), int(span_deg * 16))
        for sgn in (+1, -1):
            b = tang + math.radians(180 - 35 * sgn)
            p.drawLine(tip, QPointF(tip.x() + ah * math.cos(b), tip.y() + ah * math.sin(b)))
    p.end()
    _ROTATION_CURSOR = QCursor(pm, int(round(cx)), int(round(cy)))
    return _ROTATION_CURSOR

# -------- util: s_-style preview stretch (non-destructive) ----------
def histogram_style_autostretch(image: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    def stretch_channel(c):
        med = np.median(c); mad = np.median(np.abs(c - med))
        mad_std = mad * 1.4826
        mn, mx = float(c.min()), float(c.max())
        bp = max(mn, med - sigma * mad_std)
        wp = min(mx, med + sigma * mad_std)
        if wp - bp <= 1e-8: return np.zeros_like(c)
        out = (c - bp) / (wp - bp)
        return np.clip(out, 0, 1)

    if image.ndim == 2:
        return stretch_channel(image)
    if image.ndim == 3 and image.shape[2] == 3:
        return np.stack([stretch_channel(image[..., i]) for i in range(3)], axis=-1)
    raise ValueError("Unsupported image format for autostretch.")


# -------- fast, bit-identical preview stretch -----------------------
# np.median partitions a whole channel (~0.2 s per call on a 22 MP channel, two
# calls per channel), which made opening Crop, toggling Autostretch and building
# composite layers slow. The helpers below produce exactly the same numbers as
# histogram_style_autostretch / _img01_from_doc / _to_qimage above.

_SELECT_DIRECT_MAX = 1 << 18      # below this a plain np.median is already cheap
_SELECT_SAMPLE = 1 << 16          # strided sample size used to bracket the middle
# Work units are sized so their temporaries stay well under ~1 MB: bigger blocks
# go to the OS allocator on Windows and page-fault on every use (measured ~2x
# slower for 128-row bands than for ~32-row ones on a 22 MP frame).
_BAND_BYTES = 768 * 1024          # float32 bytes per row band
_CHUNK = 1 << 18                  # elements per piece of a flat pass
# Each parallel pass is split into this many pool tasks, each looping over its
# pieces: one future per small piece meant ~1000 futures per preview, and that
# Python bookkeeping (under the GIL) showed up as GUI hitches.
_TASKS = 16


def _pool_workers(n_jobs: int) -> int:
    return max(1, min(int(n_jobs), 8, os.cpu_count() or 1))


def _spans(n: int, unit: int) -> list:
    """range(n) as at most _TASKS contiguous spans of whole `unit`-sized
    pieces: one pool task per span, looping over its pieces."""
    n, unit = int(n), max(1, int(unit))
    if n <= 0:
        return []
    pieces = -(-n // unit)                       # ceil(n / unit)
    step = -(-pieces // _TASKS) * unit           # ceil(pieces / _TASKS) pieces per span
    return [(i, min(n, i + step)) for i in range(0, n, step)]


def _pieces(span, unit: int) -> list:
    a, b = span
    return [(i, min(b, i + unit)) for i in range(a, b, unit)]


def _band_rows(width: int) -> int:
    return max(4, _BAND_BYTES // max(1, 4 * int(width)))


def _pmap(ex, fn, items) -> list:
    """map on the pool when there is one and more than one item."""
    if ex is None or len(items) <= 1:
        return [fn(it) for it in items]
    return list(ex.map(fn, items))


def _exact_median(x: np.ndarray, ex=None):
    """np.median(x) for a NaN-free array, without partitioning all of it.

    A sorted strided sample brackets the middle rank(s); one pass counts the
    values below the bracket, a second keeps the values inside it, and only that
    small window is partitioned. The bracket is only a hint: if the middle rank
    is not inside the window this falls back to np.median, so the result is
    always the exact order statistic(s), averaged by np.mean as np.median does.
    The caller guarantees there is no NaN (comparisons would drop it).
    """
    flat = np.ascontiguousarray(x).reshape(-1)
    n = int(flat.size)
    if n <= _SELECT_DIRECT_MAX:
        return np.median(flat)
    ks = [n // 2] if (n % 2) else [n // 2 - 1, n // 2]
    samp = np.sort(flat[::max(1, n // _SELECT_SAMPLE)])
    ns = int(samp.size)
    margin = int(4.0 * math.sqrt(ns)) + 16
    i_lo = (ks[0] * ns) // n - margin
    i_hi = (ks[-1] * ns) // n + margin
    if i_lo <= 0 or i_hi >= ns - 1:
        return np.median(flat)
    lo_v, hi_v = samp[i_lo], samp[i_hi]

    def _window(span):
        n_below, wins = 0, []
        for a0, b0 in _pieces(span, _CHUNK):
            f = flat[a0:b0]
            keep = f >= lo_v
            n_below += int(f.size) - int(np.count_nonzero(keep))
            np.logical_and(keep, f <= hi_v, out=keep)
            wins.append(f[keep])
        return n_below, wins

    parts = _pmap(ex, _window, _spans(n, _CHUNK))    # comparisons release the GIL
    below = sum(p[0] for p in parts)
    wins = [w for p in parts for w in p[1]]
    win = wins[0] if len(wins) == 1 else np.concatenate(wins)
    r0, r1 = ks[0] - below, ks[-1] - below
    if r0 < 0 or r1 >= win.size:
        return np.median(flat)
    part = np.partition(win, [r0, r1] if r1 != r0 else [r0])
    return np.mean(part[r0:r1 + 1], axis=0)


def _stretch_params(c: np.ndarray, sigma: float, ex=None):
    """(bp, wp) exactly as stretch_channel in histogram_style_autostretch
    derives them, or None where it returns zeros. With a thread pool the flat
    passes run in chunks on it."""
    if ex is None:
        mn, mx = float(c.min()), float(c.max())
    else:
        # Chunked min/max: the same values (NaN still propagates through
        # np.min/np.max); only the sign of a zero minimum may differ, which
        # cannot change the 8-bit output this pooled path feeds.
        flat0 = np.ascontiguousarray(c).reshape(-1)

        def _minmax(span):
            out = []
            for a0, b0 in _pieces(span, _CHUNK):
                f = flat0[a0:b0]
                out.append((f.min(), f.max()))
            return out

        mm = [v for p in _pmap(ex, _minmax, _spans(flat0.size, _CHUNK)) for v in p]
        mn = float(np.min(np.array([v[0] for v in mm])))
        mx = float(np.max(np.array([v[1] for v in mm])))
    if mn != mn or mx != mx:                  # NaN: keep np.median's semantics
        med = np.median(c)
        mad = np.median(np.abs(c - med))
    else:
        med = _exact_median(c, ex)
        flat = np.ascontiguousarray(c).reshape(-1)
        dev = np.empty(flat.shape, dtype=(flat[:1] - med).dtype)

        def _dev(span):                       # np.abs(c - med), piece by piece
            for a0, b0 in _pieces(span, _CHUNK):
                d = dev[a0:b0]
                np.subtract(flat[a0:b0], med, out=d)
                np.abs(d, out=d)

        _pmap(ex, _dev, _spans(flat.size, _CHUNK))
        mad = _exact_median(dev, ex) if math.isfinite(float(med)) else np.median(dev)
        del dev
    mad_std = mad * 1.4826
    bp = max(mn, med - sigma * mad_std)
    wp = min(mx, med + sigma * mad_std)
    if wp - bp <= 1e-8:
        return None
    return bp, wp


def _stretch_channel_fast(c: np.ndarray, sigma: float) -> np.ndarray:
    p = _stretch_params(c, sigma)
    if p is None:
        return np.zeros_like(c)
    bp, wp = p
    out = (c - bp) / (wp - bp)
    return np.clip(out, 0, 1)


def _fast_autostretch(image: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    """histogram_style_autostretch, bit-identical; colour channels in parallel."""
    if image.ndim == 2:
        return _stretch_channel_fast(image, sigma)
    if image.ndim == 3 and image.shape[2] == 3:
        with ThreadPoolExecutor(max_workers=_pool_workers(3)) as ex:
            chans = list(ex.map(lambda i: _stretch_channel_fast(image[..., i], sigma), range(3)))
        return np.stack(chans, axis=-1)
    raise ValueError("Unsupported image format for autostretch.")


def _img01_from_array(arr) -> np.ndarray:
    """CropDialogPro._img01_from_doc on an array (or any slice of one)."""
    a = np.asarray(arr)
    src_dtype = a.dtype
    if a.dtype.kind in "ui":
        a = a.astype(np.float32) / np.iinfo(src_dtype).max
    else:
        a = a.astype(np.float32, copy=False)
    if a.ndim == 3 and a.shape[2] == 1:
        a = a[..., 0]
    return np.clip(a, 0.0, 1.0)


def _preview_u8(src, autostretch: bool, sigma: float = 3.0, out_bgra=None):
    """8-bit single-image preview, byte-identical to what CropDialogPro used to
    show: _to_qimage(_img01_from_doc()), optionally autostretched first.

    Channels are measured in parallel and mapped to 8-bit in row bands on a
    thread pool, so no full-size float copy of the image or of its stretch is
    ever held. Returns None for layouts the preview does not support, so the
    caller can take the original path (and raise the original error).

    out_bgra: optional HxWx4 uint8 view to write instead (B, G, R, 255 per
    pixel: a little-endian Format_RGB32 buffer); it is returned.
    """
    a = np.asarray(src)
    if a.ndim == 3 and a.shape[2] == 1:
        a = a[..., 0]
    if a.ndim == 2:
        nch = 1
    elif a.ndim == 3 and a.shape[2] == 3:
        nch = 3
    else:
        return None
    H, W = int(a.shape[0]), int(a.shape[1])
    is_int = a.dtype.kind in "ui"
    maxval = np.iinfo(a.dtype).max if is_int else None

    # _img01_from_doc per channel: [0,1] float32, one contiguous plane each.
    chans = [np.empty((H, W), dtype=np.float32) for _ in range(nch)]
    rows = _band_rows(W)
    row_spans = _spans(H, rows)

    def chan01(span):
        for r0, r1 in _pieces(span, rows):
            for i in range(nch):
                c = a[r0:r1] if nch == 1 else a[r0:r1, :, i]
                if is_int:
                    c = c.astype(np.float32) / maxval
                else:
                    c = c.astype(np.float32, copy=False)
                np.clip(c, 0.0, 1.0, out=chans[i][r0:r1])

    if out_bgra is not None:
        u8 = out_bgra
        dst = (0, 1, 2) if nch == 1 else (2, 1, 0)   # byte offset per output channel
    else:
        u8 = np.empty((H, W) if nch == 1 else (H, W, 3), dtype=np.uint8)
    with ThreadPoolExecutor(max_workers=_pool_workers(8)) as ex:
        list(ex.map(chan01, row_spans))
        params = ([_stretch_params(c, sigma, ex) for c in chans]
                  if autostretch else None)

        def band(span):
            for r0, r1 in _pieces(span, rows):
                _band(r0, r1)

        def _band(r0, r1):
            outs = []
            for i, c in enumerate(chans):
                cb = c[r0:r1]
                if params is None:
                    outs.append(cb)
                elif params[i] is None:
                    outs.append(np.zeros_like(cb))
                else:
                    bp, wp = params[i]
                    outs.append(np.clip((cb - bp) / (wp - bp), 0, 1))
            if out_bgra is not None:
                u8[r0:r1, :, 3] = 255
            if nch == 1:
                g = (outs[0] * 255).astype(np.uint8)
                if out_bgra is None:
                    u8[r0:r1] = g
                else:
                    for k in dst:
                        u8[r0:r1, :, k] = g
            else:
                dt = np.result_type(*outs)        # what np.stack would promote to
                for i, o in enumerate(outs):
                    k = i if out_bgra is None else dst[i]
                    u8[r0:r1, :, k] = (o.astype(dt, copy=False) * 255).astype(np.uint8)

        list(ex.map(band, row_spans))
    return u8


_PREVIEW_SYNC_MAX_PX = 2_000_000   # up to this many pixels the exact preview is built inline
_PLACEHOLDER_EDGE = 1024           # long edge of the quick placeholder shown meanwhile


def _preview_layout_ok(src) -> bool:
    """Layouts _preview_u8 handles (HxW, HxWx1, HxWx3)."""
    a = np.asarray(src)
    return a.ndim == 2 or (a.ndim == 3 and a.shape[2] in (1, 3))


def _preview_qimage_rgb32(src, autostretch: bool):
    """_preview_u8 as an owned Format_RGB32 QImage: the pixel values Qt's own
    RGB888/Grayscale8 -> RGB32 conversion gives (what QPixmap.fromImage did on
    the GUI thread), so fromImage of it is a cheap share. The bytes are written
    by numpy straight into the QImage's buffer: QImage.convertToFormat holds
    the GIL for its whole run, which froze the GUI ~60 ms per 22 MP preview."""
    a = np.asarray(src)
    if not _preview_layout_ok(a):
        return None
    if sys.byteorder != "little":
        u8 = _preview_u8(a, autostretch)
        h, w = int(u8.shape[0]), int(u8.shape[1])
        fmt = QImage.Format.Format_Grayscale8 if u8.ndim == 2 else QImage.Format.Format_RGB888
        return QImage(u8.data, w, h, int(u8.strides[0]), fmt).convertToFormat(QImage.Format.Format_RGB32)
    h, w = int(a.shape[0]), int(a.shape[1])
    q = QImage(w, h, QImage.Format.Format_RGB32)
    if q.isNull():
        return None
    ptr = q.bits()
    ptr.setsize(q.sizeInBytes())
    buf = np.frombuffer(ptr, dtype=np.uint8).reshape(h, q.bytesPerLine())[:, :4 * w].reshape(h, w, 4)
    _preview_u8(a, autostretch, out_bgra=buf)
    del buf, ptr
    return q


class _PreviewBuildJob(QThread):
    """Builds one exact single-image preview off the GUI thread. No parent:
    started with thread_keeper.start_owned_thread, so the dialog may close
    mid-build. The QImage (or None on failure) comes back through `built`."""
    built = pyqtSignal(int, object)

    def __init__(self, job_id: int, src, autostretch: bool):
        super().__init__(None)
        self._job = (int(job_id), src, bool(autostretch))

    def run(self):
        job_id, src, flag = self._job
        self._job = None
        try:
            q = _preview_qimage_rgb32(src, flag)
        except Exception:
            q = None
        src = None
        self.built.emit(job_id, q)


class _LayerBuildJob(QThread):
    """Builds missing composite layers (_overlap_layer, in parallel) off the
    GUI thread. No parent: started with thread_keeper.start_owned_thread. Each
    result is (cache key, source image, layer or None on failure)."""
    built = pyqtSignal(int, object)

    def __init__(self, job_id: int, jobs: list, autostretch: bool, target_hw):
        super().__init__(None)
        self._args = (int(job_id), list(jobs), bool(autostretch), tuple(target_hw))

    def run(self):
        job_id, jobs, flag, hw = self._args
        self._args = None

        def _one(job):
            try:
                return _overlap_layer(job[1], flag, hw)
            except Exception:
                return None

        if len(jobs) == 1:
            bases = [_one(jobs[0])]
        else:
            with ThreadPoolExecutor(max_workers=_pool_workers(len(jobs))) as ex:
                bases = list(ex.map(_one, jobs))
        out = [(key, img, base) for (key, img), base in zip(jobs, bases)]
        jobs = None
        self.built.emit(job_id, out)


def _pixmap_from_u8(u8: np.ndarray) -> QPixmap:
    """QPixmap from a C-contiguous HxW / HxWx3 uint8 buffer (deep-copied by Qt)."""
    h, w = int(u8.shape[0]), int(u8.shape[1])
    fmt = QImage.Format.Format_Grayscale8 if u8.ndim == 2 else QImage.Format.Format_RGB888
    q = QImage(u8.data, w, h, int(u8.strides[0]), fmt)
    return QPixmap.fromImage(q)


# -------- blend mode compositing -----------------------------------
_BLEND_MODES = ["Screen", "Average", "Min", "Max", "Difference"]

# Colorize palette — six perceptually distinct hues cycling for N images.
# Each entry is an RGB [0,1] multiplier applied to the luminance of the layer.
_OVERLAP_COLORS: list[tuple[float, float, float]] = [
    (0.30, 0.55, 1.00),   # blue
    (1.00, 0.30, 0.30),   # red
    (0.30, 1.00, 0.40),   # green
    (1.00, 0.30, 1.00),   # magenta
    (0.20, 0.95, 0.95),   # cyan
    (1.00, 0.90, 0.10),   # yellow
]


def _colorize_layer(layer: np.ndarray, color: tuple[float, float, float]) -> np.ndarray:
    """
    Tint a HxWx3 float32 layer with a solid hue.
    We convert to luminance first so mono and colour images both tint uniformly,
    then multiply that luma by the RGB color vector.
    Result is clamped to [0,1].
    """
    lum = (0.2126 * layer[..., 0] +
           0.7152 * layer[..., 1] +
           0.0722 * layer[..., 2])[..., np.newaxis]          # H×W×1
    tinted = lum * np.array(color, dtype=np.float32)         # broadcast to H×W×3
    return np.clip(tinted, 0.0, 1.0).astype(np.float32)

def _to_rgb01(arr: np.ndarray) -> np.ndarray:
    """Normalise any doc image array to float32 HxWx3 in [0,1]."""
    a = arr.astype(np.float32, copy=False)
    if a.dtype.kind in "ui":
        a = a / np.iinfo(arr.dtype).max
    if a.ndim == 2:
        a = np.stack([a] * 3, axis=-1)
    elif a.ndim == 3 and a.shape[2] == 1:
        a = np.repeat(a, 3, axis=2)
    elif a.ndim == 3 and a.shape[2] > 3:
        a = a[..., :3]
    return np.clip(a, 0.0, 1.0)


def _downsampled01(img: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """The resample half of _downsampled_rgb01: float32, HxW (mono) or HxWx3,
    not yet clipped or expanded to three channels."""
    a = np.asarray(img)
    if a.dtype.kind in "ui":
        a = a.astype(np.float32) / float(np.iinfo(a.dtype).max)
    else:
        a = a.astype(np.float32, copy=False)
    if a.ndim == 3 and a.shape[2] == 1:
        a = a[..., 0]
    elif a.ndim == 3 and a.shape[2] > 3:
        a = a[..., :3]
    Hd, Wd = int(target_hw[0]), int(target_hw[1])
    if a.shape[:2] != (Hd, Wd):
        a = cv2.resize(np.ascontiguousarray(a), (Wd, Hd), interpolation=cv2.INTER_AREA)
    return a


def _overlap_layer(img: np.ndarray, autostretch: bool, target_hw: tuple[int, int]) -> np.ndarray:
    """One composite layer: _downsampled_rgb01(img) and, when asked,
    histogram_style_autostretch of it - bit-identical, but a mono source is
    stretched once and replicated instead of stretching three equal channels."""
    a = _downsampled01(img, target_hw)
    if a.ndim == 2:
        a = np.clip(a, 0.0, 1.0).astype(np.float32, copy=False)
        if autostretch:
            a = _stretch_channel_fast(a, 3.0)
        return np.stack([a, a, a], axis=-1)
    base = np.clip(a, 0.0, 1.0).astype(np.float32, copy=False)
    return _fast_autostretch(base) if autostretch else base


def _downsampled_rgb01(img: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """
    Like _to_rgb01 but resamples straight to *target_hw* (H, W) so the caller
    never materialises a full-resolution float RGB copy per layer. Mono images
    are resized as a single channel and expanded to 3 afterwards. INTER_AREA is
    the correct kernel for the downscale this is always used for.
    """
    a = np.asarray(img)
    if a.dtype.kind in "ui":
        a = a.astype(np.float32) / float(np.iinfo(a.dtype).max)
    else:
        a = a.astype(np.float32, copy=False)
    if a.ndim == 3 and a.shape[2] == 1:
        a = a[..., 0]
    elif a.ndim == 3 and a.shape[2] > 3:
        a = a[..., :3]
    Hd, Wd = int(target_hw[0]), int(target_hw[1])
    if a.shape[:2] != (Hd, Wd):
        a = cv2.resize(np.ascontiguousarray(a), (Wd, Hd), interpolation=cv2.INTER_AREA)
    if a.ndim == 2:
        a = np.stack([a, a, a], axis=-1)
    return np.clip(a, 0.0, 1.0).astype(np.float32, copy=False)


def _blend_images(layers: list[np.ndarray], mode: str) -> np.ndarray:
    """
    Blend a list of HxWx3 float32 [0,1] arrays using the chosen mode.
    All layers must be the same shape (caller's responsibility to resize).

    Accumulates layer by layer instead of stacking them: the same reductions
    in the same order as _blend_images_stacked (bit-identical), without the
    N-layer copy and the per-layer temporaries. Every step is elementwise, so
    row bands are blended in parallel (numpy releases the GIL).
    """
    if not layers:
        raise ValueError("No layers to blend")
    if len(layers) == 1:
        return layers[0].copy()
    first = layers[0]
    if (mode not in _BLEND_MODES or first.dtype != np.float32 or first.ndim < 2
            or not all(lay.flags.c_contiguous for lay in layers)
            or any(lay.dtype != np.float32 or lay.shape != first.shape for lay in layers)):
        return _blend_images_stacked(layers, mode)

    out = np.empty_like(first)
    H = int(first.shape[0])
    rows = _band_rows(first[0].size)

    def _span(span):
        for r0, r1 in _pieces(span, rows):
            _blend_rows([lay[r0:r1] for lay in layers], mode, out[r0:r1])

    spans = _spans(H, rows)
    if len(spans) <= 1:
        for sp in spans:
            _span(sp)
    else:
        with ThreadPoolExecutor(max_workers=_pool_workers(len(spans))) as ex:
            list(ex.map(_span, spans))
    return out


def _blend_rows(ls: list, mode: str, out: np.ndarray) -> None:
    """One band of _blend_images (all layers float32, same shape)."""
    first = ls[0]
    if mode in ("Average", "Min", "Max"):
        out[...] = first
        op = {"Average": np.add, "Min": np.minimum, "Max": np.maximum}[mode]
        for lay in ls[1:]:
            op(out, lay, out=out)
        if mode == "Average":                 # as ndarray.mean: sum, then / N
            np.true_divide(out, len(ls), out=out, casting="unsafe")
        return
    tmp = np.empty_like(first)
    if mode == "Difference":
        # pairwise max-abs difference from first layer
        acc = np.abs(first - first)
        for lay in ls[1:]:
            np.subtract(lay, first, out=tmp)
            np.abs(tmp, out=tmp)
            np.maximum(acc, tmp, out=acc)
        np.clip(acc, 0.0, 1.0, out=out)
        return
    # Screen (default): 1 - prod(1 - layer_i)
    result = np.ones_like(first)
    for layer in ls:
        np.subtract(1.0, layer, out=tmp)
        np.multiply(result, tmp, out=result)
    np.subtract(1.0, result, out=result)
    np.clip(result, 0.0, 1.0, out=out)


def _blend_images_stacked(layers: list[np.ndarray], mode: str) -> np.ndarray:
    """Reference blend over a stacked copy (mixed dtypes/shapes fall back here)."""
    stack = np.stack(layers, axis=0)   # (N, H, W, 3)

    if mode == "Average":
        return stack.mean(axis=0).astype(np.float32)
    if mode == "Min":
        return stack.min(axis=0).astype(np.float32)
    if mode == "Max":
        return stack.max(axis=0).astype(np.float32)
    if mode == "Difference":
        # pairwise max-abs difference from first layer
        base = stack[0]
        diff = np.max(np.abs(stack - base), axis=0)
        return np.clip(diff, 0.0, 1.0).astype(np.float32)
    # Screen (default): 1 - prod(1 - layer_i)
    result = np.ones_like(stack[0])
    for layer in layers:
        result = result * (1.0 - layer)
    return np.clip(1.0 - result, 0.0, 1.0).astype(np.float32)


HANDLE_SIZE = 8  # screen pixels (handles stay constant size)
EDGE_GRAB_PX = 12  # screen-pixel tolerance for grabbing edges when zoomed out


class ResizableRotatableRectItem(QGraphicsRectItem):
    def __init__(self, rect: QRectF, parent=None):
        super().__init__(rect, parent)
        pen = QPen(Qt.GlobalColor.green, 2); pen.setCosmetic(True)
        self.setPen(pen)
        self.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsSelectable |
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable |
            QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.setAcceptHoverEvents(True)
        self._fixed_ar: Optional[float] = None
        self._rotation_cb = None  
        self._handles: dict[str, QGraphicsEllipseItem] = {}
        self._active: Optional[str] = None
        self._rotating = False
        self._angle0 = 0.0
        self._pivot_scene = QPointF()
        self._bounds_scene: QRectF | None = None
        self._clamp_eps_deg = 0.25
        self._grab_pad = 20
        self._edge_pad_px = EDGE_GRAB_PX
        # center crosshair / X
        self._crosshair = QGraphicsPathItem(self)
        pen_x = QPen(QColor(0, 255, 0), 1, Qt.PenStyle.SolidLine)
        pen_x.setCosmetic(True)
        self._crosshair.setPen(pen_x)
        self._crosshair.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        self._crosshair.setZValue(self.zValue() + 2)
        self.setZValue(100)

        self._mk_handles()
        self.setTransformOriginPoint(self.rect().center())
        self._sync_crosshair()

    def _sync_crosshair(self):
        r = self.rect()
        c = r.center()
        s = 0.12 * min(r.width(), r.height())
        s = max(10.0, min(40.0, float(s)))
        p = QPainterPath()
        p.moveTo(c.x() - s, c.y()); p.lineTo(c.x() + s, c.y())
        p.moveTo(c.x(), c.y() - s); p.lineTo(c.x(), c.y() + s)
        p.moveTo(c.x() - s, c.y() - s); p.lineTo(c.x() + s, c.y() + s)
        p.moveTo(c.x() - s, c.y() + s); p.lineTo(c.x() + s, c.y() - s)
        self._crosshair.setPath(p)

    def set_crosshair_visible(self, on: bool):
        try:
            self._crosshair.setVisible(bool(on))
        except Exception:
            pass

    def setFixedAspectRatio(self, ratio: Optional[float]):
        self._fixed_ar = ratio

    def _scene_tolerance(self, px: float) -> float:
        sc = self.scene()
        if not sc:
            return float(px)
        views = sc.views()
        if not views:
            return float(px)
        v = views[0]
        p0 = v.mapToScene(QPoint(0, 0))
        p1 = v.mapToScene(QPoint(int(px), 0))
        dx = p1.x() - p0.x()
        dy = p1.y() - p0.y()
        return math.hypot(dx, dy)

    def setBoundsSceneRect(self, r: QRectF | None):
        self._bounds_scene = QRectF(r) if r is not None else None

    def _is_unrotated(self) -> bool:
        a = float(self.rotation()) % 360.0
        if a > 180.0:
            a -= 360.0
        return abs(a) < self._clamp_eps_deg

    def _bounds_local(self) -> QRectF | None:
        if self._bounds_scene is None:
            return None
        tl = self.mapFromScene(self._bounds_scene.topLeft())
        br = self.mapFromScene(self._bounds_scene.bottomRight())
        return QRectF(tl, br).normalized()

    def _edge_under_cursor(self, scene_pos: QPointF) -> Optional[str]:
        tol = self._scene_tolerance(self._edge_pad_px)
        r = self.rect()
        p = self.mapFromScene(scene_pos)
        d = {
            "l": abs(p.x() - r.left()),
            "r": abs(p.x() - r.right()),
            "t": abs(p.y() - r.top()),
            "b": abs(p.y() - r.bottom()),
        }
        m = min(d.values())
        if m > tol:
            return None
        if d["l"] == m or d["r"] == m:
            if (r.top() - tol) <= p.y() <= (r.bottom() + tol):
                return "l" if d["l"] <= d["r"] else "r"
        else:
            if (r.left() - tol) <= p.x() <= (r.right() + tol):
                return "t" if d["t"] <= d["b"] else "b"
        return None

    def _mk_handles(self):
        pen = QPen(Qt.GlobalColor.green, 2); pen.setCosmetic(True)
        brush = QBrush(Qt.GlobalColor.white)
        for name in ("tl", "tr", "br", "bl"):
            h = QGraphicsEllipseItem(0, 0, HANDLE_SIZE, HANDLE_SIZE, self)
            h.setPen(pen); h.setBrush(brush)
            h.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)
            h.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
            h.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            h.setAcceptHoverEvents(False)
            h.setZValue(self.zValue() + 1)
            self._handles[name] = h
        self._sync_handles()

    def _handle_hit(self, h: QGraphicsEllipseItem, scene_pos: QPointF) -> bool:
        p = h.mapFromScene(scene_pos)
        r = h.rect().adjusted(-self._grab_pad, -self._grab_pad, self._grab_pad, self._grab_pad)
        return r.contains(p)

    def _sync_handles(self):
        r = self.rect(); s = HANDLE_SIZE
        pos = {
            "tl": QPointF(r.left()-s/2,  r.top()-s/2),
            "tr": QPointF(r.right()-s/2, r.top()-s/2),
            "br": QPointF(r.right()-s/2, r.bottom()-s/2),
            "bl": QPointF(r.left()-s/2,  r.bottom()-s/2),
        }
        for k, it in self._handles.items():
            it.setPos(pos[k])
        self._sync_crosshair()

    def hoverMoveEvent(self, e):
        if e.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            self.setCursor(_rotation_cursor())
            return
        for k, h in self._handles.items():
            if self._handle_hit(h, e.scenePos()):
                self.setCursor({
                    "tl": Qt.CursorShape.SizeFDiagCursor,
                    "br": Qt.CursorShape.SizeFDiagCursor,
                    "tr": Qt.CursorShape.SizeBDiagCursor,
                    "bl": Qt.CursorShape.SizeBDiagCursor,
                }.get(k, Qt.CursorShape.ArrowCursor))
                return
        edge = self._edge_under_cursor(e.scenePos())
        if edge:
            self.setCursor(
                Qt.CursorShape.SizeHorCursor if edge in ("l", "r")
                else Qt.CursorShape.SizeVerCursor
            )
            return
        self.setCursor(Qt.CursorShape.SizeAllCursor)
        super().hoverMoveEvent(e)

    def mousePressEvent(self, e):
        if e.modifiers() == Qt.KeyboardModifier.ShiftModifier:
            self._rotating = True
            self.setCursor(_rotation_cursor())   
            self._pivot_scene = self.mapToScene(self.rect().center())
            v0 = e.scenePos() - self._pivot_scene
            self._angle_ref = math.degrees(math.atan2(v0.y(), v0.x()))
            self._angle0 = self.rotation()
            e.accept(); return
        for k, h in self._handles.items():
            if self._handle_hit(h, e.scenePos()):
                self._active = k
                e.accept(); return
        edge = self._edge_under_cursor(e.scenePos())
        if edge:
            self._active = edge
            e.accept(); return
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if self._rotating:
            self.setCursor(_rotation_cursor())
            v = e.scenePos() - self._pivot_scene
            ang = math.degrees(math.atan2(v.y(), v.x()))
            self.setRotation(self._angle0 + (ang - self._angle_ref))
            e.accept(); return
        if self._active:
            self._resize_via_handle(e.scenePos()); e.accept(); return
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        self._rotating = False; self._active = None
        super().mouseReleaseEvent(e)

    def itemChange(self, change, value):
        if change in (
            QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged,
            QGraphicsItem.GraphicsItemChange.ItemRotationHasChanged,
            QGraphicsItem.GraphicsItemChange.ItemTransformHasChanged,
        ):
            self._sync_handles()
        if (change == QGraphicsItem.GraphicsItemChange.ItemRotationHasChanged
                and callable(getattr(self, "_rotation_cb", None))):
            try:
                self._rotation_cb(float(self.rotation()))
            except Exception:
                pass
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange:
            if self._bounds_scene is not None and self._is_unrotated():
                new_pos = QPointF(value)
                sr0 = self.mapRectToScene(self.rect())
                d = new_pos - self.pos()
                sr = sr0.translated(d)
                b = self._bounds_scene
                dx = 0.0; dy = 0.0
                if sr.left() < b.left():   dx = b.left()  - sr.left()
                elif sr.right() > b.right(): dx = b.right() - sr.right()
                if sr.top() < b.top():     dy = b.top()   - sr.top()
                elif sr.bottom() > b.bottom(): dy = b.bottom() - sr.bottom()
                if dx != 0.0 or dy != 0.0:
                    return new_pos + QPointF(dx, dy)
                return new_pos

        return super().itemChange(change, value)

    def _resize_via_handle(self, scene_pt: QPointF):
        r = self.rect()
        p = self.mapFromScene(scene_pt)
        if self._bounds_scene is not None and self._is_unrotated():
            bL = self._bounds_local()
            if bL is not None:
                px = min(max(p.x(), bL.left()), bL.right())
                py = min(max(p.y(), bL.top()),  bL.bottom())
                p = QPointF(px, py)
        if   self._active == "tl": r.setTopLeft(p)
        elif self._active == "tr": r.setTopRight(p)
        elif self._active == "br": r.setBottomRight(p)
        elif self._active == "bl": r.setBottomLeft(p)
        elif self._active == "l":  r.setLeft(p.x())
        elif self._active == "r":  r.setRight(p.x())
        elif self._active == "t":  r.setTop(p.y())
        elif self._active == "b":  r.setBottom(p.y())
        if self._fixed_ar:
            r = r.normalized()
            cx, cy = r.center().x(), r.center().y()
            if self._active in ("l", "r"):
                w = r.width(); h = w / self._fixed_ar
                r.setTop(cy - h/2); r.setBottom(cy + h/2)
            elif self._active in ("t", "b"):
                h = r.height(); w = h * self._fixed_ar
                r.setLeft(cx - w/2); r.setRight(cx + w/2)
            else:
                w = r.width(); h = w / self._fixed_ar
                if self._active in ("tl", "tr"):
                    r.setTop(r.bottom() - h)
                else:
                    r.setBottom(r.top() + h)
        r = r.normalized()
        self.setRect(r)
        self._sync_handles()


class _PreviewPixmapItem(QGraphicsPixmapItem):
    """The canvas image. Its scene footprint is always the primary image's
    pixel rectangle (0, 0, W, H), so scene coordinates stay image pixels at any
    pixmap resolution. A full-resolution pixmap (the single-image preview)
    behaves and paints exactly like a plain QGraphicsPixmapItem; a reduced one
    (the composite preview) is drawn scaled, bilinear, into that rectangle."""

    def __init__(self, pm: QPixmap, w: int, h: int):
        self._lrect = QRectF(0.0, 0.0, float(w), float(h))
        super().__init__(pm)

    def set_logical_size(self, w: int, h: int):
        r = QRectF(0.0, 0.0, float(w), float(h))
        if r != self._lrect:
            self.prepareGeometryChange()
            self._lrect = r

    def _full_res(self) -> bool:
        pm = self.pixmap()
        return (pm.isNull()
                or (pm.width() == int(self._lrect.width())
                    and pm.height() == int(self._lrect.height())))

    def boundingRect(self) -> QRectF:
        if self._full_res():
            return super().boundingRect()
        return QRectF(self._lrect)

    def shape(self) -> QPainterPath:
        if self._full_res():
            return super().shape()
        path = QPainterPath()
        path.addRect(self._lrect)
        return path

    def paint(self, painter, option, widget=None):
        if self._full_res():
            super().paint(painter, option, widget)
            return
        pm = self.pixmap()
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.drawPixmap(self._lrect, pm, QRectF(pm.rect()))
        painter.restore()


# =============================================================================
# Multi-image overlap panel
# =============================================================================

class OverlapPanel(QWidget):
    """
    Collapsible side panel that lists all open MDI documents as checkboxes,
    provides a blend-mode dropdown, and exposes the composite result.
    """
    composite_changed = pyqtSignal()   # emitted whenever the composite should be refreshed
    layers_ready = pyqtSignal()        # a background layer build finished

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMaximumWidth(280)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)

        self._docs: list = []          # list of document objects
        self._checks: list[QCheckBox] = []
        self._composite_mode = False   # True = composite preview active
        self._colorize = False         # True = tint each layer before blending

        # Preview composites are built downsampled (long edge capped here) and
        # cached per (doc, autostretch, target) so re-blends are cheap. The crop
        # math is resolution-independent, so this is purely a preview shortcut.
        self._preview_max_dim = 1600
        self._layer_cache: dict = {}
        # Colorized copies of cached layers, keyed (id(layer), color) and
        # holding the layer itself so a recycled id can never match.
        self._tint_cache: dict = {}
        # Background layer builds: cache key -> job id while building; keys
        # whose build failed (built synchronously next time, so the error
        # surfaces as it used to).
        self._layers_building: dict = {}
        self._layer_failed: dict = {}
        self._layer_job_seq = 0
        self._closed = False
        self._recent_hw: list = []         # preview sizes whose layers are kept

        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        # ── header ──────────────────────────────────────────────────────────
        hdr = QHBoxLayout()
        self.lbl_title = QLabel(self.tr("<b>Multi-Image Overlap</b>"))
        hdr.addWidget(self.lbl_title)
        hdr.addStretch(1)
        root.addLayout(hdr)

        sep = QFrame(); sep.setFrameShape(QFrame.Shape.HLine)
        root.addWidget(sep)

        # ── blend mode ──────────────────────────────────────────────────────
        blend_row = QHBoxLayout()
        blend_row.addWidget(QLabel(self.tr("Blend:")))
        self.cmb_blend = QComboBox()
        self.cmb_blend.addItems(_BLEND_MODES)
        self.cmb_blend.setCurrentText("Screen")
        blend_row.addWidget(self.cmb_blend, 1)
        root.addLayout(blend_row)

        # ── colorize toggle ─────────────────────────────────────────────────
        self.chk_colorize = QCheckBox(self.tr("Colorize images (blue / red / green / …)"))
        self.chk_colorize.setChecked(False)
        self.chk_colorize.setToolTip(
            self.tr("Tint each image a distinct hue before blending.\n"
            "Overlap regions converge toward white/grey; frame edges\n"
            "show their assigned color — makes coverage gaps obvious.")
        )
        root.addWidget(self.chk_colorize)

        # ── image list (scrollable) ─────────────────────────────────────────
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._list_widget = QWidget()
        self._list_layout = QVBoxLayout(self._list_widget)
        self._list_layout.setContentsMargins(2, 2, 2, 2)
        self._list_layout.setSpacing(2)
        self._list_layout.addStretch(1)
        self.scroll.setWidget(self._list_widget)
        root.addWidget(self.scroll, 1)

        # ── action buttons ──────────────────────────────────────────────────
        self.btn_refresh = QPushButton(self.tr("Refresh Image List"))
        self.btn_apply   = QPushButton(self.tr("Apply Composite Preview"))
        self.btn_apply.setCheckable(True)
        self.btn_apply.setToolTip(
            self.tr("Toggle composite preview on the canvas.\n"
            "The crop rectangle and Apply/Batch buttons still work normally.")
        )
        self.btn_clear   = QPushButton(self.tr("Back to Single Image"))
        self.btn_clear.setEnabled(False)

        root.addWidget(self.btn_refresh)
        root.addWidget(self.btn_apply)
        root.addWidget(self.btn_clear)

        # ── wiring ──────────────────────────────────────────────────────────
        self.cmb_blend.currentTextChanged.connect(self._on_blend_changed)
        self.chk_colorize.stateChanged.connect(self._on_colorize_changed)
        self.btn_refresh.clicked.connect(lambda _: self.refresh_docs())
        self.btn_apply.clicked.connect(self._on_apply_toggled)
        self.btn_clear.clicked.connect(self._on_clear)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def refresh_docs(self, all_docs: list | None = None):
        """
        Populate the checkbox list.  Pass a list of document objects,
        or call with no args to re-fetch from the parent CropDialogPro.
        """
        # The explicit Refresh button drops every cached layer. Automatic
        # refreshes (panel opened, active image changed) keep the layers whose
        # document still shows the same image array; build_composite also
        # re-checks that before using one.
        if all_docs is None:
            self._layer_cache.clear()
            self._tint_cache.clear()
        else:
            self._prune_layer_cache(all_docs)
        if all_docs is not None:
            self._docs = list(all_docs)
        else:
            # Re-fetch from the parent crop dialog if available
            parent_dlg = self.parent()
            if hasattr(parent_dlg, "_collect_all_docs"):
                self._docs = parent_dlg._collect_all_docs()
        self._rebuild_list()

    def _prune_layer_cache(self, docs: list):
        live = {id(d): getattr(d, "image", None) for d in docs}
        for key in list(self._layer_cache):
            ref = self._layer_cache[key][0]
            if key[0] not in live or ref() is not live[key[0]]:
                del self._layer_cache[key]
        self._prune_tint_cache()

    def _prune_tint_cache(self):
        bases = {id(v[1]) for v in self._layer_cache.values()}
        for key in list(self._tint_cache):
            if key[0] not in bases:
                del self._tint_cache[key]

    def checked_docs(self) -> list:
        """Return documents whose checkboxes are ticked."""
        return [
            d for d, cb in zip(self._docs, self._checks) if cb.isChecked()
        ]

    def blend_mode(self) -> str:
        return self.cmb_blend.currentText()

    def is_composite_active(self) -> bool:
        return self._composite_mode

    def _layer_hw(self, target_shape) -> tuple[int, int]:
        H, W = int(target_shape[0]), int(target_shape[1])
        cap = int(self._preview_max_dim) if self._preview_max_dim else 0
        scale = (cap / float(max(H, W))) if (cap and max(H, W) > cap) else 1.0
        hw = (max(1, int(round(H * scale))), max(1, int(round(W * scale))))
        # Keep layers for the two most recent preview sizes only: an active
        # image of another size re-targets every layer.
        if hw in self._recent_hw:
            self._recent_hw.remove(hw)
        self._recent_hw.append(hw)
        del self._recent_hw[:-2]
        stale = [k for k in self._layer_cache if k[2] not in self._recent_hw]
        for key in stale:
            del self._layer_cache[key]
        if stale:
            self._prune_tint_cache()
        return hw

    def request_layers(self, target_shape: tuple[int, int], autostretch: bool) -> bool:
        """True when build_composite can run without building any layer (or
        has nothing to build). Otherwise the missing layers are started on a
        worker, layers_ready fires when they are in, and this returns False."""
        docs = self.checked_docs()
        if len(docs) < 2:
            return True
        hw = self._layer_hw(target_shape)
        todo, waiting = [], False
        for d in docs:
            img = d.image
            key = (id(d), bool(autostretch), hw)
            ent = self._layer_cache.get(key)
            if ent is not None and ent[0]() is img:
                continue
            failed = self._layer_failed.get(key)
            if failed is not None and failed() is img:
                return True                  # let build_composite raise as before
            waiting = True
            if key not in self._layers_building:
                todo.append((key, img))
        if todo:
            self._layer_job_seq += 1
            for key, _img in todo:
                self._layers_building[key] = self._layer_job_seq
            thr = _LayerBuildJob(self._layer_job_seq, todo, autostretch, hw)
            thr.built.connect(self._on_layers_built)
            start_owned_thread(thr)
        return not waiting

    def is_building(self) -> bool:
        return bool(self._layers_building)

    def _on_layers_built(self, job_id: int, results):
        for key, img, base in results:
            if self._layers_building.get(key) == job_id:
                del self._layers_building[key]
            if self._closed:
                continue
            try:
                if base is None:
                    self._layer_failed[key] = weakref.ref(img)
                else:
                    self._layer_failed.pop(key, None)
                    self._layer_cache[key] = (weakref.ref(img), base)
            except TypeError:
                pass
        if not self._closed:
            self.layers_ready.emit()

    def build_composite(
        self,
        target_shape: tuple[int, int],
        autostretch: bool,
    ) -> np.ndarray | None:
        """
        Build the blended HxWx3 float32 preview and return it DOWNSAMPLED (long
        edge capped at self._preview_max_dim). This is a coverage/overlap preview,
        not the final crop, so full resolution buys nothing — the caller draws
        it stretched over the primary image's rectangle, and the crop itself is
        always recomputed from the originals. Returns None if <2 images checked.

        Per-layer downsampled (and optionally stretched) arrays are cached keyed
        by (doc, autostretch, target), so blend-mode / colorize / checkbox changes
        re-blend without re-reading and re-scaling every source image.
        """
        docs = self.checked_docs()
        if len(docs) < 2:
            return None

        Hd, Wd = self._layer_hw(target_shape)

        colorize = self._colorize
        # Cache entries are (weakref to the source image array, layer); a layer
        # is reused only while its document still holds that same array.
        bases: list = [None] * len(docs)
        todo = []
        for idx, d in enumerate(docs):
            img = d.image
            ent = self._layer_cache.get((id(d), bool(autostretch), (Hd, Wd)))
            if ent is not None and ent[0]() is img:
                bases[idx] = ent[1]
            else:
                todo.append((idx, d, img))
        if todo:
            # cv2.resize and the numpy stretch release the GIL: build in parallel.
            def _build(job):
                return _overlap_layer(job[2], autostretch, (Hd, Wd))
            if len(todo) == 1:
                built = [_build(todo[0])]
            else:
                with ThreadPoolExecutor(max_workers=_pool_workers(len(todo))) as ex:
                    built = list(ex.map(_build, todo))
            for (idx, d, img), base in zip(todo, built):
                bases[idx] = base
                try:
                    self._layer_cache[(id(d), bool(autostretch), (Hd, Wd))] = (weakref.ref(img), base)
                except TypeError:
                    pass                          # not weak-referenceable: no caching
        layers = list(bases)
        if colorize:
            tint_todo = []
            for idx, raw in enumerate(bases):
                color = _OVERLAP_COLORS[idx % len(_OVERLAP_COLORS)]
                ent = self._tint_cache.get((id(raw), color))
                if ent is not None and ent[0] is raw:
                    layers[idx] = ent[1]
                else:
                    tint_todo.append((idx, raw, color))

            def _tint(job):                      # new array; layer cache stays clean
                return _colorize_layer(job[1], job[2])

            if len(tint_todo) == 1:
                tinted = [_tint(tint_todo[0])]
            elif tint_todo:
                with ThreadPoolExecutor(max_workers=_pool_workers(len(tint_todo))) as ex:
                    tinted = list(ex.map(_tint, tint_todo))
            else:
                tinted = []
            for (idx, raw, color), t in zip(tint_todo, tinted):
                self._tint_cache[(id(raw), color)] = (raw, t)
                layers[idx] = t
        if todo:
            self._prune_tint_cache()

        return _blend_images(layers, self.blend_mode())

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _rebuild_list(self):
        # clear existing checkboxes
        for cb in self._checks:
            cb.deleteLater()
        self._checks.clear()

        # remove all but the trailing stretch
        while self._list_layout.count() > 1:
            item = self._list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for doc in self._docs:
            # ImageDocument.display_name() is a method; fall back through
            # metadata keys that DocManager actually populates
            title = None
            try:
                fn = getattr(doc, "display_name", None)
                if callable(fn):
                    title = fn()
            except Exception:
                pass
            if not title:
                meta = getattr(doc, "metadata", {}) or {}
                title = (
                    meta.get("display_name")
                    or meta.get("file_path")
                    or getattr(doc, "title", None)
                    or getattr(doc, "filename", None)
                    or self.tr("Untitled")
                )
                # if file_path, show just the basename
                if title and os.path.sep in title:
                    title = os.path.basename(title)
            cb = QCheckBox(str(title))
            cb.setChecked(True)
            cb.stateChanged.connect(self._on_check_changed)
            self._list_layout.insertWidget(self._list_layout.count() - 1, cb)
            self._checks.append(cb)

    def _on_check_changed(self, _):
        if self._composite_mode:
            self.composite_changed.emit()

    def _on_blend_changed(self, _):
        if self._composite_mode:
            self.composite_changed.emit()

    def _on_colorize_changed(self, state: int):
        self._colorize = bool(state)
        if self._composite_mode:
            self.composite_changed.emit()

    def _on_apply_toggled(self, checked: bool):
        self._composite_mode = bool(checked)
        self.btn_clear.setEnabled(self._composite_mode)
        self.btn_apply.setText(
            self.tr("✓ Composite Active") if self._composite_mode else self.tr("Apply Composite Preview")
        )
        self.composite_changed.emit()

    def _on_clear(self):
        self._composite_mode = False
        self.btn_apply.setChecked(False)
        self.btn_apply.setText(self.tr("Apply Composite Preview"))
        self.btn_clear.setEnabled(False)
        self.composite_changed.emit()


# =============================================================================
# Main crop dialog
# =============================================================================

class CropDialogPro(QDialog):
    """SASpro crop/rotate dialog working on a Document."""
    crop_applied = pyqtSignal(np.ndarray)

    # persistent "Load Previous"
    _prev_rect: Optional[QRectF] = None
    _prev_angle: float = 0.0
    _prev_pos: QPointF = QPointF()

    def __init__(self, parent, document):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Crop Tool"))
        self.setWindowFlag(Qt.WindowType.Window, True)
        if platform.system() == "Darwin":
            self.setWindowFlag(Qt.WindowType.Tool, True)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setModal(False)
        self._main = parent
        self.doc = document
        self._live_cross = None

        self._follow_conn = False
        if hasattr(self._main, "currentDocumentChanged"):
            try:
                self._main.currentDocumentChanged.connect(self._on_active_doc_changed)
                self._follow_conn = True
            except Exception:
                self._follow_conn = False

        self.finished.connect(self._cleanup_connections)
        try:
            self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        except Exception:
            pass
        self._rect_item: Optional[ResizableRotatableRectItem] = None
        self._pix_item: Optional[QGraphicsPixmapItem] = None
        self._drawing = False
        self._origin = QPointF()
        self._autostretch_on = True
        self._src_img = None
        self._disp_cache: OrderedDict = OrderedDict()
        self._preview_jobs: dict = {}      # job id -> (weakref to source, autostretch)
        self._preview_job_seq = 0
        self._preview_launch: list = []    # (job id, source, autostretch) not started yet

        # ── outer horizontal split: controls+canvas on left, panel on right ──
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ── left side (original layout) ───────────────────────────────────
        left_widget = QWidget()
        main = QVBoxLayout(left_widget)
        outer.addWidget(left_widget, 1)

        # ── right side: overlap panel (hidden until user opens it) ────────
        self._overlap_panel = OverlapPanel(self)
        self._overlap_panel.setVisible(False)
        self._overlap_panel.composite_changed.connect(self._on_composite_changed)
        self._overlap_panel.layers_ready.connect(self._on_layers_ready)
        outer.addWidget(self._overlap_panel, 0)

        # ── info label ───────────────────────────────────────────────────
        info = QLabel(self.tr(
            "• Click–drag to draw a crop\n"
            "• Drag corner handles to resize\n"
            "• Shift + drag on box to rotate"
        )); info.setStyleSheet("color: gray; font-style: italic;")
        main.addWidget(info)

        # aspect row
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(QLabel(self.tr("Aspect Ratio:")))
        self.cmb_ar = QComboBox()
        self.cmb_ar.addItems([
            self.tr("Free"), self.tr("Original"),
            "1:1",
            "3:2", "2:3",
            "4:3", "3:4",
            "4:5", "5:4",
            "16:9", "9:16",
            "21:9", "9:21",
            "2:1", "1:2",
            "3:5", "5:3",
        ])
        row.addWidget(self.cmb_ar)
        row.addStretch(1)
        main.addLayout(row)

        # typed margins
        margins_row = QHBoxLayout()
        margins_row.addStretch(1)
        margins_row.addWidget(QLabel(self.tr("Margins (px):")))
        self.sb_top    = QSpinBox(); self.sb_top.setSuffix(" px")
        self.sb_right  = QSpinBox(); self.sb_right.setSuffix(" px")
        self.sb_bottom = QSpinBox(); self.sb_bottom.setSuffix(" px")
        self.sb_left   = QSpinBox(); self.sb_left.setSuffix(" px")
        for sb in (self.sb_top, self.sb_bottom, self.sb_left, self.sb_right):
            sb.setRange(0, 1_000_000)
        margins_row.addWidget(QLabel(self.tr("Top")));    margins_row.addWidget(self.sb_top)
        margins_row.addSpacing(8)
        margins_row.addWidget(QLabel(self.tr("Right")));  margins_row.addWidget(self.sb_right)
        margins_row.addSpacing(8)
        margins_row.addWidget(QLabel(self.tr("Bottom"))); margins_row.addWidget(self.sb_bottom)
        margins_row.addSpacing(8)
        margins_row.addWidget(QLabel(self.tr("Left")));   margins_row.addWidget(self.sb_left)
        margins_row.addSpacing(8)
        margins_row.addWidget(QLabel(self.tr("Angle")))
        self.sb_angle = QDoubleSpinBox()
        self.sb_angle.setRange(-180.0, 180.0)
        self.sb_angle.setDecimals(1)
        self.sb_angle.setSingleStep(1.0)
        self.sb_angle.setSuffix("°")
        self.sb_angle.setWrapping(True)
        self.sb_angle.setToolTip(self.tr(
            "Rotation of the crop box (degrees).\n"
            "Tip: you can also Shift + drag the box to rotate it."))
        margins_row.addWidget(self.sb_angle)
        margins_row.addStretch(1)
        main.addLayout(margins_row)

        self._suppress_margin_sync = False
        def _on_margin_changed(_):
            if self._suppress_margin_sync:
                return
            self._apply_margin_inputs()
        for sb in (self.sb_top, self.sb_right, self.sb_bottom, self.sb_left):
            sb.valueChanged.connect(_on_margin_changed)
        self.sb_angle.valueChanged.connect(self._apply_angle_input)

        # graphics view
        self.scene = QGraphicsScene(self)
        self.view  = QGraphicsView(self.scene)
        self.view.setRenderHints(self.view.renderHints())
        self.view.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.view.viewport().installEventFilter(self)
        main.addWidget(self.view, 1)

        self._zoom = 1.0
        self._fit_mode = True
        self.view.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.view.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)

        zoom_row = QHBoxLayout()
        zoom_row.addStretch(1)
        self.btn_zoom_out = themed_toolbtn("zoom-out",      self.tr("Zoom Out"))
        self.btn_zoom_in  = themed_toolbtn("zoom-in",       self.tr("Zoom In"))
        self.btn_zoom_100 = themed_toolbtn("zoom-original", self.tr("Zoom 100%"))
        self.btn_zoom_fit = themed_toolbtn("zoom-fit-best", self.tr("Fit to View"))
        for b in (self.btn_zoom_out, self.btn_zoom_in, self.btn_zoom_100, self.btn_zoom_fit):
            zoom_row.addWidget(b)
        zoom_row.addStretch(1)
        main.addLayout(zoom_row)

        dim_row = QHBoxLayout()
        dim_row.addStretch(1)
        self.lbl_dims = QLabel(self.tr("Selection: —"))
        self.lbl_dims.setStyleSheet("color: gray;")
        dim_row.addWidget(self.lbl_dims)
        dim_row.addStretch(1)
        main.addLayout(dim_row)

        self.btn_zoom_in.clicked.connect(lambda: self._zoom_by(1.25))
        self.btn_zoom_out.clicked.connect(lambda: self._zoom_by(1/1.25))
        self.btn_zoom_100.clicked.connect(self._zoom_reset_100)
        self.btn_zoom_fit.clicked.connect(self._fit_view)

        # buttons row
        btn_row = QHBoxLayout()
        self.btn_autostretch = QPushButton(self.tr("Toggle Autostretch"))
        self.btn_prev        = QPushButton(self.tr("Load Previous Crop"))
        self.btn_apply       = QPushButton(self.tr("Apply"))
        self.btn_reset       = make_reset_button(self)
        self.btn_batch       = QPushButton(self.tr("Batch Crop (all open)"))
        self.btn_overlap     = QPushButton(self.tr("⊞ Multi-Image Overlap"))
        self.btn_overlap.setCheckable(True)
        self.btn_overlap.setToolTip(
            self.tr("Open the multi-image overlap panel.\n"
            "Select which open images to composite together so you can\n"
            "find the best shared crop region before applying.")
        )
        self.btn_close = QToolButton(); self.btn_close.setText(self.tr("Close"))
        # PI-style "new instance" grip (far left): drag to the canvas to mint a
        # Crop shortcut carrying the current rectangle/angle, or drop on an image
        # to apply the crop headlessly.
        try:
            self.preset_drag_handle = PresetDragHandle(
                "crop",
                self._crop_params,
                icon=QIcon(get_icon_path("cropicon")),
                tooltip=self.tr(
                    "Drag to the canvas to create a Crop shortcut with these exact\n"
                    "settings. Drop directly on an image to apply the crop headlessly."
                ),
                parent=self,
            )
            btn_row.addWidget(self.preset_drag_handle)
            btn_row.addSpacing(10)
        except Exception:
            self.preset_drag_handle = None
        for b in (self.btn_autostretch, self.btn_prev, self.btn_apply, self.btn_reset,
                  self.btn_batch, self.btn_overlap, self.btn_close):
            btn_row.addWidget(b)
        # small in-app documentation button at the far right (across the dialog
        # from the grip, near the resize grip)
        btn_row.addStretch(1)
        self.btn_help = make_help_button("crop", self)
        btn_row.addWidget(self.btn_help)
        main.addLayout(btn_row)

        # composite status label (hidden when not in composite mode)
        self._lbl_composite_status = QLabel()
        self._lbl_composite_status.setStyleSheet(
            "color: #c8a000; font-style: italic; font-size: 11px;"
        )
        self._lbl_composite_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl_composite_status.setVisible(False)
        main.addWidget(self._lbl_composite_status)

        # wire
        self.cmb_ar.currentTextChanged.connect(self._on_ar_changed)
        self.btn_autostretch.clicked.connect(self._toggle_autostretch)
        self.btn_prev.clicked.connect(self._load_previous)
        self.btn_apply.clicked.connect(self._apply_one)
        self.btn_reset.clicked.connect(self._reset_to_factory)
        self.btn_batch.clicked.connect(self._apply_batch)
        self.btn_overlap.clicked.connect(self._toggle_overlap_panel)
        self.btn_close.clicked.connect(self.close)

        # seed image
        self._load_from_doc()
        self._update_margin_spin_ranges()
        self.resize(1100, 720)
        self._deferred_fit()
        # Factory settings: aspect ratio and the overlap blend / colorize.
        # The crop box itself (margins, angle) is the user's selection on this
        # image and stays; so do the overlap image picks and preview toggles.
        self._factory = snapshot_controls(self, exclude=(
            self.sb_top, self.sb_right, self.sb_bottom, self.sb_left, self.sb_angle,
            self.btn_overlap, self._overlap_panel.btn_apply, self._overlap_panel.scroll))

    def _reset_to_factory(self):
        restore_controls(self._factory)

    # =========================================================================
    # Overlap panel integration
    # =========================================================================

    def _toggle_overlap_panel(self, checked: bool):
        self._overlap_panel.setVisible(checked)
        if checked:
            # populate with all open docs
            self._overlap_panel.refresh_docs(self._collect_all_docs())

    def _collect_all_docs(self) -> list:
        """Gather all open document objects from the MDI."""
        docs = []
        try:
            subs = getattr(self._main, "mdi", None)
            if subs is not None:
                for sw in subs.subWindowList():
                    vw = sw.widget()
                    d = getattr(vw, "document", None)
                    if d is not None and getattr(d, "image", None) is not None:
                        docs.append(d)
        except Exception:
            pass
        return docs

    def _on_composite_changed(self):
        """Called whenever the overlap panel wants a canvas refresh."""
        if self._overlap_panel.is_composite_active():
            self._show_composite()
        else:
            self._show_single()

    def _on_layers_ready(self):
        """Background layers landed: show the composite by swapping the
        pixmap only. The crop rectangle was already re-seated when the
        composite was requested (as it always was), and touching it now could
        cut short a drag the user started meanwhile."""
        if getattr(self, "_memory_released", False) or self._pix_item is None:
            return
        panel = self._overlap_panel
        if not panel.is_composite_active():
            return
        if not panel.request_layers((self._orig_h, self._orig_w), self._autostretch_on):
            return                           # the request changed meanwhile; more coming
        composite = panel.build_composite(
            target_shape=(self._orig_h, self._orig_w),
            autostretch=self._autostretch_on,
        )
        if composite is None:
            return                           # handled when the checks changed
        self._lbl_composite_status.setText(
            self.tr("Composite preview: {0} images · {1} blend").format(
                len(panel.checked_docs()), panel.blend_mode())
        )
        self._lbl_composite_status.setVisible(True)
        self._pix_item.setPixmap(QPixmap.fromImage(self._to_qimage(composite)))

    def _show_composite(self):
        """Build composite and update the scene pixmap."""
        if not self._overlap_panel.request_layers((self._orig_h, self._orig_w),
                                                  self._autostretch_on):
            # Layers are being built on a worker; _on_layers_ready swaps the
            # composite in when they land. Re-seat the crop rectangle and
            # refit now, exactly as the synchronous path below does, so its
            # geometry does not depend on when the worker finishes.
            self._lbl_composite_status.setText(
                self.tr("Building composite preview ({0} images)...").format(
                    len(self._overlap_panel.checked_docs()))
            )
            self._lbl_composite_status.setVisible(True)
            saved = self._snapshot_rect_state()
            if self._fit_mode:
                self._apply_zoom_transform()
            self._restore_rect_state(saved)
            return
        composite = self._overlap_panel.build_composite(
            target_shape=(self._orig_h, self._orig_w),
            autostretch=self._autostretch_on,
        )
        if composite is None:
            self._lbl_composite_status.setText(
                self.tr("Select at least 2 images in the overlap panel.")
            )
            self._lbl_composite_status.setVisible(True)
            self._show_single()
            return

        n = len(self._overlap_panel.checked_docs())
        mode = self._overlap_panel.blend_mode()
        self._lbl_composite_status.setText(
            self.tr("Composite preview: {0} images · {1} blend").format(n, mode)
        )
        self._lbl_composite_status.setVisible(True)

        # build_composite returns a downsampled array. It is shown as is: the
        # preview item stretches it over the primary image's pixel rectangle,
        # so scene coordinates (and the crop rectangle) are unchanged.
        saved = self._snapshot_rect_state()
        self._set_preview_pixmap(composite)
        self._restore_rect_state(saved)

    def _show_single(self):
        """Restore single-image preview."""
        self._lbl_composite_status.setVisible(False)
        saved = self._snapshot_rect_state()
        self._set_scene_pixmap(self._single_pixmap())
        self._restore_rect_state(saved)

    def _set_preview_pixmap(self, img01: np.ndarray):
        """Replace the scene pixmap without touching the crop rectangle."""
        q  = self._to_qimage(img01)
        self._set_scene_pixmap(QPixmap.fromImage(q))

    def _set_scene_pixmap(self, pm: QPixmap):
        if self._pix_item is not None:
            self._pix_item.set_logical_size(self._orig_w, self._orig_h)
            self._pix_item.setPixmap(pm)
        else:
            self._pix_item = _PreviewPixmapItem(pm, self._orig_w, self._orig_h)
            self._pix_item.setZValue(-1)
            self.scene.addItem(self._pix_item)

        if self._fit_mode:
            self._apply_zoom_transform()

    _DISPLAY_CACHE_MAX = 3

    def _cached_single(self, src, flag: bool):
        cache = self._disp_cache
        for k in [k for k, (ref, _pm) in cache.items() if ref() is None]:
            del cache[k]
        ent = cache.get((id(src), flag))
        if ent is not None and ent[0]() is src:
            cache.move_to_end((id(src), flag))
            return ent[1]
        return None

    def _store_single(self, src, flag: bool, pm: QPixmap):
        cache = self._disp_cache
        try:
            cache[(id(src), flag)] = (weakref.ref(src), pm)
        except TypeError:
            return                           # not weak-referenceable: no caching
        cache.move_to_end((id(src), flag))
        while len(cache) > self._DISPLAY_CACHE_MAX:
            cache.popitem(last=False)

    def _single_pixmap(self) -> QPixmap:
        """Full-resolution single-image preview for the current autostretch
        state, cached per source array (toggling Autostretch, leaving the
        composite or re-activating the same image reuses it).

        A large image that is not cached yet gets a quick placeholder (the same
        stretch on a strided subsample, drawn over the image rectangle) while
        the exact preview is built on a worker; _on_preview_built swaps it in."""
        src = self._src_img
        flag = bool(self._autostretch_on)
        pm = self._cached_single(src, flag)
        if pm is not None:
            return pm
        a = np.asarray(src)
        if not _preview_layout_ok(a):
            # Layout the fast path does not handle: original path, same errors.
            full01 = _img01_from_array(src)
            img01 = histogram_style_autostretch(full01) if flag else full01
            pm = QPixmap.fromImage(self._to_qimage(img01))
        elif int(a.shape[0]) * int(a.shape[1]) <= _PREVIEW_SYNC_MAX_PX:
            pm = _pixmap_from_u8(_preview_u8(src, flag))
        else:
            self._start_preview_build(src, flag)
            return self._placeholder_pixmap(src, flag)
        self._store_single(src, flag, pm)
        return pm

    def _placeholder_pixmap(self, src, flag: bool) -> QPixmap:
        a = np.asarray(src)
        k = max(1, int(math.ceil(max(a.shape[0], a.shape[1]) / float(_PLACEHOLDER_EDGE))))
        img01 = _img01_from_array(a[::k, ::k])
        if flag:
            img01 = _fast_autostretch(img01)
        return QPixmap.fromImage(self._to_qimage(img01))

    def _start_preview_build(self, src, flag: bool):
        for ref, f in self._preview_jobs.values():
            if f == flag and ref() is src:
                return                       # already on its way
        self._preview_job_seq += 1
        job_id = self._preview_job_seq
        try:
            self._preview_jobs[job_id] = (weakref.ref(src), flag)
        except TypeError:
            self._preview_jobs[job_id] = ((lambda s=src: s), flag)
        self._preview_launch.append((job_id, src, flag))
        QTimer.singleShot(0, self._launch_preview_jobs)

    def _launch_preview_jobs(self):
        """Start queued builds from the event loop, i.e. after the dialog has
        been built and shown, so the worker does not compete with that."""
        queued, self._preview_launch = self._preview_launch, []
        for job_id, src, flag in queued:
            if job_id not in self._preview_jobs or getattr(self, "_memory_released", False):
                continue
            thr = _PreviewBuildJob(job_id, src, flag)
            thr.built.connect(self._on_preview_built)
            start_owned_thread(thr)

    def _on_preview_built(self, job_id: int, qimg):
        job = self._preview_jobs.pop(job_id, None)
        if job is None or getattr(self, "_memory_released", False):
            return
        ref, flag = job
        src = ref()
        if src is None:
            return
        if qimg is None:
            try:                             # worker failed: try once here
                qimg = _preview_qimage_rgb32(src, flag)
            except Exception:
                qimg = None
            if qimg is None:
                return                       # keep the placeholder
        pm = QPixmap.fromImage(qimg)
        self._store_single(src, flag, pm)
        if (src is self._src_img and flag == bool(self._autostretch_on)
                and self._pix_item is not None
                and not self._overlap_panel.is_composite_active()):
            # Same footprint as the placeholder: swap the pixmap only (no
            # refit, crop rectangle untouched, a drag in progress continues).
            self._pix_item.setPixmap(pm)

    # =========================================================================
    # Deferred fit / show
    # =========================================================================

    def _deferred_fit(self):
        if self._fit_mode:
            QTimer.singleShot(0, self._fit_view)

    def showEvent(self, ev):
        super().showEvent(ev)
        if not getattr(self, "_geom_restored", False):
            self._geom_restored = True
            def _after_restore():
                self._restore_window_geometry()
                self._deferred_fit()
            QTimer.singleShot(0, _after_restore)
            return
        self._deferred_fit()

    # =========================================================================
    # Image plumbing
    # =========================================================================

    def _quad_is_axis_aligned(self, pts: np.ndarray, tol: float = 1e-2) -> bool:
        if pts.shape != (4, 2):
            return False
        left_dx  = abs(pts[0,0] - pts[3,0])
        right_dx = abs(pts[1,0] - pts[2,0])
        top_dy   = abs(pts[0,1] - pts[1,1])
        bot_dy   = abs(pts[2,1] - pts[3,1])
        return (left_dx < tol and right_dx < tol and top_dy < tol and bot_dy < tol)

    def _int_bounds_from_quad(self, pts: np.ndarray, W: int, H: int) -> tuple[int,int,int,int] | None:
        if pts.size != 8:
            return None
        xs = pts[:,0]; ys = pts[:,1]
        x0 = int(np.floor(xs.min() + 1e-6))
        y0 = int(np.floor(ys.min() + 1e-6))
        x1 = int(np.ceil (xs.max() - 1e-6))
        y1 = int(np.ceil (ys.max() - 1e-6))
        x0 = max(0, min(W, x0)); x1 = max(0, min(W, x1))
        y0 = max(0, min(H, y0)); y1 = max(0, min(H, y1))
        if x1 <= x0 or y1 <= y0:
            return None
        return x0, x1, y0, y1

    def _img01_from_doc(self) -> np.ndarray:
        return _img01_from_array(self.doc.image)

    def _img01_full(self) -> np.ndarray:
        """The [0,1] float32 working copy of the source image (what Apply
        crops from), built on demand instead of being held while the dialog
        is open."""
        return _img01_from_array(self._src_img)

    def _img01_region(self, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        """_img01_full()[y0:y1, x0:x1], converting only that region."""
        return np.ascontiguousarray(_img01_from_array(self._src_img[y0:y1, x0:x1]))

    def _on_active_doc_changed(self, doc):
        if doc is None or getattr(doc, "image", None) is None:
            return
        self.doc = doc
        self._rect_item = None
        self._load_from_doc()
        # refresh panel doc list if open
        if self._overlap_panel.isVisible():
            self._overlap_panel.refresh_docs(self._collect_all_docs())

    def _load_from_doc(self):
        # Keep a reference to the document's array, not a converted copy: the
        # preview is built from it (and cached), and Apply converts only what
        # it crops. apply_edit replaces doc.image, so this stays the snapshot
        # taken here.
        self._src_img = np.asarray(self.doc.image)
        self._orig_h, self._orig_w = (int(v) for v in self._src_img.shape[:2])

        self.scene.clear()
        self._pix_item = None
        self._rect_item = None
        self._set_scene_pixmap(self._single_pixmap())
        self._deferred_fit()
        self._set_dim_label_none()

        # if composite was active, rebuild it against the new primary image size
        if self._overlap_panel.is_composite_active():
            QTimer.singleShot(50, self._show_composite)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        if self._fit_mode:
            self._apply_zoom_transform()

    # =========================================================================
    # Selection dims label
    # =========================================================================

    def _set_dim_label_none(self):
        if hasattr(self, "lbl_dims"):
            self.lbl_dims.setText(self.tr("Selection: —"))

    def _update_dim_label_from_corners(self, corners_scene):
        if not hasattr(self, "lbl_dims") or not corners_scene or not self._pix_item:
            self._set_dim_label_none()
            return
        w_img, h_img = self._orig_w, self._orig_h
        src = np.array(
            [self._scene_to_img_pixels(p, w_img, h_img) for p in corners_scene],
            dtype=np.float32,
        )
        width  = float(np.linalg.norm(src[1] - src[0]))
        height = float(np.linalg.norm(src[3] - src[0]))
        self.lbl_dims.setText(
            self.tr("Selection: {0}×{1} px").format(int(round(height)), int(round(width)))
        )

    def _update_dim_label_from_rect_item(self):
        if not self._rect_item:
            self._set_dim_label_none()
            return
        corners = self._corners_scene()
        self._update_dim_label_from_corners(corners)

    @staticmethod
    def _to_qimage(img01: np.ndarray) -> QImage:
        if img01.ndim == 3 and img01.shape[2] == 1:
            img01 = img01[..., 0]
        if img01.ndim == 2:
            buf = np.ascontiguousarray((img01 * 255).astype(np.uint8))
            h, w = buf.shape
            return QImage(buf.tobytes(), w, h, buf.strides[0], QImage.Format.Format_Grayscale8)
        if img01.ndim == 3 and img01.shape[2] == 3:
            buf = np.ascontiguousarray((img01 * 255).astype(np.uint8))
            h, w, _ = buf.shape
            return QImage(buf.tobytes(), w, h, buf.strides[0], QImage.Format.Format_RGB888)
        raise ValueError(f"Unsupported image shape for preview: {img01.shape}")

    # =========================================================================
    # Aspect ratio
    # =========================================================================

    def _on_ar_changed(self, txt: str):
        if not self._rect_item: return
        if txt in (self.tr("Free"), "Free"):
            ar = None
        elif txt in (self.tr("Original"), "Original"):
            ar = self._orig_w / self._orig_h
        else:
            a, b = map(float, txt.split(":")); ar = a / b
        self._rect_item.setFixedAspectRatio(ar)
        if ar is not None:
            r = self._rect_item.rect()
            w = r.width(); h = w / ar
            c = r.center()
            nr = QRectF(c.x()-w/2, c.y()-h/2, w, h)
            self._rect_item.setRect(nr)
            self._rect_item.setTransformOriginPoint(nr.center())

    # =========================================================================
    # Drawing / interaction
    # =========================================================================

    def eventFilter(self, src, e):
        if src is self.view.viewport():
            if e.type() == QEvent.Type.Wheel:
                dy = e.pixelDelta().y()
                if dy != 0:
                    abs_dy = abs(dy)
                    ctrl_down = bool(e.modifiers() & Qt.KeyboardModifier.ControlModifier)
                    if abs_dy <= 3:   base_factor = 1.012 if ctrl_down else 1.010
                    elif abs_dy <= 10: base_factor = 1.025 if ctrl_down else 1.020
                    else:              base_factor = 1.040 if ctrl_down else 1.030
                    factor = base_factor if dy > 0 else 1.0 / base_factor
                else:
                    dy = e.angleDelta().y()
                    if dy == 0: return True
                    ctrl_down = bool(e.modifiers() & Qt.KeyboardModifier.ControlModifier)
                    step = 1.25 if ctrl_down else 1.15
                    factor = step if dy > 0 else 1.0 / step
                self._zoom_by(factor)
                e.accept()
                return True

            if e.type() in (QEvent.Type.MouseButtonPress, QEvent.Type.MouseMove, QEvent.Type.MouseButtonRelease):
                scene_pt = self.view.mapToScene(e.pos())

            if e.type() == QEvent.Type.MouseMove and self._rect_item is not None:
                self._update_dim_label_from_rect_item()

            if self._rect_item is None:
                if e.type() == QEvent.Type.MouseButtonPress and e.button() == Qt.MouseButton.LeftButton:
                    self._drawing = True; self._origin = scene_pt; return True

                if e.type() == QEvent.Type.MouseMove and self._drawing:
                    r = QRectF(self._origin, scene_pt).normalized()
                    r = self._apply_ar_to_rect(r, live=True, scene_pt=scene_pt)
                    r = self._clamp_rect_to_pixmap(r)
                    self._draw_live_rect(r)
                    corners = [r.topLeft(), r.topRight(), r.bottomRight(), r.bottomLeft()]
                    self._update_dim_label_from_corners(corners)
                    return True

                if e.type() == QEvent.Type.MouseButtonRelease and e.button() == Qt.MouseButton.LeftButton and self._drawing:
                    self._drawing = False
                    r = QRectF(self._origin, scene_pt).normalized()
                    r = self._apply_ar_to_rect(r, live=False, scene_pt=scene_pt)
                    r = self._clamp_rect_to_pixmap(r)
                    self._clear_live_rect()
                    self._rect_item = ResizableRotatableRectItem(r)
                    self._rect_item.setZValue(10)
                    self._rect_item.setBoundsSceneRect(self._bounds_scene_rect())
                    self._rect_item.setFixedAspectRatio(self._current_ar_value())
                    self.scene.addItem(self._rect_item)
                    self._attach_rect(self._rect_item)  
                    CropDialogPro._prev_rect  = QRectF(r)
                    CropDialogPro._prev_angle = self._rect_item.rotation()
                    CropDialogPro._prev_pos   = self._rect_item.pos()
                    self._update_dim_label_from_rect_item()
                    self._attach_rect(self._rect_item)  
                    return True

            return False
        return super().eventFilter(src, e)

    def _apply_zoom_transform(self):
        if not self._pix_item:
            return
        if self._fit_mode:
            rect = self._pix_item.mapRectToScene(self._pix_item.boundingRect())
            self.view.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
            self.view.fitInView(rect.adjusted(-1,-1,1,1), Qt.AspectRatioMode.KeepAspectRatio)
            self.view.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        else:
            self.view.resetTransform()
            self.view.scale(self._zoom, self._zoom)

    def _fit_view(self):
        self._fit_mode = True
        self._apply_zoom_transform()

    def _zoom_reset_100(self):
        self._fit_mode = False
        self._zoom = 1.0
        self._apply_zoom_transform()

    def _zoom_by(self, factor: float):
        self._fit_mode = False
        newz = min(16.0, max(0.05, self._zoom * float(factor)))
        if abs(newz - self._zoom) < 1e-4: return
        self._zoom = newz
        self._apply_zoom_transform()

    # =========================================================================
    # Typed margins
    # =========================================================================

    def _update_margin_spin_ranges(self):
        h, w = int(self._orig_h), int(self._orig_w)
        self.sb_top.setRange(0, max(0, h))
        self.sb_bottom.setRange(0, max(0, h))
        self.sb_left.setRange(0, max(0, w))
        self.sb_right.setRange(0, max(0, w))

    def _apply_margin_inputs(self):
        t = int(self.sb_top.value());    r = int(self.sb_right.value())
        b = int(self.sb_bottom.value()); l = int(self.sb_left.value())
        self._set_rect_from_margins(t, r, b, l)

    def _set_rect_from_margins(self, top: int, right: int, bottom: int, left: int):
        w_img, h_img = float(self._orig_w), float(self._orig_h)
        left   = max(0, min(int(left),   int(w_img)))
        right  = max(0, min(int(right),  int(w_img)))
        top    = max(0, min(int(top),    int(h_img)))
        bottom = max(0, min(int(bottom), int(h_img)))
        x = float(left); y = float(top)
        w = max(1.0, w_img - (left + right))
        h = max(1.0, h_img - (top + bottom))
        r = QRectF(x, y, w, h)
        if self._rect_item is None:
            self._rect_item = ResizableRotatableRectItem(r)
            self._rect_item.setZValue(10)
            self._rect_item.setBoundsSceneRect(self._bounds_scene_rect())
            self.scene.addItem(self._rect_item)
        else:
            self._rect_item.setRotation(0.0)
            self._rect_item.setPos(QPointF(0, 0))
            self._rect_item.setRect(r)
        self._rect_item.setTransformOriginPoint(r.center())
        self._update_dim_label_from_rect_item()

    def _attach_rect(self, item):
        """Wire live rotation feedback and sync the Angle spin to this rect."""
        try:
            item._rotation_cb = self._sync_angle_spin_from_item
        except Exception:
            pass
        self._sync_angle_spin_from_item()

    def _sync_angle_spin_from_item(self, angle=None):
        """Reflect the crop box's current rotation in the Angle spin box."""
        if not hasattr(self, "sb_angle") or self._rect_item is None:
            return
        ang = self._rect_item.rotation() if angle is None else float(angle)
        ang = ((ang + 180.0) % 360.0) - 180.0      # normalize to [-180, 180]
        from PyQt6.QtCore import QSignalBlocker
        _blk = QSignalBlocker(self.sb_angle)
        self.sb_angle.setValue(ang)

    def _apply_angle_input(self, value: float):
        """Rotate the crop box to the typed angle, about its center."""
        if self._rect_item is None:
            return
        r = self._rect_item.rect()
        self._rect_item.setTransformOriginPoint(r.center())
        self._rect_item.setRotation(float(value))
        self._update_dim_label_from_rect_item()

    def _current_ar_value(self) -> Optional[float]:
        txt = self.cmb_ar.currentText()
        if txt == self.tr("Free"): return None
        if txt == self.tr("Original"): return self._orig_w / self._orig_h
        a, b = map(float, txt.split(":")); return a / b

    def _apply_ar_to_rect(self, r: QRectF, live: bool, scene_pt: QPointF) -> QRectF:
        ar = self._current_ar_value()
        if ar is None:
            return r
        w = r.width(); h = w / ar
        if scene_pt.y() < self._origin.y():
            r.setTop(r.bottom() - h)
        else:
            r.setBottom(r.top() + h)
        return r.normalized()

    def _draw_live_rect(self, r: QRectF):
        if hasattr(self, "_live_rect") and self._live_rect:
            self.scene.removeItem(self._live_rect)
        if getattr(self, "_live_cross", None):
            self.scene.removeItem(self._live_cross)
            self._live_cross = None
        pen = QPen(QColor(0,255,0), 2, Qt.PenStyle.DashLine); pen.setCosmetic(True)
        self._live_rect = self.scene.addRect(r, pen)
        s = 0.12 * min(r.width(), r.height())
        s = max(10.0, min(40.0, float(s)))
        c = r.center()
        p = QPainterPath()
        p.moveTo(c.x()-s, c.y()); p.lineTo(c.x()+s, c.y())
        p.moveTo(c.x(), c.y()-s); p.lineTo(c.x(), c.y()+s)
        p.moveTo(c.x()-s, c.y()-s); p.lineTo(c.x()+s, c.y()+s)
        p.moveTo(c.x()-s, c.y()+s); p.lineTo(c.x()+s, c.y()-s)
        self._live_cross = self.scene.addPath(p, QPen(QColor(0,255,0), 1))

    def _clear_live_rect(self):
        if hasattr(self, "_live_rect") and self._live_rect:
            self.scene.removeItem(self._live_rect); self._live_rect = None
        if getattr(self, "_live_cross", None):
            self.scene.removeItem(self._live_cross); self._live_cross = None

    def _pixmap_scene_rect(self) -> QRectF | None:
        if not self._pix_item:
            return None
        return self._pix_item.mapRectToScene(self._pix_item.boundingRect())

    def _clamp_rect_to_pixmap(self, r: QRectF) -> QRectF:
        bounds = self._pixmap_scene_rect()
        if bounds is None:
            return r.normalized()
        rr = r.normalized().intersected(bounds)
        if rr.isNull() or rr.width() <= 1e-6 or rr.height() <= 1e-6:
            x = min(max(r.center().x(), bounds.left()), bounds.right())
            y = min(max(r.center().y(), bounds.top()),  bounds.bottom())
            rr = QRectF(x, y, 1.0, 1.0)
        return rr.normalized()

    def _bounds_scene_rect(self) -> QRectF | None:
        if not self._pix_item:
            return None
        return self._pix_item.mapRectToScene(self._pix_item.boundingRect())

    # =========================================================================
    # Preview toggles
    # =========================================================================

    def _toggle_autostretch(self):
        self._autostretch_on = not self._autostretch_on
        saved = self._snapshot_rect_state()
        if self._overlap_panel.is_composite_active():
            self._show_composite()
        else:
            self._set_scene_pixmap(self._single_pixmap())
        self._restore_rect_state(saved)
        self._deferred_fit()

    def _snapshot_rect_state(self):
        if not self._rect_item: return None
        return (QRectF(self._rect_item.rect()),
                float(self._rect_item.rotation()),
                QPointF(self._rect_item.pos()))

    def _restore_rect_state(self, state):
        if not state: return
        r, ang, pos = state
        # Remove old rect from scene only if it's still tracked
        if self._rect_item is not None:
            try:
                self.scene.removeItem(self._rect_item)
            except Exception:
                pass
        self._rect_item = ResizableRotatableRectItem(r)
        self._rect_item.setZValue(10)
        self._rect_item.setBoundsSceneRect(self._bounds_scene_rect())
        self._rect_item.setFixedAspectRatio(self._current_ar_value())
        self._rect_item.setRotation(ang)
        self._rect_item.setPos(pos)
        self._rect_item.setTransformOriginPoint(r.center())
        self.scene.addItem(self._rect_item)
        self._attach_rect(self._rect_item)  
        self._update_dim_label_from_rect_item()

    def _load_previous(self):
        if CropDialogPro._prev_rect is None:
            QMessageBox.information(self, self.tr("No Previous"), self.tr("No previous crop stored."))
            return
        if self._rect_item:
            self.scene.removeItem(self._rect_item)
        r = QRectF(CropDialogPro._prev_rect)
        self._rect_item = ResizableRotatableRectItem(r)
        self._rect_item.setZValue(10)
        self._rect_item.setBoundsSceneRect(self._bounds_scene_rect())
        self._rect_item.setFixedAspectRatio(self._current_ar_value())
        self._rect_item.setRotation(CropDialogPro._prev_angle)
        self._rect_item.setPos(CropDialogPro._prev_pos)
        self._rect_item.setTransformOriginPoint(r.center())
        self.scene.addItem(self._rect_item)
        self._attach_rect(self._rect_item)  
        self._update_dim_label_from_rect_item()

    # =========================================================================
    # Apply
    # =========================================================================
    def _crop_preset(self, *, x0, y0, x1, y1, angle=0.0, out_w=None, out_h=None):
        return {
            "x0": int(x0), "y0": int(y0), "x1": int(x1), "y1": int(y1),
            "angle": float(angle),
            "out_w": int(out_w) if out_w is not None else int(x1 - x0),
            "out_h": int(out_h) if out_h is not None else int(y1 - y0),
        }
    
    def _corners_scene(self):
        rl = self._rect_item.rect()
        loc = [rl.topLeft(), rl.topRight(), rl.bottomRight(), rl.bottomLeft()]
        return [self._rect_item.mapToScene(p) for p in loc]

    def _scene_to_img_pixels(self, pt_scene: QPointF, w_img: int, h_img: int):
        # Scene units are the primary image's pixels whatever the preview
        # pixmap's resolution (see _PreviewPixmapItem).
        sx, sy = w_img / self._orig_w, h_img / self._orig_h
        return np.array([pt_scene.x() * sx, pt_scene.y() * sy], dtype=np.float32)

    def _crop_params(self) -> dict:
        """Canonical preset for the drag handle — the SAME quad_norm that
        _apply_one would crop with (normalized image-pixel corners). Captures
        axis-aligned and rotated crops exactly; size-independent on replay.
        Falls back to a no-op margins preset if nothing is drawn yet."""
        try:
            if self._rect_item is not None and self._pix_item is not None:
                W, H = self._orig_w, self._orig_h
                corners = self._corners_scene()
                src = np.array(
                    [self._scene_to_img_pixels(p, W, H) for p in corners],
                    dtype=np.float32,
                )
                quad = (src / np.array([float(W), float(H)], dtype=np.float32)).tolist()
                return {"mode": "quad_norm", "quad": quad, "create_new_view": False}
        except Exception:
            pass
        return {"mode": "margins",
                "margins": {"top": 0, "right": 0, "bottom": 0, "left": 0},
                "create_new_view": False}

    def seed_from_preset(self, preset: dict | None) -> None:
        """Load a preset (margins / rect_norm / quad_norm) into the live
        rectangle so a dropped/double-clicked Crop shortcut opens the dialog
        showing exactly the crop it would apply. The preset is converted to
        image-pixel corners through the SAME crop_preset builders the headless
        runner uses, so the on-screen rectangle matches the headless result."""
        if not preset or self._pix_item is None:
            return
        try:
            import math
            from setiastro.saspro import crop_preset as _cp
            W, H = self._orig_w, self._orig_h
            mode = str(preset.get("mode", "quad_norm")).lower()
            if mode == "margins":
                q = _cp._quad_from_margins(W, H, preset.get("margins", {}) or {})
            elif mode == "rect_norm":
                q = _cp._quad_from_rect_norm(W, H, preset.get("rect", {}) or {})
            else:
                qn = np.array(preset.get("quad", []), dtype=np.float32)
                if qn.shape != (4, 2):
                    return
                q = qn * np.array([float(W), float(H)], dtype=np.float32)

            # image pixels -> scene coords (inverse of _scene_to_img_pixels)
            sx = self._orig_w / float(W)
            sy = self._orig_h / float(H)
            pts = [(float(px) * sx, float(py) * sy) for px, py in np.asarray(q, dtype=np.float32)]
            (tlx, tly), (trx, tryy), (brx, bry), (blx, bly) = pts

            w = math.hypot(trx - tlx, tryy - tly)
            h = math.hypot(blx - tlx, bly - tly)
            cx = (tlx + trx + brx + blx) / 4.0
            cy = (tly + tryy + bry + bly) / 4.0
            ang = math.degrees(math.atan2(tryy - tly, trx - tlx))

            # A rotated rectangle == axis-aligned rect centered at C, dims w×h,
            # rotated about its center by `ang` — which is exactly what
            # _restore_rect_state reproduces.
            r = QRectF(cx - w / 2.0, cy - h / 2.0, w, h)
            self._restore_rect_state((r, ang, QPointF(0.0, 0.0)))
            try:
                _blk = QSignalBlocker(self.sb_angle)
                self.sb_angle.setValue(ang)
            except Exception:
                pass
        except Exception:
            pass

    def _apply_one(self):
        try:
            self._save_window_geometry()
        except Exception:
            pass
        if not self._rect_item:
            QMessageBox.warning(self, self.tr("No Selection"), self.tr("Draw & finalize a crop first."))
            return

        corners = self._corners_scene()
        w_img, h_img = self._orig_w, self._orig_h
        src = np.array([self._scene_to_img_pixels(p, w_img, h_img) for p in corners], dtype=np.float32)

        width  = np.linalg.norm(src[1] - src[0])
        height = np.linalg.norm(src[3] - src[0])

        H_img, W_img = self._orig_h, self._orig_w
        axis_aligned = self._quad_is_axis_aligned(src)

        if axis_aligned:
            bounds = self._int_bounds_from_quad(src, W_img, H_img)
            if bounds is None:
                QMessageBox.critical(self, self.tr("Apply failed"), self.tr("Invalid crop bounds."))
                return
            x0, x1, y0, y1 = bounds
            out = self._img01_region(y0, y1, x0, x1)
            M = np.array([[1.0, 0.0, -float(x0)],
                          [0.0, 1.0, -float(y0)],
                          [0.0, 0.0,  1.0]], dtype=np.float32)
            w_out, h_out = (x1 - x0), (y1 - y0)
        else:
            dst = np.array([[0,0],[width,0],[width,height],[0,height]], dtype=np.float32)
            M = cv2.getPerspectiveTransform(src, dst)
            w_out = int(round(width)); h_out = int(round(height))
            if w_out <= 0 or h_out <= 0:
                QMessageBox.critical(self, self.tr("Apply failed"), self.tr("Invalid crop size."))
                return
            out = cv2.warpPerspective(self._img01_full(), M, (w_out, h_out), flags=cv2.INTER_LANCZOS4)

        new_meta = dict(self.doc.metadata or {})
        try:
            if update_wcs_after_crop is not None:
                new_meta = update_wcs_after_crop(new_meta, M_src_to_dst=M, out_w=w_out, out_h=h_out)
        except Exception:
            pass
        try:    # a Lunar Solver placement moves with the pixels
            from setiastro.saspro.lunar_solver import lunar_solution_after_geometry
            new_meta = lunar_solution_after_geometry(new_meta, M, (h_out, w_out),
                                                     in_shape=np.asarray(self.doc.image).shape[:2])
        except Exception:
            pass

        CropDialogPro._prev_rect  = QRectF(self._rect_item.rect())
        CropDialogPro._prev_angle = float(self._rect_item.rotation())
        CropDialogPro._prev_pos   = QPointF(self._rect_item.pos())

        try:
            # Replay preset — quad_norm captures BOTH axis-aligned and rotated
            # crops exactly, using the same corners we just cropped with.
            _preset = {
                "mode": "quad_norm",
                "quad": (src / np.array([float(W_img), float(H_img)],
                                        dtype=np.float32)).tolist(),
                "create_new_view": False,
            }
            self.doc.apply_edit(
                out.copy(),
                metadata={**new_meta, "step_name": "Crop",
                          "command_id": "crop", "preset": _preset},
                step_name="Crop",
            )
            self._maybe_notify_wcs_update(new_meta)
            self.crop_applied.emit(out)
            self.close()
        except Exception as e:
            QMessageBox.critical(self, self.tr("Apply failed"), str(e))

    def _apply_batch(self):
        try:
            self._save_window_geometry()
        except Exception:
            pass
        if not self._rect_item:
            QMessageBox.warning(self, self.tr("No Selection"), self.tr("Draw & finalize a crop first."))
            return

        # ── decide which docs to crop ─────────────────────────────────────
        # When composite is active, use exactly the checked images.
        # Otherwise fall back to the original "all open" behaviour.
        if self._overlap_panel.is_composite_active():
            docs = self._overlap_panel.checked_docs()
            if not docs:
                QMessageBox.information(
                    self, self.tr("No Images Selected"),
                    self.tr("No images are checked in the overlap panel.")
                )
                return
            source_label = self.tr(
                "Apply this crop to {0} checked image(s) from the overlap panel?"
            ).format(len(docs))
        else:
            # original behaviour: all open MDI docs
            win = self.parent()
            subs = getattr(win, "mdi", None).subWindowList() if hasattr(win, "mdi") else []
            docs = []
            for sw in subs:
                vw = sw.widget()
                d = getattr(vw, "document", None)
                if d is not None:
                    docs.append(d)
            if not docs:
                QMessageBox.information(self, self.tr("No Images"), self.tr("No open images to crop."))
                return
            source_label = self.tr("Apply this crop to {0} open image(s)?").format(len(docs))

        # Normalise crop polygon to THIS (primary) image size
        corners  = self._corners_scene()
        src_this = np.array(
            [self._scene_to_img_pixels(p, self._orig_w, self._orig_h) for p in corners],
            dtype=np.float32,
        )
        norm = src_this / np.array([self._orig_w, self._orig_h], dtype=np.float32)
        primary_h = int(self._orig_h)
        primary_w = int(self._orig_w)
        primary_pts = norm * np.array([primary_w, primary_h], dtype=np.float32)
        primary_axis = self._quad_is_axis_aligned(primary_pts)
        primary_bounds = (
            self._int_bounds_from_quad(primary_pts, primary_w, primary_h)
            if primary_axis else None
        )

        ok = QMessageBox.question(
            self, self.tr("Confirm Batch"), source_label,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if ok != QMessageBox.StandardButton.Yes:
            return

        last_cropped = None
        for d in docs:
            img = np.asarray(d.image)
            if img.dtype.kind in "ui":
                src01 = img.astype(np.float32) / np.iinfo(d.image.dtype).max
            else:
                src01 = img.astype(np.float32, copy=False)

            h, w    = src01.shape[:2]
            # Registered frames are often 1 px apart after a previous crop.
            # Reuse the primary pixel rectangle so every result has one size.
            close = abs(int(h) - primary_h) <= 4 and abs(int(w) - primary_w) <= 4
            src_pts = norm * np.array([w, h], dtype=np.float32)

            axis_aligned = self._quad_is_axis_aligned(src_pts)
            if primary_axis and primary_bounds is not None and close:
                # Same pixel rectangle as the primary, padded with zeros when
                # this frame is a pixel or two smaller. Clamping used to shrink
                # only that frame, so the batch no longer shared one size.
                cropped, M = shared_axis_crop(src01, primary_bounds)
                h_out, w_out = int(cropped.shape[0]), int(cropped.shape[1])
            elif axis_aligned:
                b = self._int_bounds_from_quad(src_pts, w, h)
                if b is None: continue
                x0, x1, y0, y1 = b
                cropped = src01[y0:y1, x0:x1].copy()
                w_out, h_out = (x1 - x0), (y1 - y0)
                M = np.array([[1.0, 0.0, -float(x0)],
                              [0.0, 1.0, -float(y0)],
                              [0.0, 0.0,  1.0]], dtype=np.float32)
            else:
                w_out = int(round(np.linalg.norm(src_pts[1] - src_pts[0])))
                h_out = int(round(np.linalg.norm(src_pts[3] - src_pts[0])))
                if w_out <= 0 or h_out <= 0: continue
                dst = np.array([[0,0],[w_out,0],[w_out,h_out],[0,h_out]], dtype=np.float32)
                M = cv2.getPerspectiveTransform(src_pts.astype(np.float32), dst)
                cropped = cv2.warpPerspective(src01, M, (w_out, h_out), flags=cv2.INTER_LANCZOS4)

            meta_this = dict(d.metadata or {})
            try:
                if update_wcs_after_crop is not None:
                    meta_this = update_wcs_after_crop(meta_this, M_src_to_dst=M, out_w=w_out, out_h=h_out)
            except Exception:
                pass
            try:    # a Lunar Solver placement moves with the pixels
                from setiastro.saspro.lunar_solver import lunar_solution_after_geometry
                meta_this = lunar_solution_after_geometry(meta_this, M, (h_out, w_out),
                                                          in_shape=np.asarray(d.image).shape[:2])
            except Exception:
                pass
            try:
                _preset = {
                    "mode": "quad_norm",
                    "quad": np.asarray(norm, dtype=np.float32).tolist(),
                    "create_new_view": False,
                }
                d.apply_edit(
                    cropped.copy(),
                    metadata={**meta_this, "step_name": "Crop",
                              "command_id": "crop", "preset": _preset},
                    step_name="Crop",
                )
                last_cropped = cropped
            except Exception:
                pass

        QMessageBox.information(
            self, self.tr("Batch Crop"),
            self.tr("Applied crop to {0} image(s). Any Astrometric Solutions have been updated.").format(len(docs))
        )
        if last_cropped is not None:
            self.crop_applied.emit(last_cropped)
        self.close()

    # =========================================================================
    # WCS notification
    # =========================================================================

    def _maybe_notify_wcs_update(self, meta: dict, batch_note: str | None = None):
        dbg = (meta or {}).get("__wcs_debug__")
        if not dbg:
            return
        try:
            before = dbg.get("before", {}); after = dbg.get("after", {}); fit = dbg.get("fit", {})
            b_ra, b_dec = before.get("crval_deg", (float("nan"), float("nan")))
            a_ra, a_dec = after.get("crval_deg",  (float("nan"), float("nan")))
            rms  = fit.get("rms_arcsec", float("nan"))
            p95  = fit.get("p95_arcsec", float("nan"))
            sip  = after.get("sip_degree")
            size = after.get("size")
            sip_txt  = f"TAN-SIP (deg={sip})" if sip is not None else "TAN"
            size_txt = f"{size[0]}×{size[1]}" if size else "?"
            extra = f"\n{batch_note}" if batch_note else ""
            msg = (
                self.tr("Astrometric solution updated ✔️\n\n") +
                self.tr("Model: {0}   Image: {1}\n").format(sip_txt, size_txt) +
                self.tr("CRVAL: ({0:.6f}, {1:.6f}) → ({2:.6f}, {3:.6f})\n").format(b_ra, b_dec, a_ra, a_dec) +
                self.tr("Fit residuals: RMS {0:.3f}\"  (p95 {1:.3f}\")").format(rms, p95) +
                f"{extra}"
            )
            QMessageBox.information(self, self.tr("WCS Updated"), msg)
        except Exception:
            pass

    # =========================================================================
    # Cleanup / geometry
    # =========================================================================

    def _cleanup_connections(self):
        try:
            if self._follow_conn and hasattr(self._main, "currentDocumentChanged"):
                self._main.currentDocumentChanged.disconnect(self._on_active_doc_changed)
        except Exception:
            pass
        self._follow_conn = False

    def _restore_window_geometry(self):
        try:
            s = QSettings()
            g = s.value("crop/window_geometry", None)
            if g is not None:
                self.restoreGeometry(g)
        except Exception:
            pass

    def _save_window_geometry(self):
        try:
            s = QSettings()
            s.setValue("crop/window_geometry", self.saveGeometry())
        except Exception:
            pass

    def closeEvent(self, ev):
        try:
            self._save_window_geometry()
        except Exception:
            pass
        self._cleanup_connections()
        try:
            self._release_memory()
        except Exception:
            pass
        super().closeEvent(ev)

    def _release_memory(self):
        """Drop full/preview buffers and scene pixmap on close."""
        if getattr(self, "_memory_released", False):
            return
        self._memory_released = True

        self._src_img = None
        self._preview_jobs.clear()
        self._preview_launch = []
        try:
            self._disp_cache.clear()
            self._overlap_panel._layer_cache.clear()
            self._overlap_panel._tint_cache.clear()
            self._overlap_panel._closed = True
        except Exception:
            pass
        self.doc = None
        try:
            if self._pix_item is not None:
                self._pix_item.setPixmap(QPixmap())
        except Exception:
            pass
        try:
            self.scene.clear()
        except Exception:
            pass
        self._pix_item = None

        try:
            from setiastro.saspro.free_torch_memory import trim_process_memory
            trim_process_memory()
        except Exception:
            pass

def open_crop_with_preset(main_window, preset: dict | None = None):
    """Open the live Crop dialog seeded from a preset. Used by the preset-open
    path (_preset_opener_for_command -> ShortcutManager) when a Crop shortcut is
    double-clicked. Resolves the active document from the active MDI subwindow
    first, matching the tool's toolbar opener."""
    doc = None
    try:
        sw = main_window.mdi.activeSubWindow()
        if sw is not None:
            doc = getattr(sw.widget(), "document", None)
    except Exception:
        doc = None
    if doc is None:
        dm = getattr(main_window, "doc_manager", getattr(main_window, "docman", None))
        if dm is not None:
            doc = (dm.get_active_document() if hasattr(dm, "get_active_document")
                   else getattr(dm, "active_document", None))
    if doc is None:
        return None

    dlg = CropDialogPro(main_window, doc)
    try:
        dlg.setWindowIcon(QIcon(get_icon_path("cropicon")))
    except Exception:
        pass
    try:
        dlg.seed_from_preset(preset or {})
    except Exception:
        pass
    dlg.show(); dlg.raise_(); dlg.activateWindow()
    return dlg