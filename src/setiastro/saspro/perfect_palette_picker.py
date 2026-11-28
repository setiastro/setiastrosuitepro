#saspro/perfect_palette_picker.py
from __future__ import annotations
import os
import re
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image
import cv2
from PyQt6.QtCore import Qt, QSize, QEvent, QTimer, QPoint, pyqtSignal, QSettings, QByteArray, QThread
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QScrollArea,
    QFileDialog, QInputDialog, QMessageBox, QGridLayout, QCheckBox, QSizePolicy, QDialog,
    QComboBox
)
from PyQt6.QtGui import QPixmap, QImage, QIcon, QPainter, QPen, QColor, QFont, QFontMetrics, QCursor

# legacy loader (same one DocManager uses)
from setiastro.saspro.legacy.image_manager import load_image as legacy_load_image

# your statistical stretch (mono + color) like SASv2
# (same signatures you use elsewhere)
from setiastro.saspro.imageops.stretch import stretch_mono_image, stretch_color_image
from setiastro.saspro.widgets.themed_buttons import themed_toolbtn
from setiastro.saspro.help_support import make_help_button
from setiastro.saspro.thread_keeper import start_owned_thread
from setiastro.saspro.widgets.factory_reset import (
    make_reset_button, snapshot_controls, restore_controls)


# ---------------------------------------------------------------------------
# Off-GUI-thread machinery.
#
# Create Palettes, palette clicks, the balancing dialog's preview and Push run
# in worker QThreads (_PPPJob). Everything a worker touches is plain numpy /
# cv2 / QImage -- no QPixmap, no widgets. The row-chunked passes below use only
# operations that are exact per element (copy, clip, min/max, nan_to_num,
# multiply/divide by one scalar, float->uint8 cast), so splitting rows across
# threads gives the same bytes as the old whole-array calls. The formula
# palettes (Realistic, Foraxx, Dynamic Inverse) are purely per-pixel too: each
# pixel goes through the same element-wise numpy operations (those exact ops
# plus the Foraxx gate's element-wise power) whether a call covers the whole
# frame or a band of rows, so _ppp_map bands them across threads as well.
# ---------------------------------------------------------------------------
_PPP_POOL = None
_PPP_CHUNK = 1 << 20          # elements per row chunk


def _ppp_pool() -> ThreadPoolExecutor:
    global _PPP_POOL
    if _PPP_POOL is None:
        _PPP_POOL = ThreadPoolExecutor(max_workers=max(2, min(8, os.cpu_count() or 4)),
                                       thread_name_prefix="ppp")
    return _PPP_POOL


def _ppp_rows(fn, rows: int, row_elems: int) -> list:
    """Run fn(i, j) over row spans [i, j) covering 0..rows, in parallel; results in order."""
    per = max(1, _PPP_CHUNK // max(1, int(row_elems)))
    spans = [(i, min(rows, i + per)) for i in range(0, rows, per)]
    if len(spans) <= 1:
        return [fn(i, j) for i, j in spans]
    return list(_ppp_pool().map(lambda s: fn(s[0], s[1]), spans))


class _PPPJob(QThread):
    """Runs fn(*args) off the GUI thread and emits ("ok", value) or ("err", message) on
    `result` (deliberately not named `finished`). Start it with start_owned_thread()."""
    result = pyqtSignal(object)

    def __init__(self, fn, *args):
        super().__init__(None)
        self._fn = fn
        self._args = args

    def run(self):
        fn, args = self._fn, self._args
        self._fn = self._args = None
        try:
            out = ("ok", fn(*args))
        except Exception as e:
            traceback.print_exc()
            out = ("err", f"{type(e).__name__}: {e}")
        del fn, args
        self.result.emit(out)


def _ppp_resize_to(arr, size):
    """Resize np array to (w,h). Keeps dtype/scale. Uses INTER_AREA for downsizing."""
    if arr is None:
        return None
    w, h = size
    if arr.ndim == 2:
        src_h, src_w = arr.shape
    else:
        src_h, src_w = arr.shape[:2]
    if (src_w, src_h) == (w, h):
        return arr
    if cv2 is None:
        # ultra-simple fallback: nearest; OK for thumbs if OpenCV isn't present
        if arr.ndim == 2:
            return np.array(Image.fromarray((arr*255).astype(np.uint8)).resize((w, h))).astype(np.float32) / 255.0
        return np.array(Image.fromarray((arr*255).astype(np.uint8)).resize((w, h))).astype(np.float32) / 255.0
    interp = cv2.INTER_AREA if (w < src_w or h < src_h) else cv2.INTER_LINEAR
    if arr.ndim == 2:
        return cv2.resize(arr, (w, h), interpolation=interp)
    return cv2.resize(arr, (w, h), interpolation=interp)


def _ppp_u8(arr) -> np.ndarray:
    """(np.clip(arr, 0, 1) * 255).astype(np.uint8), row-chunked across threads. Same bytes."""
    a = np.asarray(arr)
    if a.ndim < 2 or a.size == 0:
        return (np.clip(a, 0, 1) * 255).astype(np.uint8)
    u = np.empty(a.shape, np.uint8)

    def _job(i, j):
        t = np.clip(a[i:j], 0, 1)
        np.multiply(t, 255, out=t)
        u[i:j] = t
    _ppp_rows(_job, a.shape[0], a[0].size)
    return u


def _ppp_to_qimage(arr) -> QImage:
    """Float [0..1] mono/RGB -> QImage (Grayscale8 / RGB888)."""
    a = np.asarray(arr)
    if a.ndim == 2:
        u = _ppp_u8(a); h, w = u.shape
        return QImage(u.data, w, h, w, QImage.Format.Format_Grayscale8).copy()
    if a.ndim == 3 and a.shape[2] == 3:
        u = _ppp_u8(a); h, w, _ = u.shape
        return QImage(u.data, w, h, w*3, QImage.Format.Format_RGB888).copy()
    raise ValueError(f"Unexpected image shape: {a.shape}")


class _PPPDisplay:
    """An RGB32 QImage for an HxWx3 [0..1] float image, filled chunk by chunk with exactly the
    pixels _ppp_to_qimage gives (RGB32 = RGB888 plus an opaque alpha byte). Built in a worker:
    QPixmap.fromImage of an RGB32 image is a plain copy on the GUI thread, and zoom scaling of
    the pixmap takes Qt's fast path. view is None when unavailable (callers fall back)."""

    def __init__(self, h: int, w: int):
        self.qimg = None
        self.view = None
        if sys.byteorder != "little" or h <= 0 or w <= 0:
            return
        q = QImage(w, h, QImage.Format.Format_RGB32)
        if q.isNull() or q.bytesPerLine() != w * 4:
            return
        ptr = q.bits()
        ptr.setsize(q.sizeInBytes())
        self.view = np.frombuffer(ptr, np.uint8).reshape(h, w, 4)
        self.qimg = q

    def put(self, i, j, blk):
        t = np.clip(blk, 0, 1)
        np.multiply(t, 255, out=t)
        v = self.view[i:j]
        v[..., 2::-1] = t          # R,G,B -> bytes 2,1,0 of 0xffRRGGBB (little-endian)
        v[..., 3] = 255


def _ppp_display_qimage(rgb) -> QImage:
    """Display QImage of a [0..1] image (RGB32 when possible): the same pixels as _ppp_to_qimage."""
    a = np.asarray(rgb)
    if a.ndim == 3 and a.shape[2] == 3:
        d = _PPPDisplay(a.shape[0], a.shape[1])
        if d.view is not None:
            _ppp_rows(lambda i, j: d.put(i, j, a[i:j]), a.shape[0], a.shape[1] * 3)
            return d.qimg
    return _ppp_to_qimage(a)


def _ppp_present_count(ha, oo, si, o1, o2) -> int:
    """How many of Ha / OIII / SII the inputs provide once OSC is synthesized."""
    return int(ha is not None or o1 is not None) \
        + int(oo is not None or o1 is not None or o2 is not None) \
        + int(si is not None or o2 is not None)


def _ppp_synth(ha, oo, si, o1, o2):
    """OSC synthesis: OSC1 R~Ha, mean(G,B)~OIII; OSC2 R~SII, mean(G,B)~OIII."""
    if o1 is not None:  # OSC1: R≈Ha, mean(G,B)≈OIII
        h1 = o1[..., 0]
        g1b1 = o1[..., 1:3].mean(axis=2)
        ha = h1 if ha is None else 0.5*ha + 0.5*h1
        oo = g1b1 if oo is None else 0.5*oo + 0.5*g1b1

    if o2 is not None:  # OSC2: R≈SII, mean(G,B)≈OIII
        s2 = o2[..., 0]
        g2b2 = o2[..., 1:3].mean(axis=2)
        si = s2 if si is None else 0.5*si + 0.5*s2
        oo = g2b2 if oo is None else 0.5*oo + 0.5*g2b2
    return ha, oo, si


def _ppp_synth_shapes(ha, oo, si, o1, o2):
    """The 2-D shapes _ppp_synth would give for Ha / OIII / SII, without computing it."""
    def s(x):
        return None if x is None else tuple(x.shape[:2])
    sh_ha = s(ha) if ha is not None else s(o1)
    sh_oo = s(oo) if oo is not None else (s(o1) if o1 is not None else s(o2))
    sh_si = s(si) if si is not None else s(o2)
    return sh_ha, sh_oo, sh_si


def _ppp_match_sizes(ha, oo, si):
    """Full size: resize mismatched channels to the reference (prefer Ha, then OIII, then SII).
    Only reached once the user agreed (PerfectPalettePicker._confirm_channel_sizes)."""
    shapes = [x.shape[:2] for x in (ha, oo, si) if x is not None]
    if len(shapes) and len(set(shapes)) > 1:
        ref = ha if ha is not None else (oo if oo is not None else si)
        ref_h, ref_w = ref.shape[:2]
        ha = _ppp_resize_to(ha, (ref_w, ref_h)) if ha is not None else None
        oo = _ppp_resize_to(oo, (ref_w, ref_h)) if oo is not None else None
        si = _ppp_resize_to(si, (ref_w, ref_h)) if si is not None else None
    return ha, oo, si


def _ppp_thumb_channels(ha, oo, si, o1, o2, box_w, box_h):
    """Thumbnail Ha / OIII / SII: the same recipe as before (OSC synthesis, then size-match to
    the OIII -> Ha -> SII reference), but on inputs first shrunk to about twice the thumbnail
    box instead of to 50%. The 18 palettes then map ~0.1 MP each instead of a quarter frame;
    each result is scaled down to the button size either way. Thumbnails only."""
    srcs = [x for x in (ha, oo, si, o1, o2) if x is not None]
    if srcs:
        big_h = max(x.shape[0] for x in srcs)
        big_w = max(x.shape[1] for x in srcs)
        s = min(0.5, 2.0 * box_w / max(1, big_w), 2.0 * box_h / max(1, big_h))

        def _shrink(x):
            if x is None:
                return None
            h, w = x.shape[:2]
            return _ppp_resize_to(x, (max(1, int(w * s)), max(1, int(h * s))))
        ha, oo, si, o1, o2 = (_shrink(x) for x in (ha, oo, si, o1, o2))

    ha, oo, si = _ppp_synth(ha, oo, si, o1, o2)

    ref = oo if oo is not None else (ha if ha is not None else si)
    if ref is not None:
        ref_h, ref_w = ref.shape[:2]
        ha = _ppp_resize_to(ha, (ref_w, ref_h)) if ha is not None else None
        oo = _ppp_resize_to(oo, (ref_w, ref_h)) if oo is not None else None
        si = _ppp_resize_to(si, (ref_w, ref_h)) if si is not None else None
    return ha, oo, si


def _ppp_thumbs_job(mapper, names, picks, box_w, box_h):
    """Create Palettes, off the GUI thread: {palette name: QImage, or None if unbuildable}."""
    ha, oo, si = _ppp_thumb_channels(*picks, box_w, box_h)
    out = {}
    for name in names:
        r, g, b = mapper(name, ha, oo, si)
        if any(ch is None for ch in (r, g, b)):
            out[name] = None
            continue
        r = np.clip(np.nan_to_num(r), 0, 1)
        g = np.clip(np.nan_to_num(g), 0, 1)
        b = np.clip(np.nan_to_num(b), 0, 1)
        rgb = np.stack([r, g, b], axis=2).astype(np.float32)
        out[name] = _ppp_to_qimage(rgb)
    return out


def _ppp_finish_rgb(r, g, b):
    """The palette preview: clip(nan_to_num(ch), 0, 1) per channel, stacked, divided by the
    max -- in two threaded passes with no full-frame temporaries -- plus its display QImage.
    Returns (final, in_range, qimage).

    in_range: every channel was already finite and inside [0, 1], so the nan_to_num/clip were
    no-ops. final is then bit-identical to the balancing dialog's image when it is accepted
    untouched (that divides the same channels by the same max and clips afterwards, a no-op).
    """
    chans = (r, g, b)
    if not all(isinstance(c, np.ndarray) and c.ndim == 2 and c.shape == np.shape(r) and c.size
               for c in chans):
        r = np.clip(np.nan_to_num(r), 0, 1)
        g = np.clip(np.nan_to_num(g), 0, 1)
        b = np.clip(np.nan_to_num(b), 0, 1)
        rgb = np.stack([r, g, b], axis=2).astype(np.float32)
        mx = float(rgb.max()) or 1.0
        rgb = (rgb / mx).astype(np.float32)
        return rgb, False, _ppp_display_qimage(rgb)

    H, W = r.shape
    distinct = []
    for c in chans:
        if not any(c is d for d in distinct):
            distinct.append(c)
    rgb = np.empty((H, W, 3), np.float32)

    def _clip(i, j):
        ok = True
        for d in distinct:
            blk = d[i:j]
            if not (blk.min() >= 0 and blk.max() <= 1):      # NaN fails both
                ok = False
        out = rgb[i:j]
        for k, c in enumerate(chans):
            np.clip(c[i:j], 0, 1, out=out[..., k])
        return ok, out.max()
    parts = _ppp_rows(_clip, H, W * 3)
    in_range = all(ok for ok, _ in parts)
    if in_range:
        mx = float(np.max([m for _, m in parts]))
    else:
        def _fix(i, j):
            blk = rgb[i:j]
            np.nan_to_num(blk, copy=False)
            return blk.max()
        mx = float(np.max(_ppp_rows(_fix, H, W * 3)))
    mx = mx or 1.0
    disp = _PPPDisplay(H, W)

    def _norm(i, j):
        blk = rgb[i:j]
        np.divide(blk, mx, out=blk)
        if disp.view is not None:
            disp.put(i, j, blk)
    _ppp_rows(_norm, H, W * 3)
    return rgb, in_range, (disp.qimg if disp.view is not None else _ppp_to_qimage(rgb))


_PPP_FORMULAS = ("Realistic1", "Realistic2", "Foraxx", "Dynamic Inverse")


def _ppp_map(mapper, pal, ha, oo, si):
    """mapper(pal, ha, oo, si). Big frames of the formula palettes are mapped in bands of rows
    across threads (their math has no cross-pixel step, so the bytes are the same); the
    channel-shuffle palettes just return references and stay a single call."""
    srcs = [x for x in (ha, oo, si) if x is not None]
    if (pal not in _PPP_FORMULAS or len(srcs) < 2
            or not all(isinstance(x, np.ndarray) and x.ndim == 2 and x.shape == srcs[0].shape
                       for x in srcs)
            or srcs[0].size < 4 * _PPP_CHUNK):
        return mapper(pal, ha, oo, si)
    H, W = srcs[0].shape
    per = max(1, (2 * _PPP_CHUNK) // W)

    def _band(i, j):
        return mapper(pal, *(None if x is None else x[i:j] for x in (ha, oo, si)))
    first = _band(0, per)
    if not all(isinstance(c, np.ndarray) and c.shape == (per, W) for c in first):
        return mapper(pal, ha, oo, si)
    outs = [np.empty((H, W), c.dtype) for c in first]
    for o, c in zip(outs, first):
        o[:per] = c

    def _job(span):
        i, j = span
        for o, c in zip(outs, _band(i, j)):
            o[i:j] = c
    list(_ppp_pool().map(_job, [(i, min(H, i + per)) for i in range(per, H, per)]))
    return tuple(outs)


def _ppp_gen_job(mapper, pal, picks):
    """Palette click, off the GUI thread: prepared channels -> palette -> preview image."""
    ha, oo, si = _ppp_match_sizes(*_ppp_synth(*picks))
    if sum(x is not None for x in (ha, oo, si)) < 2:
        return None
    r, g, b = _ppp_map(mapper, pal, ha, oo, si)
    if any(ch is None for ch in (r, g, b)):
        return "unbuildable"
    return _ppp_finish_rgb(r, g, b)


def _ppp_scaled(x, f):
    """x * f (the balancing sliders), row-threaded for big 2-D arrays. Same bytes."""
    if x is None:
        return None
    if not (isinstance(x, np.ndarray) and x.ndim == 2 and x.size >= _PPP_CHUNK):
        return x * f
    out = np.empty(x.shape, (x[:1, :1] * f).dtype)
    _ppp_rows(lambda i, j: np.multiply(x[i:j], f, out=out[i:j]), x.shape[0], x.shape[1])
    return out


def _ppp_balance_compose(mapper, pal, ha_src, oiii_src, sii_src, factors, base_shape,
                         want_qimage=False):
    """The balancing dialog's image for intensity factors (fh, fo, fs): scale the sources, map
    the palette, fit to the base size, assemble, divide by the max, clip to [0, 1].
    Returns img, or (img, display QImage) when want_qimage."""
    fh, fo, fs = factors
    ha = _ppp_scaled(ha_src, fh)
    oo = _ppp_scaled(oiii_src, fo)
    si = _ppp_scaled(sii_src, fs)

    r, g, b = _ppp_map(mapper, pal, ha, oo, si)

    # --- make sure channels match the base palette size ---
    H, W = base_shape[:2]

    def fit(ch):
        if ch is None: return None
        if ch.shape[:2] != (H, W):
            return _ppp_resize_to(ch, (W, H))
        return ch
    r, g, b = fit(r), fit(g), fit(b)
    chans = (r, g, b)

    disp = None
    if (tuple(base_shape) == (H, W, 3) and H > 0 and W > 0
            and all(c is None or (isinstance(c, np.ndarray) and c.shape == (H, W)) for c in chans)):
        img = np.empty((H, W, 3), np.float32)

        def _fill(i, j):
            blk = img[i:j]
            for k, c in enumerate(chans):
                if c is None:
                    blk[..., k] = 0
                else:
                    blk[..., k] = c[i:j]
            return blk.max()
        m = float(np.max(_ppp_rows(_fill, H, W * 3))) or 1.0
        if want_qimage:
            disp = _PPPDisplay(H, W)

        def _norm(i, j):
            blk = img[i:j]
            np.divide(blk, m, out=blk)
            np.clip(blk, 0.0, 1.0, out=blk)
            if disp is not None and disp.view is not None:
                disp.put(i, j, blk)
        _ppp_rows(_norm, H, W * 3)
    else:
        img = np.zeros(base_shape, dtype=np.float32)
        if r is not None: img[...,0] = r
        if g is not None: img[...,1] = g
        if b is not None: img[...,2] = b
        m = float(img.max()) or 1.0
        img = np.clip(img/m, 0.0, 1.0)

    if not want_qimage:
        return img
    if disp is not None and disp.view is not None:
        return img, disp.qimg
    return img, _ppp_display_qimage(img)


def _ppp_channel_medians(ha, oo, si) -> dict:
    """{name: median} for channels that look like linear data.
    Values that aren't plausibly linear (median >= 0.5 or <= 1e-6) are dropped."""
    out = {}
    for name, ch in (("Ha", ha), ("OIII", oo), ("SII", si)):
        if ch is None:
            continue
        m = float(np.median(ch))
        if 1e-6 < m < 0.5:
            out[name] = m
    return out


def _ppp_target_median(meds: dict, mode: str):
    """Resolve the target-median dropdown text to one value (Max when a named channel is missing)."""
    if not meds:
        return None
    if mode.startswith("Max"):
        return max(meds.values())
    if mode.startswith("Min"):
        return min(meds.values())
    if mode == "Mean":
        vals = list(meds.values())
        return float(sum(vals) / len(vals))
    if mode.startswith("Median"):
        return float(np.median(list(meds.values())))
    # Named channel — fall back to Max if that channel isn't loaded
    if mode in meds:
        return meds[mode]
    return max(meds.values())


def _ppp_output_linear(rgb, mode, raws):
    """Output linear without the widgets: invert the pre-build stretch with a LINKED no-black-clip
    statistical stretch to the chosen raw channel median. Returns (image, (kind, value, mode))
    with kind "ok" (value = target), "none" (no plausible median) or "err" (value = message)."""
    tgt = _ppp_target_median(_ppp_channel_medians(*_ppp_synth(*raws)), mode)
    if tgt is None:
        return rgb, ("none", None, mode)
    try:
        inverted = stretch_color_image(
            rgb.astype(np.float32, copy=False),
            target_median=float(tgt),
            linked=True,
            no_black_clip=True,
            normalize=False,
        ).astype(np.float32, copy=False)
        return inverted, ("ok", tgt, mode)
    except Exception as e:
        return rgb, ("err", str(e), mode)


def _ppp_push_job(img, needs_copy, compose, olin):
    """Push, off the GUI thread. compose = (mapper, palette, picks, base_shape) builds the
    dialog's untouched image first. Returns (final, output image, preview QImage or None,
    output-linear info or None, final_is_copy)."""
    qimg = None
    if compose is not None:
        mapper, pal, picks, base_shape = compose
        ha, oo, si = _ppp_match_sizes(*_ppp_synth(*picks))
        img, qimg = _ppp_balance_compose(mapper, pal, ha, oo, si, (1.0, 1.0, 1.0), base_shape, True)
    out, info = img, None
    if olin is not None:
        out, info = _ppp_output_linear(img, *olin)
    copied = False
    if needs_copy and out is img:
        img = out = img.copy()
        copied = True
    return img, out, qimg, info, copied


# ---------------------------------------------------------------------------
# Metadata for the pushed palette: the loaded inputs' headers merged into one.
#
# Order = the full-size geometry reference of _ppp_synth_shapes / _ppp_match_sizes
# (Ha, or OSC1 standing in for it; then OIII, or OSC2; then SII), so the first
# header is the one whose pixel grid the output is on. A keyword the inputs
# disagree on keeps the first input's value. The WCS is one input's block, never a
# mix: the first input that has one and is the output's size (a resized channel's
# WCS no longer fits the pixels).
# ---------------------------------------------------------------------------
_PPP_HEADER_ORDER = ("Ha", "OSC1", "OIII", "OSC2", "SII")
_PPP_STRUCTURAL_KEYS = {"SIMPLE", "EXTEND", "XTENSION", "PCOUNT", "GCOUNT", "BITPIX",
                        "NAXIS", "NAXIS1", "NAXIS2", "NAXIS3", "NAXIS4",
                        "BSCALE", "BZERO", "END"}
_PPP_COMMENTARY_KEYS = {"HISTORY", "COMMENT", ""}
_PPP_WCS_KEY = re.compile(
    r"^(WCSAXES|WCSNAME|CTYPE\d|CUNIT\d|CRPIX\d|CRVAL\d|CDELT\d|CROTA\d|CD\d_\d|PC\d_\d|"
    r"PV\d_\d+|PS\d_\d+|LONPOLE|LATPOLE|(A|B|AP|BP)_(ORDER|DMAX|\d+_\d+)|SASWCSSY)$")


def _ppp_as_header(hdr):
    """Any header SASpro stores on a document (astropy Header or XISF-style dict) as a
    Header, or None. A Header comes back as is: callers only read it."""
    if hdr is None:
        return None
    from astropy.io import fits
    if isinstance(hdr, fits.Header):
        return hdr
    if isinstance(hdr, dict):
        out = fits.Header()
        for k, v in hdr.items():
            if not isinstance(k, str) or k.startswith("XISF:"):
                continue
            if isinstance(v, dict) and "value" in v:
                v = v["value"]
            try:
                out[k] = v
            except Exception:
                continue
        return out if len(out) else None
    return None


def _ppp_has_wcs(hdr) -> bool:
    return all(k in hdr for k in ("CTYPE1", "CTYPE2", "CRVAL1", "CRVAL2"))


def _ppp_merged_header(sources, out_hw, palette, linear_out=False):
    """One header for the pushed palette. sources: {input name: (header, (h, w))} of the
    loaded inputs; out_hw: the output's (h, w). FILTER becomes the palette name."""
    import copy
    from astropy.io import fits

    h, w = int(out_hw[0]), int(out_hw[1])
    merged = fits.Header()
    merged["SIMPLE"] = True
    merged["BITPIX"] = -32
    merged["NAXIS"] = 3
    merged["NAXIS1"] = w
    merged["NAXIS2"] = h
    merged["NAXIS3"] = 3

    used = []                                   # (input name, Header or None)
    for name in _PPP_HEADER_ORDER:
        if name in sources:
            hdr, _hw = sources[name]
            used.append((name, _ppp_as_header(hdr)))
    shapes = {name: tuple(sources[name][1]) for name, _h in used}
    donor = next((name for name, hdr in used
                  if hdr is not None and shapes[name] == (h, w) and _ppp_has_wcs(hdr)), None)

    seen = set()                                # commentary lines of the earlier inputs
    for name, hdr in used:
        if hdr is None:
            continue
        cards = list(hdr.cards)
        mine = set()
        i = 0
        while i < len(cards):
            j = i + 1
            try:
                card = cards[i]
                key = card.keyword
                if key in _PPP_COMMENTARY_KEYS:
                    # A long HISTORY/COMMENT is split over consecutive 72-character cards
                    # (trailing spaces stripped on read): keep or drop the line whole, so
                    # no input leaves a fragment. Only a line an earlier input had is dropped.
                    while (j < len(cards) and cards[j].keyword == key
                           and len(str(cards[j - 1].value)) >= 70):
                        j += 1
                    line = (key, tuple(str(c.value) for c in cards[i:j]))
                    if line not in seen:
                        mine.add(line)
                        for c in cards[i:j]:
                            merged.append(copy.copy(c))
                elif key in _PPP_STRUCTURAL_KEYS:
                    pass
                elif _PPP_WCS_KEY.match(key) and name != donor:
                    pass
                elif key not in merged:
                    merged.append(copy.copy(card))
            except Exception:
                pass
            i = j
        seen |= mine

    inputs = []
    for name, hdr in used:
        filt = ""
        if hdr is not None:
            try:
                filt = str(hdr.get("FILTER", "") or "").strip()
            except Exception:
                filt = ""
        inputs.append(f"{name} ({filt})" if filt else name)
    merged["FILTER"] = (str(palette), "Perfect Palette Picker palette")
    note = f"Perfect Palette Picker: {palette} palette from {', '.join(inputs)}"
    if linear_out:
        note += ", linear output"
    merged.add_history(note)
    return merged


def _ppp_push_metadata(sources, out_img, palette, linear_out=False):
    """Metadata for the pushed document: the merged header plus the WCS built from it."""
    meta = {"is_mono": False, "bit_depth": "32-bit floating point"}
    try:
        hdr = _ppp_merged_header(sources, out_img.shape[:2], palette, linear_out)
    except Exception as e:
        print(f"[PerfectPalettePicker] could not merge input headers: {e}")
        return meta
    meta["original_header"] = hdr
    try:
        from setiastro.saspro.legacy.image_manager import attach_wcs_to_metadata
        meta = attach_wcs_to_metadata(meta, hdr)
    except Exception as e:
        print(f"[PerfectPalettePicker] could not attach WCS: {e}")
    return meta


class PaletteAdjustDialog(QDialog):
    adjusted_image = pyqtSignal(np.ndarray)

    def __init__(self, base_rgb, palette_name, ha_src, oiii_src, sii_src, owner,
                 orig_pixmap=None, base_is_untouched=False):
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QScrollArea, QSlider
        from PyQt6.QtCore import QTimer, Qt, QPoint, QEvent
        super().__init__(owner)
        self.setWindowTitle(self.tr("Adjust Palette Intensities"))
        self.setWindowFlag(Qt.WindowType.Window, True)
        import platform
        if platform.system() == "Darwin":
            self.setWindowFlag(Qt.WindowType.Tool, True)  
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setModal(False)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)

        # Read-only here (its shape, and the untouched result), so no full-frame copy.
        self.base_rgb     = base_rgb.astype(np.float32, copy=False)
        self.palette_name = palette_name
        self.ha_src       = ha_src
        self.oiii_src     = oiii_src
        self.sii_src      = sii_src
        self.owner        = owner

        self.ha_factor = 1.0
        self.oiii_factor = 1.0
        self.sii_factor = 1.0

        self._debounce = QTimer(self); self._debounce.setInterval(150); self._debounce.setSingleShot(True)
        self._debounce.timeout.connect(self._update_preview)

        self.zoom_factor = 1.0
        self._dragging = False
        self._last_pos = QPoint()

        # Preview / accept images are built off the GUI thread (_PPPJob). A slider change while
        # one is building is queued: the latest factors are built when it lands.
        self._thr = None
        self._thr_factors = None
        self._want_rerun = False
        self._accept_pending = False
        self._result = None           # (factors, float image) of the last finished build
        self._closed = False
        # The picker checked that base_rgb is bit-identical to the all-100% image.
        self._base_is_untouched = bool(base_is_untouched)
        self.accepted_pixmap = None   # display pixmap of the image emitted on Accept

        # Original palette pixmap for A/B compare (checkbox off). The picker hands over its
        # own preview pixmap of this very image (same pixels) instead of converting it again.
        if orig_pixmap is not None and not orig_pixmap.isNull():
            self._orig_pixmap = orig_pixmap
        else:
            self._orig_pixmap = QPixmap.fromImage(_ppp_display_qimage(self.base_rgb))
        self._adj_pixmap = None

        vlayout = QVBoxLayout(self)

        # Zoom controls
        zoom_layout = QHBoxLayout()

        self.btn_zoom_in  = themed_toolbtn("zoom-in", self.tr("Zoom In"))
        self.btn_zoom_out = themed_toolbtn("zoom-out", self.tr("Zoom Out"))
        self.btn_fit      = themed_toolbtn("zoom-fit-best", self.tr("Fit to Preview"))

        self.btn_zoom_in.clicked.connect(lambda: self._change_zoom(1.25))
        self.btn_zoom_out.clicked.connect(lambda: self._change_zoom(0.8))
        self.btn_fit.clicked.connect(self._fit_to_preview)

        zoom_layout.addStretch(1)
        zoom_layout.addWidget(self.btn_zoom_out)
        zoom_layout.addWidget(self.btn_zoom_in)
        zoom_layout.addWidget(self.btn_fit)
        zoom_layout.addStretch(1)

        vlayout.addLayout(zoom_layout)

        # Preview
        self.preview_area = QScrollArea(self); self.preview_area.setWidgetResizable(True)
        self.preview_label = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setCursor(Qt.CursorShape.OpenHandCursor)
        self.preview_label.setMouseTracking(True)
        self.preview_area.setWidget(self.preview_label)
        self.preview_label.installEventFilter(self)
        vlayout.addWidget(self.preview_area, stretch=1)

        # Show adjusted vs original palette
        self.chk_show_adj = QCheckBox(self.tr("Show adjustment"), self)
        self.chk_show_adj.setChecked(True)
        self.chk_show_adj.setToolTip(self.tr(
            "Checked: preview the intensity-adjusted palette.\n"
            "Unchecked: show the original (unadjusted) palette."
        ))
        self.chk_show_adj.toggled.connect(self._on_show_adjustment_toggled)
        vlayout.addWidget(self.chk_show_adj)

        # Sliders
        for name in ("Ha","OIII","SII"):
            row = QHBoxLayout()
            row.addWidget(QLabel(self.tr("{0} Intensity:").format(name), self))
            sl = QSlider(Qt.Orientation.Horizontal, self); sl.setRange(0,200); sl.setValue(100)
            sl.valueChanged.connect(self._on_slider_change)
            setattr(self, f"_{name.lower()}_slider", sl)
            row.addWidget(sl)
            vlayout.addLayout(row)

        # Buttons
        btns = QHBoxLayout(); btns.addStretch()
        accept  = QPushButton(self.tr("Accept"), self); accept.clicked.connect(self._on_accept)
        reset   = QPushButton(self.tr("Reset"),  self); reset.clicked.connect(self._on_reset)
        discard = QPushButton(self.tr("Discard"),self); discard.clicked.connect(self.reject)
        btns.addWidget(accept); btns.addWidget(reset); btns.addWidget(discard)
        vlayout.addLayout(btns)
        self._btn_accept = accept
        self._btn_reset = reset

        # Untouched, the adjusted image is base_rgb itself when the picker vouched for it, so
        # the original pixmap is shown as-is; otherwise it is built off the GUI thread.
        if self._base_is_untouched:
            self._rescale_pixmap()
        else:
            self._update_preview()

    def _on_slider_change(self, _):
        self.ha_factor   = self._ha_slider.value()/100.0
        self.oiii_factor = self._oiii_slider.value()/100.0
        self.sii_factor  = self._sii_slider.value()/100.0
        self._debounce.start()

    def _on_show_adjustment_toggled(self, _checked: bool):
        self._rescale_pixmap()

    def _active_pixmap(self) -> QPixmap | None:
        if self.chk_show_adj.isChecked():
            return self._adj_pixmap if self._adj_pixmap is not None else self._orig_pixmap
        return self._orig_pixmap

    def _factors(self):
        return (self.ha_factor, self.oiii_factor, self.sii_factor)

    def _update_preview(self):
        """Show the adjusted palette for the current slider factors. Built off the GUI thread;
        a change while a build runs is queued and the latest factors are built when it lands."""
        if self._closed:
            return
        if self._thr is not None:
            self._want_rerun = True
            return
        f = self._factors()
        if self._result is not None and self._result[0] == f:
            return
        if self._base_is_untouched and f == (1.0, 1.0, 1.0):
            # all sliders back at 100%: that image is base_rgb, already on screen as the original
            self._result = None
            self._adj_pixmap = None
            self._rescale_pixmap()
            return
        self._start_build(f)

    def _start_build(self, f):
        owner = self.owner
        if owner is None or self._closed:
            return
        thr = _PPPJob(_ppp_balance_compose, owner._map_channels_or_special, self.palette_name,
                      self.ha_src, self.oiii_src, self.sii_src, f, self.base_rgb.shape, True)
        thr.result.connect(self._on_built)
        self._thr = thr
        self._thr_factors = f
        self._want_rerun = False
        self._result = None          # superseded; don't hold two full frames during the build
        start_owned_thread(thr)

    def _on_built(self, out):
        f = self._thr_factors
        self._thr = None
        if self._closed:
            return
        kind, val = out
        if kind == "err":
            if self._accept_pending:
                self._accept_pending = False
                self._set_controls_enabled(True)
            QMessageBox.warning(self, self.tr("Adjust Palette"), self.tr("Could not build the adjusted palette:\n{0}").format(val))
            return
        img, qimg = val
        self._result = (f, img)
        self._adj_pixmap = QPixmap.fromImage(qimg)
        cur = self._factors()
        if self._accept_pending:
            if f == cur:
                self._finish_accept(img, self._adj_pixmap)
            else:
                self._start_build(cur)
            return
        self._rescale_pixmap()
        if self._want_rerun:
            self._want_rerun = False
            self._update_preview()

    def _set_controls_enabled(self, on: bool):
        for w in (self._ha_slider, self._oiii_slider, self._sii_slider,
                  self._btn_accept, self._btn_reset):
            w.setEnabled(on)

    def _rescale_pixmap(self):
        pm = self._active_pixmap()
        if pm is None:
            return
        w = int(pm.width()  * self.zoom_factor)
        h = int(pm.height() * self.zoom_factor)
        scaled = pm.scaled(w, h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self._current_pixmap = scaled
        self.preview_label.setPixmap(scaled)
        self.preview_label.resize(scaled.size())

    def _change_zoom(self, factor: float):
        self.zoom_factor = max(0.1, min(10.0, self.zoom_factor * factor))
        self._rescale_pixmap()

    def _fit_to_preview(self):
        pm = self._active_pixmap()
        if pm is None:
            return
        vp = self.preview_area.viewport().size()
        if pm.width() <= 0 or pm.height() <= 0:
            return
        k = min(vp.width() / pm.width(), vp.height() / pm.height())
        self.zoom_factor = max(0.1, min(10.0, k))
        self._rescale_pixmap()


    def _on_reset(self):
        for s in (self._ha_slider, self._oiii_slider, self._sii_slider):
            s.setValue(100)
        self._on_slider_change(None)

    def _on_accept(self):
        if self._closed or self._accept_pending:
            return
        self._debounce.stop()
        f = self._factors()
        if self._base_is_untouched and f == (1.0, 1.0, 1.0):
            self._finish_accept(self.base_rgb, self._orig_pixmap)
            return
        if self._result is not None and self._result[0] == f:
            self._finish_accept(self._result[1], self._adj_pixmap)
            return
        # The image for these factors isn't built yet: build it, then accept. Sliders are
        # locked meanwhile; Discard still works.
        self._accept_pending = True
        self._set_controls_enabled(False)
        if self._thr is None:
            self._start_build(f)

    def _finish_accept(self, img, pm):
        self._accept_pending = False
        self.accepted_pixmap = pm
        self.adjusted_image.emit(img)
        self.accept()

    def done(self, r):
        # Accept / Discard / Esc: drop late build results and the cached image.
        self._closed = True
        self._debounce.stop()
        self._result = None
        super().done(r)

    def eventFilter(self, obj, evt):
        if obj is self.preview_label:
            if evt.type() == QEvent.Type.MouseButtonPress and evt.button() == Qt.MouseButton.LeftButton:
                self._dragging = True; self._last_pos = evt.pos()
                self.preview_label.setCursor(Qt.CursorShape.ClosedHandCursor); return True
            if evt.type() == QEvent.Type.MouseMove and self._dragging:
                d = evt.pos() - self._last_pos; self._last_pos = evt.pos()
                self.preview_area.horizontalScrollBar().setValue(self.preview_area.horizontalScrollBar().value() - d.x())
                self.preview_area.verticalScrollBar().setValue(self.preview_area.verticalScrollBar().value() - d.y())
                return True
            if evt.type() == QEvent.Type.MouseButtonRelease and evt.button() == Qt.MouseButton.LeftButton:
                self._dragging = False; self.preview_label.setCursor(Qt.CursorShape.OpenHandCursor); return True
            if evt.type() == QEvent.Type.Wheel:
                self._change_zoom(1.1 if evt.angleDelta().y() > 0 else 0.9); return True
        return super().eventFilter(obj, evt)

    def closeEvent(self, e):
        # Drop full-size float32 buffers + pixmaps as soon as the dialog closes.
        self._closed = True
        self._result = None
        self.base_rgb = None
        self.ha_src = self.oiii_src = self.sii_src = None
        self.owner = None
        self._orig_pixmap = None
        self._adj_pixmap = None
        self._current_pixmap = None
        try:
            self.preview_label.clear()
        except Exception:
            pass
        super().closeEvent(e)

class PaletteGridWindow(QDialog):
    """Non-modal companion window holding the palette thumbnail grid."""
    def __init__(self, owner, grid_host):
        super().__init__(owner)
        self.setWindowTitle(self.tr("Palette Gallery"))
        self.setWindowFlag(Qt.WindowType.Window, True)
        self.setModal(False)
        self._owner = owner

        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setWidget(grid_host)
        lay.addWidget(scroll)

    def closeEvent(self, e):
        # Don't destroy the grid; just hide so the toggle can reopen it.
        try:
            self._owner.btn_gallery.setChecked(False)
        except Exception:
            pass
        self.hide()
        e.ignore()


class PerfectPalettePicker(QWidget):
    THUMB_CROP = 512  # side length for thumbnail center crops
    PALETTES = [
        "HHO","HOO","HOS","HSO",
        "HSS","OHH","OHS","OOS",
        "OSH","OSS","SHH","SHO",
        "SOH","SOO",
        "Realistic1","Realistic2","Foraxx",
        "Dynamic Inverse",
    ]

    def __init__(self, doc_manager=None, parent=None):
        super().__init__(parent)
        self.doc_manager = doc_manager
        self.setWindowFlag(Qt.WindowType.Window, True)
        self.setWindowTitle(self.tr("Perfect Palette Picker"))
        self._settings = QSettings()
        self._persist_prefix = "perfect_palette_picker"
        self._geom_restored = False
        # raw channels (float32 ~[0..1])
        self.ha   = None
        self.oiii = None
        self.sii  = None
        self.osc1 = None
        self.osc2 = None
        self._dim_mismatch_accepted = False
        # input name -> (header as loaded, (h, w)): merged into the pushed view's metadata
        self._src_meta: dict[str, tuple] = {}

        # stretched cache (per input name → stretched array)
        self._stretched: dict[str, np.ndarray] = {}

        self.final = None
        self.current_palette = None
        self._thumb_base_pm: dict[str, QPixmap] = {}   # palette name -> base pixmap (image only)
        self._selected_name: str | None = None

        # thumbs
        self._thumb_buttons: dict[str, QPushButton] = {}

        self._base_pm: QPixmap | None = None
        self._zoom = 1.0
        self._min_zoom = 0.05
        self._max_zoom = 6.0
        self._panning = False
        self._pan_last: QPoint | None = None

        # Off-GUI-thread jobs (_PPPJob) and the FIFO of clicks that arrive while one runs.
        self._gen_thread = None      # palette click
        self._gen_req = None
        self._push_thread = None     # push
        self._push_req = None
        self._thumbs_thread = None   # Create Palettes
        self._thumbs_pending = False
        self._thumbs_serial = 0
        self._queue: list[tuple] = []
        self._inputs_rev = 0         # bumped whenever the loaded/stretched inputs change
        self._clear_serial = 0       # bumped by Clear / teardown: older results are stale
        self._final_key = None       # set while self.final == the dialog's untouched image
        self._final_shared = False   # self.final was handed to a document
        self._base_pm_src = None     # the array _base_pm shows

        self._build_ui()
        # Factory defaults for Reset, before showEvent restores the saved
        # settings. The gallery toggle is view state, not a setting.
        self._factory = snapshot_controls(self, exclude=(self.btn_gallery,))

        # Ensure PPP (and its gallery) don't outlive the application.
        from PyQt6.QtWidgets import QApplication
        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self._on_app_quit)

    def _k(self, key: str) -> str:
        return f"{self._persist_prefix}/{key}"

    # ---------------- UI ----------------
    def _build_ui(self):
        root = QHBoxLayout(self)

        # -------- left controls
        left = QVBoxLayout()
        left_host = QWidget(self); left_host.setLayout(left); left_host.setFixedWidth(300)

        left.addWidget(QLabel(self.tr("<b>Load channels</b>")))

        # Load buttons + status labels
        self.btn_ha   = QPushButton(self.tr("Load Ha…"));   self.btn_ha.clicked.connect(lambda: self._load_channel("Ha"))
        self.btn_oiii = QPushButton(self.tr("Load OIII…")); self.btn_oiii.clicked.connect(lambda: self._load_channel("OIII"))
        self.btn_sii  = QPushButton(self.tr("Load SII…"));  self.btn_sii.clicked.connect(lambda: self._load_channel("SII"))
        self.btn_osc1 = QPushButton(self.tr("Load OSC1 (Ha/OIII)…")); self.btn_osc1.clicked.connect(lambda: self._load_channel("OSC1"))
        self.btn_osc2 = QPushButton(self.tr("Load OSC2 (SII/OIII)…")); self.btn_osc2.clicked.connect(lambda: self._load_channel("OSC2"))

        self.lbl_ha   = QLabel(self.tr("No {0} loaded.").format("Ha"))
        self.lbl_oiii = QLabel(self.tr("No {0} loaded.").format("OIII"))
        self.lbl_sii  = QLabel(self.tr("No {0} loaded.").format("SII"))
        self.lbl_osc1 = QLabel(self.tr("No {0} loaded.").format("OSC1"))
        self.lbl_osc2 = QLabel(self.tr("No {0} loaded.").format("OSC2"))
        for lab in (self.lbl_ha, self.lbl_oiii, self.lbl_sii, self.lbl_osc1, self.lbl_osc2):
            lab.setWordWrap(True); lab.setStyleSheet("color:#888; margin-left:8px;")

        for btn, lab in (
            (self.btn_ha, self.lbl_ha),
            (self.btn_oiii, self.lbl_oiii),
            (self.btn_sii, self.lbl_sii),
            (self.btn_osc1, self.lbl_osc1),
            (self.btn_osc2, self.lbl_osc2),
        ):
            left.addWidget(btn); left.addWidget(lab)

        # Linear toggle (stretch BEFORE palette build)
        self.chk_linear = QCheckBox(self.tr("Linear input (apply statistical stretch before build)"))
        self.chk_linear.setChecked(True)
        self.chk_linear.stateChanged.connect(self._rebuild_stretch_cache_for_all)
        left.addSpacing(6); left.addWidget(self.chk_linear)

        # Output linear: invert the stretch on push so the new document is linear
        self.chk_output_linear = QCheckBox(self.tr("Output linear (invert stretch on push)"))
        self.chk_output_linear.setChecked(False)
        self.chk_output_linear.setToolTip(self.tr(
            "When pushing to a new view, apply a linked no-clip statistical stretch\n"
            "back to the original channel median so the output lives in linear space.\n"
            "Works for Foraxx and Dynamic Inverse too.\n"
            "The on-screen preview stays stretched so you can see what you're doing."
        ))
        self.chk_output_linear.setEnabled(self.chk_linear.isChecked())
        left.addWidget(self.chk_output_linear)

        # Which channel median to target when reverting
        row_tm = QHBoxLayout()
        self.lbl_target_median = QLabel(self.tr("Target median:"))
        self.lbl_target_median.setStyleSheet("margin-left:8px;")
        self.cmb_target_median = QComboBox(self)
        # Data = the English mode key that is persisted and parsed by
        # _ppp_target_median; the label is translated.
        for _key, _label in (
            ("Max (least stretched)", self.tr("Max (least stretched)")),
            ("Min (most stretched)", self.tr("Min (most stretched)")),
            ("Mean", self.tr("Mean")),
            ("Median of medians", self.tr("Median of medians")),
            ("Ha", "Ha"),
            ("OIII", "OIII"),
            ("SII", "SII"),
        ):
            self.cmb_target_median.addItem(_label, _key)
        self.cmb_target_median.setCurrentIndex(0)
        self.cmb_target_median.setToolTip(self.tr(
            "Which raw channel median to target when inverting the stretch.\n"
            "• Max — preserves the brightest channel's dynamic range (default)\n"
            "• Min — pushes everything deeper (noisier)\n"
            "• Mean / Median of medians — compromise\n"
            "• Ha / OIII / SII — force a specific channel (falls back to Max if missing)"
        ))
        self.cmb_target_median.setEnabled(self.chk_linear.isChecked() and
                                         self.chk_output_linear.isChecked())
        row_tm.addWidget(self.lbl_target_median)
        row_tm.addWidget(self.cmb_target_median, 1)
        left.addLayout(row_tm)

        # Enable/disable wiring: output-linear needs linear-input on;
        # the dropdown needs output-linear on.
        def _sync_output_linear_enabled(*_):
            on = self.chk_linear.isChecked()
            self.chk_output_linear.setEnabled(on)
            if not on:
                # If linear input turns off, output-linear is meaningless
                self.chk_output_linear.setChecked(False)
            self.cmb_target_median.setEnabled(on and self.chk_output_linear.isChecked())
        self.chk_linear.stateChanged.connect(_sync_output_linear_enabled)
        self.chk_output_linear.stateChanged.connect(
            lambda *_: self.cmb_target_median.setEnabled(
                self.chk_linear.isChecked() and self.chk_output_linear.isChecked()
            )
        )
        self.chk_output_linear.stateChanged.connect(lambda *_: self._save_ui_state())
        self.cmb_target_median.currentIndexChanged.connect(lambda *_: self._save_ui_state())

        # Actions
        self.btn_clear = QPushButton(self.tr("Clear Loaded Channels"))
        self.btn_clear.clicked.connect(self._clear_channels)
        left.addWidget(self.btn_clear)

        self.btn_create = QPushButton(self.tr("Create Palettes"))
        self.btn_create.clicked.connect(self._create_palettes)
        left.addWidget(self.btn_create)

        self.btn_push = QPushButton(self.tr("Push Final to New View"))
        self.btn_push.clicked.connect(self._push_final)
        left.addWidget(self.btn_push)

        self.chk_balance_on_push = QCheckBox(self.tr("Open balancing preview when pushing"))
        self.chk_balance_on_push.setChecked(True)
        self.chk_balance_on_push.setToolTip(self.tr(
            "Checked: Push Final opens the Adjust Palette Intensities dialog\n"
            "(Ha / OIII / SII intensity sliders) before creating the new view.\n"
            "Unchecked: Push Final creates the new view straight away with the palette\n"
            "as previewed \u2014 the same image as opening the dialog and clicking\n"
            "Accept without touching a slider."
        ))
        self.chk_balance_on_push.toggled.connect(self._on_balance_on_push_toggled)
        left.addWidget(self.chk_balance_on_push)

        self.btn_gallery = QPushButton(self.tr("Show Palette Gallery"))
        self.btn_gallery.setCheckable(True)
        self.btn_gallery.setChecked(True)
        self.btn_gallery.toggled.connect(self._toggle_gallery)
        left.addWidget(self.btn_gallery)

        self.btn_reset = make_reset_button(self)
        self.btn_reset.clicked.connect(self._reset_to_factory)
        left.addWidget(self.btn_reset)

        left.addStretch(1)
        # Grip + help: their own row at the bottom of the left column
        from setiastro.saspro.shortcuts import ShortcutGrip
        self.btn_doc_help = make_help_button("perfect_palette_picker", self)
        grip_row = QHBoxLayout()
        grip_row.setContentsMargins(0, 0, 0, 0)
        grip_row.addWidget(ShortcutGrip("ppp", self))
        grip_row.addStretch(1)
        grip_row.addWidget(self.btn_doc_help)
        left.addLayout(grip_row)
        root.addWidget(left_host, 0)

        # -------- right: preview + fixed-size 4×3 grid
        right = QVBoxLayout()

        # zoom toolbar
        # zoom toolbar (themed)
        tools = QHBoxLayout()
        self.btn_zoom_in  = themed_toolbtn("zoom-in", self.tr("Zoom In"))
        self.btn_zoom_out = themed_toolbtn("zoom-out", self.tr("Zoom Out"))
        self.btn_fit      = themed_toolbtn("zoom-fit-best", self.tr("Fit to Preview"))

        self.btn_zoom_in.clicked.connect(lambda: self._zoom_at(1.25))
        self.btn_zoom_out.clicked.connect(lambda: self._zoom_at(0.8))
        self.btn_fit.clicked.connect(self._fit_to_preview)

        tools.addStretch(1)
        tools.addWidget(self.btn_zoom_out)
        tools.addWidget(self.btn_zoom_in)
        tools.addWidget(self.btn_fit)
        tools.addStretch(1)
        right.addLayout(tools)


        # main preview (expands)
        self.scroll = QScrollArea(self); self.scroll.setWidgetResizable(True)
        self.scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        self.scroll.setWidget(self.preview)
        self.preview.setMouseTracking(True)
        self.preview.installEventFilter(self)
        self.scroll.viewport().installEventFilter(self)     
        self.scroll.installEventFilter(self)  
        self.scroll.horizontalScrollBar().installEventFilter(self)  # NEW
        self.scroll.verticalScrollBar().installEventFilter(self)    # NEW        
        right.addWidget(self.scroll, 1)

        # fixed-size grid
        self.grid = QGridLayout()
        self.grid.setHorizontalSpacing(8); self.grid.setVerticalSpacing(8)
        self.grid.setContentsMargins(8, 8, 8, 8)

        self.thumb_size = QSize(220, 110)
        btn_w = self.thumb_size.width() + 2
        btn_h = self.thumb_size.height() + 2
        cols = 6            # was 4  →  18 palettes = clean 3×6
        rows = (len(self.PALETTES) + cols - 1) // cols

        for idx, name in enumerate(self.PALETTES):
            r, c = divmod(idx, cols)
            b = QPushButton("")  # we draw the text onto the icon itself
            b.setToolTip(name)
            b.setIconSize(self.thumb_size)
            b.setFixedSize(btn_w, btn_h)
            b.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            b.clicked.connect(lambda _=None, n=name: self._on_palette_clicked(n))
            b.setStyleSheet("QPushButton{background:#222;border:1px solid #333;} QPushButton:hover{border-color:#555;}")
            self._thumb_buttons[name] = b
            self.grid.addWidget(b, r, c)

        grid_host = QWidget(); grid_host.setLayout(self.grid)
        hspacing = self.grid.horizontalSpacing(); vspacing = self.grid.verticalSpacing()
        m = self.grid.contentsMargins()
        grid_w = cols*btn_w + (cols-1)*hspacing + m.left() + m.right()
        grid_h = rows*btn_h + (rows-1)*vspacing + m.top() + m.bottom()
        grid_host.setMinimumSize(grid_w, grid_h)   # was setFixedSize
        self._grid_host = grid_host                 # stash for the popout

        # grid now lives in its own non-modal gallery window (created below)
        self._gallery = PaletteGridWindow(self, grid_host)
        self._gallery.resize(grid_w + 40, grid_h + 40)

        self.status = QLabel(""); right.addWidget(self.status, 0)

        right_host = QWidget(self); right_host.setLayout(right)
        root.addWidget(right_host, 1)
        self.chk_linear.stateChanged.connect(lambda *_: self._save_ui_state())
        self.setLayout(root)
        self.setMinimumSize(left_host.width() + 480, 560)

    def _resize_to(self, arr: np.ndarray | None, size: tuple[int, int]) -> np.ndarray | None:
        """Resize np array to (w,h). Keeps dtype/scale. Uses INTER_AREA for downsizing."""
        return _ppp_resize_to(arr, size)

    def _reset_to_factory(self):
        """Every option back to its factory default (each one saves on change).
        Loaded channels and the picked palette stay."""
        if self._busy() or self._thumbs_thread is not None:
            return
        restore_controls(self._factory)

    def _on_balance_on_push_toggled(self, on: bool):
        try:
            self._settings.setValue(self._k("open_balance_on_push"), bool(on))
            self._settings.sync()
        except Exception:
            pass

    def _save_ui_state(self):
        try:
            s = self._settings
            s.setValue(self._k("window_geometry"), self.saveGeometry())
            s.setValue(self._k("linear"), bool(self.chk_linear.isChecked()))
            if hasattr(self, "chk_output_linear"):
                s.setValue(self._k("output_linear"), bool(self.chk_output_linear.isChecked()))
            if hasattr(self, "cmb_target_median"):
                s.setValue(self._k("target_median_mode"), self.cmb_target_median.currentData())
            if hasattr(self, "chk_balance_on_push"):
                s.setValue(self._k("open_balance_on_push"), bool(self.chk_balance_on_push.isChecked()))
            if getattr(self, "_gallery", None) is not None:
                s.setValue(self._k("gallery_geometry"), self._gallery.saveGeometry())
                s.setValue(self._k("gallery_visible"), bool(self._gallery.isVisible()))
            try:
                s.sync()
            except Exception:
                pass
        except Exception:
            pass


    def _restore_ui_state(self):
        try:
            s = self._settings

            g = s.value(self._k("window_geometry"), None)
            if g is not None:
                self.restoreGeometry(g)

            linear = bool(s.value(self._k("linear"), True, type=bool))
            self.chk_linear.blockSignals(True)
            self.chk_linear.setChecked(linear)
            self.chk_linear.blockSignals(False)

            output_linear = bool(s.value(self._k("output_linear"), False, type=bool))
            self.chk_output_linear.blockSignals(True)
            self.chk_output_linear.setChecked(output_linear and linear)
            self.chk_output_linear.setEnabled(linear)
            self.chk_output_linear.blockSignals(False)

            tm_mode = s.value(self._k("target_median_mode"), "Max (least stretched)", type=str)
            idx = self.cmb_target_median.findData(tm_mode)
            if idx >= 0:
                self.cmb_target_median.blockSignals(True)
                self.cmb_target_median.setCurrentIndex(idx)
                self.cmb_target_median.blockSignals(False)
            self.cmb_target_median.setEnabled(linear and self.chk_output_linear.isChecked())

            balance = bool(s.value(self._k("open_balance_on_push"), True, type=bool))
            self.chk_balance_on_push.blockSignals(True)
            self.chk_balance_on_push.setChecked(balance)
            self.chk_balance_on_push.blockSignals(False)

            gg = s.value(self._k("gallery_geometry"), None)
            if gg is not None and getattr(self, "_gallery", None) is not None:
                self._gallery.restoreGeometry(gg)

        except Exception:
            pass

    def _capture_view_state(self):
        """Capture current view center in base-image coordinates + zoom."""
        if self._base_pm is None:
            return None
        vp = self.scroll.viewport()
        hbar = self.scroll.horizontalScrollBar()
        vbar = self.scroll.verticalScrollBar()

        # center of viewport in viewport coords
        anchor_vp = QPoint(vp.width() // 2, vp.height() // 2)

        # convert to label coords (scaled image coords)
        anchor_lbl = self.preview.mapFrom(vp, anchor_vp)

        # scaled -> base image coords
        base_x = anchor_lbl.x() / max(self._zoom, 1e-6)
        base_y = anchor_lbl.y() / max(self._zoom, 1e-6)

        pm = self._base_pm.size()
        fx = 0.5 if pm.width()  <= 0 else (base_x / pm.width())
        fy = 0.5 if pm.height() <= 0 else (base_y / pm.height())

        return {"zoom": float(self._zoom), "fx": float(fx), "fy": float(fy)}

    def _restore_view_state(self, state):
        """Restore zoom and pan using stored base-image fractions."""
        if not state or self._base_pm is None:
            return

        # restore zoom first
        self._zoom = max(self._min_zoom, min(self._max_zoom, float(state["zoom"])))
        self._update_preview_pixmap()

        # now restore center point
        pm = self._base_pm.size()
        fx = float(state.get("fx", 0.5))
        fy = float(state.get("fy", 0.5))

        base_x = fx * pm.width()
        base_y = fy * pm.height()

        # base -> scaled label coords
        lbl_x = int(base_x * self._zoom)
        lbl_y = int(base_y * self._zoom)

        vp = self.scroll.viewport()
        anchor_vp = QPoint(vp.width() // 2, vp.height() // 2)

        hbar = self.scroll.horizontalScrollBar()
        vbar = self.scroll.verticalScrollBar()
        hbar.setValue(max(hbar.minimum(), min(hbar.maximum(), lbl_x - anchor_vp.x())))
        vbar.setValue(max(vbar.minimum(), min(vbar.maximum(), lbl_y - anchor_vp.y())))

    # ---------- status helpers ----------
    def _set_status_label(self, which: str, text: str | None):
        lab = getattr(self, f"lbl_{which.lower()}")
        if text:
            lab.setText(text)
            lab.setStyleSheet("color:#2a7; font-weight:600; margin-left:8px;")
        else:
            lab.setText(self.tr("No {0} loaded.").format(which))
            lab.setStyleSheet("color:#888; margin-left:8px;")

    # ------------- load by view/file -------------
    def _load_channel(self, which: str):
        src, ok = QInputDialog.getItem(
            self, self.tr("Load {0}").format(which), self.tr("Source:"), ["From View", "From File"], 0, False
        )
        if not ok:
            return

        if src == "From View":
            out = self._load_from_view(which)
        else:
            out = self._load_from_file(which)
        if out is None:
            return

        img, header, bit_depth, is_mono, path, label = out

        # NB channels → mono; OSC → RGB
        if which in ("Ha","OIII","SII"):
            if img.ndim == 3:
                img = img[:, :, 0]
        else:
            if img.ndim == 2:
                img = np.stack([img]*3, axis=-1)

        # store raw, normalized
        setattr(self, which.lower(), self._as_float01(img))
        # a copy: the push edits FILTER, and a view's header must not change under it
        try:
            hdr = _ppp_as_header(header)
            hdr = hdr.copy() if hdr is not None else None
        except Exception:
            hdr = None
        self._src_meta[which] = (hdr, tuple(img.shape[:2]))
        self._set_status_label(which, label)
        self.status.setText(self.tr("{0} loaded ({1}) shape={2}").format(which, self.tr("mono") if img.ndim==2 else "RGB", img.shape))

        # build/clear stretched cache for this input
        self._cache_stretch(which)

        if self.current_palette is None:
            self.current_palette = "SHO"

    def _load_from_view(self, which):
        views = self._list_open_views()
        if not views:
            QMessageBox.warning(self, self.tr("No Views"), self.tr("No open image views were found."))
            return None

        labels = [lab for lab, _ in views]
        choice, ok = QInputDialog.getItem(
            self, self.tr("Select View for {0}").format(which), self.tr("Choose a view (by name):"), labels, 0, False
        )
        if not ok or not choice:
            return None

        sw = dict(views)[choice]
        vw = sw.widget()
        doc = getattr(vw, "document", None)

        if doc is None or getattr(doc, "image", None) is None:
            QMessageBox.warning(self, self.tr("Empty View"), self.tr("Selected view has no image."))
            return None

        img = doc.image
        meta = getattr(doc, "metadata", {}) or {}
        header = meta.get("original_header", None)
        bit_depth = meta.get("bit_depth", "Unknown")
        is_mono = (img.ndim == 2) or (img.ndim == 3 and img.shape[2] == 1)
        path = meta.get("file_path", None)
        return img, header, bit_depth, is_mono, path, self.tr("From View: {0}").format(choice)

    def _load_from_file(self, which):
        filt = "Images (*.png *.tif *.tiff *.fits *.fit *.xisf)"
        path, _ = QFileDialog.getOpenFileName(self, self.tr("Select {0} File").format(which), "", filt)
        if not path:
            return None
        img, header, bit_depth, is_mono = legacy_load_image(path)
        if img is None:
            QMessageBox.critical(self, self.tr("Load Error"), self.tr("Could not load {0}").format(os.path.basename(path)))
            return None
        label = self.tr("From File: {0}").format(os.path.basename(path))
        return img, header, bit_depth, is_mono, path, label

    def showEvent(self, e):
        super().showEvent(e)
        if self._geom_restored:
            return
        self._geom_restored = True

        def _after():
            self._restore_ui_state()
            # Open the palette gallery alongside the main window,
            # honoring last session's visibility (default: open).
            if getattr(self, "_gallery", None) is not None:
                want_visible = bool(self._settings.value(
                    self._k("gallery_visible"), True, type=bool))
                self.btn_gallery.blockSignals(True)
                self.btn_gallery.setChecked(want_visible)
                self.btn_gallery.blockSignals(False)
                if want_visible:
                    self._gallery.show()
                    self._gallery.raise_()
            # center scrollbars if nothing loaded yet
            if self._base_pm is None:
                self._center_scrollbars()
        QTimer.singleShot(0, _after)

    def _on_app_quit(self):
        try:
            self._save_ui_state()
        except Exception:
            pass
        try:
            self._release_memory()
        except Exception:
            pass
        self.close()

    def closeEvent(self, e):
        try:
            self._save_ui_state()
        except Exception:
            pass
        try:
            self._release_memory()
        except Exception:
            pass
        super().closeEvent(e)

    def _release_memory(self):
        """Drop channel arrays, caches, and Qt pixmaps so RAM can be reclaimed.

        WA_DeleteOnClose alone is not enough: full-size float32 channels and the
        stretched cache live as Python attributes and stay allocated until we
        clear them (and until any aboutToQuit connection stops retaining us).
        """
        if getattr(self, "_memory_released", False):
            return
        self._memory_released = True

        # Break the QApplication → bound-method → self retention path.
        try:
            from PyQt6.QtWidgets import QApplication
            app = QApplication.instance()
            if app is not None:
                try:
                    app.aboutToQuit.disconnect(self._on_app_quit)
                except (TypeError, RuntimeError):
                    pass
        except Exception:
            pass

        # Close any still-open adjust dialogs (they hold copies of the RGB).
        try:
            for child in self.findChildren(PaletteAdjustDialog):
                try:
                    child.close()
                except Exception:
                    pass
        except Exception:
            pass

        # Tear down the gallery companion window.
        gal = getattr(self, "_gallery", None)
        if gal is not None:
            try:
                gal.hide()
                gal.deleteLater()
            except Exception:
                pass
            self._gallery = None

        # Large numpy buffers (results of jobs still running are dropped on arrival)
        self._queue = []
        self._clear_serial += 1
        self._final_key = None
        self._base_pm_src = None
        self.ha = self.oiii = self.sii = self.osc1 = self.osc2 = None
        self._src_meta = {}
        try:
            self._stretched.clear()
        except Exception:
            self._stretched = {}
        self.final = None
        self.doc_manager = None

        # Preview / thumbnail pixmaps
        self._base_pm = None
        try:
            self._thumb_base_pm.clear()
        except Exception:
            self._thumb_base_pm = {}
        self._selected_name = None
        try:
            self.preview.clear()
        except Exception:
            pass
        for b in list(getattr(self, "_thumb_buttons", {}).values()):
            try:
                b.setIcon(QIcon())
            except Exception:
                pass

        popup = getattr(self, "_last_popup_viewer", None)
        if popup is not None:
            try:
                popup.close()
                popup.deleteLater()
            except Exception:
                pass
            self._last_popup_viewer = None

        # Encourage the allocator to give pages back (esp. Linux/glibc).
        try:
            from PyQt6.QtGui import QPixmapCache
            QPixmapCache.clear()
        except Exception:
            pass
        try:
            import gc
            gc.collect()
        except Exception:
            pass
        try:
            import sys
            if sys.platform.startswith("linux"):
                import ctypes
                ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass

    # ------------- build/caches -------------
    def _cache_stretch(self, which: str):
        """Compute and cache stretched version of a just-loaded input (if linear checked)."""
        self._inputs_rev += 1
        arr = getattr(self, which.lower())
        if arr is None:
            self._stretched.pop(which, None); return
        if not self.chk_linear.isChecked():
            self._stretched.pop(which, None); return
        self._stretched[which] = self._stretch_input(arr)

    def _rebuild_stretch_cache_for_all(self, _state: int):
        """Rebuild (or clear) stretched cache for all loaded inputs when checkbox toggles."""
        for which in ("Ha","OIII","SII","OSC1","OSC2"):
            self._cache_stretch(which)

    def _render_thumb(self, name: str):
        base = self._thumb_base_pm.get(name)
        if base is None:
            return
        pm = base.copy()

        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        font = QFont("Helvetica", 10, QFont.Weight.DemiBold)
        p.setFont(font)
        fm = QFontMetrics(font)

        pad = 6
        strip_h = fm.height() + pad * 2
        strip = pm.rect().adjusted(0, pm.height() - strip_h, 0, 0)

        # translucent bottom strip
        p.fillRect(strip, QColor(0, 0, 0, 160))
        color = QColor(102, 255, 102) if self._selected_name == name else QColor(255, 255, 255)
        p.setPen(QPen(color))
        p.drawText(strip, Qt.AlignmentFlag.AlignCenter, name)
        p.end()

        btn = self._thumb_buttons[name]
        btn.setIcon(QIcon(pm))
        btn.setIconSize(self.thumb_size)  # <- ensures no clipping

    # ------------- thumbnails -------------
    def _create_palettes(self):
        """
        Build the palette thumbnails from the stretched inputs (off the GUI thread)
        and draw the palette name directly on each thumbnail. Names turn green when selected.
        """
        picks = self._picked_inputs()
        if _ppp_present_count(*picks) < 2:
            QMessageBox.warning(self, self.tr("Need Channels"),
                self.tr("Load any two of Ha / OIII / SII, or one OSC image."))
            return
        if self._thumbs_thread is not None:
            self._thumbs_pending = True      # rebuilt from the current inputs when it lands
            return
        thr = _PPPJob(_ppp_thumbs_job, self._map_channels_or_special, list(self.PALETTES), picks,
                      self.thumb_size.width(), self.thumb_size.height())
        thr.result.connect(self._on_thumbs_built)
        self._thumbs_thread = thr
        self._thumbs_serial = self._clear_serial
        self.status.setText(self.tr("Creating palette previews\u2026"))
        start_owned_thread(thr)

    def _on_thumbs_built(self, out):
        self._thumbs_thread = None
        if getattr(self, "_memory_released", False):
            return
        if self._thumbs_pending:
            self._thumbs_pending = False
            self._create_palettes()
            return
        if self._thumbs_serial != self._clear_serial:
            return      # channels were cleared meanwhile
        kind, val = out
        if kind == "err":
            self.status.setText(self.tr("Could not create palette previews: {0}").format(val))
            return

        built = 0
        for name in self.PALETTES:
            qimg = val.get(name)
            if qimg is None:
                self._thumb_base_pm.pop(name, None)
                self._thumb_buttons[name].setIcon(QIcon())
                continue

            # scale the thumbnail to EXACTLY the button icon size first
            pm = QPixmap.fromImage(qimg).scaled(
                self.thumb_size, Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self._thumb_base_pm[name] = pm
            self._render_thumb(name)
            built += 1

        self.status.setText(self.tr("Created {0} palette previews.").format(built))


    def _on_palette_clicked(self, name: str):
        self._selected_name = name
        for n in self.PALETTES:
            self._render_thumb(n)
        if self._busy():
            self._enqueue(("gen", name))     # built after the running job, in click order
            return
        self.current_palette = name
        self._generate_for_palette(name)

    # ------------- job queue -------------
    def _busy(self) -> bool:
        return self._gen_thread is not None or self._push_thread is not None

    def _enqueue(self, item):
        # consecutive palette clicks collapse to the latest one
        if item[0] == "gen" and self._queue and self._queue[-1][0] == "gen":
            self._queue[-1] = item
        else:
            self._queue.append(item)

    def _drain_queue(self):
        """Run clicks queued during a palette build / push, in order, once nothing runs."""
        while self._queue and not self._busy():
            if getattr(self, "_memory_released", False):
                self._queue = []
                return
            kind, arg = self._queue.pop(0)
            if kind == "gen":
                self.current_palette = arg
                self._generate_for_palette(arg)
            elif kind == "push":
                self._push_final()
            elif kind == "clear":
                self._clear_channels()

    def _picked_inputs(self):
        """(Ha, OIII, SII, OSC1, OSC2) as palette inputs: the stretched cache when Linear is on."""
        lin = self.chk_linear.isChecked()

        def pick(name):
            if lin and (name in self._stretched):
                return self._stretched[name]
            return getattr(self, name.lower())
        return tuple(pick(n) for n in ("Ha", "OIII", "SII", "OSC1", "OSC2"))

    def _palette_key(self, pal, picks):
        """Same key => same prepared channels and palette (inputs untouched in between)."""
        return (self._inputs_rev, bool(self.chk_linear.isChecked()),
                tuple(id(x) for x in picks), pal)

    def _confirm_channel_sizes(self, picks) -> bool:
        """Ask once per session before warping channels of different sizes to match."""
        sh_ha, sh_oo, sh_si = _ppp_synth_shapes(*picks)
        shapes = [x for x in (sh_ha, sh_oo, sh_si) if x is not None]
        if not (len(shapes) and len(set(shapes)) > 1) or self._dim_mismatch_accepted:
            return True
        ref_name = "Ha" if sh_ha is not None else ("OIII" if sh_oo is not None else "SII")
        ref_h, ref_w = sh_ha or sh_oo or sh_si
        msg = self.tr(
            "The loaded channels have different image dimensions.\n\n"
            "\u2022 Ha:   {0}\n"
            "\u2022 OIII: {1}\n"
            "\u2022 SII:  {2}\n\n"
            "SASpro can resize (warp) the channels to match the reference frame:\n"
            "\u2022 Reference: {3}\n"
            "\u2022 Target size: ({4} \u00d7 {5})\n\n"
            "Proceed and resize mismatched channels?"
        ).format(sh_ha, sh_oo, sh_si, ref_name, ref_w, ref_h)
        ret = QMessageBox.question(
            self,
            self.tr("Channel Size Mismatch"),
            msg,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes
        )
        if ret != QMessageBox.StandardButton.Yes:
            return False
        self._dim_mismatch_accepted = True
        return True

    # ------------- palette build helpers -------------
    def _center_crop(self, img: np.ndarray, side: int) -> np.ndarray:
        """Center-crop to a square of size 'side' (no upscaling)."""
        h, w = img.shape[:2]; s = min(side, h, w)
        y0 = (h - s) // 2; x0 = (w - s) // 2
        return img[y0:y0+s, x0:x0+s] if img.ndim == 2 else img[y0:y0+s, x0:x0+s, :]

    def _center_crop_all_to_side(self, side: int, *imgs):
        """Center-crop all provided images to the same square side (no upscaling)."""
        s = None
        for im in imgs:
            if im is None: continue
            h, w = im.shape[:2]
            s = min(side, h, w) if s is None else min(s, h, w, side)
        if s is None: s = side
        return [self._center_crop(im, s) if im is not None else None for im in imgs], s

    def _prepared_channels(self, for_thumbs: bool = False):
        """
        Build Ha/OIII/SII bases from inputs. If 'Linear input' is checked,
        **use stretched versions** (cached). for_thumbs: thumbnail-sized channels
        (see _ppp_thumb_channels).
        """
        picks = self._picked_inputs()
        if for_thumbs:
            return _ppp_thumb_channels(*picks, self.thumb_size.width(), self.thumb_size.height())
        if not self._confirm_channel_sizes(picks):
            return None, None, None
        return _ppp_match_sizes(*_ppp_synth(*picks))

    def _generate_for_palette(self, pal: str):
        if self._busy():
            self._enqueue(("gen", pal))
            return
        picks = self._picked_inputs()
        if _ppp_present_count(*picks) < 2:
            return
        if not self._confirm_channel_sizes(picks):
            return
        thr = _PPPJob(_ppp_gen_job, self._map_channels_or_special, pal, picks)
        thr.result.connect(self._on_final_built)
        self._gen_thread = thr
        self._gen_req = (pal, self._palette_key(pal, picks), self._clear_serial)
        self.status.setText(self.tr("Building {0}\u2026").format(pal))
        start_owned_thread(thr)

    def _on_final_built(self, out):
        self._gen_thread = None
        if getattr(self, "_memory_released", False):
            return
        pal, key, serial = self._gen_req
        self._gen_req = None
        if serial == self._clear_serial:
            kind, val = out
            if kind == "err":
                QMessageBox.critical(self, self.tr("Palette Error"), self.tr("Could not build palette {0}.\n\n{1}").format(pal, val))
            elif isinstance(val, str):
                QMessageBox.critical(self, self.tr("Palette Error"), self.tr("Could not build palette {0}.").format(pal))
            elif val is not None:
                rgb, in_range, qimg = val
                self.final = rgb
                self._final_key = key if in_range else None
                self._final_shared = False
                # Fit only when there wasn't an existing preview yet
                first = (self._base_pm is None)
                self._set_preview_image(qimg, fit=first, preserve_view=True, src=rgb)
                self.status.setText(self.tr("Preview generated: {0}").format(pal))
        self._drain_queue()

    def _set_preview_image(self, qimg, *, fit: bool = False, preserve_view: bool = True, src=None):
        """qimg: QImage or QPixmap. src: the array it shows (lets Push reuse the pixmap)."""
        state = None
        if preserve_view and (not fit) and (self._base_pm is not None):
            state = self._capture_view_state()

        self._base_pm = qimg if isinstance(qimg, QPixmap) else QPixmap.fromImage(qimg)
        self._base_pm_src = src

        # If we’re fitting, ignore old zoom/pan.
        if fit or state is None:
            self._zoom = 1.0
            self._update_preview_pixmap()
            if fit:
                QTimer.singleShot(0, self._fit_to_preview)
            else:
                QTimer.singleShot(0, self._center_scrollbars)
            return

        # restore prior zoom/pan
        self._restore_view_state(state)


    def _update_preview_pixmap(self):
        if self._base_pm is None:
            return
        # explicit int size (QSize * float can crash on some PyQt6 builds)
        base_sz = self._base_pm.size()
        w = max(1, int(base_sz.width() * self._zoom))
        h = max(1, int(base_sz.height() * self._zoom))
        scaled = self._base_pm.scaled(
            w, h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.preview.setPixmap(scaled)
        self.preview.resize(scaled.size())

    def _set_zoom(self, new_zoom: float):
        self._zoom = max(self._min_zoom, min(self._max_zoom, new_zoom))
        self._update_preview_pixmap()

    def _zoom_at(self, factor: float = 1.25, anchor_vp: QPoint | None = None):
        if self._base_pm is None:
            return

        vp = self.scroll.viewport()
        if anchor_vp is None:
            anchor_vp = QPoint(vp.width() // 2, vp.height() // 2)  # view center

        # label coords under the anchor *before* zoom
        lbl_before = self.preview.mapFrom(vp, anchor_vp)

        old_zoom = self._zoom
        new_zoom = max(self._min_zoom, min(self._max_zoom, old_zoom * factor))
        ratio = new_zoom / max(old_zoom, 1e-6)
        if abs(ratio - 1.0) < 1e-6:
            return

        # apply zoom (updates label size & scrollbar ranges)
        self._zoom = new_zoom
        self._update_preview_pixmap()

        # desired label coords *after* zoom
        lbl_after_x = int(lbl_before.x() * ratio)
        lbl_after_y = int(lbl_before.y() * ratio)

        # move scrollbars so anchor_vp keeps the same content point
        hbar = self.scroll.horizontalScrollBar()
        vbar = self.scroll.verticalScrollBar()
        hbar.setValue(max(hbar.minimum(), min(hbar.maximum(), lbl_after_x - anchor_vp.x())))
        vbar.setValue(max(vbar.minimum(), min(vbar.maximum(), lbl_after_y - anchor_vp.y())))


    def _fit_to_preview(self):
        if self._base_pm is None:
            return
        vp = self.scroll.viewport().size()
        pm = self._base_pm.size()
        if pm.width() == 0 or pm.height() == 0:
            return
        k = min(vp.width() / pm.width(), vp.height() / pm.height())
        self._set_zoom(max(self._min_zoom, min(self._max_zoom, k)))
        self._center_scrollbars()

    def _center_scrollbars(self):
        # center the view on the image
        h = self.scroll.horizontalScrollBar()
        v = self.scroll.verticalScrollBar()
        h.setValue((h.maximum() + h.minimum()) // 2)
        v.setValue((v.maximum() + v.minimum()) // 2)

    def _foraxx_gate(self, x):
        """Foraxx-style dynamic gate: x^(1-x), clipped to [0,1] domain.
        Rises 0→1 across the input but stays high through the midrange,
        so blends lean toward the 'present' signal without hard thresholds."""
        x = np.clip(np.nan_to_num(x), 1e-6, 1.0)
        return x ** (1.0 - x)

    def _map_channels_or_special(self, name, ha, oo, si):
        # Record TRUE presence before substitution masks it, so Foraxx can
        # pick real-tricolor vs bicolor correctly.
        has_real_si = si is not None
        has_real_ha = ha is not None

        # substitution — fill the missing channel from the other two (or clone
        # the one available).  Any two of {Ha, OIII, SII} are enough.
        if ha is None and si is not None: ha = si
        if si is None and ha is not None: si = ha
        if oo is None:
            if ha is not None: oo = ha
            elif si is not None: oo = si

        basic = {
            "SHO": (si, ha, oo),
            "HOO": (ha, oo, oo),
            "HSO": (ha, si, oo),
            "HOS": (ha, oo, si),
            "OSS": (oo, si, si),
            "OHH": (oo, ha, ha),
            "OSH": (oo, si, ha),
            "OHS": (oo, ha, si),
            "HSS": (ha, si, si),
            "SOH": (si, oo, ha),   # completes the 6 full permutations
            "SHH": (si, ha, ha),   # SII-red bicolor
            "SOO": (si, oo, oo),   # SII-red / OIII bicolor
            "HHO": (ha, ha, oo),   # Ha-forward warm bicolor
            "OOS": (oo, oo, si),   # OIII-forward cool bicolor            
        }
        if name in basic:
            return basic[name]

        try:
            if name == "Realistic1":
                r = (ha + si)/2 if (ha is not None and si is not None) else (ha if ha is not None else 0)
                g = 0.3*(ha if ha is not None else 0) + 0.7*(oo if oo is not None else 0)
                b = 0.9*(oo if oo is not None else 0) + 0.1*(ha if ha is not None else 0)
                return r,g,b
            if name == "Realistic2":
                r = 0.7*(ha if ha is not None else 0) + 0.3*(si if si is not None else 0)
                g = 0.3*(si if si is not None else 0) + 0.7*(oo if oo is not None else 0)
                b = (oo if oo is not None else 0)
                return r,g,b
            if name == "Foraxx":
                # Bicolor Foraxx: Ha + OIII only (no real SII)
                if has_real_ha and oo is not None and not has_real_si:
                    r = ha
                    b = oo
                    t = self._foraxx_gate(ha * oo)
                    g = t * ha + (1.0 - t) * oo
                    return r, g, b
                # Tricolor Foraxx: real Ha + OIII + SII
                if has_real_si and oo is not None and (ha is not None):
                    t  = self._foraxx_gate(oo)
                    r  = t * si + (1.0 - t) * ha
                    t2 = self._foraxx_gate(ha * oo)
                    g  = t2 * ha + (1.0 - t2) * oo
                    b  = oo
                    return r, g, b
                return basic["SHO"]
            if name == "Dynamic Inverse":
                S = np.clip(np.nan_to_num(ha), 0.0, 1.0)
                O = np.clip(np.nan_to_num(oo), 0.0, 1.0)
                H = np.clip(np.nan_to_num(si), 0.0, 1.0)

                # Red: Ha in OIII-voids, SII in OIII-rich zones.
                tO = 1.0 - O
                r  = tO * H + (1.0 - tO) * S

                # Green: OIII fills the Ha gaps (gated by Ha absence).
                tH = self._foraxx_gate(1.0 - H)
                g  = (1.0 - tH) * H + tH * O

                # Blue: OIII, biased into Ha-free regions.
                b  = tH * O + (1.0 - tH) * (0.5 * O)

                r = np.clip(r, 0.0, 1.0)
                g = np.clip(g, 0.0, 1.0)
                b = np.clip(b, 0.0, 1.0)
                return r, g, b
        except Exception:
            return basic.get("SHO", (ha, oo, si))

        return basic.get("SHO", (ha, oo, si))

    def _toggle_gallery(self, on: bool):
        if on:
            self._gallery.show()
            self._gallery.raise_()
        else:
            self._gallery.hide()

    # ------------- output-linear (invert stretch on push) -------------
    def _prepared_channels_raw(self):
        """
        Same OSC synthesis as _prepared_channels, but ALWAYS from raw (unstretched)
        inputs. Used to recover the original linear channel medians so we can
        invert the pre-build stretch at push time.
        """
        return _ppp_synth(self.ha, self.oiii, self.sii, self.osc1, self.osc2)

    def _channel_medians_raw(self):
        """
        Returns {name: median} for channels that look like linear data.
        Values that aren't plausibly linear (median >= 0.5 or <= 1e-6) are dropped
        — those were almost certainly pre-stretched and would produce a bad target.
        """
        return _ppp_channel_medians(*self._prepared_channels_raw())

    def _target_linear_median(self) -> float | None:
        """
        Resolve the user's dropdown choice to a single target median value.
        Falls back to Max if the requested named channel is missing.
        """
        meds = self._channel_medians_raw()
        if not meds:
            return None

        mode = self.cmb_target_median.currentData() if hasattr(self, "cmb_target_median") \
               else "Max (least stretched)"
        return _ppp_target_median(meds, mode)

    def _apply_output_linear(self, rgb: np.ndarray) -> np.ndarray:
        """
        Invert the pre-build statistical stretch by running a LINKED no-black-clip
        statistical stretch on the finished palette with target_median set to the
        chosen raw channel median. This works for every palette — including
        Foraxx and Dynamic Inverse — because we're inverting the common lift
        applied to all three channels before mixing.
        """
        req = self._output_linear_request()
        if req is None:
            return rgb
        out, info = _ppp_output_linear(rgb, *req)
        self._report_output_linear(info)
        return out

    def _output_linear_request(self):
        """GUI-thread snapshot for _ppp_output_linear, or None when Output linear is off."""
        if not (self.chk_linear.isChecked() and self.chk_output_linear.isChecked()):
            return None
        mode = self.cmb_target_median.currentData() if hasattr(self, "cmb_target_median") \
               else "Max (least stretched)"
        return mode, (self.ha, self.oiii, self.sii, self.osc1, self.osc2)

    def _report_output_linear(self, info):
        kind, val, mode = info
        if kind == "none":
            self.status.setText(self.tr(
                "Output linear: no plausibly-linear channel medians found; "
                "pushed stretched instead."
            ))
        elif kind == "ok":
            _i = self.cmb_target_median.findData(mode)
            self.status.setText(self.tr(
                "Output linear: inverted stretch to target median {0:.4f} "
                "({1})."
            ).format(val, self.cmb_target_median.itemText(_i) if _i >= 0 else mode))
        else:
            QMessageBox.warning(
                self, self.tr("Output Linear Failed"),
                self.tr("Could not invert stretch:\n{0}\n\nPushing stretched image instead.").format(val)
            )

    # ------------- push to new subwindow -------------
    def _get_doc_manager(self):
        """
        Try several ways to get a DocManager:
        1) explicit doc_manager passed into PerfectPalettePicker
        2) main window's .docman or .doc_manager attribute
        """
        if self.doc_manager is not None:
            return self.doc_manager

        mw = self._find_main_window()
        if mw is None:
            return None

        return getattr(mw, "docman", None) or getattr(mw, "doc_manager", None)

    def _push_final(self):
        if self._busy():
            # A palette build or an earlier push is still running: push after it (queued).
            self._enqueue(("push", None))
            self.status.setText(self.tr("Push queued \u2014 it runs when the current palette job finishes."))
            return
        if self.final is None:
            QMessageBox.warning(self, self.tr("No Image"), self.tr("Generate a palette first."))
            return

        pal = self.current_palette or "SHO"
        picks = self._picked_inputs()
        key = self._palette_key(pal, picks)
        # self.final is bit-identical to what the balancing dialog returns when it is
        # accepted untouched (checked when the palette was built, inputs unchanged since).
        untouched = self._final_key is not None and self._final_key == key

        if not self.chk_balance_on_push.isChecked():
            # No balancing dialog: push exactly what accepting it untouched would push.
            if untouched:
                self._start_push(self.final, key=key)
                return
            if _ppp_present_count(*picks) < 2:
                QMessageBox.warning(self, self.tr("Need Channels"),
                    self.tr("Load any two of Ha / OIII / SII, or one OSC image."))
                return
            if not self._confirm_channel_sizes(picks):
                return
            # Build the dialog's untouched image with the dialog's own function.
            self._start_push(None, key=key,
                             compose=(self._map_channels_or_special, pal, picks, self.final.shape))
            return

        # Use the SAME prepared channels the palette was built with
        ha_prep, oo_prep, si_prep = self._prepared_channels()
        has_narrowband = sum(x is not None for x in (ha_prep, oo_prep, si_prep))
        if has_narrowband < 2:
            QMessageBox.warning(self, self.tr("Need Channels"),
                self.tr("Load any two of Ha / OIII / SII, or one OSC image."))
            return

        dlg = PaletteAdjustDialog(
            base_rgb     = self.final,                           # fully formed palette
            palette_name = pal,
            ha_src       = ha_prep,                              # prepared (stretched/OSC-synth)
            oiii_src     = oo_prep,
            sii_src      = si_prep,
            owner        = self,
            orig_pixmap  = self._base_pm if self._base_pm_src is self.final else None,
            base_is_untouched = untouched,
        )
        adjusted = {"img": None}
        dlg.adjusted_image.connect(lambda img: adjusted.__setitem__("img", img))
        dlg.exec()

        if adjusted["img"] is None:
            return  # user canceled
        img = adjusted["img"]
        # Untouched accept hands back self.final itself; anything else is a new image.
        self._start_push(img, key=key if img is self.final else None,
                         pm=getattr(dlg, "accepted_pixmap", None))

    def _start_push(self, img, *, key, pm=None, compose=None):
        """Finish a push: build/copy/invert off the GUI thread when needed, then open the view."""
        olin = self._output_linear_request()
        sources = dict(self._src_meta)       # the inputs this push is built from
        # two documents must never share one array
        needs_copy = img is not None and img is self.final and self._final_shared
        if compose is None and olin is None and not needs_copy:
            self._finish_push(img, img, pm, None, key, False, False, sources)
            return
        thr = _PPPJob(_ppp_push_job, img, needs_copy, compose, olin)
        thr.result.connect(self._on_push_done)
        self._push_thread = thr
        self._push_req = (key, pm, olin is not None, sources)
        self.status.setText(self.tr("Pushing palette to a new view\u2026"))
        start_owned_thread(thr)

    def _on_push_done(self, res):
        self._push_thread = None
        if getattr(self, "_memory_released", False):
            return
        key, pm, olin_on, sources = self._push_req
        self._push_req = None
        kind, val = res
        if kind == "err":
            QMessageBox.critical(self, self.tr("Error"), self.tr("Failed to open new view:\n{0}").format(val))
        else:
            final, out, qimg, info, copied = val
            self._finish_push(final, out, pm if pm is not None else qimg, info, key, olin_on, copied,
                              sources)
        self._drain_queue()

    def _finish_push(self, final, output_img, preview, info, key, olin_on, copied, sources=None):
        # Keep the stretched result as our on-screen "final" so the preview
        # stays visually meaningful; the linear invert (if requested) happens
        # only on the array we hand to the DocManager.
        if final is not self.final:
            self.final = final
            self._final_key = key
            self._final_shared = False
            if copied:
                self._base_pm_src = final        # same pixels as the preview already shows
            else:
                self._set_preview_image(preview if preview is not None
                                        else _ppp_display_qimage(final), src=final)

        # output_img is the inverse-stretched array when Output linear is on,
        # otherwise the stretched final the user is looking at.
        if info is not None:
            self._report_output_linear(info)

        title = self.current_palette or "Palette"
        if olin_on:
            title = f"{title} (linear)"

        # ---- get DocManager the robust way ----
        dm = self._get_doc_manager()

        if dm is None:
            # Fallback: open a simple viewer instead of erroring out.
            # Show the stretched version so the user can actually see it;
            # the linear array wouldn't display meaningfully in a bare QLabel.
            viewer = QDialog(self)
            viewer.setWindowTitle(title)
            vlayout = QVBoxLayout(viewer)
            lbl = QLabel()
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setPixmap(QPixmap.fromImage(self._to_qimage(self.final)))
            vlayout.addWidget(lbl)
            viewer.resize(lbl.pixmap().size())
            viewer.show()
            # keep ref so it isn't GC'd
            self._last_popup_viewer = viewer
            self.status.setText(self.tr("DocManager not found; opened palette in stand-alone viewer."))
            return

        # ---- normal SAS path: create a new document ----
        try:
            # the inputs' headers merged, FILTER = palette name
            meta = _ppp_push_metadata(sources or {}, output_img,
                                      self.current_palette or "Palette", olin_on)
            if hasattr(dm, "open_array"):
                # many of your tools already use this signature
                doc = dm.open_array(output_img, metadata=meta, title=title)
            elif hasattr(dm, "create_document"):
                doc = dm.create_document(image=output_img, metadata=meta, name=title)
            else:
                raise RuntimeError("DocManager lacks open_array/create_document")

            # If DocManager or main window auto-spawns subwindows on new docs,
            # this is all we need. If not, you can optionally keep the
            # _spawn_subwindow_for hook here.
            if output_img is self.final:
                self._final_shared = True
            if olin_on:
                self.status.setText(self.tr("Opened final palette (linear) in a new view."))
            else:
                self.status.setText(self.tr("Opened final palette in a new view."))
        except Exception as e:
            QMessageBox.critical(self, self.tr("Error"), self.tr("Failed to open new view:\n{0}").format(e))



    # ------------- utilities -------------
    def _clear_channels(self):
        if self._busy():
            self._enqueue(("clear", None))   # after the running palette job, in click order
            return
        self._inputs_rev += 1
        self._clear_serial += 1
        self._final_key = None
        self._final_shared = False
        self._base_pm_src = None
        self.ha = self.oiii = self.sii = self.osc1 = self.osc2 = None
        self._src_meta = {}
        self._stretched.clear()
        self._dim_mismatch_accepted = False
        self.final = None
        self._base_pm = None
        self.preview.clear()
        for which in ("Ha","OIII","SII","OSC1","OSC2"):
            self._set_status_label(which, None)
        self._thumb_base_pm.clear()
        self._selected_name = None
        for b in self._thumb_buttons.values():
            b.setIcon(QIcon())
        self.status.setText(self.tr("Cleared all loaded channels."))

    def _as_float01(self, arr):
        a = np.asarray(arr)
        if a.dtype == np.uint8:  return a.astype(np.float32)/255.0
        if a.dtype == np.uint16: return a.astype(np.float32)/65535.0
        return np.clip(a.astype(np.float32), 0.0, 1.0)

    def _stretch_input(self, img):
        """Run statistical stretch on mono or color inputs (target_median=0.25)."""
        if img.ndim == 2:
            return np.clip(stretch_mono_image(img, target_median=0.25), 0.0, 1.0)
        if img.ndim == 3 and img.shape[2] == 3:
            return np.clip(stretch_color_image(img, target_median=0.25, linked=False), 0.0, 1.0)
        if img.ndim == 3 and img.shape[2] == 1:
            mono = img[...,0]
            return np.clip(stretch_mono_image(mono, target_median=0.25), 0.0, 1.0)
        return img

    def _to_qimage(self, arr):
        return _ppp_to_qimage(arr)

    def _find_main_window(self):
        from PyQt6.QtWidgets import QMainWindow, QApplication

        # 1) walk parents first
        w = self
        while w is not None:
            if isinstance(w, QMainWindow):
                return w
            w = w.parentWidget()

        app = QApplication.instance()
        if app is None:
            return None

        # 2) prefer active window if it's a QMainWindow
        aw = app.activeWindow()
        if isinstance(aw, QMainWindow):
            return aw

        # 3) prefer one that has an mdi attribute (your real main window does)
        for tlw in app.topLevelWidgets():
            if isinstance(tlw, QMainWindow) and hasattr(tlw, "mdi"):
                return tlw

        # 4) last resort: any QMainWindow
        for tlw in app.topLevelWidgets():
            if isinstance(tlw, QMainWindow):
                return tlw

        return None

    def _list_open_views(self):
        mw = self._find_main_window()
        if mw is None:
            return []

        # Import ImageSubWindow (same class you showed)
        try:
            from setiastro.saspro.subwindow import ImageSubWindow
        except Exception:
            ImageSubWindow = None

        out = []

        # ── 1) Best source: MDI subWindowList() ─────────────────────────────
        mdi = getattr(mw, "mdi", None)
        if mdi is not None:
            try:
                for sub in mdi.subWindowList():
                    try:
                        view = sub.widget()  # should be ImageSubWindow
                        if ImageSubWindow is not None and not isinstance(view, ImageSubWindow):
                            continue

                        doc = getattr(view, "document", None)
                        img = getattr(doc, "image", None) if doc is not None else None
                        if img is None:
                            continue

                        # Clean, user-visible title (your view emits clean core titles already)
                        title = ""
                        try:
                            title = (sub.windowTitle() or "").strip()
                        except Exception:
                            title = ""
                        if not title:
                            try:
                                title = (view._effective_title() or "").strip()
                            except Exception:
                                title = ""
                        if not title:
                            try:
                                title = (doc.display_name() or "").strip()
                            except Exception:
                                title = "Untitled"

                        out.append((title, sub))
                    except Exception:
                        continue
            except Exception:
                pass

        # ── 2) Fallback: your registry (VERY reliable) ──────────────────────
        if not out and ImageSubWindow is not None:
            try:
                # registry holds ImageSubWindow widgets
                for view in list(ImageSubWindow._registry.values()):
                    try:
                        doc = getattr(view, "document", None)
                        img = getattr(doc, "image", None) if doc is not None else None
                        if img is None:
                            continue

                        sub = None
                        try:
                            sub = view._mdi_subwindow()
                        except Exception:
                            sub = None
                        if sub is None:
                            continue

                        title = ""
                        try:
                            title = (sub.windowTitle() or "").strip()
                        except Exception:
                            title = ""
                        if not title:
                            try:
                                title = (view._effective_title() or "").strip()
                            except Exception:
                                title = ""
                        if not title:
                            try:
                                title = (doc.display_name() or "").strip()
                            except Exception:
                                title = "Untitled"

                        out.append((title, sub))
                    except Exception:
                        continue
            except Exception:
                pass

        # De-dupe titles (keep stable order; disambiguate duplicates)
        seen = set()
        uniq = []
        for t, sub in out:
            tt = str(t)
            if tt in seen:
                i = 2
                cand = f"{tt} ({i})"
                while cand in seen:
                    i += 1
                    cand = f"{tt} ({i})"
                tt = cand
            seen.add(tt)
            uniq.append((tt, sub))

        return uniq

    
    def eventFilter(self, obj, ev):
        # Ctrl+wheel = zoom at mouse (no scrolling). Wheel without Ctrl = eaten.
        # Wheel = zoom at mouse (plain wheel + Ctrl+wheel). No scrolling.
        if ev.type() == QEvent.Type.Wheel and (
            obj is self.preview
            or obj is self.scroll
            or obj is self.scroll.viewport()
            or obj is self.scroll.horizontalScrollBar()
            or obj is self.scroll.verticalScrollBar()
        ):
            # always stop the wheel from scrolling
            ev.accept()

            # Get mouse position in global screen coords and map into the viewport
            vp = self.scroll.viewport()
            anchor_vp = vp.mapFromGlobal(ev.globalPosition().toPoint())

            # Clamp to viewport rect (robust if the event originated on scrollbars)
            r = vp.rect()
            if not r.contains(anchor_vp):
                anchor_vp.setX(max(r.left(),  min(r.right(),  anchor_vp.x())))
                anchor_vp.setY(max(r.top(),   min(r.bottom(), anchor_vp.y())))

            dy = ev.pixelDelta().y()

            if dy != 0:
                abs_dy = abs(dy)
                ctrl_down = bool(ev.modifiers() & Qt.KeyboardModifier.ControlModifier)

                if abs_dy <= 3:
                    base_factor = 1.012 if ctrl_down else 1.010
                elif abs_dy <= 10:
                    base_factor = 1.025 if ctrl_down else 1.020
                else:
                    base_factor = 1.040 if ctrl_down else 1.030

                factor = base_factor if dy > 0 else 1.0 / base_factor
            else:
                dy = ev.angleDelta().y()
                if dy == 0:
                    return True

                ctrl_down = bool(ev.modifiers() & Qt.KeyboardModifier.ControlModifier)
                step = 1.25 if ctrl_down else 1.15
                factor = step if dy > 0 else 1.0 / step

            self._zoom_at(factor, anchor_vp)
            return True
        # click-drag pan on viewport
        if obj is self.scroll.viewport():
            if ev.type() == QEvent.Type.MouseButtonPress and ev.button() == Qt.MouseButton.LeftButton:
                self._panning = True
                self._pan_last = ev.position().toPoint()
                self.scroll.viewport().setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))
                return True
            if ev.type() == QEvent.Type.MouseMove and self._panning:
                cur = ev.position().toPoint()
                delta = cur - (self._pan_last or cur)
                self._pan_last = cur
                h = self.scroll.horizontalScrollBar()
                v = self.scroll.verticalScrollBar()
                h.setValue(h.value() - delta.x())
                v.setValue(v.value() - delta.y())
                return True
            if ev.type() == QEvent.Type.MouseButtonRelease and ev.button() == Qt.MouseButton.LeftButton:
                self._panning = False
                self._pan_last = None
                self.scroll.viewport().setCursor(QCursor(Qt.CursorShape.ArrowCursor))
                return True

        return super().eventFilter(obj, ev)