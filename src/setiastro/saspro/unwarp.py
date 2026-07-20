#src/setiastro/saspro/unwarp.py
#!/usr/bin/env python3
# ======================================================
#   _____      __  _ ___         __
#  / ___/___  / /_(_)   |  _____/ /__________
#  \__ \/ _ \/ __/ / /| | / ___/ __/ ___/ __ \
# ___/ /  __/ /_/ / ___ |(__  ) /_/ /  / /_/ /
#/____/\___/\__/_/_/  |_/____/\__/_/   \____/
#  SASpro SIP Unwarp
#
#  unwarp.py  —  Remove SIP distortion from a plate-solved image.
#
#  Given a WCS with a SIP polynomial (up to 4th degree, as produced by the
#  Plate Solver), this resamples the image onto a distortion-free linear TAN
#  grid: every output pixel is projected to the sky through the *linear* WCS,
#  then back into the original frame through the *full* SIP WCS, and sampled
#  there. The canvas is expanded as needed so the corners the distortion
#  pushes outward aren't clipped. The result carries a clean linear WCS with
#  no SIP terms.
#
#  Delivers three ways: Apply in place (undoable), open as a New View, or Save
#  FITS. Also carries a Star-Stretch-style preset drag grip so the current
#  settings can be dropped on the canvas as a shortcut, or on an image to run
#  headlessly.
#
#  Written by Franklin Marek
#  www.setiastro.com
# ======================================================

from __future__ import annotations

import math
import re
from typing import Optional, Tuple, List, Callable, Dict, Any
from concurrent.futures import as_completed

import numpy as np

from PyQt6.QtCore import Qt, QThread, pyqtSignal as _Signal
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout, QLabel,
    QPushButton, QComboBox, QCheckBox, QProgressBar, QFrame, QMessageBox,
    QFileDialog, QWidget, QDialogButtonBox,
)

try:
    from astropy.wcs import WCS
    from astropy.io.fits import Header
    _HAS_ASTROPY = True
except Exception:                       # pragma: no cover
    WCS = None
    Header = None
    _HAS_ASTROPY = False

try:
    from scipy.ndimage import map_coordinates
    _HAS_SCIPY = True
except Exception:                       # pragma: no cover
    map_coordinates = None
    _HAS_SCIPY = False

# Preset drag grip (PI-style "new instance"). Optional so the module still
# imports where the shortcuts package isn't present.
try:
    from setiastro.saspro.shortcuts import PresetDragHandle
    _HAS_PRESET = True
except Exception:                       # pragma: no cover
    PresetDragHandle = None
    _HAS_PRESET = False

try:
    from setiastro.saspro.resources import unwarp_path
except Exception:                       # pragma: no cover
    unwarp_path = None

# Command id used by the shortcuts / replay system for this tool.
UNWARP_COMMAND_ID = "unwarp"
_STEP_NAME = "Unwarp (remove SIP)"


# ══════════════════════════════════════════════════════════════════════════════
#  Document adapters
#
#  The only place this module knows the host app. Mirrors the Apply/New-doc
#  pattern the other SASpro tools use (doc.apply_edit / dm.open_array).
# ══════════════════════════════════════════════════════════════════════════════

def doc_title(doc) -> str:
    for attr in ("display_name", "name", "title"):
        v = getattr(doc, attr, None)
        if callable(v):
            try:
                return str(v())
            except Exception:
                pass
        elif isinstance(v, str):
            return v
    return "Untitled"


def get_doc_metadata(doc) -> dict:
    m = getattr(doc, "metadata", None)
    return m if isinstance(m, dict) else {}


def get_doc_image(doc) -> Optional[np.ndarray]:
    """Best-effort fetch of the document's pixel array (H,W) or (H,W,C)."""
    for attr in ("image", "img", "data", "array"):
        v = getattr(doc, attr, None)
        if isinstance(v, np.ndarray):
            return v
    for meth in ("get_image", "image_data", "numpy", "as_array"):
        f = getattr(doc, meth, None)
        if callable(f):
            try:
                v = f()
                if isinstance(v, np.ndarray):
                    return v
            except Exception:
                pass
    meta = get_doc_metadata(doc)
    for k in ("image", "data"):
        v = meta.get(k)
        if isinstance(v, np.ndarray):
            return v
    return None


def get_doc_header(doc):
    meta = get_doc_metadata(doc)
    for k in ("original_header", "fits_header", "header"):
        h = meta.get(k)
        if h is not None:
            return h
    return None


def extract_wcs(doc):
    """
    Return an astropy WCS for the document, or None.

    Prefers rebuilding from the FITS header with relax=True, because a WCS
    object stored without relax may have silently dropped its SIP terms — and
    SIP is the whole point here. Falls back to a stored WCS object.
    """
    if not _HAS_ASTROPY:
        return None
    hdr = get_doc_header(doc)
    if hdr is not None:
        try:
            w = WCS(hdr, relax=True)
            if w.has_celestial:
                return w.celestial if w.naxis != 2 else w
        except Exception:
            pass
    meta = get_doc_metadata(doc)
    w = meta.get("wcs")
    if w is not None and WCS is not None and isinstance(w, WCS):
        try:
            return w.celestial if w.naxis != 2 else w
        except Exception:
            return w
    return None


def has_sip(wcs) -> bool:
    try:
        return wcs is not None and wcs.sip is not None
    except Exception:
        return False


def sip_order(wcs) -> int:
    try:
        return int(max(wcs.sip.a_order, wcs.sip.b_order))
    except Exception:
        return 0


def pixel_scale_arcsec(wcs) -> float:
    """Mean pixel scale in arcsec/px from the linear part of the WCS."""
    try:
        from astropy.wcs.utils import proj_plane_pixel_scales
        sc = proj_plane_pixel_scales(wcs.celestial if wcs.naxis != 2 else wcs)
        return float(np.mean(sc) * 3600.0)
    except Exception:
        return 0.0


def _find_main_window_from(widget):
    from PyQt6.QtWidgets import QMainWindow, QApplication
    w = widget
    while w is not None and not isinstance(w, QMainWindow):
        w = w.parent() if hasattr(w, "parent") else None
    if w is not None:
        return w
    for tlw in QApplication.topLevelWidgets():
        if isinstance(tlw, QMainWindow):
            return tlw
    return None


def _get_doc_manager_from(main_window):
    if main_window is None:
        return None
    return getattr(main_window, "doc_manager", None) or getattr(main_window, "docman", None)


def _resolve_active_doc(main_window):
    """Active document: the visible MDI subwindow first, then the doc manager."""
    if main_window is None:
        return None
    doc = None
    try:
        sw = main_window.mdi.activeSubWindow()
        if sw is not None:
            w = sw.widget()
            doc = getattr(w, "document", None)
    except Exception:
        doc = None
    if doc is None and hasattr(main_window, "current_document"):
        try:
            doc = main_window.current_document()
        except Exception:
            doc = None
    if doc is None:
        dm = _get_doc_manager_from(main_window)
        if dm is not None:
            doc = (dm.get_active_document() if hasattr(dm, "get_active_document")
                   else getattr(dm, "active_document", None))
    return doc


def apply_edit_to_doc(doc, image, metadata, step_name=_STEP_NAME):
    """Replace a document's pixels + metadata, recording undo history. Mirrors
    the Apply pattern across SASpro (apply_edit → set_image → .image)."""
    img = np.asarray(image, dtype=np.float32)
    if hasattr(doc, "apply_edit"):
        try:
            doc.apply_edit(img.copy(), metadata=metadata, step_name=step_name)
            return
        except TypeError:
            doc.apply_edit(img.copy(), metadata=metadata)
            return
    if hasattr(doc, "set_image"):
        try:
            doc.set_image(img, step_name=step_name)
            return
        except TypeError:
            doc.set_image(img)
            return
    doc.image = img


def create_doc_via_manager(dm, image, metadata, title):
    """Open a result as a new document/view. Mirrors NBN's push path."""
    img = np.asarray(image, dtype=np.float32)
    if hasattr(dm, "open_array"):
        dm.open_array(img, metadata=metadata, title=title)
        return
    if hasattr(dm, "create_document"):
        dm.create_document(image=img, metadata=metadata, name=title)
        return
    raise RuntimeError("DocManager lacks open_array/create_document.")


# ══════════════════════════════════════════════════════════════════════════════
#  Core geometry + resample
# ══════════════════════════════════════════════════════════════════════════════

def linear_wcs(wcs):
    """A copy of `wcs` with all SIP distortion removed (pure linear TAN)."""
    w = wcs.deepcopy()
    w.sip = None
    try:
        w.wcs.ctype = [str(c).replace("-SIP", "") for c in w.wcs.ctype]
    except Exception:
        pass
    return w


def _border_and_grid_pixels(W: int, H: int, n_edge: int = 96, n_grid: int = 24):
    """
    Sample points around the border (dense) plus a coarse interior grid, so the
    bounding box of the un-distorted footprint is captured even for strong SIP.
    Returns (px, py) as 0-based pixel coordinate arrays.
    """
    xs = np.linspace(0, W - 1, n_edge)
    ys = np.linspace(0, H - 1, n_edge)
    gx, gy = np.meshgrid(np.linspace(0, W - 1, n_grid),
                         np.linspace(0, H - 1, n_grid))
    px = np.concatenate([xs, xs, np.zeros(n_edge), np.full(n_edge, W - 1), gx.ravel()])
    py = np.concatenate([np.zeros(n_edge), np.full(n_edge, H - 1), ys, ys, gy.ravel()])
    return px, py


def compute_geometry(image_shape, wcs, expand: bool = True) -> Dict[str, Any]:
    """
    Work out the output canvas without resampling.

    Returns dict: outW, outH, x0, y0 (origin offset of output in the linear
    frame), sip_order, in_w, in_h, growth (out_px / in_px).
    """
    H, W = int(image_shape[0]), int(image_shape[1])
    w_lin = linear_wcs(wcs)
    px, py = _border_and_grid_pixels(W, H)
    ra, dec = wcs.all_pix2world(px, py, 0)          # full SIP forward
    tx, ty = w_lin.wcs_world2pix(ra, dec, 0)        # linear inverse (no SIP)
    finite = np.isfinite(tx) & np.isfinite(ty)
    tx, ty = tx[finite], ty[finite]
    xmin, xmax = float(np.min(tx)), float(np.max(tx))
    ymin, ymax = float(np.min(ty)), float(np.max(ty))

    if expand:
        x0 = int(math.floor(xmin)); y0 = int(math.floor(ymin))
        outW = int(math.ceil(xmax)) - x0 + 1
        outH = int(math.ceil(ymax)) - y0 + 1
    else:
        x0, y0 = 0, 0
        outW, outH = W, H

    return {
        "outW": int(outW), "outH": int(outH), "x0": x0, "y0": y0,
        "in_w": W, "in_h": H, "sip_order": sip_order(wcs),
        "growth": (outW * outH) / float(W * H) if W and H else 1.0,
    }


def output_wcs(wcs, geom: Dict[str, Any]):
    """The linear WCS for the (possibly shifted/expanded) output canvas."""
    w_out = linear_wcs(wcs)
    w_out.wcs.crpix = [w_out.wcs.crpix[0] - geom["x0"],
                       w_out.wcs.crpix[1] - geom["y0"]]
    return w_out


def unwarp_image(
    image: np.ndarray,
    wcs,
    *,
    expand: bool = True,
    order: int = 3,
    fill: float = 0.0,
    progress: Optional[Callable[[int, str], bool]] = None,
) -> Tuple[np.ndarray, Any, Dict[str, Any]]:
    """
    Resample `image` to remove SIP distortion.

    order: 0 nearest, 1 bilinear, 3 bicubic.
    fill:  value written outside the source footprint (0.0 or np.nan).
    progress(pct, msg) -> return False to cancel.

    Returns (out_image, out_wcs, geom).
    """
    if not _HAS_ASTROPY:
        raise RuntimeError("astropy is required for SIP unwarp.")
    if not _HAS_SCIPY:
        raise RuntimeError("scipy is required for SIP unwarp.")
    if not has_sip(wcs):
        raise ValueError("This WCS has no SIP distortion — nothing to unwarp.")

    wcs = wcs.celestial if wcs.naxis != 2 else wcs
    src = np.asarray(image)
    H, W = src.shape[:2]
    is_color = (src.ndim == 3)
    nchan = src.shape[2] if is_color else 1

    geom = compute_geometry(src.shape, wcs, expand=expand)
    outW, outH = geom["outW"], geom["outH"]
    w_out = output_wcs(wcs, geom)

    src_f = src.astype(np.float32, copy=False)
    if is_color:
        out = np.empty((outH, outW, nchan), dtype=np.float32)
    else:
        out = np.empty((outH, outW), dtype=np.float32)
    # after w_out is built, before the banded resample loop:
    px, py = _border_and_grid_pixels(W, H)
    ra_s, dec_s = wcs.all_pix2world(px, py, 0)          # true sky of source pixels
    ox, oy = w_out.wcs_world2pix(ra_s, dec_s, 0)         # where they land in output
    # account for the origin shift so we compare like-for-like against source px,py
    shift = np.hypot((ox - geom["x0"]) - px, (oy - geom["y0"]) - py)
    finite = np.isfinite(shift)
    geom["max_shift_px"] = float(np.nanmax(shift[finite])) if finite.any() else 0.0
    geom["med_shift_px"] = float(np.nanmedian(shift[finite])) if finite.any() else 0.0
    # Work in horizontal bands so a huge canvas doesn't allocate several
    # full-frame coordinate arrays at once, and so progress can be reported.
    band = max(16, min(256, outH))
    xline = np.arange(outW)
    done = 0
    from concurrent.futures import ThreadPoolExecutor
    import os

    src_f = src.astype(np.float32, copy=False)
    if is_color:
        out = np.empty((outH, outW, nchan), dtype=np.float32)
    else:
        out = np.empty((outH, outW), dtype=np.float32)

    band = max(16, min(256, outH))
    y_starts = list(range(0, outH, band))
    xline = np.arange(outW)

    # thread-local WCS copies (WCS objects are NOT concurrency-safe)
    import threading
    _local = threading.local()
    def _wcs_pair():
        p = getattr(_local, "pair", None)
        if p is None:
            p = (w_out.deepcopy(), wcs.deepcopy())
            _local.pair = p
        return p

    def _do_block(y0b):
        wl, ws = _wcs_pair()
        y1b = min(outH, y0b + band)
        yy, xx = np.meshgrid(np.arange(y0b, y1b), xline, indexing="ij")
        ra, dec = wl.all_pix2world(xx.ravel(), yy.ravel(), 0)   # linear fwd
        sx, sy = ws.all_world2pix(ra, dec, 0)                   # SIP inverse
        coords = np.vstack([sy, sx])
        rows = y1b - y0b
        if is_color:
            for c in range(nchan):
                samp = map_coordinates(src_f[..., c], coords, order=order,
                                       mode="constant", cval=fill)
                out[y0b:y1b, :, c] = samp.reshape(rows, outW)
        else:
            samp = map_coordinates(src_f, coords, order=order,
                                   mode="constant", cval=fill)
            out[y0b:y1b, :] = samp.reshape(rows, outW)
        return rows

    max_workers = min(len(y_starts), (os.cpu_count() or 4))
    done_rows = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_do_block, y0b): y0b for y0b in y_starts}
        for fut in as_completed(futures):
            rows = fut.result()          # re-raise worker exceptions here
            done_rows += rows
            if progress is not None:
                pct = int(done_rows / outH * 100)
                if progress(pct, f"Resampling… {done_rows}/{outH} rows") is False:
                    ex.shutdown(wait=False, cancel_futures=True)
                    raise RuntimeError("cancelled")

    return out, w_out, geom


# ── Header rewrite ────────────────────────────────────────────────────────────

_SIP_KEY = re.compile(r"^(A|B|AP|BP)_(\d+_\d+|ORDER|DMAX)$", re.IGNORECASE)
_WCS_KEY = re.compile(
    r"^(WCSAXES|CTYPE\d|CUNIT\d|CRPIX\d|CRVAL\d|CDELT\d|CROTA\d|"
    r"CD\d_\d|PC\d_\d|PV\d_\d|LONPOLE|LATPOLE)$", re.IGNORECASE)


def rewrite_header(src_header, w_out, out_shape) -> "Header":
    """
    Produce a FITS header for the unwarped image: the source header with SIP and
    old WCS keywords stripped, then the new linear WCS keywords written in.
    Non-WCS metadata (instrument, exposure, object, etc.) is preserved.
    """
    hdr = Header()
    if src_header is not None:
        try:
            hdr = src_header.copy()
        except Exception:
            try:
                hdr = Header(src_header)
            except Exception:
                hdr = Header()

    for key in list(hdr.keys()):
        k = str(key)
        if _SIP_KEY.match(k) or _WCS_KEY.match(k):
            try:
                del hdr[key]
            except Exception:
                pass

    try:
        hdr.update(w_out.to_header(relax=True))
    except Exception:
        pass

    H, W = int(out_shape[0]), int(out_shape[1])
    hdr["NAXIS1"] = W
    hdr["NAXIS2"] = H
    hdr["SASUNWRP"] = (True, "SIP distortion removed by SASpro Unwarp")
    return hdr


# ══════════════════════════════════════════════════════════════════════════════
#  Worker
# ══════════════════════════════════════════════════════════════════════════════

class _UnwarpWorker(QThread):
    progress = _Signal(int, str)
    finished = _Signal(object, object, object, object)   # image, wcs, header, geom
    failed   = _Signal(str)

    def __init__(self, image, wcs, src_header, *, expand, order, fill, parent=None):
        super().__init__(parent)
        self._image = image
        self._wcs = wcs
        self._src_header = src_header
        self._expand = expand
        self._order = order
        self._fill = fill
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            def _prog(pct, msg):
                self.progress.emit(pct, msg)
                return not self._cancel
            out, w_out, geom = unwarp_image(
                self._image, self._wcs,
                expand=self._expand, order=self._order, fill=self._fill,
                progress=_prog)
            hdr = rewrite_header(self._src_header, w_out, out.shape)
            self.finished.emit(out, w_out, hdr, geom)
        except Exception as e:
            if str(e) == "cancelled":
                self.failed.emit("Cancelled.")
            else:
                self.failed.emit(str(e))


# ══════════════════════════════════════════════════════════════════════════════
#  Presets
# ══════════════════════════════════════════════════════════════════════════════

# Canonical preset schema (single source of truth):
#   {"expand": bool, "order": int(0/1/3), "fill_nan": bool}

_INTERP = [("Bicubic (best)", 3), ("Bilinear", 1), ("Nearest", 0)]


def _order_to_index(order: int) -> int:
    return next((i for i, (_, o) in enumerate(_INTERP) if o == int(order)), 0)


def default_preset() -> dict:
    return {"expand": True, "order": 3, "fill_nan": False}


def build_unwarp_metadata(doc, wcs, header, image) -> dict:
    """Metadata for a delivered result: refreshed WCS/header + housekeeping."""
    meta = dict(get_doc_metadata(doc)) if doc is not None else {}
    meta["wcs"] = wcs
    meta["original_header"] = header
    for k in ("fits_header", "header"):
        if k in meta:
            meta[k] = header
    meta["is_mono"] = bool(np.asarray(image).ndim == 2)
    meta["is_sip_unwarped"] = True
    meta["step_name"] = _STEP_NAME
    return meta


def apply_unwarp_to_doc(doc, preset: Optional[dict] = None, *, main_window=None) -> bool:
    """
    Headless entry point (drag-grip drop-on-image / replay).

    Unwarps `doc` in place per `preset` and records undo history. Returns True
    on success, False if the image has no WCS/SIP (a no-op, logged if possible).
    """
    p = dict(default_preset())
    p.update(preset or {})
    order = int(p.get("order", 3))
    expand = bool(p.get("expand", True))
    fill = float("nan") if p.get("fill_nan", False) else 0.0

    try:
        out, w_out, hdr, geom = unwarp_doc_to_arrays(
            doc, expand=expand, order=order, fill=fill)
    except Exception as e:
        if main_window is not None and hasattr(main_window, "_log"):
            try:
                main_window._log(f"Unwarp skipped: {e}")
            except Exception:
                pass
        return False

    meta = build_unwarp_metadata(doc, w_out, hdr, out)
    apply_edit_to_doc(doc, out, meta, _STEP_NAME)

    if main_window is not None and hasattr(main_window, "_remember_last_headless_command"):
        try:
            main_window._remember_last_headless_command(
                UNWARP_COMMAND_ID, p, description="Unwarp (remove SIP)")
        except Exception:
            pass
    return True


class _UnwarpPresetDialog(QDialog):
    """
    Editor for an unwarp preset (used by the shortcut create/edit path), mirrors
    the Star Stretch preset dialog.
    """
    def __init__(self, parent=None, initial: Optional[dict] = None):
        super().__init__(parent)
        self.setWindowTitle("Unwarp — Preset")
        init = dict(default_preset())
        init.update(initial or {})

        self.combo_interp = QComboBox()
        for label, _ in _INTERP:
            self.combo_interp.addItem(label)
        self.combo_interp.setCurrentIndex(_order_to_index(init.get("order", 3)))

        self.chk_expand = QCheckBox("Expand canvas to fit corrected corners")
        self.chk_expand.setChecked(bool(init.get("expand", True)))

        self.chk_nan = QCheckBox("Fill outside footprint with NaN (else black)")
        self.chk_nan.setChecked(bool(init.get("fill_nan", False)))

        form = QFormLayout(self)
        form.addRow("Interpolation:", self.combo_interp)
        form.addRow("", self.chk_expand)
        form.addRow("", self.chk_nan)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        form.addRow(btns)

    def result_dict(self) -> dict:
        return {
            "expand": bool(self.chk_expand.isChecked()),
            "order": int(_INTERP[self.combo_interp.currentIndex()][1]),
            "fill_nan": bool(self.chk_nan.isChecked()),
        }


# ══════════════════════════════════════════════════════════════════════════════
#  Dialog
# ══════════════════════════════════════════════════════════════════════════════

class UnwarpDialog(QDialog):
    """
    Remove SIP distortion from a plate-solved image.

    Constructor mirrors the other SASpro tool dialogs:
        UnwarpDialog(parent=..., settings=..., doc_manager=...,
                     list_open_docs_fn=..., document=<active doc or None>)

    Optional injection hooks (used if provided, else the standard doc API is
    called):
        apply_fn(doc, image, metadata, step_name)   -> in-place edit w/ undo
        create_fn(image, metadata, title)           -> new document/view
    """

    _INTERP = _INTERP

    def __init__(self, parent=None, settings=None, doc_manager=None,
                 list_open_docs_fn=None, document=None,
                 apply_fn=None, create_fn=None):
        super().__init__(parent)
        self.setWindowTitle("Unwarp (Remove SIP Distortion)")
        self.setWindowFlag(Qt.WindowType.Window, True)
        import platform
        if platform.system() == "Darwin":
            self.setWindowFlag(Qt.WindowType.Tool, True)
        self.setModal(False)
        self.setMinimumWidth(460)
        self.setMinimumHeight(560) 
        try:
            self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        except Exception:
            pass

        self._settings = settings
        self._doc_manager = doc_manager
        self._list_open_docs_fn = list_open_docs_fn
        self._apply_fn = apply_fn
        self._create_fn = create_fn

        self._docs: List[tuple] = []      # (title, doc)
        self._worker: Optional[_UnwarpWorker] = None
        self._result = None               # (image, wcs, header, geom)

        self._build_ui()
        self._populate_sources(document)

    # ── UI ────────────────────────────────────────────────────────────────

    def _build_ui(self):
        v = QVBoxLayout(self)
        v.setContentsMargins(14, 14, 14, 14)
        v.setSpacing(10)

        intro = QLabel(
            "Resamples the active image onto a distortion-free grid using its "
            "plate-solved WCS + SIP polynomial. The canvas is expanded so the "
            "corrected corners aren't clipped; the result carries a clean linear WCS.")
        intro.setWordWrap(True)
        intro.setStyleSheet("color:#889; font-size:11px;")
        v.addWidget(intro)

        # Source
        src_row = QHBoxLayout()
        src_row.addWidget(QLabel("Source image:"))
        self._combo_src = QComboBox()
        self._combo_src.currentIndexChanged.connect(self._on_source_changed)
        src_row.addWidget(self._combo_src, stretch=1)
        v.addLayout(src_row)

        # Detected-solution info
        self._info = QLabel("")
        self._info.setWordWrap(True)
        self._info.setStyleSheet(
            "color:#9ab; font-size:11px; font-family:monospace; "
            "background:#0a0a18; border:1px solid #2a2a3e; border-radius:4px; "
            "padding:6px;")
        v.addWidget(self._info)

        # Options
        opt = QFrame()
        opt.setStyleSheet("QFrame{border:1px solid #2a2a3e; border-radius:6px;}")
        og = QGridLayout(opt)
        og.setContentsMargins(10, 10, 10, 10)
        og.setHorizontalSpacing(10)
        og.setVerticalSpacing(8)

        og.addWidget(QLabel("Interpolation:"), 0, 0)
        self._combo_interp = QComboBox()
        for label, _ in self._INTERP:
            self._combo_interp.addItem(label)
        og.addWidget(self._combo_interp, 0, 1)

        self._chk_expand = QCheckBox("Expand canvas to fit corrected corners")
        self._chk_expand.setChecked(True)
        self._chk_expand.setToolTip(
            "On: grow the output so nothing is clipped (recommended).\n"
            "Off: keep the original dimensions — distortion at the edges may "
            "push some pixels out of frame.")
        self._chk_expand.toggled.connect(self._refresh_info)
        og.addWidget(self._chk_expand, 1, 0, 1, 2)

        self._chk_nan = QCheckBox("Fill outside footprint with NaN (else black)")
        self._chk_nan.setChecked(False)
        self._chk_nan.setToolTip(
            "Areas the source doesn't cover after correction. NaN keeps them "
            "as 'no data'; unchecked fills them with 0 (black).")
        og.addWidget(self._chk_nan, 2, 0, 1, 2)

        v.addWidget(opt)

        # Progress
        self._bar = QProgressBar()
        self._bar.setRange(0, 100)
        self._bar.setValue(0)
        self._bar.setTextVisible(True)
        self._bar.setVisible(False)
        v.addWidget(self._bar)

        self._status = QLabel("")
        self._status.setStyleSheet("color:#778; font-size:11px;")
        self._status.setWordWrap(True)
        self._status.setMinimumHeight(78)   # room for the 4-line success readout
        v.addWidget(self._status)

        # Buttons
        btns = QHBoxLayout()
        self._btn_apply = QPushButton("Apply (in place)")
        self._btn_apply.setToolTip("Replace the source image with the unwarped "
                                   "result (undoable).")
        self._btn_apply.clicked.connect(lambda: self._run("apply"))
        btns.addWidget(self._btn_apply)

        self._btn_new = QPushButton("New View")
        self._btn_new.setToolTip("Open the unwarped result as a new image.")
        self._btn_new.clicked.connect(lambda: self._run("new"))
        btns.addWidget(self._btn_new)

        self._btn_save = QPushButton("Save FITS…")
        self._btn_save.setToolTip("Write the unwarped result straight to a FITS file.")
        self._btn_save.clicked.connect(lambda: self._run("save"))
        btns.addWidget(self._btn_save)

        btns.addStretch(1)
        self._btn_cancel = QPushButton("Cancel")
        self._btn_cancel.setVisible(False)
        self._btn_cancel.clicked.connect(self._cancel)
        btns.addWidget(self._btn_cancel)
        self._btn_close = QPushButton("Close")
        self._btn_close.clicked.connect(self.close)
        btns.addWidget(self._btn_close)
        v.addLayout(btns)

        # Preset drag grip — its own row, pinned to the bottom-left, below the buttons
        grip_row = QHBoxLayout()
        grip_row.setContentsMargins(0, 4, 0, 0)
        if _HAS_PRESET and PresetDragHandle is not None:
            try:
                self.preset_drag_handle = PresetDragHandle(
                    UNWARP_COMMAND_ID,
                    self._unwarp_params,
                    icon=QIcon(unwarp_path) if unwarp_path else QIcon(),
                    tooltip=(
                        "Drag to the canvas to create an Unwarp shortcut with "
                        "these exact settings.\n"
                        "Drop directly on a plate-solved image to unwarp it "
                        "headlessly."),
                    parent=self,
                )
                self.preset_drag_handle.setFixedSize(28, 28)
                grip_row.addWidget(self.preset_drag_handle)
            except Exception:
                self.preset_drag_handle = None
        else:
            self.preset_drag_handle = None
        grip_row.addStretch(1)          # push the grip hard to the left
        v.addLayout(grip_row)

        if not (_HAS_ASTROPY and _HAS_SCIPY):
            miss = []
            if not _HAS_ASTROPY: miss.append("astropy")
            if not _HAS_SCIPY:   miss.append("scipy")
            self._status.setText("Missing dependency: " + ", ".join(miss))
            for b in (self._btn_apply, self._btn_new, self._btn_save):
                b.setEnabled(False)

    # ── presets ──────────────────────────────────────────────────────────────

    def _unwarp_params(self) -> dict:
        """Canonical preset — single source of truth for apply + the drag grip."""
        return {
            "expand":   bool(self._chk_expand.isChecked()),
            "order":    int(self._INTERP[self._combo_interp.currentIndex()][1]),
            "fill_nan": bool(self._chk_nan.isChecked()),
        }

    def seed_from_preset(self, p: Optional[dict]) -> None:
        """Load a preset dict (same schema as _unwarp_params) into the controls,
        so a double-clicked shortcut opens the dialog pre-seeded."""
        if not p:
            return
        widgets = [self._combo_interp, self._chk_expand, self._chk_nan]
        for w in widgets:
            try:
                w.blockSignals(True)
            except Exception:
                pass
        try:
            self._combo_interp.setCurrentIndex(_order_to_index(p.get("order", 3)))
            self._chk_expand.setChecked(bool(p.get("expand", True)))
            self._chk_nan.setChecked(bool(p.get("fill_nan", False)))
        finally:
            for w in widgets:
                try:
                    w.blockSignals(False)
                except Exception:
                    pass
        self._refresh_info()

    # ── main-window / doc-manager resolution ─────────────────────────────────

    def _find_main_window(self):
        return _find_main_window_from(self.parent())

    def _get_doc_manager(self):
        if self._doc_manager is not None:
            return self._doc_manager
        return _get_doc_manager_from(self._find_main_window())

    # ── sources ─────────────────────────────────────────────────────────────

    def _normalize_docs(self, raw) -> List[tuple]:
        """Coerce whatever list_open_docs_fn returns into [(title, doc), ...]."""
        out = []
        if not raw:
            return out
        for item in raw:
            doc = None
            title = None
            if isinstance(item, (tuple, list)) and len(item) == 2:
                a, b = item
                if hasattr(a, "metadata") or get_doc_image(a) is not None:
                    doc, title = a, (b if isinstance(b, str) else None)
                elif hasattr(b, "metadata") or get_doc_image(b) is not None:
                    doc, title = b, (a if isinstance(a, str) else None)
                else:
                    doc = a
            else:
                doc = item
            if doc is None:
                continue
            out.append((title or doc_title(doc), doc))
        return out

    def _populate_sources(self, active_doc):
        raw = None
        if callable(self._list_open_docs_fn):
            try:
                raw = self._list_open_docs_fn()
            except Exception:
                raw = None
        self._docs = self._normalize_docs(raw)

        if active_doc is None:
            active_doc = _resolve_active_doc(self._find_main_window())

        if active_doc is not None:
            if not any(d is active_doc for _, d in self._docs):
                self._docs.insert(0, (doc_title(active_doc), active_doc))

        self._combo_src.blockSignals(True)
        self._combo_src.clear()
        if not self._docs:
            self._combo_src.addItem("— no open images —")
            self._combo_src.setEnabled(False)
        else:
            for title, _ in self._docs:
                self._combo_src.addItem(title)
            if active_doc is not None:
                for i, (_, d) in enumerate(self._docs):
                    if d is active_doc:
                        self._combo_src.setCurrentIndex(i)
                        break
        self._combo_src.blockSignals(False)
        self._refresh_info()

    def _current_doc(self):
        i = self._combo_src.currentIndex()
        if 0 <= i < len(self._docs):
            return self._docs[i][1]
        return None

    def _on_source_changed(self, _):
        self._refresh_info()

    def _refresh_info(self):
        doc = self._current_doc()
        for b in (self._btn_apply, self._btn_new, self._btn_save):
            b.setEnabled(False)

        if doc is None:
            self._info.setText("No image selected.")
            return
        img = get_doc_image(doc)
        if img is None:
            self._info.setText("Selected item has no readable image data.")
            return
        wcs = extract_wcs(doc)
        if wcs is None:
            self._info.setText(
                "No astrometric solution (WCS) on this image.\n"
                "Run the Plate Solver first — with SIP enabled — then unwarp.")
            return
        if not has_sip(wcs):
            self._info.setText(
                "This image is solved but its WCS has NO SIP distortion terms.\n"
                "There's nothing to unwarp — the solution is already linear.")
            return

        H, W = img.shape[:2]
        try:
            geom = compute_geometry(img.shape, wcs, expand=self._chk_expand.isChecked())
            scale = pixel_scale_arcsec(wcs)
            growth = geom["growth"]
            grow_note = (f"canvas grows ×{growth:.2f}"
                         if geom["outW"] * geom["outH"] > W * H else
                         "canvas unchanged")
            warn = ""
            if growth > 2.5:
                warn = "\n⚠ large expansion — check the SIP solution is sane"
            self._info.setText(
                f"SIP order {geom['sip_order']}   ·   {scale:.3f}\"/px\n"
                f"input   {W} × {H}\n"
                f"output  {geom['outW']} × {geom['outH']}   ({grow_note}){warn}")
            for b in (self._btn_apply, self._btn_new, self._btn_save):
                b.setEnabled(True)
        except Exception as e:
            self._info.setText(f"Could not analyze the solution: {e}")

    # ── run ─────────────────────────────────────────────────────────────────

    def _run(self, mode: str):
        doc = self._current_doc()
        if doc is None:
            return
        img = get_doc_image(doc)
        wcs = extract_wcs(doc)
        if img is None or wcs is None or not has_sip(wcs):
            self._refresh_info()
            return

        self._pending_mode = mode
        self._pending_doc = doc
        order = self._INTERP[self._combo_interp.currentIndex()][1]
        fill = float("nan") if self._chk_nan.isChecked() else 0.0

        self._set_busy(True)
        self._reset_status_style()
        self._worker = _UnwarpWorker(
            img, wcs, get_doc_header(doc),
            expand=self._chk_expand.isChecked(), order=order, fill=fill,
            parent=None)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _cancel(self):
        if self._worker is not None:
            self._worker.cancel()
            self._status.setText("Cancelling…")

    def _set_busy(self, busy: bool):
        self._bar.setVisible(busy)
        self._btn_cancel.setVisible(busy)
        for b in (self._btn_apply, self._btn_new, self._btn_save,
                  self._btn_close, self._combo_src, self._combo_interp,
                  self._chk_expand, self._chk_nan):
            b.setEnabled(not busy)
        if busy:
            self._bar.setValue(0)

    def _on_progress(self, pct: int, msg: str):
        self._bar.setValue(pct)
        self._status.setText(msg)

    def _on_failed(self, msg: str):
        self._set_busy(False)
        self._worker = None
        self._reset_status_style()
        self._refresh_info()
        if msg != "Cancelled.":
            QMessageBox.warning(self, "Unwarp", f"Unwarp failed:\n{msg}")
        self._status.setText(msg)

    def _result_summary(self, geom, verb):
        n = geom.get("sip_order", 0)
        iw, ih, ow, oh = geom["in_w"], geom["in_h"], geom["outW"], geom["outH"]
        grow = geom.get("growth", 1.0)
        max_px = geom.get("max_shift_px", 0.0)
        med_px = geom.get("med_shift_px", 0.0)
        scale = getattr(self, "_last_scale_arcsec", 0.0)

        lines = [f"✓ {verb} — SIP order {n} distortion removed."]
        if scale > 0:
            lines.append(
                f"Peak edge to edge correction: {max_px:.1f}px ({max_px*scale:.2f}\")   ·   "
                f"median {med_px:.1f}px ({med_px*scale:.2f}\")")
        else:
            lines.append(f"Peak edge correction: {max_px:.1f}px   ·   median {med_px:.1f}px")
        lines.append(
            f"Canvas {iw}×{ih} → {ow}×{oh}"
            + (f"  (×{grow:.2f})" if ow*oh != iw*ih else "  (unchanged)"))
        lines.append("The result carries a clean linear WCS. Re-solve to confirm "
                     "the residuals collapse.")
        return "\n".join(lines)

    _STATUS_PLAIN   = "color:#778; font-size:11px;"
    _STATUS_SUCCESS = ("color:#8 fdb98; font-size:13px; font-weight:600; "
                       "line-height:150%;")

    def _show_result(self, text: str):
        head, *rest = text.split("\n")
        body = "<br>".join(rest)
        self._status.setTextFormat(Qt.TextFormat.RichText)
        self._status.setStyleSheet("font-size:13px; line-height:150%;")
        self._status.setText(
            f"<span style='color:#8fdb98; font-weight:700;'>{head}</span>"
            f"<br><span style='color:#9ab;'>{body}</span>")

    def _reset_status_style(self):
        self._status.setStyleSheet(self._STATUS_PLAIN)

    def _on_finished(self, image, wcs, header, geom):
        self._worker = None
        self._set_busy(False)
        self._result = (image, wcs, header, geom)
        mode = getattr(self, "_pending_mode", "new")
        doc = getattr(self, "_pending_doc", None)
        try:
            src_wcs = extract_wcs(doc) if doc is not None else None
            self._last_scale_arcsec = pixel_scale_arcsec(src_wcs) if src_wcs else 0.0
        except Exception:
            self._last_scale_arcsec = 0.0
        metadata = build_unwarp_metadata(doc, wcs, header, image)
        title = f"{doc_title(doc)} [unwarped]" if doc is not None else "Unwarped"

        try:
            if mode == "save":
                self._save_fits(image, header)
            elif mode == "apply" and doc is not None:
                self._apply_inplace(doc, image, metadata)
                self._remember_replay()
                self._show_result(self._result_summary(geom, "Unwarped in place"))
            else:
                self._create_new(image, metadata, title)
                self._remember_replay()
                self._show_result(self._result_summary(geom, "Opened unwarped result"))
        except Exception as e:
            QMessageBox.warning(
                self, "Unwarp",
                f"The image was unwarped, but delivering it to the app failed:\n{e}\n\n"
                f"You can still use “Save FITS…”.")
            self._status.setText(f"Delivery failed: {e}")

    # ── delivery ─────────────────────────────────────────────────────────────

    def _apply_inplace(self, doc, image, metadata):
        if callable(self._apply_fn):
            self._apply_fn(doc, image, metadata, _STEP_NAME)
            return
        apply_edit_to_doc(doc, image, metadata, _STEP_NAME)

    def _create_new(self, image, metadata, title):
        if callable(self._create_fn):
            self._create_fn(image, metadata, title)
            return
        dm = self._get_doc_manager()
        if dm is None:
            raise RuntimeError("No document manager available to create a new view.")
        create_doc_via_manager(dm, image, metadata, title)

    def _remember_replay(self):
        mw = self._find_main_window()
        if mw is not None and hasattr(mw, "_remember_last_headless_command"):
            try:
                mw._remember_last_headless_command(
                    UNWARP_COMMAND_ID, self._unwarp_params(),
                    description="Unwarp (remove SIP)")
            except Exception:
                pass

    def _save_fits(self, image, header):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Unwarped FITS", "",
            "FITS files (*.fits *.fit *.fts);;All files (*)")
        if not path:
            self._status.setText("Save cancelled.")
            return
        try:
            from astropy.io import fits
        except Exception as e:
            QMessageBox.warning(self, "Unwarp", f"astropy.io.fits unavailable:\n{e}")
            return
        data = image
        if data.ndim == 3:                      # FITS wants channels-first for color
            data = np.moveaxis(data, 2, 0)
        try:
            fits.PrimaryHDU(data=np.asarray(data, dtype=np.float32),
                            header=header).writeto(path, overwrite=True)
            self._status.setText(f"Saved → {path}")
        except Exception as e:
            QMessageBox.warning(self, "Unwarp", f"Could not write FITS:\n{e}")

    # ── lifecycle ────────────────────────────────────────────────────────────

    def closeEvent(self, ev):
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait(3000)
        super().closeEvent(ev)


# ══════════════════════════════════════════════════════════════════════════════
#  Entry points
# ══════════════════════════════════════════════════════════════════════════════

def unwarp_doc_to_arrays(doc, *, expand: bool = True, order: int = 3,
                         fill: float = 0.0):
    """
    Headless helper: unwarp a document and return (image, wcs, header, geom).
    Raises ValueError if the doc has no WCS or no SIP.
    """
    img = get_doc_image(doc)
    if img is None:
        raise ValueError("Document has no image data.")
    wcs = extract_wcs(doc)
    if wcs is None:
        raise ValueError("Document has no WCS — plate solve first.")
    out, w_out, geom = unwarp_image(img, wcs, expand=expand, order=order, fill=fill)
    hdr = rewrite_header(get_doc_header(doc), w_out, out.shape)
    return out, w_out, hdr, geom


def open_unwarp_dialog(main_window, preset: Optional[dict] = None):
    """
    Open the Unwarp dialog, optionally seeded from a preset. Resolves the active
    document, doc manager, and open-docs list from the main window — matching
    the toolbar opener pattern.
    """
    dm = _get_doc_manager_from(main_window)
    list_fn = getattr(main_window, "_list_open_docs", None)
    doc = _resolve_active_doc(main_window)

    dlg = UnwarpDialog(
        parent=main_window,
        settings=getattr(main_window, "settings", None),
        doc_manager=dm,
        list_open_docs_fn=list_fn,
        document=doc,
    )
    try:
        if unwarp_path:
            dlg.setWindowIcon(QIcon(unwarp_path))
    except Exception:
        pass
    if preset:
        try:
            dlg.seed_from_preset(preset)
        except Exception:
            pass
    dlg.show()
    dlg.raise_()
    dlg.activateWindow()
    return dlg


# Alias matching the Star Stretch naming, for the double-click preset-open path.
def open_unwarp_with_preset(main_window, preset: Optional[dict] = None):
    return open_unwarp_dialog(main_window, preset=preset)