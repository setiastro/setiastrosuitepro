# ExoPlanet Detector (SASpro) — standalone plate solving, no WIMI

from __future__ import annotations

from setiastro.saspro.main_helpers import non_blocking_sleep

import os
import shutil
import tempfile
import time
from typing import List, Tuple, Set
import webbrowser
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
from urllib.parse import quote
from types import SimpleNamespace
import math
import numpy as np
import pandas as pd
import sep
import pyqtgraph as pg
import matplotlib.pyplot as plt
from types import SimpleNamespace
from astropy.io import fits
from astropy.time import Time
from astropy.stats import sigma_clipped_stats
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.wcs import WCS
from astropy.timeseries import LombScargle, BoxLeastSquares
import re

from astroquery.simbad import Simbad
from astroquery.vizier import Vizier

from astroquery.mast import Tesscut

from lightkurve import TessTargetPixelFile

import lightkurve as lk

# ---- project-local imports (adjust paths if needed) --------------------
from setiastro.saspro.legacy.numba_utils import bin2x2_numba, apply_flat_division_numba
from setiastro.saspro.imageops.stretch import stretch_mono_image, stretch_color_image

from setiastro.saspro.plate_solver import plate_solve_doc_inplace
from setiastro.saspro.star_alignment import (
    StarRegistrationWorker,
    StarRegistrationThread,
    IDENTITY_2x3,
)
from setiastro.saspro.legacy.image_manager import load_image, save_image, get_valid_header  # adjust if different
from setiastro.saspro.widgets.themed_buttons import themed_toolbtn

# ------------------------------------------------------------------------
from setiastro.saspro.xisf import XISF
from PyQt6.QtCore import Qt, QTimer, QSettings, QRectF, QPoint, QPointF
from PyQt6.QtGui import QIcon, QColor, QBrush, QPen, QPainter, QImage, QPixmap
from PyQt6.QtWidgets import (
    QAbstractItemView, QButtonGroup, QComboBox, QDialog, QDialogButtonBox, QApplication, QGraphicsView, QGraphicsPixmapItem,
    QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QListWidget, QGraphicsView, QGraphicsScene, QGraphicsEllipseItem,
    QListWidgetItem, QMessageBox, QPushButton, QProgressBar, QRadioButton, QSpinBox, QDoubleSpinBox,
    QSlider, QToolButton, QVBoxLayout, QInputDialog, QLineEdit
)
import pyqtgraph as pg

import warnings
from astropy.utils.exceptions import AstropyWarning
warnings.filterwarnings("ignore", category=AstropyWarning, message=".*more axes.*")

def _extract_ra_dec_from_header(h: fits.Header):
    """Return (ra_deg, dec_deg) if found, else (None, None)."""
    if not isinstance(h, fits.Header):
        return None, None

    # 1) If WCS is already present, use its center
    try:
        w = WCS(h)
        if w.has_celestial:
            # center of the current pixel grid
            nx = h.get("NAXIS1"); ny = h.get("NAXIS2")
            if nx and ny:
                sky = w.pixel_to_world(nx/2, ny/2)
                return float(sky.ra.deg), float(sky.dec.deg)
    except Exception:
        pass

    # 2) Common RA/DEC keyword pairs
    pairs = [
        ("OBJCTRA", "OBJCTDEC"),  # PixInsight/ASCOM style (strings)
        ("RA",      "DEC"),
        ("TELRA",   "TELDEC"),
        ("RA_OBJ",  "DEC_OBJ"),
        ("CAT-RA",  "CAT-DEC"),
        ("RA_DEG",  "DEC_DEG"),   # degrees already
    ]

    for rak, deck in pairs:
        if rak in h and deck in h:
            ra_raw, dec_raw = h[rak], h[deck]

            # Try a few parse paths
            for parser in (
                lambda r,d: SkyCoord(r, d, unit=(u.hourangle, u.deg)),
                lambda r,d: SkyCoord(float(r)*u.deg, float(d)*u.deg),
                lambda r,d: SkyCoord(r, d, unit=(u.deg, u.deg)),
            ):
                try:
                    c = parser(ra_raw, dec_raw)
                    return float(c.ra.deg), float(c.dec.deg)
                except Exception:
                    pass

    return None, None


def _estimate_scale_arcsec_per_pix(h: fits.Header):
    """Return pixel scale (arcsec/pix) if derivable, else None."""
    if not isinstance(h, fits.Header):
        return None

    # Direct scale keywords (various conventions)
    for k in ("PIXSCALE", "PIXSCL", "SECPIX", "SECPIX1"):
        if k in h:
            try:
                val = float(h[k])
                if val > 0:
                    return val
            except Exception:
                pass

    # Derive from pixel size & focal length:
    # scale["/pix] ≈ 206.265 * pixel_size_μm / focal_length_mm
    px_um = None
    for k in ("XPIXSZ", "PIXSIZE1", "PIXSIZE"):  # μm
        if k in h:
            try:
                px_um = float(h[k])
                break
            except Exception:
                pass

    foc_mm = None
    for k in ("FOCALLEN", "FOCLEN", "FOCALLENGTH"):
        if k in h:
            try:
                foc_mm = float(h[k])
                break
            except Exception:
                pass

    if px_um and foc_mm and foc_mm > 0:
        return 206.265 * px_um / foc_mm

    return None

_TZ_RE = re.compile(r'([+-])(\d{2})(\d{2})$')  # -0700 -> -07:00

def _fix_iso_tz(s: str) -> str:
    s = s.strip()
    m = _TZ_RE.search(s)
    if m:
        s = s[:m.start()] + f"{m.group(1)}{m.group(2)}:{m.group(3)}"
    return s

def _parse_obs_time_from_header(hdr) -> Time | None:
    # hdr can be fits.Header or your dict-ish header
    def _get(key):
        try:
            return hdr.get(key)
        except Exception:
            return None

    # 1) Prefer UT-OBS if present (already “UTC-ish”)
    for key in ("UT-OBS", "DATE-OBS", "DATE-END"):
        v = _get(key)
        if isinstance(v, str) and v.strip():
            try:
                return Time(_fix_iso_tz(v), format="isot", scale="utc")
            except Exception:
                pass

    # 2) MJD-OBS is super reliable
    v = _get("MJD-OBS")
    if v is not None:
        try:
            return Time(float(v), format="mjd", scale="utc")
        except Exception:
            pass

    return None

def _estimate_fov_deg(img_shape, scale_arcsec):
    """Rough FOV (deg) from image size and scale (max of X/Y)."""
    try:
        h, w = img_shape[:2]
        if scale_arcsec and h and w:
            fov_x = (w * scale_arcsec) / 3600.0
            fov_y = (h * scale_arcsec) / 3600.0
            return float(max(fov_x, fov_y))
    except Exception:
        pass
    return None


def _build_astrometry_hints(hdr: fits.Header, plane: np.ndarray):
    """Compose a hints dict for the solver."""
    ra_deg, dec_deg = _extract_ra_dec_from_header(hdr)
    scale = _estimate_scale_arcsec_per_pix(hdr)
    fov   = _estimate_fov_deg(plane.shape, scale)

    # A generous search radius: ≥1°, or 3×FOV if we have it
    radius = None
    if fov is not None:
        radius = max(1.0, 3.0 * fov)

    hints = {}
    if ra_deg is not None and dec_deg is not None:
        hints["ra_deg"]  = ra_deg
        hints["dec_deg"] = dec_deg
    if scale is not None:
        hints["pixel_scale_arcsec"] = scale
    if fov is not None:
        hints["fov_deg"] = fov
    if radius is not None:
        hints["search_radius_deg"] = radius

    # Optional: parity if you know you’re mirrored or not (None = let solver decide)
    # hints["parity"] = +1  # or -1

    return hints

# ============================================================
# Transit search helpers — ported from exoplanet_ground_telescope_script - Kyle Lynch
# ============================================================

def _robust_sigma(x: np.ndarray) -> float:
    """Median-absolute-deviation based sigma (1.4826 * MAD)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return np.nan
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    return float(1.4826 * mad) if np.isfinite(mad) else np.nan


def _sort_lightcurve(time_jd: np.ndarray, flux: np.ndarray,
                     positive_only: bool = True):
    t = np.asarray(time_jd, float)
    y = np.asarray(flux, float)
    good = np.isfinite(t) & np.isfinite(y)
    if positive_only:
        good &= (y > 0)
    if not np.any(good):
        return np.array([], float), np.array([], float)
    s = np.argsort(t[good])
    return t[good][s], y[good][s]


def _segment_slices(time: np.ndarray, gap_days: float):
    t = np.asarray(time, float)
    if t.size == 0:
        return []
    cuts = np.where(np.diff(t) > gap_days)[0]
    starts = np.r_[0, cuts + 1]
    ends   = np.r_[cuts + 1, t.size]
    return [slice(int(a), int(b)) for a, b in zip(starts, ends)]


def _upper_outlier_mask(y: np.ndarray, sigma_mult: float = 6.0) -> np.ndarray:
    arr = np.asarray(y, float)
    mask = np.zeros(arr.shape, dtype=bool)
    good = np.isfinite(arr) & (arr > 0)
    if np.sum(good) < 6:
        return mask
    med = float(np.nanmedian(arr[good]))
    sig = _robust_sigma(arr[good] - med)
    if not (np.isfinite(sig) and sig > 0):
        return mask
    return np.isfinite(arr) & (arr > med + sigma_mult * sig)


def _prepare_search_lightcurve(time_jd: np.ndarray, flux: np.ndarray):
    """Clean, sort, median-normalise, and clip upper outliers."""
    t, y = _sort_lightcurve(time_jd, flux, positive_only=True)
    if t.size == 0:
        return t, y
    med = np.nanmedian(y[np.isfinite(y) & (y > 0)])
    if np.isfinite(med) and med != 0:
        y = y / med
    bad = _upper_outlier_mask(y, sigma_mult=6.0)
    if np.any(bad):
        keep = np.isfinite(y) & (~bad)
        t, y = t[keep], y[keep]
        if y.size > 0:
            med2 = np.nanmedian(y)
            if np.isfinite(med2) and med2 != 0:
                y = y / med2
    return t, y


def _phase_centered(time: np.ndarray, period: float, t0: float) -> np.ndarray:
    """Fold to [-0.5, 0.5]."""
    ph = ((np.asarray(time, float) - t0) / period) % 1.0
    return ((ph + 0.5) % 1.0) - 0.5


def _whiten_periodogram(power: np.ndarray):
    """Subtract running-median background from BLS power spectrum."""
    from scipy.ndimage import median_filter as _mf
    power = np.asarray(power, float)
    if power.size == 0:
        return np.array([], float), np.array([], float)
    finite = np.isfinite(power)
    fill = float(np.nanmedian(power[finite])) if np.any(finite) else 0.0
    arr = np.where(finite, power, fill)
    max_window = power.size if power.size % 2 == 1 else power.size - 1
    if max_window < 3:
        bg = np.full_like(arr, fill, dtype=float)
    else:
        window = max(31, int(power.size // 150))
        if window % 2 == 0:
            window += 1
        window = max(3, min(window, max_window))
        bg = _mf(arr, size=window, mode="nearest")
    whitened = arr - bg
    whitened -= np.nanmedian(whitened[finite])
    whitened[~finite] = np.nan
    bg[~finite] = np.nan
    return bg, whitened


def _top_peak_indices(periods: np.ndarray, power: np.ndarray,
                      max_peaks: int = 12, merge_frac: float = 0.01):
    periods = np.asarray(periods, float)
    power   = np.asarray(power, float)
    if periods.size == 0:
        return []
    cand = []
    for i in range(power.size):
        if not np.isfinite(power[i]):
            continue
        left  = power[i-1] if i > 0 else -np.inf
        right = power[i+1] if i+1 < power.size else -np.inf
        if power[i] >= left and power[i] >= right:
            cand.append(i)
    if not cand:
        cand = list(np.where(np.isfinite(power))[0])
    cand = np.asarray(cand, int)
    order = cand[np.argsort(power[cand])[::-1]]
    keep = []
    for idx in order:
        p = periods[idx]
        if not (np.isfinite(p) and p > 0):
            continue
        if any(abs(p / periods[j] - 1.0) <= merge_frac for j in keep
               if np.isfinite(periods[j]) and periods[j] > 0):
            continue
        keep.append(int(idx))
        if len(keep) >= max_peaks:
            break
    return keep


def _make_binned_curve(x: np.ndarray, y: np.ndarray, bin_width: float):
    """Robust-sigma-clipped binning; returns (bx, by, be)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    if x.size < 10:
        return np.array([], float), np.array([], float), np.array([], float)
    edges = np.arange(np.nanmin(x), np.nanmax(x) + bin_width, bin_width)
    bx, by, be = [], [], []
    for i in range(len(edges) - 1):
        m = (x >= edges[i]) & (x < edges[i+1])
        if np.sum(m) < 4:
            continue
        yy = y[m]; xx = x[m]
        med = np.nanmedian(yy)
        sig = _robust_sigma(yy)
        if np.isfinite(sig) and sig > 0:
            keep = np.abs(yy - med) < 4.0 * sig
            xx, yy = xx[keep], yy[keep]
        if yy.size < 4:
            continue
        sig2 = _robust_sigma(yy)
        bx.append(float(np.nanmedian(xx)))
        by.append(float(np.nanmedian(yy)))
        be.append(float(sig2 / np.sqrt(yy.size)) if np.isfinite(sig2) and sig2 > 0
                  else float(np.nanstd(yy) / np.sqrt(max(1, yy.size))))
    return np.asarray(bx, float), np.asarray(by, float), np.asarray(be, float)


def _estimate_night_dip_events(time_jd: np.ndarray, flux: np.ndarray,
                                gap_days: float = 0.20):
    """
    Per-night dip detection: returns list of dicts with
    t_center, depth_frac, snr, duration_d.
    """
    from scipy.ndimage import median_filter as _mf
    t, y = _sort_lightcurve(time_jd, flux, positive_only=True)
    if t.size < 12:
        return []
    events = []
    for sl in _segment_slices(t, gap_days):
        tn, yn = t[sl], y[sl]
        good = np.isfinite(tn) & np.isfinite(yn) & (yn > 0)
        tn, yn = tn[good], yn[good]
        if tn.size < 8:
            continue
        med = float(np.nanmedian(yn))
        if med <= 0 or not np.isfinite(med):
            continue
        seg = yn / med
        scatter = _robust_sigma(seg)
        if not np.isfinite(scatter) or scatter <= 0:
            scatter = float(np.nanstd(seg))
        scatter = max(scatter, 1e-5)

        filled = seg.copy()
        filled[~np.isfinite(filled)] = 1.0
        w = min(11, max(5, (tn.size // 12) * 2 + 1))
        if w % 2 == 0:
            w += 1
        if w >= tn.size:
            w = tn.size if tn.size % 2 == 1 else max(5, tn.size - 1)
        smooth = _mf(filled, size=max(5, w), mode="nearest").astype(float)

        oot = float(np.nanmedian(smooth))
        min_idx = int(np.nanargmin(smooth))
        depth = max(0.0, oot - float(smooth[min_idx]))
        if not np.isfinite(depth) or depth < max(2.0 * scatter, 8e-4):
            continue

        thresh = oot - max(1.2 * scatter, 0.45 * depth)
        low = smooth <= thresh
        left = right = min_idx
        while left > 0 and low[left-1]:
            left -= 1
        while right + 1 < low.size and low[right+1]:
            right += 1

        # cadence estimate
        dts = np.diff(np.sort(tn))
        dts = dts[dts > 1e-6]
        cad = float(np.nanmedian(dts)) if dts.size > 0 else 5.0/1440.0
        dur = float((tn[right] - tn[left]) + max(cad, 5.0/1440.0))
        events.append({
            "t_center": float(np.nanmedian(tn[left:right+1])),
            "depth_frac": float(depth),
            "snr": float(depth / scatter),
            "duration_d": dur,
        })
    return events


def _alias_penalty(period_d: float, tol: float = 0.02) -> float:
    if not (np.isfinite(period_d) and period_d > 0):
        return 0.0
    penalty = 0.0
    for alias in (0.5, 1.0, 2.0, 3.0):
        frac = abs(period_d / alias - 1.0)
        if frac <= tol:
            penalty = max(penalty, 1.0 - frac / tol)
    return float(penalty)


class OverlayView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        # disable built-in hand drag
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        # always arrow cursor
        self.viewport().setCursor(Qt.CursorShape.ArrowCursor)
        self._panning = False
        self._last_pos = QPoint()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            scene_pt = self.mapToScene(event.pos())
            # if we clicked an ellipse, let it handle the event
            for it in self.scene().items(scene_pt):
                if isinstance(it, ClickableEllipseItem):
                    super().mousePressEvent(event)
                    return
            # else: start panning
            self._panning = True
            self._last_pos = event.pos()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._panning:
            delta = event.pos() - self._last_pos
            self._last_pos = event.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._panning:
            self._panning = False
            event.accept()
            return
        super().mouseReleaseEvent(event)

class ClickableEllipseItem(QGraphicsEllipseItem):
    def __init__(self, rect: QRectF, index: int, callback):
        super().__init__(rect)
        self.index = index
        self.callback = callback
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)
        self.setAcceptHoverEvents(True)

    def mousePressEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            shift = bool(ev.modifiers() & Qt.KeyboardModifier.ShiftModifier)
            self.callback(self.index, shift)
        super().mousePressEvent(ev)


class ReferenceOverlayDialog(QDialog):
    def __init__(self, plane: np.ndarray, positions: List[Tuple], target_median: float, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Reference Frame: Stars Overlay"))
        self.plane = np.asarray(plane, dtype=np.float32)
        self.positions = positions
        self.target_median = target_median
        self.autostretch = True

        self._normal_pen   = QPen(QColor('lightblue'), 3)
        self._dip_pen      = QPen(QColor('yellow'), 3)
        self._selected_pen = QPen(QColor('red'), 4)

        self.ellipse_items: dict[int, ClickableEllipseItem] = {}
        self.flagged_stars: Set[int] = set()

        self._build_ui()
        self._init_graphics()

        if parent and hasattr(parent, 'star_list'):
            parent.star_list.itemSelectionChanged.connect(self._update_highlights)

        QTimer.singleShot(0, self._fit_to_100pct)

    def _build_ui(self):
        self.view = OverlayView(self)
        self.view.setRenderHints(
            QPainter.RenderHint.Antialiasing |
            QPainter.RenderHint.SmoothPixmapTransform
        )
        self.scene = QGraphicsScene(self)
        self.view.setScene(self.scene)

        btns = QHBoxLayout()

        self.zoom_in_btn = QPushButton("Zoom In")
        self.zoom_in_btn.clicked.connect(lambda: self.view.scale(1.2, 1.2))
        btns.addWidget(self.zoom_in_btn)

        self.zoom_out_btn = QPushButton("Zoom Out")
        self.zoom_out_btn.clicked.connect(lambda: self.view.scale(1/1.2, 1/1.2))
        btns.addWidget(self.zoom_out_btn)

        self.reset_btn = QPushButton("Reset Zoom")
        self.reset_btn.clicked.connect(self._fit_to_100pct)
        btns.addWidget(self.reset_btn)

        self.fit_btn = QPushButton("Fit to Window")
        self.fit_btn.clicked.connect(self._fit_to_window)
        btns.addWidget(self.fit_btn)

        self.stretch_btn = QPushButton()
        self.stretch_btn.clicked.connect(self._toggle_autostretch)
        btns.addWidget(self.stretch_btn)

        self._update_stretch_button_text()
        btns.addStretch()

        lay = QVBoxLayout(self)
        lay.addWidget(self.view)
        lay.addLayout(btns)
        self.resize(800, 600)

    def _update_stretch_button_text(self):
        self.stretch_btn.setText(f"Stretch: {'ON' if self.autostretch else 'OFF'}")

    def _normalize_to_u8_linear(self, img: np.ndarray) -> np.ndarray:
        """
        Linear preview:
        - if already in [0,1], keep it linear
        - otherwise percentile-compress to 8-bit
        """
        arr = np.asarray(img, dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        finite = np.isfinite(arr)
        if not np.any(finite):
            return np.zeros(arr.shape, dtype=np.uint8)

        amin = float(np.min(arr[finite]))
        amax = float(np.max(arr[finite]))

        # Common SASpro case: float image already normalized to [0,1]
        if amin >= 0.0 and amax <= 1.0:
            out = np.clip(arr, 0.0, 1.0)
            return (out * 255.0).astype(np.uint8)

        # Otherwise make a robust linear preview
        lo, hi = np.percentile(arr[finite], [0.5, 99.5])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo = amin
            hi = amax if amax > amin else (amin + 1.0)

        out = (arr - lo) / max(hi - lo, 1e-8)
        out = np.clip(out, 0.0, 1.0)
        return (out * 255.0).astype(np.uint8)

    def _to_preview_u8(self, img: np.ndarray) -> np.ndarray:
        arr = np.asarray(img, dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        arr = np.clip(arr, 0.0, 1.0)
        return (arr * 255.0).astype(np.uint8)


    def _normalize_preview_source(self, img: np.ndarray) -> np.ndarray:
        arr = np.asarray(img)

        # keep original values, just move into a sane preview/stretch domain
        if arr.dtype.kind in "iu":
            arr = arr.astype(np.float32)

        else:
            arr = np.asarray(arr, dtype=np.float32)

        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        finite = np.isfinite(arr)
        if not np.any(finite):
            return np.zeros(arr.shape, dtype=np.float32)

        amin = float(np.min(arr[finite]))
        amax = float(np.max(arr[finite]))

        # already normalized
        if amin >= 0.0 and amax <= 1.0:
            return np.clip(arr, 0.0, 1.0)

        # signed / weird-range data -> normalize to 0..1 first
        if amax <= amin:
            return np.zeros(arr.shape, dtype=np.float32)

        arr01 = (arr - amin) / (amax - amin)
        return np.clip(arr01, 0.0, 1.0).astype(np.float32)

    def _init_graphics(self):
        base = self._normalize_preview_source(self.plane)
        img = base

        if self.autostretch:
            try:
                stretched = stretch_mono_image(base, target_median=0.3)
                stretched = np.asarray(stretched, dtype=np.float32)

                if np.isfinite(stretched).any() and np.nanstd(stretched) > 1e-4:
                    img = np.clip(stretched, 0.0, 1.0)
            except Exception as e:
                print(f"[Exoplanet preview] stretch_mono_image failed: {e}")
                img = base

        arr8 = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
        h, w = arr8.shape
        qimg = QImage(arr8.data, w, h, w, QImage.Format.Format_Grayscale8).copy()
        pix = QPixmap.fromImage(qimg)

        self.scene.clear()
        self.ellipse_items.clear()
        self.scene.addItem(QGraphicsPixmapItem(pix))

        radius = max(2, int(math.ceil(1.2 * self.target_median)))
        for idx, (x, y) in enumerate(self.positions):
            r = QRectF(x - radius, y - radius, 2 * radius, 2 * radius)
            ell = ClickableEllipseItem(r, idx, self._on_star_clicked)
            ell.setPen(self._normal_pen)
            ell.setBrush(QBrush(Qt.BrushStyle.NoBrush))
            self.scene.addItem(ell)
            self.ellipse_items[idx] = ell


    def _fit_to_100pct(self):
        self.view.resetTransform()
        rect = self.scene.itemsBoundingRect()
        self.view.setSceneRect(rect)
        self.view.centerOn(rect.center())

    def _fit_to_window(self):
        rect = self.scene.itemsBoundingRect()
        self.view.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)

    def _toggle_autostretch(self):
        self.autostretch = not self.autostretch
        self._update_stretch_button_text()
        self._init_graphics()

    def _on_star_clicked(self, index: int, shift: bool):
        parent = self.parent()
        if not parent or not hasattr(parent, 'star_list'):
            return

        lst = parent.star_list
        item = lst.item(index)
        if not item:
            return

        if shift:
            item.setSelected(not item.isSelected())
        else:
            lst.clearSelection()
            item.setSelected(True)

        lst.scrollToItem(item)
        self._update_highlights()

    def _update_highlights(self):
        parent = self.parent()
        if not parent or not hasattr(parent, 'star_list'):
            return

        sel = {item.data(Qt.ItemDataRole.UserRole)
               for item in parent.star_list.selectedItems()}

        for idx, ell in self.ellipse_items.items():
            if idx in sel:
                ell.setPen(self._selected_pen)
            elif idx in self.flagged_stars:
                ell.setPen(self._dip_pen)
            else:
                ell.setPen(self._normal_pen)

    def update_dip_flags(self, flagged_indices: Set[int]):
        self.flagged_stars = flagged_indices
        self._update_highlights()


class ExoPlanetWindow(QDialog):
    def __init__(self, parent=None, wrench_path=None, settings=None):
        super().__init__(parent)
        self.settings = settings
        self.main_win = parent
        self.setWindowTitle(self.tr("Exoplanet Transit Detector"))

        self.resize(900, 600)
        self.wrench_path = wrench_path
        # State
        self.image_paths      = []
        self._cached_images   = []
        self._cached_headers  = []   # parallel to _cached_images
        self.times            = None  # astropy Time array
        self.star_positions   = []
        self.fluxes           = None  # stars × frames
        self.flags            = None
        self.median_fwhm      = None
        self.master_dark      = None
        self.master_flat      = None
        self.exposure_time    = None
        self._last_ensemble   = []
        self.ensemble_map     = {}

        # --- new settings ---
        self.sep_threshold     = 5.0   # SEP σ
        self.border_fraction   = 0.10  # ignore border fraction
        self.ensemble_k        = 10    # ensemble companions

        # Analysis
        self.ls_min_frequency     = 0.01
        self.ls_max_frequency     = 10.0
        self.ls_samples_per_peak  = 10
        self.bls_min_period       = 0.05
        self.bls_max_period       = 2.0
        self.bls_n_periods        = 1000
        self.bls_duration_min_frac= 0.01
        self.bls_duration_max_frac= 0.5
        self.bls_n_durations      = 20

        # WCS (standalone; no WIMI)
        self._wcs   = None
        self.wcs_ra = None
        self.wcs_dec= None

        # — Mode selector —
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel(self.tr("Mode:")))
        self.aligned_mode_rb = QRadioButton(self.tr("Aligned Subs"))
        self.raw_mode_rb     = QRadioButton(self.tr("Raw Subs"))
        self.aligned_mode_rb.setChecked(True)
        mg = QButtonGroup(self)
        mg.addButton(self.aligned_mode_rb); mg.addButton(self.raw_mode_rb)
        mg.buttonToggled.connect(self.on_mode_changed)
        mode_layout.addWidget(self.aligned_mode_rb)
        mode_layout.addWidget(self.raw_mode_rb)
        mode_layout.addStretch()
        self.wrench_button = QToolButton()
        self.wrench_button.setIcon(QIcon(self.wrench_path))
        self.wrench_button.setToolTip("Settings…")
        self.wrench_button.setStyleSheet("""
            QToolButton {
                background-color: #FF4500;
                color: white;
                padding: 4px;
                border-radius: 4px;
            }
            QToolButton:hover {
                background-color: #FF6347;
            }
        """)
        self.wrench_button.clicked.connect(self.open_settings)
        mode_layout.addWidget(self.wrench_button)

        # — Calibration controls (hidden in Aligned) —
        cal_layout = QHBoxLayout()
        self.load_darks_btn = QPushButton(self.tr("Load Master Dark…"))
        self.load_flats_btn = QPushButton(self.tr("Load Master Flat…"))
        for w in (self.load_darks_btn, self.load_flats_btn):
            w.clicked.connect(self.load_masters)
            w.hide()
            cal_layout.addWidget(w)
        self.dark_status_label = QLabel("Dark: ❌");  self.dark_status_label.hide()
        self.flat_status_label = QLabel("Flat: ❌");  self.flat_status_label.hide()
        cal_layout.addWidget(self.dark_status_label)
        cal_layout.addWidget(self.flat_status_label)
        cal_layout.addStretch()

        # — Status & Progress —
        self.status_label = QLabel("Ready")
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)

        # — Top controls —
        top_layout = QHBoxLayout()
        self.load_raw_btn     = QPushButton(self.tr("1: Load Raw Subs…"))
        self.load_aligned_btn = QPushButton(self.tr("Load, Measure && Photometry…"))
        self.calibrate_btn    = QPushButton(self.tr("1a: Calibrate && Align Subs"))
        self.measure_btn      = QPushButton(self.tr("2: Measure && Photometry"))
        self.load_raw_btn.    clicked.connect(self.load_raw_subs)
        self.load_aligned_btn.clicked.connect(self.load_and_measure_subs)
        self.calibrate_btn.clicked.connect(self.calibrate_and_align)
        self.measure_btn.     clicked.connect(self.detect_stars)
        self.detrend_combo = QComboBox()
        self.detrend_combo.addItems(["No Detrend", "Linear", "Quadratic"])
        self.save_aligned_btn = QPushButton("Save Aligned Frames…")
        self.save_aligned_btn.clicked.connect(self.save_aligned_frames)

        top_layout.addWidget(self.load_raw_btn)
        top_layout.addWidget(self.load_aligned_btn)
        top_layout.addWidget(self.calibrate_btn)
        top_layout.addWidget(self.measure_btn)
        top_layout.addStretch()
        top_layout.addWidget(QLabel("Detrend:"))
        top_layout.addWidget(self.detrend_combo)
        top_layout.addWidget(self.save_aligned_btn)

        # — Star list & Plot —
        middle = QHBoxLayout()
        self.star_list  = QListWidget()
        self.star_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.star_list.itemSelectionChanged.connect(self.update_plot_for_selection)
        self.star_list.setStyleSheet("""
            QListWidget::item:selected {
                background: #3399ff; 
                color: white;
            }
        """)
        middle.addWidget(self.star_list, 2)
        self.plot_widget = pg.PlotWidget(title="Light Curves")
        self.plot_widget.addLegend()
        middle.addWidget(self.plot_widget, 5)

        # — Bottom rows —
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Dip threshold (ppt):"))
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(0, 100)
        self.threshold_slider.setValue(20)
        row1.addWidget(self.threshold_slider)
        self.threshold_value_label = QLabel(f"{self.threshold_slider.value()} ppt")
        row1.addWidget(self.threshold_value_label)

        row1.addSpacing(16)
        row1.addWidget(QLabel("Temporal SNR min:"))
        self.temporal_snr_spin = QDoubleSpinBox()
        self.temporal_snr_spin.setRange(0.0, 10.0)
        self.temporal_snr_spin.setSingleStep(0.1)
        self.temporal_snr_spin.setValue(2.0)
        self.temporal_snr_spin.setDecimals(1)
        self.temporal_snr_spin.setToolTip(
            "Minimum ratio of the MA's peak deviation from 1.0\n"
            "to the MAD of residuals (raw points minus MA).\n\n"
            "This is the temporal signal-to-noise of the light curve\n"
            "variation — NOT the photometric SNR of the star itself.\n\n"
            "Higher values reject more noisy/scattered stars.\n"
            "1.5 = MA must move 1.5× further than the typical scatter.\n"
            "Set to 0.0 to disable the coherence check entirely."
        )
        self.temporal_snr_spin.valueChanged.connect(
            lambda _: self.apply_threshold(self.threshold_slider.value())
        )
        row1.addWidget(self.temporal_snr_spin)

        row1.addStretch()
        self.identify_btn = QPushButton("Identify Star…")
        self.identify_btn.clicked.connect(self.on_identify_star)
        row1.addWidget(self.identify_btn)
        self.show_ensemble_btn = QPushButton("Show Ensemble Members")
        self.show_ensemble_btn.clicked.connect(self.show_ensemble_members)
        row1.addWidget(self.show_ensemble_btn)
        self.analyze_btn = QPushButton("Analyze Star…")
        self.analyze_btn.clicked.connect(self.on_analyze)
        row1.addWidget(self.analyze_btn)

        row2 = QHBoxLayout()
        self.fetch_tesscut_btn = QPushButton("Query TESScut Light Curve")
        self.fetch_tesscut_btn.setEnabled(False)
        self.fetch_tesscut_btn.clicked.connect(self.query_tesscut)
        row2.addWidget(self.fetch_tesscut_btn)
        self.export_btn = QPushButton("Export CSV/FITS")
        self.export_btn.clicked.connect(self.export_data)
        row2.addWidget(self.export_btn)
        self.export_aavso_btn = QPushButton("Export → AAVSO")
        self.export_aavso_btn.clicked.connect(self.export_to_aavso)
        row2.addWidget(self.export_aavso_btn)

        # — Assemble —
        main = QVBoxLayout(self)
        main.addLayout(mode_layout)
        main.addLayout(cal_layout)
        main.addLayout(top_layout)
        main.addLayout(middle)
        main.addLayout(row1)
        main.addLayout(row2)
        statlay = QHBoxLayout()
        statlay.addWidget(self.status_label)
        statlay.addWidget(self.progress_bar)
        main.addLayout(statlay)

        # init
        self.on_mode_changed(self.aligned_mode_rb, True)
        self.detrend_combo.setCurrentIndex(2)
        self.on_detrend_changed(2)
        self.threshold_slider.valueChanged.connect(self._on_threshold_changed)
        self.analyze_btn.setEnabled(False)
        self.calibrate_btn.hide()

    # ---------------- UI wiring ----------------

    def _on_threshold_changed(self, v: int):
        self.threshold_value_label.setText(f"{v} ppt")
        self.apply_threshold(v)
        if hasattr(self, '_ref_overlay'):
            self._ref_overlay._update_highlights()

    def open_settings(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Photometry & Analysis Settings")
        layout = QVBoxLayout(dlg)

        photo_box = QGroupBox("Photometry")
        fb = QFormLayout(photo_box)
        self.sep_spin = QDoubleSpinBox(); self.sep_spin.setRange(1.0, 20.0); self.sep_spin.setSingleStep(0.5); self.sep_spin.setValue(self.sep_threshold)
        fb.addRow("SEP detection σ:", self.sep_spin)
        self.border_spin = QDoubleSpinBox(); self.border_spin.setRange(0.0, 0.5); self.border_spin.setSingleStep(0.01); self.border_spin.setValue(self.border_fraction)
        fb.addRow("Border fraction:", self.border_spin)
        layout.addWidget(photo_box)

        ens_box = QGroupBox("Ensemble Normalization")
        ef = QFormLayout(ens_box)
        self.ensemble_spin = QSpinBox(); self.ensemble_spin.setRange(1, 50); self.ensemble_spin.setValue(self.ensemble_k)
        ef.addRow("Comparison stars (k):", self.ensemble_spin)
        layout.addWidget(ens_box)

        ana_box = QGroupBox("Analysis (period search)")
        form = QFormLayout(ana_box)
        self.ls_samp_spin = QSpinBox(); self.ls_samp_spin.setRange(1, 100); self.ls_samp_spin.setValue(self.ls_samples_per_peak)
        form.addRow("LS samples / peak:", self.ls_samp_spin)
        self.bls_min_spin = QDoubleSpinBox(); self.bls_min_spin.setRange(0.01, 10.0); self.bls_min_spin.setValue(self.bls_min_period)
        form.addRow("BLS min period [d]:", self.bls_min_spin)
        self.bls_max_spin = QDoubleSpinBox(); self.bls_max_spin.setRange(0.01, 10.0); self.bls_max_spin.setValue(self.bls_max_period)
        form.addRow("BLS max period [d]:", self.bls_max_spin)
        self.bls_nper_spin = QSpinBox(); self.bls_nper_spin.setRange(10, 20000); self.bls_nper_spin.setValue(self.bls_n_periods)
        form.addRow("BLS # periods:", self.bls_nper_spin)
        self.bls_min_frac_spin = QDoubleSpinBox(); self.bls_min_frac_spin.setRange(0.0001, 1.0); self.bls_min_frac_spin.setSingleStep(0.001); self.bls_min_frac_spin.setValue(self.bls_duration_min_frac)
        form.addRow("BLS min dur frac:", self.bls_min_frac_spin)
        self.bls_max_frac_spin = QDoubleSpinBox(); self.bls_max_frac_spin.setRange(0.01, 1.0); self.bls_max_frac_spin.setSingleStep(0.01); self.bls_max_frac_spin.setValue(self.bls_duration_max_frac)
        form.addRow("BLS max dur frac:", self.bls_max_frac_spin)
        self.bls_ndur_spin = QSpinBox(); self.bls_ndur_spin.setRange(1, 200); self.bls_ndur_spin.setValue(self.bls_n_durations)
        form.addRow("BLS # durations:", self.bls_ndur_spin)
        layout.addWidget(ana_box)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        layout.addWidget(btns)

        if dlg.exec() == QDialog.DialogCode.Accepted:
            self.sep_threshold   = self.sep_spin.value()
            self.border_fraction = self.border_spin.value()
            self.ensemble_k      = self.ensemble_spin.value()
            self.ls_samples_per_peak = self.ls_samp_spin.value()
            self.bls_min_period        = self.bls_min_spin.value()
            self.bls_max_period        = self.bls_max_spin.value()
            self.bls_n_periods         = self.bls_nper_spin.value()
            self.bls_duration_min_frac = self.bls_min_frac_spin.value()
            self.bls_duration_max_frac = self.bls_max_frac_spin.value()
            self.bls_n_durations       = self.bls_ndur_spin.value()

    def on_mode_changed(self, button, checked):
        is_raw = checked and (button is self.raw_mode_rb)
        for w in (
            self.load_raw_btn,
            self.load_darks_btn,
            self.load_flats_btn,
            self.dark_status_label,
            self.flat_status_label,
            self.calibrate_btn,
        ):
            w.setVisible(is_raw)
        self.load_aligned_btn.setVisible(not is_raw)
        self.measure_btn.setVisible(is_raw)

    def load_and_measure_subs(self):
        before = len(getattr(self, "image_paths", []))
        self.load_aligned_subs()
        after = len(getattr(self, "image_paths", []))
        if after == 0 or after == before and not self._cached_images:
            return
        self.detect_stars()
    # --------------- I/O + Calibration ----------------

    def load_raw_subs(self):
        settings = QSettings()
        start_dir = settings.value("ExoPlanet/lastRawFolder", os.path.expanduser("~"), type=str)
        paths, _ = QFileDialog.getOpenFileNames(self, "Select Raw Frames", start_dir, "FITS, TIFF or XISF (*.fit *.fits *.tif *.tiff *.xisf)")
        if not paths: return
        settings.setValue("ExoPlanet/lastRawFolder", os.path.dirname(paths[0]))

        self.status_label.setText("Reading headers…")
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(len(paths))
        self.progress_bar.setValue(0)
        QApplication.processEvents()

        datelist = []
        for i, p in enumerate(paths, start=1):
            ext = os.path.splitext(p)[1].lower()
            ds = None
            if ext == '.xisf':
                try:
                    xisf     = XISF(p)
                    img_meta = xisf.get_images_metadata()[0].get('FITSKeywords', {})
                    if 'DATE-OBS' in img_meta:
                        ds = img_meta['DATE-OBS'][0]['value']
                except Exception:
                    ds = None
            elif ext in ('.fit', '.fits', '.fz'):
                try:
                    hdr0, _ = get_valid_header(p)
                    ds      = hdr0.get('DATE-OBS')
                except Exception:
                    ds = None

            # Use robust header time parsing (UT-OBS -> DATE-OBS -> MJD-OBS fallback)
            t = None
            try:
                if ext == ".xisf":
                    # Build a tiny dict-like header for the helper
                    hdr_like = {}
                    try:
                        xisf     = XISF(p)
                        img_meta = xisf.get_images_metadata()[0]
                        kw       = img_meta.get("FITSKeywords", {}) or {}
                        # XISF FITSKeywords layout: key -> [ {value: ...}, ... ]
                        for k in ("UT-OBS", "DATE-OBS", "DATE-END", "MJD-OBS"):
                            if k in kw and kw[k]:
                                hdr_like[k] = kw[k][0].get("value")
                    except Exception:
                        hdr_like = {}
                    t = _parse_obs_time_from_header(hdr_like)

                elif ext in (".fit", ".fits", ".fz"):
                    hdr0, _ = get_valid_header(p)
                    t = _parse_obs_time_from_header(hdr0)

                else:
                    # TIFF etc. may not have FITS-like time headers; leave None
                    t = None

            except Exception as e:
                print(f"[DEBUG] Failed to parse obs time for {p}: {e}")

            datelist.append((p, t))

            self.progress_bar.setValue(i)
            QApplication.processEvents()

        datelist.sort(key=lambda x: (x[1] is None, x[1] or x[0]))
        sorted_paths = [p for p, _ in datelist]

        self.image_paths     = sorted_paths
        self._cached_images  = []
        self._cached_headers = []
        self.airmasses       = []
        self.star_list.clear()
        self.plot_widget.clear()

        self.status_label.setText("Loading raw frames…")
        self.progress_bar.setMaximum(len(sorted_paths))
        self.progress_bar.setValue(0)
        QApplication.processEvents()

        for i, p in enumerate(sorted_paths, start=1):
            self.status_label.setText(f"Loading raw frame {i}/{len(sorted_paths)}…")
            QApplication.processEvents()
            img, hdr, bit_depth, is_mono = load_image(p)
            if img is None:
                QMessageBox.warning(self, "Load Error", f"Failed to load raw frame:\n{os.path.basename(p)}")
                self._cached_images.append(None)
                self._cached_headers.append(None)
                am = 1.0
            else:
                img_binned = bin2x2_numba(img)
                self._cached_images.append(img_binned)
                self._cached_headers.append(hdr)

                if self.exposure_time is None:
                    if isinstance(hdr, fits.Header):
                        self.exposure_time = hdr.get('EXPOSURE', hdr.get('EXPTIME', None))
                    elif isinstance(hdr, dict):
                        img_meta = hdr.get('image_meta', {}) or {}
                        fits_kw  = img_meta.get('FITSKeywords', {})
                        val = None
                        if 'EXPOSURE' in fits_kw: val = fits_kw['EXPOSURE'][0].get('value')
                        elif 'EXPTIME' in fits_kw: val = fits_kw['EXPTIME'][0].get('value')
                        try:
                            self.exposure_time = float(val)
                        except (ValueError, TypeError):
                            print(f"[DEBUG] Could not parse exposure_time={val!r}")

                am = None
                if isinstance(hdr, fits.Header):
                    if 'AIRMASS' in hdr:
                        try: am = float(hdr['AIRMASS'])
                        except ValueError: am = None
                    if am is None:
                        alt = (hdr.get('OBJCTALT') or hdr.get('ALT') or hdr.get('ALTITUDE') or hdr.get('EL'))
                        try: am = self.estimate_airmass_from_altitude(float(alt))
                        except (ValueError, TypeError): am = 1.0
                elif isinstance(hdr, dict):
                    img_meta = hdr.get('image_meta', {}) or {}
                    fits_kw  = img_meta.get('FITSKeywords', {})
                    if 'AIRMASS' in fits_kw:
                        try: am = float(fits_kw['AIRMASS'][0]['value'])
                        except (ValueError, TypeError): am = None
                    if am is None:
                        for key in ('OBJCTALT','ALT','ALTITUDE','EL'):
                            ent = fits_kw.get(key)
                            if ent:
                                try:
                                    am = self.estimate_airmass_from_altitude(float(ent[0]['value']))
                                    break
                                except Exception:
                                    pass  # Ignore airmass estimation errors
                        else:
                            am = 1.0
                if am is None:
                    am = 1.0

            self.airmasses.append(am)
            self.progress_bar.setValue(i)
            QApplication.processEvents()

        # Keep full timestamps (DO NOT truncate to date-only)
        tlist = [t for _, t in datelist if t is not None]
        if tlist:
            self.times = Time(tlist)   # already utc scale from helper
        else:
            self.times = None

        self.progress_bar.setVisible(False)
        loaded = sum(1 for im in self._cached_images if im is not None)
        self.status_label.setText(f"Loaded {loaded}/{len(sorted_paths)} raw frames")

    def load_aligned_subs(self) -> bool:
        settings = QSettings()
        start_dir = settings.value("ExoPlanet/lastAlignedFolder", os.path.expanduser("~"), type=str)
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Aligned Frames", start_dir,
            "FITS or TIFF (*.fit *.fits *.tif *.tiff *.xisf)"
        )
        if not paths:
            self.status_label.setText("Load canceled.")
            return False

        settings.setValue("ExoPlanet/lastAlignedFolder", os.path.dirname(paths[0]))

        self.status_label.setText("Reading metadata from aligned frames…")
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(len(paths))
        self.progress_bar.setValue(0)
        QApplication.processEvents()

        datelist = []
        for i, p in enumerate(paths, start=1):
            ext = os.path.splitext(p)[1].lower(); ds = None
            if ext == '.xisf':
                try:
                    xisf     = XISF(p)
                    img_meta = xisf.get_images_metadata()[0]
                    kw       = img_meta.get('FITSKeywords', {})
                    if 'DATE-OBS' in kw: ds = kw['DATE-OBS'][0]['value']
                except Exception: ds = None
            elif ext in ('.fit', '.fits', '.fz'):
                try:
                    hdr0, _ = get_valid_header(p)
                    ds      = hdr0.get('DATE-OBS')
                except Exception: ds = None
            # Use robust header time parsing (UT-OBS -> DATE-OBS -> MJD-OBS fallback)
            t = None
            try:
                if ext == ".xisf":
                    # Build a tiny dict-like header for the helper
                    hdr_like = {}
                    try:
                        xisf     = XISF(p)
                        img_meta = xisf.get_images_metadata()[0]
                        kw       = img_meta.get("FITSKeywords", {}) or {}
                        # XISF FITSKeywords layout: key -> [ {value: ...}, ... ]
                        for k in ("UT-OBS", "DATE-OBS", "DATE-END", "MJD-OBS"):
                            if k in kw and kw[k]:
                                hdr_like[k] = kw[k][0].get("value")
                    except Exception:
                        hdr_like = {}
                    t = _parse_obs_time_from_header(hdr_like)

                elif ext in (".fit", ".fits", ".fz"):
                    hdr0, _ = get_valid_header(p)
                    t = _parse_obs_time_from_header(hdr0)

                else:
                    # TIFF etc. may not have FITS-like time headers; leave None
                    t = None

            except Exception as e:
                print(f"[DEBUG] Failed to parse obs time for {p}: {e}")

            datelist.append((p, t))
            self.progress_bar.setValue(i)
            QApplication.processEvents()

        datelist.sort(key=lambda x: (x[1] is None, x[1] or x[0]))
        sorted_paths = [p for p, _ in datelist]

        self.image_paths     = sorted_paths
        self._cached_images  = []
        self._cached_headers = []
        self.airmasses       = []

        self.status_label.setText("Loading aligned frames…")
        self.progress_bar.setMaximum(len(sorted_paths))
        self.progress_bar.setValue(0)
        QApplication.processEvents()

        for i, p in enumerate(sorted_paths, start=1):
            self.status_label.setText(f"Loading frame {i}/{len(sorted_paths)}…")
            QApplication.processEvents()
            img, hdr, bit_depth, is_mono = load_image(p)
            if img is None:
                QMessageBox.warning(self, "Load Error", f"Failed to load aligned frame:\n{os.path.basename(p)}")
                self._cached_images.append(None)
                self._cached_headers.append(None)
                am = 1.0
            else:
                img_binned = bin2x2_numba(img)
                self._cached_images.append(img_binned)
                self._cached_headers.append(hdr)

                if self.exposure_time is None:
                    if isinstance(hdr, fits.Header):
                        self.exposure_time = hdr.get('EXPOSURE', hdr.get('EXPTIME', None))
                    elif isinstance(hdr, dict):
                        img_meta = hdr.get('image_meta', {}) or {}
                        fits_kw  = img_meta.get('FITSKeywords', {})
                        val = None
                        if 'EXPOSURE' in fits_kw: val = fits_kw['EXPOSURE'][0].get('value')
                        elif 'EXPTIME' in fits_kw: val = fits_kw['EXPTIME'][0].get('value')
                        try: self.exposure_time = float(val)
                        except (ValueError, TypeError): print(f"[DEBUG] Could not parse exposure_time={val!r}")

                am = None
                if isinstance(hdr, fits.Header):
                    if 'AIRMASS' in hdr:
                        try: am = float(hdr['AIRMASS'])
                        except ValueError: am = None
                    if am is None:
                        alt = (hdr.get('OBJCTALT') or hdr.get('ALT') or hdr.get('ALTITUDE') or hdr.get('EL'))
                        try: am = self.estimate_airmass_from_altitude(float(alt))
                        except (ValueError, TypeError): am = 1.0
                elif isinstance(hdr, dict):
                    img_meta = hdr.get('image_meta', {}) or {}
                    fits_kw  = img_meta.get('FITSKeywords', {})
                    if 'AIRMASS' in fits_kw:
                        try: am = float(fits_kw['AIRMASS'][0]['value'])
                        except (ValueError, TypeError): am = None
                    if am is None:
                        for key in ('OBJCTALT','ALT','ALTITUDE','EL'):
                            ent = fits_kw.get(key)
                            if ent:
                                try:
                                    am = self.estimate_airmass_from_altitude(float(ent[0]['value']))
                                    break
                                except Exception:
                                    pass  # Ignore airmass estimation errors
                        else:
                            am = 1.0
                else:
                    am = 1.0

            self.airmasses.append(am)
            self.progress_bar.setValue(i)
            QApplication.processEvents()

        # Keep full timestamps (DO NOT truncate to date-only)
        tlist = [t for _, t in datelist if t is not None]
        if tlist:
            self.times = Time(tlist)   # already utc scale from helper
        else:
            self.times = None

        self.progress_bar.setVisible(False)
        loaded = sum(1 for im in self._cached_images if im is not None)
        self.status_label.setText(f"Loaded {loaded}/{len(sorted_paths)} aligned frames")
        return loaded > 0

    def load_masters(self):
        settings = QSettings()
        last_master_dir = settings.value("ExoPlanet/lastMasterFolder", os.path.expanduser("~"), type=str)
        sender = self.sender()
        dlg = QFileDialog(self, "Select Master File", last_master_dir, "FITS, TIFF or XISF (*.fit *.fits *.tif *.tiff *.xisf)")
        dlg.setFileMode(QFileDialog.FileMode.ExistingFile)
        if not dlg.exec(): return
        path = dlg.selectedFiles()[0]
        settings.setValue("ExoPlanet/lastMasterFolder", os.path.dirname(path))

        img, hdr, bit_depth, is_mono = load_image(path)
        if img is None:
            QMessageBox.warning(self, "Load Error", f"Failed to load master file:\n{path}")
            return

        img = img.astype(np.float32)
        binned = bin2x2_numba(img)

        if "Dark" in sender.text():
            if self.master_flat is not None and not self._shapes_compatible(binned, self.master_flat):
                QMessageBox.warning(self, "Shape Mismatch", "This master dark (binned) doesn’t match your existing flat.")
                return
            self.master_dark = binned
            self.dark_status_label.setText("Dark: ✅")
            self.dark_status_label.setStyleSheet("color: #00cc66; font-weight: bold;")
        else:
            if self.master_dark is not None and not self._shapes_compatible(self.master_dark, binned):
                QMessageBox.warning(self, "Shape Mismatch", "This master flat (binned) doesn’t match your existing dark.")
                return
            self.master_flat = binned
            self.flat_status_label.setText("Flat: ✅")
            self.flat_status_label.setStyleSheet("color: #00cc66; font-weight: bold;")

    def _try_wcs_from_header(self, hdr, plane):
        H = self._get_best_header_for_wcs(hdr)
        if H is None:
            return None

        try:
            w = WCS(H, naxis=2, relax=True)
            if not w.has_celestial:
                return None

            # sanity check by transforming image center
            h, wimg = plane.shape[:2]
            sky = w.pixel_to_world(wimg / 2, h / 2)
            _ = float(sky.ra.deg), float(sky.dec.deg)
            return w
        except Exception:
            return None

    def _shapes_compatible(self, master: np.ndarray, other: np.ndarray) -> bool:
        if master.shape == other.shape:
            return True
        if master.ndim == 2 and other.ndim == 3 and other.shape[:2] == master.shape:
            return True
        if other.ndim == 2 and master.ndim == 3 and master.shape[:2] == other.shape:
            return True
        return False

    def calibrate_and_align(self):
        if not self._cached_images:
            QMessageBox.warning(self, "Calibrate", "Load raw subs first.")
            return
        self.status_label.setText("Calibrating & aligning frames…")
        self.progress_bar.setVisible(True)
        n = len(self._cached_images)
        self.progress_bar.setMaximum(n)

        reference_image_2d = None
        for i, (img, hdr) in enumerate(zip(self._cached_images, self._cached_headers), start=1):
            if self.master_dark is not None:
                img = img.astype(np.float32) - self.master_dark
            if self.master_flat is not None:
                img = apply_flat_division_numba(img, self.master_flat)
            if img.ndim == 2:
                img = np.stack([img, img, img], axis=2)

            plane = img if img.ndim == 2 else img.mean(axis=2)

            if reference_image_2d is None:
                reference_image_2d = plane.copy()

            delta = StarRegistrationWorker.compute_affine_transform_astroalign(plane, reference_image_2d)
            if delta is None:
                delta = IDENTITY_2x3

            img_aligned = StarRegistrationThread.apply_affine_transform_static(img, delta)
            self._cached_images[i-1] = img_aligned
            self.progress_bar.setValue(i)
            QApplication.processEvents()

        self.progress_bar.setVisible(False)
        self.status_label.setText("Calibration & alignment complete")

    def save_aligned_frames(self):
        if not self._cached_images:
            QMessageBox.warning(self, "Save Aligned Frames", "No images to save. Run Calibrate & Align first.")
            return
        out_dir = QFileDialog.getExistingDirectory(self, "Choose Output Folder")
        if not out_dir:
            return
        for i, orig_path in enumerate(self.image_paths):
            img = self._cached_images[i]
            ext = os.path.splitext(orig_path)[1].lstrip(".").lower()
            fmt = ext if ext in ("fits","fit","tiff","tif","xisf","png","jpg","jpeg") else "fits"
            hdr = self._cached_headers[i] if hasattr(self, "_cached_headers") and i < len(self._cached_headers) else None
            base = os.path.splitext(os.path.basename(orig_path))[0]
            out_name = f"{base}_aligned.{fmt}"
            out_path = os.path.join(out_dir, out_name)
            save_image(
                img_array=img, filename=out_path, original_format=fmt, bit_depth=None,
                original_header=hdr, is_mono=(img.ndim==2), image_meta=None, file_meta=None
            )
        QMessageBox.information(self, "Save Complete", f"Saved {len(self._cached_images)} aligned frames to:\n{out_dir}")

    # --------------- Detection + Photometry ----------------
    def _seed_header_for_astap(self, ref_idx: int) -> fits.Header | None:
        """
        Build a *real* FITS header to seed ASTAP, harvested from the reference
        frame's original header. We preserve RA/Dec (OBJCTRA/OBJCTDEC or CRVAL1/2),
        size (NAXIS*), basic camera hints (PIXSZ, BINNING, FOCALLEN) if present.
        """
        if not (0 <= ref_idx < len(self._cached_headers)):
            return None

        src = self._cached_headers[ref_idx]
        H = fits.Header()

        # try to copy directly if it's already a FITS Header
        if isinstance(src, fits.Header):
            H = src.copy()
        elif isinstance(src, dict):
            # Could be a nested XISF-like dict. Look for FITSKeywords first.
            kw = None
            if "image_meta" in src and isinstance(src["image_meta"], dict):
                kw = src["image_meta"].get("FITSKeywords", None)
            if kw is None:
                kw = src.get("FITSKeywords", None)
            if isinstance(kw, dict):
                for k, v in kw.items():
                    try:
                        # XISF stores [{'value':...}], FITS-like dicts store raw
                        if isinstance(v, list):
                            vv = v[0].get("value", None)
                        else:
                            vv = v
                        if vv is not None:
                            H[k] = vv
                    except Exception:
                        pass
            else:
                # best-effort: flat dict of scalars
                for k, v in src.items():
                    try:
                        H[k] = v
                    except Exception:
                        pass

        # be sure image size is present (ASTAP likes NAXIS1/2)
        if self._cached_images and self._cached_images[ref_idx] is not None:
            img = self._cached_images[ref_idx]
            h, w = (img.shape if img.ndim == 2 else img.shape[:2])
            H.setdefault("NAXIS", 2)
            H["NAXIS1"] = w
            H["NAXIS2"] = h

        # If we only have OBJCTRA/OBJCTDEC strings, just leave them — your
        # _build_astap_seed() handles these. If we *also* have CRVAL*, keep them.
        # Do NOT inject WCS—we want ASTAP to solve, not be constrained by stale WCS.
        # (_solve_numpy_with_astap will strip WCS keys before writing temp FITS)

        # optional: exposure time
        if self.exposure_time is not None:
            H.setdefault("EXPTIME", float(self.exposure_time))

        return H if len(H) else None

    def _coerce_seed_header(self, hdr_in, plane) -> fits.Header:
        """
        Make a real fits.Header usable by _build_astap_seed():
        - copy fields from FITS, XISF-like dicts, or flat dicts
        - ensure NAXIS/NAXIS1/NAXIS2
        - try to expose RA/Dec in a form the seeder can use
            (OBJCTRA/OBJCTDEC strings and/or CRVAL1/CRVAL2 degrees).
        """
        H = fits.Header()

        # 1) copy what we can
        if isinstance(hdr_in, fits.Header):
            H = hdr_in.copy()
        elif isinstance(hdr_in, dict):
            # XISF-style nested?
            kw = None
            if "image_meta" in hdr_in and isinstance(hdr_in["image_meta"], dict):
                kw = hdr_in["image_meta"].get("FITSKeywords")
            if kw is None:
                kw = hdr_in.get("FITSKeywords")
            if isinstance(kw, dict):
                for k, v in kw.items():
                    try:
                        vv = v[0]["value"] if isinstance(v, list) else v
                        if vv is not None:
                            H[k] = vv
                    except Exception:
                        pass
            else:
                # flat dict of scalars
                for k, v in hdr_in.items():
                    try:
                        H[k] = v
                    except Exception:
                        pass

        # 2) ensure image size
        h, w = (plane.shape if plane.ndim == 2 else plane.shape[:2])
        H.setdefault("NAXIS", 2)
        H["NAXIS1"] = int(w)
        H["NAXIS2"] = int(h)

        # 3) try to normalize RA/Dec
        # If OBJCTRA/OBJCTDEC are present as strings, keep them.
        # Otherwise, if we find numeric RA/DEC/CRVAL*, ensure CRVAL1/2 are set.
        def _try_deg(val):
            try:
                return float(val)
            except Exception:
                return None

        ra_deg = None
        dec_deg = None

        # prefer CRVAL1/2 if they seem finite
        ra_deg = _try_deg(H.get("CRVAL1"))
        dec_deg = _try_deg(H.get("CRVAL2"))

        if ra_deg is None or dec_deg is None:
            # common alternates
            for rakey in ("RA", "OBJCTRA", "OBJRA"):
                if rakey in H:
                    try:
                        # could be sexagesimal string
                        ra_deg = SkyCoord(H[rakey], H.get("OBJCTDEC", None) or H.get("DEC", None), unit=(u.hourangle, u.deg)).ra.deg
                        dec_deg = SkyCoord(H[rakey], H.get("OBJCTDEC", None) or H.get("DEC", None), unit=(u.hourangle, u.deg)).dec.deg
                        break
                    except Exception:
                        pass
            if ra_deg is None and "RA" in H and "DEC" in H:
                ra_deg = _try_deg(H["RA"])
                dec_deg = _try_deg(H["DEC"])

        # Set CRVAL* if we have clean degrees
        if ra_deg is not None and dec_deg is not None:
            H["CRVAL1"] = float(ra_deg)
            H["CRVAL2"] = float(dec_deg)

            # Also supply OBJCTRA/OBJCTDEC sexagesimal (helps some ASTAP setups)
            try:
                c = SkyCoord(ra_deg*u.deg, dec_deg*u.deg, frame="icrs")
                H.setdefault("OBJCTRA",  c.ra.to_string(unit=u.hour, sep=":", precision=2, pad=True))
                H.setdefault("OBJCTDEC", c.dec.to_string(unit=u.deg, sep=":", precision=1, pad=True, alwayssign=True))
            except Exception:
                pass

        # Optional: if you have pixel size / focal length / binning in the original
        # header, leaving them in place is good; _build_astap_seed will use them.

        return H

    def _get_best_header_for_wcs(self, hdr):
        if isinstance(hdr, fits.Header):
            return hdr

        if isinstance(hdr, dict):
            # first try canonical SASpro metadata keys
            for k in ("wcs_header", "original_header", "header", "fits_header"):
                h = hdr.get(k)
                if isinstance(h, fits.Header):
                    return h

            # XISF-style FITSKeywords fallback
            kw = None
            if "image_meta" in hdr and isinstance(hdr["image_meta"], dict):
                kw = hdr["image_meta"].get("FITSKeywords")
            if kw is None:
                kw = hdr.get("FITSKeywords")

            if isinstance(kw, dict):
                H = fits.Header()
                for k, v in kw.items():
                    try:
                        vv = v[0]["value"] if isinstance(v, list) else v
                        if vv is not None:
                            H[k] = vv
                    except Exception:
                        pass
                return H if len(H) else None

        return None

    def _try_existing_wcs(self, hdr):
        H = self._get_best_header_for_wcs(hdr)
        if H is None:
            return None
        try:
            w = WCS(H, naxis=2, relax=True)
            if not w.has_celestial:
                return None
            return w
        except Exception:
            return None


    def detect_stars(self):
        self.status_label.setText("Measuring frames…")
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(len(self.image_paths))
        self.progress_bar.setValue(0)

        # 0) ensure frames are cached
        if not hasattr(self, "_cached_images") or len(self._cached_images) != len(self.image_paths):
            self._cached_images = [load_image(p)[0] for p in self.image_paths]

        n_frames = len(self._cached_images)
        self.progress_bar.setMaximum(n_frames)
        self.progress_bar.setValue(0)

        # --- PASS 1: per-frame background & SEP stats (parallel) ---
        def _process_frame(idx, img):
            plane = img.mean(axis=2) if img.ndim == 3 else img
            mean, med, std = sigma_clipped_stats(plane)
            zeroed = plane - med
            bkg    = sep.Background(zeroed)
            bkgmap = bkg.back()
            rmsmap = bkg.rms()
            data_sub = zeroed - bkgmap

            # keep arrays tight for SEP
            data_sub = np.ascontiguousarray(data_sub.astype(np.float32, copy=False))
            rmsmap   = np.ascontiguousarray(rmsmap.astype(np.float32, copy=False))

            try:
                objs = sep.extract(
                    data_sub, thresh=self.sep_threshold, err=rmsmap,
                    minarea=16, deblend_nthresh=32, clean=True
                )
            except Exception:
                objs = None

            if objs is None or len(objs) == 0:
                sc = 0; avg_fwhm = 0.0; avg_ecc = 0.0
            else:
                sc = len(objs)
                a = np.clip(objs['a'], 1e-3, None)
                b = np.clip(objs['b'], 1e-3, None)
                fwhm_vals = 2.3548 * np.sqrt(a * b)
                ecc_vals  = np.sqrt(1.0 - np.clip(b / a, 0, 1)**2)
                avg_fwhm  = float(np.nanmean(fwhm_vals))
                avg_ecc   = float(np.nanmean(ecc_vals))

            stats = {"star_count": sc, "eccentricity": avg_ecc,
                    "mean": float(np.mean(plane)), "fwhm": avg_fwhm}
            return idx, data_sub, objs, rmsmap, stats

        cpu_cnt   = multiprocessing.cpu_count()
        n_workers = max(1, int(cpu_cnt * 0.8))

        frame_data = {}
        stats_map  = {}
        with ThreadPoolExecutor(max_workers=n_workers) as exe:
            futures = [exe.submit(_process_frame, idx, img)
                    for idx, img in enumerate(self._cached_images)]
            done = 0
            for fut in as_completed(futures):
                idx, data_sub, objs, rmsmap, stats = fut.result()
                frame_data[idx] = (data_sub, objs, rmsmap)
                stats_map[idx]  = stats
                done += 1
                self.progress_bar.setValue(done)
                self.status_label.setText(f"Measured frame {done}/{n_frames}")

        # pick best reference
        def quality(i):
            s = stats_map[i]
            return s["star_count"] / (s["fwhm"] * s["mean"] + 1e-8)
        ref_idx   = max(stats_map.keys(), key=quality)
        ref_stats = stats_map[ref_idx]

        # --- Solve WCS on reference (unchanged) ---
        self.ref_idx = ref_idx
        ref_img = self._cached_images[ref_idx]
        plane = ref_img.mean(axis=2) if ref_img.ndim == 3 else ref_img
        hdr = self._cached_headers[ref_idx]
        self._solve_reference(plane, hdr)


        # --- SEP catalog on reference ---
        data_ref, objs_ref, rms_ref = frame_data[ref_idx]
        if objs_ref is None or len(objs_ref) == 0:
            QMessageBox.warning(self, "No Stars", "No stars found in reference frame.")
            self.progress_bar.setVisible(False)
            return

        xs = objs_ref['x']; ys = objs_ref['y']
        h, w = data_ref.shape
        bf = self.border_fraction
        keep_border = ((xs > w*bf) & (xs < w*(1-bf)) & (ys > h*bf) & (ys < h*(1-bf)))
        xs = np.ascontiguousarray(xs[keep_border].astype(np.float32, copy=False))
        ys = np.ascontiguousarray(ys[keep_border].astype(np.float32, copy=False))

        self.median_fwhm = ref_stats["fwhm"]
        aper_r = float(max(2.5, 1.5 * self.median_fwhm))

        # --- PASS 2: aperture + annulus sky subtraction on all frames --------
        n_stars  = len(xs)
        n_frames = len(self._cached_images)
        raw_flux     = np.empty((n_stars, n_frames), dtype=np.float32)
        raw_flux_err = np.empty((n_stars, n_frames), dtype=np.float32)
        flags        = np.zeros((n_stars, n_frames), dtype=np.int16)

        # Per-star FWHM from reference frame for adaptive aperture
        # Fall back to median_fwhm if per-star values aren't available
        if objs_ref is not None and len(objs_ref) > 0:
            a_ref = np.clip(objs_ref['a'][keep_border], 1e-3, None)
            b_ref = np.clip(objs_ref['b'][keep_border], 1e-3, None)
            per_star_fwhm = 2.3548 * np.sqrt(a_ref * b_ref)
        else:
            per_star_fwhm = np.full(n_stars, self.median_fwhm, dtype=np.float32)

        # Clamp per-star FWHM to a sane range around the median
        fwhm_med = float(np.nanmedian(per_star_fwhm))
        fwhm_med = max(fwhm_med, 1.5)
        per_star_fwhm = np.clip(per_star_fwhm, 0.5 * fwhm_med, 3.0 * fwhm_med)

        # Aperture and annulus radii based on per-star FWHM
        # aperture  : 1.5 × FWHM  (captures ~95% of flux for Gaussian PSF)
        # annulus in: 3.0 × FWHM  (clear of any PSF wings)
        # annulus out: 5.0 × FWHM (enough sky pixels for good median estimate)
        aper_r   = np.maximum(2.5, 1.5 * per_star_fwhm).astype(np.float64)
        ann_r_in  = np.maximum(aper_r + 1.0, 3.0 * per_star_fwhm).astype(np.float64)
        ann_r_out = np.maximum(ann_r_in + 2.0, 5.0 * per_star_fwhm).astype(np.float64)

        self.status_label.setText("Computing aperture photometry with annulus sky subtraction…")
        self.progress_bar.setMaximum(n_frames)
        self.progress_bar.setValue(0)

        def _sum_frame(t: int):
            data_sub, _objs, rmsmap = frame_data[t]
            ds = np.maximum(data_sub, -1.0 * rmsmap)

            fl   = np.zeros(n_stars, dtype=np.float64)
            ferr = np.zeros(n_stars, dtype=np.float64)
            flg  = np.zeros(n_stars, dtype=np.int16)

            for si in range(n_stars):
                xsi = float(xs[si])
                ysi = float(ys[si])
                r   = float(aper_r[si])
                rin  = float(ann_r_in[si])
                rout = float(ann_r_out[si])

                # --- aperture sum ---
                try:
                    f_ap, f_err, f_flag = sep.sum_circle(
                        ds, [xsi], [ysi], r, err=rmsmap,
                        gain=1.0,
                    )
                    f_ap   = float(f_ap[0])
                    f_err  = float(f_err[0])
                    f_flag = int(f_flag[0])
                except Exception:
                    fl[si] = np.nan; ferr[si] = np.nan; flg[si] = 1
                    continue

                # --- annulus sky estimate (robust median × aperture area) ---
                try:
                    ann_flux, ann_err, ann_flag = sep.sum_circann(
                        ds, [xsi], [ysi], rin, rout, err=rmsmap,
                    )
                    ann_area = sep.circann_area(rin, rout)
                    ap_area  = np.pi * r * r

                    # sky per pixel from annulus (median is more robust than mean)
                    sky_per_pix = float(ann_flux[0]) / max(ann_area, 1.0)

                    # subtract local sky from aperture
                    sky_total = sky_per_pix * ap_area
                    f_sky_sub = f_ap - sky_total

                    # propagate sky subtraction error
                    # var(sky_total) = (ap_area/ann_area)^2 * var(ann)
                    sky_err_contrib = float(ann_err[0]) * (ap_area / max(ann_area, 1.0))
                    f_err_total = float(np.sqrt(max(f_err**2 + sky_err_contrib**2, 0.0)))

                    fl[si]   = f_sky_sub
                    ferr[si] = f_err_total
                    flg[si]  = f_flag | int(ann_flag[0])

                except Exception:
                    # annulus failed — fall back to global-background-subtracted value
                    fl[si]   = f_ap
                    ferr[si] = f_err
                    flg[si]  = f_flag

            return t, fl.astype(np.float32), ferr.astype(np.float32), flg.astype(np.int16)

        done = 0
        with ThreadPoolExecutor(max_workers=n_workers) as exe:
            for t, fl, ferr, flg in exe.map(_sum_frame, range(n_frames)):
                raw_flux[:, t]     = fl
                raw_flux_err[:, t] = ferr
                flags[:, t]        = flg
                done += 1
                if (done % 4) == 0 or done == n_frames:
                    self.progress_bar.setValue(done)

        # --- ENSEMBLE NORMALIZATION (safe masks + unit-median renorm) ---
        n_stars, n_frames = raw_flux.shape
        star_refs = np.nanmedian(raw_flux, axis=1)
        rel_flux = np.full_like(raw_flux, np.nan, dtype=np.float32)
        rel_err  = np.full_like(raw_flux_err, np.nan, dtype=np.float32)

        k = int(self.ensemble_k)
        k = max(1, min(k, max(1, n_stars - 1)))  # keep in range
        self.ensemble_map = {}

        for i in range(n_stars):
            diffs = np.abs(star_refs - star_refs[i])
            diffs[i] = np.inf
            neigh = np.argpartition(diffs, k)[:k]
            self.ensemble_map[i] = list(np.asarray(neigh, dtype=int))

            ens_flux = np.nanmedian(raw_flux[neigh, :], axis=0)
            ens_err  = np.sqrt(np.nansum(raw_flux_err[neigh, :]**2, axis=0)) / np.sqrt(len(neigh))

            mask = (raw_flux[i] > 0) & (ens_flux > 0) & np.isfinite(raw_flux[i]) & np.isfinite(ens_flux)
            if not np.any(mask):
                continue

            rel_flux[i, mask] = raw_flux[i, mask] / ens_flux[mask]
            with np.errstate(divide='ignore', invalid='ignore'):
                term1 = raw_flux_err[i, mask] / raw_flux[i, mask]
                term2 = ens_err[mask]         / ens_flux[mask]
                rel_err[i, mask] = rel_flux[i, mask] * np.sqrt(term1**2 + term2**2)

        # unit-median renorm so curves are centered ~1.0
        meds = np.nanmedian(rel_flux, axis=1)
        good = (meds > 0) & np.isfinite(meds)
        rel_flux[good] /= meds[good, None]
        rel_err[good]  /= meds[good, None]

        self.fluxes      = rel_flux
        self.flux_errors = rel_err
        self.flags       = flags

        # --- detrend (then re-center) ---
        if self.detrend_degree is not None:
            n_stars = rel_flux.shape[0]
            self.status_label.setText("Detrending curves…")
            self.progress_bar.setVisible(True)
            self.progress_bar.setMaximum(n_stars)
            self.progress_bar.setValue(0)
            for i in range(n_stars):
                curve = rel_flux[i].copy()
                goodm = np.isfinite(curve) & (curve > 0)
                rel_flux[i] = self._detrend_curve(curve, self.detrend_degree, mask=goodm)
                self.progress_bar.setValue(i+1)
            self.progress_bar.setVisible(False)
            self.status_label.setText("Detrending complete")

            meds = np.nanmedian(rel_flux, axis=1)
            good = (meds > 0) & np.isfinite(meds)
            rel_flux[good] /= meds[good, None]

        # --- robust per-star outlier flagging ---
        for i in range(n_stars):
            curve = rel_flux[i, :]
            med_i = np.nanmedian(curve)
            mad_i = np.nanmedian(np.abs(curve - med_i))
            sigma_i = 1.4826 * mad_i if mad_i > 0 else np.nanstd(curve)
            if sigma_i > 0:
                outlier_mask = np.abs(curve - med_i) > 2 * sigma_i
                flags[i, outlier_mask] = 1

        # --- drop stars with too many flagged frames ---
        good_counts = np.sum(flags == 0, axis=1)
        keep        = good_counts >= (0.75 * n_frames)
        xs, ys      = xs[keep], ys[keep]
        rel_flux    = rel_flux[keep, :]
        flags       = flags[keep, :]

        self.star_positions = list(zip(xs, ys))
        self.fluxes         = rel_flux.copy()
        self.flags          = flags

        # list uses median rel flux, not the first frame
        self.star_list.clear()
        for i, (x, y) in enumerate(self.star_positions):
            fmed = np.nanmedian(rel_flux[i])
            ftxt = f"{fmed:.3f}" if np.isfinite(fmed) else "na"
            item = QListWidgetItem(
                f"#{i}: x={x:.1f}, y={y:.1f}   RelFlux≈{ftxt}   FWHM={self.median_fwhm:.2f}"
            )
            item.setData(Qt.ItemDataRole.UserRole, i)
            self.star_list.addItem(item)

        # overlay & finish: use the real reference plane, not SEP background-subtracted data
        self._show_reference_with_circles(plane, self.star_positions)
        self.status_label.setText("Ready")
        self.progress_bar.setVisible(False)
        self.analyze_btn.setEnabled(True)

        # refresh dip-highlights using MA-based thresholding
        self._on_threshold_changed(self.threshold_slider.value())

    def _solve_reference(self, plane, hdr):
        plane2d = plane.mean(axis=2) if getattr(plane, "ndim", 2) == 3 else plane

        # ── Detect XISF origin ────────────────────────────────────────────
        # XISF headers carry an approximated SIP polynomial fit to PI's TPS
        # astrometric solution. The approximation can be off by 10-30 arcsec
        # in the corners, which is enough to blow our 5 arcsec star-matching
        # radius. Always plate-solve when the source is XISF.
        def _is_xisf_source(h) -> bool:
            if h is None:
                return False
            # XISF loaded via legacy loader stores format in metadata dict
            if isinstance(h, dict):
                fmt = str(h.get("original_format", "")).lower()
                if fmt == "xisf":
                    return True
                # Any key starting with XISF: is a tell-tale from xisf_fits_header_from_meta
                for k in h.keys():
                    if isinstance(k, str) and k.upper().startswith("XISF:"):
                        return True
                # Check nested fits_header / original_header
                for sub_key in ("fits_header", "original_header", "header"):
                    sub = h.get(sub_key)
                    if isinstance(sub, fits.Header):
                        for k in sub.keys():
                            if isinstance(k, str) and k.upper().startswith("XISF:"):
                                return True
            if isinstance(h, fits.Header):
                for k in h.keys():
                    if isinstance(k, str) and k.upper().startswith("XISF:"):
                        return True
            return False

        source_is_xisf = _is_xisf_source(hdr)

        # Also check the file extension of the first loaded path as a fallback
        if not source_is_xisf and self.image_paths:
            ref_path = self.image_paths[self.ref_idx] if hasattr(self, "ref_idx") else self.image_paths[0]
            if str(ref_path).lower().endswith(".xisf"):
                source_is_xisf = True

        # ── Step 1: trust existing WCS only if NOT from XISF ─────────────
        if not source_is_xisf:
            existing_wcs = self._try_wcs_from_header(hdr, plane2d)
            if existing_wcs is not None:
                self._wcs = existing_wcs
                H, W = plane2d.shape[:2]
                try:
                    center = self._wcs.pixel_to_world(W / 2, H / 2)
                    self.wcs_ra  = float(center.ra.deg)
                    self.wcs_dec = float(center.dec.deg)
                except Exception:
                    self.wcs_ra = self.wcs_dec = None

                ra_str  = "nan" if self.wcs_ra  is None else f"{self.wcs_ra:.5f}"
                dec_str = "nan" if self.wcs_dec is None else f"{self.wcs_dec:.5f}"
                self.status_label.setText(f"WCS from header: RA={ra_str}, Dec={dec_str}")
                self.fetch_tesscut_btn.setEnabled(True)
                return
        else:
            self.status_label.setText("XISF source detected — skipping approximate WCS, running plate solve…")
            QApplication.processEvents()

        # ── Step 2: plate-solve ───────────────────────────────────────────
        seed_hdr = self._coerce_seed_header(hdr if hdr is not None else {}, plane2d)
        meta = {}
        if isinstance(hdr, fits.Header):
            meta["original_header"] = hdr.copy()
        elif isinstance(hdr, dict):
            meta.update(hdr)
            H = self._get_best_header_for_wcs(hdr)
            if H is not None:
                meta["original_header"] = H.copy()
        else:
            meta["original_header"] = seed_hdr

        # For XISF, the seed header still has useful RA/Dec/scale hints
        # even though we don't trust its distortion model — keep them.
        if source_is_xisf and "original_header" in meta:
            existing_h = meta["original_header"]
            if isinstance(existing_h, fits.Header):
                # Strip SIP and CD/CDELT so the solver starts clean,
                # but keep CRVAL*/OBJCTRA/OBJCTDEC/FOCALLEN/XPIXSZ for hints
                strip_keys = [k for k in existing_h.keys()
                              if k.startswith(("A_", "B_", "AP_", "BP_",
                                               "CD1_", "CD2_", "CDELT",
                                               "CTYPE", "CRPIX", "NAXIS"))]
                clean_hdr = existing_h.copy()
                for k in strip_keys:
                    try:
                        del clean_hdr[k]
                    except Exception:
                        pass
                meta["original_header"] = clean_hdr

        doc = SimpleNamespace(
            image=plane2d,
            metadata=meta,
        )

        settings = getattr(self, "settings", None)
        if settings is None and hasattr(self, "main_win"):
            settings = getattr(self.main_win, "settings", None)
        if settings is None and self.parent() is not None:
            settings = getattr(self.parent(), "settings", None)

        ok, res = plate_solve_doc_inplace(self, doc, settings)
        if not ok:
            if source_is_xisf:
                # For XISF, offer to fall back to the approximate WCS rather
                # than leaving the user with nothing at all
                ans = QMessageBox.question(
                    self,
                    "Plate Solve Failed",
                    f"Plate solving failed:\n{res}\n\n"
                    "The loaded files are XISF — the embedded WCS is an approximation "
                    "of PixInsight's TPS solution and may be off by 10–30 arcsec in the "
                    "frame corners.\n\n"
                    "Use the approximate XISF WCS anyway? Star matching may fail for "
                    "some targets near the edges.",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if ans == QMessageBox.StandardButton.Yes:
                    fallback_wcs = self._try_wcs_from_header(hdr, plane2d)
                    if fallback_wcs is not None:
                        self._wcs = fallback_wcs
                        H2, W2 = plane2d.shape[:2]
                        try:
                            center = self._wcs.pixel_to_world(W2 / 2, H2 / 2)
                            self.wcs_ra  = float(center.ra.deg)
                            self.wcs_dec = float(center.dec.deg)
                        except Exception:
                            self.wcs_ra = self.wcs_dec = None
                        self.status_label.setText(
                            "⚠️ Using approximate XISF WCS (star matching may be imprecise)"
                        )
                        self.fetch_tesscut_btn.setEnabled(True)
                        return
            QMessageBox.critical(self, "Plate Solve", f"Plate solving failed:\n{res}")
            self._wcs = None
            self.fetch_tesscut_btn.setEnabled(False)
            return

        # ── Step 3: grab solved WCS ───────────────────────────────────────
        self._wcs = doc.metadata.get("wcs", None)
        if self._wcs is None or not getattr(self._wcs, "has_celestial", False):
            QMessageBox.warning(self, "Plate Solve", "Solver finished but no usable WCS was found.")
            self.fetch_tesscut_btn.setEnabled(False)
            self._wcs = None
            return

        H, W = plane2d.shape[:2]
        try:
            center = self._wcs.pixel_to_world(W / 2, H / 2)
            self.wcs_ra  = float(center.ra.deg)
            self.wcs_dec = float(center.dec.deg)
        except Exception:
            self.wcs_ra = self.wcs_dec = None

        ra_str  = "nan" if self.wcs_ra  is None else f"{self.wcs_ra:.5f}"
        dec_str = "nan" if self.wcs_dec is None else f"{self.wcs_dec:.5f}"
        self.status_label.setText(f"WCS solved: RA={ra_str}, Dec={dec_str}")
        self.fetch_tesscut_btn.setEnabled(True)

    # ---------------- Plotting & helpers ----------------

    def show_ensemble_members(self):
        sels = self.star_list.selectedItems()
        if len(sels) != 1: return
        target = sels[0].data(Qt.ItemDataRole.UserRole)
        members = self.ensemble_map.get(target, [])
        for idx in self._last_ensemble:
            item = self.star_list.item(idx)
            if item:
                color = item.background().color()
                if color == QColor('lightblue'):
                    item.setBackground(QBrush())
        for idx in members:
            item = self.star_list.item(idx)
            if item and item.background().color() != QColor('yellow'):
                item.setBackground(QBrush(QColor('lightblue')))
        self._last_ensemble = members

    def on_detrend_changed(self, idx: int):
        # idx==0 → No Detrend, 1 → Linear, 2 → Quadratic
        mapping = {0: None, 1: 1, 2: 2}
        self.detrend_degree = mapping[idx]
        if getattr(self, 'fluxes', None) is not None:
            self.update_plot_for_selection()

    @staticmethod
    def _detrend_curve(curve: np.ndarray, deg: int, mask: Optional[np.ndarray] = None) -> np.ndarray:
        x = np.arange(curve.size)
        if mask is None:
            mask = np.isfinite(curve) & (curve > 0)
        n_good = int(mask.sum())
        if n_good < 2:
            return curve
        fit_deg = min(deg, n_good - 1)
        if fit_deg < 1:
            return curve
        try:
            coeffs = np.polyfit(x[mask], curve[mask], fit_deg)
        except Exception:
            return curve
        trend = np.polyval(coeffs, x)
        trend[trend == 0] = 1.0
        return curve / trend

    def _show_reference_with_circles(self, plane, positions):
        dlg = ReferenceOverlayDialog(
            plane=plane,
            positions=positions,
            target_median=self.median_fwhm,
            parent=self
        )
        self._ref_overlay = dlg
        dlg.show()

    def update_plot_for_selection(self):
        """Redraw light curves for the selected stars."""
        # 1) sanity
        if self.fluxes is None:
            QMessageBox.warning(self, "No Photometry", "Please run photometry before selecting a star.")
            return

        # 2) X axis: hours since start if we have times, else frame index
        try:
            import astropy.units as u
            x_all = (self.times - self.times[0]).to(u.hour).value
            bottom_label = "Hours since start"
        except Exception:
            x_all = np.arange(self.fluxes.shape[1])
            bottom_label = "Frame"

        # 3) prep plot
        self.plot_widget.clear()
        self.plot_widget.addLegend()
        self.plot_widget.setLabel('bottom', bottom_label)
        self.plot_widget.setLabel('left',   'Relative Flux')

        # 4) what to draw?
        inds = [it.data(Qt.ItemDataRole.UserRole) for it in self.star_list.selectedItems()]
        if not inds:
            return

        n_stars = self.fluxes.shape[0]
        medians = np.nanmedian(self.fluxes, axis=1)
        max_gap = 1.0  # hours (or frames if no time axis)

        for idx in inds:
            f = self.fluxes[idx]
            flags_star = self.flags[idx] if (self.flags is not None and idx < self.flags.shape[0]) else np.zeros_like(f, int)

            mask = np.isfinite(f) & (f > 0) & (flags_star == 0)
            if mask.sum() < 2:
                continue

            rel = f[mask] / medians[idx]
            x   = x_all[mask]

            # moving average (window=5)
            ma = self.moving_average(rel, window=5) if hasattr(self, "moving_average") else np.convolve(np.pad(rel, 2, mode="edge"), np.ones(5)/5, mode="valid")

            # split segments across large gaps
            dt = np.diff(x)
            breaks = np.where(dt > max_gap)[0]
            segments = np.split(np.arange(len(x)), breaks+1)

            color = pg.intColor(idx, hues=n_stars)
            dull  = QColor(color); dull.setAlpha(60)
            dull_pen   = pg.mkPen(color=dull, width=1)
            dull_brush = pg.mkBrush(color=dull)
            dash_pen   = pg.mkPen(color=color, width=2, style=Qt.PenStyle.DashLine)

            for seg in segments:
                xs, ys, mas = x[seg], rel[seg], ma[seg]
                # raw points
                self.plot_widget.plot(xs, ys, pen=dull_pen, symbol='o', symbolBrush=dull_brush, name=f"Star #{idx}")
                # moving average
                self.plot_widget.plot(xs, mas, pen=dash_pen, name="MA (w=5)")

    def apply_threshold(self, ppt_threshold: int, sigma_upper: float = 3.0):
        if not hasattr(self, 'fluxes') or self.fluxes is None:
            return

        rel = self.fluxes
        n_stars, n_frames = rel.shape

        flagged = set()
        for i in range(n_stars):
            f = rel[i]
            flags_star = self.flags[i] if self.flags is not None else np.zeros(n_frames, dtype=np.int16)

            mask = np.isfinite(f) & (f > 0) & (flags_star == 0)
            if mask.sum() < 5:
                continue

            med_i = float(np.nanmedian(f[mask]))
            if not (np.isfinite(med_i) and med_i > 0):
                continue

            f_norm = f[mask] / med_i
            ma = self.moving_average(f_norm, window=5)

            # Peak deviation of the MA from 1.0
            ma_dev_ppt = np.abs(1.0 - ma) * 1000.0
            finite = np.isfinite(ma_dev_ppt)
            if not finite.any():
                continue
            peak_ppt = float(np.nanmax(ma_dev_ppt[finite]))

            if peak_ppt < ppt_threshold:
                continue

            # ── Coherence check ───────────────────────────────────────────
            # Compute scatter of raw points around the MA (residuals).
            # If scatter is large relative to the MA excursion, the MA is
            # being driven by noise rather than a coherent signal.
            residuals = f_norm - ma
            mad_residuals = float(1.4826 * np.nanmedian(
                np.abs(residuals - np.nanmedian(residuals))
            )) * 1000.0  # in ppt

            if np.isfinite(mad_residuals) and mad_residuals > 0:
                min_snr = self.temporal_snr_spin.value()
                if min_snr > 0.0 and (peak_ppt / mad_residuals) < min_snr:
                    continue
            flagged.add(i)

        for dlg in self.findChildren(ReferenceOverlayDialog):
            dlg.update_dip_flags(flagged)

        for row in range(self.star_list.count()):
            self.star_list.item(row).setBackground(QBrush())
        for idx in flagged:
            item = self.star_list.item(idx)
            if item:
                item.setBackground(QBrush(QColor('yellow')))

        self.status_label.setText(f"{len(flagged)} star(s) deviate ≥ {ppt_threshold} ppt")

    def moving_average(self, curve, window=5):
        pad = window//2
        ext = np.pad(curve, pad, mode="edge")
        kernel = np.ones(window)/window
        ma = np.convolve(ext, kernel, mode="valid")
        return ma

    # ---------------- Analysis + Identify ----------------

    def on_analyze(self):
        sels = self.star_list.selectedItems()
        if len(sels) != 1:
            QMessageBox.information(self, "Analyze", "Please select exactly one star.")
            return
        idx = sels[0].data(Qt.ItemDataRole.UserRole)

        # ── 1. Build raw time/flux arrays ────────────────────────────────────
        t_jd  = self.times.utc.jd                  # full JD
        f_all = self.fluxes[idx]
        good  = np.isfinite(f_all) & (self.flags[idx] == 0)
        t_jd_good = t_jd[good]
        f_good    = f_all[good]

        if len(t_jd_good) < 10:
            QMessageBox.warning(self, "Analyze", "Not enough good points to analyze.")
            return

        # ── 2. Lomb-Scargle on relative MJD (unit-median flux) ───────────────
        t_rel = t_jd_good - t_jd_good[0]          # days from first obs
        f_norm = f_good / np.nanmedian(f_good[np.isfinite(f_good) & (f_good > 0)])

        ls = LombScargle(t_rel, f_norm)
        Tspan = float(np.ptp(t_rel))
        dt    = float(np.nanmedian(np.diff(np.sort(t_rel))))
        min_f = max(1.0 / Tspan, 1e-6)
        max_f = 0.5 / max(dt, 1e-9)
        freq, power_ls = ls.autopower(
            minimum_frequency=min_f,
            maximum_frequency=max_f,
            samples_per_peak=getattr(self, "ls_samples_per_peak", 10),
        )
        mask = (freq > 0) & np.isfinite(power_ls)
        freq, power_ls = freq[mask], power_ls[mask]
        periods_ls = 1.0 / freq
        order = np.argsort(periods_ls)
        periods_ls, power_ls = periods_ls[order], power_ls[order]
        best_ls_period = float(periods_ls[np.argmax(power_ls)])

        # ── 3. Prepare cleaned light curve for BLS ───────────────────────────
        t_clean, y_clean = _prepare_search_lightcurve(t_jd_good, f_good)
        if t_clean.size < 20:
            QMessageBox.warning(self, "Analyze", "Too few clean points for BLS.")
            return

        baseline_days = float(np.ptp(t_clean))
        bls_min_p = float(getattr(self, "bls_min_period", 0.5))
        bls_max_p = min(
            float(getattr(self, "bls_max_period", 30.0)),
            0.95 * baseline_days,
        )
        if bls_max_p <= bls_min_p:
            bls_max_p = max(bls_min_p * 2.0, 0.95 * baseline_days)

        # Duration grid
        min_p_for_dur = bls_min_p
        dur_min_frac  = float(getattr(self, "bls_duration_min_frac", 0.01))
        dur_max_frac  = float(getattr(self, "bls_duration_max_frac", 0.10))
        n_durations   = int(getattr(self, "bls_n_durations", 8))
        dur_grid = np.linspace(
            dur_min_frac * min_p_for_dur,
            dur_max_frac * min_p_for_dur,
            n_durations,
        )  # days

        # ── 4. BLS with whitened power + multi-peak scanning ─────────────────
        dy_scalar = _robust_sigma(y_clean)
        if not (np.isfinite(dy_scalar) and dy_scalar > 0):
            dy_scalar = float(np.nanstd(y_clean - np.nanmedian(y_clean)))
        if not (np.isfinite(dy_scalar) and dy_scalar > 0):
            dy_scalar = 1e-4

        bls_model = BoxLeastSquares(
            t_clean * u.day,
            y_clean,
            dy=np.full_like(y_clean, dy_scalar),
        )
        result = bls_model.autopower(
            dur_grid * u.day,
            objective="snr",
            oversample=10,
            minimum_n_transit=2,
            minimum_period=bls_min_p * u.day,
            maximum_period=bls_max_p * u.day,
        )

        power_raw = np.asarray(result.power, float).ravel()
        periods_bls = np.asarray(result.period.value
                                if hasattr(result.period, "value")
                                else result.period, float).ravel()

        if power_raw.size == 0 or not np.any(np.isfinite(power_raw)):
            QMessageBox.warning(self, "Analyze", "BLS returned no usable power spectrum.")
            return

        _, power_white = _whiten_periodogram(power_raw)
        power_metric = power_white if np.any(np.isfinite(power_white)) else power_raw

        # Compute SDE from whitened power
        fin_m = np.isfinite(power_metric)
        power_med = float(np.nanmedian(power_metric[fin_m]))
        power_std = float(max(
            _robust_sigma(power_metric[fin_m]),
            float(np.nanstd(power_metric[fin_m])),
            1e-12,
        ))

        # Scan top peaks, pick best by SNR after local refinement
        peak_idxs = _top_peak_indices(periods_bls, power_metric, max_peaks=12, merge_frac=0.01)
        if not peak_idxs:
            peak_idxs = [int(np.nanargmax(power_metric))]

        best_period_d  = np.nan
        best_duration_d = np.nan
        best_t0_jd     = np.nan
        best_depth_frac = np.nan
        best_sde       = -np.inf
        best_metric    = -np.inf

        for pi in peak_idxs:
            cand_period = float(periods_bls[pi])
            if not (np.isfinite(cand_period) and bls_min_p <= cand_period <= bls_max_p):
                continue

            # Local refinement around this peak
            half_w  = max(0.01, 0.20 * cand_period)
            p_lo    = max(bls_min_p, cand_period - half_w)
            p_hi    = min(bls_max_p, cand_period + half_w)
            n_steps = 101
            p_grid  = np.linspace(p_lo, p_hi, n_steps) * u.day

            try:
                ref = bls_model.power(
                    p_grid, dur_grid * u.day,
                    objective="snr", oversample=5,
                )
                ref_power = np.asarray(ref.power, float).ravel()
                ref_periods = np.asarray(ref.period.value
                                        if hasattr(ref.period, "value")
                                        else ref.period, float).ravel()
                _, ref_white = _whiten_periodogram(ref_power)
                ref_metric = ref_white if np.any(np.isfinite(ref_white)) else ref_power
                k = int(np.nanargmax(ref_metric))

                cand_p  = float(ref_periods[k])
                cand_d  = float(ref.duration[k].value
                                if hasattr(ref.duration[k], "value")
                                else ref.duration[k])
                cand_t0 = float(ref.transit_time[k].value
                                if hasattr(ref.transit_time[k], "value")
                                else ref.transit_time[k])
                cand_depth = abs(float(ref.depth[k].value
                                    if hasattr(ref.depth[k], "value")
                                    else ref.depth[k])) if hasattr(ref, "depth") else np.nan
                cand_metric = float(ref_metric[k]) if np.isfinite(ref_metric[k]) else float(ref_power[k])
            except Exception:
                cand_p  = cand_period
                cand_d  = float(result.duration[pi].value
                                if hasattr(result.duration[pi], "value")
                                else result.duration[pi])
                cand_t0 = float(result.transit_time[pi].value
                                if hasattr(result.transit_time[pi], "value")
                                else result.transit_time[pi])
                cand_depth = abs(float(result.depth[pi].value
                                    if hasattr(result.depth[pi], "value")
                                    else result.depth[pi])) if hasattr(result, "depth") else np.nan
                cand_metric = float(power_metric[pi]) if np.isfinite(power_metric[pi]) else 0.0

            sde = (cand_metric - power_med) / power_std if power_std > 0 else 0.0

            # Apply alias penalty
            alias_pen = _alias_penalty(cand_p)
            sde_adj = sde - 1.5 * alias_pen

            if sde_adj > best_sde:
                best_sde        = sde_adj
                best_period_d   = cand_p
                best_duration_d = cand_d
                best_t0_jd      = cand_t0
                best_depth_frac = cand_depth
                best_metric     = cand_metric

        if not np.isfinite(best_period_d):
            # Fall back to raw argmax
            k = int(np.nanargmax(power_metric))
            best_period_d   = float(periods_bls[k])
            best_duration_d = float(result.duration[k].value
                                    if hasattr(result.duration[k], "value")
                                    else result.duration[k])
            best_t0_jd      = float(result.transit_time[k].value
                                    if hasattr(result.transit_time[k], "value")
                                    else result.transit_time[k])
            best_depth_frac = abs(float(result.depth[k].value
                                        if hasattr(result.depth[k], "value")
                                        else result.depth[k])) if hasattr(result, "depth") else np.nan
            best_sde        = (float(power_metric[k]) - power_med) / power_std if power_std > 0 else 0.0

        # ── 5. Phase fold [-0.5, 0.5] ────────────────────────────────────────
        phase = _phase_centered(t_clean, best_period_d, best_t0_jd)

        # BLS box model on sorted phase
        ord_ph = np.argsort(phase)
        phase_s = phase[ord_ph]
        flux_s  = y_clean[ord_ph]

        transit_half = min(0.45, max(0.01, 0.5 * best_duration_d / best_period_d))
        depth_plot   = float(best_depth_frac) if np.isfinite(best_depth_frac) else 0.0
        model_phase = np.linspace(-0.5, 0.5, 500)
        model_flux  = np.where(np.abs(model_phase) <= transit_half,
                            1.0 - depth_plot, 1.0)

        # Binned phase curve
        bin_width = max(0.005, transit_half / 3.0)
        bx, by, be = _make_binned_curve(phase, y_clean, bin_width)

        # Dip event detection for the summary
        dip_events = _estimate_night_dip_events(t_jd_good, f_good)

        # ── 6. Build dialog ───────────────────────────────────────────────────
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Analysis: Star #{idx}  |  P = {best_period_d:.4f} d  |  SDE = {best_sde:.1f}")
        layout = QVBoxLayout(dlg)

        # Summary label
        dur_hr   = best_duration_d * 24.0
        depth_ppt = depth_plot * 1000.0
        n_dips   = len(dip_events)
        alias_pen = _alias_penalty(best_period_d)
        summary = (
            f"Period: {best_period_d:.5f} d   "
            f"Duration: {dur_hr:.2f} h   "
            f"Depth: {depth_ppt:.2f} ppt   "
            f"SDE: {best_sde:.1f}   "
            f"Alias penalty: {alias_pen:.2f}   "
            f"Night dips detected: {n_dips}"
        )
        lbl = QLabel(summary)
        lbl.setStyleSheet("font-size: 11px; color: #aaddff; padding: 4px;")
        layout.addWidget(lbl)

        tabs = __import__("PyQt6.QtWidgets", fromlist=["QTabWidget"]).QTabWidget()
        layout.addWidget(tabs)

        # ── Tab 1: Lomb-Scargle ───────────────────────────────────────────────
        pg_ls = pg.PlotWidget(title="Lomb–Scargle Periodogram")
        pg_ls.plot(periods_ls, power_ls, pen=pg.mkPen("w", width=1))
        pg_ls.addLine(x=best_ls_period,
                    pen=pg.mkPen("y", style=Qt.PenStyle.DashLine, width=2))
        pg_ls.setLabel("bottom", "Period [d]")
        pg_ls.setLabel("left", "LS Power")
        pg_ls.showGrid(True, True, alpha=0.3)
        tabs.addTab(pg_ls, "Lomb-Scargle")

        # ── Tab 2: BLS Periodogram (whitened) ────────────────────────────────
        pg_bls = pg.PlotWidget(title="BLS Periodogram (whitened SNR)")
        pg_bls.plot(periods_bls, power_metric, pen=pg.mkPen("c", width=1))
        pg_bls.addLine(x=best_period_d,
                    pen=pg.mkPen("r", style=Qt.PenStyle.DashLine, width=2))
        pg_bls.setLabel("bottom", "Period [d]")
        pg_bls.setLabel("left", "Whitened Power")
        pg_bls.showGrid(True, True, alpha=0.3)
        tabs.addTab(pg_bls, "BLS Periodogram")

        # ── Tab 3: Phase fold ─────────────────────────────────────────────────
        pg_fold = pg.PlotWidget(title=f"Phase-Folded  P = {best_period_d:.4f} d")

        # Raw points (grey, small)
        pg_fold.plot(
            phase, y_clean,
            pen=None, symbol="o",
            symbolBrush=pg.mkBrush(180, 180, 180, 80),
            symbolSize=4,
            name="Raw",
        )

        # Binned points with error bars (blue)
        if bx.size > 0:
            err = pg.ErrorBarItem(
                x=bx, y=by, height=2 * be,
                pen=pg.mkPen("b", width=1.5),
            )
            pg_fold.addItem(err)
            pg_fold.plot(
                bx, by,
                pen=None, symbol="o",
                symbolBrush=pg.mkBrush(80, 160, 255, 220),
                symbolSize=7,
                name="Binned",
            )

        # Box model (red)
        pg_fold.plot(
            model_phase, model_flux,
            pen=pg.mkPen("r", width=2),
            name="Box model",
        )
        pg_fold.addLine(y=1.0, pen=pg.mkPen("w", style=Qt.PenStyle.DashLine, width=1))
        pg_fold.setLabel("bottom", "Phase")
        pg_fold.setLabel("left", "Relative Flux")
        pg_fold.showGrid(True, True, alpha=0.3)
        tabs.addTab(pg_fold, "Phase Fold")

        # ── Tab 4: Raw light curve ────────────────────────────────────────────
        pg_lc = pg.PlotWidget(title="Raw Light Curve")
        # t_jd_good and f_good are already filtered (finite + flag==0)
        # so t_plot and f_norm are the same size — no secondary mask needed
        t_plot = t_jd_good - t_jd_good[0]
        f_norm_plot = f_good / np.nanmedian(f_good[np.isfinite(f_good) & (f_good > 0)])
        pg_lc.plot(
            t_plot, f_norm_plot,
            pen=None, symbol="o",
            symbolBrush=pg.mkBrush(180, 200, 255, 160),
            symbolSize=4,
        )
        # Mark predicted transit centres
        if np.isfinite(best_t0_jd) and np.isfinite(best_period_d):
            t0_rel = best_t0_jd - t_jd_good[0]
            span   = float(np.ptp(t_plot))
            epoch  = math.floor((0.0 - t0_rel) / best_period_d)
            while True:
                tc = t0_rel + epoch * best_period_d
                if tc > span + best_period_d:
                    break
                if tc >= -best_period_d:
                    pg_lc.addLine(
                        x=tc,
                        pen=pg.mkPen("r", style=Qt.PenStyle.DashLine, width=1),
                    )
                epoch += 1
        pg_lc.setLabel("bottom", "Days from first obs")
        pg_lc.setLabel("left", "Relative Flux")
        pg_lc.showGrid(True, True, alpha=0.3)
        tabs.addTab(pg_lc, "Light Curve")

        dlg.resize(1000, 640)
        dlg.exec()

    def on_identify_star(self):
        radec = self.get_selected_star_radec()
        if radec is None:
            QMessageBox.warning(self, "Identify Star", "Please select exactly one star first.")
            return
        ra, dec = radec
        coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')

        custom_simbad = Simbad()
        custom_simbad.reset_votable_fields()
        custom_simbad.add_votable_fields("otype")
        custom_simbad.add_votable_fields("flux(V)")

        result = None
        for attempt in range(1, 6):
            try:
                result = custom_simbad.query_region(coord, radius=5*u.arcsec)
                break
            except Exception as e:
                print(f"[DEBUG] SIMBAD attempt {attempt} failed: {e}")
                if attempt == 5:
                    QMessageBox.critical(self, "SIMBAD Error", f"Could not reach SIMBAD after 5 tries:\n{e}")
                    return
                # Use non-blocking sleep
                non_blocking_sleep(1)

        if result is None or len(result) == 0:
            QMessageBox.information(self, "No SIMBAD Matches", f"No objects found within 5″ of {ra:.6f}, {dec:.6f}.")
            return

        row = result[0]
        id_col    = next(c for c in result.colnames if c.lower()=="main_id")
        ra_col    = next(c for c in result.colnames if c.lower()=="ra")
        dec_col   = next(c for c in result.colnames if c.lower()=="dec")
        otype_col = next((c for c in result.colnames if c.lower()=="otype"), None)
        flux_col  = next((c for c in result.colnames if c.upper()=="V" or c.upper()=="FLUX_V"), None)

        main_id = row[id_col]
        if isinstance(main_id, bytes):
            main_id = main_id.decode("utf-8")

        ra_val  = float(row[ra_col]); dec_val = float(row[dec_col])
        match_coord = SkyCoord(ra=ra_val*u.deg, dec=dec_val*u.deg, frame='icrs')
        offset = coord.separation(match_coord).arcsec

        obj_type = None
        if otype_col:
            obj_type = row[otype_col]
            if isinstance(obj_type, bytes):
                obj_type = obj_type.decode("utf-8")
        obj_type = obj_type or "n/a"

        vmag = None
        if flux_col:
            raw = row[flux_col]
            try: vmag = float(raw)
            except Exception: vmag = None
        vmag_str = f"{vmag:.3f}" if vmag is not None else "n/a"

        simbad_url = "https://simbad.cds.unistra.fr/simbad/sim-id" f"?Ident={quote(main_id)}"
        msg = QMessageBox(self)
        msg.setWindowTitle("SIMBAD Lookup")
        msg.setText(
            f"Nearest object:\n"
            f"  ID:     {main_id}\n"
            f"  Type:   {obj_type}\n"
            f"  V mag:  {vmag_str}\n"
            f"  Offset: {offset:.2f}″"
        )
        open_btn = msg.addButton("Open in SIMBAD", QMessageBox.ButtonRole.ActionRole)
        ok_btn   = msg.addButton(QMessageBox.StandardButton.Ok)
        msg.exec()
        if msg.clickedButton() == open_btn:
            webbrowser.open(simbad_url)

    def _query_simbad_main_id(self):
        radec = self.get_selected_star_radec()
        if radec is None:
            return None
        coord = SkyCoord(ra=radec[0]*u.deg, dec=radec[1]*u.deg, frame="icrs")
        table = None
        for attempt in range(1, 6):
            try:
                custom = Simbad(); custom.reset_votable_fields()
                custom.add_votable_fields("otype"); custom.add_votable_fields("flux(V)")
                table = custom.query_region(coord, radius=5*u.arcsec)
                break
            except Exception as e:
                print(f"[DEBUG] SIMBAD lookup attempt {attempt} failed: {e}")
                if attempt == 5:
                    QMessageBox.critical(self, "SIMBAD Error", f"Could not reach SIMBAD after 5 tries:\n{e}")
                    return None
                non_blocking_sleep(1)
        if table is None or len(table) == 0:
            return None
        try:
            id_col = next(c for c in table.colnames if c.lower() == "main_id")
        except StopIteration:
            return None
        val = table[0][id_col]
        if isinstance(val, bytes):
            val = val.decode("utf-8")
        return val

    def _query_simbad_name_and_vmag(self, ra_deg, dec_deg, radius=5*u.arcsec):
        coord = SkyCoord(ra=ra_deg*u.deg, dec=dec_deg*u.deg, frame="icrs")
        table = None
        for attempt in range(1,6):
            try:
                custom = Simbad(); custom.reset_votable_fields()
                custom.add_votable_fields("otype","flux(V)")
                table = custom.query_region(coord, radius=radius)
                break
            except Exception as e:
                if attempt==5:
                    QMessageBox.critical(self, "SIMBAD Error", f"Could not reach SIMBAD after 5 tries:\n{e}")
                    return None, None
                non_blocking_sleep(1)
        
        if table is None or len(table)==0:
            return None, None
        try:
            id_col = next(c for c in table.colnames if c.lower()=="main_id")
        except StopIteration:
            return None, None
        raw_id = table[0][id_col]
        if isinstance(raw_id, bytes):
            raw_id = raw_id.decode()
        v_col = next((c for c in table.colnames if c.upper() in ("FLUX_V","V")), None)
        vmag = None
        if v_col:
            try:
                v = float(table[0][v_col])
                if np.isfinite(v): vmag = v
            except Exception:
                vmag = None
        return raw_id, vmag

    # ---------------- Export ----------------

    def export_data(self):
        if self.fluxes is None or self.times is None:
            QMessageBox.warning(self, "Export", "No photometry to export. Run Measure & Photometry first.")
            return
        wcs = self._wcs
        if wcs is None:
            QMessageBox.warning(self, "Export", "No WCS available. Run plate solve during photometry first.")
            return

        dlg = QFileDialog(self, "Export Light Curves")
        dlg.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
        dlg.setNameFilters(["CSV files (*.csv)", "FITS files (*.fits)"])
        if not dlg.exec():
            return
        path = dlg.selectedFiles()[0]
        fmt  = dlg.selectedNameFilter()

        times_mjd = self.times.mjd
        n_stars   = self.fluxes.shape[0]

        xs = np.array([xy[0] for xy in self.star_positions])
        ys = np.array([xy[1] for xy in self.star_positions])
        bin_factor = getattr(self, "_wcs_bin_factor", 1)
        sky = wcs.pixel_to_world(xs * bin_factor, ys * bin_factor)

        ras = sky.ra.deg
        decs = sky.dec.deg

        if fmt.startswith("CSV") or path.lower().endswith(".csv"):
            df = pd.DataFrame({"MJD": times_mjd})
            for i in range(n_stars):
                df[f"STAR_{i}"]     = self.fluxes[i]
                df[f"FLAG_{i}"]     = self.flags[i]
                df[f"STAR_{i}_RA"]  = ras[i]
                df[f"STAR_{i}_DEC"] = decs[i]
            df.to_csv(path, index=False)
            QMessageBox.information(self, "Export CSV", f"Wrote CSV →\n{path}")
            return

        hdr_out = fits.Header()
        orig_hdr = None
        if hasattr(self, "_cached_headers") and 0 <= self.ref_idx < len(self._cached_headers):
            orig_hdr = self._cached_headers[self.ref_idx]
        if isinstance(orig_hdr, fits.Header):
            for key in ("OBJECT","TELESCOP","INSTRUME","OBSERVER",
                        "DATE-OBS","EXPTIME","FILTER",
                        "CRVAL1","CRVAL2","CRPIX1","CRPIX2",
                        "CDELT1","CDELT2","CTYPE1","CTYPE2"):
                if key in orig_hdr:
                    hdr_out[key] = orig_hdr[key]
        elif isinstance(orig_hdr, dict):
            for key in ("OBJECT","TELESCOP","INSTRUME","DATE-OBS","EXPTIME","FILTER"):
                val = orig_hdr.get(key, [{}])[0].get("value")
                if val is not None:
                    hdr_out[key] = val

        hdr_out["SEPTHR"]  = (self.sep_threshold,   "SEP detection threshold (sigma)")
        hdr_out["BFRAC"]   = (self.border_fraction, "Border ignore fraction")
        hdr_out["REFIDX"]  = (self.ref_idx,         "Reference frame index")
        hdr_out["MEDFWHM"] = (self.median_fwhm,     "Median FWHM of reference")
        hdr_out.add_history("Exported by Seti Astro Suite")

        cols = [fits.Column(name="MJD", format="D", array=times_mjd)]
        for i in range(n_stars):
            cols.append(fits.Column(name=f"STAR_{i}", format="E", array=self.fluxes[i]))
            cols.append(fits.Column(name=f"FLAG_{i}", format="I", array=self.flags[i]))
        lc_hdu = fits.BinTableHDU.from_columns(cols, header=hdr_out, name="LIGHTCURVE")

        star_idx = np.arange(n_stars, dtype=int)
        cols2 = [
            fits.Column(name="INDEX", format="I", array=star_idx),
            fits.Column(name="X",     format="E", array=xs),
            fits.Column(name="Y",     format="E", array=ys),
            fits.Column(name="RA",    format="D", array=ras),
            fits.Column(name="DEC",   format="D", array=decs),
        ]
        stars_hdu = fits.BinTableHDU.from_columns(cols2, name="STARS")

        primary = fits.PrimaryHDU(header=hdr_out)
        hdul    = fits.HDUList([primary, lc_hdu, stars_hdu])
        hdul.writeto(path, overwrite=True)
        QMessageBox.information(self, "Export FITS", f"Wrote FITS →\n{path}")

    def estimate_airmass_from_altitude(self, alt_deg):
        alt_rad = np.deg2rad(np.clip(alt_deg, 0.1, 90.0))
        return 1.0 / np.sin(alt_rad)

    def export_to_aavso(self):
        if getattr(self, "fluxes", None) is None or getattr(self, "times", None) is None:
            QMessageBox.warning(self, "Export AAVSO", "No photometry available. Run Measure & Photometry first.")
            return
        wcs = self._wcs
        if wcs is None:
            QMessageBox.warning(self, "Export AAVSO", "No WCS available. Plate-solve first.")
            return

        sels = self.star_list.selectedItems()
        if len(sels) != 1:
            QMessageBox.warning(self, "Export AAVSO", "Please select exactly one star before exporting.")
            return
        idx = sels[0].data(Qt.ItemDataRole.UserRole)

        star_id = self._query_simbad_main_id()
        if star_id:
            try:
                Vizier.ROW_LIMIT = 1
                v = Vizier(columns=["Name"], catalog="B/vsx")
                tbls = v.query_object(star_id)
                if tbls and len(tbls) > 0 and len(tbls[0]) > 0:
                    star_id = tbls[0]["Name"][0]
            except Exception as e:
                print(f"[DEBUG] VSX lookup failed: {e}")
        if not star_id:
            star_id, ok = QInputDialog.getText(self, "Target Star Name", "Could not auto-identify.  Enter target star name for STARID:", QLineEdit.EchoMode.Normal, "")
            if not ok or not star_id.strip():
                return
            star_id = star_id.strip()

        if not hasattr(self, "exposure_time") or self.exposure_time is None:
            exp, ok = QInputDialog.getDouble(self, "Exposure Time", "No EXPOSURE found in headers. Please enter exposure time (s):", decimals=1)
            if not ok: return
            self.exposure_time = exp

        settings = QSettings()
        prev_code = settings.value("AAVSO/observer_code", "", type=str)
        code, ok = QInputDialog.getText(self, "Observer Code", "Enter your AAVSO observer code:", QLineEdit.EchoMode.Normal, prev_code)
        if not ok: return
        code = code.strip().upper()
        settings.setValue("AAVSO/observer_code", code)

        fmt, ok = QInputDialog.getItem(self, "AAVSO Format", "Choose submission format:", ["Variable-Star Photometry", "Exoplanet Report"], 0, False)
        if not ok: return

        raw_members = self.ensemble_map.get(idx, [])
        members     = [m for m in raw_members if 0 <= m < len(self.star_positions)]
        kname = None; kmag  = None
        for m in members:
            x, y = self.star_positions[m]
            bin_factor = getattr(self, "_wcs_bin_factor", 1)
            sky = wcs.pixel_to_world(x * bin_factor, y * bin_factor)

            name, v = self._query_simbad_name_and_vmag(sky.ra.deg, sky.dec.deg)
            if name and (v is not None) and np.isfinite(v):
                kname, kmag = name, v
                break
        if kname is None:
            kname, ok = QInputDialog.getText(self, "Check Star Name", "Could not auto-identify a check star. Enter check-star ID:")
            if not ok or not kname.strip(): return
            kname = kname.strip()
            kmag, ok = QInputDialog.getDouble(self, "Check Star Magnitude", f"Enter catalog magnitude for {kname}:", decimals=3)
            if not ok: return

        filt_choices = ["V","TG","TB","TR"]
        filt, ok = QInputDialog.getItem(self, "Filter", "Select filter code for this dataset:", filt_choices, 0, False)
        if not ok: return

        path, _ = QFileDialog.getSaveFileName(self, "Save AAVSO File", "", "Text files (*.txt *.dat *.csv)")
        if not path: return

        header_lines = [
            "#TYPE=EXTENDED",
            f"#OBSCODE={code}",
            f"#SOFTWARE=Seti Astro Suite Pro",
            "#DELIM=,",
            "#DATE=JD",
            "#OBSTYPE=CCD",
        ]
        radec = self.get_selected_star_radec()
        if radec is None:
            QMessageBox.warning(self, "Export AAVSO", "Could not determine RA/Dec of selected star.")
            return
        c = SkyCoord(ra=radec[0]*u.deg, dec=radec[1]*u.deg, frame="icrs")
        header_lines += [
            "#RA="  + c.ra.to_string(unit=u.hour, sep=":", pad=True, precision=2),
            "#DEC=" + c.dec.to_string(unit=u.degree, sep=":", pad=True, alwayssign=True, precision=1),
        ]
        header_lines.append("#NAME,DATE,MAG,MERR,FILT,TRANS,MTYPE,CNAME,CMAG,KNAME,KMAG,AMASS,GROUP,CHART,NOTES")

        jd = self.times.utc.jd
        rel_flux = self.fluxes[idx, :]
        with np.errstate(divide="ignore"):
            mags = kmag - 2.5 * np.log10(rel_flux)
        if hasattr(self, "flux_errors"):
            rel_err = self.flux_errors[idx, :]
            merr = (2.5/np.log(10)) * (rel_err / rel_flux)
        else:
            merr = np.full_like(mags, np.nan)

        try:
            with open(path, "w") as f:
                for L in header_lines: f.write(L + "\n")
                f.write("\n")
                for j, t in enumerate(jd):
                    m   = mags[j]; me  = merr[j]
                    me_str = f"{me:.3f}" if np.isfinite(me) else "na"
                    note = "MAG calc via ensemble: m=-2.5 log10(F/Fe)+K"
                    am = float(np.clip(self.airmasses[j] if j < len(self.airmasses) else 1.0, 1.0, 40.0))
                    fields = [
                        star_id,
                        f"{t:.5f}",
                        f"{m:.3f}",
                        me_str,
                        filt,
                        "NO",
                        "STD",
                        "ENSEMBLE",
                        "na",
                        kname,
                        f"{kmag:.3f}",
                        f"{am:.1f}",
                        "na",
                        "na",
                        note
                    ]
                    f.write(",".join(fields) + "\n")
        except Exception as e:
            QMessageBox.critical(self, "Export AAVSO", f"Failed to write file:\n{e}")
            return

        msg = QMessageBox(self)
        msg.setWindowTitle("Export AAVSO")
        msg.setText(f"Wrote {fmt} →\n{path}\n\nOpen AAVSO WebObs upload page now?")
        yes = msg.addButton("Yes", QMessageBox.ButtonRole.AcceptRole)
        msg.addButton("No", QMessageBox.ButtonRole.RejectRole)
        msg.exec()
        if msg.clickedButton() == yes:
            webbrowser.open("https://www.aavso.org/webobs/file")

    # ---------------- TESScut ----------------

    def query_tesscut(self):
        radec = self.get_selected_star_radec()
        if radec is None:
            QMessageBox.warning(self, "No Star Selected", "Please select a star from the list to fetch TESScut data.")
            return
        ra, dec = radec
        print(f"[DEBUG] TESScut Query Requested for RA={ra:.6f}, Dec={dec:.6f}")
        coord = SkyCoord(ra=ra, dec=dec, unit="deg")

        size = 10
        MAX_RETRIES = 5

        manifest = None
        for mtry in range(1, MAX_RETRIES+1):
            try:
                print(f"[DEBUG] Manifest attempt {mtry}/{MAX_RETRIES}…")
                manifest = Tesscut.get_cutouts(coordinates=coord, size=size)
                if manifest:
                    print(f"[DEBUG] Manifest OK: {len(manifest)} sector(s).")
                    break
                else:
                    raise RuntimeError("Empty manifest")
            except Exception as me:
                print(f"[DEBUG] Manifest attempt {mtry} failed: {me}")
                if mtry == MAX_RETRIES:
                    QMessageBox.information(self, "No TESS Data", "There are no TESS cutouts available at that position.")
                    self.status_label.setText("No TESScut data found.")
                    return
                non_blocking_sleep(2)

        self.status_label.setText("Querying TESScut…")
        QApplication.processEvents()

        cache_dir = os.path.join(os.path.expanduser("~"), ".setiastro", "tesscut_cache")
        os.makedirs(cache_dir, exist_ok=True)

        for dtry in range(1, MAX_RETRIES+1):
            try:
                print(f"[DEBUG] Download attempt {dtry}/{MAX_RETRIES}…")
                cutouts = Tesscut.download_cutouts(coordinates=coord, size=size, path=cache_dir)
                if not cutouts:
                    raise RuntimeError("No cutouts downloaded")
                print(f"[DEBUG] Downloaded {len(cutouts)} cutout(s).")

                for cutout in cutouts:
                    original_path = cutout['Local Path']
                    print(f"[DEBUG] Processing: {original_path}")
                    with fits.open(original_path, mode='readonly') as hdul:
                        sector = hdul[1].header.get('SECTOR', 'unknown')
                    ext = os.path.splitext(original_path)[1]
                    cache_key = f"tess_sector{sector}_ra{int(round(ra*10000))}_dec{int(round(dec*10000))}{ext}"
                    cached_path = os.path.join(cache_dir, cache_key)

                    if not os.path.exists(cached_path):
                        print(f"[DEBUG] Caching as: {cached_path}")
                        shutil.move(original_path, cached_path)
                    else:
                        print(f"[DEBUG] Already cached: {cached_path}")
                        os.remove(original_path)

                    tpf = TessTargetPixelFile(cached_path)
                    xpix, ypix = tpf.wcs.world_to_pixel(coord)
                    ny, nx = tpf.flux.shape[1], tpf.flux.shape[2]
                    Y, X = np.mgrid[:ny, :nx]
                    r_pix = 2.5
                    aper_mask = ((X - xpix)**2 + (Y - ypix)**2) <= r_pix**2

                    lc = (tpf.to_lightcurve(aperture_mask=aper_mask).remove_nans().normalize())
                    upper, lower = 5.0, -1.0
                    mask = (lc.flux < upper) & (lc.flux > lower)
                    n_clipped = np.sum(~mask)
                    print(f"[DEBUG] Clipping {n_clipped} points outside [{lower}, {upper}]×")
                    lc = lc[mask]

                    lc.plot(label=f"Sector {tpf.sector} (clipped)")
                    plt.title(f"TESS Light Curve - Sector {tpf.sector}")
                    plt.tight_layout()
                    plt.show()

                self.status_label.setText("TESScut fetch complete.")
                return

            except Exception as de:
                print(f"[ERROR] Download attempt {dtry} failed: {de}")
                self.status_label.setText(f"TESScut attempt {dtry}/{MAX_RETRIES} failed.")
                QApplication.processEvents()
                if dtry == MAX_RETRIES:
                    QMessageBox.critical(self, "TESScut Error", f"TESScut failed after {MAX_RETRIES} attempts.\n\n{de}")
                    self.status_label.setText("TESScut fetch failed.")
                else:
                    non_blocking_sleep(2)

    # ---------------- Pixel → Sky helper ----------------
    def get_selected_star_radec(self):
        selected_items = self.star_list.selectedItems()
        if not selected_items:
            return None

        selected_index = selected_items[0].data(Qt.ItemDataRole.UserRole)
        x, y = self.star_positions[selected_index]

        if self._wcs is None:
            return None

        bin_factor = getattr(self, "_wcs_bin_factor", 1)
        xw = x * bin_factor
        yw = y * bin_factor

        sky = self._wcs.pixel_to_world(xw, yw)
        return sky.ra.degree, sky.dec.degree

