#saspro/image_peeker_pro.py
from __future__ import annotations
import os
import math
import re
import tempfile
import numpy as np

from typing import Optional, Tuple, List

from PyQt6.QtGui import (
    QIcon, QColor, QPixmap, QPainter, QPen, QImage, QPainterPath, QFont, QGuiApplication
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QPoint, QEvent, QPointF, QCoreApplication, QSettings, QByteArray, QThread
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox, QLabel, QSlider,
    QPushButton, QComboBox, QSizePolicy, QMessageBox, QColorDialog, QWidget,
    QScrollArea, QScrollBar, QMdiSubWindow, QGraphicsScene, QGraphicsView,
    QGraphicsTextItem, QTableWidget, QTableWidgetItem, QLineEdit, QToolButton,
    QSpinBox, QDoubleSpinBox
)
from PyQt6.QtGui import QDoubleValidator, QIntValidator

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from scipy.interpolate import griddata
from scipy.ndimage import zoom as _ndzoom

import sep
sep.set_extract_pixstack(20000000)
from setiastro.saspro.widgets.themed_buttons import themed_toolbtn
from setiastro.saspro.help_support import make_help_button

# bring in your existing helpers/classes from the snippet you posted
# (we assume they live next to this file or already in pro/)
from .plate_solver import plate_solve_doc_inplace
from setiastro.saspro.imageops.stretch import stretch_mono_image, stretch_color_image
from astropy.wcs import WCS
from .plate_solver import _seed_header_from_meta, _solve_numpy_with_fallback
from astropy.wcs import WCS

def _header_from_meta(meta):
    # Prefer real Header
    hdr = _ensure_fits_header(meta.get("original_header"))
    if hdr is not None:
        return hdr

    # Next try stored WCS header
    wh = meta.get("wcs_header")
    if isinstance(wh, fits.Header):
        return wh
    if isinstance(wh, dict):
        try:
            return fits.Header(wh)
        except Exception:
            pass

    # Finally try astropy WCS object
    w = meta.get("wcs")
    if isinstance(w, WCS):
        try:
            return w.to_header(relax=True)
        except Exception:
            pass

    return None


class PreviewPane(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.zoom_factor = 1.0
        self.is_autostretched = False
        self._image_array     = None
        self.original_image = None    # QImage
        self.stretched_image = None   # QImage
        self._panning = False
        self._pan_start = QPoint()
        self._h_scroll_start = 0
        self._v_scroll_start = 0

        # the scrollable image area
        self.image_label  = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scroll_area  = QScrollArea()
        self.scroll_area.setWidget(self.image_label)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setMinimumSize(450, 450)
        self.scroll_area.viewport().installEventFilter(self)

        # zoom controls
        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setRange(1, 400)
        self.zoom_slider.setValue(100)
        self.zoom_slider.valueChanged.connect(self.on_zoom_changed)

        self.zoom_in_btn  = themed_toolbtn("zoom-in", self.tr("Zoom In"))
        self.zoom_out_btn = themed_toolbtn("zoom-out", self.tr("Zoom Out"))
        self.fit_btn      = themed_toolbtn("zoom-fit-best", self.tr("Fit to Preview"))
        self.zoom_in_btn.clicked .connect(lambda: self.adjust_zoom(10))
        self.zoom_out_btn.clicked.connect(lambda: self.adjust_zoom(-10))
        self.fit_btn.clicked .connect(self.fit_to_view)        

        self.stretch_btn  = QPushButton(self.tr("AutoStretch"))
        self.stretch_btn.clicked.connect(self.toggle_stretch)

        zl = QHBoxLayout()
        zl.addWidget(self.zoom_out_btn)
        zl.addWidget(self.zoom_slider)
        zl.addWidget(self.zoom_in_btn)
        zl.addWidget(self.fit_btn)
        zl.addWidget(self.stretch_btn)

        layout = QVBoxLayout(self)
        layout.addWidget(self.scroll_area, 1)
        layout.addLayout(zl)

        self.fit_to_view()

    def load_qimage(self, img: QImage, source_float: np.ndarray | None = None):
        """
        (Re)load a fresh image. If `source_float` is provided (H×W or H×W×3,
        float32 in [0..1]), it is kept as the true high-bit-depth source for
        stretching; otherwise we fall back to the 8-bit QImage (lossy).
        """
        self.original_image = img.copy()

        if source_float is not None:
            self._image_array = np.asarray(source_float, dtype=np.float32)
        else:
            # legacy path — only 8-bit data available
            self._image_array = self.qimage_to_numpy(self.original_image)

        self.stretched_image  = None
        self.is_autostretched = False
        self.zoom_factor      = 1.0
        self.zoom_slider.setValue(100)
        self._update_display()

    def set_overlay(self, overlays):
        """ Store and repaint overlays on top of the image. """
        self._overlays = overlays
        self._update_display()

    def toggle_stretch(self):
        if self._image_array is None:
            return

        self.is_autostretched = not self.is_autostretched

        if self.is_autostretched:
            # stretch the stored numpy array
            arr = self._image_array.copy()
            if arr.ndim == 2:
                stretched = stretch_mono_image(
                    arr,
                    target_median=0.25,
                    normalize=True,
                    apply_curves=False,
                    no_black_clip=True
                )
            else:
                stretched = stretch_color_image(
                    arr,
                    target_median=0.25,
                    linked=False,
                    normalize=True,
                    apply_curves=False,
                    no_black_clip=True
                )

            # convert back to a QImage for display
            self.stretched_image = self.numpy_to_qimage(stretched).copy()
        else:
            # go back to the original QImage
            self.stretched_image = self.original_image.copy()

        self._update_display()

    def qimage_to_numpy(self, qimg: QImage) -> np.ndarray:
        """
        Safely copy a QImage into a contiguous numpy array,
        and return float32 data normalized to [0.0, 1.0].
        Supports Grayscale8 and RGB888.
        """
        # force a copy & right format
        if qimg.format() == QImage.Format.Format_Grayscale8:
            img = qimg.convertToFormat(QImage.Format.Format_Grayscale8).copy()
            w, h = img.width(), img.height()
            ptr   = img.bits()
            ptr.setsize(h * w)
            buf   = ptr.asstring()
            arr   = np.frombuffer(buf, np.uint8).reshape((h, w))
        else:
            img = qimg.convertToFormat(QImage.Format.Format_RGB888).copy()
            w, h = img.width(), img.height()
            bpl   = img.bytesPerLine()
            ptr   = img.bits()
            ptr.setsize(h * bpl)
            buf   = ptr.asstring()
            raw   = np.frombuffer(buf, np.uint8).reshape((h, bpl))
            raw   = raw[:, : 3*w]
            arr   = raw.reshape((h, w, 3))

        # **normalize to float32 [0..1]**
        return (arr.astype(np.float32) / 255.0)

    def numpy_to_qimage(self, arr: np.ndarray) -> QImage:
        """
        Convert a H×W or H×W×3 numpy array (float in [0..1] or uint8 in [0..255])
        into a QImage (copying the buffer).
        """
        # If floating point, assume 0..1 and scale up:
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        # Otherwise convert any other integer type to uint8
        elif arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        h, w = arr.shape[:2]
        if arr.ndim == 2:
            img = QImage(arr.data, w, h, w, QImage.Format.Format_Grayscale8)
            return img.copy()
        elif arr.ndim == 3 and arr.shape[2] == 3:
            bytes_per_line = 3 * w
            img = QImage(arr.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            return img.copy()
        else:
            raise ValueError(f"Cannot convert array of shape {arr.shape} to QImage")

    def on_zoom_changed(self, val):
        self.zoom_factor = val/100
        self._update_display()

    def adjust_zoom(self, delta):
        v = self.zoom_slider.value() + delta
        self.zoom_slider.setValue(min(max(v,1),400))

    def fit_to_view(self):
        if not self.original_image:
            return
        avail = self.scroll_area.viewport().size()
        iw, ih = self.original_image.width(), self.original_image.height()
        f = min(avail.width()/iw, avail.height()/ih)
        self.zoom_factor = f
        self.zoom_slider.setValue(int(f*100))
        self._update_display()

    def _update_display(self):
        """
        Chooses original vs stretched image and repaints.
        """
        img = self.stretched_image or self.original_image
        if img is None:
            return

        pix = QPixmap.fromImage(self.stretched_image or self.original_image)
        painter = QPainter(pix)
        painter.setPen(QPen(Qt.GlobalColor.red, 2))
        # draw any overlays
        for ov in getattr(self, "_overlays", []):
            x, y, p3, p4 = ov
            # if p3 is an integer / we intended an ellipse
            if isinstance(p3, (int,)) and isinstance(p4, (int, float)):
                w = int(p3)
                h = w
                painter.drawEllipse(x, y, w, h)
                painter.drawText(x, y, f"{p4:.2f}")
            else:
                # treat as vector overlay: (angle, length)
                angle = float(p3)
                length_um = float(p4)
                # convert length from µm → pixels if necessary;
                # here we assume overlays were built in pixels:
                dx = math.cos(angle) * length_um
                dy = -math.sin(angle) * length_um
                x2 = x + dx
                y2 = y + dy
                painter.drawLine(int(x), int(y), int(x2), int(y2))
                # optional: draw a simple arrowhead
                # (two short lines at ±20° from the vector)
                ah = 5  # arrow‐head pixel length
                for sign in (+1, -1):
                    ang2 = angle + sign * math.radians(20)
                    ax = x2 - ah * math.cos(ang2)
                    ay = y2 + ah * math.sin(ang2)
                    painter.drawLine(int(x2), int(y2), int(ax), int(ay))
        painter.end()
        scaled = pix.scaled(
            pix.size() * self.zoom_factor,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.image_label.setPixmap(scaled)

    def eventFilter(self, source, evt):
        if source is self.scroll_area.viewport():
            if evt.type() == QEvent.Type.MouseButtonPress and evt.button() == Qt.MouseButton.LeftButton:
                self._panning = True
                self._pan_start = evt.position().toPoint()
                self._h_scroll_start = self.scroll_area.horizontalScrollBar().value()
                self._v_scroll_start = self.scroll_area.verticalScrollBar().value()
                self.scroll_area.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
                return True

            elif evt.type() == QEvent.Type.MouseMove and self._panning:
                delta = evt.position().toPoint() - self._pan_start
                self.scroll_area.horizontalScrollBar().setValue(self._h_scroll_start - delta.x())
                self.scroll_area.verticalScrollBar().setValue(self._v_scroll_start - delta.y())
                return True

            elif evt.type() == QEvent.Type.MouseButtonRelease and evt.button() == Qt.MouseButton.LeftButton:
                self._panning = False
                self.scroll_area.viewport().setCursor(Qt.CursorShape.ArrowCursor)
                return True

            elif evt.type() == QEvent.Type.Wheel:
                # _SASPRO_STRENGTH_WHEEL: zoom about the pointer, so the pixel under the
                # cursor stays under the cursor. Driving the slider keeps the
                # two controls from disagreeing; signals are blocked so the
                # display is updated once, after the new zoom is in place.
                try:
                    dy = evt.angleDelta().y()
                    if dy == 0:
                        return True
                    fine = bool(evt.modifiers() & (Qt.KeyboardModifier.ControlModifier
                                                   | Qt.KeyboardModifier.ShiftModifier))
                    step = (1.05 if fine else 1.18) ** (1 if dy > 0 else -1)
                    old = float(self.zoom_factor) or 1.0
                    new = min(max(old * step, 0.01), 4.0)
                    if abs(new - old) < 1e-9:
                        return True
                    pos = evt.position()
                    hb = self.scroll_area.horizontalScrollBar()
                    vb = self.scroll_area.verticalScrollBar()
                    # Point of the IMAGE currently under the cursor.
                    ix = (hb.value() + pos.x()) / old
                    iy = (vb.value() + pos.y()) / old
                    blocked = self.zoom_slider.blockSignals(True)
                    self.zoom_slider.setValue(int(round(new * 100)))
                    self.zoom_slider.blockSignals(blocked)
                    self.zoom_factor = new
                    self._update_display()
                    hb.setValue(int(round(ix * new - pos.x())))
                    vb.setValue(int(round(iy * new - pos.y())))
                except Exception:
                    return False
                return True

        return super().eventFilter(source, evt)

    def load_numpy(self, arr: np.ndarray):
        qimg = self.numpy_to_qimage(arr)
        self.load_qimage(qimg, source_float=np.asarray(arr, dtype=np.float32))

# =============================================================================
#  Shared helpers for star-based focal-plane / tilt analyses
#
#  - _detect_stars: single SEP pass with quality cuts (flag==0, min area, valid
#    a >= b > 0). Returns a filtered catalogue used by every downstream metric,
#    so Focal Plane Analysis no longer runs SEP three separate times.
#  - _eval_poly_upsampled: evaluates a fitted 2D polynomial on a small coarse
#    grid (default 128x128) and bilinearly upsamples to full resolution. The
#    fitted surface is smooth by construction, so this is visually identical
#    to full-res evaluation and ~1000x faster on multi-megapixel canvases.
#  - _fit_and_render_surface: end-to-end (fit_2d_poly + upsample) with a
#    graceful degree fallback when there are too few detections.
# =============================================================================
_FWHM_FACTOR = 2.0 * math.sqrt(2.0 * math.log(2.0))  # ~= 2.3548200450309493


def _to_gray32(img: np.ndarray) -> np.ndarray:
    """Luma-weighted grayscale (Rec.709). Passes through already-2D input."""
    if img.ndim == 3 and img.shape[2] >= 3:
        return (0.2126 * img[..., 0]
                + 0.7152 * img[..., 1]
                + 0.0722 * img[..., 2]).astype(np.float32, copy=False)
    return img.astype(np.float32, copy=False)


# =============================================================================
# Shape-measurement quality
# =============================================================================
# SEP returns a, b and theta for every detection, however small. For a star
# only a pixel or two across those numbers are not measurements -- they are
# artifacts of where the PSF happened to land on the pixel grid -- and a star
# that is genuinely round has no position angle to report at all. Feeding
# either kind into an orientation map produces a confident-looking picture of
# nothing. The constants and helpers below decide which detections carry a
# real shape measurement.
#
# Two failure modes, handled separately because they need different cuts:
#
#   Undersampling. Below roughly Nyquist (FWHM ~ 2 px) the second moments are
#   dominated by subpixel phase: the same star shifted half a pixel reports a
#   different elongation and a different angle. No amount of SNR fixes this.
#
#   Roundness. A well-sampled but nearly circular star has a position angle
#   whose uncertainty approaches 90 degrees -- it is noise with an arrow
#   drawn on it. This one is quantifiable, which is what the machinery below
#   exploits.
#
# The quantification works in the polarization (double-angle) representation
# used throughout weak lensing and PSF-anisotropy mapping:
#
#     |e| = (a^2 - b^2) / (a^2 + b^2)
#     e1  = |e| cos 2*theta,   e2 = |e| sin 2*theta
#
# theta and theta+180 deg map to the same (e1, e2), which is what we want for
# an axis with no head or tail. Given a per-component uncertainty sigma_e,
# standard error propagation through theta = 0.5*atan2(e2, e1) gives
#
#     sigma_theta = sigma_e / (2 |e|)
#
# so a single threshold on sigma_theta rejects both the too-round and the
# too-noisy in one physically meaningful cut, in degrees the user can reason
# about rather than an arbitrary axis-ratio limit.

# Nyquist: a PSF narrower than ~2 px/FWHM is undersampled and its second
# moments describe the pixel grid, not the star.
_MIN_FWHM_PX = 2.0

# The minor axis must itself be resolved, not just the geometric mean. A
# 0.75 px second-moment sigma is about 1.77 px FWHM on that axis.
_MIN_SEMIMINOR_PX = 0.75

# Shape-noise constant in sigma_e ~= _SHAPE_NOISE_K / SNR. The usual
# matched-filter rule of thumb for second-moment ellipticity errors is ~2/SNR;
# it is an approximation, not a derivation, so it is exposed as a constant to
# be re-tuned against real data rather than buried in an expression.
_SHAPE_NOISE_K = 2.0

# Pixel-grid contribution to the moment uncertainty. A pixel's own positional
# variance is 1/12 px^2 (uniform over the pixel), and that quantization
# propagates into (a^2 - b^2)/(a^2 + b^2) as roughly (1/12)/(a^2 + b^2).
# Negligible for a well-sampled star, dominant as it approaches Nyquist --
# which is exactly the behaviour wanted.
_PIX_VAR = 1.0 / 12.0

# Defaults for the orientation cut.
_MAX_SIGMA_THETA_DEG = 15.0   # direction known to better than +/- 15 deg
_MIN_ELONG_SNR = 3.0          # elongation itself detected at 3 sigma
_MIN_ORIENT_STARS = 12        # below this an orientation map is not worth drawing

# Absolute floor on |e|, independent of how well it is measured.
#
# Significance alone is not enough. sigma_e shrinks as 1/SNR, so on a bright
# star a 1% elongation clears a 3-sigma test easily and its angle comes out
# "known" to a degree or two. Both statements are true and neither is useful:
# a 1% elongation is not a tilt, a spacing error, or anything else the user
# can act on, and a map built from such stars tracks sub-percent PSF jitter
# while looking exactly like a real aberration pattern.
#
# |e| = 0.08 is about a 4% ellipticity (b/a ~ 0.96) -- comfortably below mild
# sensor tilt, which shows 8-15%, and comfortably above the noise floor.
_MIN_ABS_ELONG = 0.08

# --- _SASPRO_SHAPE_FIT_V2 : begin ------------------------------------------------
# SNR floor for stars entering a surface fit. The fit must not select on the
# elongation it is measuring (see _orientation_mask), so this is the only
# lower bound: below it the shape is noise regardless of how it looks.
_MIN_FIT_SNR = 10.0

# Weight cap. The fit weight is 1/sigma_e^2 and sigma_e ~ 2/SNR, so weight
# goes as SNR^2 and the few brightest stars would otherwise own the result --
# the same stars whose cores are clipped and therefore measure too round.
# Capping at this percentile keeps inverse-variance weighting where it is
# meaningful and stops the tail dominating.
_FIT_WEIGHT_CAP_PCT = 90.0

# Fraction of the data maximum at or above which a peak is treated as clipped.
_SAT_FRAC = 0.995

# The anchor regions that actually matter for a focal-plane judgement: four
# corners, four edge midpoints, and the centre.
_ZONE_NAMES = ("top-left", "top-centre", "top-right",
               "left", "centre", "right",
               "bottom-left", "bottom-centre", "bottom-right")
_MIN_ZONE_STARS = 8


def _estimate_sat_level(data: np.ndarray) -> Optional[float]:
    """
    Clipping level inferred from the pixel distribution, or None.

    sat_level is threaded through _sampling_mask, _quality_filter and both
    detectors, but was never given a value at any call site, so the
    saturation cut never ran. Clipped stars stayed in the catalogue, and a
    flat-topped core measures rounder than the star really is while carrying
    the largest fit weight of anything in the frame.

    A real clipped frame shows a pile-up in the top bin: many pixels at
    exactly the same maximum. A frame with headroom does not. Requiring a
    pile-up rather than always cutting at a fixed fraction means an
    unsaturated frame loses no stars.
    """
    try:
        d = np.asarray(data, dtype=np.float32)
        finite = d[np.isfinite(d)]
        if finite.size < 1000:
            return None
        mx = float(finite.max())
        if not np.isfinite(mx) or mx <= 0.0:
            return None
        n_top = int(np.count_nonzero(finite >= mx * _SAT_FRAC))
        # A genuine pile-up, not just the handful of pixels in the
        # single brightest star's core.
        if n_top < 50:
            return None
        return mx * _SAT_FRAC
    except Exception:
        return None


def _zone_index(x: np.ndarray, y: np.ndarray, H: int, W: int) -> np.ndarray:
    """3x3 anchor-region index (0..8), row-major from the top-left."""
    cx = np.clip((np.asarray(x, float) / (float(W) / 3.0)).astype(int), 0, 2)
    cy = np.clip((np.asarray(y, float) / (float(H) / 3.0)).astype(int), 0, 2)
    return cy * 3 + cx


def _anchor_zone_stats(x: np.ndarray, y: np.ndarray,
                       e1: np.ndarray, e2: np.ndarray,
                       sigma_e: np.ndarray,
                       H: int, W: int,
                       min_zone_stars: int = _MIN_ZONE_STARS) -> list:
    """
    Per-region shape summary over the nine anchor zones.

    A single global surface hides exactly the thing a focal-plane check is
    for: whether ONE corner disagrees with the rest. These are direct
    inverse-variance means of the measured ellipticity vectors inside each
    region -- no surface, no extrapolation -- so a zone reports what its own
    stars say or reports that it has too few to say anything.

    Each entry: name, n, e_mag, angle_deg, sig, ok.
    'angle_deg' is in the same y-down frame as the rest of the module, i.e.
    clockwise from horizontal as displayed. 'sig' is |e| divided by the
    standard error of the zone's own weighted mean, measured from the
    observed scatter of its stars, so ~3 or more is a real detection and a
    near-round zone lands near zero.
    """
    z = _zone_index(x, y, H, W)
    out = []
    for k in range(9):
        m = z == k
        n = int(np.count_nonzero(m))
        if n == 0:
            out.append({'name': _ZONE_NAMES[k], 'n': 0, 'e_mag': float('nan'),
                        'angle_deg': float('nan'), 'sig': 0.0, 'ok': False})
            continue
        w = 1.0 / np.maximum(sigma_e[m], 1e-6) ** 2
        w = np.nan_to_num(w, nan=0.0, posinf=0.0)
        sw = float(w.sum())
        if sw <= 0.0:
            out.append({'name': _ZONE_NAMES[k], 'n': n, 'e_mag': float('nan'),
                        'angle_deg': float('nan'), 'sig': 0.0, 'ok': False})
            continue
        m1 = float(np.sum(w * e1[m]) / sw)
        m2 = float(np.sum(w * e2[m]) / sw)
        emag = math.hypot(m1, m2)
        # Standard error from the OBSERVED scatter of this zone's own stars,
        # not from propagated sigma_e. sigma_e ~ 2/SNR is a rule of thumb that
        # badly understates real shape dispersion on bright stars, which would
        # otherwise report a 170-sigma detection of a 1% elongation.
        n_eff = (sw * sw) / max(float(np.sum(w * w)), 1e-30)
        var = (float(np.sum(w * ((e1[m] - m1) ** 2 + (e2[m] - m2) ** 2))) / sw) / 2.0
        se = math.sqrt(max(var, 0.0) / max(n_eff, 1.0)) if n_eff > 1.0 else float('inf')
        out.append({
            'name': _ZONE_NAMES[k],
            'n': n,
            'e_mag': emag,
            'angle_deg': math.degrees(0.5 * math.atan2(m2, m1)) % 180.0,
            'sig': (emag / se) if se > 0 else 0.0,
            'ok': n >= int(min_zone_stars),
        })
    return out
# --- _SASPRO_SHAPE_FIT_V2 : end --------------------------------------------------


def _shape_stats(cat: dict, rms: float = 0.0) -> dict:
    """
    Per-star shape measurements and their uncertainties.

    Takes a catalogue dict as produced by the detection helpers below and
    returns a dict of parallel float64 arrays:

        snr         estimated flux SNR (flux / (rms * sqrt(npix)))
        fwhm_px     geometric-mean FWHM in pixels, 2.3548 * sqrt(a*b)
        e_mag       |e| = (a^2 - b^2) / (a^2 + b^2), the polarization magnitude
        e1, e2      |e| cos 2*theta, |e| sin 2*theta
        sigma_e     per-component uncertainty on e1/e2
        sigma_theta position-angle uncertainty in RADIANS
        elong_snr   e_mag / sigma_e -- significance of the elongation itself

    `e_mag` is computed from moments with the pixel's own variance removed
    (Sheppard's correction: the measured second moment of a sampled source is
    the true moment plus 1/12). Skipping that correction makes every star look
    rounder than it is, and worst for the smallest stars -- precisely the ones
    whose elongation we are trying to judge. The correction is used for the
    significance test only; reported ellipticity values elsewhere keep the
    raw SEP convention so they stay comparable with other tools.

    With rms <= 0 the SNR term is unavailable and sigma_e falls back to the
    pixelization floor alone, which makes the cut purely about sampling.
    """
    a = np.asarray(cat['a'], dtype=np.float64)
    b = np.asarray(cat['b'], dtype=np.float64)
    th = np.asarray(cat['theta'], dtype=np.float64)
    flux = np.asarray(cat.get('flux', np.zeros_like(a)), dtype=np.float64)
    npix = np.asarray(cat.get('npix', np.full_like(a, 9.0)), dtype=np.float64)

    a2, b2 = a * a, b * b

    # --- SNR ---------------------------------------------------------------
    with np.errstate(divide='ignore', invalid='ignore'):
        noise = float(rms) * np.sqrt(np.maximum(npix, 1.0))
        snr = np.where(noise > 0, flux / np.maximum(noise, 1e-30), 0.0)
    snr = np.nan_to_num(snr, nan=0.0, posinf=0.0, neginf=0.0)
    snr = np.maximum(snr, 0.0)

    # --- geometric-mean FWHM ----------------------------------------------
    fwhm_px = _FWHM_FACTOR * np.sqrt(np.maximum(a * b, 0.0))

    # --- polarization magnitude, pixel variance removed --------------------
    # Only subtract where it leaves a physically sensible moment; on a star
    # too small for that the value is meaningless anyway and the sampling
    # gate will have rejected it already.
    a2c = np.where(a2 > 2.0 * _PIX_VAR, a2 - _PIX_VAR, a2)
    b2c = np.where(b2 > 2.0 * _PIX_VAR, b2 - _PIX_VAR, b2)
    denom = a2c + b2c
    with np.errstate(divide='ignore', invalid='ignore'):
        e_mag = np.where(denom > 0, (a2c - b2c) / np.maximum(denom, 1e-30), 0.0)
    e_mag = np.clip(np.nan_to_num(e_mag, nan=0.0), 0.0, 1.0)

    e1 = e_mag * np.cos(2.0 * th)
    e2 = e_mag * np.sin(2.0 * th)

    # --- uncertainty on each component -------------------------------------
    with np.errstate(divide='ignore', invalid='ignore'):
        sig_snr = np.where(snr > 0, _SHAPE_NOISE_K / np.maximum(snr, 1e-30), np.inf)
        sig_pix = _PIX_VAR / np.maximum(a2 + b2, 1e-30)
    sig_snr = np.nan_to_num(sig_snr, nan=np.inf, posinf=np.inf)
    sigma_e = np.sqrt(sig_snr ** 2 + sig_pix ** 2)

    # --- derived significance and angular uncertainty ----------------------
    with np.errstate(divide='ignore', invalid='ignore'):
        elong_snr = np.where(sigma_e > 0, e_mag / sigma_e, 0.0)
        sigma_theta = np.where(e_mag > 0,
                               sigma_e / (2.0 * np.maximum(e_mag, 1e-30)),
                               np.inf)
    elong_snr = np.nan_to_num(elong_snr, nan=0.0, posinf=0.0)
    # An angle is never known better than "somewhere in 180 degrees".
    sigma_theta = np.clip(np.nan_to_num(sigma_theta, nan=np.pi, posinf=np.pi),
                          0.0, np.pi)

    return {
        'snr': snr, 'fwhm_px': fwhm_px,
        'e_mag': e_mag, 'e1': e1, 'e2': e2,
        'sigma_e': sigma_e, 'sigma_theta': sigma_theta,
        'elong_snr': elong_snr,
    }


def _sampling_mask(cat: dict,
                   min_fwhm_px: float = _MIN_FWHM_PX,
                   min_semiminor_px: float = _MIN_SEMIMINOR_PX,
                   sat_level: Optional[float] = None) -> np.ndarray:
    """
    Boolean mask of detections whose SHAPE is measurable at all.

    This is the cut that applies to every shape-derived product -- FWHM,
    ellipticity, orientation, tilt -- because it is about whether the pixels
    support a measurement, not about what the measurement says.

    Rejects:
      * undersampled stars (FWHM below Nyquist, or an unresolved minor axis),
        whose moments track subpixel phase rather than the PSF;
      * saturated / flat-topped stars, whose profile is clipped so the wings
        dominate the moments and the core contributes nothing.

    Deliberately says nothing about roundness: an in-focus star is round and
    perfectly valid for an FWHM or tilt measurement. Only the orientation
    path adds that cut, via _orientation_mask.
    """
    a = np.asarray(cat['a'], dtype=np.float64)
    b = np.asarray(cat['b'], dtype=np.float64)
    fwhm_px = _FWHM_FACTOR * np.sqrt(np.maximum(a * b, 0.0))

    m = (fwhm_px >= float(min_fwhm_px)) & (b >= float(min_semiminor_px))

    if sat_level is not None and 'peak' in cat:
        peak = np.asarray(cat['peak'], dtype=np.float64)
        m &= (peak < float(sat_level))
    return m


def _orientation_mask(stats: dict,
                      max_sigma_theta_deg: float = _MAX_SIGMA_THETA_DEG,
                      min_elong_snr: float = _MIN_ELONG_SNR,
                      min_abs_elong: float = _MIN_ABS_ELONG,
                      for_fit: bool = False,
                      min_snr: float = _MIN_FIT_SNR) -> np.ndarray:
    """
    Boolean mask of stars whose POSITION ANGLE is worth believing.

    _SASPRO_SHAPE_FIT_V2: for_fit=True selects on measurement QUALITY only -- sampling,
    noise, a finite angle -- and never on the value being measured.

    This distinction is the whole correctness question for the surface fit.
    The three thresholds below (|e|, elongation SNR, sigma_theta) are all
    functions of the measured e_mag, so using them to choose which stars
    enter a fit of e_mag keeps only the stars whose noise made them look
    oblong, and deletes the round ones that should have pulled the result
    toward zero. On a frame of genuinely round stars that bias alone
    produces a smooth, confident ~0.20 elongation surface out of nothing.
    The (e1, e2) vector representation already handles round stars correctly
    -- they contribute near-zero vectors -- so the fit wants every
    well-measured star, elongated or not, and the elongation threshold
    belongs on the FITTED surface afterwards.

    for_fit=False keeps the original per-star behaviour described below, which
    is right where a single star's own angle is the output: do not draw an
    arrow on a star that has no axis.

    Applied on top of _sampling_mask, and only where direction is the
    quantity of interest. Three conditions, all necessary, because each
    catches a case the others let through:

      * |e| >= min_abs_elong -- the elongation is large enough to matter,
        whatever the error bars say. Without this a bright star with a 1%
        elongation passes both tests below on the strength of its SNR.
      * elongation detected at min_elong_snr sigma -- it is a real
        measurement rather than noise that happened to look oblong. This is
        what rejects the faint end.
      * the angle is pinned to better than max_sigma_theta_deg -- a
        direction known to +/- 40 degrees is not a direction.

    The three together mean: big enough to care about, real, and pointing
    somewhere definite.
    """
    if for_fit:
        # _SASPRO_SHAPE_FIT_V2: quality only. Every threshold below is a function of the
        # measured e_mag, so applying them here would select the stars whose
        # noise looked oblong and discard the round ones that should pull a
        # weighted fit toward zero.
        return (np.isfinite(stats['sigma_e'])
                & (stats['sigma_e'] > 0.0)
                & np.isfinite(stats['sigma_theta'])
                & (stats['fwhm_px'] >= _MIN_FWHM_PX)
                & (stats['snr'] >= float(min_snr)))

    max_rad = math.radians(float(max_sigma_theta_deg))
    return ((stats['e_mag'] >= float(min_abs_elong))
            & (stats['elong_snr'] >= float(min_elong_snr))
            & (stats['sigma_theta'] <= max_rad)
            & np.isfinite(stats['sigma_theta']))


def _pack_catalogue(stars, good, rms: float, sigma: Optional[float] = None) -> dict:
    """Slice a SEP output table by `good` into the catalogue dict used here."""
    names = stars.dtype.names
    n = int(np.count_nonzero(good))

    def col(key, default=0.0):
        if key in names:
            return stars[key][good].astype(np.float64, copy=False)
        return np.full(n, float(default), dtype=np.float64)

    return {
        'x':     col('x'),
        'y':     col('y'),
        'a':     col('a'),
        'b':     col('b'),
        'theta': col('theta'),
        'flux':  col('flux'),
        'npix':  col('npix', 9.0),
        'peak':  col('peak'),
        'n':     n,
        'rms':   float(rms),
        **({'sigma': float(sigma)} if sigma is not None else {}),
    }


def _quality_filter(stars, rms: float, min_area: int,
                    sat_level: Optional[float],
                    min_fwhm_px: float,
                    min_semiminor_px: float) -> Tuple[np.ndarray, dict]:
    """
    Shared quality cut for both detection helpers.

    Returns (mask, attrition) where `attrition` records how many detections
    each stage removed. The counts are not bookkeeping for its own sake: when
    an orientation map comes out empty the user needs to know whether the
    frame had no stars, or plenty of stars that were all undersampled, and
    those call for completely different fixes (longer sub vs. the realisation
    that the optical train cannot resolve its own PSF).
    """
    names = stars.dtype.names
    n_raw = len(stars)
    flag = stars['flag'] if 'flag' in names else np.zeros(n_raw, dtype=int)
    npix = stars['npix'] if 'npix' in names else np.full(n_raw, min_area + 1)
    a = stars['a']; b = stars['b']

    m_flag = (flag == 0)
    m_area = (npix >= min_area)
    m_shape = (a > 0) & (b > 0) & (b <= a)
    base = m_flag & m_area & m_shape

    cat_all = {'a': a, 'b': b, 'peak': (stars['peak'] if 'peak' in names
                                        else np.zeros(n_raw))}
    m_sample = _sampling_mask(cat_all, min_fwhm_px=min_fwhm_px,
                              min_semiminor_px=min_semiminor_px,
                              sat_level=sat_level)

    good = base & m_sample
    attrition = {
        'raw': int(n_raw),
        'rejected_flag': int(np.count_nonzero(~m_flag)),
        'rejected_area': int(np.count_nonzero(m_flag & ~m_area)),
        'rejected_shape': int(np.count_nonzero(m_flag & m_area & ~m_shape)),
        'rejected_undersampled': int(np.count_nonzero(base & ~m_sample)),
        'kept': int(np.count_nonzero(good)),
    }
    return good, attrition


def _detect_stars(gray: np.ndarray,
                  thresh_sigma: float = 5.0,
                  min_area: int = 5,
                  min_fwhm_px: float = _MIN_FWHM_PX,
                  min_semiminor_px: float = _MIN_SEMIMINOR_PX,
                  sat_level: Optional[float] = None) -> Optional[dict]:
    """
    Run SEP once on `gray` and return a quality-filtered star catalogue.

    Cuts applied:
      - flag == 0                   (no blending, no truncation)
      - npix >= min_area            (rejects hot pixels / single-pixel spikes)
      - a > 0, b > 0, b <= a        (well-formed shape ellipse)
      - FWHM >= min_fwhm_px and b >= min_semiminor_px
                                    (the PSF is actually sampled -- see the
                                     notes above _shape_stats; without this
                                     every 1-2 px detection contributes a
                                     shape measurement that is really a
                                     readout of its subpixel position)
      - peak < sat_level            (when a saturation level is supplied)

    Returns None when nothing survives. Otherwise a dict of float64 arrays:
      x, y, a, b, theta, flux, npix, peak, plus n (count), rms (background
      sigma, needed for the per-star SNR) and attrition (per-stage counts).
    """
    data = np.ascontiguousarray(gray, dtype=np.float32)
    try:
        bkg = sep.Background(data)
        stars = sep.extract(data - bkg.back(),
                            thresh=thresh_sigma,
                            err=bkg.globalrms)
    except Exception:
        return None
    if stars is None or len(stars) == 0:
        return None

    good, attrition = _quality_filter(stars, float(bkg.globalrms), min_area,
                                      sat_level, min_fwhm_px, min_semiminor_px)
    if not np.any(good):
        return None

    cat = _pack_catalogue(stars, good, float(bkg.globalrms))
    cat['attrition'] = attrition
    return cat


def _detect_stars_waterfall(gray: np.ndarray,
                            sigma_ladder: Tuple[float, ...] = (100.0, 50.0, 25.0, 12.0, 6.0, 3.0),
                            target_count: int = 1000,
                            min_area: int = 5,
                            min_fwhm_px: float = _MIN_FWHM_PX,
                            min_semiminor_px: float = _MIN_SEMIMINOR_PX,
                            sat_level: Optional[float] = None) -> Optional[dict]:
    """
    Descending-threshold star detection. Runs SEP background estimation ONCE
    (the expensive part) and then re-extracts with progressively lower SNR
    thresholds until the quality-filtered catalogue has >= target_count
    stars, or the ladder is exhausted.

    Motivation: on a rich star field, sep.extract at low sigma (say 3-5)
    can find 30k+ detections whose per-star shape measurements are noisy
    and get filtered out anyway. That's minutes of wasted work when a few
    hundred high-SNR stars are already enough to constrain a smooth
    focal-plane surface. Starting high and descending only if needed usually
    converges after 2-3 extract calls on a well-populated frame.

    The quality filter (flag==0, npix>=min_area, well-formed shape ellipse,
    adequately sampled, unsaturated) is applied at each rung, so
    `target_count` counts stars that are actually USABLE rather than raw
    detections. That matters more than it sounds: on an undersampled frame
    the raw count races past the target while the usable count stays at
    zero, and the old behaviour would stop climbing down the ladder on the
    strength of detections that every downstream fit then discarded.

    The returned catalogue is what came out of the LAST rung run -- either
    the first rung to reach target_count, or the final rung at the bottom of
    the ladder if we never got there.
    Returns None if even sigma_ladder[-1] produced no usable stars.
    """
    data = np.ascontiguousarray(gray, dtype=np.float32)
    try:
        bkg = sep.Background(data)
    except Exception:
        return None
    sub = data - bkg.back()
    rms = bkg.globalrms

    # _SASPRO_SHAPE_FIT_V2: sat_level was threaded all the way down to _sampling_mask
    # but never given a value by any caller, so the saturation cut never ran.
    # A clipped core measures rounder than the star is, and carries the
    # largest weight of anything in the frame.
    if sat_level is None:
        sat_level = _estimate_sat_level(data)

    best: Optional[dict] = None
    for sig in sigma_ladder:
        try:
            stars = sep.extract(sub, thresh=float(sig), err=rms)
        except Exception:
            # Pixstack overflow at very low sigma on a huge field will land
            # here; keep the best catalogue we had and stop descending.
            break
        if stars is None or len(stars) == 0:
            continue

        good, attrition = _quality_filter(stars, float(rms), min_area,
                                          sat_level, min_fwhm_px,
                                          min_semiminor_px)
        n_good = int(np.count_nonzero(good))
        if n_good == 0:
            continue

        best = _pack_catalogue(stars, good, float(rms), sigma=float(sig))
        best['attrition'] = attrition
        best['sat_level'] = sat_level     # _SASPRO_SHAPE_FIT_V2: reported in stars_band
        if n_good >= target_count:
            break

    return best


def _eval_poly_upsampled(sol: np.ndarray,
                         exps: list,
                         H: int, W: int,
                         coarse: int = 128) -> np.ndarray:
    """
    Evaluate a low-degree 2D polynomial on a coarse grid then bilinearly
    upsample to (H, W). Vastly faster than per-pixel evaluation on full-res.
    """
    ch = max(2, min(int(coarse), H))
    cw = max(2, min(int(coarse), W))
    yv = np.linspace(0.0, float(H - 1), ch, dtype=np.float64)
    xv = np.linspace(0.0, float(W - 1), cw, dtype=np.float64)
    Xc, Yc = np.meshgrid(xv, yv)

    # Cache x**i and y**j to avoid recomputing powers per term.
    max_i = max((i for (i, _) in exps), default=0)
    max_j = max((j for (_, j) in exps), default=0)
    Xpow = [np.ones_like(Xc)] + [None] * max_i
    for i in range(1, max_i + 1):
        Xpow[i] = Xpow[i - 1] * Xc
    Ypow = [np.ones_like(Yc)] + [None] * max_j
    for j in range(1, max_j + 1):
        Ypow[j] = Ypow[j - 1] * Yc

    Z = np.zeros_like(Xc, dtype=np.float64)
    for c, (i, j) in zip(sol, exps):
        Z += float(c) * Xpow[i] * Ypow[j]

    return _ndzoom(Z, (H / ch, W / cw), order=1, mode='nearest').astype(np.float32, copy=False)


def _fit_and_render_surface(x: np.ndarray, y: np.ndarray, values: np.ndarray,
                            H: int, W: int,
                            deg: int = 3,
                            coarse: int = 128,
                            sigma_clip: float = 3.0,
                            weights: Optional[np.ndarray] = None):
    """
    Fit a 2D polynomial of degree `deg` to scattered (x, y, values), then
    render on a full (H, W) grid via coarse eval + bilinear upsample.

    Automatically lowers `deg` if there aren't enough samples to constrain
    the requested order (need (deg+1)(deg+2)/2 samples for a unique fit).

    `weights` are passed through to fit_2d_poly as inverse-variance weights.

    Returns (surface_HxW_float32, (raw_min, raw_max)).
    """
    n = int(values.size)
    while deg > 1 and n < (deg + 1) * (deg + 2) // 2:
        deg -= 1
    sol, exps = fit_2d_poly(x, y, values, deg=deg, sigma_clip=sigma_clip,
                            weights=weights)
    surf = _eval_poly_upsampled(sol, exps, H, W, coarse=coarse)
    return surf, (float(np.min(values)), float(np.max(values)))


# --- _SASPRO_LOCAL_AXES : begin --------------------------------------------------
# Samples required per polynomial coefficient before a degree is allowed. A
# 2D polynomial of degree d has (d+1)(d+2)/2 coefficients, so degree 4 needs
# 300 stars, degree 3 needs 200, degree 2 needs 120.
_POLY_SAMPLES_PER_COEF = 20
_POLY_MAX_DEGREE = 4

# A tick cell needs at least this many stars before its own measurement is
# trusted; below it the cell falls back to the fitted surface, drawn faint.
_BIN_MIN_STARS = 6

# --- _SASPRO_KERNEL_FIELD : begin --------------------------------------------------
# Target stars per grid cell. Smaller cells follow more structure and carry
# more noise; the Gaussian below sets the real smoothing scale, so this only
# has to be fine enough not to throw structure away before smoothing.
_KERNEL_CELL_STARS = 8
# Smoothing bandwidth, in cells. ~1.6 cells at 8 stars/cell averages roughly
# 60-80 stars per output point, which is where the hold-out error bottoms out.
_KERNEL_SIGMA_CELLS = 1.6
_KERNEL_MIN_CELLS = 5
_KERNEL_MAX_CELLS = 64


def _kernel_axis_field(x, y, e1, e2, w, H: int, W: int,
                       cell_stars: float = _KERNEL_CELL_STARS,
                       sigma_cells: float = _KERNEL_SIGMA_CELLS):
    """
    _SASPRO_KERNEL_FIELD: Nadaraya-Watson kernel regression of the ellipticity vector.

        E(p) = sum_i w_i K(|p - p_i|) e_i / sum_i w_i K(|p - p_i|)

    evaluated by binning onto a coarse grid, Gaussian-blurring numerator and
    denominator, dividing, and bilinearly upsampling. O(N + grid), under a
    millisecond on a multi-megapixel frame.

    This replaces a global least-squares polynomial, which had to describe
    the whole focal plane with 15 coefficients and was therefore stiffest
    exactly at the corners -- on a real frame it reported |e| = 0.042 in a
    corner whose stars measure 0.307. Five-fold hold-out over 2073 stars:
    kernel 0.1330, polynomial degree 4 0.1420, degree 2 0.1527, thin-plate
    RBF 0.1630.

    Smoothing e1 and e2 and recovering the angle afterwards is what keeps the
    axis ambiguity harmless: +/-180 degrees is the same point in (e1, e2), so
    there is no branch to cut and no bearing to propagate around the frame.

    Returns (surf_e1, surf_e2, support, (nx, ny, sigma)). `support` is the
    smoothed weight denominator normalised to [0, 1] -- a free, smooth map of
    where the frame actually has data.
    """
    from scipy.ndimage import gaussian_filter

    x = np.asarray(x, float); y = np.asarray(y, float)
    e1 = np.asarray(e1, float); e2 = np.asarray(e2, float)
    w = np.nan_to_num(np.asarray(w, float), nan=0.0, posinf=0.0)
    n = int(x.size)

    # Grid sized for ~cell_stars per cell, keeping the frame's aspect ratio.
    target = max(4.0, n / max(float(cell_stars), 1.0))
    nx = int(round(math.sqrt(max(target * W / max(float(H), 1.0), 1.0))))
    ny = int(round(target / max(nx, 1)))
    nx = int(np.clip(nx, _KERNEL_MIN_CELLS, _KERNEL_MAX_CELLS))
    ny = int(np.clip(ny, _KERNEL_MIN_CELLS, _KERNEL_MAX_CELLS))

    cw, ch = float(W) / nx, float(H) / ny
    ix = np.clip((x / cw).astype(int), 0, nx - 1)
    iy = np.clip((y / ch).astype(int), 0, ny - 1)

    aw = np.zeros((ny, nx)); a1 = np.zeros((ny, nx)); a2 = np.zeros((ny, nx))
    np.add.at(aw, (iy, ix), w)
    np.add.at(a1, (iy, ix), w * e1)
    np.add.at(a2, (iy, ix), w * e2)

    s = float(sigma_cells)
    sw = gaussian_filter(aw, s, mode="nearest")
    g1 = gaussian_filter(a1, s, mode="nearest") / np.maximum(sw, 1e-12)
    g2 = gaussian_filter(a2, s, mode="nearest") / np.maximum(sw, 1e-12)
    sup = sw / max(float(sw.max()), 1e-12)

    def _up(a):
        b = _ndzoom(a, (H / ny, W / nx), order=1, mode="nearest")
        if b.shape != (H, W):            # zoom rounding
            out = np.zeros((H, W), dtype=b.dtype)
            hh, ww = min(H, b.shape[0]), min(W, b.shape[1])
            out[:hh, :ww] = b[:hh, :ww]
            if hh < H:
                out[hh:, :] = out[hh - 1, :]
            if ww < W:
                out[:, ww:] = out[:, ww - 1:ww]
            b = out
        return b.astype(np.float32, copy=False)

    return _up(g1), _up(g2), _up(sup), (nx, ny, s)
# --- _SASPRO_KERNEL_FIELD : end ----------------------------------------------------


def _adaptive_poly_degree(n: int, requested: int,
                          dmax: int = _POLY_MAX_DEGREE) -> int:
    """_SASPRO_LOCAL_AXES: the highest degree this many stars can actually support."""
    d = max(1, min(int(requested), int(dmax)))
    while d > 1:
        n_coef = (d + 1) * (d + 2) // 2
        if int(n) >= _POLY_SAMPLES_PER_COEF * n_coef:
            break
        d -= 1
    return d


def _local_axis_bins(x, y, e1, e2, sigma_e, H: int, W: int,
                     nx: int, ny: int,
                     min_stars: int = _BIN_MIN_STARS) -> list:
    """
    _SASPRO_LOCAL_AXES: inverse-variance mean ellipticity vector per lattice cell.

    A direct measurement of the stars inside each cell -- no surface, no
    extrapolation, nothing that can carry a corner's answer in from the rest
    of the frame. On the frame this was built against, a cell of 33 stars
    landed 1 degree from the independently stacked PSF of the same region,
    where the global degree-2 surface was 46 degrees away.

    Each entry: ix, iy, n, e_mag, theta (radians, the module's y-down frame),
    ok (n >= min_stars).
    """
    x = np.asarray(x, float); y = np.asarray(y, float)
    cw, ch = float(W) / max(int(nx), 1), float(H) / max(int(ny), 1)
    cx = np.clip((x / cw).astype(int), 0, int(nx) - 1)
    cy = np.clip((y / ch).astype(int), 0, int(ny) - 1)
    w = 1.0 / np.maximum(np.asarray(sigma_e, float), 1e-6) ** 2
    w = np.nan_to_num(w, nan=0.0, posinf=0.0)
    out = []
    for iy in range(int(ny)):
        for ix in range(int(nx)):
            m = (cx == ix) & (cy == iy)
            n = int(np.count_nonzero(m))
            if n == 0 or float(w[m].sum()) <= 0.0:
                out.append({'ix': ix, 'iy': iy, 'n': n, 'e_mag': 0.0,
                            'theta': 0.0, 'ok': False})
                continue
            sw = float(w[m].sum())
            m1 = float(np.sum(w[m] * np.asarray(e1, float)[m]) / sw)
            m2 = float(np.sum(w[m] * np.asarray(e2, float)[m]) / sw)
            out.append({'ix': ix, 'iy': iy, 'n': n,
                        'e_mag': math.hypot(m1, m2),
                        'theta': 0.5 * math.atan2(m2, m1),
                        'ok': n >= int(min_stars)})
    return out
# --- _SASPRO_LOCAL_AXES : end ----------------------------------------------------


def _fit_orientation_field(cat: dict, H: int, W: int,
                           deg: int = 2,
                           coarse: int = 128,
                           sigma_clip: float = 3.0,
                           max_sigma_theta_deg: float = _MAX_SIGMA_THETA_DEG,
                           min_elong_snr: float = _MIN_ELONG_SNR,
                           min_abs_elong: float = _MIN_ABS_ELONG,
                           min_stars: int = _MIN_ORIENT_STARS) -> Optional[dict]:
    """
    Smooth position-angle field from the stars that actually have one.

    Fits the ellipticity VECTOR (e1, e2) rather than the unit direction
    (sin 2th, cos 2th) the old code used. The difference is the whole point:

      * Unit-vector fitting throws away |e|, so a star elongated by 2% and
        one elongated by 40% pull the local direction equally hard. On a
        frame where most stars are near-round, the fit is then dominated by
        the ones carrying no directional information, and the resulting map
        is a smooth interpolation of noise -- smooth enough to look like a
        measurement of tilt or spacing.
      * Fitting (e1, e2) lets each star contribute a vector whose LENGTH is
        its elongation. A round star contributes a near-zero vector that
        barely moves the result, which is the correct weight for a star with
        no preferred axis. Recovering theta = 0.5*atan2(e2, e1) afterwards
        is unchanged.

    On top of that, stars are cut by _orientation_mask and the survivors are
    weighted by 1/sigma_e^2, so measurement quality enters twice: once as a
    threshold, once as a weight.

    A bonus of fitting the vector: |e_fit| = hypot(e1_fit, e2_fit) is a
    smoothed elongation magnitude that is guaranteed CONSISTENT with the
    direction map, because both come from the same fit. The separately
    fitted 1 - b/a surface has no such guarantee and the two can disagree.

    Returns None when too few stars survive -- which is a result, not a
    failure, and the caller should say so rather than render an empty map.
    Otherwise a dict with:
        theta_surf   (H, W) fitted angle in radians, [-pi/2, pi/2]
        theta_norm   (H, W) same mapped to [0, 1] for a hue ramp
        emag_surf    (H, W) fitted elongation magnitude |e|
        n_used       stars that passed the orientation cut
        n_input      stars offered to it
        theta_range  (min, max) measured angle over the surviving stars
        emag_range   (min, max) measured |e| over the surviving stars
        median_sigma_theta_deg   typical per-star angular uncertainty
    """
    stats = _shape_stats(cat, rms=float(cat.get('rms', 0.0)))
    # _SASPRO_SHAPE_FIT_V2: quality-only selection. Selecting on the measured |e| here
    # biases the fit upward without bound -- see _orientation_mask.
    keep = _orientation_mask(stats, for_fit=True)
    n_used = int(np.count_nonzero(keep))
    n_input = int(cat.get('n', len(keep)))
    if n_used < int(min_stars):
        return {
            'ok': False, 'n_used': n_used, 'n_input': n_input,
            'median_sigma_theta_deg': float(
                np.degrees(np.median(stats['sigma_theta']))
                if stats['sigma_theta'].size else float('nan')),
        }

    x = np.asarray(cat['x'], dtype=np.float64)[keep]
    y = np.asarray(cat['y'], dtype=np.float64)[keep]
    e1 = stats['e1'][keep]
    e2 = stats['e2'][keep]
    sig = stats['sigma_e'][keep]
    with np.errstate(divide='ignore'):
        w = 1.0 / np.maximum(sig, 1e-6) ** 2
    w = np.nan_to_num(w, nan=0.0, posinf=0.0)

    # _SASPRO_SHAPE_FIT_V2 -------------------------------------------------------
    # The nine anchor regions, measured directly from their own stars.
    zones = _anchor_zone_stats(x, y, e1, e2, sig, H, W)

    # w goes as SNR^2, so without a cap a few of the brightest stars -- the
    # ones most likely to be clipped, and so measured too round -- decide the
    # surface on their own.
    if w.size:
        cap = float(np.percentile(w[w > 0], _FIT_WEIGHT_CAP_PCT)) if np.any(w > 0) else 0.0
        if cap > 0.0:
            w = np.minimum(w, cap)

    # Equalise the anchor regions so a star-rich centre cannot outvote a
    # sparse corner. Without this the surface is whichever zone kept the most
    # stars, extrapolated over the rest of the frame.
    zidx = _zone_index(x, y, H, W)
    for _k in range(9):
        _m = zidx == _k
        if np.any(_m):
            _s = float(w[_m].sum())
            if _s > 0.0:
                w[_m] = w[_m] / _s
    # ------------------------------------------------------------------

    # A plane is usually the right model: sensor tilt and decentre both
    # produce a slowly varying axis field. Degree is capped at 2 so a
    # high-order FWHM setting does not drag the orientation fit into
    # modelling noise.
    # _SASPRO_LOCAL_AXES: degree by star count, not a flat cap of 2.
    #
    # A global quadratic has the least freedom exactly at the corners, so a
    # corner that disagrees with the rest of the frame is rotated toward the
    # frame mean and flattened. On a real 2073-star frame the degree-2 surface
    # was 46 degrees off the local measurement in one corner; degree 4 was 3.
    # The old cap of 2 dates from when round stars were excluded from the fit,
    # which made it far noisier than it is now.
    # The caller's `deg` is the FWHM setting; orientation should not inherit
    # it. Ask for the most this star count supports and let the rule decide.
    d = _adaptive_poly_degree(len(x), max(int(deg), _POLY_MAX_DEGREE))
    # _SASPRO_KERNEL_FIELD: kernel regression on the measurements, not a global fit.
    # The polynomial reported |e| = 0.042 in a corner measuring 0.307, because
    # 15 coefficients cannot describe a focal plane and are stiffest at the
    # edges. `d` above is retained only for the result dict.
    surf_e1, surf_e2, support, _kgrid = _kernel_axis_field(x, y, e1, e2, w, H, W)

    theta_surf = 0.5 * np.arctan2(surf_e2, surf_e1)
    emag_surf = np.hypot(surf_e1, surf_e2)

    th_used = 0.5 * np.arctan2(e2, e1)
    em_used = stats['e_mag'][keep]
    return {
        'ok': True,
        'theta_surf': theta_surf.astype(np.float32, copy=False),
        'theta_norm': ((theta_surf + np.pi / 2.0) / np.pi).astype(np.float32,
                                                                  copy=False),
        'emag_surf': emag_surf.astype(np.float32, copy=False),
        'n_used': n_used,
        'n_input': n_input,
        'theta_range': (float(th_used.min()), float(th_used.max())),
        'emag_range': (float(em_used.min()), float(em_used.max())),
        'median_sigma_theta_deg': float(
            np.degrees(np.median(stats['sigma_theta'][keep]))),
        # _SASPRO_SHAPE_FIT_V2: per-region numbers, plus the brightness band the fit
        # actually used, so the map can be checked against the stars.
        'zones': zones,
        # _SASPRO_LOCAL_AXES: per-cell measurements for the tick lattice.
        'bins': _local_axis_bins(x, y, e1, e2, sig, H, W,
                                 _TICK_GRID[0], _TICK_GRID[1]),
        'poly_degree': int(d),
        # _SASPRO_KERNEL_FIELD: (nx, ny, sigma_cells) of the kernel grid, and the smooth
        # map of local data support that falls out of its denominator.
        'kernel_grid': _kgrid,
        'support': support,
        'stars_band': {
            'min_snr': float(_MIN_FIT_SNR),
            'snr_pct': [float(v) for v in np.percentile(stats['snr'][keep],
                                                        [5, 50, 95])],
            'weight_cap_pct': float(_FIT_WEIGHT_CAP_PCT),
            'sat_level': cat.get('sat_level'),
        },
    }


# =============================================================================
# Aberration diagnosis: naming the cause, not just the symptom
# =============================================================================
# An orientation map says stars are elongated and which way. The two questions
# a user actually has are "is my sensor tilted?" and "is my backfocus wrong?",
# and those have distinct, separable signatures in the SAME ellipticity field
# the orientation map is built from.
#
# Work in the polar frame about the optical axis. With phi the position angle
# of a star about the field centre, project its ellipticity vector onto the
# radial frame:
#
#     e_plus  = -(e1 cos 2phi + e2 sin 2phi)      radial / tangential
#     e_cross = -(e2 cos 2phi - e1 sin 2phi)      the 45-degree component
#
# (The weak-lensing tangential/cross shear, which is exactly the same
# decomposition of exactly the same kind of field.)
#
#   e_plus > 0   stars elongated TANGENTIALLY (perpendicular to the radius)
#   e_plus < 0   stars elongated RADIALLY (pointing at the centre)
#
# Then the three causes separate:
#
#   Uniform <e1, e2> across the whole field, no radial structure
#       -> everything elongated the same way: trailing, guiding error, wind,
#          differential flexure. Not an optical problem at all.
#
#   e_plus coherent and growing with field radius, e_cross ~ 0
#       -> rotationally symmetric about the axis: corrector spacing error,
#          astigmatism, field curvature. Backfocus is the usual culprit, and
#          the SIGN says which way to move.
#
#   One-sided pattern -- magnitude varying linearly across the field rather
#   than symmetrically about the centre
#       -> tilt. The quantitative tilt number comes from the FWHM paraboloid
#          fit rather than from here; this just corroborates it.
#
#   e_cross large
#       -> the pattern is not symmetric about the assumed centre. Either the
#          optical axis is not at the frame centre (decentred corrector,
#          cropped frame) or something is going on that none of these models
#          describe.
#
# A note on which stars take part, because it looks inconsistent with the
# orientation gate and is not. _orientation_mask exists to decide whether ONE
# star's angle may be drawn, and demands that star be individually
# convincing. Fitting a GLOBAL pattern is a different question: many
# individually-marginal measurements average down, and excluding the rounder
# ones would bias every fit below -- the centre of the field is where the
# aberration is genuinely smallest, so a hard floor on |e| would carve out
# exactly the points that pin the fit's zero point. So the decomposition uses
# the sampling gate (which is about whether the measurement is real at all)
# plus inverse-variance weighting, and no elongation floor.

def _weighted_mean_se(v: np.ndarray, w: np.ndarray) -> Tuple[float, float]:
    """Inverse-variance weighted mean and its standard error."""
    sw = float(np.sum(w))
    if sw <= 0:
        return 0.0, float('inf')
    m = float(np.sum(w * v) / sw)
    return m, float(1.0 / math.sqrt(sw))


def _aberration_decomposition(cat: dict, H: int, W: int,
                              center: Optional[Tuple[float, float]] = None) -> dict:
    """
    Split the ellipticity field into uniform, radial and cross components.

    Returns a dict of measured amplitudes, each with a standard error and a
    significance (amplitude / error), so a caller can say "4.2 sigma" rather
    than eyeballing a picture:

        uniform_e, uniform_angle_deg, uniform_se, uniform_sig
            the field-wide mean ellipticity vector -- trailing

        radial_corner, radial_se, radial_sig, radial_sense
            e_plus extrapolated to the corner, from a weighted fit of
            e_plus against normalised radius squared. radial_sense is
            'tangential' for positive, 'radial' for negative.

        cross_corner, cross_se, cross_sig
            the same for e_cross; should be consistent with zero when the
            pattern really is symmetric about `center`.

        asymmetry, asymmetry_se, asymmetry_sig
            amplitude of a linear (one-sided) variation in |e| across the
            field -- the tilt-like term, in the sense that a tilted sensor
            aberrates one side more than the other.

        n_used, center
    """
    stats = _shape_stats(cat, rms=float(cat.get('rms', 0.0)))
    # Sampling gate only; see the note above on why there is no |e| floor.
    good = np.isfinite(stats['sigma_e']) & (stats['sigma_e'] > 0)
    n_used = int(np.count_nonzero(good))
    out = {'n_used': n_used, 'center': center}
    if n_used < 20:
        out['ok'] = False
        return out

    x = np.asarray(cat['x'], float)[good]
    y = np.asarray(cat['y'], float)[good]
    e1 = stats['e1'][good]
    e2 = stats['e2'][good]
    sig = stats['sigma_e'][good]
    w = 1.0 / np.maximum(sig, 1e-6) ** 2

    cx, cy = (W / 2.0, H / 2.0) if center is None else (float(center[0]),
                                                        float(center[1]))
    dx, dy = x - cx, y - cy
    phi = np.arctan2(dy, dx)
    R = np.hypot(dx, dy)
    Rmax = math.hypot(max(cx, W - cx), max(cy, H - cy))   # corner distance
    u = (R / max(Rmax, 1e-9)) ** 2                        # 0 at centre, 1 at corner

    c2, s2 = np.cos(2.0 * phi), np.sin(2.0 * phi)
    e_plus = -(e1 * c2 + e2 * s2)
    e_cross = -(e2 * c2 - e1 * s2)

    # --- uniform component (trailing) -----------------------------------
    m1, se1 = _weighted_mean_se(e1, w)
    m2, se2 = _weighted_mean_se(e2, w)
    umag = math.hypot(m1, m2)
    use = math.hypot(se1, se2) / math.sqrt(2.0)
    out.update({
        'uniform_e': umag,
        'uniform_angle_deg': math.degrees(0.5 * math.atan2(m2, m1)) % 180.0,
        'uniform_se': use,
        'uniform_sig': (umag / use) if use > 0 else 0.0,
    })

    # --- radial component, fitted against normalised R^2 ------------------
    # A free intercept is allowed rather than forcing the fit through the
    # origin: a nonzero value at the centre would mean the pattern is not
    # centred where we assumed, and it is better to measure that than to
    # absorb it into the slope.
    def _lin_fit(val):
        A = np.column_stack([np.ones_like(u), u])
        Aw = A * np.sqrt(w)[:, None]
        vw = val * np.sqrt(w)
        try:
            sol, *_ = np.linalg.lstsq(Aw, vw, rcond=None)
        except np.linalg.LinAlgError:
            return 0.0, 0.0, float('inf')
        resid = val - A.dot(sol)
        dof = max(1, val.size - 2)
        chi2 = float(np.sum(w * resid ** 2)) / dof
        try:
            cov = np.linalg.inv(Aw.T @ Aw) * max(chi2, 1.0)
            se_corner = float(math.sqrt(max(cov[0, 0] + 2 * cov[0, 1] + cov[1, 1], 0.0)))
        except np.linalg.LinAlgError:
            se_corner = float('inf')
        corner = float(sol[0] + sol[1])       # value at u = 1
        return corner, float(sol[0]), se_corner

    rc, r0, rse = _lin_fit(e_plus)
    xc, x0, xse = _lin_fit(e_cross)
    out.update({
        'radial_corner': rc,
        'radial_centre': r0,
        'radial_se': rse,
        'radial_sig': (abs(rc) / rse) if rse > 0 and np.isfinite(rse) else 0.0,
        'radial_sense': 'tangential' if rc > 0 else 'radial',
        'cross_corner': xc,
        'cross_se': xse,
        'cross_sig': (abs(xc) / xse) if xse > 0 and np.isfinite(xse) else 0.0,
    })

    # --- one-sided (tilt-like) term in |e| --------------------------------
    # Fit |e| against normalised x and y. A rotationally symmetric pattern
    # has no linear term; a tilted sensor does, because one side sits
    # further from focus than the other.
    emag = np.hypot(e1, e2)
    nx = dx / max(W / 2.0, 1e-9)
    ny = dy / max(H / 2.0, 1e-9)
    A = np.column_stack([np.ones_like(nx), nx, ny])
    Aw = A * np.sqrt(w)[:, None]
    try:
        sol, *_ = np.linalg.lstsq(Aw, emag * np.sqrt(w), rcond=None)
        resid = emag - A.dot(sol)
        dof = max(1, emag.size - 3)
        chi2 = float(np.sum(w * resid ** 2)) / dof
        cov = np.linalg.inv(Aw.T @ Aw) * max(chi2, 1.0)
        amp = float(math.hypot(sol[1], sol[2]))
        ase = float(math.sqrt(max(cov[1, 1] + cov[2, 2], 0.0)))
        out.update({
            'asymmetry': amp,
            'asymmetry_angle_deg': math.degrees(math.atan2(sol[2], sol[1])) % 360.0,
            'asymmetry_se': ase,
            'asymmetry_sig': (amp / ase) if ase > 0 else 0.0,
        })
    except np.linalg.LinAlgError:
        out.update({'asymmetry': 0.0, 'asymmetry_angle_deg': 0.0,
                    'asymmetry_se': float('inf'), 'asymmetry_sig': 0.0})

    out['ok'] = True
    return out


def _coverage_mask(x: np.ndarray, y: np.ndarray, H: int, W: int,
                   coarse: int = 128) -> np.ndarray:
    """
    Boolean (H, W) mask of the region actually sampled by stars -- the convex
    hull of the detection positions.

    Needed because a fitted polynomial keeps evaluating outside the data. On
    a frame whose stars sit in the middle (vignetted corners, a bright
    nebula, a short sub) a degree-3 surface extrapolates into the corners
    with no constraint at all, and the values it invents there are not
    measurements of anything.

    Computed on a coarse grid and upsampled, which is ample: the mask only
    decides which pixels inform the display range.
    Falls back to all-True when the hull is degenerate (collinear or too few
    points), so a caller never loses the whole frame to a geometry failure.
    """
    full = np.ones((H, W), dtype=bool)
    pts = np.column_stack([np.asarray(x, float), np.asarray(y, float)])
    if pts.shape[0] < 4:
        return full
    try:
        from scipy.spatial import Delaunay
        tri = Delaunay(pts)
        ch = max(2, min(int(coarse), H))
        cw = max(2, min(int(coarse), W))
        gy = np.linspace(0.0, float(H - 1), ch)
        gx = np.linspace(0.0, float(W - 1), cw)
        GX, GY = np.meshgrid(gx, gy)
        inside = tri.find_simplex(np.column_stack([GX.ravel(),
                                                   GY.ravel()])) >= 0
        small = inside.reshape(ch, cw).astype(np.float32)
        if not small.any():
            return full
        big = _ndzoom(small, (H / ch, W / cw), order=0, mode='nearest')
        if big.shape != (H, W):          # zoom rounding
            big = np.resize(big, (H, W))
        return big > 0.5
    except Exception:
        return full


# --- _SASPRO_MAP_CONFIDENCE : begin --------------------------------------------------
# Floor for the confidence multiplier inside the covered region. A weak area
# should read as "nothing measurable here", not as a hole in the image.
# _SASPRO_KERNEL_FIELD: raised from 0.10. With the field measured rather than modelled,
# the dark areas are now real low-elongation regions rather than places the
# polynomial happened to dip through, and they should stay legible.
_CONF_FLOOR = 0.25

# _SASPRO_STRENGTH_WHEEL: shaping for the strength channel (the ellipticity map used as the
# orientation map's brightness). 1.0 is a literal multiply.
_STRENGTH_GAMMA = 0.6

# _SASPRO_AXIS_TICKS: confidence well outside the star hull. Not zero -- a hard zero
# reads as a rendering fault, and the hull is inset by about one star spacing
# by construction, so the first band beyond it is barely extrapolating.
_COVER_OUTSIDE = 0.22

# Sampling ramp for a FITTED FWHM surface, in pixels. At _MIN_FWHM_PX the
# second moments track subpixel phase rather than the star, so neither the
# axis ratio nor the position angle means anything; half a pixel above it they
# do. Individual stars below the limit are already rejected at detection --
# this is about a REGION of the frame whose stars are all marginal.
_CONF_FWHM_LO = _MIN_FWHM_PX
_CONF_FWHM_HI = _MIN_FWHM_PX + 0.5


def _get_cmap(name: str):
    """
    Version-proof matplotlib colormap lookup.

    SurfaceDialog used cm.get_cmap(), which matplotlib deprecated in 3.7 and
    REMOVED in 3.9 -- there it raises AttributeError, which would take out
    every Image Peeker heatmap (FWHM, ellipticity, orientation) as soon as
    matplotlib is updated. matplotlib.colormaps[...] is the supported form;
    the old call is kept as a fallback for older matplotlib.
    """
    try:
        import matplotlib
        return matplotlib.colormaps[name]
    except Exception:
        from matplotlib import cm as _cm
        return _cm.get_cmap(name)


def _hsl_to_rgb(h: np.ndarray, s: np.ndarray, l: np.ndarray) -> np.ndarray:
    """
    _SASPRO_HSL_LIGHTNESS: vectorised HSL -> uint8 RGB.

    HSL, not HSV, and the difference is the whole point. In HSV every colour
    below V = 1 is a darkened version of itself, so a hue map whose strength
    is mostly mid-range renders as one murky wash. In HSL, L = 0.5 is the
    fully saturated colour, L above washes to white and below to black --
    the same black -> colour -> white structure the magma ellipticity map
    has, which is why that map reads cleanly on the same data.

    Lets the orientation map carry angle in hue and strength in lightness
    without the two fighting each other.
    """
    h = np.clip(np.asarray(h, dtype=np.float32), 0.0, 1.0)
    s = np.clip(np.asarray(s, dtype=np.float32), 0.0, 1.0)
    l = np.clip(np.asarray(l, dtype=np.float32), 0.0, 1.0)
    c = (1.0 - np.abs(2.0 * l - 1.0)) * s
    hp = h * 6.0
    x = c * (1.0 - np.abs(np.mod(hp, 2.0) - 1.0))
    z = np.zeros_like(h)
    seg = np.floor(hp).astype(np.int32) % 6
    cond = [seg == i for i in range(6)]
    r = np.select(cond, [c, x, z, z, x, c])
    g = np.select(cond, [x, c, c, x, z, z])
    b = np.select(cond, [z, z, x, c, c, x])
    m = l - c / 2.0
    out = np.stack([r + m, g + m, b + m], axis=-1)
    return (np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)


def _conf_ramp(v: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Smoothstep from 0 at `lo` to 1 at `hi`, clamped outside."""
    v = np.asarray(v, dtype=np.float32)
    if not (hi > lo):
        return (v >= hi).astype(np.float32)
    t = np.clip((v - float(lo)) / (float(hi) - float(lo)), 0.0, 1.0)
    # Smoothstep, so the transition does not read as a contour line.
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def _confidence_surface(H: int, W: int,
                        emag_surf: Optional[np.ndarray] = None,
                        fwhm_px_surf: Optional[np.ndarray] = None,
                        coverage: Optional[np.ndarray] = None,
                        emag_full: float = _MIN_ABS_ELONG,
                        floor: float = _CONF_FLOOR,
                        margin_px: float = 0.0,
                        strength: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Per-pixel confidence in [0, 1] for modulating a map's brightness.

    Hue is the wrong channel to carry strength: it has no "nothing here"
    value, since 0 and 180 degrees are both bright red. Brightness does, so
    the two are separated -- hue says which way, brightness says how much to
    believe it.

    Factors, multiplied, each omitted when its input is None:

      emag_surf      fitted |e|. Zero elongation -> zero confidence, ramping
                     to full by emag_full. A fitted angle is defined for any
                     (e1, e2) however short the vector, and that is exactly
                     the case where the angle is meaningless.
      fwhm_px_surf   fitted FWHM in PIXELS. Below _CONF_FWHM_LO the shape
                     measurement describes the pixel grid.
      coverage       boolean hull of the detections. Outside it the surface is
                     a polynomial's invention, so the factor is hard zero --
                     not floored.

    The floor applies to the measurement terms only, so a weak but sampled
    region stays visible while an unsampled one goes black.
    """
    conf = np.ones((int(H), int(W)), dtype=np.float32)
    # _SASPRO_STRENGTH_WHEEL: an already-normalised [0,1] strength map -- in practice the
    # ellipticity map -- so the hue map's brightness IS the ellipticity the
    # user is looking at in the other window. Folded in before the floor, so
    # the weakest region sits at the floor rather than at black.
    if strength is not None:
        s = np.asarray(strength, dtype=np.float32)
        if s.shape == conf.shape:
            # Gamma, because a straight multiply by a frame-normalised map
            # takes the whole picture down (mean brightness 0.35 on a real
            # frame) while barely touching the strongest corner. At 0.6 the
            # ordering is unchanged, the strong corner stays at 0.87, and the
            # mid-range stays readable. Set to 1.0 for a literal multiply.
            conf *= np.clip(s, 0.0, 1.0) ** _STRENGTH_GAMMA
    if emag_surf is not None:
        conf *= _conf_ramp(np.abs(np.asarray(emag_surf, dtype=np.float32)),
                           0.0, float(emag_full))
    if fwhm_px_surf is not None:
        conf *= _conf_ramp(fwhm_px_surf, _CONF_FWHM_LO, _CONF_FWHM_HI)
    conf = np.clip(conf, float(floor), 1.0)
    if coverage is not None:
        cov = np.asarray(coverage)
        if cov.shape == conf.shape:
            conf = conf * _soft_coverage(cov, margin_px)
    return conf.astype(np.float32, copy=False)


def _soft_coverage(cov: np.ndarray, margin_px: float = 0.0,
                   outside: float = _COVER_OUTSIDE) -> np.ndarray:
    """
    _SASPRO_AXIS_TICKS: the coverage hull as a soft falloff rather than a hard edge.

    _coverage_mask is the convex hull of the star CENTROIDS, so it is inset
    from the frame edge by roughly one star spacing no matter how good the
    frame is -- no star's centre lands in the outermost band. Multiplying by
    it directly turned that band pure black, which reads as a rendering fault
    rather than as information, and showed the 128-grid stair-stepping that
    _coverage_mask upsamples with order=0.

    So: full confidence out to `margin_px` beyond the hull -- the scale on
    which the surface is still genuinely constrained -- then a smooth fade to
    `outside`. Beyond that the polynomial really is inventing values, and a
    dim region says so without punching a hole in the picture.
    """
    cov = np.asarray(cov).astype(bool)
    if not cov.any():
        return np.ones(cov.shape, dtype=np.float32)
    m = float(max(margin_px, 0.0))
    if m <= 0.0:
        return cov.astype(np.float32)
    try:
        from scipy.ndimage import distance_transform_edt
        d_out = distance_transform_edt(~cov).astype(np.float32)
    except Exception:
        return cov.astype(np.float32)
    # Smoothstep from 1 at the margin to `outside` a further margin out.
    t = np.clip((d_out - m) / max(m, 1.0), 0.0, 1.0)
    t = t * t * (3.0 - 2.0 * t)
    return (1.0 - (1.0 - float(outside)) * t).astype(np.float32)
# --- _SASPRO_MAP_CONFIDENCE : end ----------------------------------------------------


def _normalize_for_display(surf: np.ndarray,
                           mask: Optional[np.ndarray] = None,
                           lo_pct: float = 0.5,
                           hi_pct: float = 99.5) -> Tuple[np.ndarray, Tuple[float, float]]:
    """
    Scale a fitted surface to [0, 1] and return the range that scaling used.

    The point is that the returned range is the ONE the image was normalised
    with, so a colourbar labelled from it is telling the truth. Previously
    the image was normalised over the whole rendered surface while the
    colourbar was labelled with the range of the star measurements -- two
    different numbers. Where the surface extrapolated past the stars (which
    it does by design, into every corner the stars do not reach) the colours
    and the numbers beside them described different quantities, and on a
    centrally-sampled frame the real signal was squeezed into about a third
    of the ramp while extrapolated corners took the rest.

    Percentiles rather than min/max, over the covered region only, so a
    single wild cell cannot set the scale. Values outside the range clip,
    which keeps extrapolated corners pinned at the ends of the ramp instead
    of stealing it.
    """
    s = np.asarray(surf, dtype=np.float32)
    if mask is not None and np.any(mask):
        sel = s[mask]
    else:
        sel = s.ravel()
    sel = sel[np.isfinite(sel)]
    if sel.size == 0:
        return np.zeros_like(s), (0.0, 0.0)

    vmin = float(np.percentile(sel, lo_pct))
    vmax = float(np.percentile(sel, hi_pct))
    if not (vmax > vmin):
        vmin, vmax = float(sel.min()), float(sel.max())
    if not (vmax > vmin):
        return np.zeros_like(s), (vmin, vmax)

    norm = np.clip((s - vmin) / (vmax - vmin), 0.0, 1.0)
    return norm.astype(np.float32, copy=False), (vmin, vmax)


def _fit_tilt_paraboloid(x: np.ndarray, y: np.ndarray, fwhm_um: np.ndarray,
                         f_number: float,
                         sigma_clip: float = 2.5,
                         max_iter: int = 5) -> Optional[dict]:
    """
    Fit the physical tilt model FWHM^2 = FWHM0^2 + (p*x + q*y + r)^2 / N^2
    as a general 2D quadratic in (x, y):

        FWHM^2 = c00 + c10 x + c01 y + c20 x^2 + c11 x y + c02 y^2

    and recover the signed defocus plane dz(x,y) = p*x + q*y + r plus the
    baseline (seeing + optics) FWHM0. This replaces the older
        dz = plane fit of |dz|/N
    approach, which lost sign information -- a tilted sensor has both sides
    of the field defocused with opposite sign but the same |blur|, so fitting
    a plane to blur underestimates real tilt whenever the focus ridge crosses
    the sensor.

    The (p, q, r) -> (-p, -q, -r) sign flip is intrinsically ambiguous from a
    single image (physically indistinguishable), so we fix r >= 0 by
    convention. Corner-to-corner tilt magnitudes are unambiguous.

    Recovering (p, q) from the quadratic
    ------------------------------------
    The six fitted coefficients are not independent. Writing the quadratic
    part as a matrix,

        M = [[c20,     c11/2],
             [c11/2,   c02  ]]   must equal   (1/N^2) [[p^2, pq],
                                                       [pq,  q^2]]

    which is RANK 1 -- its determinant is identically zero, because a tilted
    plane defocuses along one direction only and is flat along the
    perpendicular one. The least-squares fit does not know that and returns a
    general M, so the three routes to p (via c20, via c11, via c10) disagree.

    Reading p and q off the diagonal as sqrt(c20) and sqrt(c02) -- the obvious
    approach, and the one used here previously -- fails outright for the most
    common real case. A sensor tilted along a single axis has q = 0, hence
    c02 = 0, which is not > 0, so the "is this well defined" guard rejected it
    and fell through to a linear fallback that returns essentially zero slope.
    Axis-aligned tilt, the easiest kind to produce by overtightening one screw,
    was reported as no tilt at all. With noise, any strongly single-axis tilt
    landed in the same trap and came out 90%+ low.

    Instead take the eigendecomposition of M and keep the dominant eigenpair.
    That is the best rank-1 approximation in the least-squares sense
    (Eckart-Young), it uses the cross term rather than discarding it, and it
    has no special case: the tilt direction simply IS the principal axis of
    the defocus quadratic. r then follows by projecting the linear terms
    (c10, c01) = 2 r (p, q) / N^2 onto that direction.

    The smaller eigenvalue is not waste -- it measures how far the defocus
    surface departs from a plane. Reported as `planarity` (lambda_min /
    lambda_max): near 0 means a genuinely flat tilted sensor, while a sizeable
    value means curvature or astigmatism is present on top of (or instead of)
    tilt, and a single tilt plane is not the whole story.

    Uncertainty
    -----------
    A tilt number with no error bar cannot be acted on: 5 um +/- 1 is a shim,
    5 um +/- 20 is noise, and they print identically. The six fitted
    coefficients have a covariance that follows from the least-squares
    solution, but (p, q, r) come out of an eigendecomposition of those
    coefficients, which is nonlinear -- so the error bars are obtained by
    drawing coefficient vectors from that covariance and pushing each one
    through the same extraction. The spread of the results is the
    uncertainty, nonlinearity and all, with no linearisation to justify.

    Samples are sign-aligned to the solution before being accumulated. The
    overall sign is physically meaningless (see above), so without alignment
    half the draws would land on the opposite branch and the spread would
    come out roughly twice the magnitude regardless of how good the fit was.

    What the error bars do and do not cover. Measured against synthetic data,
    the intervals are well calibrated for SCATTER: the truth falls inside
    +/-1 sigma about 67% of the time and inside +/-2 sigma about 95%, which
    is what those intervals claim. They do not cover BIAS, and there is a
    bias near zero: |(p, q)| is N*sqrt(lambda_max), the larger eigenvalue of
    a matrix, so it cannot come out negative and on a genuinely flat sensor
    it returns some small positive value rather than zero. The bootstrap
    cloud then sits around that value with an honest spread, and `tilt_sig`
    duly reports it as several sigma from zero -- roughly one frame in seven
    at typical noise. So tilt_sig is a reliable "this is real" signal when
    it is large, and should NOT be read as a false-alarm rate when the tilt
    is small. The defensible test for a small tilt is not its significance
    but whether the blur it implies is worth chasing, which is what the
    caller should present (see TiltDialog and diagnose_optics).

    Returns None when there aren't enough stars for a 6-parameter fit, or a
    dict with:
        p, q         : defocus slopes in um / pixel
        r            : defocus offset at origin in um
        p_se, q_se   : their standard errors
        pq_cov       : 2x2 covariance of (p, q), for propagating to any
                       point on the frame
        tilt_sig     : |(p, q)| / its standard error -- how many sigma the
                       tilt is away from zero
        fwhm_baseline: sqrt(FWHM0^2) in um (0 if the fit implies negative)
        rms_residual : RMS residual of the FWHM^2 fit (um^2)
        n_used       : star count in the final (sigma-clipped) fit
        planarity    : lambda_min / lambda_max of the defocus quadratic
        well_defined : True if the paraboloid model is physically sensible;
                       False when we fell back to a linear FWHM-vs-position
                       fit because the quadratic term was ill-behaved
    """
    n = int(x.size)
    if n < 8:                                # 6 params + a little slack
        return None

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    # Fit in centred, unit-scaled coordinates. With raw pixel coordinates the
    # x^2 column runs to ~1e7 while the constant column is 1, and the normal
    # matrix is conditioned badly enough that its inverse -- which is exactly
    # what the error bars below are built from -- cannot be trusted. The
    # coefficients are converted back to per-pixel units at the end, so
    # nothing downstream sees the change.
    cx0 = float(np.mean(x)); cy0 = float(np.mean(y))
    sx = float(max(np.ptp(x) * 0.5, 1.0))
    sy = float(max(np.ptp(y) * 0.5, 1.0))
    xs = (x - cx0) / sx
    ys = (y - cy0) / sy

    z = (np.asarray(fwhm_um, dtype=np.float64) ** 2)
    A = np.vstack([
        np.ones_like(xs), xs, ys, xs * xs, xs * ys, ys * ys,
    ]).T                                     # shape (N, 6)

    def _solve_clipped(Aa, zz):
        """Sigma-clipped linear solve. Shared by the solution and by every
        bootstrap resample, so the spread measures the estimator in use."""
        m = np.ones(zz.shape[0], dtype=bool)
        s_ = None
        for _ in range(max_iter):
            nb = int(m.sum())
            if nb < 8:
                break
            try:
                s_, *_ = np.linalg.lstsq(Aa[m], zz[m], rcond=None)
            except np.linalg.LinAlgError:
                return None, m
            rr = zz - Aa.dot(s_)
            sd_ = float(np.std(rr[m]))
            if sd_ <= 0.0:
                break
            nm = np.abs(rr) < sigma_clip * sd_
            if int(nm.sum()) == nb or int(nm.sum()) < 8:
                break
            m = nm
        return s_, m

    sol, mask = _solve_clipped(A, z)
    if sol is None:
        return None

    c00, c10, c01, c20, c11, c02 = [float(v) for v in sol]
    N = float(f_number)
    k2 = 1.0 / (N * N)

    # Best rank-1 approximation of the defocus quadratic. See the docstring:
    # the tilt direction is the principal axis, and requiring both diagonal
    # entries to be positive (as this did before) rejects every axis-aligned
    # tilt, where one of them is zero by construction.
    M = np.array([[c20, 0.5 * c11],
                  [0.5 * c11, c02]], dtype=np.float64)
    try:
        evals, evecs = np.linalg.eigh(M)        # ascending, orthonormal
    except np.linalg.LinAlgError:
        evals, evecs = np.array([0.0, 0.0]), np.eye(2)
    lam_min, lam_max = float(evals[0]), float(evals[1])
    v_max = np.asarray(evecs[:, 1], dtype=np.float64)

    # A paraboloid needs positive curvature along its principal axis. That is
    # the only condition now -- the perpendicular axis is legitimately flat
    # for a pure single-axis tilt.
    well_defined = lam_max > 0.0
    planarity = (abs(lam_min) / lam_max) if lam_max > 0.0 else float('nan')

    if not well_defined:
        # Fall back to the old plane-fit shape so the UI still has something
        # to display, but flag it so the user knows it isn't a paraboloid fit.
        Ap = np.vstack([x, y, np.ones_like(x)]).T.astype(np.float64, copy=False)
        solp, *_ = np.linalg.lstsq(Ap[mask], fwhm_um[mask], rcond=None)
        # slope of FWHM per pixel times N ~= slope of |dz| per pixel
        p_fb = float(solp[0]) * N
        q_fb = float(solp[1]) * N
        r_fb = float(solp[2]) * N
        rms_fb = float(np.sqrt(np.mean((Ap.dot(solp) - fwhm_um)[mask] ** 2)))
        # Error bars on the fallback: a plain linear fit, so the coefficient
        # covariance propagates directly with no Monte Carlo needed.
        p_fb_se = q_fb_se = float('nan')
        cov_fb = None
        try:
            dof_fb = max(1, int(mask.sum()) - 3)
            s2_fb = float(np.sum((fwhm_um - Ap.dot(solp))[mask] ** 2)) / dof_fb
            C = np.linalg.inv(Ap[mask].T @ Ap[mask]) * s2_fb * (N * N)
            cov_fb = C[:2, :2]
            p_fb_se = float(math.sqrt(max(cov_fb[0, 0], 0.0)))
            q_fb_se = float(math.sqrt(max(cov_fb[1, 1], 0.0)))
        except Exception:
            pass
        mag_fb = math.hypot(p_fb, q_fb)
        sig_fb = 0.0
        if cov_fb is not None and mag_fb > 0:
            u = np.array([p_fb, q_fb]) / mag_fb
            vd = float(u @ cov_fb @ u)
            if vd > 0:
                sig_fb = mag_fb / math.sqrt(vd)
        return {
            'p': p_fb, 'q': q_fb, 'r': max(r_fb, 0.0),
            'p_se': p_fb_se, 'q_se': q_fb_se,
            'pq_cov': (cov_fb.tolist() if cov_fb is not None else None),
            'tilt_sig': float(sig_fb),
            'fwhm_baseline': float(np.median(fwhm_um[mask])),
            'rms_residual': rms_fb,
            'n_used': int(mask.sum()),
            'planarity': planarity,
            'well_defined': False,
        }

    def _extract(cc):
        """Coefficients -> (p, q, r) in SCALED coordinates. Shared by the
        solution and by every Monte Carlo draw, so the error bars describe
        the estimator that is actually used."""
        k00, k10, k01, k20, k11, k02 = cc
        Mi = np.array([[k20, 0.5 * k11], [0.5 * k11, k02]], dtype=np.float64)
        try:
            ev, V = np.linalg.eigh(Mi)
        except np.linalg.LinAlgError:
            return None
        lmax = float(ev[1])
        if lmax <= 0.0:
            return None
        vv = np.asarray(V[:, 1], dtype=np.float64)
        m = math.sqrt(lmax) * N
        P, Q = m * float(vv[0]), m * float(vv[1])
        den = 2.0 * k2 * (P * P + Q * Q)
        R = ((k10 * P + k01 * Q) / den) if den > 1e-30 else 0.0
        if R < 0.0:
            P, Q, R = -P, -Q, -R
        return P, Q, R

    base = _extract(sol)
    if base is None:
        return None
    P, Q, R = base

    # --- bootstrap the stars ---------------------------------------------
    # Resample stars with replacement, refit, and push each resample through
    # the same extraction. The spread of the results is the uncertainty.
    #
    # The obvious cheaper route -- take the analytic covariance of the six
    # coefficients and draw from it -- was measurably optimistic, giving 1
    # sigma intervals that covered the truth 50% of the time instead of 68%
    # and letting a perfectly flat sensor claim a 3 sigma tilt in 22% of
    # frames. The reason is that the quantity being fitted is FWHM squared,
    # whose errors scale with the value itself, so they vary by a factor of
    # several across a defocused frame while a single fitted variance assumes
    # they do not. Resampling makes no such assumption: it inherits whatever
    # the real error structure is, along with the effects of the sigma clip
    # and of the nonlinear eigendecomposition, none of which a linearised
    # covariance captures.
    p_se = q_se = float('nan')
    pq_cov_scaled = None
    try:
        rng = np.random.default_rng(12345)         # reproducible error bars
        n_boot = 200
        got = []
        ref = np.array([P, Q], dtype=np.float64)
        nref = float(np.hypot(P, Q))
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            s_b, _m_b = _solve_clipped(A[idx], z[idx])
            if s_b is None:
                continue
            e = _extract(np.asarray(s_b, dtype=np.float64))
            if e is None:
                continue
            v = np.array([e[0], e[1]], dtype=np.float64)
            # Align to the solution's branch; the opposite sign is the same
            # physical surface, not a different answer. Without this, half
            # the resamples land on the far branch and the spread comes out
            # about twice the magnitude however good the fit was.
            if nref > 0 and float(v @ ref) < 0.0:
                v = -v
            got.append(v)
        if len(got) >= 30:
            Gs = np.vstack(got)
            pq_cov_scaled = np.cov(Gs.T)
            p_se = float(math.sqrt(max(pq_cov_scaled[0, 0], 0.0)))
            q_se = float(math.sqrt(max(pq_cov_scaled[1, 1], 0.0)))
    except Exception:
        pass

    # --- back to per-pixel units -----------------------------------------
    # dz = P*xs + Q*ys + R with xs = (x - cx0)/sx  =>  p = P/sx, q = Q/sy,
    # and r is the value at x = y = 0.
    p = P / sx
    q = Q / sy
    r = R - P * cx0 / sx - Q * cy0 / sy
    p_se = p_se / sx if np.isfinite(p_se) else p_se
    q_se = q_se / sy if np.isfinite(q_se) else q_se
    pq_cov = None
    if pq_cov_scaled is not None:
        S = np.array([[1.0 / sx, 0.0], [0.0, 1.0 / sy]])
        pq_cov = (S @ pq_cov_scaled @ S.T)

    # Significance of the tilt vector as a whole, along its own direction.
    mag = math.hypot(p, q)
    tilt_sig = 0.0
    if pq_cov is not None and mag > 0:
        u = np.array([p, q]) / mag
        var_dir = float(u @ pq_cov @ u)
        if var_dir > 0:
            tilt_sig = mag / math.sqrt(var_dir)

    A_baseline = c00 - k2 * R * R
    fwhm_baseline = math.sqrt(A_baseline) if A_baseline > 0.0 else 0.0
    rms = float(np.sqrt(np.mean((A.dot(sol) - z)[mask] ** 2)))

    return {
        'p': float(p),
        'q': float(q),
        'r': float(r),
        'p_se': float(p_se) if np.isfinite(p_se) else float('nan'),
        'q_se': float(q_se) if np.isfinite(q_se) else float('nan'),
        'pq_cov': (pq_cov.tolist() if pq_cov is not None else None),
        'tilt_sig': float(tilt_sig),
        'fwhm_baseline': float(fwhm_baseline),
        'rms_residual': rms,
        'n_used': int(mask.sum()),
        'planarity': float(planarity),
        'well_defined': True,
    }


def diagnose_optics(cat: dict, H: int, W: int,
                    pixel_size_um: float,
                    focal_length_mm: float,
                    aperture_mm: float,
                    center: Optional[Tuple[float, float]] = None,
                    min_sig: float = 3.0) -> dict:
    """
    Answer the two questions people actually ask: is the sensor tilted, and
    is the backfocus wrong?

    Combines two independent measurements that each supply half the answer:

      * _fit_defocus_model works on the FWHM field and gives MAGNITUDES in
        microns -- how much tilt, how much centre-to-corner curvature -- by
        fitting both at once so neither absorbs the other.
      * _aberration_decomposition works on the ellipticity field and gives
        the PATTERN -- uniform, radial/tangential, or one-sided -- which is
        what distinguishes a mechanical tilt from a spacing error from plain
        trailing, and which supplies the sign that defocus magnitude cannot.

    Returns a dict carrying both, plus `findings`: a list of
    (severity, headline, detail) tuples ordered worst-first, already phrased
    for a user. Severity is 'ok', 'minor' or 'major'.

    Deliberately NOT reported: spacing error in millimetres, or whether the
    spacer is long or short. How many microns of corner defocus a millimetre
    of spacing buys is a property of the corrector -- an MPCC is far more
    sensitive to it than a Sharpstar 0.95x reducer -- so the same measured
    defocus means different spacer thicknesses on different trains, and a
    number here would be a guess wearing a unit. What IS reported is the
    corner defocus in microns and whether the corner elongation is coherent,
    which together say how far out the spacing is and whether spacing is
    even the right thing to blame. Which direction to move is one trial
    adjustment away, and both directions are equally easy to test.
    """
    out: dict = {'findings': []}
    if cat is None or int(cat.get('n', 0)) < 20:
        out['findings'].append(
            ('ok', 'Not enough stars to diagnose',
             'At least 20 well-measured stars are needed to separate tilt '
             'from field curvature.'))
        return out

    x = np.asarray(cat['x'], float)
    y = np.asarray(cat['y'], float)
    fwhm_um = _FWHM_FACTOR * np.sqrt(np.asarray(cat['a'], float)
                                     * np.asarray(cat['b'], float)) * float(pixel_size_um)
    f_number = float(focal_length_mm) / max(float(aperture_mm), 1e-9)

    defocus = _fit_defocus_model(x, y, fwhm_um, f_number, H, W, center=center)
    shape = _aberration_decomposition(cat, H, W, center=center)
    out['defocus'] = defocus
    out['shape'] = shape

    if defocus is None:
        out['findings'].append(
            ('ok', 'Defocus model did not converge',
             'Tilt and curvature could not be separated on this frame.'))
        return out

    tilt = defocus['tilt_um']
    curv = defocus['curvature_um']
    tsig = defocus.get('tilt_sig', 0.0)
    csig = defocus.get('curvature_sig', 0.0)
    f0 = defocus['fwhm0_um']

    # Scale for "does this matter": compare the defocus blur it produces
    # against the baseline FWHM. A defocus of dz microns blurs by dz/N, so
    # the fractional FWHM penalty at the worst point is the figure of merit
    # rather than the raw micron count, which means nothing without the
    # f-ratio.
    def _penalty(dz_um):
        if f0 <= 0:
            return 0.0
        return (math.sqrt(f0 * f0 + (dz_um / f_number) ** 2) - f0) / f0

    tilt_pen = _penalty(tilt / 2.0)      # half the p-p, i.e. edge vs centre
    curv_pen = _penalty(curv)

    def _sev(pen, sig):
        if sig < min_sig:
            return 'ok'
        if pen > 0.15:
            return 'major'
        if pen > 0.05:
            return 'minor'
        return 'ok'

    # ---- tilt --------------------------------------------------------
    sev = _sev(tilt_pen, tsig)
    asym_sig = shape.get('asymmetry_sig', 0.0) if shape.get('ok') else 0.0
    corroborated = asym_sig >= min_sig
    if sev == 'ok':
        out['findings'].append((
            'ok', 'Sensor tilt: not significant',
            f'Peak-to-peak tilt defocus {tilt:.1f} µm '
            f'({tsig:.1f}σ), worth about {100*tilt_pen:.0f}% on FWHM at the '
            f'edge. Nothing to chase.'))
    else:
        out['findings'].append((
            sev, f'Sensor tilt: {tilt:.1f} µm peak-to-peak',
            f'Defocus runs {tilt:.1f} ± {defocus["tilt_se"]:.1f} µm across '
            f'the frame ({tsig:.1f}σ), steepest towards '
            f'{defocus["tilt_angle_deg"]:.0f}°. That costs roughly '
            f'{100*tilt_pen:.0f}% on FWHM at the worst edge. '
            + ('The ellipticity field is one-sided in the same sense, which '
               'corroborates it. ' if corroborated else
               'The ellipticity field does not show a matching one-sided '
               'pattern, so treat this as provisional. ')
            + f'A shim of about {tilt/2.0:.0f} µm on the high side is the '
              f'scale of correction implied.'))

    # ---- field curvature / backfocus ---------------------------------
    sev = _sev(curv_pen, csig)
    sense = shape.get('radial_sense') if shape.get('ok') else None
    rsig = shape.get('radial_sig', 0.0) if shape.get('ok') else 0.0
    if sev == 'ok':
        out['findings'].append((
            'ok', 'Field curvature / backfocus: not significant',
            f'Centre-to-corner defocus {curv:.1f} µm ({csig:.1f}σ). '
            f'Spacing looks right.'))
    else:
        detail = (f'Corners sit {curv:.1f} ± {defocus["curvature_se"]:.1f} µm '
                  f'from the centre\'s focus ({csig:.1f}σ), about '
                  f'{100*curv_pen:.0f}% on FWHM in the corners. ')
        if rsig >= min_sig and sense:
            detail += (f'Corner stars are coherently elongated {sense}ly '
                       f'({rsig:.1f}σ), which points at corrector spacing '
                       f'rather than plain field curvature. ')
        else:
            detail += ('Corner stars are not coherently elongated, so this '
                       'reads as pure field curvature — the corners are '
                       'simply out of focus. ')
        detail += ('How much spacer that is depends on your corrector, so '
                   'change it a millimetre either way and re-measure — the '
                   'number above is what should come down.')
        out['findings'].append((sev, f'Backfocus / field curvature: {curv:.1f} µm', detail))

    # ---- trailing ------------------------------------------------------
    if shape.get('ok'):
        ue = shape['uniform_e']
        usig = shape['uniform_sig']
        if usig >= min_sig and ue > 0.05:
            sev = 'major' if ue > 0.15 else 'minor'
            out['findings'].append((
                sev, f'Uniform elongation: {ue:.3f} at {shape["uniform_angle_deg"]:.0f}°',
                f'Every star is elongated the same way ({usig:.1f}σ), which '
                f'is not an optical fault — it is tracking, guiding, flexure '
                f'or wind. No amount of shimming or spacing will change it.'))

        # ---- centring sanity ------------------------------------------
        xsig = shape.get('cross_sig', 0.0)
        if xsig >= min_sig and xsig > 1.5 * max(rsig, 1e-9):
            out['findings'].append((
                'minor', 'Aberration pattern is not centred on the frame',
                f'The 45° (cross) component is stronger than the radial one '
                f'({xsig:.1f}σ vs {rsig:.1f}σ). The optical axis is probably '
                f'not at the frame centre — a decentred corrector, a cropped '
                f'or offset sensor — so the curvature number above is '
                f'measured about the wrong point and is an underestimate.'))

    if not any(s in ('minor', 'major') for s, _, _ in out['findings']):
        out['findings'].insert(0, (
            'ok', 'Optics look clean',
            f'Baseline FWHM {f0:.1f} µm with no significant tilt, curvature '
            f'or trailing across {defocus["n_used"]} stars.'))

    sev_rank = {'major': 0, 'minor': 1, 'ok': 2}
    out['findings'].sort(key=lambda t: sev_rank.get(t[0], 3))
    return out


def _fit_defocus_model(x: np.ndarray, y: np.ndarray, fwhm_um: np.ndarray,
                       f_number: float, H: int, W: int,
                       center: Optional[Tuple[float, float]] = None,
                       sigma_clip: float = 3.0) -> Optional[dict]:
    """
    Joint fit of sensor tilt AND field curvature to the FWHM field.

        dz(x, y) = p*xn + q*yn + r + kappa*u
        FWHM     = sqrt(FWHM0^2 + (dz / N)^2)

    with xn, yn the position in half-frames from the centre (so p and q are
    microns of defocus from centre to edge) and u = (R/R_corner)^2, so kappa
    is microns of defocus between the centre and the corner.

    Fitting them together is the point. A tilted sensor and a mis-spaced
    corrector both make the corners worse, and a model carrying only one of
    them absorbs the other: fit tilt alone to a frame with curvature and the
    tilt comes out too large, which then gets reported as a mechanical
    problem the user cannot find. Separating them is the difference between
    "shim this corner" and "change your spacer".

    Parameters recovered:
        tilt_um       peak-to-peak defocus across the frame from the tilt
                      plane alone, in microns -- the number that corresponds
                      to a physical shim
        tilt_angle    direction of steepest defocus increase, degrees
        curvature_um  centre-to-corner defocus from the symmetric term, in
                      microns -- the backfocus / field-flatness number
        fwhm0_um      the baseline FWHM at perfect focus (seeing + optics)
        r_um          overall focus offset at the field centre

    Each comes with a standard error from the fit covariance. Signs are
    ambiguous from a single image for the same reason as always (a surface
    defocused one way is indistinguishable from the other), so magnitudes
    are what is reported; the radial/tangential sense of the ellipticity,
    measured separately in _aberration_decomposition, is what tells the user
    which way to move.

    Returns None if the fit cannot be performed.
    """
    n = int(x.size)
    if n < 20:
        return None
    try:
        from scipy.optimize import least_squares
    except Exception:
        return None

    cx, cy = (W / 2.0, H / 2.0) if center is None else (float(center[0]),
                                                        float(center[1]))
    xn = (x - cx) / max(W / 2.0, 1e-9)
    yn = (y - cy) / max(H / 2.0, 1e-9)
    R = np.hypot(x - cx, y - cy)
    Rmax = math.hypot(max(cx, W - cx), max(cy, H - cy))
    u = (R / max(Rmax, 1e-9)) ** 2

    N = float(f_number)
    f = np.asarray(fwhm_um, dtype=np.float64)

    def model(pars):
        f0, p, q, r, kap = pars
        dz = p * xn + q * yn + r + kap * u
        return np.sqrt(np.maximum(f0 * f0 + (dz / N) ** 2, 1e-12))

    # --- seeding -----------------------------------------------------------
    # Multi-start, and not optional. Tilt and curvature sit in separate local
    # minima of this cost surface, and a single seed lands in whichever one it
    # started nearest: seeding kappa at full scale and p, q near zero
    # recovered a pure 60 um tilt as 0.7 um of tilt plus 21 um of invented
    # curvature, because the symmetric explanation was the one in reach. The
    # cure is to start from each hypothesis in turn -- all tilt, all
    # curvature, all focus offset, and mixtures -- and keep whichever fits
    # best. Six small Levenberg-Marquardt solves on five parameters cost
    # nothing next to the star detection that preceded them.
    f0_0 = max(float(np.percentile(f, 10)), 1e-3)
    hi = float(np.percentile(f, 90))
    K = max(N * math.sqrt(max(hi * hi - f0_0 * f0_0, 0.0)), 1.0)
    seeds = [
        [f0_0, 0.0, 0.0, 0.0, K],            # all curvature
        [f0_0, K, 0.0, 0.0, 0.0],            # all tilt, +x
        [f0_0, 0.0, K, 0.0, 0.0],            # all tilt, +y
        [f0_0, K / 1.414, K / 1.414, 0.0, 0.0],   # tilt on the diagonal
        [f0_0, K / 1.414, -K / 1.414, 0.0, 0.0],  # the other diagonal
        [f0_0, K / 2, K / 2, 0.0, K / 2],    # mixed
        [f0_0, 0.0, 0.0, K, 0.0],            # uniform focus offset
    ]

    def _solve(seed, m):
        def resid(pars):
            return (model(pars) - f)[m]
        try:
            return least_squares(resid, seed, method='lm', max_nfev=4000)
        except Exception:
            try:
                return least_squares(resid, seed, max_nfev=4000)
            except Exception:
                return None

    mask = np.ones(n, dtype=bool)
    best = None
    for s in seeds:
        r_ = _solve(s, mask)
        if r_ is None:
            continue
        if best is None or float(r_.cost) < float(best.cost):
            best = r_
    if best is None:
        return None

    # Refine with sigma-clipping from the winning start.
    res = best
    seed = list(best.x)
    for _ in range(3):
        if int(mask.sum()) < 12:
            break
        rr = model(res.x) - f
        sd = float(np.std(rr[mask]))
        if not np.isfinite(sd) or sd <= 0:
            break
        nm = np.abs(rr) < sigma_clip * sd
        if int(nm.sum()) == int(mask.sum()) or int(nm.sum()) < 12:
            break
        mask = nm
        r_ = _solve(seed, mask)
        if r_ is None:
            break
        res = r_
        seed = list(res.x)

    f0, p, q, r, kap = [float(v) for v in res.x]
    n_used = int(mask.sum())

    # Covariance from the Jacobian at the solution, scaled by the residual
    # variance. Standard linearised error estimate -- adequate here, where
    # the model is mildly nonlinear and the residuals are many.
    se = [float('nan')] * 5
    pq_cov = None          # _SASPRO_ONE_TILT: the (p, q) block, for corner error bars
    try:
        J = res.jac
        dof = max(1, n_used - 5)
        s2 = float(np.sum(res.fun ** 2)) / dof
        cov = np.linalg.inv(J.T @ J) * s2
        se = [float(math.sqrt(max(cov[i, i], 0.0))) for i in range(5)]
        # _SASPRO_ONE_TILT: params are [f0, p, q, r, kappa], so rows/cols 1:3 are the
        # tilt pair. TiltDialog needs this to give each corner its own error
        # bar; it was being computed and thrown away.
        _c = np.asarray(cov[1:3, 1:3], dtype=float)
        if _c.shape == (2, 2) and np.all(np.isfinite(_c)):
            pq_cov = _c.tolist()
    except Exception:
        pass

    # Peak-to-peak of the tilt plane over the frame: the plane runs from
    # -|p|-|q| to +|p|+|q| in half-frame units, so p-p is twice their sum in
    # quadrature along the steepest direction.
    # _SASPRO_ONE_TILT: corner to opposite corner, which is what the comment above
    # says and what TiltDialog's corner table has always computed. A plane
    # over a RECTANGLE reaches its extremes at the corners, dz = +/-p +/-q,
    # so max - min = 2(|p| + |q|). The previous 2*hypot(p, q) is the span
    # over the inscribed CIRCLE -- the steepest direction, but only out to
    # the short radius -- and understated the tilt by up to sqrt(2) (1.38x
    # on the frame this was found on: 21.3 um reported, 29.5 um corner to
    # corner). That gap was most of the disagreement between the two dialogs.
    tilt_pp = 2.0 * (abs(p) + abs(q))
    # Propagated for THAT quantity: d/dp = 2*sign(p), d/dq = 2*sign(q), so
    # the covariance term carries its own sign rather than being dropped.
    if all(np.isfinite(se[1:3])):
        _vp, _vq = se[1] ** 2, se[2] ** 2
        _cpq = float(pq_cov[0][1]) if pq_cov is not None else 0.0
        _sgn = (1.0 if p >= 0 else -1.0) * (1.0 if q >= 0 else -1.0)
        tilt_se = 2.0 * math.sqrt(max(_vp + _vq + 2.0 * _sgn * _cpq, 0.0))
    else:
        tilt_se = float('nan')

    rms_resid = float(np.sqrt(np.mean((model(res.x) - f)[mask] ** 2)))
    return {
        'fwhm0_um': abs(f0), 'fwhm0_se': se[0],
        # _SASPRO_ONE_TILT: the tilt plane itself, in the model's half-frame units
        # (dz = p*xn + q*yn + ...), plus the half-frame scale so a caller can
        # convert to microns per pixel without re-deriving it.
        'p': float(p), 'q': float(q), 'pq_cov': pq_cov,
        'half_w': float(max(W / 2.0, 1e-9)), 'half_h': float(max(H / 2.0, 1e-9)),
        'tilt_um': tilt_pp, 'tilt_se': tilt_se,
        'tilt_angle_deg': math.degrees(math.atan2(q, p)) % 360.0,
        'curvature_um': abs(kap), 'curvature_se': se[4],
        'r_um': r, 'r_se': se[3],
        'n_used': n_used,
        'rms_resid_um': rms_resid,
        'f_number': N,
        'tilt_sig': (tilt_pp / tilt_se) if (tilt_se and np.isfinite(tilt_se)
                                            and tilt_se > 0) else 0.0,
        'curvature_sig': (abs(kap) / se[4]) if (np.isfinite(se[4]) and se[4] > 0)
                         else 0.0,
    }


def analyze_focal_plane(img: np.ndarray,
                        pixel_scale: float,
                        thresh_sigma: float = 5.0,
                        deg: int = 3,
                        coarse: int = 128,
                        min_stars: int = 12,
                        target_stars: int = 1000,
                        max_sigma_theta_deg: float = _MAX_SIGMA_THETA_DEG,
                        min_elong_snr: float = _MIN_ELONG_SNR,
                        min_abs_elong: float = _MIN_ABS_ELONG,
                        min_orient_stars: int = _MIN_ORIENT_STARS,
                        sat_level: Optional[float] = None,
                        focal_length_mm: Optional[float] = None,
                        aperture_mm: Optional[float] = None) -> Optional[dict]:
    """
    Single-pass focal-plane analysis. Runs a descending-threshold SEP
    detection ONCE (background estimate + repeated extract) and fits three
    smooth surfaces from the shared catalogue:

      - FWHM (um)      -- proper Gaussian conversion: 2.3548 * sqrt(a*b) * ps
      - Ellipticity    -- 1 - b/a (matches PixInsight / SExtractor convention)
      - Orientation    -- weighted fit of the ellipticity vector (e1, e2),
                          over only those stars with a believable position
                          angle; see _fit_orientation_field

    Detection uses the waterfall (100, 50, 25, 12, 6, 3 sigma) and stops as
    soon as `target_stars` well-formed stars have survived quality filtering.
    `thresh_sigma` is retained for API compatibility but is only used as the
    floor of the ladder when it exceeds 3.

    The three surfaces do NOT share a star list, and that is deliberate.
    FWHM and ellipticity use every adequately sampled star, round ones
    included -- an in-focus star is round, and dropping round stars would
    bias both maps. Orientation additionally requires a measurable
    elongation, because a round star has no axis to report. `orientation_ok`
    in the result says whether enough stars cleared that bar; when False
    there is no orientation map, which is the honest outcome rather than a
    smooth render of noise.

    Each surface is returned already normalised to [0, 1] for display along
    with its raw (min, max) range in physical units.
    """
    gray = _to_gray32(img)
    H, W = gray.shape

    # Build the descending ladder; ensure thresh_sigma acts as a floor.
    ladder = tuple(s for s in (100.0, 50.0, 25.0, 12.0, 6.0, 3.0)
                   if s >= float(thresh_sigma))
    if not ladder:
        ladder = (float(thresh_sigma),)
    cat = _detect_stars_waterfall(gray, sigma_ladder=ladder,
                                  target_count=target_stars,
                                  sat_level=sat_level)
    if cat is None or cat['n'] < min_stars:
        return None

    x, y = cat['x'], cat['y']
    a, b = cat['a'], cat['b']

    # Region the stars actually constrain. Everything outside it is the
    # polynomial extrapolating, and must not be allowed to set the display
    # scale -- see _normalize_for_display.
    cover = _coverage_mask(x, y, H, W)

    # ----- FWHM (geometric-mean axis FWHM, correct Gaussian conversion) -----
    fwhm_um = _FWHM_FACTOR * np.sqrt(a * b) * float(pixel_scale)
    fwhm_surf, _ = _fit_and_render_surface(x, y, fwhm_um, H, W,
                                           deg=deg, coarse=coarse)

    # ----- Ellipticity (canonical: 1 - b/a in [0, 1)) -----------------------
    ell = 1.0 - (b / a)
    # _SASPRO_STRENGTH_WHEEL: the same kernel regression the orientation field uses. This
    # surface is now also the orientation map's brightness channel, so it must
    # not reintroduce the polynomial stiffness that was just removed.
    ell_surf, _unused_e2, _ell_support, _ = _kernel_axis_field(
        x, y, ell, np.zeros_like(ell), np.ones_like(ell), H, W)

    # ----- Orientation --------------------------------------------------
    # Only the stars whose elongation is real and whose angle is pinned down
    # take part; see _fit_orientation_field. This cut is deliberately NOT
    # applied to the FWHM or ellipticity surfaces above: a round star is a
    # perfectly good FWHM measurement (in-focus stars are round), and
    # excluding round stars from an ellipticity map would bias it upward
    # everywhere. Roundness only disqualifies a star from having a DIRECTION.
    ori = _fit_orientation_field(cat, H, W, deg=deg, coarse=coarse,
                                 max_sigma_theta_deg=max_sigma_theta_deg,
                                 min_elong_snr=min_elong_snr,
                                 min_abs_elong=min_abs_elong,
                                 min_stars=min_orient_stars)

    # Normalise for display. The returned range is the one the image was
    # scaled with, so the colourbar beside it is correct by construction.
    fwhm_norm, fwhm_range = _normalize_for_display(fwhm_surf, cover)
    ell_norm, ell_range = _normalize_for_display(ell_surf, cover)

    # _SASPRO_MAP_CONFIDENCE -------------------------------------------------------
    # The sampling term has to be in pixels; fwhm_surf is in microns.
    _ps = float(pixel_scale) if float(pixel_scale) > 0 else 1.0
    # _SASPRO_AXIS_TICKS: mean nearest-neighbour star spacing for n stars over the
    # frame. The surface is still constrained about this far past the last
    # star, so it is the right distance to carry full confidence outward.
    _cov_margin = 0.5 * math.sqrt((float(H) * float(W))
                                  / max(1.0, float(cat['n'])))
    fwhm_px_surf = (fwhm_surf / _ps).astype(np.float32, copy=False)
    # Ellipticity is unreliable where the stars are undersampled, and
    # meaningless outside the star field. Roundness is NOT a reason to
    # dim it -- a round star is a true ellipticity of zero.
    ell_conf = _confidence_surface(H, W, fwhm_px_surf=fwhm_px_surf,
                                   coverage=cover, margin_px=_cov_margin)
    # FWHM only needs the coverage term: a round, well-sampled star is
    # a perfectly good FWHM measurement.
    fwhm_conf = _confidence_surface(H, W, coverage=cover,
                                    margin_px=_cov_margin)
    # ----------------------------------------------------------------

    out = {
        'n_stars':        cat['n'],
        'fwhm_norm':      fwhm_norm,
        'fwhm_range_um':  fwhm_range,
        'ell_norm':       ell_norm,
        'ell_range':      ell_range,
        'ell_conf':       ell_conf,     # _SASPRO_MAP_CONFIDENCE
        # _SASPRO_HSL_LIGHTNESS: lightness for the orientation map -- the ellipticity map
        # itself, scaled only by the sampling and coverage terms (1.0 on a
        # normal frame, so the lightness IS the ellipticity map).
        'orient_lightness': (np.clip(ell_norm, 0.0, 1.0)
                             * _confidence_surface(
                                 H, W, fwhm_px_surf=fwhm_px_surf,
                                 coverage=cover, margin_px=_cov_margin)
                             ).astype(np.float32),
        'fwhm_conf':      fwhm_conf,    # _SASPRO_MAP_CONFIDENCE
        'fwhm_px_surf':   fwhm_px_surf,  # _SASPRO_MAP_CONFIDENCE
        'coverage':       cover,
        'coverage_frac':  float(np.count_nonzero(cover)) / float(max(1, cover.size)),
        'attrition':      cat.get('attrition', {}),
        'detect_sigma':   cat.get('sigma'),
        # Raw per-star metrics kept for potential overlay drawing.
        'stars': cat,
    }

    # Name the cause, not just the symptom -- needs the optics, so it is
    # skipped when the caller did not supply them.
    if focal_length_mm and aperture_mm:
        try:
            out['diagnosis'] = diagnose_optics(
                cat, H, W, pixel_scale, focal_length_mm, aperture_mm)
        except Exception as e:
            print(f"[focal plane] diagnosis skipped: {type(e).__name__}: {e}")

    if ori.get('ok'):
        out.update({
            'orientation_ok':   True,
            'theta_norm':       ori['theta_norm'],
            'theta_surf':       ori['theta_surf'],   # _SASPRO_AXIS_TICKS: ticks need radians
            'zones':            ori.get('zones'),    # _SASPRO_AXIS_TICKS
            'axis_bins':        ori.get('bins'),     # _SASPRO_LOCAL_AXES
            'poly_degree':      ori.get('poly_degree'),   # _SASPRO_LOCAL_AXES
            'theta_range_rad':  ori['theta_range'],
            'emag_surf':        ori['emag_surf'],
            'emag_range':       ori['emag_range'],
            # _SASPRO_MAP_CONFIDENCE: orientation is the map that most needs this --
            # hue is vivid at every angle, including the angle of a
            # zero-length ellipticity vector.
            'orient_conf':      _confidence_surface(
                H, W, emag_surf=ori['emag_surf'],
                fwhm_px_surf=fwhm_px_surf, coverage=cover,
                # _SASPRO_ROUNDNESS_ONLY: no strength channel -- see above.
                margin_px=_cov_margin),
            'n_orient_stars':   ori['n_used'],
            'orient_sigma_deg': ori['median_sigma_theta_deg'],
        })
    else:
        # No orientation map rather than a meaningless one. The caller shows
        # the FWHM and ellipticity maps regardless -- those only needed the
        # sampling cut, which these stars passed.
        out.update({
            'orientation_ok':   False,
            'n_orient_stars':   ori.get('n_used', 0),
            'orient_sigma_deg': ori.get('median_sigma_theta_deg', float('nan')),
        })
    return out


# =============================================================================
def field_curvature_analysis(
    img: np.ndarray,
    grid: int,
    panel: int,
    pixel_scale: float,
    snr_thresh: float = 5.0
) -> Tuple[np.ndarray, List[Tuple[int,int,float,float]]]:
    """
    1) Estimate background + detect stars via SEP.
    2) Compute per‐star FWHM, eccentricity, and orientation theta.
    3) Bin the FWHM into a grid×grid mosaic (median per cell) → FWHM_um heatmap.
    4) Normalize that heatmap to [0..1] for display.
    5) Build an overlay list of (x_pix,y_pix,angle_rad,elongation_um) for each
       star WHOSE ANGLE IS MEANINGFUL.

    The heatmap and the arrows are filtered differently, on purpose. The
    heatmap keeps every adequately sampled star, round ones included. The
    arrows additionally require a measurable elongation: an arrow drawn on a
    round star points in a direction set by noise, and a field of those
    reads as a coherent pattern to the eye even when it is nothing.
    """
    H, W = img.shape[:2]
    # grayscale float32
    if img.ndim == 3 and img.shape[2] == 3:
        gray = (0.2126*img[...,0] + 0.7152*img[...,1] + 0.0722*img[...,2]).astype(np.float32)
    else:
        gray = img.astype(np.float32)

    # background / stats
    mean, med, std = sigma_clipped_stats(gray, sigma=3.0)
    data = gray - med

    # detect
    objs = sep.extract(data, thresh=snr_thresh, err=std)
    if objs is None or len(objs)==0:
        # empty mosaic + no overlays
        blank = np.zeros((H,W), dtype=float)
        return blank, []

    # Sampling / shape quality cut, same as every other analysis here.
    good, _att = _quality_filter(objs, float(std), min_area=5, sat_level=None,
                                 min_fwhm_px=_MIN_FWHM_PX,
                                 min_semiminor_px=_MIN_SEMIMINOR_PX)
    if not np.any(good):
        return np.zeros((H, W), dtype=float), []
    cat = _pack_catalogue(objs, good, float(std))

    x, y = cat['x'], cat['y']
    a, b, theta = cat['a'], cat['b'], cat['theta']

    # FWHM via the proper Gaussian second-moment conversion, matching the
    # rest of the module (the old 2*a underestimated it and ignored b).
    fwhm_um = _FWHM_FACTOR * np.sqrt(a * b) * pixel_scale

    # eccentricity → elongation factor e = a/b - 1
    e = np.clip(a / np.where(b>0, b, 1.0) - 1.0, 0.0, None)
    elongation_um = e * pixel_scale

    # Which of these stars may carry an arrow.
    _stats = _shape_stats(cat, rms=float(std))
    arrow_ok = _orientation_mask(_stats)

    # --- build mosaic of median‐FWHM in each grid cell ---
    cell_w, cell_h = W/grid, H/grid
    fmap = np.zeros((H,W), dtype=float)
    heat = np.full((grid, grid), np.nan, dtype=float)
    for j in range(grid):
        for i in range(grid):
            mask = (
                (x>= i*cell_w) & (x< (i+1)*cell_w) &
                (y>= j*cell_h) & (y< (j+1)*cell_h)
            )
            if np.any(mask):
                mval = np.median(fwhm_um[mask])
            else:
                mval = np.nan
            heat[j,i] = mval
            # fill that block
            y0, y1 = int(j*cell_h), int((j+1)*cell_h)
            x0, x1 = int(i*cell_w), int((i+1)*cell_w)
            fmap[y0:y1, x0:x1] = mval if not np.isnan(mval) else 0.0

    # replace empty with global median
    med_heat = np.nanmedian(heat)
    fmap = np.where(fmap==0, med_heat, fmap)

    # normalize to [0..1]
    mn, mx = fmap.min(), fmap.max()
    if mx>mn:
        norm = (fmap - mn) / (mx - mn)
    else:
        norm = np.zeros_like(fmap)

    # --- build elongation‐arrow overlays (elongated stars only) ---
    overlays: List[Tuple[int,int,float,float]] = []
    for xi, yi, ang, el, ok in zip(x, y, theta, elongation_um, arrow_ok):
        if ok:
            overlays.append((int(xi), int(yi), float(ang), float(el)))

    return norm, overlays

def tilt_analysis(
    img: np.ndarray,
    pixel_size_um: float,
    focal_length_mm: float,
    aperture_mm: float,
    sigma_clip: float = 2.5,
    thresh_sigma: float = 5.0,
    target_stars: int = 1000,
) -> Tuple[np.ndarray, Tuple[float,float,float], Tuple[int,int], Optional[dict]]:
    """
    Sensor tilt measurement via a physical FWHM^2 paraboloid fit.

    Old pipeline (broken for weak/moderate tilt): fit a plane to blur diameter
    treated as signed defocus. Because blur ~= |dz|/N, one side of a tilted
    sensor has positive dz and the other negative but both show positive
    blur -- so the fitted plane collapses toward zero whenever the focus
    ridge crosses the sensor. The old pipeline only recovered tilt correctly
    for severe cases where the whole field was defocused with the same sign.

    New pipeline:
      1) Grayscale + descending-threshold SEP detection (shared with focal
         plane; stops once target_stars well-formed detections are in hand,
         so rich fields don't waste time on 30k faint noisy stars).
      2) Compute FWHM(x,y) properly from the ellipse: 2.3548 * sqrt(a*b) * ps.
      3) Fit FWHM^2 = FWHM0^2 + (p*x + q*y + r)^2 / N^2 as a general 2D
         quadratic, extract signed (p, q, r) plus baseline FWHM0.
      4) Rasterise the signed defocus plane over the frame for display.

    Returns
    -------
    plane_norm : (H, W) float32 in [0, 1]
        Signed defocus plane normalised for the display colour ramp.
    (p, q, r)  : plane coefficients in um and um / pixel; the same tuple
                 shape the old code returned so TiltDialog corner-delta
                 arithmetic is unchanged.
    (H, W)     : image shape.
    fit_info   : dict with diagnostics (fwhm_baseline_um, rms_residual,
                 n_used, well_defined, f_number) -- or None if the fit failed.
                 New in this version; TiltDialog uses it if provided.
    """
    gray = _to_gray32(img)
    H, W = gray.shape

    ladder = tuple(s for s in (100.0, 50.0, 25.0, 12.0, 6.0, 3.0)
                   if s >= float(thresh_sigma))
    if not ladder:
        ladder = (float(thresh_sigma),)
    cat = _detect_stars_waterfall(gray, sigma_ladder=ladder, target_count=target_stars)
    if cat is None or cat['n'] < 15:      # paraboloid needs 6+; 15 for stability
        return np.zeros((H, W), dtype=np.float32), (0.0, 0.0, 0.0), (H, W), None

    x = cat['x']; y = cat['y']
    a_pix = cat['a']; b_pix = cat['b']

    # FWHM (um) via proper Gaussian second-moment conversion
    fwhm_um = _FWHM_FACTOR * np.sqrt(a_pix * b_pix) * float(pixel_size_um)

    f_number = float(focal_length_mm) / float(aperture_mm)
    # _SASPRO_ONE_TILT: the SAME estimator the Optical Diagnosis uses, so the two
    # dialogs cannot report different tilts for one frame. The paraboloid it
    # replaces had no field-curvature term, so curvature leaked into its tilt
    # -- which is what TiltDialog's own "surface is not flat" warning was
    # describing. Expect smaller, better-determined numbers than before.
    # NB: deliberately NOT passing this function's `sigma_clip` (2.5). The
    # Optical Diagnosis uses the model's own default, and a different clip
    # selects a different subset of stars -- 1009 against 1059 on a real
    # frame -- which is enough on its own to make the two dialogs disagree.
    # One estimator means one clip too.
    fit = _fit_defocus_model(x, y, fwhm_um, f_number, H, W)
    if fit is None:
        return np.zeros((H, W), dtype=np.float32), (0.0, 0.0, 0.0), (H, W), None

    # The model works in half-frame units; the corner table wants microns per
    # PIXEL, delta = p*dx + q*dy measured from the frame centre.
    _hw = float(fit.get('half_w') or max(W / 2.0, 1e-9))
    _hh = float(fit.get('half_h') or max(H / 2.0, 1e-9))
    p = float(fit.get('p', 0.0)) / _hw
    q = float(fit.get('q', 0.0)) / _hh
    r = float(fit.get('r_um', 0.0))

    # Transform the covariance with the same scaling, so each corner keeps an
    # exact error bar rather than a shared global one.
    _pq = fit.get('pq_cov')
    if _pq is not None:
        try:
            _S = np.array([[1.0 / _hw, 0.0], [0.0, 1.0 / _hh]], dtype=float)
            _pq = (_S @ np.asarray(_pq, dtype=float) @ _S).tolist()
        except Exception:
            _pq = None

    fit = dict(fit)
    fit.update({
        'p': p, 'q': q, 'r': r, 'pq_cov': _pq,
        'fwhm_baseline': fit.get('fwhm0_um'),
        # now a residual in microns, not microns^2 -- the label says so too
        'rms_residual': fit.get('rms_resid_um'),
        'well_defined': True,
        # Retired: it flagged the old model having nowhere to put curvature.
        # Curvature is a fitted term now, so the corner numbers are the tilt
        # part by construction, and the dialog shows the curvature instead.
        'planarity': None,
    })

    # Rasterise the (linear) signed defocus plane. Uses shared coarse-grid
    # upsampler for consistency, though a plane could be evaluated directly.
    plane_norm = _eval_poly_upsampled(
        np.array([p, q, r]),
        [(1, 0), (0, 1), (0, 0)],
        H, W, coarse=64,
    )
    pmin, pmax = float(plane_norm.min()), float(plane_norm.max())
    if pmax > pmin:
        plane_norm = (plane_norm - pmin) / (pmax - pmin)
    else:
        plane_norm = np.zeros_like(plane_norm)

    fit['f_number'] = f_number
    return plane_norm, (p, q, r), (H, W), fit

def focal_plane_curvature_overlay(img: np.ndarray, grid: int, panel: int):
    """
    Compute the best-fit sphere radius through each local panel,
    return a list of QPainter-friendly overlay primitives,
    e.g. [(x,y,radius,quality), …].
    """
    overlays = []
    h, w = img.shape[:2]
    xs = np.linspace(0, w-panel, grid, dtype=int)
    ys = np.linspace(0, h-panel, grid, dtype=int)
    for y in ys:
        for x in xs:
            patch = img[y:y+panel, x:x+panel]
            # Fit a circle to the intensity → radius
            radius = fit_circle_radius(patch)
            overlays.append((x, y, panel, radius))
    return overlays



def build_mosaic_numpy(
    arr: np.ndarray,
    grid: int,
    panel: int,
    sep: int = 4,
    background: float = 0.0
) -> np.ndarray:
    """
    Tile `arr` into a grid×grid mosaic of size `panel` each, separated by `sep` pixels.
    If arr is 2D, result is 2D; if 3D (H×W×3), result is 3D.
    """
    h, w = arr.shape[:2]
    out_h = grid * panel + (grid - 1) * sep
    out_w = grid * panel + (grid - 1) * sep
    if arr.ndim == 2:
        mosaic = np.full((out_h, out_w), background, dtype=arr.dtype)
    else:
        c = arr.shape[2]
        mosaic = np.full((out_h, out_w, c), background, dtype=arr.dtype)

    # evenly spaced top-left corners
    xs = [int((w - panel) * i / (grid - 1)) for i in range(grid)]
    ys = [int((h - panel) * j / (grid - 1)) for j in range(grid)]

    for row, y in enumerate(ys):
        for col, x in enumerate(xs):
            patch = arr[y : y + panel, x : x + panel]
            dy = row * (panel + sep)
            dx = col * (panel + sep)
            mosaic[dy:dy + panel, dx:dx + panel, ...] = patch

    return mosaic




def fit_circle_radius(patch: np.ndarray) -> float:
    """
    Very rough radius estimate by thresholding + edge points + circle fit.
    Returns radius in pixels (caller scales to physical units).
    """
    # 1) threshold at ~50% max:
    thr = patch.max() * 0.5
    mask = patch > thr
    ys, xs = np.nonzero(mask)
    if len(xs) < 5:
        return 0.0

    # 2) algebraic circle fit (Taubin)
    x = xs.astype(float)
    y = ys.astype(float)
    x_m = x.mean();  y_m = y.mean()
    u = x - x_m;   v = y - y_m
    Suu = (u*u).sum();  Suv = (u*v).sum();  Svv = (v*v).sum()
    Suuu = (u*u*u).sum();  Svvv = (v*v*v).sum()
    Suvv = (u*v*v).sum();  Svuu = (v*u*u).sum()
    # Solved system:
    A = np.array([[Suu, Suv], [Suv, Svv]])
    B = np.array([(Suuu + Suvv)/2.0, (Svvv + Svuu)/2.0])
    try:
        uc, vc = np.linalg.solve(A, B)
    except np.linalg.LinAlgError:
        return 0.0
    radius = math.hypot(uc, vc)
    return radius

def focal_plane_curvature_overlay(
    img: np.ndarray,
    grid: int,
    panel: int,
    pixel_size_um: Optional[float] = None
) -> List[Tuple[int,int,int,float]]:
    """
    Divide `img` into grid×grid panels, estimate per-panel best-focus radius,
    and return overlay tuples (x, y, panel, radius_um).
    If pixel_size_um is given, radius is returned in microns; else in pixels.
    """
    overlays: List[Tuple[int,int,int,float]] = []
    h, w = img.shape[:2]
    xs = [int((w - panel) * i / (grid - 1)) for i in range(grid)]
    ys = [int((h - panel) * j / (grid - 1)) for j in range(grid)]

    for y in ys:
        for x in xs:
            patch = img[y : y + panel, x : x + panel]
            r_px = fit_circle_radius(patch)
            r = (r_px * pixel_size_um) if pixel_size_um else r_px
            overlays.append((x, y, panel, r))

    return overlays

# Import centralized widgets
from setiastro.saspro.widgets.spinboxes import CustomSpinBox, CustomDoubleSpinBox


class TiltDialog(QDialog):
    def __init__(self,
                 title: str,
                 img: np.ndarray,
                 plane: Optional[Tuple[float,float,float]] = None,
                 img_shape: Optional[Tuple[int,int]]    = None,
                 pixel_size_um: float                   = 1.0,
                 overlays: Optional[List[Tuple]]        = None,
                 fit_info: Optional[dict]               = None,
                 parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.pixel_size_um = pixel_size_um

        # ----- image / overlays -------------------------------------------------
        self.view = PreviewPane()
        self.view.load_numpy(img)
        if overlays:
            self.view.set_overlay(overlays)

        # ----- corner tilt table + diagnostics ---------------------------------
        range_label = None
        table = None
        diag_label = None
        note_label = None
        curve_label = None
        insig_label = None

        if plane and img_shape:
            a, b, c = plane
            H, W = img_shape
            cx, cy = W / 2.0, H / 2.0
            corners = [
                ("Top Left",     0, 0),
                ("Top Right",    W, 0),
                ("Bottom Left",  0, H),
                ("Bottom Right", W, H),
            ]
            # Covariance of (p, q) from the fit, so each corner delta can
            # carry its own error bar. delta = p*dx + q*dy is linear in the
            # parameters, so the variance is exact given the covariance:
            #   var = dx^2 var(p) + dy^2 var(q) + 2 dx dy cov(p, q)
            # A corner near the tilt axis is better determined than one out
            # along it, and the per-corner numbers show that rather than
            # sharing one global figure.
            pq_cov = None
            if fit_info and fit_info.get("pq_cov"):
                try:
                    pq_cov = np.asarray(fit_info["pq_cov"], dtype=float)
                    if pq_cov.shape != (2, 2) or not np.all(np.isfinite(pq_cov)):
                        pq_cov = None
                except Exception:
                    pq_cov = None

            def _delta_se(dx, dy):
                if pq_cov is None:
                    return None
                v = (dx * dx * pq_cov[0, 0] + dy * dy * pq_cov[1, 1]
                     + 2.0 * dx * dy * pq_cov[0, 1])
                return math.sqrt(v) if v > 0 else 0.0

            rows = []
            corner_deltas = []
            for name, x, y in corners:
                dx, dy = (x - cx), (y - cy)
                delta = a * dx + b * dy               # relative to centre plane
                corner_deltas.append(delta)
                se = _delta_se(dx, dy)
                rows.append((name, f"{delta:.1f}" if se is None
                             else f"{delta:.1f} ± {se:.1f}"))

            min_d, max_d = min(corner_deltas), max(corner_deltas)
            span = max_d - min_d
            # Peak-to-peak runs corner to opposite corner, so its uncertainty
            # is that of a single delta doubled along the same direction.
            span_se = None
            if pq_cov is not None:
                i_max = int(np.argmax(corner_deltas))
                sx_, sy_ = (corners[i_max][1] - cx), (corners[i_max][2] - cy)
                s1 = _delta_se(sx_, sy_)
                span_se = (2.0 * s1) if s1 is not None else None
            tsig = (fit_info or {}).get("tilt_sig")
            range_label = QLabel(self.tr(
                "Tilt span (corner Δ vs centre): {0:.1f} µm … {1:.1f} µm   "
                "|  peak-to-peak {2}"
            ).format(min_d, max_d,
                     f"{span:.1f} µm" if span_se is None
                     else f"{span:.1f} ± {span_se:.1f} µm"
                     + (f"  ({tsig:.1f}σ)" if tsig else "")))

            table = QTableWidget(len(rows), 2, self)
            table.setHorizontalHeaderLabels([self.tr("Corner"), self.tr("Δ µm")])
            table.verticalHeader().setVisible(False)
            for i, (name, val) in enumerate(rows):
                table.setItem(i, 0, QTableWidgetItem(name))
                table.setItem(i, 1, QTableWidgetItem(val))
            table.resizeColumnsToContents()

            # Two independent reasons not to chase a tilt, and the second
            # matters more than the first.
            #
            #   (a) it is not distinguishable from zero;
            #   (b) it is real but too small to cost anything.
            #
            # (b) is the robust one. The magnitude estimator is biased upward
            # near zero -- it is the square root of a largest eigenvalue, so
            # it cannot return zero -- which means a flat sensor still clears
            # a significance threshold now and then. Whether the implied blur
            # is worth a screwdriver does not depend on that bias at all, so
            # it is the test shown whenever the optics are known.
            f_num = (fit_info or {}).get("f_number")
            f_base = (fit_info or {}).get("fwhm_baseline")
            penalty = None
            if f_num and f_base and f_base > 0:
                worst = span / 2.0          # edge vs centre
                penalty = (math.sqrt(f_base ** 2 + (worst / float(f_num)) ** 2)
                           - f_base) / f_base

            if penalty is not None and penalty < 0.05:
                insig_label = QLabel(self.tr(
                    "✓  Even taken at face value this tilt costs about {0:.1f}% "
                    "on FWHM at the worst edge — not worth chasing. "
                    "(Significance {1:.1f}σ; note that this estimator cannot "
                    "return zero, so a perfectly flat sensor still reports a "
                    "small tilt.)"
                ).format(100.0 * penalty, float(tsig or 0.0)))
                insig_label.setStyleSheet("color:#2d7d46; font-style:italic;")
                insig_label.setWordWrap(True)
            elif tsig is not None and tsig < 3.0:
                insig_label = QLabel(self.tr(
                    "⚠  This tilt is only {0:.1f}σ from zero — consistent with "
                    "no tilt at all. The numbers above are what the fit "
                    "returned, not evidence of a problem. More stars, better "
                    "seeing, or a longer sub would tighten it."
                ).format(float(tsig)))
                insig_label.setStyleSheet("color:#c07000; font-style:italic;")
                insig_label.setWordWrap(True)

            # ----- fit diagnostics (paraboloid FWHM^2 model) -----------------
            if fit_info:
                diag_bits = []
                if fit_info.get("fwhm_baseline") is not None:
                    diag_bits.append(self.tr("Baseline FWHM: {0:.2f} µm")
                                     .format(float(fit_info["fwhm_baseline"])))
                if fit_info.get("n_used") is not None:
                    diag_bits.append(self.tr("Stars used: {0}")
                                     .format(int(fit_info["n_used"])))
                if fit_info.get("f_number") is not None:
                    diag_bits.append(self.tr("f/{0:.1f}")
                                     .format(float(fit_info["f_number"])))
                if fit_info.get("rms_residual") is not None:
                    # _SASPRO_ONE_TILT: the defocus model's residual is in µm, not
                    # µm² -- it no longer works in FWHM².
                    diag_bits.append(self.tr("RMS residual: {0:.2f} µm")
                                     .format(float(fit_info["rms_residual"])))
                # _SASPRO_ONE_TILT: curvature is fitted separately now, so show it
                # here too rather than leaving it to the other dialog.
                if fit_info.get("curvature_um") is not None:
                    _cv = float(fit_info["curvature_um"])
                    _cs = fit_info.get("curvature_sig")
                    diag_bits.append(self.tr("Curvature: {0:.1f} µm").format(_cv)
                                     + (f" ({float(_cs):.1f}σ)" if _cs else ""))
                if diag_bits:
                    diag_label = QLabel("   |   ".join(diag_bits))

                # Planarity: how close the defocus surface is to a plane.
                # A tilted sensor defocuses along one direction and stays
                # flat along the perpendicular one, so a pure tilt has
                # planarity ~ 0. A large value means something curved is
                # present as well, and a single tilt plane is not the whole
                # story -- worth saying, because the corner numbers above
                # are reported as if it were.
                _pl = fit_info.get("planarity")
                if _pl is not None and np.isfinite(_pl) and _pl > 0.35:
                    curve_label = QLabel(self.tr(
                        "⚠  The defocus surface is not flat (planarity "
                        "{0:.2f}; a pure tilt gives ~0). Field curvature or "
                        "astigmatism is contributing, so these corner "
                        "numbers describe only the tilt-like part of it."
                    ).format(float(_pl)))
                    curve_label.setStyleSheet("color:#c07000; font-style:italic;")
                    curve_label.setWordWrap(True)

                if not fit_info.get("well_defined", True):
                    note_label = QLabel(self.tr(
                        "⚠  Paraboloid model is ill-conditioned for this frame — "
                        "showing linear FWHM-vs-position fallback. Tilt magnitudes "
                        "may be less reliable; consider a longer sub with more stars."
                    ))
                    note_label.setStyleSheet("color:#c07000; font-style:italic;")
                    note_label.setWordWrap(True)
                else:
                    note_label = QLabel(self.tr(
                        "Note: the sign of tilt (which corner sits high vs low) "
                        "is not recoverable from a single image — only the "
                        "magnitude is. Through-focus data is required to break "
                        "the ambiguity."
                    ))
                    note_label.setStyleSheet("color:#555; font-style:italic;")
                    note_label.setWordWrap(True)

        # ----- layout ---------------------------------------------------------
        layout = QVBoxLayout(self)
        layout.addWidget(self.view, 1)
        if range_label is not None:
            layout.addWidget(range_label, 0)
        if diag_label is not None:
            layout.addWidget(diag_label, 0)
        if table is not None:
            layout.addWidget(table, 0)
        if insig_label is not None:
            layout.addWidget(insig_label, 0)
        if curve_label is not None:
            layout.addWidget(curve_label, 0)
        if note_label is not None:
            layout.addWidget(note_label, 0)
        close_btn = QPushButton(self.tr("Close"), self)
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn, 0)

        self.view.fit_to_view()

def compute_fwhm_heatmap_full(
    img: np.ndarray,
    pixel_scale: float,
    thresh_sigma: float = 5.0
) -> np.ndarray:
    """
    1) Detect stars with SEP, measure fwhm_um = 2*a*pixel_scale
    2) Interpolate fwhm_um onto the full H×W grid with cubic+nearest
    3) Normalize to [0..1] and return that heatmap
    """
    gray = img.mean(axis=2).astype(np.float32) if img.ndim==3 else img.astype(np.float32)
    H, W = gray.shape
    data = np.ascontiguousarray(gray, dtype=np.float32)
    bkg  = sep.Background(data)
    objs = sep.extract(data - bkg.back(), thresh=thresh_sigma, err=bkg.globalrms)
    if objs is None or len(objs) < 5:
        return np.zeros((H,W),dtype=float)

    x = objs['x']; y = objs['y']
    fwhm_um = 2.0 * objs['a'] * pixel_scale

    # create interpolation grid
    grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
    points = np.vstack([x, y]).T

    # first cubic, then nearest for NaNs
    heat = griddata(points, fwhm_um, (grid_x, grid_y), method='cubic')
    mask = np.isnan(heat)
    if mask.any():
        heat[mask] = griddata(points, fwhm_um, (grid_x, grid_y), method='nearest')[mask]

    # normalize
    mn, mx = heat.min(), heat.max()
    return (heat - mn)/max(mx-mn,1e-9)



def fit_2d_poly(x, y, z, deg=2, sigma_clip=3.0, max_iter=3, weights=None):
    """
    Fit z(x,y) = Σ_{i+j≤deg} c_{ij} x^i y^j
    by linear least squares + sigma-clipping.
    Returns the flattened coeff array.

    `weights`, when given, are per-sample weights (NOT sigmas) applied as a
    standard weighted least squares: both sides are scaled by sqrt(w) before
    the solve. The natural choice is inverse variance, 1/sigma^2.

    Weighting matters for shape surfaces specifically. Star measurements span
    orders of magnitude in quality across one frame -- a bright well-sampled
    star pins its local ellipticity to a few percent while a marginal one is
    almost pure noise -- and an unweighted fit lets them vote equally. The
    sigma-clip alone cannot fix that: a noisy value near the surface is kept
    at full strength, and one far from it is discarded entirely, when what it
    deserves is partial weight.

    Residuals are scaled by sqrt(w) for the clipping test as well, so the
    clip is in units of sigma rather than raw value -- otherwise a precise
    star with a small real residual gets clipped before an imprecise one
    with a large expected residual.
    """
    # Build list of (i,j) exponents
    exps = [(i, j) for total in range(deg+1)
                  for i in range(total+1)
                  for j in [total - i]]
    # Design matrix
    A = np.vstack([ (x**i)*(y**j) for (i,j) in exps ]).T  # shape (N,len(exps))
    z = np.asarray(z, dtype=np.float64)
    mask = np.ones_like(z, bool)

    if weights is None:
        w = None
        sw = np.ones_like(z)
    else:
        w = np.asarray(weights, dtype=np.float64)
        w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
        w = np.maximum(w, 0.0)
        sw = np.sqrt(w)

    n_params = len(exps)
    sol = np.zeros(n_params, dtype=np.float64)

    for _ in range(max_iter):
        if int(np.count_nonzero(mask)) < n_params:
            break
        if w is None:
            sol, *_ = np.linalg.lstsq(A[mask], z[mask], rcond=None)
        else:
            Aw = A[mask] * sw[mask, None]
            zw = z[mask] * sw[mask]
            if not np.any(sw[mask] > 0):
                break
            sol, *_ = np.linalg.lstsq(Aw, zw, rcond=None)
        resid = (z - A.dot(sol)) * sw
        std = np.std(resid[mask])
        if not np.isfinite(std) or std <= 0:
            break
        newm = np.abs(resid) < sigma_clip*std
        if int(newm.sum()) < n_params:
            break
        if newm.sum() == mask.sum():
            break
        mask = newm

    return sol, exps

def eval_2d_poly(sol, exps, X, Y):
    """
    Evaluate the polynomial with coeffs sol and exponents exps
    on a full grid X,Y.
    """
    Z = np.zeros_like(X, float)
    for c,(i,j) in zip(sol, exps):
        Z += c * (X**i) * (Y**j)
    return Z

def compute_fwhm_surface(img, pixel_scale, thresh_sigma=5.0, deg=3):
    """
    FWHM (um) heatmap. Uses proper Gaussian conversion 2.3548 * sqrt(a*b) * ps
    rather than the historical 2*a underestimate.
    """
    gray = _to_gray32(img)
    H, W = gray.shape
    cat = _detect_stars(gray, thresh_sigma=thresh_sigma)
    if cat is None or cat['n'] < 10:
        return np.zeros((H, W), dtype=np.float32), (0.0, 0.0)

    fwhm_um = _FWHM_FACTOR * np.sqrt(cat['a'] * cat['b']) * float(pixel_scale)
    surf, _ = _fit_and_render_surface(cat['x'], cat['y'], fwhm_um, H, W, deg=deg)
    cover = _coverage_mask(cat['x'], cat['y'], H, W)
    # Range returned is the one the heatmap was scaled with, so a colourbar
    # built from it matches the image.
    heat, rng = _normalize_for_display(surf, cover)
    return heat, rng


def compute_eccentricity_surface(
    img: np.ndarray,
    pixel_scale: float,
    thresh_sigma: float = 5.0,
    deg: int = 3
) -> Tuple[np.ndarray, Tuple[float,float]]:
    """
    Ellipticity heatmap. Canonical astro convention: 1 - b/a, in [0, 1).
    (Reserving the name 'eccentricity' since that's what the rest of the code
    uses, but the value is ellipticity to match PixInsight / SExtractor.)
    """
    gray = _to_gray32(img)
    H, W = gray.shape
    cat = _detect_stars(gray, thresh_sigma=thresh_sigma)
    if cat is None or cat['n'] < 6:
        return np.zeros((H, W), dtype=np.float32), (0.0, 0.0)

    ell = 1.0 - (cat['b'] / cat['a'])
    surf, _ = _fit_and_render_surface(cat['x'], cat['y'], ell, H, W, deg=deg)
    cover = _coverage_mask(cat['x'], cat['y'], H, W)
    heat, rng = _normalize_for_display(surf, cover)
    return heat, rng


def compute_orientation_surface(
    img: np.ndarray,
    thresh_sigma: float = 5.0,
    deg: int = 1,            # for pure tilt a plane is enough
    sigma_clip: float = 3.0,
    max_iter: int = 3,
    max_sigma_theta_deg: float = _MAX_SIGMA_THETA_DEG,
    min_elong_snr: float = _MIN_ELONG_SNR,
    min_abs_elong: float = _MIN_ABS_ELONG,
    min_stars: int = _MIN_ORIENT_STARS,
    return_info: bool = False,
):
    """
    Smooth orientation surface theta(x, y) from the stars that have a
    measurable position angle.

    Fits the ellipticity vector (e1, e2) weighted by measurement quality,
    over stars that pass both the sampling cut (applied during detection)
    and the elongation cut -- see _fit_orientation_field for why the vector
    form matters and why a unit-direction fit over every detection produces
    a convincing map of nothing.

    Returns hue in [0, 1] and the raw theta range (radians) over the
    CONTRIBUTING stars. When too few stars qualify, the hue map is all zeros
    and the range is (0, 0); pass return_info=True to get the third element
    describing what happened, so a caller can tell "no usable stars" from a
    genuine field of zero orientation.
    """
    gray = _to_gray32(img)
    H, W = gray.shape
    cat = _detect_stars(gray, thresh_sigma=thresh_sigma)
    blank = np.zeros((H, W), dtype=np.float32)
    if cat is None or cat['n'] < 6:
        info = {'ok': False, 'n_used': 0, 'n_input': 0 if cat is None else cat['n'],
                'reason': 'no adequately sampled stars detected'}
        return (blank, (0.0, 0.0), info) if return_info else (blank, (0.0, 0.0))

    ori = _fit_orientation_field(cat, H, W, deg=deg, sigma_clip=sigma_clip,
                                 max_sigma_theta_deg=max_sigma_theta_deg,
                                 min_elong_snr=min_elong_snr,
                                 min_abs_elong=min_abs_elong,
                                 min_stars=min_stars)
    if not ori.get('ok'):
        info = dict(ori)
        info['reason'] = (f"only {ori.get('n_used', 0)} of {cat['n']} stars have a "
                          f"position angle good to "
                          f"{max_sigma_theta_deg:.0f} deg")
        return (blank, (0.0, 0.0), info) if return_info else (blank, (0.0, 0.0))

    hue = ori['theta_norm']
    rng = ori['theta_range']
    return (hue, rng, ori) if return_info else (hue, rng)

class DiagnosisDialog(QDialog):
    """
    Plain-language verdict from diagnose_optics: what is wrong, how sure we
    are, and what to adjust. Worst finding first.

    Deliberately leads with the conclusion rather than the measurements. The
    heatmaps already show the symptom; the thing a user cannot get by looking
    at them is whether the corners are soft because the sensor is tilted or
    because the spacer is wrong, and those call for completely different
    actions.
    """
    _COLOURS = {'major': '#c0392b', 'minor': '#c07000', 'ok': '#2d7d46'}
    _MARKS = {'major': '●', 'minor': '●', 'ok': '✓'}

    def __init__(self, diagnosis: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Optical Diagnosis"))
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        findings = (diagnosis or {}).get('findings', [])
        if not findings:
            layout.addWidget(QLabel(self.tr("No diagnosis available.")))

        for sev, headline, detail in findings:
            box = QGroupBox()
            inner = QVBoxLayout(box)
            h = QLabel(f"{self._MARKS.get(sev, '•')}  {headline}")
            h.setStyleSheet(
                f"color:{self._COLOURS.get(sev, '#555')};"
                f"font-weight:bold; font-size:11pt;")
            h.setWordWrap(True)
            inner.addWidget(h)
            d = QLabel(detail)
            d.setWordWrap(True)
            d.setStyleSheet("color:#bbb;")
            inner.addWidget(d)
            layout.addWidget(box)

        # Measured numbers, for anyone who wants to check the reasoning.
        dfc = (diagnosis or {}).get('defocus')
        if dfc:
            t = QTableWidget(5, 2)
            t.setHorizontalHeaderLabels([self.tr("Quantity"), self.tr("Value")])
            t.verticalHeader().setVisible(False)
            rows = [
                (self.tr("Baseline FWHM (best focus)"),
                 f"{dfc['fwhm0_um']:.2f} µm"),
                (self.tr("Tilt, peak-to-peak"),
                 f"{dfc['tilt_um']:.1f} ± {dfc['tilt_se']:.1f} µm "
                 f"({dfc.get('tilt_sig', 0):.1f}σ)"),
                (self.tr("Curvature, centre to corner"),
                 f"{dfc['curvature_um']:.1f} ± {dfc['curvature_se']:.1f} µm "
                 f"({dfc.get('curvature_sig', 0):.1f}σ)"),
                (self.tr("Stars used / RMS residual"),
                 f"{dfc['n_used']} / {dfc['rms_resid_um']:.2f} µm"),
                (self.tr("Working f-ratio"), f"f/{dfc['f_number']:.1f}"),
            ]
            for i, (k, v) in enumerate(rows):
                t.setItem(i, 0, QTableWidgetItem(k))
                t.setItem(i, 1, QTableWidgetItem(v))
            t.resizeColumnsToContents()
            t.setMaximumHeight(180)
            layout.addWidget(t)

        note = QLabel(self.tr(
            "Signs are not recoverable from a single frame — which side sits "
            "high, or whether the spacer is long or short, needs either a "
            "second frame at a different spacing or one trial adjustment. "
            "The magnitudes above are unambiguous."))
        note.setWordWrap(True)
        note.setStyleSheet("color:#777; font-style:italic;")
        layout.addWidget(note)

        btn = QPushButton(self.tr("Close"))
        btn.clicked.connect(self.accept)
        layout.addWidget(btn)
        self.resize(560, 520)


# --- _SASPRO_AXIS_TICKS : begin --------------------------------------------------
# Drawing an AXIS as a hue is the core legibility problem. A hue ramp has no
# natural zero (both ends of "hsv" are red, because +90 and -90 degrees are
# the same axis) and cannot resolve the 20-degree differences that matter
# here -- 17 vs 45 degrees is green versus slightly different green. A line
# segment lying along the measured axis is read instantly and can be compared
# directly against the stars underneath it.
_TICK_GRID = (10, 7)          # columns, rows of the fitted-field tick lattice
_TICK_MIN_PX = 14.0           # a tick this short means "no measurable axis"
_TICK_GAIN = 420.0            # pixels of length per unit |e|
_TICK_BACKDROP = 0.45         # dim the frame so the ticks carry the eye


def _axis_tick_overlay(pane, base_image, theta_surf, emag_surf,
                       zones=None, conf=None, bins_in=None):   # _SASPRO_LOCAL_AXES
    """
    _SASPRO_AXIS_TICKS: render the orientation fit as axis ticks over the frame.

    Returns an H x W x 3 uint8 array, or None if it cannot be built -- the
    caller then simply offers no toggle.

    theta_surf is in the module's y-down frame, so a tick is drawn with
    (cos t, sin t) directly in widget coordinates and comes out lying along
    the elongation as it appears on screen. No sign flip anywhere: that
    equivalence is the reason this view cannot disagree with the heatmap.
    """
    try:
        a = np.asarray(base_image)
        if a.ndim == 3:
            a = a.mean(axis=2)
        a = a.astype(np.float32)
        H, W = a.shape
        if theta_surf.shape != (H, W):
            return None

        # Percentile stretch so stars are visible without blowing out nebula.
        lo, hi = np.percentile(a[np.isfinite(a)], (20.0, 99.3))
        if not np.isfinite(hi) or hi <= lo:
            lo, hi = float(np.nanmin(a)), float(np.nanmax(a)) or 1.0
        g = np.clip((a - lo) / max(hi - lo, 1e-9), 0.0, 1.0) * _TICK_BACKDROP
        rgb = (np.repeat(g[..., None], 3, axis=2) * 255.0).astype(np.uint8)

        qimg = pane.numpy_to_qimage(rgb).convertToFormat(
            QImage.Format.Format_RGB888)
        p = QPainter(qimg)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        def tick(cx, cy, t, length, colour, width, alpha=255):
            dx = 0.5 * length * math.cos(t)
            dy = 0.5 * length * math.sin(t)     # y-down: no sign flip
            col = QColor(colour)
            col.setAlpha(int(alpha))
            pen = QPen(col, float(width))
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            p.setPen(pen)
            p.drawLine(int(cx - dx), int(cy - dy), int(cx + dx), int(cy + dy))

        # The fitted field, on a lattice.
        # _SASPRO_LOCAL_AXES: draw each cell's OWN stars. Sampling the fitted surface
        # here was wrong by 48 degrees in a real corner, because a global
        # polynomial cannot follow a corner that disagrees with the frame.
        # A cell without enough stars falls back to the surface and is drawn
        # faint, so "too few stars" never looks like "measured, and weak".
        nx, ny = _TICK_GRID
        bins = {}
        for b in (bins_in or []):
            bins[(int(b.get("ix", -1)), int(b.get("iy", -1)))] = b
        for iy in range(ny):
            for ix in range(nx):
                px = int((ix + 0.5) * W / nx)
                py = int((iy + 0.5) * H / ny)
                b = bins.get((ix, iy))
                if b is not None and b.get("ok"):
                    e, t, measured = float(b["e_mag"]), float(b["theta"]), True
                else:
                    e, t, measured = (float(emag_surf[py, px]),
                                      float(theta_surf[py, px]), False)
                a_ = 255
                if conf is not None and conf.shape == (H, W):
                    a_ = int(70 + 185 * float(np.clip(conf[py, px], 0.0, 1.0)))
                if not measured:
                    a_ = int(a_ * 0.45)
                tick(px, py, t, _TICK_MIN_PX + _TICK_GAIN * e,
                     "#19d9ff" if measured else "#6f8fa0", 2.2, a_)

        # The nine anchor zones, measured straight from their own stars.
        if zones:
            p.setFont(QFont("", max(8, int(min(H, W) / 90))))
            for k, z in enumerate(zones):
                if not z.get("n"):
                    continue
                r_, c_ = divmod(k, 3)
                px = int((c_ + 0.5) * W / 3)
                py = int((r_ + 0.5) * H / 3)
                e = float(z.get("e_mag") or 0.0)
                if not np.isfinite(e):
                    continue
                strong = float(z.get("sig") or 0.0) >= 3.0
                tick(px, py, math.radians(float(z.get("angle_deg") or 0.0)),
                     _TICK_MIN_PX * 2.0 + _TICK_GAIN * 1.7 * e,
                     "#ffd400" if strong else "#ff8c3a", 4.2,
                     255 if strong else 130)
                col = QColor("#ffd400" if strong else "#ff8c3a")
                col.setAlpha(255 if strong else 150)
                p.setPen(QPen(col))
                p.drawText(int(px - 42), int(py + int(min(H, W) / 22)),
                           f"|e|={e:.3f}")
        p.end()
        return pane.qimage_to_numpy(qimg)
    except Exception as exc:
        print(f"[image peeker] tick overlay skipped: {type(exc).__name__}: {exc}")
        return None
# --- _SASPRO_AXIS_TICKS : end ----------------------------------------------------


class SurfaceDialog(QDialog):
    def __init__(self, title, heatmap, vmin, vmax, units:str="", cmap="gray", parent=None,
                 confidence=None,   # _SASPRO_MAP_CONFIDENCE
                 base_image=None, ticks=None,   # _SASPRO_AXIS_TICKS
                 lightness=None):   # _SASPRO_HSL_LIGHTNESS
        super().__init__(parent)
        self.setWindowTitle(title)

        # image
        # image (apply the chosen colormap if it’s a 2D heatmap,
        # and load RGB directly if it’s already color)
        from matplotlib import cm
        import matplotlib.pyplot as plt

        view = PreviewPane()
        if heatmap.ndim == 2:
            # 1) map to RGBA via colormap
            cmap_obj = _get_cmap(cmap)   # _SASPRO_MAP_CONFIDENCE: 3.9-safe
            rgba = cmap_obj(heatmap)            # shape H×W×4, floats 0–1
            rgb  = (rgba[...,:3] * 255).astype(np.uint8)
            # _SASPRO_HSL_LIGHTNESS: for a cyclic hue map, compose in HSL and take the
            # lightness from the companion map instead of scaling RGB. See
            # _hsl_to_rgb for why value is the wrong channel for this.
            if lightness is not None:
                try:
                    _l = np.asarray(lightness, dtype=np.float32)
                    if _l.shape == heatmap.shape:
                        rgb = _hsl_to_rgb(heatmap, np.ones_like(_l), _l)
                except Exception:
                    pass
            # _SASPRO_MAP_CONFIDENCE: brightness carries how much to believe the
            # colour. Hue cannot: 0 and 180 degrees are both bright red,
            # so a region of round stars would otherwise render as a
            # confident direction that is not there.
            if confidence is not None:
                try:
                    c = np.clip(np.asarray(confidence, dtype=np.float32),
                                0.0, 1.0)
                    if c.shape == rgb.shape[:2]:
                        rgb = (rgb.astype(np.float32)
                               * c[..., None]).astype(np.uint8)
                except Exception:
                    pass
            view.load_numpy(rgb)
        else:
            # assume already float32 [0..1] RGB or uint8 RGB
            view.load_numpy(heatmap)
            rgb = heatmap
        view.fit_to_view()

        # _SASPRO_AXIS_TICKS: build the tick view now so the toggle is instant.
        self._view = view
        self._rgb_heat = rgb
        self._rgb_ticks = None
        if ticks is not None and base_image is not None:
            self._rgb_ticks = _axis_tick_overlay(
                view, base_image,
                ticks.get("theta"), ticks.get("emag"),
                zones=ticks.get("zones"), conf=ticks.get("conf"),
                bins_in=ticks.get("bins"))   # _SASPRO_LOCAL_AXES

        # colorbar pixmap
        cb = self._make_colorbar(cmap, vmin, vmax, units)
        lbl_cb = QLabel()
        lbl_cb.setPixmap(cb)

        # layout
        h = QHBoxLayout()
        h.addWidget(view, 1)
        h.addWidget(lbl_cb, 0)

        # _SASPRO_AXIS_TICKS: one fit, two readings, one click apart.
        self._btn_toggle = None
        if self._rgb_ticks is not None:
            self._btn_toggle = QPushButton(self.tr("Show axis ticks"))
            self._btn_toggle.setCheckable(True)
            self._btn_toggle.clicked.connect(self._toggle_ticks)

        btn = QPushButton(self.tr("Close"))
        btn.clicked.connect(self.accept)
        lbl_span = QLabel(self.tr("Span: {0:.2f} … {1:.2f} {2}").format(vmin, vmax, units))
        if confidence is not None:   # _SASPRO_MAP_CONFIDENCE
            lbl_span.setText(lbl_span.text() + self.tr(
                "    •    dimmed where the measurement is weak: near-round "
                "stars, below ~2 px FWHM, or outside the star coverage"))

        v = QVBoxLayout(self)
        v.addLayout(h)
        v.addWidget(lbl_span)
        if self._btn_toggle is not None:   # _SASPRO_AXIS_TICKS
            v.addWidget(self._btn_toggle)
        v.addWidget(btn)

    def _toggle_ticks(self, on: bool):
        """_SASPRO_AXIS_TICKS: swap between the tick view and the heatmap, keeping zoom."""
        try:
            arr = self._rgb_ticks if on else self._rgb_heat
            if arr is None:
                return
            pane = self._view
            z = getattr(pane, "zoom_factor", 1.0)
            pane.load_numpy(arr)
            try:
                pane.zoom_factor = z
                pane.zoom_slider.setValue(int(round(z * 100)))
            except Exception:
                pane.fit_to_view()
            self._btn_toggle.setText(
                self.tr("Show heatmap") if on else self.tr("Show axis ticks"))
        except Exception as exc:
            print(f"[image peeker] tick toggle failed: "
                  f"{type(exc).__name__}: {exc}")

    def _make_colorbar(self, cmap_name, vmin, vmax, units):
        # build a 256×20 gradient in RGBA
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib import cm
        grad = np.linspace(0,1,256)[:,None]
        bar  = _get_cmap(cmap_name)(grad)   # _SASPRO_MAP_CONFIDENCE: 3.9-safe
        bar  = (bar[:,:,:3]*255).astype(np.uint8)
        # make a QImage
        H,W,_ = bar.shape
        img = QImage(bar.data, 1, 256, 3*1, QImage.Format.Format_RGB888)
        # rotate to vertical
        return QPixmap.fromImage(img.mirrored(False, True).scaled(20,256))

def distortion_vectors_sip(x_pix, y_pix, sip, pixel_size_um):
    """
    Evaluate the SIP Δ‐pixels at the given star positions,
    return (dx_um, dy_um) and also the raw dx_pix,dy_pix arrays.
    """
    A = sip.a
    B = sip.b
    order = A.shape[0] - 1

    # pull off CRPIX so u,v are relative to the SIP origin
    crpix1, crpix2 = sip.forward_origin   # equivalent to wcs.wcs.crpix
    u = x_pix - crpix1
    v = y_pix - crpix2

    dx_pix = np.zeros_like(u)
    dy_pix = np.zeros_like(u)

    # vectorized polynomial evaluation
    for i in range(order+1):
        for j in range(order+1-i):
            a_ij = A[i, j]
            b_ij = B[i, j]
            if a_ij:
                dx_pix += a_ij * (u**i) * (v**j)
            if b_ij:
                dy_pix += b_ij * (u**i) * (v**j)

    dx_um = dx_pix * pixel_size_um
    dy_um = dy_pix * pixel_size_um

    return dx_pix, dy_pix, dx_um, dy_um

def distortion_vectors(img: np.ndarray,
                       sip_meta: dict,
                       pixel_size_um: float):
    """
    1) SEP detect stars → x_pix,y_pix
    2) extract A,B,crpix from sip_meta
    3) eval dx_pix,dy_pix → dx_um,dy_um
    4) return overlays
    """
    # 1) detect stars
    gray = img.mean(-1).astype(np.float32) if img.ndim==3 else img.astype(np.float32)
    data = np.ascontiguousarray(gray, np.float32)
    bkg  = sep.Background(data)
    stars = sep.extract(data - bkg.back(),
                        thresh=5.0, err=bkg.globalrms)
    if stars is None:
        return []

    x_pix = stars['x']; y_pix = stars['y']

    # 2) pull SIP from meta (now robust to missing A_ORDER)
    A, B, crpix1, crpix2 = extract_sip_from_meta(sip_meta)

    # 3) vector‐polynomial evaluation
    u = x_pix - crpix1
    v = y_pix - crpix2
    dx_pix = np.zeros_like(u)
    dy_pix = np.zeros_like(u)
    order  = A.shape[0] - 1
    for i in range(order+1):
        for j in range(order+1-i):
            a_ij = A[i, j]
            b_ij = B[i, j]
            if a_ij:
                dx_pix += a_ij * (u**i) * (v**j)
            if b_ij:
                dy_pix += b_ij * (u**i) * (v**j)

    # 4) to microns & pack
    dx_um = dx_pix * pixel_size_um
    dy_um = dy_pix * pixel_size_um

    overlays = []
    for x,y,dx,dy in zip(x_pix, y_pix, dx_um, dy_um):
        ang    = math.atan2(dy, dx)
        length = math.hypot(dx, dy)
        overlays.append((int(x), int(y), ang, length))
    return overlays

def eval_sip(A, B, u, v):
    """
    Vectorized SIP evaluation: given coefficient arrays A,B and 
    coordinate offsets u=x-crpix1, v=y-crpix2, returns dx_pix, dy_pix.
    """
    dx = np.zeros_like(u)
    dy = np.zeros_like(u)
    order = A.shape[0]-1
    for i in range(order+1):
        for j in range(order+1-i):
            a = A[i, j]
            b = B[i, j]
            if a:
                dx += a * (u**i)*(v**j)
            if b:
                dy += b * (u**i)*(v**j)
    return dx, dy

def extract_sip_from_meta(sm: dict):
    """
    Given the metadata dict that ASTAP wrote into your slot,
    pull out the forward SIP polynomials A and B (and the reference pixel).
    We no longer rely on A_ORDER existing; we infer it from the A_i_j keys.
    """
    # 1) find all the A_i_j keys that actually made it into sm
    a_keys = [k for k in sm.keys() if re.match(r"A_\d+_\d+", k)]
    if not a_keys:
        raise ValueError("No SIP A_?_? coefficients found in metadata!")

    # 2) parse out all the (i,j) pairs and infer the polynomial order as max(i+j)
    pairs = [tuple(map(int, k.split("_")[1:])) for k in a_keys]
    order = max(i+j for i,j in pairs)

    # 3) allocate forward‐SIP coefficient arrays
    A = np.zeros((order+1, order+1), float)
    B = np.zeros((order+1, order+1), float)

    for i, j in pairs:
        A[i, j] = float(sm[f"A_{i}_{j}"])
        B[i, j] = float(sm[f"B_{i}_{j}"])

    # 4) pull the reference pixel
    crpix1 = float(sm["CRPIX1"])
    crpix2 = float(sm["CRPIX2"])

    return A, B, crpix1, crpix2

class DistortionGridDialog(QDialog):
    def __init__(self,
                img: np.ndarray,
                sip_meta: dict,
                arcsec_per_pix: float,
                n_grid_lines: int = 10,
                amplify: float    = 20.0,
                document=None,
                main_window=None,
                parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Astrometric Distortion & Histogram"))

        self._doc = document
        self._main_window = main_window

        # — 1) detect stars — single high-sigma pass, then subsample if too many
        gray = img.mean(-1).astype(np.float32) if img.ndim==3 else img.astype(np.float32)
        data = np.ascontiguousarray(gray, np.float32)
        bkg  = sep.Background(data)
        bkg_sub = data - bkg.back()

        _target_max = 1000
        stars = sep.extract(bkg_sub, thresh=50.0, err=bkg.globalrms)
        n = len(stars) if stars is not None else 0

        if stars is None or n == 0:
            stars = sep.extract(bkg_sub, thresh=10.0, err=bkg.globalrms)
            n = len(stars) if stars is not None else 0

        if n > _target_max:
            order = np.argsort(stars['flux'])[::-1][:_target_max]
            stars = stars[order]

        if stars is None or len(stars) < 10:
            QMessageBox.warning(self, self.tr("Distortion"), self.tr("Not enough stars found."))
            self.reject()
            return

        x_pix = stars['x']
        y_pix = stars['y']

        # — 2) extract SIP A,B and reference pixel from metadata dict —
        A, B, crpix1, crpix2 = extract_sip_from_meta(sip_meta)

        # — 3) per-star residuals in pixels → arc-sec —
        u_star = x_pix - crpix1
        v_star = y_pix - crpix2
        dx_star_pix, dy_star_pix = eval_sip(A, B, u_star, v_star)
        disp_star_pix    = np.hypot(dx_star_pix, dy_star_pix)
        disp_star_arcsec = disp_star_pix * arcsec_per_pix

        # — store everything the grid rebuild needs —
        H, W = data.shape
        self._A, self._B = A, B
        self._crpix1, self._crpix2 = crpix1, crpix2
        self._H, self._W = H, W
        self._arcsec_per_pix = float(arcsec_per_pix)
        self._n_grid_lines = int(n_grid_lines)

        # — 4) grid scene + view (populated by _rebuild_grid) —
        self._scene = QGraphicsScene(self)
        self._scene.setBackgroundBrush(QColor(30,30,30))
        self._view = QGraphicsView(self._scene)
        self._view.setRenderHint(QPainter.RenderHint.Antialiasing)

        title = QLabel(self.tr("Astrometric Distortion Grid"))
        title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("color: white;")

        left_layout = QVBoxLayout()
        left_layout.addWidget(title)
        left_layout.addWidget(self._view, 1)

        init_amp = max(0.0, min(200.0, float(amplify)))
        self._rebuild_grid(init_amp)

        # — 5) histogram of per-star residuals (arcsec) —
        fig    = Figure(figsize=(4,4))
        canvas = FigureCanvas(fig)
        ax     = fig.add_subplot(111)
        ax.hist(disp_star_arcsec, bins=30, edgecolor='black')
        ax.set_xlabel(self.tr("Distortion (″)"))
        ax.set_ylabel(self.tr("Number of stars"))
        ax.set_title(self.tr("Residual histogram"))
        fig.tight_layout()

        hl = QHBoxLayout()
        hl.addLayout(left_layout, 1)
        hl.addWidget(canvas, 1)

        # — 6) controls: warp slider + Unwarp hook —
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel(self.tr("Grid warp:")))
        self._amp_slider = QSlider(Qt.Orientation.Horizontal)
        self._amp_slider.setRange(0, 200)
        self._amp_slider.setValue(int(round(init_amp)))
        self._amp_slider.setMinimumWidth(160)
        self._amp_slider.valueChanged.connect(self._on_amp_changed)
        self._amp_value = QLabel(f"×{int(round(init_amp))}")
        ctrl.addWidget(self._amp_slider, 1)
        ctrl.addWidget(self._amp_value)
        ctrl.addSpacing(16)

        self._unwarp_btn = QPushButton(self.tr("Unwarp This Image…"))
        self._unwarp_btn.setToolTip(self.tr(
            "Resample the image onto a distortion-free grid using this SIP "
            "solution, then re-solve to confirm a clean field."))
        self._unwarp_btn.clicked.connect(self._open_unwarp)
        if self._doc is None:
            self._unwarp_btn.setEnabled(False)
            self._unwarp_btn.setToolTip(self.tr("No document is associated with this analysis view."))
        ctrl.addWidget(self._unwarp_btn)

        # close button
        btn = QPushButton(self.tr("Close"))
        btn.clicked.connect(self.accept)

        # final
        v = QVBoxLayout(self)
        v.addLayout(hl)
        v.addLayout(ctrl)
        v.addWidget(btn, 0)

    def _on_amp_changed(self, v: int):
        self._amp_value.setText(f"×{v}")
        self._rebuild_grid(float(v))

    def _rebuild_grid(self, amplify: float):
        self._amplify = float(amplify)
        A, B = self._A, self._B
        crpix1, crpix2 = self._crpix1, self._crpix2
        H, W = self._H, self._W
        n = self._n_grid_lines

        scene = self._scene
        scene.clear()
        pen = QPen(QColor(255,100,100), 1)
        label_font = QFont("Arial", 12, QFont.Weight.Bold)

        def warp_points(xs, ys):
            xs = np.asarray(xs, dtype=np.float64)
            ys = np.asarray(ys, dtype=np.float64)
            dx_pix, dy_pix = eval_sip(A, B, xs - crpix1, ys - crpix2)
            return dx_pix * self._amplify, dy_pix * self._amplify

        # horizontal grid lines
        for i in range(n+1):
            y0 = i*(H-1)/n
            xs = np.linspace(0, W-1, 200); ys = np.full_like(xs, y0)
            dxl, dyl = warp_points(xs, ys)
            warped = np.column_stack([xs + dxl, ys + dyl])
            path = QPainterPath(QPointF(*warped[0]))
            for px, py in warped[1:]:
                path.lineTo(QPointF(px, py))
            scene.addPath(path, pen)

        # vertical grid lines
        for j in range(n+1):
            x0 = j*(W-1)/n
            ys = np.linspace(0, H-1, 200); xs = np.full_like(ys, x0)
            dxl, dyl = warp_points(xs, ys)
            warped = np.column_stack([xs + dxl, ys + dyl])
            path = QPainterPath(QPointF(*warped[0]))
            for px, py in warped[1:]:
                path.lineTo(QPointF(px, py))
            scene.addPath(path, pen)

        # annotate intersections — label is the true (un-amplified) residual,
        # position uses the current amplification so it tracks the warped node
        ii, jj = np.meshgrid(np.arange(n+1), np.arange(n+1))
        x0_all = jj.ravel() * (W-1) / n
        y0_all = ii.ravel() * (H-1) / n
        dxr, dyr = eval_sip(A, B, x0_all - crpix1, y0_all - crpix2)  # raw pixels
        for idx in range(x0_all.size):
            d_arcsec = math.hypot(dxr[idx], dyr[idx]) * self._arcsec_per_pix
            px = x0_all[idx] + dxr[idx] * self._amplify
            py = y0_all[idx] + dyr[idx] * self._amplify
            txt = QGraphicsTextItem(f"{d_arcsec:.1f}\"")
            txt.setFont(label_font)
            txt.setScale(5.0)
            txt.setDefaultTextColor(QColor(200,200,200))
            txt.setPos(px + 4, py + 4)
            scene.addItem(txt)

        scene.setSceneRect(scene.itemsBoundingRect())
        self._view.fitInView(scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def showEvent(self, ev):
        super().showEvent(ev)
        QTimer.singleShot(0, lambda: self._view.fitInView(
            self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio))

    def _open_unwarp(self):
        doc = self._doc
        if doc is None:
            return
        mw = self._main_window
        try:
            from setiastro.saspro.unwarp import UnwarpDialog
        except Exception as e:
            QMessageBox.warning(self, self.tr("Unwarp"),
                                self.tr("Unwarp tool unavailable:\n{0}").format(e))
            return
        dm = None
        if mw is not None:
            dm = getattr(mw, "doc_manager", None) or getattr(mw, "docman", None)
        dlg = UnwarpDialog(
            parent=mw or self,
            settings=getattr(mw, "settings", None) if mw is not None else None,
            doc_manager=dm,
            list_open_docs_fn=getattr(mw, "_list_open_docs", None) if mw is not None else None,
            document=doc,
        )
        try:
            from setiastro.saspro.resources import unwarp_path
            dlg.setWindowIcon(QIcon(unwarp_path))
        except Exception:
            pass
        dlg.show(); dlg.raise_(); dlg.activateWindow()

def make_header_from_xisf_meta(meta: dict) -> fits.Header:
    """
    meta is the dict you returned as original_header for XISF:
      {
        'file_meta': ...,
        'image_meta': ...,
        'astrometry': {
           'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2',
           'crpix1', 'crpix2',
           'sip': {'order', 'A', 'B'}
        }
      }
    This builds a real fits.Header with WCS+SIP cards.
    """
    hdr = fits.Header()
    ast = meta['astrometry']

    # WCS linear part
    hdr['CTYPE1'] = 'RA---TAN-SIP'
    hdr['CTYPE2'] = 'DEC--TAN-SIP'
    hdr['CRPIX1'] = ast['crpix1']
    hdr['CRPIX2'] = ast['crpix2']
    hdr['CD1_1']  = ast['CD1_1']
    hdr['CD1_2']  = ast['CD1_2']
    hdr['CD2_1']  = ast['CD2_1']
    hdr['CD2_2']  = ast['CD2_2']

    # SIP coefficients
    sip = ast['sip']
    order = sip['order']
    hdr['A_ORDER'] = order
    hdr['B_ORDER'] = order

    for i in range(order+1):
        for j in range(order+1-i):
            hdr[f'A_{i}_{j}'] = float(sip['A'][i,j])
            hdr[f'B_{i}_{j}'] = float(sip['B'][i,j])

    # If you have file_meta FITSKeywords you can also copy those here:
    # for kw, vals in meta['file_meta'].get('FITSKeywords', {}).items():
    #     for entry in vals:
    #         hdr[kw] = entry['value']

    return hdr

def plate_solve_current_image(image_manager, settings, parent=None):
    """
    Plate-solve the current slot image using the SASpro plate solver logic
    (ASTAP + Astrometry.net fallback) and update the slot's metadata in-place.

    Returns the updated metadata dict for the current slot.
    """
    # 1) Grab pixel data + metadata from Image Peeker Pro
    arr, meta = image_manager.get_current_image_and_metadata()
    if meta is None:
        meta = {}
    elif not isinstance(meta, dict):
        meta = dict(meta)

    # 2) Build the seed header from metadata (original_header / wcs / wcs_header)
    seed_h = _seed_header_from_meta(meta)

    # 3) Pick a parent for UI/status if none is given
    if parent is None and hasattr(image_manager, "parent"):
        try:
            parent = image_manager.parent()
        except Exception:
            parent = None

    # 4) Run the actual solve (ASTAP first, then astrometry.net if needed)
    ok, res = _solve_numpy_with_fallback(parent, settings, arr, seed_h)
    if not ok:
        # You can raise, return None, or bubble the error string.
        # Here we raise to make failures obvious.
        raise RuntimeError(f"Plate solve failed: {res}")

    hdr = res  # this is a real astropy.io.fits.Header

    # 5) Store back into metadata
    meta["original_header"] = hdr

    try:
        wcs_obj = WCS(hdr)
        meta["wcs"] = wcs_obj
    except Exception as e:
        print("Image Peeker: WCS build failed:", e)

    # 6) Update image_manager’s internal metadata for this slot
    slot = image_manager.current_slot
    if hasattr(image_manager, "_metadata"):
        image_manager._metadata[slot] = meta

    return meta



# ----------------------------- small utils -----------------------------------

def _ensure_fits_header(orig_hdr):
    if isinstance(orig_hdr, fits.Header):
        return orig_hdr
    if isinstance(orig_hdr, dict) and "astrometry" in orig_hdr:
        try:
            return make_header_from_xisf_meta(orig_hdr)   # use local function
        except Exception:
            return None
    return None
def _arcsec_per_pix_from_header(hdr: fits.Header, fallback_px_um: float|None=None, fallback_fl_mm: float|None=None):
    """Try PC*CDELT scale; fallback to CD matrix; then pixel_size & focal length."""
    if hdr is None:
        if fallback_px_um and fallback_fl_mm:
            return 206.264806 * (fallback_px_um / fallback_fl_mm)
        return None

    # 1) PC matrix * CDELT (modern WCS convention used by astropy fit_wcs_from_points)
    try:
        cdelt1 = float(hdr.get("CDELT1", 1.0))
        cdelt2 = float(hdr.get("CDELT2", 1.0))
        pc1_1  = float(hdr.get("PC1_1", 1.0))
        pc1_2  = float(hdr.get("PC1_2", 0.0))
        pc2_1  = float(hdr.get("PC2_1", 0.0))
        pc2_2  = float(hdr.get("PC2_2", 1.0))
        if "PC1_1" in hdr or "PC2_2" in hdr:
            cd11 = cdelt1 * pc1_1
            cd12 = cdelt1 * pc1_2
            cd21 = cdelt2 * pc2_1
            cd22 = cdelt2 * pc2_2
            scale_deg = np.sqrt(abs(cd11 * cd22 - cd12 * cd21))
            if scale_deg > 0:
                return scale_deg * 3600.0
    except Exception:
        pass

    # 2) CD matrix directly
    try:
        cd11 = float(hdr["CD1_1"])
        cd12 = float(hdr.get("CD1_2", 0.0))
        cd21 = float(hdr.get("CD2_1", 0.0))
        cd22 = float(hdr["CD2_2"])
        scale_deg = np.sqrt(abs(cd11 * cd22 - cd12 * cd21))
        return scale_deg * 3600.0
    except Exception:
        pass

# =============================================================================
#  Background worker for SEP-based analyses
#
#  The Focal Plane and Tilt analyses used to run on the GUI thread (the old
#  QTimer.singleShot(0, ...) trick just posts back to the same event loop, so
#  the app would freeze for the entire pipeline on multi-megapixel frames).
#  This QThread runs everything off-thread and posts the result back via a
#  signal for the dialog to consume.
# =============================================================================
class _AnalysisWorker(QThread):
    progress_signal = pyqtSignal(str)             # human-readable status
    finished_signal = pyqtSignal(str, object)     # (mode, result_or_None)
    error_signal    = pyqtSignal(str, str)        # (mode, message)

    def __init__(self, mode: str, arr: np.ndarray, params: dict, parent=None):
        super().__init__(parent)
        self._mode   = mode
        self._arr    = arr
        self._params = params

    def run(self):
        mode = self._mode
        p = self._params
        try:
            if mode == "Focal Plane Analysis":
                self.progress_signal.emit(self.tr("Detecting stars..."))
                res = analyze_focal_plane(
                    self._arr,
                    pixel_scale  = float(p["pixel_size_um"]),
                    thresh_sigma = float(p.get("snr_threshold", 3.0)),
                    deg          = int(p.get("deg", 3)),
                    coarse       = int(p.get("coarse", 128)),
                    target_stars = int(p.get("target_stars", 1000)),
                    # Optics let the result name the cause (tilt vs backfocus)
                    # rather than only mapping the symptom.
                    focal_length_mm = float(p.get("focal_length_mm") or 0) or None,
                    aperture_mm     = float(p.get("aperture_mm") or 0) or None,
                )
                self.finished_signal.emit(mode, res)

            elif mode == "Tilt Analysis":
                self.progress_signal.emit(self.tr("Detecting stars..."))
                norm_plane, plane_coefs, hw, fit_info = tilt_analysis(
                    self._arr,
                    pixel_size_um   = float(p["pixel_size_um"]),
                    focal_length_mm = float(p["focal_length_mm"]),
                    aperture_mm     = float(p["aperture_mm"]),
                    sigma_clip      = float(p.get("sigma_clip", 2.5)),
                    thresh_sigma    = float(p.get("snr_threshold", 3.0)),
                    target_stars    = int(p.get("target_stars", 1000)),
                )
                self.finished_signal.emit(mode, {
                    "plane_norm":  norm_plane,
                    "plane_coefs": plane_coefs,
                    "shape":       hw,
                    "fit_info":    fit_info,
                })

            else:
                self.finished_signal.emit(mode, None)

        except Exception as e:
            self.error_signal.emit(mode, str(e))
            self.finished_signal.emit(mode, None)


class ImagePeekerDialogPro(QDialog):
    def __init__(self, parent, document, settings):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Image Peeker"))
        self.setWindowFlag(Qt.WindowType.Window, True)
        import platform
        if platform.system() == "Darwin":
            self.setWindowFlag(Qt.WindowType.Tool, True)  
        import platform
        if platform.system() == "Darwin":
            self.setWindowFlag(Qt.WindowType.Tool, True)  
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setModal(False)
        try:
            self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        except Exception:
            pass  # older PyQt6 versions        
        self.document = self._coerce_doc(document)   # <- ensure we hold a real doc
        self.settings = settings
        self._persist_prefix = "image_peeker"
        self._geom_restored = False
        self._restoring_ui = True  # block saves during restore
        self._analysis_worker: Optional[_AnalysisWorker] = None
        # status / progress line
        self.status_lbl = QLabel("")
        self.status_lbl.setStyleSheet("color:#bbb;")


        self.params = QGroupBox(self.tr("Grid parameters"))
        self.params.setMinimumWidth(180)
        self.params.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        gl = QGridLayout(self.params)

        from PyQt6.QtWidgets import QSpinBox, QDoubleSpinBox
        self.grid_spin = QSpinBox(); self.grid_spin.setRange(2, 10); self.grid_spin.setValue(3)
        self.panel_slider = QSlider(Qt.Orientation.Horizontal); self.panel_slider.setRange(32, 512); self.panel_slider.setValue(256)
        self.panel_value_label = QLabel(str(self.panel_slider.value()))
        self.sep_slider = QSlider(Qt.Orientation.Horizontal); self.sep_slider.setRange(0, 50); self.sep_slider.setValue(4)
        self.sep_value_label = QLabel(str(self.sep_slider.value()))

        self.pixel_size_input = QDoubleSpinBox(); self.pixel_size_input.setRange(0.01, 50.0); self.pixel_size_input.setSingleStep(0.1)
        self.focal_length_input = QDoubleSpinBox(); self.focal_length_input.setRange(10.0, 5000.0); self.focal_length_input.setSingleStep(10.0)
        self.aperture_input = QDoubleSpinBox(); self.aperture_input.setRange(1.0, 5000.0); self.aperture_input.setSingleStep(1.0)

        px = self.settings.value("pixel_size_um", 4.8, type=float)
        fl = self.settings.value("focal_length_mm", 800.0, type=float)
        ap = self.settings.value("aperture_mm", 100.0, type=float)
        self.pixel_size_input.setValue(px); self.focal_length_input.setValue(fl); self.aperture_input.setValue(ap)

        row = 0
        gl.addWidget(QLabel(self.tr("Grid size:")), row, 0); gl.addWidget(self.grid_spin, row, 1); row += 1
        gl.addWidget(QLabel(self.tr("Panel size:")), row, 0)
        pr = QHBoxLayout(); pr.addWidget(self.panel_slider, 1); pr.addWidget(self.panel_value_label)
        gl.addLayout(pr, row, 1); row += 1
        gl.addWidget(QLabel(self.tr("Separation:")), row, 0)
        sr = QHBoxLayout(); sr.addWidget(self.sep_slider, 1); sr.addWidget(self.sep_value_label)
        gl.addLayout(sr, row, 1); row += 1
        gl.addWidget(QLabel(self.tr("Pixel size (µm):")), row, 0); gl.addWidget(self.pixel_size_input, row, 1); row += 1
        gl.addWidget(QLabel(self.tr("Focal length (mm):")), row, 0); gl.addWidget(self.focal_length_input, row, 1); row += 1
        gl.addWidget(QLabel(self.tr("Aperture (mm):")), row, 0); gl.addWidget(self.aperture_input, row, 1); row += 1

        # Right side
        from PyQt6.QtWidgets import QTabWidget
        self.preview_pane = PreviewPane()
        analysis_row = QHBoxLayout()
        analysis_row.addWidget(QLabel(self.tr("Analysis:")))
        self.analysis_combo = QComboBox()
        self.analysis_combo.addItem(self.tr("None"), "None")
        self.analysis_combo.addItem(self.tr("Tilt Analysis"), "Tilt Analysis")
        self.analysis_combo.addItem(self.tr("Focal Plane Analysis"), "Focal Plane Analysis")
        self.analysis_combo.addItem(self.tr("Astrometric Distortion Analysis"), "Astrometric Distortion Analysis")
        analysis_row.addWidget(self.analysis_combo)
        self.analyze_btn = QPushButton(self.tr("Analyze"))
        analysis_row.addWidget(self.analyze_btn)
        analysis_row.addStretch(1)
        analysis_row.addWidget(make_help_button("image_peeker", self))

        btns = QHBoxLayout(); btns.addStretch(1)
        ok_btn = QPushButton(self.tr("Save Settings && Exit")); cancel_btn = QPushButton(self.tr("Exit without Saving"))
        btns.addWidget(ok_btn); btns.addWidget(cancel_btn)

        main = QHBoxLayout(self)
        main.addWidget(self.params)
        right = QVBoxLayout(); right.addLayout(analysis_row); right.addWidget(self.status_lbl, 0), right.addWidget(self.preview_pane, 1); right.addLayout(btns)
        main.addLayout(right, 1)

        # Signals
        self.grid_spin.valueChanged.connect(self._refresh_mosaic)
        self.panel_slider.valueChanged.connect(lambda v: (self.panel_value_label.setText(str(v)), self._refresh_mosaic()))
        self.sep_slider.valueChanged.connect(lambda v: (self.sep_value_label.setText(str(v)), self._refresh_mosaic()))
        self.analyze_btn.clicked.connect(self._run_analysis)
        ok_btn.clicked.connect(self.accept); cancel_btn.clicked.connect(self.reject)
        def _save_after_change(*_):
            self._save_ui_state()

        self.grid_spin.valueChanged.connect(_save_after_change)
        self.panel_slider.sliderReleased.connect(_save_after_change)
        self.sep_slider.sliderReleased.connect(_save_after_change)
        self.analysis_combo.currentIndexChanged.connect(_save_after_change)

        self.pixel_size_input.editingFinished.connect(_save_after_change)
        self.focal_length_input.editingFinished.connect(_save_after_change)
        self.aperture_input.editingFinished.connect(_save_after_change)
        QTimer.singleShot(0, self._refresh_mosaic)

    def _k(self, key: str) -> str:
        return f"{self._persist_prefix}/{key}"

    def _save_ui_state(self):
        # Don't save until we've actually shown/restored once
        if getattr(self, "_restoring_ui", False) or not getattr(self, "_geom_restored", False):
            return
        try:
            s = self.settings  # <-- use your passed-in QSettings
            s.setValue(self._k("window_geometry"), self.saveGeometry())

            s.setValue(self._k("grid_n"), int(self.grid_spin.value()))
            s.setValue(self._k("panel_sz"), int(self.panel_slider.value()))
            s.setValue(self._k("sep_px"), int(self.sep_slider.value()))

            # analysis selection by data key (stable)
            s.setValue(self._k("analysis_key"), str(self.analysis_combo.currentData()))

            # your optics values are already saved in accept(), but saving here helps if user closes w/out accept
            s.setValue("pixel_size_um",   float(self.pixel_size_input.value()))
            s.setValue("focal_length_mm", float(self.focal_length_input.value()))
            s.setValue("aperture_mm",     float(self.aperture_input.value()))

            try:
                s.sync()
            except Exception:
                pass
        except Exception:
            pass

    def _restore_ui_state(self):
        self._restoring_ui = True
        try:
            s = self.settings

            g = s.value(self._k("window_geometry"), None)
            if g is not None:
                self.restoreGeometry(g)

            # grid controls
            n = int(s.value(self._k("grid_n"), 3))
            ps = int(s.value(self._k("panel_sz"), 256))
            sep = int(s.value(self._k("sep_px"), 4))

            self.grid_spin.blockSignals(True)
            self.panel_slider.blockSignals(True)
            self.sep_slider.blockSignals(True)
            try:
                self.grid_spin.setValue(max(2, min(10, n)))
                self.panel_slider.setValue(max(32, min(512, ps)))
                self.sep_slider.setValue(max(0, min(50, sep)))
            finally:
                self.grid_spin.blockSignals(False)
                self.panel_slider.blockSignals(False)
                self.sep_slider.blockSignals(False)

            # reflect labels
            try:
                self.panel_value_label.setText(str(self.panel_slider.value()))
                self.sep_value_label.setText(str(self.sep_slider.value()))
            except Exception:
                pass

            # analysis selection
            key = str(s.value(self._k("analysis_key"), "None"))
            # find by userData
            idx = -1
            for i in range(self.analysis_combo.count()):
                if str(self.analysis_combo.itemData(i)) == key:
                    idx = i
                    break
            if idx >= 0:
                self.analysis_combo.blockSignals(True)
                self.analysis_combo.setCurrentIndex(idx)
                self.analysis_combo.blockSignals(False)

        finally:
            self._restoring_ui = False

    def showEvent(self, ev):
        super().showEvent(ev)
        if not self._geom_restored:
            self._geom_restored = True
            QTimer.singleShot(0, lambda: (self._restore_ui_state(), self._refresh_mosaic()))

    def closeEvent(self, ev):
        try:
            self._save_ui_state()
        except Exception:
            pass
        # If a worker is still running, ask it to finish (or drop the handle if
        # it never will). We don't wait forever -- SEP + a plane fit is at most
        # a few seconds even on a large frame.
        worker = getattr(self, "_analysis_worker", None)
        if worker is not None:
            try:
                worker.wait(3000)   # ms
            except Exception:
                pass
            self._analysis_worker = None
        # Never leave the override cursor pushed if the dialog closes mid-run.
        try:
            self._set_busy_cursor(False)
        except Exception:
            pass
        super().closeEvent(ev)

    def _set_busy(self, on: bool, text: str = ""):
        if not text: text = self.tr("Processing…")
        self.status_lbl.setText(text if on else "")
        for w in (self.params, self.analysis_combo):
            w.setEnabled(not on)
        self._set_busy_cursor(on)
        QCoreApplication.processEvents()

    def _set_busy_cursor(self, on: bool):
        """
        Push/pop the application-wide busy cursor at most once.

        QGuiApplication's override cursor is a STACK, not a flag: every
        setOverrideCursor() must be matched by exactly one
        restoreOverrideCursor(). This used to push on every _set_busy(True),
        and the worker's progress_signal was wired to _set_busy(True, ...),
        so a successful Tilt / Focal Plane run pushed twice (launch, then
        "Detecting stars...") and popped once -- leaving the busy cursor on
        for the whole application while the dialog sat idle and responsive.
        Pairing against a flag also stops a stray _set_busy(False) from
        popping an override cursor this dialog never set.
        """
        active = bool(getattr(self, "_busy_cursor_active", False))
        if on and not active:
            QGuiApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            self._busy_cursor_active = True
        elif active and not on:
            QGuiApplication.restoreOverrideCursor()
            self._busy_cursor_active = False

    def _set_busy_text(self, text: str):
        """Progress updates: status label only, never the cursor stack."""
        self.status_lbl.setText(text)

    def _arr_and_meta(self):
        doc = self._coerce_doc(self.document)
        if doc is None or getattr(doc, "image", None) is None:
            return None, {}
        arr = np.asarray(doc.image, dtype=np.float32)
        meta = dict(getattr(doc, "metadata", {}) or {})
        return arr, meta

    def accept(self):
        self.settings.setValue("pixel_size_um",   self.pixel_size_input.value())
        self.settings.setValue("focal_length_mm", self.focal_length_input.value())
        self.settings.setValue("aperture_mm",     self.aperture_input.value())
        super().accept()


    def _run_analysis(self, *_):
        mode_key = self.analysis_combo.currentData()
        mode_disp = self.analysis_combo.currentText()
        if mode_key == "None":
            self._set_busy(False, "")
            self._refresh_mosaic()
            return
        # Guard against re-entrancy while a worker is running.
        if getattr(self, "_analysis_worker", None) is not None:
            return
        self._set_busy(True, self.tr("Running {0}…").format(mode_disp))
        # Give the busy cursor a chance to render before we spin up the worker.
        QTimer.singleShot(0, lambda: self._run_analysis_dispatch(mode_key))

    def _run_analysis_dispatch(self, mode: str):
        """
        Route analysis modes. SEP-based analyses (Focal Plane, Tilt) run in a
        background QThread so the GUI stays responsive; the plate-solve /
        distortion path stays synchronous because it involves modal solver
        dialogs and metadata mutation on the doc.
        """
        arr, meta = self._arr_and_meta()
        if arr is None or arr.size == 0:
            self._set_busy(False, "")
            return

        ps_um  = float(meta.get("pixel_size_um",   self.pixel_size_input.value()))
        fl_mm  = float(meta.get("focal_length_mm", self.focal_length_input.value()))
        ap_mm  = float(meta.get("aperture_mm",     self.aperture_input.value()))
        snr_th = float(meta.get("snr_threshold",   3.0))
        n_tgt  = int(meta.get("target_stars",      1000))

        if mode in ("Tilt Analysis", "Focal Plane Analysis"):
            params = {
                "pixel_size_um":   ps_um,
                "focal_length_mm": fl_mm,
                "aperture_mm":     ap_mm,
                "snr_threshold":   snr_th,
                "target_stars":    n_tgt,
                "deg":             3,
                "coarse":          128,
                "sigma_clip":      2.5,
            }
            # np.ascontiguousarray so the worker's SEP call doesn't hit a
            # non-contiguous view; float32 keeps SEP happy without a copy.
            arr_c = np.ascontiguousarray(arr, dtype=np.float32)
            worker = _AnalysisWorker(mode, arr_c, params, parent=self)
            self._analysis_worker = worker
            worker.progress_signal.connect(
                lambda t, m=mode: self._set_busy_text(f"{m}: {t}"))
            worker.error_signal.connect(self._on_analysis_error)
            worker.finished_signal.connect(self._on_analysis_finished)
            worker.start()
            return

        # ---- Synchronous path: Astrometric Distortion Analysis --------------
        try:
            if mode == "Astrometric Distortion Analysis":
                hdr = _header_from_meta(meta)

                # If we truly have no WCS, plate-solve
                if hdr is None or not WCS(hdr, naxis=2, relax=True).has_celestial:
                    ok, hdr_or_err = plate_solve_doc_inplace(
                        parent=self, doc=self._coerce_doc(self.document), settings=self.settings
                    )
                    if not ok:
                        QMessageBox.warning(self, self.tr("Plate Solve"),
                                            self.tr("ASTAP/Astrometry failed:\n{0}").format(hdr_or_err))
                        return

                    if isinstance(hdr_or_err, fits.Header):
                        doc = self._coerce_doc(self.document)
                        if doc and isinstance(getattr(doc, "metadata", None), dict):
                            doc.metadata["original_header"] = hdr_or_err
                            doc.metadata["wcs_header"] = hdr_or_err
                            try:
                                doc.metadata["wcs"] = WCS(hdr_or_err, relax=True)
                            except Exception:
                                pass

                    arr, meta = self._arr_and_meta()
                    hdr = _header_from_meta(meta)

                if hdr is None:
                    QMessageBox.critical(self, self.tr("WCS Error"),
                                         self.tr("Plate solve did not produce a readable WCS header."))
                    return

                has_sip = (any(k.startswith("A_") for k in hdr.keys())
                           and any(k.startswith("B_") for k in hdr.keys()))
                if not has_sip:
                    QMessageBox.warning(
                        self, self.tr("No Distortion Model"),
                        self.tr("This image has a valid WCS, but no SIP distortion terms (A_*, B_*).\n"
                                "Astrometric distortion analysis requires a SIP-enabled solve.\n\n"
                                "Re-solve with distortion fitting enabled in ASTAP.")
                    )
                    return

                asp = _arcsec_per_pix_from_header(hdr, fallback_px_um=ps_um, fallback_fl_mm=fl_mm)
                if asp is None:
                    QMessageBox.critical(self, self.tr("WCS Error"),
                                         self.tr("Cannot determine pixel scale."))
                    return

                DistortionGridDialog(
                    img=np.clip(arr, 0, 1), sip_meta=hdr, arcsec_per_pix=float(asp),
                    n_grid_lines=10, amplify=60.0,
                    document=self._coerce_doc(self.document),
                    main_window=self.parent(),
                    parent=self
                ).show()

            else:
                self._refresh_mosaic()
        finally:
            self._set_busy(False, "")

    def _on_analysis_finished(self, mode: str, res):
        """Handles a worker's finished_signal on the GUI thread."""
        try:
            if res is None:
                if mode == "Focal Plane Analysis":
                    QMessageBox.warning(
                        self, self.tr("Focal Plane Analysis"),
                        self.tr("Not enough well-shaped stars for a reliable "
                                "focal-plane fit. Try lowering the SNR threshold "
                                "or using a longer sub with more detections."))
                elif mode == "Tilt Analysis":
                    QMessageBox.warning(
                        self, self.tr("Tilt Analysis"),
                        self.tr("Not enough stars for a reliable tilt fit."))
                return

            if mode == "Focal Plane Analysis":
                # Verdict first — it is the answer; the maps are the evidence.
                if res.get("diagnosis"):
                    DiagnosisDialog(res["diagnosis"], parent=self).show()

                mn_f, mx_f = res["fwhm_range_um"]
                mn_e, mx_e = res["ell_range"]
                SurfaceDialog(
                    self.tr("FWHM Heatmap ({0} stars)").format(res["n_stars"]),
                    res["fwhm_norm"], mn_f, mx_f, "µm", "viridis", parent=self,
                    confidence=res.get("fwhm_conf")   # _SASPRO_MAP_CONFIDENCE
                ).show()
                SurfaceDialog(
                    self.tr("Ellipticity Map (1 − b/a)"),
                    res["ell_norm"], mn_e, mx_e, "", "magma", parent=self,
                    confidence=res.get("ell_conf")   # _SASPRO_MAP_CONFIDENCE
                ).show()

                # Orientation only when the stars support one. A map drawn
                # from round or undersampled stars looks exactly like a real
                # tilt signature, so showing nothing is the safer failure.
                if res.get("orientation_ok"):
                    mn_o, mx_o = res["theta_range_rad"]
                    SurfaceDialog(
                        self.tr("Orientation Map ({0} elongated stars, ±{1:.0f}°)")
                            .format(res.get("n_orient_stars", 0),
                                    res.get("orient_sigma_deg", float("nan"))),
                        res["theta_norm"], mn_o, mx_o,
                        self.tr("rad  (cyan=horizontal, +=clockwise on screen)"),
                        "hsv",
                        parent=self,
                        # _SASPRO_ROUNDNESS_ONLY: brightness from roundness alone.
                        # Driving it from the ellipticity map muddied the whole
                        # frame in HSV and washed the strongest region to white
                        # in HSL -- one channel cannot carry strength without
                        # taking it out of hue.
                        confidence=res.get("orient_conf"),
                        # _SASPRO_AXIS_TICKS: an axis drawn as an axis.
                        base_image=self._arr_and_meta()[0],
                        ticks={"theta": res.get("theta_surf"),
                               "emag": res.get("emag_surf"),
                               "zones": res.get("zones"),
                               "bins": res.get("axis_bins"),   # _SASPRO_LOCAL_AXES
                               "conf": res.get("orient_conf")}
                    ).show()
                else:
                    att = res.get("attrition", {}) or {}
                    under = att.get("rejected_undersampled", 0)
                    detail = ""
                    if under:
                        detail = self.tr(
                            "\n\n{0} detections were also dropped before this "
                            "as too small to measure a shape at all (below "
                            "~2 px FWHM, where the second moments track "
                            "subpixel position rather than the star)."
                        ).format(under)
                    QMessageBox.information(
                        self, self.tr("Orientation Map"),
                        self.tr(
                            "No orientation map — only {0} of {1} stars are "
                            "elongated enough for their position angle to "
                            "mean anything.\n\n"
                            "A star that is round within the noise has no "
                            "axis to report, so mapping its angle would draw "
                            "a smooth picture of nothing. Typical angular "
                            "uncertainty here is ±{2:.0f}°.\n\n"
                            "FWHM and ellipticity maps are still valid and "
                            "are shown — those do not need elongated stars."
                        ).format(res.get("n_orient_stars", 0),
                                 res.get("n_stars", 0),
                                 res.get("orient_sigma_deg", float("nan")))
                        + detail)

            elif mode == "Tilt Analysis":
                a, b, c = res["plane_coefs"]
                H, W = res["shape"]
                TiltDialog(self.tr("Sensor Tilt (µm)"),
                           res["plane_norm"], (a, b, c), (H, W),
                           float(self.pixel_size_input.value()),
                           fit_info=res.get("fit_info"),
                           parent=self).show()
        finally:
            self._set_busy(False, "")
            worker = getattr(self, "_analysis_worker", None)
            if worker is not None:
                worker.deleteLater()
                self._analysis_worker = None

    def _on_analysis_error(self, mode: str, message: str):
        self._set_busy(False, "")
        worker = getattr(self, "_analysis_worker", None)
        if worker is not None:
            worker.deleteLater()
            self._analysis_worker = None
        QMessageBox.critical(self, mode,
                             self.tr("Analysis failed:\n{0}").format(message))

    def _coerce_doc(self, obj):
        """Return a document object that has .image and .metadata, or None."""
        if obj is None:
            return None
        # If it already looks like a document
        if hasattr(obj, "image") and not isinstance(obj, QMdiSubWindow):
            return obj
        # If it's a subwindow, try its widget().document
        if isinstance(obj, QMdiSubWindow):
            w = obj.widget()
            return getattr(w, "document", None)
        # If it's a view-type wrapper
        if hasattr(obj, "document"):
            return getattr(obj, "document")
        return None


    def _on_panel_changed(self, v):
        self.panel_value_label.setText(str(v))
        self._refresh_mosaic()

    def _on_sep_changed(self, v):
        self.sep_value_label.setText(str(v))
        self._refresh_mosaic()

    def _update_sep_color_button(self):
        # show current color
        pix = QIcon().pixmap(16,16)
        pix.fill(self._sep_color)
        self.sep_color_btn.setIcon(QIcon(pix))

    def _choose_sep_color(self):
        col = QColorDialog.getColor(self._sep_color, self, self.tr("Choose separation color"))
        if col.isValid():
            self._sep_color = col
            self._update_sep_color_button()

    def _refresh_mosaic(self):
        arr, _ = self._arr_and_meta()          # this is the 32-bit float image
        if arr is None or arr.size == 0:
            return
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=2)
        arr = np.clip(arr, 0.0, 1.0).astype(np.float32, copy=False)

        n   = max(2, int(self.grid_spin.value()))
        ps  = int(self.panel_slider.value())
        sep = int(self.sep_slider.value())

        # tile in FLOAT so the stretch sees full precision
        mosaic_f = build_mosaic_numpy(arr, grid=n, panel=ps, sep=sep, background=0.0)

        qimg = self._to_qimage(mosaic_f)       # 8-bit, for immediate display only
        self.preview_pane.load_qimage(qimg, source_float=mosaic_f)

    def _on_ok(self):
        # user clicked OK → generate & display the mosaic
        n        = self.grid_spin.value()
        panel_sz = self.panel_slider.value()
        sep      = self.sep_slider.value()
        sep_col  = self._sep_color

        # fetch the currently loaded image (you’ll adapt to your image_manager API)
        img = self.image_manager.current_qimage()
        if img is None:
            QMessageBox.warning(self, self.tr("No image"), self.tr("No image loaded to peek at!"))
            return

        mosaic = self._build_mosaic(img, n, panel_sz, sep, sep_col)
        self.preview.setPixmap(QPixmap.fromImage(mosaic))
        # keep dialog open so user can tweak parameters

    def _build_mosaic(self, img, n, panel_sz, sep, sep_col):
        from PyQt6.QtGui import QImage
        W = n*panel_sz + (n-1)*sep; H = n*panel_sz + (n-1)*sep
        mosaic = QImage(W, H, img.format()); p = QPainter(mosaic)
        p.fillRect(0,0,W,H, sep_col)
        src_w, src_h = img.width(), img.height()
        xs = [int((src_w - panel_sz) * i / max(n-1, 1)) for i in range(n)]
        ys = [int((src_h - panel_sz) * j / max(n-1, 1)) for j in range(n)]
        for row, y in enumerate(ys):
            for col, x in enumerate(xs):
                patch = img.copy(x, y, panel_sz, panel_sz)
                dx = col * (panel_sz + sep); dy = row * (panel_sz + sep)
                p.drawImage(dx, dy, patch)
        p.end()
        return mosaic

    def _to_qimage(self, arr: np.ndarray):
        # same as your _to_qimage in the snippet
        if arr.dtype.kind == "f":
            arr8 = np.clip(arr * 255, 0, 255).astype(np.uint8)
        elif arr.dtype != np.uint8:
            arr8 = arr.astype(np.uint8)
        else:
            arr8 = arr
        h, w = arr8.shape[:2]
        buf = arr8.tobytes(); self._last_qimage_buffer = buf
        from PyQt6.QtGui import QImage
        if arr8.ndim == 2:
            return QImage(buf, w, h, w, QImage.Format.Format_Grayscale8)
        elif arr8.ndim == 3 and arr8.shape[2] == 3:
            return QImage(buf, w, h, 3*w, QImage.Format.Format_RGB888)
        raise ValueError(f"Unsupported array shape {arr.shape}")