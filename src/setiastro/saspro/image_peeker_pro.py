# pro/image_peeker_pro.py
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


def _detect_stars(gray: np.ndarray,
                  thresh_sigma: float = 5.0,
                  min_area: int = 5) -> Optional[dict]:
    """
    Run SEP once on `gray` and return a quality-filtered star catalogue.

    Cuts applied:
      - flag == 0                   (no saturation, no blending, no truncation)
      - npix >= min_area            (rejects hot pixels / single-pixel spikes)
      - a > 0, b > 0, b <= a        (well-formed shape ellipse)

    Returns None when nothing survives. Otherwise a dict of float64 arrays:
      x, y, a, b, theta, flux, and n (int, surviving count).
    """
    data = np.ascontiguousarray(gray, dtype=np.float32)
    bkg = sep.Background(data)
    try:
        stars = sep.extract(data - bkg.back(),
                            thresh=thresh_sigma,
                            err=bkg.globalrms)
    except Exception:
        return None
    if stars is None or len(stars) == 0:
        return None

    names = stars.dtype.names
    flag = stars['flag'] if 'flag' in names else np.zeros(len(stars), dtype=int)
    npix = stars['npix'] if 'npix' in names else np.full(len(stars), min_area + 1, dtype=int)
    a = stars['a']; b = stars['b']

    good = (flag == 0) & (npix >= min_area) & (a > 0) & (b > 0) & (b <= a)
    if not np.any(good):
        return None

    return {
        'x':     stars['x'][good].astype(np.float64, copy=False),
        'y':     stars['y'][good].astype(np.float64, copy=False),
        'a':     a[good].astype(np.float64, copy=False),
        'b':     b[good].astype(np.float64, copy=False),
        'theta': stars['theta'][good].astype(np.float64, copy=False),
        'flux':  (stars['flux'] if 'flux' in names
                  else np.zeros(int(good.sum())))[good].astype(np.float64, copy=False),
        'n':     int(good.sum()),
    }


def _detect_stars_waterfall(gray: np.ndarray,
                            sigma_ladder: Tuple[float, ...] = (100.0, 50.0, 25.0, 12.0, 6.0, 3.0),
                            target_count: int = 1000,
                            min_area: int = 5) -> Optional[dict]:
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

    The quality filter (flag==0, npix>=min_area, well-formed shape ellipse)
    is applied at each rung; the returned catalogue is what came out of the
    LAST rung run -- either the first rung to reach target_count, or the
    final rung at the bottom of the ladder if we never got there.
    Returns None if even sigma_ladder[-1] produced no usable stars.
    """
    data = np.ascontiguousarray(gray, dtype=np.float32)
    try:
        bkg = sep.Background(data)
    except Exception:
        return None
    sub = data - bkg.back()
    rms = bkg.globalrms

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

        names = stars.dtype.names
        flag  = stars['flag'] if 'flag' in names else np.zeros(len(stars), dtype=int)
        npix  = stars['npix'] if 'npix' in names else np.full(len(stars), min_area + 1, dtype=int)
        a = stars['a']; b = stars['b']

        good = (flag == 0) & (npix >= min_area) & (a > 0) & (b > 0) & (b <= a)
        n_good = int(good.sum())
        if n_good == 0:
            continue

        best = {
            'x':     stars['x'][good].astype(np.float64, copy=False),
            'y':     stars['y'][good].astype(np.float64, copy=False),
            'a':     a[good].astype(np.float64, copy=False),
            'b':     b[good].astype(np.float64, copy=False),
            'theta': stars['theta'][good].astype(np.float64, copy=False),
            'flux':  (stars['flux'] if 'flux' in names
                      else np.zeros(n_good))[good].astype(np.float64, copy=False),
            'n':     n_good,
            'sigma': float(sig),
        }
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
                            sigma_clip: float = 3.0):
    """
    Fit a 2D polynomial of degree `deg` to scattered (x, y, values), then
    render on a full (H, W) grid via coarse eval + bilinear upsample.

    Automatically lowers `deg` if there aren't enough samples to constrain
    the requested order (need (deg+1)(deg+2)/2 samples for a unique fit).

    Returns (surface_HxW_float32, (raw_min, raw_max)).
    """
    n = int(values.size)
    while deg > 1 and n < (deg + 1) * (deg + 2) // 2:
        deg -= 1
    sol, exps = fit_2d_poly(x, y, values, deg=deg, sigma_clip=sigma_clip)
    surf = _eval_poly_upsampled(sol, exps, H, W, coarse=coarse)
    return surf, (float(np.min(values)), float(np.max(values)))


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

    Returns None when there aren't enough stars for a 6-parameter fit, or a
    dict with:
        p, q         : defocus slopes in um / pixel
        r            : defocus offset at origin in um
        fwhm_baseline: sqrt(FWHM0^2) in um (0 if the fit implies negative)
        rms_residual : RMS residual of the FWHM^2 fit (um^2)
        n_used       : star count in the final (sigma-clipped) fit
        well_defined : True if the paraboloid model is physically sensible;
                       False when we fell back to a linear FWHM-vs-position
                       fit because the quadratic term was ill-behaved
    """
    n = int(x.size)
    if n < 8:                                # 6 params + a little slack
        return None

    z = (fwhm_um * fwhm_um).astype(np.float64, copy=False)
    A = np.vstack([
        np.ones_like(x, dtype=np.float64),
        x.astype(np.float64, copy=False),
        y.astype(np.float64, copy=False),
        (x * x).astype(np.float64, copy=False),
        (x * y).astype(np.float64, copy=False),
        (y * y).astype(np.float64, copy=False),
    ]).T                                     # shape (N, 6)

    mask = np.ones_like(z, dtype=bool)
    sol = None
    for _ in range(max_iter):
        n_before = int(mask.sum())
        if n_before < 8:
            break
        sol, *_ = np.linalg.lstsq(A[mask], z[mask], rcond=None)
        resid = z - A.dot(sol)
        sd = float(np.std(resid[mask]))
        if sd <= 0.0:
            break
        newm = np.abs(resid) < sigma_clip * sd
        if int(newm.sum()) == n_before:
            break
        mask = newm

    if sol is None:
        return None

    c00, c10, c01, c20, c11, c02 = [float(v) for v in sol]
    N = float(f_number)
    k2 = 1.0 / (N * N)

    # Well-defined paraboloid: both quadratic-axis coefficients must be
    # positive (FWHM^2 is bounded below and grows away from best focus).
    # Small negative values from noise are treated as zero.
    c20_pos = max(c20, 0.0)
    c02_pos = max(c02, 0.0)
    well_defined = (c20 > 0.0) and (c02 > 0.0)

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
        return {
            'p': p_fb, 'q': q_fb, 'r': max(r_fb, 0.0),
            'fwhm_baseline': float(np.median(fwhm_um[mask])),
            'rms_residual': rms_fb,
            'n_used': int(mask.sum()),
            'well_defined': False,
        }

    p = math.sqrt(c20_pos) * N
    q = math.sqrt(c02_pos) * N
    # sign(p*q) from cross term
    if c11 < 0.0:
        q = -q
    # sign(p*r) or sign(q*r) from linear terms
    if abs(p) > 1e-12:
        r = c10 / (2.0 * k2 * p)
    elif abs(q) > 1e-12:
        r = c01 / (2.0 * k2 * q)
    else:
        r = 0.0
    # Convention: pin r >= 0 (overall sign flip is physically ambiguous).
    if r < 0.0:
        p, q, r = -p, -q, -r

    A_baseline = c00 - k2 * r * r
    fwhm_baseline = math.sqrt(A_baseline) if A_baseline > 0.0 else 0.0
    rms = float(np.sqrt(np.mean((A.dot(sol) - z)[mask] ** 2)))

    return {
        'p': float(p),
        'q': float(q),
        'r': float(r),
        'fwhm_baseline': float(fwhm_baseline),
        'rms_residual': rms,
        'n_used': int(mask.sum()),
        'well_defined': True,
    }


def analyze_focal_plane(img: np.ndarray,
                        pixel_scale: float,
                        thresh_sigma: float = 5.0,
                        deg: int = 3,
                        coarse: int = 128,
                        min_stars: int = 12,
                        target_stars: int = 1000) -> Optional[dict]:
    """
    Single-pass focal-plane analysis. Runs a descending-threshold SEP
    detection ONCE (background estimate + repeated extract) and fits three
    smooth surfaces from the shared catalogue:

      - FWHM (um)      -- proper Gaussian conversion: 2.3548 * sqrt(a*b) * ps
      - Ellipticity    -- 1 - b/a (matches PixInsight / SExtractor convention)
      - Orientation    -- circular fit on (sin 2th, cos 2th), recovered via atan2

    Detection uses the waterfall (100, 50, 25, 12, 6, 3 sigma) and stops as
    soon as `target_stars` well-formed stars have survived quality filtering.
    `thresh_sigma` is retained for API compatibility but is only used as the
    floor of the ladder when it exceeds 3.

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
    cat = _detect_stars_waterfall(gray, sigma_ladder=ladder, target_count=target_stars)
    if cat is None or cat['n'] < min_stars:
        return None

    x, y = cat['x'], cat['y']
    a, b = cat['a'], cat['b']
    theta = cat['theta']

    # ----- FWHM (geometric-mean axis FWHM, correct Gaussian conversion) -----
    fwhm_um = _FWHM_FACTOR * np.sqrt(a * b) * float(pixel_scale)
    fwhm_surf, fwhm_range = _fit_and_render_surface(x, y, fwhm_um, H, W,
                                                    deg=deg, coarse=coarse)

    # ----- Ellipticity (canonical: 1 - b/a in [0, 1)) -----------------------
    ell = 1.0 - (b / a)
    ell_surf, ell_range = _fit_and_render_surface(x, y, ell, H, W,
                                                  deg=deg, coarse=coarse)

    # ----- Orientation via circular fit -------------------------------------
    # Fit sin(2th) and cos(2th) as independent smooth surfaces, then recover
    # th_fit = 0.5 * atan2(surf_s, surf_c). Uses lower degree because pure
    # sensor tilt / decentre produces a slowly-varying orientation field.
    ori_deg = max(1, min(deg, 2))
    s = np.sin(2.0 * theta)
    c = np.cos(2.0 * theta)
    surf_s, _ = _fit_and_render_surface(x, y, s, H, W, deg=ori_deg, coarse=coarse)
    surf_c, _ = _fit_and_render_surface(x, y, c, H, W, deg=ori_deg, coarse=coarse)
    theta_surf = 0.5 * np.arctan2(surf_s, surf_c)   # in [-pi/2, pi/2]

    # Normalise each surface to [0, 1] for the SurfaceDialog display.
    def _norm(z: np.ndarray) -> np.ndarray:
        zmn = float(z.min()); zmx = float(z.max())
        rng = zmx - zmn
        return (z - zmn) / (rng if rng > 1e-9 else 1.0)

    return {
        'n_stars':        cat['n'],
        'fwhm_norm':      _norm(fwhm_surf),
        'fwhm_range_um':  fwhm_range,
        'ell_norm':       _norm(ell_surf),
        'ell_range':      ell_range,
        'theta_norm':     (theta_surf + np.pi / 2.0) / np.pi,   # [0, 1] hue
        'theta_range_rad': (float(theta.min()), float(theta.max())),
        # Raw per-star metrics kept for potential overlay drawing.
        'stars': cat,
    }


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
    2) Compute per‐star FWHM (≈2*a), eccentricity, and orientation theta.
    3) Bin the FWHM into a grid×grid mosaic (median per cell) → FWHM_um heatmap.
    4) Normalize that heatmap to [0..1] for display.
    5) Build an overlay list of (x_pix,y_pix,angle_rad,elongation_um) for each star.
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

    x, y = objs['x'], objs['y']
    a, b, theta = objs['a'], objs['b'], objs['theta']

    # FWHM ≈ 2 * a  (in pixels) → µm
    fwhm_um = 2.0 * a * pixel_scale

    # eccentricity → elongation factor e = a/b - 1
    e = np.clip(a / np.where(b>0, b, 1.0) - 1.0, 0.0, None)
    elongation_um = e * pixel_scale

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

    # --- build elongation‐arrow overlays ---
    overlays: List[Tuple[int,int,float,float]] = []
    for xi, yi, ang, el in zip(x, y, theta, elongation_um):
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
    fit = _fit_tilt_paraboloid(x, y, fwhm_um, f_number,
                               sigma_clip=sigma_clip, max_iter=5)
    if fit is None:
        return np.zeros((H, W), dtype=np.float32), (0.0, 0.0, 0.0), (H, W), None

    p, q, r = fit['p'], fit['q'], fit['r']

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
            rows = []
            corner_deltas = []
            for name, x, y in corners:
                delta = a * (x - cx) + b * (y - cy)   # relative to centre plane
                corner_deltas.append(delta)
                rows.append((name, f"{delta:.1f}"))

            min_d, max_d = min(corner_deltas), max(corner_deltas)
            span = max_d - min_d
            range_label = QLabel(self.tr(
                "Tilt span (corner Δ vs centre): {0:.1f} µm … {1:.1f} µm   "
                "|  peak-to-peak {2:.1f} µm"
            ).format(min_d, max_d, span))

            table = QTableWidget(len(rows), 2, self)
            table.setHorizontalHeaderLabels([self.tr("Corner"), self.tr("Δ µm")])
            table.verticalHeader().setVisible(False)
            for i, (name, val) in enumerate(rows):
                table.setItem(i, 0, QTableWidgetItem(name))
                table.setItem(i, 1, QTableWidgetItem(val))
            table.resizeColumnsToContents()

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
                    diag_bits.append(self.tr("RMS residual: {0:.2f} µm²")
                                     .format(float(fit_info["rms_residual"])))
                if diag_bits:
                    diag_label = QLabel("   |   ".join(diag_bits))

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



def fit_2d_poly(x, y, z, deg=2, sigma_clip=3.0, max_iter=3):
    """
    Fit z(x,y) = Σ_{i+j≤deg} c_{ij} x^i y^j
    by linear least squares + sigma-clipping.
    Returns the flattened coeff array.
    """
    # Build list of (i,j) exponents
    exps = [(i, j) for total in range(deg+1)
                  for i in range(total+1)
                  for j in [total - i]]
    # Design matrix
    A = np.vstack([ (x**i)*(y**j) for (i,j) in exps ]).T  # shape (N,len(exps))
    mask = np.ones_like(z, bool)

    for _ in range(max_iter):
        sol, *_ = np.linalg.lstsq(A[mask], z[mask], rcond=None)
        zfit    = A.dot(sol)
        resid   = z - zfit
        std     = np.std(resid[mask])
        newm    = np.abs(resid) < sigma_clip*std
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
    surf, raw = _fit_and_render_surface(cat['x'], cat['y'], fwhm_um, H, W, deg=deg)
    mn, mx = float(surf.min()), float(surf.max())
    heat = (surf - mn) / max(mx - mn, 1e-9)
    return heat.astype(np.float32, copy=False), raw


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
    surf, raw = _fit_and_render_surface(cat['x'], cat['y'], ell, H, W, deg=deg)
    mn, mx = float(surf.min()), float(surf.max())
    heat = (surf - mn) / max(mx - mn, 1e-9)
    return heat.astype(np.float32, copy=False), raw


def compute_orientation_surface(
    img: np.ndarray,
    thresh_sigma: float = 5.0,
    deg: int = 1,            # for pure tilt a plane is enough
    sigma_clip: float = 3.0,
    max_iter: int = 3
) -> Tuple[np.ndarray, Tuple[float,float]]:
    """
    Fits a smooth orientation surface theta(x,y) via a circular fit on the
    double-angle representation (sin 2th, cos 2th) so that theta and theta+pi
    are treated identically. Returns hue in [0, 1] and the raw theta range
    (radians) at the detection sites for the legend.
    """
    gray = _to_gray32(img)
    H, W = gray.shape
    cat = _detect_stars(gray, thresh_sigma=thresh_sigma)
    if cat is None or cat['n'] < 6:
        return np.zeros((H, W), dtype=np.float32), (0.0, 0.0)

    theta = cat['theta']
    s = np.sin(2.0 * theta)
    c = np.cos(2.0 * theta)
    surf_s, _ = _fit_and_render_surface(cat['x'], cat['y'], s, H, W,
                                        deg=max(1, deg), sigma_clip=sigma_clip)
    surf_c, _ = _fit_and_render_surface(cat['x'], cat['y'], c, H, W,
                                        deg=max(1, deg), sigma_clip=sigma_clip)
    theta_fit = 0.5 * np.arctan2(surf_s, surf_c)          # [-pi/2, pi/2]
    hue = (theta_fit + np.pi / 2.0) / np.pi               # [0, 1]
    return hue.astype(np.float32, copy=False), (float(theta.min()), float(theta.max()))

class SurfaceDialog(QDialog):
    def __init__(self, title, heatmap, vmin, vmax, units:str="", cmap="gray", parent=None):
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
            cmap_obj = cm.get_cmap(cmap)
            rgba = cmap_obj(heatmap)            # shape H×W×4, floats 0–1
            rgb  = (rgba[...,:3] * 255).astype(np.uint8)
            view.load_numpy(rgb)
        else:
            # assume already float32 [0..1] RGB or uint8 RGB
            view.load_numpy(heatmap)
        view.fit_to_view()

        # colorbar pixmap
        cb = self._make_colorbar(cmap, vmin, vmax, units)
        lbl_cb = QLabel()
        lbl_cb.setPixmap(cb)

        # layout
        h = QHBoxLayout()
        h.addWidget(view, 1)
        h.addWidget(lbl_cb, 0)

        btn = QPushButton(self.tr("Close"))
        btn.clicked.connect(self.accept)
        lbl_span = QLabel(self.tr("Span: {0:.2f} … {1:.2f} {2}").format(vmin, vmax, units))

        v = QVBoxLayout(self)
        v.addLayout(h)
        v.addWidget(lbl_span)
        v.addWidget(btn)

    def _make_colorbar(self, cmap_name, vmin, vmax, units):
        # build a 256×20 gradient in RGBA
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib import cm
        grad = np.linspace(0,1,256)[:,None]
        bar  = cm.get_cmap(cmap_name)(grad)
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
        super().closeEvent(ev)

    def _set_busy(self, on: bool, text: str = ""):
        if not text: text = self.tr("Processing…")
        self.status_lbl.setText(text if on else "")
        for w in (self.params, self.analysis_combo):
            w.setEnabled(not on)
        QGuiApplication.setOverrideCursor(Qt.CursorShape.WaitCursor) if on else QGuiApplication.restoreOverrideCursor()
        QCoreApplication.processEvents()

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
                lambda t, m=mode: self._set_busy(True, f"{m}: {t}"))
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
                mn_f, mx_f = res["fwhm_range_um"]
                mn_e, mx_e = res["ell_range"]
                mn_o, mx_o = res["theta_range_rad"]
                SurfaceDialog(
                    self.tr("FWHM Heatmap ({0} stars)").format(res["n_stars"]),
                    res["fwhm_norm"], mn_f, mx_f, "µm", "viridis", parent=self
                ).show()
                SurfaceDialog(
                    self.tr("Ellipticity Map (1 − b/a)"),
                    res["ell_norm"], mn_e, mx_e, "", "magma", parent=self
                ).show()
                SurfaceDialog(
                    self.tr("Orientation Map"),
                    res["theta_norm"], mn_o, mx_o, "rad", "hsv", parent=self
                ).show()

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