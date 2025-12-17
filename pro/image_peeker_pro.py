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
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QPoint, QEvent, QPointF, QCoreApplication
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

import sep
from pro.widgets.themed_buttons import themed_toolbtn

# bring in your existing helpers/classes from the snippet you posted
# (we assume they live next to this file or already in pro/)
from .plate_solver import plate_solve_doc_inplace
from imageops.stretch import stretch_mono_image, stretch_color_image
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

        self.zoom_in_btn  = themed_toolbtn("zoom-in", "Zoom In")
        self.zoom_out_btn = themed_toolbtn("zoom-out", "Zoom Out")
        self.fit_btn      = themed_toolbtn("zoom-fit-best", "Fit to Preview")
        self.zoom_in_btn.clicked .connect(lambda: self.adjust_zoom(10))
        self.zoom_out_btn.clicked.connect(lambda: self.adjust_zoom(-10))
        self.fit_btn.clicked .connect(self.fit_to_view)        

        self.stretch_btn  = QPushButton("AutoStretch")
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

    def load_qimage(self, img: QImage):
        """
        Call this to (re)load a fresh image.
        We immediately convert it to a numpy array once
        so we never have to touch the QImage bits again.
        """
        # keep a local copy of the QImage (for fast redisplay)
        self.original_image   = img.copy()

        # one & only time we go QImage→numpy
        self._image_array     = self.qimage_to_numpy(self.original_image)

        # reset any existing stretch state
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
                    apply_curves=False
                )
            else:
                stretched = stretch_color_image(
                    arr,
                    target_median=0.25,
                    linked=False,
                    normalize=True,
                    apply_curves=False
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
        """
        Convenience wrapper: take an H×W or H×W×3 NumPy array (float in [0..1] or uint8),
        convert it to a QImage and display.
        """
        # Convert to QImage
        qimg = self.numpy_to_qimage(arr)
        # Delegate to your existing loader
        self.load_qimage(qimg)


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
    sigma_clip: float = 2.0,
    thresh_sigma: float = 5.0,
) -> Tuple[np.ndarray, Tuple[float,float,float], Tuple[int,int]]:
    """
    Robust sensor‐tilt measurement via direct plane fit, with a thin‐lens defocus model.

    1) Convert to 2-D luminance if needed.
    2) Detect stars & measure half-light radius via SEP → rad (pixels).
    3) Compute blur diameter d_um = 2*a_px * pixel_size_um.
    4) Convert blur → defocus via thin‐lens: Δz_um = d_um * (focal_length_mm / aperture_mm).
    5) Fit plane Δz = a x + b y + c to all stars (sigma‐clipped).
    6) Return that best‐fit plane evaluated over every pixel, normalized 0–1 for display.
    """
    # 0) grayscale float32
    if img.ndim == 3 and img.shape[2] == 3:
        gray = (0.2126*img[...,0] + 0.7152*img[...,1] + 0.0722*img[...,2]).astype(np.float32)
    else:
        gray = img.astype(np.float32)
    H, W = gray.shape

    # 1) SEP star detection
    data = np.ascontiguousarray(gray, dtype=np.float32)
    bkg  = sep.Background(data)
    stars = sep.extract(data - bkg.back(),
                        thresh=thresh_sigma,
                        err=bkg.globalrms)
    if stars is None or len(stars) < 10:
        return np.zeros((H,W), dtype=float), (0.0,0.0,0.0), (H,W)

    x     = stars['x']
    y     = stars['y']
    a_pix = stars['a']   # semi-major axis
    flags = stars['flag'] if 'flag' in stars.dtype.names else np.zeros_like(a_pix, dtype=int)

    # 2) map to defocus distance (µm) via thin-lens:
    #    blur diameter ≈ 2*a_pix * px_size_um
    #    Δz_um = blur_um * (focal_length_mm / aperture_mm)
    blur_um     = 2.0 * a_pix * pixel_size_um
    f_number    = focal_length_mm / aperture_mm
    defocus_um  = blur_um * f_number

    # 3) initial least‐squares plane fit
    A     = np.vstack([x, y, np.ones_like(x)]).T  # (N,3)
    sol, *_ = np.linalg.lstsq(A, defocus_um, rcond=None)
    a, b, c = sol

    # 4) sigma‐clip outliers and re-fit
    z_pred = A.dot(sol)
    resid  = defocus_um - z_pred
    mask   = np.abs(resid) < sigma_clip * np.std(resid)
    if mask.sum() > 10:
        sol, *_ = np.linalg.lstsq(A[mask], defocus_um[mask], rcond=None)
        a, b, c = sol

    # 5) build full‐frame plane
    Y, X      = np.mgrid[0:H, 0:W]
    plane_full = a*X + b*Y + c

    # 6) normalize to [0..1] for display
    pmin, pmax = plane_full.min(), plane_full.max()
    if pmax > pmin:
        norm_plane = (plane_full - pmin) / (pmax - pmin)
    else:
        norm_plane = np.zeros_like(plane_full)

    return norm_plane, (a, b, c), (H, W)

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
from pro.widgets.spinboxes import CustomSpinBox, CustomDoubleSpinBox


class TiltDialog(QDialog):
    def __init__(self,
                 title: str,
                 img: np.ndarray,
                 plane: Optional[Tuple[float,float,float]] = None,
                 img_shape: Optional[Tuple[int,int]]    = None,
                 pixel_size_um: float                   = 1.0,
                 overlays: Optional[List[Tuple]]        = None,
                 parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.pixel_size_um = pixel_size_um

        # ––––– Create the view and load the image –––––
        self.view = PreviewPane()
        self.view.load_numpy(img)
        if overlays:
            self.view.set_overlay(overlays)

        # ––––– Corner tilt table –––––
        table = None
        if plane and img_shape:
            a, b, c = plane
            H, W = img_shape
            cx, cy = W/2, H/2
            corners = {
                "Top Left":    (0,   0),
                "Top Right":   (W,   0),
                "Bottom Left": (0,   H),
                "Bottom Right":(W,   H),
            }
            rows = []
            corner_deltas = []
            for name,(x,y) in corners.items():
                delta = a*(x - cx) + b*(y - cy)
                corner_deltas.append(delta)

            min_d, max_d = min(corner_deltas), max(corner_deltas)

            # 2) now build a more meaningful label:
            range_label = QLabel(f"Tilt span: {min_d:.1f} µm … {max_d:.1f} µm")            
            for name, (x, y) in corners.items():
                # how far above/below the center plane
                delta = a*(x - cx) + b*(y - cy)
                rows.append((name, f"{delta:.1f}"))

            table = QTableWidget(len(rows), 2, self)
            table.setHorizontalHeaderLabels(["Corner", "Δ µm"])
            # hide the vertical header
            table.verticalHeader().setVisible(False)
            for i, (name, val) in enumerate(rows):
                table.setItem(i, 0, QTableWidgetItem(name))
                table.setItem(i, 1, QTableWidgetItem(val))
            table.resizeColumnsToContents()

        # ––––– Layout everything –––––
        layout = QVBoxLayout(self)
        layout.addWidget(self.view,        1)  # stretch = 1
        layout.addWidget(range_label,     0)  # stretch = 0
        if table:
            layout.addWidget(table,       0)
        close_btn = QPushButton("Close", self)
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn,       0)

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
    # grayscale
    gray = img.mean(axis=2).astype(np.float32) if img.ndim==3 else img.astype(np.float32)
    H, W = gray.shape
    data = np.ascontiguousarray(gray, np.float32)
    bkg  = sep.Background(data)
    stars = sep.extract(data - bkg.back(), thresh=thresh_sigma, err=bkg.globalrms)
    if stars is None or len(stars)<10:
        return np.zeros((H,W), float)

    x = stars['x']; y = stars['y']
    fwhm_um = 2.0 * stars['a'] * pixel_scale

    # 1) fit
    sol, exps = fit_2d_poly(x, y, fwhm_um, deg=deg)

    # 2) evaluate
    Y, X     = np.mgrid[0:H, 0:W]
    surf     = eval_2d_poly(sol, exps, X, Y)

    # 3) normalize
    mn, mx = surf.min(), surf.max()
    heat = (surf - mn)/max(mx-mn,1e-9)
    return heat, (mn, mx)


def compute_eccentricity_surface(
    img: np.ndarray,
    pixel_scale: float,
    thresh_sigma: float = 5.0,
    deg: int = 3
) -> Tuple[np.ndarray, Tuple[float,float]]:
    """
    1) SEP → x,y,a,b
    2) e = clip(1 - b/a)
    3) Fit e(x,y) with a 2D poly of degree 'deg' + sigma-clip
    4) Evaluate on full H×W grid, normalize to [0..1]
    """
    gray = img.mean(axis=2).astype(np.float32) if img.ndim==3 else img.astype(np.float32)
    H, W = gray.shape
    data = np.ascontiguousarray(gray, np.float32)
    bkg  = sep.Background(data)
    stars = sep.extract(data - bkg.back(), thresh=thresh_sigma, err=bkg.globalrms)
    if stars is None or len(stars)<6:
        return np.zeros((H,W),dtype=float), (0.0, 0.0)

    x = stars['x']; y = stars['y']
    a = stars['a']; b = stars['b']
    e = np.clip(1.0 - b/a, 0.0, 1.0)
    e_min, e_max = float(e.min()), float(e.max())

    # fit polynomial
    sol, exps = fit_2d_poly(x, y, e, deg=deg)
    Y, X = np.mgrid[0:H,0:W]
    surf = eval_2d_poly(sol, exps, X, Y)

    mn, mx = surf.min(), surf.max()
    norm = (surf - mn)/max(mx-mn,1e-9)
    return norm, (e_min, e_max)


def compute_orientation_surface(
    img: np.ndarray,
    thresh_sigma: float = 5.0,
    deg: int = 1,            # for pure tilt a plane is enough
    sigma_clip: float = 3.0,
    max_iter: int = 3
) -> Tuple[np.ndarray, Tuple[float,float]]:
    """
    Fits a smooth orientation surface θ(x,y) via circular least squares.

    Returns
    -------
    norm_hue : H×W array
      Hue = (θ_fit + π/2)/π in [0..1], ready for display.
    (h_min, h_max) :
      min/max of the raw hue samples at star positions.
    """
    # → 1) make a 2D grayscale
    gray = img.mean(axis=2).astype(np.float32) if img.ndim==3 else img.astype(np.float32)
    H, W = gray.shape

    # → 2) SEP detect
    data = np.ascontiguousarray(gray, np.float32)
    bkg  = sep.Background(data)
    stars = sep.extract(data - bkg.back(), thresh=thresh_sigma, err=bkg.globalrms)
    if stars is None or len(stars) < 6:
        return np.zeros((H, W), dtype=float), (0.0, 0.0)

    x     = stars['x']
    y     = stars['y']
    theta = stars['theta']  # in radians

    # → 3) form double‐angle sine/cosine
    s = np.sin(2*theta)
    c = np.cos(2*theta)

    # compute raw hue range for legend
    # compute **actual** θ range for legend (in radians)
    theta_min, theta_max = float(theta.min()), float(theta.max())

    # → 4) build design matrix for deg‐th 2D poly
    exps = [(i,j) for total in range(deg+1)
                  for i in range(total+1)
                  for j in [total-i]]
    A    = np.vstack([ (x**i)*(y**j) for (i,j) in exps ]).T  # shape (N, M)

    # → 5) sigma‐clip loops on residual length
    mask = np.ones_like(s, bool)
    for _ in range(max_iter):
        sol_s, *_ = np.linalg.lstsq(A[mask], s[mask], rcond=None)
        sol_c, *_ = np.linalg.lstsq(A[mask], c[mask], rcond=None)
        fit_s = A.dot(sol_s)
        fit_c = A.dot(sol_c)
        resid = np.hypot(s - fit_s, c - fit_c)
        std   = np.std(resid[mask])
        newm  = resid < sigma_clip*std
        if newm.sum() == mask.sum():
            break
        mask = newm

    # → 6) evaluate both polys on the full image grid
    Y, X = np.mgrid[0:H, 0:W]
    surf_s = sum(coeff*(X**i)*(Y**j) for coeff,(i,j) in zip(sol_s, exps))
    surf_c = sum(coeff*(X**i)*(Y**j) for coeff,(i,j) in zip(sol_c, exps))

    # → 7) recover the smooth θ_fit and map to hue [0..1]
    theta_fit = 0.5 * np.arctan2(surf_s, surf_c)       # in [−π/2..π/2]
    hue       = (theta_fit + np.pi/2) / np.pi          # now [0..1]

    return hue, (theta_min, theta_max)

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

        btn = QPushButton("Close")
        btn.clicked.connect(self.accept)
        lbl_span = QLabel(f"Span: {vmin:.2f} … {vmax:.2f} {units}")

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
                parent=None):
        super().__init__(parent)
        self.setWindowTitle("Astrometric Distortion & Histogram")

        # — 1) detect stars —
        gray = img.mean(-1).astype(np.float32) if img.ndim==3 else img.astype(np.float32)
        data = np.ascontiguousarray(gray, np.float32)
        bkg  = sep.Background(data)
        stars = sep.extract(data - bkg.back(), thresh=5.0, err=bkg.globalrms)
        if stars is None or len(stars) < 10:
            QMessageBox.warning(self, "Distortion", "Not enough stars found.")
            self.reject()
            return

        x_pix = stars['x']
        y_pix = stars['y']

        # — 2) extract SIP A,B and reference pixel from metadata dict —
        A, B, crpix1, crpix2 = extract_sip_from_meta(sip_meta)

        # — 4) per-star residuals in pixels → arc-sec —
        u_star = x_pix - crpix1
        v_star = y_pix - crpix2
        dx_star_pix, dy_star_pix = eval_sip(A, B, u_star, v_star)
        disp_star_pix    = np.hypot(dx_star_pix, dy_star_pix)
        disp_star_arcsec = disp_star_pix * arcsec_per_pix

        # — 5) full‐image warp maps (pixels) for drawing grid —
        H, W = data.shape
        YY, XX = np.mgrid[0:H, 0:W]
        U = XX - crpix1
        V = YY - crpix2
        DX_pix, DY_pix = eval_sip(A, B, U, V)
        DX = DX_pix * amplify
        DY = DY_pix * amplify

        # — 6) build the distortion grid scene —
        scene = QGraphicsScene(self)
        scene.setBackgroundBrush(QColor(30,30,30))
        pen  = QPen(QColor(255,100,100), 1)
        label_font = QFont("Arial", 12, QFont.Weight.Bold)

        # title above the grid
        title = QLabel("Astrometric Distortion Grid")
        title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("color: white;")

        # draw horizontal + vertical lines
        for i in range(n_grid_lines+1):
            y0  = i*(H-1)/n_grid_lines
            xs  = np.linspace(0, W-1, 200)
            ys  = np.full_like(xs, y0)
            xi  = np.clip(xs.astype(int), 0, W-1)
            yi  = np.clip(ys.astype(int), 0, H-1)
            warped = np.column_stack([ xs + DX[yi,xi], ys + DY[yi,xi] ])
            path = QPainterPath(QPointF(*warped[0]))
            for px,py in warped[1:]:
                path.lineTo(QPointF(px,py))
            scene.addPath(path, pen)

        for j in range(n_grid_lines+1):
            x0  = j*(W-1)/n_grid_lines
            ys  = np.linspace(0, H-1, 200)
            xs  = np.full_like(ys, x0)
            xi  = np.clip(xs.astype(int), 0, W-1)
            yi  = np.clip(ys.astype(int), 0, H-1)
            warped = np.column_stack([ xs + DX[yi,xi], ys + DY[yi,xi] ])
            path = QPainterPath(QPointF(*warped[0]))
            for px,py in warped[1:]:
                path.lineTo(QPointF(px,py))
            scene.addPath(path, pen)

        # annotate each grid‐intersection
        for i in range(n_grid_lines+1):
            for j in range(n_grid_lines+1):
                y0 = i*(H-1)/n_grid_lines
                x0 = j*(W-1)/n_grid_lines
                xi, yi = int(round(x0)), int(round(y0))

                # local distortion in pixels → arcsec
                d_pix    = math.hypot(DX_pix[yi, xi], DY_pix[yi, xi])
                d_arcsec = d_pix * arcsec_per_pix

                px = x0 + DX[yi, xi]
                py = y0 + DY[yi, xi]

                txt = QGraphicsTextItem(f"{d_arcsec:.1f}\"")
                txt.setFont(label_font)
                txt.setScale(5.0)
                txt.setDefaultTextColor(QColor(200,200,200))
                txt.setPos(px + 4, py + 4)
                scene.addItem(txt)

        view = QGraphicsView(scene)
        view.setRenderHint(QPainter.RenderHint.Antialiasing)
        view.fitInView(scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

        # pack title + view vertically
        left_layout = QVBoxLayout()
        left_layout.addWidget(title)
        left_layout.addWidget(view, 1)

        # — 7) histogram of per-star residuals (arcsec) —
        fig    = Figure(figsize=(4,4))
        canvas = FigureCanvas(fig)
        ax     = fig.add_subplot(111)
        ax.hist(disp_star_arcsec, bins=30, edgecolor='black')
        ax.set_xlabel("Distortion (″)")
        ax.set_ylabel("Number of stars")
        ax.set_title("Residual histogram")
        fig.tight_layout()

        # side-by-side layout
        hl = QHBoxLayout()
        hl.addLayout(left_layout, 1)
        hl.addWidget(canvas, 1)

        # close button
        btn = QPushButton("Close")
        btn.clicked.connect(self.accept)

        # final
        v = QVBoxLayout(self)
        v.addLayout(hl)
        v.addWidget(btn, 0)

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
    """Try CDELT-based scale; fallback to CD matrix; then pixel_size & focal length."""
    if hdr is None:
        if fallback_px_um and fallback_fl_mm:
            return 206.264806 * (fallback_px_um / fallback_fl_mm)
        return None
    try:
        return abs(float(hdr["CDELT1"])) * 3600.0
    except Exception:
        try:
            cd11 = float(hdr["CD1_1"])
            cd12 = float(hdr.get("CD1_2", 0.0))
            cd21 = float(hdr.get("CD2_1", 0.0))
            cd22 = float(hdr["CD2_2"])
            scale_deg = np.sqrt(abs(cd11 * cd22 - cd12 * cd21))
            return scale_deg * 3600.0
        except Exception:
            if fallback_px_um and fallback_fl_mm:
                return 206.264806 * (fallback_px_um / fallback_fl_mm)
            return None

class ImagePeekerDialogPro(QDialog):
    def __init__(self, parent, document, settings):
        super().__init__(parent)
        self.setWindowTitle("Image Peeker")
        self.document = self._coerce_doc(document)   # <- ensure we hold a real doc
        self.settings = settings
        # status / progress line
        self.status_lbl = QLabel("")
        self.status_lbl.setStyleSheet("color:#bbb;")


        self.params = QGroupBox("Grid parameters")
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
        gl.addWidget(QLabel("Grid size:"), row, 0); gl.addWidget(self.grid_spin, row, 1); row += 1
        gl.addWidget(QLabel("Panel size:"), row, 0)
        pr = QHBoxLayout(); pr.addWidget(self.panel_slider, 1); pr.addWidget(self.panel_value_label)
        gl.addLayout(pr, row, 1); row += 1
        gl.addWidget(QLabel("Separation:"), row, 0)
        sr = QHBoxLayout(); sr.addWidget(self.sep_slider, 1); sr.addWidget(self.sep_value_label)
        gl.addLayout(sr, row, 1); row += 1
        gl.addWidget(QLabel("Pixel size (µm):"), row, 0); gl.addWidget(self.pixel_size_input, row, 1); row += 1
        gl.addWidget(QLabel("Focal length (mm):"), row, 0); gl.addWidget(self.focal_length_input, row, 1); row += 1
        gl.addWidget(QLabel("Aperture (mm):"), row, 0); gl.addWidget(self.aperture_input, row, 1); row += 1

        # Right side
        from PyQt6.QtWidgets import QTabWidget
        self.preview_pane = PreviewPane()
        analysis_row = QHBoxLayout()
        analysis_row.addWidget(QLabel("Analysis:"))
        self.analysis_combo = QComboBox()
        self.analysis_combo.addItems(["None", "Tilt Analysis", "Focal Plane Analysis", "Astrometric Distortion Analysis"])
        analysis_row.addWidget(self.analysis_combo); analysis_row.addStretch(1)

        btns = QHBoxLayout(); btns.addStretch(1)
        ok_btn = QPushButton("Save Settings && Exit"); cancel_btn = QPushButton("Exit without Saving")
        btns.addWidget(ok_btn); btns.addWidget(cancel_btn)

        main = QHBoxLayout(self)
        main.addWidget(self.params)
        right = QVBoxLayout(); right.addLayout(analysis_row); right.addWidget(self.status_lbl, 0), right.addWidget(self.preview_pane, 1); right.addLayout(btns)
        main.addLayout(right, 1)

        # Signals
        self.grid_spin.valueChanged.connect(self._refresh_mosaic)
        self.panel_slider.valueChanged.connect(lambda v: (self.panel_value_label.setText(str(v)), self._refresh_mosaic()))
        self.sep_slider.valueChanged.connect(lambda v: (self.sep_value_label.setText(str(v)), self._refresh_mosaic()))
        self.analysis_combo.currentTextChanged.connect(self._run_analysis)
        ok_btn.clicked.connect(self.accept); cancel_btn.clicked.connect(self.reject)

        QTimer.singleShot(0, self._refresh_mosaic)

    def _set_busy(self, on: bool, text: str = "Processing…"):
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
        mode = self.analysis_combo.currentText()
        if mode == "None":
            self._set_busy(False, "")
            self._refresh_mosaic()
            return
        self._set_busy(True, f"Running {mode}…")
        QTimer.singleShot(0, lambda: self._run_analysis_dispatch(mode))

    def _run_analysis_dispatch(self, mode: str):
        try:
            arr, meta = self._arr_and_meta()
            if arr is None or arr.size == 0:
                return
            ps_um = float(meta.get("pixel_size_um", self.pixel_size_input.value()))
            fl_mm = float(meta.get("focal_length_mm", self.focal_length_input.value()))
            ap_mm = float(meta.get("aperture_mm", self.aperture_input.value()))
            snr_th = float(meta.get("snr_threshold", 5.0))

            if mode == "Tilt Analysis":
                norm_plane, (a,b,c), (H,W) = tilt_analysis(
                    arr, pixel_size_um=ps_um, focal_length_mm=fl_mm, aperture_mm=ap_mm,
                    sigma_clip=2.5, thresh_sigma=snr_th
                )
                TiltDialog("Sensor Tilt (µm)", norm_plane, (a,b,c), (H,W), ps_um, parent=self).show()

            elif mode == "Focal Plane Analysis":
                fwhm_heat, (mn_f, mx_f) = compute_fwhm_surface(arr, ps_um, thresh_sigma=snr_th, deg=3)
                SurfaceDialog("FWHM Heatmap", fwhm_heat, mn_f, mx_f, "µm", "viridis", parent=self).show()
                ecc_heat, (mn_e, mx_e) = compute_eccentricity_surface(arr, ps_um, thresh_sigma=snr_th, deg=3)
                SurfaceDialog("Eccentricity Map", ecc_heat, mn_e, mx_e, "e = 1−b/a", "magma", parent=self).show()
                ori_heat, (mn_o, mx_o) = compute_orientation_surface(arr, thresh_sigma=snr_th, deg=3)
                SurfaceDialog("Orientation Map", ori_heat, mn_o, mx_o, "rad", "hsv", parent=self).show()

            elif mode == "Astrometric Distortion Analysis":
                hdr = _header_from_meta(meta)

                # If we truly have no WCS, plate-solve
                if hdr is None or not WCS(hdr, relax=True).has_celestial:
                    ok, hdr_or_err = plate_solve_doc_inplace(
                        parent=self, doc=self._coerce_doc(self.document), settings=self.settings
                    )
                    if not ok:
                        QMessageBox.warning(self, "Plate Solve", f"ASTAP/Astrometry failed:\n{hdr_or_err}")
                        return

                    # IMPORTANT: if solver returned a Header, store it
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

                # Now WCS exists, but do we have SIP?
                if hdr is None:
                    QMessageBox.critical(self, "WCS Error", "Plate solve did not produce a readable WCS header.")
                    return

                has_sip = any(k.startswith("A_") for k in hdr.keys()) and any(k.startswith("B_") for k in hdr.keys())
                if not has_sip:
                    QMessageBox.warning(
                        self, "No Distortion Model",
                        "This image has a valid WCS, but no SIP distortion terms (A_*, B_*).\n"
                        "Astrometric distortion analysis requires a SIP-enabled solve.\n\n"
                        "Re-solve with distortion fitting enabled in ASTAP."
                    )
                    return

                asp = _arcsec_per_pix_from_header(hdr, fallback_px_um=ps_um, fallback_fl_mm=fl_mm)
                if asp is None:
                    QMessageBox.critical(self, "WCS Error", "Cannot determine pixel scale.")
                    return

                DistortionGridDialog(
                    img=np.clip(arr, 0, 1), sip_meta=hdr, arcsec_per_pix=float(asp),
                    n_grid_lines=10, amplify=60.0, parent=self
                ).show()


            else:
                self._refresh_mosaic()
        finally:
            self._set_busy(False, "")

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
        col = QColorDialog.getColor(self._sep_color, self, "Choose separation color")
        if col.isValid():
            self._sep_color = col
            self._update_sep_color_button()

    def _refresh_mosaic(self):
        arr, _ = self._arr_and_meta()
        if arr is None or arr.size == 0:
            return
        # ensure RGB for preview
        if arr.ndim == 2: arr = np.repeat(arr[...,None], 3, axis=2)
        qimg = self._to_qimage(np.clip(arr, 0, 1))
        n = max(2, int(self.grid_spin.value()))
        ps  = int(self.panel_slider.value())
        sep = int(self.sep_slider.value())
        mosaic = self._build_mosaic(qimg, n, ps, sep, QColor(0,0,0))
        self.preview_pane.load_qimage(mosaic)

    def _on_ok(self):
        # user clicked OK → generate & display the mosaic
        n        = self.grid_spin.value
        panel_sz = self.panel_slider.value()
        sep      = self.sep_slider.value()
        sep_col  = self._sep_color

        # fetch the currently loaded image (you’ll adapt to your image_manager API)
        img = self.image_manager.current_qimage()
        if img is None:
            QMessageBox.warning(self, "No image", "No image loaded to peek at!")
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

