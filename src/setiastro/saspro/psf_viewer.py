# saspro/psf_viewer.py
from __future__ import annotations

import math
import numpy as np
import sep
sep.set_extract_pixstack(20000000)
from astropy.table import Table

from PyQt6.QtCore import Qt, QTimer, QRectF, QPointF, QSettings
from PyQt6.QtGui import QPainter, QPen, QFont, QPixmap, QColor, QBrush, QImage
from PyQt6.QtWidgets import (
    QDialog, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QScrollArea,
    QSlider, QTableWidget, QTableWidgetItem, QApplication,
    QSizePolicy, QCheckBox,
)
from setiastro.saspro.widgets.themed_buttons import themed_toolbtn

from PyQt6.QtCore import QThread, pyqtSignal, QObject
from PyQt6.QtWidgets import QWidget

from .psf_utils import detect_stars_waterfall


# ---------------------------------------------------------------------------
# Processing overlay
# ---------------------------------------------------------------------------
class _ProcessingOverlay(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setStyleSheet("""
            QWidget { background: rgba(0,0,0,140); border-radius: 10px; }
            QLabel  { color: white; font-size: 14px; font-weight: 600; }
        """)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 14, 18, 14)
        self.lbl = QLabel("Processing…", self)
        self.lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self.lbl)

    def setText(self, s: str):
        self.lbl.setText(s)


# ---------------------------------------------------------------------------
# Star shape widget  (pre-renders to QPixmap, scales on resize — no per-pixel loop at runtime)
# ---------------------------------------------------------------------------
class _StarWidget(QWidget):
    """
    Renders a synthetic median star stamp.
    The heavy Gaussian is computed once into a QPixmap; resize just scales that pixmap.
    """
    _RENDER_SIZE = 300   # internal render resolution

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(150, 150)
        self._cached_smooth_pm: QPixmap | None = None
        self._cached_pixel_pm:  QPixmap | None = None
        self._stamp: np.ndarray | None = None      # 2D float, real median star sampled at native pixel resolution
        self._pixel_mode = False
        self._a     = 2.0
        self._b     = 1.8
        self._theta = 0.0
        self._fwhm  = 4.7
        self._hfr   = 4.0
        self._ecc   = 0.0
        self._valid = False
        # Click to toggle smooth ↔ true-pixel view
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("Click to toggle: smooth ↔ true pixels")

    def set_star(self, a: float, b: float, theta: float, fwhm: float, hfr: float, ecc: float,
                 pixel_stamp: np.ndarray | None = None):
        self._a     = max(float(a),    0.1)
        self._b     = max(float(b),    0.1)
        self._theta = float(theta)
        self._fwhm  = max(float(fwhm), 0.5)
        self._hfr   = max(float(hfr),  0.5)
        self._ecc   = float(ecc)
        self._valid = True
        self._stamp = pixel_stamp if pixel_stamp is not None else None
        self._rebuild_smooth_cache()
        self._rebuild_pixel_cache()
        # If we lost the pixel stamp (e.g. star near edges), fall back to smooth.
        if self._pixel_mode and self._cached_pixel_pm is None:
            self._pixel_mode = False
        self.update()

    def clear(self):
        self._valid = False
        self._cached_smooth_pm = None
        self._cached_pixel_pm = None
        self._stamp = None
        self._pixel_mode = False
        self.update()

    def mousePressEvent(self, e):
        # Toggle between smooth Gaussian view and real-pixel median stamp.
        # Only meaningful if we actually have a pixel stamp to show.
        if (e.button() == Qt.MouseButton.LeftButton
                and self._valid
                and self._cached_pixel_pm is not None):
            self._pixel_mode = not self._pixel_mode
            self.update()
        super().mousePressEvent(e)

    # ------------------------------------------------------------------
    def _rebuild_smooth_cache(self):
        """Render the smooth (analytic Gaussian) star at _RENDER_SIZE into a QPixmap.
        Called once per new star data — resize just scales the cached pixmap."""
        N = self._RENDER_SIZE
        cx = cy = N / 2.0

        # --- Gaussian blob on numpy grid ----------------------------------
        scale = (N * 0.16) / max(self._a, self._b)
        scale = max(scale, 2.0)

        a_px  = self._a * scale
        b_px  = self._b * scale
        theta = self._theta

        xs = np.arange(N, dtype=np.float32) - cx
        ys = np.arange(N, dtype=np.float32) - cy
        xg, yg = np.meshgrid(xs, ys)

        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        xr =  xg * cos_t + yg * sin_t
        yr = -xg * sin_t + yg * cos_t

        gauss = np.exp(-0.5 * ((xr / a_px) ** 2 + (yr / b_px) ** 2))
        gauss = (gauss / gauss.max() * 255).astype(np.uint8)

        rgb = np.stack([gauss, gauss, gauss], axis=2)
        rgb = np.ascontiguousarray(rgb)
        h, w, _ = rgb.shape
        qi      = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)
        base_pm = QPixmap.fromImage(qi)

        # --- Paint overlays onto the pixmap --------------------------------
        pm = QPixmap(N, N)
        pm.fill(QColor(30, 30, 40))
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Gaussian blob
        p.drawPixmap(0, 0, base_pm)

        # Pre-compute ellipse semi-axes in screen pixels
        # Major axis (along a) and minor axis (along b), scaled to match blob
        hfr_a  = (self._hfr  / 2.0) * scale
        hfr_b  = hfr_a  * (self._b / max(self._a, 1e-9))
        fwhm_a = (self._fwhm / 2.0) * scale
        fwhm_b = fwhm_a * (self._b / max(self._a, 1e-9))

        # HFR ellipse (orange) — rotated to match star orientation
        p.save()
        p.translate(cx, cy)
        p.rotate(math.degrees(theta))     # was: 90.0 - math.degrees(theta)
        p.setPen(QPen(QColor(255, 140, 0), 1.8, Qt.PenStyle.SolidLine))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(QRectF(-hfr_a, -hfr_b, hfr_a * 2, hfr_b * 2))
        p.restore()

        # FWHM ellipse (green) — rotated to match star orientation
        p.save()
        p.translate(cx, cy)
        p.rotate(math.degrees(theta))     # was: 90.0 - math.degrees(theta)
        p.setPen(QPen(QColor(80, 200, 80), 1.8, Qt.PenStyle.SolidLine))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(QRectF(-fwhm_a, -fwhm_b, fwhm_a * 2, fwhm_b * 2))
        p.restore()

        # Crosshair (cyan)
        p.setPen(QPen(QColor(0, 200, 220), 1.0, Qt.PenStyle.SolidLine))
        p.drawLine(0, int(cy), N, int(cy))
        p.drawLine(int(cx), 0, int(cx), N)

        # Major axis — red
        arm_a = a_px * 2.4
        dx_a  =  arm_a * cos_t
        dy_a  =  arm_a * sin_t  
        p.setPen(QPen(QColor(220, 60, 60), 2.0))
        p.drawLine(QPointF(cx - dx_a, cy - dy_a), QPointF(cx + dx_a, cy + dy_a))

        # Minor axis — blue
        arm_b = b_px * 2.4
        dx_b  = -arm_b * sin_t
        dy_b  =  arm_b * cos_t
        p.setPen(QPen(QColor(80, 120, 220), 2.0))
        p.drawLine(QPointF(cx - dx_b, cy - dy_b), QPointF(cx + dx_b, cy + dy_b))

        # Legend
        p.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        lx, ly, lh = 8, 14, 17
        p.setPen(QColor(220, 60, 60));  p.drawText(lx, ly,         "PSF X")
        p.setPen(QColor(80, 120, 220)); p.drawText(lx, ly + lh,    "PSF Y")
        p.setPen(QColor(80, 200, 80));  p.drawText(lx, ly + lh*2,  "FWHM")
        p.setPen(QColor(255, 140, 0));  p.drawText(lx, ly + lh*3,  "HFR")

        # Eccentricity readout bottom-left
        p.setFont(QFont("Segoe UI", 8))
        p.setPen(QColor(200, 200, 200))
        p.drawText(8, N - 8, f"ecc: {self._ecc:.3f}")

        # Mode hint bottom-right
        p.setFont(QFont("Segoe UI", 8))
        p.setPen(QColor(150, 150, 165))
        p.drawText(N - 130, N - 8, "click → true pixels")

        p.end()
        self._cached_smooth_pm = pm

    # ------------------------------------------------------------------
    def _rebuild_pixel_cache(self):
        """Render the median star stamp at *native pixel resolution*, scaled up
        with nearest-neighbor into _RENDER_SIZE so each image pixel is a big
        crisp square. Same FWHM/HFR overlays as the smooth view so it's clear
        this is the same star, just sampled honestly."""
        if self._stamp is None:
            self._cached_pixel_pm = None
            return

        stamp = np.asarray(self._stamp, dtype=np.float32)
        if stamp.ndim != 2 or stamp.size == 0:
            self._cached_pixel_pm = None
            return

        H, W = stamp.shape
        N = self._RENDER_SIZE

        # Normalize 0..1 for display; the stamp fed in is already peak≈1
        # but background can dip slightly negative after median-of-edges sub.
        s = stamp - float(stamp.min())
        m = float(s.max())
        if m > 0:
            s = s / m
        img8 = (np.clip(s, 0.0, 1.0) * 255.0).astype(np.uint8)
        rgb  = np.ascontiguousarray(np.stack([img8, img8, img8], axis=2))
        qi   = QImage(rgb.data, W, H, W * 3, QImage.Format.Format_RGB888)
        base_pm = QPixmap.fromImage(qi)     # copies buffer internally

        # Fit stamp into the render area with an *integer* pixel size, so
        # every image pixel is exactly the same number of screen pixels.
        margin = 10
        avail  = N - 2 * margin
        pix_size = max(1, avail // max(W, H))
        draw_w   = pix_size * W
        draw_h   = pix_size * H
        ox = (N - draw_w) // 2
        oy = (N - draw_h) // 2

        pm = QPixmap(N, N)
        pm.fill(QColor(30, 30, 40))
        p  = QPainter(pm)
        # NB: NO antialiasing hint here — we want crisp square pixels.

        scaled = base_pm.scaled(
            draw_w, draw_h,
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.FastTransformation,   # nearest neighbor
        )
        p.drawPixmap(ox, oy, scaled)

        # Faint grid between pixels
        p.setPen(QPen(QColor(70, 70, 85), 1))
        for i in range(W + 1):
            x = ox + i * pix_size
            p.drawLine(x, oy, x, oy + draw_h)
        for j in range(H + 1):
            y = oy + j * pix_size
            p.drawLine(ox, y, ox + draw_w, y)

        # Ellipses & axes — same overlays as smooth view, but in screen-pixel
        # units where 1 image pixel == pix_size screen pixels. That means the
        # FWHM ellipse literally measures the FWHM against the grid.
        p.setRenderHint(QPainter.RenderHint.Antialiasing)   # for the overlays only
        cx = ox + (W / 2.0) * pix_size
        cy = oy + (H / 2.0) * pix_size
        scale = float(pix_size)

        cos_t = math.cos(self._theta)
        sin_t = math.sin(self._theta)

        # HFR ellipse (orange)
        hfr_a = (self._hfr / 2.0) * scale
        hfr_b = hfr_a * (self._b / max(self._a, 1e-9))
        p.save(); p.translate(cx, cy); p.rotate(math.degrees(self._theta))
        p.setPen(QPen(QColor(255, 140, 0), 1.6))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(QRectF(-hfr_a, -hfr_b, hfr_a * 2, hfr_b * 2))
        p.restore()

        # FWHM ellipse (green)
        fwhm_a = (self._fwhm / 2.0) * scale
        fwhm_b = fwhm_a * (self._b / max(self._a, 1e-9))
        p.save(); p.translate(cx, cy); p.rotate(math.degrees(self._theta))
        p.setPen(QPen(QColor(80, 200, 80), 1.6))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(QRectF(-fwhm_a, -fwhm_b, fwhm_a * 2, fwhm_b * 2))
        p.restore()

        # Crosshair (cyan, subtle) — clipped to the stamp
        p.setPen(QPen(QColor(0, 200, 220, 160), 1.0))
        p.drawLine(QPointF(ox, cy), QPointF(ox + draw_w, cy))
        p.drawLine(QPointF(cx, oy), QPointF(cx, oy + draw_h))

        # Major / minor PSF axes (red / blue)
        arm_a = self._a * scale * 2.0
        p.setPen(QPen(QColor(220, 60, 60), 1.8))
        p.drawLine(QPointF(cx - arm_a * cos_t, cy - arm_a * sin_t),
                   QPointF(cx + arm_a * cos_t, cy + arm_a * sin_t))
        arm_b = self._b * scale * 2.0
        p.setPen(QPen(QColor(80, 120, 220), 1.8))
        p.drawLine(QPointF(cx + arm_b * sin_t, cy - arm_b * cos_t),
                   QPointF(cx - arm_b * sin_t, cy + arm_b * cos_t))

        # Header — stamp dimensions in image pixels + scale factor
        p.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        p.setPen(QColor(230, 230, 230))
        p.drawText(8, 16, f"True Pixels  {W}×{H} px  ·  1 px → {pix_size}px")

        # FWHM readout bottom-left
        p.setFont(QFont("Segoe UI", 8))
        p.setPen(QColor(200, 200, 200))
        p.drawText(8, N - 8, f"FWHM: {self._fwhm:.2f} px")

        # Mode hint bottom-right
        p.setPen(QColor(150, 150, 165))
        p.drawText(N - 120, N - 8, "click → smooth")

        p.end()
        self._cached_pixel_pm = pm

    def paintEvent(self, event):
        pm = self._cached_pixel_pm if self._pixel_mode else self._cached_smooth_pm
        if pm is None:
            # No data — draw placeholder
            p = QPainter(self)
            p.fillRect(self.rect(), QColor(30, 30, 40))
            p.setPen(QColor(100, 100, 120))
            p.setFont(QFont("Segoe UI", 9))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No stars\ndetected")
            p.end()
            return

        # For pixel mode use nearest-neighbor so the squares stay crisp when
        # the widget size doesn't evenly divide _RENDER_SIZE. For smooth mode
        # keep bilinear — it's a continuous Gaussian.
        tmode = (Qt.TransformationMode.FastTransformation if self._pixel_mode
                 else Qt.TransformationMode.SmoothTransformation)
        scaled = pm.scaled(
            self.width(), self.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            tmode,
        )
        p = QPainter(self)
        ox = (self.width()  - scaled.width())  // 2
        oy = (self.height() - scaled.height()) // 2
        p.drawPixmap(ox, oy, scaled)
        p.end()


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------
class _PSFWorker(QObject):
    finished = pyqtSignal(object, str)
    failed   = pyqtSignal(str)

    def __init__(self, image: np.ndarray, threshold_sigma: float,
                 auto_mode: bool = True, target_stars: int = 1000):
        super().__init__()
        self.image = image
        self.threshold_sigma = float(threshold_sigma)
        self.auto_mode       = bool(auto_mode)
        self.target_stars    = int(target_stars)

    def run(self):
        try:
            if self.image is None:
                self.finished.emit(None, "Status: No image.")
                return

            if self.image.ndim == 3:
                image_gray = np.mean(self.image, axis=2)
            else:
                image_gray = self.image
            gray = image_gray.astype(np.float32, copy=False)

            if self.auto_mode:
                # Waterfall: descend from sigma=100 to sigma=3, stop as soon
                # as target_stars quality-filtered detections are in hand.
                cat = detect_stars_waterfall(
                    gray,
                    sigma_ladder=(100.0, 50.0, 25.0, 12.0, 6.0, 3.0),
                    target_count=self.target_stars,
                    quality_filter=True,
                )
                if cat is None or cat['n'] == 0:
                    self.finished.emit(None, "Status: Extraction completed — 0 stars.")
                    return
                sig_used = cat['sigma_stopped']
                raw_at   = cat['total_at_stop']
                status = (f"Status: Auto — {cat['n']} stars at σ={sig_used:g} "
                          f"(kept {cat['n']}/{raw_at} at that threshold).")
            else:
                # Manual: single-shot at the user's exact sigma. Still filter
                # so medians aren't polluted by saturated / blended detections.
                cat = detect_stars_waterfall(
                    gray,
                    sigma_ladder=(self.threshold_sigma,),
                    target_count=10**9,          # never trigger early stop
                    quality_filter=True,
                )
                if cat is None or cat['n'] == 0:
                    self.finished.emit(None, f"Status: Extraction completed — 0 stars at σ={self.threshold_sigma:g}.")
                    return
                raw_at = cat['total_at_stop']
                status = (f"Status: Manual — {cat['n']} stars at σ={self.threshold_sigma:g} "
                          f"(kept {cat['n']}/{raw_at} after quality filter).")

            a_arr = cat['a'].astype(np.float32, copy=False)
            b_arr = cat['b'].astype(np.float32, copy=False)

            tbl = Table()
            tbl["xcentroid"] = cat['x']
            tbl["ycentroid"] = cat['y']
            tbl["flux"]      = cat['flux']
            tbl["HFR"]       = 2.0 * a_arr
            tbl["FWHM"]      = 2.3548 * a_arr
            tbl["a"]         = a_arr
            tbl["b"]         = b_arr
            tbl["theta"]     = cat['theta']

            self.finished.emit(tbl, status)
        except Exception as e:
            self.failed.emit(f"Extraction failed: {e}")


# ---------------------------------------------------------------------------
# Main dialog
# ---------------------------------------------------------------------------
class PSFViewer(QDialog):
    def __init__(self, view_or_doc, parent=None):
        super().__init__(parent)
        self.setWindowTitle("PSF Viewer")

        doc = getattr(view_or_doc, "document", None)
        self.doc = doc if doc is not None else view_or_doc
        try:
            self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        except Exception:
            pass

        self.image = self._grab_image()
        self.log_scale = False
        self.star_list = None
        self.histogram_mode = "PSF"
        self.detection_threshold = 15

        # Auto-detect (waterfall) vs manual (single-shot at slider sigma).
        # Auto is the sane default: descends the sigma ladder until ~1000
        # quality-filtered stars are in hand, so a 53k-source frame doesn't
        # spend seconds fitting shape moments we would just throw away.
        self._settings = QSettings()
        self.auto_detect = bool(self._settings.value(
            "psf_viewer/auto_detect", True, type=bool))
        self._target_stars = int(self._settings.value(
            "psf_viewer/target_stars", 1000, type=int))

        self.threshold_timer = QTimer(self)
        self.threshold_timer.setSingleShot(True)
        self.threshold_timer.setInterval(500)
        self.threshold_timer.timeout.connect(self._applyThreshold)

        self._psf_thread = None
        self._psf_worker = None
        self._doc_conn = False
        if hasattr(self.doc, "changed"):
            try:
                self.doc.changed.connect(self._on_doc_changed)
                self._doc_conn = True
            except Exception:
                self._doc_conn = False

        self.finished.connect(self._cleanup)
        self._build_ui()
        QTimer.singleShot(0, self._applyThreshold)

    def _grab_image(self):
        img = getattr(self.doc, "image", None)
        if img is None:
            return None
        try:
            return np.asarray(img)
        except Exception:
            return None

    def _on_doc_changed(self, *_):
        self.image = self._grab_image()
        if self.threshold_timer.isActive():
            self.threshold_timer.stop()
        self.threshold_timer.start()

    # ------------------------------------------------------------------
    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(8)

        # ── Top row: histogram + star graphic ──────────────────────────
        top_layout = QHBoxLayout()
        top_layout.setSpacing(10)

        # Histogram scroll area — expands with window
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setMinimumSize(400, 180)
        self.scroll_area.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.scroll_area.setWidgetResizable(False)
        self.hist_label = QLabel(self)
        self.hist_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scroll_area.setWidget(self.hist_label)
        top_layout.addWidget(self.scroll_area, stretch=3)

        # Star graphic column — expands with window
        star_col = QVBoxLayout()
        star_col.setSpacing(4)
        star_lbl = QLabel("Median Star Profile")
        star_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        star_lbl.setStyleSheet("font-size: 11px; color: #aaa;")
        self._star_widget = _StarWidget(self)
        self._star_widget.setMinimumSize(150, 150)
        self._star_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        star_col.addWidget(star_lbl)
        star_col.addWidget(self._star_widget)
        top_layout.addLayout(star_col, stretch=1)

        main_layout.addLayout(top_layout, stretch=1)

        # ── Stats table — tall enough to avoid scrollbar ────────────────
        self.stats_table = QTableWidget(self)
        self.stats_table.setRowCount(4)
        self.stats_table.setColumnCount(0)
        self.stats_table.setVerticalHeaderLabels(["Median", "Min", "Max", "StdDev"])
        self.stats_table.setMinimumHeight(138)
        self.stats_table.setMaximumHeight(160)
        self.stats_table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        main_layout.addWidget(self.stats_table)

        # ── Status ──────────────────────────────────────────────────────
        self.status_label = QLabel("Status: Ready", self)
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.status_label)

        # ── Zoom controls ───────────────────────────────────────────────
        controls_layout = QHBoxLayout()
        controls_layout.addWidget(QLabel("Zoom:"))

        btn_zoom_out = themed_toolbtn("zoom-out",      "Zoom Out")
        btn_zoom_in  = themed_toolbtn("zoom-in",       "Zoom In")
        btn_fit      = themed_toolbtn("zoom-fit-best", "Fit")
        btn_zoom_out.clicked.connect(lambda: self._step_zoom(1 / 1.25))
        btn_zoom_in.clicked.connect(lambda:  self._step_zoom(1.25))
        btn_fit.clicked.connect(self._fit_histogram)
        controls_layout.addWidget(btn_zoom_out)
        controls_layout.addWidget(btn_zoom_in)
        controls_layout.addWidget(btn_fit)

        self.zoom_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.zoom_slider.setRange(50, 1000)
        self.zoom_slider.setValue(100)
        self.zoom_slider.setTickInterval(10)
        self.zoom_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.zoom_slider.valueChanged.connect(self.updateZoom)
        controls_layout.addWidget(self.zoom_slider, 1)

        self.log_toggle_button = QPushButton("X-Axis: Linear", self)
        self.log_toggle_button.setCheckable(True)
        self.log_toggle_button.setChecked(False)
        self.log_toggle_button.toggled.connect(self.toggleLogScale)
        controls_layout.addWidget(self.log_toggle_button)

        self.mode_toggle_button = QPushButton("Show Flux Histogram", self)
        self.mode_toggle_button.clicked.connect(self.toggleHistogramMode)
        controls_layout.addWidget(self.mode_toggle_button)

        main_layout.addLayout(controls_layout)

        # ── Threshold ───────────────────────────────────────────────────
        thresh_layout = QHBoxLayout()
        thresh_layout.addWidget(QLabel("Detection Threshold (σ):", self))
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.threshold_slider.setRange(1, 50)
        self.threshold_slider.setValue(self.detection_threshold)
        self.threshold_slider.setTickInterval(1)
        self.threshold_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.threshold_slider.valueChanged.connect(self.onThresholdChange)
        thresh_layout.addWidget(self.threshold_slider)
        self.threshold_value_label = QLabel(str(self.detection_threshold), self)
        thresh_layout.addWidget(self.threshold_value_label)

        self.chk_auto = QCheckBox("Auto (waterfall → ~1000 stars)", self)
        self.chk_auto.setToolTip(
            "When checked, descend through σ = 100, 50, 25, 12, 6, 3 and stop "
            "as soon as ~1000 quality stars are detected. Fast on rich fields "
            "and gives a more representative PSF median than dropping the "
            "slider low. Uncheck to use the slider value literally."
        )
        self.chk_auto.setChecked(self.auto_detect)
        self.chk_auto.toggled.connect(self._on_auto_toggled)
        thresh_layout.addWidget(self.chk_auto)

        # Slider is meaningless while auto-detect is on.
        self.threshold_slider.setEnabled(not self.auto_detect)
        self.threshold_value_label.setEnabled(not self.auto_detect)

        main_layout.addLayout(thresh_layout)

        # ── Close ───────────────────────────────────────────────────────
        close_btn = QPushButton("Close", self)
        close_btn.clicked.connect(self.close)
        main_layout.addWidget(close_btn)

        self.setLayout(main_layout)
        self.drawHistogram()

    # ------------------------------------------------------------------
    def onThresholdChange(self, value: int):
        self.detection_threshold = int(value)
        self.threshold_value_label.setText(str(value))
        if self.threshold_timer.isActive():
            self.threshold_timer.stop()
        self.threshold_timer.start()

    def _on_auto_toggled(self, checked: bool):
        self.auto_detect = bool(checked)
        try:
            self._settings.setValue("psf_viewer/auto_detect", self.auto_detect)
        except Exception:
            pass
        self.threshold_slider.setEnabled(not self.auto_detect)
        self.threshold_value_label.setEnabled(not self.auto_detect)
        # Re-run detection immediately -- toggling mode IS the "give me a
        # different result" signal. Route through the debounce so we don't
        # start work if the user is rapid-toggling.
        if self.threshold_timer.isActive():
            self.threshold_timer.stop()
        self.threshold_timer.start()

    def _step_zoom(self, factor: float):
        v = int(round(self.zoom_slider.value() * factor))
        v = max(self.zoom_slider.minimum(), min(self.zoom_slider.maximum(), v))
        self.zoom_slider.setValue(v)

    def _fit_histogram(self):
        if not hasattr(self, "_base_hist_pm") or self._base_hist_pm is None:
            return
        vp_w   = self.scroll_area.viewport().width()
        base_w = max(1, self._base_hist_pm.width())
        self.zoom_slider.setValue(int(round(vp_w / base_w * 100)))

    def _apply_hist_zoom(self):
        if not hasattr(self, "_base_hist_pm") or self._base_hist_pm is None:
            return
        z = self.zoom_slider.value() / 100.0
        w = max(1, int(self._base_hist_pm.width()  * z))
        h = max(1, int(self._base_hist_pm.height() * z))
        scaled = self._base_hist_pm.scaled(
            w, h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.hist_label.setPixmap(scaled)
        self.hist_label.resize(scaled.size())

    def _applyThreshold(self):
        if self.image is None:
            self.star_list = None
            self.status_label.setText("Status: No image.")
            self.drawHistogram()
            self._star_widget.clear()
            return

        self._show_processing("Processing… extracting stars / PSFs")
        self._stop_psf_worker()

        self._psf_thread = QThread(self)
        self._psf_worker = _PSFWorker(
            self.image,
            self.detection_threshold,
            auto_mode=self.auto_detect,
            target_stars=self._target_stars,
        )
        self._psf_worker.moveToThread(self._psf_thread)
        self._psf_thread.started.connect(self._psf_worker.run)
        self._psf_worker.finished.connect(self._on_psf_done)
        self._psf_worker.failed.connect(self._on_psf_fail)
        self._psf_worker.finished.connect(lambda *_: self._stop_psf_worker(quit_only=False))
        self._psf_worker.failed.connect(lambda *_:   self._stop_psf_worker(quit_only=False))
        self._psf_thread.start()

    def _stop_psf_worker(self, quit_only: bool = False):
        thr = getattr(self, "_psf_thread", None)
        wkr = getattr(self, "_psf_worker", None)
        if thr is None:
            return
        try: thr.quit()
        except Exception: pass
        try: thr.wait(250)
        except Exception: pass
        if not quit_only:
            try:
                if wkr is not None: wkr.deleteLater()
            except Exception: pass
            try: thr.deleteLater()
            except Exception: pass
            self._psf_worker = None
            self._psf_thread = None

    def _on_psf_done(self, tbl, status: str):
        self.star_list = tbl
        self.status_label.setText(status)
        self._hide_processing()
        self.drawHistogram()
        self._update_star_widget()

    def _on_psf_fail(self, msg: str):
        self.star_list = None
        self.status_label.setText(f"Status: {msg}")
        self._hide_processing()
        self.drawHistogram()
        self._star_widget.clear()

    def _update_star_widget(self):
        if self.star_list is None or len(self.star_list) == 0:
            self._star_widget.clear()
            return
        try:
            a_arr   = np.array(self.star_list["a"],     dtype=float)
            b_arr   = np.array(self.star_list["b"],     dtype=float)
            ecc_arr = np.sqrt(1.0 - (b_arr / np.maximum(a_arr, 1e-9)) ** 2)
            a     = float(np.median(a_arr))
            b     = float(np.median(b_arr))
            fwhm  = float(np.median(np.array(self.star_list["FWHM"],  dtype=float)))
            hfr   = float(np.median(np.array(self.star_list["HFR"],   dtype=float)))
            ecc   = float(np.median(ecc_arr))

            # SEP theta is in the array frame (x=col, y=row). The star stamp bakes
            # the blob row-0-at-top (y down), so raw theta matches a main view that
            # also renders row-0-at-top. If the main image view shows FITS
            # orientation (row 0 at bottom / y up), the on-screen display is flipped
            # vertically relative to the stamp → negate theta so the stamp's
            # elongation matches the real star's lean in the image.
            MAIN_VIEW_IS_YUP = False   # set True if the main image view is FITS/y-up
            theta = float(np.median(np.array(self.star_list["theta"], dtype=float)))
            if MAIN_VIEW_IS_YUP:
                theta = -theta

            stamp = self._compute_median_stamp(fwhm)
            self._star_widget.set_star(a, b, theta, fwhm, hfr, ecc, pixel_stamp=stamp)
        except Exception:
            self._star_widget.clear()

    def _compute_median_stamp(self, fwhm: float, max_stars: int = 500) -> np.ndarray | None:
        """Median-stack small real-image cutouts around detected stars, so the
        pixel view of the star widget shows what an actual star in this image
        looks like at native pixel resolution — noise, sampling and all.

        Each cutout is background-subtracted (edge median) and peak-normalized
        before stacking so bright stars don't dominate the median."""
        if self.image is None or self.star_list is None or len(self.star_list) == 0:
            return None

        img = self.image
        if img.ndim == 3:
            img = np.mean(img, axis=2)
        img = np.ascontiguousarray(img, dtype=np.float32)
        H, W = img.shape

        # ~±2.5 FWHM on each side gives you core + wings without being huge.
        # Clamp so the stamp always shows something (very small FWHM) but never
        # dominates on wide seeing.
        half = int(np.clip(np.ceil(2.5 * max(fwhm, 1.0)), 5, 25))

        xs = np.asarray(self.star_list["xcentroid"], dtype=float)
        ys = np.asarray(self.star_list["ycentroid"], dtype=float)
        if xs.size == 0:
            return None

        # Cap for speed — 500 stamps is plenty to nail a median.
        if xs.size > max_stars:
            idx = np.linspace(0, xs.size - 1, max_stars).astype(int)
            xs = xs[idx]; ys = ys[idx]

        stamps = []
        side = 2 * half + 1
        for xi, yi in zip(xs, ys):
            cx = int(round(xi)); cy = int(round(yi))
            x0 = cx - half; x1 = cx + half + 1
            y0 = cy - half; y1 = cy + half + 1
            if x0 < 0 or y0 < 0 or x1 > W or y1 > H:
                continue
            s = img[y0:y1, x0:x1]
            # Background from the 1-pixel edge of the stamp
            border = np.concatenate([s[0], s[-1], s[1:-1, 0], s[1:-1, -1]])
            bg = float(np.median(border))
            s  = s - bg
            pk = float(s.max())
            if pk > 0.0:
                stamps.append(s / pk)

        if not stamps:
            return None
        return np.median(np.stack(stamps, axis=0), axis=0).astype(np.float32)

    def updateImage(self, new_image):
        self.image = np.asarray(new_image) if new_image is not None else None
        if self.threshold_timer.isActive():
            self.threshold_timer.stop()
        self.threshold_timer.start()

    def updateZoom(self, _=None):
        self._apply_hist_zoom()

    def toggleLogScale(self, checked: bool):
        self.log_scale = bool(checked)
        self.log_toggle_button.setText(
            "X-Axis: Log  ✓" if checked else "X-Axis: Linear"
        )
        self.drawHistogram()

    def toggleHistogramMode(self):
        if self.histogram_mode == "PSF":
            self.histogram_mode = "Flux"
            self.mode_toggle_button.setText("Show PSF Histogram")
            # Flux spans huge range — default to log
            self.log_toggle_button.setChecked(True)
        else:
            self.histogram_mode = "PSF"
            self.mode_toggle_button.setText("Show Flux Histogram")
            # PSF/HFR is compact — default back to linear
            self.log_toggle_button.setChecked(False)
        self.drawHistogram()

    def _show_processing(self, msg="Processing…"):
        if not hasattr(self, "_overlay") or self._overlay is None:
            self._overlay = _ProcessingOverlay(self.scroll_area)
            self._overlay.hide()
        self._overlay.setText(msg)
        self._overlay.resize(self.scroll_area.viewport().size())
        self._overlay.move(0, 0)
        self._overlay.show()
        self._overlay.raise_()

    def _hide_processing(self):
        if hasattr(self, "_overlay") and self._overlay is not None:
            self._overlay.hide()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if hasattr(self, "_overlay") and self._overlay is not None and self._overlay.isVisible():
            self._overlay.resize(self.scroll_area.viewport().size())
        # Redraw histogram to fill new viewport size
        if hasattr(self, "_base_hist_pm"):
            self.drawHistogram()

    # ------------------------------------------------------------------
    def drawHistogram(self):
        base_w = max(self.scroll_area.viewport().width(), 400)
        h      = max(self.scroll_area.viewport().height() - 20, 150)

        pix = QPixmap(base_w, h)
        pix.fill(Qt.GlobalColor.white)
        painter = QPainter(pix)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if self.star_list is None or len(self.star_list) == 0:
            data  = np.array([], dtype=float)
            edges = np.linspace(0, 1, 51)
            low, high = float(edges[0]), float(edges[-1])
        else:
            if self.histogram_mode == "PSF":
                data  = np.array(self.star_list["HFR"],  dtype=float)
                hi    = float(data.max()) if data.size else 7.5
                edges = np.linspace(0.0, max(hi, 1e-6), 51)
            else:
                data  = np.array(self.star_list["flux"], dtype=float)
                edges = (np.linspace(data.min(), data.max(), 51)
                         if data.size else np.linspace(0, 1, 51))
            low, high = float(edges[0]), float(edges[-1])

        if self.log_scale and high > max(low, 1e-9):
            low   = max(low, 1e-4)
            edges = np.logspace(np.log10(low), np.log10(high if high > low else low * 10), 51)
            lo_l  = np.log10(low)
            hi_l  = np.log10(high) if high > low else lo_l + 1.0
            def xfun(v):
                lv = np.log10(max(v, low))
                return int((lv - lo_l) / (hi_l - lo_l) * base_w) if hi_l > lo_l else 0
        else:
            def xfun(v):
                return int((v - low) / (high - low) * base_w) if high > low else 0

        hist = np.histogram(data, bins=edges)[0].astype(float)
        if hist.size and hist.max() > 0:
            hist /= hist.max()

        painter.setPen(QPen(Qt.GlobalColor.black))
        for i in range(len(hist)):
            x0 = xfun(float(edges[i]))
            x1 = xfun(float(edges[i + 1]))
            bw = max(x1 - x0, 1)
            bh = float(hist[i]) * h
            painter.drawRect(x0, int(h - bh), bw, int(bh))

        painter.setPen(QPen(Qt.GlobalColor.black, 2))
        painter.drawLine(0, h - 1, base_w, h - 1)
        painter.setFont(QFont("Arial", 10))

        ticks = (
            np.logspace(np.log10(max(low, 1e-4)), np.log10(max(high, low * 10)), 6)
            if self.log_scale and high > low
            else np.linspace(low, high, 6)
        )
        for t in ticks:
            x = xfun(float(t))
            painter.drawLine(x, h - 1, x, h - 6)
            painter.drawText(x - 28, h - 10, f"{t:.3f}" if self.log_scale else f"{t:.2f}")

        painter.end()
        self._base_hist_pm = pix
        self._apply_hist_zoom()
        self.updateStatistics()

    def updateStatistics(self):
        data_map = {}
        if self.star_list is not None and len(self.star_list) > 0:
            a   = np.array(self.star_list["a"], float)
            b   = np.array(self.star_list["b"], float)
            ecc = np.nan_to_num(np.sqrt(1 - (b / np.maximum(a, 1e-9)) ** 2))
            data_map["eccentricity"] = ecc
            for c in self.star_list.colnames:
                try:
                    data_map[c] = np.array(self.star_list[c], float)
                except Exception:
                    pass

        col_order = ["HFR", "FWHM", "eccentricity", "a", "b", "theta", "flux"]
        cols = [c for c in col_order if c in data_map]

        self.stats_table.setColumnCount(len(cols))
        self.stats_table.setHorizontalHeaderLabels(cols)
        self.stats_table.setRowCount(4)
        # Row order: Median, Min, Max, StdDev
        self.stats_table.setVerticalHeaderLabels(["Median", "Min", "Max", "StdDev"])

        for ci, col in enumerate(cols):
            arr  = data_map.get(col, np.zeros(0, dtype=float))
            vals = (
                [np.median(arr), np.min(arr), np.max(arr), np.std(arr)]
                if arr.size else [0.0, 0.0, 0.0, 0.0]
            )
            for ri, v in enumerate(vals):
                it = QTableWidgetItem(f"{v:.3f}")
                it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.stats_table.setItem(ri, ci, it)

        self.stats_table.resizeColumnsToContents()

    # ------------------------------------------------------------------
    def _cleanup(self):
        try:
            if getattr(self, "threshold_timer", None) is not None:
                self.threshold_timer.stop()
        except Exception:
            pass
        try:
            if self._doc_conn and hasattr(self.doc, "changed"):
                self.doc.changed.disconnect(self._on_doc_changed)
        except Exception:
            pass
        self._doc_conn = False
        try:
            thr = getattr(self, "_psf_thread", None)
            wkr = getattr(self, "_psf_worker", None)
            if wkr is not None:
                try: wkr.deleteLater()
                except Exception: pass
            if thr is not None:
                try: thr.requestInterruption()
                except Exception: pass
                try: thr.quit()
                except Exception: pass
                try: thr.wait(250)
                except Exception: pass
                try: thr.deleteLater()
                except Exception: pass
        except Exception:
            pass
        self._psf_worker = None
        self._psf_thread = None

    def closeEvent(self, e):
        self._cleanup()
        super().closeEvent(e)