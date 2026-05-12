# pro/psf_viewer.py
from __future__ import annotations

import math
import numpy as np
import sep
from astropy.table import Table

from PyQt6.QtCore import Qt, QTimer, QRectF, QPointF
from PyQt6.QtGui import QPainter, QPen, QFont, QPixmap, QColor, QBrush, QImage
from PyQt6.QtWidgets import (
    QDialog, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QScrollArea,
    QSlider, QTableWidget, QTableWidgetItem, QApplication,
    QSizePolicy,
)
from setiastro.saspro.widgets.themed_buttons import themed_toolbtn

from PyQt6.QtCore import QThread, pyqtSignal, QObject
from PyQt6.QtWidgets import QWidget


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
        self.setMinimumSize(180, 180)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._cached_pm: QPixmap | None = None
        self._a     = 2.0
        self._b     = 1.8
        self._theta = 0.0
        self._fwhm  = 4.7
        self._hfr   = 4.0
        self._valid = False

    def set_star(self, a: float, b: float, theta: float, fwhm: float, hfr: float):
        self._a     = max(float(a),    0.1)
        self._b     = max(float(b),    0.1)
        self._theta = float(theta)
        self._fwhm  = max(float(fwhm), 0.5)
        self._hfr   = max(float(hfr),  0.5)
        self._valid = True
        self._rebuild_cache()
        self.update()

    def clear(self):
        self._valid = False
        self._cached_pm = None
        self.update()

    # ------------------------------------------------------------------
    def _rebuild_cache(self):
        """Render the star at _RENDER_SIZE into a QPixmap. Called once per new star data."""
        N = self._RENDER_SIZE
        cx = cy = N / 2.0

        # --- Gaussian blob on numpy grid ----------------------------------
        scale = (N * 0.16) / max(self._a, self._b)
        scale = max(scale, 2.0)

        a_px = self._a * scale
        b_px = self._b * scale
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
        qi = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)
        base_pm = QPixmap.fromImage(qi)

        # --- Paint overlays onto the pixmap --------------------------------
        pm = QPixmap(N, N)
        pm.fill(QColor(30, 30, 40))
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw gaussian blob
        p.drawPixmap(0, 0, base_pm)

        # HFR ellipse (orange) — rotated to match star orientation
        hfr_a = (self._hfr / 2.0) * scale
        hfr_b = hfr_a * (self._b / max(self._a, 1e-9))
        p.save()
        p.translate(cx, cy)
        p.rotate(-math.degrees(theta))
        p.setPen(QPen(QColor(255, 140, 0), 1.8, Qt.PenStyle.SolidLine))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(QRectF(-hfr_a, -hfr_b, hfr_a * 2, hfr_b * 2))
        p.restore()

        # FWHM ellipse (green) — rotated to match star orientation
        fwhm_a = (self._fwhm / 2.0) * scale
        fwhm_b = fwhm_a * (self._b / max(self._a, 1e-9))
        p.save()
        p.translate(cx, cy)
        p.rotate(-math.degrees(theta))
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
        dy_a  = -arm_a * sin_t
        p.setPen(QPen(QColor(220, 60, 60), 2.0))
        p.drawLine(QPointF(cx - dx_a, cy - dy_a), QPointF(cx + dx_a, cy + dy_a))

        # Minor axis — blue
        arm_b = b_px * 2.4
        dx_b  = -arm_b * sin_t
        dy_b  = -arm_b * cos_t
        p.setPen(QPen(QColor(80, 120, 220), 2.0))
        p.drawLine(QPointF(cx - dx_b, cy - dy_b), QPointF(cx + dx_b, cy + dy_b))

        # Legend + eccentricity readout
        p.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        lx, ly, lh = 8, 14, 17
        p.setPen(QColor(220, 60, 60));  p.drawText(lx, ly,        "PSF X")
        p.setPen(QColor(80, 120, 220)); p.drawText(lx, ly + lh,   "PSF Y")
        p.setPen(QColor(80, 200, 80));  p.drawText(lx, ly + lh*2, "FWHM")
        p.setPen(QColor(255, 140, 0));  p.drawText(lx, ly + lh*3, "HFR")

        ecc = math.sqrt(1.0 - (self._b / max(self._a, 1e-9)) ** 2)
        p.setFont(QFont("Segoe UI", 8))
        p.setPen(QColor(200, 200, 200))
        p.drawText(8, N - 8, f"ecc: {ecc:.3f}")

        p.end()
        self._cached_pm = pm

    def paintEvent(self, event):
        if self._cached_pm is None:
            # No data — draw placeholder
            p = QPainter(self)
            p.fillRect(self.rect(), QColor(30, 30, 40))
            p.setPen(QColor(100, 100, 120))
            p.setFont(QFont("Segoe UI", 9))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No stars\ndetected")
            p.end()
            return

        # Scale cached pixmap to widget size (fast — no computation)
        scaled = self._cached_pm.scaled(
            self.width(), self.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        p = QPainter(self)
        # Center in widget
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

    def __init__(self, image: np.ndarray, threshold_sigma: float):
        super().__init__()
        self.image = image
        self.threshold_sigma = float(threshold_sigma)

    def run(self):
        try:
            if self.image is None:
                self.finished.emit(None, "Status: No image.")
                return

            if self.image.ndim == 3:
                image_gray = np.mean(self.image, axis=2)
            else:
                image_gray = self.image
            data = image_gray.astype(np.float32, copy=False)

            bkg = sep.Background(data)
            data_sub = data - bkg.back()
            try:
                err_val = bkg.globalrms
            except Exception:
                err_val = float(np.median(bkg.rms()))

            sources = sep.extract(data_sub, self.threshold_sigma, err=err_val)
            if sources is None or len(sources) == 0:
                self.finished.emit(None, "Status: Extraction completed — 0 sources.")
                return

            a_arr = np.array(sources["a"], dtype=np.float32)
            tbl = Table()
            tbl["xcentroid"] = sources["x"]
            tbl["ycentroid"] = sources["y"]
            tbl["flux"]      = sources["flux"]
            tbl["HFR"]       = 2.0 * a_arr
            tbl["FWHM"]      = 2.3548 * a_arr
            tbl["a"]         = a_arr
            tbl["b"]         = sources["b"]
            tbl["theta"]     = sources["theta"]

            self.finished.emit(tbl, f"Status: Extraction completed — {len(sources)} sources.")
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

        # Histogram
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setFixedSize(520, 230)
        self.scroll_area.setWidgetResizable(False)
        self.hist_label = QLabel(self)
        self.hist_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scroll_area.setWidget(self.hist_label)
        top_layout.addWidget(self.scroll_area)

        # Star graphic column
        star_col = QVBoxLayout()
        star_col.setSpacing(4)
        star_col.setAlignment(Qt.AlignmentFlag.AlignTop)
        star_lbl = QLabel("Median Star Profile")
        star_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        star_lbl.setStyleSheet("font-size: 11px; color: #aaa;")
        self._star_widget = _StarWidget(self)
        self._star_widget.setFixedSize(200, 200)
        star_col.addWidget(star_lbl)
        star_col.addWidget(self._star_widget)
        # No addStretch() — Fixed size policy handles it
        top_layout.addLayout(star_col)
        top_layout.setAlignment(star_col, Qt.AlignmentFlag.AlignTop)

        main_layout.addLayout(top_layout)

        # ── Stats table — tall enough to avoid scrollbar ────────────────
        self.stats_table = QTableWidget(self)
        self.stats_table.setRowCount(4)
        self.stats_table.setColumnCount(0)
        # Order: Median first, then Min / Max / StdDev
        self.stats_table.setVerticalHeaderLabels(["Median", "Min", "Max", "StdDev"])
        # 4 rows × ~26px + header ~30px + 2px border = ~138px
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
        self._psf_worker = _PSFWorker(self.image, self.detection_threshold)
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
            a     = float(np.median(np.array(self.star_list["a"],     dtype=float)))
            b     = float(np.median(np.array(self.star_list["b"],     dtype=float)))
            theta = float(np.median(np.array(self.star_list["theta"], dtype=float)))
            fwhm  = float(np.median(np.array(self.star_list["FWHM"],  dtype=float)))
            hfr   = float(np.median(np.array(self.star_list["HFR"],   dtype=float)))
            self._star_widget.set_star(a, b, theta, fwhm, hfr)
        except Exception:
            self._star_widget.clear()

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

    # ------------------------------------------------------------------
    def drawHistogram(self):
        base_w, h = 512, 210

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
                edges = np.linspace(0, 7.5, 51)
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