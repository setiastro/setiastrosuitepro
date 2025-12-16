# pro/psf_viewer.py
from __future__ import annotations

import numpy as np
import sep
from astropy.table import Table

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPainter, QPen, QFont, QPixmap
from PyQt6.QtWidgets import (
    QDialog, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QScrollArea,
    QSlider, QTableWidget, QTableWidgetItem, QApplication, QMessageBox
)
from pro.widgets.themed_buttons import themed_toolbtn

class PSFViewer(QDialog):
    """
    A lightweight PSF/Flux histogram viewer.
    Pass an ImageSubWindow instance *or* a document (with .image and .changed).
    Listens to doc.changed to keep results fresh.
    """
    def __init__(self, view_or_doc, parent=None):
        super().__init__(parent)
        self.setWindowTitle("PSF Viewer")

        # Accept either a view (with .document) or a doc directly
        doc = getattr(view_or_doc, "document", None)
        self.doc = doc if doc is not None else view_or_doc

        # Image + state
        self.image = self._grab_image()
        self.zoom_factor = 1.0
        self.log_scale = False
        self.star_list = None
        self.histogram_mode = "PSF"  # or "Flux"
        self.detection_threshold = 5  # sigma

        # Debounce timer for threshold slider
        self.threshold_timer = QTimer(self)
        self.threshold_timer.setSingleShot(True)
        self.threshold_timer.setInterval(500)
        self.threshold_timer.timeout.connect(self._applyThreshold)

        # Auto-update when the document changes
        if hasattr(self.doc, "changed"):
            try:
                self.doc.changed.connect(self._on_doc_changed)
            except Exception:
                pass

        self._build_ui()
        # Defer first compute until after the dialog is shown/layouted
        QTimer.singleShot(0, self._applyThreshold)

    # ---------- internals ----------
    def _grab_image(self):
        img = getattr(self.doc, "image", None)
        if img is None:
            return None
        # Ensure ndarray
        try:
            return np.asarray(img)
        except Exception:
            return None

    def _on_doc_changed(self, *_):
        self.image = self._grab_image()
        self.compute_star_list()
        self.drawHistogram()

    # ---------- UI ----------
    def _build_ui(self):
        main_layout = QVBoxLayout(self)

        # Top: histogram + stats
        top_layout = QHBoxLayout()
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setFixedSize(520, 310)
        self.scroll_area.setWidgetResizable(False)
        self.hist_label = QLabel(self)
        self.hist_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scroll_area.setWidget(self.hist_label)
        top_layout.addWidget(self.scroll_area)

        self.stats_table = QTableWidget(self)
        self.stats_table.setRowCount(4)
        self.stats_table.setColumnCount(0)
        self.stats_table.setVerticalHeaderLabels(["Min", "Max", "Median", "StdDev"])
        self.stats_table.setFixedWidth(360)
        top_layout.addWidget(self.stats_table)
        main_layout.addLayout(top_layout)

        self.status_label = QLabel("Status: Ready", self)
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.status_label)

        # Controls
        controls_layout = QHBoxLayout()

        controls_layout.addWidget(QLabel("Zoom:"))
        self.zoom_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.zoom_slider.setRange(50, 1000)
        self.zoom_slider.setValue(100)
        self.zoom_slider.setTickInterval(10)
        self.zoom_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.zoom_slider.valueChanged.connect(self.updateZoom)
        controls_layout.addWidget(self.zoom_slider)

        self.log_toggle_button = QPushButton("Toggle Log X-Axis", self)
        self.log_toggle_button.setCheckable(True)
        self.log_toggle_button.setToolTip("Toggle between linear and logarithmic x-axis.")
        self.log_toggle_button.toggled.connect(self.toggleLogScale)
        controls_layout.addWidget(self.log_toggle_button)

        self.mode_toggle_button = QPushButton("Show Flux Histogram", self)
        self.mode_toggle_button.setToolTip("Switch between PSF (HFR) and Flux histograms.")
        self.mode_toggle_button.clicked.connect(self.toggleHistogramMode)
        controls_layout.addWidget(self.mode_toggle_button)

        main_layout.addLayout(controls_layout)

        # Threshold
        thresh_layout = QHBoxLayout()
        thresh_layout.addWidget(QLabel("Detection Threshold (σ):", self))
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.threshold_slider.setRange(1, 20)
        self.threshold_slider.setValue(self.detection_threshold)
        self.threshold_slider.setTickInterval(1)
        self.threshold_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.threshold_slider.valueChanged.connect(self.onThresholdChange)
        thresh_layout.addWidget(self.threshold_slider)

        self.threshold_value_label = QLabel(str(self.detection_threshold), self)
        thresh_layout.addWidget(self.threshold_value_label)
        main_layout.addLayout(thresh_layout)

        # Close
        close_btn = QPushButton("Close", self)
        close_btn.clicked.connect(self.accept)
        main_layout.addWidget(close_btn)

        self.setLayout(main_layout)
        self.drawHistogram()

    # ---------- interactions ----------
    def onThresholdChange(self, value: int):
        self.detection_threshold = int(value)
        self.threshold_value_label.setText(str(value))
        if self.threshold_timer.isActive():
            self.threshold_timer.stop()
        self.threshold_timer.start()

    def _applyThreshold(self):
        self.compute_star_list()
        self.drawHistogram()

    def updateImage(self, new_image):
        self.image = np.asarray(new_image) if new_image is not None else None
        self.compute_star_list()
        self.drawHistogram()

    def updateZoom(self, val: int):
        self.zoom_factor = max(0.1, val / 100.0)
        self.drawHistogram()

    def toggleLogScale(self, checked: bool):
        self.log_scale = bool(checked)
        self.drawHistogram()

    def toggleHistogramMode(self):
        if self.histogram_mode == "PSF":
            self.histogram_mode = "Flux"
            self.mode_toggle_button.setText("Show PSF Histogram")
        else:
            self.histogram_mode = "PSF"
            self.mode_toggle_button.setText("Show Flux Histogram")
        self.drawHistogram()

    # ---------- compute ----------
    def compute_star_list(self):
        if self.image is None:
            self.status_label.setText("Status: No image.")
            self.star_list = None
            return

        # Convert to grayscale
        if self.image.ndim == 3:
            image_gray = np.mean(self.image, axis=2)
        else:
            image_gray = self.image
        data = image_gray.astype(np.float32, copy=False)

        # Background
        try:
            bkg = sep.Background(data)
            data_sub = data - bkg.back()
            try:
                err_val = bkg.globalrms
            except Exception:
                err_val = float(np.median(bkg.rms()))
        except Exception as e:
            self.status_label.setText(f"Status: Background failed: {e}")
            self.star_list = None
            return

        threshold = float(self.detection_threshold)

        self.status_label.setText("Status: Starting star extraction...")
        QApplication.processEvents()

        try:
            sources = sep.extract(data_sub, threshold, err=err_val)
            n = len(sources) if sources is not None else 0
            self.status_label.setText(f"Status: Extraction completed — {n} sources.")
        except Exception as e:
            self.status_label.setText(f"Status: Extraction failed: {e}")
            sources = None

        QApplication.processEvents()

        if sources is None or len(sources) == 0:
            self.star_list = None
            return

        # HFR (quick proxy): 2 * a  (a ≈ semi-major Gaussian sigma in pixels for SEP)
        try:
            a = sources["a"]
            r = 2 * a
        except Exception:
            r = np.zeros(len(sources), dtype=np.float32)

        tbl = Table()
        tbl["xcentroid"] = sources["x"]
        tbl["ycentroid"] = sources["y"]
        tbl["flux"] = sources["flux"]
        tbl["HFR"] = r
        tbl["a"] = sources["a"]
        tbl["b"] = sources["b"]
        tbl["theta"] = sources["theta"]
        self.star_list = tbl

    # ---------- drawing ----------
    def drawHistogram(self):
        base_w, h = 512, 300
        w = max(64, int(base_w * self.zoom_factor))
        pix = QPixmap(w, h)
        pix.fill(Qt.GlobalColor.white)

        painter = QPainter(pix)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Prepare data
        if self.star_list is None or len(self.star_list) == 0:
            data = np.array([], dtype=float)
            edges = np.linspace(0, 1, 51)
            low, high = edges[0], edges[-1]
        else:
            if self.histogram_mode == "PSF":
                data = np.array(self.star_list["HFR"], dtype=float)
                edges = np.linspace(0, 7.5, 51)
            else:
                data = np.array(self.star_list["flux"], dtype=float)
                edges = np.linspace(data.min(), data.max(), 51) if data.size else np.linspace(0, 1, 51)
            low, high = float(edges[0]), float(edges[-1])

        # Axis scale
        if self.log_scale and high > max(low, 1e-9):
            low = max(low, 1e-4)
            edges = np.logspace(np.log10(low), np.log10(high if high > low else low * 10), 51)
            def xfun(v: float) -> int:
                lv = np.log10(max(v, low))
                return int((lv - np.log10(low)) / (np.log10(high) - np.log10(low)) * w) if high > low else 0
        else:
            def xfun(v: float) -> int:
                return int((v - low) / (high - low) * w) if high > low else 0

        # Histogram
        hist = np.histogram(data, bins=edges)[0].astype(float)
        if hist.size and hist.max() > 0:
            hist /= hist.max()

        # Bars
        painter.setPen(QPen(Qt.GlobalColor.black))
        for i in range(len(hist)):
            x0 = xfun(edges[i])
            x1 = xfun(edges[i + 1])
            bw = max(x1 - x0, 1)
            bh = float(hist[i]) * h
            painter.drawRect(x0, int(h - bh), bw, int(bh))

        # X axis
        painter.setPen(QPen(Qt.GlobalColor.black, 2))
        painter.drawLine(0, h - 1, w, h - 1)
        painter.setFont(QFont("Arial", 10))

        ticks = (np.logspace(np.log10(max(low, 1e-4)), np.log10(max(high, low * 10)), 6)
                 if self.log_scale and high > low
                 else np.linspace(low, high, 6))
        for t in ticks:
            x = xfun(float(t))
            painter.drawLine(x, h - 1, x, h - 6)
            painter.drawText(x - 28, h - 10, f"{t:.3f}" if self.log_scale else f"{t:.2f}")

        painter.end()
        self.hist_label.setPixmap(pix)
        self.hist_label.resize(pix.size())
        self.updateStatistics()

    def updateStatistics(self):
        data_map = {}
        if self.star_list is not None and len(self.star_list) > 0:
            cols = ["HFR", "eccentricity", "a", "b", "theta", "flux"]
            a = np.array(self.star_list["a"], float)
            b = np.array(self.star_list["b"], float)
            ecc = np.nan_to_num(np.sqrt(1 - (b / np.maximum(a, 1e-9)) ** 2))
            data_map["eccentricity"] = ecc
            for c in self.star_list.colnames:
                try:
                    data_map[c] = np.array(self.star_list[c], float)
                except Exception:
                    pass
            cols = [c for c in cols if c in data_map]
        else:
            cols = []

        self.stats_table.setColumnCount(len(cols))
        self.stats_table.setHorizontalHeaderLabels(cols)
        self.stats_table.setRowCount(4)
        self.stats_table.setVerticalHeaderLabels(["Min", "Max", "Median", "StdDev"])

        for ci, col in enumerate(cols):
            arr = data_map.get(col, np.zeros(0, dtype=float))
            if arr.size:
                vals = [np.min(arr), np.max(arr), np.median(arr), np.std(arr)]
            else:
                vals = [0.0, 0.0, 0.0, 0.0]
            for ri, v in enumerate(vals):
                it = QTableWidgetItem(f"{v:.3f}")
                it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.stats_table.setItem(ri, ci, it)

    # ---------- lifecycle ----------
    def closeEvent(self, e):
        # Best-effort disconnect
        if hasattr(self.doc, "changed"):
            try:
                self.doc.changed.disconnect(self._on_doc_changed)
            except Exception:
                pass
        super().closeEvent(e)
