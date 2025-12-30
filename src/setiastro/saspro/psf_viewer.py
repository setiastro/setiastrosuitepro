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
from setiastro.saspro.widgets.themed_buttons import themed_toolbtn

from PyQt6.QtCore import QThread, pyqtSignal, QObject
from PyQt6.QtWidgets import QWidget

class _ProcessingOverlay(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setStyleSheet("""
            QWidget {
                background: rgba(0,0,0,140);
                border-radius: 10px;
            }
            QLabel {
                color: white;
                font-size: 14px;
                font-weight: 600;
            }
        """)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 14, 18, 14)
        self.lbl = QLabel("Processing…", self)
        self.lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self.lbl)

    def setText(self, s: str):
        self.lbl.setText(s)

class _PSFWorker(QObject):
    finished = pyqtSignal(object, str)   # (Table or None, status_text)
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

            # grayscale
            if self.image.ndim == 3:
                image_gray = np.mean(self.image, axis=2)
            else:
                image_gray = self.image
            data = image_gray.astype(np.float32, copy=False)

            # background
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

            # HFR proxy
            try:
                r = 2.0 * sources["a"]
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

            self.finished.emit(tbl, f"Status: Extraction completed — {len(sources)} sources.")
        except Exception as e:
            self.failed.emit(f"Extraction failed: {e}")


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
        try:
            self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        except Exception:
            pass  # older PyQt6 versions
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

        self._psf_thread = None
        self._psf_worker = None
        self._doc_conn = False
        if hasattr(self.doc, "changed"):
            try:
                self.doc.changed.connect(self._on_doc_changed)
                self._doc_conn = True
            except Exception:
                self._doc_conn = False

        # cleanup no matter how the dialog is dismissed (accept/reject/done)
        self.finished.connect(self._cleanup)
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
        # reuse the existing debounce timer instead of immediate recompute
        if self.threshold_timer.isActive():
            self.threshold_timer.stop()
        self.threshold_timer.start()

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

        # themed zoom buttons
        btn_zoom_out = themed_toolbtn("zoom-out", "Zoom Out")
        btn_zoom_in  = themed_toolbtn("zoom-in", "Zoom In")
        btn_fit      = themed_toolbtn("zoom-fit-best", "Fit")

        btn_zoom_out.clicked.connect(lambda: self._step_zoom(1/1.25))
        btn_zoom_in.clicked.connect(lambda: self._step_zoom(1.25))
        btn_fit.clicked.connect(self._fit_histogram)

        controls_layout.addWidget(btn_zoom_out)
        controls_layout.addWidget(btn_zoom_in)
        controls_layout.addWidget(btn_fit)

        # keep the slider (nice for big jumps)
        self.zoom_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.zoom_slider.setRange(50, 1000)
        self.zoom_slider.setValue(100)
        self.zoom_slider.setTickInterval(10)
        self.zoom_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.zoom_slider.valueChanged.connect(self.updateZoom)
        controls_layout.addWidget(self.zoom_slider, 1)

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
        close_btn.clicked.connect(self.close)
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

    def _step_zoom(self, factor: float):
        v = int(round(self.zoom_slider.value() * factor))
        v = max(self.zoom_slider.minimum(), min(self.zoom_slider.maximum(), v))
        self.zoom_slider.setValue(v)  # drives updateZoom()

    def _fit_histogram(self):
        # Fit the histogram pixmap to the scroll viewport width.
        # Keeps behavior consistent with your other preview dialogs.
        if not hasattr(self, "_base_hist_pm") or self._base_hist_pm is None:
            return
        vp_w = self.scroll_area.viewport().width()
        base_w = max(1, self._base_hist_pm.width())
        z = vp_w / base_w
        self.zoom_slider.setValue(int(round(z * 100)))

    def _apply_hist_zoom(self):
        if not hasattr(self, "_base_hist_pm") or self._base_hist_pm is None:
            return
        z = self.zoom_slider.value() / 100.0
        w = max(1, int(self._base_hist_pm.width()  * z))
        h = max(1, int(self._base_hist_pm.height() * z))
        scaled = self._base_hist_pm.scaled(
            w, h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.hist_label.setPixmap(scaled)
        self.hist_label.resize(scaled.size())

    def _applyThreshold(self):
        if self.image is None:
            self.star_list = None
            self.status_label.setText("Status: No image.")
            self.drawHistogram()
            return

        self._show_processing("Processing… extracting stars / PSFs")

        # stop any previous run cleanly
        self._stop_psf_worker()

        self._psf_thread = QThread(self)
        self._psf_worker = _PSFWorker(self.image, self.detection_threshold)
        self._psf_worker.moveToThread(self._psf_thread)

        self._psf_thread.started.connect(self._psf_worker.run)
        self._psf_worker.finished.connect(self._on_psf_done)
        self._psf_worker.failed.connect(self._on_psf_fail)

        # ensure thread quits once worker reports anything
        self._psf_worker.finished.connect(lambda *_: self._stop_psf_worker(quit_only=False))
        self._psf_worker.failed.connect(lambda *_: self._stop_psf_worker(quit_only=False))


        self._psf_thread.start()

    def _stop_psf_worker(self, quit_only: bool = False):
        thr = getattr(self, "_psf_thread", None)
        wkr = getattr(self, "_psf_worker", None)

        if thr is None:
            return

        try:
            thr.quit()
        except Exception:
            pass
        try:
            thr.wait(250)
        except Exception:
            pass

        if not quit_only:
            try:
                if wkr is not None:
                    wkr.deleteLater()
            except Exception:
                pass
            try:
                thr.deleteLater()
            except Exception:
                pass
            self._psf_worker = None
            self._psf_thread = None

    def _on_psf_done(self, tbl, status: str):
        # tbl is an astropy Table or None
        self.star_list = tbl
        self.status_label.setText(status)
        self._hide_processing()
        self.drawHistogram()

    def _on_psf_fail(self, msg: str):
        self.star_list = None
        self.status_label.setText(f"Status: {msg}")
        self._hide_processing()
        self.drawHistogram()


    def updateImage(self, new_image):
        self.image = np.asarray(new_image) if new_image is not None else None
        if self.threshold_timer.isActive():
            self.threshold_timer.stop()
        self.threshold_timer.start()

    def updateZoom(self, _=None):
        self._apply_hist_zoom()


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

        # Render at fixed base resolution (no zoom here)
        pix = QPixmap(base_w, h)
        pix.fill(Qt.GlobalColor.white)

        painter = QPainter(pix)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Prepare data
        if self.star_list is None or len(self.star_list) == 0:
            data = np.array([], dtype=float)
            edges = np.linspace(0, 1, 51)
            low, high = float(edges[0]), float(edges[-1])
        else:
            if self.histogram_mode == "PSF":
                data = np.array(self.star_list["HFR"], dtype=float)
                edges = np.linspace(0, 7.5, 51)
            else:
                data = np.array(self.star_list["flux"], dtype=float)
                edges = np.linspace(data.min(), data.max(), 51) if data.size else np.linspace(0, 1, 51)
            low, high = float(edges[0]), float(edges[-1])

        # Axis scale helpers (map value -> x in [0..base_w])
        if self.log_scale and high > max(low, 1e-9):
            low = max(low, 1e-4)
            edges = np.logspace(np.log10(low), np.log10(high if high > low else low * 10), 51)

            lo_l = np.log10(low)
            hi_l = np.log10(high) if high > low else lo_l + 1.0

            def xfun(v: float) -> int:
                lv = np.log10(max(v, low))
                return int((lv - lo_l) / (hi_l - lo_l) * base_w) if hi_l > lo_l else 0
        else:
            def xfun(v: float) -> int:
                return int((v - low) / (high - low) * base_w) if high > low else 0

        # Histogram
        hist = np.histogram(data, bins=edges)[0].astype(float)
        if hist.size and hist.max() > 0:
            hist /= hist.max()

        # Bars
        painter.setPen(QPen(Qt.GlobalColor.black))
        for i in range(len(hist)):
            x0 = xfun(float(edges[i]))
            x1 = xfun(float(edges[i + 1]))
            bw = max(x1 - x0, 1)
            bh = float(hist[i]) * h
            painter.drawRect(x0, int(h - bh), bw, int(bh))

        # X axis
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

        # Store base pixmap for zooming
        self._base_hist_pm = pix
        self._apply_hist_zoom()      # scales into hist_label
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


    def _cleanup(self):
        # stop debounce timer
        try:
            if getattr(self, "threshold_timer", None) is not None:
                self.threshold_timer.stop()
        except Exception:
            pass

        # disconnect doc listener
        try:
            if self._doc_conn and hasattr(self.doc, "changed"):
                self.doc.changed.disconnect(self._on_doc_changed)
        except Exception:
            pass
        self._doc_conn = False

        # stop worker/thread
        try:
            thr = getattr(self, "_psf_thread", None)
            wkr = getattr(self, "_psf_worker", None)

            if wkr is not None:
                try:
                    wkr.deleteLater()
                except Exception:
                    pass

            if thr is not None:
                try:
                    thr.requestInterruption()
                except Exception:
                    pass
                try:
                    thr.quit()
                except Exception:
                    pass
                try:
                    thr.wait(250)
                except Exception:
                    pass
                try:
                    thr.deleteLater()
                except Exception:
                    pass
        except Exception:
            pass

        self._psf_worker = None
        self._psf_thread = None

    # ---------- lifecycle ----------
    def closeEvent(self, e):
        self._cleanup()
        super().closeEvent(e)

