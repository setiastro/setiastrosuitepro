import os
import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QHBoxLayout, QLineEdit, QPushButton, QFileDialog,
    QListWidget, QSlider, QCheckBox, QMessageBox, QTextEdit, QDialog, QApplication,
    QTreeWidget, QTreeWidgetItem, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGridLayout,
    QToolBar, QSizePolicy, QSpinBox, QDoubleSpinBox, QProgressBar
)
from PyQt6.QtGui import QImage, QPixmap, QIcon, QPainter, QAction, QTransform, QCursor
from PyQt6.QtCore import Qt, pyqtSignal, QRectF, QPointF, QTimer, QThread, QObject


from pathlib import Path
import tempfile

from astropy.wcs import WCS
from astropy.time import Time
from astropy import units as u


from legacy.image_manager import load_image, save_image
from numba_utils import bulk_cosmetic_correction_numba
from imageops.stretch import stretch_mono_image, stretch_color_image
from pro.star_alignment import PolyGradientRemoval 
from pro import minorbodycatalog as mbc
from pro.plate_solver import PlateSolverDialog as PlateSolver

from pro.plate_solver import (
    _solve_numpy_with_fallback,
    _as_header,
    _strip_wcs_keys,
    _merge_wcs_into_base_header,
)

def _xisf_kw_value(xisf_meta: dict, key: str, default=None):
    """
    Return the first 'value' for FITSKeywords[key] from a XISF meta dict.

    xisf_meta: the dict stored in doc.metadata["xisf_meta"]
    """
    if not xisf_meta:
        return default

    fk = xisf_meta.get("FITSKeywords", {})
    if key not in fk:
        return default

    entry = fk[key]
    # In your sample, it's a list of {"value": "...", "comment": "..."}
    if isinstance(entry, list) and entry:
        v = entry[0].get("value", default)
    elif isinstance(entry, dict):
        v = entry.get("value", default)
    else:
        v = entry
    return v

def ensure_jd_from_xisf_meta(meta: dict) -> None:
    """
    If this document came from a XISF and we haven't stored a JD yet,
    derive JD / MJD from XISF FITSKeywords (DATE-OBS + EXPOSURE).

    Safe no-op if anything is missing.
    """
    # Already have it? Don't overwrite.
    if "jd" in meta and np.isfinite(meta["jd"]):
        return

    xisf_meta = meta.get("xisf_meta")
    if not isinstance(xisf_meta, dict):
        return

    # 1) Get UTC observation timestamp and exposure
    date_obs = _xisf_kw_value(xisf_meta, "DATE-OBS")
    if not date_obs:
        # Optional fallback to local time if you *really* want:
        # date_obs = _xisf_kw_value(xisf_meta, "DATE-LOC")
        return

    exp_str = (_xisf_kw_value(xisf_meta, "EXPOSURE") or
               _xisf_kw_value(xisf_meta, "EXPTIME"))
    exposure = None
    if exp_str is not None:
        try:
            exposure = float(exp_str)
        except Exception:
            exposure = None

    # 2) Parse the date string → Time
    # SGP / PI are emitting ISO8601 with fractional seconds: 2024-04-22T06:58:08.4217144
    try:
        t = Time(date_obs, format="isot", scale="utc")
    except Exception:
        # Last-resort: let astropy guess; if that fails, bail out
        try:
            t = Time(date_obs, scale="utc")
        except Exception:
            return

    # 3) Move to mid-exposure if we know the exposure length
    if exposure and exposure > 0:
        t = t + 0.5 * exposure * u.s

    # 4) Store JD/MJD for later minor-body prediction
    meta["jd"] = float(t.jd)
    meta["mjd"] = float(t.mjd)
    # Optional: keep a cleaned-up timestamp string too
    meta.setdefault("date_obs", t.isot)

def _numpy_to_qimage(img: np.ndarray) -> QImage:
    """
    Accepts:
      - float32/float64 in [0..1], mono or RGB
      - uint8 mono/RGB
    Returns QImage (RGB888 or Grayscale8).
    """
    if img is None:
        return QImage()

    # Normalize dtype
    if img.dtype != np.uint8:
        img = (np.clip(img, 0, 1) * 255.0).astype(np.uint8)

    if img.ndim == 2:
        h, w = img.shape
        return QImage(img.data, w, h, img.strides[0], QImage.Format.Format_Grayscale8)
    elif img.ndim == 3:
        h, w, c = img.shape
        if c == 3:
            # assume RGB
            return QImage(img.data, w, h, img.strides[0], QImage.Format.Format_RGB888)
        elif c == 4:
            return QImage(img.data, w, h, img.strides[0], QImage.Format.Format_RGBA8888)
        else:
            # collapse/expand as needed
            if c == 1:
                img = np.repeat(img, 3, axis=2)
                h, w, _ = img.shape
                return QImage(img.data, w, h, img.strides[0], QImage.Format.Format_RGB888)
    # fallback empty
    return QImage()

class MinorBodyWorker(QObject):
    """
    Runs the heavy minor-body prediction in a background thread.
    Does NOT touch any widgets directly.
    """
    finished = pyqtSignal(list, str)   # (bodies, error_message or "")
    progress = pyqtSignal(int, str)    # (percent, message)

    def __init__(self, owner, jd_for_calc: float):
        super().__init__()
        self._owner = owner      # SupernovaAsteroidHunterDialog
        self._jd = jd_for_calc

    def run(self):
        try:
            # Kick off with a low percentage
            self.progress.emit(0, "Minor-body search: preparing catalog query...")
            bodies = self._owner._get_predicted_minor_bodies_for_field(
                H_ast_max=self._owner.minor_H_ast_max,
                H_com_max=self._owner.minor_H_com_max,
                jd=self._jd,
                progress_cb=self.progress.emit,   # pass our signal as callback
            )
            if bodies is None:
                bodies = []
            self.finished.emit(bodies, "")
        except Exception as e:
            self.finished.emit([], str(e))

class ZoomableImageView(QGraphicsView):
    zoomChanged = pyqtSignal(float)  # emits current scale (1.0 = 100%)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self._pix = QGraphicsPixmapItem()
        self.scene().addItem(self._pix)
        self.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self._fit_mode = False
        self._scale = 1.0

    def set_image(self, np_img_rgb_or_gray_uint8_or_float):
        qimg = _numpy_to_qimage(np_img_rgb_or_gray_uint8_or_float)
        pix = QPixmap.fromImage(qimg)
        self._pix.setPixmap(pix)
        self.scene().setSceneRect(QRectF(pix.rect()))
        self.reset_view()

    def reset_view(self):
        self._fit_mode = False
        self._scale = 1.0
        self.setTransform(QTransform())
        self.centerOn(self._pix)
        self.zoomChanged.emit(self._scale)

    def fit_to_view(self):
        if self._pix.pixmap().isNull():
            return
        self._fit_mode = True
        self.setTransform(QTransform())
        self.fitInView(self._pix, Qt.AspectRatioMode.KeepAspectRatio)
        # derive scale from transform.m11
        self._scale = self.transform().m11()
        self.zoomChanged.emit(self._scale)

    def set_1to1(self):
        self._fit_mode = False
        self.setTransform(QTransform())
        self._scale = 1.0
        self.zoomChanged.emit(self._scale)

    def zoom(self, factor: float, anchor_pos: QPointF | None = None):
        if self._pix.pixmap().isNull():
            return
        self._fit_mode = False
        # clamp
        new_scale = self._scale * factor
        new_scale = max(0.05, min(32.0, new_scale))
        factor = new_scale / self._scale
        if abs(factor - 1.0) < 1e-6:
            return

        # zoom around cursor
        if anchor_pos is not None:
            self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        else:
            self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)

        self.scale(factor, factor)
        self._scale = new_scale
        self.zoomChanged.emit(self._scale)

    # --- input handling ---
    def wheelEvent(self, event):
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            delta = event.angleDelta().y()
            step = 1.25 if delta > 0 else 0.8
            self.zoom(step, anchor_pos=event.position())
            event.accept()
        else:
            super().wheelEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.viewport().setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.viewport().unsetCursor()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._fit_mode and not self._pix.pixmap().isNull():
            # keep image fitted when the window is resized
            # (doesn't steal state if user switched to manual zoom)
            self.fit_to_view()

class ImagePreviewWindow(QDialog):
    pushed = pyqtSignal(object, str)           # (numpy_image, title)
    minorBodySearchRequested = pyqtSignal()    # emitted when user clicks MB button

    def __init__(self, np_img_rgb_or_gray, title="Preview", parent=None, icon: QIcon | None = None):
        super().__init__(parent)
        self.setWindowTitle(title)
        if icon:
            self.setWindowIcon(icon)
        self._original = np_img_rgb_or_gray  # keep in memory to push upstream

        lay = QVBoxLayout(self)

        # toolbar
        tb = QToolBar(self)
        self.act_fit = QAction("Fit", self)
        self.act_1to1 = QAction("1:1", self)
        self.act_zoom_in = QAction("Zoom In", self)
        self.act_zoom_out = QAction("Zoom Out", self)
        self.act_push = QAction("Push to New View", self)
        #self.act_minor = QAction("Check Catalogued Minor Bodies in Field", self)

        self.act_zoom_in.setShortcut("Ctrl++")
        self.act_zoom_out.setShortcut("Ctrl+-")
        self.act_fit.setShortcut("F")
        self.act_1to1.setShortcut("1")

        tb.addAction(self.act_fit)
        tb.addAction(self.act_1to1)
        tb.addSeparator()
        tb.addAction(self.act_zoom_in)
        tb.addAction(self.act_zoom_out)
        tb.addSeparator()
        tb.addAction(self.act_push)
        #tb.addSeparator()
        #tb.addAction(self.act_minor)

        # zoom label spacer
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        tb.addWidget(spacer)
        self._zoom_label = QLabel("100%")
        tb.addWidget(self._zoom_label)

        lay.addWidget(tb)

        # view
        self.view = ZoomableImageView(self)
        lay.addWidget(self.view)
        self.view.set_image(np_img_rgb_or_gray)
        self.view.zoomChanged.connect(self._on_zoom_changed)

        # connect actions
        self.act_fit.triggered.connect(self.view.fit_to_view)
        self.act_1to1.triggered.connect(self.view.set_1to1)
        self.act_zoom_in.triggered.connect(lambda: self.view.zoom(1.25))
        self.act_zoom_out.triggered.connect(lambda: self.view.zoom(0.8))
        self.act_push.triggered.connect(self._on_push)
        #self.act_minor.triggered.connect(self._on_minor_body_search)

        # start in "Fit"
        self.view.fit_to_view()
        self.resize(900, 700)

    def _on_zoom_changed(self, s: float):
        self._zoom_label.setText(f"{round(s*100)}%")

    def _on_push(self):
        # Emit the original (float or uint8) image up to the parent/dialog
        self.pushed.emit(self._original, self.windowTitle())
        QMessageBox.information(self, "Pushed", "New View Created.")

    def _on_minor_body_search(self):
        # Just emit a signal; the parent dialog will handle the heavy lifting.
        self.minorBodySearchRequested.emit()

    def showEvent(self, e):
        super().showEvent(e)
        # Defer one tick so the view has its final size
        QTimer.singleShot(0, self.view.fit_to_view)


class SupernovaAsteroidHunterDialog(QDialog):
    def __init__(self, parent=None, settings=None,
                 image_manager=None, doc_manager=None,
                 supernova_path=None, wrench_path=None, spinner_path=None):
        super().__init__(parent)
        self.setWindowTitle("Supernova / Asteroid Hunter")
        if supernova_path:
            self.setWindowIcon(QIcon(supernova_path))
        # keep icon path for previews
        self.supernova_path = supernova_path

        self.settings = settings
        self.image_manager = image_manager
        self.doc_manager = doc_manager

        # one layout for the dialog
        self.setLayout(QVBoxLayout())

        # state
        self.parameters = {
            "referenceImagePath": "",
            "searchImagePaths": [],
            "threshold": 0.10
        }
        self.preprocessed_reference = None
        self.preprocessed_search = []
        self.anomalyData = []

        # WCS / timing / minor bodies
        self.ref_header = None
        self.ref_wcs = None
        self.ref_jd = None
        self.ref_site = None  # you can fill this from settings later
        self.predicted_minor_bodies = None

        # default H limits for minor bodies (you can later expose via UI)
        self.minor_H_ast_max = 20.0
        self.minor_H_com_max = 15.0
        self.minor_ast_max_count = 50000
        self.minor_com_max_count = 5000
        self.minor_time_offset_hours = 0.0
        self.initUI()
        self.resize(900, 700)

    def initUI(self):
        layout = self.layout()

        # Instruction Label
        instructions = QLabel(
            "Select the reference image and search images. "
            "Then click Process to hunt for anomalies."
        )
        layout.addWidget(instructions)

        # --- Reference Image Selection ---
        ref_layout = QHBoxLayout()
        self.ref_line_edit = QLineEdit(self)
        self.ref_line_edit.setPlaceholderText("No reference image selected")
        self.ref_button = QPushButton("Select Reference Image", self)
        self.ref_button.clicked.connect(self.selectReferenceImage)
        ref_layout.addWidget(self.ref_line_edit)
        ref_layout.addWidget(self.ref_button)
        layout.addLayout(ref_layout)

        # --- Search Images Selection ---
        search_layout = QHBoxLayout()
        self.search_list = QListWidget(self)
        self.search_button = QPushButton("Select Search Images", self)
        self.search_button.clicked.connect(self.selectSearchImages)
        search_layout.addWidget(self.search_list)
        search_layout.addWidget(self.search_button)
        layout.addLayout(search_layout)

        # --- Cosmetic Correction Checkbox ---
        self.cosmetic_checkbox = QCheckBox(
            "Apply Cosmetic Correction before Preprocessing", self
        )
        layout.addWidget(self.cosmetic_checkbox)

        # --- Threshold Slider ---
        thresh_layout = QHBoxLayout()
        self.thresh_label = QLabel("Anomaly Detection Threshold: 0.10", self)
        self.thresh_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.thresh_slider.setMinimum(1)
        self.thresh_slider.setMaximum(50)    # Represents 0.01 to 0.50
        self.thresh_slider.setValue(10)      # 10 => 0.10 threshold
        self.thresh_slider.valueChanged.connect(self.updateThreshold)
        thresh_layout.addWidget(self.thresh_label)
        thresh_layout.addWidget(self.thresh_slider)
        layout.addLayout(thresh_layout)

        # --- Process Button ---
        self.process_button = QPushButton(
            "Process (Cosmetic Correction, Preprocess, and Search)", self
        )
        self.process_button.clicked.connect(self.process)
        layout.addWidget(self.process_button)

        # --- Progress Labels ---
        self.preprocess_progress_label = QLabel("Preprocessing progress: 0 / 0", self)
        self.search_progress_label = QLabel("Processing progress: 0 / 0", self)
        layout.addWidget(self.preprocess_progress_label)
        layout.addWidget(self.search_progress_label)

        # -- Status label --
        self.status_label = QLabel("Status: Idle", self)
        layout.addWidget(self.status_label)

        # Minor-body progress bar (hidden by default)
        self.minor_progress = QProgressBar(self)
        self.minor_progress.setRange(0, 100)
        self.minor_progress.setValue(0)
        self.minor_progress.setVisible(False)
        layout.addWidget(self.minor_progress)

        # --- New Instance Button ---
        self.new_instance_button = QPushButton("New Instance", self)
        self.new_instance_button.clicked.connect(self.newInstance)
        layout.addWidget(self.new_instance_button)

        self.setLayout(layout)
        self.setWindowTitle("Supernova/Asteroid Hunter")



    def updateThreshold(self, value):
        threshold = value / 100.0  # e.g. slider value 10 becomes 0.10
        self.parameters["threshold"] = threshold
        self.thresh_label.setText(f"Anomaly Detection Threshold: {threshold:.2f}")

    def selectReferenceImage(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Reference Image", "",
                                                   "Images (*.png *.tif *.tiff *.fits *.fit *.xisf)")
        if file_path:
            self.parameters["referenceImagePath"] = file_path
            self.ref_line_edit.setText(os.path.basename(file_path))

    def selectSearchImages(self):
        file_paths, _ = QFileDialog.getOpenFileNames(self, "Select Search Images", "",
                                                     "Images (*.png *.tif *.tiff *.fits *.fit *.xisf)")
        if file_paths:
            self.parameters["searchImagePaths"] = file_paths
            self.search_list.clear()
            for path in file_paths:
                self.search_list.addItem(os.path.basename(path))

    def process(self):
        self.status_label.setText("Process started...")
        QApplication.processEvents()

        # If cosmetic correction is enabled, run it first
        if self.cosmetic_checkbox.isChecked():
            self.status_label.setText("Running Cosmetic Correction...")
            QApplication.processEvents()
            self.runCosmeticCorrectionIfNeeded()

        self.status_label.setText("Preprocessing images...")
        QApplication.processEvents()
        self.preprocessImages()

        self.status_label.setText("Analyzing anomalies...")
        QApplication.processEvents()
        self.runSearch()

        self.status_label.setText("Process complete.")
        QApplication.processEvents()


    def runCosmeticCorrectionIfNeeded(self):
        """
        Runs cosmetic correction on each search image...
        """
        # Dictionary to hold corrected images
        self.cosmetic_images = {}

        for idx, image_path in enumerate(self.parameters["searchImagePaths"]):
            try:
                # Update status label to show which image is being handled
                self.status_label.setText(f"Cosmetic Correction: {idx+1}/{len(self.parameters['searchImagePaths'])} => {os.path.basename(image_path)}")
                QApplication.processEvents()

                img, header, bit_depth, is_mono = load_image(image_path)
                if img is None:
                    print(f"Unable to load image: {image_path}")
                    continue

                # Numba correction
                corrected = bulk_cosmetic_correction_numba(
                    img,
                    hot_sigma=5.0,
                    cold_sigma=5.0,
                    window_size=3
                )
                self.cosmetic_images[image_path] = corrected
                print(f"Cosmetic correction (Numba) applied to: {image_path}")

            except Exception as e:
                print(f"Error in cosmetic correction for {image_path}: {e}")


    def preprocessImages(self):
        # Update status label for reference image
        self.status_label.setText("Preprocessing reference image...")
        print("[Preprocessing] Preprocessing reference image...")
        QApplication.processEvents()

        ref_path = self.parameters["referenceImagePath"]
        if not ref_path:
            QMessageBox.warning(self, "Error", "No reference image selected.")
            return

        try:
            # --- Load reference with metadata so we can grab header / XISF info ---
            ref_res = load_image(ref_path, return_metadata=True)
            if not ref_res or ref_res[0] is None:
                raise ValueError("load_image() returned no data for reference image.")

            ref_img, header, bit_depth, is_mono, meta = ref_res

            # Prefer synthesized FITS header from meta if present
            self.ref_header = meta.get("fits_header", header) if isinstance(meta, dict) else header

            # Try to build WCS directly from header (if it already has one).
            try:
                self.ref_wcs = WCS(self.ref_header)
            except Exception:
                self.ref_wcs = None

            # --- Derive mid-exposure JD ---
            self.ref_jd = None

            # 1) XISF-aware path: use FITSKeywords (DATE-OBS + EXPOSURE/EXPTIME)
            if isinstance(meta, dict):
                ensure_jd_from_xisf_meta(meta)
                jd_val = meta.get("jd", None)
                if jd_val is not None:
                    self.ref_jd = float(jd_val)

            # 2) FITS-style fallback from header (for non-XISF, or if XISF path failed)
            if self.ref_jd is None and isinstance(self.ref_header, (dict, Header)):
                try:
                    date_obs = self.ref_header.get("DATE-OBS")
                    exptime = float(
                        self.ref_header.get("EXPTIME", self.ref_header.get("EXPOSURE", 0.0))
                    )
                    if date_obs:
                        t = Time(str(date_obs), scale="utc")
                        # mid-exposure
                        t_mid = t + (exptime / 2.0) * u.s
                        self.ref_jd = float(t_mid.tt.jd)
                except Exception:
                    self.ref_jd = None

            print(f"[Preprocessing] ref JD={self.ref_jd!r}")
            print("[Preprocessing] (Minor-body prediction is now manual only.)")

            # --- Background neutralization + ABE + stretch for reference ---
            debug_prefix_ref = os.path.splitext(ref_path)[0] + "_debug_ref"

            self.status_label.setText(
                "Applying background neutralization & ABE on reference..."
            )
            QApplication.processEvents()

            ref_processed = self.preprocessImage(ref_img, debug_prefix=debug_prefix_ref)
            self.preprocessed_reference = ref_processed
            self.preprocess_progress_label.setText(
                "Preprocessing reference image... Done."
            )

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to preprocess reference image: {e}")
            return

        # --- Preprocess search images ---
        self.preprocessed_search = []
        search_paths = self.parameters["searchImagePaths"]
        total = len(search_paths)

        for i, path in enumerate(search_paths):
            try:
                self.status_label.setText(
                    f"Preprocessing search image {i+1}/{total} => {os.path.basename(path)}"
                )
                QApplication.processEvents()

                debug_prefix_search = os.path.splitext(path)[0] + f"_debug_search_{i+1}"

                if hasattr(self, "cosmetic_images") and path in self.cosmetic_images:
                    img = self.cosmetic_images[path]
                else:
                    img, header, bit_depth, is_mono = load_image(path)

                processed = self.preprocessImage(img, debug_prefix=debug_prefix_search)
                self.preprocessed_search.append({"path": path, "image": processed})

                self.preprocess_progress_label.setText(
                    f"Preprocessing image {i+1} of {total}... Done."
                )
                QApplication.processEvents()

            except Exception as e:
                print(f"Failed to preprocess {path}: {e}")

        self.status_label.setText("All search images preprocessed.")
        QApplication.processEvents()

    def _ensure_wcs(self, ref_path: str):
        """
        Ensure we have a WCS (and, if possible, JD) for the reference frame.
        This does NOT do any minor-body catalog work.
        """
        # If we already have a WCS and header, don't re-solve.
        if self.ref_wcs is not None and self.ref_header is not None:
            return

        try:
            image_data, original_header, bit_depth, is_mono = load_image(ref_path)
        except Exception as e:
            print(f"[SupernovaHunter] Failed to load reference image for plate solve: {e}")
            self.ref_wcs = None
            return

        if image_data is None:
            print("[SupernovaHunter] Reference image is unsupported or unreadable for plate solve.")
            self.ref_wcs = None
            return

        # Seed header from original_header (dict/Header/etc.)
        seed_h = _as_header(original_header) if isinstance(original_header, (dict, Header)) else None

        # Acquisition base for merge (strip any existing WCS)
        acq_base: Header | None = None
        if isinstance(seed_h, Header):
            acq_base = _strip_wcs_keys(seed_h)

        # Run the same solver core used by PlateSolverDialog
        ok, res = _solve_numpy_with_fallback(self, self.settings, image_data, seed_h)
        if not ok:
            print(f"[SupernovaHunter] Plate solve failed for {ref_path}: {res}")
            self.ref_wcs = None
            return

        solver_hdr: Header = res if isinstance(res, Header) else Header()

        # Merge solver WCS into acquisition header
        if isinstance(acq_base, Header) and isinstance(solver_hdr, Header):
            hdr_final = _merge_wcs_into_base_header(acq_base, solver_hdr)
        else:
            hdr_final = solver_hdr

        self.ref_header = hdr_final
        try:
            self.ref_wcs = WCS(hdr_final)
        except Exception as e:
            print("[SupernovaHunter] WCS build failed after plate solve:", e)
            self.ref_wcs = None

        # If we still lack JD, try to derive it from the header
        if self.ref_jd is None and isinstance(self.ref_header, Header):
            try:
                date_obs = self.ref_header.get("DATE-OBS")
                exptime = float(
                    self.ref_header.get("EXPTIME", self.ref_header.get("EXPOSURE", 0.0))
                )
                if date_obs:
                    t = Time(str(date_obs), scale="utc")
                    t_mid = t + (exptime / 2.0) * u.s
                    self.ref_jd = float(t_mid.tt.jd)
            except Exception:
                pass

    def _prompt_minor_body_limits(self) -> bool:
        """
        Modal dialog to configure minor-body search limits.

        Returns True if the user pressed OK (and updates self.* attributes),
        False if they cancelled.
        """
        dlg = QDialog(self)
        dlg.setWindowTitle("Minor-body Search Limits")
        layout = QVBoxLayout(dlg)

        row_layout = QGridLayout()
        layout.addLayout(row_layout)

        # Defaults / existing values
        ast_H_default = getattr(self, "minor_H_ast_max", 9.0)
        com_H_default = getattr(self, "minor_H_com_max", 10.0)
        ast_max_default = getattr(self, "minor_ast_max_count", 5000)
        com_max_default = getattr(self, "minor_com_max_count", 1000)

        # Time offset in *hours* now; if old days-based attr exists, convert.
        if hasattr(self, "minor_time_offset_hours"):
            dt_default = float(self.minor_time_offset_hours)
        else:
            dt_default = float(getattr(self, "minor_time_offset_days", 0.0)) * 24.0

        # Row 0: Asteroids
        row_layout.addWidget(QLabel("Asteroid H ≤"), 0, 0)
        ast_H_spin = QDoubleSpinBox(dlg)
        ast_H_spin.setDecimals(1)
        ast_H_spin.setRange(-5.0, 40.0)
        ast_H_spin.setSingleStep(0.1)
        ast_H_spin.setValue(ast_H_default)
        row_layout.addWidget(ast_H_spin, 0, 1)

        row_layout.addWidget(QLabel("Max asteroid"), 0, 2)
        ast_max_spin = QSpinBox(dlg)
        ast_max_spin.setRange(1, 2000000)
        ast_max_spin.setValue(ast_max_default)
        row_layout.addWidget(ast_max_spin, 0, 3)

        # Row 1: Comets
        row_layout.addWidget(QLabel("Comet H ≤"), 1, 0)
        com_H_spin = QDoubleSpinBox(dlg)
        com_H_spin.setDecimals(1)
        com_H_spin.setRange(-5.0, 40.0)
        com_H_spin.setSingleStep(0.1)
        com_H_spin.setValue(com_H_default)
        row_layout.addWidget(com_H_spin, 1, 1)

        row_layout.addWidget(QLabel("Max comet"), 1, 2)
        com_max_spin = QSpinBox(dlg)
        com_max_spin.setRange(1, 200000)
        com_max_spin.setValue(com_max_default)
        row_layout.addWidget(com_max_spin, 1, 3)

        # Row 2: Time offset (hours)
        row_layout.addWidget(QLabel("Time offset (hours)"), 2, 0)
        dt_spin = QDoubleSpinBox(dlg)
        dt_spin.setDecimals(1)
        dt_spin.setRange(-72.0, 72.0)   # ±3 days in hours
        dt_spin.setSingleStep(1.0)
        dt_spin.setValue(dt_default)
        row_layout.addWidget(dt_spin, 2, 1, 1, 3)

        # Buttons
        btn_row = QHBoxLayout()
        layout.addLayout(btn_row)
        btn_row.addStretch(1)
        ok_btn = QPushButton("OK", dlg)
        cancel_btn = QPushButton("Cancel", dlg)
        btn_row.addWidget(ok_btn)
        btn_row.addWidget(cancel_btn)

        def on_ok():
            self.minor_H_ast_max = float(ast_H_spin.value())
            self.minor_H_com_max = float(com_H_spin.value())
            self.minor_ast_max_count = int(ast_max_spin.value())
            self.minor_com_max_count = int(com_max_spin.value())
            hours = float(dt_spin.value())
            self.minor_time_offset_hours = hours
            # backward compat if anything still reads the old name:
            self.minor_time_offset_days = hours / 24.0
            dlg.accept()

        def on_cancel():
            dlg.reject()

        ok_btn.clicked.connect(on_ok)
        cancel_btn.clicked.connect(on_cancel)

        return dlg.exec() == QDialog.DialogCode.Accepted

    def _on_minor_body_progress(self, pct: int, msg: str):
        self.status_label.setText(msg)
        if hasattr(self, "minor_progress"):
            self.minor_progress.setVisible(True)
            self.minor_progress.setValue(int(pct))
        QApplication.processEvents()

    def _on_minor_body_finished(self, bodies: list, error: str):
        if hasattr(self, "minor_progress"):
            # show as done, then hide
            self.minor_progress.setValue(100 if not error else 0)
            self.minor_progress.setVisible(False)        
        if error:
            print("[MinorBodies] prediction failed:", error)
            QMessageBox.critical(
                self,
                "Minor-body Search",
                f"Minor-body prediction failed:\n{error}"
            )
            self.status_label.setText("Minor-body search failed.")
            return

        self.predicted_minor_bodies = bodies or []

        if not self.predicted_minor_bodies:
            self.status_label.setText(
                "Minor-body search complete: no catalogued objects in this field "
                "for the current magnitude limits."
            )
            QMessageBox.information(
                self,
                "Minor-body Search",
                "No catalogued minor bodies (within the configured magnitude limits) "
                "were found in this field."
            )
            return

        self.status_label.setText(
            f"Minor-body search complete: {len(self.predicted_minor_bodies)} objects in field."
        )
        QApplication.processEvents()

        # Now cross-match on the UI thread if we already have anomalies
        try:
            if self.anomalyData:
                print(f"[MinorBodies] cross-matching anomalies to "
                      f"{len(self.predicted_minor_bodies)} predicted bodies...")
                self._match_anomalies_to_minor_bodies(
                    self.predicted_minor_bodies,
                    search_radius_arcsec=60.0
                )
                self.showDetailedResultsDialog(self.anomalyData)
            else:
                QMessageBox.information(
                    self,
                    "Minor-body Search",
                    "Minor bodies in field have been computed.\n\n"
                    "Run the anomaly search (Process) to cross-match detections "
                    "against the predicted objects."
                )
        except Exception as e:
            print("[MinorBodies] cross-match failed:", e)


    def runMinorBodySearch(self):
        """
        Optional, slow step:
        - Ensure we have WCS + JD for the reference frame (plate-solve if needed).
        - Ask the user for H limits / max counts.
        - Query the minor-body catalog and compute predicted objects in the FOV.
        - Cross-match with existing anomalies (if any) and refresh the summary dialog.
        """
        ref_path = self.parameters.get("referenceImagePath") or ""
        if not ref_path:
            QMessageBox.warning(
                self,
                "Minor-body Search",
                "No reference image selected.\n\n"
                "Please select a reference image and run Process first."
            )
            return

        if self.preprocessed_reference is None:
            QMessageBox.warning(
                self,
                "Minor-body Search",
                "Reference image has not been preprocessed yet.\n\n"
                "Please click 'Process' before running the minor-body search."
            )
            return

        if self.settings is None:
            QMessageBox.warning(
                self,
                "Minor-body Search",
                "Settings object is not available; cannot locate the minor-body database path."
            )
            return

        # Configure limits (H, max counts, time offset)
        if not self._prompt_minor_body_limits():
            # user cancelled
            return

        # Step 1: Ensure WCS (plate-solve if necessary)
        self.status_label.setText("Minor-body search: solving plate / ensuring WCS...")
        QApplication.processEvents()

        self._ensure_wcs(ref_path)

        if self.ref_wcs is None:
            QMessageBox.warning(
                self,
                "Minor-body Search",
                "No valid WCS (astrometric solution) is available for the reference image.\n\n"
                "Minor-body prediction requires a solved WCS."
            )
            self.status_label.setText("Minor-body search aborted: no WCS.")
            return

        # Ensure we have JD (time of observation) for ephemerides
        if self.ref_jd is None:
            QMessageBox.warning(
                self,
                "Minor-body Search",
                "No valid observation time (JD) is available for the reference image.\n\n"
                "Minor-body prediction requires DATE-OBS/EXPTIME or equivalent."
            )
            self.status_label.setText("Minor-body search aborted: no JD.")
            return

        # Optional observatory site
        try:
            print("[MinorBodies] fetching observatory site from settings...")
            lat = self.settings.value("site/latitude_deg", None, type=float)
            lon = self.settings.value("site/longitude_deg", None, type=float)
            elev = self.settings.value("site/elevation_m", 0.0, type=float)
            if lat is not None and lon is not None:
                self.ref_site = (lat, lon, elev)
            else:
                self.ref_site = None
        except Exception as e:
            print("[MinorBodies] failed to fetch observatory site from settings:", e)
            self.ref_site = None

        # JD adjusted by time offset (hours → days)
        offset_hours = getattr(self, "minor_time_offset_hours", 0.0)
        jd_for_calc = self.ref_jd + (offset_hours / 24.0)

        # Kick off the heavy catalog + ephemeris work in a background thread
        self.status_label.setText(
            "Minor-body search: starting background catalog query..."
        )
        QApplication.processEvents()
        if hasattr(self, "minor_progress"):
            self.minor_progress.setVisible(True)
            self.minor_progress.setValue(0)

        self._mb_thread = QThread(self)
        self._mb_worker = MinorBodyWorker(self, jd_for_calc)
        self._mb_worker.moveToThread(self._mb_thread)

        # Wire up thread lifecycle
        self._mb_thread.started.connect(self._mb_worker.run)
        self._mb_worker.progress.connect(self._on_minor_body_progress)
        self._mb_worker.finished.connect(self._on_minor_body_finished)
        self._mb_worker.finished.connect(self._mb_thread.quit)
        self._mb_worker.finished.connect(self._mb_worker.deleteLater)
        self._mb_thread.finished.connect(self._mb_thread.deleteLater)

        self._mb_thread.start()

    def _get_predicted_minor_bodies_for_field(
        self,
        H_ast_max: float,
        H_com_max: float,
        jd: float | None = None,
        progress_cb=None,
    ):
        """
        Return a list of predicted minor bodies in the current ref image FOV
        at 'jd' (or self.ref_jd if jd is None), with pixel coords.
        """
        # Need WCS and an image
        if self.ref_wcs is None or self.preprocessed_reference is None:
            return []

        def emit(pct, msg):
            if progress_cb is not None:
                try:
                    progress_cb(int(pct), msg)
                except TypeError:
                    # fallback if callback only wants a message
                    progress_cb(msg)


        # Resolve JD: explicit first, then self.ref_jd
        if jd is None:
            jd = self.ref_jd
        if jd is None:
            return []

        if self.settings is None:
            print("[MinorBodies] settings object is None; cannot resolve DB path.")
            return []

        # Per-type max counts with safe defaults
        ast_limit = getattr(self, "minor_ast_max_count", 50000)
        com_limit = getattr(self, "minor_com_max_count", 5000)

        # 1) open DB (reuse WIMI’s ensure logic)
        emit(5, "Minor-body search: opening minor-body database...")
        try:
            data_dir = Path(
                self.settings.value("wimi/minorbody_data_dir", "", type=str)
                or os.path.join(os.path.expanduser("~"), ".saspro_minor_bodies")
            )
            db_path, manifest = mbc.ensure_minor_body_db(data_dir)
            catalog = mbc.MinorBodyCatalog(db_path)
        except Exception as e:
            print("[MinorBodies] could not open DB:", e)
            emit(100, "Minor-body search: failed to open database.")
            return []

        try:
            emit(20, "Minor-body search: selecting bright asteroids/comets...")
            ast_df = catalog.get_bright_asteroids(H_max=H_ast_max, limit=ast_limit)
            com_df = catalog.get_bright_comets(H_max=H_com_max,   limit=com_limit)

            emit(40, "Minor-body search: computing asteroid positions...")
            ast_pos = catalog.compute_positions_skyfield(
                ast_df,
                jd,
                topocentric=self.ref_site,
                debug=False,
            )
            emit(60, "Minor-body search: computing comet positions...")
            com_pos = catalog.compute_positions_skyfield(
                com_df,
                jd,
                topocentric=self.ref_site,
                debug=False,
            )

            emit(80, "Minor-body search: projecting onto image pixels...")

            # 4) map RA/Dec -> pixel with ref WCS, and drop those outside FOV
            h, w = self.preprocessed_reference.shape[:2]
            bodies = []
            for src, kind, df in (
                (ast_pos, "asteroid", ast_df),
                (com_pos, "comet",   com_df),
            ):
                df_by_name = {row["designation"]: row for _, row in df.iterrows()}
                for row in src:
                    ra = row["ra_deg"]
                    dec = row["dec_deg"]
                    x, y = self.ref_wcs.world_to_pixel_values(ra, dec)
                    if 0 <= x < w and 0 <= y < h:
                        base = df_by_name.get(row["designation"], {})
                        bodies.append({
                            "designation": row["designation"],
                            "kind": kind,
                            "ra_deg": ra,
                            "dec_deg": dec,
                            "x": float(x),
                            "y": float(y),
                            "H": float(base.get("magnitude_H", np.nan)),
                            "distance_au": row.get("distance_au", np.nan),
                        })
            emit(100, "Minor-body search: finished computing positions.")            
            return bodies
        finally:
            try:
                catalog.close()
            except Exception:
                pass


    def preprocessImage(self, img, debug_prefix=None):
        """
        Runs the full preprocessing chain on a single image:
        1. Background Neutralization
        2. Automatic Background Extraction (ABE)
        3. Pixel-math stretching

        Optionally saves debug images if debug_prefix is provided.
        """


        # --- Step 1: Background Neutralization ---
        if img.ndim == 3 and img.shape[2] == 3:
            h, w, _ = img.shape
            sample_x = int(w * 0.45)
            sample_y = int(h * 0.45)
            sample_w = max(1, int(w * 0.1))
            sample_h = max(1, int(h * 0.1))
            sample_region = img[sample_y:sample_y+sample_h, sample_x:sample_x+sample_w, :]
            medians = np.median(sample_region, axis=(0, 1))
            average_median = np.mean(medians)
            neutralized = img.copy()
            for c in range(3):
                diff = medians[c] - average_median
                numerator = neutralized[:, :, c] - diff
                denominator = 1.0 - diff
                if abs(denominator) < 1e-8:
                    denominator = 1e-8
                neutralized[:, :, c] = np.clip(numerator / denominator, 0, 1)
        else:
            neutralized = img


        # --- Step 2: Automatic Background Extraction (ABE) ---
        pgr = PolyGradientRemoval(
            neutralized,
            poly_degree=2,          # or pass in a user choice
            downsample_scale=4,
            num_sample_points=100
        )
        abe = pgr.process()  # returns final polynomial-corrected image in original domain


        # --- Step 3: Pixel Math Stretch ---
        stretched = self.pixel_math_stretch(abe)

        return stretched



    def pixel_math_stretch(self, image):
        """
        Replaces the old pixel math stretch logic by using the existing
        stretch_mono_image or stretch_color_image methods. 
        """
        # Choose a target median (the default you’ve used elsewhere is often 0.25)
        target_median = 0.25

        # Check if the image is mono or color
        if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
            # Treat it as mono
            stretched = stretch_mono_image(
                image.squeeze(),  # squeeze in case it's (H,W,1)
                target_median=target_median,
                normalize=False,  # Adjust if you want normalization
                apply_curves=False,
                curves_boost=0.0
            )
            # If it was (H,W,1), replicate to 3 channels (optional)
            # or just keep it mono if you prefer
            # For now, replicate to 3 channels:
            stretched = np.stack([stretched]*3, axis=-1)
        else:
            # Full-color image
            stretched = stretch_color_image(
                image,
                target_median=target_median,
                linked=False,      # or False if you want per-channel stretches
                normalize=False,  
                apply_curves=False,
                curves_boost=0.0
            )

        return np.clip(stretched, 0, 1)

    def runSearch(self):
        if self.preprocessed_reference is None:
            QMessageBox.warning(self, "Error", "Reference image not preprocessed.")
            return
        if not self.preprocessed_search:
            QMessageBox.warning(self, "Error", "No search images preprocessed.")
            return

        ref_gray = self.to_grayscale(self.preprocessed_reference)

        self.anomalyData = []
        total = len(self.preprocessed_search)
        for i, search_dict in enumerate(self.preprocessed_search):
            search_img = search_dict["image"]
            search_gray = self.to_grayscale(search_img)

            diff_img = self.subtractImagesOnce(search_gray, ref_gray)
            anomalies = self.detectAnomaliesConnected(
                diff_img,
                threshold=self.parameters["threshold"],
            )

            self.anomalyData.append({
                "imageName": os.path.basename(search_dict["path"]),
                "anomalyCount": len(anomalies),
                "anomalies": anomalies,
            })

            self.search_progress_label.setText(f"Processing image {i+1} of {total}...")
            QApplication.processEvents()

        self.search_progress_label.setText("Search for anomalies complete.")

        # Minor-body cross-match (optional)
        try:
            bodies = getattr(self, "predicted_minor_bodies", None)
            if bodies:
                print(f"[MinorBodies] cross-matching anomalies to {len(bodies)} predicted bodies...")
                self._match_anomalies_to_minor_bodies(bodies, search_radius_arcsec=60.0)
        except Exception as e:
            print("[MinorBodies] cross-match failed:", e)

        # Show text-based summary & tree
        self.showDetailedResultsDialog(self.anomalyData)
        self.showAnomalyListDialog()

    def showAnomalyListDialog(self):
        """
        Build a QDialog with a QTreeWidget listing each image and its anomaly count.
        Double-clicking an item will open a non-modal preview.
        """
        if not self.anomalyData:
            QMessageBox.information(self, "Info", "No anomalies or no images processed.")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Anomaly Results")

        layout = QVBoxLayout(dialog)

        self.anomaly_tree = QTreeWidget(dialog)
        self.anomaly_tree.setColumnCount(2)
        self.anomaly_tree.setHeaderLabels(["Image", "Anomaly Count"])
        layout.addWidget(self.anomaly_tree)

        # Populate the tree
        for i, data in enumerate(self.anomalyData):
            item = QTreeWidgetItem([
                data["imageName"],
                str(data["anomalyCount"])
            ])
            # Store an index or reference so we know which image to open
            item.setData(0, Qt.ItemDataRole.UserRole, i)
            self.anomaly_tree.addTopLevelItem(item)

        # Connect double-click
        self.anomaly_tree.itemDoubleClicked.connect(self.onAnomalyItemDoubleClicked)

        dialog.setLayout(layout)
        dialog.resize(300, 200)
        dialog.show()  # non-modal, so the user can keep using the main window

    def onAnomalyItemDoubleClicked(self, item, column):
        idx = item.data(0, Qt.ItemDataRole.UserRole)
        if idx is None:
            return
        anomalies = self.anomalyData[idx]["anomalies"]
        image_name = self.anomalyData[idx]["imageName"]
        search_img = self.preprocessed_search[idx]["image"]  # float in [0..1]

        # Show zoomable preview with overlays
        self.showAnomaliesOnImage(search_img, anomalies, window_title=f"Anomalies in {image_name}")

    def _match_anomalies_to_minor_bodies(self, bodies, search_radius_arcsec=20.0):
        """
        For each anomaly, compute center pixel and find
        all predicted minor bodies within search_radius_arcsec.

        Adds:
          - anomaly["matched_bodies"] = [body, ...]
          - anomaly["matched_body"]   = closest body or None
        """
        if self.ref_wcs is None or not bodies:
            return

        # search radius in pixels — crude average plate scale from WCS
        try:
            cd = self.ref_wcs.pixel_scale_matrix  # 2x2
            from numpy.linalg import det
            deg_per_pix = np.sqrt(abs(det(cd)))
            arcsec_per_pix = deg_per_pix * 3600.0
        except Exception:
            arcsec_per_pix = 1.0  # fallback

        pix_radius = search_radius_arcsec / arcsec_per_pix

        for entry in self.anomalyData:
            for anomaly in entry["anomalies"]:
                cx = 0.5 * (anomaly["minX"] + anomaly["maxX"])
                cy = 0.5 * (anomaly["minY"] + anomaly["maxY"])

                matches = []
                for body in bodies:
                    dx = body["x"] - cx
                    dy = body["y"] - cy
                    r_pix = np.hypot(dx, dy)
                    if r_pix <= pix_radius:
                        matches.append((r_pix, body))

                if matches:
                    matches.sort(key=lambda t: t[0])
                    anomaly["matched_body"] = matches[0][1]
                    anomaly["matched_bodies"] = [b for _, b in matches]
                else:
                    anomaly["matched_body"] = None
                    anomaly["matched_bodies"] = []


    def draw_bounding_boxes_on_stretched(self,
        stretched_image: np.ndarray, 
        anomalies: list
    ) -> np.ndarray:
        """
        1) Convert 'stretched_image' [0..1] -> [0..255] 8-bit color
        2) Draw red rectangles for each anomaly in 'anomalies'.
        Each anomaly is assumed to have keys: minX, minY, maxX, maxY
        3) Return the 8-bit color image (H,W,3).
        """
        # Ensure 3 channels
        if stretched_image.ndim == 2:
            stretched_3ch = np.stack([stretched_image]*3, axis=-1)
        elif stretched_image.ndim == 3 and stretched_image.shape[2] == 1:
            stretched_3ch = np.concatenate([stretched_image]*3, axis=2)
        else:
            stretched_3ch = stretched_image

        # Convert float [0..1] => uint8 [0..255]
        img_bgr = (stretched_3ch * 255).clip(0,255).astype(np.uint8)

        # Define the margin
        margin = 15

        # Draw red boxes in BGR color = (0, 0, 255)
        for anomaly in anomalies:
            x1, y1 = anomaly["minX"], anomaly["minY"]
            x2, y2 = anomaly["maxX"], anomaly["maxY"]

            # Expand the bounding box by a 10-pixel margin
            x1_exp = x1 - margin
            y1_exp = y1 - margin
            x2_exp = x2 + margin
            y2_exp = y2 + margin
            cv2.rectangle(img_bgr, (x1_exp, y1_exp), (x2_exp, y2_exp), color=(0, 0, 255), thickness=5)

        return img_bgr


    def subtractImagesOnce(self, search_img, ref_img, debug_prefix=None):
        result = search_img - ref_img
        result = np.clip(result, 0, 1)  # apply the clip
        return result

    def debug_save_image(self, image, prefix="debug", step_name="step", ext=".tif"):
        """
        Saves 'image' to disk for debugging. 
        - 'prefix' can be a directory path or prefix for your debug images.
        - 'step_name' is appended to the filename to indicate which step.
        - 'ext' could be '.tif', '.png', or another format you support.

        This example uses your 'save_image' function from earlier or can
        directly use tiff.imwrite or similar.
        """

        # Ensure the image is float32 in [0..1] before saving
        image = image.astype(np.float32, copy=False)

        # Build debug filename
        filename = f"{prefix}_{step_name}{ext}"

        # E.g., if you have a global 'save_image' function:
        save_image(
            image, 
            filename,
            original_format="tif",  # or "png", "fits", etc.
            bit_depth="16-bit"
        )
        print(f"[DEBUG] Saved {step_name} => {filename}")

    def to_grayscale(self, image):
        """
        Converts an image to grayscale by averaging channels if needed.
        If the image is already 2D, return it as is.
        """
        if image.ndim == 2:
            # Already grayscale
            return image
        elif image.ndim == 3 and image.shape[2] == 3:
            # Average the three channels
            return np.mean(image, axis=2)
        elif image.ndim == 3 and image.shape[2] == 1:
            # Squeeze out that single channel
            return image[:, :, 0]
        else:
            raise ValueError(f"Unsupported image shape for grayscale: {image.shape}")

    def detectAnomaliesConnected(self, diff_img: np.ndarray, threshold: float = 0.1):
        """
        1) Build mask = diff_img > threshold.
        2) Optionally skip 5% border by zeroing out that region in the mask.
        3) connectedComponentsWithStats => bounding boxes.
        4) Filter by min_area, etc.
        5) Return a list of anomalies, each with minX, minY, maxX, maxY, area.
        """
        h, w = diff_img.shape

        # 1) Create the mask
        mask = (diff_img > threshold).astype(np.uint8)

        # 2) Skip 5% border (optional)
        border_x = int(0.05 * w)
        border_y = int(0.05 * h)
        mask[:border_y, :] = 0
        mask[h - border_y:, :] = 0
        mask[:, :border_x] = 0
        mask[:, w - border_x:] = 0

        # 3) connectedComponentsWithStats => label each region
        # connectivity=8 => 8-way adjacency
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

        # stats[i] = [x, y, width, height, area], for i in [1..num_labels-1]
        # label_id=0 => background

        anomalies = []
        for label_id in range(1, num_labels):
            x, y, width_, height_, area_ = stats[label_id]

            # bounding box corners
            minX = x
            minY = y
            maxX = x + width_ - 1
            maxY = y + height_ - 1

            # 4) Filter out tiny or huge areas if you want:
            # e.g., skip anything <4x4 => area<16
            if area_ < 25:
                continue
            # e.g., skip bounding boxes bigger than 40 in either dimension if you want
            if width_ > 200 or height_ > 200:
                continue

            anomalies.append({
                "minX": minX,
                "minY": minY,
                "maxX": maxX,
                "maxY": maxY,
                "area": area_
            })

        return anomalies


    def showDetailedResultsDialog(self, anomalyData):
        dialog = QDialog(self)
        dialog.setWindowTitle("Anomaly Detection Results")
        layout = QVBoxLayout(dialog)
        text_edit = QTextEdit(dialog)
        text_edit.setReadOnly(True)
        result_text = "Detailed Anomaly Results:\n\n"

        for data in anomalyData:
            result_text += f"Image: {data['imageName']}\nAnomalies: {data['anomalyCount']}\n"
            for group in data["anomalies"]:
                result_text += (
                    f"  Group Bounding Box: "
                    f"Top-Left ({group['minX']}, {group['minY']}), "
                    f"Bottom-Right ({group['maxX']}, {group['maxY']})\n"
                )
                mbs = group.get("matched_bodies") or []
                if mbs:
                    result_text += "    → Candidate matches:\n"
                    for mb in mbs:
                        H_str = (
                            f"{mb['H']:.1f}"
                            if np.isfinite(mb.get("H", np.nan))
                            else "?"
                        )
                        result_text += (
                            f"      - {mb['kind']} {mb['designation']} "
                            f"(H={H_str})\n"
                        )
                # if no matches, leave as a pure candidate box
            result_text += "\n"

        text_edit.setText(result_text)
        layout.addWidget(text_edit)
        dialog.setLayout(layout)
        dialog.show()


    def showAnomaliesOnImage(self, image: np.ndarray, anomalies: list, window_title="Anomalies"):
        """
        Shows a zoomable, pannable preview. CTRL+wheel zoom, buttons for fit/1:1.
        Pushing emits a signal you can wire to your main UI.
        """
        # Ensure 3-ch so we can draw boxes
        if image.ndim == 2:
            img3 = np.stack([image]*3, axis=-1)
        elif image.ndim == 3 and image.shape[2] == 1:
            img3 = np.concatenate([image]*3, axis=2)
        else:
            img3 = image

        # Make a copy in uint8 RGB for overlays
        if img3.dtype != np.uint8:
            img_u8 = (np.clip(img3, 0, 1) * 255).astype(np.uint8)
        else:
            img_u8 = img3.copy()

        margin = 10
        h, w = img_u8.shape[:2]
        for a in anomalies:
            x1, y1, x2, y2 = a["minX"], a["minY"], a["maxX"], a["maxY"]
            x1 = max(0, x1 - margin); y1 = max(0, y1 - margin)
            x2 = min(w - 1, x2 + margin); y2 = min(h - 1, y2 + margin)

            mbs = a.get("matched_bodies") or []
            if mbs:
                # anomalies with known bodies -> green box
                color = (0, 255, 0)
            else:
                # pure candidates -> red box
                color = (255, 0, 0)

            cv2.rectangle(img_u8, (x1, y1), (x2, y2), color=color, thickness=5)

        # NEW: overlay all predicted minor bodies as circles
        bodies = getattr(self, "predicted_minor_bodies", None)
        if bodies:
            for body in bodies:
                x = int(round(body["x"]))
                y = int(round(body["y"]))
                if 0 <= x < w and 0 <= y < h:
                    # yellow circle so it stands out from red/green boxes
                    cv2.circle(img_u8, (x, y), 8, (255, 255, 0), thickness=2)


        # Launch preview window
        icon = None
        try:
            if hasattr(self, "supernova_path") and self.supernova_path:
                icon = QIcon(self.supernova_path)
        except Exception:
            pass

        prev = ImagePreviewWindow(img_u8, title=window_title, parent=self, icon=icon)
        prev.pushed.connect(self._handle_preview_push)
        prev.minorBodySearchRequested.connect(self._on_preview_minor_body_search)
        prev.show()  # non-modal

    def _on_preview_minor_body_search(self):
        """
        Called when the user clicks 'Check Catalogued Minor Bodies in Field'
        on any anomaly preview window.
        """
        self.runMinorBodySearch()


    def _handle_preview_push(self, np_img, title: str):
        """
        Try to push the preview up to your main UI using doc_manager.
        Customize this to your actual document API.
        """
        # If your doc_manager has a known API, call it here:
        if self.doc_manager and hasattr(self.doc_manager, "open_numpy"):
            try:
                self.doc_manager.open_numpy(np_img, title=title)
                return
            except Exception as e:
                print("doc_manager.open_numpy failed:", e)

        if self.doc_manager and hasattr(self.doc_manager, "open_image_array"):
            try:
                self.doc_manager.open_image_array(np_img, title)
                return
            except Exception as e:
                print("doc_manager.open_image_array failed:", e)

        # Fallback: write a temp PNG and let the user open it
        try:
            tmpdir = tempfile.gettempdir()
            safe_title = "".join(c if c.isalnum() or c in " _-" else "_" for c in title)
            out = os.path.join(tmpdir, f"{safe_title}.png")
            # Ensure RGB uint8
            arr = np_img
            if arr.dtype != np.uint8:
                arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
            if arr.ndim == 2:
                arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
            elif arr.ndim == 3 and arr.shape[2] == 3:
                pass  # already RGB
            elif arr.ndim == 3 and arr.shape[2] == 4:
                arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
            cv2.imwrite(out, cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
            QMessageBox.information(self, "Pushed",
                f"No document API found; saved preview to:\n{out}")
        except Exception as e:
            QMessageBox.warning(self, "Push failed", f"Could not export preview: {e}")

    def newInstance(self):
        # Reset parameters and UI elements for a new run
        self.parameters = {
            "referenceImagePath": "",
            "searchImagePaths": [],
            "threshold": 0.10
        }

        self.ref_line_edit.clear()
        self.search_list.clear()
        self.cosmetic_checkbox.setChecked(False)
        self.thresh_slider.setValue(10)

        self.preprocess_progress_label.setText("Preprocessing progress: 0 / 0")
        self.search_progress_label.setText("Processing progress: 0 / 0")
        self.status_label.setText("Status: Idle")

        # Image + results state
        self.preprocessed_reference = None
        self.preprocessed_search = []
        self.anomalyData = []

        # WCS / timing / minor-body state
        self.ref_header = None
        self.ref_wcs = None
        self.ref_jd = None
        self.ref_site = None
        self.predicted_minor_bodies = None

        QMessageBox.information(self, "New Instance", "Reset for a new instance.")

