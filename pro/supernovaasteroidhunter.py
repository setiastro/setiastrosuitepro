import os
import cv2
import numpy as np
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QHBoxLayout, QLineEdit, QPushButton, QFileDialog,
                             QListWidget, QSlider, QCheckBox, QMessageBox, QTextEdit, QDialog, QApplication,
                             QTreeWidget, QTreeWidgetItem, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QToolBar, QSizePolicy)
from PyQt6.QtGui import QImage, QPixmap, QIcon, QPainter, QAction, QTransform, QCursor
from PyQt6.QtCore import Qt, pyqtSignal, QRectF, QPointF, QTimer

from legacy.image_manager import load_image, save_image
from numba_utils import bulk_cosmetic_correction_numba
from imageops.stretch import stretch_mono_image, stretch_color_image
from pro.star_alignment import PolyGradientRemoval 

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
    pushed = pyqtSignal(object, str)  # (numpy_image, title)

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

        # start in "Fit"
        self.view.fit_to_view()
        self.resize(900, 700)

    def _on_zoom_changed(self, s: float):
        self._zoom_label.setText(f"{round(s*100)}%")

    def _on_push(self):
        # Emit the original (float or uint8) image up to the parent/dialog
        self.pushed.emit(self._original, self.windowTitle())
        QMessageBox.information(self, "Pushed", "New View Created.")

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

        self.initUI()
        self.resize(900, 700)

    def initUI(self):
        layout = self.layout() 

        # Instruction Label
        instructions = QLabel("Select the reference image and search images. Then click Process to hunt for anomalies.")
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
        self.cosmetic_checkbox = QCheckBox("Apply Cosmetic Correction before Preprocessing", self)
        layout.addWidget(self.cosmetic_checkbox)

        # --- Threshold Slider ---
        thresh_layout = QHBoxLayout()
        self.thresh_label = QLabel("Anomaly Detection Threshold: 0.10", self)
        self.thresh_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.thresh_slider.setMinimum(1)
        self.thresh_slider.setMaximum(50)  # Represents 0.01 to 0.50
        self.thresh_slider.setValue(10)      # 10 => 0.10 threshold
        self.thresh_slider.valueChanged.connect(self.updateThreshold)
        thresh_layout.addWidget(self.thresh_label)
        thresh_layout.addWidget(self.thresh_slider)
        layout.addLayout(thresh_layout)

        # --- Process Button ---
        self.process_button = QPushButton("Process (Cosmetic Correction, Preprocess, and Search)", self)
        self.process_button.clicked.connect(self.process)
        layout.addWidget(self.process_button)

        # --- Progress Labels ---
        self.preprocess_progress_label = QLabel("Preprocessing progress: 0 / 0", self)
        self.search_progress_label = QLabel("Processing progress: 0 / 0", self)
        layout.addWidget(self.preprocess_progress_label)
        layout.addWidget(self.search_progress_label)

        # -- Add a new status label --
        self.status_label = QLabel("Status: Idle", self)
        layout.addWidget(self.status_label)

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
        QApplication.processEvents()

        ref_path = self.parameters["referenceImagePath"]
        if not ref_path:
            QMessageBox.warning(self, "Error", "No reference image selected.")
            return

        try:
            ref_img, header, bit_depth, is_mono = load_image(ref_path)

            # Create a debug prefix from the reference path (e.g. "C:/data/ref_debug")
            debug_prefix_ref = os.path.splitext(ref_path)[0] + "_debug_ref"

            self.status_label.setText("Applying background neutralization & ABE on reference...")
            QApplication.processEvents()

            # Pass debug_prefix_ref to preprocessImage
            ref_processed = self.preprocessImage(ref_img, debug_prefix=debug_prefix_ref)
            self.preprocessed_reference = ref_processed
            self.preprocess_progress_label.setText("Preprocessing reference image... Done.")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to preprocess reference image: {e}")
            return

        self.preprocessed_search = []
        search_paths = self.parameters["searchImagePaths"]
        for i, path in enumerate(search_paths):
            try:
                self.status_label.setText(f"Preprocessing search image {i+1}/{len(search_paths)} => {os.path.basename(path)}")
                QApplication.processEvents()

                # Create a debug prefix from the search path
                debug_prefix_search = os.path.splitext(path)[0] + f"_debug_search_{i+1}"

                if hasattr(self, 'cosmetic_images') and path in self.cosmetic_images:
                    img = self.cosmetic_images[path]
                else:
                    img, header, bit_depth, is_mono = load_image(path)

                # Pass debug_prefix_search to preprocessImage
                processed = self.preprocessImage(img, debug_prefix=debug_prefix_search)
                self.preprocessed_search.append({"path": path, "image": processed})

                self.preprocess_progress_label.setText(f"Preprocessing image {i+1} of {len(search_paths)}... Done.")
                QApplication.processEvents()

            except Exception as e:
                print(f"Failed to preprocess {path}: {e}")

        self.status_label.setText("All search images preprocessed.")
        QApplication.processEvents()



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
            anomalies = self.detectAnomaliesConnected(diff_img, threshold=self.parameters["threshold"])

            # Just store the anomalies
            self.anomalyData.append({
                "imageName": os.path.basename(search_dict["path"]),
                "anomalyCount": len(anomalies),
                "anomalies": anomalies
            })

            self.search_progress_label.setText(f"Processing image {i+1} of {total}...")
            QApplication.processEvents()

        self.search_progress_label.setText("Search for anomalies complete.")

        # Optionally still show the text-based summary:
        self.showDetailedResultsDialog(self.anomalyData)

        # Now build & show the anomaly tree for user double-click
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
                # Now refer to 'minX', 'minY', 'maxX', 'maxY'
                result_text += (
                    f"  Group Bounding Box: "
                    f"Top-Left ({group['minX']}, {group['minY']}), "
                    f"Bottom-Right ({group['maxX']}, {group['maxY']})\n"
                )
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

        # Draw red rectangles (we’ll do it in RGB here for consistency)
        margin = 10
        h, w = img_u8.shape[:2]
        for a in anomalies:
            x1, y1, x2, y2 = a["minX"], a["minY"], a["maxX"], a["maxY"]
            x1 = max(0, x1 - margin); y1 = max(0, y1 - margin)
            x2 = min(w - 1, x2 + margin); y2 = min(h - 1, y2 + margin)
            cv2.rectangle(img_u8, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=5)  # RGB red

        # Launch preview window
        icon = None
        try:
            # if you passed supernova_path into the dialog:
            if hasattr(self, "supernova_path") and self.supernova_path:
                icon = QIcon(self.supernova_path)
        except Exception:
            pass

        prev = ImagePreviewWindow(img_u8, title=window_title, parent=self, icon=icon)
        prev.pushed.connect(self._handle_preview_push)
        prev.show()  # non-modal

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
        self.parameters = {"referenceImagePath": "", "searchImagePaths": [], "threshold": 0.10}
        self.ref_line_edit.clear()
        self.search_list.clear()
        self.cosmetic_checkbox.setChecked(False)
        self.thresh_slider.setValue(10)
        self.preprocess_progress_label.setText("Preprocessing progress: 0 / 0")
        self.search_progress_label.setText("Processing progress: 0 / 0")
        self.preprocessed_reference = None
        self.preprocessed_search = []
        self.anomalyData = []
        QMessageBox.information(self, "New Instance", "Reset for a new instance.")
