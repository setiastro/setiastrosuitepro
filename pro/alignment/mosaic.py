from __future__ import annotations
import os
import math
import sys
import numpy as np
import cv2
import sep
import random
import json
import time
import shutil
import tempfile
import re
import traceback
import warnings
import numpy.ma as ma

from PyQt6.QtCore import (
    Qt, QThread, QRunnable, QThreadPool, pyqtSignal, QObject, QTimer, 
    QProcess, QPoint, QEvent, QSettings
)
from PyQt6.QtGui import (
    QImage, QPixmap, QIcon, QMovie, QPalette, QColor, QBrush, QAction
)
from PyQt6.QtWidgets import (
    QDialog, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, 
    QGroupBox, QAbstractItemView, QListWidget, QInputDialog, QApplication, 
    QProgressBar, QProgressDialog, QRadioButton, QFileDialog, QComboBox, 
    QMessageBox, QTextEdit, QDialogButtonBox, QTreeWidget, QCheckBox, 
    QFormLayout, QListWidgetItem, QScrollArea, QTreeWidgetItem, QSpinBox, 
    QDoubleSpinBox
)
from astropy.io import fits
from astropy.io.fits import Header
from astropy.wcs import WCS, FITSFixedWarning
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import skycoord_to_pixel
import astropy.units as u
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
import astroalign

# Core
from pro.alignment.core import (
    _align_prefs, _gray2d, aa_find_transform_with_backoff, 
    sanitize_wcs_header, get_wcs_from_header,
    load_api_key, save_api_key, robust_api_request, 
    generate_minimal_fits_header, ASTROMETRY_API_URL
)
from legacy.image_manager import load_image, save_image
from legacy.numba_utils import (
    numba_mono_final_formula, numba_color_final_formula_unlinked, numba_unstretch
)
try:
    from imageops.stretch import stretch_mono_image, stretch_color_image
except ImportError:
    stretch_mono_image = None
    stretch_color_image = None

# ABE / Poly
from pro.memory_utils import smart_zeros
from pro.stacking.functions import debayer_raw_fast, debayer_fits_fast
from reproject import reproject_interp

# ABE / Poly
try:
    from pro.abe import build_poly_terms, evaluate_polynomial, abe_generate_sample_points
except ImportError:
    # Fallback or stub if not available?
    build_poly_terms = None
    evaluate_polynomial = None
    abe_generate_sample_points = None


# Helpers for UI
class CustomDoubleSpinBox(QDoubleSpinBox):
    def __init__(self, minimum, maximum, initial, step):
        super().__init__()
        self.setRange(minimum, maximum)
        self.setValue(initial)
        self.setSingleStep(step)

class CustomSpinBox(QSpinBox):
    def __init__(self, minimum, maximum, initial, step):
        super().__init__()
        self.setRange(minimum, maximum)
        self.setValue(initial)
        self.setSingleStep(step)


class MosaicPreviewWindow(QDialog):
    def __init__(self, image_array, title="", parent=None, push_cb=None):
        super().__init__(parent)
        self.setWindowTitle(title if title else "Preview")
        self._push_cb = push_cb

        # Keep the original array around for re-stretch or reset
        self.original_array = image_array.copy()
        # Current displayed array (8-bit or whatever you want)
        self.image_array = image_array.copy()

        # Zoom state
        self.zoom_factor = 1.0

        # Variables for panning (dragging)
        self.dragging = False
        self.last_mouse_pos = QPoint()
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout(self)

        # 1) QScrollArea to enable scrollbars for large images
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        layout.addWidget(self.scroll_area)

        # 2) Label inside the scroll area
        self.preview_label = QLabel("No image yet.")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scroll_area.setWidget(self.preview_label)
        self.scroll_area.viewport().installEventFilter(self)

        # 3) Auto-Stretch Toggle
        self.stretch_toggle = QCheckBox("Auto-Stretch for Display")
        self.stretch_toggle.setChecked(True)
        self.stretch_toggle.stateChanged.connect(self.update_display)
        layout.addWidget(self.stretch_toggle)

        # 4) Button row (Zoom, Fit, etc.)
        button_layout = QHBoxLayout()

        self.zoom_in_button = QPushButton("Zoom In")
        self.zoom_in_button.clicked.connect(self.zoom_in)
        button_layout.addWidget(self.zoom_in_button)

        self.zoom_out_button = QPushButton("Zoom Out")
        self.zoom_out_button.clicked.connect(self.zoom_out)
        button_layout.addWidget(self.zoom_out_button)

        self.fit_button = QPushButton("Fit to Preview")
        self.fit_button.clicked.connect(self.fit_to_preview)
        button_layout.addWidget(self.fit_button)

        self.autostretch_button = QPushButton("Reapply Stretch")
        self.autostretch_button.clicked.connect(self.autostretch_image)
        button_layout.addWidget(self.autostretch_button)

        self.push_btn = QPushButton("Push to New View")
        self.push_btn.clicked.connect(lambda: self._push_cb() if callable(self._push_cb) else None)
        button_layout.addWidget(self.push_btn)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        button_layout.addWidget(close_btn)

        layout.addLayout(button_layout)

        self.setLayout(layout)
        # Finally, display the initial image
        self.display_image(self.image_array)

    def display_image(self, arr):
        """
        Convert array to QPixmap and display in preview_label.
        We'll respect self.zoom_factor to scale the pixmap.
        """
        if arr is None or arr.size == 0:
            print("WARNING: Trying to display an empty image.")
            return

        # Possibly apply auto-stretch
        if self.stretch_toggle.isChecked():
            arr_display = self.stretch_for_display(arr)
        else:
            # If it's already 8-bit or float, just convert to 8-bit safely
            arr_display = self.to_8bit(arr)
        
        # Convert single-channel => 3 channels if needed
        if arr_display.ndim == 2:
            arr_3ch = np.stack([arr_display]*3, axis=-1)
        elif arr_display.ndim == 3 and arr_display.shape[2] == 1:
            arr_3ch = np.concatenate([arr_display, arr_display, arr_display], axis=2)
        else:
            arr_3ch = arr_display

        # Make QImage => QPixmap
        h, w, c = arr_3ch.shape
        bytes_per_line = w * c
        qimg = QImage(arr_3ch.tobytes(), w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        # Apply zoom factor
        new_w = int(w * self.zoom_factor)
        new_h = int(h * self.zoom_factor)
        if new_w < 1: new_w = 1
        if new_h < 1: new_h = 1

        scaled_pixmap = pixmap.scaled(
            new_w, new_h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

        # Set the label to the scaled pixmap
        self.preview_label.setPixmap(scaled_pixmap)
        # Important: set the label size so the scroll area can scroll if it's bigger
        self.preview_label.resize(scaled_pixmap.size())

    def stretch_for_display(self, arr):
        """
        Applies an auto-stretch to improve visualization:
          1) Compute 0.5 and 99.5 percentiles
          2) Rescale to [0..255]
        """
        arr = arr.astype(np.float32, copy=False)
        mn, mx = np.percentile(arr, (0.5, 99.5))
        if mx > mn:
            arr = (arr - mn) / (mx - mn)
        else:
            arr = np.zeros_like(arr)
        arr = (arr * 255).clip(0, 255).astype(np.uint8)
        return arr

    def eventFilter(self, source, event):
        """
        Capture mouse events on the scroll_area.viewport():
          - Left-button press => start dragging
          - Mouse move => if dragging, pan
          - Left-button release => stop dragging
          - Wheel => zoom in/out
        """
        if source == self.scroll_area.viewport():
            if event.type() == QEvent.Type.MouseButtonPress:
                if event.button() == Qt.MouseButton.LeftButton:
                    self.dragging = True
                    self.last_mouse_pos = event.pos()
                    return True  # We handled it
            elif event.type() == QEvent.Type.MouseMove:
                if self.dragging:
                    # Compute how far we moved
                    delta = event.pos() - self.last_mouse_pos
                    self.last_mouse_pos = event.pos()

                    # Adjust scrollbars
                    h_bar = self.scroll_area.horizontalScrollBar()
                    v_bar = self.scroll_area.verticalScrollBar()
                    h_bar.setValue(h_bar.value() - delta.x())
                    v_bar.setValue(v_bar.value() - delta.y())

                    return True
            elif event.type() == QEvent.Type.MouseButtonRelease:
                if event.button() == Qt.MouseButton.LeftButton:
                    self.dragging = False
                    return True
            elif event.type() == QEvent.Type.Wheel:
                # Zoom in or out
                if event.angleDelta().y() > 0:
                    self.zoom_in()
                else:
                    self.zoom_out()
                event.accept()
                return True
        return super().eventFilter(source, event)

    def to_8bit(self, arr):
        """
        Simple fallback if not using auto-stretch:
        - If float in [0..1], multiply by 255
        - If already 8-bit, do nothing
        """
        if arr.dtype == np.uint8:
            return arr
        # else assume float [0..1]
        return (arr*255).clip(0,255).astype(np.uint8)

    def update_display(self):
        """
        Called when toggling the stretch checkbox. Re-display the image.
        """
        self.display_image(self.image_array)

    def autostretch_image(self):
        """
        Auto-stretch the original image and update preview.
        If the image has multiple channels, we do the same approach as stretch_for_display
        but typically you'd do something more advanced for color.
        """
        arr = self.original_array.copy()
        # e.g., if color, you might do a channel-by-channel approach
        # For simplicity, let's do a grayscale approach using the mean:
        if arr.ndim == 3:
            # we create a single channel for stretch
            g = np.mean(arr, axis=-1)
            arr_stretched = self.stretch_for_display(g)
            # Then broadcast back to 3 channels
            arr_stretched = np.stack([arr_stretched]*3, axis=-1)
        else:
            arr_stretched = self.stretch_for_display(arr)
        self.image_array = arr_stretched
        self.display_image(self.image_array)

    # ---------------------
    # ZOOM Methods
    # ---------------------
    
    def zoom_in(self):
        self.zoom_factor *= 1.2
        self.display_image(self.image_array)

    
    def zoom_out(self):
        self.zoom_factor /= 1.2
        if self.zoom_factor < 0.05:
            self.zoom_factor = 0.05
        self.display_image(self.image_array)

    def fit_to_preview(self):
        """
        Scale the image so it fits inside the scroll_area's viewport.
        We'll measure the image's actual size, compare to the viewport,
        and adjust zoom_factor accordingly.
        """
        if self.image_array is None or self.image_array.size == 0:
            return

        # We'll figure out the image's *unzoomed* dimensions
        arr_display = self.image_array
        if self.stretch_toggle.isChecked():
            arr_display = self.stretch_for_display(arr_display)
        else:
            arr_display = self.to_8bit(arr_display)

        # Convert single-channel => 3 channels if needed, to find w,h
        if arr_display.ndim == 2:
            arr_3ch = np.stack([arr_display]*3, axis=-1)
        elif arr_display.ndim == 3 and arr_display.shape[2] == 1:
            arr_3ch = np.concatenate([arr_display, arr_display, arr_display], axis=2)
        else:
            arr_3ch = arr_display

        h, w, c = arr_3ch.shape

        # The scroll area viewport size
        viewport_size = self.scroll_area.viewport().size()
        vw, vh = viewport_size.width(), viewport_size.height()

        # Compute the scale factor to fit image inside viewport
        scale_w = vw / w if w else 1.0
        scale_h = vh / h if h else 1.0
        new_zoom = min(scale_w, scale_h)
        if new_zoom <= 0:
            new_zoom = 0.01

        self.zoom_factor = new_zoom
        self.display_image(self.image_array)

    def resizeEvent(self, event):
        """
        Refresh displayed pixmap when window is resized,
        only if we want the displayed image to keep fitting automatically.
        But typically, we won't auto-fit on window resize if user is controlling zoom.
        """
        super().resizeEvent(event)
        # Optionally do:
        # self.fit_to_preview()
        # or if you want to keep the user-chosen zoom, just re-display:
        self.display_image(self.image_array)

class MosaicSettingsDialog(QDialog):
    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.setWindowTitle("Mosaic Master Settings")
        self.initUI()

    def initUI(self):
        layout = QFormLayout(self)

        # Number of Stars to Attempt to Use
        self.starCountSpin = CustomSpinBox(minimum=1, maximum=1000,
                                        initial=self.settings.value("mosaic/num_stars", 150, type=int),
                                        step=1)
        layout.addRow("Number of Stars:", self.starCountSpin)

        # Translation Max Tolerance
        self.transTolSpin = CustomDoubleSpinBox(minimum=0.0, maximum=10.0,
                                                initial=self.settings.value("mosaic/translation_max_tolerance", 3.0, type=float),
                                                step=0.1)
        layout.addRow("Translation Max Tolerance:", self.transTolSpin)

        # Scale Min Tolerance
        self.scaleMinSpin = CustomDoubleSpinBox(minimum=0.0, maximum=10.0,
                                                initial=self.settings.value("mosaic/scale_min_tolerance", 0.8, type=float),
                                                step=0.1)
        layout.addRow("Scale Min Tolerance:", self.scaleMinSpin)

        # Scale Max Tolerance
        self.scaleMaxSpin = CustomDoubleSpinBox(minimum=0.0, maximum=10.0,
                                                initial=self.settings.value("mosaic/scale_max_tolerance", 1.25, type=float),
                                                step=0.1)
        layout.addRow("Scale Max Tolerance:", self.scaleMaxSpin)

        # Rotation Max Tolerance
        self.rotationMaxSpin = CustomDoubleSpinBox(minimum=0.0, maximum=180.0,
                                                initial=self.settings.value("mosaic/rotation_max_tolerance", 45.0, type=float),
                                                step=0.1)
        # Force two decimals in display
        self.rotationMaxSpin.lineEdit().setText(f"{self.rotationMaxSpin.value():.2f}")
        layout.addRow("Rotation Max Tolerance (°):", self.rotationMaxSpin)

        # Skew Max Tolerance
        self.skewMaxSpin = CustomDoubleSpinBox(minimum=0.0, maximum=1.0,
                                            initial=self.settings.value("mosaic/skew_max_tolerance", 0.1, type=float),
                                            step=0.01)
        layout.addRow("Skew Max Tolerance:", self.skewMaxSpin)

        # FWHM for Star Detection
        self.fwhmSpin = CustomDoubleSpinBox(minimum=0.0, maximum=20.0,
                                            initial=self.settings.value("mosaic/star_fwhm", 3.0, type=float),
                                            step=0.1)
        self.fwhmSpin.lineEdit().setText(f"{self.fwhmSpin.value():.2f}")
        layout.addRow("FWHM for Star Detection:", self.fwhmSpin)

        # Sigma for Star Detection
        self.sigmaSpin = CustomDoubleSpinBox(minimum=0.0, maximum=10.0,
                                            initial=self.settings.value("mosaic/star_sigma", 3.0, type=float),
                                            step=0.1)
        self.sigmaSpin.lineEdit().setText(f"{self.sigmaSpin.value():.2f}")
        layout.addRow("Sigma for Star Detection:", self.sigmaSpin)

        # Polynomial Degree
        self.polyDegreeSpin = CustomSpinBox(minimum=1, maximum=6,
                                            initial=self.settings.value("mosaic/poly_degree", 3, type=int),
                                            step=1)
        layout.addRow("Polynomial Degree:", self.polyDegreeSpin)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def accept(self):
        # Save the values to QSettings
        self.settings.setValue("mosaic/num_stars", self.starCountSpin.value())
        self.settings.setValue("mosaic/translation_max_tolerance", self.transTolSpin.value())
        self.settings.setValue("mosaic/scale_min_tolerance", self.scaleMinSpin.value())
        self.settings.setValue("mosaic/scale_max_tolerance", self.scaleMaxSpin.value())
        self.settings.setValue("mosaic/rotation_max_tolerance", self.rotationMaxSpin.value())
        self.settings.setValue("mosaic/skew_max_tolerance", self.skewMaxSpin.value())
        self.settings.setValue("mosaic/star_fwhm", self.fwhmSpin.value())
        self.settings.setValue("mosaic/star_sigma", self.sigmaSpin.value())
        self.settings.setValue("mosaic/poly_degree", self.polyDegreeSpin.value())
        super().accept()

# Imported from core to avoid duplication
# Imported from core to avoid duplication
from .core import PolyGradientRemoval



settings = QSettings("SetiAstro", "Seti Astro Suite Pro")

class MosaicMasterDialog(QDialog):
    def __init__(self, settings: QSettings, parent=None, image_manager=None,
                 doc_manager=None, wrench_path=None, spinner_path=None,
                 list_open_docs_fn=None):                                 # ← add param
        super().__init__(parent)
        self.settings = settings
        self.image_manager = image_manager
        self._docman = doc_manager or getattr(parent, "doc_manager", None)
        # same pattern as StellarAlignmentDialog
        if list_open_docs_fn is None:
            cand = getattr(parent, "_list_open_docs", None)
            self._list_open_docs_fn = cand if callable(cand) else None
        else:
            self._list_open_docs_fn = list_open_docs_fn

        self.setWindowTitle("Mosaic Master")
        self.wrench_path = wrench_path
        self.spinner_path = spinner_path

        self.resize(600, 400)
        self.loaded_images = []  
        self.final_mosaic = None
        self.weight_mosaic = None
        self.wcs_metadata = None  # To store mosaic WCS header
        self.astap_exe = self.settings.value("paths/astap", "", type=str)
        # Variables to store stretching parameters:
        self.stretch_original_mins = []
        self.stretch_original_medians = []
        self.was_single_channel = False
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout(self)

        instructions = QLabel(
            "Mosaic Master:\n"
            "1) Add images - Highly Recommend Images be Linear FITS\n"
            "2) Choose Transformation Type:\n"
            "....Partial Affine - Great for Images with Translation, Rotation, and Scaling Needs\n"
            "....Affine - Great for Images that also have skew distortions\n"
            "....Homography - Great for Images that also have lens or perspective distortion\n"
            "....Polynomial Warp - Useful in large mosaics to bend the images together\n"
            "3) Align & Create Mosaic\n"
            "4) Save to Image Manager"
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        btn_layout = QHBoxLayout()
        # Button to add image from disk
        add_btn = QPushButton("Add Image")
        add_btn.clicked.connect(self.add_image)
        btn_layout.addWidget(add_btn)

        # New button to add an image from one of the ImageManager slots
        add_from_view_btn = QPushButton("Add from View")
        add_from_view_btn.setToolTip("Add an image from any open View")
        add_from_view_btn.clicked.connect(self.add_image_from_view)
        btn_layout.addWidget(add_from_view_btn)

        remove_btn = QPushButton("Remove Selected")
        remove_btn.clicked.connect(self.remove_selected)
        btn_layout.addWidget(remove_btn)

        preview_btn = QPushButton("Preview Selected")
        preview_btn.clicked.connect(self.preview_selected)
        btn_layout.addWidget(preview_btn)

        align_btn = QPushButton("Align and Create Mosaic")
        align_btn.clicked.connect(self.align_images)
        btn_layout.addWidget(align_btn)

        save_btn = QPushButton("Save to New View")
        save_btn.clicked.connect(self.save_mosaic_to_new_view)
        btn_layout.addWidget(save_btn)

        # Add the wrench button for settings.
        wrench_btn = QPushButton()
        wrench_btn.setIcon(QIcon(self.wrench_path))
        wrench_btn.setToolTip("Mosaic Settings")
        wrench_btn.clicked.connect(self.openSettings)
        btn_layout.addWidget(wrench_btn)

        layout.addLayout(btn_layout)

        # Horizontal sizer for checkboxes.
        checkbox_layout = QHBoxLayout()
        self.forceBlindCheckBox = QCheckBox("Force Blind Solve (ignore existing WCS)")
        checkbox_layout.addWidget(self.forceBlindCheckBox)
        # New Seestar Mode checkbox:
        self.seestarCheckBox = QCheckBox("Seestar Mode")
        self.seestarCheckBox.setToolTip("Wwen enabled, images are aligned iteratively using astroalign without plate solving.")
        checkbox_layout.addWidget(self.seestarCheckBox)
        layout.addLayout(checkbox_layout)

        self.normalizeCheckBox = QCheckBox("Normalize images (median match)")
        self.normalizeCheckBox.setChecked(True)
        layout.addWidget(self.normalizeCheckBox)

        self.autostretchPreviewCheck = QCheckBox("Autostretch previews only")
        self.autostretchPreviewCheck.setToolTip("Only stretch display previews; keep data linear for processing.")
        self.autostretchPreviewCheck.setChecked(True)
        layout.addWidget(self.autostretchPreviewCheck)

        # Reprojection mode (persisted)
        self.reprojectModeLabel = QLabel("Reprojection mode:")
        self.reprojectModeCombo = QComboBox()
        self.reprojectModeCombo.addItems([
            "Fast — SIP-aware (Exact Remap)",    # default
            "Fast — Homography (Global H)",
            "Precise — Full WCS (astropy.reproject)"
        ])
        self.reprojectModeCombo.setToolTip(
            "Fast — SIP-aware: Dense inverse WCS remap (tiled), honors SIP; removes double stars.\n"
            "Fast — Homography: One global H; very fast, can double stars near edges.\n"
            "Precise — Full WCS: astropy.reproject per channel; slowest, most exact."
        )

        # Persist user choice
        _settings = QSettings("SetiAstro", "SASpro")
        _default_mode = _settings.value("mosaic/reproject_mode",
                                        "Fast — SIP-aware (Exact Remap)")
        if _default_mode not in [self.reprojectModeCombo.itemText(i) for i in range(self.reprojectModeCombo.count())]:
            _default_mode = "Fast — SIP-aware (Exact Remap)"
        self.reprojectModeCombo.setCurrentText(_default_mode)
        self.reprojectModeCombo.currentTextChanged.connect(
            lambda t: QSettings("SetiAstro", "SASpro").setValue("mosaic/reproject_mode", t)
        )

        # Add to layout where the old checkbox lived
        row = QHBoxLayout()
        row.addWidget(self.reprojectModeLabel)
        row.addWidget(self.reprojectModeCombo, 1)
        layout.addLayout(row)


        self.transform_combo = QComboBox()
        self.transform_combo.addItems([
            "Partial Affine Transform",
            "Affine Transform",
            "Homography Transform",
            "Polynomial Warp Based Transform"
        ])
        # Set the default selection to "Affine Transform" (index 1)
        self.transform_combo.setCurrentIndex(1)
        layout.addWidget(QLabel("Select Transformation Method:"))
        layout.addWidget(self.transform_combo)

        self.images_list = QListWidget()
        self.images_list.setSelectionMode(self.images_list.SelectionMode.SingleSelection)
        layout.addWidget(self.images_list)

        self.status_label = QLabel("Status: no images")
        layout.addWidget(self.status_label)

        self.spinnerLabel = QLabel(self)
        self.spinnerLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.spinnerMovie = QMovie(self.spinner_path)
        self.spinnerLabel.setMovie(self.spinnerMovie)
        self.spinnerLabel.hide() 
        layout.addWidget(self.spinnerLabel)

        self.setLayout(layout)

    def _target_median_from_first(self, items):
        # Pick a stable target (median of first image after safe clipping)
        if not items:
            return 0.1
        a0 = items[0]["image"].astype(np.float32)
        if a0.ndim == 3:
            a0 = np.mean(a0, axis=2)
        m = np.median(np.clip(a0, np.percentile(a0, 1), np.percentile(a0, 99)))
        return float(max(m, 1e-6))

    def _normalize_linear(self, arr, target_med):
        """Linear median match only (no stretch/curves). Returns float32 array."""
        img = arr.astype(np.float32, copy=False)
        mono = img if img.ndim == 2 else np.mean(img, axis=2)
        med = np.median(mono)
        if med <= 0:
            return img
        gain = target_med / med
        return img * gain

    def _autostretch_if_requested(self, arr):
        """Used only for preview windows based on the checkbox."""
        if not self.autostretchPreviewCheck.isChecked():
            # Convert linear [0..1-ish] to 8-bit safely for preview
            a = arr.astype(np.float32)
            a = a / max(1e-6, np.percentile(a, 99.9))
            return (np.clip(a, 0, 1) * 255).astype(np.uint8)

        # Your existing stretch_for_display logic:
        return self.stretch_for_display(arr)

    def _sample_grid_pts(self, w, h):
        """9 control points: corners, edge midpoints, center."""
        xs = [0, w/2, w-1]
        ys = [0, h/2, h-1]
        pts = np.array([[x, y] for y in ys for x in xs], dtype=np.float32)  # shape (9,2)
        return pts

    def _compute_wcs_homography(self, src_wcs, dst_wcs, src_shape, dst_shape):
        """
        Build a single 3x3 homography H that maps src pixel coords -> dst pixel coords
        by sampling a small grid of tie-points via WCS.
        """
        h, w = src_shape[:2]
        pts_src = self._sample_grid_pts(w, h)                           # (N,2)
        # src pixels -> world (RA,DEC)
        ra, dec = src_wcs.pixel_to_world_values(pts_src[:,0], pts_src[:,1])
        # world -> dst pixels
        x_dst, y_dst = dst_wcs.world_to_pixel_values(ra, dec)

        pts_dst = np.column_stack([x_dst, y_dst]).astype(np.float32)    # (N,2)

        # Robust H (covers rotation/scale/shear and mild curvature over small fields)
        H, mask = cv2.findHomography(pts_src, pts_dst, method=cv2.RANSAC, ransacReprojThreshold=2.5)
        return H

    def _warp_via_wcs_homography(self, img, src_wcs, dst_wcs, out_shape, H_cache=None):
        """
        Fast path: single homography warp per image (per mosaic WCS).
        If H_cache is provided as a dict, we reuse computed H.
        """
        H, W = out_shape
        key = id(src_wcs)
        if H_cache is not None and key in H_cache:
            H33 = H_cache[key]
        else:
            H33 = self._compute_wcs_homography(src_wcs, dst_wcs, img.shape, (H, W))
            if H_cache is not None:
                H_cache[key] = H33

        if img.ndim == 2:
            return cv2.warpPerspective(img, H33, (W, H), flags=cv2.INTER_LANCZOS4)
        else:
            # Warp each channel
            out = np.empty((H, W, img.shape[2]), dtype=np.float32)
            for c in range(img.shape[2]):
                out[..., c] = cv2.warpPerspective(img[..., c], H33, (W, H), flags=cv2.INTER_LANCZOS4)
            return out

    def _warp_via_wcs_remap_exact(self, src_img, src_wcs, dst_wcs, out_shape, tile=512):
        """
        Inverse mapping: for each mosaic pixel (x_d,y_d), find (x_s,y_s) via
        world = dst_wcs.pixel_to_world(x_d, y_d)
        x_s,y_s = src_wcs.world_to_pixel(world)
        and cv2.remap() from src -> dst. Handles SIP & distortions accurately.
        Tiled to reduce RAM and reuses the same map for all 3 channels if color.
        """
        H, W = out_shape
        is_color = (src_img.ndim == 3)
        dst = np.zeros((H, W, 3), np.float32) if is_color else np.zeros((H, W), np.float32)

        # process in tiles to keep memory bounded
        for y0 in range(0, H, tile):
            y1 = min(y0 + tile, H)
            h = y1 - y0
            ys = np.arange(y0, y1, dtype=np.float64)[:, None]  # (h,1)
            for x0 in range(0, W, tile):
                x1 = min(x0 + tile, W)
                w = x1 - x0
                xs = np.arange(x0, x1, dtype=np.float64)[None, :]  # (1,w)

                # meshgrid of dst pixels in this tile
                Xd, Yd = np.broadcast_to(xs, (h, w)), np.broadcast_to(ys, (h, w))

                # dst->world->src (vectorized)
                # NOTE: astropy wants x, y order
                world = dst_wcs.pixel_to_world(Xd, Yd)
                Xs, Ys = src_wcs.world_to_pixel(world)  # float32/64

                # OpenCV remap expects float32 maps (mapx = x, mapy = y in source image coords)
                mapx = Xs.astype(np.float32)
                mapy = Ys.astype(np.float32)

                if is_color:
                    # remap each channel with same map
                    patch = np.empty((h, w, 3), np.float32)
                    for c in range(3):
                        patch[..., c] = cv2.remap(
                            src_img[..., c], mapx, mapy,
                            interpolation=cv2.INTER_LANCZOS4,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0.0
                        )
                else:
                    patch = cv2.remap(
                        src_img, mapx, mapy,
                        interpolation=cv2.INTER_LANCZOS4,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0.0
                    )

                dst[y0:y1, x0:x1] = patch

        return dst


    def _get_astrometry_api_key(self) -> str:
        # Prefer QSettings; fall back to your legacy loader if present.
        key = self.settings.value("api/astrometry_key", "", type=str)
        if key:
            return key
        try:
            return load_api_key() or ""
        except Exception:
            return ""

    def _is_view_path(self, p) -> bool:
        return isinstance(p, str) and p.startswith("view://")

    def _push_mosaic_to_new_doc(self):
        if self.final_mosaic is None:
            QMessageBox.information(self, "Mosaic Master", "No mosaic available to push.")
            return
        img = self.final_mosaic.astype(np.float32, copy=False)

        # ⚠️ KEEP DIMENSIONALITY — don't force 3ch here
        meta = dict(self.wcs_metadata or {})

        dm = self._docman
        if dm is not None:
            newdoc = dm.open_array(img, metadata=meta, title="Mosaic")
            if hasattr(self.parent(), "_spawn_subwindow_for"):
                self.parent()._spawn_subwindow_for(newdoc)
            QMessageBox.information(self, "Mosaic Master", "Pushed to new view.")
        elif self.image_manager and hasattr(self.image_manager, "create_document"):
            self.image_manager.create_document(image=img, metadata=meta)
            QMessageBox.information(self, "Mosaic Master", "Pushed via image manager.")


    # ---- view helpers (same spirit as StellarAlignmentDialog) ----
    def _safe_call(self, maybe_callable):
        try:
            return maybe_callable() if callable(maybe_callable) else maybe_callable
        except Exception:
            return None

    def _title_for_doc(self, doc, preferred=None):
        # If we were given a candidate title and it's usable, keep it.
        if isinstance(preferred, str) and preferred.strip():
            return preferred.strip()

        # Try common “nice” sources (call methods when present).
        for attr in ("display_name", "windowTitle", "objectName", "title", "name"):
            val = self._safe_call(getattr(doc, attr, None))
            if isinstance(val, str) and val.strip():
                return val.strip()

        # Try metadata fallbacks.
        meta = getattr(doc, "metadata", None)
        if isinstance(meta, dict):
            for k in ("display_name", "title", "name"):
                val = meta.get(k)
                if isinstance(val, str) and val.strip():
                    return val.strip()

        # Last resort.
        return f"Untitled View {id(doc)}"

    def _iter_docs(self):
        """
        Returns a list of (title:str, doc:any) pairs.
        Accepts helpers that yield either bare docs or (title, doc) tuples.
        """
        # Prefer host helper
        items = []
        if self._list_open_docs_fn:
            try:
                raw = list(self._list_open_docs_fn()) or []
            except Exception as e:
                print("[Mosaic] _list_open_docs_fn failed:", e)
                raw = []
            for it in raw:
                if isinstance(it, (tuple, list)) and len(it) >= 2:
                    t, d = it[0], it[1]
                    items.append((self._title_for_doc(d, preferred=t), d))
                else:
                    d = it
                    items.append((self._title_for_doc(d), d))

        if items:
            return items

        # Fallbacks from the doc manager
        docs = []
        try:
            if self._docman and hasattr(self._docman, "iter_documents") and callable(self._docman.iter_documents):
                docs = list(self._docman.iter_documents())
            elif self._docman and hasattr(self._docman, "documents"):
                docs = list(self._docman.documents)
        except Exception:
            docs = []

        return [(self._title_for_doc(d), d) for d in docs]


    # --- title helpers -------------------------------------------------
    def _callmaybe(self, obj, attr):
        """Return obj.attr() if callable, else obj.attr; else None."""
        try:
            v = getattr(obj, attr, None)
            return v() if callable(v) else v
        except Exception:
            return None

    def _best_title_from_obj(self, o):
        """Try common title/name providers on a QWidget/ImageDocument."""
        if o is None:
            return None
        # 1) display_name (method or attr)
        t = self._callmaybe(o, "display_name")
        if isinstance(t, str) and t.strip():
            return t
        # 2) windowTitle / objectName (Qt)
        for a in ("windowTitle", "objectName", "title", "name"):
            t = self._callmaybe(o, a)
            if isinstance(t, str) and t.strip():
                return t
        # 3) metadata.display_name
        try:
            md = getattr(o, "metadata", {}) or {}
            t = md.get("display_name")
            if isinstance(t, str) and t.strip():
                return t
        except Exception:
            pass
        return None

    def _resolve_view_title(self, view, doc, title_hint=None):
        """Prefer the host-provided title, then view, then doc, then a safe default."""
        # host-provided title from _list_open_docs()
        if isinstance(title_hint, str) and title_hint.strip():
            return title_hint.strip()
        # view wins over doc
        t = self._best_title_from_obj(view) or self._best_title_from_obj(doc)
        return t if t else "Untitled View"


    def _pick_view_dialog(self):
        items = self._iter_docs()
        if not items:
            QMessageBox.information(self, "Add from View", "No open views found.")
            return None

        dlg = QDialog(self)
        dlg.setWindowTitle("Select View")
        v = QVBoxLayout(dlg)
        v.addWidget(QLabel("Choose a view to add:"))

        combo = QComboBox()
        for (_, doc) in items:
            # We don't have _fmt_doc_title here, use _title_for_doc?
            # Original code used self._fmt_doc_title(doc) which was missing? 
            # Or maybe _title_for_doc?
            # It seems _fmt_doc_title was never defined in what I extracted.
            # I'll use _title_for_doc.
            title = self._title_for_doc(doc)
            # append dims if available
            try:
                # _doc_image is not defined here.
                # Assuming doc has image attr?
                img = getattr(doc, "image", None)
                if img is not None:
                    h, w = img.shape[:2]
                    title = f"{title} — {w}×{h}"
            except Exception:
                pass
            combo.addItem(title, userData=doc)
        v.addWidget(combo)

        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                            QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        v.addWidget(bb)

        return combo.currentData() if dlg.exec() else None


    def openSettings(self):
        dlg = MosaicSettingsDialog(self.settings, self)
        if dlg.exec():
            self.status_label.setText("Mosaic settings updated.")

    # ---------- Add / Remove ----------
    def add_image(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Add Image(s)",
            "",
            "Images (*.png *.jpg *.jpeg *.tif *.tiff *.fits *.fit *.fz *.fz *.xisf *.cr2 *.nef *.arw *.dng *.raf *.orf *.rw2 *.pef)"
        )
        if paths:
            for path in paths:
                arr, header, bitdepth, ismono = load_image(path)
                header = sanitize_wcs_header(header) if header else None
                wcs_obj = get_wcs_from_header(header) if header else None
                d = {
                    "path": path,
                    "image": arr,
                    "header": header,
                    "wcs": wcs_obj,
                    "bit_depth": bitdepth,
                    "is_mono": ismono,
                    "transform": None
                }
                self.loaded_images.append(d)

                text = os.path.basename(path)
                if wcs_obj is not None:
                    text += " [WCS]"
                item = QListWidgetItem(text)
                item.setToolTip(path)
                self.images_list.addItem(item)
            self.update_status()

    def add_image_from_view(self):
        items = self._iter_docs()
        if not items:
            QMessageBox.information(self, "Add from View", "No open views found.")
            return

        candidates = []
        for title, doc in items:
            # image
            img = None
            try:
                if hasattr(doc, "get_image") and callable(doc.get_image):
                    img = doc.get_image()
                else:
                    img = getattr(doc, "image", None)
            except Exception:
                img = None
            if not isinstance(img, np.ndarray):
                continue

            # metadata/header
            meta = {}
            try:
                if hasattr(doc, "get_metadata") and callable(doc.get_metadata):
                    meta = doc.get_metadata() or {}
                else:
                    meta = getattr(doc, "metadata", {}) or {}
            except Exception:
                meta = {}

            header = (meta.get("original_header")
                    or meta.get("header")
                    or meta.get("fits_header"))
            header = sanitize_wcs_header(header) if header else None
            wcs_obj = get_wcs_from_header(header) if header else None

            h, w = img.shape[:2]
            label = f"{title} — {w}×{h}"
            key = f"view://{id(doc)}"  # stable pseudo-path for removal

            candidates.append((label, img, meta, key, header, wcs_obj))

        if not candidates:
            QMessageBox.information(self, "Add from View", "No views with image data are open.")
            return

        labels = [c[0] for c in candidates]
        choice, ok = QInputDialog.getItem(self, "Select View", "Choose a view to add:", labels, 0, False)
        if not ok or not choice:
            return

        label, image, metadata, path_key, header, wcs_obj = candidates[labels.index(choice)]

        d = {
            "path": path_key,
            "image": image,
            "header": header,
            "wcs": wcs_obj,
            "bit_depth": (metadata.get("bit_depth") if isinstance(metadata, dict) else None),
            "is_mono": (metadata.get("is_mono", image.ndim == 2) if isinstance(metadata, dict) else (image.ndim == 2)),
            "transform": None,
        }
        self.loaded_images.append(d)

        txt = label + (" [WCS]" if wcs_obj is not None else "")
        item = QListWidgetItem(txt)
        item.setToolTip(path_key)
        self.images_list.addItem(item)
        self.update_status()


    def remove_selected(self):
        s = self.images_list.selectedItems()
        if not s:
            QMessageBox.information(self, "Remove", "No item selected.")
            return
        for itm in s:
            row = self.images_list.row(itm)
            self.images_list.takeItem(row)
            p = itm.toolTip()
            self.loaded_images = [x for x in self.loaded_images if x["path"] != p]
        self.update_status()

    def update_status(self):
        c = len(self.loaded_images)
        self.status_label.setText(f"{c} images loaded.")

    # ---------- Preview ----------
    def preview_selected(self):
        s = self.images_list.selectedItems()
        if not s:
            QMessageBox.information(self, "Preview", "No item selected.")
            return
        path = s[0].toolTip()
        for d in self.loaded_images:
            if d["path"] == path:
                preview_image = d["image"]
                disp = self._autostretch_if_requested(preview_image)
                MosaicPreviewWindow(disp, title="Preview Selected", parent=self,
                                    push_cb=self._push_mosaic_to_new_doc).show()
                break

    def _resolve_view_title(self, *objs) -> str:
        """
        Try hard to get a meaningful title from any of the passed objects
        (view, subwindow, doc, etc.). Returns a non-empty string or 'Untitled View'.
        """
        def first_str(x):
            return str(x).strip() if x is not None and str(x).strip() else None

        for o in objs:
            if o is None:
                continue
            # Methods that return a title
            for meth in ("windowTitle", "title", "displayTitle", "text"):
                fn = getattr(o, meth, None)
                if callable(fn):
                    s = first_str(fn())
                    if s: return s
            # Properties/attributes that might hold a title
            for attr in ("title", "display_title", "name", "objectName", "display_name"):
                s = first_str(getattr(o, attr, None))
                if s: return s
            # From metadata
            meta = getattr(o, "metadata", None)
            if isinstance(meta, dict):
                s = first_str(meta.get("display_name") or meta.get("title") or meta.get("name"))
                if s: return s
            # From file path-ish things
            for attr in ("file_path", "filepath", "path", "filename"):
                p = getattr(o, attr, None)
                if p:
                    base = os.path.basename(str(p))
                    s = first_str(base)
                    if s: return s

        return "Untitled View"

    # --- WCS/SIP helpers ---------------------------------------------------------
    def _ensure_image_naxis(self, hdr: dict | fits.Header, shape):
        try:
            h, w = (shape[0], shape[1]) if len(shape) >= 2 else (None, None)
            if h is not None and w is not None:
                hdr["NAXIS"]  = 2
                hdr["NAXIS1"] = int(w)
                hdr["NAXIS2"] = int(h)
        except Exception:
            pass

    def _coerce_float(self, hdr: dict, key: str):
        if key in hdr:
            try:
                hdr[key] = float(hdr[key])
            except Exception:
                try:
                    hdr[key] = float(str(hdr[key]).strip().strip("'"))
                except Exception:
                    del hdr[key]

    def _normalize_wcs_header(self, hdr: dict, img_shape):
        # Required CTYPE + RADESYS
        hdr.setdefault("CTYPE1", "RA---TAN")
        hdr.setdefault("CTYPE2", "DEC--TAN")
        # RADECSYS is deprecated → RADESYS
        if "RADECSYS" in hdr and "RADESYS" not in hdr:
            hdr["RADESYS"] = hdr.pop("RADECSYS")
        hdr.setdefault("RADESYS", "ICRS")
        hdr.setdefault("WCSAXES", 2)

        # Coerce numeric offenders occasionally written as strings
        for k in ("CRPIX1","CRPIX2","CRVAL1","CRVAL2","CD1_1","CD1_2","CD2_1","CD2_2","CDELT1","CDELT2","EQUINOX","CROTA1","CROTA2"):
            self._coerce_float(hdr, k)

        # SIP orders should be ints if present
        for k in ("A_ORDER","B_ORDER","AP_ORDER","BP_ORDER"):
            if k in hdr:
                try:
                    hdr[k] = int(hdr[k])
                except Exception:
                    del hdr[k]

        # Fill missing A/B vs AP/BP if only one is present
        if "A_ORDER" in hdr and "B_ORDER" not in hdr: hdr["B_ORDER"] = hdr["A_ORDER"]
        if "B_ORDER" in hdr and "A_ORDER" not in hdr: hdr["A_ORDER"] = hdr["B_ORDER"]
        if "AP_ORDER" in hdr and "BP_ORDER" not in hdr: hdr["BP_ORDER"] = hdr["AP_ORDER"]
        if "BP_ORDER" in hdr and "AP_ORDER" not in hdr: hdr["AP_ORDER"] = hdr["BP_ORDER"]

        # Make sure dimensions are present (avoids “more axes than image” warning)
        self._ensure_image_naxis(hdr, img_shape)

        return hdr

    def _build_wcs(self, hdr: dict, img_shape):
        # normalize then pass relax=True so SIP and non-standard keys are parsed
        nh = self._normalize_wcs_header(dict(hdr), img_shape)
        return WCS(nh, relax=True)


    # ---------- Align (Entry Point) ----------
    def align_images(self):
        if self.seestarCheckBox.isChecked():
            self.align_images_seestar_mode()
            return

        if len(self.loaded_images) == 0:
            QMessageBox.warning(self, "Align", "No images to align.")
            return

        # Show spinner and start animation.
        self.spinnerLabel.show()
        self.spinnerMovie.start()
        QApplication.processEvents()

        # Step 1: Force blind solve if requested.
        force_blind = self.forceBlindCheckBox.isChecked()
        images_to_process = (self.loaded_images if force_blind
                            else [item for item in self.loaded_images if item.get("wcs") is None])

        # Process each image for plate solving.
        for item in images_to_process:
            # Check if ASTAP is set.
            if not self.astap_exe or not os.path.exists(self.astap_exe):
                executable_filter = "Executables (*.exe);;All Files (*)" if sys.platform.startswith("win") else "Executables (*)"
                new_path, _ = QFileDialog.getOpenFileName(self, "Select ASTAP Executable", "", executable_filter)
                if new_path:
                    self._save_astap_exe_to_settings(new_path)
                    QMessageBox.information(self, "Mosaic Master", "ASTAP path updated successfully.")
                else:
                    QMessageBox.warning(self, "Mosaic Master", "ASTAP path not provided. Falling back to blind solve.")
                    solved_header = self.perform_blind_solve(item)
                    if solved_header:
                        # normalize + build WCS with relax=True so SIP is retained
                        solved_header = self._normalize_wcs_header(solved_header, item["image"].shape)
                        item["wcs"] = self._build_wcs(solved_header, item["image"].shape)
                    continue

            # Attempt ASTAP solve.
            self.status_label.setText(f"Attempting ASTAP solve for {item['path']}...")
            QApplication.processEvents()
            solved_header = self.attempt_astap_solve(item)

            if solved_header is None:
                self.status_label.setText(f"ASTAP failed for {item['path']}. Falling back to blind solve...")
                QApplication.processEvents()
                solved_header = self.perform_blind_solve(item)
            else:
                self.status_label.setText(f"Plate solve successful using ASTAP for {item['path']}.")

            if solved_header:
                # Single, centralized sanitize → keeps SIP and fixes types
                solved_header = self._normalize_wcs_header(solved_header, item["image"].shape)
                item["wcs"] = self._build_wcs(solved_header, item["image"].shape)
            else:
                print(f"Plate solving failed for {item['path']}.")

        # After processing, get all images with valid WCS.
        wcs_items = [x for x in self.loaded_images if x.get("wcs") is not None]
        if not wcs_items:
            print("No images have WCS, skipping WCS alignment.")
            self.spinnerMovie.stop()
            self.spinnerLabel.hide()
            return


        # Use the first image's WCS as reference and compute the mosaic bounding box.
        # (Rebuild with relax=True just in case, then deepcopy.)
        reference_wcs = self._build_wcs(wcs_items[0]["wcs"].to_header(relax=True), wcs_items[0]["image"].shape).deepcopy()
        min_x, min_y, max_x, max_y = self.compute_mosaic_bounding_box(wcs_items, reference_wcs)
        mosaic_width = int(max_x - min_x)
        mosaic_height = int(max_y - min_y)

        if mosaic_width < 1 or mosaic_height < 1:
            print("ERROR: Computed mosaic size is invalid. Check WCS or inputs.")
            self.spinnerMovie.stop()
            self.spinnerLabel.hide()
            return

        # Adjust the reference WCS so that (min_x, min_y) becomes (0,0).
        mosaic_wcs = reference_wcs.deepcopy()
        mosaic_wcs.wcs.crpix[0] -= min_x
        mosaic_wcs.wcs.crpix[1] -= min_y
        # keep SIP in the stored header
        self.wcs_metadata = mosaic_wcs.to_header(relax=True)

        # Set up accumulators.
        is_color = any(not item["is_mono"] for item in wcs_items)

        # stats for optional "unstretch"
        self.stretch_original_mins = []
        self.stretch_original_medians = []
        self.was_single_channel = (not is_color)

        if is_color:
            self.final_mosaic, _ = smart_zeros((mosaic_height, mosaic_width, 3), dtype=np.float32)
        else:
            self.final_mosaic, _ = smart_zeros((mosaic_height, mosaic_width), dtype=np.float32)
        self.weight_mosaic, _ = smart_zeros((mosaic_height, mosaic_width), dtype=np.float32)

        first_image = True
        for idx, itm in enumerate(wcs_items):
            arr = itm["image"]
            self.status_label.setText(f"Mapping {itm['path']} into mosaic frame...")
            QApplication.processEvents()

            img_lin = arr.astype(np.float32, copy=False)

            # --- record original stats for optional "unstretch" ---
            mono_for_stats = img_lin if img_lin.ndim == 2 else np.mean(img_lin, axis=2)
            self.stretch_original_mins.append(float(np.min(mono_for_stats)))
            self.stretch_original_medians.append(float(np.median(mono_for_stats)))

            # 1) optional median normalization only
            if self.normalizeCheckBox.isChecked():
                target_med = getattr(self, "_mosaic_target_median", None)
                if target_med is None:
                    self._mosaic_target_median = self._target_median_from_first(wcs_items)
                    target_med = self._mosaic_target_median
                img_lin = self._normalize_linear(img_lin, target_med)

            # 2) Reprojection (3 modes)
            if not hasattr(self, "_H_cache"):
                self._H_cache = {}

            mode = self.reprojectModeCombo.currentText()

            if mode.startswith("Fast — SIP"):
                # Exact dense remap (SIP-aware), tiled; keep mono as 2D
                reprojected = self._warp_via_wcs_remap_exact(
                    img_lin, itm["wcs"], mosaic_wcs, (mosaic_height, mosaic_width), tile=512
                ).astype(np.float32)
                reproj_red = reprojected[..., 0] if reprojected.ndim == 3 else reprojected

            elif mode.startswith("Fast — Homography"):
                # Single global homography; keep mono as 2D
                reprojected = self._warp_via_wcs_homography(
                    img_lin, itm["wcs"], mosaic_wcs, (mosaic_height, mosaic_width), H_cache=self._H_cache
                ).astype(np.float32)
                reproj_red = reprojected[..., 0] if reprojected.ndim == 3 else reprojected

            else:
                # Precise — Full WCS (astropy.reproject); mono stays 2D
                if img_lin.ndim == 3:
                    channels = []
                    for c in range(3):
                        rpj, _ = reproject_interp((img_lin[..., c], itm["wcs"]), mosaic_wcs,
                                                shape_out=(mosaic_height, mosaic_width))
                        channels.append(np.nan_to_num(rpj, nan=0.0).astype(np.float32))
                    reprojected = np.stack(channels, axis=-1)
                    reproj_red = reprojected[..., 0]
                else:
                    reproj_red, _ = reproject_interp((img_lin, itm["wcs"]), mosaic_wcs,
                                                    shape_out=(mosaic_height, mosaic_width))
                    reprojected = np.nan_to_num(reproj_red, nan=0.0).astype(np.float32)  # 2D mono
                    # no fake stacking here

            self.status_label.setText(f"WCS map: {itm['path']} processed.")
            QApplication.processEvents()

            # --- Stellar Alignment ---
            if not first_image:
                transform_method = self.transform_combo.currentText()
                mosaic_gray = (self.final_mosaic if self.final_mosaic.ndim == 2
                            else np.mean(self.final_mosaic, axis=-1))
                try:
                    self.status_label.setText("Computing affine transform with astroalign...")
                    QApplication.processEvents()
                    # Updated to use core function call with SWAPPED arguments (target=mosaic, source=reproj)
                    transform_obj, (src_pts, dst_pts) = aa_find_transform_with_backoff(mosaic_gray, reproj_red)
                    transform_matrix = transform_obj.params[0:2, :].astype(np.float32)
                    self.status_label.setText("Astroalign computed transform successfully.")
                except Exception as e:
                    self.status_label.setText(f"Astroalign failed: {e}. Using identity transform.")
                    transform_matrix = np.eye(2, 3, dtype=np.float32)

                A = transform_matrix[:, :2]
                scale1 = np.linalg.norm(A[:, 0])
                scale2 = np.linalg.norm(A[:, 1])
                print(f"Computed affine scales: {scale1:.6f}, {scale2:.6f}")

                self.status_label.setText("Affine alignment computed. Warping image...")
                QApplication.processEvents()
                affine_aligned = cv2.warpAffine(reprojected, transform_matrix, (mosaic_width, mosaic_height),
                                                flags=cv2.INTER_LANCZOS4)
                aligned = affine_aligned

                if transform_method in ["Homography Transform", "Polynomial Warp Based Transform"]:
                    self.status_label.setText(f"Starting refined alignment using {transform_method}...")
                    QApplication.processEvents()
                    refined_result = self.refined_alignment(affine_aligned, mosaic_gray, method=transform_method)
                    if refined_result is not None:
                        aligned, best_inliers2 = refined_result
                        self.status_label.setText(f"Refined alignment succeeded with {best_inliers2} inliers.")
                        self.status_label.setText("Refined alignment failed; falling back to affine alignment.")
            else:
                aligned = reprojected
                first_image = False

            # If mosaic is color but aligned is mono, expand for accumulation only
            if is_color and aligned.ndim == 2:
                aligned = np.repeat(aligned[..., None], 3, axis=2)

            gray_aligned = aligned[..., 0] if aligned.ndim == 3 else aligned

            # Compute weight mask
            binary_mask = (gray_aligned > 0).astype(np.uint8)
            smooth_mask = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
            smooth_mask = (smooth_mask / np.max(smooth_mask)) if np.max(smooth_mask) > 0 else binary_mask.astype(np.float32)
            smooth_mask = cv2.GaussianBlur(smooth_mask, (15, 15), 0)

            # Accumulate
            if is_color:
                self.final_mosaic += aligned * smooth_mask[..., np.newaxis]
            else:
                self.final_mosaic += gray_aligned * smooth_mask
            self.weight_mosaic += smooth_mask

            self.status_label.setText(f"Processed: {itm['path']}")
            QApplication.processEvents()

        # Final blending.
        nonzero_mask = (self.weight_mosaic > 0)
        if is_color:
            self.final_mosaic = np.where(self.weight_mosaic[..., None] > 0,
                                        self.final_mosaic / self.weight_mosaic[..., None],
                                        self.final_mosaic)
        else:
            self.final_mosaic[nonzero_mask] = self.final_mosaic[nonzero_mask] / self.weight_mosaic[nonzero_mask]

        print("WCS + Star Alignment Complete.")
        self.status_label.setText("WCS + Star Alignment Complete. De-Normalizing Mosaic...")

        # Call-guard: only unstretch if we normalized AND we actually recorded stats
        did_normalize = self.normalizeCheckBox.isChecked()
        if (did_normalize and
            hasattr(self, "_mosaic_target_median") and self._mosaic_target_median > 0 and
            getattr(self, "stretch_original_medians", None) and len(self.stretch_original_medians) > 0 and
            getattr(self, "stretch_original_mins", None) and len(self.stretch_original_mins) > 0):
            self.final_mosaic = self.unstretch_image(self.final_mosaic)

        self.status_label.setText("Final Mosaic Ready.")
        QApplication.processEvents()

        display_image = (np.stack([self.final_mosaic]*3, axis=-1)
                        if self.final_mosaic.ndim == 2 else self.final_mosaic)
        display_image = self._autostretch_if_requested(display_image)
        MosaicPreviewWindow(display_image, title="Final Mosaic", parent=self,
                            push_cb=self._push_mosaic_to_new_doc).show()

        self.spinnerMovie.stop()
        self.spinnerLabel.hide()
        QApplication.processEvents()
      

    def debayer_image(self, image, file_path, header):
        if file_path.lower().endswith(('.cr2', '.cr3', '.nef', '.arw', '.dng', '.raf', '.orf', '.rw2', '.pef')):
            print(f"Debayering RAW image: {file_path}")
            return debayer_raw_fast(image)
        elif file_path.lower().endswith(('.fits', '.fit', '.fz', '.fz')):
            bayer_pattern = header.get('BAYERPAT')
            if bayer_pattern:
                print(f"Debayering FITS image: {file_path} with Bayer pattern {bayer_pattern}")
                return debayer_fits_fast(image, bayer_pattern)
        return image

    def refine_via_overlap(self, new_gray, mosaic_gray, rough_matrix):
        """
        1) Warp new_gray into mosaic coords using rough_matrix.
        2) Compute overlap region by checking where both warped_new_gray and mosaic_gray > 0.
        3) Restrict astroalign.find_transform() to that overlap only.
        4) Combine the refinement transform with rough_matrix so you have a single final 2×3 matrix
        mapping the original new_gray -> mosaic_gray coordinates.

        Returns the final 2×3 transform matrix for cv2.warpAffine, or None if no overlap or alignment fails.
        """

        # ------------------------------------------
        # A) Warp new_gray with rough_matrix
        # ------------------------------------------
        h_m, w_m = mosaic_gray.shape
        warped_new_gray = cv2.warpAffine(
            new_gray,
            rough_matrix,
            (w_m, h_m),   # (width, height)
            flags=cv2.INTER_LANCZOS4
        )

        # ------------------------------------------
        # B) Determine overlap region (both > 0)
        # ------------------------------------------
        mask_warped = (warped_new_gray > 0)
        mask_mosaic = (mosaic_gray > 0)
        overlap_mask = mask_warped & mask_mosaic

        overlap_pixels = np.count_nonzero(overlap_mask)
        print(f"[DEBUG] Overlap region has {overlap_pixels} pixels.")

        # If there’s not enough overlap, bail out
        if overlap_pixels < 50:
            # Not enough area for star matching
            return None

        # ------------------------------------------
        # C) Mask out everything except the overlap
        # ------------------------------------------
        # We'll use numpy.ma arrays so astroalign sees "valid" data only in overlap region
        warped_new_ma = ma.array(warped_new_gray, mask=~overlap_mask)
        mosaic_ma = ma.array(mosaic_gray, mask=~overlap_mask)

        # ------------------------------------------
        # D) Attempt astroalign on that overlap
        #    Note: the transform we get is from warped_new_gray -> mosaic_gray,
        #    i.e., in "mosaic coordinate space." So it should be *almost* identity
        #    if the rough transform was close, but with some tweak for rotation/translation.
        # ------------------------------------------
        try:
            refine_obj, (src_pts, dst_pts) = astroalign.find_transform(
                warped_new_ma, mosaic_ma,
                max_control_points=50,
                detection_sigma=2
            )
            print(f"[DEBUG] Overlap-limited astroalign success with {len(src_pts)} stars.")
        except Exception as e:
            print(f"[DEBUG] Overlap-limited astroalign failed: {e}")
            return None

        # ------------------------------------------
        # E) Combine refine_obj with the original rough_matrix
        # ------------------------------------------
        # refine_obj.params is typically a 3×3 (similarity).
        # rough_matrix is 2×3 for warpAffine.
        # We'll promote rough_matrix to 3×3, multiply, then convert back.
        refine_mat_3x3 = refine_obj.params  # e.g. a 3×3
        rough_mat_3x3 = np.eye(3, dtype=np.float32)
        rough_mat_3x3[:2, :] = rough_matrix

        # The final transform is refine_mat_3x3 * rough_mat_3x3
        #   (in linear-algebra order, the right-hand transform applies first)
        combined_3x3 = refine_mat_3x3 @ rough_mat_3x3

        # Convert that back to 2×3 for cv2
        final_transform = combined_3x3[:2, :].astype(np.float32)
        return final_transform


    def align_images_seestar_mode(self):
        """ 
        Align images in Seestar Mode by first selecting the center-most image (based on WCS centers)
        as the initial mosaic. Then, sort and add images by increasing distance from the mosaic center.
        If WCS information is present in the header, a rough pre-alignment is computed before refining
        with astroalign.find_transform. After processing all images, failed images are reattempted.

        Final image is produced by summing all warped images and dividing by the pixel counts 
        (sum-then-divide approach).
        """

        if len(self.loaded_images) == 0:
            QMessageBox.warning(self, "Align", "No images to align.")
            return

        self.status_label.setText("🔄 Image Registration Started...")
        self.spinnerLabel.show()
        self.spinnerMovie.start()
        QApplication.processEvents()

        total_files = len(self.loaded_images)  # how many images in total
        if total_files == 0:
            QMessageBox.warning(self, "Align", "No images to align.")
            return

        # Create a QProgressDialog
        progress = QProgressDialog("Aligning images...", "Cancel", 0, total_files, self)
        progress.setWindowTitle("Seestar Alignment")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        progress.setMinimumDuration(0)
        progress.show()

        # We define how many pixels to zero out on each edge AFTER warp
        POST_WARP_BORDER = 10
        THRESHOLD_RATIO = 0.9  # if new pixel < 0.5 * mosaic pixel, skip it

        # -----------------------------------------------------
        # Helper: Extract the world-coordinate center from an image header
        # -----------------------------------------------------
        def get_wcs_center(item):
            header = item.get("header", None)
            if header is None:
                print(f"[DEBUG] No header for {item.get('path','')}")
                return None
            try:
                wcs = WCS(header)
                naxis1 = int(header.get("NAXIS1", 0))
                naxis2 = int(header.get("NAXIS2", 0))
                center_pix = np.array([naxis1 / 2.0, naxis2 / 2.0])
                center_world = wcs.all_pix2world(center_pix[None, :], 1)[0]
                print(f"[DEBUG] {item.get('path','')} center_world: {center_world}")
                return center_world
            except Exception as e:
                print(f"[DEBUG] Failed to get WCS center for {item.get('path','')}: {e}")
                return None

        # -----------------------------------------------------
        # Helper: Count how many stars appear in an image (2D or 3D)
        # -----------------------------------------------------
        def count_stars_in_image(image):
            """
            Use DAOStarFinder to return number of detected stars.
            Expects a 2D image; if 3D, we average over channels.
            You might need to tweak fwhm/threshold for your data.
            """
            if image.ndim == 3:
                image = np.mean(image, axis=2)

            mean_val, median_val, std_val = sigma_clipped_stats(image, sigma=3.0)
            daofind = DAOStarFinder(fwhm=3.0, threshold=5.0 * std_val)
            sources = daofind(image - median_val)

            if sources is None:
                return 0
            else:
                return len(sources)

        # -----------------------------------------------------
        # 1) Gather all images with valid WCS and compute average center
        # -----------------------------------------------------
        centers = []
        valid_items = []
        for item in self.loaded_images:
            center = get_wcs_center(item)
            if center is not None:
                centers.append(center)
                valid_items.append(item)

        if not centers:
            QMessageBox.warning(self, "Align", "No images with valid WCS found.")
            return

        centers = np.array(centers)
        avg_center = np.mean(centers, axis=0)  # or np.median if you prefer

        # -----------------------------------------------------
        # Sort valid items by distance from average center
        # -----------------------------------------------------
        def distance_from_avg(item):
            center = get_wcs_center(item)
            dist = np.linalg.norm(center - avg_center) if center is not None else np.inf
            print(f"[DEBUG] {item['path']} distance from avg: {dist}")
            return dist

        valid_items.sort(key=distance_from_avg)
        for item in valid_items:
            print(f"[DEBUG] Sorted valid item: {item['path']}")

        # -----------------------------------------------------
        # Select from the top few “closest to center,” pick the one with the most stars as the base.
        # -----------------------------------------------------
        top_n = 5
        candidate_subset = valid_items[:top_n]

        best_item = None
        best_star_count = 0
        for candidate in candidate_subset:
            star_count = count_stars_in_image(candidate["image"])
            print(f"[DEBUG] {candidate['path']} star count: {star_count}")
            if star_count > best_star_count:
                best_star_count = star_count
                best_item = candidate

        if best_item is None:
            best_item = valid_items[0]

        base_item = best_item
        print(f"[DEBUG] Selected base image: {base_item['path']} with star count = {best_star_count}")

        # -----------------------------------------------------
        # Reorder loaded_images so that valid (WCS) items come first
        # -----------------------------------------------------
        remaining_items = []
        valid_paths = {v["path"] for v in valid_items}  # set of valid item paths

        for item in self.loaded_images:
            if item["path"] not in valid_paths:
                remaining_items.append(item)

        self.loaded_images = valid_items + remaining_items


        # -----------------------------------------------------
        # Helper: Crop a 5-pixel border from each edge
        # -----------------------------------------------------
        def crop_5px_border(img):
            if img.ndim == 3:
                h, w, c = img.shape
                if h <= 10 or w <= 10:
                    return np.empty((0, 0, c), dtype=img.dtype)
                return img[10:-10, 10:-10, :]
            else:
                h, w = img.shape
                if h <= 10 or w <= 10:
                    return np.empty((0, 0), dtype=img.dtype)
                return img[10:-10, 10:-10]

        # -----------------------------------------------------
        # Helper: Convert image to float32 and normalize if needed
        # -----------------------------------------------------
        def ensure_float32_in_01(img):
            img = img.astype(np.float32, copy=False)
            mx = np.max(img)
            if mx <= 1.0:
                return img
            elif mx <= 255.0:
                return img / 255.0
            elif mx <= 65535.0:
                return img / 65535.0
            else:
                return img / mx if mx > 0 else img

        # -----------------------------------------------------
        # Helper: Compute a rough transform (rotation+translation) using WCS
        # -----------------------------------------------------
        def compute_rough_transform(new_header, ref_header):
            """Compute a rough transformation (rotation and translation) using WCS."""
            try:
                new_wcs = WCS(new_header)
                ref_wcs = WCS(ref_header)
                ref_naxis1 = int(ref_header.get('NAXIS1', 0))
                ref_naxis2 = int(ref_header.get('NAXIS2', 0))
                ref_center = np.array([ref_naxis1 / 2.0, ref_naxis2 / 2.0])
                world_center = ref_wcs.all_pix2world(ref_center[None, :], 1)
                new_center = new_wcs.all_world2pix(world_center, 1)[0]
                ref_offset = ref_center + np.array([1, 0])
                world_offset = ref_wcs.all_pix2world(ref_offset[None, :], 1)[0]
                new_offset = new_wcs.all_world2pix(world_offset[None, :], 1)[0]
                vector = new_offset - new_center
                angle = np.arctan2(vector[1], vector[0])
                cos_a = np.cos(angle)
                sin_a = np.sin(angle)
                R = np.array([[cos_a, -sin_a],
                            [sin_a,  cos_a]])
                t = ref_center - R @ new_center
                rough = np.hstack([R, t.reshape(2, 1)]).astype(np.float32)
                return rough
            except Exception as e:
                print(f"Rough transform skipped: {e}")
                return None

        def compute_rough_transform_seestar(new_header, ref_header):
            """
            Builds a 2×3 transform matrix (rotation + translation) that
            roughly aligns an image with RA/DEC, pixel scale, and focal length
            to a reference image with the same kind of data.

            Required header fields (example):
            new_header["RA"]       (degrees)
            new_header["DEC"]      (degrees)
            new_header["XPIXSZ"]   (microns)
            new_header["YPIXSZ"]   (microns)
            new_header["FOCALLEN"] (mm)
            Similarly for ref_header.

            Returns:
            A 2×3 np.float32 array suitable for cv2.warpAffine,
            or None if missing data or something fails.
            """

            # 1) Parse required fields from new_header
            try:
                ra_new_deg = float(new_header["RA"])       # degrees
                dec_new_deg = float(new_header["DEC"])     # degrees
                xpixsz_new = float(new_header["XPIXSZ"])   # microns
                ypixsz_new = float(new_header["YPIXSZ"])   # microns
                flen_new   = float(new_header["FOCALLEN"]) # mm
            except KeyError as e:
                print(f"[DEBUG] new_header missing key {e}. No rough transform.")
                return None
            except Exception as e:
                print(f"[DEBUG] parse error in new_header => {e}")
                return None

            # 2) Parse required fields from ref_header
            try:
                ra_ref_deg = float(ref_header["RA"])
                dec_ref_deg = float(ref_header["DEC"])
                xpixsz_ref = float(ref_header["XPIXSZ"])
                ypixsz_ref = float(ref_header["YPIXSZ"])
                flen_ref   = float(ref_header["FOCALLEN"])
            except KeyError as e:
                print(f"[DEBUG] ref_header missing key {e}. No rough transform.")
                return None
            except Exception as e:
                print(f"[DEBUG] parse error in ref_header => {e}")
                return None

            # 3) Compute average arcsec/pixel for each image
            #    arcsec_per_pix = 206265 * (XPIXSZ [um]) / (FOCALLEN [mm])
            #    (assuming 1 mm = 1000 um)
            #    If you want to be more precise, you might average XPIXSZ and YPIXSZ,
            #    or handle them separately if the camera has rectangular pixels.
            def arcsec_per_pixel(xpixsz, flen):
                return 206265.0 * (xpixsz / 1000.0) / flen

            scale_new = arcsec_per_pixel(xpixsz_new, flen_new)
            scale_ref = arcsec_per_pixel(xpixsz_ref, flen_ref)

            # For simplicity, let's assume the "reference" image defines the final scale.
            # If the images have different scales, we can do a ratio of scales => ~some "zoom."
            # Here we assume they are close enough, or you can do scale = scale_ref / scale_new
            # to get a small difference in pixel scale if needed.
            scale_ratio = scale_ref / scale_new  # how many ref pixels = 1 new pixel

            # 4) Compute difference in RA/DEC in arcseconds
            #    RA/DEC difference in degrees => multiply by 3600 to get arcseconds
            #    Then convert to pixel shift in reference coordinates.
            d_ra_deg  = (ra_new_deg  - ra_ref_deg)
            d_dec_deg = (dec_new_deg - dec_ref_deg)

            # For small fields, we can do a simple approximation ignoring spherical geometry:
            d_ra_arcsec  = d_ra_deg  * 3600.0 * np.cos(np.radians(dec_ref_deg))  # cos(dec) if you want
            d_dec_arcsec = d_dec_deg * 3600.0

            # If you want a more precise approach, you'd do a spherical trig approach,
            # but for small fields, this is usually good enough.

            # 5) Convert arcseconds => pixel shift in "reference" image coords
            dx_pix = d_ra_arcsec  / scale_ref
            dy_pix = - d_dec_arcsec / scale_ref
            # Note: we do "dy_pix = -d_dec_arcsec" if we assume +DEC => "up" in the image.
            # You may need to invert or swap RA/DEC sign depending on your orientation.

            # 6) Combine the scale ratio (if needed) and shift into a 2×3 transform.
            #    If you want to guess rotation from the difference in rotation angles,
            #    you can do that here. For now, we do no rotation => angle=0 => cos=1, sin=0.
            angle_rad = 0.0
            cos_a = np.cos(angle_rad)
            sin_a = np.sin(angle_rad)

            # scale_ratio is for "zoom," if you want to handle that.
            # If you prefer no scale difference, set scale_ratio = 1.0
            # or if the difference is small, ignore it.
            # scale_ratio = 1.0

            M = np.array([
                [scale_ratio*cos_a, -scale_ratio*sin_a, dx_pix],
                [scale_ratio*sin_a,  scale_ratio*cos_a, dy_pix]
            ], dtype=np.float32)

            print(f"[DEBUG] Seestar rough transform => dx_pix={dx_pix:.2f}, dy_pix={dy_pix:.2f}, scale={scale_ratio:.3f}")
            return M

        # -----------------------------------------------------
        # 1) Initialize mosaic_sum/mosaic_count from the base image
        # -----------------------------------------------------
        base_img = ensure_float32_in_01(base_item["image"])
        if base_item["path"].lower().endswith(('.fits', '.fit', '.fz')):
            if (base_img.ndim == 2 
                and "header" in base_item 
                and base_item["header"] is not None 
                and base_item["header"].get('BAYERPAT')):
                self.status_label.setText(f"Debayering {base_item['path']}")
                QApplication.processEvents()
                base_img = self.debayer_image(base_img, base_item["path"], base_item["header"])
                print(f"[DEBUG] Finished debayering base image: {base_item['path']}")

        base_img = crop_5px_border(base_img)
        if base_img.size == 0:
            QMessageBox.warning(self, "Crop Error", "Initial image is too small after cropping.")
            return
    
        base_median = np.median(base_img)

        # mosaic_sum holds the sum of pixel intensities
        mosaic_sum = base_img.astype(np.float32, copy=True)
        # mosaic_count tracks how many images contributed at each pixel
        if mosaic_sum.ndim == 3:
            # For color: count “contributed” if any channel is >0
            contrib_mask = np.any(mosaic_sum > 0, axis=2)
            mosaic_count = np.zeros_like(mosaic_sum, dtype=np.float32)
            for c in range(mosaic_sum.shape[2]):
                mosaic_count[..., c][contrib_mask] = 1.0
        else:
            contrib_mask = (mosaic_sum > 0)
            mosaic_count = np.zeros_like(mosaic_sum, dtype=np.float32)
            mosaic_count[contrib_mask] = 1.0

        # We’ll use a helper that returns a mono “view” for alignment:
        def get_current_mosaic_gray():
            """Return current mosaic as a float32 array in [0..1], averaged if color."""
            with np.errstate(divide='ignore', invalid='ignore'):
                stacked = mosaic_sum / np.maximum(mosaic_count, 1e-6)
            # If color, reduce to mono for astroalign
            if stacked.ndim == 3:
                return np.mean(stacked, axis=2).astype(np.float32)
            else:
                return stacked

        self.status_label.setText("Starting alignment of subsequent images...")
        QApplication.processEvents()

        # --- Helper: Process alignment for one image, then accumulate into mosaic_sum/count ---
        def process_alignment(item):
            nonlocal mosaic_sum, mosaic_count

            print(f"[DEBUG] Aligning image: {item['path']}")

            try:
                new_img = item["image"].astype(np.float32)
                print(f"[DEBUG] new_img initial shape={new_img.shape}, dtype={new_img.dtype}, "
                    f"min={np.min(new_img):.3f}, max={np.max(new_img):.3f}")

                # Debayer if needed
                if item["path"].lower().endswith(('.fits', '.fit')):
                    if (new_img.ndim == 2 
                        and "header" in item 
                        and item["header"] is not None 
                        and item["header"].get('BAYERPAT')):
                        print(f"[DEBUG] Debayering image: {item['path']}")
                        new_img = self.debayer_image(new_img, item["path"], item["header"])
                        print(f"[DEBUG] debayered new_img shape={new_img.shape}, dtype={new_img.dtype}, "
                            f"min={np.min(new_img):.3f}, max={np.max(new_img):.3f}")

                new_img = crop_5px_border(new_img)
                if new_img.size == 0:
                    print(f"[DEBUG] new_img is empty after cropping => skip.")
                    self.status_label.setText(f"Skipping {item['path']} (too small after crop).")
                    QApplication.processEvents()
                    return False

                self.status_label.setText(f"Removing linear gradient from {item['path']}...")
                QApplication.processEvents()
                # Create an instance of PolyGradientRemoval with degree 1.
                poly_remover = PolyGradientRemoval(new_img, poly_degree=2, downsample_scale=5, num_sample_points=100)
                # Process the image to remove the gradient
                new_img = poly_remover.process()
                print("[DEBUG] Finished polynomial gradient removal on subframe.")

                # If mono:
                self.status_label.setText(f"Normalizing {item['path']}.")
                QApplication.processEvents()
                new_img = new_img-np.min(new_img)
                if new_img.ndim == 2:
                    new_img = stretch_mono_image(new_img, target_median=base_median, normalize=False, apply_curves=False, curves_boost=0.0)

                else:
                    # color approach - you might do a single ratio or call your stretch_color_image
                    new_img = stretch_color_image(new_img, target_median=base_median, linked=False, normalize=False, apply_curves=False, curves_boost=0.0)


                # If mosaic is color and new_img is mono, expand new_img
                # But remember, mosaic_sum is our "canvas"
                if mosaic_sum.ndim == 3 and new_img.ndim == 2:
                    print("[DEBUG] Expanding new_img from 2D => 3D to match mosaic channels.")
                    new_img = np.repeat(new_img[:, :, np.newaxis], mosaic_sum.shape[2], axis=2)
                elif mosaic_sum.ndim == 2 and new_img.ndim == 3:
                    print("[DEBUG] new_img is color but mosaic is mono => average new_img channels.")
                    new_img = np.mean(new_img, axis=2, dtype=np.float32)

                # Create the grayscale for astroalign from mosaic_sum/mosaic_count
                mosaic_gray = get_current_mosaic_gray()

                new_gray = new_img
                if new_gray.ndim == 3:
                    new_gray = np.mean(new_gray, axis=2)

                # Optional: mild stretch to help astroalign
                mosaic_gray = stretch_mono_image(mosaic_gray, target_median=0.1, normalize=False, apply_curves=False, curves_boost=0.0)
                new_gray = stretch_mono_image(new_gray, target_median=0.1, normalize=False, apply_curves=False, curves_boost=0.0)

                # Attempt rough transform via WCS
                rough_matrix = None
                if ("header" in item and item["header"] is not None
                    and "CTYPE1" in item["header"] and "CTYPE2" in item["header"]):
                    # Normal WCS approach
                    rough_matrix = compute_rough_transform(item["header"], base_item["header"])
                    if rough_matrix is not None:
                        print("[DEBUG] WCS-based rough transform:\n", rough_matrix)
                else:
                    # Fallback: check if we have RA,DEC,XPIXSZ,FOCALLEN, etc.
                    rough_matrix = compute_rough_transform_seestar(item["header"], base_item["header"])
                    if rough_matrix is not None:
                        print("[DEBUG] Seestar rough transform:\n", rough_matrix)

                self.status_label.setText(f"Astroaligning for {item['path']}...")
                QApplication.processEvents()

                # Find transform
                transform_matrix = None
                try:
                    # astroalign wants double precision
                    transform_obj, (src_pts, dst_pts) = astroalign.find_transform(
                        new_gray,
                        mosaic_gray,
                        max_control_points=50,
                        detection_sigma=4,
                        min_area=5
                    )
                    transform_matrix = transform_obj.params[0:2, :].astype(np.float32)
                    print(f"[DEBUG] Astroalign success with {len(src_pts)} stars. transform_matrix:\n{transform_matrix}")

                except Exception as e:
                    print(f"[DEBUG] astroalign.find_transform failed: {e}")
                    self.status_label.setText(f"Alignment failed: {e}")
                    QApplication.processEvents()

                    # Fallback if rough_matrix is available
                    if rough_matrix is not None:
                        print("[DEBUG] Attempting refine_via_overlap fallback with rough_matrix.")
                        transform_matrix = self.refine_via_overlap(new_gray, mosaic_gray, rough_matrix)
                        if transform_matrix is None:
                            print("[DEBUG] Overlap approach also failed => skip this image.")
                            return False
                        print("[DEBUG] Overlap-based refinement succeeded. transform_matrix:\n", transform_matrix)
                    else:
                        print("[DEBUG] No rough transform => skipping image.")
                        return False

                if transform_matrix is None:
                    self.status_label.setText("No transform found; skipping image.")
                    print("[DEBUG] transform_matrix is None => returning False")
                    return False

                self.status_label.setText(f"Astroalign success for {item['path']}")
                QApplication.processEvents()

                # Warp the new image onto mosaic_sum's coordinate system
                h_m, w_m = mosaic_sum.shape[:2]
                new_h, new_w = new_img.shape[:2]

                mosaic_corners = np.array([[0, 0], [w_m, 0], [0, h_m], [w_m, h_m]], dtype=np.float32)
                new_img_corners = np.array([[0, 0], [new_w, 0], [0, new_h], [new_w, new_h]], dtype=np.float32)
                ones = np.ones((4, 1), dtype=np.float32)
                new_img_corners_hom = np.hstack([new_img_corners, ones])

                warped_corners = (transform_matrix @ new_img_corners_hom.T).T
                all_corners = np.vstack([mosaic_corners, warped_corners])
                min_xy = np.min(all_corners, axis=0)
                max_xy = np.max(all_corners, axis=0)

                margin = 10
                new_canvas_width = int(np.ceil(max_xy[0] - min_xy[0])) + margin
                new_canvas_height = int(np.ceil(max_xy[1] - min_xy[1])) + margin
                shift = -min_xy + np.array([margin / 2, margin / 2])
                shift_int = np.round(shift).astype(int)
                y0, x0 = shift_int[1], shift_int[0]

                print(f"[DEBUG] new_canvas_width={new_canvas_width}, new_canvas_height={new_canvas_height}")
                print(f"[DEBUG] shift={shift}, shift_int={shift_int}, y0={y0}, x0={x0}")

                # Expand mosaic_sum/mosaic_count to fit new canvas if needed
                expanded_sum = None
                expanded_count = None

                if mosaic_sum.ndim == 3:
                    channels = mosaic_sum.shape[2]
                    expanded_sum, _ = smart_zeros((new_canvas_height, new_canvas_width, channels), dtype=np.float32)
                    expanded_count = np.zeros_like(expanded_sum, dtype=np.float32)
                    # Copy existing mosaic_sum/mosaic_count
                    expanded_sum[y0:y0+h_m, x0:x0+w_m, :] = mosaic_sum
                    expanded_count[y0:y0+h_m, x0:x0+w_m, :] = mosaic_count
                else:
                    expanded_sum, _ = smart_zeros((new_canvas_height, new_canvas_width), dtype=np.float32)
                    expanded_count = np.zeros_like(expanded_sum, dtype=np.float32)
                    expanded_sum[y0:y0+h_m, x0:x0+w_m] = mosaic_sum
                    expanded_count[y0:y0+h_m, x0:x0+w_m] = mosaic_count

                # Adjust transform for the shift
                new_transform = transform_matrix.copy()
                new_transform[0, 2] += shift[0]
                new_transform[1, 2] += shift[1]

                # Warp the new_img
                try:
                    warped_new = cv2.warpAffine(
                        new_img,
                        new_transform,
                        (new_canvas_width, new_canvas_height),
                        flags=cv2.INTER_LANCZOS4
                    )
                except cv2.error as cv2_err:
                    print(f"[OpenCV] warpAffine error => {cv2_err}")
                    return False

                # 1) Build the "border_mask" that excludes the outer POST_WARP_BORDER region
                h_w, w_w = warped_new.shape[:2]
                border_mask = np.ones((h_w, w_w), dtype=bool)
                b = POST_WARP_BORDER
                border_mask[:b, :] = False
                border_mask[-b:, :] = False
                border_mask[:, :b] = False
                border_mask[:, -b:] = False

                # 2) Build a "valid_mask" by warping an image of ones using INTER_NEAREST.
                ones_img = np.ones(new_img.shape[:2], dtype=np.uint8)
                warped_mask = cv2.warpAffine(
                    ones_img,
                    new_transform,
                    (new_canvas_width, new_canvas_height),
                    flags=cv2.INTER_NEAREST
                )
                valid_mask = (warped_mask == 1)

                # 2b) Compute a feathering weight from the valid_mask:
                # Compute the distance transform of the valid_mask.
                # This gives, for each pixel inside the valid region, its distance (in pixels)
                # from the nearest 0 (i.e. from the border of the valid region).
                FEATHER_RADIUS = 20  # adjust as needed; this is the maximum distance for feathering
                # Convert valid_mask to uint8 for distance transform.
                valid_uint8 = valid_mask.astype(np.uint8)
                dist = cv2.distanceTransform(valid_uint8, cv2.DIST_L2, 5)
                # Compute weights: linear ramp from 0 (at distance=0) to 1 (at distance>=FEATHER_RADIUS).
                feather_weights = np.clip(dist / FEATHER_RADIUS, 0, 1)

                # 3) Build "mosaic_gray" = the grayscale of (expanded_sum / expanded_count)
                mosaic_gray = np.zeros((new_canvas_height, new_canvas_width), dtype=np.float32)
                if expanded_sum.ndim == 2:
                    nonzero_mask = (expanded_count > 0)
                    mosaic_gray[nonzero_mask] = expanded_sum[nonzero_mask] / expanded_count[nonzero_mask]
                else:
                    sum_channels = np.sum(expanded_sum, axis=2)
                    sum_counts = np.sum(expanded_count, axis=2)
                    nonzero_mask = (sum_counts > 0)
                    mosaic_gray[nonzero_mask] = sum_channels[nonzero_mask] / sum_counts[nonzero_mask]

                # 4) Build new_gray for the warped image
                if warped_new.ndim == 2:
                    new_gray = warped_new
                else:
                    new_gray = np.mean(warped_new, axis=2)

                # 5) Build the brightness_mask: accept pixels where new_gray >= THRESHOLD_RATIO * mosaic_gray.
                brightness_mask = (new_gray >= THRESHOLD_RATIO * mosaic_gray)

                # 6) Combine masks.
                # Here we combine border_mask, brightness_mask, and valid_mask.
                # The final effective weight will be the feathering weight from within the valid_mask,
                # applied only where the other masks also pass.
                combined_mask = border_mask & valid_mask & brightness_mask

                # 7) Add only those pixels, using the feathering weights.
                if warped_new.ndim == 2:
                    # MONO: weight each pixel accordingly.
                    expanded_sum[combined_mask] += warped_new[combined_mask] * feather_weights[combined_mask]
                    expanded_count[combined_mask] += feather_weights[combined_mask]
                else:
                    # COLOR: do it per channel.
                    for c in range(warped_new.shape[2]):
                        expanded_sum[..., c][combined_mask] += warped_new[..., c][combined_mask] * feather_weights[combined_mask]
                        expanded_count[..., c][combined_mask] += feather_weights[combined_mask]

                # 8) Update mosaic_sum/mosaic_count.
                mosaic_sum = expanded_sum
                mosaic_count = expanded_count


                self.status_label.setText(f"Integrated image: {item['path']}")
                QApplication.processEvents()

                print("[DEBUG] process_alignment done successfully for this image.")
                return True

            except Exception as e:
                print(f"[DEBUG] process_alignment error => {e}")
                traceback.print_exc()
                return False

        # ---------------------------------------------------------
        # Process each subsequent image once
        # ---------------------------------------------------------
        failed_items = []
        # We'll do a normal for-loop with enumerate so we can pass index to progress
        for i, item in enumerate(self.loaded_images):
            # If user cancels
            if progress.wasCanceled():
                self.status_label.setText("Alignment canceled by user.")
                break

            # Update the progress label
            progress.setLabelText(f"Aligning image {i+1}/{total_files}: {item['path']}")
            progress.setValue(i)  # update the progress bar

            if item is base_item:
                continue

            success = process_alignment(item)
            if not success:
                failed_items.append(item)

            QApplication.processEvents()  # allow UI updates

        # Final step: mark progress done
        progress.setValue(total_files)


        # ---------------------------------------------------------
        # Reattempt failed images (max 1 additional attempt)
        # ---------------------------------------------------------
        max_retries = 1
        attempt = 1
        while failed_items and attempt <= max_retries:
            reattempt_count = len(failed_items)
            self.status_label.setText(f"Reattempting alignment for {reattempt_count} images (Attempt {attempt})")
            print(f"Reattempting alignment for {reattempt_count} images (Attempt {attempt})")
            QApplication.processEvents()

            # 2) Reset the existing progress bar to track the reattempt pass
            progress.setRange(0, reattempt_count)
            progress.setValue(0)
            progress.setLabelText(f"Reattempt pass {attempt}...")

            current_failures = []
            for i, item in enumerate(failed_items):
                if progress.wasCanceled():
                    self.status_label.setText("Reattempt canceled by user.")
                    break

                progress.setLabelText(f"Reattempting {item['path']} ({i+1}/{reattempt_count})")
                progress.setValue(i)
                QApplication.processEvents()

                success = process_alignment(item)
                if not success:
                    current_failures.append(item)

            progress.setValue(reattempt_count)  # done reattempt pass

            failed_items = current_failures
            attempt += 1

        # All passes complete
        progress.close()
        self.status_label.setText("All alignment attempts complete.")
        QApplication.processEvents()

        # ---------------------------------------------------------
        # Finally, build the average mosaic = mosaic_sum / mosaic_count
        # ---------------------------------------------------------
        with np.errstate(divide='ignore', invalid='ignore'):
            final_mosaic = np.zeros_like(mosaic_sum, dtype=np.float32)
            if final_mosaic.ndim == 2:
                nonzero = (mosaic_count > 0)
                final_mosaic[nonzero] = mosaic_sum[nonzero] / mosaic_count[nonzero]
            else:
                # color
                for c in range(final_mosaic.shape[2]):
                    chan_nonzero = (mosaic_count[..., c] > 0)
                    final_mosaic[..., c][chan_nonzero] = (
                        mosaic_sum[..., c][chan_nonzero] 
                        / mosaic_count[..., c][chan_nonzero]
                    )

        # Optional: you could do a final normalization
        max_val = np.max(final_mosaic)
        if max_val > 0:
            final_mosaic /= max_val

        self.final_mosaic = final_mosaic
        self.spinnerMovie.stop()
        self.spinnerLabel.hide()
        self.status_label.setText("Seestar Mode alignment (sum/divide) complete.")
        QApplication.processEvents()

        display_image = self.stretch_for_display(self.final_mosaic)
        mosaic_win = MosaicPreviewWindow(display_image, title="Robot Telescope Mosaic", parent=self,
                    push_cb=self._push_mosaic_to_new_doc).show()

    # ---------- Star alignment using triangle matching ----------

    def refined_alignment(self, affine_aligned, mosaic_img, method="Homography Transform"):
        """
        Refined alignment that assumes affine_aligned is the result of the affine alignment step.
        It re-detects stars in the candidate (overlap) region and computes a refined transform from
        affine_aligned to mosaic_img. Then it applies the refined transform to affine_aligned and
        returns the fully warped image.
        
        Returns:
        - For "Homography Transform": (warped_image, inlier_count)
        - For "Polynomial Warp Based Transform": (warped_image, inlier_count)
        - If refinement fails, returns None.
        """
        print("\n--- Starting Refined Alignment ---")
        poly_degree = self.settings.value("mosaic/poly_degree", 3, type=int)
        self.status_label.setText("Refinement: Converting images to grayscale...")
        QApplication.processEvents()
        
        # Convert images to grayscale
        if affine_aligned.ndim == 3:
            affine_aligned_gray = np.mean(affine_aligned, axis=-1)
        else:
            affine_aligned_gray = affine_aligned
        if mosaic_img.ndim == 3:
            mosaic_gray = np.mean(mosaic_img, axis=-1)
        else:
            mosaic_gray = mosaic_img

        print("Grayscale conversion done.")
        
        # Compute overlap mask
        self.status_label.setText("Refinement: Computing overlap mask...")
        QApplication.processEvents()
        overlap_mask = (mosaic_gray > 0) & (affine_aligned_gray > 0)

        # Detect stars
        self.status_label.setText("Refinement: Detecting stars in mosaic and affine-aligned images...")
        QApplication.processEvents()
        # Increase max_stars to 50 for debugging purposes.
        mosaic_stars_masked = self.detect_stars(np.where(overlap_mask, mosaic_gray, 0), max_stars=300)
        new_stars_aligned = self.detect_stars(np.where(overlap_mask, affine_aligned_gray, 0), max_stars=300)

        # Debug: Print out the star lists.
        print("Mosaic stars (refined alignment):")
        for s in mosaic_stars_masked:
            print(f"({s[0]:.2f}, {s[1]:.2f}) flux: {s[2]:.2f}")
        print("New stars (refined alignment):")
        for s in new_stars_aligned:
            print(f"({s[0]:.2f}, {s[1]:.2f}) flux: {s[2]:.2f}")

        self.status_label.setText(f"Refinement: Detected {len(mosaic_stars_masked)} mosaic stars and {len(new_stars_aligned)} new stars.")
        QApplication.processEvents()

        if len(mosaic_stars_masked) < 4 or len(new_stars_aligned) < 4:
            self.status_label.setText("Refinement: Not enough stars detected in candidate region.")
            return None

        # Match stars using position and flux.
        self.status_label.setText("Refinement: Matching stars...")
        QApplication.processEvents()
        # For debugging, you might try a higher threshold.
        matches = self.match_stars(new_stars_aligned, mosaic_stars_masked, distance_thresh=10.0, flux_thresh=1.0)
        print(f"Matched stars: {len(matches)}")
        self.status_label.setText(f"Refinement: {len(matches)} matches found.")
        if len(matches) < 4:
            self.status_label.setText("Refinement: Not enough matched stars for refined transform.")
            return None

        src_pts = np.float32([match[0][:2] for match in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([match[1][:2] for match in matches]).reshape(-1, 1, 2)

        if method == "Homography Transform":
            self.status_label.setText("Refinement: Computing homography transform...")
            QApplication.processEvents()
            H_refined, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            inliers = int(np.count_nonzero(mask)) if mask is not None else 0
            self.status_label.setText(f"Refinement: Homography computed with {inliers} inliers. Warping image...")
            QApplication.processEvents()
            if H_refined is None:
                self.status_label.setText("Refinement: Homography estimation failed.")
                return None
            warped_image = cv2.warpPerspective(affine_aligned, H_refined,
                                                (affine_aligned.shape[1], affine_aligned.shape[0]),
                                                flags=cv2.INTER_LANCZOS4)
            return (warped_image, inliers)
        elif method == "Polynomial Warp Based Transform":
            self.status_label.setText("Refinement: Computing Polynomial Warp based transform...")
            QApplication.processEvents()
            # Extract control points from matches.
            src_pts = np.float32([match[0][:2] for match in matches])
            dst_pts = np.float32([match[1][:2] for match in matches])
            # Call the static Polynomial Warp warp function.
            try:
                warped_image = MosaicMasterDialog.poly_warp(affine_aligned, src_pts, dst_pts, degree=poly_degree, status_label=self.status_label)
                inliers = len(matches)
                self.status_label.setText(f"Refinement: Polynomial Warp warp applied with {inliers} matched points.")
                return (warped_image, inliers)
            except Exception as e:
                self.status_label.setText(f"Refinement: Polynomial Warp warp failed: {e}")
                return None
        else:
            self.status_label.setText("Refinement: Unexpected transformation method.")
            return None

    @staticmethod
    def poly_warp(image, src_pts, dst_pts, degree=3, status_label=None):
        """
        Warp `image` using a polynomial transformation of the specified degree.
        
        The transformation is defined as:
        u = sum_{i+j<=degree} a_{ij} * x^i * y^j
        v = sum_{i+j<=degree} b_{ij} * x^i * y^j
        
        where the coefficients are solved via least squares using the control points.
        
        Parameters:
        image: Input image.
        src_pts: numpy array of shape (N,2) with control points in the source image.
        dst_pts: numpy array of shape (N,2) with corresponding control points in the destination.
        degree: Degree of the polynomial transformation (allowed 1 to 6, default=3).
        status_label: If provided, a Qt widget where progress messages are displayed.
        
        Returns:
        The warped image.
        """
        h, w = image.shape[:2]
        
        # Function to build the design matrix for points given a degree.
        def build_design_matrix(pts, degree):
            # pts: (N,2) array, where each row is (x, y)
            N = pts.shape[0]
            terms = []
            x = pts[:, 0]
            y = pts[:, 1]
            # Loop over exponents i and j with i+j <= degree.
            for i in range(degree + 1):
                for j in range(degree + 1 - i):
                    terms.append((x**i) * (y**j))
            X = np.vstack(terms).T  # shape: (N, number_of_terms)
            return X

        # Build the design matrix for the control points.
        if status_label is not None:
            status_label.setText("Polynomial warp: Building design matrix for control points...")
            QApplication.processEvents()
        X = build_design_matrix(src_pts, degree)
        
        # Destination coordinates.
        U = dst_pts[:, 0]
        V = dst_pts[:, 1]
        
        if status_label is not None:
            status_label.setText("Polynomial warp: Solving for polynomial coefficients...")
            QApplication.processEvents()
        
        # Solve the least-squares problem.
        coeffs_u, _, _, _ = np.linalg.lstsq(X, U, rcond=None)
        coeffs_v, _, _, _ = np.linalg.lstsq(X, V, rcond=None)
        
        if status_label is not None:
            status_label.setText("Polynomial warp: Computing full mapping for image...")
            QApplication.processEvents()
        
        # Build a full grid of coordinates.
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        flat_x = grid_x.ravel()
        flat_y = grid_y.ravel()
        pts_full = np.vstack([flat_x, flat_y]).T  # shape: (h*w, 2)
        
        # Build design matrix for the full grid.
        X_full = build_design_matrix(pts_full, degree)
        
        # Evaluate the polynomial mappings.
        map_u = np.dot(X_full, coeffs_u).reshape(h, w).astype(np.float32)
        map_v = np.dot(X_full, coeffs_v).reshape(h, w).astype(np.float32)
        
        if status_label is not None:
            status_label.setText("Polynomial warp: Full mapping computed. Remapping image...")
            QApplication.processEvents()
        
        # Remap the image.
        warped = cv2.remap(image, map_u, map_v, interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT)
        
        if status_label is not None:
            status_label.setText("Polynomial warp: Image warped.")
            QApplication.processEvents()
        
        return warped



    def stretch_for_display(self, arr):
        """
        Uses your global stretch_mono_image or stretch_color_image to produce
        a display-ready 8-bit image. For color images, uses unlinked stretching.
        """
        arr = arr.astype(np.float32)

        # Decide if it's mono or color based on shape
        if arr.ndim == 3 and arr.shape[2] == 3:
            # Color image => use stretch_color_image with unlinked stretching
            stretched = stretch_color_image(
                image=arr,
                target_median=0.25,   # Adjust if you prefer a different default
                linked=False,         # "unlinked" mode
                normalize=True,       # Ensures final values are in [0,1]
                apply_curves=False,   # Adjust if needed
                curves_boost=0.0
            )
        else:
            # Mono image => use stretch_mono_image
            stretched = stretch_mono_image(
                image=arr,
                target_median=0.25,
                normalize=True,
                apply_curves=False,
                curves_boost=0.0
            )

        # Convert [0,1] => [0,255]
        disp = (stretched * 255.0).clip(0, 255).astype(np.uint8)
        return disp


    def detect_stars(self, image2d, max_stars=50):
        # Retrieve user-defined values for sigma and fwhm.
        sigma_val = self.settings.value("mosaic/star_sigma", 3.0, type=float)
        fwhm_val = self.settings.value("mosaic/star_fwhm", 3.0, type=float)
        
        mean_val, median_val, std_val = sigma_clipped_stats(image2d, sigma=3.0)
        # Use the user-defined fwhm and scale the threshold by the standard deviation.
        daofind = DAOStarFinder(threshold=sigma_val * std_val, fwhm=fwhm_val)
        sources = daofind(image2d - median_val)
        if sources is None or len(sources) == 0:
            return []
        x_coords = sources['xcentroid'].data
        y_coords = sources['ycentroid'].data
        flux = sources['flux'].data
        # Sort stars by brightness (flux) and select the top ones.
        sorted_indices = np.argsort(-flux)
        top_indices = sorted_indices[:max_stars]
        stars = [(x_coords[i], y_coords[i], flux[i]) for i in top_indices]
        return stars


    def match_stars(self, new_stars, mosaic_stars, distance_thresh=10.0, flux_thresh=0.2):
        """
        Matches stars between two lists.
        new_stars and mosaic_stars should be lists of (x, y, flux).
        
        This version normalizes the flux values in each list by dividing by the median flux.
        
        distance_thresh: maximum distance (in pixels) allowed for a match.
        flux_thresh: allowed absolute difference in normalized flux.
                    For example, if set to 0.2, then the normalized flux difference must be less than 0.2.
                    
        Returns a list of matched pairs: [(new_star, mosaic_star), ...]
        """
        # If either list is empty, return an empty match list.
        if not new_stars or not mosaic_stars:
            return []
        
        # Compute median fluxes.
        new_fluxes = [s[2] for s in new_stars]
        mosaic_fluxes = [s[2] for s in mosaic_stars]
        new_median = np.median(new_fluxes) if new_fluxes else 1.0
        mosaic_median = np.median(mosaic_fluxes) if mosaic_fluxes else 1.0

        # Normalize the flux for each star.
        norm_new_stars = [(s[0], s[1], s[2] / new_median) for s in new_stars]
        norm_mosaic_stars = [(s[0], s[1], s[2] / mosaic_median) for s in mosaic_stars]

        matches = []
        for ns in norm_new_stars:
            best_match = None
            best_distance = float('inf')
            for ms in norm_mosaic_stars:
                dx = ns[0] - ms[0]
                dy = ns[1] - ms[1]
                dist = np.hypot(dx, dy)
                # Check spatial proximity.
                if dist < distance_thresh and dist < best_distance:
                    # Check if the normalized fluxes are similar.
                    # Here, flux_thresh is an absolute threshold on the difference.
                    if abs(ns[2] - ms[2]) < flux_thresh:
                        best_match = ms
                        best_distance = dist
            if best_match is not None:
                matches.append((ns, best_match))
        return matches

    def estimate_transform(self, source_stars, dest_stars):
        min_len = min(len(source_stars), len(dest_stars))
        if min_len < 3:
            return None
        src_pts = np.float32(source_stars[:min_len])
        dst_pts = np.float32(dest_stars[:min_len])
        matrix, inliers = cv2.estimateAffinePartial2D(
            src_pts, dst_pts,
            method=cv2.RANSAC, ransacReprojThreshold=3.0
        )
        if matrix is None:
            return None
        full_mat = np.eye(3, dtype=np.float32)
        full_mat[:2] = matrix
        return full_mat


    def compute_mosaic_bounding_box(self, wcs_items, reference_wcs):
        """
        Compute the mosaic bounding box in pixel coordinates relative to a shared WCS frame,
        properly accounting for rotation and orientation dynamically.
        """
        all_pixels = []

        for itm in wcs_items:
            wcs = itm["wcs"]
            H, W = itm["image"].shape[:2]  # Use only the height and width

            # Get image corner coordinates in world coordinates (RA/Dec)
            pixel_corners = np.array([
                [0, 0],         # Top-left
                [W - 1, 0],     # Top-right
                [0, H - 1],     # Bottom-left
                [W - 1, H - 1]  # Bottom-right
            ])

            # Convert pixel to world coordinates (RA, Dec)
            world_coords = np.column_stack(wcs.pixel_to_world_values(pixel_corners[:, 0], pixel_corners[:, 1]))

            # Convert RA/Dec to pixel coordinates in the reference WCS
            sky_coords = SkyCoord(ra=world_coords[:, 0] * u.deg, dec=world_coords[:, 1] * u.deg, frame='icrs')
            pixel_coords = skycoord_to_pixel(sky_coords, reference_wcs)

            # Ensure we're only using the first two values (x, y)
            all_pixels.append(np.column_stack(pixel_coords[:2]))

        # Stack all pixel coordinates and compute bounding box
        all_pixels = np.vstack(all_pixels)

        min_x, max_x = np.min(all_pixels[:, 0]), np.max(all_pixels[:, 0])
        min_y, max_y = np.min(all_pixels[:, 1]), np.max(all_pixels[:, 1])

        # **Determine whether the mosaic is wider or taller dynamically**
        width = max_x - min_x
        height = max_y - min_y

        print(f"Detected Bounding Box (X={min_x} to {max_x}, Y={min_y} to {max_y})")
        print(f"Calculated Mosaic Size: Width={width}, Height={height}")
        self.status_label.setText(f"Detected Bounding Box: X={min_x} to {max_x}, Y={min_y} to {max_y}")
        self.status_label.setText(f"Calculated Mosaic Size: Width={width}, Height={height}")
        QApplication.processEvents()

        return int(np.floor(min_x)), int(np.floor(min_y)), int(np.ceil(max_x)), int(np.ceil(max_y))

    def save_mosaic_to_new_view(self):
        """
        Finalize and create a brand-new document/view in SASpro.
        """
        finalized = self.finalize_mosaic()
        if finalized is None:
            QMessageBox.information(self, "Mosaic Master", "No mosaic to save.")
            return

        img, meta = finalized

        if not self._docman:
            QMessageBox.warning(self, "Mosaic Master", "No document manager available.")
            return

        title = self._proposed_mosaic_title()

        try:
            if hasattr(self._docman, "open_array") and callable(self._docman.open_array):
                newdoc = self._docman.open_array(img, metadata=meta, title=title)
            elif hasattr(self._docman, "open_numpy") and callable(self._docman.open_numpy):
                newdoc = self._docman.open_numpy(img, metadata=meta, title=title)
            else:
                # very old API
                newdoc = self._docman.create_document(image=img, metadata=meta, name=title)

            # show it in a subwindow if the host provides the helper
            parent = self.parent() if hasattr(self, "parent") else None
            if parent and hasattr(parent, "_spawn_subwindow_for"):
                parent._spawn_subwindow_for(newdoc)

            QMessageBox.information(self, "Mosaic Master", "Created a new view for the mosaic.")
            self.accept()

        except Exception as e:
            QMessageBox.critical(self, "Mosaic Master", f"Failed to create new view:\n{e}")

    def _proposed_mosaic_title(self) -> str:
        return "Mosaic"

    def finalize_mosaic(self):
        """
        Prepare the final mosaic (and its WCS metadata) and return (image, metadata).
        Does not push anywhere. Returns None on failure.
        """
        if self.final_mosaic is None:
            print("No mosaic to finalize.")
            return None

        # Build metadata header
        if not self.wcs_metadata or not any(self.wcs_metadata.values()):
            print("WCS metadata not available; creating minimal header.")
            is_mono = (self.final_mosaic.ndim == 2 or
                    (self.final_mosaic.ndim == 3 and self.final_mosaic.shape[2] == 1))
            minimal_header = self.create_minimal_fits_header(self.final_mosaic, is_mono)
            meta = dict(minimal_header)
        else:
            meta = dict(self.wcs_metadata)

        # Add helpful tags
        is_mono = (self.final_mosaic.ndim == 2 or
                (self.final_mosaic.ndim == 3 and self.final_mosaic.shape[2] == 1))
        meta["step_name"] = "Mosaic Master"
        meta["is_mono"] = bool(is_mono)

        # Keep the array as-is (2D mono or 3-channel)
        return self.final_mosaic, meta

    def unstretch_image(self, image: np.ndarray) -> np.ndarray:
        """
        Best-effort inverse of the earlier median normalization.
        If mono, scale so median(image) -> original_median[0].
        If color, scale each channel so median(channel) -> original_median[c].
        Requires that self.stretch_original_medians / mins were populated per input.
        If stats are unavailable or inconsistent, returns image unchanged.
        """
        try:
            img = image.astype(np.float32, copy=True)

            orig_meds = getattr(self, "stretch_original_medians", None)
            orig_mins = getattr(self, "stretch_original_mins", None)
            if not orig_meds or not orig_mins:
                return img  # nothing to do

            # Use the FIRST frame’s recorded stats as the anchor for de-normalization.
            # (This is a heuristic; a true per-frame inverse is impossible after blending.)
            anchor_med = float(orig_meds[0])
            anchor_min = float(orig_mins[0])

            if anchor_med <= 0:
                return img

            if img.ndim == 2:
                cur_med = float(np.median(img))
                if cur_med > 0:
                    gain = anchor_med / cur_med
                    img = img * gain
                # keep dynamic range plausible
                img = np.clip(img, 0.0, 1.0)
                return img

            # color: per-channel median scaling (unlinked)
            for c in range(min(3, img.shape[2])):
                cur_med = float(np.median(img[..., c]))
                # choose a per-channel original median if available; fallback to anchor
                src_med = float(orig_meds[c]) if len(orig_meds) >= 3 else anchor_med
                if cur_med > 0 and src_med > 0:
                    gain = src_med / cur_med
                    img[..., c] *= gain

            img = np.clip(img, 0.0, 1.0)

            # If the mosaic was originally single-channel, keep it mono.
            if getattr(self, "was_single_channel", False) and img.ndim == 3:
                img = np.mean(img, axis=2).astype(np.float32)

            return img

        except Exception as e:
            print(f"[unstretch_image] fallback (no-op) due to: {e}")
            return image

    def _save_astap_exe_to_settings(self, path: str) -> None:
        path = os.path.normpath(os.path.expanduser(os.path.expandvars(path)))
        self.astap_exe = path
        self.settings.setValue("paths/astap", path)
        self.settings.sync()

    def _load_astrometry_api_key(self) -> str:
        # primary source: QSettings (matches SettingsDialog)
        key = self.settings.value("api/astrometry_key", "", type=str) or ""
        if key:
            return key
        # optional fallback to any app-level file helpers you already had
        try:
            k = load_api_key()
            return k or ""
        except Exception:
            return ""

    def _save_astrometry_api_key(self, key: str) -> None:
        key = (key or "").strip()
        self.settings.setValue("api/astrometry_key", key)
        self.settings.sync()
        # keep old helper in sync if you want
        try:
            save_api_key(key)
        except Exception:
            pass


    # ---------- Blind Solve via Astrometry.net ----------
    def perform_blind_solve(self, item):
        """
        Blind solve with Astrometry.net, construct WCS, keep SIP, and (optionally) update on-disk FITS.
        """
        while True:
            self.status_label.setText("Status: Logging in to Astrometry.net...")
            QApplication.processEvents()
            api_key = self._get_astrometry_api_key()
            if not api_key:
                api_key, ok = QInputDialog.getText(self, "Enter API Key", "Please enter your Astrometry.net API key:")
                if ok and api_key:
                    self.settings.setValue("api/astrometry_key", api_key)
                    self.settings.sync()
                else:
                    QMessageBox.warning(self, "API Key Required", "Blind solve canceled (no API key).")
                    return None

            session_key = self.login_to_astrometry(api_key)
            if session_key is None:
                if QMessageBox.question(self, "Login Failed",
                                        "Could not log in to Astrometry.net. Try again?",
                                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes:
                    continue
                else:
                    return None

            self.status_label.setText("Status: Uploading image to Astrometry.net...")
            QApplication.processEvents()

            # Determine the file extension of the original image
            ext = os.path.splitext(str(item.get("path","")))[1].lower()
            if ext not in ('.fits', '.fit'):
                tmp = tempfile.NamedTemporaryFile(suffix=".fit", delete=False)
                tmp.close()
                minimal_header = generate_minimal_fits_header(item["image"])
                save_image(
                    img_array=item["image"],
                    filename=tmp.name,
                    original_format="fit",
                    bit_depth="16-bit",
                    original_header=minimal_header,
                    is_mono=item.get("is_mono", False)
                )
                upload_path = tmp.name
            else:
                upload_path = item["path"]

            subid = self.upload_image_to_astrometry(upload_path, session_key)
            if not subid:
                if QMessageBox.question(self, "Upload Failed",
                                        "Image upload failed or no subid returned. Try again?",
                                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes:
                    continue
                else:
                    return None

            self.status_label.setText("Status: Waiting for job ID...")
            QApplication.processEvents()
            job_id = self.poll_submission_status(subid)
            if not job_id:
                if QMessageBox.question(self, "Blind Solve Failed",
                                        "Failed to retrieve job ID from Astrometry.net. Try again?",
                                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes:
                    continue
                else:
                    return None

            self.status_label.setText("Status: Retrieving calibration data...")
            QApplication.processEvents()
            calibration_data = self.poll_calibration_data(job_id)
            if not calibration_data:
                if QMessageBox.question(self, "Blind Solve Failed",
                                        "Calibration data did not arrive from Astrometry.net. Try again?",
                                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes:
                    continue
                else:
                    return None

            # Clean up tmp if we created one
            if ext not in ('.fits', '.fit', '.tif', '.tiff'):
                try:
                    os.remove(upload_path)
                except Exception:
                    pass

            break  # success

        # Build + normalize header, keep SIP, and build WCS with relax=True
        wcs_header = self.construct_wcs_header(calibration_data, item["image"].shape)
        wcs_header = self._normalize_wcs_header(wcs_header, item["image"].shape)
        item["wcs"] = self._build_wcs(wcs_header, item["image"].shape)

        dest = str(item.get("path", ""))
        if dest.lower().endswith((".fits", ".fit")) and not self._is_view_path(dest):
            self.update_fits_with_wcs(dest, calibration_data, wcs_header)
        else:
            print("Blind solve succeeded; not updating file on disk (view or non-FITS source).")

        return wcs_header


    def login_to_astrometry(self, api_key):
        url = ASTROMETRY_API_URL + "login"
        data = {'request-json': json.dumps({"apikey": api_key})}
        response = robust_api_request("POST", url, data=data, prompt_on_failure=True)
        if response and response.get("status") == "success":
            return response["session"]
        print("Login failed after multiple attempts.")
        QMessageBox.critical(self, "Login Failed", "Could not log in to Astrometry.net. Check your API key or internet connection.")
        return None

    def upload_image_to_astrometry(self, image_path, session_key):
        url = ASTROMETRY_API_URL + "upload"
        with open(image_path, 'rb') as f:
            files = {'file': f}
            data = {
                'request-json': json.dumps({
                    "publicly_visible": "y",
                    "allow_modifications": "d",
                    "session": session_key,
                    "allow_commercial_use": "d"
                })
            }
            response = robust_api_request("POST", url, data=data, files=files)
        if response and response.get("status") == "success":
            return response["subid"]
        QMessageBox.critical(self, "Upload Failed", "Image upload failed after multiple attempts.")
        return None

    def poll_submission_status(self, subid):
        url = ASTROMETRY_API_URL + f"submissions/{subid}"
        for attempt in range(90):  # up to ~15 minutes
            response = robust_api_request("GET", url)
            if response:
                jobs = response.get("jobs", [])
                if jobs and jobs[0] is not None:
                    return jobs[0]
            print(f"Polling attempt {attempt+1}: Job ID not ready yet.")
            time.sleep(10)
        QMessageBox.critical(self, "Blind Solve Failed", "Failed to retrieve job ID from Astrometry.net after multiple attempts.")
        return None

    def poll_calibration_data(self, job_id):
        url = ASTROMETRY_API_URL + f"jobs/{job_id}/calibration/"
        for attempt in range(90):
            response = robust_api_request("GET", url)
            if response and 'ra' in response and 'dec' in response:
                print("Calibration data retrieved:", response)
                return response
            print(f"Calibration data not available yet (attempt {attempt+1})")
            time.sleep(10)
        QMessageBox.critical(self, "Blind Solve Failed", "Calibration data did not complete in the expected timeframe.")
        return None

    # (truncated in thought, but assuming I write the rest or next part)

    def construct_wcs_header(self, calibration_data, image_shape):
        h = Header()
        h['CTYPE1'] = 'RA---TAN'
        h['CTYPE2'] = 'DEC--TAN'
        h['CRPIX1'] = image_shape[1] / 2
        h['CRPIX2'] = image_shape[0] / 2
        h['CRVAL1'] = calibration_data['ra']
        h['CRVAL2'] = calibration_data['dec']
        scale = calibration_data['pixscale'] / 3600.0  # degrees/pixel
        orientation = math.radians(calibration_data['orientation'])
        h['CD1_1'] = -scale * np.cos(orientation)
        h['CD1_2'] = scale * np.sin(orientation)
        h['CD2_1'] = -scale * np.sin(orientation)
        h['CD2_2'] = -scale * np.cos(orientation)
        h['RADECSYS'] = 'ICRS'
        h['WCSAXES'] = 2
        print("Generated WCS header from calibration data.")
        return h

    def update_fits_with_wcs(self, filepath, calibration_data, wcs_header):
        if not filepath.lower().endswith(('.fits','.fit')):
            print("Not a FITS, skipping WCS update.")
            return
        try:
            with fits.open(filepath, mode='update') as hdul:
                hdr = hdul[0].header
                if 'NAXIS3' in hdr:
                    del hdr['NAXIS3']
                hdr['NAXIS'] = 2
                hdr['CTYPE1'] = 'RA---TAN'
                hdr['CTYPE2'] = 'DEC--TAN'
                hdr['CRVAL1'] = calibration_data['ra']
                hdr['CRVAL2'] = calibration_data['dec']
                # Determine H and W based on the data's dimensionality.
                if hdul[0].data.ndim == 3:
                    # Assume data are stored as (channels, height, width)
                    _, H, W = hdul[0].data.shape
                else:
                    H, W = hdul[0].data.shape[:2]
                hdr['CRPIX1'] = W/2.0
                hdr['CRPIX2'] = H/2.0
                scale = calibration_data['pixscale']/3600.0
                orientation = math.radians(calibration_data.get('orientation', 0.0))
                hdr['CD1_1'] = -scale * np.cos(orientation)
                hdr['CD1_2'] = scale * np.sin(orientation)
                hdr['CD2_1'] = -scale * np.sin(orientation)
                hdr['CD2_2'] = -scale * np.cos(orientation)
                hdr['WCSAXES'] = 2
                hdr['RADECSYS'] = 'ICRS'
                hdul.flush()
                print("WCS updated in FITS.")
            # Re-open to verify changes:
            with fits.open(filepath) as hdul_verify:
                print("Updated header keys:", hdul_verify[0].header.keys())
        except Exception as e:
            print(f"Error updating FITS with WCS: {e}")

    # ---------- Blind Solve via ASTAP ----------
    def attempt_astap_solve(self, item):
        """
        Attempt to plate-solve the image using ASTAP.
        Returns a solved header (as a dict) on success or None on failure.
        """
        # 1) Normalize the image (using your stretch_image).
        normalized_image = self.stretch_image(item["image"])
        
        # 2) Save normalized image to a temporary FITS file for ASTAP.
        try:
            tmp_path = self.save_temp_fits_image(normalized_image, item["path"])
        except Exception as e:
            print("Failed to save temporary FITS file:", e)
            return None

        # 3) Run ASTAP on the temporary file.
        process = QProcess(self)
        args = ["-f", tmp_path, "-r", "179", "-fov", "0", "-z", "0", "-wcs", "-sip"]
        print("Running ASTAP with arguments:", args)
        process.start(self.astap_exe, args)
        if not process.waitForStarted(5000):
            print("Failed to start ASTAP process:", process.errorString())
            os.remove(tmp_path)
            return None
        if not process.waitForFinished(300000):  # wait up to 5 minutes
            print("ASTAP process timed out.")
            os.remove(tmp_path)
            return None

        exit_code = process.exitCode()
        stdout = process.readAllStandardOutput().data().decode()
        stderr = process.readAllStandardError().data().decode()
        print("ASTAP exit code:", exit_code)
        print("ASTAP STDOUT:\n", stdout)
        print("ASTAP STDERR:\n", stderr)

        if exit_code != 0:
            try:
                os.remove(tmp_path)
            except Exception as e:
                print("Error removing temporary file:", e)
            return None

        # 4) Retrieve updated header from the temporary file.
        try:
            with fits.open(tmp_path, memmap=False) as hdul:
                solved_header = dict(hdul[0].header)
            # Remove some extraneous keywords
            solved_header.pop("COMMENT", None)
            solved_header.pop("HISTORY", None)
        except Exception as e:
            print("Error reading solved header:", e)
            os.remove(tmp_path)
            return None

        # 5) Merge .wcs file (if ASTAP wrote one) into the solved_header.
        wcs_path = os.path.splitext(tmp_path)[0] + ".wcs"
        if os.path.exists(wcs_path):
            try:
                with open(wcs_path, "r") as f:
                    content = f.read()
                pattern = r"(\w+)\s*=\s*('?[^/']*'?)[\s/]"
                for match in re.finditer(pattern, content):
                    key = match.group(1).strip().upper()
                    val = match.group(2).strip()
                    if val.startswith("'") and val.endswith("'"):
                        val = val[1:-1].strip()
                    solved_header[key] = val
            except Exception as e:
                print("Error reading .wcs file:", e)
            finally:
                try:
                    os.remove(wcs_path)
                except Exception as e:
                    print("Error removing .wcs file:", e)

        # Remove the END keyword if present
        solved_header.pop("END", None)
        # Remove any unneeded keywords
        for keyword in ["RANGE_LOW", "RANGE_HIGH", "HISTORY"]:
            solved_header.pop(keyword, None)

        # --- Ensure required WCS keys are present ---
        if "CTYPE1" not in solved_header or not solved_header["CTYPE1"].strip():
            solved_header["CTYPE1"] = "RA---TAN"
        if "CTYPE2" not in solved_header or not solved_header["CTYPE2"].strip():
            solved_header["CTYPE2"] = "DEC--TAN"
        if "RADECSYS" not in solved_header:
            solved_header["RADECSYS"] = "ICRS"
        if "WCSAXES" not in solved_header:
            solved_header["WCSAXES"] = 2

        # Convert known WCS keys to float or int
        expected_float_keys = {"CRPIX1", "CRPIX2", "CRVAL1", "CRVAL2", "CDELT1", "CDELT2", "CD1_1", "CD1_2", "CD2_1", "CD2_2"}
        expected_int_keys = {"NAXIS", "WCSAXES"}

        for key in expected_float_keys:
            if key in solved_header:
                try:
                    solved_header[key] = float(solved_header[key])
                except ValueError:
                    print(f"Warning: Could not convert {key}='{solved_header[key]}' to float.")

        for key in expected_int_keys:
            if key in solved_header:
                try:
                    solved_header[key] = int(float(solved_header[key]))
                except ValueError:
                    print(f"Warning: Could not convert {key}='{solved_header[key]}' to int.")

        dest = str(item.get("path", ""))

        ok_to_overwrite = (
            bool(dest)
            and not self._is_view_path(dest)
            and dest.lower().endswith((".fits", ".fit"))
        )

        if ok_to_overwrite:
            try:
                save_image(
                    img_array=normalized_image,
                    filename=dest,
                    original_format="fit",
                    bit_depth="32",
                    original_header=solved_header,
                    is_mono=False
                )
                print(f"✅ Updated FITS header with full WCS solution for {dest}.")
            except Exception as e:
                print("Error saving updated FITS file using save_image():", e)
                # don't re-raise; we can still continue in-memory
        else:
            print("Skipping on-disk write (source is a view or not a FITS path).")

        # cleanup temp files as you already do...
        return solved_header


    def save_temp_fits_image(self, normalized_image, image_path: str):
        """
        Save the normalized_image as a FITS file to a temporary file.
        
        If the original image is FITS, this method retrieves the stored metadata
        from the ImageManager and passes it directly to save_image().
        If not, it generates a minimal header.
        
        Returns the path to the temporary FITS file.
        """
        # Always save as FITS.
        selected_format = "fits"
        bit_depth = "32-bit floating point"
        is_mono = (normalized_image.ndim == 2 or 
                   (normalized_image.ndim == 3 and normalized_image.shape[2] == 1))
        
        # If the original image is FITS, try to get its stored metadata.
        original_header = None
        is_fits_path = isinstance(image_path, str) and image_path.lower().endswith((".fits", ".fit"))
        if is_fits_path and not self._is_view_path(image_path):
            if self.parent() and hasattr(self.parent(), "image_manager"):
                # Use the metadata from the current slot.
                _, meta = self.parent().image_manager.get_current_image_and_metadata()
                # Assume that meta already contains a proper 'original_header'
                # (or the entire meta is the header).
                original_header = meta.get("original_header", None)
            # If nothing is stored, fall back to creating a minimal header.
            if original_header is None:
                print("No stored FITS header found; creating a minimal header.")
                original_header = self.create_minimal_fits_header(normalized_image, is_mono)
        else:
            # For non-FITS images, generate a minimal header.
            original_header = self.create_minimal_fits_header(normalized_image, is_mono)
        
        # Create a temporary filename.
        tmp_file = tempfile.NamedTemporaryFile(suffix=".fits", delete=False)
        tmp_path = tmp_file.name
        tmp_file.close()
        
        try:
            # Call your global save_image() exactly as in AstroEditingSuite.
            save_image(
                img_array=normalized_image,
                filename=tmp_path,
                original_format=selected_format,
                bit_depth=bit_depth,
                original_header=original_header,
                is_mono=is_mono
                # (image_meta and file_meta can be omitted if not needed)
            )
            print(f"Temporary normalized FITS saved to: {tmp_path}")
        except Exception as e:
            print("Error saving temporary FITS file using save_image():", e)
            raise e
        return tmp_path

    def create_minimal_fits_header(self, img_array, is_mono=False):
        """
        Creates a minimal FITS header when the original header is missing.
        """

        header = Header()
        header['SIMPLE'] = (True, 'Standard FITS file')
        header['BITPIX'] = -32  # 32-bit floating-point data
        header['NAXIS'] = 2 if is_mono else 3
        header['NAXIS1'] = img_array.shape[2] if img_array.ndim == 3 and not is_mono else img_array.shape[1]  # Image width
        header['NAXIS2'] = img_array.shape[1] if img_array.ndim == 3 and not is_mono else img_array.shape[0]  # Image height
        if not is_mono:
            header['NAXIS3'] = img_array.shape[0] if img_array.ndim == 3 else 1  # Number of color channels
        header['BZERO'] = 0.0  # No offset
        header['BSCALE'] = 1.0  # No scaling


        return header

    def stretch_image(self, image):
        """
        Perform an unlinked linear stretch on the image.
        Each channel is stretched independently by subtracting its own minimum,
        recording its own median, and applying the stretch formula.
        Returns the stretched image in [0,1].
        """
        was_single_channel = False  # Flag to check if image was single-channel

        # If the image is 2D or has one channel, convert to 3-channel
        if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
            was_single_channel = True
            image = np.stack([image] * 3, axis=-1)

        image = image.astype(np.float32).copy()
        stretched_image = image.copy()
        self.stretch_original_mins = []
        self.stretch_original_medians = []
        target_median = 0.02

        for c in range(3):
            channel_min = np.min(stretched_image[..., c])
            self.stretch_original_mins.append(channel_min)
            stretched_image[..., c] -= channel_min
            channel_median = np.median(stretched_image[..., c])
            self.stretch_original_medians.append(channel_median)
            if channel_median != 0:
                numerator = (channel_median - 1) * target_median * stretched_image[..., c]
                denominator = (
                    channel_median * (target_median + stretched_image[..., c] - 1)
                    - target_median * stretched_image[..., c]
                )
                denominator = np.where(denominator == 0, 1e-6, denominator)
                stretched_image[..., c] = numerator / denominator
            else:
                print(f"Channel {c} - Median is zero. Skipping stretch.")

        stretched_image = np.clip(stretched_image, 0.0, 1.0)
        self.was_single_channel = was_single_channel
        return stretched_image

