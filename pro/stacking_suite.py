from __future__ import annotations
import os, glob, shutil, tempfile, datetime as _dt
from time import perf_counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import re
import math            # used in compute_safe_chunk
import psutil          # used in bytes_available / compute_safe_chunk
from typing import List 
from PyQt6.QtCore import Qt, QTimer, QSettings, pyqtSignal, QObject, pyqtSlot, QThread, QEvent, QPoint, QSize, QEventLoop
from PyQt6.QtGui import QIcon, QImage, QPixmap, QAction, QIntValidator, QDoubleValidator, QFontMetrics, QTextCursor
from PyQt6.QtWidgets import (QDialog, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QLineEdit, QTreeWidget, QHeaderView, QTreeWidgetItem, QProgressBar, QProgressDialog,
                             QFormLayout, QDialogButtonBox, QToolBar, QToolButton, QFileDialog, QTabWidget, QAbstractItemView, QSpinBox, QDoubleSpinBox,
                             QSizePolicy, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QApplication, QScrollArea, QTextEdit, QMenu, QPlainTextEdit,
                             QMessageBox, QSlider, QCheckBox, QInputDialog, QComboBox)
from datetime import datetime, timezone
import pyqtgraph as pg
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from PIL import Image                 # used by _get_image_size (and elsewhere)
import tifffile as tiff               # _get_image_size -> tiff.imread(...)
import cv2                            # _get_image_size -> cv2.imread(...)

import rawpy

import exifread


import sep


# your helpers/utilities
from imageops.stretch import stretch_mono_image, stretch_color_image
from legacy.numba_utils import *   # adjust names if different
from legacy.image_manager import load_image, save_image, get_valid_header
from pro.star_alignment import StarRegistrationWorker, StarRegistrationThread, IDENTITY_2x3


def load_fits_tile(filepath, y_start, y_end, x_start, x_end):
    """
    Loads a sub-region from a FITS file, detecting which axes are spatial vs. color.
    
    * If the data is 2D, it might be (height, width) or (width, height).
    * If the data is 3D, it might be:
        - (height, width, 3)
        - (3, height, width)
        - (width, height, 3)
        - (3, width, height)
      We only slice the two spatial dimensions; the color axis remains intact.
    
    The returned tile will always have the shape:
      - (tile_height, tile_width) for mono
      - (tile_height, tile_width, 3) for color
    (though the color dimension may still be first if it was first in the file).
    It's up to the caller to reorder if needed.
    """
    with fits.open(filepath, memmap=False) as hdul:
        data = hdul[0].data
        if data is None:
            return None

        # Save the original data type for normalization later.
        orig_dtype = data.dtype

        shape = data.shape
        ndim = data.ndim

        if ndim == 2:
            # Data is 2D; shape could be (height, width) or (width, height)
            dim0, dim1 = shape
            if (y_end <= dim0) and (x_end <= dim1):
                tile_data = data[y_start:y_end, x_start:x_end]
            else:
                tile_data = data[x_start:x_end, y_start:y_end]
        elif ndim == 3:
            # Data is 3D; could be (height, width, 3) or (3, height, width), etc.
            dim0, dim1, dim2 = shape

            def do_slice_spatial(data3d, spat0, spat1, color_axis):
                slicer = [slice(None)] * 3
                slicer[spat0] = slice(y_start, y_end)
                slicer[spat1] = slice(x_start, x_end)
                tile = data3d[tuple(slicer)]
                return tile

            # Identify the color axis (assumed to have size 3)
            color_axis = None
            spat_axes = []
            for idx, d in enumerate((dim0, dim1, dim2)):
                if d == 3:
                    color_axis = idx
                else:
                    spat_axes.append(idx)

            if color_axis is None:
                # No axis with size 3; assume the image is mono and use the first two dims.
                tile_data = data[y_start:y_end, x_start:x_end]
            else:
                # Ensure we have two spatial axes.
                if len(spat_axes) != 2:
                    spat_axes = [0, 1]
                spat0, spat1 = spat_axes
                d0 = shape[spat0]
                d1 = shape[spat1]
                if (y_end <= d0) and (x_end <= d1):
                    tile_data = do_slice_spatial(data, spat0, spat1, color_axis)
                else:
                    tile_data = do_slice_spatial(data, spat1, spat0, color_axis)
        else:
            return None

        # Normalize based on the original data type.
        if orig_dtype == np.uint8:
            tile_data = tile_data.astype(np.float32) / 255.0
        elif orig_dtype == np.uint16:
            tile_data = tile_data.astype(np.float32) / 65535.0
        elif orig_dtype == np.uint32:
            # 32-bit data: convert to float32 but leave values as is.
            tile_data = tile_data.astype(np.float32)
        elif orig_dtype == np.float32:
            # Already 32-bit float; assume it's in the desired range.
            tile_data = tile_data
        else:
            tile_data = tile_data.astype(np.float32)

    return tile_data



# --------------------------------------------------
# Stacking Suite
# --------------------------------------------------
class BatchSettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Set Type, Exposure, and Filter for All Files")

        layout = QVBoxLayout(self)

        # 1) IMAGETYP Combo
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Image Type (IMAGETYP):"))
        self.type_combo = QComboBox()
        self.type_combo.addItems(["LIGHT", "DARK", "FLAT", "BIAS", "UNKNOWN"])
        type_layout.addWidget(self.type_combo)
        layout.addLayout(type_layout)

        # 2) Exposure Time
        exp_layout = QHBoxLayout()
        exp_layout.addWidget(QLabel("Exposure Time (seconds):"))
        self.exptime_edit = QLineEdit()
        self.exptime_edit.setText("Unknown")  # default
        exp_layout.addWidget(self.exptime_edit)
        layout.addLayout(exp_layout)

        # 3) Filter
        filt_layout = QHBoxLayout()
        filt_layout.addWidget(QLabel("Filter:"))
        self.filter_edit = QLineEdit()
        self.filter_edit.setText("None")  # default
        filt_layout.addWidget(self.filter_edit)
        layout.addLayout(filt_layout)

        # Buttons
        btn_layout = QHBoxLayout()
        ok_btn = QPushButton("OK")
        cancel_btn = QPushButton("Cancel")
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

        ok_btn.clicked.connect(self.accept)
        cancel_btn.clicked.connect(self.reject)

        # Final layout
        self.setLayout(layout)

    def get_values(self):
        """
        Returns (imagetyp, exptime_str, filter_str)
        after the dialog is accepted.
        """
        return (
            self.type_combo.currentText(),
            self.exptime_edit.text(),
            self.filter_edit.text()
        )

class ReferenceFrameReviewDialog(QDialog):
    def __init__(self, ref_frame_path, stats, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Reference Frame Review")
        self.ref_frame_path = ref_frame_path
        self.stats = stats  # e.g., {"star_count": 250, "eccentricity": 0.12, "mean": 0.45}
        self.autostretch_enabled = False
        self.original_image = None  # Will store the loaded image array
        self.target_median = self.stats.get("mean", 0.25)
        self.user_choice = None  # Will be set to 'use' or 'select_other'
        self.zoom_factor = 1.0
        self.current_preview_image = None  # Store the image array currently shown in preview

        # For panning functionality
        self._panning = False
        self._last_mouse_pos = QPoint()

        self.initUI()
        self.loadImageArray()  # Load the image into self.original_image
        if self.original_image is not None:
            self.updatePreview(self.original_image)  # Ensure the first image is shown
        if self.original_image is not None:
            QTimer.singleShot(0, self.zoomIn)            


    def initUI(self):
        main_layout = QVBoxLayout(self)
        
        # Create a scroll area for the preview image
        self.scrollArea = QScrollArea(self)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setMinimumSize(QSize(600, 400))
        self.previewLabel = QLabel("Reference Preview", self)
        self.previewLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scrollArea.setWidget(self.previewLabel)
        main_layout.addWidget(self.scrollArea)
        self.scrollArea.viewport().installEventFilter(self)
        
        # Zoom control buttons
        zoom_layout = QHBoxLayout()
        self.zoomInButton = QPushButton("Zoom In", self)
        self.zoomInButton.clicked.connect(self.zoomIn)
        zoom_layout.addWidget(self.zoomInButton)
        self.zoomOutButton = QPushButton("Zoom Out", self)
        self.zoomOutButton.clicked.connect(self.zoomOut)
        zoom_layout.addWidget(self.zoomOutButton)
        main_layout.addLayout(zoom_layout)
        
        # Stats display
        stats_text = (
            f"Star Count: {self.stats.get('star_count', 'N/A')}\n"
            f"Eccentricity: {self.stats.get('eccentricity', 'N/A'):.4f}\n"
            f"Mean: {self.stats.get('mean', 'N/A'):.4f}"
        )
        self.statsLabel = QLabel(stats_text, self)
        main_layout.addWidget(self.statsLabel)
        
        # Buttons layout for reference selection and autostretch toggle
        button_layout = QHBoxLayout()
        self.toggleAutoStretchButton = QPushButton("Enable Autostretch", self)
        self.toggleAutoStretchButton.clicked.connect(self.toggleAutostretch)
        button_layout.addWidget(self.toggleAutoStretchButton)
        
        # New button to let the user select a new reference frame file
        self.selectNewRefButton = QPushButton("Select New Reference Frame", self)
        self.selectNewRefButton.clicked.connect(self.selectNewReferenceFrame)
        button_layout.addWidget(self.selectNewRefButton)
        
        self.useRefButton = QPushButton("Use This Reference Frame", self)
        self.useRefButton.clicked.connect(self.useReference)
        button_layout.addWidget(self.useRefButton)
        
        self.selectOtherButton = QPushButton("Cancel", self)
        self.selectOtherButton.clicked.connect(self.reject)
        button_layout.addWidget(self.selectOtherButton)
        
        main_layout.addLayout(button_layout)
        self.setLayout(main_layout)
        self.zoomIn()
    
    def fitToPreview(self):
        """Calculate and set the zoom factor so that the image fills the preview area."""
        if self.original_image is None:
            return
        # Get the available size from the scroll area's viewport.
        available_size = self.scrollArea.viewport().size()
        # Determine the original image dimensions.
        if self.original_image.ndim == 2:
            orig_height, orig_width = self.original_image.shape
        elif self.original_image.ndim == 3:
            orig_height, orig_width = self.original_image.shape[:2]
        else:
            return
        # Calculate the zoom factor that will allow the image to fit.
        factor = min(available_size.width() / orig_width,
                    available_size.height() / orig_height)
        self.zoom_factor = factor
        # Choose the current preview image if available, otherwise use the original image.
        if self.current_preview_image is not None:
            image = self.current_preview_image
        else:
            image = self.original_image
        self.updatePreview(image)



    def loadImageArray(self):
        """
        Load the image from the reference frame file using the global load_image function.
        """
        image_data, header, _, _ = load_image(self.ref_frame_path)
        if image_data is not None:
            if image_data.ndim == 3 and image_data.shape[-1] == 1:
                image_data = np.squeeze(image_data, axis=-1)
            self.original_image = image_data
        else:
            QMessageBox.critical(self, "Error", "Failed to load the reference image.")
    
    def updatePreview(self, image):
        """
        Convert a given image array to a QPixmap and update the preview label.
        """
        self.current_preview_image = image
        pixmap = self.convertArrayToPixmap(image)
        if pixmap is None or pixmap.isNull():
            self.previewLabel.setText("Unable to load preview.")
        else:
            available_size = self.scrollArea.viewport().size()
            new_size = QSize(int(available_size.width() * self.zoom_factor),
                             int(available_size.height() * self.zoom_factor))
            scaled_pixmap = pixmap.scaled(new_size, Qt.AspectRatioMode.KeepAspectRatio,
                                          Qt.TransformationMode.SmoothTransformation)
            self.previewLabel.setPixmap(scaled_pixmap)
    
    def convertArrayToPixmap(self, image):
        if image is None:
            return None
        display_image = (image * 255).clip(0, 255).astype(np.uint8)
        if display_image.ndim == 2:
            h, w = display_image.shape
            bytes_per_line = w
            q_image = QImage(display_image.tobytes(), w, h, bytes_per_line, QImage.Format.Format_Grayscale8)
        elif display_image.ndim == 3 and display_image.shape[2] == 3:
            h, w, _ = display_image.shape
            bytes_per_line = 3 * w
            q_image = QImage(display_image.tobytes(), w, h, bytes_per_line, QImage.Format.Format_RGB888)
        else:
            return None
        return QPixmap.fromImage(q_image)
    
    def toggleAutostretch(self):
        if self.original_image is None:
            QMessageBox.warning(self, "Error", "Reference image not loaded.")
            return
        self.autostretch_enabled = not self.autostretch_enabled
        if self.autostretch_enabled:
            if self.original_image.ndim == 2:
                new_image = stretch_mono_image(self.original_image, target_median=0.3,
                                               normalize=True, apply_curves=False)
            elif self.original_image.ndim == 3 and self.original_image.shape[2] == 3:
                new_image = stretch_color_image(self.original_image, target_median=0.3,
                                                linked=False, normalize=True, apply_curves=False)
            else:
                new_image = self.original_image
            self.toggleAutoStretchButton.setText("Disable Autostretch")
        else:
            new_image = self.original_image
            self.toggleAutoStretchButton.setText("Enable Autostretch")
        self.updatePreview(new_image)
    
    def zoomIn(self):
        self.zoom_factor *= 1.2
        if self.current_preview_image is not None:
            self.updatePreview(self.current_preview_image)
    
    def zoomOut(self):
        self.zoom_factor /= 1.2
        if self.current_preview_image is not None:
            self.updatePreview(self.current_preview_image)
    
    def eventFilter(self, source, event):
        if source is self.scrollArea.viewport():
            if event.type() == QEvent.Type.Wheel:
                if event.angleDelta().y() > 0:
                    self.zoomIn()
                else:
                    self.zoomOut()
                return True
            if event.type() == QEvent.Type.MouseButtonPress:
                if event.button() == Qt.MouseButton.LeftButton:
                    self._panning = True
                    self._last_mouse_pos = event.pos()
                    self.scrollArea.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
                    return True
            if event.type() == QEvent.Type.MouseMove:
                if self._panning:
                    delta = event.pos() - self._last_mouse_pos
                    self._last_mouse_pos = event.pos()
                    h_bar = self.scrollArea.horizontalScrollBar()
                    v_bar = self.scrollArea.verticalScrollBar()
                    h_bar.setValue(h_bar.value() - delta.x())
                    v_bar.setValue(v_bar.value() - delta.y())
                    return True
            if event.type() == QEvent.Type.MouseButtonRelease:
                if event.button() == Qt.MouseButton.LeftButton:
                    self._panning = False
                    self.scrollArea.viewport().setCursor(Qt.CursorShape.ArrowCursor)
                    return True
        return super().eventFilter(source, event)
    
    def resizeEvent(self, event):
        if self.current_preview_image is not None:
            self.updatePreview(self.current_preview_image)
        super().resizeEvent(event)
    
    def selectNewReferenceFrame(self):
        """Open a file dialog to select a new reference frame, update preview accordingly."""
        new_file, _ = QFileDialog.getOpenFileName(
            self,
            "Select New Reference Frame",
            "",
            "FITS Files (*.fits *.fit);;All Files (*)"
        )
        if new_file:
            self.ref_frame_path = new_file
            self.loadImageArray()          # Reload the new image
            self.updatePreview(self.original_image)  # Update the preview
            # Optionally, you could also update stats if needed.
    
    def useReference(self):
        self.user_choice = "use"
        self.accept()
    
    def selectOtherReference(self):
        self.user_choice = "select_other"
        self.reject()
    
    def getUserChoice(self):
        return self.user_choice

def bytes_available():
    vm = psutil.virtual_memory()
    # Keep a safety margin (e.g. leave 10% free)
    return int(vm.available * 0.9)


def compute_safe_chunk(height, width, N, channels, dtype, pref_h, pref_w):
    vm    = psutil.virtual_memory()
    avail = vm.free * 0.9
    bpe64 = np.dtype(dtype).itemsize      # 8 bytes
    workers = os.cpu_count() or 1

    # budget *all* float64 copies (master + per-thread)
    bytes_per_pixel = (N + workers) * channels * bpe64 / 2
    max_pixels      = int(avail // bytes_per_pixel)
    if max_pixels < 1:
        raise MemoryError("Not enough RAM for even a 1×1 tile")

    raw_side = int(math.sqrt(max_pixels))
    # **shrink by √workers to be super-safe**
    fudge    = int(math.sqrt(workers)) or 1
    safe_side = max(1, raw_side // fudge)

    # clamp to user prefs and image dims
    ch = min(pref_h, height, safe_side)
    cw = min(pref_w, width,  safe_side)

    # final area clamp
    if ch * cw > max_pixels // fudge**2:
        # extra safety: adjust cw so area ≤ max_pixels/fudge²
        cw = max(1, (max_pixels // (fudge**2)) // ch)

    if ch < 1 or cw < 1:
        raise MemoryError(f"Chunk too small after fudge: {ch}×{cw}")

    print(f"[DEBUG] raw_side={raw_side}, workers={workers} ⇒ safe_side={safe_side}")
    print(f"[DEBUG] final chunk: {ch}×{cw}")
    return ch, cw

_DIM_RE = re.compile(r"\s*\(\d+\s*x\s*\d+\)\s*")


class AfterAlignWorker(QObject):
    progress = pyqtSignal(str)                 # emits status lines
    finished = pyqtSignal(bool, str)           # (success, message)

    def __init__(self, dialog, *,
                 light_files,
                 frame_weights,
                 transforms_dict,
                 drizzle_dict,
                 autocrop_enabled,
                 autocrop_pct):
        super().__init__()
        self.dialog = dialog                    # we will call pure methods on it
        self.light_files = light_files
        self.frame_weights = frame_weights
        self.transforms_dict = transforms_dict
        self.drizzle_dict = drizzle_dict
        self.autocrop_enabled = autocrop_enabled
        self.autocrop_pct = autocrop_pct

    @pyqtSlot()
    def run(self):
        dlg = self.dialog  # the StackingSuiteDialog you passed in

        try:
            result = dlg.stack_images_mixed_drizzle(
                grouped_files=self.light_files,
                frame_weights=self.frame_weights,
                transforms_dict=self.transforms_dict,
                drizzle_dict=self.drizzle_dict,
                autocrop_enabled=self.autocrop_enabled,
                autocrop_pct=self.autocrop_pct,
                status_cb=self.progress.emit,   # stream status back to UI
            )
            summary = "\n".join(result["summary_lines"])
            self.finished.emit(True, f"Post-alignment complete.\n\n{summary}")
        except Exception as e:
            self.finished.emit(False, f"Post-alignment failed: {e}")

class StatusLogWindow(QDialog):
    MAX_BLOCKS = 2000

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Stacking Suite Log")

        # ── key flags ─────────────────────────────────────────────
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)   # hide, don't delete
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setWindowFlag(Qt.WindowType.Tool, True)                    # tool-style (no taskbar)
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)    # stay on top of app windows
        self.setWindowModality(Qt.WindowModality.NonModal)              # never block UI
        # ─────────────────────────────────────────────────────────

        self.resize(800, 250)

        lay = QVBoxLayout(self)
        self.view = QPlainTextEdit(self)
        self.view.setReadOnly(True)
        self.view.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.view.setStyleSheet(
            "background-color: black; color: white; font-family: Monospace; padding: 6px;"
        )
        lay.addWidget(self.view)

        row = QHBoxLayout()
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.view.clear)
        row.addWidget(self.clear_btn)
        row.addStretch(1)
        lay.addLayout(row)

    def show_raise(self):
        # bring forward without stealing focus
        self.show()
        self.raise_()

    @pyqtSlot(str)
    def append_line(self, message: str):
        doc = self.view.document()

        if message.startswith("🔄 Normalizing") and doc.blockCount() > 0:
            last = doc.findBlockByNumber(doc.blockCount() - 1)
            if last.isValid() and last.text().startswith("🔄 Normalizing"):
                cur = self.view.textCursor()
                cur.movePosition(QTextCursor.MoveOperation.End)
                cur.movePosition(QTextCursor.MoveOperation.StartOfBlock,
                                 QTextCursor.MoveMode.KeepAnchor)
                cur.removeSelectedText()
                cur.insertText(message)
                self.view.setTextCursor(cur)
            else:
                self.view.appendPlainText(message)
        else:
            self.view.appendPlainText(message)

        # trim
        if doc.blockCount() > self.MAX_BLOCKS:
            extra = doc.blockCount() - self.MAX_BLOCKS
            cur = self.view.textCursor()
            cur.movePosition(QTextCursor.MoveOperation.Start)
            cur.movePosition(QTextCursor.MoveOperation.Down,
                             QTextCursor.MoveMode.KeepAnchor, extra)
            cur.removeSelectedText()
            self.view.setTextCursor(self.view.textCursor())

        # autoscroll
        sb = self.view.verticalScrollBar()
        sb.setValue(sb.maximum())

    @pyqtSlot(str)
    def _on_post_status(self, msg: str):
        # update small “last status” indicator in the dialog (GUI thread slot)
        self._update_status_gui(msg)
        # append to the shared log window
        self.status_signal.emit(msg)
        # reflect in progress dialog label if it exists
        try:
            if getattr(self, "post_progress", None):
                self.post_progress.setLabelText(msg)
                QApplication.processEvents()
        except Exception:
            pass



class StackingSuiteDialog(QDialog):
    requestRelaunch = pyqtSignal(str, str)  # old_dir, new_dir
    status_signal = pyqtSignal(str)

    def __init__(self, parent=None, wrench_path=None, spinner_path=None, **_ignored):
        super().__init__(parent)
        self.settings = QSettings()         
        self._wrench_path = wrench_path
        self._spinner_path = spinner_path
        self._post_progress_label = None
        # ...
        self.wrench_button = QPushButton()
        self.wrench_button.setIcon(QIcon(self._wrench_path))
        self.setWindowTitle("Stacking Suite")
        self.setGeometry(300, 200, 800, 600)
        self.per_group_drizzle = {}
        self.manual_dark_overrides = {}  
        self.manual_flat_overrides = {}
        self.conversion_output_directory = None
        self.reg_files = {}
        self.session_tags = {}  # 🔑 file_path => session_tag (e.g., "Session1", "Blue Flats", etc.)
        self.deleted_calibrated_files = []
        self._norm_map = {}
        self._gui_thread = QThread.currentThread()        # remember GUI thread
        self.status_signal.connect(self._update_status_gui)  # queued to GUI
 
        self.auto_rot180 = self.settings.value("stacking/auto_rot180", True, type=bool)
        self.auto_rot180_tol_deg = self.settings.value("stacking/auto_rot180_tol_deg", 89.0, type=float)             

        # QSettings for your app

        dtype_str = self.settings.value("stacking/internal_dtype", "float64", type=str)
        self.internal_dtype = np.float64 if dtype_str == "float64" else np.float32  
        self.star_trail_mode = self.settings.value("stacking/star_trail_mode", False, type=bool)        

        self.align_refinement_passes = self.settings.value("stacking/refinement_passes", 3, type=int)
        self.align_shift_tolerance = self.settings.value("stacking/shift_tolerance_px", 0.2, type=float)

        # Load or default these
        self.stacking_directory = self.settings.value("stacking/dir", "", type=str)
        self.sigma_high = self.settings.value("stacking/sigma_high", 3.0, type=float)
        self.sigma_low = self.settings.value("stacking/sigma_low", 3.0, type=float)
        self.rejection_algorithm = self.settings.value(
            "stacking/rejection_algorithm",
            "Weighted Windsorized Sigma Clipping",
            type=str
        )
        self.kappa = self.settings.value("stacking/kappa", 2.5, type=float)
        self.iterations = self.settings.value("stacking/iterations", 3, type=int)
        self.esd_threshold = self.settings.value("stacking/esd_threshold", 3.0, type=float)
        self.biweight_constant = self.settings.value("stacking/biweight_constant", 6.0, type=float)
        self.trim_fraction = self.settings.value("stacking/trim_fraction", 0.1, type=float)
        self.modz_threshold = self.settings.value("stacking/modz_threshold", 3.5, type=float)
        self.chunk_height = self.settings.value("stacking/chunk_height", 2048, type=int)
        self.chunk_width = self.settings.value("stacking/chunk_width", 2048, type=int)        

        # Dictionaries to store file paths
        self.conversion_files = {}
        self.dark_files = {}
        self.flat_files = {}
        self.light_files = {}
        self.master_files = {}
        self.master_sizes = {}

        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        self.dir_path_edit = QLineEdit(self.stacking_directory)  # Add this here
        # Create the new Conversion tab.
        self.conversion_tab = self.create_conversion_tab()
        # Existing tabs...
        self.dark_tab = self.create_dark_tab()
        self.flat_tab = self.create_flat_tab()
        self.light_tab = self.create_light_tab()
        self.image_integration_tab = self.create_image_registration_tab()

        # Add the tabs in desired order. (Conversion first)
        self.tabs.addTab(self.conversion_tab, "Convert Non-FITS Formats")
        self.tabs.addTab(self.dark_tab, "Darks")
        self.tabs.addTab(self.flat_tab, "Flats")
        self.tabs.addTab(self.light_tab, "Lights")
        self.tabs.addTab(self.image_integration_tab, "Image Integration")
        self.tabs.setCurrentIndex(1)  # Default to Darks tab

        # Wrench button, status bar, etc.
        self.wrench_button = QPushButton()
        self.wrench_button.setIcon(QIcon(wrench_path))
        self.wrench_button.setToolTip("Set Stacking Directory & Sigma Clipping")
        self.wrench_button.clicked.connect(self.open_stacking_settings)
        self.wrench_button.setStyleSheet("""
            QPushButton {
                background-color: #FF4500;
                color: white;
                font-size: 16px;
                padding: 8px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #FF6347;
            }
        """)
        header_row = QHBoxLayout()
        header_row.addWidget(self.wrench_button)

        self.stacking_path_display = QLineEdit(self.stacking_directory or "")
        self.stacking_path_display.setReadOnly(True)
        self.stacking_path_display.setPlaceholderText("No stacking folder selected")
        self.stacking_path_display.setFrame(False)  # nicer, label-like look
        self.stacking_path_display.setToolTip(self.stacking_directory or "No stacking folder selected")
        header_row.addWidget(self.stacking_path_display, 1)  # stretch

        layout.addLayout(header_row)
        self.log_btn = QToolButton(self)
        self.log_btn.setText("Open Log")
        self.log_btn.setToolTip("Show the Stacking Suite log window")
        self.log_btn.clicked.connect(self._show_log_window)
        header_row.addWidget(self.log_btn)
        self.tabs.currentChanged.connect(self.on_tab_changed)
        self.restore_saved_master_calibrations()
        self._update_stacking_path_display()

        app = QApplication.instance()
        if not hasattr(app, "_sasd_status_console"):
            # parent=None so it floats independent of any single dialog
            app._sasd_status_console = StatusLogWindow(parent=None)
        self._status_console = app._sasd_status_console

        # every StackingSuiteDialog writes to the same console
        self.status_signal.connect(self._status_console.append_line)

        # show it (front) on open without stealing focus
        self._status_console.show_raise()

        # Keep a tiny “last line” label (optional)
        #self._last_status_label = QLabel("")
        #self._last_status_label.setStyleSheet("color:#bbb; font: 11px 'Monospace';")
        #header_row.addWidget(self._last_status_label)
        #self._last_status_max_chars = 50 

    def _elide_chars(self, s: str, max_chars: int) -> str:
        if not s:
            return ""
        s = s.replace("\n", " ")
        if len(s) <= max_chars:
            return s
        # reserve one char for the ellipsis
        return s[:max_chars - 1] + "…"

    def _dtype(self):
        return self.internal_dtype

    def _dtype_name(self):
        return np.dtype(self.internal_dtype).name

    def _set_last_status(self, message: str):
        disp = self._elide_chars(message, getattr(self, "_last_status_max_chars", 50))
        self._last_status_label.setText(disp)
        # keep full message available on hover
        self._last_status_label.setToolTip(message)

    def _show_log_window(self):
        # Prefer the instance we already cached during __init__
        if hasattr(self, "_status_console") and self._status_console:
            self._status_console.show_raise()
            return

        # Fallback to the app-global singleton (create if missing)
        app = QApplication.instance()
        console = getattr(app, "_sasd_status_console", None)
        if console is None:
            app._sasd_status_console = StatusLogWindow(parent=None)
            console = app._sasd_status_console
        self._status_console = console
        console.show_raise()

    def _label_with_dims(self, label: str, width: int, height: int) -> str:
        """Replace or append (WxH) in a human label."""
        clean = _DIM_RE.sub("", label).rstrip()
        return f"{clean} ({width}x{height})"

    def _update_stacking_path_display(self):
        txt = self.stacking_directory or ""
        self.stacking_path_display.setText(txt)
        self.stacking_path_display.setToolTip(txt or "No stacking folder selected")

    def restore_saved_master_calibrations(self):
        saved_darks = self.settings.value("stacking/master_darks", [], type=list)
        saved_flats = self.settings.value("stacking/master_flats", [], type=list)

        if saved_darks:
            self.add_master_files(self.master_dark_tree, "DARK", saved_darks)

        if saved_flats:
            self.add_master_files(self.master_flat_tree, "FLAT", saved_flats)

    def create_conversion_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.addWidget(QLabel("Batch Convert Files to Debayered FITS (.fit)"))

        # 1) Create the tree
        self.conversion_tree = QTreeWidget()
        self.conversion_tree.setColumnCount(2)
        self.conversion_tree.setHeaderLabels(["File", "Status"])

        # 2) Make columns user-resizable (Interactive)
        header = self.conversion_tree.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Interactive)

        # 3) After populating the tree, do an initial auto-resize
        self.conversion_tree.resizeColumnToContents(0)
        self.conversion_tree.resizeColumnToContents(1)
        layout.addWidget(self.conversion_tree)

        # Buttons for adding files, adding a directory,
        # selecting an output directory, and clearing the list.
        btn_layout = QHBoxLayout()
        self.add_conversion_files_btn = QPushButton("Add Conversion Files")
        self.add_conversion_files_btn.clicked.connect(self.add_conversion_files)
        self.add_conversion_dir_btn = QPushButton("Add Conversion Directory")
        self.add_conversion_dir_btn.clicked.connect(self.add_conversion_directory)
        self.select_conversion_output_btn = QPushButton("Select Output Directory")
        self.select_conversion_output_btn.clicked.connect(self.select_conversion_output_dir)
        self.clear_conversion_btn = QPushButton("Clear List")
        self.clear_conversion_btn.clicked.connect(self.clear_conversion_list)
        btn_layout.addWidget(self.add_conversion_files_btn)
        btn_layout.addWidget(self.add_conversion_dir_btn)
        btn_layout.addWidget(self.select_conversion_output_btn)
        btn_layout.addWidget(self.clear_conversion_btn)
        layout.addLayout(btn_layout)

        # Convert All button (converts all files in the tree).
        self.convert_btn = QPushButton("Convert All Files to FITS")
        self.convert_btn.clicked.connect(self.convert_all_files)
        layout.addWidget(self.convert_btn)

        return tab

    def add_conversion_files(self):
        last_dir = self.settings.value("last_opened_folder", "", type=str)
        files, _ = QFileDialog.getOpenFileNames(self, "Select Files for Conversion", last_dir,
                                                "Supported Files (*.fits *.fit *.fz *.fz *.fits.gz *.fit.gz *.tiff *.tif *.png *.jpg *.jpeg *.cr2 *.cr3 *.nef *.arw *.dng *.orf *.rw2 *.pef *.xisf)")
        if files:
            self.settings.setValue("last_opened_folder", os.path.dirname(files[0]))
            for file in files:
                item = QTreeWidgetItem([os.path.basename(file), "Pending"])
                item.setData(0, 1000, file)  # store full path in role 1000
                self.conversion_tree.addTopLevelItem(item)

    def add_conversion_directory(self):
        last_dir = self.settings.value("last_opened_folder", "", type=str)
        directory = QFileDialog.getExistingDirectory(self, "Select Directory for Conversion", last_dir)
        if directory:
            self.settings.setValue("last_opened_folder", directory)
            for file in os.listdir(directory):
                if file.lower().endswith((".fits", ".fit", ".fz", ".fz", ".fit.gz", ".fits.gz", ".tiff", ".tif", ".png", ".jpg", ".jpeg", 
                                           ".cr2", ".cr3", ".nef", ".arw", ".dng", ".orf", ".rw2", ".pef", ".xisf")):
                    full_path = os.path.join(directory, file)
                    item = QTreeWidgetItem([file, "Pending"])
                    item.setData(0, 1000, full_path)
                    self.conversion_tree.addTopLevelItem(item)

    def select_conversion_output_dir(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Conversion Output Directory")
        if directory:
            self.conversion_output_directory = directory
            self.update_status(f"Conversion output directory set to: {directory}")

    def clear_conversion_list(self):
        self.conversion_tree.clear()
        self.update_status("Conversion list cleared.")

    def convert_all_files(self):
        # If no output directory is set, ask the user if they want to set it now.
        if not self.conversion_output_directory:
            reply = QMessageBox.question(
                self,
                "No Output Directory",
                "No output directory is set. Do you want to select one now?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.select_conversion_output_dir()  # Let them pick a folder
            else:
                # They chose 'No' → just stop
                return

            # If it's still empty after that, bail out
            if not self.conversion_output_directory:
                QMessageBox.warning(self, "No Output Directory", "Please select a conversion output directory first.")
                return

        count = self.conversion_tree.topLevelItemCount()
        if count == 0:
            QMessageBox.information(self, "No Files", "There are no files to convert.")
            return

        # 1) Show the batch settings dialog
        dialog = BatchSettingsDialog(self)
        result = dialog.exec()
        if result == int(QDialog.DialogCode.Rejected):
            # user canceled
            return
        # user pressed OK => get the values
        imagetyp_user, exptime_user, filter_user = dialog.get_values()

        for i in range(count):
            item = self.conversion_tree.topLevelItem(i)
            file_path = item.data(0, 1000)
            result = load_image(file_path)
            if result[0] is None:
                item.setText(1, "Failed to load")
                self.update_status(f"Failed to load {os.path.basename(file_path)}")
                continue

            image, header, bit_depth, is_mono = result

            if image is None:
                item.setText(1, "Failed to load")
                self.update_status(f"Failed to load {os.path.basename(file_path)}")
                continue

            # 🔹 If the file has no header (TIFF, PNG, JPG, etc.), create a minimal one
            if header is None:
                header = fits.Header()
                header["SIMPLE"]   = True
                header["BITPIX"]   = 16  # Or 16, depending on your preference
                header["CREATOR"]  = "SetiAstroSuite"
                header["IMAGETYP"] = "UNKNOWN"  # We'll set it properly below
                header["EXPTIME"]  = "Unknown"  # Just a placeholder
                # You can add more default keywords as needed

            # Debayer if needed:
            image = self.debayer_image(image, file_path, header)
            if image.ndim == 3:
                is_mono = False

            # If it's a RAW format, definitely treat as color
            if file_path.lower().endswith(('.cr2', '.cr3', '.nef', '.arw', '.dng', '.orf', '.rw2', '.pef')):
                is_mono = False

                # Try extracting EXIF metadata
                try:
                    with open(file_path, 'rb') as f:
                        tags = exifread.process_file(f, details=False)

                    exptime_tag = tags.get("EXIF ExposureTime")  # e.g. "1/125"
                    iso_tag = tags.get("EXIF ISOSpeedRatings")
                    date_obs_tag = tags.get("EXIF DateTimeOriginal")

                    # Create or replace with a fresh header, but keep some existing fields if desired
                    new_header = fits.Header()
                    new_header['SIMPLE'] = True
                    new_header['BITPIX'] = 16
                    new_header['IMAGETYP'] = header.get('IMAGETYP', "UNKNOWN")

                    # Attempt to parse exptime. If fraction or numeric fails, store 'Unknown'.
                    if exptime_tag:
                        exptime_str = str(exptime_tag.values)  # or exptime_tag.printable
                        # Attempt fraction or float
                        try:
                            if '/' in exptime_str:  
                                # e.g. "1/125"
                                top, bot = exptime_str.split('/', 1)
                                fexp = float(top) / float(bot)
                                new_header['EXPTIME'] = (fexp, "Exposure Time in seconds")
                            else:
                                # e.g. "0.008" or "8"
                                fexp = float(exptime_str)
                                new_header['EXPTIME'] = (fexp, "Exposure Time in seconds")
                        except (ValueError, ZeroDivisionError):
                            new_header['EXPTIME'] = 'Unknown'
                    # If no exptime_tag, set Unknown
                    else:
                        new_header['EXPTIME'] = 'Unknown'

                    if iso_tag:
                        new_header['ISO'] = str(iso_tag.values)
                    if date_obs_tag:
                        new_header['DATE-OBS'] = str(date_obs_tag.values)

                    # Replace old header with new
                    header = new_header

                except Exception as e:
                    # If exif extraction fails for any reason, we just keep the existing header
                    # but ensure we set EXPTIME if missing
                    self.update_status(f"Warning: Failed to extract RAW header from {os.path.basename(file_path)}: {e}")

            header['IMAGETYP'] = imagetyp_user
            header['FILTER'] = filter_user

            # For exptime_user, try to parse float or fraction
            try:
                if '/' in exptime_user:
                    top, bot = exptime_user.split('/', 1)
                    exptime_val = float(top) / float(bot)
                    header['EXPTIME'] = (exptime_val, "User-specified exposure (s)")
                else:
                    exptime_val = float(exptime_user)
                    header['EXPTIME'] = (exptime_val, "User-specified exposure (s)")
            except (ValueError, ZeroDivisionError):
                # If user typed "Unknown" or something non-numeric
                header['EXPTIME'] = exptime_user

            # Remove any existing NAXIS keywords
            for key in ["NAXIS", "NAXIS1", "NAXIS2", "NAXIS3"]:
                header.pop(key, None)

            if image.ndim == 2:
                header['NAXIS'] = 2
                header['NAXIS1'] = image.shape[1]
                header['NAXIS2'] = image.shape[0]
            elif image.ndim == 3:
                header['NAXIS'] = 3
                header['NAXIS1'] = image.shape[1]
                header['NAXIS2'] = image.shape[0]
                header['NAXIS3'] = image.shape[2]

            # -- Ensure EXPTIME is defined --
            if 'EXPTIME' not in header:
                # If the camera or exif didn't provide it, we set it to 'Unknown'
                header['EXPTIME'] = 'Unknown'

            # Build output filename and save
            base = os.path.basename(file_path)
            name, _ = os.path.splitext(base)
            output_filename = os.path.join(self.conversion_output_directory, f"{name}.fit")
            image=image/np.max(image)

            try:
                save_image(
                    img_array=image,
                    filename=output_filename,
                    original_format="fit",
                    bit_depth="16-bit",
                    original_header=header,
                    is_mono=is_mono
                )
                item.setText(1, "Converted")
                self.update_status(
                    f"Converted {os.path.basename(file_path)} to FITS with "
                    f"IMAGETYP={header['IMAGETYP']}, EXPTIME={header['EXPTIME']}."
                )
            except Exception as e:
                item.setText(1, f"Error: {e}")
                self.update_status(f"Error converting {os.path.basename(file_path)}: {e}")

            QApplication.processEvents()

        self.update_status("Conversion complete.")



    def debayer_image(self, image, file_path, header):
        if file_path.lower().endswith(('.cr2', '.cr3', '.nef', '.arw', '.dng', '.orf', '.rw2', '.pef')):
            print(f"Debayering RAW image: {file_path}")
            return debayer_raw_fast(image)
        elif file_path.lower().endswith(('.fits', '.fit', '.fz')):
            bayer_pattern = header.get('BAYERPAT')
            if bayer_pattern:
                print(f"Debayering FITS image: {file_path} with Bayer pattern {bayer_pattern}")
                return debayer_fits_fast(image, bayer_pattern)
        return image

    def setup_status_bar(self, layout):
        """ Sets up a scrollable status log at the bottom of the UI. """
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        self.status_text.setStyleSheet(
            "background-color: black; color: white; font-family: Monospace; padding: 4px;"
        )

        self.status_scroll = QScrollArea()
        self.status_scroll.setWidgetResizable(True)
        self.status_scroll.setWidget(self.status_text)
        # Make the scroll area respect a fixed height
        self.status_scroll.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed
        )
        layout.addWidget(self.status_scroll)

        # show ~10 lines
        self.set_status_visible_lines(6)

    def set_status_visible_lines(self, n_lines: int):
        fm = QFontMetrics(self.status_text.font())
        line_h = fm.lineSpacing()

        # Add margins/frames (a small fudge keeps things from clipping)
        frame = self.status_text.frameWidth()
        docm  = int(self.status_text.document().documentMargin())
        extra = 2 * frame + 2 * docm + 8

        self.status_scroll.setFixedHeight(int(n_lines * line_h + extra))

    @pyqtSlot(str)
    def _update_status_gui(self, message: str):
        # tiny ‘now’ indicator in the dialog header
        if hasattr(self, "_last_status_label"):
            self._last_status_label.setText(message)
            self._set_last_status(message)


    def update_status(self, message: str):
        if QThread.currentThread() is self._gui_thread:
            self._update_status_gui(message)
            # ALSO emit so the log window gets the line if we’re on the GUI thread
            self.status_signal.emit(message)
        else:
            self.status_signal.emit(message)

    @pyqtSlot(str)
    def _on_post_status(self, msg: str):
        # 1) your central logger
        self.update_status(msg)
        # 2) also reflect in the progress dialog label if it exists
        try:
            if getattr(self, "post_progress", None):
                self.post_progress.setLabelText(msg)
                QApplication.processEvents()
        except Exception:
            pass


    def _norm_dir(self, p: str) -> str:
        if not p:
            return ""
        p = os.path.expanduser(os.path.expandvars(p))
        p = os.path.abspath(p)
        p = os.path.normpath(p)
        if os.name == "nt":
            p = p.lower()
        return p

    def _choose_dir_into(self, line_edit: QLineEdit):
        d = QFileDialog.getExistingDirectory(self, "Select Stacking Directory",
                                            line_edit.text() or self.stacking_directory or "")
        if d:
            line_edit.setText(d)

    def open_stacking_settings(self):
        """ Opens a dialog to set the stacking directory, sigma values, rejection algorithm, and algorithm parameters. """
        dialog = QDialog(self)
        dialog.setWindowTitle("Stacking Settings")
        layout = QVBoxLayout(dialog)

        # --- Stacking directory selection (DIALOG-SCOPED, no collisions) ---
        dir_layout = QHBoxLayout()
        dir_label = QLabel("Stacking Directory:")
        dir_edit = QLineEdit(self.stacking_directory or "")       # <<< changed: local, not self.dir_path_edit
        dialog._dir_edit = dir_edit                               # <<< changed: actually assign the defined variable
        dir_button = QPushButton("Browse")
        dir_button.clicked.connect(lambda: self._choose_dir_into(dir_edit))  # <<< changed
        dir_layout.addWidget(dir_label)
        dir_layout.addWidget(dir_edit)                            # <<< changed
        dir_layout.addWidget(dir_button)
        layout.addLayout(dir_layout)

        prec_layout = QHBoxLayout()
        prec_layout.addWidget(QLabel("Internal stacking precision:"))
        self.precision_combo = QComboBox()
        self.precision_combo.addItems(["32-bit float", "64-bit float"])
        # reflect current value
        self.precision_combo.setCurrentIndex(1 if self.internal_dtype is np.float64 else 0)
        self.precision_combo.setToolTip("64-bit uses ~2× RAM but can reduce rounding; 32-bit is faster/lighter.")
        prec_layout.addWidget(self.precision_combo)
        prec_layout.addStretch(1)
        layout.addLayout(prec_layout)

        # Sigma High & Low settings
        sigma_layout = QHBoxLayout()
        sigma_layout.addWidget(QLabel("Sigma High:"))
        self.sigma_high_spinbox = QDoubleSpinBox()
        self.sigma_high_spinbox.setRange(0.1, 10.0)
        self.sigma_high_spinbox.setDecimals(2)
        self.sigma_high_spinbox.setValue(self.sigma_high)
        sigma_layout.addWidget(self.sigma_high_spinbox)
        sigma_layout.addWidget(QLabel("Sigma Low:"))
        self.sigma_low_spinbox = QDoubleSpinBox()
        self.sigma_low_spinbox.setRange(0.1, 10.0)
        self.sigma_low_spinbox.setDecimals(2)
        self.sigma_low_spinbox.setValue(self.sigma_low)
        sigma_layout.addWidget(self.sigma_low_spinbox)
        layout.addLayout(sigma_layout)

        chunk_layout = QHBoxLayout()
        chunk_layout.addWidget(QLabel("Chunk Height:"))
        self.chunkHeightSpinBox = QSpinBox()
        self.chunkHeightSpinBox.setRange(128, 8192)
        self.chunkHeightSpinBox.setValue(self.settings.value("stacking/chunk_height", 2048, type=int))
        chunk_layout.addWidget(self.chunkHeightSpinBox)

        chunk_layout.addWidget(QLabel("Chunk Width:"))
        self.chunkWidthSpinBox = QSpinBox()
        self.chunkWidthSpinBox.setRange(128, 8192)
        self.chunkWidthSpinBox.setValue(self.settings.value("stacking/chunk_width", 2048, type=int))
        chunk_layout.addWidget(self.chunkWidthSpinBox)
        layout.addLayout(chunk_layout)

        # Alignment refinement passes
        align_layout = QHBoxLayout()
        align_layout.addWidget(QLabel("Alignment refinement:"))
        self.align_passes_combo = QComboBox()
        self.align_passes_combo.addItems(["Fast (1 pass)", "Accurate (3 passes)"])
        self.align_passes_combo.setToolTip(
            "Fast (1 pass): single astroalign pass; all successfully transformed frames are accepted.\n"
            "Accurate (3 passes): iterative refinement for sub-pixel accuracy; final outliers can be rejected."
        )        
        curr_passes = self.settings.value("stacking/refinement_passes", 3, type=int)
        self.align_passes_combo.setCurrentIndex(0 if curr_passes <= 1 else 1)
        align_layout.addWidget(self.align_passes_combo)

        align_layout.addSpacing(16)
        align_layout.addWidget(QLabel("Accept tolerance (px):"))
        self.shift_tol_spin = QDoubleSpinBox()
        self.shift_tol_spin.setRange(0.05, 5.0)
        self.shift_tol_spin.setDecimals(2)
        self.shift_tol_spin.setSingleStep(0.05)
        self.shift_tol_spin.setValue(self.settings.value("stacking/shift_tolerance", 0.2, type=float))
        self.shift_tol_spin.setToolTip("Convergence threshold per pass; fast mode ignores this for early stop but it is still used for progress messages.")
        align_layout.addWidget(self.shift_tol_spin)
        align_layout.addStretch(1)
        layout.addLayout(align_layout)

        # Rejection algorithm selection
        algo_layout = QHBoxLayout()
        algo_label = QLabel("Rejection Algorithm:")
        self.rejection_algo_combo = QComboBox()
        self.rejection_algo_combo.addItems([
            "Weighted Windsorized Sigma Clipping",
            "Kappa-Sigma Clipping",
            "Simple Average (No Rejection)",
            "Simple Median (No Rejection)",
            "Trimmed Mean",
            "Extreme Studentized Deviate (ESD)",
            "Biweight Estimator",
            "Modified Z-Score Clipping",
            "Max Value"
        ])
        saved_algo = self.settings.value("stacking/rejection_algorithm", "Weighted Windsorized Sigma Clipping")
        index = self.rejection_algo_combo.findText(saved_algo)
        if index >= 0:
            self.rejection_algo_combo.setCurrentIndex(index)
        algo_layout.addWidget(algo_label)
        algo_layout.addWidget(self.rejection_algo_combo)
        layout.addLayout(algo_layout)

        # --- Additional Parameters ---

        # Kappa-Sigma Clipping: Kappa Value
        kappa_layout = QHBoxLayout()
        kappa_label = QLabel("Kappa Value:")
        self.kappa_spinbox = QDoubleSpinBox()
        self.kappa_spinbox.setRange(0.1, 10.0)
        self.kappa_spinbox.setDecimals(2)
        self.kappa_spinbox.setValue(self.settings.value("stacking/kappa", 2.5, type=float))
        kappa_help = QPushButton("?")
        kappa_help.setFixedSize(20, 20)
        kappa_help.clicked.connect(lambda: QMessageBox.information(self, "Kappa Value", 
            "Kappa determines how many standard deviations away from the median are considered outliers. Higher values are more lenient."))
        kappa_layout.addWidget(kappa_label)
        kappa_layout.addWidget(self.kappa_spinbox)
        kappa_layout.addWidget(kappa_help)
        layout.addLayout(kappa_layout)

        # Kappa-Sigma Clipping: Iterations
        iterations_layout = QHBoxLayout()
        iterations_label = QLabel("Iterations:")
        self.iterations_spinbox = QSpinBox()
        self.iterations_spinbox.setRange(1, 10)
        self.iterations_spinbox.setValue(self.settings.value("stacking/iterations", 3, type=int))
        iterations_help = QPushButton("?")
        iterations_help.setFixedSize(20, 20)
        iterations_help.clicked.connect(lambda: QMessageBox.information(self, "Iterations", 
            "The number of iterations to perform kappa-sigma clipping. More iterations may remove more outliers."))
        iterations_layout.addWidget(iterations_label)
        iterations_layout.addWidget(self.iterations_spinbox)
        iterations_layout.addWidget(iterations_help)
        layout.addLayout(iterations_layout)

        # ESD: ESD Threshold
        esd_layout = QHBoxLayout()
        esd_label = QLabel("ESD Threshold:")
        self.esd_spinbox = QDoubleSpinBox()
        self.esd_spinbox.setRange(0.1, 10.0)
        self.esd_spinbox.setDecimals(2)
        self.esd_spinbox.setValue(self.settings.value("stacking/esd_threshold", 3.0, type=float))
        esd_help = QPushButton("?")
        esd_help.setFixedSize(20, 20)
        esd_help.clicked.connect(lambda: QMessageBox.information(self, "ESD Threshold", 
            "Threshold for the Extreme Studentized Deviate test. Lower values are more aggressive in rejecting outliers."))
        esd_layout.addWidget(esd_label)
        esd_layout.addWidget(self.esd_spinbox)
        esd_layout.addWidget(esd_help)
        layout.addLayout(esd_layout)

        # Biweight Estimator: Tuning Constant
        biweight_layout = QHBoxLayout()
        biweight_label = QLabel("Biweight Tuning Constant:")
        self.biweight_spinbox = QDoubleSpinBox()
        self.biweight_spinbox.setRange(1.0, 10.0)
        self.biweight_spinbox.setDecimals(2)
        self.biweight_spinbox.setValue(self.settings.value("stacking/biweight_constant", 6.0, type=float))
        biweight_help = QPushButton("?")
        biweight_help.setFixedSize(20, 20)
        biweight_help.clicked.connect(lambda: QMessageBox.information(self, "Biweight Tuning Constant", 
            "Tuning constant for the biweight estimator; it controls the aggressiveness of down-weighting outliers."))
        biweight_layout.addWidget(biweight_label)
        biweight_layout.addWidget(self.biweight_spinbox)
        biweight_layout.addWidget(biweight_help)
        layout.addLayout(biweight_layout)

        # Trimmed Mean: Trim Fraction
        trim_layout = QHBoxLayout()
        trim_label = QLabel("Trim Fraction:")
        self.trim_spinbox = QDoubleSpinBox()
        self.trim_spinbox.setRange(0.0, 0.5)
        self.trim_spinbox.setDecimals(2)
        self.trim_spinbox.setValue(self.settings.value("stacking/trim_fraction", 0.1, type=float))
        trim_help = QPushButton("?")
        trim_help.setFixedSize(20, 20)
        trim_help.clicked.connect(lambda: QMessageBox.information(self, "Trim Fraction", 
            "Fraction of values to trim from each end before averaging. For example, 0.1 will trim 10% from each end."))
        trim_layout.addWidget(trim_label)
        trim_layout.addWidget(self.trim_spinbox)
        trim_layout.addWidget(trim_help)
        layout.addLayout(trim_layout)

        # Modified Z-Score Clipping: Threshold
        modz_layout = QHBoxLayout()
        modz_label = QLabel("Modified Z-Score Threshold:")
        self.modz_spinbox = QDoubleSpinBox()
        self.modz_spinbox.setRange(0.1, 10.0)
        self.modz_spinbox.setDecimals(2)
        self.modz_spinbox.setValue(self.settings.value("stacking/modz_threshold", 3.5, type=float))
        modz_help = QPushButton("?")
        modz_help.setFixedSize(20, 20)
        modz_help.clicked.connect(lambda: QMessageBox.information(self, "Modified Z-Score Threshold", 
            "Threshold for the modified z-score clipping using the median absolute deviation. Lower values are more aggressive."))
        modz_layout.addWidget(modz_label)
        modz_layout.addWidget(self.modz_spinbox)
        modz_layout.addWidget(modz_help)
        layout.addLayout(modz_layout)

        # Save button
        save_button = QPushButton("Save Settings")
        save_button.clicked.connect(lambda: self.save_stacking_settings(dialog))
        layout.addWidget(save_button)

        dialog.exec()

    def closeEvent(self, e):
        # Graceful shutdown for any running workers
        try:
            if hasattr(self, "alignment_thread") and self.alignment_thread and self.alignment_thread.isRunning():
                self.alignment_thread.requestInterruption()
                self.alignment_thread.wait(1500)
        except Exception:
            pass
        super().closeEvent(e)

    def save_stacking_settings(self, dialog):
        """
        Save settings and restart the Stacking Suite if the directory OR internal dtype changed.
        Uses dialog-scoped dir_edit and normalized path comparison.
        """
        # --- capture previous state BEFORE we change anything ---
        prev_dir_raw   = self.stacking_directory or ""
        prev_dir       = self._norm_dir(prev_dir_raw)
        prev_dtype_str = "float64" if (getattr(self, "internal_dtype", np.float64) is np.float64) else "float32"

        # --- read dialog widgets ---
        dir_edit   = getattr(dialog, "_dir_edit", None)
        new_dir_raw = (dir_edit.text() if dir_edit else prev_dir_raw)
        new_dir     = self._norm_dir(new_dir_raw)

        # Persist the rest
        self.sigma_high       = self.sigma_high_spinbox.value()
        self.sigma_low        = self.sigma_low_spinbox.value()
        self.rejection_algorithm = self.rejection_algo_combo.currentText()
        self.kappa           = self.kappa_spinbox.value()
        self.iterations      = self.iterations_spinbox.value()
        self.esd_threshold   = self.esd_spinbox.value()
        self.biweight_constant = self.biweight_spinbox.value()
        self.trim_fraction   = self.trim_spinbox.value()
        self.modz_threshold  = self.modz_spinbox.value()
        self.chunk_height    = self.chunkHeightSpinBox.value()
        self.chunk_width     = self.chunkWidthSpinBox.value()

        # Update instance + QSettings (write RAW path; use normalized only for comparison)
        self.stacking_directory = new_dir_raw
        self.settings.setValue("stacking/dir", new_dir_raw)
        self.settings.setValue("stacking/sigma_high", self.sigma_high)
        self.settings.setValue("stacking/sigma_low", self.sigma_low)
        self.settings.setValue("stacking/rejection_algorithm", self.rejection_algorithm)
        self.settings.setValue("stacking/kappa", self.kappa)
        self.settings.setValue("stacking/iterations", self.iterations)
        self.settings.setValue("stacking/esd_threshold", self.esd_threshold)
        self.settings.setValue("stacking/biweight_constant", self.biweight_constant)
        self.settings.setValue("stacking/trim_fraction", self.trim_fraction)
        self.settings.setValue("stacking/modz_threshold", self.modz_threshold)
        self.settings.setValue("stacking/chunk_height", self.chunk_height)
        self.settings.setValue("stacking/chunk_width", self.chunk_width)
        self.settings.setValue("stacking/autocrop_enabled", self.autocrop_cb.isChecked())
        self.settings.setValue("stacking/autocrop_pct", float(self.autocrop_pct.value()))

        passes = 1 if self.align_passes_combo.currentIndex() == 0 else 3
        self.settings.setValue("stacking/refinement_passes", passes)
        self.settings.setValue("stacking/shift_tolerance", self.shift_tol_spin.value())

        # --- precision (internal dtype) ---
        chosen = self.precision_combo.currentText()  # "32-bit float" or "64-bit float"
        new_dtype_str = "float64" if "64" in chosen else "float32"
        dtype_changed = (new_dtype_str != prev_dtype_str)

        self.internal_dtype = np.float64 if new_dtype_str == "float64" else np.float32
        self.settings.setValue("stacking/internal_dtype", new_dtype_str)

        # Make sure everything is flushed
        self.settings.sync()

        # Logging
        self.update_status("✅ Saved stacking settings.")
        self.update_status(f"• Internal precision: {new_dtype_str}")
        self._update_stacking_path_display()

        # --- restart if needed ---
        dir_changed = (new_dir != prev_dir)
        if dir_changed or dtype_changed:
            reasons = []
            if dir_changed:
                reasons.append("folder change")
            if dtype_changed:
                reasons.append(f"precision → {new_dtype_str}")
            self.update_status(f"🔁 Restarting Stacking Suite to apply {', '.join(reasons)}…")
            dialog.accept()
            self._restart_self()
            return

        dialog.accept()


    def _restart_self(self):
        geom = self.saveGeometry()
        try:
            cur_tab = self.tabs.currentIndex()
        except Exception:
            cur_tab = None

        parent = self.parent()  # may be None

        app = QApplication.instance()
        # Keep a global strong ref so GC can't collect the new dialog
        if not hasattr(app, "_stacking_suite_ref"):
            app._stacking_suite_ref = None

        def spawn():
            new = StackingSuiteDialog(parent=parent)
            if geom:
                new.restoreGeometry(geom)
            if cur_tab is not None:
                try:
                    new.tabs.setCurrentIndex(cur_tab)
                except Exception:
                    pass
            new.show()
            app._stacking_suite_ref = new  # <<< strong ref lives for app lifetime

        QTimer.singleShot(0, spawn)
        self.close()

    def _on_stacking_directory_changed(self, old_dir: str, new_dir: str):
        # Stop any running worker safely
        if hasattr(self, "alignment_thread") and self.alignment_thread:
            try:
                if self.alignment_thread.isRunning():
                    self.alignment_thread.requestInterruption()
                    self.alignment_thread.wait(1500)
            except Exception:
                pass

        self._ensure_stacking_subdirs(new_dir)
        self._clear_integration_state()

        # 🔁 RESCAN + REPOPULATE (the key bit you’re missing)
        self._reload_lists_for_new_dir()

        # If your tabs populate on change, poke the active one:
        if hasattr(self, "on_tab_changed"):
            self.on_tab_changed(self.tabs.currentIndex())

        # Update any path labels
        self._update_stacking_path_display()

        # Reload any persisted master selections
        try:
            self.restore_saved_master_calibrations()
        except Exception:
            pass

        self.update_status(f"📂 Stacking directory changed:\n    {old_dir or '(none)'} → {new_dir}")

    def _reload_lists_for_new_dir(self):
        """
        Re-scan the new stacking directory and repopulate internal dicts AND UI.
        """
        base = self.stacking_directory or ""
        self.conversion_output_directory = os.path.join(base, "Converted_Images")

        # Rebuild dictionaries from disk
        self.dark_files  = self._discover_grouped(os.path.join(base, "Calibrated_Darks"))
        self.flat_files  = self._discover_grouped(os.path.join(base, "Calibrated_Flats"))
        self.light_files = self._discover_grouped(os.path.join(base, "Calibrated_Lights"))

        # If you store master lists/sizes by path, clear/reseed minimally
        self.master_files.clear()
        self.master_sizes.clear()

        # 🔄 Update the tab UIs if you have builders; try common method names safely
        # Darks
        if hasattr(self, "rebuild_dark_tree"):
            self.rebuild_dark_tree(self.dark_files)
        elif hasattr(self, "populate_dark_tab"):
            self.populate_dark_tab()

        # Flats
        if hasattr(self, "rebuild_flat_tree"):
            self.rebuild_flat_tree(self.flat_files)
        elif hasattr(self, "populate_flat_tab"):
            self.populate_flat_tab()

        # Lights
        if hasattr(self, "rebuild_light_tree"):
            self.rebuild_light_tree(self.light_files)
        elif hasattr(self, "populate_light_tab"):
            self.populate_light_tab()

        # Image Integration (registration) tab often shows counts/paths
        if hasattr(self, "refresh_integration_tab"):
            self.refresh_integration_tab()

        self.update_status(f"🔄 Re-scanned calibrated sets in: {base}")

    def _discover_grouped(self, root_dir: str) -> dict:
        """
        Walk 'root_dir' and return {group_name: [file_paths,...]}.
        Group = immediate subfolder name; if files are directly in root, group 'Ungrouped'.
        """
        groups = {}
        if not root_dir or not os.path.isdir(root_dir):
            return groups

        valid_ext = (".fit", ".fits", ".xisf", ".tif", ".tiff")
        root_dir = os.path.normpath(root_dir)

        for dirpath, _, files in os.walk(root_dir):
            for fn in files:
                if not fn.lower().endswith(valid_ext):
                    continue
                fpath = os.path.normpath(os.path.join(dirpath, fn))
                parent = os.path.basename(os.path.dirname(fpath))
                group  = parent if os.path.dirname(fpath) != root_dir else "Ungrouped"
                groups.setdefault(group, []).append(fpath)

        # Stable ordering helps
        for g in groups:
            groups[g].sort()
        return groups

    def _refresh_all_tabs_once(self):
        current = self.tabs.currentIndex()
        if hasattr(self, "on_tab_changed"):
            for idx in range(self.tabs.count()):
                self.on_tab_changed(idx)
        self.tabs.setCurrentIndex(current)

    def _ensure_stacking_subdirs(self, base_dir: str):
        try:
            os.makedirs(base_dir, exist_ok=True)
            for sub in (
                "Aligned_Images",
                "Normalized_Images",
                "Calibrated_Darks",
                "Calibrated_Flats",
                "Calibrated_Lights",
                "Converted_Images",
                "Masters",
            ):
                os.makedirs(os.path.join(base_dir, sub), exist_ok=True)
        except Exception as e:
            self.update_status(f"⚠️ Could not ensure subfolders in '{base_dir}': {e}")

    def _clear_integration_state(self):
        # wipe per-run state so we don't “blend” two directories
        self.per_group_drizzle.clear()
        self.manual_dark_overrides.clear()
        self.manual_flat_overrides.clear()
        self.reg_files.clear()
        self.session_tags.clear()
        self.deleted_calibrated_files.clear()
        self._norm_map.clear()
        setattr(self, "valid_transforms", {})
        setattr(self, "frame_weights", {})
        setattr(self, "_global_autocrop_rect", None)

    def _rebuild_tabs_after_dir_change(self):
        # Rebuild the tab widgets so any path assumptions inside them reset to the new dir
        current = self.tabs.currentIndex()

        # Remove all tabs & delete widgets
        while self.tabs.count():
            w = self.tabs.widget(0)
            self.tabs.removeTab(0)
            try:
                w.deleteLater()
            except Exception:
                pass

        # Recreate against the new base path
        self.conversion_tab = self.create_conversion_tab()
        self.dark_tab       = self.create_dark_tab()
        self.flat_tab       = self.create_flat_tab()
        self.light_tab      = self.create_light_tab()
        self.image_integration_tab = self.create_image_registration_tab()

        self.tabs.addTab(self.conversion_tab, "Convert Non-FITS Formats")
        self.tabs.addTab(self.dark_tab,       "Darks")
        self.tabs.addTab(self.flat_tab,       "Flats")
        self.tabs.addTab(self.light_tab,      "Lights")
        self.tabs.addTab(self.image_integration_tab, "Image Integration")

        # Restore previously active tab if possible
        if 0 <= current < self.tabs.count():
            self.tabs.setCurrentIndex(current)
        else:
            self.tabs.setCurrentIndex(1)  # Darks by default

    def select_stacking_directory(self):
        """ Opens a dialog to choose a stacking directory. """
        directory = QFileDialog.getExistingDirectory(self, "Select Stacking Directory")
        if directory:
            self.stacking_directory = directory
            self.dir_path_edit.setText(directory)  # No more AttributeError
            self.settings.setValue("stacking/dir", directory)  # Save the new directory
            self._update_stacking_path_display()



    def create_dark_tab(self):
        tab = QWidget()
        main_layout = QVBoxLayout(tab)  # Vertical layout to separate sections

        # --- DARK FRAMES TREEBOX (TOP) ---
        darks_layout = QHBoxLayout()  # Left = Dark Tree, Right = Controls

        # Left Side - Dark Frames
        dark_frames_layout = QVBoxLayout()
        dark_frames_layout.addWidget(QLabel("Dark Frames"))
        # 1) Create the tree
        self.dark_tree = QTreeWidget()
        self.dark_tree.setColumnCount(2)
        self.dark_tree.setHeaderLabels(["Exposure Time", "Metadata"])
        self.dark_tree.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)

        # 2) Make columns user-resizable
        header = self.dark_tree.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Interactive)

        # 3) After you fill the tree with items, auto-resize
        self.dark_tree.resizeColumnToContents(0)
        self.dark_tree.resizeColumnToContents(1)

        # Then add it to the layout
        dark_frames_layout.addWidget(self.dark_tree)

        # Buttons to Add Dark Files & Directories
        btn_layout = QHBoxLayout()
        self.add_dark_files_btn = QPushButton("Add Dark Files")
        self.add_dark_files_btn.clicked.connect(self.add_dark_files)
        self.add_dark_dir_btn = QPushButton("Add Dark Directory")
        self.add_dark_dir_btn.clicked.connect(self.add_dark_directory)
        btn_layout.addWidget(self.add_dark_files_btn)
        btn_layout.addWidget(self.add_dark_dir_btn)
        dark_frames_layout.addLayout(btn_layout)

        self.clear_dark_selection_btn = QPushButton("Clear Selection")
        self.clear_dark_selection_btn.clicked.connect(lambda: self.clear_tree_selection(self.dark_tree, self.dark_files))
        dark_frames_layout.addWidget(self.clear_dark_selection_btn)

        darks_layout.addLayout(dark_frames_layout, 2)  # Dark Frames Tree takes more space


        # --- RIGHT SIDE: Exposure Tolerance & Master Darks Button ---
        right_controls_layout = QVBoxLayout()

        # Exposure Tolerance
        exposure_tolerance_layout = QHBoxLayout()
        exposure_tolerance_label = QLabel("Exposure Tolerance (seconds):")
        self.exposure_tolerance_spinbox = QSpinBox()
        self.exposure_tolerance_spinbox.setRange(0, 30)  # Acceptable range
        self.exposure_tolerance_spinbox.setValue(5)  # Default: ±5 sec
        exposure_tolerance_layout.addWidget(exposure_tolerance_label)
        exposure_tolerance_layout.addWidget(self.exposure_tolerance_spinbox)
        right_controls_layout.addLayout(exposure_tolerance_layout)

        # --- "Turn Those Darks Into Master Darks" Button ---
        self.create_master_dark_btn = QPushButton("Turn Those Darks Into Master Darks")
        self.create_master_dark_btn.clicked.connect(self.create_master_dark)

        # Apply a bold font, padding, and a highlighted effect
        self.create_master_dark_btn.setStyleSheet("""
            QPushButton {
                background-color: #333;  /* Dark gray */
                color: white;
                font-size: 14px;
                padding: 8px;
                border-radius: 5px;
                border: 2px solid yellow;  /* Subtle yellow border */
            }
            QPushButton:hover {
                border: 2px solid #FFD700;  /* Brighter yellow on hover */
            }
            QPushButton:pressed {
                background-color: #222;  /* Darker gray on press */
                border: 2px solid #FFA500;  /* Orange border when pressed */
            }
        """)

        right_controls_layout.addWidget(self.create_master_dark_btn)


        darks_layout.addLayout(right_controls_layout, 1)  # Right side takes less space

        main_layout.addLayout(darks_layout)

        # --- MASTER DARKS TREEBOX (BOTTOM) ---
        main_layout.addWidget(QLabel("Master Darks"))
        self.master_dark_tree = QTreeWidget()
        self.master_dark_tree.setColumnCount(2)
        self.master_dark_tree.setHeaderLabels(["Exposure Time", "Master File"])
        self.master_dark_tree.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        main_layout.addWidget(self.master_dark_tree)

        # Master Dark Selection Button
        self.master_dark_btn = QPushButton("Load Master Dark")
        self.master_dark_btn.clicked.connect(self.load_master_dark)
        main_layout.addWidget(self.master_dark_btn)

        # Add "Clear Selection" button for Master Darks
        self.clear_master_dark_selection_btn = QPushButton("Clear Selection")
        self.clear_master_dark_selection_btn.clicked.connect(
            lambda: self.clear_tree_selection(self.master_dark_tree, self.master_files)
        )
        self.clear_master_dark_selection_btn.clicked.connect(
            lambda: (self.clear_tree_selection(self.master_dark_tree, self.master_files),
                    self.save_master_paths_to_settings())
        )        
        main_layout.addWidget(self.clear_master_dark_selection_btn)

        return tab



    def create_flat_tab(self):
        tab = QWidget()
        main_layout = QVBoxLayout(tab)  # Main layout to organize sections

        # --- FLAT FRAMES TREEBOX (TOP) ---
        flats_layout = QHBoxLayout()  # Left = Flat Tree, Right = Controls

        # Left Side - Flat Frames
        flat_frames_layout = QVBoxLayout()
        flat_frames_layout.addWidget(QLabel("Flat Frames"))

        self.flat_tree = QTreeWidget()
        self.flat_tree.setColumnCount(3)  # Added 3rd column for Master Dark Used
        self.flat_tree.setHeaderLabels(["Filter & Exposure", "Metadata", "Master Dark Used"])
        self.flat_tree.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.flat_tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.flat_tree.customContextMenuRequested.connect(self.flat_tree_context_menu)
        flat_frames_layout.addWidget(self.flat_tree)

        # Buttons to Add Flat Files & Directories
        btn_layout = QHBoxLayout()
        self.add_flat_files_btn = QPushButton("Add Flat Files")
        self.add_flat_files_btn.clicked.connect(self.add_flat_files)
        self.add_flat_dir_btn = QPushButton("Add Flat Directory")
        self.add_flat_dir_btn.clicked.connect(self.add_flat_directory)
        btn_layout.addWidget(self.add_flat_files_btn)
        btn_layout.addWidget(self.add_flat_dir_btn)
        flat_frames_layout.addLayout(btn_layout)
        # 🔧 Session Tag Hint
        session_hint_label = QLabel("Right Click to Assign Session Keys if desired")
        session_hint_label.setStyleSheet("color: #888; font-style: italic; font-size: 11px; margin-left: 4px;")
        flat_frames_layout.addWidget(session_hint_label)

        # Add "Clear Selection" button for Flat Frames
        self.clear_flat_selection_btn = QPushButton("Clear Selection")
        self.clear_flat_selection_btn.clicked.connect(lambda: self.clear_tree_selection_flat(self.flat_tree, self.flat_files))
        flat_frames_layout.addWidget(self.clear_flat_selection_btn)

        flats_layout.addLayout(flat_frames_layout, 2)  # Left side takes more space

        # --- RIGHT SIDE: Exposure Tolerance & Master Dark Selection ---
        right_controls_layout = QVBoxLayout()

        # Exposure Tolerance
        exposure_tolerance_layout = QHBoxLayout()
        exposure_tolerance_label = QLabel("Exposure Tolerance (seconds):")
        self.flat_exposure_tolerance_spinbox = QSpinBox()
        self.flat_exposure_tolerance_spinbox.setRange(0, 30)  # Allow ±0 to 30 seconds
        self.flat_exposure_tolerance_spinbox.setValue(5)  # Default: ±5 sec
        exposure_tolerance_layout.addWidget(exposure_tolerance_label)
        exposure_tolerance_layout.addWidget(self.flat_exposure_tolerance_spinbox)
        right_controls_layout.addLayout(exposure_tolerance_layout)
        self.flat_exposure_tolerance_spinbox.valueChanged.connect(self.rebuild_flat_tree)


        # Auto-Select Master Dark
        self.auto_select_dark_checkbox = QCheckBox("Auto-Select Closest Master Dark")
        self.auto_select_dark_checkbox.setChecked(True)  # Default enabled
        right_controls_layout.addWidget(self.auto_select_dark_checkbox)

        # Manual Override: Select a Master Dark
        self.override_dark_combo = QComboBox()
        self.override_dark_combo.addItem("None (Use Auto-Select)")
        self.override_dark_combo.currentIndexChanged.connect(self.override_selected_master_dark)
        right_controls_layout.addWidget(QLabel("Override Master Dark Selection"))
        right_controls_layout.addWidget(self.override_dark_combo)

        self.create_master_flat_btn = QPushButton("Turn Those Flats Into Master Flats")
        self.create_master_flat_btn.clicked.connect(self.create_master_flat)

        # Apply a bold font, padding, and a glowing effect
        self.create_master_flat_btn.setStyleSheet("""
            QPushButton {
                background-color: #333;  /* Dark gray */
                color: white;
                font-size: 14px;
                padding: 8px;
                border-radius: 5px;
                border: 2px solid yellow;  /* Subtle yellow border */
            }
            QPushButton:hover {
                border: 2px solid #FFD700;  /* Brighter yellow on hover */
            }
            QPushButton:pressed {
                background-color: #222;  /* Darker gray on press */
                border: 2px solid #FFA500;  /* Orange border when pressed */
            }
        """)


        right_controls_layout.addWidget(self.create_master_flat_btn)

        flats_layout.addLayout(right_controls_layout, 1)  # Right side takes less space

        main_layout.addLayout(flats_layout)

        # --- MASTER FLATS TREEBOX (BOTTOM) ---
        main_layout.addWidget(QLabel("Master Flats"))
        self.master_flat_tree = QTreeWidget()
        self.master_flat_tree.setColumnCount(2)
        self.master_flat_tree.setHeaderLabels(["Filter", "Master File"])
        self.master_flat_tree.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        
        main_layout.addWidget(self.master_flat_tree)

        # Master Flat Selection Button
        self.master_flat_btn = QPushButton("Load Master Flat")
        self.master_flat_btn.clicked.connect(self.load_master_flat)
        main_layout.addWidget(self.master_flat_btn)

        self.clear_master_flat_selection_btn = QPushButton("Clear Selection")
        self.clear_master_flat_selection_btn.clicked.connect(
            lambda: (self.clear_tree_selection(self.master_flat_tree, self.master_files),
                    self.save_master_paths_to_settings())
        )
        main_layout.addWidget(self.clear_master_flat_selection_btn)
        return tab

    def flat_tree_context_menu(self, position):
        item = self.flat_tree.itemAt(position)
        if item:
            menu = QMenu()
            set_session_action = menu.addAction("Set Session Tag")
            action = menu.exec(self.flat_tree.viewport().mapToGlobal(position))
            if action == set_session_action:
                self.prompt_set_session(item, "flat")

    def create_light_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Tree widget for light frames
        self.light_tree = QTreeWidget()
        self.light_tree.setColumnCount(5)  # Added columns for Master Dark and Flat
        self.light_tree.setHeaderLabels(["Filter & Exposure", "Metadata", "Master Dark", "Master Flat", "Corrections"])
        self.light_tree.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)

        layout.addWidget(QLabel("Light Frames"))
        layout.addWidget(self.light_tree)

        # Buttons for adding files and directories
        btn_layout = QHBoxLayout()
        self.add_light_files_btn = QPushButton("Add Light Files")
        self.add_light_files_btn.clicked.connect(self.add_light_files)
        self.add_light_dir_btn = QPushButton("Add Light Directory")
        self.add_light_dir_btn.clicked.connect(self.add_light_directory)
        btn_layout.addWidget(self.add_light_files_btn)
        btn_layout.addWidget(self.add_light_dir_btn)
        layout.addLayout(btn_layout)
        session_hint_label = QLabel("Right Click to Assign Session Keys if desired")
        session_hint_label.setStyleSheet("color: #888; font-style: italic; font-size: 11px; margin-left: 4px;")
        layout.addWidget(session_hint_label)

        clear_selection_btn = QPushButton("Remove Selected")
        clear_selection_btn.clicked.connect(lambda: self.clear_tree_selection_light(self.light_tree))
        layout.addWidget(clear_selection_btn)

        # Cosmetic Correction & Pedestal Controls
        correction_layout = QHBoxLayout()

        self.cosmetic_checkbox = QCheckBox("Enable Cosmetic Correction")
        self.pedestal_checkbox = QCheckBox("Apply Pedestal")
        self.bias_checkbox = QCheckBox("Apply Bias Subtraction (For CCD Users)")

        # Pedestal Value (0-1000, converted to 0-1)
        pedestal_layout = QHBoxLayout()
        self.pedestal_spinbox = QSpinBox()
        self.pedestal_spinbox.setRange(0, 1000)
        self.pedestal_spinbox.setValue(50)  # Default pedestal
        pedestal_layout.addWidget(QLabel("Pedestal (0-1000):"))
        pedestal_layout.addWidget(self.pedestal_spinbox)
        layout.addLayout(pedestal_layout)        

        # Tooltip for Bias Checkbox
        self.bias_checkbox.setToolTip(
            "CMOS users: Bias Subtraction is not needed.\n"
            "Modern CMOS cameras use Correlated Double Sampling (CDS),\n"
            "meaning bias is already subtracted at the sensor level."
        )

        # Connect checkboxes to update function
        self.cosmetic_checkbox.stateChanged.connect(self.update_light_corrections)
        self.pedestal_checkbox.stateChanged.connect(self.update_light_corrections)
        self.bias_checkbox.stateChanged.connect(self.update_light_corrections)

        # Add checkboxes to layout
        correction_layout.addWidget(self.cosmetic_checkbox)
        correction_layout.addWidget(self.pedestal_checkbox)
        correction_layout.addWidget(self.bias_checkbox)

        layout.addLayout(correction_layout)        

        # --- RIGHT SIDE CONTROLS: Override Dark & Flat ---
        override_layout = QHBoxLayout()

        self.override_dark_btn = QPushButton("Override Dark Frame")
        self.override_dark_btn.clicked.connect(self.override_selected_master_dark)
        override_layout.addWidget(self.override_dark_btn)

        self.override_flat_btn = QPushButton("Override Flat Frame")
        self.override_flat_btn.clicked.connect(self.override_selected_master_flat)
        override_layout.addWidget(self.override_flat_btn)

        layout.addLayout(override_layout)

        # Calibrate Lights Button
        self.calibrate_lights_btn = QPushButton("🚀 Calibrate Light Frames 🚀")
        self.calibrate_lights_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF4500;
                color: white;
                font-size: 16px;
                padding: 8px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #FF6347;
            }
        """)
        self.calibrate_lights_btn.clicked.connect(self.calibrate_lights)
        layout.addWidget(self.calibrate_lights_btn)

        # Enable Context Menu
        self.light_tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.light_tree.customContextMenuRequested.connect(self.light_tree_context_menu)

        return tab



    def prompt_set_session(self, item, frame_type):
        text, ok = QInputDialog.getText(self, "Set Session Tag", "Enter session name:")
        if not (ok and text.strip()):
            return

        session_name = text.strip()
        is_flat = frame_type.upper() == "FLAT"
        tree = self.flat_tree if is_flat else self.light_tree
        target_dict = self.flat_files if is_flat else self.light_files

        selected_items = tree.selectedItems()

        def update_file_session(filename, widget_item):
            for key in list(target_dict.keys()):
                if isinstance(key, tuple) and len(key) == 2:
                    group_key, old_session = key
                else:
                    continue  # Skip malformed keys

                files = target_dict.get(key, [])
                for f in list(files):
                    if os.path.basename(f) == filename:
                        if old_session != session_name:
                            new_key = (group_key, session_name)
                            if new_key not in target_dict:
                                target_dict[new_key] = []
                            target_dict[new_key].append(f)
                            target_dict[key].remove(f)
                            if not target_dict[key]:
                                del target_dict[key]

                        # Update internal session tag
                        self.session_tags[f] = session_name

                        # Update leaf's metadata column
                        old_meta = widget_item.text(1)
                        if "Session:" in old_meta:
                            new_meta = re.sub(r"Session: [^|]*", f"Session: {session_name}", old_meta)
                        else:
                            new_meta = f"{old_meta} | Session: {session_name}"
                        widget_item.setText(1, new_meta)
                        return

        def recurse_all_leaf_items(parent_item):
            for i in range(parent_item.childCount()):
                child = parent_item.child(i)
                if child.childCount() == 0:
                    update_file_session(child.text(0), child)
                else:
                    recurse_all_leaf_items(child)

        # Case 1: Multi-leaf selection (e.g. Shift/Ctrl-click)
        if selected_items and any(i.childCount() == 0 for i in selected_items):
            for leaf in selected_items:
                if leaf.childCount() == 0:
                    update_file_session(leaf.text(0), leaf)

        # Case 2: Right-clicked on a group (e.g. filter+exposure node)
        elif item and item.childCount() > 0:
            recurse_all_leaf_items(item)

        # ✅ Reassign matching master flats/darks per leaf
        self.assign_best_master_files()

    def _quad_coverage_add(self, cov: np.ndarray, quad: np.ndarray):
        """
        Rasterize a convex quad (4x2 float array of (x,y) in aligned coords) into 'cov' by +1 filling.
        Bounds/clipping are handled. Small, robust scanline fill.
        """
        H, W = cov.shape
        pts = quad.astype(np.float32)

        ymin = max(int(np.floor(np.min(pts[:,1]))), 0)
        ymax = min(int(np.ceil (np.max(pts[:,1]))), H-1)
        if ymin > ymax: return

        # Edges (x0,y0)->(x1,y1), 4 of them
        edges = []
        for i in range(4):
            x0, y0 = pts[i]
            x1, y1 = pts[(i+1) % 4]
            edges.append((x0, y0, x1, y1))

        for y in range(ymin, ymax+1):
            xs = []
            yf = float(y) + 0.5  # sample at pixel center
            for (x0, y0, x1, y1) in edges:
                # Skip horizontal edges
                if (y0 <= yf < y1) or (y1 <= yf < y0):
                    # Linear interpolate X at scanline yf
                    t = (yf - y0) / (y1 - y0)
                    xs.append(x0 + t * (x1 - x0))

            if len(xs) < 2:
                continue
            xs.sort()
            # Fill between pairs
            for i in range(0, len(xs), 2):
                xL = int(np.floor(min(xs[i], xs[i+1])))
                xR = int(np.ceil (max(xs[i], xs[i+1])))
                if xR < 0 or xL > W-1: 
                    continue
                xL = max(xL, 0); xR = min(xR, W)
                if xR > xL:
                    cov[y, xL:xR] += 1


    def _max_rectangle_in_binary(self, mask: np.ndarray):
        """
        Largest axis-aligned rectangle of 1s in a binary mask (H×W, dtype=bool).
        Returns (x0, y0, x1, y1) where x1,y1 are exclusive, or None if empty.
        O(H*W) using 'largest rectangle in histogram' per row.
        """
        H, W = mask.shape
        heights = np.zeros(W, dtype=np.int32)
        best = (0, 0, 0, 0, 0)  # (area, x0, y0, x1, y1)

        for y in range(H):
            row = mask[y]
            heights[row] += 1
            heights[~row] = 0

            # Largest rectangle in histogram 'heights'
            stack = []
            i = 0
            while i <= W:
                h = heights[i] if i < W else 0
                if not stack or h >= heights[stack[-1]]:
                    stack.append(i); i += 1
                else:
                    top = stack.pop()
                    height = heights[top]
                    left = stack[-1] + 1 if stack else 0
                    right = i
                    area = height * (right - left)
                    if area > best[0]:
                        # rectangle spans rows [y-height+1 .. y], columns [left .. right-1]
                        y0 = y - height + 1
                        y1 = y + 1
                        best = (area, left, y0, right, y1)

        if best[0] == 0:
            return None
        _, x0, y0, x1, y1 = best
        return (x0, y0, x1, y1)

    def _compute_common_autocrop_rect(self, grouped_files: dict, coverage_pct: float, status_cb=None):
        """
        Fast global crop using alignment transforms only.

        - If coverage_pct >= 99: exact convex intersection of all transformed rectangles (no black edges).
        - If coverage_pct < 99: fast percentile-based AABB approximation (very quick; good in practice).

        Returns (x0, y0, x1, y1) in *aligned/reference* pixel coords, or None if not computable.
        """
        import math
        import json
        from time import perf_counter
        import numpy as np
        from astropy.io import fits

        def log(msg: str):
            try:
                (status_cb or self.update_status)(msg)
            except Exception:
                pass

        t0 = perf_counter()
        transforms_path = os.path.join(self.stacking_directory, "alignment_transforms.sasd")

        # ---- Helpers -----------------------------------------------------------
        def _load_transforms(path: str):
            """
            Returns dict {normalized_path: 3x3 float64 affine matrix}
            Tries your existing loader if present; otherwise tries JSON / pickle / npy/npz.
            """
            # Prefer your existing method if available
            if hasattr(self, "load_alignment_matrices_sasd"):
                try:
                    d = self.load_alignment_matrices_sasd(path)
                    # normalize into numpy arrays
                    out = {}
                    for k, v in (d or {}).items():
                        M = np.asarray(v, dtype=np.float64)
                        if M.shape == (3, 3):
                            out[os.path.normpath(k)] = M
                    return out
                except Exception:
                    pass

            # Fallbacks
            try:
                with open(path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                out = {}
                for k, v in raw.items():
                    M = np.asarray(v, dtype=np.float64)
                    if M.shape == (3, 3):
                        out[os.path.normpath(k)] = M
                return out
            except Exception:
                pass

            try:
                import pickle
                with open(path, "rb") as f:
                    raw = pickle.load(f)
                out = {}
                for k, v in raw.items():
                    M = np.asarray(v, dtype=np.float64)
                    if M.shape == (3, 3):
                        out[os.path.normpath(k)] = M
                return out
            except Exception:
                pass

            try:
                npz = np.load(path, allow_pickle=True)
                out = {}
                # support either dict-like npz or (keys, mats)
                if hasattr(npz, "files") and "keys" in npz.files and "mats" in npz.files:
                    keys = npz["keys"]
                    mats = npz["mats"]
                    for k, M in zip(keys, mats):
                        M = np.asarray(M, dtype=np.float64)
                        if M.shape == (3, 3):
                            out[os.path.normpath(str(k))] = M
                    return out
            except Exception:
                pass
            return {}

        def _infer_dims_from_any_aligned_file() -> tuple[int, int] | None:
            for _g, lst in grouped_files.items():
                for p in lst:
                    if os.path.exists(p):
                        try:
                            hdr = fits.getheader(p, ext=0)
                            W = int(hdr.get("NAXIS1") or 0)
                            H = int(hdr.get("NAXIS2") or 0)
                            if W > 0 and H > 0:
                                return (W, H)
                        except Exception:
                            pass
            return None

        def _rev_map_aligned_to_normalized() -> dict:
            """
            Map aligned path -> normalized path using either self.valid_transforms or filename pattern.
            """
            rev = {}
            vt = getattr(self, "valid_transforms", None)
            if isinstance(vt, dict) and vt:
                for norm_p, aligned_p in vt.items():
                    rev[os.path.normpath(aligned_p)] = os.path.normpath(norm_p)
            else:
                # best-effort: derive from naming convention
                for _g, lst in grouped_files.items():
                    for aligned in lst:
                        base = os.path.basename(aligned)
                        if base.endswith("_n_r.fit"):
                            nn = base.replace("_n_r.fit", "_n.fit")
                        elif base.endswith("_r.fit"):
                            nn = base.replace("_r.fit", "_n.fit")
                        else:
                            continue
                        rev[os.path.normpath(aligned)] = os.path.normpath(
                            os.path.join(self.stacking_directory, "Normalized_Images", nn)
                        )
            return rev

        def _transform_rect(M: np.ndarray, W: int, H: int):
            # Apply 3x3 affine to the 4 corners of the source image (0,0)-(W,H)
            corners = np.array([[0, 0, 1],
                                [W, 0, 1],
                                [W, H, 1],
                                [0, H, 1]], dtype=np.float64)
            tp = (M @ corners.T).T
            # assume affine (no perspective), ignore homogeneous w if present
            return [(float(tp[i, 0]), float(tp[i, 1])) for i in range(4)]

        # Sutherland–Hodgman convex polygon intersection
        def _poly_intersection(subject, clip):
            def _inside(p, a, b):
                # keep left of directed edge a->b
                return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= 0.0
            def _intersection(a1, a2, b1, b2):
                # line-line intersection
                x1,y1 = a1; x2,y2 = a2; x3,y3 = b1; x4,y4 = b2
                den = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
                if abs(den) < 1e-12:
                    return a2  # parallel; fallback
                px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / den
                py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / den
                return (px, py)

            output = subject
            if not output:
                return []
            for i in range(len(clip)):
                input_list = output
                output = []
                A = clip[i]
                B = clip[(i+1) % len(clip)]
                if not input_list:
                    break
                S = input_list[-1]
                for E in input_list:
                    if _inside(E, A, B):
                        if not _inside(S, A, B):
                            output.append(_intersection(S, E, A, B))
                        output.append(E)
                    elif _inside(S, A, B):
                        output.append(_intersection(S, E, A, B))
                    S = E
                if not output:
                    break
            return output

        # ---- Load transforms + dims -------------------------------------------
        log("✂️ Auto-crop: loading transforms…")
        transforms = _load_transforms(transforms_path)
        if not transforms:
            log("✂️ Auto-crop: no transforms found → disabling global crop.")
            return None

        dims = _infer_dims_from_any_aligned_file()
        if not dims:
            log("✂️ Auto-crop: could not infer image dimensions → disabling global crop.")
            return None
        W, H = dims

        rev_map = _rev_map_aligned_to_normalized()

        # ---- Build per-frame polygons (transformed corners on the reference grid) ----
        polys = []
        n_frames = 0
        for _g, flist in grouped_files.items():
            for aligned_path in flist:
                n_frames += 1
                norm_p = rev_map.get(os.path.normpath(aligned_path))
                if not norm_p:
                    continue
                M = transforms.get(os.path.normpath(norm_p))
                if M is None or np.asarray(M).shape != (3, 3):
                    continue
                polys.append(_transform_rect(np.asarray(M, dtype=np.float64), W, H))

        if not polys:
            log("✂️ Auto-crop: no usable polygons from transforms → disabling global crop.")
            return None

        # ---- Exact 100% coverage (convex intersection) ----
        base_rect = [(0.0, 0.0), (W*1.0, 0.0), (W*1.0, H*1.0), (0.0, H*1.0)]
        inter_poly = base_rect
        for p in polys:
            inter_poly = _poly_intersection(inter_poly, p)
            if len(inter_poly) < 3:
                inter_poly = []
                break

        # If user wants ≈100% coverage, prefer exact polygon intersection (fast & safe).
        if coverage_pct >= 99 or not polys:
            if not inter_poly:
                log("✂️ Auto-crop: empty intersection at 100% coverage.")
                return None
            xs = [pt[0] for pt in inter_poly]
            ys = [pt[1] for pt in inter_poly]
            x0 = max(0, int(math.floor(min(xs))))
            y0 = max(0, int(math.floor(min(ys))))
            x1 = min(W, int(math.ceil(max(xs))))
            y1 = min(H, int(math.ceil(max(ys))))
            if x1 - x0 <= 4 or y1 - y0 <= 4:
                log("✂️ Auto-crop: intersection too small to be useful.")
                return None
            log(f"✂️ Global crop (100% coverage): {x0},{y0} → {x1},{y1}  "
                f"({x1-x0}×{y1-y0})  in {perf_counter()-t0:.1f}s")
            return (x0, y0, x1, y1)

        # ---- Fast percentile AABB fallback for coverage_pct < 99 ---------------
        # This is a conservative, very fast approximation (no per-pixel masks).
        xmins = np.array([min(x for x, _ in p) for p in polys], dtype=np.float64)
        xmaxs = np.array([max(x for x, _ in p) for p in polys], dtype=np.float64)
        ymins = np.array([min(y for _, y in p) for p in polys], dtype=np.float64)
        ymaxs = np.array([max(y for _, y in p) for p in polys], dtype=np.float64)

        k = max(1, int(math.ceil((coverage_pct / 100.0) * len(polys))))
        # k-th largest for mins, k-th smallest for maxes
        # (np.partition is O(n) and fast)
        x0 = float(np.partition(xmins, -k)[-k])   # k-th largest left boundary
        y0 = float(np.partition(ymins, -k)[-k])
        x1 = float(np.partition(xmaxs,  k-1)[k-1])  # k-th smallest right boundary
        y1 = float(np.partition(ymaxs,  k-1)[k-1])

        x0 = max(0, int(math.floor(x0)))
        y0 = max(0, int(math.floor(y0)))
        x1 = min(W, int(math.ceil(x1)))
        y1 = min(H, int(math.ceil(y1)))

        if x1 - x0 <= 4 or y1 - y0 <= 4 or x0 >= x1 or y0 >= y1:
            log("✂️ Auto-crop: percentile AABB produced too small/invalid rect → disabling global crop.")
            return None

        log(f"✂️ Global crop (~{coverage_pct:.0f}% coverage): {x0},{y0} → {x1},{y1}  "
            f"({x1-x0}×{y1-y0})  in {perf_counter()-t0:.1f}s")
        return (x0, y0, x1, y1)


    def _first_non_none(self, *vals):
        for v in vals:
            if v is not None:
                return v
        return None

    def _compute_coverage_mask(self, file_list: List[str], transforms_path: str, coverage_pct: float):
        """
        Build a coverage-count image on the aligned canvas for 'file_list'.
        Threshold at coverage_pct, but use the number of frames we ACTUALLY rasterized (N_eff).
        Returns a bool mask (H×W) or None if nothing rasterized.
        """
        if not file_list:
            return None

        # Canvas from first aligned image
        ref_img, _, _, _ = load_image(file_list[0])
        if ref_img is None:
            self.update_status("✂️ Auto-crop: could not load first aligned ref.")
            return None
        H, W = (ref_img.shape if ref_img.ndim == 2 else ref_img.shape[:2])

        if not os.path.exists(transforms_path):
            self.update_status(f"✂️ Auto-crop: no transforms file at {transforms_path}")
            return None

        transforms = self.load_alignment_matrices_custom(transforms_path)

        # --- Robust transform lookup: key by normalized full path AND by basename ---
        def _normcase(p):  # windows-insensitive
            p = os.path.normpath(os.path.abspath(p))
            return p.lower() if os.name == "nt" else p

        xforms_by_full = { _normcase(k): v for k, v in transforms.items() }
        xforms_by_name = {}
        for k, v in transforms.items():
            xforms_by_name.setdefault(os.path.basename(k), v)

        cov = np.zeros((H, W), dtype=np.uint16)
        used = 0

        for aligned_path in file_list:
            base = os.path.basename(aligned_path)
            if base.endswith("_n_r.fit"):
                raw_base = base.replace("_n_r.fit", "_n.fit")
            elif base.endswith("_r.fit"):
                raw_base = base.replace("_r.fit", ".fit")
            else:
                raw_base = base

            # try normalized-Images location first
            raw_path_guess = os.path.join(self.stacking_directory, "Normalized_Images", raw_base)

            # find transform
            M = self._first_non_none(
                xforms_by_full.get(_normcase(raw_path_guess)),
                xforms_by_full.get(_normcase(aligned_path)),
                transforms.get(raw_path_guess),
                transforms.get(os.path.normpath(aligned_path)),
                xforms_by_name.get(raw_base),
            )

            if M is None:
                # Can't rasterize this frame
                continue

            # raw size
            h_raw = w_raw = None
            if os.path.exists(raw_path_guess):
                raw_img, _, _, _ = load_image(raw_path_guess)
                if raw_img is not None:
                    h_raw, w_raw = (raw_img.shape if raw_img.ndim == 2 else raw_img.shape[:2])

            if h_raw is None or w_raw is None:
                # fallback to aligned canvas size (still okay; affine provides placement)
                h_raw, w_raw = H, W

            corners = np.array([[0,0],[w_raw-1,0],[w_raw-1,h_raw-1],[0,h_raw-1]], dtype=np.float32)
            A = M[:, :2]; t = M[:, 2]
            quad = (corners @ A.T) + t

            self._quad_coverage_add(cov, quad)
            used += 1

        if used == 0:
            self.update_status("✂️ Auto-crop: 0/{} frames had usable transforms; skipping.".format(len(file_list)))
            return None

        need = int(np.ceil((coverage_pct / 100.0) * used))
        mask = (cov >= need)
        self.update_status(f"✂️ Auto-crop: rasterized {used}/{len(file_list)} frames; need {need} per-pixel.")
        if not mask.any():
            self.update_status("✂️ Auto-crop: threshold produced empty mask.")
            return None
        return mask



    def _compute_autocrop_rect(self, file_list: List[str], transforms_path: str, coverage_pct: float):
        """
        Build a coverage-count image (aligned canvas), threshold at pct, and extract largest rectangle.e
        Returns (x0, y0, x1, y1) or None.
        """
        if not file_list:
            return None

        # Load aligned reference to get canvas size
        ref_img, ref_hdr, _, _ = load_image(file_list[0])
        if ref_img is None:
            return None
        if ref_img.ndim == 2:
            H, W = ref_img.shape
        else:
            H, W = ref_img.shape[:2]

        # Load transforms (raw _n path -> 2x3 matrix mapping raw->aligned)
        if not os.path.exists(transforms_path):
            return None
        transforms = self.load_alignment_matrices_custom(transforms_path)

        # We need the raw (normalized) image size for each file to transform its corners
        # From aligned name "..._n_r.fit" get raw name "..._n.fit" (like in your drizzle code)
        cov = np.zeros((H, W), dtype=np.uint16)
        for aligned_path in file_list:
            base = os.path.basename(aligned_path)
            if base.endswith("_n_r.fit"):
                raw_base = base.replace("_n_r.fit", "_n.fit")
            elif base.endswith("_r.fit"):
                raw_base = base.replace("_r.fit", ".fit")  # fallback
            else:
                raw_base = base  # fallback

            raw_path = os.path.join(self.stacking_directory, "Normalized_Images", raw_base)
            # Fallback if normalized folder differs:
            raw_key = os.path.normpath(raw_path)
            M = transforms.get(raw_key, None)
            if M is None:
                # Try direct key (some pipelines use normalized path equal to aligned key)
                M = transforms.get(os.path.normpath(aligned_path), None)
            if M is None:
                continue

            # Determine raw size
            raw_img, _, _, _ = load_image(raw_key) if os.path.exists(raw_key) else (None, None, None, None)
            if raw_img is None:
                # last resort: assume same canvas; still yields a conservative crop
                h_raw, w_raw = H, W
            else:
                if raw_img.ndim == 2:
                    h_raw, w_raw = raw_img.shape
                else:
                    h_raw, w_raw = raw_img.shape[:2]

            # Transform raw rectangle corners into aligned coords
            corners = np.array([
                [0,       0      ],
                [w_raw-1, 0      ],
                [w_raw-1, h_raw-1],
                [0,       h_raw-1]
            ], dtype=np.float32)

            # Apply affine: [x' y']^T = A*[x y]^T + t
            A = M[:, :2]; t = M[:, 2]
            quad = (corners @ A.T) + t  # shape (4,2)

            # Rasterize into coverage
            self._quad_coverage_add(cov, quad)

        # Threshold at requested coverage
        N = len(file_list)
        need = int(np.ceil((coverage_pct / 100.0) * N))
        mask = (cov >= need)

        # Largest rectangle of 1s
        rect = self._max_rectangle_in_binary(mask)
        return rect


    def create_image_registration_tab(self):
        """
        Creates an Image Registration tab that mimics how the Light tab handles
        cosmetic corrections—i.e., we have global Drizzle controls (checkbox, combo, spin),
        and we update a text column in the QTreeWidget to show each group's drizzle state.
        """
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # ─────────────────────────────────────────
        # 1) QTreeWidget
        # ─────────────────────────────────────────
        self.reg_tree = QTreeWidget()
        self.reg_tree.setColumnCount(3)
        self.reg_tree.setHeaderLabels([
            "Filter - Exposure - Size",
            "Metadata",
            "Drizzle"  # We'll display "Drizzle: True, Scale: 2x, Drop:0.65" here
        ])
        self.reg_tree.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)

        # Optional: make columns resize nicely
        header = self.reg_tree.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)

        layout.addWidget(QLabel("Calibrated Light Frames"))
        layout.addWidget(self.reg_tree)

        tol_layout = QHBoxLayout()
        tol_layout.addWidget(QLabel("Exposure Tolerance (sec):"))
        self.exposure_tolerance_spin = QSpinBox()
        self.exposure_tolerance_spin.setRange(0, 900)
        self.exposure_tolerance_spin.setValue(0)
        self.exposure_tolerance_spin.setSingleStep(5)
        tol_layout.addWidget(self.exposure_tolerance_spin)
        tol_layout.addStretch()
        self.split_dualband_cb = QCheckBox("Split dual-band OSC before integration")
        self.split_dualband_cb.setToolTip("For OSC dual-band data: SII/OIII → R=SII, G=OIII; Ha/OIII → R=Ha, G=OIII")
        tol_layout.addWidget(self.split_dualband_cb)
        layout.addLayout(tol_layout)

        self.exposure_tolerance_spin.valueChanged.connect(lambda _: self.populate_calibrated_lights())



        # ─────────────────────────────────────────
        # 2) Buttons for Managing Files
        # ─────────────────────────────────────────
        btn_layout = QHBoxLayout()
        self.add_reg_files_btn = QPushButton("Add Light Files")
        self.add_reg_files_btn.clicked.connect(self.add_light_files_to_registration)
        btn_layout.addWidget(self.add_reg_files_btn)

        self.clear_selection_btn = QPushButton("Remove Selected")
        self.clear_selection_btn.clicked.connect(lambda: self.clear_tree_selection_registration(self.reg_tree))

        btn_layout.addWidget(self.clear_selection_btn)

        layout.addLayout(btn_layout)

        # ─────────────────────────────────────────
        # 3) Global Drizzle Controls
        # ─────────────────────────────────────────
        drizzle_layout = QHBoxLayout()

        self.drizzle_checkbox = QCheckBox("Enable Drizzle (beta)")
        self.drizzle_checkbox.toggled.connect(self._on_drizzle_checkbox_toggled) # <─ connect signal
        drizzle_layout.addWidget(self.drizzle_checkbox)

        drizzle_layout.addWidget(QLabel("Scale:"))
        self.drizzle_scale_combo = QComboBox()
        self.drizzle_scale_combo.addItems(["1x", "2x", "3x"])
        self.drizzle_scale_combo.currentIndexChanged.connect(self._on_drizzle_param_changed)  # <─ connect
        drizzle_layout.addWidget(self.drizzle_scale_combo)

        drizzle_layout.addWidget(QLabel("Drop Shrink:"))
        self.drizzle_drop_shrink_spin = QDoubleSpinBox()
        self.drizzle_drop_shrink_spin.setRange(0.0, 1.0)
        self.drizzle_drop_shrink_spin.setSingleStep(0.05)
        self.drizzle_drop_shrink_spin.setValue(0.65)
        self.drizzle_drop_shrink_spin.valueChanged.connect(self._on_drizzle_param_changed)  # <─ connect
        drizzle_layout.addWidget(self.drizzle_drop_shrink_spin)

        layout.addLayout(drizzle_layout)

        # ─────────────────────────────────────────
        # 4) Reference Frame Selection
        # ─────────────────────────────────────────
        self.ref_frame_label = QLabel("Select Reference Frame:")
        self.ref_frame_path = QLabel("No file selected")
        self.ref_frame_path.setWordWrap(True)
        self.select_ref_frame_btn = QPushButton("Select Reference Frame")
        self.select_ref_frame_btn.clicked.connect(self.select_reference_frame)

        ref_layout = QHBoxLayout()
        ref_layout.addWidget(self.ref_frame_label)
        ref_layout.addWidget(self.ref_frame_path)
        ref_layout.addWidget(self.select_ref_frame_btn)
        layout.addLayout(ref_layout)

        crop_row = QHBoxLayout()
        self.autocrop_cb = QCheckBox("Auto-crop output")
        self.autocrop_cb.setToolTip("Crop the final image to pixels covered by ≥ Coverage % of frames")
        self.autocrop_pct = QDoubleSpinBox()
        self.autocrop_pct.setRange(50.0, 100.0)
        self.autocrop_pct.setSingleStep(1.0)
        self.autocrop_pct.setSuffix(" %")
        self.autocrop_pct.setValue(self.settings.value("stacking/autocrop_pct", 95.0, type=float))
        self.autocrop_cb.setChecked(self.settings.value("stacking/autocrop_enabled", True, type=bool))
        crop_row.addWidget(self.autocrop_cb)
        crop_row.addWidget(QLabel("Coverage:"))
        crop_row.addWidget(self.autocrop_pct)
        crop_row.addStretch(1)
        layout.addLayout(crop_row)

        # ★★ Star-Trail Mode ★★
        trail_layout = QHBoxLayout()
        self.trail_cb = QCheckBox("★★ Star-Trail Mode ★★ (Max-Value Stack)")
        self.trail_cb.setChecked(self.star_trail_mode)
        self.trail_cb.setToolTip(
            "Skip registration/alignment and use Maximum-Intensity projection for star trails"
        )
        self.trail_cb.stateChanged.connect(self._on_star_trail_toggled)
        trail_layout.addWidget(self.trail_cb)
        layout.addLayout(trail_layout)
        # ─────────────────────────────────────────
        # 5) Register & Integrate Buttons
        # ─────────────────────────────────────────

        self.register_images_btn = QPushButton("🔥🚀Register and Integrate Images🔥🚀")
        self.register_images_btn.clicked.connect(self.register_images)
        self.register_images_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF4500;
                color: white;
                font-size: 16px;
                padding: 8px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #FF6347;
            }
        """)
        layout.addWidget(self.register_images_btn)

        self.integrate_registered_btn = QPushButton("Integrate Previously Registered Images")
        self.integrate_registered_btn.clicked.connect(self.integrate_registered_images)
        self.integrate_registered_btn.setStyleSheet("""
            QPushButton {
                background-color: #333;
                color: white;
                font-size: 14px;
                padding: 8px;
                border-radius: 5px;
                border: 2px solid yellow;
            }
            QPushButton:hover {
                border: 2px solid #FFD700;
            }
            QPushButton:pressed {
                background-color: #222;
                border: 2px solid #FFA500;
            }
        """)
        layout.addWidget(self.integrate_registered_btn)

        # Populate the tree from your calibrated folder
        self.populate_calibrated_lights()
        tab.setLayout(layout)
        return tab

    def _on_star_trail_toggled(self, state):
        self.star_trail_mode = bool(state)
        self.settings.setValue("stacking/star_trail_mode", self.star_trail_mode)
        # if they turn it on, immediately override the rejection combo:
        if self.star_trail_mode:
            self.rejection_algorithm = "Maximum Value"
        else:
            # reload whatever the user picked
            self.rejection_algorithm = self.settings.value("stacking/rejection_algorithm",
                                                          self.rejection_algorithm,
                                                          type=str)

    def select_reference_frame(self):
        """ Opens a file dialog to select the reference frame. """
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Reference Frame", "", 
                                                "FITS Images (*.fits *.fit);;All Files (*)")
        if file_path:
            self.reference_frame = file_path
            self.ref_frame_path.setText(os.path.basename(file_path))

    def save_master_paths_to_settings(self):
        """Save current master dark and flat paths to QSettings using their actual trees."""

        # Master Darks
        dark_paths = []
        for i in range(self.master_dark_tree.topLevelItemCount()):
            group = self.master_dark_tree.topLevelItem(i)
            for j in range(group.childCount()):
                fname = group.child(j).text(0)
                for path in self.master_files.values():
                    if os.path.basename(path) == fname:
                        dark_paths.append(path)

        # Master Flats
        flat_paths = []
        for i in range(self.master_flat_tree.topLevelItemCount()):
            group = self.master_flat_tree.topLevelItem(i)
            for j in range(group.childCount()):
                fname = group.child(j).text(0)
                for path in self.master_files.values():
                    if os.path.basename(path) == fname:
                        flat_paths.append(path)

        self.settings.setValue("stacking/master_darks", dark_paths)
        self.settings.setValue("stacking/master_flats", flat_paths)

    def clear_tree_selection(self, tree, file_dict):
        """Clears selected items from a simple (non-tuple-keyed) tree like Master Darks or Darks tab."""
        selected_items = tree.selectedItems()
        if not selected_items:
            return

        for item in selected_items:
            parent = item.parent()
            if parent is None:
                # Top-level group item
                key = item.text(0)
                if key in file_dict:
                    del file_dict[key]
                tree.takeTopLevelItem(tree.indexOfTopLevelItem(item))
            else:
                # Child item
                key = parent.text(0)
                filename = item.text(0)
                if key in file_dict:
                    file_dict[key] = [f for f in file_dict[key] if os.path.basename(f) != filename]
                    if not file_dict[key]:
                        del file_dict[key]
                parent.removeChild(item)


    def clear_tree_selection_light(self, tree):
        """Clears the selection in the light tree and updates self.light_files accordingly."""
        selected_items = tree.selectedItems()
        if not selected_items:
            return

        for item in selected_items:
            parent = item.parent()
            if parent is None:
                # Top-level filter node selected
                filter_name = item.text(0)
                # Remove all composite keys whose group_key starts with filter_name
                keys_to_remove = [key for key in list(self.light_files.keys())
                                if isinstance(key, tuple) and key[0].startswith(f"{filter_name} - ")]
                for key in keys_to_remove:
                    del self.light_files[key]
                tree.takeTopLevelItem(tree.indexOfTopLevelItem(item))
            else:
                if parent.parent() is None:
                    # Exposure node selected (child)
                    filter_name = parent.text(0)
                    exposure_text = item.text(0)
                    group_key = f"{filter_name} - {exposure_text}"
                    keys_to_remove = [key for key in list(self.light_files.keys())
                                    if isinstance(key, tuple) and key[0] == group_key]
                    for key in keys_to_remove:
                        del self.light_files[key]
                    parent.removeChild(item)
                else:
                    # Grandchild file node selected
                    filter_name = parent.parent().text(0)
                    exposure_text = parent.text(0)
                    group_key = f"{filter_name} - {exposure_text}"
                    filename = item.text(0)

                    keys_to_check = [key for key in list(self.light_files.keys())
                                    if isinstance(key, tuple) and key[0] == group_key]

                    for key in keys_to_check:
                        self.light_files[key] = [
                            f for f in self.light_files[key] if os.path.basename(f) != filename
                        ]
                        if not self.light_files[key]:
                            del self.light_files[key]
                    parent.removeChild(item)

    def clear_tree_selection_flat(self, tree, file_dict):
        """Clears the selection in the given tree widget and removes items from the corresponding dictionary."""
        selected_items = tree.selectedItems()
        if not selected_items:
            return

        for item in selected_items:
            parent = item.parent()

            if parent:
                # Grandchild level (actual file)
                if parent.parent() is not None:
                    filter_name = parent.parent().text(0)
                    exposure_text = parent.text(0)
                    group_key = f"{filter_name} - {exposure_text}"
                else:
                    # Exposure level
                    filter_name = parent.text(0)
                    exposure_text = item.text(0)
                    group_key = f"{filter_name} - {exposure_text}"

                filename = item.text(0)

                # Remove from all matching (group_key, session) tuples
                keys_to_check = [key for key in list(file_dict.keys())
                                if isinstance(key, tuple) and key[0] == group_key]

                for key in keys_to_check:
                    file_dict[key] = [f for f in file_dict[key] if os.path.basename(f) != filename]
                    if not file_dict[key]:
                        del file_dict[key]

                parent.removeChild(item)
            else:
                # Top-level (filter group) selected
                filter_name = item.text(0)
                keys_to_remove = [key for key in list(file_dict.keys())
                                if isinstance(key, tuple) and key[0].startswith(f"{filter_name} - ")]
                for key in keys_to_remove:
                    del file_dict[key]
                tree.takeTopLevelItem(tree.indexOfTopLevelItem(item))

    def _sync_group_userrole(self, top_item: QTreeWidgetItem):
        paths = []
        for i in range(top_item.childCount()):
            child = top_item.child(i)
            fp = child.data(0, Qt.ItemDataRole.UserRole)
            if fp:
                paths.append(fp)
        top_item.setData(0, Qt.ItemDataRole.UserRole, paths)

    def clear_tree_selection_registration(self, tree):
        selected_items = tree.selectedItems()
        if not selected_items:
            return

        for item in selected_items:
            parent = item.parent()

            if parent is None:
                # Remove entire group
                group_key = item.text(0)
                # Track deleted files (optional)
                full_paths = item.data(0, Qt.ItemDataRole.UserRole) or []
                self.deleted_calibrated_files.extend(p for p in full_paths
                                                    if p not in self.deleted_calibrated_files)
                # Remove from dict + tree
                self.reg_files.pop(group_key, None)
                tree.takeTopLevelItem(tree.indexOfTopLevelItem(item))

            else:
                # Remove a single child
                group_key = parent.text(0)
                filename = item.text(0)

                if group_key in self.reg_files:
                    self.reg_files[group_key] = [
                        f for f in self.reg_files[group_key]
                        if os.path.basename(f) != filename
                    ]
                    if not self.reg_files[group_key]:
                        del self.reg_files[group_key]

                # Track deleted path (optional)
                fp = item.data(0, Qt.ItemDataRole.UserRole)
                if fp and fp not in self.deleted_calibrated_files:
                    self.deleted_calibrated_files.append(fp)

                # Remove child from tree
                parent.removeChild(item)

                # 🔑 keep parent’s stored list in sync
                self._sync_group_userrole(parent)

    def rebuild_flat_tree(self):
        """Regroup flat frames in the flat_tree based on the exposure tolerance."""
        self.flat_tree.clear()

        if not self.flat_files:
            return

        tolerance = self.flat_exposure_tolerance_spinbox.value()

        # Flatten all flats into a list
        all_flats = []
        for (filter_exp_size, session_tag), files in self.flat_files.items():
            for file in files:
                all_flats.append((filter_exp_size, session_tag, file))

        # Group the flats
        grouped = {}

        for (filter_exp_size, session_tag, file_path) in all_flats:
            try:
                header = fits.getheader(file_path, ext=0)
                filter_name = header.get("FILTER", "Unknown")
                filter_name     = self._sanitize_name(filter_name)
                exposure = header.get("EXPOSURE", header.get("EXPTIME", "Unknown"))
                width = header.get("NAXIS1", 0)
                height = header.get("NAXIS2", 0)
                image_size = f"{width}x{height}" if width and height else "Unknown"
                exposure = float(exposure)

                found_group = None
                for group_key in grouped.keys():
                    g_filter, g_min_exp, g_max_exp, g_size = group_key
                    if (
                        filter_name == g_filter and
                        image_size == g_size and
                        g_min_exp - tolerance <= exposure <= g_max_exp + tolerance
                    ):
                        found_group = group_key
                        break

                if found_group:
                    grouped[found_group].append((file_path, exposure))
                else:
                    new_key = (filter_name, exposure, exposure, image_size)
                    grouped[new_key] = [(file_path, exposure)]

            except Exception as e:
                print(f"⚠️ Failed reading {file_path}: {e}")

        # Now create the tree
        for (filter_name, min_exp, max_exp, image_size), files in grouped.items():
            top_item = QTreeWidgetItem()
            expmin = np.floor(min_exp)
            tolerance = self.flat_exposure_tolerance_spinbox.value()

            if len(files) > 1:
                exposure_str = f"{expmin:.1f}s–{(expmin + tolerance):.1f}s"
            else:
                exposure_str = f"{min_exp:.1f}s"

            top_item.setText(0, f"{filter_name} - {exposure_str} ({image_size})")
            top_item.setText(1, f"{len(files)} files")
            top_item.setText(2, "Auto-Selected Dark" if self.auto_select_dark_checkbox.isChecked() else "None")

            self.flat_tree.addTopLevelItem(top_item)

            for file_path, _ in files:
                session_tag = self.session_tags.get(file_path, "Default")
                leaf_item = QTreeWidgetItem([
                    os.path.basename(file_path),
                    f"Size: {image_size} | Session: {session_tag}"
                ])
                top_item.addChild(leaf_item)


    def exposures_within_tolerance(self, exp1, exp2, tolerance):
        try:
            return abs(float(exp1) - float(exp2)) <= tolerance
            
        except Exception:
            return False

    def parse_group_key(self, group_key):
        """
        Parses a group key string like 'Luminance - 90s (3000x2000)'
        into filter_name, exposure (float), and image_size (str).
        """
        try:
            parts = group_key.split(' - ')
            filter_name = parts[0]
            exp_size_part = parts[1] if len(parts) > 1 else ""

            # Separate exposure and size correctly
            if '(' in exp_size_part and ')' in exp_size_part:
                exposure_str, size_part = exp_size_part.split('(', 1)
                exposure = exposure_str.replace('s', '').strip()
                size = size_part.strip(') ').strip()
            else:
                exposure = exp_size_part.replace('s', '').strip()
                size = "Unknown"

            
            return filter_name, float(exposure), size

        except Exception as e:
            
            return "Unknown", 0.0, "Unknown"

    def _get_image_size(self, fp):
        ext = os.path.splitext(fp)[1].lower()
        # first try FITS
        if ext in (".fits", ".fit"):
            hdr0 = fits.getheader(fp, ext=0)
            data0 = fits.getdata(fp, ext=0)
            h, w = data0.shape[-2:]
        else:
            # try Pillow
            try:
                with Image.open(fp) as img:
                    w, h = img.size
            except Exception:
                # Pillow failed on TIFF or exotic format → try tifffile
                try:
                    arr = tiff.imread(fp)
                    h, w = arr.shape[:2]
                except Exception:
                    # last resort: OpenCV
                    arr = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
                    if arr is None:
                        raise IOError(f"Cannot read image size for {fp}")
                    h, w = arr.shape[:2]
        return w, h


    def populate_calibrated_lights(self):
        """
        Reads both the Calibrated folder and any manually-added files,
        groups them by FILTER, EXPOSURE±tol, SIZE, and fills self.reg_tree.
        Also sets each group's Drizzle column from saved per-group state,
        or from the current global controls if none exists yet.
        """
        from PIL import Image

        # Fallback in case helper wasn't added
        def _fmt(enabled, scale, drop):
            return (f"Drizzle: True, Scale: {scale:g}x, Drop: {drop:.2f}"
                    if enabled else "Drizzle: False")

        # 1) clear tree
        self.reg_tree.clear()
        self.reg_tree.setColumnCount(3)
        self.reg_tree.setHeaderLabels(["Filter - Exposure - Size", "Metadata", "Drizzle"])
        hdr = self.reg_tree.header()
        for col in (0, 1, 2):
            hdr.setSectionResizeMode(col, QHeaderView.ResizeMode.Interactive)

        # 2) gather files
        calibrated_folder = os.path.join(self.stacking_directory or "", "Calibrated")
        files = []
        if os.path.isdir(calibrated_folder):
            for fn in os.listdir(calibrated_folder):
                files.append(os.path.join(calibrated_folder, fn))
        files += getattr(self, "manual_light_files", [])

        if not files:
            return

        # 3) group by header (or defaults)
        grouped = {}  # key -> list of dicts: {"path", "exp", "size"}
        tol = self.exposure_tolerance_spin.value()

        for fp in files:
            ext = os.path.splitext(fp)[1].lower()
            filt = "Unknown"
            exp = 0.0
            size = "Unknown"

            if ext in (".fits", ".fit"):
                try:
                    hdr0 = fits.getheader(fp, ext=0)
                    filt = self._sanitize_name(hdr0.get("FILTER", "Unknown"))
                    exp_raw = hdr0.get("EXPOSURE", hdr0.get("EXPTIME", None))
                    try:
                        exp = float(exp_raw)
                    except (TypeError, ValueError):
                        print(f"⚠️ Exposure invalid in {fp}, defaulting to 0.0s")
                        exp = 0.0
                    data0 = fits.getdata(fp, ext=0)
                    h, w = data0.shape[-2:]
                    size = f"{w}x{h}"
                except Exception as e:
                    print(f"⚠️ Could not read FITS {fp}: {e}; treating as generic image")

            if filt == "Unknown" and ext not in (".fits", ".fit"):
                # generic image via PIL/utility
                try:
                    w, h = self._get_image_size(fp)
                    size = f"{w}x{h}"
                except Exception as e:
                    print(f"⚠️ Cannot read image size for {fp}: {e}")
                    continue

            # find existing group
            match_key = None
            for key in grouped:
                f2, e2, s2 = self.parse_group_key(key)
                if filt == f2 and s2 == size and abs(exp - e2) <= tol:
                    match_key = key
                    break

            key = match_key or f"{filt} - {exp:.1f}s ({size})"
            grouped.setdefault(key, []).append({"path": fp, "exp": exp, "size": size})

        # 4) populate tree & self.light_files
        self.light_files = {}

        # read current global drizzle controls (used as default)
        global_enabled = self.drizzle_checkbox.isChecked()
        try:
            global_scale = float(self.drizzle_scale_combo.currentText().replace("x", "", 1))
        except Exception:
            global_scale = 1.0
        global_drop = self.drizzle_drop_shrink_spin.value()

        for key, entries in grouped.items():
            paths = [d["path"] for d in entries]
            exps  = [d["exp"]  for d in entries]

            top = QTreeWidgetItem()
            top.setText(0, key)
            if len(exps) > 1:
                mn, mx = min(exps), max(exps)
                top.setText(1, f"{len(paths)} files, {mn:.0f}s–{mx:.0f}s")
            else:
                top.setText(1, f"{len(paths)} file")

            # Use saved per-group drizzle state if present; else default to global controls
            state = self.per_group_drizzle.get(key)
            if state is None:
                state = {
                    "enabled": bool(global_enabled),
                    "scale":   float(global_scale),
                    "drop":    float(global_drop),
                }
                self.per_group_drizzle[key] = state  # persist default for this group

            # Show in column 2
            try:
                top.setText(2, self._format_drizzle_text(state["enabled"], state["scale"], state["drop"]))
            except AttributeError:
                top.setText(2, _fmt(state["enabled"], state["scale"], state["drop"]))

            top.setData(0, Qt.ItemDataRole.UserRole, paths)
            self.reg_tree.addTopLevelItem(top)

            # leaf rows: show basename + *per-file* size (fixes the old "same size for all leaves" issue)
            for d in entries:
                fp = d["path"]
                leaf = QTreeWidgetItem([os.path.basename(fp), f"Size: {d['size']}"])
                leaf.setData(0, Qt.ItemDataRole.UserRole, fp)
                top.addChild(leaf)

            top.setExpanded(True)
            self.light_files[key] = paths

    def _iter_group_items(self):
        for i in range(self.reg_tree.topLevelItemCount()):
            yield self.reg_tree.topLevelItem(i)

    def _format_drizzle_text(self, enabled: bool, scale: float, drop: float) -> str:
        return (f"Drizzle: True, Scale: {scale:g}x, Drop: {drop:.2f}"
                if enabled else "Drizzle: False")

    def _set_drizzle_on_items(self, items, enabled: bool, scale: float, drop: float):
        txt_on  = self._format_drizzle_text(True,  scale, drop)
        txt_off = self._format_drizzle_text(False, scale, drop)
        for it in items:
            # dedupe child selection → parent group
            if it.parent() is not None:
                it = it.parent()
            group_key = it.text(0)
            it.setText(2, txt_on if enabled else txt_off)
            self.per_group_drizzle[group_key] = {
                "enabled": bool(enabled),
                "scale": float(scale),
                "drop":  float(drop),
            }

    def update_drizzle_settings(self):
        """
        Called whenever the user toggles the 'Enable Drizzle' checkbox,
        changes the scale combo, or changes the drop shrink spinbox.
        Applies to all *selected* top-level items in the reg_tree.
        """
        # Current states from global controls
        drizzle_enabled = self.drizzle_checkbox.isChecked()
        scale_str = self.drizzle_scale_combo.currentText()  # e.g. "1x","2x","3x"
        drop_val = self.drizzle_drop_shrink_spin.value()    # e.g. 0.65

        # Gather selected items
        selected_items = self.reg_tree.selectedItems()
        if not selected_items:
            return

        for item in selected_items:
            # If the user selected a child row, go up to its parent group
            if item.parent() is not None:
                item = item.parent()

            group_key = item.text(0)

            if drizzle_enabled:
                # Show scale + drop shrink
                drizzle_text = (f"Drizzle: True, "
                                f"Scale: {scale_str}, "
                                f"Drop: {drop_val:.2f}")
            else:
                # Just show "Drizzle: False"
                drizzle_text = "Drizzle: False"

            # Update column 2 with the new text
            item.setText(2, drizzle_text)

            # If you also store it in a dictionary:
            self.per_group_drizzle[group_key] = {
                "enabled": drizzle_enabled,
                "scale": float(scale_str.replace("x","", 1)),
                "drop": drop_val
            }

    def _on_drizzle_checkbox_toggled(self, checked: bool):
        scale = float(self.drizzle_scale_combo.currentText().replace("x","",1))
        drop  = self.drizzle_drop_shrink_spin.value()
        targets = list(self._iter_group_items())  # ALWAYS all groups
        self._set_drizzle_on_items(targets, checked, scale, drop)

    def _on_drizzle_param_changed(self, *_):
        enabled = self.drizzle_checkbox.isChecked()
        scale   = float(self.drizzle_scale_combo.currentText().replace("x","",1))
        drop    = self.drizzle_drop_shrink_spin.value()

        sel = self.reg_tree.selectedItems()
        if sel:
            # update selected groups
            seen, targets = set(), []
            for it in sel:
                top = it if it.parent() is None else it.parent()
                key = top.text(0)
                if key not in seen:
                    seen.add(key); targets.append(top)
        else:
            # no selection → update ALL groups (keeps UI intuitive)
            targets = list(self._iter_group_items())

        self._set_drizzle_on_items(targets, enabled, scale, drop)

    def gather_drizzle_settings_from_tree(self):
        """
        Returns: { group_key: {files:[...], drizzle_enabled:bool,
                            scale_factor:float, drop_shrink:float} }
        """
        dd = {}
        for i in range(self.reg_tree.topLevelItemCount()):
            item = self.reg_tree.topLevelItem(i)
            key  = item.text(0)
            files= item.data(0, Qt.ItemDataRole.UserRole) or []
            txt  = item.text(2).lower()

            ena = txt.startswith("drizzle: true")
            sf  = 1.0
            ds  = 0.65
            if ena:
                m = re.search(r"scale\s*:\s*([\d\.]+)x?", txt)
                if m: sf = float(m.group(1))
                m = re.search(r"drop\s*:\s*([\d\.]+)", txt)
                if m: ds = float(m.group(1))

            dd[key] = {
                "files": files,
                "drizzle_enabled": ena,
                "scale_factor": sf,
                "drop_shrink": ds
            }

        # backfill any group that lived only in self.light_files
        for key, fl in self.light_files.items():
            if key not in dd:
                dd[key] = {
                    "files": fl,
                    "drizzle_enabled": False,
                    "scale_factor": 1.0,
                    "drop_shrink": 0.65
                }

        return dd



    def add_light_files_to_registration(self):
        """
        Let the user pick some new LIGHT frames, then
        immediately re-populate the tree so they show up
        in the same Filter–Exposure–Size groups as everything else.
        """
        last_dir = self.settings.value("last_opened_folder", "", type=str)
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Light Frames",
            last_dir,
            "FITS Files (*.fits *.fit *.fz *.fz *.xisf *.tif *.tiff *.png *.jpg *.jpeg)"
        )
        if not files:
            return

        # remember for next time
        self.settings.setValue("last_opened_folder", os.path.dirname(files[0]))

        # store these in a manual list, then rebuild the tree
        if not hasattr(self, "manual_light_files"):
            self.manual_light_files = []
        self.manual_light_files.extend(files)

        # rebuild the registration tree (it reads manual_light_files + calibrated folder)
        self.populate_calibrated_lights()





    def on_tab_changed(self, index):
        """ Detects when user switches to the Flats tab and triggers auto-assign. """
        if self.tabs.tabText(index) == "Flats":
            print("🔄 Auto-checking best Master Darks for Flats...")
            self.assign_best_master_dark()


    def add_dark_files(self):
        self.add_files(self.dark_tree, "Select Dark Files", "DARK")
    
    def add_dark_directory(self):
        self.add_directory(self.dark_tree, "Select Dark Directory", "DARK")

    def add_flat_files(self):
        self.prompt_session_before_adding("FLAT")


    def add_flat_directory(self):
        self.prompt_session_before_adding("FLAT", directory_mode=True)


    
    def add_light_files(self):
        self.prompt_session_before_adding("LIGHT")

    
    def add_light_directory(self):
        self.prompt_session_before_adding("LIGHT", directory_mode=True)


    def prompt_session_before_adding(self, frame_type, directory_mode=False):
        # 🔥 Prompt user first
        text, ok = QInputDialog.getText(self, "Set Session Tag", "Enter session name:", text="Default")
        if not (ok and text.strip()):
            return

        session_name = text.strip()

        # 🔥 Set it globally before adding
        self.current_session_tag = session_name

        # 🔥 Then add files or directory
        if frame_type.upper() == "FLAT":
            if directory_mode:
                self.add_directory(self.flat_tree, "Select Flat Directory", "FLAT")
            else:
                self.add_files(self.flat_tree, "Select Flat Files", "FLAT")
            self.assign_best_master_dark()
            self.rebuild_flat_tree()

        elif frame_type.upper() == "LIGHT":
            if directory_mode:
                self.add_directory(self.light_tree, "Select Light Directory", "LIGHT")
            else:
                self.add_files(self.light_tree, "Select Light Files", "LIGHT")
            self.assign_best_master_files()

    def load_master_dark(self):
        """ Loads a Master Dark and updates the UI. """
        last_dir = self.settings.value("last_opened_folder", "", type=str)  # Get last folder
        files, _ = QFileDialog.getOpenFileNames(self, "Select Master Dark", last_dir, "FITS Files (*.fits *.fit)")
        
        if files:
            self.settings.setValue("last_opened_folder", os.path.dirname(files[0]))  # Save last used folder
            self.add_master_files(self.master_dark_tree, "DARK", files)
            self.save_master_paths_to_settings() 

        self.update_override_dark_combo()
        self.assign_best_master_dark()
        self.assign_best_master_files()
        print("DEBUG: Loaded Master Darks and updated assignments.")


    def load_master_flat(self):
        last_dir = self.settings.value("last_opened_folder", "", type=str)
        files, _ = QFileDialog.getOpenFileNames(self, "Select Master Flat", last_dir, "FITS Files (*.fits *.fit)")

        if files:
            self.settings.setValue("last_opened_folder", os.path.dirname(files[0]))
            self.add_master_files(self.master_flat_tree, "FLAT", files)
            self.save_master_paths_to_settings() 


    def add_files(self, tree, title, expected_type):
        """ Adds FITS files and assigns best master files if needed. """
        last_dir = self.settings.value("last_opened_folder", "", type=str)
        files, _ = QFileDialog.getOpenFileNames(self, title, last_dir, "FITS Files (*.fits *.fit *.fz *.fz)")

        if files:
            self.settings.setValue("last_opened_folder", os.path.dirname(files[0]))  # Save last opened folder
            for file in files:
                self.process_fits_header(file, tree, expected_type)

            # 🔥 Auto-assign Master Dark & Flat **if adding LIGHTS**
            if expected_type == "LIGHT":
                self.assign_best_master_files()



    def add_directory(self, tree, title, expected_type):
        """ Adds all FITS files from a directory and assigns best master files if needed. """
        last_dir = self.settings.value("last_opened_folder", "", type=str)
        directory = QFileDialog.getExistingDirectory(self, title, last_dir)

        if directory:
            self.settings.setValue("last_opened_folder", directory)  # Save last opened folder
            for file in os.listdir(directory):
                if file.lower().endswith((".fits", ".fit", ".fz", ".fz")):
                    self.process_fits_header(os.path.join(directory, file), tree, expected_type)

            # 🔥 Auto-assign Master Dark & Flat **if adding LIGHTS**
            if expected_type == "LIGHT":
                self.assign_best_master_files()

    def _sanitize_name(self, name: str) -> str:
        """
        Replace any character that isn’t a letter, digit, space, dash or underscore
        with an underscore so it’s safe to use in filenames, dict-keys, tree labels, etc.
        """
        return re.sub(r"[^\w\s\-]", "_", name)
    
    def process_fits_header(self, file_path, tree, expected_type):
        try:
            # Read only the FITS header (fast)
            header, _ = get_valid_header(file_path)

            try:
                width = int(header.get("NAXIS1"))
                height = int(header.get("NAXIS2"))
            except Exception as e:
                self.update_status(f"Warning: Could not convert dimensions to int for {file_path}: {e}")
                width, height = None, None

            if width is not None and height is not None:
                image_size = f"{width}x{height}"
            else:
                image_size = "Unknown"

            # Retrieve IMAGETYP (default to "UNKNOWN" if not present)
            imagetyp = header.get("IMAGETYP", "UNKNOWN").lower()

            # Retrieve exposure from either EXPOSURE or EXPTIME
            exposure_val = header.get("EXPOSURE")
            if not exposure_val:
                exposure_val = header.get("EXPTIME")
            if not exposure_val:
                exposure_val = "Unknown"  # fallback if neither keyword is present

            # Define forbidden keywords per expected type.
            if expected_type.upper() == "DARK":
                forbidden = ["light", "flat"]
            elif expected_type.upper() == "FLAT":
                forbidden = ["dark", "light"]
            elif expected_type.upper() == "LIGHT":
                forbidden = ["dark", "flat"]
            else:
                forbidden = []

            # Determine attribute name for auto-confirm decision (per expected type)
            decision_attr = f"auto_confirm_{expected_type.lower()}"
            # If a decision has already been made, use it.
            if hasattr(self, decision_attr):
                decision = getattr(self, decision_attr)
                if decision is False:
                    # Skip this file automatically.
                    return
                # If decision is True, then add without prompting.
            elif any(word in imagetyp for word in forbidden):
                # Prompt the user with Yes, Yes to All, No, and No to All options.
                msgBox = QMessageBox(self)
                msgBox.setWindowTitle("Mismatched Image Type")
                msgBox.setText(
                    f"The file:\n{os.path.basename(file_path)}\n"
                    f"has IMAGETYP = {header.get('IMAGETYP')} "
                    f"which does not match the expected type ({expected_type}).\n\n"
                    f"Do you want to add it anyway?"
                )
                yesButton = msgBox.addButton("Yes", QMessageBox.ButtonRole.YesRole)
                yesToAllButton = msgBox.addButton("Yes to All", QMessageBox.ButtonRole.YesRole)
                noButton = msgBox.addButton("No", QMessageBox.ButtonRole.NoRole)
                noToAllButton = msgBox.addButton("No to All", QMessageBox.ButtonRole.NoRole)
                msgBox.exec()
                clicked = msgBox.clickedButton()
                if clicked == yesToAllButton:
                    setattr(self, decision_attr, True)
                elif clicked == noToAllButton:
                    setattr(self, decision_attr, False)
                    return
                elif clicked == noButton:
                    return

            # Now handle each expected type
            if expected_type.upper() == "DARK":
                key = f"{exposure_val} ({image_size})"
                if key not in self.dark_files:
                    self.dark_files[key] = []
                self.dark_files[key].append(file_path)

                items = tree.findItems(key, Qt.MatchFlag.MatchExactly, 0)
                if not items:
                    exposure_item = QTreeWidgetItem([key])
                    tree.addTopLevelItem(exposure_item)
                else:
                    exposure_item = items[0]
                metadata = f"Size: {image_size}"
                exposure_item.addChild(QTreeWidgetItem([os.path.basename(file_path), metadata]))

            elif expected_type.upper() == "FLAT":
                filter_name = header.get("FILTER", "Unknown")
                filter_name = self._sanitize_name(filter_name)
                flat_key = f"{filter_name} - {exposure_val} ({image_size})"
                session_tag = getattr(self, "current_session_tag", "Default")
                composite_key = (flat_key, session_tag)

                if composite_key not in self.flat_files:
                    self.flat_files[composite_key] = []
                self.flat_files[composite_key].append(file_path)

                # ✅ Also store session tag internally
                self.session_tags[file_path] = session_tag

                # Tree UI update
                filter_items = tree.findItems(filter_name, Qt.MatchFlag.MatchExactly, 0)
                if not filter_items:
                    filter_item = QTreeWidgetItem([filter_name])
                    tree.addTopLevelItem(filter_item)
                else:
                    filter_item = filter_items[0]

                exposure_items = [filter_item.child(i) for i in range(filter_item.childCount())]
                exposure_item = next((item for item in exposure_items
                                    if item.text(0) == f"{exposure_val} ({image_size})"), None)
                if not exposure_item:
                    exposure_item = QTreeWidgetItem([f"{exposure_val} ({image_size})"])
                    filter_item.addChild(exposure_item)

                metadata = f"Size: {image_size} | Session: {session_tag}"
                exposure_item.addChild(QTreeWidgetItem([os.path.basename(file_path), metadata]))


            elif expected_type.upper() == "LIGHT":
                filter_name = header.get("FILTER", "Unknown")
                filter_name = self._sanitize_name(filter_name)
                session_tag = getattr(self, "current_session_tag", "Default")  # ⭐️ Step 1: Get session label

                light_key = f"{filter_name} - {exposure_val} ({image_size})"
                composite_key = (light_key, session_tag)

                if composite_key not in self.light_files:
                    self.light_files[composite_key] = []
                self.light_files[composite_key].append(file_path)

                # Update Tree UI
                filter_items = tree.findItems(filter_name, Qt.MatchFlag.MatchExactly, 0)
                if not filter_items:
                    filter_item = QTreeWidgetItem([filter_name])
                    tree.addTopLevelItem(filter_item)
                else:
                    filter_item = filter_items[0]

                exposure_items = [filter_item.child(i) for i in range(filter_item.childCount())]
                exposure_item = next((item for item in exposure_items
                                    if item.text(0) == f"{exposure_val} ({image_size})"), None)
                if not exposure_item:
                    exposure_item = QTreeWidgetItem([f"{exposure_val} ({image_size})"])
                    filter_item.addChild(exposure_item)

                leaf_item = QTreeWidgetItem([os.path.basename(file_path), f"Size: {image_size} | Session: {session_tag}"])
                exposure_item.addChild(leaf_item)
                self.session_tags[file_path] = session_tag  # ✅ Store per-file session tag here


            self.update_status(f"✅ Added {os.path.basename(file_path)} as {expected_type}")
            QApplication.processEvents()

        except Exception as e:
            self.update_status(f"❌ ERROR: Could not read FITS header for {file_path} - {e}")
            QApplication.processEvents()


    def add_master_files(self, tree, file_type, files):
        """ 
        Adds multiple master calibration files to the correct treebox with metadata including image dimensions.
        This version only reads the FITS header to extract image dimensions, making it much faster.
        """
        for file_path in files:
            try:
                # Read only the FITS header (fast)
                header = fits.getheader(file_path)
                
                # Check for both EXPOSURE and EXPTIME
                exposure = header.get("EXPOSURE", header.get("EXPTIME", "Unknown"))
                filter_name = header.get("FILTER", "Unknown")
                filter_name     = self._sanitize_name(filter_name)
                # Extract image dimensions from header keywords NAXIS1 and NAXIS2
                width = header.get("NAXIS1")
                height = header.get("NAXIS2")
                if width is not None and height is not None:
                    image_size = f"{width}x{height}"
                else:
                    image_size = "Unknown"
                
                # Construct key based on file type
                if file_type.upper() == "DARK":
                    key = f"{exposure}s ({image_size})"
                    self.master_files[key] = file_path  # Store master dark
                    self.master_sizes[file_path] = image_size  # Store size
                elif file_type.upper() == "FLAT":
                    # Attempt to extract session name from filename
                    session_name = "Default"
                    filename = os.path.basename(file_path)
                    if filename.lower().startswith("masterflat_"):
                        parts = filename.split("_")
                        if len(parts) > 1:
                            session_name = parts[1]

                    key = f"{filter_name} ({image_size}) [{session_name}]"
                    self.master_files[key] = file_path
                    self.master_sizes[file_path] = image_size

                # Extract additional metadata from header.
                sensor_temp = header.get("CCD-TEMP", "N/A")
                date_obs = header.get("DATE-OBS", "Unknown")
                metadata = f"Size: {image_size}, Temp: {sensor_temp}°C, Date: {date_obs}"

                # Check if category item already exists in the tree.
                items = tree.findItems(key, Qt.MatchFlag.MatchExactly, 0)
                if not items:
                    item = QTreeWidgetItem([key])
                    tree.addTopLevelItem(item)
                else:
                    item = items[0]

                # Add the master file as a child node with metadata.
                item.addChild(QTreeWidgetItem([os.path.basename(file_path), metadata]))

                print(f"✅ DEBUG: Added Master {file_type} -> {file_path} under {key} with metadata: {metadata}")
                self.update_status(f"✅ Added Master {file_type} -> {file_path} under {key} with metadata: {metadata}")
                print(f"📂 DEBUG: Master Files Stored: {self.master_files}")
                self.update_status(f"📂 DEBUG: Master Files Stored: {self.master_files}")
                QApplication.processEvents()
                self.assign_best_master_files()

            except Exception as e:
                print(f"❌ ERROR: Failed to load master file {file_path} - {e}")
                self.update_status(f"❌ ERROR: Failed to load master file {file_path} - {e}")
                QApplication.processEvents()



    def create_master_dark(self):
        """ Creates master darks with minimal RAM usage by loading frames in small tiles. """

        if not self.stacking_directory:
            self.select_stacking_directory()
            if not self.stacking_directory:
                QMessageBox.warning(self, "Error", "Output directory is not set.")
                return

        exposure_tolerance = self.exposure_tolerance_spinbox.value()
        dark_files_by_group = {}

        # 1) Group dark files by exposure time & image size within tolerance
        for exposure_key, file_list in self.dark_files.items():
            exposure_time_str, image_size = exposure_key.split(" (")
            image_size = image_size.rstrip(")")
            exposure_time = float(exposure_time_str.replace("s", "")) if "Unknown" not in exposure_time_str else 0

            matched_group = None
            for (existing_exposure, existing_size) in dark_files_by_group.keys():
                if abs(existing_exposure - exposure_time) <= exposure_tolerance and existing_size == image_size:
                    matched_group = (existing_exposure, existing_size)
                    break

            if matched_group is None:
                matched_group = (exposure_time, image_size)
                dark_files_by_group[matched_group] = []

            dark_files_by_group[matched_group].extend(file_list)

        # 2) Create Master Calibration Directory
        master_dir = os.path.join(self.stacking_directory, "Master_Calibration_Files")
        os.makedirs(master_dir, exist_ok=True)

        # 3) Stack Each Group in a Chunked Manner
        chunk_height = self.chunk_height
        chunk_width  = self.chunk_width

        for (exposure_time, image_size), file_list in dark_files_by_group.items():
            if len(file_list) < 2:
                self.update_status(f"⚠️ Skipping {exposure_time}s ({image_size}) - Not enough frames to stack.")
                QApplication.processEvents()
                continue

            self.update_status(f"🟢 Processing {len(file_list)} darks for {exposure_time}s ({image_size}) exposure...")
            QApplication.processEvents()

            # (A) Identify reference shape from the first file
            ref_file = file_list[0]
            ref_data, ref_header, bit_depth, is_mono = load_image(ref_file)
            if ref_data is None:
                self.update_status(f"❌ Failed to load reference {os.path.basename(ref_file)}")
                continue

            height, width = ref_data.shape[:2]
            channels = 1 if (ref_data.ndim == 2) else 3

            # (B) Create a memmap for the final stacked result
            # shape=(height, width, channels)
            memmap_path = os.path.join(master_dir, f"temp_dark_{exposure_time}_{image_size}.dat")
            final_stacked = np.memmap(
                memmap_path,
                dtype=self._dtype(),
                mode='w+',
                shape=(height, width, channels)
            )

            # (C) For each tile, load that tile from all frames, do outlier rejection, store in final_stacked
            num_frames = len(file_list)

            for y_start in range(0, height, chunk_height):
                y_end = min(y_start + chunk_height, height)
                tile_h = y_end - y_start

                for x_start in range(0, width, chunk_width):
                    x_end = min(x_start + chunk_width, width)
                    tile_w = x_end - x_start

                    # tile_stack shape => (num_frames, tile_h, tile_w, channels)
                    tile_stack = np.zeros((num_frames, tile_h, tile_w, channels), dtype=np.float32)


                    num_cores = os.cpu_count() or 4
                    with ThreadPoolExecutor(max_workers=num_cores) as executor:
                        future_to_index = {}
                        # 1) Submit each file’s tile load in parallel
                        for i, fpath in enumerate(file_list):
                            future = executor.submit(load_fits_tile, fpath, y_start, y_end, x_start, x_end)
                            future_to_index[future] = i

                        # 2) Collect results as they complete
                        for future in as_completed(future_to_index):
                            i = future_to_index[future]
                            sub_img = future.result()
                            if sub_img is None:
                                continue

                            # --- shape handling (same as before) ---
                            # If sub_img is (H,W) & channels=3 => expand
                            if sub_img.ndim == 2 and channels == 3:
                                sub_img = np.repeat(sub_img[:, :, np.newaxis], 3, axis=2)
                            elif sub_img.ndim == 2 and channels == 1:
                                sub_img = sub_img[:, :, np.newaxis]

                            # If sub_img is (3,H,W) but we want (H,W,3), transpose
                            if sub_img.ndim == 3 and sub_img.shape[0] == 3 and channels == 3:
                                sub_img = sub_img.transpose(1, 2, 0)

                            sub_img = sub_img.astype(np.float32, copy=False)
                            tile_stack[i] = sub_img

                    # (D) Outlier rejection => tile_result
                    # Use your existing 3D or 4D Windsorized Sigma Clip depending on channels
                    if channels == 3:
                        # tile_stack => shape (F,H,W,3)
                        tile_result = windsorized_sigma_clip_4d(
                            tile_stack,
                            lower=self.sigma_low,
                            upper=self.sigma_high
                        )
                        # If the function returns a tuple, extract the first element.
                        if isinstance(tile_result, tuple):
                            tile_result = tile_result[0]
                    else:
                        # tile_stack => shape (F,H,W,1) or (F,H,W)
                        # If shape=(F,H,W,1), we can squeeze or just call 3D version
                        tile_stack_3d = tile_stack[..., 0] if tile_stack.ndim == 4 else tile_stack
                        tile_result_3d = windsorized_sigma_clip_3d(tile_stack_3d, lower=self.sigma_low, upper=self.sigma_high)
                        # If the function returns a tuple, extract the first element.
                        if isinstance(tile_result_3d, tuple):
                            tile_result_3d = tile_result_3d[0]
                        # Now, ensure the result has shape (H, W, 1)
                        tile_result = tile_result_3d[..., np.newaxis]

                    # (E) Store tile_result in final_stacked
                    final_stacked[y_start:y_end, x_start:x_end, :] = tile_result

            # Convert final_stacked to a normal array
            master_dark_data = np.array(final_stacked)
            del final_stacked

            # (F) Save Master Dark
            master_dark_path = os.path.join(master_dir, f"MasterDark_{int(exposure_time)}s_{image_size}.fit")

            # Build a minimal header
            # Possibly store EXPTIME, IMAGETYP="DARK", etc.
            master_header = fits.Header()
            master_header["IMAGETYP"] = "DARK"
            master_header["EXPTIME"]  = (exposure_time, "User-specified or from grouping")
            # plus any other fields you want

            # Remove NAXIS from the old ref_header if you want
            # or define them fresh
            master_header["NAXIS"] = 3 if channels==3 else 2
            master_header["NAXIS1"] = master_dark_data.shape[1]
            master_header["NAXIS2"] = master_dark_data.shape[0]
            if channels==3:
                master_header["NAXIS3"] = 3

            save_image(
                img_array=master_dark_data,
                filename=master_dark_path,
                original_format="fit",
                bit_depth="32-bit floating point",
                original_header=master_header,
                is_mono=(channels==1)
            )

            # (G) Add to tree, status, etc.
            self.add_master_dark_to_tree(f"{exposure_time}s ({image_size})", master_dark_path)
            self.update_status(f"✅ Master Dark saved: {master_dark_path}")
            self.assign_best_master_files()
            self.save_master_paths_to_settings()

        # Finally, assign best master dark, etc.
        self.assign_best_master_dark()
        self.update_override_dark_combo()
        self.assign_best_master_files()



    def save_master_dark(self, master_dark, output_path, exposure_time, is_mono):
        """Saves the master dark as 32-bit floating point FITS while maintaining OSC structure."""
        if is_mono:
            # Mono => shape (H, W)
            h, w = master_dark.shape
            # Wrap in an HDU
            hdu_data = master_dark.astype(np.float32)
            hdu = fits.PrimaryHDU(hdu_data)
            image_size = f"{w}x{h}"  # Width x Height
        else:
            # Color => shape (H, W, C)
            h, w, c = master_dark.shape
            # Transpose to (C, H, W)
            hdu_data = master_dark.transpose(2, 0, 1).astype(np.float32)
            hdu = fits.PrimaryHDU(hdu_data)
            image_size = f"{w}x{h}"

        # Now 'hdu' is a fits.PrimaryHDU in both branches
        hdr = hdu.header
        hdr["SIMPLE"]   = True
        hdr["BITPIX"]   = -32
        hdr["NAXIS"]    = 3 if not is_mono else 2
        hdr["NAXIS1"]   = w  # Width
        hdr["NAXIS2"]   = h  # Height
        if not is_mono:
            hdr["NAXIS3"] = c
        hdr["BSCALE"]   = 1.0
        hdr["BZERO"]    = 0.0
        hdr["IMAGETYP"] = "MASTER DARK"
        hdr["EXPOSURE"] = exposure_time
        hdr["DATE-OBS"] = datetime.utcnow().isoformat()
        hdr["CREATOR"]  = "SetiAstroSuite"

        # Write the FITS file
        hdu.writeto(output_path, overwrite=True)

        # Store Master Dark path with correct key
        key = f"{exposure_time}s ({image_size})"
        self.master_files[key] = output_path
        self.master_sizes[output_path] = image_size

        print(f"✅ Master Dark FITS saved: {output_path}")
        self.update_status(f"✅ Stored Master Dark -> {key}: {output_path}")



            
    def add_master_dark_to_tree(self, exposure_time, master_dark_path):
        """ Adds the newly created Master Dark to the Master Dark TreeBox and updates the dropdown. """

        exposure_key = f"{exposure_time}s"

        # ✅ Store in the dictionary
        self.master_files[exposure_key] = master_dark_path  # Store master dark
        print(f"📝 DEBUG: Stored Master Dark -> {exposure_key}: {master_dark_path}")

        # ✅ Update UI Tree
        existing_items = self.master_dark_tree.findItems(exposure_key, Qt.MatchFlag.MatchExactly, 0)

        if existing_items:
            exposure_item = existing_items[0]
        else:
            exposure_item = QTreeWidgetItem([exposure_key])
            self.master_dark_tree.addTopLevelItem(exposure_item)

        master_item = QTreeWidgetItem([os.path.basename(master_dark_path)])
        exposure_item.addChild(master_item)

        # ✅ Refresh the override dropdown
        self.update_override_dark_combo()
        self.assign_best_master_dark()  # 🔥 Ensure auto-selection works

        self.update_status(f"✅ Master Dark saved and added to UI: {master_dark_path}")



    def assign_best_master_dark(self):
        """ Assigns the closest matching master dark based on exposure & image size. """
        print("\n🔍 DEBUG: Assigning best master darks to flats...\n")

        if not self.master_files:
            print("⚠️ WARNING: No Master Darks available.")
            self.update_status("⚠️ WARNING: No Master Darks available.")
            return  # Exit early if there are no master darks

        print(f"📂 Loaded Master Darks ({len(self.master_files)} total):")
        for key, value in self.master_files.items():
            print(f"   📌 {key} -> {value}")

        # Iterate through all flat filters
        for i in range(self.flat_tree.topLevelItemCount()):
            filter_item = self.flat_tree.topLevelItem(i)

            for j in range(filter_item.childCount()):
                exposure_item = filter_item.child(j)
                exposure_text = exposure_item.text(0)  # Example: "0.0007s (8288x5644)"

                # Extract exposure time
                match = re.match(r"([\d.]+)s?", exposure_text)
                if not match:
                    print(f"⚠️ WARNING: Could not parse exposure time from {exposure_text}")
                    continue  # Skip if exposure is invalid

                exposure_time = float(match.group(1))  # Extracted number
                print(f"🟢 Checking Flat Group: {exposure_text} (Parsed: {exposure_time}s)")

                # Extract image size from metadata
                if exposure_item.childCount() > 0:
                    metadata_text = exposure_item.child(0).text(1)  # Metadata column
                    size_match = re.search(r"Size: (\d+x\d+)", metadata_text)
                    image_size = size_match.group(1) if size_match else "Unknown"
                else:
                    image_size = "Unknown"

                print(f"✅ Parsed Flat Size: {image_size}")

                # Find the best matching master dark
                best_match = None
                best_diff = float("inf")

                for master_dark_exposure, master_dark_path in self.master_files.items():
                    master_dark_exposure_match = re.match(r"([\d.]+)s?", master_dark_exposure)
                    if not master_dark_exposure_match:
                        continue  # Skip if master dark exposure is invalid

                    master_dark_exposure_time = float(master_dark_exposure_match.group(1))
                    master_dark_size = self.master_sizes.get(master_dark_path, "Unknown")
                    if master_dark_size == "Unknown":
                        with fits.open(master_dark_path) as hdul:
                            master_dark_size = f"{hdul[0].data.shape[1]}x{hdul[0].data.shape[0]}"
                            self.master_sizes[master_dark_path] = master_dark_size  # ✅ Store it

                    print(f"🔎 Comparing with Master Dark: {master_dark_exposure_time}s ({master_dark_size})")

                    # Match both image size and exposure time
                    if image_size == master_dark_size:
                        diff = abs(master_dark_exposure_time - exposure_time)
                        if diff < best_diff:
                            best_match = master_dark_path
                            best_diff = diff

                # Assign best match in column 3
                if best_match:
                    exposure_item.setText(2, os.path.basename(best_match))
                    print(f"🔵 Assigned Master Dark: {os.path.basename(best_match)}")
                else:
                    exposure_item.setText(2, "None")
                    print(f"⚠️ No matching Master Dark found for {exposure_text}")

        # 🔥 Force UI update to reflect changes
        self.flat_tree.viewport().update()

        print("\n✅ DEBUG: Finished assigning best matching Master Darks to Flats.\n")



    def update_override_dark_combo(self):
        """ Populates the dropdown with available Master Darks and prevents duplicate entries. """
        self.override_dark_combo.clear()
        self.override_dark_combo.addItem("None (Use Auto-Select)")
        self.override_dark_combo.addItem("None (Use no Dark to Calibrate)")

        seen_files = set()
        for exposure, path in self.master_files.items():
            file_name = os.path.basename(path)
            if file_name not in seen_files:
                self.override_dark_combo.addItem(f"{file_name} ({exposure})")
                seen_files.add(file_name)

        print("✅ DEBUG: Updated Override Master Dark dropdown with unique entries.")


    def override_selected_master_dark(self):
        """ Overrides the selected master dark for the currently highlighted flat group. """
        selected_items = self.flat_tree.selectedItems()
        if not selected_items:
            return

        new_dark = self.override_dark_combo.currentText()

        # ✅ Handle "None (Use no Dark to Calibrate)" explicitly
        if new_dark == "None (Use no Dark to Calibrate)":
            new_dark = "No Calibration"  # Show "No Calibration" in the UI
        elif new_dark == "None (Use Auto-Select)":
            new_dark = None  # Auto-select behavior

        for item in selected_items:
            if item.parent():  # Ensure it's an exposure group, not the top filter name
                item.setText(2, new_dark if new_dark else "Auto")

        print(f"✅ DEBUG: Override Master Dark set to: {new_dark}")

    def create_master_flat(self):
        """ Creates master flats using per-frame dark subtraction before stacking. """

        if not self.stacking_directory:
            QMessageBox.warning(self, "Error", "Please set the stacking directory first using the wrench button.")
            return

        exposure_tolerance = self.flat_exposure_tolerance_spinbox.value()
        flat_files_by_group = {}  # Group by (Exposure, Image Size, Filter, Session)

        # Group Flats by Filter, Exposure & Size within Tolerance
        for (filter_exposure, session), file_list in self.flat_files.items():
            try:
                filter_name, exposure_size = filter_exposure.split(" - ")
                exposure_time_str, image_size = exposure_size.split(" (")
                image_size = image_size.rstrip(")")
            except ValueError:
                self.update_status(f"⚠️ ERROR: Could not parse {filter_exposure}")
                continue

            match = re.match(r"([\d.]+)s?", exposure_time_str)
            exposure_time = float(match.group(1)) if match else -10.0

            matched_group = None
            for key in flat_files_by_group:
                existing_exposure, existing_size, existing_filter, existing_session = key
                if (
                    abs(existing_exposure - exposure_time) <= exposure_tolerance
                    and existing_size == image_size
                    and existing_filter == filter_name
                    and existing_session == session
                ):
                    matched_group = key
                    break

            if matched_group is None:
                matched_group = (exposure_time, image_size, filter_name, session)
                flat_files_by_group[matched_group] = []

            flat_files_by_group[matched_group].extend(file_list)

        # Create output folder
        master_dir = os.path.join(self.stacking_directory, "Master_Calibration_Files")
        os.makedirs(master_dir, exist_ok=True)

        # Stack each grouped flat set
        for (exposure_time, image_size, filter_name, session), file_list in flat_files_by_group.items():
            if len(file_list) < 2:
                self.update_status(f"⚠️ Skipping {exposure_time}s ({image_size}) [{filter_name}] [{session}] - Not enough frames to stack.")
                continue

            self.update_status(f"🟢 Processing {len(file_list)} flats for {exposure_time}s ({image_size}) [{filter_name}] in session '{session}'...")
            QApplication.processEvents()

            # Load master dark
            best_diff = float("inf")
            selected_master_dark = None
            for key, path in self.master_files.items():
                match = re.match(r"([\d.]+)s", key)
                if not match:
                    continue
                dark_exposure = float(match.group(1))
                dark_size = self.master_sizes.get(path, "Unknown")
                if dark_size == image_size:
                    diff = abs(dark_exposure - exposure_time)
                    if diff < best_diff:
                        best_diff = diff
                        selected_master_dark = path

            if selected_master_dark:
                dark_data, _, _, _ = load_image(selected_master_dark)
            else:
                dark_data = None
                self.update_status("DEBUG: No matching Master Dark found.")

            # Load reference image
            ref_data, _, _, _ = load_image(file_list[0])
            if ref_data is None:
                self.update_status(f"❌ Failed to load reference {os.path.basename(file_list[0])}")
                continue

            height, width = ref_data.shape[:2]
            channels = 1 if ref_data.ndim == 2 else 3
            memmap_path = os.path.join(master_dir, f"temp_flat_{session}_{exposure_time}_{image_size}_{filter_name}.dat")
            final_stacked = np.memmap(memmap_path, dtype=self._dtype(), mode="w+", shape=(height, width, channels))
            num_frames = len(file_list)

            for y_start in range(0, height, self.chunk_height):
                y_end = min(y_start + self.chunk_height, height)
                tile_h = y_end - y_start
                for x_start in range(0, width, self.chunk_width):
                    x_end = min(x_start + self.chunk_width, width)
                    tile_w = x_end - x_start
                    tile_stack = np.zeros((num_frames, tile_h, tile_w, channels), dtype=np.float32)

                    with ThreadPoolExecutor() as executor:
                        
                        futures = {
                            executor.submit(load_fits_tile, f, y_start, y_end, x_start, x_end): idx
                            for idx, f in enumerate(file_list)
                        }

                        for future in as_completed(futures):
                            i = futures[future]
                            sub_img = future.result()
                            if sub_img is None:
                                self.update_status(f"⚠️ Skipping tile {i} due to load failure.")
                                continue

                            # Ensure correct shape
                            if sub_img.ndim == 2:
                                sub_img = sub_img[:, :, np.newaxis]
                                if channels == 3:
                                    sub_img = np.repeat(sub_img, 3, axis=2)
                            elif sub_img.shape[0] == 3:
                                sub_img = sub_img.transpose(1, 2, 0)

                            tile_stack[i] = sub_img


                    if dark_data is not None:
                        dark_tile = dark_data[y_start:y_end, x_start:x_end]
                        if dark_tile.ndim == 2:
                            dark_tile = dark_tile[..., np.newaxis]
                            if channels == 3:
                                dark_tile = np.repeat(dark_tile, 3, axis=2)
                        elif dark_tile.shape[0] == 3:
                            dark_tile = dark_tile.transpose(1, 2, 0)

                        if dark_tile.shape == (tile_h, tile_w, channels):
                            tile_stack = subtract_dark(tile_stack, dark_tile)

                    if channels == 3:
                        tile_result = windsorized_sigma_clip_4d(tile_stack, lower=self.sigma_low, upper=self.sigma_high)[0]
                    else:
                        stack_3d = tile_stack[..., 0]
                        tile_result = windsorized_sigma_clip_3d(stack_3d, lower=self.sigma_low, upper=self.sigma_high)[0]
                        tile_result = tile_result[..., np.newaxis]

                    final_stacked[y_start:y_end, x_start:x_end, :] = tile_result

            master_flat_data = np.array(final_stacked)
            del final_stacked

            master_flat_path = os.path.join(
                master_dir,
                f"MasterFlat_{session}_{int(exposure_time)}s_{image_size}_{filter_name}.fit"
            )

            header = fits.Header()
            header["IMAGETYP"] = "FLAT"
            header["EXPTIME"] = (exposure_time, "grouped exposure")
            header["FILTER"] = filter_name
            header["NAXIS"] = 3 if channels == 3 else 2
            header["NAXIS1"] = width
            header["NAXIS2"] = height
            if channels == 3:
                header["NAXIS3"] = 3

            save_image(
                img_array=master_flat_data,
                filename=master_flat_path,
                original_format="fit",
                bit_depth="32-bit floating point",
                original_header=header,
                is_mono=(channels == 1)
            )

            key = f"{filter_name} ({image_size}) [{session}]"
            self.master_files[key] = master_flat_path
            self.master_sizes[master_flat_path] = image_size
            self.add_master_flat_to_tree(filter_name, master_flat_path)
            self.update_status(f"✅ Master Flat saved: {master_flat_path}")
            self.save_master_paths_to_settings()

        self.assign_best_master_dark()
        self.assign_best_master_files()



    def save_master_flat(self, master_flat, output_path, exposure_time, filter_name):
        """ Saves master flat as both a 32-bit floating point FITS and TIFF while ensuring no unintended normalization. """

        # ✅ Retrieve FITS header from a sample flat (to check if it's mono or color)
        original_header = None
        is_mono = True  # Default to mono

        if self.flat_files:
            sample_flat = next(iter(self.flat_files.values()))[0]  # Get the first flat file
            try:
                with fits.open(sample_flat) as hdul:
                    original_header = hdul[0].header

                    # **🔍 Detect if the flat is color by checking NAXIS3**
                    if original_header.get("NAXIS", 2) == 3 and original_header.get("NAXIS3", 1) == 3:
                        is_mono = False  # ✅ It's a color flat

            except Exception as e:
                print(f"⚠️ Warning: Could not retrieve FITS header from {sample_flat}: {e}")

        # ✅ Explicitly ensure we are saving raw values (NO normalization)
        fits_header = original_header if original_header else fits.Header()
        fits_header["BSCALE"] = 1.0  # 🔹 Prevent rescaling
        fits_header["BZERO"] = 0.0   # 🔹 Prevent offset

        # ✅ Save as FITS
        save_image(
            img_array=master_flat,
            filename=output_path,
            original_format="fit",
            bit_depth="32-bit floating point",
            original_header=fits_header,
            is_mono=is_mono
        )

        print(f"✅ Master Flat FITS saved: {output_path}")




    def add_master_flat_to_tree(self, filter_name, master_flat_path):
        """ Adds the newly created Master Flat to the Master Flat TreeBox and stores it. """

        key = f"{filter_name} ({self.master_sizes[master_flat_path]})"
        self.master_files[key] = master_flat_path  # ✅ Store the flat file for future use
        print(f"📝 DEBUG: Stored Master Flat -> {key}: {master_flat_path}")

        existing_items = self.master_flat_tree.findItems(filter_name, Qt.MatchFlag.MatchExactly, 0)

        if existing_items:
            filter_item = existing_items[0]
        else:
            filter_item = QTreeWidgetItem([filter_name])
            self.master_flat_tree.addTopLevelItem(filter_item)

        master_item = QTreeWidgetItem([os.path.basename(master_flat_path)])
        filter_item.addChild(master_item)

    def assign_best_master_files(self):
        """ Assign best matching Master Dark and Flat to each Light Frame (per leaf). """
        print("\n🔍 DEBUG: Assigning best Master Darks & Flats to Lights...\n")

        if not self.master_files:
            print("⚠️ WARNING: No Master Calibration Files available.")
            self.update_status("⚠️ WARNING: No Master Calibration Files available.")
            return

        for i in range(self.light_tree.topLevelItemCount()):
            filter_item = self.light_tree.topLevelItem(i)
            filter_name = filter_item.text(0)
            filter_name     = self._sanitize_name(filter_name)

            for j in range(filter_item.childCount()):
                exposure_item = filter_item.child(j)
                exposure_text = exposure_item.text(0)

                match = re.match(r"([\d.]+)s?", exposure_text)
                if not match:
                    print(f"⚠️ WARNING: Could not parse exposure time from {exposure_text}")
                    continue

                exposure_time = float(match.group(1))

                for k in range(exposure_item.childCount()):
                    leaf_item = exposure_item.child(k)
                    meta_text = leaf_item.text(1)

                    size_match = re.search(r"Size: (\d+x\d+)", meta_text)
                    session_match = re.search(r"Session: ([^|]+)", meta_text)
                    image_size = size_match.group(1) if size_match else "Unknown"
                    session_name = session_match.group(1).strip() if session_match else "Default"

                    print(f"🧠 Leaf: {leaf_item.text(0)} | Size: {image_size} | Session: {session_name}")

                    # 🔍 Match Dark
                    best_dark_match = None
                    best_dark_diff = float("inf")

                    for master_key, master_path in self.master_files.items():
                        # ✅ Only consider keys that start with an exposure (i.e. darks)
                        dark_match = re.match(r"^([\d.]+)s\b", master_key)
                        if not dark_match:
                            continue
                        master_dark_exposure_time = float(dark_match.group(1))

                        # Ensure we know the dark’s size
                        master_dark_size = self.master_sizes.get(master_path, "Unknown")
                        if master_dark_size == "Unknown":
                            with fits.open(master_path) as hdul:
                                master_dark_size = f"{hdul[0].data.shape[1]}x{hdul[0].data.shape[0]}"
                                self.master_sizes[master_path] = master_dark_size

                        # Only compare if sizes match
                        if master_dark_size == image_size:
                            diff = abs(master_dark_exposure_time - exposure_time)
                            if diff < best_dark_diff:
                                best_dark_match = master_path
                                best_dark_diff = diff

                    # 🔍 Match Flat
                    best_flat_match = None
                    for flat_key, flat_path in self.master_files.items():
                        if filter_name not in flat_key or f"({image_size})" not in flat_key:
                            continue
                        if session_name in flat_key:
                            best_flat_match = flat_path
                            break
                    if not best_flat_match:
                        fallback_key = f"{filter_name} ({image_size})"
                        best_flat_match = self.master_files.get(fallback_key)

                    # 🔄 Assign to leaf
                    leaf_item.setText(2, os.path.basename(best_dark_match) if best_dark_match else "None")
                    leaf_item.setText(3, os.path.basename(best_flat_match) if best_flat_match else "None")

                    print(f"📌 Assigned to {leaf_item.text(0)} -> Dark: {leaf_item.text(2)}, Flat: {leaf_item.text(3)}")

        self.light_tree.viewport().update()
        print("\n✅ DEBUG: Finished assigning Master Files per leaf.\n")

    def update_light_corrections(self):
        """ Updates the light frame corrections when checkboxes change. """
        corrections = []
        if self.cosmetic_checkbox.isChecked():
            corrections.append("Cosmetic: True")
        else:
            corrections.append("Cosmetic: False")

        if self.pedestal_checkbox.isChecked():
            corrections.append("Pedestal: True")
        else:
            corrections.append("Pedestal: False")

        if self.bias_checkbox.isChecked():
            # Show file dialog to select a Master Bias
            bias_file, _ = QFileDialog.getOpenFileName(self, "Select Master Bias Frame", "", "FITS Files (*.fits *.fit)")
            if bias_file:
                self.master_files["Bias"] = bias_file  # ✅ Store bias path
                corrections.append(f"Bias: {os.path.basename(bias_file)}")
            else:
                self.bias_checkbox.setChecked(False)  # If no file selected, uncheck
                return

        # Update all rows
        for i in range(self.light_tree.topLevelItemCount()):
            filter_item = self.light_tree.topLevelItem(i)
            for j in range(filter_item.childCount()):
                exposure_item = filter_item.child(j)
                exposure_item.setText(4, ", ".join(corrections))

    def light_tree_context_menu(self, pos):
        item = self.light_tree.itemAt(pos)
        if not item:
            return

        menu = QMenu(self.light_tree)
        override_dark_action = menu.addAction("Override Dark Frame")
        override_flat_action = menu.addAction("Override Flat Frame")
        set_session_action = menu.addAction("Set Session Tag...")

        action = menu.exec(self.light_tree.viewport().mapToGlobal(pos))

        if action == override_dark_action:
            self.override_selected_master_dark()
        elif action == override_flat_action:
            self.override_selected_master_flat()
        elif action == set_session_action:
            self.prompt_set_session(item, "LIGHT")


    def set_session_tag_for_group(self, item):
        """
        Prompt the user to assign a session tag to all frames in this group.
        """
        session_name, ok = QInputDialog.getText(self, "Set Session Tag", "Enter session label (e.g., Night1, RedFilterSet2):")
        if not ok or not session_name.strip():
            return

        session_name = session_name.strip()
        filter_name = item.text(0)

        for i in range(item.childCount()):
            exposure_item = item.child(i)
            exposure_label = exposure_item.text(0)

            # Update metadata text
            if exposure_item.childCount() > 0:
                metadata_item = exposure_item.child(0)
                metadata_text = metadata_item.text(1)
                metadata_text = re.sub(r"Session: [^|]+", f"Session: {session_name}", metadata_text)
                if "Session:" not in metadata_text:
                    metadata_text += f" | Session: {session_name}"
                metadata_item.setText(1, metadata_text)

            # Update internal session tag mapping
            composite_key = (f"{filter_name} - {exposure_label}", session_name)
            original_key = f"{filter_name} - {exposure_label}"

            if original_key in self.light_files:
                self.light_files[composite_key] = self.light_files.pop(original_key)

                for path in self.light_files[composite_key]:
                    self.session_tags[path] = session_name

        self.update_status(f"🟢 Assigned session '{session_name}' to group '{filter_name}'")


    def override_selected_master_dark(self):
        """ Opens a file dialog to manually select a Master Dark for the selected group and stores it. """
        selected_items = self.light_tree.selectedItems()
        if not selected_items:
            print("⚠️ No light group selected for dark frame override.")
            return

        file_path, _ = QFileDialog.getOpenFileName(self, "Select Master Dark", "", "FITS Files (*.fits *.fit)")
        if not file_path:
            return  # User canceled

        for item in selected_items:
            if item.parent():  # Ensure it's an exposure group, not the top filter name
                item.setText(2, os.path.basename(file_path))  # Update tree UI
                self.manual_dark_overrides[item.text(0)] = file_path  # Store override

        print(f"✅ DEBUG: Overrode Master Dark for {item.text(0)} with {file_path}")



    def override_selected_master_flat(self):
        """ Opens a file dialog to manually select a Master Flat for the selected group and stores it. """
        selected_items = self.light_tree.selectedItems()
        if not selected_items:
            print("⚠️ No light group selected for flat frame override.")
            return

        file_path, _ = QFileDialog.getOpenFileName(self, "Select Master Flat", "", "FITS Files (*.fits *.fit)")
        if not file_path:
            return  # User canceled

        for item in selected_items:
            if item.parent():  # Ensure it's an exposure group, not the top filter name
                item.setText(3, os.path.basename(file_path))  # Update tree UI
                self.manual_flat_overrides[item.text(0)] = file_path  # Store override

        print(f"✅ DEBUG: Overrode Master Flat for {item.text(0)} with {file_path}")


    def toggle_group_correction(self, group_item, which):
        """
        group_item: a top-level item in the light_tree
        which: either "cosmetic" or "pedestal"
        """
        old_text = group_item.text(4)  # e.g. "Cosmetic: True, Pedestal: False"
        # If there's nothing, default them to False
        if not old_text:
            old_text = "Cosmetic: False, Pedestal: False"

        # Parse
        # old_text might be "Cosmetic: True, Pedestal: False"
        # split by comma
        # part[0] => "Cosmetic: True"
        # part[1] => " Pedestal: False"
        parts = old_text.split(",")
        cosmetic_str = "False"
        pedestal_str = "False"
        if len(parts) == 2:
            # parse cosmetic
            cos_part = parts[0].split(":")[-1].strip()  # "True" or "False"
            cosmetic_str = cos_part
            # parse pedestal
            ped_part = parts[1].split(":")[-1].strip()
            pedestal_str = ped_part

        # Convert to bool
        cosmetic_bool = (cosmetic_str.lower() == "true")
        pedestal_bool = (pedestal_str.lower() == "true")

        # Toggle whichever was requested
        if which == "cosmetic":
            cosmetic_bool = not cosmetic_bool
        elif which == "pedestal":
            pedestal_bool = not pedestal_bool

        # Rebuild the new text
        new_text = f"Cosmetic: {str(cosmetic_bool)}, Pedestal: {str(pedestal_bool)}"
        group_item.setText(4, new_text)


    def calibrate_lights(self):
        """Performs calibration on selected light frames using Master Darks and Flats, considering overrides."""
        if not self.stacking_directory:
            QMessageBox.warning(self, "Error", "Please set the stacking directory first.")
            return

        calibrated_dir = os.path.join(self.stacking_directory, "Calibrated")
        os.makedirs(calibrated_dir, exist_ok=True)

        total_files = sum(len(files) for files in self.light_files.values())
        processed_files = 0

        master_bias_path = self.master_files.get("Bias", None)
        master_bias = None
        if master_bias_path:
            with fits.open(master_bias_path) as bias_hdul:
                master_bias = bias_hdul[0].data.astype(np.float32)
            self.update_status(f"Using Master Bias: {os.path.basename(master_bias_path)}")

        for i in range(self.light_tree.topLevelItemCount()):
            filter_item = self.light_tree.topLevelItem(i)
            filter_name = filter_item.text(0)

            for j in range(filter_item.childCount()):
                exposure_item = filter_item.child(j)
                exposure_text = exposure_item.text(0)

                # Get default corrections
                correction_text = exposure_item.text(4)
                apply_cosmetic = False
                apply_pedestal = False
                if correction_text:
                    parts = correction_text.split(",")
                    if len(parts) == 2:
                        apply_cosmetic = parts[0].split(":")[-1].strip().lower() == "true"
                        apply_pedestal = parts[1].split(":")[-1].strip().lower() == "true"

                pedestal_value = self.pedestal_spinbox.value() / 65535 if apply_pedestal else 0

                for k in range(exposure_item.childCount()):
                    leaf = exposure_item.child(k)
                    filename = leaf.text(0)
                    meta = leaf.text(1)

                    # Get session from metadata
                    session_name = "Default"
                    match = re.search(r"Session: ([^|]+)", meta)
                    if match:
                        session_name = match.group(1).strip()

                    # Look up the light file from session-specific group
                    composite_key = (f"{filter_name} - {exposure_text}", session_name)
                    light_file_list = self.light_files.get(composite_key, [])
                    light_file = next((f for f in light_file_list if os.path.basename(f) == filename), None)
                    if not light_file:
                        continue

                    # Determine size from header
                    header, _ = get_valid_header(light_file)
                    width = int(header.get("NAXIS1", 0))
                    height = int(header.get("NAXIS2", 0))
                    image_size = f"{width}x{height}"

                    # Determine Master Dark (manual override or best match)
                    manual_dark_key = f"{filter_name} - {exposure_text}"
                    master_dark_path = self.manual_dark_overrides.get(manual_dark_key)
                    if not master_dark_path:
                        for key, path in self.master_files.items():
                            if os.path.basename(path) == exposure_item.text(2):
                                master_dark_path = path
                                break

                    # Determine Master Flat (manual override or best session match)
                    manual_flat_key = f"{filter_name} - {exposure_text}"
                    master_flat_path = self.manual_flat_overrides.get(manual_flat_key)
                    if not master_flat_path:
                        flat_key = f"{filter_name} ({image_size}) [{session_name}]"
                        master_flat_path = self.master_files.get(flat_key)

                    self.update_status(f"Processing: {os.path.basename(light_file)}")
                    QApplication.processEvents()

                    light_data, hdr, bit_depth, is_mono = load_image(light_file)
                    if light_data is None or hdr is None:
                        self.update_status(f"❌ ERROR: Failed to load {os.path.basename(light_file)}")
                        continue

                    if not is_mono and light_data.shape[-1] == 3:
                        light_data = light_data.transpose(2, 0, 1)

                    if master_bias is not None:
                        if is_mono:
                            light_data -= master_bias
                        else:
                            light_data -= master_bias[np.newaxis, :, :]
                        self.update_status("Bias Subtracted")
                        QApplication.processEvents()

                    if master_dark_path:
                        dark_data, _, _, dark_is_mono = load_image(master_dark_path)
                        if dark_data is not None:
                            if not dark_is_mono and dark_data.shape[-1] == 3:
                                dark_data = dark_data.transpose(2, 0, 1)
                            light_data = subtract_dark_with_pedestal(
                                light_data[np.newaxis, :, :], dark_data, pedestal_value
                            )[0]
                            self.update_status(f"Dark Subtracted: {os.path.basename(master_dark_path)}")
                            QApplication.processEvents()

                    if master_flat_path:
                        flat_data, _, _, flat_is_mono = load_image(master_flat_path)
                        if flat_data is not None:
                            if not flat_is_mono and flat_data.shape[-1] == 3:
                                flat_data = flat_data.transpose(2, 0, 1)
                            flat_data[flat_data == 0] = 1.0
                            light_data = apply_flat_division_numba(light_data, flat_data)
                            self.update_status(f"Flat Applied: {os.path.basename(master_flat_path)}")
                            QApplication.processEvents()

                    if apply_cosmetic:
                        if hdr.get("BAYERPAT"):
                            light_data = bulk_cosmetic_correction_bayer(light_data)
                            self.update_status("Cosmetic Correction Applied for Bayer Pattern")
                        else:
                            light_data = bulk_cosmetic_correction_numba(light_data)
                            self.update_status("Cosmetic Correction Applied")
                        QApplication.processEvents()

                    if not is_mono and light_data.shape[0] == 3:
                        light_data = light_data.transpose(1, 2, 0)

                    min_val = light_data.min()
                    max_val = light_data.max()
                    self.update_status(f"Before saving: min = {min_val:.4f}, max = {max_val:.4f}")
                    print(f"Before saving: min = {min_val:.4f}, max = {max_val:.4f}")
                    QApplication.processEvents()

                    calibrated_filename = os.path.join(
                        calibrated_dir, os.path.basename(light_file).replace(".fit", "_c.fit")
                    )

                    save_image(
                        img_array=light_data,
                        filename=calibrated_filename,
                        original_format="fit",
                        bit_depth=bit_depth,
                        original_header=hdr,
                        is_mono=is_mono
                    )

                    processed_files += 1
                    self.update_status(f"Saved: {os.path.basename(calibrated_filename)} ({processed_files}/{total_files})")
                    QApplication.processEvents()

        self.update_status("✅ Calibration Complete!")
        QApplication.processEvents()
        self.populate_calibrated_lights()

    def extract_light_files_from_tree(self):
        """
        Walks self.reg_tree and rebuilds self.light_files as
        { group_key: [abs_path1, abs_path2, ...], ... }
        """
        new = {}
        for i in range(self.reg_tree.topLevelItemCount()):
            group = self.reg_tree.topLevelItem(i)
            key   = group.text(0)
            files = []

            # dive into exposure → leaf or direct leaf
            for j in range(group.childCount()):
                sub = group.child(j)
                leaves = []
                if sub.childCount()>0:
                    for k in range(sub.childCount()):
                        leaves.append(sub.child(k))
                else:
                    leaves.append(sub)

                for leaf in leaves:
                    fp = leaf.data(0, Qt.ItemDataRole.UserRole)
                    if fp and os.path.exists(fp):
                        files.append(fp)
                    else:
                        self.update_status(f"⚠️ WARNING: File not found: {fp}")
            if files:
                new[key] = files

        self.light_files = new
        total = sum(len(v) for v in new.values())
        self.update_status(f"✅ Extracted Light Files: {total} total")


    def select_reference_frame_robust(self, frame_weights, sigma_threshold=1.0):
        """
        Instead of sigma filtering, pick the frame at the 75th percentile of frame weights.
        This assumes that higher weights are better and that the 75th percentile represents
        a good-quality frame.
        
        Parameters
        ----------
        frame_weights : dict
            Mapping { file_path: weight_value } for each frame.
        
        Returns
        -------
        best_frame : str or None
            The file path of the chosen reference frame, or None if no frames are available.
        """
        items = list(frame_weights.items())  # List of (file_path, weight) pairs
        if not items:
            return None

        # Sort frames by weight in ascending order.
        items.sort(key=lambda x: x[1])
        n = len(items)
        # Get the index corresponding to the 75th percentile.
        index = int(0.75 * (n - 1))
        best_frame = items[index][0]
        return best_frame

    def prompt_for_reference_frame(self):
        new_ref, _ = QFileDialog.getOpenFileName(
            self,
            "Select New Reference Frame",
            "",  # default directory
            "FITS Files (*.fit *.fits);;All Files (*)"
        )
        return new_ref if new_ref else None

    def extract_light_files_from_tree(self, *, debug: bool = False):
        """
        Rebuild self.light_files from what's *currently shown* in reg_tree.
        - Only uses leaf items (childCount()==0)
        - Repairs missing leaf UserRole by matching basename against parent's cached list
        - Filters non-existent paths
        """
        light_files: dict[str, list[str]] = {}
        total_leafs = 0
        total_paths = 0

        for i in range(self.reg_tree.topLevelItemCount()):
            top = self.reg_tree.topLevelItem(i)
            group_key = top.text(0)
            repaired_from_parent = 0

            # Parent's cached list (may be stale but useful for repairing)
            parent_cached = top.data(0, Qt.ItemDataRole.UserRole) or []

            paths: list[str] = []
            for j in range(top.childCount()):
                leaf = top.child(j)
                # Only accept real leaf rows (no grandchildren expected in this tree)
                if leaf.childCount() != 0:
                    continue

                total_leafs += 1

                fp = leaf.data(0, Qt.ItemDataRole.UserRole)
                if not fp:
                    # Try to repair by basename match against parent's cached list
                    name = leaf.text(0).lstrip("⚠️ ").strip()
                    match = next((p for p in parent_cached if os.path.basename(p) == name), None)
                    if match:
                        leaf.setData(0, Qt.ItemDataRole.UserRole, match)
                        fp = match
                        repaired_from_parent += 1

                if fp and isinstance(fp, str) and os.path.exists(fp):
                    paths.append(fp)

            if paths:
                light_files[group_key] = paths
                # keep the parent cache in sync for future repairs
                top.setData(0, Qt.ItemDataRole.UserRole, paths)
                total_paths += len(paths)

            if debug:
                self.update_status(
                    f"⤴ {group_key}: {len(paths)} files"
                    + (f" (repaired {repaired_from_parent})" if repaired_from_parent else "")
                )

        self.light_files = light_files
        if debug:
            self.update_status(f"🧭 Tree snapshot → groups: {len(light_files)}, leaves seen: {total_leafs}, paths kept: {total_paths}")
        return light_files

    def _norm_filter_key(self, s: str) -> str:
        s = (s or "").lower()
        # map greek letters to ascii
        s = s.replace("α", "a").replace("β", "b")
        return re.sub(r"[^a-z0-9]+", "", s)

    def _classify_filter(self, filt_str: str) -> str:
        """
        Return one of:
        'DUAL_HA_OIII', 'DUAL_SII_OIII', 'DUAL_SII_HB',
        'MONO_HA', 'MONO_SII', 'MONO_OIII', 'MONO_HB',
        'UNKNOWN'
        """
        k = self._norm_filter_key(filt_str)
        comps = set()

        # explicit component tokens
        if "ha"    in k or "halpha" in k: comps.add("ha")
        if "sii"   in k or "s2"     in k: comps.add("sii")
        if "oiii"  in k or "o3"     in k: comps.add("oiii")
        if "hb"    in k or "hbeta"  in k: comps.add("hb")

        # common vendor aliases → Ha/OIII
        vendor_aliases = (
            "lextreme", "lenhance", "lultimate",
            "nbz", "nbzu", "alpt", "alp",
            "duo-band", "duoband", "dual band", "dual-band", "dualband"
        )
        if any(alias in k for alias in vendor_aliases):
            comps.update({"ha", "oiii"})

        # generic dual/duo/bicolor markers → assume Ha/OIII (most OSC duals)
        dual_markers = (
            "dual", "duo", "2band", "2-band", "two band",
            "bicolor", "bi-color", "bicolour", "bi-colour",
            "dualnb", "dual-nb", "duo-nb", "duonb",
            "duo narrow", "dual narrow"
        )
        if any(m in k for m in dual_markers):
            comps.update({"ha", "oiii"})

        # decide
        if {"ha","oiii"}.issubset(comps):  return "DUAL_HA_OIII"
        if {"sii","oiii"}.issubset(comps): return "DUAL_SII_OIII"
        if {"sii","hb"}.issubset(comps):   return "DUAL_SII_HB"

        if comps == {"ha"}:   return "MONO_HA"
        if comps == {"sii"}:  return "MONO_SII"
        if comps == {"oiii"}: return "MONO_OIII"
        if comps == {"hb"}:   return "MONO_HB"

        # NEW: if user explicitly asked to split dual-band, default to Ha/OIII
        try:
            if hasattr(self, "split_dualband_cb") and self.split_dualband_cb.isChecked():
                return "DUAL_HA_OIII"
        except Exception:
            pass

        return "UNKNOWN"

    def _get_filter_name(self, path: str) -> str:
        # Prefer FITS header 'FILTER'; fall back to filename tokens
        try:
            hdr = fits.getheader(path, ext=0)
            for key in ("FILTER", "FILTER1", "HIERARCH INDI FILTER", "HIERARCH ESO INS FILT1 NAME"):
                if key in hdr and str(hdr[key]).strip():
                    return str(hdr[key]).strip()
        except Exception:
            pass
        return os.path.basename(path)

    def _current_global_drizzle(self):
        # read from the “global” controls (used as a template)
        return {
            "enabled": self.drizzle_checkbox.isChecked(),
            "scale": float(self.drizzle_scale_combo.currentText().replace("x","", 1)),
            "drop": float(self.drizzle_drop_shrink_spin.value())
        }

    def _split_dual_band_osc(self, selected_groups=None):
        """
        Create mono Ha/SII/OIII frames from dual-band OSC files and
        update self.light_files so integration sees separate channels.
        """
        selected_groups = selected_groups or set()
        out_dir = os.path.join(self.stacking_directory, "DualBand_Split")
        os.makedirs(out_dir, exist_ok=True)

        ha_files, sii_files, oiii_files, hb_files = [], [], [], []
        inherit_map = {}                      # gk -> set(parent_group names)   # <<< NEW
        parent_of = {}                        # path -> parent_group            # <<< NEW

        # Walk all groups/files you already collected
        old_groups = list(self.light_files.items())
        old_drizzle = dict(self.per_group_drizzle)
        for group, files in old_groups:
            for fp in files:
                try:
                    img, hdr, _, _ = load_image(fp)
                    if img is None:
                        self.update_status(f"⚠️ Cannot load {fp}; skipping.")
                        continue

                    if hdr and hdr.get("BAYERPAT"):
                        img = self.debayer_image(img, fp, hdr)

                    # 3-channel split; otherwise treat mono via classifier
                    if img.ndim != 3 or img.shape[-1] < 2:
                        filt = self._get_filter_name(fp)
                        cls  = self._classify_filter(filt)
                        if cls == "MONO_HA":
                            ha_files.append(fp);   parent_of[fp] = group        # <<< NEW
                        elif cls == "MONO_SII":
                            sii_files.append(fp);  parent_of[fp] = group        # <<< NEW
                        elif cls == "MONO_OIII":
                            oiii_files.append(fp); parent_of[fp] = group        # <<< NEW
                        elif cls == "MONO_HB":   hb_files.append(fp);  parent_of[fp] = group        # <<< NEW
                        # else: leave in original groups
                        continue

                    filt = self._get_filter_name(fp)
                    cls  = self._classify_filter(filt)

                    R = img[..., 0]; G = img[..., 1]
                    base = os.path.splitext(os.path.basename(fp))[0]

                    if cls == "DUAL_HA_OIII":
                        ha_path   = os.path.join(out_dir, f"{base}_Ha.fit")
                        oiii_path = os.path.join(out_dir, f"{base}_OIII.fit")
                        self._write_band_fit(ha_path,  R, hdr, "Ha",  src_filter=filt)
                        self._write_band_fit(oiii_path, G, hdr, "OIII", src_filter=filt)
                        ha_files.append(ha_path);     parent_of[ha_path]   = group   # <<< NEW
                        oiii_files.append(oiii_path); parent_of[oiii_path] = group   # <<< NEW

                    elif cls == "DUAL_SII_OIII":
                        sii_path  = os.path.join(out_dir, f"{base}_SII.fit")
                        oiii_path = os.path.join(out_dir, f"{base}_OIII.fit")
                        self._write_band_fit(sii_path, R, hdr, "SII",  src_filter=filt)
                        self._write_band_fit(oiii_path, G, hdr, "OIII", src_filter=filt)
                        sii_files.append(sii_path);    parent_of[sii_path]  = group  # <<< NEW
                        oiii_files.append(oiii_path);  parent_of[oiii_path] = group  # <<< NEW

                    elif cls == "DUAL_SII_HB":  # NEW → R=SII, G=Hb  (G works well; we can add G+B later if you want)
                        sii_path = os.path.join(out_dir, f"{base}_SII.fit")
                        hb_path  = os.path.join(out_dir, f"{base}_Hb.fit")
                        self._write_band_fit(sii_path, R, hdr, "SII", src_filter=filt)
                        self._write_band_fit(hb_path,  G, hdr, "Hb",  src_filter=filt)
                        sii_files.append(sii_path); parent_of[sii_path] = group
                        hb_files.append(hb_path);   parent_of[hb_path]  = group

                    else:
                        pass

                except Exception as e:
                    self.update_status(f"⚠️ Split error on {os.path.basename(fp)}: {e}")

        # Group the new files
        def _group_key(band: str, path: str) -> str:
            try:
                h = fits.getheader(path, ext=0)
                exp = h.get("EXPTIME") or h.get("EXPOSURE") or ""
                w   = h.get("NAXIS1","?"); hgt = h.get("NAXIS2","?")
                exp_str = f"{float(exp):.1f}s" if isinstance(exp, (int,float)) else str(exp)
                return f"{band} - {exp_str} - {w}x{hgt}"
            except Exception:
                return f"{band} - ? - ?x?"

        new_groups = {}
        for band, flist in (("Ha", ha_files), ("SII", sii_files), ("OIII", oiii_files), ("Hb", hb_files)):  # NEW Hb
            for p in flist:
                gk = _group_key(band, p)
                new_groups.setdefault(gk, []).append(p)
                parent = parent_of.get(p)
                if parent:
                    inherit_map.setdefault(gk, set()).add(parent)

        if new_groups:
            self.light_files = new_groups

            # Seed drizzle for the new groups based on parents
            seeded = 0
            global_template = self._current_global_drizzle()   # make sure this helper exists
            self.per_group_drizzle = {}  # rebuild for the new groups

            for gk, parents in inherit_map.items():
                parent_cfgs = [old_drizzle.get(pg) for pg in parents if old_drizzle.get(pg)]
                chosen = None
                for cfg in parent_cfgs:
                    if cfg.get("enabled"):
                        chosen = cfg
                        break
                if not chosen and parent_cfgs:
                    chosen = parent_cfgs[0]

                if not chosen and (parents & selected_groups) and global_template.get("enabled"):
                    chosen = global_template

                if chosen:
                    self.per_group_drizzle[gk] = dict(chosen)
                    seeded += 1


            self.update_status(
                f"✅ Dual-band split complete: Ha={len(ha_files)}, SII={len(sii_files)}, "
                f"OIII={len(oiii_files)}, Hb={len(hb_files)} (drizzle seeded on {seeded} new group(s))"
            )
        else:
            self.update_status("ℹ️ No dual-band frames detected or split.")

    def _write_band_fit(self, out_path: str, data: np.ndarray, src_header: Optional[fits.Header],
                        band: str, src_filter: str):

        arr = np.ascontiguousarray(data.astype(np.float32))

        hdr = (src_header.copy() if isinstance(src_header, fits.Header) else fits.Header())

        # --- strip CFA/Bayer-related cards so we never try to debayer these ---
        cfa_like = (
            "BAYERPAT", "BAYER_PATTERN", "DEBAYER", "DEBAYERING", "DEMAT", "DEMOSAIC",
            "XBAYROFF", "YBAYROFF", "COLORTYP", "COLORSPACE", "HIERARCH CFA", "HIERARCH OSC",
            "HIERARCH ASI BAYERPATTERN", "HIERARCH DNG CFA", "HIERARCH ZWO CFA"
        )
        for k in list(hdr.keys()):
            kk = str(k).upper()
            if any(token in kk for token in ("BAYER", "CFA", "DEMOSA")) or kk in cfa_like:
                try:
                    del hdr[k]
                except Exception:
                    pass

        # Mark these as mono split files & set the band as the filter
        hdr["FILTER"] = (band, "Channel from dual-band split")
        hdr["SPLITDB"] = (True, "This frame was generated by dual-band splitting")
        hdr.add_history(f"Dual-band split: {band} from {src_filter}")

        fits.PrimaryHDU(data=arr, header=hdr).writeto(out_path, overwrite=True)

    def _drizzle_text_for_group(self, group_key: str) -> str:
        d = self.per_group_drizzle.get(group_key)
        if not d:
            return ""
        return f"Drizzle: {d.get('enabled', False)}, Scale: {d.get('scale','1x')}, Drop:{d.get('drop',0.65)}"

    def _refresh_reg_tree_from_light_files(self):
        self.reg_tree.clear()
        for group, files in self.light_files.items():
            top = QTreeWidgetItem([group, f"{len(files)} file(s)", self._drizzle_text_for_group(group)])
            self.reg_tree.addTopLevelItem(top)
            for fp in files:
                # Optional: show some header metadata
                meta = ""
                try:
                    hdr = fits.getheader(fp, ext=0)
                    filt = hdr.get("FILTER", "")
                    exp  = hdr.get("EXPTIME") or hdr.get("EXPOSURE") or ""
                    if isinstance(exp, (int, float)): exp = f"{exp:.1f}s"
                    meta = f"Filter={filt}  Exp={exp}"
                except Exception:
                    pass
                child = QTreeWidgetItem([os.path.basename(fp), meta, ""])
                top.addChild(child)
        self.reg_tree.expandAll()

    def _norm_ang(self, a):
        a = a % 360.0
        return a + 360.0 if a < 0 else a

    def _angdiff(self, a, b):
        # smallest absolute difference in degrees
        return abs((self._norm_ang(a) - self._norm_ang(b) + 180.0) % 360.0 - 180.0)

    def _extract_pa_deg(self, hdr):
        """
        Try common FITS keys for camera/sky position angle.
        Fallback: estimate from WCS CD/PC matrix (CROTA2-ish).
        Returns float degrees or None.
        """
        if hdr is None:
            return None
        keys = ("POSANGLE","ANGLE","ROTANGLE","ROTSKYPA","ROTATOR",
                "PA","CROTA2","CROTA1")
        for k in keys:
            if k in hdr:
                try:
                    return float(hdr[k])
                except Exception:
                    pass
        # crude WCS fallback (angle of +Y axis on detector)
        try:
            cd11 = float(hdr.get('CD1_1', hdr.get('PC1_1')))
            cd12 = float(hdr.get('CD1_2', hdr.get('PC1_2')))
            cd22 = float(hdr.get('CD2_2', hdr.get('PC2_2')))
            # common CROTA2-style estimate
            pa = np.degrees(np.arctan2(-cd12, cd22))
            return float(pa)
        except Exception:
            return None

    def _maybe_rot180(self, img, pa_cur, pa_ref, tol_deg):
        """
        If |(pa_cur - pa_ref)| ≈ 180° (within tol), rotate image 180°.
        Works for (H,W) or (H,W,3).
        Returns (img_out, rotated_bool).
        """
        
        if pa_cur is None or pa_ref is None:
            return img, False
        d = self._angdiff(pa_cur, pa_ref)
        if abs(d - 180.0) <= tol_deg:
            # 180° is just two 90° rotations; cheap & exact
            self.update_status(f"Flipping Image")
            QApplication.processEvents()
            return np.rot90(img, 2).copy(), True
        return img, False


    def register_images(self):
        """Measure → choose reference → DEBAYER ref → DEBAYER+normalize all → align."""
        if self.star_trail_mode:
            self.update_status("🌠 Star-Trail Mode enabled: skipping registration & using max-value stack")
            QApplication.processEvents()
            return self._make_star_trail()

        self.update_status("🔄 Image Registration Started...")
        self.extract_light_files_from_tree(debug=True)

        if not self.light_files:
            self.update_status("⚠️ No light files to register!")
            return

        # Which groups are selected? (used for optional dual-band split)
        selected_groups = set()
        for it in self.reg_tree.selectedItems():
            top = it if it.parent() is None else it.parent()
            selected_groups.add(top.text(0))

        if self.split_dualband_cb.isChecked():
            self.update_status("🌈 Splitting dual-band OSC frames into Ha / SII / OIII...")
            self._split_dual_band_osc(selected_groups=selected_groups)
            self._refresh_reg_tree_from_light_files()

        # Flatten to get all files
        all_files = [f for lst in self.light_files.values() for f in lst]
        self.update_status(f"📊 Found {len(all_files)} total frames. Now measuring in parallel batches...")

        # ─────────────────────────────────────────────────────────────────────
        # Helpers
        # ─────────────────────────────────────────────────────────────────────
        def mono_preview_for_stats(img: np.ndarray, hdr) -> np.ndarray:
            """
            2D float32 preview without full demosaic:
            - If color (HxWx3): return luma.
            - If raw CFA 2D with BAYERPAT: 2x2 superpixel average.
            - Else: return mono as float32.
            """
            if img is None:
                return None
            if img.ndim == 3 and img.shape[-1] == 3:
                img = img.astype(np.float32, copy=False)
                r = img[..., 0]; g = img[..., 1]; b = img[..., 2]
                return 0.2126 * r + 0.7152 * g + 0.0722 * b
            if hdr and hdr.get('BAYERPAT') and not hdr.get('SPLITDB', False) and img.ndim == 2:
                h, w = img.shape
                h2, w2 = h - (h % 2), w - (w % 2)
                cfa = img[:h2, :w2].astype(np.float32, copy=False)
                return (cfa[0:h2:2, 0:w2:2] + cfa[0:h2:2, 1:w2:2] +
                        cfa[1:h2:2, 0:w2:2] + cfa[1:h2:2, 1:w2:2]) * 0.25
            return img.astype(np.float32, copy=False)

        def chunk_list(lst, size):
            for i in range(0, len(lst), size):
                yield lst[i:i+size]

        # ─────────────────────────────────────────────────────────────────────
        # PHASE 1: measure (NO demosaic here)
        # ─────────────────────────────────────────────────────────────────────
        self.frame_weights = {}
        mean_values = {}
        star_counts = {}
        measured_frames = []

        max_workers = os.cpu_count() or 4
        chunk_size = max_workers
        chunks = list(chunk_list(all_files, chunk_size))
        total_chunks = len(chunks)

        from concurrent.futures import ThreadPoolExecutor, as_completed

        for idx, chunk in enumerate(chunks, 1):
            self.update_status(f"📦 Measuring chunk {idx}/{total_chunks} ({len(chunk)} frames)")
            chunk_images = []
            chunk_valid_files = []

            self.update_status(f"🌍 Loading {len(chunk)} images in parallel (up to {max_workers} threads)...")
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_file = {executor.submit(load_image, fp): fp for fp in chunk}
                for fut in as_completed(future_to_file):
                    fp = future_to_file[fut]
                    try:
                        img, hdr, _, _ = fut.result()
                        if img is None:
                            continue
                        preview = mono_preview_for_stats(img, hdr)
                        if preview is None:
                            continue
                        chunk_images.append(preview)
                        chunk_valid_files.append(fp)
                        self.update_status(f"  Loaded {fp}")
                    except Exception as e:
                        self.update_status(f"⚠️ Error loading {fp}: {e}")
                    QApplication.processEvents()

            if not chunk_images:
                self.update_status("⚠️ No valid images in this chunk.")
                continue

            self.update_status("🌍 Measuring global means in parallel...")
            means = parallel_measure_frames(chunk_images)  # expects list/stack of 2D previews

            # star counts per preview
            for i, fp in enumerate(chunk_valid_files):
                mv = float(means[i])
                mean_values[fp] = mv
                c, ecc = compute_star_count(chunk_images[i])  # 2D preview
                star_counts[fp] = {"count": c, "eccentricity": ecc}
                measured_frames.append(fp)

            del chunk_images

        if not measured_frames:
            self.update_status("⚠️ No frames could be measured!")
            return

        self.update_status(f"✅ All chunks complete! Measured {len(measured_frames)} frames total.")

        # ─────────────────────────────────────────────────────────────────────
        # Pick reference & compute weights
        # ─────────────────────────────────────────────────────────────────────
        self.update_status("⚖️ Computing frame weights...")
        debug_log = "\n📊 **Frame Weights Debug Log:**\n"
        for fp in measured_frames:
            c = star_counts[fp]["count"]
            ecc = star_counts[fp]["eccentricity"]
            m = mean_values[fp]
            c = max(c, 1)
            m = max(m, 1e-6)
            raw_w = (c * min(1.0, max(1.0 - ecc, 0.0))) / m
            self.frame_weights[fp] = raw_w
            debug_log += f"📂 {os.path.basename(fp)} → StarCount={c}, Ecc={ecc:.4f}, Mean={m:.4f}, Weight={raw_w:.4f}\n"
        self.update_status(debug_log)

        max_w = max(self.frame_weights.values()) if self.frame_weights else 0.0
        if max_w > 0:
            for k in self.frame_weights:
                self.frame_weights[k] /= max_w

        # Choose reference (path)
        if getattr(self, "reference_frame", None):
            self.update_status(f"📌 Using user-specified reference: {self.reference_frame}")
        else:
            self.reference_frame = self.select_reference_frame_robust(self.frame_weights, sigma_threshold=2.0)
            self.update_status(f"📌 Auto-selected robust reference frame: {self.reference_frame}")

        # Stats for the chosen reference from the measurement pass
        ref_stats_meas = star_counts.get(self.reference_frame, {"count": 0, "eccentricity": 0.0})
        ref_count = ref_stats_meas["count"]
        ref_ecc   = ref_stats_meas["eccentricity"]

        # ─────────────────────────────────────────────────────────────────────
        # Debayer the reference ONCE and compute ref_median from debayered ref
        # ─────────────────────────────────────────────────────────────────────
        ref_img_raw, ref_hdr, _, _ = load_image(self.reference_frame)
        if ref_img_raw is None:
            self.update_status(f"🚨 Could not load reference {self.reference_frame}. Aborting.")
            return

        # If CFA, debayer; if already color, keep; if mono but 3D with last=1, squeeze.
        if ref_hdr and ref_hdr.get('BAYERPAT') and not ref_hdr.get('SPLITDB', False) and (ref_img_raw.ndim == 2 or (ref_img_raw.ndim == 3 and ref_img_raw.shape[-1] == 1)):
            self.update_status("📦 Debayering reference frame…")
            ref_img = self.debayer_image(ref_img_raw, self.reference_frame, ref_hdr)  # HxWx3
        else:
            ref_img = ref_img_raw
            if ref_img.ndim == 3 and ref_img.shape[-1] == 1:
                ref_img = np.squeeze(ref_img, axis=-1)

        # Use luma median if color, else direct median
        if ref_img.ndim == 3 and ref_img.shape[-1] == 3:
            r, g, b = ref_img[..., 0], ref_img[..., 1], ref_img[..., 2]
            ref_luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
            ref_median = float(np.median(ref_luma))
        else:
            ref_median = float(np.median(ref_img))

        self.update_status(f"📊 Reference (debayered) median: {ref_median:.4f}")

        # Show review dialog; if user changes reference, redo debayer+median
        stats_payload = {"star_count": ref_count, "eccentricity": ref_ecc, "mean": ref_median}
        dialog = ReferenceFrameReviewDialog(self.reference_frame, stats_payload, parent=self)

        # Make it non-modal but raised/focused
        dialog.setModal(False)
        dialog.setWindowModality(Qt.WindowModality.NonModal)
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

        # Wait here without freezing UI (modeless pseudo-modal)
        _loop = QEventLoop(self)
        dialog.finished.connect(_loop.quit)   # finished(int) -> quit loop
        _loop.exec()

        # After the user closes the dialog, proceed exactly as before:
        result = dialog.result()
        user_choice = dialog.getUserChoice()   # "use", "select_other", or None

        if result == QDialog.DialogCode.Accepted:
            self.update_status("User accepted the reference frame.")
        elif user_choice == "select_other":
            new_ref = self.prompt_for_reference_frame()
            if new_ref:
                self.reference_frame = new_ref
                self.update_status(f"User selected a new reference frame: {new_ref}")
                # re-load and debayer/median the new reference
                ref_img_raw, ref_hdr, _, _ = load_image(self.reference_frame)
                if ref_img_raw is None:
                    self.update_status(f"🚨 Could not load reference {self.reference_frame}. Aborting.")
                    return
                if ref_hdr and ref_hdr.get('BAYERPAT') and not ref_hdr.get('SPLITDB', False) and (ref_img_raw.ndim == 2 or (ref_img_raw.ndim == 3 and ref_img_raw.shape[-1] == 1)):
                    self.update_status("📦 Debayering reference frame…")
                    ref_img = self.debayer_image(ref_img_raw, self.reference_frame, ref_hdr)
                else:
                    ref_img = ref_img_raw
                    if ref_img.ndim == 3 and ref_img.shape[-1] == 1:
                        ref_img = np.squeeze(ref_img, axis=-1)
                if ref_img.ndim == 3 and ref_img.shape[-1] == 3:
                    r, g, b = ref_img[..., 0], ref_img[..., 1], ref_img[..., 2]
                    ref_luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
                    ref_median = float(np.median(ref_luma))
                else:
                    ref_median = float(np.median(ref_img))
                self.update_status(f"📊 (New) reference median: {ref_median:.4f}")
            else:
                self.update_status("No new reference selected; using previous reference.")
        else:
            self.update_status("Dialog closed without explicit choice; using selected reference.")


        # ─────────────────────────────────────────────────────────────────────
        # PHASE 1b: Meridian flips
        # ─────────────────────────────────────────────────────────────────────
        ref_pa = self._extract_pa_deg(ref_hdr)
        self.update_status(f"🧭 Reference PA: {ref_pa:.2f}°" if ref_pa is not None else "🧭 Reference PA: (unknown)")

        # ─────────────────────────────────────────────────────────────────────
        # PHASE 2: normalize (DEBAYER everything once here)
        # ─────────────────────────────────────────────────────────────────────
        norm_dir = os.path.join(self.stacking_directory, "Normalized_Images")
        os.makedirs(norm_dir, exist_ok=True)

        normalized_files = []
        chunks = list(chunk_list(measured_frames, chunk_size))
        total_chunks = len(chunks)

        for idx, chunk in enumerate(chunks, 1):
            self.update_status(f"🌀 Normalizing chunk {idx}/{total_chunks} ({len(chunk)} frames)…")
            QApplication.processEvents()

            loaded_images = []
            valid_paths = []

            self.update_status(f"🌍 Loading {len(chunk)} images in parallel for normalization (up to {max_workers} threads)…")
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_file = {executor.submit(load_image, fp): fp for fp in chunk}
                for fut in as_completed(future_to_file):
                    fp = future_to_file[fut]
                    try:
                        img, hdr, _, _ = fut.result()
                        if img is None:
                            self.update_status(f"⚠️ No data for {fp}")
                            continue

                        # Debayer ONCE here if CFA
                        if hdr and hdr.get('BAYERPAT') and not hdr.get('SPLITDB', False) and (img.ndim == 2 or (img.ndim == 3 and img.shape[-1] == 1)):
                            img = self.debayer_image(img, fp, hdr)  # → HxWx3
                        else:
                            if img.ndim == 3 and img.shape[-1] == 1:
                                img = np.squeeze(img, axis=-1)

                        # --- Meridian flip assist: auto 180° rotate by header PA ---
                        if self.auto_rot180 and ref_pa is not None:
                            pa = self._extract_pa_deg(hdr)
                            img, did = self._maybe_rot180(img, pa, ref_pa, self.auto_rot180_tol_deg)
                            if did:
                                self.update_status(f"↻ 180° rotate (PA Δ≈180°): {os.path.basename(fp)}")
                                # Optional: mark header so we remember we pre-rotated
                                QApplication.processEvents()
                                try:
                                    hdr['ROT180'] = (True, 'Rotated 180° pre-align by SAS')
                                except Exception:
                                    pass

                        loaded_images.append(img)
                        valid_paths.append(fp)
                    except Exception as e:
                        self.update_status(f"⚠️ Error loading {fp} for normalization: {e}")
                    QApplication.processEvents()

            if not loaded_images:
                continue

            stack = np.array(loaded_images, dtype=np.float32)  # (F,H,W) or (F,H,W,3)
            normalized_stack = normalize_images(stack, ref_median)

            # Save each with "_n.fit"
            for i, orig_file in enumerate(valid_paths):
                base = os.path.basename(orig_file)
                # normalize suffix handling for .fit/.fits
                if base.endswith("_n.fit"):
                    base = base.replace("_n.fit", ".fit")
                if base.lower().endswith(".fits"):
                    out_name = base[:-5] + "_n.fit"
                elif base.lower().endswith(".fit"):
                    out_name = base[:-4] + "_n.fit"
                else:
                    out_name = base + "_n.fit"

                out_path = os.path.join(norm_dir, out_name)
                frame_data = normalized_stack[i]

                try:
                    orig_header = fits.getheader(orig_file, ext=0)
                except Exception:
                    orig_header = fits.Header()

                # Mark representation to avoid re-debayering later
                if isinstance(frame_data, np.ndarray) and frame_data.ndim == 3 and frame_data.shape[-1] == 3:
                    orig_header["DEBAYERED"] = (True, "Color debayered normalized")
                else:
                    orig_header["DEBAYERED"] = (False, "Mono normalized")

                hdu = fits.PrimaryHDU(data=frame_data.astype(np.float32), header=orig_header)
                hdu.writeto(out_path, overwrite=True)
                normalized_files.append(out_path)

            del loaded_images, stack, normalized_stack

        # Update self.light_files to *_n.fit
        for group, file_list in self.light_files.items():
            new_list = []
            for old_path in file_list:
                base = os.path.basename(old_path)
                if base.endswith("_n.fit"):
                    new_list.append(os.path.join(norm_dir, base))
                else:
                    if base.lower().endswith(".fits"):
                        n_name = base[:-5] + "_n.fit"
                    elif base.lower().endswith(".fit"):
                        n_name = base[:-4] + "_n.fit"
                    else:
                        n_name = base + "_n.fit"
                    new_list.append(os.path.join(norm_dir, n_name))
            self.light_files[group] = new_list

        self.update_status("✅ Updated self.light_files to use debayered, normalized *_n.fit frames.")

        # Pick normalized reference path to align against
        ref_base = os.path.basename(self.reference_frame)
        if ref_base.endswith("_n.fit"):
            norm_ref_path = os.path.join(norm_dir, ref_base)
        else:
            if ref_base.lower().endswith(".fits"):
                norm_ref_base = ref_base[:-5] + "_n.fit"
            elif ref_base.lower().endswith(".fit"):
                norm_ref_base = ref_base[:-4] + "_n.fit"
            else:
                norm_ref_base = ref_base + "_n.fit"
            norm_ref_path = os.path.join(norm_dir, norm_ref_base)

        # ─────────────────────────────────────────────────────────────────────
        # Start alignment on the normalized files
        # ─────────────────────────────────────────────────────────────────────
        align_dir = os.path.join(self.stacking_directory, "Aligned_Images")
        os.makedirs(align_dir, exist_ok=True)

        passes = self.settings.value("stacking/refinement_passes", 3, type=int)
        shift_tol = self.settings.value("stacking/shift_tolerance", 0.2, type=float)

        self.alignment_thread = StarRegistrationThread(
            norm_ref_path,
            normalized_files,
            align_dir,
            max_refinement_passes=passes,
            shift_tolerance=shift_tol,
            parent_window=self
        )
        self.alignment_thread.progress_update.connect(self.update_status)
        self.alignment_thread.registration_complete.connect(self.on_registration_complete)

        self.align_progress = QProgressDialog("Aligning stars…", None, 0, 0, self)
        self.align_progress.setWindowModality(Qt.WindowModality.WindowModal)
        self.align_progress.setMinimumDuration(0)
        self.align_progress.setCancelButton(None)
        self.align_progress.setWindowTitle("Stellar Alignment")
        self.align_progress.setValue(0)
        self.align_progress.show()

        self.alignment_thread.progress_step.connect(self._on_align_progress)
        self.alignment_thread.registration_complete.connect(self._on_align_done)
        self.alignment_thread.start()

        
    @pyqtSlot(int, int)
    def _on_align_progress(self, done, total):
        self.align_progress.setLabelText(f"Aligning stars… ({done}/{total})")
        self.align_progress.setMaximum(total)
        self.align_progress.setValue(done)
        QApplication.processEvents()

    @pyqtSlot(bool, str)
    def _on_align_done(self, success, message):
        if hasattr(self, "align_progress"):
            self.align_progress.close()
            del self.align_progress
        self.update_status(message)

    def save_alignment_matrices_sasd(self, transforms_dict):
        out_path = os.path.join(self.stacking_directory, "alignment_transforms.sasd")
        try:
            with open(out_path, "w") as f:
                for norm_path, matrix in transforms_dict.items():
                    # Use the original normalized input path (e.g., *_n.fit)
                    orig_path = os.path.normpath(norm_path)

                    a, b, tx = matrix[0]
                    c, d, ty = matrix[1]

                    f.write(f"FILE: {orig_path}\n")
                    f.write("MATRIX:\n")
                    f.write(f"{a:.4f}, {b:.4f}, {tx:.4f}\n")
                    f.write(f"{c:.4f}, {d:.4f}, {ty:.4f}\n")
                    f.write("\n")  # blank line
            self.update_status(f"✅ Transform file saved as {os.path.basename(out_path)}")
        except Exception as e:
            self.update_status(f"⚠️ Failed to save transform file: {e}")



    def load_alignment_matrices_custom(self, file_path):

        transforms = {}
        with open(file_path, "r") as f:
            content = f.read()

        blocks = re.split(r"\n\s*\n", content.strip())

        for block in blocks:
            lines = [line.strip() for line in block.splitlines() if line.strip()]
            if not lines:
                continue
            if lines[0].startswith("FILE:"):
                raw_file_path = lines[0].replace("FILE:", "").strip()
                # *** KEY FIX: normalize here
                curr_file = os.path.normpath(raw_file_path)
            else:
                continue
            
            if len(lines) < 4 or not lines[1].startswith("MATRIX:"):
                continue

            row0 = lines[2].split(",")
            row1 = lines[3].split(",")
            a, b, tx = [float(x) for x in row0]
            c, d, ty = [float(x) for x in row1]

            transforms[curr_file] = np.array([[a, b, tx],
                                            [c, d, ty]], dtype=np.float32)
        return transforms

    def _make_star_trail(self):
        # 1) collect all your calibrated light frames
        all_files = [f for flist in self.light_files.values() for f in flist]
        n_frames = len(all_files)
        if not all_files:
            self.update_status("⚠️ No calibrated lights available for star trails.")
            return

        # 2) load every frame (once), compute its median, and remember its header
        frames: list[tuple[np.ndarray, fits.Header]] = []
        medians: list[float] = []

        for fn in all_files:
            img, hdr, _, _ = load_image(fn)
            if img is None:
                self.update_status(f"⚠️ Failed to load {os.path.basename(fn)}; skipping")
                QApplication.processEvents()
                continue

            arr = img.astype(np.float32)
            medians.append(float(np.median(arr)))
            frames.append((arr, hdr))

        if not frames:
            self.update_status("⚠️ No valid frames to compute reference median; aborting star-trail.")
            return

        # reference median is the median of per-frame medians
        ref_median = float(np.median(medians))

        # grab the header from the first valid frame, strip out extra NAXIS keywords
        first_hdr = frames[0][1]
        if first_hdr is not None:
            hdr_to_use = first_hdr.copy()
            for key in list(hdr_to_use):
                if key.startswith("NAXIS") and key not in ("NAXIS", "NAXIS1", "NAXIS2"):
                    hdr_to_use.pop(key, None)
        else:
            hdr_to_use = None

        # 3) normalize each frame and write to a temp dir
        with tempfile.TemporaryDirectory(prefix="startrail_norm_") as norm_dir:
            normalized_paths = []
            for idx, (arr, hdr) in enumerate(frames, start=1):
                self.update_status(f"🔄 Normalizing frame {idx}/{len(frames)}")
                QApplication.processEvents()

                # guard against divide-by-zero
                m = float(np.median(arr))
                scale = ref_median / (m + 1e-12)
                img_norm = arr * scale

                stem = Path(all_files[idx-1]).stem
                out_path = os.path.join(norm_dir, f"{stem}_st.fit")
                fits.PrimaryHDU(data=img_norm, header=hdr).writeto(out_path, overwrite=True)
                normalized_paths.append(out_path)

            # 4) stack and do max-value projection
            self.update_status(f"📊 Stacking {len(normalized_paths)} frames")
            QApplication.processEvents()
            stack = np.stack([fits.getdata(p).astype(np.float32) for p in normalized_paths], axis=0)
            trail_img, _ = max_value_stack(stack)

            # 5) stretch final image and prompt user for save location & format
            trail_img = trail_img.astype(np.float32)
            # normalize to [0–1] for our save helper
            trail_norm = trail_img / (trail_img.max() + 1e-12)

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_name = f"StarTrail_{n_frames:03d}frames_{ts}"
            filters = "TIFF (*.tif);;PNG (*.png);;JPEG (*.jpg *.jpeg);;FITS (*.fits);;XISF (*.xisf')"
            path, chosen_filter = QFileDialog.getSaveFileName(
                self,
                "Save Star-Trail Image",
                os.path.join(self.stacking_directory, default_name),
                filters
            )
            if not path:
                self.update_status("✖ Star-trail save cancelled.")
                return

            # figure out extension
            ext = os.path.splitext(path)[1].lower().lstrip('.')
            if not ext:
                ext = chosen_filter.split('(')[1].split(')')[0].lstrip('*.').lower()
                path += f".{ext}"

            # if user picked FITS, supply the first frame’s header; else None
            use_hdr = hdr_to_use if ext in ('fits', 'fit') else None

            # 16-bit everywhere
            save_image(
                img_array=trail_norm,
                filename=path,
                original_format=ext,
                bit_depth="16-bit",
                original_header=use_hdr,
                is_mono=False
            )

        # once we exit the with-block, all the _st.fit files are deleted
        self.update_status(f"✅ Star‐Trail image written to {path}")
        return


    def _apply_autocrop(self, arr, file_list, header, scale=1.0, rect_override=None):
        """
        If rect_override is provided, use it; else compute per-file_list.
        """
        try:
            enabled = self.autocrop_cb.isChecked()
            pct = float(self.autocrop_pct.value())
        except Exception:
            enabled = self.settings.value("stacking/autocrop_enabled", False, type=bool)
            pct = float(self.settings.value("stacking/autocrop_pct", 95.0, type=float))

        if not enabled or not file_list:
            return arr, header

        rect = rect_override
        if rect is None:
            transforms_path = os.path.join(self.stacking_directory, "alignment_transforms.sasd")
            rect = self._compute_autocrop_rect(file_list, transforms_path, pct)

        if not rect:
            self.update_status("✂️ Auto-crop: no common area found; skipping.")
            return arr, header

        x0, y0, x1, y1 = rect
        if scale != 1.0:
            # scale rect to drizzle resolution
            x0 = int(math.floor(x0 * scale))
            y0 = int(math.floor(y0 * scale))
            x1 = int(math.ceil (x1 * scale))
            y1 = int(math.ceil (y1 * scale))

        # Clamp to image bounds
        H, W = arr.shape[:2]
        x0 = max(0, min(W, x0)); x1 = max(x0, min(W, x1))
        y0 = max(0, min(H, y0)); y1 = max(y0, min(H, y1))

        # --- Crop while preserving channels ---
        if arr.ndim == 2:
            arr = arr[y0:y1, x0:x1]
        else:
            arr = arr[y0:y1, x0:x1, :]
            # If this is actually mono stored as (H,W,1), squeeze back to (H,W)
            if arr.shape[-1] == 1:
                arr = arr[..., 0]

        # Update header dims (+ shift CRPIX if present)
        if header is None:
            header = fits.Header()

        # NAXIS / sizes consistent with the new array
        if arr.ndim == 2:
            header["NAXIS"]  = 2
            header["NAXIS1"] = arr.shape[1]
            header["NAXIS2"] = arr.shape[0]
            # Remove any stale NAXIS3
            if "NAXIS3" in header:
                del header["NAXIS3"]
        else:
            header["NAXIS"]  = 3
            header["NAXIS1"] = arr.shape[1]
            header["NAXIS2"] = arr.shape[0]
            header["NAXIS3"] = arr.shape[2]

        if "CRPIX1" in header:
            header["CRPIX1"] = float(header["CRPIX1"]) - x0
        if "CRPIX2" in header:
            header["CRPIX2"] = float(header["CRPIX2"]) - y0

        self.update_status(f"✂️ Auto-cropped to [{x0}:{x1}]×[{y0}:{y1}] (scale {scale}×)")
        return arr, header


    def on_registration_complete(self, success, msg):
        self.update_status(msg)
        if not success:
            return

        alignment_thread = self.alignment_thread
        if alignment_thread is None:
            self.update_status("⚠️ Error: No alignment data available.")
            return

        # ----------------------------
        # Gather results from the thread
        # ----------------------------
        all_transforms = dict(alignment_thread.alignment_matrices)  # {orig_norm_path -> 2x3 or None}
        keys = list(all_transforms.keys())

        # Build a per-file shift map (last pass), defaulting to 0 when missing
        shift_map = {}
        if alignment_thread.transform_deltas and alignment_thread.transform_deltas[-1]:
            last = alignment_thread.transform_deltas[-1]
            for i, k in enumerate(keys):
                if i < len(last):
                    shift_map[k] = float(last[i])
                else:
                    shift_map[k] = 0.0
        else:
            shift_map = {k: 0.0 for k in keys}

        # Fast mode if only 1 pass was requested
        fast_mode = (getattr(alignment_thread, "max_refinement_passes", 3) <= 1)
        # Threshold is only used in normal mode
        accept_thresh = float(self.settings.value("stacking/accept_shift_px", 2.0, type=float))

        def _accept(k: str) -> bool:
            """Accept criteria for a frame."""
            if all_transforms.get(k) is None:
                # real failure (e.g., astroalign couldn't find a transform)
                return False
            if fast_mode:
                # In fast mode we keep everything that didn't fail
                return True
            # Normal (multi-pass) behavior: keep small last-pass shifts
            return shift_map.get(k, 0.0) <= accept_thresh

        accepted = [k for k in keys if _accept(k)]
        rejected = [k for k in keys if not _accept(k)]

        # ----------------------------
        # Persist numeric transforms we accepted (for drizzle, etc.)
        # ----------------------------
        valid_matrices = {k: all_transforms[k] for k in accepted}
        self.save_alignment_matrices_sasd(valid_matrices)

        # ----------------------------
        # Build mapping from normalized -> aligned paths
        # Use the *actual* final paths produced by the thread.
        # ----------------------------
        final_map = alignment_thread.file_key_to_current_path  # {orig_norm_path -> final_aligned_path}
        self.valid_transforms = {
            os.path.normpath(k): os.path.normpath(final_map[k])
            for k in accepted
            if k in final_map and os.path.exists(final_map[k])
        }

        # finalize alignment phase
        self.alignment_thread = None

        # Status
        prefix = "⚡ Fast mode: " if fast_mode else ""
        self.update_status(f"{prefix}Alignment summary: {len(accepted)} succeeded, {len(rejected)} rejected.")
        if (not fast_mode) and rejected:
            self.update_status(f"🚨 Rejected {len(rejected)} frame(s) due to shift > {accept_thresh}px.")
            for rf in rejected:
                self.update_status(f"  ❌ {os.path.basename(rf)}")

        if not self.valid_transforms:
            self.update_status("⚠️ No frames to stack; aborting.")
            return

        # ----------------------------
        # Build aligned file groups (unchanged)
        # ----------------------------
        filtered_light_files = {}
        for group, file_list in self.light_files.items():
            filtered = [f for f in file_list if os.path.normpath(f) in self.valid_transforms]
            filtered_light_files[group] = filtered
            self.update_status(f"Group '{group}' has {len(filtered)} file(s) after filtering.")

        aligned_light_files = {}
        for group, file_list in filtered_light_files.items():
            new_list = []
            for f in file_list:
                normed = os.path.normpath(f)
                aligned = self.valid_transforms.get(normed)
                if aligned and os.path.exists(aligned):
                    new_list.append(aligned)
                else:
                    self.update_status(f"DEBUG: File '{aligned}' does not exist on disk.")
            aligned_light_files[group] = new_list

        # ----------------------------
        # Snapshot UI-dependent settings
        # ----------------------------
        drizzle_dict = self.gather_drizzle_settings_from_tree()
        try:
            autocrop_enabled = self.autocrop_cb.isChecked()
            autocrop_pct = float(self.autocrop_pct.value())
        except Exception:
            autocrop_enabled = self.settings.value("stacking/autocrop_enabled", False, type=bool)
            autocrop_pct = float(self.settings.value("stacking/autocrop_pct", 95.0, type=float))

        # ----------------------------
        # Kick off post-align worker (unchanged)
        # ----------------------------
        self.post_thread = QThread(self)
        self.post_worker = AfterAlignWorker(
            self,
            light_files=aligned_light_files,
            frame_weights=dict(self.frame_weights),
            transforms_dict=dict(self.valid_transforms),
            drizzle_dict=drizzle_dict,
            autocrop_enabled=autocrop_enabled,
            autocrop_pct=autocrop_pct,
        )

        self.post_worker.progress.connect(self._on_post_status)
        self.post_worker.finished.connect(self._on_post_pipeline_finished)

        self.post_worker.moveToThread(self.post_thread)
        self.post_thread.started.connect(self.post_worker.run)
        self.post_thread.start()

        self.post_progress = QProgressDialog("Stacking & drizzle (if enabled)…", None, 0, 0, self)
        self.post_progress.setWindowModality(Qt.WindowModality.WindowModal)
        self.post_progress.setCancelButton(None)
        self.post_progress.setMinimumDuration(0)
        self.post_progress.setWindowTitle("Post-Alignment")
        self.post_progress.show()


    @pyqtSlot(bool, str)
    def _on_post_pipeline_finished(self, ok: bool, message: str):
        try:
            if getattr(self, "post_progress", None):
                self.post_progress.close()
                self.post_progress = None
        except Exception:
            pass

        try:
            self.post_thread.quit()
            self.post_thread.wait()
        except Exception:
            pass
        try:
            self.post_worker.deleteLater()
            self.post_thread.deleteLater()
        except Exception:
            pass

        self.update_status(message)


    def save_rejection_map_sasr(self, rejection_map, out_file):
        """
        Writes the per-file rejection map to a custom text file.
        Format:
            FILE: path/to/file1
            x1, y1
            x2, y2

            FILE: path/to/file2
            ...
        """
        with open(out_file, "w") as f:
            for fpath, coords_list in rejection_map.items():
                f.write(f"FILE: {fpath}\n")
                for (x, y) in coords_list:
                    # Convert to Python int in case they're NumPy int64
                    f.write(f"{int(x)}, {int(y)}\n")
                f.write("\n")  # blank line to separate blocks

    def load_rejection_map_sasr(self, in_file):
        """
        Reads a .sasr text file and rebuilds the rejection map dictionary.
        Returns a dict { fpath: [(x, y), (x, y), ...], ... }
        """
        rejections = {}
        with open(in_file, "r") as f:
            content = f.read().strip()

        # Split on blank lines
        blocks = re.split(r"\n\s*\n", content)
        for block in blocks:
            lines = [line.strip() for line in block.splitlines() if line.strip()]
            if not lines:
                continue

            # First line should be 'FILE: <path>'
            if lines[0].startswith("FILE:"):
                raw_path = lines[0].replace("FILE:", "").strip()
                coords = []
                for line in lines[1:]:
                    # Each subsequent line is "x, y"
                    parts = line.split(",")
                    if len(parts) == 2:
                        x_str, y_str = parts
                        x = int(x_str.strip())
                        y = int(y_str.strip())
                        coords.append((x, y))
                rejections[raw_path] = coords
        return rejections

    def stack_images_mixed_drizzle(
        self,
        grouped_files,           # { group_key: [aligned _n_r.fit paths] }
        frame_weights,           # { file_path: weight }
        transforms_dict,         # { normalized_path -> aligned_path } (kept for compatibility)
        drizzle_dict,            # { group_key: {drizzle_enabled, scale_factor, drop_shrink} }
        *,
        autocrop_enabled: bool,
        autocrop_pct: float,
        status_cb=None
    ):
        """Runs normal integration (to get rejection coords), saves masters,
        and (optionally) runs drizzle. Designed to run in a worker thread.

        Returns:
            {
            "summary_lines": [str, ...],
            "autocrop_outputs": [(group_key, out_path_crop), ...]
            }
        """
        log = status_cb or (lambda *_: None)

        n_groups = len(grouped_files)
        n_frames = sum(len(v) for v in grouped_files.values())
        log(f"📁 Post-align: {n_groups} group(s), {n_frames} aligned frame(s).")

        # Precompute a single global crop rect if enabled (pure computation, no UI).
        global_rect = None
        if autocrop_enabled:
            t0 = perf_counter()
            log("✂️ Auto-crop: computing common bounding box…")
            global_rect = self._compute_common_autocrop_rect(grouped_files, autocrop_pct,
                                                            status_cb=log)  # ← pass logger in
            dt = perf_counter() - t0
            if global_rect is None:
                log(f"✂️ Auto-crop: no common box (took {dt:.1f}s) — will crop per group.")
            else:
                log(f"✂️ Auto-crop: bounding box ready (took {dt:.1f}s).")

        group_integration_data = {}
        summary_lines = []
        autocrop_outputs = []

        for gi, (group_key, file_list) in enumerate(grouped_files.items(), 1):
            t_g = perf_counter()
            log(f"🔹 [{gi}/{n_groups}] Integrating '{group_key}' with {len(file_list)} file(s)…")

            integrated_image, rejection_map, ref_header = self.normal_integration_with_rejection(
                group_key, file_list, frame_weights, status_cb=log  # ensure inner loop emits!
            )
            log(f"   ↳ Integration done in {perf_counter() - t_g:.1f}s.")
            if integrated_image is None:
                continue

            if ref_header is None:
                ref_header = fits.Header()

            # --- Save the non-cropped master ---
            hdr_orig = ref_header.copy()
            hdr_orig["IMAGETYP"] = "MASTER STACK"
            hdr_orig["BITPIX"]   = -32
            hdr_orig["STACKED"]  = (True, "Stacked using normal_integration_with_rejection")
            hdr_orig["CREATOR"]  = "SetiAstroSuite"
            hdr_orig["DATE-OBS"] = datetime.utcnow().isoformat()

            is_mono_orig = (integrated_image.ndim == 2)
            if is_mono_orig:
                hdr_orig["NAXIS"]  = 2
                hdr_orig["NAXIS1"] = integrated_image.shape[1]
                hdr_orig["NAXIS2"] = integrated_image.shape[0]
                if "NAXIS3" in hdr_orig:
                    del hdr_orig["NAXIS3"]
            else:
                hdr_orig["NAXIS"]  = 3
                hdr_orig["NAXIS1"] = integrated_image.shape[1]
                hdr_orig["NAXIS2"] = integrated_image.shape[0]
                hdr_orig["NAXIS3"] = integrated_image.shape[2]

            n_frames = len(file_list)
            H, W = integrated_image.shape[:2]
            display_group = self._label_with_dims(group_key, W, H)
            base_name = f"MasterLight_{display_group}_{n_frames}stacked"
            out_path_orig = os.path.join(self.stacking_directory, f"{base_name}.fit")

            save_image(
                img_array=integrated_image,
                filename=out_path_orig,
                original_format="fit",
                bit_depth="32-bit floating point",
                original_header=hdr_orig,
                is_mono=is_mono_orig
            )
            log(f"✅ Saved integrated image (original) for '{group_key}': {out_path_orig}")

            # --- Optional: auto-cropped copy (uses global_rect if provided) ---
            if autocrop_enabled:
                cropped_img, hdr_crop = self._apply_autocrop(
                    integrated_image,
                    file_list,
                    ref_header.copy(),
                    scale=1.0,
                    rect_override=global_rect
                )
                is_mono_crop = (cropped_img.ndim == 2)
                Hc, Wc = (cropped_img.shape[:2] if cropped_img.ndim >= 2 else (H, W))
                display_group_crop = self._label_with_dims(group_key, Wc, Hc)
                base_name_crop = f"MasterLight_{display_group_crop}_{n_frames}stacked"
                out_path_crop = os.path.join(self.stacking_directory, f"{base_name_crop}_autocrop.fit")

                save_image(
                    img_array=cropped_img,
                    filename=out_path_crop,
                    original_format="fit",
                    bit_depth="32-bit floating point",
                    original_header=hdr_crop,
                    is_mono=is_mono_crop
                )
                log(f"✂️ Saved auto-cropped image for '{group_key}': {out_path_crop}")
                autocrop_outputs.append((group_key, out_path_crop))

            # Bookkeeping for drizzle
            dconf = drizzle_dict.get(group_key, {})
            if dconf.get("drizzle_enabled", False):
                sasr_path = os.path.join(self.stacking_directory, f"{group_key}_rejections.sasr")
                self.save_rejection_map_sasr(rejection_map, sasr_path)
                log(f"✅ Saved rejection map to {sasr_path}")
                group_integration_data[group_key] = {
                    "integrated_image": integrated_image,
                    "rejection_map": rejection_map,
                    "n_frames": n_frames,
                    "drizzled": True
                }
            else:
                group_integration_data[group_key] = {
                    "integrated_image": integrated_image,
                    "rejection_map": None,
                    "n_frames": n_frames,
                    "drizzled": False
                }
                log(f"ℹ️ Skipping rejection map save for '{group_key}' (drizzle disabled).")

        # Drizzle pass (only for groups with drizzle enabled)
        for group_key, file_list in grouped_files.items():
            dconf = drizzle_dict.get(group_key)
            if not (dconf and dconf.get("drizzle_enabled", False)):
                log(f"✅ Group '{group_key}' not set for drizzle. Integrated image already saved.")
                continue

            scale_factor = float(dconf["scale_factor"])
            drop_shrink  = float(dconf["drop_shrink"])
            rejections_for_group = group_integration_data[group_key]["rejection_map"]
            n_frames = group_integration_data[group_key]["n_frames"]

            log(f"📐 Drizzle for '{group_key}' at {scale_factor}× (drop={drop_shrink}) using {n_frames} frame(s).")

            self.drizzle_stack_one_group(
                group_key=group_key,
                file_list=file_list,
                transforms_dict=transforms_dict,   # kept for compatibility; method reloads from disk
                frame_weights=frame_weights,
                scale_factor=scale_factor,
                drop_shrink=drop_shrink,
                rejection_map=rejections_for_group,
                autocrop_enabled=autocrop_enabled,
                rect_override=global_rect,
                status_cb=log
            )

        # Build summary lines
        for group_key, info in group_integration_data.items():
            n_frames = info["n_frames"]
            drizzled = info["drizzled"]
            summary_lines.append(f"• {group_key}: {n_frames} stacked{' + drizzle' if drizzled else ''}")

        if autocrop_outputs:
            summary_lines.append("")
            summary_lines.append("Auto-cropped files saved:")
            for g, p in autocrop_outputs:
                summary_lines.append(f"  • {g} → {p}")

        return {
            "summary_lines": summary_lines,
            "autocrop_outputs": autocrop_outputs
        }

    def save_registered_images(self, success, msg, frame_weights):
        if not success:
            self.update_status(f"⚠️ Image registration failed: {msg}")
            return

        self.update_status("✅ All frames registered successfully!")
        
        # Use the grouped files already stored from the tree view.
        if not self.light_files:
            self.update_status("⚠️ No light frames available for stacking!")
            return
        
        self.update_status(f"📂 Preparing to stack {sum(len(v) for v in self.light_files.values())} frames in {len(self.light_files)} groups.")
        
        # Pass the dictionary (grouped by filter, exposure, dimensions) to the stacking function.
        self.stack_registered_images(self.light_files, frame_weights)


    def stack_registered_images_chunked(
        self,
        grouped_files,           # dict of { group_key: [list_of_aligned_and_already_normalized_file_paths] }
        frame_weights,           # dict of { file_path: weight }
        chunk_height=2048,
        chunk_width=2048
    ):
        """
        Chunked stacking of already-aligned and pre-normalized FITS images.
        Reads small tiles from each image, applies outlier rejection (using the new rejection-map output),
        writes the result into a memory-mapped array, and saves a final stacked FITS.
        """
        self.update_status(f"✅ Chunked stacking {len(grouped_files)} group(s)...")

        # We'll also accumulate a list of rejected pixel positions (global coordinates)
        all_rejection_coords = []

        for group_key, file_list in grouped_files.items():
            num_files = len(file_list)
            self.update_status(f"📊 Group '{group_key}' has {num_files} aligned file(s).")
            QApplication.processEvents()

            if num_files < 2:
                self.update_status(f"⚠️ Group '{group_key}' does not have enough frames to stack.")
                continue

            # 1) Identify the reference file to get shape and header
            ref_file = file_list[0]
            if not os.path.exists(ref_file):
                self.update_status(f"⚠️ Reference file '{ref_file}' not found, skipping group.")
                continue

            ref_data, ref_header, _, _ = load_image(ref_file)
            if ref_data is None:
                self.update_status(f"⚠️ Could not load reference '{ref_file}', skipping group.")
                continue

            is_color = (ref_data.ndim == 3 and ref_data.shape[2] == 3)
            height, width = ref_data.shape[:2]
            channels = 3 if is_color else 1

            # 2) Prepare a memmap for the final stacked image.
            memmap_path = os.path.join(self.stacking_directory, f"chunked_{group_key}.dat")
            final_stacked = np.memmap(
                memmap_path,
                dtype=np.float32,
                mode='w+',
                shape=(height, width, channels)
            )

            # Build list of valid files and corresponding weights.
            aligned_paths = []
            weights_list = []
            for fpath in file_list:
                if os.path.exists(fpath):
                    aligned_paths.append(fpath)
                    w = frame_weights.get(fpath, 1.0)
                    weights_list.append(w)
                else:
                    self.update_status(f"⚠️ File not found: {fpath}, skipping.")

            if len(aligned_paths) < 2:
                self.update_status(f"⚠️ Not enough valid frames in group '{group_key}' to stack.")
                continue

            weights_list = np.array(weights_list, dtype=np.float32)
            self.update_status(f"📊 Stacking group '{group_key}' with {self.rejection_algorithm}")
            QApplication.processEvents()

            # Initialize a list to collect rejected pixel coordinates for this group.
            rejection_coords = []
            N = len(aligned_paths)
            DTYPE    = self._dtype()
            pref_h   = self.chunk_height
            pref_w   = self.chunk_width

            # 3) Compute a safe chunk size once, up front
            try:
                chunk_h, chunk_w = compute_safe_chunk(
                    height, width, N, channels, DTYPE, pref_h, pref_w
                )
                self.update_status(f"🔧 Using chunk size {chunk_h}×{chunk_w} for {self._dtype()}")
            except MemoryError as e:
                self.update_status(f"⚠️ {e}")
                return None, {}, None

            # 3) Loop over tiles
            from concurrent.futures import ThreadPoolExecutor, as_completed
            for y_start in range(0, height, chunk_height):
                y_end = min(y_start + chunk_height, height)
                tile_h = y_end - y_start

                for x_start in range(0, width, chunk_width):
                    x_end = min(x_start + chunk_width, width)
                    tile_w = x_end - x_start

                    # Build tile stack: shape (N, tile_h, tile_w, channels)
                    N = len(aligned_paths)
                 
                    tile_stack = np.zeros((N, tile_h, tile_w, channels), dtype=np.float32)
                    num_cores = os.cpu_count() or 4
                    with ThreadPoolExecutor(max_workers=num_cores) as executor:
                        future_to_index = {}
                        for i, path in enumerate(aligned_paths):
                            future = executor.submit(load_fits_tile, path, y_start, y_end, x_start, x_end)
                            future_to_index[future] = i

                        for future in as_completed(future_to_index):
                            i = future_to_index[future]
                            sub_img = future.result()
                            if sub_img is None:
                                continue
                            # Ensure sub_img is shaped (tile_h, tile_w, channels)
                            if sub_img.ndim == 2:
                                sub_img = sub_img[:, :, np.newaxis]
                                if channels == 3:
                                    sub_img = np.repeat(sub_img, 3, axis=2)
                            elif sub_img.ndim == 3 and sub_img.shape[0] == 3 and channels == 3:
                                sub_img = sub_img.transpose(1, 2, 0)
                            sub_img = sub_img.astype(np.float32, copy=False)
                            tile_stack[i] = sub_img

                    # 4) Apply the chosen rejection algorithm and get the rejection map.
                    algo = self.rejection_algorithm
                    if algo == "Simple Median (No Rejection)":
                        tile_result = np.median(tile_stack, axis=0)
                        tile_rej_map = np.zeros(tile_stack.shape[1:], dtype=np.bool_)
                    elif algo == "Simple Average (No Rejection)":
                        tile_result = np.average(tile_stack, axis=0, weights=weights_list)
                        tile_rej_map = np.zeros(tile_stack.shape[1:], dtype=np.bool_)
                    elif algo == "Weighted Windsorized Sigma Clipping":
                        tile_result, tile_rej_map = windsorized_sigma_clip_weighted(tile_stack, weights_list,
                            lower=self.sigma_low, upper=self.sigma_high)
                    elif algo == "Kappa-Sigma Clipping":
                        tile_result, tile_rej_map = kappa_sigma_clip_weighted(tile_stack, weights_list,
                            kappa=self.kappa, iterations=self.iterations)
                    elif algo == "Trimmed Mean":
                        tile_result, tile_rej_map = trimmed_mean_weighted(tile_stack, weights_list,
                            trim_fraction=self.trim_fraction)
                    elif algo == "Extreme Studentized Deviate (ESD)":
                        tile_result, tile_rej_map = esd_clip_weighted(tile_stack, weights_list,
                            threshold=self.esd_threshold)
                    elif algo == "Biweight Estimator":
                        tile_result, tile_rej_map = biweight_location_weighted(tile_stack, weights_list,
                            tuning_constant=self.biweight_constant)
                    elif algo == "Modified Z-Score Clipping":
                        tile_result, tile_rej_map = modified_zscore_clip_weighted(tile_stack, weights_list,
                            threshold=self.modz_threshold)
                    elif algo == "Max Value":
                        tile_result, tile_rej_map = max_value_stack(tile_stack, weights_list)
                    else:
                        tile_result, tile_rej_map = windsorized_sigma_clip_weighted(tile_stack, weights_list,
                            lower=self.sigma_low, upper=self.sigma_high)

                    # 5) Insert integrated tile into final image.
                    final_stacked[y_start:y_end, x_start:x_end, :] = tile_result

                    # 6) Use the returned tile_rej_map to record rejected pixel positions.
                    # For rejection maps with per-frame output, combine along the frame axis.
                    if tile_rej_map.ndim == 3:  # mono: (N, tile_h, tile_w)
                        combined_rej = np.any(tile_rej_map, axis=0)  # shape: (tile_h, tile_w)
                    elif tile_rej_map.ndim == 4:  # color: (N, tile_h, tile_w, channels)
                        # First combine along the frame axis, then across channels.
                        combined_rej = np.any(tile_rej_map, axis=0)  # shape: (tile_h, tile_w, channels)
                        combined_rej = np.any(combined_rej, axis=-1)  # shape: (tile_h, tile_w)
                    else:
                        combined_rej = np.zeros(tile_stack.shape[1:3], dtype=np.bool_)

                    ys_tile, xs_tile = np.where(combined_rej)
                    for dx, dy in zip(xs_tile, ys_tile):
                        global_x = x_start + dx
                        global_y = y_start + dy
                        rejection_coords.append((global_x, global_y))

            # 7) After processing all tiles, finish up the integrated image.
            final_array = np.array(final_stacked)
            del final_stacked

            # Apply a black-point offset and scale if needed.
            flat_array = final_array.ravel()
            nonzero_indices = np.where(flat_array > 0)[0]
            if nonzero_indices.size > 0:
                first_nonzero = flat_array[nonzero_indices[0]]
                final_array -= first_nonzero

            new_max = final_array.max()
            if new_max > 1.0:
                new_min = final_array.min()
                range_val = new_max - new_min
                if range_val != 0:
                    final_array = (final_array - new_min) / range_val
                else:
                    final_array = np.zeros_like(final_array, dtype=np.float32)

            if final_array.ndim == 3 and final_array.shape[-1] == 1:
                final_array = final_array[..., 0]
            is_mono = (final_array.ndim == 2)

            # 8) Save the final stacked image.
            if ref_header is None:
                ref_header = fits.Header()

            ref_header["IMAGETYP"] = "MASTER STACK"
            ref_header["BITPIX"] = -32
            ref_header["STACKED"] = (True, "Stacked using chunked approach")
            ref_header["CREATOR"] = "SetiAstroSuite"
            ref_header["DATE-OBS"] = datetime.utcnow().isoformat()

            if is_mono:
                ref_header["NAXIS"] = 2
                ref_header["NAXIS1"] = final_array.shape[1]
                ref_header["NAXIS2"] = final_array.shape[0]
            else:
                ref_header["NAXIS"] = 3
                ref_header["NAXIS1"] = final_array.shape[1]
                ref_header["NAXIS2"] = final_array.shape[0]
                ref_header["NAXIS3"] = 3

            output_filename = f"MasterLight_{group_key}_{len(aligned_paths)}stacked.fit"
            output_path = os.path.join(self.stacking_directory, output_filename)
            save_image(
                img_array=final_array,
                filename=output_path,
                original_format="fit",
                bit_depth="32-bit floating point",
                original_header=ref_header,
                is_mono=is_mono
            )

            self.update_status(f"✅ Group '{group_key}' stacked {len(aligned_paths)} frame(s)! Saved: {output_path}")

            print(f"✅ Master Light saved for group '{group_key}': {output_path}")

            # Optionally, you might want to store or log 'rejection_coords' (here appended to all_rejection_coords)
            all_rejection_coords.extend(rejection_coords)

            # Clean up memmap file
            try:
                os.remove(memmap_path)
            except OSError:
                pass

        # Optionally, you could return the global rejection coordinate list.
        return all_rejection_coords

        QMessageBox.information(
            self,
            "Stacking Complete",
            f"All stacking finished successfully.\n"
            f"Frames per group:\n" +
            "\n".join([f"{group_key}: {len(files)} frame(s)" for group_key, files in grouped_files.items()])
        )



    def integrate_registered_images(self):
        """ 
        Integrates previously registered images (already aligned) without re-aligning them,
        but uses a chunked measurement approach so we don't load all frames at once.
        """
        self.update_status("🔄 Integrating Previously Registered Images...")

        # 1) Extract files from the registration tree
        self.extract_light_files_from_tree()
        if not self.light_files:
            self.update_status("⚠️ No registered images found!")
            return

        # Flatten the dictionary to get all registered files
        all_files = [f for file_list in self.light_files.values() for f in file_list]
        if not all_files:
            self.update_status("⚠️ No frames found in the registration tree!")
            return

        # 2) We'll measure means + star counts in chunks, so we don't load everything at once
        self.update_status(f"📊 Found {len(all_files)} total aligned frames. Measuring in parallel batches...")

        self.frame_weights = {}
        mean_values = {}
        star_counts = {}
        measured_frames = []

        # Decide how many images to load at once. Typically # of CPU cores:
        max_workers = os.cpu_count() or 4
        chunk_size = max_workers  # or a custom formula if you prefer

        def chunk_list(lst, size):
            for i in range(0, len(lst), size):
                yield lst[i : i + size]

        chunked_files = list(chunk_list(all_files, chunk_size))
        total_chunks = len(chunked_files)

        # 3) Process each chunk
        chunk_index = 0
        for chunk in chunked_files:
            chunk_index += 1
            self.update_status(f"📦 Loading and measuring chunk {chunk_index}/{total_chunks} with {len(chunk)} frames...")
            QApplication.processEvents()

            # Load this chunk of images
            images = []
            valid_files_for_this_chunk = []
            for file in chunk:
                image_data, _, _, _ = load_image(file)
                if image_data is not None:
                    images.append(image_data)
                    valid_files_for_this_chunk.append(file)
                else:
                    self.update_status(f"⚠️ Could not load {file}, skipping.")

            if not images:
                self.update_status("⚠️ No valid images in this chunk.")
                continue

            # Parallel measure the mean pixel value
            self.update_status("🌍 Measuring global statistics (mean) in parallel...")
            QApplication.processEvents()
            means = parallel_measure_frames(images)

            # Now measure star counts
            for i, file in enumerate(valid_files_for_this_chunk):
                mean_signal = means[i]
                mean_values[file] = mean_signal
                measured_frames.append(file)

                self.update_status(f"⭐ Measuring star stats for {file}...")
                QApplication.processEvents()
                count, ecc = compute_star_count(images[i])
                star_counts[file] = {"count": count, "eccentricity": ecc}

            # Clear the images from memory before moving on
            del images

        # If we never measured any frames at all
        if not measured_frames:
            self.update_status("⚠️ No frames could be measured!")
            return

        self.update_status(f"✅ All chunks complete! Measured {len(measured_frames)} frames total.")
        QApplication.processEvents()

        # 4) Compute Weights
        self.update_status("⚖️ Computing frame weights...")

        debug_weight_log = "\n📊 **Frame Weights Debug Log:**\n"
        QApplication.processEvents()
        for file in measured_frames:
            c = star_counts[file]["count"]
            ecc = star_counts[file]["eccentricity"]
            mean_val = mean_values[file]

            star_weight = max(c, 1e-6)
            mean_weight = max(mean_val, 1e-6)

            # Basic ratio-based weight: star_count / mean
            raw_weight = star_weight / mean_weight
            self.frame_weights[file] = raw_weight

            debug_weight_log += (
                f"📂 {os.path.basename(file)} → "
                f"Star Count: {c}, Mean: {mean_val:.4f}, Final Weight: {raw_weight:.4f}\n"
            )
            QApplication.processEvents()

        self.update_status(debug_weight_log)
        self.update_status("✅ Frame weights computed!")
        QApplication.processEvents()

        # 5) Pick the best reference frame if not user-specified
        if hasattr(self, "reference_frame") and self.reference_frame:
            self.update_status(f"📌 Using user-specified reference frame: {self.reference_frame}")
            QApplication.processEvents()
        else:
            self.reference_frame = max(self.frame_weights, key=self.frame_weights.get)
            self.update_status(f"📌 Auto-selected reference frame: {self.reference_frame} (Best Weight)")
            
        chunk_h = self.chunk_height  # or self.settings.value("stacking/chunk_height", 1024, type=int)
        chunk_w = self.chunk_width   # or self.settings.value("stacking/chunk_width", 1024, type=int)

        # 6) Finally, call the chunked stacking method using the already registered images
        self.stack_registered_images_chunked(self.light_files, self.frame_weights, chunk_height=chunk_h, chunk_width=chunk_w)

    @staticmethod
    def invert_affine_transform(matrix):
        """
        Inverts a 2x3 affine transformation matrix.
        Given matrix = [[a, b, tx],
                        [c, d, ty]],
        returns the inverse matrix.
        """
        A = matrix[:, :2]
        t = matrix[:, 2]
        A_inv = np.linalg.inv(A)
        t_inv = -A_inv @ t
        inv = np.hstack([A_inv, t_inv.reshape(2, 1)])
        return inv

    @staticmethod
    def apply_affine_transform_point(matrix, x, y):
        """
        Applies a 2x3 affine transformation to a point (x, y).
        Returns the transformed (x, y) coordinates.
        """
        point = np.array([x, y])
        result = matrix[:, :2] @ point + matrix[:, 2]
        return result[0], result[1]

    def drizzle_stack_one_group(
        self,
        group_key,
        file_list,
        transforms_dict,   # kept for API compatibility; transforms reloaded from disk
        frame_weights,
        scale_factor=2.0,
        drop_shrink=0.65,
        rejection_map=None,
        *,
        autocrop_enabled: bool = False,
        rect_override=None,
        status_cb=None
    ):
        """
        Drizzle a single group. Skips only per-file rejected pixels.
        Designed to run in a worker thread (no UI calls).
        """
        log = status_cb or (lambda *_: None)

        total_rej = sum(len(v) for v in (rejection_map or {}).values())
        log(f"🔭 Drizzle stacking for group '{group_key}' with {total_rej} total rejected pixels.")

        if len(file_list) < 2:
            log(f"⚠️ Group '{group_key}' does not have enough frames to drizzle.")
            return

        transforms_path = os.path.join(self.stacking_directory, "alignment_transforms.sasd")
        if not os.path.exists(transforms_path):
            log(f"⚠️ No alignment_transforms.sasd found at {transforms_path}!")
            return

        new_transforms_dict = self.load_alignment_matrices_custom(transforms_path)
        log(f"✅ Loaded {len(new_transforms_dict)} transforms from disk for drizzle.")

        first_file = file_list[0]
        first_img, hdr, _, _ = load_image(first_file)
        if first_img is None:
            log(f"⚠️ Could not load {first_file} to determine drizzle shape!")
            return

        if first_img.ndim == 2:
            is_mono = True
            h, w = first_img.shape
            c = 1
        else:
            is_mono = False
            h, w, c = first_img.shape

        # Choose depositor
        if drop_shrink >= 0.99:
            deposit_func = drizzle_deposit_numba_naive if is_mono else drizzle_deposit_color_naive
            log(f"Using naive drizzle deposit ({'mono' if is_mono else 'color'}).")
        else:
            deposit_func = drizzle_deposit_numba_footprint if is_mono else drizzle_deposit_color_footprint
            log(f"Using footprint drizzle deposit ({'mono' if is_mono else 'color'}).")

        out_h = int(h * scale_factor)
        out_w = int(w * scale_factor)
        drizzle_buffer  = np.zeros((out_h, out_w) if is_mono else (out_h, out_w, c), dtype=self._dtype())
        coverage_buffer = np.zeros_like(drizzle_buffer, dtype=self._dtype())
        finalize_func   = finalize_drizzle_2d if is_mono else finalize_drizzle_3d

        for aligned_file in file_list:
            aligned_base = os.path.basename(aligned_file)
            raw_base = aligned_base.replace("_n_r.fit", "_n.fit") if aligned_base.endswith("_n_r.fit") else aligned_base
            raw_file = os.path.join(self.stacking_directory, "Normalized_Images", raw_base)

            raw_img_data, _, _, _ = load_image(raw_file)
            if raw_img_data is None:
                log(f"⚠️ Could not load raw file '{raw_file}' for drizzle!")
                continue

            raw_key = os.path.normpath(raw_file)
            transform = new_transforms_dict.get(raw_key, None)
            if transform is None:
                log(f"⚠️ No transform found for raw '{raw_base}'! Skipping drizzle.")
                continue

            log(f"🧩 Drizzling (raw): {raw_base}")
            log(
                f"    Matrix: [[{transform[0,0]:.4f}, {transform[0,1]:.4f}, {transform[0,2]:.4f}], "
                f"[{transform[1,0]:.4f}, {transform[1,1]:.4f}, {transform[1,2]:.4f}]]"
            )

            weight = frame_weights.get(aligned_file, 1.0)
            if transform.dtype != np.float32:
                transform = transform.astype(np.float32)

            coords_for_this_file = rejection_map.get(aligned_file, []) if rejection_map else []

            if coords_for_this_file:
                inv_transform = self.invert_affine_transform(transform)
                for (x_r, y_r) in coords_for_this_file:
                    x_raw, y_raw = self.apply_affine_transform_point(inv_transform, x_r, y_r)
                    x_raw = int(round(x_raw))
                    y_raw = int(round(y_raw))
                    if 0 <= x_raw < raw_img_data.shape[1] and 0 <= y_raw < raw_img_data.shape[0]:
                        raw_img_data[y_raw, x_raw] = 0.0

            drizzle_buffer, coverage_buffer = deposit_func(
                raw_img_data, transform, drizzle_buffer, coverage_buffer,
                scale_factor, drop_shrink, weight
            )

        final_drizzle = np.zeros_like(drizzle_buffer, dtype=np.float32)
        final_drizzle = finalize_func(drizzle_buffer, coverage_buffer, final_drizzle)

        # Save original drizzle
        Hd, Wd = final_drizzle.shape[:2] if final_drizzle.ndim >= 2 else (0, 0)
        display_group_driz = self._label_with_dims(group_key, Wd, Hd)
        base_name = f"MasterLight_{display_group_driz}_{len(file_list)}stacked_drizzle"
        out_path_orig = os.path.join(self.stacking_directory, f"{base_name}.fit")

        hdr_orig = hdr.copy() if hdr is not None else fits.Header()
        hdr_orig["IMAGETYP"]   = "MASTER STACK - DRIZZLE"
        hdr_orig["DRIZFACTOR"] = (float(scale_factor), "Drizzle scale factor")
        hdr_orig["DROPFRAC"]   = (float(drop_shrink),  "Drizzle drop shrink/pixfrac")
        hdr_orig["CREATOR"]    = "SetiAstroSuite"
        hdr_orig["DATE-OBS"]   = datetime.utcnow().isoformat()

        if final_drizzle.ndim == 2:
            hdr_orig["NAXIS"]  = 2
            hdr_orig["NAXIS1"] = final_drizzle.shape[1]
            hdr_orig["NAXIS2"] = final_drizzle.shape[0]
            if "NAXIS3" in hdr_orig:
                del hdr_orig["NAXIS3"]
        else:
            hdr_orig["NAXIS"]  = 3
            hdr_orig["NAXIS1"] = final_drizzle.shape[1]
            hdr_orig["NAXIS2"] = final_drizzle.shape[0]
            hdr_orig["NAXIS3"] = final_drizzle.shape[2]

        save_image(
            img_array=final_drizzle,
            filename=out_path_orig,
            original_format="fit",
            bit_depth="32-bit floating point",
            original_header=hdr_orig,
            is_mono=(final_drizzle.ndim == 2)
        )
        log(f"✅ Drizzle (original) saved: {out_path_orig}")

        # Optional auto-crop (respects global rect if provided)
        if autocrop_enabled:
            cropped_drizzle, hdr_crop = self._apply_autocrop(
                final_drizzle,
                file_list,
                hdr.copy() if hdr is not None else fits.Header(),
                scale=float(scale_factor),
                rect_override=rect_override
            )
            is_mono_crop = (cropped_drizzle.ndim == 2)
            display_group_driz_crop = self._label_with_dims(group_key, cropped_drizzle.shape[1], cropped_drizzle.shape[0])
            base_name_crop = f"MasterLight_{display_group_driz_crop}_{len(file_list)}stacked_drizzle"
            out_path_crop = os.path.join(self.stacking_directory, f"{base_name_crop}_autocrop.fit")

            save_image(
                img_array=cropped_drizzle,
                filename=out_path_crop,
                original_format="fit",
                bit_depth="32-bit floating point",
                original_header=hdr_crop,
                is_mono=is_mono_crop
            )
            if not hasattr(self, "_autocrop_outputs"):
                self._autocrop_outputs = []
            self._autocrop_outputs.append((group_key, out_path_crop))
            log(f"✂️ Drizzle (auto-cropped) saved: {out_path_crop}")

    def normal_integration_with_rejection(self, group_key, file_list, frame_weights, status_cb=None):
        """
        Chunked integration of aligned (_n_r) images with outlier rejection.
        Returns: (integrated_image, per_file_rejections, ref_header)
        """
        log = status_cb or (lambda *_: None)

        log(f"Starting integration for group '{group_key}' with {len(file_list)} files.")

        if not file_list:
            log(f"DEBUG: Empty file_list for group '{group_key}'.")
            return None, {}, None

        ref_file = file_list[0]
        if not os.path.exists(ref_file):
            log(f"⚠️ Reference file '{ref_file}' not found for group '{group_key}'.")
            return None, {}, None

        ref_data, ref_header, _, _ = load_image(ref_file)
        if ref_data is None:
            log(f"⚠️ Could not load reference '{ref_file}' for group '{group_key}'.")
            return None, {}, None
        if ref_header is None:
            ref_header = fits.Header()

        is_color = (ref_data.ndim == 3 and ref_data.shape[2] == 3)
        height, width = ref_data.shape[:2]
        channels = 3 if is_color else 1

        log(f"📊 Stacking group '{group_key}' with {self.rejection_algorithm}")

        N = len(file_list)
        integrated_image = np.zeros((height, width, channels), dtype=self._dtype())
        per_file_rejections = {f: [] for f in file_list}

        DTYPE  = self._dtype()
        pref_h = self.chunk_height
        pref_w = self.chunk_width
        try:
            chunk_h, chunk_w = compute_safe_chunk(
                height, width, N, channels, DTYPE, pref_h, pref_w
            )
            log(f"🔧 Using chunk size {chunk_h}×{chunk_w} for {self._dtype()}")
        except MemoryError as e:
            log(f"⚠️ {e}")
            return None, {}, None

        n_rows  = math.ceil(height / chunk_h)
        n_cols  = math.ceil(width  / chunk_w)
        total_tiles = n_rows * n_cols
        tile_idx = 0

        from concurrent.futures import ThreadPoolExecutor, as_completed

        for y_start in range(0, height, chunk_h):
            y_end  = min(y_start + chunk_h, height)
            tile_h = y_end - y_start

            for x_start in range(0, width, chunk_w):
                x_end  = min(x_start + chunk_w, width)
                tile_w = x_end - x_start

                tile_idx += 1
                log(f"Integrating tile {tile_idx}/{total_tiles}…")

                tile_stack   = np.zeros((N, tile_h, tile_w, channels), dtype=self._dtype())
                weights_list = []

                num_cores = os.cpu_count() or 4
                with ThreadPoolExecutor(max_workers=num_cores) as executor:
                    future_to_i = {}
                    for i, fpath in enumerate(file_list):
                        future = executor.submit(load_fits_tile, fpath, y_start, y_end, x_start, x_end)
                        future_to_i[future] = i
                        weights_list.append(frame_weights.get(fpath, 1.0))

                    for fut in as_completed(future_to_i):
                        i = future_to_i[fut]
                        sub_img = fut.result()
                        if sub_img is None:
                            log(f"DEBUG: Tile load returned None for file: {file_list[i]}")
                            continue
                        if sub_img.ndim == 2:
                            sub_img = sub_img[:, :, np.newaxis]
                            if channels == 3:
                                sub_img = np.repeat(sub_img, 3, axis=2)
                        elif sub_img.ndim == 3 and sub_img.shape[0] == 3 and channels == 3:
                            sub_img = sub_img.transpose(1, 2, 0)
                        tile_stack[i] = sub_img.astype(np.float32, copy=False)

                weights_array = np.array(weights_list, dtype=np.float32)

                # Rejection
                algo = self.rejection_algorithm
                if algo == "Simple Median (No Rejection)":
                    tile_result  = np.median(tile_stack, axis=0)
                    tile_rej_map = np.zeros((N, tile_h, tile_w), dtype=bool)
                elif algo == "Simple Average (No Rejection)":
                    tile_result  = np.average(tile_stack, axis=0, weights=weights_array)
                    tile_rej_map = np.zeros((N, tile_h, tile_w), dtype=bool)
                elif algo == "Weighted Windsorized Sigma Clipping":
                    tile_result, tile_rej_map = windsorized_sigma_clip_weighted(
                        tile_stack, weights_array,
                        lower=self.sigma_low, upper=self.sigma_high
                    )
                elif algo == "Kappa-Sigma Clipping":
                    tile_result, tile_rej_map = kappa_sigma_clip_weighted(
                        tile_stack, weights_array,
                        kappa=self.kappa, iterations=self.iterations
                    )
                elif algo == "Trimmed Mean":
                    tile_result, tile_rej_map = trimmed_mean_weighted(
                        tile_stack, weights_array,
                        trim_fraction=self.trim_fraction
                    )
                elif algo == "Extreme Studentized Deviate (ESD)":
                    tile_result, tile_rej_map = esd_clip_weighted(
                        tile_stack, weights_array,
                        threshold=self.esd_threshold
                    )
                elif algo == "Biweight Estimator":
                    tile_result, tile_rej_map = biweight_location_weighted(
                        tile_stack, weights_array,
                        tuning_constant=self.biweight_constant
                    )
                elif algo == "Modified Z-Score Clipping":
                    tile_result, tile_rej_map = modified_zscore_clip_weighted(
                        tile_stack, weights_array,
                        threshold=self.modz_threshold
                    )
                elif algo == "Max Value":
                    tile_result, tile_rej_map = max_value_stack(
                        tile_stack, weights_array
                    )
                else:
                    tile_result, tile_rej_map = windsorized_sigma_clip_weighted(
                        tile_stack, weights_array,
                        lower=self.sigma_low, upper=self.sigma_high
                    )

                # Commit tile
                integrated_image[y_start:y_end, x_start:x_end, :] = tile_result

                # Collect per-file rejections
                if tile_rej_map.ndim == 4:
                    tile_rej_map = np.any(tile_rej_map, axis=-1)
                for i, fpath in enumerate(file_list):
                    ys, xs = np.where(tile_rej_map[i])
                    for dy, dx in zip(ys, xs):
                        per_file_rejections[fpath].append((x_start + dx, y_start + dy))

        if channels == 1:
            integrated_image = integrated_image[..., 0]

        log(f"Integration complete for group '{group_key}'.")
        return integrated_image, per_file_rejections, ref_header




    def outlier_rejection_with_mask(self, tile_stack, weights_array):
        """
        Example outlier rejection routine that computes the weighted median of the tile stack
        and returns both the integrated tile and a rejection mask.
        
        Parameters:
        tile_stack: numpy array of shape (N, H, W, C)
        weights_array: numpy array of shape (N,)
        
        Returns:
        tile_result: numpy array of shape (H, W, C)
        rejection_mask: boolean numpy array of shape (H, W) where True indicates a rejected pixel.
        
        This is a simple example. Replace this logic with your actual rejection algorithm.
        """
        # Compute the weighted median along axis 0.
        # For simplicity, we'll use the unweighted median here.
        tile_result = np.median(tile_stack, axis=0)
        
        # Compute the absolute deviation for each frame and take the median deviation.
        # Then mark as rejected any pixel in any frame that deviates by more than a threshold.
        # Here we define a threshold factor (this value may need tuning).
        threshold_factor = 1.5
        abs_deviation = np.abs(tile_stack - tile_result)
        # Compute the median deviation per pixel over the frames.
        median_deviation = np.median(abs_deviation, axis=0)
        # Define a rejection mask: True if the median deviation exceeds a threshold.
        # (For demonstration, assume threshold = threshold_factor * some constant; here we choose 0.05.)
        rejection_mask = median_deviation[..., 0] > (threshold_factor * 0.05)
        # If color, you might combine channels or process each channel separately.
        
        return tile_result, rejection_mask
