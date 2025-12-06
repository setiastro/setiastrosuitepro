# Stacking module - imports from functions.py for shared utilities
from __future__ import annotations
import os
import sys
import math
import time
import numpy as np
import cv2
cv2.setNumThreads(0)

from PyQt6.QtCore import Qt, QTimer, QSettings, pyqtSignal, QObject, pyqtSlot, QThread, QEvent, QPoint, QSize, QEventLoop, QCoreApplication, QRectF, QPointF, QMetaObject
from PyQt6.QtGui import QIcon, QImage, QPixmap, QAction, QIntValidator, QDoubleValidator, QFontMetrics, QTextCursor, QPalette, QPainter, QPen, QTransform, QColor, QBrush, QCursor
from PyQt6.QtWidgets import (QDialog, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QLineEdit, QTreeWidget, QHeaderView, QTreeWidgetItem, QProgressBar, QProgressDialog,
                             QFormLayout, QDialogButtonBox, QToolBar, QToolButton, QFileDialog, QTabWidget, QAbstractItemView, QSpinBox, QDoubleSpinBox, QGroupBox, QRadioButton,
                             QSizePolicy, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QApplication, QScrollArea, QTextEdit, QMenu, QPlainTextEdit, QGraphicsEllipseItem,
                             QMessageBox, QSlider, QCheckBox, QInputDialog, QComboBox)

from astropy.io import fits
from datetime import datetime

# Import shared utilities from functions.py
from .functions import (
    _asarray, _WINDOWS_RESERVED, _FITS_EXTS,
    get_valid_header, LRUDict, load_image, save_image,
    _torch_ok, _gpu_algo_supported, _torch_reduce_tile,
    windsorized_sigma_clip_weighted, kappa_sigma_clip_weighted,
    debayer_raw_fast, drizzle_deposit_numba_kernel_mono, drizzle_deposit_color_kernel,
    finalize_drizzle_2d, finalize_drizzle_3d,
    bulk_cosmetic_correction_numba, drizzle_deposit_numba_naive, drizzle_deposit_color_naive,
    bulk_cosmetic_correction_bayer,
    compute_star_count_fast_preview, siril_style_autostretch,
)
from .functions import *


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
        """Load image for preview; ensure float32 in [0,1] and HxWx{1,3}."""
        image_data, header, _, _ = load_image(self.ref_frame_path)
        if image_data is None:
            QMessageBox.critical(self, "Error", "Failed to load the reference image.")
            return

        # If CHW (3,H,W), convert to HWC for preview
        if image_data.ndim == 3 and image_data.shape[0] == 3 and image_data.shape[-1] != 3:
            image_data = np.transpose(image_data, (1, 2, 0))  # CHW -> HWC

        # Squeeze last singleton channel
        if image_data.ndim == 3 and image_data.shape[-1] == 1:
            image_data = np.squeeze(image_data, axis=-1)

        img = image_data.astype(np.float32, copy=False)

        # Preview-normalize: if not already ~[0,1], bring it into [0,1]
        mn = float(np.nanmin(img)); mx = float(np.nanmax(img))
        if not np.isfinite(mn) or not np.isfinite(mx):
            img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
            mn = float(img.min()); mx = float(img.max())

        if mx > 1.0 or mn < 0.0:
            ptp = mx - mn
            img = (img - mn) / ptp if ptp > 0.0 else np.zeros_like(img, dtype=np.float32)

        self.original_image = np.clip(img, 0.0, 1.0)

    
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
    
    def _preview_boost(self, img: np.ndarray) -> np.ndarray:
        """Robust, very gentle stretch for display when image would quantize to black."""
        # Use your implemented siril_style_autostretch
        try:
            out = siril_style_autostretch(img, sigma=3.0).astype(np.float32, copy=False)
            mx = float(out.max())
            if mx > 0: out /= mx  # keep in [0,1]
            return np.clip(out, 0.0, 1.0)
        except Exception:
            return np.clip(img, 0.0, 1.0)

    def convertArrayToPixmap(self, image):
        if image is None:
            return None

        img = image.astype(np.float32, copy=False)

        # If image is so dim or flat that 8-bit will zero-out, boost for preview
        ptp = float(img.max() - img.min())
        needs_boost = (float(img.max()) <= (1.0 / 255.0)) or (ptp < 1e-6) or (not np.isfinite(img).all())
        if needs_boost:
            img = self._preview_boost(np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0))

        # Convert to 8-bit for QImage
        display_image = (img * 255.0).clip(0, 255).astype(np.uint8)

        if display_image.ndim == 2:
            h, w = display_image.shape
            q_image = QImage(display_image.data, w, h, w, QImage.Format.Format_Grayscale8)
        elif display_image.ndim == 3 and display_image.shape[2] == 3:
            h, w, _ = display_image.shape
            q_image = QImage(display_image.data, w, h, 3 * w, QImage.Format.Format_RGB888)
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

class _Responder(QObject):
    finished = pyqtSignal(object)   # emits the edited dict or None



class StatusLogWindow(QDialog):
    MAX_BLOCKS = 2000

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Stacking Suite Log")

        # ── key flags ─────────────────────────────────────────────
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)   # hide, don't delete
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setWindowFlag(Qt.WindowType.Tool, True)                    # tool window (no taskbar)
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, False)   # ❗ not global topmost
        self.setWindowModality(Qt.WindowModality.NonModal)              # don't block UI
        self._was_visible_on_deactivate = False   
        # ─────────────────────────────────────────────────────────
        # follow app activation/deactivation
        QApplication.instance().applicationStateChanged.connect(self._on_app_state_changed)

        # watch the parent to keep the log above it while the app is active
        #if parent is not None:
        #    parent.installEventFilter(self)

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


    def _apply_topmost(self, enable: bool):
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, enable)
        # Re-apply the native flags and stacking
        if enable:
            # When re-activating, make sure it’s shown even if the OS hid it
            self.show()               # ← **always** show on enable
            self.raise_()
        else:
            # When deactivating, keep whatever visible state it had
            # (don’t force-hide here—let the OS do whatever it wants)
            self.show()  # reapply flags without changing visibility

    @pyqtSlot(Qt.ApplicationState)
    def _on_app_state_changed(self, state):
        if state == Qt.ApplicationState.ApplicationActive:
            # If it was visible when we lost focus, ensure it’s back
            if self._was_visible_on_deactivate:
                self.show()
                self.raise_()
            self._apply_topmost(True)
        else:
            # Remember whether we should bring it back later
            self._was_visible_on_deactivate = self.isVisible()
            self._apply_topmost(False)

    def eventFilter(self, obj, event):
        if obj is self.parent() and event.type() in (
            QEvent.Type.WindowActivate,
            QEvent.Type.ZOrderChange,
            QEvent.Type.ActivationChange,
        ):
            if (QApplication.instance().applicationState() == Qt.ApplicationState.ApplicationActive
                and self.isVisible()):
                self.raise_()
        return super().eventFilter(obj, event)

    def show_raise(self):
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

def _save_master_with_rejection_layers(
    img_array: np.ndarray,
    hdr: "fits.Header",
    out_path: str,
    *,
    rej_any: "np.ndarray | None" = None,     # 2D bool
    rej_frac: "np.ndarray | None" = None,    # 2D float32 [0..1]
):
    """
    Writes a MEF (multi-extension FITS) file:
      - Primary HDU: the master image (2D or 3D) as float32
        * Mono: (H, W)
        * Color: (3, H, W)  <-- channels-first for FITS
      - Optional EXTNAME=REJ_COMB: uint8 (0/1) combined rejection mask
      - Optional EXTNAME=REJ_FRAC: float32 fraction-of-frames rejected per pixel
    """
    # --- sanitize/shape primary data ---
    data = np.asarray(img_array, dtype=np.float32, order="C")

    # If channels-last, move to channels-first for FITS
    if data.ndim == 3:
        # squeeze accidental singleton channels
        if data.shape[-1] == 1:
            data = np.squeeze(data, axis=-1)  # becomes (H, W)
        elif data.shape[-1] in (3, 4):       # RGB or RGBA
            data = np.transpose(data, (2, 0, 1))  # (C, H, W)
        # If already (C, H, W) leave it as-is.

    # After squeeze/transpose, re-evaluate dims
    if data.ndim not in (2, 3):
        raise ValueError(f"Unsupported master image shape for FITS: {data.shape}")

    # --- clone + annotate header, and align NAXIS* with 'data' ---
    H = (hdr.copy() if hdr is not None else fits.Header())
    # purge prior NAXIS keys to avoid conflicts after transpose/squeeze
    for k in ("NAXIS", "NAXIS1", "NAXIS2", "NAXIS3", "NAXIS4"):
        if k in H:
            del H[k]

    H["IMAGETYP"] = "MASTER STACK"
    H["BITPIX"]   = -32
    H["STACKED"]  = (True, "Stacked with rejection; channels-first in FITS if color")
    H["CREATOR"]  = "SetiAstroSuite"
    H["DATE-OBS"] = datetime.utcnow().isoformat()

    # Fill NAXIS* to match data (optional; Astropy will infer if omitted)
    if data.ndim == 2:
        H["NAXIS"]  = 2
        H["NAXIS1"] = int(data.shape[1])  # width
        H["NAXIS2"] = int(data.shape[0])  # height
    else:
        # data.shape == (C, H, W)
        C, Hh, Ww = data.shape
        H["NAXIS"]  = 3
        H["NAXIS1"] = int(Ww)  # width
        H["NAXIS2"] = int(Hh)  # height
        H["NAXIS3"] = int(C)   # channels/planes

    # --- build HDU list ---
    prim = fits.PrimaryHDU(data=data, header=H)
    hdul = [prim]

    # Optional layers: must be 2D (H, W). Convert types safely.
    if rej_any is not None:
        rej_any_2d = np.asarray(rej_any, dtype=bool)
        if rej_any_2d.ndim != 2:
            raise ValueError(f"REJ_COMB must be 2D, got {rej_any_2d.shape}")
        h = fits.Header()
        h["EXTNAME"] = "REJ_COMB"
        h["COMMENT"] = "Combined rejection mask (any algorithm / any frame)"
        hdul.append(fits.ImageHDU(data=rej_any_2d.astype(np.uint8, copy=False), header=h))

    #if rej_frac is not None:
    #    rej_frac_2d = np.asarray(rej_frac, dtype=np.float32)
    #    if rej_frac_2d.ndim != 2:
    #        raise ValueError(f"REJ_FRAC must be 2D, got {rej_frac_2d.shape}")
    #    h = fits.Header()
    #    h["EXTNAME"] = "REJ_FRAC"
    #    h["COMMENT"] = "Per-pixel fraction of frames rejected [0..1]"
    #    hdul.append(fits.ImageHDU(data=rej_frac_2d, header=h))

    fits.HDUList(hdul).writeto(out_path, overwrite=True)

class _SimplePickDialog(QDialog):
    def __init__(self, np_image, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Click the comet center")
        self._orig = np.clip(np.asarray(np_image, dtype=np.float32), 0.0, 1.0)
        if self._orig.ndim == 3 and self._orig.shape[-1] == 1:
            self._orig = np.squeeze(self._orig, axis=-1)

        self._autostretch = False
        self._zoom = 1.0
        self._marker_xy = None  # image coords (float)

        v = QVBoxLayout(self)

        # ---- Graphics View scaffold ----
        self.scene = QGraphicsScene(self)
        self.view = _ZoomableGraphicsView(self.scene, self)
        self.view.setRenderHints(self.view.renderHints() | QPainter.RenderHint.SmoothPixmapTransform)
        self.view.setDragMode(QGraphicsView.DragMode.NoDrag)  # keep arrow; pan via wheel/scrollbars or your own handler
        self.view.setCursor(QCursor(CursorShape.ArrowCursor))
        self.view.viewport().setCursor(QCursor(CursorShape.ArrowCursor))
        v.addWidget(self.view)

        # pixmap item that holds the image
        self.pix_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pix_item)

        # marker (crosshair as small ellipse; we’ll draw lines into it)
        self.marker = QGraphicsEllipseItem(-6, -6, 12, 12)

        pen = QPen(QColor(0, 255, 0))   # bright green outline
        pen.setWidth(2)
        pen.setCosmetic(True)           # stays 2px regardless of zoom
        self.marker.setPen(pen)

        # optional: translucent green fill; or use Qt.NoBrush for hollow circle
        #self.marker.setBrush(QBrush(QColor(0, 255, 0, 60)))

        self.marker.setZValue(1_000_000)  # ensure it draws on top
        self.marker.setVisible(False)
        self.scene.addItem(self.marker)

        # ---- Controls ----
        row = QHBoxLayout()
        self.btn_fit = QPushButton("Fit")
        self.btn_fit.clicked.connect(self.fitToView)
        row.addWidget(self.btn_fit)

        self.btn_1x = QPushButton("1:1")
        self.btn_1x.clicked.connect(self.zoom1x)
        row.addWidget(self.btn_1x)

        self.btn_zi = QPushButton("Zoom In")
        self.btn_zi.clicked.connect(lambda: self.zoomBy(1.2))
        row.addWidget(self.btn_zi)

        self.btn_zo = QPushButton("Zoom Out")
        self.btn_zo.clicked.connect(lambda: self.zoomBy(1/1.2))
        row.addWidget(self.btn_zo)

        self.btn_st = QPushButton("Enable Autostretch")
        self.btn_st.setCheckable(True)
        self.btn_st.toggled.connect(self._toggle_autostretch)
        row.addWidget(self.btn_st)

        row.addStretch(1)
        v.addLayout(row)

        # ---- OK/Cancel ----
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        v.addWidget(btns)

        # image render + first fit
        self._update_pixmap()
        QTimer.singleShot(0, self.fitToView)

        # click to place marker
        self.view.mousePressOnScene = self._on_scene_click
        self.view.viewport().installEventFilter(self)

    def eventFilter(self, obj, ev):
        if obj is self.view.viewport():
            if ev.type() == QEvent.Type.CursorChange:
                obj.setCursor(QCursor(CursorShape.ArrowCursor))
                return True
        return super().eventFilter(obj, ev)

    # ---------- Display pipeline ----------
    def _make_display(self):
        if self._autostretch:
            if self._orig.ndim == 2:
                from imageops.stretch import stretch_mono_image
                disp = stretch_mono_image(self._orig, target_median=0.30, normalize=True, apply_curves=False)
            elif self._orig.ndim == 3 and self._orig.shape[2] == 3:
                from imageops.stretch import stretch_color_image
                disp = stretch_color_image(self._orig, target_median=0.30, linked=False, normalize=True, apply_curves=False)
            else:
                disp = self._orig
        else:
            disp = self._orig
        return (np.clip(disp, 0, 1) * 255).astype(np.uint8)

    def _update_pixmap(self):
        disp = self._make_display()
        if disp.ndim == 2:
            h, w = disp.shape; bpl = w
            qimg = QImage(disp.tobytes(), w, h, bpl, QImage.Format.Format_Grayscale8)
        else:
            h, w, _ = disp.shape; bpl = 3*w
            qimg = QImage(disp.tobytes(), w, h, bpl, QImage.Format.Format_RGB888)
        pm = QPixmap.fromImage(qimg)
        self.pix_item.setPixmap(pm)
        # reset scene rect to image bounds so fit works
        self.scene.setSceneRect(QRectF(0, 0, pm.width(), pm.height()))

        # keep marker visible at same image coordinate
        if self._marker_xy is not None:
            self._place_marker_at(self._marker_xy[0], self._marker_xy[1])

    # ---------- Zoom / Fit ----------
    def fitToView(self):
        if self.pix_item.pixmap().isNull():
            return
        self.view.fitInView(self.pix_item, Qt.AspectRatioMode.KeepAspectRatio)
        # record current zoom (approx) from view transform
        m = self.view.transform().m11()
        self._zoom = m

    def zoom1x(self):
        self.view.setTransform(QTransform())  # identity
        self._zoom = 1.0

    def zoomBy(self, factor):
        self._zoom *= factor
        self.view.scale(factor, factor)

    def _toggle_autostretch(self, checked):
        self._autostretch = bool(checked)
        self.btn_st.setText("Disable Autostretch" if checked else "Enable Autostretch")
        self._update_pixmap()

    # ---------- Picking ----------
    def _on_scene_click(self, scene_pos: QPointF, button: Qt.MouseButton):
        if button != Qt.MouseButton.LeftButton:
            return
        # clamp to image rect
        img_rect = self.scene.sceneRect()
        x = float(max(img_rect.left(), min(img_rect.right() - 1.0, scene_pos.x())))
        y = float(max(img_rect.top(),  min(img_rect.bottom() - 1.0, scene_pos.y())))
        self._marker_xy = (x, y)
        self._place_marker_at(x, y)

    def _place_marker_at(self, x, y):
        self.marker.setPos(QPointF(x, y))
        self.marker.setVisible(True)

    def point(self):
        """Return (x, y) in native image coordinates (float), or (0.0, 0.0) if none."""
        return self._marker_xy or (0.0, 0.0)

class _ZoomableGraphicsView(QGraphicsView):
    """
    Small helper view: Ctrl+wheel to zoom centered on cursor.
    Right or middle mouse drag pans (ScrollHandDrag is enabled).
    We call back to owner for clicks in scene coords.
    """
    def __init__(self, scene, owner):
        super().__init__(scene)
        self.owner = owner
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.SmartViewportUpdate)
        self.mousePressOnScene = None  # set by owner
        self.setMouseTracking(True)

    def wheelEvent(self, e):
        if e.modifiers() & Qt.KeyboardModifier.ControlModifier:
            factor = 1.15 if e.angleDelta().y() > 0 else 1/1.15
            # zoom around cursor
            pos = e.position()
            old_pos = self.mapToScene(int(pos.x()), int(pos.y()))
            self.scale(factor, factor)
            new_pos = self.mapToScene(int(pos.x()), int(pos.y()))
            delta = new_pos - old_pos
            self.translate(delta.x(), delta.y())
            e.accept()
        else:
            super().wheelEvent(e)

    def mousePressEvent(self, e):
        if self.mousePressOnScene and e.button() in (Qt.MouseButton.LeftButton,):
            sp = self.mapToScene(e.pos())
            self.mousePressOnScene(sp, e.button())
        super().mousePressEvent(e)

def _canonize_img(img: np.ndarray) -> np.ndarray:
    """Return image as float32, shape (H,W,C) where C is 1 or 3."""
    x = np.asarray(img, dtype=np.float32)
    if x.ndim == 2:
        return x[..., None]                # (H,W) -> (H,W,1)
    if x.ndim == 3 and x.shape[2] in (1, 3):
        return x
    # unexpected (e.g., more channels)
    raise ValueError(f"Unexpected image shape {x.shape}")

def _canonize_mask(mask: np.ndarray, channels: int) -> np.ndarray:
    """Return mask in [0,1], shape (H,W,channels)."""
    m = np.asarray(mask, dtype=np.float32)
    if m.ndim == 2:
        m = m[..., None]                   # (H,W) -> (H,W,1)
    elif m.ndim == 3 and m.shape[2] == 1:
        pass
    else:
        raise ValueError(f"Unexpected mask shape {m.shape}")
    # repeat to RGB if needed
    if channels == 3 and m.shape[2] == 1:
        m = np.repeat(m, 3, axis=2)
    return np.clip(m, 0.0, 1.0)

def feather_mask(mask_hw_or_hw1: np.ndarray, feather_px: int, channels: int) -> np.ndarray:
    """Feather with Gaussian blur; returns (H,W,channels) in [0,1]."""
    m = _canonize_mask(mask_hw_or_hw1, 1)[..., 0]   # work in 2-D
    if feather_px and feather_px > 0:
        # Convert feather_px (approx radius) to sigma and odd kernel size
        sigma = max(0.1, float(feather_px) / 2.0)
        k = int(2 * int(3 * sigma) + 1)            # ~3σ radius, odd
        m = cv2.GaussianBlur(m, (k, k), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REPLICATE)
        m = np.clip(m, 0.0, 1.0)
    # expand to channels
    if channels == 3:
        m = np.repeat(m[..., None], 3, axis=2)
    else:
        m = m[..., None]
    return m

def blend_stars_comet(stars_img, comet_img, comet_mask, feather_px=16, mix=1.0):
    """
    stars_img: (H,W) or (H,W,1/3)
    comet_img: (H,W) or (H,W,1/3)
    comet_mask: (H,W) or (H,W,1)
    mix in [0..1]: 0 = only stars, 1 = only comet in masked areas
    """
    S = _canonize_img(stars_img)
    C = _canonize_img(comet_img)

    # If one is mono and the other RGB, upcast mono to RGB for a consistent result
    if S.shape[2] != C.shape[2]:
        if S.shape[2] == 1 and C.shape[2] == 3:
            S = np.repeat(S, 3, axis=2)
        elif S.shape[2] == 3 and C.shape[2] == 1:
            C = np.repeat(C, 3, axis=2)
        else:
            raise ValueError(f"Unsupported channel combination: {S.shape} vs {C.shape}")

    A = feather_mask(comet_mask, feather_px, S.shape[2]) * float(np.clip(mix, 0.0, 1.0))
    out = (1.0 - A) * S + A * C
    return np.clip(out, 0.0, 1.0)

def _match_channels(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (A', B') with matching channel counts (1 or 3), float32."""
    A = np.asarray(A, dtype=np.float32)
    B = np.asarray(B, dtype=np.float32)
    def _c(x):
        if x.ndim == 2:    return 1
        if x.ndim == 3:    return x.shape[2]
        raise ValueError(f"Unexpected image shape {x.shape}")
    ca, cb = _c(A), _c(B)
    if ca == cb:
        return A, B
    if ca == 1 and cb == 3:
        A = np.repeat(A[..., None] if A.ndim == 2 else A, 3, axis=2)
    elif ca == 3 and cb == 1:
        B = np.repeat(B[..., None] if B.ndim == 2 else B, 3, axis=2)
    else:
        # handle (H,W,1) vs (H,W): squeeze to 2D then upcast consistently
        if A.ndim == 3 and A.shape[2] == 1: A = A[..., 0]
        if B.ndim == 3 and B.shape[2] == 1: B = B[..., 0]
        A = A[..., None] if A.ndim == 2 else A
        B = B[..., None] if B.ndim == 2 else B
    return A, B

# --- comet-friendly reducers ---
def _lower_trimmed_mean(ts: np.ndarray, trim_hi_frac: float = 0.30) -> np.ndarray:
    """
    Per-pixel lower-trimmed mean: drop the brightest t% (star trails) then mean.
    ts: (N, th, tw, C) float32
    """
    n = ts.shape[0]
    k = int(np.floor(n * (1.0 - trim_hi_frac)))
    if k <= 0:
        return np.median(ts, axis=0)
    part = np.partition(ts, k-1, axis=0)[:k, ...]  # keep lowest k across N
    return np.mean(part, axis=0)

def _percentile40(ts: np.ndarray) -> np.ndarray:
    """Low-percentile combiner; good at suppressing bright streaks."""
    return np.percentile(ts, 40.0, axis=0)

def _high_clip_percentile(ts: np.ndarray, k: float = 2.5, p: float = 40.0) -> np.ndarray:
    """
    Robust high-side winsorize per pixel, then take a low percentile.
    ts: (N, th, tw, C) float32
    k:  MAD multiplier for high tail clamp
    p:  percentile to return (e.g. 35..45)
    """
    med = np.median(ts, axis=0)
    mad = np.median(np.abs(ts - med), axis=0) + 1e-6
    hi = med + (k * 1.4826 * mad)
    clipped = np.minimum(ts, hi, dtype=ts.dtype)
    return np.percentile(clipped, p, axis=0)

def _high_clip_percentile_fast(ts: np.ndarray, k: float = 2.5, p: float = 40.0,
                               _work: dict = {}) -> np.ndarray:
    """
    Same math as _high_clip_percentile, but ~2–4× faster on CPU:
      - median / MAD via np.partition (no full sort)
      - percentile via partition index (nearest-rank)
      - reuse scratch buffers between calls (pass via _work dict)
    ts: (N, th, tw, C) float32
    """
    N = ts.shape[0]
    assert N >= 2, "need at least two frames"

    # allocate / reuse scratch buffers
    tmp  = _work.get("tmp");   # for med
    tmp2 = _work.get("tmp2");  # for mad & hi
    tmp3 = _work.get("tmp3");  # for percentile
    if (tmp is None) or (tmp.shape != ts.shape):
        tmp  = np.empty_like(ts)
        tmp2 = np.empty_like(ts)
        tmp3 = np.empty_like(ts)
        _work["tmp"], _work["tmp2"], _work["tmp3"] = tmp, tmp2, tmp3

    # ---- median along axis 0 (copy -> partition -> take middle slice)
    np.copyto(tmp, ts)
    np.partition(tmp, N // 2, axis=0)
    med = tmp[N // 2]  # shape: (th, tw, C)

    # ---- MAD: median(|ts - med|)
    np.subtract(ts, med, out=tmp2)
    np.abs(tmp2, out=tmp2)
    np.copyto(tmp, tmp2)
    np.partition(tmp, N // 2, axis=0)
    mad = tmp[N // 2] + 1e-6

    # ---- high-side clip threshold: hi = med + k * 1.4826 * MAD
    np.multiply(mad, (k * 1.4826), out=tmp)   # tmp = k*1.4826*MAD
    np.add(med, tmp, out=tmp)                 # tmp = hi

    # ---- clip high side into tmp2 (reuse): clipped = min(ts, hi)
    np.minimum(ts, tmp, out=tmp2)

    # ---- percentile via nearest-rank partition
    # index of p-th percentile along axis=0
    p = float(p)
    idx = int(np.clip(round((p / 100.0) * (N - 1)), 0, N - 1))
    np.copyto(tmp3, tmp2)
    np.partition(tmp3, idx, axis=0)
    return tmp3[idx]

def _parse_binning_from_header(hdr) -> tuple[int, int]:
    """
    Return (xbin, ybin), defaulting to (1,1). Handles common FITS keys and string forms.
    Accepts: XBINNING/YBINNING, CCDXBIN/CCDYBIN, XBIN/YBIN, BINNING="2 2" | "2x2" | "2,2"
    """
    def _coerce(v):
        try:
            return int(float(v))
        except Exception:
            return None

    if hdr is None:
        return 1, 1

    # direct numeric keys
    for kx, ky in (("XBINNING", "YBINNING"),
                   ("CCDXBIN", "CCDYBIN"),
                   ("XBIN", "YBIN")):
        if kx in hdr and ky in hdr:
            bx = _coerce(hdr.get(kx)); by = _coerce(hdr.get(ky))
            if bx and by:
                return max(1, bx), max(1, by)

    # combined string key
    if "BINNING" in hdr:
        s = str(hdr.get("BINNING", "")).lower().replace("x", " ").replace(",", " ")
        parts = [p for p in s.split() if p.strip()]
        if len(parts) >= 2:
            bx = _coerce(parts[0]); by = _coerce(parts[1])
            if bx and by:
                return max(1, bx), max(1, by)

    return 1, 1


def _resize_to_scale(img: np.ndarray, scale_x: float, scale_y: float) -> np.ndarray:
    """
    Resample image to apply per-axis scale. Works for 2D or HxWx3.
    Uses OpenCV if present (fast, high quality). Fallbacks to SciPy; last resort = NumPy kron for integer upscales.
    """
    if scale_x == 1.0 and scale_y == 1.0:
        return img

    h, w = img.shape[:2]
    new_w = max(1, int(round(w * (scale_x))))
    new_h = max(1, int(round(h * (scale_y))))

    # Prefer OpenCV
    try:
        import cv2
        interp = cv2.INTER_CUBIC if (scale_x > 1.0 or scale_y > 1.0) else cv2.INTER_AREA
        if img.ndim == 2:
            return cv2.resize(img.astype(np.float32, copy=False), (new_w, new_h), interpolation=interp)
        else:
            return cv2.resize(img.astype(np.float32, copy=False), (new_w, new_h), interpolation=interp)
    except Exception:
        pass

    # SciPy fallback
    try:
        from scipy.ndimage import zoom
        if img.ndim == 2:
            return zoom(img.astype(np.float32, copy=False), (scale_y, scale_x), order=3)
        else:
            # zoom each channel; zoom expects zoom per axis
            return zoom(img.astype(np.float32, copy=False), (scale_y, scale_x, 1.0), order=3)
    except Exception:
        pass

    # Last resort (integer upscale only): kron
    sx_i = int(round(scale_x)); sy_i = int(round(scale_y))
    if abs(scale_x - sx_i) < 1e-6 and abs(scale_y - sy_i) < 1e-6 and sx_i >= 1 and sy_i >= 1:
        if img.ndim == 2:
            return np.kron(img, np.ones((sy_i, sx_i), dtype=np.float32))
        else:
            up2d = np.kron(img[..., 0], np.ones((sy_i, sx_i), dtype=np.float32))
            out = np.empty((up2d.shape[0], up2d.shape[1], img.shape[2]), dtype=np.float32)
            for c in range(img.shape[2]):
                out[..., c] = np.kron(img[..., c], np.ones((sy_i, sx_i), dtype=np.float32))
            return out

    # If all else fails, return original
    return img

def _center_crop_2d(a: np.ndarray, Ht: int, Wt: int) -> np.ndarray:
    H, W = a.shape[:2]
    if H == Ht and W == Wt:
        return a
    y0 = max(0, (H - Ht) // 2)
    x0 = max(0, (W - Wt) // 2)
    return a[y0:y0+Ht, x0:x0+Wt]

def _median_fast_sample(img: np.ndarray, stride: int = 8) -> float:
    """
    Fast robust median on a small strided sample.
    Works on mono or RGB (uses luma).
    Assumes 'img' is float32 and non-NaN/-Inf already.
    """
    if img.ndim == 3 and img.shape[-1] == 3:
        r, g, b = img[..., 0], img[..., 1], img[..., 2]
        L = 0.2126 * r + 0.7152 * g + 0.0722 * b
        sample = L[::stride, ::stride]
    else:
        sample = img[::stride, ::stride]
    sample = sample - float(np.nanmin(sample))
    return float(np.median(sample))

def _luma_view(img: np.ndarray) -> np.ndarray:
    """Return a float32 2D luma view (no copy if mono)."""
    if img.ndim == 2:
        return img.astype(np.float32, copy=False)
    if img.ndim == 3 and img.shape[-1] == 3:
        r, g, b = img[..., 0], img[..., 1], img[..., 2]
        # compute in float32, no allocation explosion
        return (0.2126 * r + 0.7152 * g + 0.0722 * b).astype(np.float32, copy=False)
    if img.ndim == 3 and img.shape[-1] == 1:
        return np.squeeze(img, axis=-1).astype(np.float32, copy=False)
    return img.astype(np.float32, copy=False)


def _median_fast_sample(img: np.ndarray, stride: int = 8) -> float:
    """
    Very fast robust median from a decimated grid.
    Works on mono or color (uses luma).
    """
    v = _luma_view(img)
    v = v[::stride, ::stride]
    # subtract a tiny pedestal so transparency/G gain differences dominate
    vmin = float(np.nanmin(v))
    return float(np.median(v - vmin)) if v.size else 0.0


def _compute_scale(ref_target_median: float, preview_median: float, img: np.ndarray,
                   refine_stride: int = 8, refine_if_rel_err: float = 0.10) -> float:
    """
    Start from preview-based scale, optionally refine once on a tiny decimated sample.
    """
    eps = 1e-6
    s0 = ref_target_median / max(preview_median, eps)  # first guess (from preview)
    # quick refinement only if we're likely off by >10%
    m_post = _median_fast_sample(img, stride=refine_stride)
    if m_post > eps:
        s1 = ref_target_median / m_post
        # if preview and refined differ a lot, trust refined; otherwise keep s0 (avoids jitter)
        if abs(s1 - s0) / max(s0, eps) >= refine_if_rel_err:
            return float(s1)
    return float(s0)


def _apply_scale_inplace(img: np.ndarray, s: float) -> np.ndarray:
    """
    Scale in-place when safe (saves alloc), else returns a new float32 array.
    """
    # ensure float32 for consistent stack later; try in-place if possible
    if img.dtype != np.float32:
        img = img.astype(np.float32, copy=False)
    # numpy multiply is already vectorized/SIMD
    img *= np.float32(s)
    return img

def _fits_first_image_hdu(hdul):
    """
    Return (hdu_index, hdu) for the first HDU that actually contains an image
    (PrimaryHDU or ImageHDU or CompImageHDU) with a numeric ndarray.
    """
    for i, h in enumerate(hdul):
        try:
            data = h.data  # astropy will auto-scale with BSCALE/BZERO
        except Exception:
            continue
        if isinstance(data, np.ndarray) and data.size > 0:
            # Restrict to 2D or 3D images only
            if data.ndim in (2, 3):
                return i, h
    return None, None

def _fits_read_any_hdu(path, prefer_float32=True, memmap=True):
    """
    Open FITS and return (img, header, hdu_index). Finds the *first* HDU that has
    real image data. Handles PrimaryHDU, ImageHDU, CompImageHDU.
    Converts to float32 unless prefer_float32=False.
    Squeezes trailing singleton channel dim.
    """
    with fits.open(path, memmap=memmap) as hdul:
        idx, h = _fits_first_image_hdu(hdul)
        if h is None:
            return None, None, None
        img = h.data
        hdr = h.header
        # standardize dtype
        if prefer_float32 and img is not None:
            img = np.asarray(img, dtype=np.float32, order="C")
        # squeeze trailing channel=1
        if img is not None and img.ndim == 3 and img.shape[-1] == 1:
            img = np.squeeze(img, axis=-1)
        return img, hdr, idx

def _apply_blank_bscale_bzero(arr: np.ndarray, hdr: fits.Header) -> np.ndarray:
    """Manual FITS image scaling: apply BLANK (→ 0), then BSCALE/BZERO. Return float32."""
    if ma.isMaskedArray(arr):
        arr = arr.filled(0)
    # Replace BLANK sentinel for integer images
    if hdr is not None and "BLANK" in hdr and np.issubdtype(arr.dtype, np.integer):
        blank = int(hdr["BLANK"])
        arr = np.where(arr == blank, 0, arr)

    # Manual scaling
    bscale = float(hdr.get("BSCALE", 1.0)) if hdr is not None else 1.0
    bzero  = float(hdr.get("BZERO", 0.0)) if hdr is not None else 0.0
    arr = arr.astype(np.float32, copy=False)
    if (bscale != 1.0) or (bzero != 0.0):
        arr = arr * np.float32(bscale) + np.float32(bzero)
    return arr

def _fits_read_any_hdu_noscale(path: str, memmap: bool) -> tuple[np.ndarray | None, fits.Header | None, int | None]:
    """
    Open FITS with do_not_scale_image_data=True and return (img, header, hdu_index)
    for the first HDU that has 2D/3D numeric image data. Applies BLANK/BSCALE/BZERO manually.
    """
    with fits.open(path,
                   memmap=memmap,
                   do_not_scale_image_data=True,
                   ignore_missing_end=True,
                   uint=False) as hdul:
        for i, h in enumerate(hdul):
            try:
                d = h.data
            except Exception:
                continue
            if not isinstance(d, np.ndarray):
                continue
            if d.ndim not in (2, 3) or d.size == 0:
                continue
            # Ensure native byte order (avoid later copies)
            d = _to_native_endian(d)
            # Squeeze trivial leading/ending dims
            d = np.squeeze(d)
            # Manual scaling to float32
            d = _apply_blank_bscale_bzero(d, h.header)

            # If 3D and trailing channel is 1, squeeze it
            if d.ndim == 3 and d.shape[-1] == 1:
                d = np.squeeze(d, axis=-1)

            # Clean numerics
            d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
            return d, h.header, i
    return None, None, None

class _Progress:
    def __init__(self, owner, title: str, maximum: int):
        self._owner = owner
        self._pd = QProgressDialog(title, "Cancel", 0, max(1, int(maximum)), owner)
        self._pd.setWindowModality(Qt.WindowModality.ApplicationModal)
        self._pd.setMinimumDuration(0)
        self._pd.setAutoClose(False)
        self._pd.setAutoReset(False)
        self._pd.setValue(0)
        self._pd.setWindowTitle(title)
        self._pd.setMinimumWidth(520)
        self._cancelled = False
        self._value = 0

        def _on_cancel():
            self._cancelled = True
        self._pd.canceled.connect(_on_cancel, Qt.ConnectionType.QueuedConnection)
        self._pd.show()
        QApplication.processEvents()

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def set_label(self, text: str):
        try:
            self._pd.setLabelText(text)
        except Exception:
            pass

    def set_value(self, v: int):
        self._value = max(0, min(int(v), self._pd.maximum()))
        try:
            self._pd.setValue(self._value)
        except Exception:
            pass
        QApplication.processEvents()

    def step(self, n: int = 1, label: str | None = None):
        if label:
            self.set_label(label)
        self.set_value(self._value + n)

    def close(self):
        try:
            self._pd.reset()
            self._pd.deleteLater()
        except Exception:
            pass

def _count_tiles(h: int, w: int, ch: int, cw: int) -> int:
    ty = (h + ch - 1) // ch
    tx = (w + cw - 1) // cw
    return ty * tx



def _nearest_index(src_len: int, dst_len: int) -> np.ndarray:
    """Nearest-neighbor index map from dst→src (no rounding drift)."""
    if dst_len <= 0:
        return np.zeros((0,), dtype=np.int32)
    scale = src_len / float(dst_len)
    idx = np.floor((np.arange(dst_len, dtype=np.float32) + 0.5) * scale).astype(np.int32)
    idx[idx < 0] = 0
    idx[idx >= src_len] = src_len - 1
    return idx

def _nearest_resize_2d(m: np.ndarray, H: int, W: int) -> np.ndarray:
    """Resize 2-D array to (H,W) using nearest-neighbor. Uses shared implementation."""
    return _nearest_resize_2d_shared(m, H, W)

def _expand_mask_for(image_like: np.ndarray, mask_like: np.ndarray) -> np.ndarray:
    """
    Return a float32 mask in [0..1] shaped for image_like:
      - If image is HxW  (mono): mask -> HxW
      - If image is HxWxC: mask -> HxWxC (repeated per channel)
    Accepts mask as HxW, HxWx1, or HxWxC (we'll reduce to 2-D then repeat).
    """
    im = np.asarray(image_like)
    m  = np.asarray(mask_like, dtype=np.float32)

    # squeeze trivial last dim
    if m.ndim == 3 and m.shape[2] == 1:
        m = m[..., 0]

    # normalize to [0..1] if mask is 0..255 or arbitrary
    mmin, mmax = float(m.min(initial=0.0)), float(m.max(initial=1.0))
    if mmax > 1.0 or mmin < 0.0:
        if mmax > 0:
            m = m / mmax
        m = np.clip(m, 0.0, 1.0)

    if im.ndim == 2:
        # want 2-D
        if m.ndim == 3:
            m = m[..., 0]
        if m.shape != im.shape:
            m = _nearest_resize_2d(m, im.shape[0], im.shape[1])
        return m.astype(np.float32, copy=False)

    # RGB path
    H, W, C = im.shape[:3]
    if m.ndim == 3 and m.shape[:2] == (H, W):
        # reduce multi-channel mask to 2-D via max then repeat
        m = np.max(m, axis=2)
    if m.ndim == 2:
        if m.shape != (H, W):
            m = _nearest_resize_2d(m, H, W)
        m = np.repeat(m[..., None], C, axis=2)
    return np.clip(m, 0.0, 1.0).astype(np.float32, copy=False)

def _debug_dump_mask_blend(prefix: str, img: np.ndarray, mask: np.ndarray, out_dir: str):
    try:
        os.makedirs(out_dir, exist_ok=True)
        m = _expand_mask_for(img, mask)
        # save per-channel PNGs to spot one-channel issues instantly
        def _to8(u): 
            u = np.clip(u.astype(np.float32), 0, 1) * 255.0
            return u.astype(np.uint8)
        if img.ndim == 3:
            for i, ch in enumerate(["R","G","B"]):
                cv2.imwrite(os.path.join(out_dir, f"{prefix}_img_{ch}.png"), _to8(img[..., i]))
                cv2.imwrite(os.path.join(out_dir, f"{prefix}_mask_{ch}.png"), _to8(m[..., i]))
        else:
            cv2.imwrite(os.path.join(out_dir, f"{prefix}_img.png"), _to8(img))
            cv2.imwrite(os.path.join(out_dir, f"{prefix}_mask.png"), _to8(m if m.ndim==2 else m[...,0]))
    except Exception:
        pass

def _float01_to_u16(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=np.float32), 0.0, 1.0)
    return (x * 65535.0 + 0.5).astype(np.uint16, copy=False)

def _u16_to_float01(x: np.ndarray) -> np.ndarray:
    if x.dtype != np.uint16:
        # be forgiving: if someone returns 8-bit, still normalize
        if x.dtype == np.uint8:
            return (x.astype(np.float32) / 255.0).astype(np.float32)
        return np.clip(x.astype(np.float32), 0.0, 1.0)
    return (x.astype(np.float32) / 65535.0).astype(np.float32)

def _write_poly_or_unknown(fh, file_key: str, kind_name: str):
    """Helper: write a block with no numeric matrix (loader maps to (kind, None))."""
    fh.write(f"FILE: {file_key}\n")
    fh.write(f"KIND: {kind_name}\n")
    fh.write("MATRIX:\nUNSUPPORTED\n\n")

_SESSION_PATTERNS = [
    # "Night1", "night_02", "Session-3", "sess7"
    (re.compile(r"(session|sess|night|noche|nuit)[ _-]?(\d{1,2})", re.I),
    lambda m: f"Session-{int(m.group(2)):02d}"),

    # ISO-ish dates in folder names: 2024-10-09, 2024_10_09, 20241009
    (re.compile(r"\b(20\d{2})[-_]?([01]\d)[-_]?([0-3]\d)\b"),
    lambda m: f"{m.group(1)}-{m.group(2)}-{m.group(3)}"),
]

def _to_writable_f32(arr):
    import numpy as np
    a = np.asarray(arr)
    if a.dtype != np.float32:
        a = a.astype(np.float32, copy=False)
    # Make sure we can write into it (mmap/XISF often returns read-only views)
    if (not a.flags.writeable) or (not a.flags.c_contiguous):
        a = np.ascontiguousarray(a.copy())
    return a

def _is_fits(path: str) -> bool:
    p = path.lower()
    return p.endswith(".fit") or p.endswith(".fits") or p.endswith(".fz")

def _is_xisf(path: str) -> bool:
    return path.lower().endswith(".xisf")

def _xisf_first_kw(meta_dict: dict, key: str, cast=str):
    """
    XISF images_meta['FITSKeywords'] → { KEY: [ {value, comment}, ...], ... }.
    Return first value if present, casted.
    """
    try:
        v = meta_dict.get("FITSKeywords", {}).get(key, [])
        if not v: return None
        val = v[0]["value"]
        return cast(val) if cast and val is not None else val
    except Exception:
        return None

def _get_header_fast(path: str):
    """
    FITS: std fast header. XISF: synthesize a dict with common keys used upstream.
    """
    if _is_fits(path):
        try:
            return fits.getheader(path, ext=0)
        except Exception:
            return {}
    if _is_xisf(path):
        try:
            x = XISF(path)
            im = x.get_images_metadata()[0]  # first image block
            # Build a tiny FITS-like dict
            hdr = {}
            # Common bits we use elsewhere
            filt = _xisf_first_kw(im, "FILTER", str)
            if filt is not None:
                hdr["FILTER"] = filt
            # Exposure: EXPTIME/EXPOSURE
            for k in ("EXPOSURE", "EXPTIME"):
                v = _xisf_first_kw(im, k, float)
                if v is not None:
                    hdr[k] = v
                    break
            # Bayer pattern (PI often writes CFA pattern as a keyword too)
            bp = _xisf_first_kw(im, "BAYERPAT", str)
            if bp:
                hdr["BAYERPAT"] = bp
            # Some PI exports use XISF properties instead of FITS keywords.
            props = im.get("XISFProperties", {})
            # Try a few common property ids for convenience
            for cand in ("Instrument:FILTER", "FILTER", "Filter:Name", "FilterName"):
                if cand in props and "value" in props[cand]:
                    hdr.setdefault("FILTER", str(props[cand]["value"]))
                    break
            for cand in ("Exposure:Time", "EXPTIME"):
                if cand in props and "value" in props[cand]:
                    try:
                        hdr.setdefault("EXPTIME", float(props[cand]["value"]))
                    except Exception:
                        pass
                    break
            return hdr
        except Exception:
            return {}
    return {}

def _quick_preview_from_path(path: str, *, target_xbin=1, target_ybin=1) -> np.ndarray | None:
    """
    Debayer-aware, tiny, 2D float32 preview for FITS or XISF.
    Mirrors your _quick_preview_from_fits behavior but supports XISF.
    Returned image is small-ish, writeable, contiguous.
    """
    def _superpixel2x2(x: np.ndarray) -> np.ndarray:
        h, w = x.shape[:2]
        h2, w2 = h - (h % 2), w - (w % 2)
        if h2 <= 0 or w2 <= 0:
            return x.astype(np.float32, copy=False)
        x = x[:h2, :w2].astype(np.float32, copy=False)
        if x.ndim == 2:
            return (x[0:h2:2, 0:w2:2] + x[0:h2:2, 1:w2:2] +
                    x[1:h2:2, 0:w2:2] + x[1:h2:2, 1:w2:2]) * 0.25
        # RGB → luma then superpixel
        r, g, b = x[..., 0], x[..., 1], x[..., 2]
        L = 0.2126 * r + 0.7152 * g + 0.0722 * b
        return (L[0:h2:2, 0:w2:2] + L[0:h2:2, 1:w2:2] +
                L[1:h2:2, 0:w2:2] + L[1:h2:2, 1:w2:2]) * 0.25

    try:
        if _is_fits(path):
            # Reuse your existing routine if you like; otherwise a tiny inline path:
            from astropy.io import fits
            with fits.open(path, memmap=True) as hdul:
                # Prefer primary image HDU; fall back to first image-like HDU
                hdu = None
                for cand in (0,):
                    try:
                        h = hdul[cand]
                        if getattr(h, "data", None) is not None:
                            hdu = h; break
                    except Exception:
                        pass
                if hdu is None:
                    for h in hdul:
                        if getattr(h, "data", None) is not None:
                            hdu = h; break
                if hdu is None or hdu.data is None:
                    return None
                data = np.asanyarray(hdu.data)
                # Make small, 2D, float32
                if data.ndim == 3 and data.shape[-1] == 1:
                    data = np.squeeze(data, axis=-1)
                data = data.astype(np.float32, copy=False)
                prev = _superpixel2x2(data)
                return np.ascontiguousarray(prev, dtype=np.float32)
        elif _is_xisf(path):
            x = XISF(path)
            im = x.read_image(0)  # channels_last
            if im is None:
                return None
            if im.ndim == 3 and im.shape[-1] == 1:
                im = np.squeeze(im, axis=-1)
            im = im.astype(np.float32, copy=False)
            prev = _superpixel2x2(im)
            return np.ascontiguousarray(prev, dtype=np.float32)
        else:
            # Non-FITS raster: you can fall back to PIL if desired
            return None
    except Exception:
        return None

def _synth_header_from_xisf_meta(im_meta: dict) -> fits.Header:
    """Make a minimal FITS-like Header from XISF image metadata."""
    h = fits.Header()
    try:
        # FITSKeywords shape: { KEY: [ {value, comment}, ...] }
        kwords = im_meta.get("FITSKeywords", {}) or {}
        def _first(key):
            v = kwords.get(key)
            return None if not v else v[0].get("value")
        # Filter / Exposure
        flt = _first("FILTER")
        if flt is not None: h["FILTER"] = str(flt)
        for k in ("EXPOSURE", "EXPTIME"):
            v = _first(k)
            if v is not None:
                try: h[k] = float(v)
                except Exception: h[k] = v
                break
        bp = _first("BAYERPAT")
        if bp: h["BAYERPAT"] = str(bp)
        # A couple of safe hints
        h["ORIGIN"] = "XISF-import"
    except Exception:
        pass
    return h

def _load_image_for_stack(path: str):
    """
    Return (img_float32_contig, header, is_mono) for either FITS or XISF.
    Never returns a read-only view; always writeable contiguous float32.
    """
    p = path.lower()
    if p.endswith((".fit", ".fits", ".fz")):
        with fits.open(path, memmap=True) as hdul:
            # choose first image-like HDU
            hdu = None
            for h in hdul:
                if getattr(h, "data", None) is not None:
                    hdu = h; break
            if hdu is None or hdu.data is None:
                raise IOError(f"No image data in FITS: {path}")
            arr = np.asanyarray(hdu.data)
            # Make a safe float32 copy (writeable, native-endian, contiguous)
            img = np.array(arr, dtype=np.float32, copy=True, order="C")
            if img.ndim == 3 and img.shape[-1] == 1:
                img = np.squeeze(img, axis=-1)
            hdr = hdu.header or fits.Header()
            is_mono = bool(img.ndim == 2 or (img.ndim == 3 and img.shape[-1] == 1))
            return img, hdr, is_mono

    if p.endswith(".xisf"):
        x = XISF(path)
        im_meta = x.get_images_metadata()[0]
        arr = x.read_image(0)  # channels_last
        if arr is None:
            raise IOError(f"No image data in XISF: {path}")
        img = np.array(arr, dtype=np.float32, copy=True, order="C")
        if img.ndim == 3 and img.shape[-1] == 1:
            img = np.squeeze(img, axis=-1)
        hdr = _synth_header_from_xisf_meta(im_meta)
        is_mono = bool(img.ndim == 2 or (img.ndim == 3 and img.shape[-1] == 1))
        return img, hdr, is_mono

    # Fallback: try FITS loader so we get a useful error
    with fits.open(path, memmap=True) as hdul:
        hdu = hdul[0]
        arr = np.asanyarray(hdu.data)
        img = np.array(arr, dtype=np.float32, copy=True, order="C")
        hdr = hdu.header or fits.Header()
        is_mono = bool(img.ndim == 2 or (img.ndim == 3 and img.shape[-1] == 1))
        return img, hdr, is_mono

class _MMImage:
    """
    Unified memory-friendly reader for FITS (memmap) and XISF.
    Exposes: .shape, .ndim, .read_tile(y0,y1,x0,x1), .read_full(), .close()
    Always returns float32 arrays (writeable) with color last if present.
    """

    def __init__(self, path: str):
        self.path = path
        self._kind = None          # "fits" | "xisf"
        self._scale = None         # integer -> [0..1] scale
        self._fits_hdul = None
        self._xisf = None
        self._xisf_memmap = None   # np.memmap when possible
        self._xisf_arr = None      # decompressed ndarray when needed
        self._xisf_color_axis = None
        self._xisf_spat_axes = (0, 1)
        self._xisf_dtype = None

        p = path.lower()
        if p.endswith((".fit", ".fits", ".fz")):
            self._open_fits(path)
            self._kind = "fits"
        elif p.endswith(".xisf"):
            if XISF is None:
                raise RuntimeError("XISF support not available (import failed).")
            self._open_xisf(path)
            self._kind = "xisf"
        else:
            # let FITS try anyway so you get a useful error text
            self._open_fits(path)
            self._kind = "fits"

    # ---------------- FITS ----------------
    def _open_fits(self, path: str):
        self._fits_hdul = fits.open(path, memmap=True)
        # choose first image-like HDU (don’t assume PRIMARY only)
        hdu = None
        for h in self._fits_hdul:
            if getattr(h, "data", None) is not None:
                hdu = h; break
        if hdu is None or hdu.data is None:
            raise ValueError(f"Empty FITS: {path}")

        d = hdu.data
        self._fits_data = d
        self.shape = d.shape
        self.ndim  = d.ndim
        self._orig_dtype = d.dtype

        # Detect color axis (size==3) and spatial axes once
        if self.ndim == 2:
            self._color_axis = None
            self._spat_axes  = (0, 1)
        elif self.ndim == 3:
            dims = self.shape
            self._color_axis = next((i for i, s in enumerate(dims) if s == 3), None)
            self._spat_axes  = (0, 1) if self._color_axis is None else tuple(i for i in range(3) if i != self._color_axis)
        else:
            raise ValueError(f"Unsupported ndim={self.ndim} for {path}")

        # late normalization scale for integer data
        if   self._orig_dtype == np.uint8:  self._scale = 1.0/255.0
        elif self._orig_dtype == np.uint16: self._scale = 1.0/65535.0
        else:                                self._scale = None

    # ---------------- XISF ----------------
    def _open_xisf(self, path: str):
        x = XISF(path)
        ims = x.get_images_metadata()
        if not ims:
            raise ValueError(f"Empty XISF: {path}")
        m0 = ims[0]
        self._xisf = x
        self._xisf_dtype = m0["dtype"]              # numpy dtype
        w, h, chc = m0["geometry"]                  # (width, height, channels)
        self.shape = (h, w) if chc == 1 else (h, w, chc)
        self.ndim  = 2 if chc == 1 else 3

        # color and spatial axes (image data is planar CHW on disk, we expose HWC)
        self._xisf_color_axis = None if chc == 1 else 2
        self._xisf_spat_axes  = (0, 1)

        # choose integer scale (same convention as FITS branch)
        if   self._xisf_dtype == np.dtype("uint8"):  self._scale = 1.0/255.0
        elif self._xisf_dtype == np.dtype("uint16"): self._scale = 1.0/65535.0
        else:                                        self._scale = None

        # Try zero-copy memmap only when the image block is an uncompressed attachment.
        loc = m0.get("location", None)
        comp = m0.get("compression", None)
        if isinstance(loc, tuple) and loc[0] == "attachment" and not comp:
            # location = ("attachment", pos, size)
            pos = int(loc[1])
            # on-disk order for XISF planar is (C,H,W). We memmap that and rearrange at slice time.
            chc = 1 if self.ndim == 2 else self.shape[2]
            shp_on_disk = (chc, self.shape[0], self.shape[1])
            # Align dtype endianness to native when mapping
            dt = self._xisf_dtype.newbyteorder("<") if self._xisf_dtype.byteorder == ">" else self._xisf_dtype
            self._xisf_memmap = np.memmap(path, mode="r", dtype=dt, offset=pos, shape=shp_on_disk)
            self._xisf_arr = None
        else:
            # Compressed / inline / embedded → must decompress whole image once
            arr = x.read_image(0)  # HWC float/uint per metadata
            # Ensure we own a writeable, contiguous float32 buffer
            self._xisf_arr = np.array(arr, dtype=np.float32, copy=True, order="C")
            self._xisf_memmap = None

    # ---------------- common API ----------------
    def read_tile(self, y0, y1, x0, x1) -> np.ndarray:
        if self._kind == "fits":
            d = self._fits_data
            if self.ndim == 2:
                tile = d[y0:y1, x0:x1]
            else:
                sl = [slice(None)]*3
                sl[self._spat_axes[0]] = slice(y0, y1)
                sl[self._spat_axes[1]] = slice(x0, x1)
                tile = d[tuple(sl)]
                if (self._color_axis is not None) and (self._color_axis != 2):
                    tile = np.moveaxis(tile, self._color_axis, -1)
        else:
            if self._xisf_memmap is not None:
                # memmapped (C,H,W) → slice, then move to (H,W,C)
                C = 1 if self.ndim == 2 else self.shape[2]
                if C == 1:
                    tile = self._xisf_memmap[0, y0:y1, x0:x1]
                else:
                    tile = np.moveaxis(self._xisf_memmap[:, y0:y1, x0:x1], 0, -1)
            else:
                # decompressed full array (H,W[,C]) → slice
                tile = self._xisf_arr[y0:y1, x0:x1]  # already HWC or HW

        # late normalize → float32 writeable, contiguous
        if self._scale is None:
            out = np.array(tile, dtype=np.float32, copy=True, order="C")
        else:
            out = np.array(tile, dtype=np.float32, copy=True, order="C")
            out *= self._scale

        # ensure (h,w,3) or (h,w)
        if out.ndim == 3 and out.shape[-1] not in (1, 3):
            if out.shape[0] == 3 and out.shape[-1] != 3:
                out = np.moveaxis(out, 0, -1)
        if out.ndim == 3 and out.shape[-1] == 1:
            out = np.squeeze(out, axis=-1)
        return out

    def read_full(self) -> np.ndarray:
        if self._kind == "fits":
            d = self._fits_data
            if self.ndim == 2:
                full = d
            else:
                full = d
                if (self._color_axis is not None) and (self._color_axis != 2):
                    full = np.moveaxis(full, self._color_axis, -1)
        else:
            if self._xisf_memmap is not None:
                C = 1 if self.ndim == 2 else self.shape[2]
                full = self._xisf_memmap[0] if C == 1 else np.moveaxis(self._xisf_memmap, 0, -1)
            else:
                full = self._xisf_arr

        # late normalize → float32 writeable, contiguous
        if self._scale is None:
            out = np.array(full, dtype=np.float32, copy=True, order="C")
        else:
            out = np.array(full, dtype=np.float32, copy=True, order="C")
            out *= self._scale

        if out.ndim == 3 and out.shape[-1] not in (1, 3):
            if out.shape[0] == 3 and out.shape[-1] != 3:
                out = np.moveaxis(out, 0, -1)
        if out.ndim == 3 and out.shape[-1] == 1:
            out = np.squeeze(out, axis=-1)
        return out

    def close(self):
        try:
            if self._fits_hdul is not None:
                self._fits_hdul.close()
        except Exception:
            pass

def _open_sources_for_mfdeconv(paths, log):
    srcs = []
    try:
        for p in paths:
            srcs.append(_MMImage(p))   # <-- handles FITS or XISF
        return srcs
    except Exception as e:
        # close anything we already opened
        for s in srcs:
            try: s.close()
            except Exception as e:
                import logging
                logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")
        raise RuntimeError(f"{e}")

# --- RAW helpers ------------------------------------------------------------

def _rawpy_pattern_to_token(rp) -> Optional[str]:
    """
    Turn rawpy's 2x2 raw_pattern + color_desc into 'RGGB' | 'BGGR' | 'GRBG' | 'GBRG'.
    Returns None if not Bayer.
    """
    try:
        pat = np.array(getattr(rp, "raw_pattern"))
        desc = getattr(rp, "color_desc")  # e.g. b'RGBG' or similar
        if pat.shape != (2, 2) or desc is None:
            return None
        # Map indices in raw_pattern (0..3) to letters in color_desc
        letters = []
        for y in range(2):
            for x in range(2):
                idx = int(pat[y, x])
                c = chr(desc[idx]).upper() if isinstance(desc, (bytes, bytearray)) else str(desc[idx]).upper()
                # Normalize to R/G/B only
                if c.startswith('R'): c = 'R'
                elif c.startswith('G'): c = 'G'
                elif c.startswith('B'): c = 'B'
                else: return None
                letters.append(c)
        token = ''.join(letters)
        return token if token in {"RGGB","BGGR","GRBG","GBRG"} else None
    except Exception:
        return None


def _is_xtrans_from_path(path: str) -> Optional[bool]:
    """
    Returns True if X-Trans, False if definitely Bayer, None if unknown.
    """
    try:
        import rawpy
        with rawpy.imread(path) as rp:
            if getattr(rp, "xtrans_pattern", None) is not None:
                return True
            if getattr(rp, "raw_pattern", None) is not None:
                return False
    except Exception:
        pass
    return None

def _probably_fuji_xtrans(file_path: str, header) -> bool:
    p = str(file_path).lower()
    if p.endswith(".raf"):
        return True  # RAF is Fujifilm; X-Series uses X-Trans (GFX uses Bayer but extension still RAF)
    # Look at header if present
    try:
        make  = str(header.get("MAKE", "") or header.get("CAMERAM", "")).upper()
        model = str(header.get("MODEL", "")).upper()
    except Exception:
        make = model = ""
    if "FUJIFILM" in make and any(t in model for t in (
        "X-T", "X-E", "X-PRO", "X-H", "X100", "X-S", "X30", "X70"
    )):
        return True
    return False


def _rawpy_is_xtrans_or_bayer(rp) -> str:
    """
    Returns 'XTRANS', 'BAYER', or 'UNKNOWN' from a rawpy handle without trusting a single flag.
    """
    try:
        # Prefer explicit X-Trans structure if available
        if getattr(rp, "xtrans_pattern", None) is not None:
            return "XTRANS"
    except Exception:
        pass
    try:
        pat = getattr(rp, "raw_pattern", None)
        if pat is not None and np.asarray(pat).shape == (2, 2):
            return "BAYER"
    except Exception:
        pass
    # LibRaw builds vary; check metadata as a heuristic
    try:
        md = getattr(rp, "metadata", None)
        make  = (md.make or "").upper() if md else ""
        model = (md.model or "").upper() if md else ""
        if "FUJIFILM" in make and ("GFX" not in model):  # GFX medium format is Bayer
            return "XTRANS"
    except Exception:
        pass
    return "UNKNOWN"

def _fmt2(x):
    return "NA" if x is None else f"{float(x):.2f}"

def _norm(p: str | Path) -> str:
    # normalize + collapse .. + native separators + case-insensitive key on Windows
    s = os.path.normpath(os.path.abspath(os.fspath(p)))
    return os.path.normcase(s) if os.name == "nt" else s

def _native(p: str | Path) -> str:
    # for UI display only (pretty backslashes on Windows)
    return os.path.normpath(os.fspath(p))



RAW_EXTS = ('.cr2', '.cr3', '.nef', '.arw', '.dng', '.raf', '.orf', '.rw2', '.pef')


def _is_raw_file(path: str) -> bool:
    return path.lower().endswith(RAW_EXTS)


def _parse_fraction_or_float(val) -> float | None:
    """
    Accepts things like '1/125', '0.008', 8, or exifread Ratio objects.
    Returns float seconds or None.
    """
    s = str(val).strip()
    if not s:
        return None
    try:
        # exifread often gives a single Ratio or list of one Ratio
        if hasattr(val, "num") and hasattr(val, "den"):
            return float(val.num) / float(val.den)
        if isinstance(val, (list, tuple)) and val and hasattr(val[0], "num"):
            r = val[0]
            return float(r.num) / float(r.den)

        if '/' in s:
            num, den = s.split('/', 1)
            return float(num) / float(den)
        return float(s)
    except Exception:
        return None


def _parse_exif_datetime(dt_str: str) -> str | None:
    """
    EXIF typically: 'YYYY:MM:DD HH:MM:SS'.
    Returns ISO-like 'YYYY-MM-DDTHH:MM:SS' or None.
    """
    s = str(dt_str).strip()
    if not s:
        return None

    # exifread sometimes formats as "YYYY:MM:DD HH:MM:SS"
    try:
        date_part, time_part = s.split(' ', 1)
        y, m, d = date_part.split(':', 2)
        return f"{int(y):04d}-{int(m):02d}-{int(d):02d}T{time_part}"
    except Exception:
        return None


def _ensure_minimal_header(header, file_path: str) -> fits.Header:
    """
    Guarantee we have a FITS Header. For non-FITS sources (TIFF/PNG/JPG/etc),
    synthesize a basic header and fill DATE-OBS from file mtime if missing.
    """
    if header is None:
        header = fits.Header()
        header["SIMPLE"]  = True
        header["BITPIX"]  = 16
        header["CREATOR"] = "SetiAstroSuite"

    # Try to provide DATE-OBS if not present
    if "DATE-OBS" not in header:
        try:
            ts = os.path.getmtime(file_path)
            dt = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc)
            header["DATE-OBS"] = (
                dt.isoformat(timespec="seconds"),
                "File modification time (UTC) used as DATE-OBS"
            )
        except Exception:
            pass

    return header

def _try_load_raw_with_rawpy(filename, allow_thumb_preview=True, debug_thumb=False):
    import rawpy
    from astropy.io import fits

    raw = rawpy.imread(filename)
    img = raw.postprocess(output_bps=16, no_auto_bright=True, gamma=(1,1))

    hdr = fits.Header()
    m = raw.metadata

    if m.exposure is not None:
        hdr['EXPTIME'] = (float(m.exposure), "Exposure time (s) from RAW metadata")
    if m.iso is not None:
        hdr['ISO'] = (int(m.iso), "ISO from RAW metadata")
    if m.aperture is not None:
        hdr['FNUMBER'] = (float(m.aperture), "F-number from RAW metadata")
    if m.focal_len is not None:
        hdr['FOCALLEN'] = (float(m.focal_len), "Focal length (mm) from RAW metadata")
    if m.make or m.model:
        cam = f"{m.make or ''} {m.model or ''}".strip()
        if cam:
            hdr['INSTRUME'] = cam
            hdr['CAMERA']   = cam

    # timestamp is usually epoch seconds
    if m.timestamp:
        dt = datetime.datetime.fromtimestamp(m.timestamp, tz=datetime.timezone.utc)
        hdr['DATE-OBS'] = (dt.isoformat(timespec='seconds'), "RAW timestamp (UTC)")

    bit_depth = "16-bit"
    is_mono = False

    return img.astype(np.float32) / 65535.0, hdr, bit_depth, is_mono


def _enrich_header_from_exif(header: fits.Header, file_path: str) -> fits.Header:
    """
    Merge EXIF metadata from a RAW file into an existing header without
    blowing away other keys. Only fills keys that are missing.
    """
    header = header.copy() if header is not None else fits.Header()
    header.setdefault("SIMPLE", True)
    header.setdefault("BITPIX", 16)
    header.setdefault("CREATOR", "SetiAstroSuite")

    try:
        with open(file_path, "rb") as f:
            tags = exifread.process_file(f, details=False)
    except Exception:
        # Can't read EXIF → just return what we have
        return header

    def get_tag(*names):
        for n in names:
            t = tags.get(n)
            if t is not None:
                return t
        return None

    # Exposure time
    exptime_tag = get_tag("EXIF ExposureTime", "EXIF ShutterSpeedValue")
    if exptime_tag and "EXPTIME" not in header:
        val = _parse_fraction_or_float(exptime_tag.values)
        if val is not None:
            header["EXPTIME"] = (float(val), "Exposure time (s) from EXIF")

    # ISO
    iso_tag = get_tag("EXIF ISOSpeedRatings", "EXIF PhotographicSensitivity")
    if iso_tag and "ISO" not in header:
        try:
            header["ISO"] = (int(str(iso_tag.values)), "ISO from EXIF")
        except Exception:
            header["ISO"] = (str(iso_tag.values), "ISO from EXIF")

    # Date/time
    date_tag = get_tag(
        "EXIF DateTimeOriginal",
        "EXIF DateTimeDigitized",
        "Image DateTime",
    )
    if date_tag and "DATE-OBS" not in header:
        dt = _parse_exif_datetime(date_tag.values)
        if dt:
            header["DATE-OBS"] = (dt, "Start of exposure (camera local time)")

    # Aperture
    fnum_tag = get_tag("EXIF FNumber")
    if fnum_tag and "FNUMBER" not in header:
        val = _parse_fraction_or_float(fnum_tag.values)
        if val is not None:
            header["FNUMBER"] = (float(val), "F-number (aperture)")

    # Focal length
    fl_tag = get_tag("EXIF FocalLength")
    if fl_tag and "FOCALLEN" not in header:
        val = _parse_fraction_or_float(fl_tag.values)
        if val is not None:
            header["FOCALLEN"] = (float(val), "Focal length (mm)")

    # Camera make/model
    make_tag  = get_tag("Image Make")
    model_tag = get_tag("Image Model")
    cam_parts = []
    if make_tag:
        cam_parts.append(str(make_tag.values).strip())
    if model_tag:
        cam_parts.append(str(model_tag.values).strip())
    camera_str = " ".join(p for p in cam_parts if p)
    if camera_str:
        header.setdefault("INSTRUME", camera_str)  # instrument / camera
        header.setdefault("CAMERA", camera_str)    # custom keyword

    return header




