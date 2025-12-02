"""
Multiscale HDR Transform for astronomical images.
Implementation of À Trous (Starlet) wavelet decomposition with dynamic compression.
"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import convolve1d

from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSpinBox,
    QDoubleSpinBox, QCheckBox, QMessageBox, QGroupBox, QProgressBar
)
from PyQt6.QtGui import QFont

from .doc_manager import ImageDocument

class AstroMultiscaleHDR:
    """À Trous wavelet-based HDR transform for astronomical images."""
    
    def __init__(self):
        self._base_kernel = np.array([1.0/16.0, 1.0/4.0, 3.0/8.0, 1.0/4.0, 1.0/16.0], dtype=np.float64)

    def _dilated_kernel(self, step: int) -> np.ndarray:
        """Generate dilated kernel for À Trous decomposition."""
        if step <= 1:
            return self._base_kernel
        n = len(self._base_kernel)
        out_len = (n - 1) * step + 1
        out = np.zeros(out_len, dtype=np.float64)
        out[::step] = self._base_kernel
        return out

    def _starlet_transform(self, image: np.ndarray, num_scales: int) -> tuple[list[np.ndarray], np.ndarray]:
        """Perform Starlet (À Trous) wavelet decomposition."""
        layer_image = image.astype(np.float64, copy=True)
        details: list[np.ndarray] = []

        for scale in range(num_scales):
            step = 1 << scale
            k = self._dilated_kernel(step)
            smoothed = convolve1d(layer_image, k, axis=0, mode='mirror')
            smoothed = convolve1d(smoothed, k, axis=1, mode='mirror')
            detail = layer_image - smoothed
            details.append(detail)
            layer_image = smoothed

        return details, layer_image

    def _apply_compression(self, data: np.ndarray, strength: float) -> np.ndarray:
        """Apply exponential compression to brighten regions."""
        if strength <= 1e-6:
            return data
        safe = np.clip(data, 0.0, 1.0)
        gamma = 1.0 + (strength * 0.5)
        return np.power(safe, gamma)

    def _compute_mask(self, data: np.ndarray, threshold: float, softness: float) -> np.ndarray:
        """Generate soft mask for highlight protection."""
        if softness < 1e-4:
            return (data > threshold).astype(np.float64)
        k = 10.0 / softness
        mask = 1.0 / (1.0 + np.exp(-k * (data - threshold)))
        return mask

    def process(self,
                image: np.ndarray,
                num_layers: int = 6,
                threshold: float = 0.05,
                softness: float = 0.2,
                strength: float = 0.8,
                to_lightness: bool = True) -> np.ndarray:
        """Execute the Multiscale HDR transform."""
        image = np.ascontiguousarray(image, dtype=np.float64)
        is_color = (image.ndim == 3 and image.shape[2] in (3, 4))

        chroma_ratio = None
        target_channel = image
        alpha_channel = None

        if is_color and to_lightness:
            lum = np.dot(image[..., :3], [0.2126, 0.7152, 0.0722])
            chroma_ratio = image[..., :3] / (lum[..., np.newaxis] + 1e-9)
            if image.shape[2] == 4:
                alpha_channel = image[..., 3]
            target_channel = lum

        details, residual = self._starlet_transform(target_channel, num_layers)
        mask = self._compute_mask(residual, threshold, softness)
        compressed_residual = self._apply_compression(residual, strength)
        new_residual = (compressed_residual * mask) + (residual * (1.0 - mask))

        processed_lum = new_residual
        for d in details:
            processed_lum = processed_lum + d

        processed_lum = np.clip(processed_lum, 0.0, 1.0)

        if is_color and to_lightness:
            result_rgb = processed_lum[..., np.newaxis] * chroma_ratio
            if alpha_channel is not None:
                result = np.dstack((result_rgb, alpha_channel))
            else:
                result = result_rgb
            return np.clip(result, 0.0, 1.0)
        else:
            return processed_lum


class MultiscaleHDRThread(QThread):
    done = pyqtSignal(np.ndarray)
    error = pyqtSignal(str)

    def __init__(self, image: np.ndarray, params: dict, parent=None):
        super().__init__(parent)
        self.image = image.astype(np.float32, copy=False)
        self.params = params

    def run(self):
        try:
            proc = AstroMultiscaleHDR()
            out = proc.process(self.image,
                               num_layers=int(self.params.get('num_layers', 6)),
                               threshold=float(self.params.get('threshold', 0.05)),
                               softness=float(self.params.get('softness', 0.2)),
                               strength=float(self.params.get('strength', 0.8)),
                               to_lightness=bool(self.params.get('to_lightness', True)))
            self.done.emit(out.astype(np.float32))
        except Exception as e:
            self.error.emit(str(e))


class MultiscaleHDRTab(QDialog):
    """UI Panel for Multiscale HDR Transform."""
    
    def __init__(self, image_manager=None, doc_manager=None, parent=None, document: ImageDocument | None = None):
        try:
            super().__init__(parent)
            self.doc_manager = doc_manager or image_manager
            self.doc = document
            self.image = None
            self.thread = None
            self._last_result = None
            self.status_label = None  # Initialize as None first
            self.setMinimumSize(550, 400)
            
            # Build UI - this creates status_label
            try:
                self._build_ui()
            except Exception as e:
                import traceback
                traceback.print_exc()
                # Create a minimal status label for error display
                if self.status_label is None:
                    self.status_label = QLabel(f"UI Build Error: {e}", self)
                    layout = QVBoxLayout(self)
                    layout.addWidget(self.status_label)
                    self.setLayout(layout)
                return

            # Try to load image from document
            if self.doc is not None and getattr(self.doc, 'image', None) is not None:
                try:
                    self.set_image_from_doc(np.asarray(self.doc.image))
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    if self.status_label:
                        self.status_label.setText(f"Warning: Could not load image: {e}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            # Try to show error in a label
            try:
                layout = QVBoxLayout(self)
                error_label = QLabel(f"Initialization Error:\n{e}\n\nSee console for details.", self)
                error_label.setWordWrap(True)
                layout.addWidget(error_label)
                self.setLayout(layout)
            except Exception:
                pass
            raise

    def _build_ui(self):
        """Build the user interface."""
        main = QVBoxLayout(self)
        main.setContentsMargins(12, 12, 12, 12)
        main.setSpacing(10)
        self.setLayout(main)

        # Title
        title = QLabel("Multiscale HDR Transform")
        font = title.font()
        font.setPointSize(12)
        font.setBold(True)
        title.setFont(font)
        main.addWidget(title)

        desc = QLabel(
            "À Trous (Starlet) wavelet decomposition with dynamic compression.\n"
            "Recovers details in bright regions (nuclei, nebulosae) by selective compression."
        )
        desc.setStyleSheet("color: #666; font-size: 10px;")
        desc.setWordWrap(True)
        main.addWidget(desc)

        # Parameters group
        params_group = QGroupBox("Processing Parameters", self)
        params_layout = QVBoxLayout()

        # Row 1: Scales
        r1 = QHBoxLayout()
        r1.addWidget(QLabel("Wavelet Scales:"))
        self.spin_scales = QSpinBox(self)
        self.spin_scales.setRange(1, 10)
        self.spin_scales.setValue(6)
        self.spin_scales.setToolTip("Number of decomposition levels (higher = more detail recovery)")
        r1.addWidget(self.spin_scales)
        r1.addWidget(QLabel("(1–10)"))
        r1.addStretch()
        params_layout.addLayout(r1)

        # Row 2: Threshold
        r2 = QHBoxLayout()
        r2.addWidget(QLabel("Threshold:"))
        self.spin_thresh = QDoubleSpinBox(self)
        self.spin_thresh.setRange(0.0, 1.0)
        self.spin_thresh.setSingleStep(0.01)
        self.spin_thresh.setValue(0.05)
        self.spin_thresh.setToolTip("Bright region cutoff (0.0–1.0)")
        r2.addWidget(self.spin_thresh)
        r2.addWidget(QLabel("(0.0–1.0)"))
        r2.addStretch()
        params_layout.addLayout(r2)

        # Row 3: Softness
        r3 = QHBoxLayout()
        r3.addWidget(QLabel("Softness:"))
        self.spin_soft = QDoubleSpinBox(self)
        self.spin_soft.setRange(0.0, 1.0)
        self.spin_soft.setSingleStep(0.01)
        self.spin_soft.setValue(0.2)
        self.spin_soft.setToolTip("Mask transition smoothness")
        r3.addWidget(self.spin_soft)
        r3.addWidget(QLabel("(0.0–1.0)"))
        r3.addStretch()
        params_layout.addLayout(r3)

        # Row 4: Strength
        r4 = QHBoxLayout()
        r4.addWidget(QLabel("Strength (Overdrive):"))
        self.spin_strength = QDoubleSpinBox(self)
        self.spin_strength.setRange(0.0, 5.0)
        self.spin_strength.setSingleStep(0.05)
        self.spin_strength.setValue(0.8)
        self.spin_strength.setToolTip("Compression intensity for bright areas")
        r4.addWidget(self.spin_strength)
        r4.addWidget(QLabel("(0.0–5.0)"))
        r4.addStretch()
        params_layout.addLayout(r4)

        params_group.setLayout(params_layout)
        main.addWidget(params_group)

        # Color mode checkbox
        self.chk_light = QCheckBox("Process luminance only (preserve colors)", self)
        self.chk_light.setChecked(True)
        self.chk_light.setToolTip("When checked, only brightness is processed; colors remain unchanged")
        main.addWidget(self.chk_light)

        # Progress bar
        self.progress = QProgressBar(self)
        self.progress.setVisible(False)
        main.addWidget(self.progress)

        # Status label
        self.status_label = QLabel("Ready.", self)
        self.status_label.setStyleSheet("color: #666; font-size: 9px;")
        main.addWidget(self.status_label)

        # Buttons
        btns = QHBoxLayout()
        btns.setSpacing(8)

        self.btn_apply = QPushButton("Apply Transform", self)
        self.btn_apply.clicked.connect(self._on_apply)
        self.btn_apply.setToolTip("Apply the Multiscale HDR transform to the current image")
        btns.addWidget(self.btn_apply)

        self.btn_push = QPushButton("Push Result to New View", self)
        self.btn_push.clicked.connect(self._on_push)
        self.btn_push.setEnabled(False)
        self.btn_push.setToolTip("Create a new image window with the processed result")
        btns.addWidget(self.btn_push)

        main.addLayout(btns)
        main.addStretch()

    def set_image_from_doc(self, img: np.ndarray):
        """Load image from document."""
        self.image = img.astype(np.float32, copy=False)
        h, w = self.image.shape[:2]
        self.status_label.setText(f"Image loaded: {w}x{h}")

    def _on_apply(self):
        """Handle Apply button click."""
        if self.image is None:
            QMessageBox.information(self, "No Image", "Please load an image first (use Tools > Multiscale HDR from the main menu).")
            return

        params = {
            'num_layers': int(self.spin_scales.value()),
            'threshold': float(self.spin_thresh.value()),
            'softness': float(self.spin_soft.value()),
            'strength': float(self.spin_strength.value()),
            'to_lightness': bool(self.chk_light.isChecked())
        }

        self.btn_apply.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.status_label.setText("Processing... (this may take a moment)")

        try:
            self.thread = MultiscaleHDRThread(self.image, params, parent=self)
            self.thread.done.connect(self._on_done)
            self.thread.error.connect(self._on_error)
            self.thread.start()
        except Exception as e:
            self.btn_apply.setEnabled(True)
            self.progress.setVisible(False)
            self.status_label.setText("Error starting processing.")
            QMessageBox.critical(self, "Error", f"Failed to start processing:\n{e}")

    def _on_done(self, out: np.ndarray):
        """Handle processing completion."""
        self.btn_apply.setEnabled(True)
        self.progress.setVisible(False)
        self._last_result = out
        self.btn_push.setEnabled(True)

        # Try to apply to current document if available
        applied = False
        if isinstance(self.doc, ImageDocument):
            try:
                self.doc.apply_edit(lambda img: out, step_name="Multiscale HDR")
                self.status_label.setText("Successfully applied to current image. Use Ctrl+Z to undo.")
                applied = True
            except Exception as e:
                self.status_label.setText(f"Applied but could not update display: {e}")

        if not applied:
            self.status_label.setText("Processing complete! Click 'Push Result to New View' to save.")

        QMessageBox.information(
            self,
            "Success",
            "Multiscale HDR transform completed successfully!"
        )

    def _on_error(self, msg: str):
        """Handle processing error."""
        self.btn_apply.setEnabled(True)
        self.progress.setVisible(False)
        self.status_label.setText(f"Error: {msg}")
        QMessageBox.critical(self, "Processing Error", f"Transform failed:\n{msg}")

    def _on_push(self):
        """Push result to new view."""
        if self._last_result is None:
            QMessageBox.information(self, "No Result", "Run 'Apply Transform' first to generate a result.")
            return

        if self.doc_manager is None:
            QMessageBox.warning(self, "Error", "No document manager available.")
            return

        try:
            arr = np.asarray(self._last_result)
            self.doc_manager.open_array(arr, metadata={}, title="Multiscale HDR Result")
            self.status_label.setText("Result pushed to new view.")
        except Exception as e:
            self.status_label.setText(f"Push failed: {e}")
            QMessageBox.critical(self, "Push Failed", f"Could not create new image:\n{e}")
