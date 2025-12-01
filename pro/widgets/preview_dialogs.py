# pro/widgets/preview_dialogs.py
"""
Centralized preview dialog widgets for Seti Astro Suite Pro.
"""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt, QTimer, QPoint, QEvent
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QScrollArea
)

# Import stretch functions - these are core to the app
from imageops.stretch import stretch_mono_image, stretch_color_image


class ImagePreviewDialog(QDialog):
    """
    A dialog for previewing images with autostretch and zoom capabilities.
    
    Features:
    - AutoStretch toggle
    - Zoom in/out (buttons and mouse wheel)
    - Scroll area for panning
    - Supports mono and RGB images
    
    Usage:
        dialog = ImagePreviewDialog(np_image, is_mono=False)
        dialog.exec()
    """
    
    def __init__(self, np_image: np.ndarray, is_mono: bool = False, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Image Preview")
        self.resize(640, 480)
        
        self.autostretch_enabled = False
        self.is_mono = is_mono
        self.zoom_factor = 1.0
        
        # Store the image, ensure float32 [0,1]
        self.np_image = np.clip(np.asarray(np_image, dtype=np.float32), 0, 1)
        
        # Layout
        layout = QVBoxLayout(self)
        
        # Button row
        button_layout = QHBoxLayout()
        
        self.autostretch_button = QPushButton("AutoStretch (Off)")
        self.autostretch_button.setCheckable(True)
        self.autostretch_button.toggled.connect(self._toggle_autostretch)
        button_layout.addWidget(self.autostretch_button)
        
        self.zoom_in_button = QPushButton("Zoom In")
        self.zoom_in_button.clicked.connect(self._zoom_in)
        button_layout.addWidget(self.zoom_in_button)
        
        self.zoom_out_button = QPushButton("Zoom Out")
        self.zoom_out_button.clicked.connect(self._zoom_out)
        button_layout.addWidget(self.zoom_out_button)
        
        layout.addLayout(button_layout)
        
        # Scroll area
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        layout.addWidget(self.scroll_area)
        
        # Image label
        self.image_label = QLabel()
        self.scroll_area.setWidget(self.image_label)
        
        # Display initial image
        self._display_image(self.np_image)
        
        # Enable mouse wheel zoom
        self.image_label.installEventFilter(self)
        
        # Center scrollbars after layout
        QTimer.singleShot(0, self._center_scrollbars)
    
    def _display_image(self, np_img: np.ndarray):
        """Convert numpy array to QImage and display at current zoom."""
        # Convert to uint8
        arr = (np.clip(np_img, 0, 1) * 255).astype(np.uint8)
        
        if arr.ndim == 3 and arr.shape[2] == 3:
            h, w, _ = arr.shape
            qimg = QImage(arr.tobytes(), w, h, 3 * w, QImage.Format.Format_RGB888)
        elif arr.ndim == 2:
            h, w = arr.shape
            qimg = QImage(arr.tobytes(), w, h, w, QImage.Format.Format_Grayscale8)
        elif arr.ndim == 3 and arr.shape[2] == 1:
            arr = arr.squeeze()
            h, w = arr.shape
            qimg = QImage(arr.tobytes(), w, h, w, QImage.Format.Format_Grayscale8)
        else:
            raise ValueError(f"Unexpected image shape: {arr.shape}")
        
        # Apply zoom
        pixmap = QPixmap.fromImage(qimg)
        scaled_w = int(pixmap.width() * self.zoom_factor)
        scaled_h = int(pixmap.height() * self.zoom_factor)
        scaled_pixmap = pixmap.scaled(
            scaled_w, scaled_h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)
        self.image_label.adjustSize()
    
    def _toggle_autostretch(self, checked: bool):
        """Toggle autostretch on/off."""
        self.autostretch_enabled = checked
        self.autostretch_button.setText(
            "AutoStretch (On)" if checked else "AutoStretch (Off)"
        )
        self._apply_display()
    
    def _apply_display(self):
        """Apply current display settings (autostretch, zoom)."""
        target_median = 0.25
        
        if self.autostretch_enabled:
            if self.np_image.ndim == 2:
                stretched = stretch_mono_image(self.np_image, target_median)
                display_img = np.stack([stretched] * 3, axis=-1)
            elif self.np_image.ndim == 3 and self.np_image.shape[2] == 3:
                display_img = stretch_color_image(self.np_image, target_median, linked=False)
            elif self.np_image.ndim == 3 and self.np_image.shape[2] == 1:
                stretched = stretch_mono_image(self.np_image.squeeze(), target_median)
                display_img = np.stack([stretched] * 3, axis=-1)
            else:
                display_img = self.np_image
        else:
            if self.np_image.ndim == 2:
                display_img = np.stack([self.np_image] * 3, axis=-1)
            elif self.np_image.ndim == 3 and self.np_image.shape[2] == 1:
                display_img = np.repeat(self.np_image, 3, axis=2)
            else:
                display_img = self.np_image
        
        self._display_image(display_img)
    
    def _zoom_in(self):
        """Zoom in by 20%."""
        self.zoom_factor *= 1.2
        self._apply_display()
    
    def _zoom_out(self):
        """Zoom out by 20%."""
        self.zoom_factor /= 1.2
        self._apply_display()
    
    def _center_scrollbars(self):
        """Center the scroll area on the image."""
        h_bar = self.scroll_area.horizontalScrollBar()
        v_bar = self.scroll_area.verticalScrollBar()
        h_bar.setValue((h_bar.maximum() + h_bar.minimum()) // 2)
        v_bar.setValue((v_bar.maximum() + v_bar.minimum()) // 2)
    
    def eventFilter(self, source, event):
        """Handle mouse wheel for zooming."""
        if source == self.image_label and event.type() == QEvent.Type.Wheel:
            if event.angleDelta().y() > 0:
                self._zoom_in()
            else:
                self._zoom_out()
            return True
        return super().eventFilter(source, event)
