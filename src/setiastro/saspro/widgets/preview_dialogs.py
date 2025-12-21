# pro/widgets/preview_dialogs.py
"""
Centralized preview dialog widgets for Seti Astro Suite Pro.
"""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt, QTimer, QPoint, QEvent
from PyQt6.QtGui import QImage, QPixmap, QIcon
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QScrollArea, QToolButton
)

from setiastro.saspro.widgets.themed_buttons import themed_toolbtn

# Import stretch functions - these are core to the app
from setiastro.saspro.imageops.stretch import stretch_mono_image, stretch_color_image

def _as_float01(img: np.ndarray) -> np.ndarray:
    a = np.asarray(img)

    # Integers: scale by dtype max
    if a.dtype.kind in "ui":
        maxv = float(np.iinfo(a.dtype).max)
        if maxv <= 0:
            return np.zeros_like(a, dtype=np.float32)
        return (a.astype(np.float32) / maxv).clip(0.0, 1.0)

    # Floats: if already 0..1, keep; otherwise scale down
    af = a.astype(np.float32, copy=False)

    mx = float(np.nanmax(af)) if af.size else 0.0
    if not np.isfinite(mx) or mx <= 1.0:
        return np.clip(af, 0.0, 1.0)

    # Common “display buffers”: 0..255 or 0..65535 stored as float
    if mx <= 255.5:
        return np.clip(af / 255.0, 0.0, 1.0)
    if mx <= 65535.5:
        return np.clip(af / 65535.0, 0.0, 1.0)

    # Fallback: normalize by max
    return np.clip(af / mx, 0.0, 1.0)


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
        self.setWindowTitle(self.tr("Image Preview"))
        self.resize(640, 480)
        
        self.autostretch_enabled = False
        self.is_mono = is_mono
        self.zoom_factor = 1.0
        # Drag-to-pan state
        self._panning = False
        self._pan_last = QPoint()
        
        # Store the image, ensure float32 [0,1]
        self.np_image = _as_float01(np_image)
        
        # Store the image, ensure float32 [0,1]
        self.np_image = _as_float01(np_image)

        # Layout
        layout = QVBoxLayout(self)

        # Toolbar row (themed)
        bar = QHBoxLayout()

        self.autostretch_button = QToolButton()
        self.autostretch_button.setText(self.tr("AutoStretch (Off)"))
        self.autostretch_button.setToolTip(self.tr("Toggle AutoStretch"))
        self.autostretch_button.setCheckable(True)
        self.autostretch_button.toggled.connect(self._toggle_autostretch)
        bar.addWidget(self.autostretch_button)

        bar.addStretch(1)

        self.zoom_in_button = themed_toolbtn("zoom-in", self.tr("Zoom In"))
        self.zoom_out_button = themed_toolbtn("zoom-out", self.tr("Zoom Out"))
        self.zoom_1to1_button = themed_toolbtn("zoom-original", self.tr("1:1 (100%)"))
        self.fit_button = themed_toolbtn("zoom-fit-best", self.tr("Fit to Preview"))

        self.zoom_in_button.clicked.connect(self._zoom_in)
        self.zoom_out_button.clicked.connect(self._zoom_out)
        self.zoom_1to1_button.clicked.connect(self._one_to_one)
        self.fit_button.clicked.connect(self._fit_to_preview)

        bar.addWidget(self.zoom_in_button)
        bar.addWidget(self.zoom_out_button)
        bar.addWidget(self.zoom_1to1_button)
        bar.addWidget(self.fit_button)

        layout.addLayout(bar)
        
        # Scroll area
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        layout.addWidget(self.scroll_area)
        
        # Image label
        self.image_label = QLabel()
        self.scroll_area.setWidget(self.image_label)
        self.image_label.installEventFilter(self)
        self.scroll_area.viewport().installEventFilter(self)
        self.image_label.setText("")
        self.image_label.setMouseTracking(True)        
        # Display initial image
        self._display_image(self.np_image)
        
        # Enable mouse wheel zoom
        self.image_label.installEventFilter(self)
        
        # Center scrollbars after layout
        QTimer.singleShot(0, self._fit_to_preview)
    
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
            self.tr("AutoStretch (On)") if checked else self.tr("AutoStretch (Off)")
        )
        self._apply_display()
    
    def _one_to_one(self):
        self.zoom_factor = 1.0
        self._apply_display()

    def _fit_to_preview(self):
        # Fit image into the scroll viewport
        if self.image_label.pixmap() is None or self.image_label.pixmap().isNull():
            return
        vp = self.scroll_area.viewport().size()
        pm = self.image_label.pixmap()
        if pm.width() <= 0 or pm.height() <= 0:
            return

        # Compute zoom that fits current *source image* (not already scaled label)
        # So, recompute based on original image dims:
        base_h, base_w = self.np_image.shape[:2]
        if base_w <= 0 or base_h <= 0:
            return

        zx = vp.width() / float(base_w)
        zy = vp.height() / float(base_h)
        self.zoom_factor = max(0.01, min(zx, zy))
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
        # --- wheel zoom (keep exactly as you had) ---
        if source in (self.image_label, self.scroll_area.viewport()) and event.type() == QEvent.Type.Wheel:
            if event.angleDelta().y() > 0:
                self._zoom_in()
            else:
                self._zoom_out()
            return True

        # --- drag-to-pan ---
        if source in (self.image_label, self.scroll_area.viewport()):

            if event.type() == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
                self._panning = True
                self._pan_last = event.globalPosition().toPoint()
                # nice UX
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
                return True

            if event.type() == QEvent.Type.MouseMove and self._panning:
                cur = event.globalPosition().toPoint()
                delta = cur - self._pan_last
                self._pan_last = cur

                h = self.scroll_area.horizontalScrollBar()
                v = self.scroll_area.verticalScrollBar()
                h.setValue(h.value() - delta.x())
                v.setValue(v.value() - delta.y())
                return True

            if event.type() == QEvent.Type.MouseButtonRelease and event.button() == Qt.MouseButton.LeftButton and self._panning:
                self._panning = False
                self.unsetCursor()
                return True

            # if mouse leaves while dragging, stop panning so it doesn't get "stuck"
            if event.type() == QEvent.Type.Leave and self._panning:
                self._panning = False
                self.unsetCursor()
                return True

        return super().eventFilter(source, event)

