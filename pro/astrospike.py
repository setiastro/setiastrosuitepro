# pro/astrospike.py
"""
AstroSpike integration module for Seti Astro Suite Pro.

This module provides a wrapper to use the AstroSpike script as a native dialog.
"""
from __future__ import annotations

import numpy as np
from PyQt6.QtWidgets import QDialog, QMessageBox, QApplication
from PyQt6.QtGui import QIcon
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QSizePolicy

# Import the script module from pro folder
from pro.astrospike_python import (
    AstroSpikeWindow,
)


class AstroSpikeContext:
    """
    Context object that mimics the script runner context for AstroSpike.
    """
    def __init__(self, get_image_callback=None, set_image_callback=None):
        self._get_image_callback = get_image_callback
        self._set_image_callback = set_image_callback
        self._log_messages = []
    
    def get_image(self):
        """Get the current image from the application."""
        if self._get_image_callback:
            return self._get_image_callback()
        return None
    
    def set_image(self, data: np.ndarray, step_name: str = "AstroSpike"):
        """Apply the modified image back to the application."""
        if self._set_image_callback:
            self._set_image_callback(data, step_name)
    
    def log(self, message: str):
        self._log_messages.append(message)
        print(f"[AstroSpike] {message}")


class AstroSpikeDialog(AstroSpikeWindow):
    """
    AstroSpike dialog adapted for use within the main application.
    """
    
    def __init__(self, parent=None, icon_path: str = None, get_image_callback=None, set_image_callback=None):
        """
        Initialize AstroSpike dialog.
        
        Args:
            parent: Parent widget
            icon_path: Path to window icon
            get_image_callback: Callback function to get the current image from the app
            set_image_callback: Callback function to set the result image back to the app
        """
        # Create context
        self.ctx = AstroSpikeContext(get_image_callback, set_image_callback)
        
        # Try to get image from callback, otherwise create blank
        image_data = None
        if get_image_callback:
            try:
                print("[AstroSpike] Calling get_image_callback...")
                image_data = get_image_callback()
                print(f"[AstroSpike] Callback returned image: {image_data.shape if image_data is not None else 'None'}")
            except Exception as e:
                print(f"[AstroSpike] Error getting image from callback: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("[AstroSpike] No get_image_callback provided")
        
        # If no image, create a blank one
        if image_data is None:
            print("[AstroSpike] No image provided, creating blank image")
            image_data = np.ones((512, 512, 3), dtype=np.float32) * 0.5
        
        # Ensure correct format
        if image_data.dtype != np.float32:
            if image_data.dtype == np.uint8:
                print(f"[AstroSpike] Converting uint8 to float32")
                image_data_float = image_data.astype(np.float32) / 255.0
            else:
                print(f"[AstroSpike] Converting {image_data.dtype} to float32")
                image_data_float = image_data.astype(np.float32)
        else:
            image_data_float = image_data
            if image_data_float.max() > 1.0:
                print(f"[AstroSpike] Image max is {image_data_float.max()}, dividing by 255")
                image_data_float = image_data_float / 255.0
        
        print(f"[AstroSpike] Formatted image: shape {image_data_float.shape}, dtype {image_data_float.dtype}, min {image_data_float.min():.3f}, max {image_data_float.max():.3f}")
        
        # Handle grayscale
        if len(image_data_float.shape) == 2:
            image_data_float = np.stack([image_data_float, image_data_float, image_data_float], axis=-1)
        elif image_data_float.shape[2] == 1:
            image_data_float = np.concatenate([image_data_float, image_data_float, image_data_float], axis=-1)
        
        # Ensure RGB (not RGBA)
        if image_data_float.shape[2] == 4:
            image_data_float = image_data_float[:, :, :3]
        
        # Convert to uint8 for QImage
        image_data_uint8 = (np.clip(image_data_float, 0, 1) * 255).astype(np.uint8)
        
        # Initialize parent with properly formatted images
        super().__init__(image_data_uint8, image_data_float, self.ctx)
        
        # Set parent
        if parent:
            self.setParent(parent)
        
        # Set icon
        if icon_path:
            self.setWindowIcon(QIcon(icon_path))
        
        self.setWindowTitle("AstroSpike - Star Diffraction Spikes")
        
        # Enable window frame with title bar and control buttons
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.Window)
        
    
    def resizeEvent(self, event):
        """Handle resize events to update internal layouts."""
        super().resizeEvent(event)
        # Force layout recalculation
        if self.layout():
            self.layout().update()
            self.layout().activate()


def open_astrospike_dialog(parent=None, icon_path: str = None, get_image_callback=None) -> bool:
    """
    Open the AstroSpike dialog.
    
    Args:
        parent: Parent widget
        icon_path: Window icon path
        get_image_callback: Callback to get current image
        
    Returns:
        True if dialog was accepted, False otherwise
    """
    dlg = AstroSpikeDialog(parent, icon_path, get_image_callback)
    dlg.showMaximized() 
    result = dlg.exec()
    return result == QDialog.DialogCode.Accepted
