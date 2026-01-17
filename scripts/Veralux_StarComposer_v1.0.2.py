##############################################
# VeraLux — StarComposer (SASPro Port)
# High-Fidelity Star Reconstruction Engine
# Author: Riccardo Paterniti (2025)
# Contact: info@veralux.space
##############################################

# (c) 2025 Riccardo Paterniti
# VeraLux — StarComposer
# SPDX-License-Identifier: GPL-3.0-or-later
# Version 1.0.2 (SASPro Port)
#
# Credits / Origin
# ----------------
#   • Architecture: Powered by VeraLux Core v1.3 (Shared Math Engine)
#   • Math basis: Anchored Inverse Hyperbolic Stretch (IHS)
#   • Color Science: Vector-based Chrominance preservation (VCP)
#

"""
Overview
--------
A specialized photometric reconstruction engine designed for deep-sky 
astrophotography.

VeraLux StarComposer solves the "bloating" and "bleaching" issues inherent in 
standard star stretching by decoupling the stellar field from the main object.
It leverages the rigorous **Inverse Hyperbolic Stretch (IHS)** to develop linear 
star masks with vector precision, preserving true stellar color and geometry (PSF) 
before compositing them onto non-linear starless images.

Key Features v1.0
-------------------
• **Smart Composition Modes**: Exclusive toggles for 'Linear Add' vs 'Screen' blending.
• **Dynamic LSR**: Large Structure Rejection scales with image resolution.
• **Soft Landing Engine**: Highlights rolloff protection to prevent core saturation.
• **True Color Pipeline**: Full vector preservation (Zero Hue Shift).
• **SASPro Integration**: Opens result as new document for seamless workflow.

Design Goals
------------
• Treat stars as geometric entities (Gaussian/Moffat profiles).
• Prevent "white core" saturation using physical color convergence models.
• Repair optical defects (Chromatic Aberration) via Chroma-only convolution.
• Automate the cleanup of star mask residual artifacts (Shadow Convergence).

Core Features
-------------
• **The VeraLux Engine**: Anchored IHS function for conical star profiles.
• **Hybrid Physics**: Color Grip (Vector Lock) + Shadow Convergence (Damping).
• **Star Surgery**: Includes Optical Healing for halos and Dynamic LSR for galaxy core removal.

Usage
-----
1. **Load Data**: Import Starless (Non-Linear Base) and Starmask (Linear).
2. **Blend Mode**: Use 'Screen' if mask has bright residuals, 'Linear Add' otherwise.
3. **Intensity**: Increase "Star Intensity" (Gold Slider) to define brightness.
4. **Geometry**: Adjust "Profile Hardness" (b) to sculpt the PSF.
5. **Generate**: Click PROCESS. Result opens as a new document in SASPro.

Inputs & Outputs
----------------
Input: Linear Starmask FITS/TIFF + Stretched Starless FITS/TIFF.
Output: Recombined RGB (32-bit Float) in new document.

Compatibility
-------------
• Seti Astro Pro (SASPro)
• Python 3.10+
• Dependencies: PyQt6, numpy
• Optional: opencv-python (for Star Surgery features)

License
-------
Released under GPL-3.0-or-later.
"""

import sys
import os
import math
import traceback

import numpy as np

# OpenCV is optional - only needed for Star Surgery features
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                            QWidget, QLabel, QDoubleSpinBox, QSlider,
                            QPushButton, QGroupBox, QMessageBox, QProgressBar,
                            QComboBox, QCheckBox, QFileDialog, QGraphicsView, 
                            QGraphicsScene, QGraphicsPixmapItem, QButtonGroup)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QEvent, QSettings, QObject
from PyQt6.QtGui import QImage, QPixmap, QPainter, QColor, QPen, QBrush, QPainterPath

# ---------------------
#  SASPRO SCRIPT METADATA
# ---------------------
SCRIPT_NAME = "VeraLux StarComposer"
SCRIPT_GROUP = "Composition"

VERSION = "1.0.3"

# ------------------------------------------------------------------------------
# VERSION HISTORY
# ------------------------------------------------------------------------------
# 1.0.3: SASPro Integration Enhancement.
#        • Added "Open Current View" support alongside "Open File" for both
#          starmask and starless inputs.
#        • View selector combos to pick from open views using ctx.list_views().
#        • Refresh button to update open views list.
#        • Improved UX with clearer input source options.
# 1.0.2: SASPro Port.
#        • Ported from Siril to SASPro platform.
#        • Uses SASPro context for image loading/saving.
#        • Sensor DB update (same as VeraLux HyperMetric Stretch).
# 1.0.1: Import fix.
# ------------------------------------------------------------------------------

# ---------------------
#  THEME & STYLING
# ---------------------

DARK_STYLESHEET = """
QWidget { background-color: #2b2b2b; color: #e0e0e0; font-size: 10pt; }
QToolTip { background-color: #333333; color: #ffffff; border: 1px solid #88aaff; }
QGroupBox { border: 1px solid #444444; margin-top: 5px; font-weight: bold; border-radius: 4px; padding-top: 12px; }
QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 3px; color: #88aaff; }
QLabel { color: #cccccc; }

QCheckBox { spacing: 5px; color: #cccccc; }
QCheckBox::indicator { width: 14px; height: 14px; border: 1px solid #666666; background: #3c3c3c; border-radius: 3px; }
QCheckBox::indicator:checked { background-color: #285299; border: 1px solid #88aaff; }

/* Robust Slider Styling */
QSlider { min-height: 24px; }
QSlider::groove:horizontal { background: #444444; height: 6px; border-radius: 3px; }
QSlider::sub-page:horizontal { background: transparent; }
QSlider::add-page:horizontal { background: transparent; }
QSlider::handle:horizontal { 
    background-color: #cccccc; border: 1px solid #666666; 
    width: 14px; height: 14px; margin: -4px 0; border-radius: 7px; 
}
QSlider::handle:horizontal:hover { background-color: #ffffff; border-color: #88aaff; }
QSlider::handle:horizontal:pressed { background-color: #ffffff; border-color: #ffffff; }

/* Master Control Handle (Solid Gold) */
QSlider#MainSlider::handle:horizontal { background-color: #ffb000; border: 1px solid #cc8800; }
QSlider#MainSlider::handle:horizontal:hover { background-color: #ffcc00; border-color: #ffffff; }
QSlider#MainSlider::groove:horizontal { background: #554400; }

QPushButton { background-color: #444444; color: #dddddd; border: 1px solid #666666; border-radius: 4px; padding: 6px; font-weight: bold;}
QPushButton:hover { background-color: #555555; border-color: #777777; }

QPushButton#ProcessButton { background-color: #285299; border: 1px solid #1e3f7a; }
QPushButton#ProcessButton:hover { background-color: #355ea1; }

QPushButton#CloseButton { background-color: #5a2a2a; border: 1px solid #804040; }
QPushButton#CloseButton:hover { background-color: #7a3a3a; }

/* GHOST HELP BUTTON */
QPushButton#HelpButton { 
    background-color: transparent; 
    color: #555555; 
    border: none; 
    font-weight: bold; 
    min-width: 20px;
}
QPushButton#HelpButton:hover { 
    color: #aaaaaa; 
}

QPushButton#ZoomBtn { min-width: 30px; font-weight: bold; background-color: #3c3c3c; }

/* HMS-Style ComboBox */
QComboBox { background-color: #3c3c3c; color: #ffffff; border: 1px solid #555555; padding: 3px; border-radius: 3px; }
QComboBox:hover { border-color: #777777; }
QComboBox::drop-down { border: none; width: 20px; }
QComboBox::down-arrow { width: 0; height: 0; border-left: 4px solid transparent; border-right: 4px solid transparent; border-top: 6px solid #aaaaaa; margin-right: 6px; }
QComboBox QAbstractItemView { background-color: #3c3c3c; color: #ffffff; selection-background-color: #285299; border: 1px solid #555555; }

QProgressBar { border: 1px solid #555555; border-radius: 3px; text-align: center; }
QProgressBar::chunk { background-color: #285299; width: 10px; }
"""

# =============================================================================
#  SENSOR PROFILES (Database v2.1)
# =============================================================================

SENSOR_PROFILES = {
    "Rec.709 (Recommended)": (0.2126, 0.7152, 0.0722),
    "Sony IMX571 (ASI2600/QHY268)": (0.2944, 0.5021, 0.2035),
    "Sony IMX533 (ASI533)": (0.2910, 0.5072, 0.2018),
    "Sony IMX455 (ASI6200/QHY600)": (0.2987, 0.5001, 0.2013),
    "Sony IMX294 (ASI294)": (0.3068, 0.5008, 0.1925),
    "Sony IMX183 (ASI183)": (0.2967, 0.4983, 0.2050),
    "Sony IMX178 (ASI178)": (0.2346, 0.5206, 0.2448),
    "Sony IMX224 (ASI224)": (0.3402, 0.4765, 0.1833),
    "Sony IMX585 (ASI585) - STARVIS 2": (0.3431, 0.4822, 0.1747),
    "Sony IMX662 (ASI662) - STARVIS 2": (0.3430, 0.4821, 0.1749),
    "Sony IMX678/715 - STARVIS 2": (0.3426, 0.4825, 0.1750),
    "Panasonic MN34230 (ASI1600/QHY163)": (0.2650, 0.5250, 0.2100),
    "Canon EOS (Modern - 60D/6D/R)": (0.2550, 0.5250, 0.2200),
    "Canon EOS (Legacy - 300D/40D)": (0.2400, 0.5400, 0.2200),
    "Nikon DSLR (Modern - D5300/D850)": (0.2600, 0.5100, 0.2300),
    "ZWO Seestar S50": (0.3333, 0.4866, 0.1801),
    "ZWO Seestar S30": (0.2928, 0.5053, 0.2019),
    "Narrowband HOO": (0.5000, 0.2500, 0.2500),
    "Narrowband SHO": (0.3333, 0.3400, 0.3267),
}
DEFAULT_PROFILE = "Rec.709 (Recommended)"

# =============================================================================
#  CORE MATH (HMS v1.3.0)
# =============================================================================

class VeraLuxCore:
    @staticmethod
    def normalize_input(img_data):
        input_dtype = img_data.dtype
        img_float = img_data.astype(np.float32)
        if np.issubdtype(input_dtype, np.integer):
            if input_dtype == np.uint8: return img_float / 255.0
            elif input_dtype == np.uint16: return img_float / 65535.0
            else: return img_float / float(np.iinfo(input_dtype).max)
        elif np.issubdtype(input_dtype, np.floating):
            current_max = np.max(img_data)
            if current_max <= 1.0 + 1e-5: return img_float
            if current_max <= 65535.0: return img_float / 65535.0
            return img_float
        return img_float

    @staticmethod
    def calculate_anchor_adaptive(data_norm, weights):
        stride = max(1, data_norm.size // 1000000) 
        if data_norm.ndim == 3:
            r, g, b = weights
            L = r * data_norm[0] + g * data_norm[1] + b * data_norm[2]
            sample = L.flatten()[::stride]
        else:
            sample = data_norm.flatten()[::stride]
        valid = sample[sample > 0]
        if valid.size == 0: return 0.0
        return max(0.0, np.percentile(valid, 5.0))

    @staticmethod
    def extract_luminance(data_norm, anchor, weights):
        r_w, g_w, b_w = weights
        img_anchored = np.maximum(data_norm - anchor, 0.0)
        if data_norm.ndim == 3:
            L = (r_w * img_anchored[0] + g_w * img_anchored[1] + b_w * img_anchored[2])
        else:
            L = img_anchored
        return L, img_anchored

    @staticmethod
    def hyperbolic_stretch(data, D, b):
        D = max(D, 0.1); b = max(b, 0.1)
        term1 = np.arcsinh(D * data + b)
        term2 = np.arcsinh(b)
        norm_factor = np.arcsinh(D + b) - term2
        if norm_factor == 0: norm_factor = 1e-6
        return (term1 - term2) / norm_factor

# =============================================================================
#  STAR PIPELINE
# =============================================================================

def soft_clip_stars(img_rgb, threshold=0.98, rolloff=2.0):
    """HMS Soft Landing for Star Profiles"""
    mask = img_rgb > threshold
    if not np.any(mask): return img_rgb
    result = img_rgb.copy()
    t = np.clip((img_rgb[mask] - threshold) / (1.0 - threshold + 1e-9), 0.0, 1.0)
    result[mask] = threshold + (1.0 - threshold) * (1.0 - np.power(1.0 - t, rolloff))
    return np.clip(result, 0.0, 1.0)

def apply_optical_healing(img_rgb, strength):
    if strength <= 0: return img_rgb
    if not HAS_CV2:
        return img_rgb  # Skip if OpenCV not available
    img_cv = img_rgb.transpose(1, 2, 0)
    img_cv_8 = np.clip(img_cv * 255, 0, 255).astype(np.uint8)
    ycrcb = cv2.cvtColor(img_cv_8, cv2.COLOR_RGB2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    
    ksize = int(strength * 2) + 1
    cr = cv2.GaussianBlur(cr, (ksize, ksize), 0)
    cb = cv2.GaussianBlur(cb, (ksize, ksize), 0)
    
    merged = cv2.merge([y, cr, cb])
    rgb_heal = cv2.cvtColor(merged, cv2.COLOR_YCrCb2RGB)
    return rgb_heal.transpose(2, 0, 1).astype(np.float32) / 255.0

def apply_star_reduction(img_rgb, intensity):
    if intensity <= 0: return img_rgb
    if not HAS_CV2:
        return img_rgb  # Skip if OpenCV not available
    k_size = 3 if intensity < 0.5 else 5
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    img_hwc = img_rgb.transpose(1, 2, 0)
    eroded = cv2.erode(img_hwc, kernel, iterations=1)
    return (img_hwc * (1.0 - intensity) + eroded * intensity).transpose(2, 0, 1)

def apply_large_structure_rejection(img_rgb, intensity):
    """
    Core Rejection (LSR): Removes large blobs using Difference of Gaussians.
    Dynamic Kernel Size: scales with image to target actual structures.
    """
    if intensity <= 0: return img_rgb
    if not HAS_CV2:
        return img_rgb  # Skip if OpenCV not available
    
    h, w = img_rgb.shape[1], img_rgb.shape[2]
    # Dynamic Kernel: 1/15th of image
    k_size_val = int(min(h, w) / 15.0) 
    if k_size_val % 2 == 0: k_size_val += 1 
    if k_size_val < 3: k_size_val = 3
    
    img_hwc = img_rgb.transpose(1, 2, 0)
    low_pass = cv2.GaussianBlur(img_hwc, (k_size_val, k_size_val), 0)
    high_pass = np.maximum(img_hwc - low_pass, 0.0)
    result = img_hwc * (1.0 - intensity) + high_pass * intensity
    
    return result.transpose(2, 0, 1)

def process_star_pipeline(starmask, D, b, conv, grip, shadow, reduction, healing, lsr, weights, use_adaptive):
    img = VeraLuxCore.normalize_input(starmask)
    if img.ndim == 2: img = np.array([img, img, img]) 
    
    anchor = VeraLuxCore.calculate_anchor_adaptive(img, weights) if use_adaptive else 0.0
    L_anchored, img_anchored = VeraLuxCore.extract_luminance(img, anchor, weights)
    
    L_str = VeraLuxCore.hyperbolic_stretch(L_anchored, 10.0**D, b)
    L_str = np.clip(L_str, 0.0, 1.0)
    
    epsilon = 1e-9; L_safe = L_anchored + epsilon
    r_ratio = img_anchored[0] / L_safe
    g_ratio = img_anchored[1] / L_safe
    b_ratio = img_anchored[2] / L_safe
    
    k = np.power(L_str, conv)
    r_final = r_ratio * (1.0 - k) + 1.0 * k
    g_final = g_ratio * (1.0 - k) + 1.0 * k
    b_final = b_ratio * (1.0 - k) + 1.0 * k
    
    final = np.zeros_like(img)
    final[0] = L_str * r_final
    final[1] = L_str * g_final
    final[2] = L_str * b_final
    
    # Hybrid Engine (Shadow Convergence)
    D_val = 10.0 ** D
    scalar = np.zeros_like(final)
    scalar[0] = VeraLuxCore.hyperbolic_stretch(img_anchored[0], D_val, b)
    scalar[1] = VeraLuxCore.hyperbolic_stretch(img_anchored[1], D_val, b)
    scalar[2] = VeraLuxCore.hyperbolic_stretch(img_anchored[2], D_val, b)
    scalar = np.clip(scalar, 0.0, 1.0)
    
    grip_map = np.full_like(L_str, grip)
    if shadow > 0.01:
        damping = np.power(L_str, shadow)
        grip_map = grip_map * damping
        
    final = (final * grip_map) + (scalar * (1.0 - grip_map))
    final = np.clip(final, 0.0, 1.0).astype(np.float32)
    
    # 5. Surgery
    if lsr > 0: final = apply_large_structure_rejection(final, lsr)
    if healing > 0: final = apply_optical_healing(final, healing)
    if reduction > 0: final = apply_star_reduction(final, reduction)
        
    final = soft_clip_stars(final, threshold=0.98, rolloff=2.0)
    return final

# =============================================================================
#  GUI WIDGETS
# =============================================================================

class ResetSlider(QSlider):
    def __init__(self, orientation, default_val=0, parent=None):
        super().__init__(orientation, parent)
        self.default_val = default_val
    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.setValue(self.default_val); event.accept()
        else: super().mouseDoubleClickEvent(event)

class HistogramOverlay(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)
        self.hist_data = None
        self.clip_white = False
        self.pct_white = 0.0
        self.use_log = False
        self.setFixedSize(200, 100)
        
    def set_log_scale(self, val):
        self.use_log = val
        self.update()

    def set_data(self, img):
        if img is None: return
        bins = 256
        r = np.histogram(img[0], bins=bins, range=(0, 1))[0]
        g = np.histogram(img[1], bins=bins, range=(0, 1))[0]
        b = np.histogram(img[2], bins=bins, range=(0, 1))[0]
        mx = max(np.max(r), np.max(g), np.max(b))
        if mx > 0: self.hist_data = (r/mx, g/mx, b/mx)
        else: self.hist_data = None
        
        tot = img.shape[1] * img.shape[2]
        eps = 1e-5
        self.pct_white = (np.count_nonzero(img >= 1.0-eps) / tot) * 100
        self.clip_white = self.pct_white > 0.01
        self.update()

    def paintEvent(self, event):
        if not self.hist_data: return
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setBrush(QColor(0,0,0,100)); p.setPen(Qt.PenStyle.NoPen)
        h_widget = self.height()
        p.drawRoundedRect(0,0,200,h_widget,5,5)
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Plus)
        
        cols = [QColor(255,50,50,180), QColor(50,255,50,180), QColor(50,100,255,180)]
        step = 200/256
        log_norm = math.log10(1001)
        
        for i, ch in enumerate(self.hist_data):
            path = QPainterPath(); path.moveTo(0, h_widget)
            for x, val in enumerate(ch):
                h_val = val
                if self.use_log and val > 0: h_val = math.log10(1 + val * 1000) / log_norm
                path.lineTo(x*step, h_widget - h_val*(h_widget-10))
            path.lineTo(200, h_widget); path.closeSubpath()
            p.setBrush(cols[i]); p.drawPath(path)
            
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
        # White Indicator Only
        if self.clip_white:
            c = QColor(255,0,0) if self.pct_white > 0.1 else QColor(255,140,0)
            p.setPen(c); p.setBrush(c); p.drawRect(196,0,4,h_widget)

# =============================================================================
#  MAIN GUI (Wrapper Class Pattern - matches HyperMetric Stretch architecture)
# =============================================================================

class StarComposerInterface(QObject):
    """
    Wrapper class that manages a QMainWindow internally.
    This pattern matches the working HyperMetric Stretch port architecture.
    Inherits from QObject to support eventFilter.
    """
    def __init__(self, ctx, qt_app):
        super().__init__()
        self.ctx = ctx
        self.app = qt_app
        
        # --- HEADER LOG STARTUP ---
        header_msg = (
            "##############################################\n"
            "# VeraLux — StarComposer (SASPro)\n"
            "# High-Fidelity Star Reconstruction Engine\n"
            "# Author: Riccardo Paterniti (2025)\n"
            "##############################################"
        )
        self.ctx.log(header_msg)
        
        # State variables
        self.sm_full = None; self.sl_full = None
        self.sm_proxy = None; self.sl_proxy = None
        self.comp_proxy = None
        self.request_fit = False 
        self.working_dir = os.getcwd()
        
        # Create the main window
        self.window = QMainWindow()
        self.window.closeEvent = self.handle_close_event
        
        # --- Persistent settings (QSettings) ---
        self.settings = QSettings("VeraLux", "StarComposer")
        
        self.window.setWindowTitle(f"VeraLux StarComposer v{VERSION}")
        self.app.setStyle("Fusion")
        self.window.setStyleSheet(DARK_STYLESHEET)
        self.window.resize(1350, 650) 
        self.window.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint)
        
        # Debounce timer for preview updates
        self.debounce = QTimer()
        self.debounce.setSingleShot(True); self.debounce.setInterval(150)
        self.debounce.timeout.connect(self.run_preview_logic)
        
        # Build the UI
        self.init_ui()
        
        # Populate view list at startup
        self.refresh_view_list()
        
        # Show the window
        self.window.show()
        self.center_window()

    def handle_close_event(self, event):
        """Save settings on close."""
        try:
            self.settings.setValue("sensor_profile", self.cmb_prof.currentText())
        except:
            pass
        event.accept()

    def center_window(self):
        """Center the window on screen."""
        screen = self.app.primaryScreen()
        if screen:
            frame_geo = self.window.frameGeometry()
            frame_geo.moveCenter(screen.availableGeometry().center())
            self.window.move(frame_geo.topLeft())

    def init_ui(self):
        main = QWidget(); self.window.setCentralWidget(main)
        layout = QHBoxLayout(main)
        
        # --- LEFT PANEL ---
        left_container = QWidget(); left_container.setFixedWidth(360)
        left = QVBoxLayout(left_container); left.setContentsMargins(0,0,0,0)
        
        # 1. Input
        g1 = QGroupBox("1. Input"); l1 = QVBoxLayout(g1)
        
        # Refresh button for views
        row_refresh = QHBoxLayout()
        b_refresh = QPushButton("↻ Refresh Views")
        b_refresh.setToolTip("Refresh the list of open views from SASPro.")
        b_refresh.clicked.connect(self.refresh_view_list)
        row_refresh.addStretch()
        row_refresh.addWidget(b_refresh)
        l1.addLayout(row_refresh)
        
        # --- STARMASK INPUT ---
        l1.addWidget(QLabel("Starmask (Linear):"))
        self.cmb_sm_view = QComboBox()
        self.cmb_sm_view.setToolTip("Select an open view to use as the <b>Linear</b> starmask.")
        self.cmb_sm_view.addItem("[Select from open views...]")
        l1.addWidget(self.cmb_sm_view)
        
        row_sm = QHBoxLayout()
        b_sm_view = QPushButton("Use Selected View")
        b_sm_view.setToolTip("Load the selected view as the <b>Linear</b> starmask.")
        b_sm_view.clicked.connect(self.load_starmask_from_view)
        row_sm.addWidget(b_sm_view)
        
        b_sm_file = QPushButton("Open File...")
        b_sm_file.setToolTip("Browse for a <b>Linear</b> starmask file (FITS/TIFF/etc).")
        b_sm_file.clicked.connect(self.load_starmask_from_file)
        row_sm.addWidget(b_sm_file)
        l1.addLayout(row_sm)
        
        self.lbl_sm = QLabel("Starmask: [Empty]")
        self.lbl_sm.setStyleSheet("color: #888888; font-style: italic;")
        l1.addWidget(self.lbl_sm)
        
        # --- STARLESS INPUT ---
        l1.addWidget(QLabel("Starless (Stretched):"))
        self.cmb_sl_view = QComboBox()
        self.cmb_sl_view.setToolTip("Select an open view to use as the <b>Non-Linear</b> starless base.")
        self.cmb_sl_view.addItem("[Select from open views...]")
        l1.addWidget(self.cmb_sl_view)
        
        row_sl = QHBoxLayout()
        b_sl_view = QPushButton("Use Selected View")
        b_sl_view.setToolTip("Load the selected view as the <b>Non-Linear</b> starless base.")
        b_sl_view.clicked.connect(self.load_starless_from_view)
        row_sl.addWidget(b_sl_view)
        
        b_sl_file = QPushButton("Open File...")
        b_sl_file.setToolTip("Browse for a <b>Non-Linear</b> starless file (FITS/TIFF/etc).")
        b_sl_file.clicked.connect(self.load_starless_from_file)
        row_sl.addWidget(b_sl_file)
        l1.addLayout(row_sl)
        
        self.lbl_sl = QLabel("Starless: [Empty]")
        self.lbl_sl.setStyleSheet("color: #888888; font-style: italic;")
        l1.addWidget(self.lbl_sl)
        
        # Blend Mode (Exclusive Checkboxes)
        l1.addWidget(QLabel("Composition Mode:"))
        row_blend = QHBoxLayout()
        
        self.chk_add = QCheckBox("Linear Add (Physical)")
        self.chk_screen = QCheckBox("Screen (Safe)")
        self.chk_add.setChecked(True)
        
        self.chk_add.setToolTip("<b>Linear Add:</b> Physical light addition. High contrast, but risk of core clipping if background is bright.")
        self.chk_screen.setToolTip("<b>Screen:</b> Soft blend. Preserves galaxy cores and prevents explosion, but lowers contrast.")
        
        # Logic Group for exclusivity
        self.grp_blend = QButtonGroup()
        self.grp_blend.addButton(self.chk_add, 0)
        self.grp_blend.addButton(self.chk_screen, 1)
        self.grp_blend.setExclusive(True)
        self.grp_blend.buttonToggled.connect(self.trigger_update)
        
        row_blend.addWidget(self.chk_add)
        row_blend.addWidget(self.chk_screen)
        l1.addLayout(row_blend)
        
        left.addWidget(g1)
        
        # 2. Sensor
        g2 = QGroupBox("2. Sensor Profile"); l2 = QVBoxLayout(g2)
        self.cmb_prof = QComboBox()
        self.cmb_prof.setToolTip("<b>Sensor Profile:</b><br>Defines the specific Quantum Efficiency weights used to calculate star luminance.<br>Use <b>Rec.709</b> for general purpose.")
        for k in SENSOR_PROFILES:
            self.cmb_prof.addItem(k)

        # Restore last used profile
        try:
            saved_profile = self.settings.value("sensor_profile", DEFAULT_PROFILE, type=str)
        except Exception:
            saved_profile = DEFAULT_PROFILE
        if saved_profile in SENSOR_PROFILES:
            self.cmb_prof.setCurrentText(saved_profile)
        else:
            self.cmb_prof.setCurrentText(DEFAULT_PROFILE)

        # Persist selection on change
        self.cmb_prof.currentIndexChanged.connect(self.on_sensor_profile_changed)
        l2.addWidget(self.cmb_prof)
        left.addWidget(g2)
        
        # 3. Stretch
        g3 = QGroupBox("3. VeraLux Stretch"); l3 = QVBoxLayout(g3)
        self.lbl_D = QLabel("Star Intensity (Log D): 0.00")
        l3.addWidget(self.lbl_D)
        self.s_D = ResetSlider(Qt.Orientation.Horizontal, 0); self.s_D.setObjectName("MainSlider")
        self.s_D.setToolTip("<b>Star Intensity (Log D):</b><br>Master gain control. Increases the brightness of the star field.<br>Start here to make stars visible.")
        self.s_D.setRange(0, 600); self.s_D.setValue(0)
        self.s_D.valueChanged.connect(self.update_labels)
        self.s_D.valueChanged.connect(self.trigger_update); l3.addWidget(self.s_D)
        
        self.lbl_b = QLabel("Profile Hardness (b): 6.0")
        l3.addWidget(self.lbl_b)
        self.s_b = ResetSlider(Qt.Orientation.Horizontal, 60); self.s_b.setRange(10, 200); self.s_b.setValue(60)
        self.s_b.setToolTip("<b>Profile Hardness (b):</b><br>Controls the star profile geometry.<br>• <b>High:</b> Pinpoint, sharp stars (prevents bloating).<br>• <b>Low:</b> Soft stars with larger halos.")
        self.s_b.valueChanged.connect(self.update_labels)
        self.s_b.valueChanged.connect(self.trigger_update); l3.addWidget(self.s_b)
        
        self.chk_adapt = QCheckBox("Adaptive Anchor"); self.chk_adapt.setChecked(True)
        self.chk_adapt.setToolTip("<b>Adaptive Anchor:</b><br>Automatically detects the black point of the starmask to maximize contrast.<br>Keep enabled for best results.")
        self.chk_adapt.toggled.connect(self.trigger_update); l3.addWidget(self.chk_adapt)
        left.addWidget(g3)
        
        # 4. Physics
        g4 = QGroupBox("4. Physics"); l4 = QVBoxLayout(g4)
        self.lbl_grip = QLabel("Color Grip (Vibrance): 100%")
        l4.addWidget(self.lbl_grip)
        self.s_grip = ResetSlider(Qt.Orientation.Horizontal, 100); self.s_grip.setRange(0, 100); self.s_grip.setValue(100)
        self.s_grip.setToolTip("<b>Color Grip (VCP):</b><br>Controls Vector Color Preservation.<br>• <b>100%:</b> Fully preserves original star color ratios (Zero Hue Shift).<br>• <b><100%:</b> Allows core desaturation towards white.")
        self.s_grip.valueChanged.connect(self.update_labels)
        self.s_grip.valueChanged.connect(self.trigger_update); l4.addWidget(self.s_grip)
        
        self.lbl_shad = QLabel("Shadow Conv (Hide Artifacts): 0.0")
        l4.addWidget(self.lbl_shad)
        self.s_shad = ResetSlider(Qt.Orientation.Horizontal, 0); self.s_shad.setRange(0, 300); self.s_shad.setValue(0)
        self.s_shad.setToolTip("<b>Shadow Convergence:</b><br>Damps chromatic noise in the background.<br>Use this to hide artifacts left by star removal tools in the black point.")
        self.s_shad.valueChanged.connect(self.update_labels)
        self.s_shad.valueChanged.connect(self.trigger_update); l4.addWidget(self.s_shad)
        left.addWidget(g4)
        
        # 5. Surgery
        self.chk_surgery = QCheckBox("Show Star Surgery (Advanced)")
        self.chk_surgery.setToolTip("Reveal advanced tools for morphological reduction and optical correction.")
        self.chk_surgery.toggled.connect(self.toggle_surgery)
        left.addWidget(self.chk_surgery)
        
        self.g5 = QGroupBox("5. Star Surgery"); l5 = QVBoxLayout(self.g5)
        
        # LSR
        self.lbl_lsr = QLabel("Core Rejection (LSR): 0%")
        l5.addWidget(self.lbl_lsr)
        self.s_lsr = ResetSlider(Qt.Orientation.Horizontal, 0); self.s_lsr.setRange(0, 100); self.s_lsr.setValue(0)
        self.s_lsr.setToolTip("<b>Large Structure Rejection (LSR):</b><br>Uses Dynamic High-Pass filtering to remove large non-stellar blobs (galaxy cores).")
        self.s_lsr.valueChanged.connect(self.update_labels)
        self.s_lsr.valueChanged.connect(self.trigger_update); l5.addWidget(self.s_lsr)

        self.lbl_red = QLabel("Reduction (Erosion): 0%")
        l5.addWidget(self.lbl_red)
        self.s_red = ResetSlider(Qt.Orientation.Horizontal, 0); self.s_red.setRange(0, 100); self.s_red.setValue(0)
        self.s_red.setToolTip("<b>Morphological Reduction:</b><br>Applies circular erosion <i>after</i> stretching to physically shrink star diameters.<br><i>Use with caution.</i>")
        self.s_red.valueChanged.connect(self.update_labels)
        self.s_red.valueChanged.connect(self.trigger_update); l5.addWidget(self.s_red)
        
        self.lbl_heal = QLabel("Optical Healing (Halos): 0.0")
        l5.addWidget(self.lbl_heal)
        self.s_heal = ResetSlider(Qt.Orientation.Horizontal, 0); self.s_heal.setRange(0, 20); self.s_heal.setValue(0)
        self.s_heal.setToolTip("<b>Optical Healing:</b><br>Repairs chromatic aberration / color fringing (magenta/green halos) by blurring chrominance channels into the luminance structure.<br><br><b>Note:</b> Works best close to star cores. Not intended to remove large-scale halos or starless residual artifacts.")
        self.s_heal.valueChanged.connect(self.update_labels)
        self.s_heal.valueChanged.connect(self.trigger_update); l5.addWidget(self.s_heal)
        self.g5.hide() 
        left.addWidget(self.g5)
        
        # Buttons
        footer = QHBoxLayout()
        
        self.btn_help = QPushButton("?"); self.btn_help.setObjectName("HelpButton"); self.btn_help.setFixedWidth(20)
        self.btn_help.setToolTip("Print Operational Guide to Console")
        self.btn_help.clicked.connect(self.print_help_to_console)
        footer.addWidget(self.btn_help)
        
        b_res = QPushButton("Defaults"); b_res.clicked.connect(self.set_defaults)
        b_res.setToolTip("Reset all parameters to optimal starting values.")
        footer.addWidget(b_res)
        
        b_cls = QPushButton("Close"); b_cls.setObjectName("CloseButton")
        b_cls.setToolTip("Close the application.")
        b_cls.clicked.connect(self.window.close)
        footer.addWidget(b_cls)
        
        b_proc = QPushButton("PROCESS"); b_proc.setObjectName("ProcessButton")
        b_proc.setToolTip("Compute full-resolution image and apply to active document.")
        b_proc.clicked.connect(self.process_full_resolution)
        footer.addWidget(b_proc)
        
        left.addLayout(footer)
        left.addStretch()
        
        layout.addWidget(left_container)
        
        # --- RIGHT PANEL (Preview) ---
        right = QVBoxLayout()
        
        # Toolbar
        tb = QHBoxLayout()
        b_out = QPushButton("-"); b_out.setObjectName("ZoomBtn"); b_out.clicked.connect(self.zoom_out)
        b_fit = QPushButton("Fit"); b_fit.setObjectName("ZoomBtn"); b_fit.clicked.connect(self.fit_view)
        b_11 = QPushButton("1:1"); b_11.setObjectName("ZoomBtn"); b_11.clicked.connect(self.zoom_1to1)
        b_in = QPushButton("+"); b_in.setObjectName("ZoomBtn"); b_in.clicked.connect(self.zoom_in)
        
        lbl_hint = QLabel("Double-click to fit")
        lbl_hint.setStyleSheet("color: #777777; font-size: 8pt; font-style: italic; margin-left: 10px;")
        
        self.chk_ontop = QCheckBox("On Top")
        self.chk_ontop.setToolTip("Keep window above other windows.")
        self.chk_ontop.setChecked(True)
        self.chk_ontop.toggled.connect(self.toggle_ontop)
        self.chk_ontop.setStyleSheet("color: #cccccc; font-weight: bold; margin-left: 10px;")
        
        self.b_hist = QPushButton("Hist"); self.b_hist.setObjectName("ZoomBtn"); self.b_hist.setCheckable(True)
        self.b_hist.setToolTip("Toggle Histogram overlay.")
        self.b_hist.setChecked(True); self.b_hist.clicked.connect(self.toggle_hist)
        
        tb.addWidget(b_out); tb.addWidget(b_fit); tb.addWidget(b_11); tb.addWidget(b_in)
        tb.addWidget(lbl_hint)
        tb.addStretch()
        tb.addWidget(self.chk_ontop)
        tb.addWidget(self.b_hist)
        right.addLayout(tb)
        
        # View
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.view.setStyleSheet("background-color: #151515; border: none;")
        self.view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.view.viewport().installEventFilter(self)
        self.view.installEventFilter(self)
        right.addWidget(self.view)
        
        self.pix_item = QGraphicsPixmapItem(); self.scene.addItem(self.pix_item)
        
        # Floating Overlays
        self.chk_log = QCheckBox("Log Scale", self.view)
        self.chk_log.setToolTip("Toggle Logarithmic scale.")
        self.chk_log.setStyleSheet("color:#aaa; background:transparent; font-weight:bold;")
        self.chk_log.toggled.connect(self.toggle_log)
        self.chk_log.hide()
        
        self.hist = HistogramOverlay(self.view)
        self.hist.hide()
        
        layout.addLayout(right)
        
        self.update_overlays()
        self.update_labels()

    # --- HELP ---
    def print_help_to_console(self):
        guide_lines = [
            "==========================================================================",
            "   VERALUX STARCOMPOSER v1.0 - OPERATIONAL GUIDE",
            "   High-Fidelity Star Reconstruction & Compositing Engine",
            "==========================================================================",
            "",
            "OVERVIEW",
            "-----------------",
            "StarComposer is a photometric workstation that decouples star stretching",
            "from the main image. It applies the VeraLux HyperMetric engine to linear star masks",
            "to create pinpoint, colorful stars, and composites them onto a starless base.",
            "",
            "[1] INPUT REQUIREMENTS",
            "    • Starmask: Must be LINEAR (unstretched). This is critical for color fidelity.",
            "    • Starless: Must be NON-LINEAR (stretched/processed). This serves as the base.",
            "",
            "[2] THE MAIN WORKFLOW",
            "    • Composition Mode: Choose 'Linear Add' (default) for physical accuracy or",
            "      'Screen' to prevent core explosion if your starmask has bright galaxy residuals.",
            "    • Sensor Profile: Select the correct profile to weight star luminosity correctly.",
            "    • Star Intensity (Gold Slider): This is your master gain. Start here.",
            "      Increase until you see the desired amount of faint stars.",
            "    • Profile Hardness (b): Controls star geometry.",
            "      - Higher values = Sharp, pinpoint stars (prevents bloating).",
            "      - Lower values = Soft, large stars.",
            "",
            "[3] PHYSICS & COLOR",
            "    • Color Grip: Keeps star cores colorful. Default 100% recommended.",
            "    • Shadow Convergence: The 'Cleaner'.",
            "      Star removal tools often leave messy artifacts in the background.",
            "      Increase Shadow Conv to hide these artifacts into the black point.",
            "",
            "[4] STAR SURGERY (Advanced)",
            "    Enable this section only if necessary.",
            "    • Core Rejection (LSR): Dynamically removes large blobs (e.g. galaxy cores) from",
            "      the star mask, keeping only stars.",
            "    • Reduction: Applies morphological erosion *after* stretching.",
            "    • Optical Healing: Blurs chrominance to fix magenta/green halos.",
            "      Note: Targets near-star chromatic fringing. Not a large-scale halo remover.",
            "",
            "[5] INTERPRETING THE HISTOGRAM",
            "    • Left Side (Blacks): Ignore clipping here. Starmasks are mostly black.",
            "    • Right Side (Whites): Red bar means star cores are saturating.",
            "      If red appears, increase 'Profile Hardness (b)' or reduce 'Intensity'.",
            "",
            "Support & Info: info@veralux.space",
            "=========================================================================="
        ]
        for line in guide_lines:
            msg = line if line.strip() else " "
            self.ctx.log(msg)

    # --- INPUT HANDLING ---
    def refresh_view_list(self):
        """Refresh the list of open views from SASPro."""
        try:
            views = self.ctx.list_views()
            
            # Update starmask combo
            current_sm = self.cmb_sm_view.currentText()
            self.cmb_sm_view.clear()
            self.cmb_sm_view.addItem("[Select from open views...]")
            
            # Update starless combo
            current_sl = self.cmb_sl_view.currentText()
            self.cmb_sl_view.clear()
            self.cmb_sl_view.addItem("[Select from open views...]")
            
            for v in views:
                name = v.get('name', v.get('title', 'Unknown'))
                uid = v.get('uid', '')
                display = f"{name}" if name else f"View {uid}"
                self.cmb_sm_view.addItem(display, uid)
                self.cmb_sl_view.addItem(display, uid)
            
            # Try to restore previous selection
            idx_sm = self.cmb_sm_view.findText(current_sm)
            if idx_sm >= 0:
                self.cmb_sm_view.setCurrentIndex(idx_sm)
            idx_sl = self.cmb_sl_view.findText(current_sl)
            if idx_sl >= 0:
                self.cmb_sl_view.setCurrentIndex(idx_sl)
                
            view_count = len(views)
            self.ctx.log(f"StarComposer: Found {view_count} open view(s).")
        except Exception as e:
            self.ctx.log(f"Error refreshing views: {str(e)}")
    
    def _convert_to_chw(self, img):
        """Convert image array to (C, H, W) format for internal processing."""
        if img.ndim == 2:
            # Mono image
            return np.array([img, img, img], dtype=np.float32)
        elif img.ndim == 3:
            if img.shape[2] in [1, 3, 4]:
                # (H, W, C) format - transpose to (C, H, W)
                if img.shape[2] == 1:
                    return np.array([img[:,:,0], img[:,:,0], img[:,:,0]], dtype=np.float32)
                elif img.shape[2] == 4:
                    return img[:,:,:3].transpose(2, 0, 1).astype(np.float32)
                else:
                    return img.transpose(2, 0, 1).astype(np.float32)
            elif img.shape[0] in [1, 3, 4]:
                # Already (C, H, W) format
                if img.shape[0] == 1:
                    return np.array([img[0], img[0], img[0]], dtype=np.float32)
                elif img.shape[0] == 4:
                    return img[:3].astype(np.float32)
                else:
                    return img.astype(np.float32)
        return img.astype(np.float32)
    
    def load_starmask_from_view(self):
        """Load starmask from selected open view."""
        idx = self.cmb_sm_view.currentIndex()
        if idx <= 0:
            QMessageBox.warning(self.window, "No View Selected", 
                "Please select a view from the dropdown or click 'Refresh Views' first.")
            return
        
        uid = self.cmb_sm_view.currentData()
        name = self.cmb_sm_view.currentText()
        
        try:
            img = self.ctx.get_image_for(uid)
            if img is None:
                QMessageBox.warning(self.window, "Load Error", f"Could not get image data from view: {name}")
                return
            
            self.sm_full = self._convert_to_chw(img)
            self.sm_proxy = self.make_proxy(self.sm_full)
            self.lbl_sm.setText(f"Mask: {name} (view)")
            self.lbl_sm.setStyleSheet("color: #88ff88; font-style: normal;")
            self.request_fit = True
            self.trigger_update()
            self.ctx.log(f"StarComposer: Loaded starmask from view '{name}'")
        except Exception as e:
            self.ctx.log(f"Error loading starmask from view: {str(e)}")
            QMessageBox.critical(self.window, "Load Error", f"Failed to load from view:\n{str(e)}")
    
    def load_starless_from_view(self):
        """Load starless from selected open view."""
        idx = self.cmb_sl_view.currentIndex()
        if idx <= 0:
            QMessageBox.warning(self.window, "No View Selected", 
                "Please select a view from the dropdown or click 'Refresh Views' first.")
            return
        
        uid = self.cmb_sl_view.currentData()
        name = self.cmb_sl_view.currentText()
        
        try:
            img = self.ctx.get_image_for(uid)
            if img is None:
                QMessageBox.warning(self.window, "Load Error", f"Could not get image data from view: {name}")
                return
            
            self.sl_full = self._convert_to_chw(img)
            self.sl_proxy = self.make_proxy(self.sl_full)
            self.lbl_sl.setText(f"Base: {name} (view)")
            self.lbl_sl.setStyleSheet("color: #88ff88; font-style: normal;")
            self.request_fit = True
            self.trigger_update()
            self.ctx.log(f"StarComposer: Loaded starless from view '{name}'")
        except Exception as e:
            self.ctx.log(f"Error loading starless from view: {str(e)}")
            QMessageBox.critical(self.window, "Load Error", f"Failed to load from view:\n{str(e)}")
    
    def load_image_file(self):
        """Load an image file using SASPro's ctx.load_image"""
        f, _ = QFileDialog.getOpenFileName(
            self.window, "Load Image", self.working_dir, 
            "Image Files (*.fit *.fits *.tif *.tiff *.png *.jpg *.jpeg *.xisf);;All Files (*)"
        )
        if not f: return None, None
        
        self.working_dir = os.path.dirname(f)
        
        try:
            # Use SASPro's load_image function
            img, hdr, bit, mono = self.ctx.load_image(f)
            img = self._convert_to_chw(img)
            return img, os.path.basename(f)
        except Exception as e:
            self.ctx.log(f"Error loading image: {str(e)}")
            QMessageBox.critical(self.window, "Load Error", f"Failed to load image:\n{str(e)}")
            return None, None

    def make_proxy(self, img):
        if img is None: return None
        h, w = img.shape[1], img.shape[2]
        scale = 1600 / max(h, w)
        if scale >= 1.0: return img
        step = int(1/scale)
        return img[:, ::step, ::step].copy()

    def load_starmask_from_file(self):
        """Load starmask from file dialog."""
        d, n = self.load_image_file()
        if d is not None:
            self.sm_full = d
            self.sm_proxy = self.make_proxy(d)
            self.lbl_sm.setText(f"Mask: {n} (file)")
            self.lbl_sm.setStyleSheet("color: #ffcc88; font-style: normal;")
            self.request_fit = True
            self.trigger_update()
            self.ctx.log(f"StarComposer: Loaded starmask from file '{n}'")

    def load_starless_from_file(self):
        """Load starless from file dialog."""
        d, n = self.load_image_file()
        if d is not None:
            self.sl_full = d
            self.sl_proxy = self.make_proxy(d)
            self.lbl_sl.setText(f"Base: {n} (file)")
            self.lbl_sl.setStyleSheet("color: #ffcc88; font-style: normal;")
            self.request_fit = True
            self.trigger_update()
            self.ctx.log(f"StarComposer: Loaded starless from file '{n}'")

    def set_defaults(self):
        self.s_D.setValue(0); self.s_b.setValue(60); self.s_grip.setValue(100)
        self.s_shad.setValue(0); self.s_lsr.setValue(0); self.s_red.setValue(0); self.s_heal.setValue(0)
        self.chk_adapt.setChecked(True)
        self.chk_add.setChecked(True)
        self.update_labels()
        self.trigger_update()

    def toggle_surgery(self, checked):
        self.g5.setVisible(checked)
        QApplication.processEvents()
        if not checked:
            QTimer.singleShot(10, lambda: self.window.resize(self.window.width(), 0))

    def on_sensor_profile_changed(self, _index=None):
        try: self.settings.setValue("sensor_profile", self.cmb_prof.currentText())
        except: pass
        self.trigger_update()

    def update_labels(self):
        val_D = self.s_D.value() / 100.0 * 2 + 1.0
        val_b = self.s_b.value() / 10.0
        val_grip = int(self.s_grip.value())
        val_shad = self.s_shad.value() / 100.0
        val_lsr = int(self.s_lsr.value())
        val_red = int(self.s_red.value())
        val_heal = self.s_heal.value()
        self.lbl_D.setText(f"Star Intensity (Log D): <b style='color:#ffb000'>{val_D:.2f}</b>")
        self.lbl_b.setText(f"Profile Hardness (b): <b>{val_b:.1f}</b>")
        self.lbl_grip.setText(f"Color Grip (Vibrance): <b>{val_grip}%</b>")
        self.lbl_shad.setText(f"Shadow Conv (Hide Artifacts): <b>{val_shad:.2f}</b>")
        self.lbl_lsr.setText(f"Core Rejection (LSR): <b>{val_lsr}%</b>")
        self.lbl_red.setText(f"Reduction (Erosion): <b>{val_red}%</b>")
        self.lbl_heal.setText(f"Optical Healing (Halos): <b>{val_heal:.1f}</b>")

    def trigger_update(self):
        if self.sm_proxy is None: return
        self.debounce.start()

    def run_preview_logic(self):
        D = self.s_D.value() / 100.0 * 2 + 1.0
        b = self.s_b.value() / 10.0
        conv = 3.5
        grip = self.s_grip.value() / 100.0
        shadow = self.s_shad.value() / 100.0
        lsr = self.s_lsr.value() / 100.0
        red = self.s_red.value() / 100.0
        heal = self.s_heal.value()
        adapt = self.chk_adapt.isChecked()
        w = SENSOR_PROFILES[self.cmb_prof.currentText()]
        
        stars = process_star_pipeline(self.sm_proxy, D, b, conv, grip, shadow, red, heal, lsr, w, adapt)
        
        # Composition Logic
        use_screen = self.chk_screen.isChecked()
        if self.sl_proxy is not None:
            if self.sl_proxy.shape != stars.shape:
                min_h = min(self.sl_proxy.shape[1], stars.shape[1])
                min_w = min(self.sl_proxy.shape[2], stars.shape[2])
                sl = self.sl_proxy[:, :min_h, :min_w]
                st = stars[:, :min_h, :min_w]
                if use_screen: comp = 1.0 - (1.0 - sl) * (1.0 - st)
                else: comp = np.clip(sl + st, 0.0, 1.0)
            else:
                if use_screen: comp = 1.0 - (1.0 - self.sl_proxy) * (1.0 - stars)
                else: comp = np.clip(self.sl_proxy + stars, 0.0, 1.0)
        else:
            comp = stars
            
        self.comp_proxy = comp
        self.update_view()

    def update_view(self):
        if self.comp_proxy is None: return
        
        self.hist.set_data(self.comp_proxy)
        if self.b_hist.isChecked(): 
            self.hist.show(); self.chk_log.show()
        
        disp = np.clip(self.comp_proxy * 255, 0, 255).astype(np.uint8)
        disp = np.ascontiguousarray(np.flipud(disp.transpose(1, 2, 0)))
        h, w, c = disp.shape
        qimg = QImage(disp.data.tobytes(), w, h, c*w, QImage.Format.Format_RGB888)
        self.pix_item.setPixmap(QPixmap.fromImage(qimg))
        self.scene.setSceneRect(0, 0, w, h)
        
        if self.request_fit:
            self.fit_view()
            self.request_fit = False

    def process_full_resolution(self):
        if self.sm_full is None: 
            QMessageBox.warning(self.window, "Missing Input", "Please load a starmask first.")
            return
        self.window.setEnabled(False)
        
        D = self.s_D.value() / 100.0 * 2 + 1.0
        b = self.s_b.value() / 10.0
        conv = 3.5
        grip = self.s_grip.value() / 100.0
        shadow = self.s_shad.value() / 100.0
        lsr = self.s_lsr.value() / 100.0
        red = self.s_red.value() / 100.0
        heal = self.s_heal.value()
        adapt = self.chk_adapt.isChecked()
        w = SENSOR_PROFILES[self.cmb_prof.currentText()]
        use_screen = self.chk_screen.isChecked()

        try:
            stars = process_star_pipeline(self.sm_full, D, b, conv, grip, shadow, red, heal, lsr, w, adapt)
            
            if self.sl_full is not None:
                if self.sl_full.shape != stars.shape:
                    min_h = min(self.sl_full.shape[1], stars.shape[1])
                    min_w = min(self.sl_full.shape[2], stars.shape[2])
                    sl = self.sl_full[:, :min_h, :min_w]
                    st = stars[:, :min_h, :min_w]
                    if use_screen: final = 1.0 - (1.0 - sl) * (1.0 - st)
                    else: final = np.clip(sl + st, 0.0, 1.0)
                else:
                    if use_screen: final = 1.0 - (1.0 - self.sl_full) * (1.0 - stars)
                    else: final = np.clip(self.sl_full + stars, 0.0, 1.0)
            else:
                final = stars
            
            # Convert from (C, H, W) to (H, W, C) for SASPro
            output = final.astype(np.float32).transpose(1, 2, 0)
            
            # Open as new document in SASPro
            self.ctx.open_new_document(output, name="VeraLux_StarComposer_Result")
            self.ctx.log(f"VeraLux StarComposer v{VERSION}: Result created as new document.")
            
            self.window.close()
        except Exception as e:
            QMessageBox.critical(self.window, "Error", str(e))
            traceback.print_exc()
        finally:
            self.window.setEnabled(True)

    # --- VIEWPORT EVENTS ---
    def toggle_ontop(self, checked):
        pos = self.window.pos()
        if checked:
            self.window.setWindowFlags(self.window.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
        else:
            self.window.setWindowFlags(self.window.windowFlags() & ~Qt.WindowType.WindowStaysOnTopHint)
        self.window.show()
        self.window.move(pos)

    def toggle_hist(self):
        v = self.b_hist.isChecked()
        if self.comp_proxy is not None:
            self.hist.setVisible(v); self.chk_log.setVisible(v)
        else:
            self.hist.hide(); self.chk_log.hide()

    def toggle_log(self, v):
        self.hist.set_log_scale(v)

    def update_overlays(self):
        w, h = self.view.width(), self.view.height()
        hx, hy = 10, h - 110
        self.hist.move(hx, hy)
        self.chk_log.move(hx, hy - 20)

    def zoom_in(self): self.view.scale(1.2, 1.2)
    def zoom_out(self): self.view.scale(1/1.2, 1/1.2)
    def zoom_1to1(self): self.view.resetTransform()
    def fit_view(self): self.view.fitInView(self.pix_item, Qt.AspectRatioMode.KeepAspectRatio)
    
    def eventFilter(self, source, event):
        if not hasattr(self, 'view') or self.view is None:
            return False
        if source == self.view and event.type() == QEvent.Type.Resize:
            self.update_overlays()
        
        if source == self.view.viewport():
            if event.type() == QEvent.Type.Wheel:
                if event.angleDelta().y() > 0: self.zoom_in()
                else: self.zoom_out()
                return True
            elif event.type() == QEvent.Type.MouseButtonDblClick:
                self.fit_view()
                return True
        return False

# =============================================================================
#  SASPRO SCRIPT ENTRYPOINT
# =============================================================================

def run(ctx):
    """
    SASPro script entrypoint.
    Launches the VeraLux StarComposer GUI.
    """
    try:
        # Get or create QApplication instance
        app = QApplication.instance()
        if not app:
            app = QApplication(sys.argv)
        
        # Create the GUI (constructor shows the window)
        gui = StarComposerInterface(ctx, app)
        
        # Run the event loop
        app.exec()
        
    except Exception as e:
        ctx.log(f"VeraLux StarComposer Error: {str(e)}")
        traceback.print_exc()


if __name__ == "__main__":
    print("VeraLux StarComposer v" + VERSION)
    print("This script must be run from within SASPro.")
    print("Please open SASPro and run this script from the Scripts menu.")
