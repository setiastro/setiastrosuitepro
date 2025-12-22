##############################################
# VeraLux — HyperMetric Stretch (SASPro Port)
# Photometric Hyperbolic Stretch Engine
# Author: Riccardo Paterniti (2025)
# Contact: info@veralux.space
##############################################

# (c) 2025 Riccardo Paterniti
# VeraLux — HyperMetric Stretch
# SPDX-License-Identifier: GPL-3.0-or-later
# Version 1.3.1 (SASPro Port)
#
# Credits / Origin
# ----------------
#   • Inspired by: The "True Color" methodology of Dr. Roger N. Clark
#   • Math basis: Inverse Hyperbolic Stretch (IHS) & Vector Color Preservation
#   • Sensor Science: Hardware-specific Quantum Efficiency weighting
#

"""
Overview
--------
A precision linear-to-nonlinear stretching engine designed to maximize sensor 
fidelity while managing the transition to the visible domain.

HyperMetric Stretch (HMS) operates on a fundamental axiom: standard histogram 
transformations often destroy the photometric relationships between color channels 
(hue shifts) and clip high-dynamic range data. HMS solves this by decoupling 
Luminance geometry from Chromatic vectors.

New Features in v1.3
----------------------
• Smart Iterative Solver: The Auto-Calculator now performs a "Floating Sky Check".
  It simulates the post-stretch scaling pipeline to detect black clipping on the 
  full dataset. If clipping is detected, it iteratively adjusts the target to find 
  the optimal Log D that maximizes contrast without data loss.
• Live Histogram: Real-time RGB overlay with smart clipping warnings (>0.1%).
• Unified Color Strategy: Single control for balancing noise vs highlights.
• Adaptive Anchor: Morphological black point detection (Default ON).

Design Goals
------------
• Preserve original vector color ratios during extreme stretching (True Color)
• Optimize Luminance extraction based on specific hardware (Sensor Profiles)
• Provide a mathematically "Safe" expansion for high-dynamic targets
• Bridge the gap between numerical processing and visual feedback (Live Preview)
• Allow controlled hybrid tone-mapping via Color Grip & Shadow Convergence

Core Features
-------------
• Live Preview Engine:
  - Interactive floating window offering real-time feedback on parameter changes.
  - Features Smart Proxy technology for fluid response even on massive files.
  - Includes professional navigation controls (Zoom, Pan, Fit-to-Screen).
• Hybrid Color Engine:
  - Scientific Mode: Full manual control over Color Grip and Shadow Convergence.
  - Ready-to-Use Mode: Orchestrated "Color Strategy" slider for intuitive adjustments.
• Unified Math Core:
  - Implements a "Single Source of Truth" architecture. The Auto-Solver, Live 
    Preview, and Main Processor share the exact same logic.

Usage
-----
1. Pre-requisite: Image MUST be Linear and Color Calibrated (SPCC).
2. Setup: Select Sensor Profile and Processing Mode.
3. Calibrate: 
   - Adaptive Anchor is ON by default (recommended for max dynamic range).
   - Click Calculate Optimal Log D. The solver will find the safe limit.
4. Refine (Live Preview): 
   - Adjust Color Strategy to clean noise (Left) or save highlights (Right).
   - Use the Histogram Overlay to verify clipping (Red bars).
5. Process: Click PROCESS.

Inputs & Outputs
----------------
Input: Linear FITS/TIFF (RGB/Mono). 16/32-bit Int or Float.
Output: Non-linear (Stretched) 32-bit Float.

Compatibility
-------------
• Seti Astro Pro 
• Python 3.10+ 
• Dependencies: PyQt6, numpy

License
-------
Released under GPL-3.0-or-later.
"""

import sys
import os
import traceback

import numpy as np
import math

from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                            QWidget, QLabel, QDoubleSpinBox, QSlider,
                            QPushButton, QGroupBox, QMessageBox, QProgressBar,
                            QComboBox, QRadioButton, QButtonGroup, QCheckBox, QFrame,
                            QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QStackedWidget)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QEvent, QSettings, QPointF
from PyQt6.QtGui import (QImage, QPixmap, QPainter, QColor, QWheelEvent, QMouseEvent, 
                        QPen, QBrush, QPainterPath)

# ---------------------
#  SASPRO SCRIPT METADATA
# ---------------------
SCRIPT_NAME = "VeraLux HyperMetric Stretch"
SCRIPT_GROUP = "Stretch"

VERSION = "1.3.1"

# ------------------------------------------------------------------------------
# VERSION HISTORY
# ------------------------------------------------------------------------------
# 1.3.1: Maintenance Release (SASPro Port).
#        • Sync with Siril v1.3.1 release.
# 1.3.0: Major Science & Engineering Upgrade (SASPro Port).
#        • Ported from Siril to SASPro platform.
#        • Smart Iterative Solver: Auto-calculator now employs a predictive 
#          feedback loop to optimize dynamic range allocation and preserve 
#          deep shadow structure ("Floating Sky" optimization) especially useful
#          in Ready-to-use mode.
#        • Visual Feedback: Added Live RGB Histogram with smart clipping analysis.
#        • Unified Color Strategy: Single intuitive slider for Ready-to-Use mode.
#        • Shadow Convergence: Photometric noise damping for the Scientific engine.
# 1.2.2: UX Upgrade. Added persistent settings (QSettings). VeraLux now remembers
#        Sensor Profile, Processing Mode, and Target Background between sessions.
# 1.2.1: Nomenclature refinement. Replaced generic GHS terms with accurate
#        "Inverse Hyperbolic Stretch" definitions. Minor UI text polish.
# 1.2.0: Major Upgrade. Added Live Preview Engine with Smart Proxy technology.
#        Introduced "Color Grip" Hybrid Stretch for star control.
# 1.1.0: Architecture Upgrade. Introduced VeraLuxCore (Single Source of Truth).
#        Fixed 32-bit/Mono input handling & visual refresh issues (visu reset).
#        Added robust input normalization & improved Solver precision.
# 1.0.3: Added help button (?) that prints Operational Guide to Console.
#        Added contact e-mail. Texts consistency minor fixes.
# 1.0.2: Sensor Database Update (v2.0). Added real QE weights for 15+ sensors.
# 1.0.1: Fix Windows GUI artifacts (invisible checkboxes) and UI polish.
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

/* Windows Fix: Explicitly style indicators to ensure visibility on custom dark backgrounds */
QRadioButton, QCheckBox { color: #cccccc; spacing: 5px; }
QRadioButton::indicator, QCheckBox::indicator { width: 14px; height: 14px; border: 1px solid #666666; background: #3c3c3c; border-radius: 7px; }
QCheckBox::indicator { border-radius: 3px; }
QRadioButton::indicator:checked { background-color: #285299; border: 1px solid #88aaff; image: none; }
QCheckBox::indicator:checked { background-color: #285299; border: 1px solid #88aaff; image: none; }
QRadioButton::indicator:checked { background: qradialgradient(cx:0.5, cy:0.5, radius: 0.4, fx:0.5, fy:0.5, stop:0 #ffffff, stop:1 #285299); }
QCheckBox::indicator:checked { background: #285299; }

QDoubleSpinBox { background-color: #3c3c3c; color: #ffffff; border: 1px solid #555555; padding: 3px; border-radius: 3px; }
QComboBox { background-color: #3c3c3c; color: #ffffff; border: 1px solid #555555; padding: 3px; border-radius: 3px; }
QComboBox:hover { border-color: #777777; }
QComboBox::drop-down { border: none; width: 20px; }
QComboBox::down-arrow { width: 0; height: 0; border-left: 4px solid transparent; border-right: 4px solid transparent; border-top: 6px solid #aaaaaa; margin-right: 6px; }
QComboBox QAbstractItemView { background-color: #3c3c3c; color: #ffffff; selection-background-color: #285299; border: 1px solid #555555; }

/* Robust styling to prevent flickering and transparency */
QSlider { min-height: 22px; }
QSlider::groove:horizontal { 
    background: #444444; 
    height: 6px; 
    border-radius: 3px; 
}
QSlider::handle:horizontal { 
    background-color: #aaaaaa; 
    width: 14px; 
    height: 14px; 
    margin: -4px 0; 
    border-radius: 7px; 
    border: 1px solid #555555; 
}
QSlider::handle:horizontal:hover { 
    background-color: #ffffff; 
    border: 1px solid #888888; 
}
QSlider::handle:horizontal:pressed { 
    background-color: #ffffff; 
    border: 1px solid #dddddd; 
}

QPushButton { background-color: #444444; color: #dddddd; border: 1px solid #666666; border-radius: 4px; padding: 6px; font-weight: bold;}
QPushButton:hover { background-color: #555555; border-color: #777777; }
QPushButton#ProcessButton { background-color: #285299; border: 1px solid #1e3f7a; }
QPushButton#ProcessButton:hover { background-color: #355ea1; }
QPushButton#AutoButton { background-color: #8c6a00; border: 1px solid #a37c00; }
QPushButton#AutoButton:hover { background-color: #bfa100; color: #000000;}
QPushButton#PreviewButton { background-color: #2a5a2a; border: 1px solid #408040; }
QPushButton#PreviewButton:hover { background-color: #3a7a3a; }
QPushButton#CloseButton { background-color: #5a2a2a; border: 1px solid #804040; }
QPushButton#CloseButton:hover { background-color: #7a3a3a; }

/* Preview Toolbar Buttons */
QPushButton#ZoomBtn { min-width: 30px; font-weight: bold; background-color: #3c3c3c; }

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

QProgressBar { border: 1px solid #555555; border-radius: 3px; text-align: center; }
QProgressBar::chunk { background-color: #285299; width: 10px; }
"""

# =============================================================================
#  WORKING SPACE PROFILES (Database v2.1 - SPCC Derived)
# =============================================================================

SENSOR_PROFILES = {
    # --- STANDARD ---
    "Rec.709 (Recommended)": {
        'weights': (0.2126, 0.7152, 0.0722),
        'description': "ITU-R BT.709 standard for sRGB/HDTV",
        'info': "Default choice. Best for general use, DSLR and unknown sensors.",
        'category': 'standard'
    },
    
    # --- SONY MODERN BSI (Consumer) ---
    "Sony IMX571 (ASI2600/QHY268)": {
        'weights': (0.2944, 0.5021, 0.2035),
        'description': "Sony IMX571 26MP APS-C BSI (STARVIS)",
        'info': "Gold standard APS-C. Excellent balance for broadband.",
        'category': 'sensor-specific'
    },
    
    "Sony IMX533 (ASI533)": {
        'weights': (0.2910, 0.5072, 0.2018),
        'description': "Sony IMX533 9MP 1\" Square BSI (STARVIS)",
        'info': "Popular square format. Very low noise.",
        'category': 'sensor-specific'
    },
    
    "Sony IMX455 (ASI6200/QHY600)": {
        'weights': (0.2987, 0.5001, 0.2013),
        'description': "Sony IMX455 61MP Full Frame BSI (STARVIS)",
        'info': "Full frame reference sensor.",
        'category': 'sensor-specific'
    },
    
    "Sony IMX294 (ASI294)": {
        'weights': (0.3068, 0.5008, 0.1925),
        'description': "Sony IMX294 11.7MP 4/3\" BSI",
        'info': "High sensitivity 4/3 format.",
        'category': 'sensor-specific'
    },
    
    "Sony IMX183 (ASI183)": {
        'weights': (0.2967, 0.4983, 0.2050),
        'description': "Sony IMX183 20MP 1\" BSI",
        'info': "High resolution 1-inch sensor.",
        'category': 'sensor-specific'
    },
    
    "Sony IMX178 (ASI178)": {
        'weights': (0.2346, 0.5206, 0.2448),
        'description': "Sony IMX178 6.4MP 1/1.8\" BSI",
        'info': "High resolution entry-level sensor.",
        'category': 'sensor-specific'
    },
    
    "Sony IMX224 (ASI224)": {
        'weights': (0.3402, 0.4765, 0.1833),
        'description': "Sony IMX224 1.27MP 1/3\" BSI",
        'info': "Classic planetary sensor. High Red response.",
        'category': 'sensor-specific'
    },
    
    # --- SONY STARVIS 2 (NIR Optimized) ---
    "Sony IMX585 (ASI585) - STARVIS 2": {
        'weights': (0.3431, 0.4822, 0.1747),
        'description': "Sony IMX585 8.3MP 1/1.2\" BSI (STARVIS 2)",
        'info': "NIR optimized. Excellent for H-Alpha/Narrowband.",
        'category': 'sensor-specific'
    },
    
    "Sony IMX662 (ASI662) - STARVIS 2": {
        'weights': (0.3430, 0.4821, 0.1749),
        'description': "Sony IMX662 2.1MP 1/2.8\" BSI (STARVIS 2)",
        'info': "Planetary/Guiding. High Red/NIR sensitivity.",
        'category': 'sensor-specific'
    },
    
    "Sony IMX678/715 - STARVIS 2": {
        'weights': (0.3426, 0.4825, 0.1750),
        'description': "Sony IMX678/715 BSI (STARVIS 2)",
        'info': "High resolution planetary/security sensors.",
        'category': 'sensor-specific'
    },
    
    # --- PANASONIC / OTHERS ---
    "Panasonic MN34230 (ASI1600/QHY163)": {
        'weights': (0.2650, 0.5250, 0.2100),
        'description': "Panasonic MN34230 4/3\" CMOS",
        'info': "Classic Mono/OSC sensor. Optimized weights.",
        'category': 'sensor-specific'
    },
    
    # --- CANON DSLR (Averaged Profiles) ---
    "Canon EOS (Modern - 60D/6D/R)": {
        'weights': (0.2550, 0.5250, 0.2200),
        'description': "Canon CMOS Profile (Modern)",
        'info': "Balanced profile for most Canon EOS cameras (60D, 6D, 5D, R-series).",
        'category': 'sensor-specific'
    },
    
    "Canon EOS (Legacy - 300D/40D)": {
        'weights': (0.2400, 0.5400, 0.2200),
        'description': "Canon CMOS Profile (Legacy)",
        'info': "For older Canon models (Digic 2/3 era).",
        'category': 'sensor-specific'
    },
    
    # --- NIKON DSLR (Averaged Profiles) ---
    "Nikon DSLR (Modern - D5300/D850)": {
        'weights': (0.2600, 0.5100, 0.2300),
        'description': "Nikon CMOS Profile (Modern)",
        'info': "Balanced profile for Nikon Expeed 4+ cameras.",
        'category': 'sensor-specific'
    },
    
    # --- SMART TELESCOPES ---
    "ZWO Seestar S50": {
        'weights': (0.3333, 0.4866, 0.1801),
        'description': "ZWO Seestar S50 (IMX462)",
        'info': "Specific profile for Seestar S50 smart telescope.",
        'category': 'sensor-specific'
    },
    
    "ZWO Seestar S30": {
        'weights': (0.2928, 0.5053, 0.2019),
        'description': "ZWO Seestar S30",
        'info': "Specific profile for Seestar S30 smart telescope.",
        'category': 'sensor-specific'
    },
    
    # --- NARROWBAND ---
    "Narrowband HOO": {
        'weights': (0.5000, 0.2500, 0.2500),
        'description': "Bicolor palette: Hα=Red, OIII=Green+Blue",
        'info': "Balanced weighting for HOO synthetic palette processing.",
        'category': 'narrowband'
    },
    
    "Narrowband SHO": {
        'weights': (0.3333, 0.3400, 0.3267),
        'description': "Hubble palette: SII=Red, Hα=Green, OIII=Blue",
        'info': "Nearly uniform weighting for SHO tricolor narrowband.",
        'category': 'narrowband'
    }
}

DEFAULT_PROFILE = "Rec.709 (Recommended)"

# =============================================================================
#  CUSTOM WIDGETS
# =============================================================================

class ResetSlider(QSlider):
    """
    A specialized QSlider that resets to its default value on double-click.
    Used for the 'Color Strategy' unified control.
    """
    def __init__(self, orientation, default_value=0, parent=None):
        super().__init__(orientation, parent)
        self.default_val = default_value

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self.setValue(self.default_val)
            event.accept()
        else:
            super().mouseDoubleClickEvent(event)

class HistogramOverlay(QWidget):
    """
    A transparent overlay that draws an RGB histogram.
    Designed for the Preview Window. Implements Smart Clipping Warning with % readout.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)
        self.hist_data = None
        self.is_visible = True
        self.use_log = False  
        self.tracker_val = None 
        
        self.clip_black = False
        self.clip_white = False
        self.pct_black = 0.0
        self.pct_white = 0.0
        
        self.black_severity = 0
        self.white_severity = 0
        
        self.source_saturated = False
        self.processing_mode = "ready_to_use"
        
        self.setFixedHeight(120)
        self.setFixedWidth(200)
        
    def set_context(self, source_saturated, mode):
        """Sets external context for diagnostic tooltips."""
        self.source_saturated = source_saturated
        self.processing_mode = mode

    def set_data(self, img_data):
        """
        Calculates histogram from image data (C, H, W) normalized 0-1.
        Includes Smart Clipping detection (Exact Boundary Check).
        """
        if img_data is None: return
        
        bins = 256
        if img_data.ndim == 3:
            r = np.histogram(img_data[0], bins=bins, range=(0, 1))[0]
            g = np.histogram(img_data[1], bins=bins, range=(0, 1))[0]
            b = np.histogram(img_data[2], bins=bins, range=(0, 1))[0]
            
            max_val = max(np.max(r), np.max(g), np.max(b))
            if max_val > 0:
                self.hist_data = (r / max_val, g / max_val, b / max_val)
            else:
                self.hist_data = None
            
            total_pixels = img_data.shape[1] * img_data.shape[2]
            epsilon = 1e-7
            
            mask_black = np.any(img_data <= epsilon, axis=0)
            black_count = np.count_nonzero(mask_black)
            
            mask_white = np.any(img_data >= (1.0 - epsilon), axis=0)
            white_count = np.count_nonzero(mask_white)
            
        else:
            l = np.histogram(img_data, bins=bins, range=(0, 1))[0]
            max_val = np.max(l)
            if max_val > 0:
                self.hist_data = (l / max_val,)
            else:
                self.hist_data = None
                
            total_pixels = img_data.shape[0] * img_data.shape[1]
            epsilon = 1e-7
            
            black_count = np.count_nonzero(img_data <= epsilon)
            white_count = np.count_nonzero(img_data >= (1.0 - epsilon))
        
        self.pct_black = (black_count / total_pixels) * 100.0
        self.pct_white = (white_count / total_pixels) * 100.0
        
        threshold_warn = 0.01
        threshold_critical = 0.1
        
        self.clip_black = self.pct_black > threshold_warn
        self.black_severity = 2 if self.pct_black > threshold_critical else (1 if self.clip_black else 0)
        
        self.clip_white = self.pct_white > threshold_warn
        self.white_severity = 2 if self.pct_white > threshold_critical else (1 if self.clip_white else 0)
        
        tips = ["<b>Histogram RGB</b>"]
        
        if self.clip_black:
            color = "Red" if self.black_severity == 2 else "Orange"
            tips.append(f"• Black Clipping: <font color='{color}'><b>{self.pct_black:.2f}%</b></font>")
            
        if self.clip_white:
            color = "Red" if self.white_severity == 2 else "Orange"
            tips.append(f"• White Clipping: <font color='{color}'><b>{self.pct_white:.2f}%</b></font>")
            
        if not self.clip_black and not self.clip_white:
            tips.append("No clipping detected.")
            
        self.setToolTip("<br>".join(tips))

        self.update()

    def set_tracker(self, val):
        self.tracker_val = val
        self.update()

    def toggle_visibility(self):
        self.is_visible = not self.is_visible
        self.setVisible(self.is_visible)

    def set_log_scale(self, enabled):
        self.use_log = enabled
        self.update()

    def paintEvent(self, event):
        if not self.hist_data or not self.is_visible: return
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        w = self.width()
        h = self.height()
        
        painter.setBrush(QColor(0, 0, 0, 100))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(0, 0, w, h, 5, 5)
        
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Plus)
        
        step_w = w / 256.0
        
        colors = [QColor(255, 50, 50, 180), QColor(50, 255, 50, 180), QColor(50, 100, 255, 180)]
        if len(self.hist_data) == 1: colors = [QColor(200, 200, 200, 180)]
        
        log_scale_factor = 1000.0
        log_norm = math.log10(1 + log_scale_factor)
        
        for i, channel in enumerate(self.hist_data):
            path = QPainterPath()
            path.moveTo(0, h)
            
            for x_idx, val in enumerate(channel):
                x_pos = x_idx * step_w
                
                draw_val = val
                if self.use_log and val > 0:
                    draw_val = math.log10(1 + val * log_scale_factor) / log_norm
                
                y_pos = h - (draw_val * (h - 10))
                path.lineTo(x_pos, y_pos)
            
            path.lineTo(w, h)
            path.closeSubpath()
            
            painter.setBrush(QBrush(colors[i]))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawPath(path)
            
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(colors[i].lighter(120), 1))
            painter.drawPath(path)

        if self.tracker_val is not None:
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
            tx = self.tracker_val * w
            
            tracker_col = QColor(140, 106, 0, 255)
            
            painter.setPen(QPen(tracker_col, 1, Qt.PenStyle.DashLine))
            painter.drawLine(int(tx), 0, int(tx), h)
            
            painter.setPen(tracker_col)
            painter.drawText(int(tx) + 3, h - 5, f"{self.tracker_val:.3f}")

        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
        font = painter.font()
        font.setPointSize(8); font.setBold(True); painter.setFont(font)
        
        if self.clip_black:
            c = QColor(255, 0, 0) if self.black_severity == 2 else QColor(255, 140, 0)
            
            painter.setPen(c)
            painter.setBrush(c)
            painter.drawRect(0, 0, 4, h)
            
            painter.setPen(c.lighter(130))
            painter.drawText(6, 15, f"{self.pct_black:.2f}%")
            
        if self.clip_white:
            c = QColor(255, 0, 0) if self.white_severity == 2 else QColor(255, 140, 0)
            
            painter.setPen(c)
            painter.setBrush(c)
            painter.drawRect(w-4, 0, 4, h)
            
            painter.setPen(c.lighter(130))
            text = f"{self.pct_white:.2f}%"
            fm = painter.fontMetrics()
            t_w = fm.horizontalAdvance(text)
            painter.drawText(w - t_w - 6, 15, text)

# =============================================================================
#  CORE ENGINE (Single Source of Truth) - V1.3 Implementation
# =============================================================================

class VeraLuxCore:
    @staticmethod
    def normalize_input(img_data):
        input_dtype = img_data.dtype
        img_float = img_data.astype(np.float32)
        if np.issubdtype(input_dtype, np.integer):
            if input_dtype == np.uint8: return img_float / 255.0
            elif input_dtype == np.uint16: return img_float / 65535.0
            elif input_dtype == np.uint32: return img_float / 4294967295.0
            else: return img_float / float(np.iinfo(input_dtype).max)
        elif np.issubdtype(input_dtype, np.floating):
            current_max = np.max(img_data)
            if current_max <= 1.0 + 1e-5: return img_float
            if current_max <= 255.0: return img_float / 255.0
            if current_max <= 65535.0: return img_float / 65535.0
            return img_float / 4294967295.0
        return img_float

    @staticmethod
    def calculate_anchor(data_norm):
        if data_norm.ndim == 3 and data_norm.shape[0] == 3:
            floors = []
            stride = max(1, data_norm.size // 500000)
            for c in range(3):
                floors.append(np.percentile(data_norm[c].flatten()[::stride], 0.5))
            anchor = max(0.0, min(floors) - 0.00025)

        elif data_norm.ndim == 2 and data_norm.shape[0] == 3:
            floors = []
            stride = max(1, data_norm.size // 500000)
            for c in range(3):
                floors.append(np.percentile(data_norm[c].flatten()[::stride], 0.5))
            anchor = max(0.0, min(floors) - 0.00025)

        elif data_norm.ndim == 3 and data_norm.shape[0] == 1:
            stride = max(1, data_norm.size // 200000)
            floor = np.percentile(data_norm[0].flatten()[::stride], 0.5)
            anchor = max(0.0, floor - 0.00025)

        else:
            stride = max(1, data_norm.size // 200000)
            floor = np.percentile(data_norm.flatten()[::stride], 0.5)
            anchor = max(0.0, floor - 0.00025)

        return anchor

    @staticmethod
    def calculate_anchor_adaptive(data_norm, weights=None):
        """
        Adaptive (morphological) black point estimation.
        """
        if weights is None:
            weights = (0.2126, 0.7152, 0.0722)

        if data_norm.ndim == 3 and data_norm.shape[0] == 3:
            r_w, g_w, b_w = weights
            L = r_w * data_norm[0] + g_w * data_norm[1] + b_w * data_norm[2]
            base = L
        elif data_norm.ndim == 2 and data_norm.shape[0] == 3:
            r_w, g_w, b_w = weights
            L = r_w * data_norm[0] + g_w * data_norm[1] + b_w * data_norm[2]
            base = L
        elif data_norm.ndim == 3 and data_norm.shape[0] == 1:
            base = data_norm[0]
        else:
            base = data_norm

        stride = max(1, base.size // 2000000)
        sample = base.flatten()[::stride]

        hist, bin_edges = np.histogram(sample, bins=65536, range=(0.0, 1.0))
        hist_smooth = np.convolve(hist, np.ones(50)/50, mode='same')

        search_start = 100
        if search_start >= len(hist_smooth):
            search_start = 0

        peak_idx = np.argmax(hist_smooth[search_start:]) + search_start
        peak_val = hist_smooth[peak_idx]
        target_val = peak_val * 0.06

        left_side = hist_smooth[:peak_idx]
        candidates = np.where(left_side < target_val)[0]

        if len(candidates) > 0:
            anchor_idx = candidates[-1]
            anchor = bin_edges[anchor_idx]
        else:
            anchor = np.percentile(sample, 0.5)

        return max(0.0, anchor)

    @staticmethod
    def extract_luminance(data_norm, anchor, weights):
        r_w, g_w, b_w = weights
        img_anchored = np.maximum(data_norm - anchor, 0.0)

        if data_norm.ndim == 3 and data_norm.shape[0] == 3:
            L_anchored = (r_w * img_anchored[0] + g_w * img_anchored[1] + b_w * img_anchored[2])

        elif data_norm.ndim == 2 and data_norm.shape[0] == 3:
            L_anchored = (r_w * img_anchored[0] + g_w * img_anchored[1] + b_w * img_anchored[2])

        elif data_norm.ndim == 3 and data_norm.shape[0] == 1:
            L_anchored = img_anchored[0]
            img_anchored = img_anchored[0]

        else:
            L_anchored = img_anchored

        return L_anchored, img_anchored

    @staticmethod
    def hyperbolic_stretch(data, D, b, SP=0.0):
        D = max(D, 0.1); b = max(b, 0.1)
        term1 = np.arcsinh(D * (data - SP) + b)
        term2 = np.arcsinh(b)
        norm_factor = np.arcsinh(D * (1.0 - SP) + b) - term2
        if norm_factor == 0: norm_factor = 1e-6
        return (term1 - term2) / norm_factor

    @staticmethod
    def solve_log_d(luma_sample, target_median, b_val):
        median_in = np.median(luma_sample)
        if median_in < 1e-9: return 2.0 
        low_log = 0.0; high_log = 7.0; best_log_D = 2.0
        for _ in range(40):
            mid_log = (low_log + high_log) / 2.0
            mid_D = 10.0 ** mid_log
            test_val = VeraLuxCore.hyperbolic_stretch(median_in, mid_D, b_val)
            if abs(test_val - target_median) < 0.0001: best_log_D = mid_log; break
            if test_val < target_median: low_log = mid_log
            else: high_log = mid_log
        return best_log_D

    @staticmethod
    def apply_mtf(data, m):
        term1 = (m - 1.0) * data
        term2 = (2.0 * m - 1.0) * data - m
        with np.errstate(divide='ignore', invalid='ignore'): res = term1 / term2
        return np.nan_to_num(res, nan=0.0, posinf=1.0, neginf=0.0)

# =============================================================================
#  HELPER FUNCTIONS (Ready-to-Use Logic)
# =============================================================================

def adaptive_output_scaling(img_data, working_space="Rec.709 (Recommended)", 
                            target_bg=0.20, progress_callback=None):
    if progress_callback: progress_callback("Adaptive Scaling: Analyzing Dynamic Range...")
    luma_r, luma_g, luma_b = SENSOR_PROFILES[working_space]['weights']
    is_rgb = (img_data.ndim == 3 and img_data.shape[0] == 3)
    
    if is_rgb:
        R, G, B = img_data[0], img_data[1], img_data[2]
        L_raw = luma_r * R + luma_g * G + luma_b * B
    else:
        L_raw = img_data
    
    median_L = float(np.median(L_raw))
    std_L = float(np.std(L_raw)); min_L = float(np.min(L_raw))
    global_floor = max(min_L, median_L - 2.7 * std_L)
    PEDESTAL = 0.001
    
    if is_rgb:
        stride = max(1, R.size // 500000)
        soft_ceil = max(np.percentile(R.flatten()[::stride], 99.0), np.percentile(G.flatten()[::stride], 99.0), np.percentile(B.flatten()[::stride], 99.0))
        hard_ceil = max(np.percentile(R.flatten()[::stride], 99.99), np.percentile(G.flatten()[::stride], 99.99), np.percentile(B.flatten()[::stride], 99.99))
    else:
        stride = max(1, L_raw.size // 200000)
        soft_ceil = np.percentile(L_raw.flatten()[::stride], 99.0); hard_ceil = np.percentile(L_raw.flatten()[::stride], 99.99)
        
    if soft_ceil <= global_floor: soft_ceil = global_floor + 1e-6
    if hard_ceil <= soft_ceil: hard_ceil = soft_ceil + 1e-6
    
    scale_contrast = (0.98 - PEDESTAL) / (soft_ceil - global_floor + 1e-9)
    scale_safety = (1.0 - PEDESTAL) / (hard_ceil - global_floor + 1e-9)
    final_scale = min(scale_contrast, scale_safety)
    
    def expand_channel(c): return np.clip((c - global_floor) * final_scale + PEDESTAL, 0.0, 1.0)
    
    if is_rgb:
        img_data[0] = expand_channel(R); img_data[1] = expand_channel(G); img_data[2] = expand_channel(B)
        L = luma_r * img_data[0] + luma_g * img_data[1] + luma_b * img_data[2]
    else:
        img_data = expand_channel(L_raw); L = img_data
    
    current_bg = float(np.median(L))
    if current_bg > 0.0 and current_bg < 1.0 and abs(current_bg - target_bg) > 1e-3:
        if progress_callback: progress_callback(f"Applying MTF (Bg: {current_bg:.3f} -> {target_bg})")
        m = (current_bg * (target_bg - 1.0)) / (current_bg * (2.0 * target_bg - 1.0) - target_bg)
        if is_rgb:
            for i in range(3): img_data[i] = VeraLuxCore.apply_mtf(img_data[i], m)
        else:
            img_data = VeraLuxCore.apply_mtf(img_data, m)
    return img_data

def apply_ready_to_use_soft_clip(img_data, threshold=0.98, rolloff=2.0, progress_callback=None):
    if progress_callback: progress_callback(f"Final Polish: Soft-clip > {threshold:.2f}")
    def soft_clip_channel(c, thresh, roll):
        mask = c > thresh
        result = c.copy()
        if np.any(mask):
            t = np.clip((c[mask] - thresh) / (1.0 - thresh + 1e-9), 0.0, 1.0)
            result[mask] = thresh + (1.0 - thresh) * (1.0 - np.power(1.0 - t, roll))
        return np.clip(result, 0.0, 1.0)
    if img_data.ndim == 3:
        for i in range(img_data.shape[0]): img_data[i] = soft_clip_channel(img_data[i], threshold, rolloff)
    else:
        img_data = soft_clip_channel(img_data, threshold, rolloff)
    return img_data

def process_veralux_v6(img_data, log_D, protect_b, convergence_power, 
                       working_space="Rec.709 (Recommended)", 
                       processing_mode="ready_to_use",
                       target_bg=None,
                       color_grip=1.0, 
                       shadow_convergence=0.0, 
                       use_adaptive_anchor=False,
                       progress_callback=None):
    
    if progress_callback: progress_callback("Normalization & Analysis...")
    img = VeraLuxCore.normalize_input(img_data)
    if img.ndim == 3 and img.shape[0] != 3 and img.shape[2] == 3: img = img.transpose(2, 0, 1)

    luma_weights = SENSOR_PROFILES[working_space]['weights']
    is_rgb = (img.ndim == 3)

    if use_adaptive_anchor:
        if progress_callback: progress_callback("Calculating Anchor (Adaptive)...")
        anchor = VeraLuxCore.calculate_anchor_adaptive(img, weights=luma_weights)
    else:
        if progress_callback: progress_callback("Calculating Anchor (Statistical)...")
        anchor = VeraLuxCore.calculate_anchor(img)
    
    if progress_callback: progress_callback(f"Extracting Luminance ({working_space})...")
    L_anchored, img_anchored = VeraLuxCore.extract_luminance(img, anchor, luma_weights)
    
    epsilon = 1e-9; L_safe = L_anchored + epsilon
    if is_rgb:
        r_ratio = img_anchored[0] / L_safe
        g_ratio = img_anchored[1] / L_safe
        b_ratio = img_anchored[2] / L_safe

    if progress_callback: progress_callback(f"Stretching (Log D={log_D:.2f})...")
    L_str = VeraLuxCore.hyperbolic_stretch(L_anchored, 10.0 ** log_D, protect_b)
    L_str = np.clip(L_str, 0.0, 1.0)
    
    if progress_callback: progress_callback("Color Convergence & Hybrid Engine...")
    final = np.zeros_like(img)
    
    if is_rgb:
        k = np.power(L_str, convergence_power)
        r_final = r_ratio * (1.0 - k) + 1.0 * k
        g_final = g_ratio * (1.0 - k) + 1.0 * k
        b_final = b_ratio * (1.0 - k) + 1.0 * k
        
        final[0] = L_str * r_final; final[1] = L_str * g_final; final[2] = L_str * b_final
        
        needs_hybrid = (color_grip < 1.0) or (shadow_convergence > 0.01)
        
        if needs_hybrid:
            if progress_callback: progress_callback("Applying Hybrid Grip & Shadow Convergence...")
            D_val = 10.0 ** log_D
            scalar = np.zeros_like(final)
            scalar[0] = VeraLuxCore.hyperbolic_stretch(img_anchored[0], D_val, protect_b)
            scalar[1] = VeraLuxCore.hyperbolic_stretch(img_anchored[1], D_val, protect_b)
            scalar[2] = VeraLuxCore.hyperbolic_stretch(img_anchored[2], D_val, protect_b)
            scalar = np.clip(scalar, 0.0, 1.0)
            
            grip_map = np.full_like(L_str, color_grip)
            
            if shadow_convergence > 0.01:
                damping = np.power(L_str, shadow_convergence)
                grip_map = grip_map * damping
            
            final = (final * grip_map) + (scalar * (1.0 - grip_map))
    else:
        final = L_str

    final = final * (1.0 - 0.005) + 0.005
    final = np.clip(final, 0.0, 1.0).astype(np.float32)
    
    if processing_mode == "ready_to_use":
        effective_bg = 0.20 if target_bg is None else float(target_bg)
        final = adaptive_output_scaling(final, working_space, effective_bg, progress_callback)
        final = apply_ready_to_use_soft_clip(final, 0.98, 2.0, progress_callback)
    
    return final

# =============================================================================
#  LIVE PREVIEW SYSTEM
# =============================================================================

class VeraLuxPreviewWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__()
        self.setWindowTitle("VeraLux Live Preview")
        self.resize(800, 600)
        self.setWindowFlags(Qt.WindowType.Window | Qt.WindowType.WindowStaysOnTopHint)
        self.setStyleSheet(DARK_STYLESHEET) 

        layout = QVBoxLayout(self); layout.setContentsMargins(0, 0, 0, 0); layout.setSpacing(0)
        
        toolbar = QWidget(); toolbar.setStyleSheet("background-color: #333333; border-bottom: 1px solid #555555;")
        tb_layout = QHBoxLayout(toolbar); tb_layout.setContentsMargins(5, 5, 5, 5); tb_layout.setSpacing(10)
        
        btn_in = QPushButton("+"); btn_in.setObjectName("ZoomBtn"); btn_in.clicked.connect(self.zoom_in)
        btn_out = QPushButton("-"); btn_out.setObjectName("ZoomBtn"); btn_out.clicked.connect(self.zoom_out)
        btn_fit = QPushButton("Fit"); btn_fit.setObjectName("ZoomBtn"); btn_fit.clicked.connect(self.fit_to_view)
        
        self.btn_hist = QPushButton("Hist"); self.btn_hist.setObjectName("ZoomBtn"); self.btn_hist.setCheckable(True)
        self.btn_hist.setChecked(True); self.btn_hist.clicked.connect(self.toggle_histogram)
        
        tb_layout.addWidget(btn_out); tb_layout.addWidget(btn_fit); tb_layout.addWidget(btn_in); tb_layout.addStretch()
        tb_layout.addWidget(self.btn_hist)
        layout.addWidget(toolbar)
        
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        
        self.view.setDragMode(QGraphicsView.DragMode.NoDrag) 
        self.view.setCursor(Qt.CursorShape.CrossCursor) 
        self._last_drag_pos = None 
        
        self.view.setMouseTracking(True) 
        self.view.viewport().installEventFilter(self) 
        self.view.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.view.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.view.setStyleSheet("background-color: #1e1e1e; border: none;")
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        layout.addWidget(self.view)
        
        self.pixmap_item = QGraphicsPixmapItem(); self.scene.addItem(self.pixmap_item)
        self.processed_pixmap = None
        self.last_img_data = None 
        
        self.lbl_info = QLabel("Loading...", self.view)
        self.lbl_info.setStyleSheet("background-color: rgba(0,0,0,150); color: white; padding: 5px; border-radius: 3px;")
        self.lbl_info.move(10, 10)
        
        self.lbl_hint = QLabel("Double-click to fit", self.view)
        self.lbl_hint.setStyleSheet("color: rgba(255,255,255,100); font-size: 8pt; font-weight: bold;")
        self.lbl_hint.setAlignment(Qt.AlignmentFlag.AlignRight)
        
        self.chk_log = QCheckBox("Log Scale", self.view)
        self.chk_log.setToolTip("Toggle Logarithmic Scale")
        self.chk_log.setStyleSheet("""
            QCheckBox { color: #aaaaaa; font-weight: bold; background: transparent; }
            QCheckBox::indicator { border: 1px solid #666666; border-radius: 2px; background: #222222; width: 12px; height: 12px; }
            QCheckBox::indicator:checked { background: #88aaff; border: 1px solid #88aaff; }
        """)
        
        self.histogram = HistogramOverlay(self.view)
        self.chk_log.toggled.connect(self.histogram.set_log_scale)
        self.histogram.move(10, self.view.height() - 130)

    def set_image(self, qimg, img_data_for_hist, source_saturated=False, mode="ready_to_use"):
        self.last_img_data = img_data_for_hist
        pixmap = QPixmap.fromImage(qimg)
        self.processed_pixmap = pixmap
        self.pixmap_item.setPixmap(pixmap)
        self.scene.setSceneRect(self.pixmap_item.boundingRect())
        self.lbl_info.setText("Preview Updated"); self.lbl_info.adjustSize()
        
        self.histogram.set_context(source_saturated, mode)
        self.histogram.set_data(img_data_for_hist)
        
        self.update_overlays_pos()

    def update_overlays_pos(self):
        w, h = self.view.width(), self.view.height()
        self.lbl_hint.move(w - 110, h - 25); self.lbl_hint.raise_()
        
        hist_x = 10
        hist_y = h - 130
        self.histogram.move(hist_x, hist_y); self.histogram.raise_()
        
        self.chk_log.move(hist_x, hist_y - 20)
        self.chk_log.raise_()

    def eventFilter(self, source, event):
        if event.type() == QEvent.Type.MouseButtonPress:
            if event.button() == Qt.MouseButton.LeftButton:
                self._last_drag_pos = event.pos()
                self.view.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
                return True
                
        elif event.type() == QEvent.Type.MouseButtonRelease:
            if event.button() == Qt.MouseButton.LeftButton:
                self.view.viewport().setCursor(Qt.CursorShape.CrossCursor)
                self._last_drag_pos = None
                return True

        elif event.type() == QEvent.Type.MouseMove:
            if event.buttons() & Qt.MouseButton.LeftButton:
                if self._last_drag_pos is not None:
                    delta = event.pos() - self._last_drag_pos
                    self._last_drag_pos = event.pos()
                    
                    hs = self.view.horizontalScrollBar()
                    vs = self.view.verticalScrollBar()
                    hs.setValue(hs.value() - delta.x())
                    vs.setValue(vs.value() - delta.y())
                return True

            self.view.viewport().setCursor(Qt.CursorShape.CrossCursor)
            
            if self.last_img_data is not None:
                pos = self.view.mapToScene(event.pos())
                if self.pixmap_item.contains(pos):
                    ix, iy = int(pos.x()), int(pos.y())
                    
                    radius = 1
                    h, w = self.last_img_data.shape[-2:]
                    
                    iy_flipped = h - 1 - iy
                    
                    x_start, x_end = max(0, ix - radius), min(w, ix + radius + 1)
                    y_start, y_end = max(0, iy_flipped - radius), min(h, iy_flipped + radius + 1)
                    
                    if x_start < x_end and y_start < y_end:
                        if self.last_img_data.ndim == 3:
                            roi = self.last_img_data[:, y_start:y_end, x_start:x_end]
                            luma_roi = 0.2126*roi[0] + 0.7152*roi[1] + 0.0722*roi[2]
                        else:
                            luma_roi = self.last_img_data[y_start:y_end, x_start:x_end]
                        
                        flat = np.sort(luma_roi.flatten())
                        n_pixels = max(1, len(flat) // 2 + 1)
                        val = np.mean(flat[-n_pixels:])
                        
                        self.histogram.set_tracker(val)
                    else:
                        self.histogram.set_tracker(None)
                else:
                    self.histogram.set_tracker(None)
                    
        return super().eventFilter(source, event)

    def resizeEvent(self, event):
        self.update_overlays_pos()
        super().resizeEvent(event)
        
    def toggle_histogram(self):
        self.histogram.toggle_visibility()
        self.chk_log.setVisible(self.histogram.isVisible())

    def fit_to_view(self):
        if self.pixmap_item.pixmap(): self.view.fitInView(self.pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
    def zoom_in(self): self.view.scale(1.2, 1.2)
    def zoom_out(self): self.view.scale(1/1.2, 1/1.2)
    def wheelEvent(self, event: QWheelEvent):
        if event.angleDelta().y() > 0: self.zoom_in()
        else: self.zoom_out()
    def mouseDoubleClickEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton: self.fit_to_view()

# =============================================================================
#  THREADING
# =============================================================================

class AutoSolverThread(QThread):
    result_ready = pyqtSignal(float)
    def __init__(self, data, target, b_val, luma_weights, adaptive, processing_mode):
        super().__init__()
        self.data = data; self.target = target; self.b_val = b_val
        self.luma_weights = luma_weights; self.adaptive = adaptive
        self.processing_mode = processing_mode
        
    def run(self):
        try:
            img_norm = VeraLuxCore.normalize_input(self.data) 
            if img_norm.ndim == 3 and img_norm.shape[0] != 3 and img_norm.shape[2] == 3:
                img_norm = img_norm.transpose(2, 0, 1)
            
            if img_norm.ndim == 3:
                h, w = img_norm.shape[1], img_norm.shape[2]
                num_pixels = h * w
                indices = np.random.choice(num_pixels, min(num_pixels, 100000), replace=False)
                c0 = img_norm[0].flatten()[indices]
                c1 = img_norm[1].flatten()[indices]
                c2 = img_norm[2].flatten()[indices]
                sub_data = np.vstack((c0, c1, c2))
            else:
                h, w = img_norm.shape
                num_pixels = h * w
                indices = np.random.choice(num_pixels, min(num_pixels, 100000), replace=False)
                sub_data = img_norm.flatten()[indices]

            if self.adaptive:
                anchor = VeraLuxCore.calculate_anchor_adaptive(sub_data, weights=self.luma_weights)
            else:
                anchor = VeraLuxCore.calculate_anchor(sub_data)

            L_anchored, _ = VeraLuxCore.extract_luminance(sub_data, anchor, self.luma_weights)
            
            valid = L_anchored[L_anchored > 1e-7]

            if len(valid) == 0:
                self.result_ready.emit(2.0)
                return

            target_temp = self.target
            best_log_d = 2.0
            
            for _ in range(15):
                best_log_d = VeraLuxCore.solve_log_d(valid, target_temp, self.b_val)

                if self.processing_mode != "ready_to_use":
                    break

                D = 10.0 ** best_log_d
                valid_str = VeraLuxCore.hyperbolic_stretch(valid, D, self.b_val)

                med = float(np.median(valid_str))
                std = float(np.std(valid_str))
                min_v = float(np.min(valid_str))
                global_floor = max(min_v, med - (2.7 * std))

                if global_floor <= 0.001:
                    break

                target_temp -= 0.015
                if target_temp < 0.05: break

            self.result_ready.emit(best_log_d)
            
        except Exception as e:
            print(f"Solver Error: {e}")
            self.result_ready.emit(2.0)

class ProcessingThread(QThread):
    finished = pyqtSignal(object); progress = pyqtSignal(str)
    def __init__(self, img, D, b, conv, working_space, processing_mode, target_bg, color_grip, shadow_convergence, adaptive):
        super().__init__()
        self.img = img; self.D = D; self.b = b; self.conv = conv
        self.working_space = working_space; self.processing_mode = processing_mode; self.target_bg = target_bg
        self.color_grip = color_grip; self.shadow_convergence = shadow_convergence; self.adaptive = adaptive
    def run(self):
        try:
            res = process_veralux_v6(self.img, self.D, self.b, self.conv, self.working_space, 
                                   self.processing_mode, self.target_bg, self.color_grip, 
                                   self.shadow_convergence, self.adaptive, self.progress.emit)
            self.finished.emit(res)
        except Exception as e: 
            traceback.print_exc(); self.progress.emit(f"Error: {str(e)}")

# =============================================================================
#  GUI
# =============================================================================

class VeraLuxInterface:
    def __init__(self, ctx, qt_app):
        self.ctx = ctx
        self.app = qt_app
        
        # Log header
        header_msg = (
            "##############################################\n"
            "# VeraLux — HyperMetric Stretch (SASPro)\n"
            "# Photometric Hyperbolic Stretch Engine\n"
            "# Author: Riccardo Paterniti (2025)\n"
            "##############################################"
        )
        self.ctx.log(header_msg)

        self.linear_cache = None
        self.is_source_saturated = False
        self.preview_proxy = None
        self.preview_window = None
        
        self.window = QMainWindow()
        self.window.closeEvent = self.handle_close_event
        
        self.settings = QSettings("VeraLux", "HyperMetricStretch")
        
        self.window.setWindowTitle(f"VeraLux v{VERSION}")
        self.app.setStyle("Fusion") 
        self.window.setStyleSheet(DARK_STYLESHEET)
        self.window.setMinimumWidth(620) 
        self.window.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint)
        central = QWidget()
        self.window.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(8) 
        
        # Header
        head_title = QLabel(f"VeraLux HyperMetric Stretch v{VERSION}")
        head_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        head_title.setStyleSheet("font-size: 14pt; font-weight: bold; color: #88aaff;")
        layout.addWidget(head_title)
        
        subhead = QLabel("Requirement: Linear Data • Color Calibration Applied")
        subhead.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subhead.setStyleSheet("font-size: 9pt; color: #999999; font-style: italic; margin-bottom: 5px;")
        layout.addWidget(subhead)
        
        # --- GUI BLOCKS ---
        # 0. Mode
        grp_mode = QGroupBox("0. Processing Mode")
        l_mode = QVBoxLayout(grp_mode)
        self.radio_ready = QRadioButton("Ready-to-Use (Aesthetic)")
        self.radio_ready.setToolTip("<b>Ready-to-Use Mode:</b><br>Produces an aesthetic, export-ready image.")
        self.radio_scientific = QRadioButton("Scientific (Preserve)")
        self.radio_scientific.setToolTip("<b>Scientific Mode:</b><br>Produces a 100% mathematically consistent output.")
        self.radio_ready.setChecked(True) 
        self.mode_group = QButtonGroup()
        self.mode_group.addButton(self.radio_ready, 0)
        self.mode_group.addButton(self.radio_scientific, 1)
        l_mode.addWidget(self.radio_ready)
        l_mode.addWidget(self.radio_scientific)
        self.label_mode_info = QLabel("✓ Ready-to-Use selected")
        self.label_mode_info.setStyleSheet("color: #999999; font-size: 9pt;")
        l_mode.addWidget(self.label_mode_info)
        
        # 1. Sensor
        grp_space = QGroupBox("1. Sensor Calibration")
        l_space = QVBoxLayout(grp_space)
        l_combo = QHBoxLayout()
        l_combo.addWidget(QLabel("Sensor Profile:")) 
        self.combo_profile = QComboBox()
        self.combo_profile.setToolTip("<b>Sensor Profile:</b><br>Defines the Luminance coefficients.")
        for profile_name in SENSOR_PROFILES.keys(): self.combo_profile.addItem(profile_name)
        self.combo_profile.setCurrentText(DEFAULT_PROFILE)
        l_combo.addWidget(self.combo_profile)
        l_space.addLayout(l_combo)
        self.label_profile_info = QLabel("Rec.709 Standard")
        self.label_profile_info.setStyleSheet("color: #999999; font-size: 9pt;")
        l_space.addWidget(self.label_profile_info)
        
        top_row = QHBoxLayout()
        top_row.addWidget(grp_mode); top_row.addWidget(grp_space)
        layout.addLayout(top_row)
        
        # 2. Stretch & Calibration
        grp_combined = QGroupBox("2. Stretch Engine & Calibration")
        l_combined = QVBoxLayout(grp_combined)
        
        l_calib = QHBoxLayout()
        l_calib.addWidget(QLabel("Target Bg:"))
        self.spin_target = QDoubleSpinBox()
        self.spin_target.setToolTip("<b>Target Background (Median):</b><br>The desired median value for the background sky.")
        self.spin_target.setRange(0.05, 0.50); self.spin_target.setValue(0.20); self.spin_target.setSingleStep(0.01)
        l_calib.addWidget(self.spin_target)
        
        self.chk_adaptive = QCheckBox("Adaptive Anchor")
        self.chk_adaptive.setChecked(True)
        self.chk_adaptive.setToolTip("<b>Adaptive Anchor:</b><br>Analyzes histogram to find true signal start.")
        l_calib.addWidget(self.chk_adaptive)
        
        self.btn_auto = QPushButton("⚡ Auto-Calc Log D")
        self.btn_auto.setToolTip("<b>Auto-Solver:</b><br>Finds optimal Stretch Factor (Log D).")
        self.btn_auto.setObjectName("AutoButton")
        self.btn_auto.clicked.connect(self.run_solver)
        l_calib.addWidget(self.btn_auto)
        
        self.btn_preview = QPushButton("👁️ Live Preview")
        self.btn_preview.setObjectName("PreviewButton")
        self.btn_preview.setToolTip("Toggle Real-Time Interactive Preview Window")
        self.btn_preview.clicked.connect(self.toggle_preview)
        l_calib.addWidget(self.btn_preview)
        
        l_combined.addLayout(l_calib)
        l_combined.addSpacing(5)
        
        l_manual = QHBoxLayout()
        l_manual.addWidget(QLabel("Log D:"))
        self.spin_d = QDoubleSpinBox()
        self.spin_d.setToolTip("<b>Hyperbolic Intensity (Log D):</b><br>Controls the strength of the stretch.")
        self.spin_d.setRange(0.0, 7.0); self.spin_d.setValue(2.0); self.spin_d.setDecimals(2); self.spin_d.setSingleStep(0.1)
        self.slide_d = QSlider(Qt.Orientation.Horizontal)
        self.slide_d.setRange(0, 700); self.slide_d.setValue(200)
        l_manual.addWidget(self.spin_d); l_manual.addWidget(self.slide_d)
        
        l_manual.addSpacing(15)
        l_manual.addWidget(QLabel("Protect b:"))
        self.spin_b = QDoubleSpinBox()
        self.spin_b.setToolTip("<b>Highlight Protection (b):</b><br>Controls the 'knee' of the Hyperbolic curve.")
        self.spin_b.setRange(0.1, 15.0); self.spin_b.setValue(6.0); self.spin_b.setSingleStep(0.1)
        l_manual.addWidget(self.spin_b)
        l_combined.addLayout(l_manual)
        layout.addWidget(grp_combined)
        
        # 3. Physics
        grp_phys = QGroupBox("3. Physics & Color Engine")
        l_phys = QVBoxLayout(grp_phys)
        l_conv = QHBoxLayout()
        l_conv.addWidget(QLabel("Star Core Recovery (White Point):"))
        self.spin_conv = QDoubleSpinBox()
        self.spin_conv.setToolTip("<b>Color Convergence:</b><br>Controls how quickly saturated colors transition to white.")
        self.spin_conv.setRange(1.0, 10.0); self.spin_conv.setValue(3.5)
        l_conv.addWidget(self.spin_conv)
        l_phys.addLayout(l_conv)
        
        # Ready-to-Use controls
        self.container_ready = QWidget()
        l_ready = QVBoxLayout(self.container_ready)
        l_ready.setContentsMargins(0,0,0,0)
        l_ready.setSpacing(2)
        
        l_uni = QHBoxLayout()
        l_uni.addWidget(QLabel("Color Strategy:"))
        self.slide_unified = ResetSlider(Qt.Orientation.Horizontal, default_value=0)
        self.slide_unified.setToolTip("<b>Unified Color Strategy:</b><br>• Center (0): Balanced.<br>• Left: Clean Noise.<br>• Right: Soften Highlights.")
        self.slide_unified.setRange(-100, 100); self.slide_unified.setValue(0)
        l_uni.addWidget(self.slide_unified)
        l_ready.addLayout(l_uni)
        
        self.lbl_strategy_feedback = QLabel("Balanced (Pure Vector)")
        self.lbl_strategy_feedback.setStyleSheet("color: #999999; font-size: 8pt; font-style: italic; margin-left: 80px;")
        l_ready.addWidget(self.lbl_strategy_feedback)
        
        # Scientific controls
        self.container_scientific = QWidget()
        l_sci = QVBoxLayout(self.container_scientific)
        l_sci.setContentsMargins(0,0,0,0)
        
        l_grip = QHBoxLayout()
        l_grip.addWidget(QLabel("Color Grip (Global):"))
        self.spin_grip = QDoubleSpinBox()
        self.spin_grip.setToolTip("<b>Color Grip:</b> Controls vector preservation rigor.")
        self.spin_grip.setRange(0.0, 1.0); self.spin_grip.setValue(1.0); self.spin_grip.setSingleStep(0.05)
        self.slide_grip = QSlider(Qt.Orientation.Horizontal)
        self.slide_grip.setRange(0, 100); self.slide_grip.setValue(100)
        self.slide_grip.valueChanged.connect(lambda v: self.spin_grip.setValue(v/100.0))
        self.spin_grip.valueChanged.connect(lambda v: self.slide_grip.setValue(int(v*100)))
        l_grip.addWidget(self.spin_grip); l_grip.addWidget(self.slide_grip)
        l_sci.addLayout(l_grip)
        
        l_shadow = QHBoxLayout()
        l_shadow.addWidget(QLabel("Shadow Conv. (Noise):"))
        self.spin_shadow = QDoubleSpinBox()
        self.spin_shadow.setToolTip("<b>Shadow Convergence:</b><br>Damps vector preservation in shadows.")
        self.spin_shadow.setRange(0.0, 3.0); self.spin_shadow.setValue(0.0); self.spin_shadow.setSingleStep(0.1)
        self.slide_shadow = QSlider(Qt.Orientation.Horizontal)
        self.slide_shadow.setRange(0, 300); self.slide_shadow.setValue(0)
        self.slide_shadow.valueChanged.connect(lambda v: self.spin_shadow.setValue(v/100.0))
        self.spin_shadow.valueChanged.connect(lambda v: self.slide_shadow.setValue(int(v*100)))
        l_shadow.addWidget(self.spin_shadow); l_shadow.addWidget(self.slide_shadow)
        l_sci.addLayout(l_shadow)
        
        l_phys.addWidget(self.container_ready)
        l_phys.addWidget(self.container_scientific)
        layout.addWidget(grp_phys)
        
        # Footer
        self.progress = QProgressBar(); self.progress.setTextVisible(True)
        layout.addWidget(self.progress)
        self.status = QLabel("Ready. Please cache input first.")
        self.status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status)
        
        # Buttons
        btns = QHBoxLayout()
        self.btn_help = QPushButton("?"); self.btn_help.setObjectName("HelpButton"); self.btn_help.setFixedWidth(20)
        self.btn_help.setToolTip("Print Operational Guide to Console")
        self.chk_ontop = QCheckBox("Always on top"); self.chk_ontop.setChecked(True)
        self.chk_ontop.setToolTip("Keep this window above other windows")
        b_reset = QPushButton("Defaults")
        b_reset.setToolTip("Reset all sliders and dropdowns to default values.")
        b_reload = QPushButton("Reload Input")
        b_reload.setToolTip("Reload linear image from active document. Use Undo in SASPro to revert changes.")
        b_proc = QPushButton("PROCESS"); b_proc.setObjectName("ProcessButton")
        b_proc.setToolTip("Apply the stretch to the image.")
        b_close = QPushButton("Close"); b_close.setObjectName("CloseButton")
        
        btns.addWidget(self.btn_help); btns.addWidget(self.chk_ontop)
        btns.addWidget(b_reset); btns.addWidget(b_reload); btns.addWidget(b_proc); btns.addWidget(b_close)
        layout.addLayout(btns)
        
        # CONNECT SIGNALS
        self.chk_ontop.toggled.connect(self.toggle_ontop)
        self.btn_help.clicked.connect(self.print_help_to_console)
        b_reset.clicked.connect(self.set_defaults)
        b_reload.clicked.connect(self.cache_input)
        b_proc.clicked.connect(self.run_process)
        b_close.clicked.connect(self.window.close)
        
        self.slide_d.valueChanged.connect(lambda v: self.spin_d.setValue(v/100.0))
        self.spin_d.valueChanged.connect(lambda v: self.slide_d.setValue(int(v*100)))
        
        self.radio_ready.toggled.connect(self.update_mode_ui)
        self.radio_scientific.toggled.connect(self.update_mode_ui)
        
        self.combo_profile.currentTextChanged.connect(self.update_profile_info)
        self.slide_unified.valueChanged.connect(self.update_unified_feedback)
        
        # LIVE PREVIEW CONNECTIONS (Debounced)
        self.debounce_timer = QTimer()
        self.debounce_timer.setSingleShot(True)
        self.debounce_timer.setInterval(150)
        self.debounce_timer.timeout.connect(self.update_preview_image)
        
        for widget in [self.spin_d, self.spin_b, self.spin_conv, self.spin_target, self.spin_grip, self.spin_shadow]:
            widget.valueChanged.connect(self.trigger_preview_update)
        self.slide_unified.valueChanged.connect(self.trigger_preview_update)
        self.combo_profile.currentTextChanged.connect(self.trigger_preview_update)
        self.radio_ready.toggled.connect(self.trigger_preview_update)
        self.slide_d.valueChanged.connect(self.trigger_preview_update)
        self.slide_grip.valueChanged.connect(self.trigger_preview_update)
        self.slide_shadow.valueChanged.connect(self.trigger_preview_update)
        self.chk_adaptive.toggled.connect(self.trigger_preview_update)

        self.update_profile_info(DEFAULT_PROFILE)
        
        # Load saved settings
        saved_profile = self.settings.value("sensor_profile", DEFAULT_PROFILE)
        if saved_profile in SENSOR_PROFILES:
            self.combo_profile.setCurrentText(saved_profile)
        is_ready = self.settings.value("mode_ready", True, type=bool)
        if is_ready: self.radio_ready.setChecked(True)
        else: self.radio_scientific.setChecked(True)
        saved_target = self.settings.value("target_bg", 0.20, type=float)
        self.spin_target.setValue(saved_target)
        
        self.update_mode_ui()
        self.window.show()
        self.center_window()
        self.cache_input()

    def get_effective_params(self):
        if self.radio_ready.isChecked():
            val = self.slide_unified.value()
            if val < 0:
                shadow = (abs(val) / 100.0) * 3.0
                grip = 1.0
            else:
                grip = 1.0 - ((val / 100.0) * 0.6)
                shadow = 0.0
        else:
            grip = self.spin_grip.value()
            shadow = self.spin_shadow.value()
        return grip, shadow

    def update_unified_feedback(self):
        val = self.slide_unified.value()
        grip, shadow = self.get_effective_params()
        if val < 0:
            txt = f"Action: Noise Cleaning (Shadow Conv: {shadow:.1f})"
        elif val > 0:
            txt = f"Action: Highlight Softening (Grip: {grip:.2f})"
        else:
            txt = "Balanced (Pure Vector)"
        self.lbl_strategy_feedback.setText(txt)

    def update_mode_ui(self):
        is_ready = self.radio_ready.isChecked()
        self.container_ready.setVisible(is_ready)
        self.container_scientific.setVisible(not is_ready)
        if is_ready:
            self.label_mode_info.setText("✓ Ready-to-Use: Unified Color Strategy enabled.")
        else:
            self.label_mode_info.setText("✓ Scientific: Full manual parameter control.")
        QTimer.singleShot(10, self.window.adjustSize)

    def toggle_preview(self):
        if not self.preview_window:
            self.preview_window = VeraLuxPreviewWindow()
        if self.preview_window.isVisible():
            self.preview_window.hide()
        else:
            if self.preview_proxy is None:
                self.prepare_preview_proxy()
            self.preview_window.show()
            self.preview_window.raise_()
            self.preview_window.activateWindow()
            self.update_preview_image()
            self.preview_window.fit_to_view()

    def prepare_preview_proxy(self):
        if self.linear_cache is None: return
        img = VeraLuxCore.normalize_input(self.linear_cache)
        if img.ndim == 3 and img.shape[0] != 3 and img.shape[2] == 3:
            img = img.transpose(2, 0, 1)
        h = img.shape[1] if img.ndim == 3 else img.shape[0]
        w = img.shape[2] if img.ndim == 3 else img.shape[1]
        scale = 1600 / max(h, w)
        if scale >= 1.0:
            self.preview_proxy = img
        else:
            step = int(1 / scale)
            if img.ndim == 3:
                self.preview_proxy = img[:, ::step, ::step]
            else:
                self.preview_proxy = img[::step, ::step]

    def trigger_preview_update(self):
        if self.preview_window and self.preview_window.isVisible():
            self.debounce_timer.start()

    def update_preview_image(self):
        if self.preview_proxy is None: return
        D = self.spin_d.value()
        b = self.spin_b.value()
        conv = self.spin_conv.value()
        grip, shadow = self.get_effective_params()
        adaptive = self.chk_adaptive.isChecked()
        ws = self.combo_profile.currentText()
        target_bg = self.spin_target.value()
        mode_str = "ready_to_use" if self.radio_ready.isChecked() else "scientific"
        res = process_veralux_v6(self.preview_proxy.copy(), D, b, conv, ws, mode_str, target_bg, grip, shadow, adaptive, None)
        qimg = self.numpy_to_qimage(res)
        self.preview_window.set_image(qimg, res, self.is_source_saturated, mode_str)

    def numpy_to_qimage(self, img_data):
        if img_data.ndim == 3:
            disp = img_data.transpose(1, 2, 0)
        else:
            disp = img_data
        disp = np.clip(disp * 255.0, 0, 255).astype(np.uint8)     
        disp = np.flipud(disp)
        disp = np.ascontiguousarray(disp)
        h, w = disp.shape[0], disp.shape[1]
        bytes_per_line = disp.strides[0]
        data_bytes = disp.data.tobytes()
        if disp.ndim == 2:
            qimg = QImage(data_bytes, w, h, bytes_per_line, QImage.Format.Format_Grayscale8)
        else:
            if disp.shape[2] == 3:
                qimg = QImage(data_bytes, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            else:
                return QImage()
        return qimg.copy()

    def handle_close_event(self, event):
        self.settings.setValue("sensor_profile", self.combo_profile.currentText())
        self.settings.setValue("mode_ready", self.radio_ready.isChecked())
        self.settings.setValue("target_bg", self.spin_target.value())
        if self.preview_window:
            self.preview_window.close()
        event.accept()

    def print_help_to_console(self):
        guide_lines = [
            "==========================================================================",
            "   VERALUX HYPERMETRIC STRETCH v1.3 - OPERATIONAL GUIDE",
            "   Physics-Based Photometric Hyperbolic Stretch Engine",
            "==========================================================================",
            "",
            "OVERVIEW",
            "-----------------",
            "VeraLux provides a mathematically precise linear-to-nonlinear stretch that",
            "preserves the photometric color ratios (Vector Color) often destroyed by",
            "standard histogram transformations (Hue Shift).",
            "",
            "[1] CRITICAL PREREQUISITES",
            "    • Input MUST be Linear (not yet stretched).",
            "    • Background gradients must have been removed.",
            "    • RGB input must be Color Calibrated.",
            "",
            "[2] THE DUAL PHILOSOPHY (MODES)",
            "    VeraLux allows you to choose between ease of use and manual control.",
            "",
            "    A. Ready-to-Use (Default)",
            "       Designed for aesthetic results. Features the 'Unified Color Strategy' slider.",
            "       • SLIDER CENTER (0): Pure VeraLux. Maximum vector color fidelity.",
            "       • MOVE LEFT (< 0): Increases 'Shadow Convergence'.",
            "         - Use this to suppress background color noise/speckles.",
            "       • MOVE RIGHT (> 0): Decreases 'Color Grip'.",
            "         - Use this to soften highlights and recover detail in saturated cores.",
            "       • TIP: Double-click the slider to reset it to center.",
            "",
            "    B. Scientific Mode",
            "       For purists. Exposes the raw parameters independently.",
            "       • Color Grip: Global blend factor (1.0 = Vivid, 0.5 = Soft).",
            "       • Shadow Convergence: Damps vector logic in shadows to kill noise.",
            "",
            "[3] CALIBRATION & ANCHORING",
            "    • Adaptive Anchor: If checked, the script analyzes the histogram shape",
            "      to find the 'physical' start of the signal. This maximizes contrast",
            "      on well-calibrated frames but may clip if gradients are present.",
            "    • Auto-Calculate: Automatically finds the optimal 'Log D' to place",
            "      the background at the target level (0.20 is standard).",
            "      The solver is now Iterative and Smart. It simulates the entire",
            "      pipeline to ensure the Black Point is not clipped.",
            "",
            "[4] PHYSICS TUNING",
            "    • Stretch (Log D): The intensity of the curve. Higher = Brighter.",
            "    • Protect b: Controls the highlight protection 'knee'.",
            "      - Higher (> 6.0): Sharper stars, more contrast.",
            "      - Lower (< 2.0): Brighter nebulosity, softer stars.",
            "    • Star Core Recovery: Controls how fast saturated colors turn white.",
            "",
            "[5] LIVE HISTOGRAM (Preview Window)",
            "    • Real-time feedback with Smart Indicators (Traffic Light system).",
            "    • ORANGE BAR (> 0.01%): Marginal clipping. Usually safe to ignore.",
            "      (In Ready-to-Use mode, this indicates intentional noise floor trimming).",
            "    • RED BAR (> 0.1%): Structural data loss. Action required.",
            "      - LEFT RED: Crushed blacks. Reduce Log D or Target Bg.",
            "      - RIGHT RED: Blown highlights. Increase Color Convergence.",
            "",
            "[6] TROUBLESHOOTING",
            "    • 'My background is noisy/colorful': Move Strategy Slider to LEFT.",
            "    • 'My stars/nebula cores are flat/burned': Move Strategy Slider to RIGHT.",
            "    • 'The image is too dark': Increase 'Target Bg' or lower 'Protect b'.",
            "",
            "Support & Info: info@veralux.space",
            "=========================================================================="
        ]
        
        for line in guide_lines:
            msg = line if line.strip() else " "
            self.ctx.log(msg)
        self.status.setText("Full Guide printed to Console.")

    def center_window(self):
        screen = self.app.primaryScreen()
        if screen:
            frame_geo = self.window.frameGeometry()
            frame_geo.moveCenter(screen.availableGeometry().center())
            self.window.move(frame_geo.topLeft())

    def toggle_ontop(self, checked):
        pos = self.window.pos()
        if checked: self.window.setWindowFlags(self.window.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
        else: self.window.setWindowFlags(self.window.windowFlags() & ~Qt.WindowType.WindowStaysOnTopHint)
        self.window.show(); self.window.move(pos)

    def update_profile_info(self, profile_name):
        if profile_name in SENSOR_PROFILES:
            profile = SENSOR_PROFILES[profile_name]
            r, g, b = profile['weights']
            self.label_profile_info.setText(f"{profile['description']} (R:{r:.2f} G:{g:.2f} B:{b:.2f})")

    def set_defaults(self):
        self.spin_d.setValue(2.0); self.spin_b.setValue(6.0); self.spin_target.setValue(0.20)
        self.spin_conv.setValue(3.5); self.spin_grip.setValue(1.0); self.spin_shadow.setValue(0.0)
        self.slide_unified.setValue(0)
        self.chk_adaptive.setChecked(True)
        self.combo_profile.setCurrentText(DEFAULT_PROFILE)
        self.radio_ready.setChecked(True)

    def cache_input(self):
        try:
            self.status.setText("Caching Linear Data...")
            self.app.processEvents()
            
            # Get image from SASPro context
            img = self.ctx.get_image()
            if img is None:
                self.status.setText("Error: No active image.")
                return
            
            # SASPro returns (H,W,C), convert to (C,H,W) for internal processing
            if img.ndim == 3 and img.shape[2] in [1, 3, 4]:
                self.linear_cache = img.transpose(2, 0, 1)
            else:
                self.linear_cache = img
                
            self.status.setText("Input Cached.")
            self.ctx.log("VeraLux: Input Cached.")
            
            check_norm = VeraLuxCore.normalize_input(self.linear_cache)
            self.is_source_saturated = np.max(check_norm) > 0.999
            
            self.preview_proxy = None
            if self.preview_window and self.preview_window.isVisible():
                self.prepare_preview_proxy()
                self.update_preview_image()
        except Exception as e:
            self.status.setText(f"Error: {str(e)}")
            self.ctx.log(f"VeraLux Error: {str(e)}")

    def run_solver(self):
        if self.linear_cache is None: return
        self.status.setText("Solving..."); self.btn_auto.setEnabled(False); self.progress.setRange(0, 0)
        tgt = self.spin_target.value(); b = self.spin_b.value(); ws = self.combo_profile.currentText()
        luma = SENSOR_PROFILES[ws]['weights']
        adaptive = self.chk_adaptive.isChecked()
        mode = "ready_to_use" if self.radio_ready.isChecked() else "scientific"
        
        self.solver = AutoSolverThread(self.linear_cache, tgt, b, luma, adaptive, mode)
        self.solver.result_ready.connect(self.apply_solver_result)
        self.solver.start()
        
    def apply_solver_result(self, log_d):
        self.spin_d.setValue(log_d); self.progress.setRange(0, 100); self.progress.setValue(100)
        self.btn_auto.setEnabled(True); ws = self.combo_profile.currentText()
        self.status.setText(f"Solved: Log D = {log_d:.2f}")
        self.ctx.log(f"VeraLux Solver: Optimal Log D={log_d:.2f} [{ws}]")

    def run_process(self):
        if self.linear_cache is None: return
        D = self.spin_d.value(); b = self.spin_b.value(); conv = self.spin_conv.value()
        ws = self.combo_profile.currentText(); t_bg = self.spin_target.value()
        grip, shadow = self.get_effective_params()
        adaptive = self.chk_adaptive.isChecked()
        mode = "ready_to_use" if self.radio_ready.isChecked() else "scientific"
        
        self.status.setText("Processing..."); self.progress.setRange(0, 0)
        img_copy = self.linear_cache.copy()
        
        self.worker = ProcessingThread(img_copy, D, b, conv, ws, mode, t_bg, grip, shadow, adaptive)
        self.worker.progress.connect(self.status.setText)
        self.worker.finished.connect(self.finish_process)
        self.worker.start()
        
    def finish_process(self, result_img):
        self.progress.setRange(0, 100); self.progress.setValue(100); self.status.setText("Complete.")
        mode = "Ready-to-Use" if self.radio_ready.isChecked() else "Scientific"
        ws = self.combo_profile.currentText()
        if result_img is not None:
            # Convert back to (H,W,C) for SASPro
            if result_img.ndim == 3:
                output = result_img.transpose(1, 2, 0)
            else:
                output = result_img
            
            # Set image via SASPro context (handles undo automatically)
            self.ctx.set_image(output, step_name=f"VeraLux v{VERSION} Stretch")
            self.ctx.log(f"VeraLux v{VERSION}: {mode} mode applied [{ws}]")

# =============================================================================
#  SASPRO SCRIPT ENTRYPOINT
# =============================================================================

def run(ctx):
    """
    SASPro script entrypoint.
    Launches the VeraLux HyperMetric Stretch GUI.
    """
    try:
        # Check for active image
        img = ctx.get_image()
        if img is None:
            ctx.log("VeraLux Error: No active image. Please open an image first.")
            return
        
        # Get or create QApplication instance
        app = QApplication.instance()
        if not app:
            app = QApplication(sys.argv)
        
        # Create and show the GUI
        gui = VeraLuxInterface(ctx, app)
        
        # Run the event loop
        app.exec()
        
    except Exception as e:
        ctx.log(f"VeraLux Error: {str(e)}")
        traceback.print_exc()


if __name__ == "__main__":
    print("VeraLux HyperMetric Stretch v" + VERSION)
    print("This script must be run from within SASPro.")
    print("Please open SASPro and run this script from the Scripts menu.")

