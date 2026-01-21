#!/usr/bin/env python3
##############################################################################
# HDR_Blender.py (SAS Pro) - Version 1.0
# HDR Blending with Multiple Stretch Method Options
# Author: Dark Energy
# Copyright (C) 2025
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
##############################################################################
# ATTRIBUTION & DERIVATIVE WORK NOTICE
##############################################################################
#
# The VeraLux stretch method in this script is based on:
#
#   VeraLux — HyperMetric Stretch
#   Photometric Hyperbolic Stretch Engine
#   Original Author: Riccardo Paterniti (2025)
#   Contact: info@veralux.space
#   License: GPL-3.0-or-later
#
# The VeraLux algorithm implements luminance-chrominance separation with
# color vector preservation during arcsinh stretching. 
#
# Modifications for SAS Pro (December 2025):
#   - Integrated into HDR exposure blending workflow
#   - Added Per-Channel and Linked stretch method alternatives
#   - Added luminosity mask-based exposure blending
#   - Added background clipping options
#   - Adapted UI for SAS Pro framework
#
##############################################################################
"""
HDR_Blender.py - HDR blending with multiple stretch method options.

Stretch Methods:
- VeraLux (Color Vector Preservation): Stretch luminance only, preserve R/G/B ratios
- Per-Channel Unlinked: Arcsinh stretch each channel independently
- Linked: Arcsinh stretch all channels with same parameters

All methods use arcsinh stretch with auto Log D solving.
"""

from __future__ import annotations
import numpy as np
import json
from datetime import datetime

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton,
    QMessageBox, QGroupBox, QGridLayout, QCheckBox, QDoubleSpinBox,
    QScrollArea, QFileDialog, QFrame, QSlider
)
from PyQt6.QtCore import Qt, QEvent, QPointF
from PyQt6.QtGui import QPixmap, QImage

# Preview helpers
try:
    from pro.curve_editor_pro import _float_to_qimage_rgb8, _downsample_for_preview, ImageLabel
except Exception:
    from PyQt6.QtWidgets import QLabel as _QLabel
    from PyQt6.QtCore import pyqtSignal

    class ImageLabel(_QLabel):
        mouseMoved = pyqtSignal(float, float)
        def mouseMoveEvent(self, event):
            if self.pixmap() is not None:
                self.mouseMoved.emit(event.position().x(), event.position().y())
            super().mouseMoveEvent(event)

    def _float_to_qimage_rgb8(img):
        arr = np.asarray(img, dtype=np.float32)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        elif arr.ndim == 3 and arr.shape[2] == 1:
            arr = np.concatenate([arr, arr, arr], axis=-1)
        arr = np.clip(arr, 0.0, 1.0)
        arr8 = (arr * 255.0 + 0.5).astype(np.uint8)
        arr8 = np.ascontiguousarray(arr8)
        h, w = arr8.shape[:2]
        qimg = QImage(arr8.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        return qimg.copy()

    def _downsample_for_preview(arr, maxdim=1200):
        a = np.asarray(arr, dtype=np.float32)
        h, w = a.shape[:2]
        if max(h, w) <= maxdim:
            return a.copy()
        scale = float(maxdim) / float(max(h, w))
        new_h, new_w = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
        yi = np.linspace(0, h - 1, new_h).astype(np.int32)
        xi = np.linspace(0, w - 1, new_w).astype(np.int32)
        if a.ndim == 2:
            return a[np.ix_(yi, xi)].copy()
        else:
            return a[np.ix_(yi, xi)].copy()


# =============================================================================
# Constants
# =============================================================================

SCRIPT_NAME = "HDR Blender"
SCRIPT_VERSION = "1.0"
_NONE_PLACEHOLDER = "<None>"


# =============================================================================
# Stretch Methods
# =============================================================================

class StretchMethod:
    VERALUX = 0          # Color vector preservation (stretch L only)
    PER_CHANNEL = 1      # Each channel independently
    LINKED = 2           # Same parameters for all channels


STRETCH_METHOD_NAMES = [
    "VeraLux (Color Vector Preservation)",
    "Per-Channel (Unlinked)",
    "Linked (Same Parameters)",
]

STRETCH_METHOD_TOOLTIPS = [
    "Stretch luminance only, then reconstruct with preserved R/G/B ratios.\n"
    "Best color preservation. Recommended for most HDR work.",

    "Stretch each R, G, B channel independently with its own anchor/Log D.\n"
    "Each channel optimized separately. May cause color shifts.",

    "Stretch all channels with the same anchor and Log D.\n"
    "Preserves relative brightness between channels.",
]


# =============================================================================
# Sensor Profiles
# =============================================================================

SENSOR_PROFILES = {
    # --- STANDARD ---
    "Rec.709 (Recommended)": {
        'weights': (0.2126, 0.7152, 0.0722),
        'description': "ITU-R BT.709 standard for sRGB/HDTV",
    },
    # --- SONY STARVIS ---
    "Sony IMX571 (ASI2600/QHY268)": {
        'weights': (0.2944, 0.5021, 0.2035),
        'description': "Sony IMX571 26MP APS-C BSI (STARVIS)",
    },
    "Sony IMX533 (ASI533/QHY533)": {
        'weights': (0.2910, 0.5072, 0.2018),
        'description': "Sony IMX533 9MP 1\" Square BSI (STARVIS)",
    },
    "Sony IMX455 (ASI6200/QHY600)": {
        'weights': (0.2987, 0.5001, 0.2013),
        'description': "Sony IMX455 61MP Full Frame BSI (STARVIS)",
    },
    "Sony IMX294 (ASI294)": {
        'weights': (0.3068, 0.5008, 0.1925),
        'description': "Sony IMX294 11.7MP 4/3\" BSI",
    },
    "Sony IMX183 (ASI183/QHY183)": {
        'weights': (0.2967, 0.4983, 0.2050),
        'description': "Sony IMX183 20MP 1\" BSI",
    },
    # --- SONY STARVIS 2 ---
    "Sony IMX585 (ASI585/462/678)": {
        'weights': (0.3431, 0.4822, 0.1747),
        'description': "Sony IMX585 8.3MP 1/1.2\" BSI (STARVIS 2) - NIR optimized",
    },
    # --- PANASONIC ---
    "Panasonic MN34230 (ASI1600)": {
        'weights': (0.2650, 0.5250, 0.2100),
        'description': "Panasonic MN34230 4/3\" CMOS",
    },
    # --- DSLR ---
    "Canon EOS (Modern)": {
        'weights': (0.2550, 0.5250, 0.2200),
        'description': "Canon CMOS Profile (60D, 6D, 5D, R-series)",
    },
    "Nikon DSLR (Modern)": {
        'weights': (0.2600, 0.5100, 0.2300),
        'description': "Nikon Expeed 4+ cameras (D5300/D850)",
    },
    # --- SMART TELESCOPES ---
    "ZWO Seestar S50": {
        'weights': (0.3333, 0.4866, 0.1801),
        'description': "ZWO Seestar S50 (IMX462)",
    },
    # --- NARROWBAND PALETTES ---
    "HOO (Ha-OIII-OIII)": {
        'weights': (0.5000, 0.2500, 0.2500),
        'description': "Bicolor palette: Ha=Red, OIII=Green+Blue",
    },
    "SHO (Hubble Palette)": {
        'weights': (0.3333, 0.3400, 0.3267),
        'description': "Hubble palette: SII=Red, Ha=Green, OIII=Blue",
    },
    # --- NEUTRAL ---
    "Equal Weights": {
        'weights': (0.3333, 0.3333, 0.3334),
        'description': "Equal channel contribution",
    },
}

DEFAULT_PROFILE = "Rec.709 (Recommended)"


# =============================================================================
# Core Engine
# =============================================================================

class HDRCore:
    """Core math for HDR blending and stretching."""

    @staticmethod
    def normalize_input(img_data):
        """Normalize input to float32 [0, 1]."""
        input_dtype = img_data.dtype
        img_float = img_data.astype(np.float32)

        if np.issubdtype(input_dtype, np.integer):
            if input_dtype == np.uint8:
                return img_float / 255.0
            elif input_dtype == np.uint16:
                return img_float / 65535.0
            elif input_dtype == np.uint32:
                return img_float / 4294967295.0
            else:
                info = np.iinfo(input_dtype)
                return img_float / float(info.max)

        elif np.issubdtype(input_dtype, np.floating):
            current_max = np.nanmax(img_data)
            if current_max <= 1.0 + 1e-5:
                return img_float
            if current_max <= 255.0:
                return img_float / 255.0
            if current_max <= 65535.0:
                return img_float / 65535.0
            return img_float / 4294967295.0

        return img_float

    @staticmethod
    def ensure_hwc(img):
        """Ensure image is (H, W, C) format with 3 channels."""
        if img.ndim == 2:
            return np.stack([img, img, img], axis=-1).astype(np.float32)
        if img.ndim == 3:
            if img.shape[2] == 1:
                return np.repeat(img, 3, axis=2).astype(np.float32)
            elif img.shape[2] == 4:
                r = img[..., 0]
                g = (img[..., 1] + img[..., 2]) / 2.0
                b = img[..., 3]
                return np.stack([r, g, b], axis=-1).astype(np.float32)
            elif img.shape[2] > 4:
                return img[..., :3].astype(np.float32)
        return img.astype(np.float32)

    @staticmethod
    def calculate_anchor(data_norm):
        """Calculate black point anchor."""
        if data_norm.ndim == 3:
            floors = []
            stride = max(1, data_norm.size // 500000)
            for c in range(data_norm.shape[2]):
                channel_floor = np.percentile(data_norm[..., c].flatten()[::stride], 0.5)
                floors.append(channel_floor)
            anchor = max(0.0, min(floors) - 0.00025)
        else:
            stride = max(1, data_norm.size // 200000)
            floor = np.percentile(data_norm.flatten()[::stride], 0.5)
            anchor = max(0.0, floor - 0.00025)
        return float(anchor)

    @staticmethod
    def calculate_channel_anchor(channel):
        """Calculate anchor for a single channel."""
        stride = max(1, channel.size // 200000)
        floor = np.percentile(channel.flatten()[::stride], 0.5)
        return max(0.0, float(floor) - 0.00025)

    @staticmethod
    def extract_luminance(img, weights):
        """Extract luminance using sensor weights."""
        r_w, g_w, b_w = weights
        if img.ndim == 2:
            return img.astype(np.float32)
        return (r_w * img[..., 0] + g_w * img[..., 1] + b_w * img[..., 2]).astype(np.float32)

    @staticmethod
    def hyperbolic_stretch(data, D, b, SP=0.0):
        """Inverse hyperbolic (arcsinh) stretch."""
        D = max(D, 0.1)
        b = max(b, 0.1)
        term1 = np.arcsinh(D * (data - SP) + b)
        term2 = np.arcsinh(b)
        norm_factor = np.arcsinh(D * (1.0 - SP) + b) - term2
        if norm_factor == 0:
            norm_factor = 1e-6
        stretched = (term1 - term2) / norm_factor
        return np.clip(stretched, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def solve_log_d(data_sample, target_median, b_val):
        """Binary search for optimal Log D."""
        median_in = np.median(data_sample)
        if median_in < 1e-9:
            return 2.0

        low_log, high_log = 0.0, 7.0
        best_log_D = 2.0

        for _ in range(40):
            mid_log = (low_log + high_log) / 2.0
            mid_D = 10.0 ** mid_log
            test_val = HDRCore.hyperbolic_stretch(median_in, mid_D, b_val)

            if abs(test_val - target_median) < 0.0001:
                best_log_D = mid_log
                break

            if test_val < target_median:
                low_log = mid_log
            else:
                high_log = mid_log
            best_log_D = mid_log

        return best_log_D


# =============================================================================
# HDR Blend Function
# =============================================================================

def hdr_blend(
    short_exp, long_exp, mid_exp=None,
    weights=(0.2126, 0.7152, 0.0722),
    # Stretch method
    stretch_method=StretchMethod.VERALUX,
    # Stretch parameters
    log_D=2.5,
    protect_b=6.0,
    # Background clipping options
    clip_background=False,
    clip_method=0,  # 0=target black level, 1=statistical (N*sigma)
    target_black=0.01,
    clip_sigma=2.5,
    # VeraLux color parameters (only for VERALUX stretch method)
    convergence_power=3.5,
    color_grip=1.0,
    # Blend parameters
    blend_threshold=0.05,
    blend_feather=0.10,
    protect_highlights=True,
    # VeraLux blend options
    lum_chrom_blend=False,
    # Saturation (for non-VeraLux methods)
    saturation=1.0,
    # Output
    output_boost=1.0,
):
    """
    HDR blend with selectable stretch method.

    Parameters
    ----------
    short_exp, long_exp : ndarray
        Short and long exposure images
    mid_exp : ndarray, optional
        Medium exposure for 3-way blend
    weights : tuple
        Sensor-specific luminance weights (R, G, B)
    stretch_method : int
        0=VeraLux, 1=Per-Channel, 2=Linked
    log_D : float
        Stretch intensity (log10 scale)
    protect_b : float
        Highlight protection
    clip_background : bool
        Enable background clipping
    clip_method : int
        0 = Target black level (set percentile to target value)
        1 = Statistical clipping (median - N*sigma)
    target_black : float
        Target value for black point (used by both methods)
    clip_sigma : float
        Sigma multiplier for statistical clipping (e.g., 2.5)
    convergence_power : float
        Star core convergence (VeraLux stretch only)
    color_grip : float
        Color control for VeraLux stretch:
        0.0-1.0 = blend between scalar (soft) and vector (vivid)
        >1.0 = saturation boost (compensate for L/C blend washout)
    blend_threshold : float
        Luminosity mask threshold
    blend_feather : float
        Mask feather amount
    protect_highlights : bool
        Force short exp where long is clipped
    lum_chrom_blend : bool
        If True, blend luminance and color ratios separately (VeraLux-style)
        using mask-weighted blending for both. Use color_grip > 1.0 to
        compensate for any color washout from ratio averaging.
        If False, blend RGB channels directly (standard approach).
    saturation : float
        Saturation adjustment for all stretch methods (1.0 = no change).
        Useful to compensate for washout from L/C ratio blending.
    output_boost : float
        Final brightness multiplier

    Returns
    -------
    ndarray
        Blended and stretched result
    """
    # Normalize inputs
    short_exp = HDRCore.normalize_input(short_exp)
    short_exp = HDRCore.ensure_hwc(short_exp)
    long_exp = HDRCore.normalize_input(long_exp)
    long_exp = HDRCore.ensure_hwc(long_exp)

    if mid_exp is not None:
        mid_exp = HDRCore.normalize_input(mid_exp)
        mid_exp = HDRCore.ensure_hwc(mid_exp)

    epsilon = 1e-9

    # =========================================================================
    # Step 1: Create luminosity mask from long exposure
    # =========================================================================
    lum_long = HDRCore.extract_luminance(long_exp, weights)

    if blend_feather < 0.001:
        highlight_mask = (lum_long > blend_threshold).astype(np.float32)
    else:
        low = blend_threshold - blend_feather / 2
        high = blend_threshold + blend_feather / 2
        highlight_mask = np.clip((lum_long - low) / max(high - low, 1e-6), 0.0, 1.0)

    if protect_highlights:
        clipped = (lum_long > 0.98).astype(np.float32)
        highlight_mask = np.maximum(highlight_mask, clipped)

    shadow_mask = 1.0 - highlight_mask

    # =========================================================================
    # Step 2: Blend exposures
    # =========================================================================
    if lum_chrom_blend:
        # VeraLux-style: blend luminance and color ratios separately
        L_short = HDRCore.extract_luminance(short_exp, weights)
        L_long = HDRCore.extract_luminance(long_exp, weights)

        # Extract color ratios (R/L, G/L, B/L)
        L_short_safe = L_short + epsilon
        L_long_safe = L_long + epsilon
        r_ratio_short = short_exp[..., 0] / L_short_safe
        g_ratio_short = short_exp[..., 1] / L_short_safe
        b_ratio_short = short_exp[..., 2] / L_short_safe
        r_ratio_long = long_exp[..., 0] / L_long_safe
        g_ratio_long = long_exp[..., 1] / L_long_safe
        b_ratio_long = long_exp[..., 2] / L_long_safe

        if mid_exp is not None:
            L_mid = HDRCore.extract_luminance(mid_exp, weights)
            L_mid_safe = L_mid + epsilon
            r_ratio_mid = mid_exp[..., 0] / L_mid_safe
            g_ratio_mid = mid_exp[..., 1] / L_mid_safe
            b_ratio_mid = mid_exp[..., 2] / L_mid_safe

            # Three-zone blend for luminance
            thresh_high = blend_threshold
            thresh_low = blend_threshold * 0.4
            high_mask = highlight_mask
            low_mask = 1.0 - np.clip((lum_long - thresh_low) / max(thresh_high - thresh_low, 1e-6), 0.0, 1.0)
            mid_mask = np.clip(1.0 - high_mask - low_mask, 0.0, 1.0)
            total = high_mask + mid_mask + low_mask + epsilon

            # Blend luminances with mask
            L_blended = (L_short * high_mask + L_mid * mid_mask + L_long * low_mask) / total

            # Blend color ratios using same mask as luminance (mask-weighted)
            r_ratio = (r_ratio_short * high_mask + r_ratio_mid * mid_mask + r_ratio_long * low_mask) / total
            g_ratio = (g_ratio_short * high_mask + g_ratio_mid * mid_mask + g_ratio_long * low_mask) / total
            b_ratio = (b_ratio_short * high_mask + b_ratio_mid * mid_mask + b_ratio_long * low_mask) / total
        else:
            # Two-zone blend for luminance
            L_blended = L_short * highlight_mask + L_long * shadow_mask

            # Blend color ratios using same mask as luminance (mask-weighted)
            r_ratio = r_ratio_short * highlight_mask + r_ratio_long * shadow_mask
            g_ratio = g_ratio_short * highlight_mask + g_ratio_long * shadow_mask
            b_ratio = b_ratio_short * highlight_mask + b_ratio_long * shadow_mask

        # Reconstruct blended RGB from luminance and ratios
        blended = np.zeros_like(short_exp)
        blended[..., 0] = L_blended * r_ratio
        blended[..., 1] = L_blended * g_ratio
        blended[..., 2] = L_blended * b_ratio

    else:
        # Standard RGB channel blend
        if mid_exp is not None:
            thresh_high = blend_threshold
            thresh_low = blend_threshold * 0.4

            high_mask = highlight_mask
            low_mask = 1.0 - np.clip((lum_long - thresh_low) / max(thresh_high - thresh_low, 1e-6), 0.0, 1.0)
            mid_mask = np.clip(1.0 - high_mask - low_mask, 0.0, 1.0)

            total = high_mask + mid_mask + low_mask + epsilon
            blended = (short_exp * (high_mask / total)[..., None] +
                       mid_exp * (mid_mask / total)[..., None] +
                       long_exp * (low_mask / total)[..., None])
        else:
            blended = (short_exp * highlight_mask[..., None] +
                       long_exp * shadow_mask[..., None])

    # =========================================================================
    # Step 3: Apply stretch based on method
    # =========================================================================
    D = 10.0 ** log_D

    if stretch_method == StretchMethod.VERALUX:
        # VeraLux: Stretch luminance only, preserve color ratios
        anchor = HDRCore.calculate_anchor(blended)
        blended_anchored = np.maximum(blended - anchor, 0.0)

        L_anchored = HDRCore.extract_luminance(blended_anchored, weights)
        L_stretched = HDRCore.hyperbolic_stretch(L_anchored, D, protect_b)

        # Extract and preserve color ratios
        L_safe = L_anchored + epsilon
        r_ratio = blended_anchored[..., 0] / L_safe
        g_ratio = blended_anchored[..., 1] / L_safe
        b_ratio = blended_anchored[..., 2] / L_safe

        # Dynamic color convergence for bright areas
        k = np.power(L_stretched, convergence_power)
        r_final = r_ratio * (1.0 - k) + 1.0 * k
        g_final = g_ratio * (1.0 - k) + 1.0 * k
        b_final = b_ratio * (1.0 - k) + 1.0 * k

        # Reconstruct
        result = np.zeros_like(blended)
        result[..., 0] = L_stretched * r_final
        result[..., 1] = L_stretched * g_final
        result[..., 2] = L_stretched * b_final

        # Color grip: 0-1 blends vector/scalar, >1 boosts saturation
        if color_grip < 1.0:
            # Blend with scalar stretch for softer result
            scalar_result = np.zeros_like(result)
            for c in range(3):
                scalar_result[..., c] = HDRCore.hyperbolic_stretch(
                    blended_anchored[..., c], D, protect_b
                )
            result = result * color_grip + scalar_result * (1.0 - color_grip)
        elif color_grip > 1.0:
            # Saturation boost to compensate for L/C blend washout
            L = HDRCore.extract_luminance(result, weights)
            result = L[..., None] + (result - L[..., None]) * color_grip
            result = np.clip(result, 0.0, 1.0)

    elif stretch_method == StretchMethod.PER_CHANNEL:
        # Per-channel: Each channel gets its own anchor and stretch
        result = np.zeros_like(blended)

        for c in range(3):
            channel = blended[..., c]
            anchor = HDRCore.calculate_channel_anchor(channel)
            ch_anchored = np.maximum(channel - anchor, 0.0)
            result[..., c] = HDRCore.hyperbolic_stretch(ch_anchored, D, protect_b)

        # Apply saturation adjustment
        if saturation != 1.0:
            L = HDRCore.extract_luminance(result, weights)
            result = L[..., None] + (result - L[..., None]) * saturation
            result = np.clip(result, 0.0, 1.0)

    elif stretch_method == StretchMethod.LINKED:
        # Linked: Same anchor and parameters for all channels
        anchor = HDRCore.calculate_anchor(blended)
        blended_anchored = np.maximum(blended - anchor, 0.0)

        result = np.zeros_like(blended)
        for c in range(3):
            result[..., c] = HDRCore.hyperbolic_stretch(blended_anchored[..., c], D, protect_b)

        # Apply saturation adjustment
        if saturation != 1.0:
            L = HDRCore.extract_luminance(result, weights)
            result = L[..., None] + (result - L[..., None]) * saturation
            result = np.clip(result, 0.0, 1.0)

    else:
        result = blended

    # =========================================================================
    # Step 4: Background clipping (optional)
    # =========================================================================
    if clip_background:
        L_result = HDRCore.extract_luminance(result, weights)

        if clip_method == 0:
            # Target black level: find low percentile and shift to target
            current_black = float(np.percentile(L_result, 1.0))
            if current_black > target_black:
                # Shift down so current black becomes target black
                shift = current_black - target_black
                result = result - shift
                result = np.maximum(result, 0.0)
        else:
            # Statistical clipping: median - N*sigma
            median_L = float(np.median(L_result))
            std_L = float(np.std(L_result))
            clip_point = median_L - clip_sigma * std_L

            if clip_point > target_black:
                # Shift so clip_point becomes target_black
                shift = clip_point - target_black
                result = result - shift
                result = np.maximum(result, 0.0)

        result = np.clip(result, 0.0, 1.0).astype(np.float32)

    # =========================================================================
    # Step 5: Output adjustments
    # =========================================================================
    if output_boost != 1.0:
        result = result * output_boost

    # Safety pedestal
    result = result * (1.0 - 0.005) + 0.005

    return np.clip(result, 0.0, 1.0).astype(np.float32)


# =============================================================================
# Main Dialog
# =============================================================================

class HDRBlenderDialog(QDialog):
    """Dialog for HDR blending with multiple stretch methods."""

    def __init__(self, ctx):
        super().__init__(parent=ctx.app)
        self.ctx = ctx
        self.setWindowTitle(SCRIPT_NAME)
        self.resize(1400, 850)
        self._title_to_doc = {}
        self._preview_short = None
        self._preview_long = None
        self._preview_mid = None
        self._pix = None
        self._zoom = 1.0
        self._panning = False
        self._pan_start = QPointF()
        self._pan_hval = 0
        self._pan_vval = 0
        self._build_ui()
        self._populate_views()
        self._update_method_visibility()
        self._connect_signals()
        self._update_preview()

    def _build_ui(self):
        root = QVBoxLayout(self)
        main = QHBoxLayout()
        root.addLayout(main)

        # === Left Panel: Image Selection ===
        img_box = QGroupBox("Exposure Selection")
        img_layout = QVBoxLayout(img_box)

        img_layout.addWidget(QLabel("Short Exposure (highlights):"))
        self.combo_short = QComboBox()
        self.combo_short.setToolTip("Short exposure to capture bright regions")
        img_layout.addWidget(self.combo_short)

        img_layout.addWidget(QLabel("Long Exposure (faint details):"))
        self.combo_long = QComboBox()
        self.combo_long.setToolTip("Long exposure for faint outer regions")
        img_layout.addWidget(self.combo_long)

        img_layout.addWidget(QLabel("Medium Exposure (optional):"))
        self.combo_mid = QComboBox()
        self.combo_mid.setToolTip("Optional medium exposure for 3-zone blend")
        img_layout.addWidget(self.combo_mid)

        img_layout.addStretch()
        main.addWidget(img_box)

        # === Middle Panel: Controls (in scroll area) ===
        ctrl_scroll = QScrollArea()
        ctrl_scroll.setWidgetResizable(True)
        ctrl_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        ctrl_scroll.setMinimumWidth(420)
        ctrl_scroll.setMaximumWidth(500)

        ctrl_box = QGroupBox("Blend & Stretch Controls")
        ctrl_layout = QVBoxLayout(ctrl_box)

        # Stretch Method Selection
        method_group = QGroupBox("Stretch Method")
        method_layout = QVBoxLayout(method_group)

        self.combo_stretch_method = QComboBox()
        for i, name in enumerate(STRETCH_METHOD_NAMES):
            self.combo_stretch_method.addItem(name)
            self.combo_stretch_method.setItemData(i, STRETCH_METHOD_TOOLTIPS[i], Qt.ItemDataRole.ToolTipRole)
        self.combo_stretch_method.setCurrentIndex(0)
        method_layout.addWidget(self.combo_stretch_method)

        self.lbl_method_desc = QLabel("")
        self.lbl_method_desc.setStyleSheet("color: #666666; font-size: 9pt;")
        self.lbl_method_desc.setWordWrap(True)
        method_layout.addWidget(self.lbl_method_desc)

        ctrl_layout.addWidget(method_group)

        # Sensor Profile (for VeraLux and Linked)
        sensor_group = QGroupBox("Sensor Profile")
        sensor_layout = QVBoxLayout(sensor_group)

        self.combo_profile = QComboBox()
        for name in SENSOR_PROFILES.keys():
            self.combo_profile.addItem(name)
        self.combo_profile.setCurrentText(DEFAULT_PROFILE)
        self.combo_profile.setToolTip(
            "Sensor-specific luminance weights (R, G, B).\n"
            "Used for luminance extraction in VeraLux stretch,\n"
            "luminosity mask creation, and background clipping."
        )
        sensor_layout.addWidget(self.combo_profile)

        self.lbl_profile_info = QLabel("")
        self.lbl_profile_info.setStyleSheet("color: #666666; font-size: 9pt;")
        sensor_layout.addWidget(self.lbl_profile_info)

        ctrl_layout.addWidget(sensor_group)

        # Stretch Parameters
        stretch_group = QGroupBox("Arcsinh Stretch")
        stretch_layout = QGridLayout(stretch_group)

        stretch_layout.addWidget(QLabel("Target Background:"), 0, 0)
        self.spin_target_bg = QDoubleSpinBox()
        self.spin_target_bg.setRange(0.01, 0.50)
        self.spin_target_bg.setValue(0.02)
        self.spin_target_bg.setSingleStep(0.01)
        self.spin_target_bg.setDecimals(3)
        self.spin_target_bg.setToolTip(
            "Target median brightness for Auto Log D solver.\n"
            "Higher values = brighter background after stretch."
        )
        stretch_layout.addWidget(self.spin_target_bg, 0, 1)

        self.btn_auto_solve = QPushButton("Auto Log D")
        self.btn_auto_solve.setToolTip(
            "Automatically calculate optimal Log D value\n"
            "to achieve the target background brightness."
        )
        self.btn_auto_solve.clicked.connect(self._auto_solve)
        stretch_layout.addWidget(self.btn_auto_solve, 0, 2)

        stretch_layout.addWidget(QLabel("Log D (Intensity):"), 1, 0)
        self.spin_log_d = QDoubleSpinBox()
        self.spin_log_d.setRange(0.0, 7.0)
        self.spin_log_d.setValue(2.5)
        self.spin_log_d.setSingleStep(0.1)
        self.spin_log_d.setDecimals(2)
        self.spin_log_d.setToolTip(
            "Stretch intensity (log10 scale).\n"
            "Higher values = stronger stretch, brighter faint details.\n"
            "Typical range: 1.5-4.0"
        )
        stretch_layout.addWidget(self.spin_log_d, 1, 1)

        self.slide_log_d = QSlider(Qt.Orientation.Horizontal)
        self.slide_log_d.setRange(0, 700)
        self.slide_log_d.setValue(250)
        self.slide_log_d.setToolTip("Stretch intensity slider")
        stretch_layout.addWidget(self.slide_log_d, 1, 2)

        stretch_layout.addWidget(QLabel("Protect b:"), 2, 0)
        self.spin_protect_b = QDoubleSpinBox()
        self.spin_protect_b.setRange(0.1, 15.0)
        self.spin_protect_b.setValue(6.0)
        self.spin_protect_b.setSingleStep(0.5)
        self.spin_protect_b.setDecimals(1)
        self.spin_protect_b.setToolTip(
            "Highlight protection parameter.\n"
            "Higher values = gentler highlight rolloff.\n"
            "Lower values = more aggressive stretch in highlights."
        )
        stretch_layout.addWidget(self.spin_protect_b, 2, 1, 1, 2)

        ctrl_layout.addWidget(stretch_group)

        # Background Clipping
        bg_group = QGroupBox("Background Clipping")
        bg_layout = QGridLayout(bg_group)

        self.chk_clip_background = QCheckBox("Enable background clipping")
        self.chk_clip_background.setChecked(False)
        self.chk_clip_background.setToolTip("Shift background down to target black level")
        bg_layout.addWidget(self.chk_clip_background, 0, 0, 1, 3)

        bg_layout.addWidget(QLabel("Method:"), 1, 0)
        self.combo_clip_method = QComboBox()
        self.combo_clip_method.addItem("Target Black Level")
        self.combo_clip_method.addItem("Statistical (N×σ)")
        self.combo_clip_method.setToolTip(
            "Target Black Level: Shift 1st percentile to target value\n"
            "Statistical: Shift (median - N×σ) to target value"
        )
        bg_layout.addWidget(self.combo_clip_method, 1, 1, 1, 2)

        bg_layout.addWidget(QLabel("Target Black:"), 2, 0)
        self.spin_target_black = QDoubleSpinBox()
        self.spin_target_black.setRange(0.0, 0.1)
        self.spin_target_black.setValue(0.01)
        self.spin_target_black.setSingleStep(0.005)
        self.spin_target_black.setDecimals(3)
        self.spin_target_black.setToolTip("Target value for black point (0.01 = nearly black)")
        bg_layout.addWidget(self.spin_target_black, 2, 1, 1, 2)

        bg_layout.addWidget(QLabel("Sigma (N):"), 3, 0)
        self.spin_clip_sigma = QDoubleSpinBox()
        self.spin_clip_sigma.setRange(1.0, 5.0)
        self.spin_clip_sigma.setValue(2.5)
        self.spin_clip_sigma.setSingleStep(0.1)
        self.spin_clip_sigma.setDecimals(1)
        self.spin_clip_sigma.setToolTip("Sigma multiplier for statistical clipping (median - N×σ)")
        bg_layout.addWidget(self.spin_clip_sigma, 3, 1, 1, 2)

        ctrl_layout.addWidget(bg_group)

        # VeraLux Color Options (only visible for VeraLux method)
        self.veralux_group = QGroupBox("VeraLux Color Options")
        veralux_layout = QGridLayout(self.veralux_group)

        veralux_layout.addWidget(QLabel("Star Core Recovery:"), 0, 0)
        self.spin_convergence = QDoubleSpinBox()
        self.spin_convergence.setRange(1.0, 10.0)
        self.spin_convergence.setValue(3.5)
        self.spin_convergence.setSingleStep(0.5)
        self.spin_convergence.setToolTip("How fast star cores converge to white")
        veralux_layout.addWidget(self.spin_convergence, 0, 1)

        veralux_layout.addWidget(QLabel("Color Grip:"), 1, 0)
        self.spin_color_grip = QDoubleSpinBox()
        self.spin_color_grip.setRange(0.0, 2.0)
        self.spin_color_grip.setValue(1.0)
        self.spin_color_grip.setSingleStep(0.05)
        self.spin_color_grip.setDecimals(2)
        self.spin_color_grip.setToolTip(
            "0.0 = Soft (scalar stretch)\n"
            "1.0 = Vivid (vector preservation)\n"
            "> 1.0 = Saturation boost (compensate for L/C blend washout)"
        )
        veralux_layout.addWidget(self.spin_color_grip, 1, 1)

        self.slide_color_grip = QSlider(Qt.Orientation.Horizontal)
        self.slide_color_grip.setRange(0, 200)
        self.slide_color_grip.setValue(100)
        self.slide_color_grip.setToolTip("Color grip slider (0-2)")
        veralux_layout.addWidget(self.slide_color_grip, 1, 2)

        ctrl_layout.addWidget(self.veralux_group)

        # Non-VeraLux Color Options (only visible for Per-Channel and Linked)
        self.nonveralux_group = QGroupBox("Color Options")
        nonveralux_layout = QGridLayout(self.nonveralux_group)

        nonveralux_layout.addWidget(QLabel("Saturation:"), 0, 0)
        self.spin_saturation = QDoubleSpinBox()
        self.spin_saturation.setRange(0.0, 3.0)
        self.spin_saturation.setValue(1.0)
        self.spin_saturation.setSingleStep(0.05)
        self.spin_saturation.setDecimals(2)
        self.spin_saturation.setToolTip(
            "Saturation adjustment (1.0 = no change).\n"
            "< 1.0 reduces saturation, > 1.0 boosts saturation.\n\n"
            "Useful to compensate for color washout when using\n"
            "Luminance-chrominance blend with Mask-weighted ratios."
        )
        nonveralux_layout.addWidget(self.spin_saturation, 0, 1)

        self.slide_saturation = QSlider(Qt.Orientation.Horizontal)
        self.slide_saturation.setRange(0, 300)
        self.slide_saturation.setValue(100)
        self.slide_saturation.setToolTip("Saturation slider (0-3)")
        nonveralux_layout.addWidget(self.slide_saturation, 0, 2)

        ctrl_layout.addWidget(self.nonveralux_group)

        # Blend Parameters
        blend_group = QGroupBox("Luminosity Mask & Blend")
        blend_layout = QGridLayout(blend_group)

        blend_layout.addWidget(QLabel("Threshold:"), 0, 0)
        self.spin_threshold = QDoubleSpinBox()
        self.spin_threshold.setRange(0.01, 0.99)
        self.spin_threshold.setValue(0.05)
        self.spin_threshold.setSingleStep(0.01)
        self.spin_threshold.setDecimals(3)
        self.spin_threshold.setToolTip(
            "Luminosity mask threshold.\n"
            "Below threshold: use long exposure (faint details).\n"
            "Above threshold: use short exposure (bright regions)."
        )
        blend_layout.addWidget(self.spin_threshold, 0, 1)

        blend_layout.addWidget(QLabel("Feather:"), 1, 0)
        self.spin_feather = QDoubleSpinBox()
        self.spin_feather.setRange(0.0, 0.5)
        self.spin_feather.setValue(0.10)
        self.spin_feather.setSingleStep(0.01)
        self.spin_feather.setDecimals(3)
        self.spin_feather.setToolTip(
            "Mask feather/transition width.\n"
            "Higher values = smoother blend between exposures.\n"
            "0 = hard transition at threshold."
        )
        blend_layout.addWidget(self.spin_feather, 1, 1)

        self.chk_protect_highlights = QCheckBox("Protect clipped highlights")
        self.chk_protect_highlights.setChecked(True)
        self.chk_protect_highlights.setToolTip(
            "Force short exposure where long exposure is clipped (>98%).\n"
            "Prevents blown-out star cores and bright regions."
        )
        blend_layout.addWidget(self.chk_protect_highlights, 2, 0, 1, 2)

        # VeraLux L/C blend option
        self.chk_lum_chrom_blend = QCheckBox("Luminance-chrominance blend")
        self.chk_lum_chrom_blend.setChecked(False)
        self.chk_lum_chrom_blend.setToolTip(
            "VeraLux-style blend: separately blend luminance\n"
            "and color ratios, then reconstruct.\n\n"
            "Unchecked: Standard RGB channel blend."
        )
        blend_layout.addWidget(self.chk_lum_chrom_blend, 3, 0, 1, 2)


        ctrl_layout.addWidget(blend_group)

        # Output
        output_group = QGroupBox("Output")
        output_layout = QGridLayout(output_group)

        output_layout.addWidget(QLabel("Boost:"), 0, 0)
        self.spin_output_boost = QDoubleSpinBox()
        self.spin_output_boost.setRange(0.5, 2.0)
        self.spin_output_boost.setValue(1.0)
        self.spin_output_boost.setSingleStep(0.05)
        self.spin_output_boost.setToolTip(
            "Final brightness multiplier.\n"
            "1.0 = no change, >1.0 = brighter output."
        )
        output_layout.addWidget(self.spin_output_boost, 0, 1)

        ctrl_layout.addWidget(output_group)

        ctrl_layout.addStretch()
        ctrl_scroll.setWidget(ctrl_box)
        main.addWidget(ctrl_scroll)

        # === Right Panel: Preview ===
        preview_box = QGroupBox("Preview")
        preview_layout = QVBoxLayout(preview_box)

        preview_ctrl = QHBoxLayout()
        self.chk_auto_preview = QCheckBox("Auto-update")
        self.chk_auto_preview.setChecked(True)
        preview_ctrl.addWidget(self.chk_auto_preview)

        btn_refresh = QPushButton("Refresh")
        btn_refresh.clicked.connect(self._update_preview)
        preview_ctrl.addWidget(btn_refresh)

        btn_fit = QPushButton("Fit")
        btn_fit.clicked.connect(self._fit_preview)
        preview_ctrl.addWidget(btn_fit)

        btn_100 = QPushButton("100%")
        btn_100.clicked.connect(self._zoom_100)
        preview_ctrl.addWidget(btn_100)

        preview_ctrl.addStretch()
        preview_layout.addLayout(preview_ctrl)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setMinimumSize(500, 400)
        self.preview_label = ImageLabel()
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scroll.setWidget(self.preview_label)
        self.scroll.viewport().installEventFilter(self)
        preview_layout.addWidget(self.scroll)

        self.status_label = QLabel("")
        preview_layout.addWidget(self.status_label)

        main.addWidget(preview_box, 1)

        # === Bottom Buttons ===
        btn_layout = QHBoxLayout()

        btn_save = QPushButton("Save Settings...")
        btn_save.clicked.connect(self._save_settings)
        btn_layout.addWidget(btn_save)

        btn_load = QPushButton("Load Settings...")
        btn_load.clicked.connect(self._load_settings)
        btn_layout.addWidget(btn_load)

        btn_reset = QPushButton("Reset")
        btn_reset.clicked.connect(self._reset)
        btn_layout.addWidget(btn_reset)

        btn_layout.addStretch()

        btn_apply = QPushButton("Apply to Short Exp")
        btn_apply.clicked.connect(self._apply_to_existing)
        btn_layout.addWidget(btn_apply)

        btn_new = QPushButton("Create New Image")
        btn_new.clicked.connect(self._create_new)
        btn_layout.addWidget(btn_new)

        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.close)
        btn_layout.addWidget(btn_close)

        root.addLayout(btn_layout)

    def _connect_signals(self):
        # Image selection
        self.combo_short.currentIndexChanged.connect(self._on_image_changed)
        self.combo_long.currentIndexChanged.connect(self._on_image_changed)
        self.combo_mid.currentIndexChanged.connect(self._on_image_changed)

        # Stretch method
        self.combo_stretch_method.currentIndexChanged.connect(self._update_method_visibility)
        self.combo_stretch_method.currentIndexChanged.connect(self._schedule_preview)

        # Profile
        self.combo_profile.currentTextChanged.connect(self._update_profile_info)
        self.combo_profile.currentTextChanged.connect(self._schedule_preview)

        # Slider sync
        self.slide_log_d.valueChanged.connect(lambda v: self.spin_log_d.setValue(v / 100.0))
        self.spin_log_d.valueChanged.connect(lambda v: self.slide_log_d.setValue(int(v * 100)))
        self.slide_color_grip.valueChanged.connect(lambda v: self.spin_color_grip.setValue(v / 100.0))
        self.spin_color_grip.valueChanged.connect(lambda v: self.slide_color_grip.setValue(int(v * 100)))
        self.slide_saturation.valueChanged.connect(lambda v: self.spin_saturation.setValue(v / 100.0))
        self.spin_saturation.valueChanged.connect(lambda v: self.slide_saturation.setValue(int(v * 100)))

        # All spinboxes
        spinboxes = [
            self.spin_log_d, self.spin_protect_b,
            self.spin_target_black, self.spin_clip_sigma,
            self.spin_convergence, self.spin_color_grip,
            self.spin_saturation,
            self.spin_threshold, self.spin_feather,
            self.spin_output_boost,
        ]
        for spin in spinboxes:
            spin.valueChanged.connect(self._schedule_preview)

        # Checkboxes
        self.chk_clip_background.toggled.connect(self._schedule_preview)
        self.chk_protect_highlights.toggled.connect(self._schedule_preview)
        self.chk_lum_chrom_blend.toggled.connect(self._schedule_preview)

        # Combo boxes
        self.combo_clip_method.currentIndexChanged.connect(self._schedule_preview)

        # Initial info
        self._update_profile_info(DEFAULT_PROFILE)
        self._update_method_description()

    def _schedule_preview(self):
        if self.chk_auto_preview.isChecked():
            self._update_preview()

    def _update_method_visibility(self):
        method = self.combo_stretch_method.currentIndex()
        # VeraLux options only visible for VeraLux method
        self.veralux_group.setVisible(method == StretchMethod.VERALUX)
        # Saturation control only for non-VeraLux (VeraLux uses extended color grip)
        self.nonveralux_group.setVisible(method != StretchMethod.VERALUX)
        self._update_method_description()

    def _update_method_description(self):
        idx = self.combo_stretch_method.currentIndex()
        if 0 <= idx < len(STRETCH_METHOD_TOOLTIPS):
            self.lbl_method_desc.setText(STRETCH_METHOD_TOOLTIPS[idx])

    def _update_profile_info(self, profile_name):
        if profile_name in SENSOR_PROFILES:
            profile = SENSOR_PROFILES[profile_name]
            r, g, b = profile['weights']
            self.lbl_profile_info.setText(
                f"{profile['description']} (R:{r:.3f} G:{g:.3f} B:{b:.3f})"
            )

    def _populate_views(self):
        for combo in [self.combo_short, self.combo_long, self.combo_mid]:
            combo.blockSignals(True)
            combo.clear()
        self._title_to_doc.clear()

        self.combo_mid.addItem(_NONE_PLACEHOLDER)

        try:
            views = self.ctx.list_image_views()
        except Exception:
            views = []

        for title, doc in views:
            key = title
            if key in self._title_to_doc:
                n = 2
                while f"{title} ({n})" in self._title_to_doc:
                    n += 1
                key = f"{title} ({n})"
            self._title_to_doc[key] = doc
            self.combo_short.addItem(key)
            self.combo_long.addItem(key)
            self.combo_mid.addItem(key)

        for combo in [self.combo_short, self.combo_long, self.combo_mid]:
            combo.blockSignals(False)

        if self.combo_short.count() > 1:
            self.combo_long.setCurrentIndex(1)

    def _resolve_doc(self, combo):
        title = combo.currentText()
        if title == _NONE_PLACEHOLDER:
            return None, None
        doc = self._title_to_doc.get(title)
        return title, doc

    def _on_image_changed(self):
        self._preview_short = None
        self._preview_long = None
        self._preview_mid = None
        self._schedule_preview()

    def _load_preview_images(self):
        _, short_doc = self._resolve_doc(self.combo_short)
        _, long_doc = self._resolve_doc(self.combo_long)
        _, mid_doc = self._resolve_doc(self.combo_mid)

        if short_doc is not None and self._preview_short is None:
            img = HDRCore.normalize_input(short_doc.image)
            img = HDRCore.ensure_hwc(img)
            self._preview_short = _downsample_for_preview(img)

        if long_doc is not None and self._preview_long is None:
            img = HDRCore.normalize_input(long_doc.image)
            img = HDRCore.ensure_hwc(img)
            self._preview_long = _downsample_for_preview(img)

        if mid_doc is not None and self._preview_mid is None:
            img = HDRCore.normalize_input(mid_doc.image)
            img = HDRCore.ensure_hwc(img)
            self._preview_mid = _downsample_for_preview(img)

    def _get_weights(self):
        profile = self.combo_profile.currentText()
        if profile in SENSOR_PROFILES:
            return SENSOR_PROFILES[profile]['weights']
        return SENSOR_PROFILES[DEFAULT_PROFILE]['weights']

    def _update_preview(self):
        try:
            self._load_preview_images()

            if self._preview_short is None or self._preview_long is None:
                self.status_label.setText("Select short and long exposures")
                return

            result = hdr_blend(
                self._preview_short, self._preview_long, self._preview_mid,
                weights=self._get_weights(),
                stretch_method=self.combo_stretch_method.currentIndex(),
                log_D=self.spin_log_d.value(),
                protect_b=self.spin_protect_b.value(),
                clip_background=self.chk_clip_background.isChecked(),
                clip_method=self.combo_clip_method.currentIndex(),
                target_black=self.spin_target_black.value(),
                clip_sigma=self.spin_clip_sigma.value(),
                convergence_power=self.spin_convergence.value(),
                color_grip=self.spin_color_grip.value(),
                blend_threshold=self.spin_threshold.value(),
                blend_feather=self.spin_feather.value(),
                protect_highlights=self.chk_protect_highlights.isChecked(),
                lum_chrom_blend=self.chk_lum_chrom_blend.isChecked(),
                saturation=self.spin_saturation.value(),
                output_boost=self.spin_output_boost.value(),
            )

            qimg = _float_to_qimage_rgb8(result)
            scaled_w = int(qimg.width() * self._zoom)
            scaled_h = int(qimg.height() * self._zoom)
            pix = QPixmap.fromImage(qimg).scaled(
                scaled_w, scaled_h,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self._pix = pix
            self.preview_label.setPixmap(pix)

            method_name = STRETCH_METHOD_NAMES[self.combo_stretch_method.currentIndex()].split()[0]
            self.status_label.setText(
                f"{result.shape[1]}x{result.shape[0]} | "
                f"{method_name} | Log D: {self.spin_log_d.value():.2f}"
            )

        except Exception as e:
            self.status_label.setText(f"Error: {e}")
            import traceback
            traceback.print_exc()

    def _fit_preview(self):
        if self._pix is None:
            return
        vp = self.scroll.viewport()
        scale_w = vp.width() / max(1, self._pix.width() / self._zoom)
        scale_h = vp.height() / max(1, self._pix.height() / self._zoom)
        self._zoom = min(scale_w, scale_h) * 0.95
        self._update_preview()

    def _zoom_100(self):
        self._zoom = 1.0
        self._update_preview()

    def _auto_solve(self):
        """Auto-calculate optimal Log D for each stretch method."""
        try:
            self._load_preview_images()
            if self._preview_long is None:
                QMessageBox.warning(self, "Error", "Load a long exposure first")
                return

            weights = self._get_weights()
            target = self.spin_target_bg.value()
            b_val = self.spin_protect_b.value()

            img = self._preview_long
            method = self.combo_stretch_method.currentIndex()

            if method == StretchMethod.VERALUX:
                # VeraLux: Compute based on luminance (this works well)
                anchor = HDRCore.calculate_anchor(img)
                img_anchored = np.maximum(img - anchor, 0.0)
                L_anchored = HDRCore.extract_luminance(img_anchored, weights)
                valid = L_anchored[L_anchored > 1e-7].flatten()
                if len(valid) > 100000:
                    valid = valid[np.random.choice(len(valid), 100000, replace=False)]
                log_d = HDRCore.solve_log_d(valid, target, b_val) if len(valid) > 0 else 2.0
                method_note = "luminance-based"

            elif method == StretchMethod.PER_CHANNEL:
                # Per-Channel: Each channel gets its own anchor, so compute for the
                # BRIGHTEST channel to avoid over-stretching. Use lower target since
                # background normalization will adjust afterward.
                adjusted_target = target * 0.35  # More conservative target
                log_ds = []
                for c in range(3):
                    ch = img[..., c]
                    ch_anchor = HDRCore.calculate_channel_anchor(ch)
                    ch_anchored = np.maximum(ch - ch_anchor, 0.0)
                    valid = ch_anchored[ch_anchored > 1e-7].flatten()
                    if len(valid) > 50000:
                        valid = valid[np.random.choice(len(valid), 50000, replace=False)]
                    if len(valid) > 0:
                        log_ds.append(HDRCore.solve_log_d(valid, adjusted_target, b_val))
                # Use the MINIMUM Log D to avoid over-stretching any channel
                log_d = min(log_ds) if log_ds else 2.0
                method_note = f"per-channel (conservative, adj target={adjusted_target:.2f})"

            elif method == StretchMethod.LINKED:
                # Linked: All channels use same anchor and D, so compute based on
                # the actual channel values (not luminance). Use median of all pixels.
                anchor = HDRCore.calculate_anchor(img)
                img_anchored = np.maximum(img - anchor, 0.0)
                # Flatten all channels together - this is what actually gets stretched
                all_channels = img_anchored.flatten()
                valid = all_channels[all_channels > 1e-7]
                if len(valid) > 100000:
                    valid = valid[np.random.choice(len(valid), 100000, replace=False)]
                # Use much lower target since stretching RGB directly amplifies more
                adjusted_target = target * 0.4
                log_d = HDRCore.solve_log_d(valid, adjusted_target, b_val) if len(valid) > 0 else 2.0
                method_note = f"all-channel (adj target={adjusted_target:.2f})"

            else:
                log_d = 2.0
                method_note = "default"

            self.spin_log_d.setValue(log_d)

            QMessageBox.information(
                self, "Auto-Solve",
                f"Optimal Log D: {log_d:.2f}\n"
                f"Method: {method_note}\n"
                f"Original target: {target:.2f}"
            )

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def eventFilter(self, obj, ev):
        if obj == self.scroll.viewport():
            if ev.type() == QEvent.Type.Wheel:
                delta = ev.angleDelta().y()
                factor = 1.15 if delta > 0 else 1.0 / 1.15
                self._zoom = max(0.1, min(10.0, self._zoom * factor))
                self._update_preview()
                return True
            elif ev.type() == QEvent.Type.MouseButtonPress:
                if ev.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton):
                    self._panning = True
                    self._pan_start = ev.position()
                    self._pan_hval = self.scroll.horizontalScrollBar().value()
                    self._pan_vval = self.scroll.verticalScrollBar().value()
                    self.scroll.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
                    return True
            elif ev.type() == QEvent.Type.MouseMove:
                if self._panning:
                    delta = ev.position() - self._pan_start
                    self.scroll.horizontalScrollBar().setValue(int(self._pan_hval - delta.x()))
                    self.scroll.verticalScrollBar().setValue(int(self._pan_vval - delta.y()))
                    return True
            elif ev.type() == QEvent.Type.MouseButtonRelease:
                if self._panning:
                    self._panning = False
                    self.scroll.viewport().setCursor(Qt.CursorShape.ArrowCursor)
                    return True
        return super().eventFilter(obj, ev)

    def _reset(self):
        self.combo_stretch_method.setCurrentIndex(0)
        self.combo_profile.setCurrentText(DEFAULT_PROFILE)
        self.spin_target_bg.setValue(0.02)
        self.spin_log_d.setValue(2.5)
        self.spin_protect_b.setValue(6.0)
        self.chk_clip_background.setChecked(False)
        self.combo_clip_method.setCurrentIndex(0)
        self.spin_target_black.setValue(0.01)
        self.spin_clip_sigma.setValue(2.5)
        self.spin_convergence.setValue(3.5)
        self.spin_color_grip.setValue(1.0)
        self.spin_saturation.setValue(1.0)
        self.spin_threshold.setValue(0.05)
        self.spin_feather.setValue(0.10)
        self.chk_protect_highlights.setChecked(True)
        self.chk_lum_chrom_blend.setChecked(False)
        self.spin_output_boost.setValue(1.0)
        self._update_method_visibility()
        self._update_preview()

    def _get_settings_dict(self):
        return {
            "_comment": "HDR Blender Settings",
            "_saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "version": SCRIPT_VERSION,
            "stretch_method": self.combo_stretch_method.currentIndex(),
            "sensor_profile": self.combo_profile.currentText(),
            "stretch": {
                "target_bg": self.spin_target_bg.value(),
                "log_D": self.spin_log_d.value(),
                "protect_b": self.spin_protect_b.value(),
            },
            "background_clip": {
                "enabled": self.chk_clip_background.isChecked(),
                "method": self.combo_clip_method.currentIndex(),
                "target_black": self.spin_target_black.value(),
                "clip_sigma": self.spin_clip_sigma.value(),
            },
            "veralux": {
                "convergence": self.spin_convergence.value(),
                "color_grip": self.spin_color_grip.value(),
            },
            "color": {
                "saturation": self.spin_saturation.value(),
            },
            "blend": {
                "threshold": self.spin_threshold.value(),
                "feather": self.spin_feather.value(),
                "protect_highlights": self.chk_protect_highlights.isChecked(),
                "lum_chrom_blend": self.chk_lum_chrom_blend.isChecked(),
            },
            "output": {
                "boost": self.spin_output_boost.value(),
            },
        }

    def _apply_settings_dict(self, settings):
        widgets = [
            self.combo_stretch_method, self.combo_profile,
            self.spin_target_bg, self.spin_log_d, self.spin_protect_b,
            self.chk_clip_background, self.combo_clip_method,
            self.spin_target_black, self.spin_clip_sigma,
            self.spin_convergence, self.spin_color_grip,
            self.spin_saturation,
            self.spin_threshold, self.spin_feather,
            self.spin_output_boost,
            self.chk_protect_highlights, self.chk_lum_chrom_blend,
        ]
        for w in widgets:
            w.blockSignals(True)

        try:
            if "stretch_method" in settings:
                self.combo_stretch_method.setCurrentIndex(settings["stretch_method"])
            if "sensor_profile" in settings:
                self.combo_profile.setCurrentText(settings["sensor_profile"])

            if "stretch" in settings:
                st = settings["stretch"]
                self.spin_target_bg.setValue(st.get("target_bg", 0.02))
                self.spin_log_d.setValue(st.get("log_D", 2.5))
                self.spin_protect_b.setValue(st.get("protect_b", 6.0))

            if "background_clip" in settings:
                bc = settings["background_clip"]
                self.chk_clip_background.setChecked(bc.get("enabled", False))
                self.combo_clip_method.setCurrentIndex(bc.get("method", 0))
                self.spin_target_black.setValue(bc.get("target_black", 0.01))
                self.spin_clip_sigma.setValue(bc.get("clip_sigma", 2.5))

            if "veralux" in settings:
                vl = settings["veralux"]
                self.spin_convergence.setValue(vl.get("convergence", 3.5))
                self.spin_color_grip.setValue(vl.get("color_grip", 1.0))

            if "color" in settings:
                col = settings["color"]
                self.spin_saturation.setValue(col.get("saturation", 1.0))

            if "blend" in settings:
                bl = settings["blend"]
                self.spin_threshold.setValue(bl.get("threshold", 0.05))
                self.spin_feather.setValue(bl.get("feather", 0.10))
                self.chk_protect_highlights.setChecked(bl.get("protect_highlights", True))
                self.chk_lum_chrom_blend.setChecked(bl.get("lum_chrom_blend", False))

            if "output" in settings:
                out = settings["output"]
                self.spin_output_boost.setValue(out.get("boost", 1.0))

        finally:
            for w in widgets:
                w.blockSignals(False)

        self._update_method_visibility()
        self._update_preview()

    def _save_settings(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Settings", "hdr_blender_settings.json", "JSON (*.json)"
        )
        if not path:
            return
        try:
            with open(path, 'w') as f:
                json.dump(self._get_settings_dict(), f, indent=2)
            QMessageBox.information(self, "Saved", f"Settings saved to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _load_settings(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Settings", "", "JSON (*.json)"
        )
        if not path:
            return
        try:
            with open(path, 'r') as f:
                settings = json.load(f)
            self._apply_settings_dict(settings)
            QMessageBox.information(self, "Loaded", f"Settings loaded from:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _gather_inputs(self):
        short_key, short_doc = self._resolve_doc(self.combo_short)
        long_key, long_doc = self._resolve_doc(self.combo_long)
        _, mid_doc = self._resolve_doc(self.combo_mid)

        if short_doc is None:
            raise RuntimeError("Select a short exposure image")
        if long_doc is None:
            raise RuntimeError("Select a long exposure image")

        short_img = HDRCore.normalize_input(short_doc.image)
        short_img = HDRCore.ensure_hwc(short_img)
        long_img = HDRCore.normalize_input(long_doc.image)
        long_img = HDRCore.ensure_hwc(long_img)
        mid_img = None
        if mid_doc is not None:
            mid_img = HDRCore.normalize_input(mid_doc.image)
            mid_img = HDRCore.ensure_hwc(mid_img)

        return short_img, long_img, mid_img, short_doc, short_key

    def _do_blend(self, short_img, long_img, mid_img):
        return hdr_blend(
            short_img, long_img, mid_img,
            weights=self._get_weights(),
            stretch_method=self.combo_stretch_method.currentIndex(),
            log_D=self.spin_log_d.value(),
            protect_b=self.spin_protect_b.value(),
            clip_background=self.chk_clip_background.isChecked(),
            clip_method=self.combo_clip_method.currentIndex(),
            target_black=self.spin_target_black.value(),
            clip_sigma=self.spin_clip_sigma.value(),
            convergence_power=self.spin_convergence.value(),
            color_grip=self.spin_color_grip.value(),
            blend_threshold=self.spin_threshold.value(),
            blend_feather=self.spin_feather.value(),
            protect_highlights=self.chk_protect_highlights.isChecked(),
            lum_chrom_blend=self.chk_lum_chrom_blend.isChecked(),
            saturation=self.spin_saturation.value(),
            output_boost=self.spin_output_boost.value(),
        )

    def _apply_to_existing(self):
        try:
            short_img, long_img, mid_img, short_doc, _ = self._gather_inputs()
            result = self._do_blend(short_img, long_img, mid_img)
            short_doc.image[...] = result.astype(short_doc.image.dtype)
            QMessageBox.information(self, "Done", "HDR blend applied")
            self._preview_short = None
            self._update_preview()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _create_new(self):
        try:
            short_img, long_img, mid_img, _, short_key = self._gather_inputs()
            result = self._do_blend(short_img, long_img, mid_img)
            method_name = ["VeraLux", "PerCh", "Linked"][self.combo_stretch_method.currentIndex()]
            name = f"{short_key}_HDR_{method_name}"
            self.ctx.open_new_document(result, name=name)
            QMessageBox.information(self, "Done", f"Created: {name}")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))


# =============================================================================
# Entry Point
# =============================================================================

def run(ctx):
    """Entry point for SAS Pro."""
    dlg = HDRBlenderDialog(ctx)
    dlg.exec()
