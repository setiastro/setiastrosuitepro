"""
LRGB_NB_Combiner.py (SAS Pro) - Version 4.0
--------------------------------------------
Advanced LRGB + Narrowband combiner with multiple blend modes,
luminance processing, per-channel blackpoint control,
external star mask support, and post-processing curves.
"""

from __future__ import annotations
import numpy as np
import json
from datetime import datetime

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton,
    QMessageBox, QGroupBox, QGridLayout, QCheckBox, QDoubleSpinBox,
    QScrollArea, QFileDialog, QFrame
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


def _auto_stretch(img, low_pct=0.25, high_pct=99.75):
    a = np.asarray(img, dtype=np.float32)
    if a.size == 0:
        return a
    if a.ndim == 3 and a.shape[2] == 3:
        lum = 0.25 * a[..., 0] + 0.5 * a[..., 1] + 0.25 * a[..., 2]
    else:
        lum = a
    lo, hi = float(np.percentile(lum, low_pct)), float(np.percentile(lum, high_pct))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.clip(a, 0.0, 1.0)
    stretched = (a - lo) / max(hi - lo, 1e-6)
    return np.clip(stretched, 0.0, 1.0)


# Constants
SCRIPT_NAME = "LRGB + Narrowband Combiner"
SCRIPT_VERSION = "4.0"
_NONE_PLACEHOLDER = "<None>"


class CombinationMethod:
    HA_ENHANCE = 0
    HOO_RGB = 1
    SHO_RGB = 2
    LRGB_NB = 3
    FORAXX = 4
    HOO_SHO_MORPH = 5
    GOLDEN_PALETTE = 6
    CUSTOM_MAP = 7


METHOD_NAMES = [
    "Hα Enhancement (RGB + Hα)",
    "HOO + RGB (Bicolor with RGB Stars)",
    "SHO + RGB (Hubble Palette Blend)",
    "L-RGB-NB (Narrowband as Luminance)",
    "Foraxx Palette (Modified HOO)",
    "HOO↔SHO Morph (Palette Blend)",
    "Golden Palette (Ha+SII Warm)",
    "Custom Channel Mapping",
]

METHOD_TOOLTIPS = [
    # HA_ENHANCE
    "Enhances the red channel using Hα data.\n\n"
    "Formula: R = max(R_rgb, Hα × strength)\n\n"
    "Use for: Adding nebula detail to broadband RGB images.\n"
    "Requires: Hα",

    # HOO_RGB
    "Classic bicolor palette blended with RGB.\n\n"
    "Formula:\n"
    "  R = Hα × strength × (1-mix) + R_rgb × mix\n"
    "  G = OIII × strength × (1-mix) + G_rgb × mix\n"
    "  B = OIII × strength × (1-mix) + B_rgb × mix\n\n"
    "Use for: Emission nebulae with two-filter data.\n"
    "Requires: Hα, OIII",

    # SHO_RGB
    "Hubble Space Telescope palette blended with RGB.\n\n"
    "Formula:\n"
    "  R = SII × strength × (1-mix) + R_rgb × mix\n"
    "  G = Hα × strength × (1-mix) + G_rgb × mix\n"
    "  B = OIII × strength × (1-mix) + B_rgb × mix\n\n"
    "Use for: Maximum color separation in emission nebulae.\n"
    "Requires: Hα, OIII, SII",

    # LRGB_NB
    "Uses narrowband data as synthetic luminance.\n\n"
    "Formula: Scales RGB channels by ratio of NB luminance\n"
    "to RGB luminance, preserving original color ratios.\n\n"
    "NB Lum = weighted sum of available NB channels.\n\n"
    "Use for: Adding NB detail while keeping RGB colors.\n"
    "Requires: Hα (OIII, SII optional)",

    # FORAXX
    "Modified HOO with enhanced red/orange tones.\n\n"
    "Formula:\n"
    "  R = (0.3×OIII + 0.7×Hα) × (1-mix) + R_rgb × mix\n"
    "  G = OIII × (1-mix) + G_rgb × mix\n"
    "  B = OIII × (1-mix) + B_rgb × mix\n\n"
    "Use for: Warmer, more vibrant bicolor images.\n"
    "Requires: Hα, OIII",

    # HOO_SHO_MORPH
    "Smoothly blend between HOO and SHO palettes.\n\n"
    "Morph slider: 0.0 = pure HOO, 1.0 = pure SHO\n\n"
    "Channel strengths are applied before morphing.\n"
    "Intermediate values create hybrid palettes.\n\n"
    "Use for: Finding the ideal palette between bicolor and Hubble.\n"
    "Requires: Hα, OIII, SII",

    # GOLDEN_PALETTE
    "Warm golden-teal palette using all three NB channels.\n\n"
    "Formula:\n"
    "  R = 0.5×Hα + 0.5×SII (golden/orange)\n"
    "  G = 0.7×Hα + 0.3×OIII (yellow-green)\n"
    "  B = OIII (teal/cyan)\n\n"
    "Use for: Warm, aesthetically pleasing nebula images.\n"
    "Requires: Hα, OIII, SII",

    # CUSTOM_MAP
    "User-defined channel mapping with full flexibility.\n\n"
    "Map any source (Hα, OIII, SII, RGB channels) to any\n"
    "output channel (R, G, B). Use presets for common\n"
    "palettes or create your own combinations.\n\n"
    "Strength sliders apply to NB sources before mapping.\n"
    "RGB Mix blends NB result with original RGB.\n\n"
    "Use for: Any custom palette or experimental blends.",
]


class BlendMode:
    LINEAR = 0
    SCREEN = 1
    MAX = 2
    SOFT_LIGHT = 3


BLEND_MODE_NAMES = ["Linear", "Screen", "Maximum", "Soft Light"]


class ChannelSource:
    NONE = 0
    HA = 1
    OIII = 2
    SII = 3
    RGB_R = 4
    RGB_G = 5
    RGB_B = 6


CHANNEL_SOURCE_NAMES = ["None", "Hα", "OIII", "SII", "RGB Red", "RGB Green", "RGB Blue"]

MAPPING_PRESETS = [
    ("SHO (Hubble)", ChannelSource.SII, ChannelSource.HA, ChannelSource.OIII),
    ("HOO (Bicolor)", ChannelSource.HA, ChannelSource.OIII, ChannelSource.OIII),
    ("OHS (Inverted)", ChannelSource.OIII, ChannelSource.HA, ChannelSource.SII),
    ("HSO", ChannelSource.HA, ChannelSource.SII, ChannelSource.OIII),
    ("HHO", ChannelSource.HA, ChannelSource.HA, ChannelSource.OIII),
    ("SOO", ChannelSource.SII, ChannelSource.OIII, ChannelSource.OIII),
    ("RGB Only", ChannelSource.RGB_R, ChannelSource.RGB_G, ChannelSource.RGB_B),
]


# Utility Functions
def _to_float01(img):
    a = np.asarray(img)
    if a.dtype not in (np.float32, np.float64):
        a = a.astype(np.float32)
    else:
        a = np.asarray(a, dtype=np.float32)
    if a.size == 0:
        return a
    vmax = float(np.nanmax(a))
    if np.isfinite(vmax) and vmax > 1.5:
        a = a / vmax
    return np.clip(a, 0.0, 1.0)


def _ensure_rgb(img):
    if img.ndim == 2:
        img = img[..., None]
    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    return img.astype(np.float32)


def _ensure_mono(img):
    if img.ndim == 2:
        return img.astype(np.float32)
    if img.ndim == 3 and img.shape[2] == 3:
        return (0.3 * img[..., 0] + 0.6 * img[..., 1] + 0.1 * img[..., 2]).astype(np.float32)
    if img.ndim == 3 and img.shape[2] == 1:
        return img[..., 0].astype(np.float32)
    return img.astype(np.float32)


def _linear_fit(target, reference):
    ref_med = float(np.median(reference))
    tgt_med = float(np.median(target))
    ref_mad = float(np.median(np.abs(reference - ref_med)))
    tgt_mad = float(np.median(np.abs(target - tgt_med)))
    if tgt_mad < 1e-10:
        tgt_mad = 1e-10
    if ref_mad < 1e-10:
        return (target - tgt_med + ref_med).astype(np.float32)
    scale = ref_mad / tgt_mad
    return ((target - tgt_med) * scale + ref_med).astype(np.float32)


def _neutralize_background(img, percentile=15.0):
    bg = float(np.percentile(img, percentile))
    return np.maximum(img - bg, 0.0).astype(np.float32)


def _star_mask_from_rgb(rgb):
    avg = (rgb[..., 0] + rgb[..., 1] + rgb[..., 2]) / 3.0
    mask = (avg - 0.7) * 10.0
    return np.clip(mask, 0.0, 1.0).astype(np.float32)


def _estimate_background(img, percentile=15.0):
    """Estimate background level using a low percentile (more robust than median for nebulae)."""
    return float(np.percentile(img, percentile))


def _apply_blackpoint(nb, strength, blackpoint, clip_negative=True):
    """Apply blackpoint subtraction to narrowband data.
    
    Parameters
    ----------
    nb : ndarray
        Narrowband image data
    strength : float
        Intensity multiplier applied after blackpoint subtraction
    blackpoint : float
        Background subtraction level (0-1 range):
        - 0.0 = no subtraction (use raw data)
        - 0.5 = subtract half the estimated background
        - 1.0 = subtract full estimated background (default, neutral)
        - >1.0 = aggressive subtraction (clips into signal)
    clip_negative : bool
        If True, clip negative values to zero after subtraction
    
    Returns
    -------
    ndarray
        Processed narrowband data
    """
    if nb is None:
        return None
    
    # Use low percentile for background (better than median for nebulae)
    bg_level = _estimate_background(nb, percentile=15.0)
    
    # Blackpoint of 1.0 = subtract full background, 0.0 = no subtraction
    subtracted = nb - (blackpoint * bg_level)
    
    if clip_negative:
        subtracted = np.maximum(subtracted, 0.0)
    
    # Apply strength after subtraction
    result = subtracted * strength
    
    return result.astype(np.float32)


def _blend_screen(base, overlay):
    return 1.0 - (1.0 - base) * (1.0 - overlay)


def _blend_soft_light(base, overlay):
    return (1.0 - 2.0 * overlay) * base**2 + 2.0 * overlay * base


def _blend_channels(rgb_ch, nb_ch, rgb_weight, mode):
    if nb_ch is None:
        return rgb_ch.copy()
    nb_clipped = np.clip(nb_ch, 0.0, 1.0)
    nb_weight = 1.0 - rgb_weight
    if mode == BlendMode.LINEAR:
        result = nb_clipped * nb_weight + rgb_ch * rgb_weight
    elif mode == BlendMode.SCREEN:
        screened = _blend_screen(rgb_ch, nb_clipped)
        result = screened * nb_weight + rgb_ch * rgb_weight
    elif mode == BlendMode.MAX:
        maxed = np.maximum(rgb_ch, nb_clipped)
        result = maxed * nb_weight + rgb_ch * rgb_weight
    elif mode == BlendMode.SOFT_LIGHT:
        soft = _blend_soft_light(rgb_ch, nb_clipped)
        result = soft * nb_weight + rgb_ch * rgb_weight
    else:
        result = nb_clipped * nb_weight + rgb_ch * rgb_weight
    return result.astype(np.float32)


def _apply_external_star_mask(nb_r, nb_g, nb_b, star_mask, star_balance):
    if star_mask is None or star_balance <= 0:
        return nb_r, nb_g, nb_b
    subtract = star_balance * star_mask
    if nb_r is not None:
        nb_r = nb_r - subtract
    if nb_g is not None:
        nb_g = nb_g - subtract
    if nb_b is not None:
        nb_b = nb_b - subtract
    return nb_r, nb_g, nb_b


def _apply_vibrance(img, vibrance):
    if abs(vibrance) < 0.001:
        return img
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    lum = 0.3 * r + 0.6 * g + 0.1 * b
    factor = 1.0 + vibrance * 0.5
    r_out = lum + (r - lum) * factor
    g_out = lum + (g - lum) * factor
    b_out = lum + (b - lum) * factor
    return np.stack([r_out, g_out, b_out], axis=-1).astype(np.float32)


def _apply_highlights_compression(img, highlights):
    if highlights >= 0.999:
        return img
    pivot = 0.75
    out = img.copy()
    high_mask = img > pivot
    out[high_mask] = pivot + (img[high_mask] - pivot) * highlights
    return out.astype(np.float32)


def _apply_luminance(img, luminance, mode, strength, preserve_saturation):
    """Apply luminance to a combined RGB image.
    
    Parameters
    ----------
    img : ndarray (H, W, 3)
        Combined RGB image
    luminance : ndarray (H, W)
        Luminance image (mono)
    mode : int
        0 = Replace, 1 = Blend
    strength : float
        How strongly to apply luminance (0-1)
    preserve_saturation : bool
        Boost saturation to compensate for luminance application
    
    Returns
    -------
    ndarray (H, W, 3)
        Image with luminance applied
    """
    if luminance is None or strength <= 0:
        return img
    
    # Get current luminance of the combined image
    R, G, B = img[..., 0], img[..., 1], img[..., 2]
    current_lum = 0.3 * R + 0.6 * G + 0.1 * B
    current_lum = np.maximum(current_lum, 1e-6)  # Avoid division by zero
    
    # Target luminance
    target_lum = np.clip(luminance, 0.0, 1.0)
    
    if mode == 0:  # Replace
        # Scale RGB to match target luminance
        if strength >= 0.999:
            new_lum = target_lum
        else:
            new_lum = current_lum * (1 - strength) + target_lum * strength
        
        scale = new_lum / current_lum
        
        if preserve_saturation:
            # Calculate saturation before scaling
            max_rgb = np.maximum(np.maximum(R, G), B)
            min_rgb = np.minimum(np.minimum(R, G), B)
            chroma = max_rgb - min_rgb
            old_sat = chroma / np.maximum(max_rgb, 1e-6)
        
        R_out = R * scale
        G_out = G * scale
        B_out = B * scale
        
        if preserve_saturation:
            # Restore saturation
            new_lum_out = 0.3 * R_out + 0.6 * G_out + 0.1 * B_out
            new_lum_out = np.maximum(new_lum_out, 1e-6)
            
            # Boost saturation proportionally to how much we changed luminance
            sat_boost = np.where(current_lum > 0.01, 
                                 np.sqrt(current_lum / np.maximum(new_lum, 1e-6)), 
                                 1.0)
            sat_boost = np.clip(sat_boost, 0.5, 2.0)
            
            R_out = new_lum_out + (R_out - new_lum_out) * sat_boost
            G_out = new_lum_out + (G_out - new_lum_out) * sat_boost
            B_out = new_lum_out + (B_out - new_lum_out) * sat_boost
    
    else:  # Blend
        # Simple blend between original and L-replaced version
        scale = target_lum / current_lum
        R_lum = R * scale
        G_lum = G * scale
        B_lum = B * scale
        
        R_out = R * (1 - strength) + R_lum * strength
        G_out = G * (1 - strength) + G_lum * strength
        B_out = B * (1 - strength) + B_lum * strength
        
        if preserve_saturation and strength > 0.01:
            # Mild saturation boost for blend mode
            out_lum = 0.3 * R_out + 0.6 * G_out + 0.1 * B_out
            out_lum = np.maximum(out_lum, 1e-6)
            sat_boost = 1.0 + 0.3 * strength  # Gentle boost
            R_out = out_lum + (R_out - out_lum) * sat_boost
            G_out = out_lum + (G_out - out_lum) * sat_boost
            B_out = out_lum + (B_out - out_lum) * sat_boost
    
    out = np.stack([R_out, G_out, B_out], axis=-1)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _create_synthetic_luminance(rgb, r_weight=0.3, g_weight=0.6, b_weight=0.1):
    """Create synthetic luminance from RGB image.
    
    Parameters
    ----------
    rgb : ndarray (H, W, 3)
        RGB image
    r_weight, g_weight, b_weight : float
        Channel weights (should sum to ~1.0)
    
    Returns
    -------
    ndarray (H, W)
        Luminance image
    """
    total = r_weight + g_weight + b_weight
    if total > 0:
        r_weight /= total
        g_weight /= total
        b_weight /= total
    
    lum = r_weight * rgb[..., 0] + g_weight * rgb[..., 1] + b_weight * rgb[..., 2]
    return lum.astype(np.float32)


def _apply_rescale(img, rescale):
    if rescale >= 0.999:
        return img
    max_val = float(np.max(img))
    if max_val > 1.0:
        return (img / max_val * rescale).astype(np.float32)
    return img


# Core combination function
def combine_rgb_nb(
    rgb, ha, oiii, sii,
    method, ha_strength, oiii_strength, sii_strength,
    rgb_weight, boost, do_preserve_stars, do_remove_magenta,
    morph_factor=0.5,
    custom_r_source=ChannelSource.HA,
    custom_g_source=ChannelSource.OIII,
    custom_b_source=ChannelSource.OIII,
    do_linear_fit=False,
    do_neutralize_bg=False,
    rgb_r_weight=1.0, rgb_g_weight=1.0, rgb_b_weight=1.0,
    blackpoint_ha=1.0, blackpoint_oiii=1.0, blackpoint_sii=1.0,
    use_blackpoint=False,
    blend_mode=BlendMode.LINEAR,
    external_star_mask=None,
    star_balance=0.2,
    # Luminance options
    apply_luminance=False,
    luminance_image=None,
    lum_source=0,  # 0=Synthetic RGB, 1=External
    lum_mode=0,    # 0=Replace, 1=Blend
    lum_strength=1.0,
    lum_preserve_sat=True,
    lum_r_weight=0.3,
    lum_g_weight=0.6,
    lum_b_weight=0.1,
    # Post-processing
    rescale=1.0, vibrance=0.0, highlights=1.0,
):
    """Core RGB+NB combiner."""
    if rgb is None:
        raise ValueError("RGB image cannot be None")
    
    rgb = _ensure_rgb(_to_float01(rgb))
    
    if ha is not None:
        ha = _ensure_mono(_to_float01(ha))
    if oiii is not None:
        oiii = _ensure_mono(_to_float01(oiii))
    if sii is not None:
        sii = _ensure_mono(_to_float01(sii))
    if external_star_mask is not None:
        external_star_mask = _ensure_mono(_to_float01(external_star_mask))

    # Background neutralization
    if do_neutralize_bg:
        rgb[..., 0] = _neutralize_background(rgb[..., 0])
        rgb[..., 1] = _neutralize_background(rgb[..., 1])
        rgb[..., 2] = _neutralize_background(rgb[..., 2])
        if ha is not None:
            ha = _neutralize_background(ha)
        if oiii is not None:
            oiii = _neutralize_background(oiii)
        if sii is not None:
            sii = _neutralize_background(sii)

    # Linear fit
    if do_linear_fit:
        if ha is not None:
            ha = np.clip(_linear_fit(ha, rgb[..., 0]), 0.0, 1.0)
        if oiii is not None:
            ref_gb = (rgb[..., 1] + rgb[..., 2]) / 2.0
            oiii = np.clip(_linear_fit(oiii, ref_gb), 0.0, 1.0)
        if sii is not None:
            sii = np.clip(_linear_fit(sii, rgb[..., 0]), 0.0, 1.0)

    R0 = rgb[..., 0].copy()
    G0 = rgb[..., 1].copy()
    B0 = rgb[..., 2].copy()

    r_mix = float(rgb_weight) * float(rgb_r_weight)
    g_mix = float(rgb_weight) * float(rgb_g_weight)
    b_mix = float(rgb_weight) * float(rgb_b_weight)

    # Prepare NB with optional blackpoint
    if use_blackpoint:
        ha_prep = _apply_blackpoint(ha, ha_strength, blackpoint_ha) if ha is not None else None
        oiii_prep = _apply_blackpoint(oiii, oiii_strength, blackpoint_oiii) if oiii is not None else None
        sii_prep = _apply_blackpoint(sii, sii_strength, blackpoint_sii) if sii is not None else None
    else:
        ha_prep = (ha * ha_strength) if ha is not None else None
        oiii_prep = (oiii * oiii_strength) if oiii is not None else None
        sii_prep = (sii * sii_strength) if sii is not None else None

    # External star mask
    if external_star_mask is not None and star_balance > 0:
        ha_prep, oiii_prep, sii_prep = _apply_external_star_mask(
            ha_prep, oiii_prep, sii_prep, external_star_mask, star_balance
        )

    R, G, B = R0.copy(), G0.copy(), B0.copy()

    # Combination methods
    if method == CombinationMethod.HA_ENHANCE:
        if ha_prep is not None:
            R = _blend_channels(R0, ha_prep, rgb_weight, BlendMode.MAX)
            if do_remove_magenta:
                G = np.maximum(G0, 0.3 * np.clip(ha_prep, 0, 1))
            R, G, B = R * boost, G * boost, B * boost

    elif method == CombinationMethod.HOO_RGB:
        if ha_prep is not None and oiii_prep is not None:
            R_nb = ha_prep
            G_nb = (oiii_prep * 0.8 + ha_prep * 0.2) if do_remove_magenta else oiii_prep
            B_nb = oiii_prep
            R = _blend_channels(R0, R_nb, r_mix, blend_mode) * boost
            G = _blend_channels(G0, G_nb, g_mix, blend_mode) * boost
            B = _blend_channels(B0, B_nb, b_mix, blend_mode) * boost

    elif method == CombinationMethod.SHO_RGB:
        if ha_prep is not None and oiii_prep is not None and sii_prep is not None:
            R = _blend_channels(R0, sii_prep, r_mix, blend_mode) * boost
            G = _blend_channels(G0, ha_prep, g_mix, blend_mode) * boost
            B = _blend_channels(B0, oiii_prep, b_mix, blend_mode) * boost

    elif method == CombinationMethod.LRGB_NB:
        if ha_prep is not None:
            if oiii_prep is not None and sii_prep is not None:
                lum = (np.clip(ha_prep,0,1) + np.clip(oiii_prep,0,1) + np.clip(sii_prep,0,1)) / 3.0
            elif oiii_prep is not None:
                lum = (np.clip(ha_prep,0,1) + np.clip(oiii_prep,0,1)) / 2.0
            else:
                lum = np.clip(ha_prep, 0, 1)
            rgb_lum = np.maximum(0.001, R0 * 0.3 + G0 * 0.6 + B0 * 0.1)
            scale = lum / rgb_lum
            R, G, B = R0 * scale * boost, G0 * scale * boost, B0 * scale * boost

    elif method == CombinationMethod.FORAXX:
        if ha_prep is not None and oiii_prep is not None:
            R_nb = oiii_prep * 0.3 + ha_prep * 0.7
            R = _blend_channels(R0, R_nb, r_mix, blend_mode) * boost
            G = _blend_channels(G0, oiii_prep, g_mix, blend_mode) * boost
            B = _blend_channels(B0, oiii_prep, b_mix, blend_mode) * boost

    elif method == CombinationMethod.HOO_SHO_MORPH:
        if ha_prep is not None and oiii_prep is not None and sii_prep is not None:
            t = np.clip(morph_factor, 0.0, 1.0)
            R_nb = (1-t) * ha_prep + t * sii_prep
            G_nb = (1-t) * oiii_prep + t * ha_prep
            B_nb = oiii_prep
            R = _blend_channels(R0, R_nb, r_mix, blend_mode) * boost
            G = _blend_channels(G0, G_nb, g_mix, blend_mode) * boost
            B = _blend_channels(B0, B_nb, b_mix, blend_mode) * boost

    elif method == CombinationMethod.GOLDEN_PALETTE:
        if ha_prep is not None and oiii_prep is not None and sii_prep is not None:
            R_nb = 0.5 * ha_prep + 0.5 * sii_prep
            G_nb = 0.7 * ha_prep + 0.3 * oiii_prep
            R = _blend_channels(R0, R_nb, r_mix, blend_mode) * boost
            G = _blend_channels(G0, G_nb, g_mix, blend_mode) * boost
            B = _blend_channels(B0, oiii_prep, b_mix, blend_mode) * boost

    elif method == CombinationMethod.CUSTOM_MAP:
        sources = {
            ChannelSource.NONE: None,
            ChannelSource.HA: ha_prep,
            ChannelSource.OIII: oiii_prep,
            ChannelSource.SII: sii_prep,
            ChannelSource.RGB_R: R0,
            ChannelSource.RGB_G: G0,
            ChannelSource.RGB_B: B0,
        }
        R_src = sources.get(custom_r_source)
        G_src = sources.get(custom_g_source)
        B_src = sources.get(custom_b_source)
        
        if custom_r_source >= ChannelSource.RGB_R:
            R = R_src if R_src is not None else R0
        else:
            R = _blend_channels(R0, R_src, r_mix, blend_mode) if R_src is not None else R0
        if custom_g_source >= ChannelSource.RGB_R:
            G = G_src if G_src is not None else G0
        else:
            G = _blend_channels(G0, G_src, g_mix, blend_mode) if G_src is not None else G0
        if custom_b_source >= ChannelSource.RGB_R:
            B = B_src if B_src is not None else B0
        else:
            B = _blend_channels(B0, B_src, b_mix, blend_mode) if B_src is not None else B0
        R, G, B = R * boost, G * boost, B * boost

    # Star preservation
    if do_preserve_stars:
        star_mask = _star_mask_from_rgb(np.stack([R0, G0, B0], axis=2))
        inv_mask = 1.0 - star_mask
        R = R * inv_mask + R0 * star_mask
        G = G * inv_mask + G0 * star_mask
        B = B * inv_mask + B0 * star_mask

    out = np.stack([R, G, B], axis=2)
    
    # Luminance application
    if apply_luminance and lum_strength > 0:
        if lum_source == 0:  # Synthetic from RGB
            lum = _create_synthetic_luminance(rgb, lum_r_weight, lum_g_weight, lum_b_weight)
        else:  # External L image
            if luminance_image is not None:
                lum = _ensure_mono(_to_float01(luminance_image))
            else:
                lum = None
        
        if lum is not None:
            out = _apply_luminance(out, lum, lum_mode, lum_strength, lum_preserve_sat)
    
    # Post-processing
    out = _apply_rescale(out, rescale)
    out = _apply_vibrance(out, vibrance)
    out = _apply_highlights_compression(out, highlights)
    
    return np.clip(out, 0.0, 1.0).astype(np.float32)


# Dialog
class RGBNBCombinerDialog(QDialog):
    def __init__(self, ctx):
        super().__init__(parent=ctx.app)
        self.ctx = ctx
        self.setWindowTitle(SCRIPT_NAME)
        self.resize(1200, 750)
        self._title_to_doc = {}
        self._preview_rgb = None
        self._preview_ha = None
        self._preview_oiii = None
        self._preview_sii = None
        self._preview_star_mask = None
        self._preview_luminance = None
        self._pix = None
        self._zoom = 1.0
        self._panning = False
        self._pan_start = QPointF()
        self._pan_hval = 0
        self._pan_vval = 0
        self._control_labels = {}
        self._build_ui()
        self._populate_views()
        self._update_control_visibility()
        self._update_rgb_weights_visibility()
        self._update_luminance_visibility()
        self._update_warning_label()
        self._fit_preview()

    def _build_ui(self):
        root = QVBoxLayout(self)
        main = QHBoxLayout()
        root.addLayout(main)

        # Left: Image Selection
        img_box = QGroupBox("Image Selection")
        img_layout = QVBoxLayout(img_box)
        
        img_layout.addWidget(QLabel("RGB Image:"))
        self.combo_rgb = QComboBox()
        self.combo_rgb.setToolTip(
            "Base RGB broadband image.\n\n"
            "This provides the star colors and overall structure.\n"
            "Narrowband data will be blended into this image."
        )
        img_layout.addWidget(self.combo_rgb)
        
        img_layout.addWidget(QLabel("Hα Image:"))
        self.combo_ha = QComboBox()
        self.combo_ha.setToolTip(
            "Hydrogen-alpha narrowband image (656nm).\n\n"
            "Captures emission from ionized hydrogen.\n"
            "Primary signal for most emission nebulae.\n\n"
            "Required for: All methods except Custom."
        )
        img_layout.addWidget(self.combo_ha)
        
        img_layout.addWidget(QLabel("OIII Image:"))
        self.combo_oiii = QComboBox()
        self.combo_oiii.setToolTip(
            "Oxygen-III narrowband image (500nm).\n\n"
            "Captures doubly-ionized oxygen emission.\n"
            "Common in planetary nebulae and supernova remnants.\n\n"
            "Required for: HOO, SHO, Foraxx, Morph, Golden palettes."
        )
        img_layout.addWidget(self.combo_oiii)
        
        img_layout.addWidget(QLabel("SII Image:"))
        self.combo_sii = QComboBox()
        self.combo_sii.setToolTip(
            "Sulfur-II narrowband image (672nm).\n\n"
            "Captures ionized sulfur emission.\n"
            "Reveals shock fronts and ionization boundaries.\n\n"
            "Required for: SHO (Hubble), Morph, Golden palettes."
        )
        img_layout.addWidget(self.combo_sii)
        
        img_layout.addWidget(QLabel("Star Mask (optional):"))
        self.combo_star_mask = QComboBox()
        self.combo_star_mask.setToolTip(
            "External star mask for NB star reduction.\n\n"
            "White = stars, Black = nebula/background.\n"
            "Used with 'Star Balance' to subtract star signal\n"
            "from narrowband channels before combination.\n\n"
            "Optional: Leave as <None> to skip this step."
        )
        img_layout.addWidget(self.combo_star_mask)
        
        img_layout.addWidget(QLabel("Luminance (optional):"))
        self.combo_luminance = QComboBox()
        self.combo_luminance.setToolTip(
            "External luminance image for LRGB-style processing.\n\n"
            "Use a dedicated L filter capture to add structure\n"
            "and detail to the narrowband combination.\n\n"
            "Ideal for: Galaxies, targets with fine detail,\n"
            "or when you have mono L data available.\n\n"
            "Optional: Leave as <None> to use synthetic L or skip."
        )
        img_layout.addWidget(self.combo_luminance)
        
        img_layout.addStretch()
        main.addWidget(img_box)

        # Middle: Controls
        ctrl_box = QGroupBox("Controls")
        ctrl_layout = QVBoxLayout(ctrl_box)
        
        ctrl_layout.addWidget(QLabel("Method:"))
        self.combo_method = QComboBox()
        for i, name in enumerate(METHOD_NAMES):
            self.combo_method.addItem(name)
            self.combo_method.setItemData(i, METHOD_TOOLTIPS[i], Qt.ItemDataRole.ToolTipRole)
        ctrl_layout.addWidget(self.combo_method)
        
        ctrl_layout.addWidget(QLabel("Blend Mode:"))
        self.combo_blend_mode = QComboBox()
        self.combo_blend_mode.setToolTip(
            "How narrowband data is combined with RGB:\n\n"
            "Linear: Simple weighted average (NB × weight + RGB × (1-weight))\n"
            "  Best for: General use, predictable results\n\n"
            "Screen: Brightening blend (1 - (1-NB) × (1-RGB))\n"
            "  Best for: Adding NB glow without clipping highlights\n\n"
            "Maximum: Takes brighter pixel from NB or RGB\n"
            "  Best for: Preserving brightest details from both\n\n"
            "Soft Light: Gentle contrast enhancement\n"
            "  Best for: Subtle NB integration with natural look"
        )
        for name in BLEND_MODE_NAMES:
            self.combo_blend_mode.addItem(name)
        ctrl_layout.addWidget(self.combo_blend_mode)

        grid = QGridLayout()
        row = 0
        
        def add_spin(key, label, default, minv, maxv, step=0.05):
            nonlocal row
            lbl = QLabel(label)
            self._control_labels[key] = lbl
            grid.addWidget(lbl, row, 0)
            spin = QDoubleSpinBox()
            spin.setRange(minv, maxv)
            spin.setValue(default)
            spin.setSingleStep(step)
            spin.setDecimals(3)
            spin.setMinimumWidth(80)
            grid.addWidget(spin, row, 1)
            row += 1
            return spin

        self.spin_ha = add_spin("ha", "Hα Strength:", 0.8, 0, 3, 0.05)
        self.spin_oiii = add_spin("oiii", "OIII Strength:", 0.6, 0, 3, 0.05)
        self.spin_sii = add_spin("sii", "SII Strength:", 0.4, 0, 3, 0.05)
        self.spin_morph = add_spin("morph", "Morph:", 0.5, 0, 1, 0.02)
        self.spin_rgb = add_spin("rgb", "RGB Mix:", 0.3, 0, 1, 0.02)
        self.spin_rgb.setToolTip(
            "Controls blend between narrowband and original RGB.\n"
            "0.0 = pure narrowband, 1.0 = pure RGB.\n"
            "Default 0.3 = 70% NB + 30% RGB."
        )
        
        # Per-channel RGB weights
        self.spin_rgb_r = add_spin("rgb_r", "  R Weight:", 1.0, 0.0, 2.0)
        self.spin_rgb_r.setToolTip("Weight multiplier for the original RGB red channel.")
        self.spin_rgb_g = add_spin("rgb_g", "  G Weight:", 1.0, 0.0, 2.0)
        self.spin_rgb_g.setToolTip("Weight multiplier for the original RGB green channel.")
        self.spin_rgb_b = add_spin("rgb_b", "  B Weight:", 1.0, 0.0, 2.0)
        self.spin_rgb_b.setToolTip("Weight multiplier for the original RGB blue channel.")
        
        self.spin_boost = add_spin("boost", "Boost:", 1.0, 0.5, 2, 0.02)
        self.spin_boost.setToolTip(
            "Global brightness multiplier applied after combination.\n\n"
            "1.0 = No change\n"
            "<1.0 = Darken result\n"
            ">1.0 = Brighten result (may clip highlights)\n\n"
            "Use to compensate for darkening from blend operations."
        )
        
        ctrl_layout.addLayout(grid)

        # Checkbox to show/hide per-channel RGB weights
        self.chk_show_rgb_weights = QCheckBox("Per-channel RGB weights")
        self.chk_show_rgb_weights.setChecked(False)
        self.chk_show_rgb_weights.setToolTip(
            "Show individual weight controls for each RGB channel."
        )
        self.chk_show_rgb_weights.toggled.connect(self._update_rgb_weights_visibility)
        ctrl_layout.addWidget(self.chk_show_rgb_weights)

        # Mapping group with presets
        self.mapping_group = QGroupBox("Channel Mapping")
        self.mapping_group.setToolTip(
            "Custom channel mapping for full control.\n\n"
            "Assign any source (Hα, OIII, SII, or RGB channels)\n"
            "to any output channel (Red, Green, Blue).\n\n"
            "Use presets for common palettes or create your own."
        )
        map_layout = QGridLayout(self.mapping_group)
        
        # Presets
        map_layout.addWidget(QLabel("Preset:"), 0, 0)
        self.combo_preset = QComboBox()
        self.combo_preset.setToolTip(
            "Quick-select common channel mappings:\n\n"
            "SHO (Hubble): S→R, H→G, O→B (classic Hubble palette)\n"
            "HOO (Bicolor): H→R, O→G, O→B (two-filter palette)\n"
            "OHS (Inverted): O→R, H→G, S→B\n"
            "HSO: H→R, S→G, O→B\n"
            "HHO: H→R, H→G, O→B\n"
            "SOO: S→R, O→G, O→B\n"
            "RGB Only: Pass through original RGB"
        )
        self.combo_preset.addItem("(Select preset...)")
        for preset_name, _, _, _ in MAPPING_PRESETS:
            self.combo_preset.addItem(preset_name)
        self.combo_preset.currentIndexChanged.connect(self._apply_preset)
        map_layout.addWidget(self.combo_preset, 0, 1)
        
        self.combo_map_r = QComboBox()
        self.combo_map_g = QComboBox()
        self.combo_map_b = QComboBox()
        for c in [self.combo_map_r, self.combo_map_g, self.combo_map_b]:
            for n in CHANNEL_SOURCE_NAMES:
                c.addItem(n)
        self.combo_map_r.setCurrentIndex(ChannelSource.SII)
        self.combo_map_g.setCurrentIndex(ChannelSource.HA)
        self.combo_map_b.setCurrentIndex(ChannelSource.OIII)
        self.combo_map_r.setToolTip("Source channel for Red output")
        self.combo_map_g.setToolTip("Source channel for Green output")
        self.combo_map_b.setToolTip("Source channel for Blue output")
        map_layout.addWidget(QLabel("R←"), 1, 0)
        map_layout.addWidget(self.combo_map_r, 1, 1)
        map_layout.addWidget(QLabel("G←"), 2, 0)
        map_layout.addWidget(self.combo_map_g, 2, 1)
        map_layout.addWidget(QLabel("B←"), 3, 0)
        map_layout.addWidget(self.combo_map_b, 3, 1)
        ctrl_layout.addWidget(self.mapping_group)

        self.chk_preserve_stars = QCheckBox("Preserve RGB star colors")
        self.chk_preserve_stars.setChecked(True)
        self.chk_preserve_stars.setToolTip(
            "Blend original RGB star colors back into the result.\n\n"
            "Uses a brightness-based mask to detect stars and\n"
            "preserve their natural colors from the RGB image.\n\n"
            "Recommended: ON for most combinations to avoid\n"
            "colored halos or unnatural star tints from NB data."
        )
        ctrl_layout.addWidget(self.chk_preserve_stars)
        
        self.chk_remove_magenta = QCheckBox("Reduce magenta")
        self.chk_remove_magenta.setToolTip(
            "Adds Hα signal to the green channel to reduce\n"
            "magenta/pink color cast.\n\n"
            "Common in HOO and Hα enhancement methods where\n"
            "strong red + blue with weak green creates magenta.\n\n"
            "Effect varies by method - most useful for HOO/Hα enhance."
        )
        ctrl_layout.addWidget(self.chk_remove_magenta)
        
        self.chk_linear_fit = QCheckBox("Linear fit NB to RGB")
        self.chk_linear_fit.setToolTip(
            "Normalize narrowband intensity to match RGB levels.\n\n"
            "Scales each NB channel so its median and spread\n"
            "match the corresponding RGB channel.\n\n"
            "Use when: NB and RGB have very different exposure\n"
            "levels or stretching, causing imbalanced blends.\n\n"
            "Hα/SII → matched to Red channel\n"
            "OIII → matched to average of Green+Blue"
        )
        ctrl_layout.addWidget(self.chk_linear_fit)
        
        self.chk_neutralize_bg = QCheckBox("Neutralize background")
        self.chk_neutralize_bg.setToolTip(
            "Subtract sky background from all channels.\n\n"
            "Removes the 15th percentile value from RGB and\n"
            "all NB channels before combination.\n\n"
            "Use when: Images have light pollution gradients,\n"
            "uneven backgrounds, or color casts in the sky.\n\n"
            "Applied before linear fit and blackpoint."
        )
        ctrl_layout.addWidget(self.chk_neutralize_bg)

        ctrl_layout.addStretch()
        main.addWidget(ctrl_box)

        # Column 3: Advanced Features
        adv_box = QGroupBox("Advanced")
        adv_layout = QVBoxLayout(adv_box)
        
        self.chk_use_blackpoint = QCheckBox("Per-channel blackpoint")
        self.chk_use_blackpoint.setToolTip(
            "Enable background subtraction for each narrowband channel.\n\n"
            "Subtracts the estimated sky background before combining.\n"
            "Use this to reduce background glow and improve contrast.\n\n"
            "Values:\n"
            "  0.0 = No subtraction (raw data)\n"
            "  1.0 = Subtract full background (default)\n"
            "  >1.0 = Aggressive (clips into faint signal)"
        )
        self.chk_use_blackpoint.toggled.connect(self._update_blackpoint_visibility)
        adv_layout.addWidget(self.chk_use_blackpoint)
        
        self.bp_grid_widget = QFrame()
        bp_grid = QGridLayout(self.bp_grid_widget)
        bp_grid.setContentsMargins(0, 0, 0, 0)
        self.spin_bp_ha = QDoubleSpinBox()
        self.spin_bp_oiii = QDoubleSpinBox()
        self.spin_bp_sii = QDoubleSpinBox()
        bp_labels = [("Hα BP:", self.spin_bp_ha), ("OIII BP:", self.spin_bp_oiii), ("SII BP:", self.spin_bp_sii)]
        for i, (lbl, spin) in enumerate(bp_labels):
            spin.setRange(0, 2)
            spin.setValue(1.0)
            spin.setSingleStep(0.05)
            spin.setDecimals(3)
            spin.setToolTip(
                "Blackpoint level for background subtraction.\n"
                "1.0 = subtract estimated background\n"
                "0.0 = no subtraction, >1.0 = aggressive"
            )
            bp_grid.addWidget(QLabel(lbl), i, 0)
            bp_grid.addWidget(spin, i, 1)
        self.bp_grid_widget.setVisible(False)
        adv_layout.addWidget(self.bp_grid_widget)

        # Star balance widget (hidden, uses default value 0.2)
        self.spin_star_balance = QDoubleSpinBox()
        self.spin_star_balance.setRange(0, 1)
        self.spin_star_balance.setValue(0.2)
        self.spin_star_balance.setSingleStep(0.05)
        self.spin_star_balance.setDecimals(3)
        # Not added to layout - widget exists but is not visible

        # Luminance controls
        self.chk_apply_luminance = QCheckBox("Apply Luminance")
        self.chk_apply_luminance.setToolTip(
            "Enable luminance processing after NB combination.\n\n"
            "Applies a luminance layer to control brightness and\n"
            "add structural detail to the combined result.\n\n"
            "Source can be synthetic (from RGB) or an external\n"
            "L image selected in Image Selection above."
        )
        self.chk_apply_luminance.toggled.connect(self._update_luminance_visibility)
        adv_layout.addWidget(self.chk_apply_luminance)
        
        self.lum_controls_widget = QFrame()
        lum_layout = QVBoxLayout(self.lum_controls_widget)
        lum_layout.setContentsMargins(10, 0, 0, 0)
        lum_layout.setSpacing(4)
        
        # L Source
        lum_source_row = QHBoxLayout()
        lum_source_row.addWidget(QLabel("Source:"))
        self.combo_lum_source = QComboBox()
        self.combo_lum_source.addItems(["Synthetic from RGB", "External L Image"])
        self.combo_lum_source.setToolTip(
            "Where to get the luminance data:\n\n"
            "Synthetic from RGB:\n"
            "  Creates L from the RGB image using weighted average.\n"
            "  Good for OSC data or when no L capture available.\n"
            "  Formula: L = 0.3×R + 0.6×G + 0.1×B\n\n"
            "External L Image:\n"
            "  Uses the L image selected in Image Selection.\n"
            "  Best for mono cameras with dedicated L filter data."
        )
        self.combo_lum_source.currentIndexChanged.connect(self._update_luminance_visibility)
        lum_source_row.addWidget(self.combo_lum_source, 1)
        lum_layout.addLayout(lum_source_row)
        
        # L Mode
        lum_mode_row = QHBoxLayout()
        lum_mode_row.addWidget(QLabel("Mode:"))
        self.combo_lum_mode = QComboBox()
        self.combo_lum_mode.addItems(["Replace", "Blend"])
        self.combo_lum_mode.setToolTip(
            "How luminance is applied to the combined image:\n\n"
            "Replace:\n"
            "  Classic LRGB - L completely controls brightness.\n"
            "  Combined image provides color/chrominance only.\n"
            "  Best for: Maximum detail from L data.\n\n"
            "Blend:\n"
            "  Mix between original NB brightness and L.\n"
            "  Use L Strength to control the balance.\n"
            "  Best for: Subtle enhancement, preserving NB intensity."
        )
        lum_mode_row.addWidget(self.combo_lum_mode, 1)
        lum_layout.addLayout(lum_mode_row)
        
        # L Strength
        lum_strength_row = QHBoxLayout()
        lum_strength_row.addWidget(QLabel("Strength:"))
        self.spin_lum_strength = QDoubleSpinBox()
        self.spin_lum_strength.setRange(0.0, 1.0)
        self.spin_lum_strength.setValue(1.0)
        self.spin_lum_strength.setSingleStep(0.05)
        self.spin_lum_strength.setDecimals(3)
        self.spin_lum_strength.setToolTip(
            "How strongly luminance affects the result.\n\n"
            "For Replace mode:\n"
            "  1.0 = Full L replacement\n"
            "  0.5 = Half L, half original brightness\n"
            "  0.0 = No luminance effect\n\n"
            "For Blend mode:\n"
            "  Controls the mix between original and L.\n"
            "  Lower values preserve more NB character."
        )
        lum_strength_row.addWidget(self.spin_lum_strength, 1)
        lum_layout.addLayout(lum_strength_row)
        
        # Preserve saturation checkbox
        self.chk_preserve_sat = QCheckBox("Preserve color saturation")
        self.chk_preserve_sat.setChecked(True)
        self.chk_preserve_sat.setToolTip(
            "Boost saturation to compensate for luminance application.\n\n"
            "When L replaces brightness, colors can appear washed out.\n"
            "This option restores saturation proportionally.\n\n"
            "Recommended: ON for most cases.\n"
            "Turn OFF if colors become oversaturated."
        )
        lum_layout.addWidget(self.chk_preserve_sat)
        
        # Advanced L controls (hidden by default)
        self.chk_advanced_lum = QCheckBox("Advanced L controls")
        self.chk_advanced_lum.setToolTip("Show additional luminance options")
        self.chk_advanced_lum.toggled.connect(self._update_luminance_visibility)
        lum_layout.addWidget(self.chk_advanced_lum)
        
        self.adv_lum_widget = QFrame()
        adv_lum_layout = QGridLayout(self.adv_lum_widget)
        adv_lum_layout.setContentsMargins(10, 0, 0, 0)
        
        # Custom RGB weights for synthetic L
        adv_lum_layout.addWidget(QLabel("Synthetic L weights:"), 0, 0, 1, 2)
        self.spin_lum_r_weight = QDoubleSpinBox()
        self.spin_lum_r_weight.setRange(0.0, 1.0)
        self.spin_lum_r_weight.setValue(0.3)
        self.spin_lum_r_weight.setSingleStep(0.05)
        self.spin_lum_r_weight.setDecimals(2)
        self.spin_lum_r_weight.setToolTip("Red channel contribution to synthetic L")
        adv_lum_layout.addWidget(QLabel("R:"), 1, 0)
        adv_lum_layout.addWidget(self.spin_lum_r_weight, 1, 1)
        
        self.spin_lum_g_weight = QDoubleSpinBox()
        self.spin_lum_g_weight.setRange(0.0, 1.0)
        self.spin_lum_g_weight.setValue(0.6)
        self.spin_lum_g_weight.setSingleStep(0.05)
        self.spin_lum_g_weight.setDecimals(2)
        self.spin_lum_g_weight.setToolTip("Green channel contribution to synthetic L")
        adv_lum_layout.addWidget(QLabel("G:"), 2, 0)
        adv_lum_layout.addWidget(self.spin_lum_g_weight, 2, 1)
        
        self.spin_lum_b_weight = QDoubleSpinBox()
        self.spin_lum_b_weight.setRange(0.0, 1.0)
        self.spin_lum_b_weight.setValue(0.1)
        self.spin_lum_b_weight.setSingleStep(0.05)
        self.spin_lum_b_weight.setDecimals(2)
        self.spin_lum_b_weight.setToolTip("Blue channel contribution to synthetic L")
        adv_lum_layout.addWidget(QLabel("B:"), 3, 0)
        adv_lum_layout.addWidget(self.spin_lum_b_weight, 3, 1)
        
        self.adv_lum_widget.setVisible(False)
        lum_layout.addWidget(self.adv_lum_widget)
        
        self.lum_controls_widget.setVisible(False)
        adv_layout.addWidget(self.lum_controls_widget)
        
        adv_layout.addWidget(QLabel("Post-Processing:"))
        post_grid = QGridLayout()
        self.spin_rescale = QDoubleSpinBox()
        self.spin_rescale.setRange(0.5, 1)
        self.spin_rescale.setValue(1.0)
        self.spin_rescale.setSingleStep(0.01)
        self.spin_rescale.setDecimals(3)
        self.spin_rescale.setToolTip(
            "Rescale maximum value to prevent clipping.\n\n"
            "If result exceeds 1.0, scales entire image so\n"
            "the brightest pixel equals this value.\n\n"
            "1.0 = No rescaling (may clip)\n"
            "0.95 = Leave 5% headroom\n"
            "0.8 = Significant darkening to preserve highlights"
        )
        self.spin_vibrance = QDoubleSpinBox()
        self.spin_vibrance.setRange(-1, 1)
        self.spin_vibrance.setValue(0)
        self.spin_vibrance.setSingleStep(0.02)
        self.spin_vibrance.setDecimals(3)
        self.spin_vibrance.setToolTip(
            "Adjust color saturation around neutral tones.\n\n"
            "Similar to vibrance in photo editors - affects\n"
            "less-saturated colors more than already-vivid ones.\n\n"
            "-1.0 = Fully desaturated (grayscale)\n"
            " 0.0 = No change\n"
            "+1.0 = Maximum vibrance boost\n\n"
            "Useful for enhancing subtle nebula colors."
        )
        self.spin_highlights = QDoubleSpinBox()
        self.spin_highlights.setRange(0.5, 1)
        self.spin_highlights.setValue(1.0)
        self.spin_highlights.setSingleStep(0.01)
        self.spin_highlights.setDecimals(3)
        self.spin_highlights.setToolTip(
            "Compress bright highlight regions.\n\n"
            "Reduces intensity of pixels above 75% brightness\n"
            "to recover detail in bright nebula cores.\n\n"
            "1.0 = No compression\n"
            "0.8 = Gentle highlight recovery\n"
            "0.5 = Strong compression\n\n"
            "Apply after other adjustments to tame hot spots."
        )
        post_grid.addWidget(QLabel("Rescale:"), 0, 0)
        post_grid.addWidget(self.spin_rescale, 0, 1)
        post_grid.addWidget(QLabel("Vibrance:"), 1, 0)
        post_grid.addWidget(self.spin_vibrance, 1, 1)
        post_grid.addWidget(QLabel("Highlights:"), 2, 0)
        post_grid.addWidget(self.spin_highlights, 2, 1)
        adv_layout.addLayout(post_grid)

        main.addWidget(adv_box)

        # Column 4: Preview / Image Viewer
        prev_box = QGroupBox("Preview")
        prev_layout = QVBoxLayout(prev_box)
        
        zoom_row = QHBoxLayout()
        self.btn_zoom_out = QPushButton("−")
        self.btn_zoom_out.setToolTip("Zoom out (make image smaller)")
        self.btn_zoom_in = QPushButton("+")
        self.btn_zoom_in.setToolTip("Zoom in (make image larger)")
        self.btn_zoom_fit = QPushButton("Fit")
        self.btn_zoom_fit.setToolTip("Fit entire image in preview area")
        self.btn_zoom_100 = QPushButton("100%")
        self.btn_zoom_100.setToolTip("View at actual pixel size (1:1)")
        zoom_row.addWidget(self.btn_zoom_out)
        zoom_row.addWidget(self.btn_zoom_in)
        zoom_row.addWidget(self.btn_zoom_fit)
        zoom_row.addWidget(self.btn_zoom_100)
        self.chk_autostretch = QCheckBox("Auto stretch")
        self.chk_autostretch.setChecked(True)
        self.chk_autostretch.setToolTip(
            "Apply screen stretch for preview display only.\n\n"
            "Makes linear data visible without affecting\n"
            "the actual output image.\n\n"
            "Turn OFF to see true output levels."
        )
        zoom_row.addWidget(self.chk_autostretch)
        self.lbl_zoom = QLabel("100%")
        self.lbl_zoom.setMinimumWidth(50)
        self.lbl_zoom.setToolTip("Current zoom level")
        zoom_row.addWidget(self.lbl_zoom)
        zoom_row.addStretch()
        prev_layout.addLayout(zoom_row)

        # Use a custom scroll area that handles mouse events properly
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(False)  # We manage size ourselves
        self.scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        self.label = QLabel()
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setScaledContents(False)
        self.scroll.setWidget(self.label)
        
        # Install event filter on scroll area for wheel events
        self.scroll.installEventFilter(self)
        # Install on viewport for mouse pan events
        self.scroll.viewport().installEventFilter(self)
        
        prev_layout.addWidget(self.scroll, 1)
        
        self.lbl_warning = QLabel("")
        self.lbl_warning.setStyleSheet("color: #FFAA00; font-weight: bold;")
        self.lbl_warning.setWordWrap(True)
        self.lbl_warning.setVisible(False)
        prev_layout.addWidget(self.lbl_warning)

        main.addWidget(prev_box, 2)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.btn_load = QPushButton("Load...")
        self.btn_load.setToolTip("Load combination settings from a JSON file")
        self.btn_save = QPushButton("Save...")
        self.btn_save.setToolTip("Save current settings to a JSON file")
        self.btn_reset = QPushButton("Reset")
        self.btn_reset.setToolTip("Reset all settings to default values")
        self.btn_apply = QPushButton("Apply to RGB")
        self.btn_apply.setToolTip(
            "Apply combination to the selected RGB image.\n"
            "Modifies the original image in place."
        )
        self.btn_create = QPushButton("Create New")
        self.btn_create.setToolTip(
            "Create a new image with the combination result.\n"
            "Original RGB image is not modified."
        )
        self.btn_close = QPushButton("Close")
        self.btn_close.setToolTip("Close dialog without applying")
        btn_row.addWidget(self.btn_load)
        btn_row.addWidget(self.btn_save)
        btn_row.addWidget(self.btn_reset)
        btn_row.addWidget(self.btn_apply)
        btn_row.addWidget(self.btn_create)
        btn_row.addWidget(self.btn_close)
        root.addLayout(btn_row)

        # Connections
        self.btn_close.clicked.connect(self.reject)
        self.btn_reset.clicked.connect(self._reset)
        self.btn_apply.clicked.connect(self._apply_to_existing)
        self.btn_create.clicked.connect(self._create_new)
        self.btn_load.clicked.connect(self._load_settings)
        self.btn_save.clicked.connect(self._save_settings)
        self.btn_zoom_out.clicked.connect(lambda: self._set_zoom(self._zoom / 1.25))
        self.btn_zoom_in.clicked.connect(lambda: self._set_zoom(self._zoom * 1.25))
        self.btn_zoom_fit.clicked.connect(self._fit_preview)
        self.btn_zoom_100.clicked.connect(lambda: self._set_zoom(1.0))

        for c in [self.combo_rgb, self.combo_ha, self.combo_oiii, self.combo_sii, 
                  self.combo_star_mask, self.combo_luminance]:
            c.currentIndexChanged.connect(self._reload_sources)
        
        self.combo_method.currentIndexChanged.connect(self._update_control_visibility)
        self.combo_method.currentIndexChanged.connect(self._update_warning_label)
        self.combo_method.currentIndexChanged.connect(self._update_preview)
        
        for c in [self.combo_blend_mode, self.combo_map_r, self.combo_map_g, self.combo_map_b]:
            c.currentIndexChanged.connect(self._update_preview)
        
        # Connect mapping combos to warning update
        for c in [self.combo_map_r, self.combo_map_g, self.combo_map_b]:
            c.currentIndexChanged.connect(self._update_warning_label)
        
        # Connect ALL spinboxes to preview update
        all_spins = [
            self.spin_ha, self.spin_oiii, self.spin_sii, self.spin_morph, 
            self.spin_rgb, self.spin_boost,
            self.spin_rgb_r, self.spin_rgb_g, self.spin_rgb_b,
            self.spin_bp_ha, self.spin_bp_oiii, self.spin_bp_sii, 
            self.spin_star_balance,
            self.spin_lum_strength, self.spin_lum_r_weight, self.spin_lum_g_weight, self.spin_lum_b_weight,
            self.spin_rescale, self.spin_vibrance, self.spin_highlights
        ]
        for s in all_spins:
            s.valueChanged.connect(self._update_preview)
        
        # Connect checkboxes to preview update
        for c in [self.chk_preserve_stars, self.chk_remove_magenta, self.chk_linear_fit,
                  self.chk_neutralize_bg, self.chk_autostretch, self.chk_use_blackpoint,
                  self.chk_apply_luminance, self.chk_preserve_sat]:
            c.toggled.connect(self._update_preview)
        
        # Connect luminance combos to preview
        for c in [self.combo_lum_source, self.combo_lum_mode]:
            c.currentIndexChanged.connect(self._update_preview)

    def _populate_views(self):
        for c in [self.combo_rgb, self.combo_ha, self.combo_oiii, self.combo_sii, 
                  self.combo_star_mask, self.combo_luminance]:
            c.clear()
        self._title_to_doc.clear()
        
        try:
            views = self.ctx.list_image_views()
        except:
            views = []
        
        for c in [self.combo_ha, self.combo_oiii, self.combo_sii, 
                  self.combo_star_mask, self.combo_luminance]:
            c.addItem(_NONE_PLACEHOLDER)
        
        for title, doc in views:
            key = title
            if key in self._title_to_doc:
                n = 2
                while f"{title} ({n})" in self._title_to_doc:
                    n += 1
                key = f"{title} ({n})"
            self._title_to_doc[key] = doc
            self.combo_rgb.addItem(key)
            self.combo_ha.addItem(key)
            self.combo_oiii.addItem(key)
            self.combo_sii.addItem(key)
            self.combo_star_mask.addItem(key)
            self.combo_luminance.addItem(key)
        
        for c in [self.combo_ha, self.combo_oiii, self.combo_sii]:
            if c.count() > 1:
                c.setCurrentIndex(1)
        
        self._reload_sources()

    def _resolve_doc(self, combo):
        key = combo.currentText()
        if key == _NONE_PLACEHOLDER:
            return None, None
        return key, self._title_to_doc.get(key)

    def _reload_sources(self):
        self._preview_rgb = None
        self._preview_ha = None
        self._preview_oiii = None
        self._preview_sii = None
        self._preview_star_mask = None
        self._preview_luminance = None

        _, rgb_doc = self._resolve_doc(self.combo_rgb)
        if rgb_doc is not None:
            try:
                self._preview_rgb = _downsample_for_preview(_ensure_rgb(_to_float01(rgb_doc.image)), 1200)
            except:
                pass

        for attr, combo in [("_preview_ha", self.combo_ha), ("_preview_oiii", self.combo_oiii),
                            ("_preview_sii", self.combo_sii), ("_preview_star_mask", self.combo_star_mask),
                            ("_preview_luminance", self.combo_luminance)]:
            _, doc = self._resolve_doc(combo)
            if doc is not None:
                try:
                    setattr(self, attr, _downsample_for_preview(_ensure_mono(_to_float01(doc.image)), 1200))
                except:
                    pass
        
        self._update_warning_label()
        self._update_preview()

    def _update_control_visibility(self):
        method = self.combo_method.currentIndex()
        
        # SII visibility
        sii_visible = method in (
            CombinationMethod.SHO_RGB,
            CombinationMethod.LRGB_NB,
            CombinationMethod.HOO_SHO_MORPH,
            CombinationMethod.GOLDEN_PALETTE,
            CombinationMethod.CUSTOM_MAP,
        )
        self.spin_sii.setVisible(sii_visible)
        self._control_labels["sii"].setVisible(sii_visible)
        
        # OIII visibility
        oiii_visible = method not in (CombinationMethod.HA_ENHANCE,)
        self.spin_oiii.setVisible(oiii_visible)
        self._control_labels["oiii"].setVisible(oiii_visible)
        
        # RGB Mix visibility
        rgb_visible = method in (
            CombinationMethod.HOO_RGB,
            CombinationMethod.SHO_RGB,
            CombinationMethod.FORAXX,
            CombinationMethod.HOO_SHO_MORPH,
            CombinationMethod.GOLDEN_PALETTE,
            CombinationMethod.CUSTOM_MAP,
        )
        self.spin_rgb.setVisible(rgb_visible)
        self._control_labels["rgb"].setVisible(rgb_visible)
        
        # Morph visibility
        morph_visible = method == CombinationMethod.HOO_SHO_MORPH
        self.spin_morph.setVisible(morph_visible)
        self._control_labels["morph"].setVisible(morph_visible)
        
        # Custom mapping group
        self.mapping_group.setVisible(method == CombinationMethod.CUSTOM_MAP)
        
        # Per-channel RGB weights checkbox
        self.chk_show_rgb_weights.setVisible(rgb_visible)
        if not rgb_visible:
            self.spin_rgb_r.setVisible(False)
            self._control_labels["rgb_r"].setVisible(False)
            self.spin_rgb_g.setVisible(False)
            self._control_labels["rgb_g"].setVisible(False)
            self.spin_rgb_b.setVisible(False)
            self._control_labels["rgb_b"].setVisible(False)
        else:
            self._update_rgb_weights_visibility()

        # Update tooltips based on method
        if method == CombinationMethod.HA_ENHANCE:
            self.spin_ha.setToolTip("Controls how much Hα enhances the red channel.")
            self.spin_oiii.setToolTip("")
            self.spin_sii.setToolTip("")
        elif method == CombinationMethod.HOO_RGB:
            self.spin_ha.setToolTip("Hα contribution to red channel (HOO).")
            self.spin_oiii.setToolTip("OIII contribution to green and blue channels (HOO).")
            self.spin_sii.setToolTip("")
        elif method == CombinationMethod.SHO_RGB:
            self.spin_ha.setToolTip("Hα contribution to green channel (Hubble palette).")
            self.spin_oiii.setToolTip("OIII contribution to blue channel (Hubble palette).")
            self.spin_sii.setToolTip("SII contribution to red channel (Hubble palette).")
        elif method == CombinationMethod.LRGB_NB:
            self.spin_ha.setToolTip("Hα weight in luminance calculation.")
            self.spin_oiii.setToolTip("OIII weight in luminance calculation.")
            self.spin_sii.setToolTip("SII weight in luminance calculation.")
        elif method == CombinationMethod.FORAXX:
            self.spin_ha.setToolTip("Hα contribution (70% of red channel).")
            self.spin_oiii.setToolTip("OIII contribution (30% of red, 100% of green/blue).")
            self.spin_sii.setToolTip("")
        elif method == CombinationMethod.HOO_SHO_MORPH:
            self.spin_ha.setToolTip("Hα channel strength before morphing.")
            self.spin_oiii.setToolTip("OIII channel strength before morphing.")
            self.spin_sii.setToolTip("SII channel strength before morphing.")
            self.spin_morph.setToolTip("0.0 = pure HOO palette, 1.0 = pure SHO palette.")
        elif method == CombinationMethod.GOLDEN_PALETTE:
            self.spin_ha.setToolTip("Hα contribution to red and green channels.")
            self.spin_oiii.setToolTip("OIII contribution to green and blue channels.")
            self.spin_sii.setToolTip("SII contribution to red channel (warm tones).")
        elif method == CombinationMethod.CUSTOM_MAP:
            self.spin_ha.setToolTip("Strength multiplier for Hα when mapped to any channel.")
            self.spin_oiii.setToolTip("Strength multiplier for OIII when mapped to any channel.")
            self.spin_sii.setToolTip("Strength multiplier for SII when mapped to any channel.")
        else:
            self.spin_ha.setToolTip("")
            self.spin_oiii.setToolTip("")
            self.spin_sii.setToolTip("")

    def _update_rgb_weights_visibility(self):
        show = self.chk_show_rgb_weights.isChecked() and self.chk_show_rgb_weights.isVisible()
        self.spin_rgb_r.setVisible(show)
        self._control_labels["rgb_r"].setVisible(show)
        self.spin_rgb_g.setVisible(show)
        self._control_labels["rgb_g"].setVisible(show)
        self.spin_rgb_b.setVisible(show)
        self._control_labels["rgb_b"].setVisible(show)

    def _update_blackpoint_visibility(self):
        show = self.chk_use_blackpoint.isChecked()
        self.bp_grid_widget.setVisible(show)

    def _update_luminance_visibility(self):
        show_lum = self.chk_apply_luminance.isChecked()
        self.lum_controls_widget.setVisible(show_lum)
        
        if show_lum:
            # Show advanced controls if checkbox is checked
            show_adv = self.chk_advanced_lum.isChecked()
            self.adv_lum_widget.setVisible(show_adv)
            
            # Show warning if external L selected but no image loaded
            if self.combo_lum_source.currentIndex() == 1:  # External L
                lum_selected = self.combo_luminance.currentText() != _NONE_PLACEHOLDER
                if not lum_selected:
                    self.lbl_warning.setText("⚠ External L selected but no L image loaded")
                    self.lbl_warning.setVisible(True)
                    return
        
        self._update_warning_label()

    def _update_warning_label(self):
        """Show warning if required images are missing for the selected method."""
        method = self.combo_method.currentIndex()
        
        ha_selected = self.combo_ha.currentText() != _NONE_PLACEHOLDER
        oiii_selected = self.combo_oiii.currentText() != _NONE_PLACEHOLDER
        sii_selected = self.combo_sii.currentText() != _NONE_PLACEHOLDER
        
        missing = []
        
        if method == CombinationMethod.HA_ENHANCE:
            if not ha_selected:
                missing.append("Hα")
                
        elif method in (CombinationMethod.HOO_RGB, CombinationMethod.FORAXX):
            if not ha_selected:
                missing.append("Hα")
            if not oiii_selected:
                missing.append("OIII")
                
        elif method in (CombinationMethod.SHO_RGB, CombinationMethod.HOO_SHO_MORPH,
                        CombinationMethod.GOLDEN_PALETTE):
            if not ha_selected:
                missing.append("Hα")
            if not oiii_selected:
                missing.append("OIII")
            if not sii_selected:
                missing.append("SII")
                
        elif method == CombinationMethod.LRGB_NB:
            if not ha_selected:
                missing.append("Hα")
                
        elif method == CombinationMethod.CUSTOM_MAP:
            r_src = self.combo_map_r.currentIndex()
            g_src = self.combo_map_g.currentIndex()
            b_src = self.combo_map_b.currentIndex()
            
            if ChannelSource.HA in (r_src, g_src, b_src) and not ha_selected:
                missing.append("Hα")
            if ChannelSource.OIII in (r_src, g_src, b_src) and not oiii_selected:
                missing.append("OIII")
            if ChannelSource.SII in (r_src, g_src, b_src) and not sii_selected:
                missing.append("SII")
        
        if missing:
            self.lbl_warning.setText(f"⚠ Missing required images: {', '.join(missing)}")
            self.lbl_warning.setVisible(True)
        else:
            self.lbl_warning.setVisible(False)

    def _apply_preset(self, index):
        """Apply a preset channel mapping."""
        if index <= 0:
            return
        
        preset_idx = index - 1
        if preset_idx < len(MAPPING_PRESETS):
            _, r_src, g_src, b_src = MAPPING_PRESETS[preset_idx]
            self.combo_map_r.setCurrentIndex(r_src)
            self.combo_map_g.setCurrentIndex(g_src)
            self.combo_map_b.setCurrentIndex(b_src)
        
        self.combo_preset.blockSignals(True)
        self.combo_preset.setCurrentIndex(0)
        self.combo_preset.blockSignals(False)

    def _update_preview(self):
        if self._preview_rgb is None:
            self.label.clear()
            return
        
        try:
            # Determine luminance image for preview
            lum_img = None
            if self.chk_apply_luminance.isChecked():
                if self.combo_lum_source.currentIndex() == 1:  # External
                    lum_img = self._preview_luminance
            
            img = combine_rgb_nb(
                self._preview_rgb, self._preview_ha, self._preview_oiii, self._preview_sii,
                method=self.combo_method.currentIndex(),
                ha_strength=self.spin_ha.value(),
                oiii_strength=self.spin_oiii.value(),
                sii_strength=self.spin_sii.value(),
                rgb_weight=self.spin_rgb.value(),
                boost=self.spin_boost.value(),
                do_preserve_stars=self.chk_preserve_stars.isChecked(),
                do_remove_magenta=self.chk_remove_magenta.isChecked(),
                morph_factor=self.spin_morph.value(),
                custom_r_source=self.combo_map_r.currentIndex(),
                custom_g_source=self.combo_map_g.currentIndex(),
                custom_b_source=self.combo_map_b.currentIndex(),
                do_linear_fit=self.chk_linear_fit.isChecked(),
                do_neutralize_bg=self.chk_neutralize_bg.isChecked(),
                rgb_r_weight=self.spin_rgb_r.value(),
                rgb_g_weight=self.spin_rgb_g.value(),
                rgb_b_weight=self.spin_rgb_b.value(),
                blackpoint_ha=self.spin_bp_ha.value(),
                blackpoint_oiii=self.spin_bp_oiii.value(),
                blackpoint_sii=self.spin_bp_sii.value(),
                use_blackpoint=self.chk_use_blackpoint.isChecked(),
                blend_mode=self.combo_blend_mode.currentIndex(),
                external_star_mask=self._preview_star_mask,
                star_balance=self.spin_star_balance.value(),
                apply_luminance=self.chk_apply_luminance.isChecked(),
                luminance_image=lum_img,
                lum_source=self.combo_lum_source.currentIndex(),
                lum_mode=self.combo_lum_mode.currentIndex(),
                lum_strength=self.spin_lum_strength.value(),
                lum_preserve_sat=self.chk_preserve_sat.isChecked(),
                lum_r_weight=self.spin_lum_r_weight.value(),
                lum_g_weight=self.spin_lum_g_weight.value(),
                lum_b_weight=self.spin_lum_b_weight.value(),
                rescale=self.spin_rescale.value(),
                vibrance=self.spin_vibrance.value(),
                highlights=self.spin_highlights.value(),
            )
            
            if img is None:
                return
            
            if self.chk_autostretch.isChecked():
                img = _auto_stretch(img)
            
            img = np.clip(img, 0.0, 1.0)
            qimg = _float_to_qimage_rgb8(img)
            self._pix = QPixmap.fromImage(qimg)
            self._apply_zoom()
            
        except Exception as e:
            print(f"[Preview Error] {e}")
            import traceback
            traceback.print_exc()

    def _apply_zoom(self):
        if self._pix is None:
            return
        new_size = self._pix.size() * self._zoom
        scaled = self._pix.scaled(
            new_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.label.setPixmap(scaled)
        self.label.setFixedSize(scaled.size())
        self.lbl_zoom.setText(f"{int(self._zoom * 100)}%")

    def _set_zoom(self, z):
        old_zoom = self._zoom
        self._zoom = max(0.1, min(z, 10.0))
        if self._pix is None:
            return
        
        # Get scroll position before zoom (center of viewport)
        hbar = self.scroll.horizontalScrollBar()
        vbar = self.scroll.verticalScrollBar()
        vp = self.scroll.viewport()
        
        # Calculate center point in image coordinates
        center_x = (hbar.value() + vp.width() / 2) / old_zoom
        center_y = (vbar.value() + vp.height() / 2) / old_zoom
        
        self._apply_zoom()
        
        # Restore center point after zoom
        new_hval = int(center_x * self._zoom - vp.width() / 2)
        new_vval = int(center_y * self._zoom - vp.height() / 2)
        hbar.setValue(max(0, new_hval))
        vbar.setValue(max(0, new_vval))

    def _fit_preview(self):
        if self._pix is None:
            return
        vp = self.scroll.viewport().size()
        pw, ph = self._pix.width(), self._pix.height()
        if pw == 0 or ph == 0:
            return
        scale = min(vp.width() / pw, vp.height() / ph)
        self._set_zoom(max(0.1, min(scale, 10.0)))

    def eventFilter(self, obj, ev):
        """Handle mouse wheel zoom and click-drag panning."""
        # Handle wheel events on scroll area or viewport
        if ev.type() == QEvent.Type.Wheel:
            if obj is self.scroll or obj is self.scroll.viewport():
                # Ctrl+wheel = zoom, plain wheel = also zoom (more intuitive for image viewers)
                if ev.modifiers() & Qt.KeyboardModifier.ControlModifier or True:
                    delta = ev.angleDelta().y()
                    if delta != 0:
                        factor = 1.15 if delta > 0 else (1.0 / 1.15)
                        
                        # Zoom toward mouse position
                        mouse_pos = ev.position()
                        vp = self.scroll.viewport()
                        hbar = self.scroll.horizontalScrollBar()
                        vbar = self.scroll.verticalScrollBar()
                        
                        # Mouse position in image coordinates before zoom
                        img_x = (hbar.value() + mouse_pos.x()) / self._zoom
                        img_y = (vbar.value() + mouse_pos.y()) / self._zoom
                        
                        old_zoom = self._zoom
                        self._zoom = max(0.1, min(self._zoom * factor, 10.0))
                        
                        if self._zoom != old_zoom:
                            self._apply_zoom()
                            
                            # Adjust scroll to keep mouse over same image point
                            new_hval = int(img_x * self._zoom - mouse_pos.x())
                            new_vval = int(img_y * self._zoom - mouse_pos.y())
                            hbar.setValue(max(0, new_hval))
                            vbar.setValue(max(0, new_vval))
                        
                        ev.accept()
                        return True
        
        # Handle mouse events on viewport for panning
        if obj is self.scroll.viewport():
            if ev.type() == QEvent.Type.MouseButtonPress:
                if ev.button() == Qt.MouseButton.LeftButton or ev.button() == Qt.MouseButton.MiddleButton:
                    self._panning = True
                    self._pan_start = ev.position()
                    self._pan_hval = self.scroll.horizontalScrollBar().value()
                    self._pan_vval = self.scroll.verticalScrollBar().value()
                    self.scroll.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
                    ev.accept()
                    return True
                    
            elif ev.type() == QEvent.Type.MouseMove:
                if self._panning:
                    delta = ev.position() - self._pan_start
                    self.scroll.horizontalScrollBar().setValue(self._pan_hval - int(delta.x()))
                    self.scroll.verticalScrollBar().setValue(self._pan_vval - int(delta.y()))
                    ev.accept()
                    return True
                    
            elif ev.type() == QEvent.Type.MouseButtonRelease:
                if ev.button() == Qt.MouseButton.LeftButton or ev.button() == Qt.MouseButton.MiddleButton:
                    if self._panning:
                        self._panning = False
                        self.scroll.viewport().setCursor(Qt.CursorShape.ArrowCursor)
                        ev.accept()
                        return True
        
        return super().eventFilter(obj, ev)

    def _reset(self):
        self.spin_ha.setValue(0.8)
        self.spin_oiii.setValue(0.6)
        self.spin_sii.setValue(0.4)
        self.spin_morph.setValue(0.5)
        self.spin_rgb.setValue(0.3)
        self.spin_rgb_r.setValue(1.0)
        self.spin_rgb_g.setValue(1.0)
        self.spin_rgb_b.setValue(1.0)
        self.spin_boost.setValue(1.0)
        self.spin_bp_ha.setValue(1.0)
        self.spin_bp_oiii.setValue(1.0)
        self.spin_bp_sii.setValue(1.0)
        self.spin_star_balance.setValue(0.2)
        self.spin_lum_strength.setValue(1.0)
        self.spin_lum_r_weight.setValue(0.3)
        self.spin_lum_g_weight.setValue(0.6)
        self.spin_lum_b_weight.setValue(0.1)
        self.spin_rescale.setValue(1.0)
        self.spin_vibrance.setValue(0.0)
        self.spin_highlights.setValue(1.0)
        self.chk_preserve_stars.setChecked(True)
        self.chk_remove_magenta.setChecked(False)
        self.chk_linear_fit.setChecked(False)
        self.chk_neutralize_bg.setChecked(False)
        self.chk_use_blackpoint.setChecked(False)
        self.chk_show_rgb_weights.setChecked(False)
        self.chk_apply_luminance.setChecked(False)
        self.chk_preserve_sat.setChecked(True)
        self.chk_advanced_lum.setChecked(False)
        self.combo_lum_source.setCurrentIndex(0)
        self.combo_lum_mode.setCurrentIndex(0)
        self.combo_map_r.setCurrentIndex(ChannelSource.SII)
        self.combo_map_g.setCurrentIndex(ChannelSource.HA)
        self.combo_map_b.setCurrentIndex(ChannelSource.OIII)
        self.combo_preset.setCurrentIndex(0)
        self._update_rgb_weights_visibility()
        self._update_luminance_visibility()

    def _get_settings_dict(self):
        """Collect all current settings into a dictionary."""
        rgb_name = self.combo_rgb.currentText()
        ha_name = self.combo_ha.currentText()
        oiii_name = self.combo_oiii.currentText()
        sii_name = self.combo_sii.currentText()
        mask_name = self.combo_star_mask.currentText()
        
        return {
            "_comment": "RGB+Narrowband Combiner v3 Settings",
            "_saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "version": "3.0",
            "image_sources": {
                "_note": "For reference only - not restored on load",
                "RGB": rgb_name if rgb_name != _NONE_PLACEHOLDER else None,
                "Ha": ha_name if ha_name != _NONE_PLACEHOLDER else None,
                "OIII": oiii_name if oiii_name != _NONE_PLACEHOLDER else None,
                "SII": sii_name if sii_name != _NONE_PLACEHOLDER else None,
                "StarMask": mask_name if mask_name != _NONE_PLACEHOLDER else None,
                "Luminance": self.combo_luminance.currentText() if self.combo_luminance.currentText() != _NONE_PLACEHOLDER else None,
            },
            "method": self.combo_method.currentIndex(),
            "method_name": METHOD_NAMES[self.combo_method.currentIndex()],
            "blend_mode": self.combo_blend_mode.currentIndex(),
            "ha_strength": self.spin_ha.value(),
            "oiii_strength": self.spin_oiii.value(),
            "sii_strength": self.spin_sii.value(),
            "morph_factor": self.spin_morph.value(),
            "rgb_mix": self.spin_rgb.value(),
            "rgb_channel_weights": {
                "r": self.spin_rgb_r.value(),
                "g": self.spin_rgb_g.value(),
                "b": self.spin_rgb_b.value(),
            },
            "boost": self.spin_boost.value(),
            "preserve_stars": self.chk_preserve_stars.isChecked(),
            "remove_magenta": self.chk_remove_magenta.isChecked(),
            "linear_fit": self.chk_linear_fit.isChecked(),
            "neutralize_bg": self.chk_neutralize_bg.isChecked(),
            "use_blackpoint": self.chk_use_blackpoint.isChecked(),
            "blackpoints": {
                "ha": self.spin_bp_ha.value(),
                "oiii": self.spin_bp_oiii.value(),
                "sii": self.spin_bp_sii.value(),
            },
            "star_balance": self.spin_star_balance.value(),
            "luminance": {
                "apply": self.chk_apply_luminance.isChecked(),
                "source": self.combo_lum_source.currentIndex(),
                "mode": self.combo_lum_mode.currentIndex(),
                "strength": self.spin_lum_strength.value(),
                "preserve_saturation": self.chk_preserve_sat.isChecked(),
                "synthetic_weights": {
                    "r": self.spin_lum_r_weight.value(),
                    "g": self.spin_lum_g_weight.value(),
                    "b": self.spin_lum_b_weight.value(),
                },
            },
            "post_processing": {
                "rescale": self.spin_rescale.value(),
                "vibrance": self.spin_vibrance.value(),
                "highlights": self.spin_highlights.value(),
            },
            "custom_mapping": {
                "red_source": self.combo_map_r.currentIndex(),
                "red_source_name": CHANNEL_SOURCE_NAMES[self.combo_map_r.currentIndex()],
                "green_source": self.combo_map_g.currentIndex(),
                "green_source_name": CHANNEL_SOURCE_NAMES[self.combo_map_g.currentIndex()],
                "blue_source": self.combo_map_b.currentIndex(),
                "blue_source_name": CHANNEL_SOURCE_NAMES[self.combo_map_b.currentIndex()],
            },
        }

    def _apply_settings_dict(self, settings):
        """Apply settings from a dictionary to the UI controls."""
        widgets = [
            self.combo_method, self.combo_blend_mode,
            self.spin_ha, self.spin_oiii, self.spin_sii, self.spin_morph,
            self.spin_rgb, self.spin_rgb_r, self.spin_rgb_g, self.spin_rgb_b,
            self.spin_boost, self.spin_bp_ha, self.spin_bp_oiii, self.spin_bp_sii,
            self.spin_star_balance, self.spin_lum_strength, 
            self.spin_lum_r_weight, self.spin_lum_g_weight, self.spin_lum_b_weight,
            self.spin_rescale, self.spin_vibrance, self.spin_highlights, 
            self.chk_preserve_stars, self.chk_remove_magenta,
            self.chk_linear_fit, self.chk_neutralize_bg, self.chk_use_blackpoint,
            self.chk_show_rgb_weights, self.chk_apply_luminance, self.chk_preserve_sat,
            self.chk_advanced_lum, self.combo_lum_source, self.combo_lum_mode,
            self.combo_map_r, self.combo_map_g, self.combo_map_b,
        ]
        for w in widgets:
            w.blockSignals(True)

        try:
            if "method" in settings:
                idx = settings["method"]
                if 0 <= idx < len(METHOD_NAMES):
                    self.combo_method.setCurrentIndex(idx)
            if "blend_mode" in settings:
                self.combo_blend_mode.setCurrentIndex(settings["blend_mode"])
            if "ha_strength" in settings:
                self.spin_ha.setValue(float(settings["ha_strength"]))
            if "oiii_strength" in settings:
                self.spin_oiii.setValue(float(settings["oiii_strength"]))
            if "sii_strength" in settings:
                self.spin_sii.setValue(float(settings["sii_strength"]))
            if "morph_factor" in settings:
                self.spin_morph.setValue(float(settings["morph_factor"]))
            if "rgb_mix" in settings:
                self.spin_rgb.setValue(float(settings["rgb_mix"]))
            if "boost" in settings:
                self.spin_boost.setValue(float(settings["boost"]))
            if "rgb_channel_weights" in settings:
                cw = settings["rgb_channel_weights"]
                if "r" in cw:
                    self.spin_rgb_r.setValue(float(cw["r"]))
                if "g" in cw:
                    self.spin_rgb_g.setValue(float(cw["g"]))
                if "b" in cw:
                    self.spin_rgb_b.setValue(float(cw["b"]))
                if cw.get("r", 1.0) != 1.0 or cw.get("g", 1.0) != 1.0 or cw.get("b", 1.0) != 1.0:
                    self.chk_show_rgb_weights.setChecked(True)
            if "preserve_stars" in settings:
                self.chk_preserve_stars.setChecked(bool(settings["preserve_stars"]))
            if "remove_magenta" in settings:
                self.chk_remove_magenta.setChecked(bool(settings["remove_magenta"]))
            if "linear_fit" in settings:
                self.chk_linear_fit.setChecked(bool(settings["linear_fit"]))
            if "neutralize_bg" in settings:
                self.chk_neutralize_bg.setChecked(bool(settings["neutralize_bg"]))
            if "use_blackpoint" in settings:
                self.chk_use_blackpoint.setChecked(bool(settings["use_blackpoint"]))
            if "blackpoints" in settings:
                bp = settings["blackpoints"]
                if "ha" in bp:
                    self.spin_bp_ha.setValue(float(bp["ha"]))
                if "oiii" in bp:
                    self.spin_bp_oiii.setValue(float(bp["oiii"]))
                if "sii" in bp:
                    self.spin_bp_sii.setValue(float(bp["sii"]))
            if "star_balance" in settings:
                self.spin_star_balance.setValue(float(settings["star_balance"]))
            if "luminance" in settings:
                lum = settings["luminance"]
                if "apply" in lum:
                    self.chk_apply_luminance.setChecked(bool(lum["apply"]))
                if "source" in lum:
                    self.combo_lum_source.setCurrentIndex(lum["source"])
                if "mode" in lum:
                    self.combo_lum_mode.setCurrentIndex(lum["mode"])
                if "strength" in lum:
                    self.spin_lum_strength.setValue(float(lum["strength"]))
                if "preserve_saturation" in lum:
                    self.chk_preserve_sat.setChecked(bool(lum["preserve_saturation"]))
                if "synthetic_weights" in lum:
                    sw = lum["synthetic_weights"]
                    if "r" in sw:
                        self.spin_lum_r_weight.setValue(float(sw["r"]))
                    if "g" in sw:
                        self.spin_lum_g_weight.setValue(float(sw["g"]))
                    if "b" in sw:
                        self.spin_lum_b_weight.setValue(float(sw["b"]))
                    # Show advanced controls if non-default weights
                    if sw.get("r", 0.3) != 0.3 or sw.get("g", 0.6) != 0.6 or sw.get("b", 0.1) != 0.1:
                        self.chk_advanced_lum.setChecked(True)
            if "post_processing" in settings:
                pp = settings["post_processing"]
                if "rescale" in pp:
                    self.spin_rescale.setValue(float(pp["rescale"]))
                if "vibrance" in pp:
                    self.spin_vibrance.setValue(float(pp["vibrance"]))
                if "highlights" in pp:
                    self.spin_highlights.setValue(float(pp["highlights"]))
            if "custom_mapping" in settings:
                cm = settings["custom_mapping"]
                if "red_source" in cm:
                    self.combo_map_r.setCurrentIndex(cm["red_source"])
                if "green_source" in cm:
                    self.combo_map_g.setCurrentIndex(cm["green_source"])
                if "blue_source" in cm:
                    self.combo_map_b.setCurrentIndex(cm["blue_source"])
        finally:
            for w in widgets:
                w.blockSignals(False)

        self._update_control_visibility()
        self._update_rgb_weights_visibility()
        self._update_blackpoint_visibility()
        self._update_luminance_visibility()
        self._update_warning_label()
        self._update_preview()

    def _save_settings(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Settings", "rgbnb_v3_settings.json", "JSON Files (*.json)"
        )
        if not file_path:
            return
        try:
            settings = self._get_settings_dict()
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(settings, f, indent=2)
            QMessageBox.information(self, "Saved", f"Settings saved to:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save:\n{e}")

    def _load_settings(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Settings", "", "JSON Files (*.json)"
        )
        if not file_path:
            return
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                settings = json.load(f)
            self._apply_settings_dict(settings)
            QMessageBox.information(self, "Loaded", f"Settings loaded from:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load:\n{e}")

    def _gather_inputs(self):
        rgb_key, rgb_doc = self._resolve_doc(self.combo_rgb)
        if rgb_doc is None:
            raise RuntimeError("Select an RGB image")
        
        _, ha_doc = self._resolve_doc(self.combo_ha)
        _, oiii_doc = self._resolve_doc(self.combo_oiii)
        _, sii_doc = self._resolve_doc(self.combo_sii)
        _, mask_doc = self._resolve_doc(self.combo_star_mask)
        _, lum_doc = self._resolve_doc(self.combo_luminance)
        
        method = self.combo_method.currentIndex()
        
        # Validation
        if method == CombinationMethod.HA_ENHANCE and ha_doc is None:
            raise RuntimeError("Hα image required for Hα Enhancement")
        if method in (CombinationMethod.HOO_RGB, CombinationMethod.FORAXX):
            if ha_doc is None or oiii_doc is None:
                raise RuntimeError("Hα and OIII required for this method")
        if method in (CombinationMethod.SHO_RGB, CombinationMethod.HOO_SHO_MORPH, CombinationMethod.GOLDEN_PALETTE):
            if ha_doc is None or oiii_doc is None or sii_doc is None:
                raise RuntimeError("Hα, OIII, and SII required for this method")
        if method == CombinationMethod.LRGB_NB and ha_doc is None:
            raise RuntimeError("At least Hα required for L-RGB-NB")
        
        # Check luminance if external source selected
        if self.chk_apply_luminance.isChecked() and self.combo_lum_source.currentIndex() == 1:
            if lum_doc is None:
                raise RuntimeError("External L source selected but no L image loaded")
        
        rgb = _to_float01(rgb_doc.image)
        ha = _to_float01(ha_doc.image) if ha_doc else None
        oiii = _to_float01(oiii_doc.image) if oiii_doc else None
        sii = _to_float01(sii_doc.image) if sii_doc else None
        mask = _to_float01(mask_doc.image) if mask_doc else None
        luminance = _to_float01(lum_doc.image) if lum_doc else None
        
        return rgb, ha, oiii, sii, mask, luminance, rgb_doc, rgb_key

    def _do_combine(self, rgb, ha, oiii, sii, mask, luminance):
        return combine_rgb_nb(
            rgb, ha, oiii, sii,
            method=self.combo_method.currentIndex(),
            ha_strength=self.spin_ha.value(),
            oiii_strength=self.spin_oiii.value(),
            sii_strength=self.spin_sii.value(),
            rgb_weight=self.spin_rgb.value(),
            boost=self.spin_boost.value(),
            do_preserve_stars=self.chk_preserve_stars.isChecked(),
            do_remove_magenta=self.chk_remove_magenta.isChecked(),
            morph_factor=self.spin_morph.value(),
            custom_r_source=self.combo_map_r.currentIndex(),
            custom_g_source=self.combo_map_g.currentIndex(),
            custom_b_source=self.combo_map_b.currentIndex(),
            do_linear_fit=self.chk_linear_fit.isChecked(),
            do_neutralize_bg=self.chk_neutralize_bg.isChecked(),
            rgb_r_weight=self.spin_rgb_r.value(),
            rgb_g_weight=self.spin_rgb_g.value(),
            rgb_b_weight=self.spin_rgb_b.value(),
            blackpoint_ha=self.spin_bp_ha.value(),
            blackpoint_oiii=self.spin_bp_oiii.value(),
            blackpoint_sii=self.spin_bp_sii.value(),
            use_blackpoint=self.chk_use_blackpoint.isChecked(),
            blend_mode=self.combo_blend_mode.currentIndex(),
            external_star_mask=_ensure_mono(mask) if mask is not None else None,
            star_balance=self.spin_star_balance.value(),
            apply_luminance=self.chk_apply_luminance.isChecked(),
            luminance_image=luminance,
            lum_source=self.combo_lum_source.currentIndex(),
            lum_mode=self.combo_lum_mode.currentIndex(),
            lum_strength=self.spin_lum_strength.value(),
            lum_preserve_sat=self.chk_preserve_sat.isChecked(),
            lum_r_weight=self.spin_lum_r_weight.value(),
            lum_g_weight=self.spin_lum_g_weight.value(),
            lum_b_weight=self.spin_lum_b_weight.value(),
            rescale=self.spin_rescale.value(),
            vibrance=self.spin_vibrance.value(),
            highlights=self.spin_highlights.value(),
        )

    def _apply_to_existing(self):
        try:
            rgb, ha, oiii, sii, mask, luminance, rgb_doc, _ = self._gather_inputs()
            out = self._do_combine(rgb, ha, oiii, sii, mask, luminance)
            rgb_doc.image[...] = out.astype(rgb_doc.image.dtype)
            QMessageBox.information(self, "Done", "Applied to RGB image")
            self._reload_sources()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _create_new(self):
        try:
            rgb, ha, oiii, sii, mask, luminance, _, rgb_key = self._gather_inputs()
            out = self._do_combine(rgb, ha, oiii, sii, mask, luminance)
            name = f"{rgb_key}_NB"
            self.ctx.open_new_document(out, name=name)
            QMessageBox.information(self, "Done", f"Created: {name}")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))


def run(ctx):
    dlg = RGBNBCombinerDialog(ctx)
    dlg.exec()