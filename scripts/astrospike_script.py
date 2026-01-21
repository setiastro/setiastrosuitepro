"""
AstroSpike Script for SETI Astro
================================
Applies diffraction spikes, halos and soft flares to stars in astrophotography images.

This script opens a full GUI interface with live preview, star detection,
and the ability to add/remove stars manually. Includes save functionality.
"""

SCRIPT_NAME = "AstroSpike - Star Diffraction Spikes"
SCRIPT_GROUP = "Effects"

import numpy as np
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from enum import Enum

# PyQt6 imports (available in SETI Astro)
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout, 
                             QVBoxLayout, QPushButton, QFileDialog, QLabel, QSlider,
                             QScrollArea, QCheckBox, QGroupBox, QDialog)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QPointF, QRectF
from PyQt6.QtGui import (QImage, QPixmap, QPainter, QColor, QPalette, QLinearGradient, 
                         QRadialGradient, QBrush, QPen, QPaintEvent, QMouseEvent, 
                         QWheelEvent, QPainterPath)


# =============================================================================
# DATA TYPES
# =============================================================================

@dataclass
class Color:
    r: float
    g: float
    b: float

@dataclass
class Star:
    x: float
    y: float
    brightness: float
    radius: float
    color: Color

class ToolMode(Enum):
    NONE = 'none'
    ADD = 'add'
    ERASE = 'erase'

@dataclass
class SpikeConfig:
    # Detection
    threshold: int = 100
    star_amount: float = 100.0
    min_star_size: float = 0.0
    max_star_size: float = 100.0
    # Main Spikes
    quantity: int = 4
    length: float = 300.0
    global_scale: float = 1.0
    angle: float = 45.0
    intensity: float = 1.0
    spike_width: float = 1.0
    sharpness: float = 0.5
    # Appearance
    color_saturation: float = 1.0
    hue_shift: float = 0.0
    # Secondary Spikes
    secondary_intensity: float = 0.5
    secondary_length: float = 120.0
    secondary_offset: float = 45.0
    # Soft Flare
    soft_flare_intensity: float = 3.0
    soft_flare_size: float = 15.0
    # Halo
    enable_halo: bool = False
    halo_intensity: float = 0.5
    halo_scale: float = 5.0
    halo_width: float = 1.0
    halo_blur: float = 0.5
    halo_saturation: float = 1.0
    # Rainbow
    enable_rainbow: bool = False
    rainbow_spikes: bool = True
    rainbow_spike_intensity: float = 0.8
    rainbow_spike_frequency: float = 1.0
    rainbow_spike_length: float = 0.8

DEFAULT_CONFIG = SpikeConfig()


# =============================================================================
# STAR DETECTION
# =============================================================================

def map_threshold_to_internal(ui_threshold: int) -> float:
    """Maps UI threshold (1-100) to internal threshold (140-240)."""
    return 140 + (ui_threshold - 1) * (240 - 140) / (100 - 1)


def find_local_peak(lum_data: np.ndarray, x: int, y: int, width: int, height: int) -> Tuple[int, int, float]:
    """Finds the local maximum brightness starting from (x, y)."""
    curr_x, curr_y = x, y
    curr_lum = lum_data[y, x]
    
    for _ in range(20):
        best_lum = curr_lum
        best_x, best_y = curr_x, curr_y
        changed = False
        
        y_min = max(0, curr_y - 1)
        y_max = min(height, curr_y + 2)
        x_min = max(0, curr_x - 1)
        x_max = min(width, curr_x + 2)
        
        window = lum_data[y_min:y_max, x_min:x_max]
        max_val = np.max(window)
        
        if max_val > best_lum:
            local_y, local_x = np.unravel_index(np.argmax(window), window.shape)
            best_x = x_min + local_x
            best_y = y_min + local_y
            best_lum = max_val
            changed = True
        
        if not changed:
            break
            
        curr_x, curr_y = best_x, best_y
        curr_lum = best_lum
        
    return curr_x, curr_y, curr_lum


def flood_fill_star(data: np.ndarray, lum_data: np.ndarray, width: int, height: int, 
                   start_x: int, start_y: int, threshold: int, checked: np.ndarray) -> Optional[Star]:
    """Flood fill to determine star extent and properties."""
    sum_x = 0.0
    sum_y = 0.0
    sum_lum = 0.0
    
    sum_r = 0.0
    sum_g = 0.0
    sum_b = 0.0
    sum_color_weight = 0.0
    
    pixel_count = 0
    max_lum = lum_data[start_y, start_x]
    
    min_x, max_x = start_x, start_x
    min_y, max_y = start_y, start_y
    
    pixel_coords_x = []
    pixel_coords_y = []
    
    stack = [(start_x, start_y)]
    max_pixels = int(1000 + (max_lum / 255.0) * 50000)
    min_lum_ratio = 0.20
    path_min_lum = max_lum
    
    while stack and pixel_count < max_pixels:
        cx, cy = stack.pop()
        
        if cx < 0 or cx >= width or cy < 0 or cy >= height:
            continue
            
        if checked[cy, cx]:
            continue
            
        l = lum_data[cy, cx]
        
        if l > threshold:
            if max_lum > 0 and l < (max_lum * min_lum_ratio):
                continue
            
            checked[cy, cx] = True
            
            if l < path_min_lum:
                path_min_lum = l
            
            min_x = min(min_x, cx)
            max_x = max(max_x, cx)
            min_y = min(min_y, cy)
            max_y = max(max_y, cy)
            
            pixel_coords_x.append(cx)
            pixel_coords_y.append(cy)
            
            sum_x += cx * l
            sum_y += cy * l
            sum_lum += l
            
            pr = float(data[cy, cx, 0])
            pg = float(data[cy, cx, 1])
            pb = float(data[cy, cx, 2])
            
            max_rgb = max(pr, pg, pb)
            min_rgb = min(pr, pg, pb)
            saturation = (max_rgb - min_rgb) / 255.0 if max_rgb > 0 else 0
            
            if pr > 245 and pg > 245 and pb > 245:
                color_weight = 0.01
            else:
                color_weight = (l / 255.0) + saturation * 2.0
            
            sum_r += pr * color_weight
            sum_g += pg * color_weight
            sum_b += pb * color_weight
            sum_color_weight += color_weight
            
            pixel_count += 1
            
            neighbors = [
                (cx + 1, cy), (cx - 1, cy),
                (cx, cy + 1), (cx, cy - 1)
            ]
            
            for nx, ny in neighbors:
                if 0 <= nx < width and 0 <= ny < height:
                    nl = lum_data[ny, nx]
                    valley_climb_tolerance = max(10, path_min_lum * 0.15)
                    if nl > path_min_lum + valley_climb_tolerance:
                        continue
                    stack.append((nx, ny))
            
    if pixel_count == 0:
        return None
    
    # Shape analysis - reject irregular blobs
    if pixel_count >= 10:
        coords_x = np.array(pixel_coords_x, dtype=float)
        coords_y = np.array(pixel_coords_y, dtype=float)
        
        cx = np.mean(coords_x)
        cy = np.mean(coords_y)
        
        dx = coords_x - cx
        dy = coords_y - cy
        
        mu20 = np.mean(dx * dx)
        mu02 = np.mean(dy * dy)
        mu11 = np.mean(dx * dy)
        
        trace = mu20 + mu02
        det = mu20 * mu02 - mu11 * mu11
        discriminant = trace * trace - 4 * det
        
        if discriminant >= 0 and trace > 0:
            sqrt_disc = math.sqrt(discriminant)
            lambda1 = (trace + sqrt_disc) / 2.0
            lambda2 = (trace - sqrt_disc) / 2.0
            
            if lambda2 > 0:
                axis_ratio = math.sqrt(lambda1 / lambda2)
                if axis_ratio > 1.5:
                    return None
    
    # Compactness check
    bbox_width = max_x - min_x + 1
    bbox_height = max_y - min_y + 1
    bbox_area = bbox_width * bbox_height
    
    aspect_ratio = max(bbox_width, bbox_height) / max(min(bbox_width, bbox_height), 1)
    if aspect_ratio > 5.0:
        return None
    
    fill_ratio = pixel_count / max(bbox_area, 1)
    if fill_ratio < 0.10 and pixel_count > 50:
        return None
    
    calculated_radius = math.sqrt(pixel_count / math.pi)
    
    if sum_color_weight > 0:
        avg_r = sum_r / sum_color_weight
        avg_g = sum_g / sum_color_weight
        avg_b = sum_b / sum_color_weight
    else:
        avg_r, avg_g, avg_b = 255, 255, 255
        
    return Star(
        x=sum_x / sum_lum,
        y=sum_y / sum_lum,
        brightness=max_lum / 255.0,
        radius=calculated_radius,
        color=Color(avg_r, avg_g, avg_b)
    )


def sample_halo_color(data: np.ndarray, width: int, height: int, star: Star) -> Color:
    """Sample color from the star's halo region."""
    inner_radius = star.radius * 1.5
    outer_radius = star.radius * 3.0
    
    sum_r = 0.0
    sum_g = 0.0
    sum_b = 0.0
    sample_count = 0
    
    samples = 24
    for i in range(samples):
        angle = (i / samples) * math.pi * 2
        radius = (inner_radius + outer_radius) / 2
        
        x = int(round(star.x + math.cos(angle) * radius))
        y = int(round(star.y + math.sin(angle) * radius))
        
        if 0 <= x < width and 0 <= y < height:
            sum_r += data[y, x, 0]
            sum_g += data[y, x, 1]
            sum_b += data[y, x, 2]
            sample_count += 1
            
    if sample_count == 0:
        return Color(255, 255, 255)
        
    return Color(
        r=sum_r / sample_count,
        g=sum_g / sample_count,
        b=sum_b / sample_count
    )


def detect_stars(image_data: np.ndarray, threshold: int) -> List[Star]:
    """Detect stars in the image using peak finding and flood fill."""
    height, width = image_data.shape[:2]
    internal_threshold = map_threshold_to_internal(threshold)
    
    # Convert to 0-255 range if needed
    if image_data.dtype == np.float32 or image_data.dtype == np.float64:
        if image_data.max() <= 1.0:
            image_data_255 = (image_data * 255).astype(np.uint8)
        else:
            image_data_255 = image_data.astype(np.uint8)
    else:
        image_data_255 = image_data
    
    # Calculate luminance
    r = image_data_255[:, :, 0].astype(float)
    g = image_data_255[:, :, 1].astype(float)
    b = image_data_255[:, :, 2].astype(float)
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    
    stride = 4
    lum_strided = lum[0:height:stride, 0:width:stride]
    cy_indices, cx_indices = np.where(lum_strided > internal_threshold)
    
    unique_peaks: Dict[Tuple[int, int], float] = {}
    
    for i in range(len(cy_indices)):
        y = cy_indices[i] * stride
        x = cx_indices[i] * stride
        
        px, py, plum = find_local_peak(lum, x, y, width, height)
        
        if plum > internal_threshold:
            unique_peaks[(px, py)] = plum
            
    sorted_peaks = sorted(unique_peaks.items(), key=lambda item: item[1], reverse=True)
    
    stars: List[Star] = []
    checked = np.zeros((height, width), dtype=bool)
    
    for (px, py), plum in sorted_peaks:
        if checked[py, px]:
            continue
            
        star = flood_fill_star(image_data_255, lum, width, height, px, py, internal_threshold, checked)
        if star:
            stars.append(star)
            
    # Merge overlapping stars
    stars.sort(key=lambda s: s.radius, reverse=True)
    merged_stars: List[Star] = []
    
    for star in stars:
        merged = False
        for existing in merged_stars:
            dx = star.x - existing.x
            dy = star.y - existing.y
            dist = math.sqrt(dx*dx + dy*dy)
            
            brightness_ratio = star.brightness / max(existing.brightness, 0.01)
            is_much_dimmer = brightness_ratio < 0.4
            is_tiny = star.radius < 5
            
            should_merge = False
            
            if is_much_dimmer or is_tiny:
                if dist < (existing.radius + star.radius) * 1.2:
                    should_merge = True
            else:
                if dist < (existing.radius + star.radius) * 0.25:
                    should_merge = True
            
            if should_merge:
                merged = True
                break
        
        if not merged:
            merged_stars.append(star)
            
    # Sample halo colors
    for star in merged_stars:
        star.color = sample_halo_color(image_data_255, width, height, star)
        
    merged_stars.sort(key=lambda s: s.brightness * s.radius, reverse=True)
    return merged_stars


# =============================================================================
# RENDERING
# =============================================================================

def hsl_to_rgb(h: float, s: float, l: float) -> Tuple[float, float, float]:
    """Convert HSL to RGB (all values 0-1)."""
    if s == 0:
        return l, l, l
    
    def hue_to_rgb(p, q, t):
        if t < 0:
            t += 1
        if t > 1:
            t -= 1
        if t < 1/6:
            return p + (q - p) * 6 * t
        if t < 1/2:
            return q
        if t < 2/3:
            return p + (q - p) * (2/3 - t) * 6
        return p
    
    q = l * (1 + s) if l < 0.5 else l + s - l * s
    p = 2 * l - q
    
    r = hue_to_rgb(p, q, h + 1/3)
    g = hue_to_rgb(p, q, h)
    b = hue_to_rgb(p, q, h - 1/3)
    
    return r, g, b


def get_star_color(star: Star, hue_shift: float, saturation_input: float, alpha: float) -> Tuple[float, float, float, float]:
    """Calculate star color with saturation control. Returns (r, g, b, a) in 0-1 range."""
    r, g, b = star.color.r / 255.0, star.color.g / 255.0, star.color.b / 255.0
    
    # RGB to HSL
    max_c = max(r, g, b)
    min_c = min(r, g, b)
    l = (max_c + min_c) / 2.0
    h = 0.0
    s = 0.0
    
    if max_c != min_c:
        d = max_c - min_c
        s = d / (2.0 - max_c - min_c) if l > 0.5 else d / (max_c + min_c)
        
        if max_c == r:
            h = (g - b) / d + (6.0 if g < b else 0.0)
        elif max_c == g:
            h = (b - r) / d + 2.0
        elif max_c == b:
            h = (r - g) / d + 4.0
        h /= 6.0
    else:
        h = ((star.x * 0.618 + star.y * 0.382) % 1.0)

    new_h = (h * 360.0) + hue_shift
    
    # Saturation logic
    boosted_s = min(1.0, s * 16.0)
    
    if saturation_input <= 1.0:
        final_s = boosted_s * saturation_input
        final_l = max(l, 0.65)
    else:
        hyper_factor = saturation_input - 1.0
        final_s = boosted_s + (1.0 - boosted_s) * hyper_factor
        base_l = max(l, 0.65)
        target_l = 0.5
        final_l = base_l + (target_l - base_l) * hyper_factor
    
    final_s = max(0.0, min(1.0, final_s))
    final_l = max(0.4, min(0.95, final_l))
    final_h = ((new_h % 360.0) / 360.0)
    
    r_out, g_out, b_out = hsl_to_rgb(final_h, final_s, final_l)
    return (r_out, g_out, b_out, alpha)


def create_glow_sprite(size: int = 256) -> np.ndarray:
    """Create a radial gradient glow sprite."""
    sprite = np.zeros((size, size, 4), dtype=np.float32)
    center = size / 2
    
    for y in range(size):
        for x in range(size):
            dx = x - center
            dy = y - center
            dist = math.sqrt(dx*dx + dy*dy) / center
            
            if dist <= 1.0:
                if dist <= 0.2:
                    alpha = 1.0 - (dist / 0.2) * 0.6
                elif dist <= 0.6:
                    alpha = 0.4 - ((dist - 0.2) / 0.4) * 0.35
                else:
                    alpha = 0.05 - ((dist - 0.6) / 0.4) * 0.05
                
                alpha = max(0, alpha)
                sprite[y, x] = [1.0, 1.0, 1.0, alpha]
    
    return sprite


def blend_screen(base: np.ndarray, overlay: np.ndarray, x: int, y: int, opacity: float = 1.0):
    """Apply screen blending mode for overlay at position (x, y)."""
    h, w = overlay.shape[:2]
    bh, bw = base.shape[:2]
    
    # Calculate bounds
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(bw, x + w), min(bh, y + h)
    
    ox1, oy1 = x1 - x, y1 - y
    ox2, oy2 = ox1 + (x2 - x1), oy1 + (y2 - y1)
    
    if x2 <= x1 or y2 <= y1:
        return
    
    base_region = base[y1:y2, x1:x2]
    overlay_region = overlay[oy1:oy2, ox1:ox2]
    
    if overlay.shape[2] == 4:
        alpha = overlay_region[:, :, 3:4] * opacity
        overlay_rgb = overlay_region[:, :, :3]
    else:
        alpha = np.ones((overlay_region.shape[0], overlay_region.shape[1], 1)) * opacity
        overlay_rgb = overlay_region
    
    # Screen blend: 1 - (1-a)(1-b)
    result = 1.0 - (1.0 - base_region) * (1.0 - overlay_rgb * alpha)
    base[y1:y2, x1:x2] = result


def draw_line_gradient(output: np.ndarray, x1: float, y1: float, x2: float, y2: float,
                       color_start: Tuple[float, float, float, float],
                       color_end: Tuple[float, float, float, float],
                       thickness: float, sharpness: float = 0.5):
    """Draw a gradient line with screen blending."""
    height, width = output.shape[:2]
    
    dx = x2 - x1
    dy = y2 - y1
    length = math.sqrt(dx*dx + dy*dy)
    
    if length < 1:
        return
    
    # Normalize direction
    dx /= length
    dy /= length
    
    # Perpendicular for thickness
    px, py = -dy, dx
    
    # Number of steps along the line
    steps = int(length * 2)
    
    for i in range(steps):
        t = i / max(1, steps - 1)
        
        # Position along line
        lx = x1 + dx * length * t
        ly = y1 + dy * length * t
        
        # Interpolate color with sharpness
        if t < sharpness:
            color_t = t / sharpness if sharpness > 0 else 1
            r = color_start[0] * (1 - color_t * 0.2)
            g = color_start[1] * (1 - color_t * 0.2)
            b = color_start[2] * (1 - color_t * 0.2)
            a = color_start[3] * (1 - color_t * 0.2)
        else:
            fade_t = (t - sharpness) / (1 - sharpness) if sharpness < 1 else 0
            r = color_start[0] * 0.8 * (1 - fade_t)
            g = color_start[1] * 0.8 * (1 - fade_t)
            b = color_start[2] * 0.8 * (1 - fade_t)
            a = color_start[3] * 0.8 * (1 - fade_t)
        
        # Draw across thickness
        half_thick = thickness / 2
        for ti in range(-int(half_thick), int(half_thick) + 1):
            px_x = int(lx + px * ti)
            px_y = int(ly + py * ti)
            
            if 0 <= px_x < width and 0 <= px_y < height:
                # Distance from center of line for anti-aliasing
                thick_factor = 1.0 - abs(ti) / (half_thick + 1)
                
                # Screen blend
                final_a = a * thick_factor
                if final_a > 0.001:
                    base = output[px_y, px_x]
                    overlay = np.array([r, g, b]) * final_a
                    output[px_y, px_x] = 1.0 - (1.0 - base) * (1.0 - overlay)


def render_spikes(output: np.ndarray, stars: List[Star], config: SpikeConfig, ctx=None):
    """Render all spike effects onto the output image."""
    height, width = output.shape[:2]
    
    if not stars:
        return
    
    # Apply quantity limit
    limit = int(len(stars) * (config.star_amount / 100.0))
    active_stars = stars[:limit]
    
    # Apply min size filtering
    if config.min_star_size > 0:
        internal_min_size = config.min_star_size * 0.02
        active_stars = [star for star in active_stars if star.radius >= internal_min_size]
    
    # Apply max size filtering
    internal_max_size = 96 + (config.max_star_size * 0.04)
    
    if internal_max_size < 100 and len(active_stars) > 0:
        sorted_by_size = sorted(active_stars, key=lambda s: s.radius, reverse=True)
        removal_percentage = (100 - internal_max_size) / 100.0
        num_to_remove = int(len(sorted_by_size) * removal_percentage)
        
        if num_to_remove > 0:
            stars_to_remove_ids = set(id(star) for star in sorted_by_size[:num_to_remove])
            active_stars = [star for star in active_stars if id(star) not in stars_to_remove_ids]
    
    if ctx:
        ctx.log(f"Processing {len(active_stars)} stars...")
    
    deg_to_rad = math.pi / 180.0
    main_angle_rad = config.angle * deg_to_rad
    sec_angle_rad = (config.angle + config.secondary_offset) * deg_to_rad
    
    # Create glow sprite for soft flare
    glow_sprite = create_glow_sprite(256)
    
    # Render soft flare
    if config.soft_flare_intensity > 0:
        for star in active_stars:
            glow_r = (star.radius * config.soft_flare_size * 0.4 + (star.radius * 2))
            if glow_r > 2:
                opacity = config.soft_flare_intensity * 0.8 * star.brightness
                opacity = min(1.0, opacity)
                
                # Resize glow sprite
                draw_size = int(glow_r * 2)
                if draw_size > 4:
                    # Simple resize using nearest neighbor
                    scale = draw_size / 256
                    resized_glow = np.zeros((draw_size, draw_size, 4), dtype=np.float32)
                    
                    for y in range(draw_size):
                        for x in range(draw_size):
                            src_x = int(x / scale)
                            src_y = int(y / scale)
                            src_x = min(255, src_x)
                            src_y = min(255, src_y)
                            resized_glow[y, x] = glow_sprite[src_y, src_x]
                    
                    # Apply star color tint
                    star_color = get_star_color(star, config.hue_shift, config.color_saturation, 1.0)
                    resized_glow[:, :, 0] *= star_color[0]
                    resized_glow[:, :, 1] *= star_color[1]
                    resized_glow[:, :, 2] *= star_color[2]
                    
                    # Blend
                    x_pos = int(star.x - glow_r)
                    y_pos = int(star.y - glow_r)
                    blend_screen(output, resized_glow, x_pos, y_pos, opacity)
    
    # Render spikes
    for star in active_stars:
        radius_factor = math.pow(star.radius, 1.2)
        base_length = radius_factor * (config.length / 40.0) * config.global_scale
        thickness = max(0.5, star.radius * config.spike_width * 0.15 * config.global_scale)
        
        if base_length < 2:
            continue
            
        color = get_star_color(star, config.hue_shift, config.color_saturation, config.intensity)
        sec_color = get_star_color(star, config.hue_shift, config.color_saturation, config.secondary_intensity)
        
        # Main spikes
        if config.intensity > 0:
            for i in range(config.quantity):
                theta = main_angle_rad + (i * (math.pi * 2) / config.quantity)
                cos_t = math.cos(theta)
                sin_t = math.sin(theta)
                
                start_x = star.x + cos_t * 0.5
                start_y = star.y + sin_t * 0.5
                end_x = star.x + cos_t * base_length
                end_y = star.y + sin_t * base_length
                
                draw_line_gradient(output, start_x, start_y, end_x, end_y,
                                  color, (0, 0, 0, 0), thickness, config.sharpness)
        
        # Secondary spikes
        if config.secondary_intensity > 0:
            sec_len = base_length * (config.secondary_length / config.length)
            for i in range(config.quantity):
                theta = sec_angle_rad + (i * (math.pi * 2) / config.quantity)
                cos_t = math.cos(theta)
                sin_t = math.sin(theta)
                
                start_x = star.x + cos_t * 1.0
                start_y = star.y + sin_t * 1.0
                end_x = star.x + cos_t * sec_len
                end_y = star.y + sin_t * sec_len
                
                draw_line_gradient(output, start_x, start_y, end_x, end_y,
                                  sec_color, (0, 0, 0, 0), thickness * 0.6, config.sharpness)
        
        # Halo
        if config.enable_halo and config.halo_intensity > 0:
            classification_score = star.radius * star.brightness
            intensity_weight = math.pow(min(1.0, classification_score / 10.0), 2)
            
            if intensity_weight > 0.01:
                final_halo_intensity = config.halo_intensity * intensity_weight
                halo_color = get_star_color(star, config.hue_shift, config.halo_saturation, final_halo_intensity)
                
                r_halo = star.radius * config.halo_scale
                if r_halo > 0.5:
                    # Draw halo ring
                    ring_width = r_halo * config.halo_width * 0.15
                    inner_r = max(0.5, r_halo - ring_width / 2)
                    outer_r = r_halo + ring_width / 2
                    
                    for angle in np.linspace(0, 2 * math.pi, 72):
                        for r in np.linspace(inner_r, outer_r, max(1, int(ring_width))):
                            px = int(star.x + math.cos(angle) * r)
                            py = int(star.y + math.sin(angle) * r)
                            
                            if 0 <= px < width and 0 <= py < height:
                                # Distance from ring center for falloff
                                ring_center = (inner_r + outer_r) / 2
                                dist_from_center = abs(r - ring_center) / (ring_width / 2 + 0.1)
                                falloff = max(0, 1 - dist_from_center)
                                falloff *= (1 - config.halo_blur * 0.5)
                                
                                alpha = halo_color[3] * falloff
                                if alpha > 0.001:
                                    overlay = np.array([halo_color[0], halo_color[1], halo_color[2]]) * alpha
                                    output[py, px] = 1.0 - (1.0 - output[py, px]) * (1.0 - overlay)


# =============================================================================
# PyQt6 RENDERER (for GUI preview)
# =============================================================================

class Renderer:
    """PyQt6-based renderer for spike effects."""
    
    def __init__(self):
        self.glow_sprite = self._create_glow_sprite()

    def _create_glow_sprite(self) -> QImage:
        size = 256
        image = QImage(size, size, QImage.Format.Format_ARGB32_Premultiplied)
        image.fill(Qt.GlobalColor.transparent)
        
        painter = QPainter(image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        half = size / 2
        grad = QRadialGradient(half, half, half)
        grad.setColorAt(0, QColor(255, 255, 255, 255))
        grad.setColorAt(0.2, QColor(255, 255, 255, 100))
        grad.setColorAt(0.6, QColor(255, 255, 255, 13))
        grad.setColorAt(1, QColor(255, 255, 255, 0))
        
        painter.setBrush(QBrush(grad))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRect(0, 0, size, size)
        painter.end()
        return image

    def get_star_color(self, star: Star, hue_shift: float, saturation_input: float, alpha: float) -> QColor:
        r, g, b = int(star.color.r), int(star.color.g), int(star.color.b)
        r1, g1, b1 = r / 255.0, g / 255.0, b / 255.0
        max_c = max(r1, g1, b1)
        min_c = min(r1, g1, b1)
        l = (max_c + min_c) / 2.0
        h = 0.0
        s = 0.0
        
        if max_c != min_c:
            d = max_c - min_c
            s = d / (2.0 - max_c - min_c) if l > 0.5 else d / (max_c + min_c)
            if max_c == r1:
                h = (g1 - b1) / d + (6.0 if g1 < b1 else 0.0)
            elif max_c == g1:
                h = (b1 - r1) / d + 2.0
            elif max_c == b1:
                h = (r1 - g1) / d + 4.0
            h /= 6.0
        else:
            h = ((star.x * 0.618 + star.y * 0.382) % 1.0)

        new_h = (h * 360.0) + hue_shift
        boosted_s = min(1.0, s * 16.0)
        
        if saturation_input <= 1.0:
            final_s = boosted_s * saturation_input
            final_l = max(l, 0.65)
        else:
            hyper_factor = saturation_input - 1.0
            final_s = boosted_s + (1.0 - boosted_s) * hyper_factor
            base_l = max(l, 0.65)
            final_l = base_l + (0.5 - base_l) * hyper_factor
        
        final_s = max(0.0, min(1.0, final_s))
        final_l = max(0.4, min(0.95, final_l))
        final_h = (new_h % 360.0) / 360.0
        
        return QColor.fromHslF(final_h, final_s, final_l, alpha)

    def render(self, painter: QPainter, width: int, height: int, stars: List[Star], config: SpikeConfig):
        if not stars:
            return

        limit = int(len(stars) * (config.star_amount / 100.0))
        active_stars = stars[:limit]
        
        if config.min_star_size > 0:
            internal_min_size = config.min_star_size * 0.02
            active_stars = [star for star in active_stars if star.radius >= internal_min_size]
        
        internal_max_size = 96 + (config.max_star_size * 0.04)
        if internal_max_size < 100 and len(active_stars) > 0:
            sorted_by_size = sorted(active_stars, key=lambda s: s.radius, reverse=True)
            removal_percentage = (100 - internal_max_size) / 100.0
            num_to_remove = int(len(sorted_by_size) * removal_percentage)
            if num_to_remove > 0:
                stars_to_remove_ids = set(id(star) for star in sorted_by_size[:num_to_remove])
                active_stars = [star for star in active_stars if id(star) not in stars_to_remove_ids]
        
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Screen)
        
        deg_to_rad = math.pi / 180.0
        main_angle_rad = config.angle * deg_to_rad
        sec_angle_rad = (config.angle + config.secondary_offset) * deg_to_rad
        
        # Soft Flare
        if config.soft_flare_intensity > 0:
            for star in active_stars:
                glow_r = (star.radius * config.soft_flare_size * 0.4 + (star.radius * 2))
                if glow_r > 2:
                    draw_size = glow_r * 2
                    opacity = config.soft_flare_intensity * 0.8 * star.brightness
                    painter.setOpacity(min(1.0, opacity))
                    target_rect = QRectF(star.x - glow_r, star.y - glow_r, draw_size, draw_size)
                    painter.drawImage(target_rect, self.glow_sprite, QRectF(self.glow_sprite.rect()))
            painter.setOpacity(1.0)

        # Spikes
        for star in active_stars:
            radius_factor = math.pow(star.radius, 1.2)
            base_length = radius_factor * (config.length / 40.0) * config.global_scale
            thickness = max(0.5, star.radius * config.spike_width * 0.15 * config.global_scale)
            
            if base_length < 2:
                continue
                
            color = self.get_star_color(star, config.hue_shift, config.color_saturation, config.intensity)
            sec_color = self.get_star_color(star, config.hue_shift, config.color_saturation, config.secondary_intensity)
            
            # Main Spikes
            if config.intensity > 0:
                rainbow_str = config.rainbow_spike_intensity if (config.enable_rainbow and config.rainbow_spikes) else 0
                
                for i in range(int(config.quantity)):
                    theta = main_angle_rad + (i * (math.pi * 2) / config.quantity)
                    cos_t = math.cos(theta)
                    sin_t = math.sin(theta)
                    
                    start_x = star.x + cos_t * 0.5
                    start_y = star.y + sin_t * 0.5
                    end_x = star.x + cos_t * base_length
                    end_y = star.y + sin_t * base_length
                    
                    # 1. Standard Spike (dimmed if rainbow enabled)
                    if rainbow_str > 0:
                        painter.setOpacity(0.4)
                    
                    grad = QLinearGradient(star.x, star.y, end_x, end_y)
                    grad.setColorAt(0, color)
                    fade_point = max(0.0, min(0.99, config.sharpness))
                    if fade_point > 0:
                        c_mid = QColor(color)
                        c_mid.setAlphaF(min(1.0, config.intensity * 0.8))
                        grad.setColorAt(fade_point, c_mid)
                    c_end = QColor(int(star.color.r), int(star.color.g), int(star.color.b), 0)
                    grad.setColorAt(1, c_end)
                    
                    pen = QPen(QBrush(grad), thickness)
                    pen.setCapStyle(Qt.PenCapStyle.FlatCap)
                    painter.setPen(pen)
                    painter.drawLine(QPointF(start_x, start_y), QPointF(end_x, end_y))
                    
                    if rainbow_str > 0:
                        painter.setOpacity(1.0)
                    
                    # 2. Rainbow Overlay (IF ENABLED)
                    if rainbow_str > 0:
                        r_grad = QLinearGradient(star.x, star.y, end_x, end_y)
                        r_grad.setColorAt(0, color)
                        
                        stops = 10
                        for s in range(1, stops + 1):
                            pos = s / stops
                            if pos > config.rainbow_spike_length:
                                break
                            
                            hue = (pos * 360.0 * config.rainbow_spike_frequency) % 360.0
                            a = min(1.0, config.intensity * rainbow_str * 2.0) * (1.0 - pos)
                            c = QColor.fromHslF(hue / 360.0, 0.8, 0.6, min(1.0, a))
                            r_grad.setColorAt(pos, c)
                        
                        r_grad.setColorAt(1, QColor(0, 0, 0, 0))
                        
                        r_pen = QPen(QBrush(r_grad), thickness)
                        r_pen.setCapStyle(Qt.PenCapStyle.FlatCap)
                        painter.setPen(r_pen)
                        painter.drawLine(QPointF(start_x, start_y), QPointF(end_x, end_y))
                               
            # Secondary Spikes
            if config.secondary_intensity > 0:
                sec_len = base_length * (config.secondary_length / config.length)
                for i in range(int(config.quantity)):
                    theta = sec_angle_rad + (i * (math.pi * 2) / config.quantity)
                    cos_t = math.cos(theta)
                    sin_t = math.sin(theta)
                    
                    start_x = star.x + cos_t * 1.0
                    start_y = star.y + sin_t * 1.0
                    end_x = star.x + cos_t * sec_len
                    end_y = star.y + sin_t * sec_len
                    
                    grad = QLinearGradient(star.x, star.y, end_x, end_y)
                    grad.setColorAt(0, sec_color)
                    grad.setColorAt(1, QColor(0, 0, 0, 0))
                    
                    pen = QPen(QBrush(grad), thickness * 0.6)
                    pen.setCapStyle(Qt.PenCapStyle.FlatCap)
                    painter.setPen(pen)
                    painter.drawLine(QPointF(start_x, start_y), QPointF(end_x, end_y))
                               
            # Halo
            if config.enable_halo and config.halo_intensity > 0:
                classification_score = star.radius * star.brightness
                intensity_weight = math.pow(min(1.0, classification_score / 10.0), 2)
                
                if intensity_weight > 0.01:
                    final_halo_intensity = config.halo_intensity * intensity_weight
                    halo_color = self.get_star_color(star, config.hue_shift, config.halo_saturation, final_halo_intensity)
                    
                    r_halo = star.radius * config.halo_scale
                    if r_halo > 0.5:
                        blur_expand = config.halo_blur * 20.0
                        relative_width = r_halo * (config.halo_width * 0.15)
                        inner_r = max(0.0, r_halo - relative_width/2.0)
                        outer_r = r_halo + relative_width/2.0
                        draw_outer = outer_r + blur_expand
                        
                        grad = QRadialGradient(star.x, star.y, draw_outer)
                        stop_start = inner_r / draw_outer
                        stop_end = outer_r / draw_outer
                        
                        grad.setColorAt(0, QColor(0,0,0,0))
                        grad.setColorAt(max(0, stop_start - 0.05), QColor(0,0,0,0))
                        grad.setColorAt((stop_start + stop_end)/2, halo_color)
                        grad.setColorAt(min(1, stop_end + 0.05), QColor(0,0,0,0))
                        grad.setColorAt(1, QColor(0,0,0,0))
                        
                        painter.setBrush(QBrush(grad))
                        painter.setPen(Qt.PenStyle.NoPen)
                        painter.drawEllipse(QPointF(star.x, star.y), draw_outer, draw_outer)


# =============================================================================
# MAIN SCRIPT ENTRY POINT
# =============================================================================

def run(ctx):
    """Main entry point for SETI Astro script - Opens GUI interface."""
    ctx.log("AstroSpike - Opening interface...")
    
    # Get the active image
    img = ctx.get_image()
    if img is None:
        ctx.log("Error: No active image found. Please open an image first.")
        return
    
    # Ensure image is float32 and in 0-1 range
    if img.dtype != np.float32:
        img = img.astype(np.float32)
    
    if img.max() > 1.0:
        img = img / 255.0
    
    # Handle grayscale images
    if len(img.shape) == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.shape[2] == 1:
        img = np.concatenate([img, img, img], axis=-1)
    
    # Ensure we have RGB (not RGBA)
    if img.shape[2] == 4:
        img = img[:, :, :3]
    
    ctx.log(f"Image size: {img.shape[1]}x{img.shape[0]}")
    
    # Convert to 0-255 uint8 for QImage
    img_255 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    
    # Create and show the GUI window
    window = AstroSpikeWindow(img_255, img, ctx)
    window.exec()  # Modal dialog
    
    ctx.log("AstroSpike completed.")


# =============================================================================
# UI CLASSES
# =============================================================================

class StarDetectionThread(QThread):
    """Thread for star detection to keep UI responsive."""
    stars_detected = pyqtSignal(list)

    def __init__(self, image_data, threshold):
        super().__init__()
        self.image_data = image_data
        self.threshold = threshold

    def run(self):
        stars = detect_stars(self.image_data, self.threshold)
        self.stars_detected.emit(stars)


class AstroSpikeWindow(QDialog):
    """Main AstroSpike GUI window."""
    
    def __init__(self, image_data_255: np.ndarray, image_data_float: np.ndarray, ctx):
        super().__init__()
        self.setWindowTitle("AstroSpike - Star Diffraction Spikes")
        self.resize(1200, 800)
        self.setModal(True)
        
        self.ctx = ctx
        self.image_data = image_data_255  # uint8 for detection
        self.image_data_float = image_data_float  # float for output
        self.config = DEFAULT_CONFIG
        self.thread = None
        
        # Tool state
        self.tool_mode = ToolMode.NONE
        self.star_input_radius = 4.0
        self.eraser_input_size = 20.0
        
        # History management
        self.history: List[List[Star]] = []
        self.history_index = -1
        
        # Debounce timer
        self.detect_timer = QTimer()
        self.detect_timer.setSingleShot(True)
        self.detect_timer.setInterval(200)
        self.detect_timer.timeout.connect(self.detect_stars)
        
        # Convert numpy to QImage
        height, width = self.image_data.shape[:2]
        bytes_per_line = 3 * width
        self.qimage = QImage(self.image_data.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        
        self._init_ui()
        self._apply_dark_theme()
        
        # Auto-detect stars on open
        QTimer.singleShot(100, self.detect_stars)
        
    def _apply_dark_theme(self):
        """Apply dark theme to window."""
        dark_palette = QPalette()
        dark_palette.setColor(QPalette.ColorRole.Window, QColor(30, 30, 30))
        dark_palette.setColor(QPalette.ColorRole.WindowText, QColor(224, 224, 224))
        dark_palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
        dark_palette.setColor(QPalette.ColorRole.AlternateBase, QColor(45, 45, 48))
        dark_palette.setColor(QPalette.ColorRole.Text, QColor(224, 224, 224))
        dark_palette.setColor(QPalette.ColorRole.Button, QColor(45, 45, 48))
        dark_palette.setColor(QPalette.ColorRole.ButtonText, QColor(224, 224, 224))
        dark_palette.setColor(QPalette.ColorRole.Highlight, QColor(0, 122, 204))
        dark_palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
        self.setPalette(dark_palette)
        
    def _init_ui(self):
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        
        # Top Toolbar
        top_bar = QWidget()
        top_bar.setObjectName("topBar")
        top_bar.setFixedHeight(50)
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(10, 0, 10, 0)
        top_layout.setSpacing(15)
        
        # Apply to Document button
        btn_apply = QPushButton("✓ Apply to Document")
        btn_apply.setToolTip("Apply effect and close")
        btn_apply.clicked.connect(self.apply_to_document)
        btn_apply.setStyleSheet("background: #007acc; color: white; font-weight: bold; padding: 8px 16px;")
        
        btn_save = QPushButton("💾 Save Image")
        btn_save.setToolTip("Save Image to File")
        btn_save.clicked.connect(self.save_image)
        
        top_layout.addWidget(btn_apply)
        top_layout.addWidget(btn_save)
        
        # Separator
        line1 = QWidget()
        line1.setFixedWidth(1)
        line1.setStyleSheet("background: #444;")
        top_layout.addWidget(line1)
        
        # Undo/Redo
        self.btn_undo = QPushButton("↩️ Undo")
        self.btn_undo.clicked.connect(self.undo)
        self.btn_undo.setEnabled(False)
        
        self.btn_redo = QPushButton("↪️ Redo")
        self.btn_redo.clicked.connect(self.redo)
        self.btn_redo.setEnabled(False)
        
        top_layout.addWidget(self.btn_undo)
        top_layout.addWidget(self.btn_redo)
        
        # Separator
        line2 = QWidget()
        line2.setFixedWidth(1)
        line2.setStyleSheet("background: #444;")
        top_layout.addWidget(line2)
        
        # Tools
        lbl_tools = QLabel("Tools:")
        lbl_tools.setStyleSheet("color: #888; font-weight: bold;")
        top_layout.addWidget(lbl_tools)
        
        self.btn_pan = QPushButton("✋ Pan")
        self.btn_pan.setCheckable(True)
        self.btn_pan.setChecked(True)
        self.btn_pan.clicked.connect(lambda: self.set_tool_mode(ToolMode.NONE))
        
        self.btn_brush_add = QPushButton("⭐ Add Star")
        self.btn_brush_add.setCheckable(True)
        self.btn_brush_add.clicked.connect(lambda: self.set_tool_mode(ToolMode.ADD))
        
        self.btn_eraser = QPushButton("🧹 Eraser")
        self.btn_eraser.setCheckable(True)
        self.btn_eraser.clicked.connect(lambda: self.set_tool_mode(ToolMode.ERASE))
        
        top_layout.addWidget(self.btn_pan)
        top_layout.addWidget(self.btn_brush_add)
        top_layout.addWidget(self.btn_eraser)
        
        # Tool Size Controls
        line_tool_sep = QWidget()
        line_tool_sep.setFixedWidth(1)
        line_tool_sep.setStyleSheet("background: #444;")
        top_layout.addWidget(line_tool_sep)
        
        lbl_star_size = QLabel("Star Size:")
        lbl_star_size.setStyleSheet("color: #888;")
        top_layout.addWidget(lbl_star_size)
        
        self.slider_star_size = QSlider(Qt.Orientation.Horizontal)
        self.slider_star_size.setRange(1, 50)
        self.slider_star_size.setValue(int(self.star_input_radius))
        self.slider_star_size.setFixedWidth(80)
        self.slider_star_size.valueChanged.connect(self.on_star_size_changed)
        top_layout.addWidget(self.slider_star_size)
        
        self.lbl_star_size_val = QLabel(f"{self.star_input_radius:.0f}")
        self.lbl_star_size_val.setFixedWidth(25)
        top_layout.addWidget(self.lbl_star_size_val)
        
        lbl_eraser_size = QLabel("Eraser:")
        lbl_eraser_size.setStyleSheet("color: #888;")
        top_layout.addWidget(lbl_eraser_size)
        
        self.slider_eraser_size = QSlider(Qt.Orientation.Horizontal)
        self.slider_eraser_size.setRange(5, 100)
        self.slider_eraser_size.setValue(int(self.eraser_input_size))
        self.slider_eraser_size.setFixedWidth(80)
        self.slider_eraser_size.valueChanged.connect(self.on_eraser_size_changed)
        top_layout.addWidget(self.slider_eraser_size)
        
        self.lbl_eraser_size_val = QLabel(f"{self.eraser_input_size:.0f}")
        self.lbl_eraser_size_val.setFixedWidth(25)
        top_layout.addWidget(self.lbl_eraser_size_val)
        
        # Separator
        line3 = QWidget()
        line3.setFixedWidth(1)
        line3.setStyleSheet("background: #444;")
        top_layout.addWidget(line3)
        
        # Zoom Controls
        btn_zoom_in = QPushButton("➕")
        btn_zoom_out = QPushButton("➖")
        btn_fit = QPushButton("⛶ Fit")
        
        top_layout.addWidget(btn_zoom_in)
        top_layout.addWidget(btn_zoom_out)
        top_layout.addWidget(btn_fit)
        
        top_layout.addStretch()
        
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #aaa;")
        top_layout.addWidget(self.status_label)
        
        root_layout.addWidget(top_bar)
        
        # Content Area
        content_area = QWidget()
        content_layout = QHBoxLayout(content_area)
        content_layout.setContentsMargins(0, 0, 0, 0)
        
        # Canvas
        self.canvas = CanvasPreview()
        self.canvas.stars_updated.connect(self.on_stars_updated)
        self.canvas.set_image(self.qimage)
        content_layout.addWidget(self.canvas, stretch=1)
        
        # Connect Zoom
        btn_zoom_in.clicked.connect(self.canvas.zoom_in)
        btn_zoom_out.clicked.connect(self.canvas.zoom_out)
        btn_fit.clicked.connect(self.canvas.fit_to_view)
        
        # Controls Panel
        self.controls = ControlPanel(self.config)
        self.controls.setFixedWidth(340)
        self.controls.config_changed.connect(self.on_config_changed)
        self.controls.reset_requested.connect(self.reset_config)
        
        controls_container = QWidget()
        controls_container.setObjectName("controlsContainer")
        controls_layout = QVBoxLayout(controls_container)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.addWidget(self.controls)
        
        content_layout.addWidget(controls_container)
        
        root_layout.addWidget(content_area)
        
        # Style
        self.setStyleSheet("""
            QDialog { background-color: #1e1e1e; }
            QWidget { color: #e0e0e0; font-family: 'Segoe UI', sans-serif; font-size: 13px; }
            #topBar { background-color: #252526; border-bottom: 1px solid #333; }
            #controlsContainer { background-color: #252526; border-left: 1px solid #333; }
            QPushButton { 
                background-color: transparent; border: 1px solid transparent; 
                padding: 6px 12px; border-radius: 4px; color: #ccc;
            }
            QPushButton:hover { background-color: #3e3e42; color: white; }
            QPushButton:pressed { background-color: #007acc; color: white; }
            QPushButton:checked { background-color: #007acc; color: white; border: 1px solid #005a9e; }
            QGroupBox { 
                font-weight: bold; border: 1px solid #3e3e42; margin-top: 16px; 
                padding-top: 16px; border-radius: 4px; background: #2d2d30; 
            }
            QGroupBox::title { 
                subcontrol-origin: margin; subcontrol-position: top left;
                left: 10px; top: 0px; padding: 0 5px; color: #007acc; 
            }
            QSlider::groove:horizontal { border: 1px solid #3e3e42; height: 4px; background: #1e1e1e; margin: 2px 0; border-radius: 2px; }
            QSlider::handle:horizontal { background: #007acc; border: 1px solid #007acc; width: 14px; height: 14px; margin: -6px 0; border-radius: 7px; }
            QScrollArea { border: none; background: transparent; }
            QScrollBar:vertical { border: none; background: #1e1e1e; width: 10px; margin: 0; }
            QScrollBar::handle:vertical { background: #424242; min-height: 20px; border-radius: 5px; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
        """)
        
    def detect_stars(self):
        if self.image_data is None:
            return
            
        if self.thread and self.thread.isRunning():
            self.detect_timer.start(100)
            return

        self.status_label.setText("Detecting stars...")
        self.thread = StarDetectionThread(self.image_data, self.config.threshold)
        self.thread.stars_detected.connect(self.on_stars_detected)
        self.thread.start()

    def on_stars_detected(self, stars):
        self.status_label.setText(f"Found {len(stars)} stars")
        self.canvas.set_stars(stars)
        self.canvas.set_config(self.config)
        self._reset_history(stars)

    def on_config_changed(self, config):
        current_threshold = self.thread.threshold if self.thread else -1
        if config.threshold != current_threshold:
            self.detect_timer.start()
        self.config = config
        self.canvas.set_config(config)

    def reset_config(self):
        self.config = SpikeConfig()
        self.controls.set_config(self.config)
        self.canvas.set_config(self.config)
        self.detect_stars()
    
    def set_tool_mode(self, mode: ToolMode):
        self.tool_mode = mode
        self.canvas.set_tool_mode(mode)
        self.btn_pan.setChecked(mode == ToolMode.NONE)
        self.btn_brush_add.setChecked(mode == ToolMode.ADD)
        self.btn_eraser.setChecked(mode == ToolMode.ERASE)
    
    def on_star_size_changed(self, value: int):
        self.star_input_radius = float(value)
        self.lbl_star_size_val.setText(f"{value}")
        self.canvas.set_star_input_radius(self.star_input_radius)
    
    def on_eraser_size_changed(self, value: int):
        self.eraser_input_size = float(value)
        self.lbl_eraser_size_val.setText(f"{value}")
        self.canvas.set_eraser_input_size(self.eraser_input_size)
    
    def on_stars_updated(self, new_stars: list, push_history: bool):
        self.canvas.stars = new_stars
        if push_history:
            self.history = self.history[:self.history_index + 1]
            self.history.append(list(new_stars))
            self.history_index += 1
        self._update_history_buttons()
        self.canvas.update()
    
    def _update_history_buttons(self):
        self.btn_undo.setEnabled(self.history_index > 0)
        self.btn_redo.setEnabled(self.history_index < len(self.history) - 1)
    
    def undo(self):
        if self.history_index > 0:
            self.history_index -= 1
            self.canvas.stars = list(self.history[self.history_index])
            self.canvas.update()
            self._update_history_buttons()
    
    def redo(self):
        if self.history_index < len(self.history) - 1:
            self.history_index += 1
            self.canvas.stars = list(self.history[self.history_index])
            self.canvas.update()
            self._update_history_buttons()
    
    def _reset_history(self, initial_stars: list):
        self.history = [list(initial_stars)]
        self.history_index = 0
        self._update_history_buttons()

    def save_image(self):
        """Save rendered image to file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Image", "astrospike_output.png", 
            "PNG Images (*.png);;JPEG Images (*.jpg);;TIFF Images (*.tif)"
        )
        if file_path:
            final_image = self.qimage.copy()
            painter = QPainter(final_image)
            self.canvas.renderer.render(painter, final_image.width(), final_image.height(), 
                                        self.canvas.stars, self.config)
            painter.end()
            final_image.save(file_path)
            self.status_label.setText(f"Saved to {file_path}")
            self.ctx.log(f"Saved image to {file_path}")

    def apply_to_document(self):
        """Apply the effect to the SETI Astro document and close."""
        self.status_label.setText("Applying to document...")
        
        # Render to numpy array
        output = self.image_data_float.copy()
        render_spikes(output, self.canvas.stars, self.config)
        output = np.clip(output, 0.0, 1.0)
        
        # Apply to document
        self.ctx.set_image(output, step_name="AstroSpike Effect")
        self.ctx.log(f"Applied AstroSpike effect with {len(self.canvas.stars)} stars")
        
        self.accept()  # Close dialog

    def closeEvent(self, event):
        if self.thread and self.thread.isRunning():
            self.thread.terminate()
            self.thread.wait()
        event.accept()


class SliderControl(QWidget):
    """Slider control with label and value display."""
    value_changed = pyqtSignal(float)

    def __init__(self, label: str, min_val: float, max_val: float, step: float, initial: float, unit: str = ""):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.step = step
        self.unit = unit
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 5, 0, 5)
        layout.setSpacing(2)
        
        header = QHBoxLayout()
        self.label = QLabel(label)
        self.value_label = QLabel(f"{initial:.2f}{unit}")
        header.addWidget(self.label)
        header.addStretch()
        header.addWidget(self.value_label)
        
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 1000)
        self.slider.setValue(self._float_to_int(initial))
        self.slider.valueChanged.connect(self._on_slider_change)
        
        layout.addLayout(header)
        layout.addWidget(self.slider)
        
    def _float_to_int(self, val: float) -> int:
        ratio = (val - self.min_val) / (self.max_val - self.min_val)
        return int(ratio * 1000)
        
    def _int_to_float(self, val: int) -> float:
        ratio = val / 1000.0
        return self.min_val + ratio * (self.max_val - self.min_val)
        
    def _on_slider_change(self, val: int):
        f_val = self._int_to_float(val)
        if self.step > 0:
            f_val = round(f_val / self.step) * self.step
        self.value_label.setText(f"{f_val:.2f}{self.unit}")
        self.value_changed.emit(f_val)
        
    def set_value(self, val: float):
        self.slider.blockSignals(True)
        self.slider.setValue(self._float_to_int(val))
        self.value_label.setText(f"{val:.2f}{self.unit}")
        self.slider.blockSignals(False)


class ControlPanel(QWidget):
    """Control panel with all spike parameters."""
    config_changed = pyqtSignal(SpikeConfig)
    reset_requested = pyqtSignal()

    def __init__(self, config: SpikeConfig):
        super().__init__()
        self.config = config
        self._init_ui()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(10, 10, 10, 0)
        
        lbl_title = QLabel("PARAMETERS")
        lbl_title.setStyleSheet("font-weight: bold; color: #888; letter-spacing: 1px;")
        header_layout.addWidget(lbl_title)
        header_layout.addStretch()
        
        btn_reset = QPushButton("↺ Reset")
        btn_reset.setStyleSheet("background: #333; border: 1px solid #555; padding: 4px 8px; font-size: 11px;")
        btn_reset.clicked.connect(self.reset_requested.emit)
        header_layout.addWidget(btn_reset)
        
        main_layout.addLayout(header_layout)
        
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        self.content = QWidget()
        self.layout = QVBoxLayout(self.content)
        self.layout.setSpacing(15)
        
        self._build_controls()
        
        self.scroll.setWidget(self.content)
        main_layout.addWidget(self.scroll)

    def _build_controls(self):
        while self.layout.count():
            child = self.layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        self._add_group("Detection", [
            ("Threshold", 1, 100, 1, self.config.threshold, "threshold", ""),
            ("Quantity Limit %", 0, 100, 1, self.config.star_amount, "star_amount", "%"),
            ("Min Star Size", 0, 100, 1, self.config.min_star_size, "min_star_size", ""),
            ("Max Star Size", 0, 100, 1, self.config.max_star_size, "max_star_size", "")
        ])
        
        self._add_group("Geometry", [
            ("Global Scale", 0.2, 3.0, 0.1, self.config.global_scale, "global_scale", ""),
            ("Points", 2, 8, 1, self.config.quantity, "quantity", ""),
            ("Length", 10, 1500, 10, self.config.length, "length", ""),
            ("Angle", 0, 180, 1, self.config.angle, "angle", "°"),
            ("Thickness", 0.1, 5.0, 0.1, self.config.spike_width, "spike_width", "")
        ])
        
        self._add_group("Appearance", [
            ("Intensity", 0, 1.0, 0.05, self.config.intensity, "intensity", ""),
            ("Color Saturation", 0, 2.0, 0.05, self.config.color_saturation, "color_saturation", ""),
            ("Hue Shift", -180, 180, 1, self.config.hue_shift, "hue_shift", "°")
        ])
        
        # Halo
        halo_group = QGroupBox("Star Halo / Rings")
        halo_layout = QVBoxLayout()
        
        self.halo_check = QCheckBox("Enable Halo")
        self.halo_check.setChecked(self.config.enable_halo)
        self.halo_check.toggled.connect(lambda c: self._update_config("enable_halo", c))
        halo_layout.addWidget(self.halo_check)
        
        self._add_slider(halo_layout, "Intensity", 0, 1.0, 0.05, self.config.halo_intensity, "halo_intensity", "")
        self._add_slider(halo_layout, "Radius", 0.1, 5.0, 0.1, self.config.halo_scale, "halo_scale", "")
        self._add_slider(halo_layout, "Width", 0.2, 10.0, 0.2, self.config.halo_width, "halo_width", "")
        self._add_slider(halo_layout, "Blur", 0, 10.0, 0.1, self.config.halo_blur, "halo_blur", "")
        self._add_slider(halo_layout, "Saturation", 0, 3.0, 0.1, self.config.halo_saturation, "halo_saturation", "")
        
        halo_group.setLayout(halo_layout)
        self.layout.addWidget(halo_group)
        
        self._add_group("Secondary Spikes", [
            ("Intensity", 0, 1.0, 0.05, self.config.secondary_intensity, "secondary_intensity", ""),
            ("Length", 0, 500, 10, self.config.secondary_length, "secondary_length", ""),
            ("Offset Angle", 0, 90, 1, self.config.secondary_offset, "secondary_offset", "°")
        ])
        
        self._add_group("Soft Flare", [
            ("Glow Intensity", 0, 3.0, 0.05, self.config.soft_flare_intensity, "soft_flare_intensity", ""),
            ("Glow Size", 0, 200, 5, self.config.soft_flare_size, "soft_flare_size", "")
        ])
        
        # Spectral
        spectral_group = QGroupBox("Spectral Effects")
        spectral_layout = QVBoxLayout()
        
        self.rainbow_check = QCheckBox("Enable Rainbow FX")
        self.rainbow_check.setChecked(self.config.enable_rainbow)
        self.rainbow_check.toggled.connect(lambda c: self._update_config("enable_rainbow", c))
        spectral_layout.addWidget(self.rainbow_check)
        
        self._add_slider(spectral_layout, "Intensity", 0, 1.0, 0.05, self.config.rainbow_spike_intensity, "rainbow_spike_intensity", "")
        self._add_slider(spectral_layout, "Frequency", 0.1, 3.0, 0.1, self.config.rainbow_spike_frequency, "rainbow_spike_frequency", "")
        self._add_slider(spectral_layout, "Coverage", 0.1, 1.0, 0.1, self.config.rainbow_spike_length, "rainbow_spike_length", "")
        
        spectral_group.setLayout(spectral_layout)
        self.layout.addWidget(spectral_group)
        
        self.layout.addStretch()

    def _add_group(self, title, sliders):
        group = QGroupBox(title)
        layout = QVBoxLayout()
        for label, min_v, max_v, step, init, key, unit in sliders:
            self._add_slider(layout, label, min_v, max_v, step, init, key, unit)
        group.setLayout(layout)
        self.layout.addWidget(group)

    def _add_slider(self, layout, label, min_v, max_v, step, init, key, unit):
        slider = SliderControl(label, min_v, max_v, step, init, unit)
        slider.value_changed.connect(lambda v, k=key: self._update_config(k, v))
        layout.addWidget(slider)

    def _update_config(self, key, value):
        setattr(self.config, key, value)
        self.config_changed.emit(self.config)

    def set_config(self, config: SpikeConfig):
        self.config = config
        self._build_controls()


class CanvasPreview(QWidget):
    """Canvas widget for image preview with pan/zoom."""
    stars_updated = pyqtSignal(list, bool)
    
    def __init__(self):
        super().__init__()
        self.setMouseTracking(True)
        self.image: Optional[QImage] = None
        self.stars: List[Star] = []
        self.config: Optional[SpikeConfig] = None
        self.renderer = Renderer()
        
        self.scale = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        
        self.is_dragging = False
        self.last_mouse_pos = QPointF()
        
        self.tool_mode: ToolMode = ToolMode.NONE
        self.star_input_radius: float = 4.0
        self.eraser_input_size: float = 20.0
        self.cursor_pos = QPointF(-9999, -9999)
        self.is_erasing = False

    def set_image(self, image: QImage):
        self.image = image
        self.fit_to_view()
        self.update()

    def set_stars(self, stars: List[Star]):
        self.stars = stars
        self.update()

    def set_config(self, config: SpikeConfig):
        self.config = config
        self.update()
    
    def set_tool_mode(self, mode: ToolMode):
        self.tool_mode = mode
        if mode == ToolMode.NONE:
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        else:
            self.setCursor(Qt.CursorShape.BlankCursor)
        self.update()
    
    def set_star_input_radius(self, radius: float):
        self.star_input_radius = radius
        self.update()
    
    def set_eraser_input_size(self, size: float):
        self.eraser_input_size = size
        self.update()

    def fit_to_view(self):
        if not self.image:
            return
        w_ratio = self.width() / self.image.width()
        h_ratio = self.height() / self.image.height()
        self.scale = min(w_ratio, h_ratio) * 0.9
        self.center_image()

    def zoom_in(self):
        self.scale *= 1.2
        self.center_image()
        
    def zoom_out(self):
        self.scale /= 1.2
        self.center_image()

    def center_image(self):
        if not self.image:
            return
        self.offset_x = (self.width() - self.image.width() * self.scale) / 2
        self.offset_y = (self.height() - self.image.height() * self.scale) / 2
        self.update()

    def resizeEvent(self, event):
        if self.image:
            self.center_image()
        super().resizeEvent(event)
    
    def _screen_to_image(self, screen_pos: QPointF) -> QPointF:
        img_x = (screen_pos.x() - self.offset_x) / self.scale
        img_y = (screen_pos.y() - self.offset_y) / self.scale
        return QPointF(img_x, img_y)

    def paintEvent(self, event: QPaintEvent):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(10, 10, 12))
        
        if not self.image:
            painter.setPen(QColor(100, 100, 100))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No Image Loaded")
            return

        painter.save()
        painter.translate(self.offset_x, self.offset_y)
        painter.scale(self.scale, self.scale)
        
        target_rect = QRectF(0, 0, self.image.width(), self.image.height())
        painter.drawImage(target_rect, self.image)
        
        if self.config:
            self.renderer.render(painter, self.image.width(), self.image.height(), self.stars, self.config)
            
        painter.restore()
        
        if self.tool_mode != ToolMode.NONE and self.cursor_pos.x() > -9000:
            self._draw_cursor_preview(painter)
        
        painter.setPen(QColor(200, 200, 200))
        mode_str = self.tool_mode.value.upper() if self.tool_mode else "NONE"
        painter.drawText(10, 20, f"Zoom: {self.scale*100:.0f}% | Stars: {len(self.stars)} | Tool: {mode_str}")

    def _draw_cursor_preview(self, painter: QPainter):
        if self.tool_mode == ToolMode.ADD:
            preview_radius = self.star_input_radius * self.scale
            color = QColor(56, 189, 248, 150)
            border_color = QColor(56, 189, 248, 255)
        elif self.tool_mode == ToolMode.ERASE:
            preview_radius = self.eraser_input_size * self.scale
            color = QColor(248, 113, 113, 80)
            border_color = QColor(248, 113, 113, 200)
        else:
            return
        
        preview_radius = max(4, preview_radius)
        painter.setBrush(QBrush(color))
        painter.setPen(QPen(border_color, 2))
        painter.drawEllipse(self.cursor_pos, preview_radius, preview_radius)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            if self.tool_mode == ToolMode.NONE:
                self.is_dragging = True
                self.last_mouse_pos = event.position()
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
            elif self.tool_mode == ToolMode.ADD:
                self._add_star_at(event.position())
            elif self.tool_mode == ToolMode.ERASE:
                self.is_erasing = True
                self._erase_stars_at(event.position(), push_history=False)

    def mouseMoveEvent(self, event: QMouseEvent):
        self.cursor_pos = event.position()
        
        if self.is_dragging and self.tool_mode == ToolMode.NONE:
            delta = event.position() - self.last_mouse_pos
            self.offset_x += delta.x()
            self.offset_y += delta.y()
            self.last_mouse_pos = event.position()
        elif self.is_erasing and self.tool_mode == ToolMode.ERASE:
            self._erase_stars_at(event.position(), push_history=False)
        
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            if self.is_dragging and self.tool_mode == ToolMode.NONE:
                self.is_dragging = False
                self.setCursor(Qt.CursorShape.OpenHandCursor)
            elif self.is_erasing and self.tool_mode == ToolMode.ERASE:
                self.is_erasing = False
                self.stars_updated.emit(list(self.stars), True)

    def leaveEvent(self, event):
        self.cursor_pos = QPointF(-9999, -9999)
        self.update()
        super().leaveEvent(event)

    def wheelEvent(self, event: QWheelEvent):
        zoom_sensitivity = 0.001
        delta = event.angleDelta().y() * zoom_sensitivity
        
        old_scale = self.scale
        new_scale = max(0.05, min(20.0, self.scale * (1 + delta)))
        
        mouse_pos = event.position()
        rel_x = (mouse_pos.x() - self.offset_x) / old_scale
        rel_y = (mouse_pos.y() - self.offset_y) / old_scale
        
        self.offset_x = mouse_pos.x() - rel_x * new_scale
        self.offset_y = mouse_pos.y() - rel_y * new_scale
        self.scale = new_scale
        
        self.update()

    def _add_star_at(self, screen_pos: QPointF):
        if not self.image:
            return
        
        img_pos = self._screen_to_image(screen_pos)
        
        if img_pos.x() < 0 or img_pos.x() >= self.image.width():
            return
        if img_pos.y() < 0 or img_pos.y() >= self.image.height():
            return
        
        new_star = Star(
            x=img_pos.x(),
            y=img_pos.y(),
            brightness=1.0,
            radius=self.star_input_radius,
            color=Color(255, 255, 255)
        )
        
        new_stars = list(self.stars) + [new_star]
        self.stars = new_stars
        self.stars_updated.emit(new_stars, True)
        self.update()

    def _erase_stars_at(self, screen_pos: QPointF, push_history: bool = False):
        if not self.image:
            return
        
        img_pos = self._screen_to_image(screen_pos)
        erase_radius_sq = self.eraser_input_size * self.eraser_input_size
        
        initial_count = len(self.stars)
        
        filtered_stars = [
            star for star in self.stars
            if (star.x - img_pos.x()) ** 2 + (star.y - img_pos.y()) ** 2 > erase_radius_sq
        ]
        
        if len(filtered_stars) != initial_count:
            self.stars = filtered_stars
            self.stars_updated.emit(filtered_stars, push_history)
            self.update()


if __name__ == "__main__":
    print("AstroSpike Script for SETI Astro")
    print("=" * 40)
    print("This script is designed to run within SETI Astro.")
    print("To use it:")
    print("1. Copy this file to your SETI Astro scripts folder")
    print("2. Open an image in SETI Astro")
    print("3. Run the script from the Scripts menu")
    print()
    print("Configuration options can be adjusted at the top of this file.")
