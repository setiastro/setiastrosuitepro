# src/setiastro/saspro/flythrough.py — SASpro Nebula Flythrough Tool
# =============================================================================
#
#  Composites a stars-only image over a starless image using Screen blend,
#  with independent per-layer zoom center and zoom rate, producing the effect
#  of flying into a nebula along a curved/skewed trajectory.
#
#  Optional per-layer effects:
#    • Nebula depth warp  — luminance-driven per-pixel zoom (no banding)
#    • Radial edge stretch
#    • Zoom blur (warp-speed star streaks)
#    • Chromatic aberration (R/G/B channel split)
#
#  Written by Franklin Marek  |  www.setiastro.com
#
# =============================================================================
from __future__ import annotations

import os
import numpy as np

from PyQt6.QtCore import (
    Qt, QThread, pyqtSignal, QSettings, QTimer,
)
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel,
    QPushButton, QComboBox, QDoubleSpinBox, QSpinBox,
    QGroupBox, QSlider, QFileDialog, QMessageBox, QProgressBar,
    QWidget, QSizePolicy, QCheckBox,
)
from PyQt6.QtGui import QImage, QPixmap

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


# ── SetiAstro copyright header ────────────────────────────────────────────────
#
#   _____      __  _ ___         __
#  / ___/___  / /_(_)   |  _____/ /__________
#  \__ \/ _ \/ __/ / /| | / ___/ __/ ___/ __ \
# ___/ /  __/ /_/ / ___ |(__  ) /_/ /  / /_/ /
#/____/\___/\__/_/_/  |_/____/\__/_/   \____/
#
# =============================================================================


# ---------------------------------------------------------------------------
# Easing functions  (t in [0,1] -> eased t in [0,1])
# ---------------------------------------------------------------------------

def _ease_linear(t: float) -> float:
    return float(t)

def _ease_in(t: float) -> float:
    return float(t * t * t)

def _ease_out(t: float) -> float:
    t = 1.0 - t
    return float(1.0 - t * t * t)

def _ease_in_out(t: float) -> float:
    if t < 0.5:
        return float(4.0 * t * t * t)
    return float(1.0 - (-2.0 * t + 2.0) ** 3 / 2.0)

EASE_FUNCTIONS = {
    "Linear":      _ease_linear,
    "Ease In":     _ease_in,
    "Ease Out":    _ease_out,
    "Ease In-Out": _ease_in_out,
}


# Working-resolution presets
WORKING_RES_PRESETS = {
    "Full (original)": None,
    "4K  (3840x2160)": (3840, 2160),
    "2K  (2560x1440)": (2560, 1440),
    "HD  (1920x1080)": (1920, 1080),
    "HD  (1280x720)":  (1280,  720),
    "SD  (854x480)":   ( 854,  480),
    "SD  (640x360)":   ( 640,  360),
}


# ---------------------------------------------------------------------------
# Array helpers
# ---------------------------------------------------------------------------

def _to_f32(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    mx = float(np.nanmax(a)) if a.size else 1.0
    if mx > 2.0:
        a = a / mx
    return np.clip(a, 0.0, 1.0)


def _ensure_rgb(arr: np.ndarray) -> np.ndarray:
    a = _to_f32(arr)
    if a.ndim == 2:
        return np.stack([a, a, a], axis=2)
    if a.ndim == 3 and a.shape[2] == 1:
        return np.repeat(a, 3, axis=2)
    if a.ndim == 3 and a.shape[2] >= 3:
        return a[:, :, :3]
    raise ValueError(f"Unsupported image shape: {a.shape}")


def _downsize_to_fit(arr: np.ndarray, max_w: int, max_h: int) -> np.ndarray:
    h, w = arr.shape[:2]
    if w <= max_w and h <= max_h:
        return arr.astype(np.float32, copy=False)
    scale  = min(max_w / w, max_h / h)
    new_w  = max(1, int(round(w * scale)))
    new_h  = max(1, int(round(h * scale)))
    if HAS_CV2:
        return cv2.resize(arr, (new_w, new_h),
                          interpolation=cv2.INTER_AREA).astype(np.float32)
    ys = np.linspace(0, h - 1, new_h).astype(np.int32)
    xs = np.linspace(0, w - 1, new_w).astype(np.int32)
    return (arr[ys][:, xs] if arr.ndim == 2
            else arr[np.ix_(ys, xs)]).astype(np.float32)


def _screen_blend(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return 1.0 - (1.0 - a) * (1.0 - b)


# ---------------------------------------------------------------------------
# Depth map builder  (used by both CPU and GPU paths)
# ---------------------------------------------------------------------------

def _build_depth_map(img: np.ndarray,
                     blur_sigma: float = 8.0,
                     depth_gamma: float = 1.0,
                     invert: bool = False) -> np.ndarray:
    """
    Build a smooth normalised depth map from the luminance of img.
    Returns float32 H x W in [-1, 1].
      +1 = brightest region  = closest to camera
      -1 = darkest region    = furthest from camera
    (flip with invert=True for dark=closer)

    Uses MAD normalisation so bright stars don't dominate the field.
    The heavy blur is intentional — we want large-scale nebula structure
    to drive the depth, not individual stars or noise.
    """
    img = _ensure_rgb(img)
    lum = (0.299 * img[:, :, 0] +
           0.587 * img[:, :, 1] +
           0.114 * img[:, :, 2]).astype(np.float32)

    if HAS_CV2 and blur_sigma > 0:
        lum_s = cv2.GaussianBlur(lum, (0, 0), float(blur_sigma))
    else:
        lum_s = lum

    # Centre around median so sky background = 0
    h = lum_s - float(np.median(lum_s))

    # Optional gamma shaping
    g = float(max(1e-6, depth_gamma))
    if abs(g - 1.0) > 1e-6:
        h = np.sign(h) * (np.abs(h) ** g)

    if invert:
        h = -h

    # MAD normalisation — robust against stars
    mad = float(np.median(np.abs(h))) + 1e-9
    h = h / (6.0 * mad)
    return np.clip(h, -1.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# CPU zoom-crop  (flat — no depth)
# ---------------------------------------------------------------------------

def _zoom_crop(img: np.ndarray,
               zoom: float,
               cx_frac: float, cy_frac: float,
               out_h: int, out_w: int) -> np.ndarray:
    """Standard flat zoom crop — no depth."""
    H, W = img.shape[:2]
    crop_w = max(1, int(round(W / zoom)))
    crop_h = max(1, int(round(H / zoom)))
    cx_px  = cx_frac * W
    cy_px  = cy_frac * H
    x0 = int(round(cx_px - crop_w / 2))
    y0 = int(round(cy_px - crop_h / 2))
    x0 = max(0, min(x0, W - crop_w))
    y0 = max(0, min(y0, H - crop_h))
    crop = img[y0:y0 + crop_h, x0:x0 + crop_w]
    if HAS_CV2:
        return cv2.resize(crop, (out_w, out_h),
                          interpolation=cv2.INTER_LANCZOS4).astype(np.float32)
    ys = np.linspace(0, crop.shape[0] - 1, out_h).astype(np.int32)
    xs = np.linspace(0, crop.shape[1] - 1, out_w).astype(np.int32)
    return (crop[ys][:, xs] if crop.ndim == 2
            else crop[np.ix_(ys, xs)]).astype(np.float32)


# ---------------------------------------------------------------------------
# CPU depth-aware zoom crop  (the correct nebula-depth implementation)
# ---------------------------------------------------------------------------

def _zoom_crop_depth(img, depth_map, zoom_base, depth_strength,
                     cx_frac, cy_frac, out_h, out_w):
    """
    Camera flying toward a height-field surface draped with img.
    
    zoom_base:      flat zoom (camera has advanced this much toward mean surface)
    depth_strength: how much height deviation adds extra radial parallax
    
    For each output pixel:
      1. Find where it maps in source space under flat zoom (same as _zoom_crop)
      2. Read the depth at that source location  
      3. Apply additional radial shift based on depth * camera_advance
         (close pixels shift outward MORE than far pixels)
    """
    if not HAS_CV2 or depth_strength < 1e-4:
        return _zoom_crop(img, zoom_base, cx_frac, cy_frac, out_h, out_w)

    H, W = img.shape[:2]
    cx_px = cx_frac * W
    cy_px = cy_frac * H

    # Step 1: flat zoom source coords (where each output pixel reads from)
    cols = (np.arange(out_w, dtype=np.float32) + 0.5) / out_w - 0.5
    rows = (np.arange(out_h, dtype=np.float32) + 0.5) / out_h - 0.5
    u, v = np.meshgrid(cols, rows)

    crop_w = W / zoom_base
    crop_h = H / zoom_base
    base_src_x = cx_px + u * crop_w
    base_src_y = cy_px + v * crop_h

    # Step 2: sample depth map at flat-zoom source coords
    sx = np.clip(base_src_x, 0.0, W - 1.0).astype(np.float32)
    sy = np.clip(base_src_y, 0.0, H - 1.0).astype(np.float32)
    dm_sampled = cv2.remap(depth_map, sx, sy,
                           interpolation=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)

    # Step 3: radial parallax from height
    # Direction from zoom centre in OUTPUT space
    dx = u * crop_w   # displacement from centre in source pixels
    dy = v * crop_h
    dist = np.sqrt(dx*dx + dy*dy)
    safe_dist = np.maximum(dist, 1e-6)
    nx = dx / safe_dist
    ny = dy / safe_dist

    # Height-driven extra shift: tall pixels shift outward by depth*strength*zoom_progress
    # This is the parallax — tall things move more than short things as camera advances
    parallax_px = dm_sampled * depth_strength * (zoom_base - 1.0)

    final_src_x = (base_src_x - nx * parallax_px).astype(np.float32)
    final_src_y = (base_src_y - ny * parallax_px).astype(np.float32)

    final_src_x = np.clip(final_src_x, 0.0, W - 1.0)
    final_src_y = np.clip(final_src_y, 0.0, H - 1.0)

    return cv2.remap(img, final_src_x, final_src_y,
                     interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT_101).astype(np.float32)

# ---------------------------------------------------------------------------
# CPU optical effects  (applied AFTER zoom crop, on output-res frame)
# ---------------------------------------------------------------------------

def apply_radial_stretch(img: np.ndarray,
                          strength: float,
                          cx_frac: float = 0.5,
                          cy_frac: float = 0.5) -> np.ndarray:
    """Edges zoom outward faster than centre (rubber-sheet effect)."""
    if abs(strength) < 1e-4 or not HAS_CV2:
        return img
    h, w = img.shape[:2]
    cx_px, cy_px = cx_frac * w, cy_frac * h
    xs = np.arange(w, dtype=np.float32)
    ys = np.arange(h, dtype=np.float32)
    xg, yg = np.meshgrid(xs, ys)
    dx, dy = xg - cx_px, yg - cy_px
    half_diag = max(1.0, (w * w + h * h) ** 0.5 / 2.0)
    r = np.sqrt(dx * dx + dy * dy) / half_diag
    factor = strength * r * r
    src_x = (xg - dx * factor).astype(np.float32)
    src_y = (yg - dy * factor).astype(np.float32)
    return cv2.remap(img, src_x, src_y,
                     interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT_101).astype(np.float32)


def apply_zoom_blur(img: np.ndarray,
                    strength: float,
                    cx_frac: float = 0.5,
                    cy_frac: float = 0.5,
                    samples: int = 12) -> np.ndarray:
    """Radial motion blur (warp-speed streaks)."""
    if strength < 1e-4 or not HAS_CV2:
        return img
    h, w = img.shape[:2]
    max_zoom_spread = 1.0 + strength * 0.25
    acc = np.zeros_like(img, dtype=np.float32)
    for i in range(samples):
        frac  = i / max(1, samples - 1)
        zoom  = 1.0 + (max_zoom_spread - 1.0) * frac
        cw = max(1, int(round(w / zoom)))
        ch = max(1, int(round(h / zoom)))
        cx_px, cy_px = cx_frac * w, cy_frac * h
        x0 = max(0, min(int(round(cx_px - cw / 2)), w - cw))
        y0 = max(0, min(int(round(cy_px - ch / 2)), h - ch))
        crop = img[y0:y0 + ch, x0:x0 + cw]
        resized = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        acc += resized * (1.0 - frac * 0.6)
    total_w = sum(1.0 - (i / max(1, samples - 1)) * 0.6 for i in range(samples))
    acc /= max(1e-8, total_w)
    blend = strength * 0.7
    return np.clip((1.0 - blend) * img + blend * acc, 0.0, 1.0).astype(np.float32)


def apply_chromatic_aberration(img: np.ndarray,
                                strength: float,
                                cx_frac: float = 0.5,
                                cy_frac: float = 0.5) -> np.ndarray:
    """R channel zoomed out, B zoomed in, G unchanged."""
    if abs(strength) < 1e-4 or not HAS_CV2:
        return img
    h, w = img.shape[:2]
    max_shift = strength * 0.03

    def _shift_channel(ch: int, zoom_factor: float) -> np.ndarray:
        cw = max(1, int(round(w / zoom_factor)))
        ch_h = max(1, int(round(h / zoom_factor)))
        cx_px, cy_px = cx_frac * w, cy_frac * h
        x0 = max(0, min(int(round(cx_px - cw / 2)), w - cw))
        y0 = max(0, min(int(round(cy_px - ch_h / 2)), h - ch_h))
        return cv2.resize(img[y0:y0 + ch_h, x0:x0 + cw, ch],
                          (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)

    out = img.copy()
    r_zoom = 1.0 - max_shift
    b_zoom = 1.0 + max_shift
    if r_zoom > 0.01:
        out[:, :, 0] = _shift_channel(0, r_zoom)
    out[:, :, 2] = _shift_channel(2, b_zoom)
    return np.clip(out, 0.0, 1.0).astype(np.float32)

def _apply_layer_effects(frame: np.ndarray,
                          t: float,
                          cx_frac: float, cy_frac: float,
                          fx: dict) -> np.ndarray:
    """
    Post-zoom optical effects (radial stretch, zoom blur, chromatic aberration).
    Effects scale with zoom VELOCITY not position — fast zoom = strong effects.
    """
    if not fx:
        return frame

    animate = bool(fx.get("animate_effects", True))

    if animate:
        # Approximate instantaneous zoom velocity via central finite difference.
        # We need the eased t values at t-eps and t+eps to get d(eased_t)/dt.
        # The ease function is stored in fx so we can't call it directly here,
        # but we can approximate velocity from the zoom_start/zoom_end and
        # the eased position passed in via fx, OR we can just differentiate
        # the ease function numerically using t.
        #
        # Simplest correct approach: the effects should track the derivative
        # of whichever ease curve the layer uses. We pass zoom velocity in via fx.
        zoom_vel = float(fx.get("zoom_velocity", 1.0))   # 0..~3, normalised below
        # Normalise: velocity=1 means "average speed for this clip"
        # Cap at 1.0 so effects never exceed their set strength
        ramp = float(np.clip(zoom_vel, 0.0, 1.0))
    else:
        ramp = 1.0

    out = _ensure_rgb(frame)

    rs = float(fx.get("radial_stretch", 0.0)) * ramp
    if abs(rs) > 1e-4:
        out = apply_radial_stretch(out, rs, cx_frac, cy_frac)

    zb = float(fx.get("zoom_blur", 0.0)) * ramp
    if zb > 1e-4:
        out = apply_zoom_blur(out, zb, cx_frac, cy_frac,
                               int(fx.get("zoom_blur_samples", 12)))

    ca = float(fx.get("chroma", 0.0)) * ramp
    if abs(ca) > 1e-4:
        out = apply_chromatic_aberration(out, ca, cx_frac, cy_frac)

    return out


# ---------------------------------------------------------------------------
# GPU helpers
# ---------------------------------------------------------------------------

def _get_torch_device():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    except ImportError:
        return None


def _np_to_tensor(arr: np.ndarray, device):
    import torch
    t = torch.from_numpy(np.ascontiguousarray(arr)).to(device)
    return t.permute(2, 0, 1).unsqueeze(0)   # HWC -> NCHW


def _tensor_to_np(t) -> np.ndarray:
    return t.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float32)


def _np_depth_to_tensor(dm: np.ndarray, device):
    """H x W float32 depth map -> 1 x 1 x H x W tensor."""
    import torch
    return torch.from_numpy(np.ascontiguousarray(dm)).to(device).unsqueeze(0).unsqueeze(0)


# ---------------------------------------------------------------------------
# GPU flat zoom crop
# ---------------------------------------------------------------------------

def _zoom_crop_gpu(t, zoom: float, cx_frac: float, cy_frac: float,
                   out_h: int, out_w: int):
    import torch
    import torch.nn.functional as F

    _, C, H, W = t.shape
    cx_n = float(np.clip(cx_frac * 2.0 - 1.0, -1.0, 1.0))
    cy_n = float(np.clip(cy_frac * 2.0 - 1.0, -1.0, 1.0))
    hx = min(1.0 / max(zoom, 1e-4), 1.0)
    hy = min(1.0 / max(zoom, 1e-4), 1.0)
    cx_n = float(np.clip(cx_n, -1.0 + hx, 1.0 - hx))
    cy_n = float(np.clip(cy_n, -1.0 + hy, 1.0 - hy))

    gx = torch.linspace(cx_n - hx, cx_n + hx, out_w, device=t.device, dtype=torch.float32)
    gy = torch.linspace(cy_n - hy, cy_n + hy, out_h, device=t.device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(gy, gx, indexing="ij")
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
    return F.grid_sample(t, grid, mode="bilinear", padding_mode="border", align_corners=True)


# ---------------------------------------------------------------------------
# GPU depth-aware zoom crop  (the key function)
# ---------------------------------------------------------------------------
def _zoom_crop_depth_gpu(t, depth_t, zoom_base, depth_strength,
                          cx_frac, cy_frac, out_h, out_w):
    import torch
    import torch.nn.functional as F

    if depth_strength < 1e-4:
        return _zoom_crop_gpu(t, zoom_base, cx_frac, cy_frac, out_h, out_w)

    _, C, H, W = t.shape
    dev = t.device

    cx_px = cx_frac * W
    cy_px = cy_frac * H
    crop_w = W / zoom_base
    crop_h = H / zoom_base

    # Output pixel normalised coords u,v in [-0.5, 0.5]
    cols = (torch.arange(out_w, device=dev, dtype=torch.float32) + 0.5) / out_w - 0.5
    rows = (torch.arange(out_h, device=dev, dtype=torch.float32) + 0.5) / out_h - 0.5
    v_g, u_g = torch.meshgrid(rows, cols, indexing="ij")

    # Flat zoom source coords in source pixels
    base_src_x = cx_px + u_g * crop_w
    base_src_y = cy_px + v_g * crop_h

    # Sample depth at those source coords
    # Normalise to [-1,1] for grid_sample
    norm_bx = (base_src_x.clamp(0, W-1) / (W - 1)) * 2.0 - 1.0
    norm_by = (base_src_y.clamp(0, H-1) / (H - 1)) * 2.0 - 1.0
    depth_grid = torch.stack([norm_bx, norm_by], dim=-1).unsqueeze(0)
    dm_sampled = F.grid_sample(depth_t, depth_grid,
                                mode="bilinear", padding_mode="border",
                                align_corners=True).squeeze(0).squeeze(0)

    # Radial direction from zoom centre in source pixel space
    dx = u_g * crop_w
    dy = v_g * crop_h
    dist = torch.sqrt(dx*dx + dy*dy).clamp(min=1e-6)
    nx = dx / dist
    ny = dy / dist

    # Parallax: grows with camera advance (zoom_base - 1)
    parallax_px = dm_sampled * depth_strength * (zoom_base - 1.0)

    final_src_x = (base_src_x - nx * parallax_px).clamp(0, W - 1.0)
    final_src_y = (base_src_y - ny * parallax_px).clamp(0, H - 1.0)

    norm_fx = final_src_x / (W - 1) * 2.0 - 1.0
    norm_fy = final_src_y / (H - 1) * 2.0 - 1.0
    grid = torch.stack([norm_fx, norm_fy], dim=-1).unsqueeze(0)

    return F.grid_sample(t, grid, mode="bilinear",
                          padding_mode="border", align_corners=True)

# ---------------------------------------------------------------------------
# GPU optical effects
# ---------------------------------------------------------------------------

def _radial_stretch_gpu(t, strength: float, cx_frac: float, cy_frac: float):
    import torch
    import torch.nn.functional as F
    if abs(strength) < 1e-4:
        return t
    _, C, H, W = t.shape
    dev = t.device
    xs = torch.linspace(-1, 1, W, device=dev, dtype=torch.float32)
    ys = torch.linspace(-1, 1, H, device=dev, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    cx_n = cx_frac * 2.0 - 1.0
    cy_n = cy_frac * 2.0 - 1.0
    dx, dy = grid_x - cx_n, grid_y - cy_n
    half_diag = (2.0 ** 2 + 2.0 ** 2) ** 0.5 / 2.0
    r = torch.sqrt(dx * dx + dy * dy) / half_diag
    factor = strength * r * r
    src_x = grid_x - dx * factor
    src_y = grid_y - dy * factor
    grid = torch.stack([src_x, src_y], dim=-1).unsqueeze(0)
    return F.grid_sample(t, grid, mode="bilinear", padding_mode="reflection", align_corners=True)


def _zoom_blur_gpu(t, strength: float, cx_frac: float, cy_frac: float, samples: int = 12):
    import torch
    if strength < 1e-4:
        return t
    _, C, H, W = t.shape
    max_spread = 1.0 + strength * 0.25
    acc = torch.zeros_like(t)
    total_w = 0.0
    for i in range(samples):
        frac   = i / max(1, samples - 1)
        zoom   = 1.0 + (max_spread - 1.0) * frac
        weight = 1.0 - frac * 0.6
        acc    += _zoom_crop_gpu(t, zoom, cx_frac, cy_frac, H, W) * weight
        total_w += weight
    acc /= max(1e-8, total_w)
    blend = strength * 0.7
    return torch.clamp((1.0 - blend) * t + blend * acc, 0.0, 1.0)


def _chroma_gpu(t, strength: float, cx_frac: float, cy_frac: float):
    import torch
    import torch.nn.functional as F
    if abs(strength) < 1e-4:
        return t
    _, C, H, W = t.shape
    dev = t.device
    max_shift = strength * 0.03
    r_zoom = 1.0 - max_shift
    b_zoom = 1.0 + max_shift

    def _channel_grid(zoom_factor: float):
        cw = W / zoom_factor
        ch = H / zoom_factor
        cx_px = cx_frac * W
        cy_px = cy_frac * H
        x0 = float(np.clip(cx_px - cw / 2.0, 0.0, W - cw))
        y0 = float(np.clip(cy_px - ch / 2.0, 0.0, H - ch))
        out_xs = torch.arange(W, device=dev, dtype=torch.float32)
        src_xs = x0 + out_xs * (cw / W)
        norm_xs = (src_xs / (W - 1)) * 2.0 - 1.0
        out_ys = torch.arange(H, device=dev, dtype=torch.float32)
        src_ys = y0 + out_ys * (ch / H)
        norm_ys = (src_ys / (H - 1)) * 2.0 - 1.0
        grid_x = norm_xs.unsqueeze(0).expand(H, W)
        grid_y = norm_ys.unsqueeze(1).expand(H, W)
        return torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)

    out = t.clone()
    if r_zoom > 0.01:
        out[:, 0:1] = F.grid_sample(t[:, 0:1], _channel_grid(r_zoom),
                                     mode="bilinear", padding_mode="border", align_corners=True)
    out[:, 2:3] = F.grid_sample(t[:, 2:3], _channel_grid(b_zoom),
                                 mode="bilinear", padding_mode="border", align_corners=True)
    return torch.clamp(out, 0.0, 1.0)


def _screen_blend_gpu(a, b):
    return 1.0 - (1.0 - a) * (1.0 - b)


# ---------------------------------------------------------------------------
# CPU render_frame
# ---------------------------------------------------------------------------

def render_frame(
    starless: np.ndarray,
    stars:    np.ndarray,
    t: float,
    sl_zoom_start: float, sl_zoom_end: float,
    sl_cx_start: float,   sl_cy_start: float,
    sl_cx_end: float,     sl_cy_end: float,
    sl_ease_fn,
    sl_fx: dict,
    st_zoom_start: float, st_zoom_end: float,
    st_cx_start: float,   st_cy_start: float,
    st_cx_end: float,     st_cy_end: float,
    st_ease_fn,
    st_fx: dict,
    out_h: int, out_w: int,
    sl_depth_map: np.ndarray | None = None,
    st_depth_map: np.ndarray | None = None,
) -> np.ndarray:
    """Render one composited frame at normalised time t in [0,1]."""
    te_sl = sl_ease_fn(t)
    te_st = st_ease_fn(t)

    sl_zoom = sl_zoom_start + (sl_zoom_end - sl_zoom_start) * te_sl
    sl_cx   = sl_cx_start   + (sl_cx_end   - sl_cx_start)   * te_sl
    sl_cy   = sl_cy_start   + (sl_cy_end   - sl_cy_start)   * te_sl
    st_zoom = st_zoom_start + (st_zoom_end - st_zoom_start) * te_st
    st_cx   = st_cx_start   + (st_cx_end   - st_cx_start)   * te_st
    st_cy   = st_cy_start   + (st_cy_end   - st_cy_start)   * te_st

    # Zoom velocity via central finite difference on the eased t curve.
    # This gives d(zoom)/dt which is what drives motion-linked effects.
    # At Ease In-Out: slow at start and end, fast in middle — effects follow that.
    eps = 0.005
    t_lo = max(0.0, t - eps)
    t_hi = min(1.0, t + eps)

    sl_zoom_range = max(sl_zoom_end - sl_zoom_start, 1e-6)
    st_zoom_range = max(st_zoom_end - st_zoom_start, 1e-6)

    # d(eased_t)/dt * zoom_range gives zoom speed in zoom-units/frame
    sl_vel_raw = (sl_ease_fn(t_hi) - sl_ease_fn(t_lo)) / max(t_hi - t_lo, 1e-9)
    st_vel_raw = (st_ease_fn(t_hi) - st_ease_fn(t_lo)) / max(t_hi - t_lo, 1e-9)

    # At linear easing vel_raw = 1.0 always.
    # At ease-in-out: peaks at ~2.0 in the middle, 0 at ends.
    # Normalise to [0,1] so strength slider remains intuitive:
    # divide by the theoretical max of the derivative (which is 3.0 for cubic ease)
    sl_zoom_vel = float(np.clip(sl_vel_raw / 3.0, 0.0, 1.0))
    st_zoom_vel = float(np.clip(st_vel_raw / 3.0, 0.0, 1.0))

    dw_sl = float(sl_fx.get("depth_warp", 0.0))
    dw_st = float(st_fx.get("depth_warp", 0.0))

    if sl_depth_map is not None and dw_sl > 1e-4:
        sl_frame = _zoom_crop_depth(starless, sl_depth_map, sl_zoom, dw_sl,
                                     sl_cx, sl_cy, out_h, out_w)
    else:
        sl_frame = _zoom_crop(starless, sl_zoom, sl_cx, sl_cy, out_h, out_w)

    if st_depth_map is not None and dw_st > 1e-4:
        st_frame = _zoom_crop_depth(stars, st_depth_map, st_zoom, dw_st,
                                     st_cx, st_cy, out_h, out_w)
    else:
        st_frame = _zoom_crop(stars, st_zoom, st_cx, st_cy, out_h, out_w)

    # Inject velocity into fx dicts for _apply_layer_effects
    sl_fx_post = {k: v for k, v in sl_fx.items() if k != "depth_warp"}
    st_fx_post = {k: v for k, v in st_fx.items() if k != "depth_warp"}
    sl_fx_post["zoom_velocity"] = sl_zoom_vel
    st_fx_post["zoom_velocity"] = st_zoom_vel

    sl_rgb = _apply_layer_effects(_ensure_rgb(sl_frame), t, sl_cx, sl_cy, sl_fx_post)
    st_rgb = _apply_layer_effects(_ensure_rgb(st_frame), t, st_cx, st_cy, st_fx_post)

    return np.clip(_screen_blend(sl_rgb, st_rgb), 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# GPU render_frame
# ---------------------------------------------------------------------------

def render_frame_gpu(
    starless_t, stars_t,
    t: float,
    sl_zoom_start, sl_zoom_end,
    sl_cx_start, sl_cy_start,
    sl_cx_end,   sl_cy_end,
    sl_ease_fn,  sl_fx: dict,
    st_zoom_start, st_zoom_end,
    st_cx_start, st_cy_start,
    st_cx_end,   st_cy_end,
    st_ease_fn,  st_fx: dict,
    out_h: int,  out_w: int,
    sl_depth_t=None,
    st_depth_t=None,
) -> np.ndarray:
    """Full GPU frame render. Returns H x W x 3 float32 numpy."""
    import torch

    te_sl = sl_ease_fn(t);  te_st = st_ease_fn(t)

    sl_zoom = sl_zoom_start + (sl_zoom_end - sl_zoom_start) * te_sl
    sl_cx   = sl_cx_start   + (sl_cx_end   - sl_cx_start)   * te_sl
    sl_cy   = sl_cy_start   + (sl_cy_end   - sl_cy_start)   * te_sl
    st_zoom = st_zoom_start + (st_zoom_end - st_zoom_start) * te_st
    st_cx   = st_cx_start   + (st_cx_end   - st_cx_start)   * te_st
    st_cy   = st_cy_start   + (st_cy_end   - st_cy_start)   * te_st

    dw_sl = float(sl_fx.get("depth_warp", 0.0))
    dw_st = float(st_fx.get("depth_warp", 0.0))

    if sl_depth_t is not None and dw_sl > 1e-4:
        sl_frame = _zoom_crop_depth_gpu(starless_t, sl_depth_t, sl_zoom, dw_sl,
                                         sl_cx, sl_cy, out_h, out_w)
    else:
        sl_frame = _zoom_crop_gpu(starless_t, sl_zoom, sl_cx, sl_cy, out_h, out_w)

    if st_depth_t is not None and dw_st > 1e-4:
        st_frame = _zoom_crop_depth_gpu(stars_t, st_depth_t, st_zoom, dw_st,
                                         st_cx, st_cy, out_h, out_w)
    else:
        st_frame = _zoom_crop_gpu(stars_t, st_zoom, st_cx, st_cy, out_h, out_w)

    # Post-zoom optical effects
    # Zoom velocity via central finite difference — same as CPU path
    eps = 0.005
    t_lo, t_hi = max(0.0, t - eps), min(1.0, t + eps)
    sl_vel_raw = (sl_ease_fn(t_hi) - sl_ease_fn(t_lo)) / max(t_hi - t_lo, 1e-9)
    st_vel_raw = (st_ease_fn(t_hi) - st_ease_fn(t_lo)) / max(t_hi - t_lo, 1e-9)

    animate_sl = bool(sl_fx.get("animate_effects", True))
    animate_st = bool(st_fx.get("animate_effects", True))
    ramp_sl = float(np.clip(sl_vel_raw / 3.0, 0.0, 1.0)) if animate_sl else 1.0
    ramp_st = float(np.clip(st_vel_raw / 3.0, 0.0, 1.0)) if animate_st else 1.0

    rs = float(sl_fx.get("radial_stretch", 0.0)) * ramp_sl
    if abs(rs) > 1e-4:
        sl_frame = _radial_stretch_gpu(sl_frame, rs, sl_cx, sl_cy)
    zb = float(sl_fx.get("zoom_blur", 0.0)) * ramp_sl
    if zb > 1e-4:
        sl_frame = _zoom_blur_gpu(sl_frame, zb, sl_cx, sl_cy,
                                   int(sl_fx.get("zoom_blur_samples", 12)))
    ca = float(sl_fx.get("chroma", 0.0)) * ramp_sl
    if abs(ca) > 1e-4:
        sl_frame = _chroma_gpu(sl_frame, ca, sl_cx, sl_cy)

    rs = float(st_fx.get("radial_stretch", 0.0)) * ramp_st
    if abs(rs) > 1e-4:
        st_frame = _radial_stretch_gpu(st_frame, rs, st_cx, st_cy)
    zb = float(st_fx.get("zoom_blur", 0.0)) * ramp_st
    if zb > 1e-4:
        st_frame = _zoom_blur_gpu(st_frame, zb, st_cx, st_cy,
                                   int(st_fx.get("zoom_blur_samples", 12)))
    ca = float(st_fx.get("chroma", 0.0)) * ramp_st
    if abs(ca) > 1e-4:
        st_frame = _chroma_gpu(st_frame, ca, st_cx, st_cy)

    composited = torch.clamp(_screen_blend_gpu(sl_frame, st_frame), 0.0, 1.0)
    return _tensor_to_np(composited)


# ---------------------------------------------------------------------------
# Render worker thread
# ---------------------------------------------------------------------------

class _FlythroughWorker(QThread):
    progress = pyqtSignal(int, int, str)
    finished = pyqtSignal(bool, str)

    def __init__(self, params: dict, parent=None):
        super().__init__(parent)
        self._p      = params
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        p = self._p
        try:
            starless_np = _ensure_rgb(p["starless"])
            stars_np    = _ensure_rgb(p["stars"])

            fps      = int(p["fps"])
            duration = float(p["duration"])
            n_frames = max(1, int(round(fps * duration)))
            out_w    = int(p["out_w"])
            out_h    = int(p["out_h"])
            out_path = str(p["out_path"])

            sl_ease_fn = EASE_FUNCTIONS.get(p.get("sl_ease", "Ease In-Out"), _ease_in_out)
            st_ease_fn = EASE_FUNCTIONS.get(p.get("st_ease", "Ease In-Out"), _ease_in_out)
            sl_fx = p.get("sl_fx", {})
            st_fx = p.get("st_fx", {})

            if not HAS_CV2:
                self.finished.emit(False, "OpenCV (cv2) is required for video export.")
                return

            # Pre-compute depth maps from full-res source (once, not per frame)
            sl_depth_map = None
            if sl_fx.get("depth_warp", 0.0) > 1e-4:
                sl_depth_map = _build_depth_map(
                    starless_np,
                    blur_sigma=float(sl_fx.get("depth_blur_sigma", 8.0)),
                    invert=bool(sl_fx.get("depth_invert", False)))

            st_depth_map = None
            if st_fx.get("depth_warp", 0.0) > 1e-4:
                st_depth_map = _build_depth_map(
                    stars_np,
                    blur_sigma=float(st_fx.get("depth_blur_sigma", 8.0)),
                    invert=bool(st_fx.get("depth_invert", False)))

            # GPU setup
            device  = _get_torch_device()
            use_gpu = device is not None and str(device) != "cpu"
            starless_t = stars_t = sl_depth_t = st_depth_t = None

            if use_gpu:
                try:
                    import torch
                    starless_t = _np_to_tensor(starless_np, device)
                    stars_t    = _np_to_tensor(stars_np,    device)
                    if sl_depth_map is not None:
                        sl_depth_t = _np_depth_to_tensor(sl_depth_map, device)
                    if st_depth_map is not None:
                        st_depth_t = _np_depth_to_tensor(st_depth_map, device)
                    self.progress.emit(0, n_frames,
                                       f"GPU render on {device} — {n_frames} frames")
                except Exception as e:
                    print(f"[Flythrough] GPU init failed, falling back to CPU: {e}")
                    use_gpu = False

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, fps, (out_w, out_h))
            if not writer.isOpened():
                self.finished.emit(False,
                    f"Could not open video file for writing:\n{out_path}")
                return

            for i in range(n_frames):
                if self._cancel:
                    writer.release()
                    try:
                        os.unlink(out_path)
                    except Exception:
                        pass
                    self.finished.emit(False, "Cancelled.")
                    return

                t = i / max(1, n_frames - 1)

                if use_gpu:
                    try:
                        frame_f32 = render_frame_gpu(
                            starless_t, stars_t, t,
                            p["sl_zoom_start"], p["sl_zoom_end"],
                            p["sl_cx_start"],   p["sl_cy_start"],
                            p["sl_cx_end"],     p["sl_cy_end"],
                            sl_ease_fn, sl_fx,
                            p["st_zoom_start"], p["st_zoom_end"],
                            p["st_cx_start"],   p["st_cy_start"],
                            p["st_cx_end"],     p["st_cy_end"],
                            st_ease_fn, st_fx,
                            out_h, out_w,
                            sl_depth_t=sl_depth_t,
                            st_depth_t=st_depth_t,
                        )
                    except Exception as e:
                        print(f"[Flythrough] GPU frame {i} failed, switching to CPU: {e}")
                        use_gpu = False

                if not use_gpu:
                    frame_f32 = render_frame(
                        starless_np, stars_np, t,
                        p["sl_zoom_start"], p["sl_zoom_end"],
                        p["sl_cx_start"],   p["sl_cy_start"],
                        p["sl_cx_end"],     p["sl_cy_end"],
                        sl_ease_fn, sl_fx,
                        p["st_zoom_start"], p["st_zoom_end"],
                        p["st_cx_start"],   p["st_cy_start"],
                        p["st_cx_end"],     p["st_cy_end"],
                        st_ease_fn, st_fx,
                        out_h, out_w,
                        sl_depth_map=sl_depth_map,
                        st_depth_map=st_depth_map,
                    )

                frame_u8  = (frame_f32 * 255.0).clip(0, 255).astype(np.uint8)
                frame_bgr = cv2.cvtColor(frame_u8, cv2.COLOR_RGB2BGR)
                writer.write(frame_bgr)
                backend = "GPU" if use_gpu else "CPU"
                self.progress.emit(i + 1, n_frames,
                                   f"[{backend}] Frame {i+1}/{n_frames}")

            writer.release()
            self.finished.emit(True, out_path)

        except Exception as e:
            self.finished.emit(False, str(e))


# ---------------------------------------------------------------------------
# Centre-picker label
# ---------------------------------------------------------------------------

class _CentrePickerLabel(QLabel):
    pointPicked = pyqtSignal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(240, 160)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self._markers: list = []
        self._qimg: QImage | None = None

    def set_image(self, arr: np.ndarray):
        rgb  = _ensure_rgb(arr)
        buf8 = np.ascontiguousarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
        h, w, _ = buf8.shape
        self._qimg = QImage(buf8.tobytes(), w, h, w * 3, QImage.Format.Format_RGB888)
        self._repaint_scaled()

    def set_markers(self, markers: list):
        self._markers = list(markers)
        self._repaint_scaled()

    def _repaint_scaled(self):
        if self._qimg is None:
            return
        pm = QPixmap.fromImage(self._qimg).scaled(
            self.width(), self.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        from PyQt6.QtGui import QPainter, QPen, QColor, QFont
        painter = QPainter(pm)
        font = QFont(); font.setPixelSize(11); font.setBold(True)
        painter.setFont(font)
        for fx, fy, colour, label_text in self._markers:
            px = int(fx * pm.width())
            py = int(fy * pm.height())
            pen = QPen(QColor(colour), 2)
            painter.setPen(pen)
            r = 6
            painter.drawEllipse(px - r, py - r, 2 * r, 2 * r)
            painter.drawLine(px - r - 3, py, px + r + 3, py)
            painter.drawLine(px, py - r - 3, px, py + r + 3)
            painter.setPen(QPen(QColor("white"), 1))
            painter.drawText(px + r + 3, py - 3, label_text)
        painter.end()
        full = QPixmap(self.width(), self.height())
        full.fill(Qt.GlobalColor.black)
        p2 = QPainter(full)
        off_x = max(0, (self.width()  - pm.width())  // 2)
        off_y = max(0, (self.height() - pm.height()) // 2)
        p2.drawPixmap(off_x, off_y, pm)
        p2.end()
        self.setPixmap(full)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._repaint_scaled()

    def mousePressEvent(self, ev):
        if ev.button() != Qt.MouseButton.LeftButton or self._qimg is None:
            return
        pm_w, pm_h   = self._qimg.width(), self._qimg.height()
        lbl_w, lbl_h = self.width(), self.height()
        scale  = min(lbl_w / pm_w, lbl_h / pm_h)
        disp_w = int(pm_w * scale); disp_h = int(pm_h * scale)
        off_x  = (lbl_w - disp_w) // 2; off_y = (lbl_h - disp_h) // 2
        px = ev.position().x() - off_x
        py = ev.position().y() - off_y
        if px < 0 or py < 0 or px > disp_w or py > disp_h:
            return
        self.pointPicked.emit(
            float(np.clip(px / max(1, disp_w), 0.0, 1.0)),
            float(np.clip(py / max(1, disp_h), 0.0, 1.0)))


# ---------------------------------------------------------------------------
# Slider row helper
# ---------------------------------------------------------------------------

def _slider_row(label: str, lo: float, hi: float, default: float,
                decimals: int = 2, scale: int = 100):
    row = QWidget()
    h   = QHBoxLayout(row)
    h.setContentsMargins(0, 0, 0, 0); h.setSpacing(6)
    h.addWidget(QLabel(label))
    sld = QSlider(Qt.Orientation.Horizontal)
    sld.setRange(int(lo * scale), int(hi * scale))
    sld.setValue(int(default * scale))
    h.addWidget(sld, 1)
    lbl = QLabel(f"{default:.{decimals}f}")
    lbl.setFixedWidth(42)
    h.addWidget(lbl)
    sld.valueChanged.connect(lambda v, l=lbl, d=decimals, s=scale:
                             l.setText(f"{v/s:.{d}f}"))
    return row, sld, lbl


# ---------------------------------------------------------------------------
# Per-layer panel
# ---------------------------------------------------------------------------

class _LayerPanel(QGroupBox):
    def __init__(self, title: str, accent_colour: str, parent=None):
        super().__init__(title, parent)
        self._accent    = accent_colour
        self._pick_mode = "start"

        self.setStyleSheet(f"""
            QGroupBox {{
                border: 2px solid {accent_colour};
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 6px;
                font-weight: bold;
                color: {accent_colour};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 6px;
                color: {accent_colour};
            }}
        """)

        form = QFormLayout(self)

        # Zoom
        self.sp_zoom_start = QDoubleSpinBox()
        self.sp_zoom_start.setRange(1.0, 50.0); self.sp_zoom_start.setSingleStep(0.5)
        self.sp_zoom_start.setValue(1.0);        self.sp_zoom_start.setDecimals(2)
        self.sp_zoom_end = QDoubleSpinBox()
        self.sp_zoom_end.setRange(1.0, 50.0); self.sp_zoom_end.setSingleStep(0.5)
        self.sp_zoom_end.setValue(6.0);        self.sp_zoom_end.setDecimals(2)
        zoom_row = QHBoxLayout()
        zoom_row.addWidget(QLabel("Start:")); zoom_row.addWidget(self.sp_zoom_start)
        zoom_row.addSpacing(12)
        zoom_row.addWidget(QLabel("End:"));   zoom_row.addWidget(self.sp_zoom_end)
        form.addRow("Zoom x:", zoom_row)

        def _dsb(v=0.5):
            s = QDoubleSpinBox()
            s.setRange(0.0, 1.0); s.setSingleStep(0.01); s.setDecimals(4); s.setValue(v)
            return s

        cx_s = QHBoxLayout()
        self.sp_cx_start = _dsb(); self.sp_cy_start = _dsb()
        cx_s.addWidget(QLabel("X:")); cx_s.addWidget(self.sp_cx_start)
        cx_s.addWidget(QLabel("Y:")); cx_s.addWidget(self.sp_cy_start)
        form.addRow("Centre start:", cx_s)

        cx_e = QHBoxLayout()
        self.sp_cx_end = _dsb(); self.sp_cy_end = _dsb()
        cx_e.addWidget(QLabel("X:")); cx_e.addWidget(self.sp_cx_end)
        cx_e.addWidget(QLabel("Y:")); cx_e.addWidget(self.sp_cy_end)
        form.addRow("Centre end:", cx_e)

        self.cmb_ease = QComboBox()
        self.cmb_ease.addItems(list(EASE_FUNCTIONS.keys()))
        self.cmb_ease.setCurrentText("Ease In-Out")
        form.addRow("Easing:", self.cmb_ease)

        pick_row = QHBoxLayout()
        self.btn_pick_start = QPushButton("Pick Start Centre")
        self.btn_pick_end   = QPushButton("Pick End Centre")
        self.btn_pick_start.setCheckable(True); self.btn_pick_end.setCheckable(True)
        pick_row.addWidget(self.btn_pick_start); pick_row.addWidget(self.btn_pick_end)
        form.addRow("", pick_row)
        self.btn_pick_start.clicked.connect(self._on_pick_start)
        self.btn_pick_end.clicked.connect(self._on_pick_end)

        # Effects group
        fx_box = QGroupBox("Lens Effects")
        fx_box.setStyleSheet(f"""
            QGroupBox {{
                border: 1px solid {accent_colour}88;
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 4px;
                font-weight: normal;
                color: {accent_colour};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin; subcontrol-position: top left; padding: 0 4px;
            }}
        """)
        fx_v = QVBoxLayout(fx_box); fx_v.setSpacing(4)

        self.chk_animate_fx = QCheckBox("Ramp effects with zoom (0 -> full at end)")
        self.chk_animate_fx.setChecked(True)
        self.chk_animate_fx.setToolTip(
            "All effects start at zero and reach their set strength at the end of the clip.")
        fx_v.addWidget(self.chk_animate_fx)

        # Depth warp  — listed first as the headline feature
        self.chk_depth_warp = QCheckBox("Nebula depth warp  (bright = closer)")
        self.chk_depth_warp.setToolTip(
            "Treats luminance as a height field and moves the camera toward it.\n"
            "Bright nebula regions zoom in faster than the dark background —\n"
            "genuine per-pixel parallax with zero banding.\n"
            "Best applied to the Starless layer.\n"
            "Strength 0.2-0.4 is usually plenty.  Depth blur controls smoothness.")
        fx_v.addWidget(self.chk_depth_warp)
        dw_row, self.sld_depth_warp, self.lbl_depth_warp = _slider_row(
            "  Strength:", 0.0, 10.0, 0.3, decimals=2, scale=100)
        dw_row.setEnabled(False)
        self.chk_depth_warp.toggled.connect(dw_row.setEnabled)
        fx_v.addWidget(dw_row)
        self._dw_row = dw_row

        dblur_row, self.sld_depth_blur, self.lbl_depth_blur = _slider_row(
            "  Depth blur:", 1.0, 40.0, 8.0, decimals=1, scale=10)
        dblur_row.setEnabled(False)
        self.chk_depth_warp.toggled.connect(dblur_row.setEnabled)
        fx_v.addWidget(dblur_row)
        self._dblur_row = dblur_row

        self.chk_depth_invert = QCheckBox("  Invert depth  (dark = closer)")
        self.chk_depth_invert.setEnabled(False)
        self.chk_depth_warp.toggled.connect(self.chk_depth_invert.setEnabled)
        fx_v.addWidget(self.chk_depth_invert)

        # Radial edge stretch
        self.chk_barrel = QCheckBox("Radial edge stretch")
        self.chk_barrel.setToolTip(
            "Edges zoom outward faster than the centre — rubber-sheet effect.")
        fx_v.addWidget(self.chk_barrel)
        barrel_row, self.sld_barrel, self.lbl_barrel = _slider_row(
            "  Strength:", -0.5, 0.5, 0.25, decimals=3, scale=1000)
        barrel_row.setEnabled(False)
        self.chk_barrel.toggled.connect(barrel_row.setEnabled)
        fx_v.addWidget(barrel_row)
        self._barrel_row = barrel_row

        # Zoom blur
        self.chk_zoom_blur = QCheckBox("Zoom blur  (warp-speed streaks)")
        self.chk_zoom_blur.setToolTip(
            "Radial motion blur from zoom centre.  Best on stars layer.")
        fx_v.addWidget(self.chk_zoom_blur)
        zb_row, self.sld_zoom_blur, self.lbl_zoom_blur = _slider_row(
            "  Strength:", 0.0, 1.0, 0.4, decimals=2, scale=100)
        zb_row.setEnabled(False)
        self.chk_zoom_blur.toggled.connect(zb_row.setEnabled)
        fx_v.addWidget(zb_row)
        self._zb_row = zb_row

        # Chromatic aberration
        self.chk_chroma = QCheckBox("Chromatic aberration")
        self.chk_chroma.setToolTip(
            "R zooms out, B zooms in.  Very effective on bright stars.")
        fx_v.addWidget(self.chk_chroma)
        ca_row, self.sld_chroma, self.lbl_chroma = _slider_row(
            "  Strength:", 0.0, 1.0, 0.5, decimals=2, scale=100)
        ca_row.setEnabled(False)
        self.chk_chroma.toggled.connect(ca_row.setEnabled)
        fx_v.addWidget(ca_row)
        self._ca_row = ca_row

        form.addRow(fx_box)

    def _on_pick_start(self, checked):
        self._pick_mode = "start" if checked else None
        self.btn_pick_end.setChecked(False)

    def _on_pick_end(self, checked):
        self._pick_mode = "end" if checked else None
        self.btn_pick_start.setChecked(False)

    def receive_picked_point(self, fx: float, fy: float):
        if self._pick_mode == "start":
            self.sp_cx_start.setValue(fx); self.sp_cy_start.setValue(fy)
            self.btn_pick_start.setChecked(False); self._pick_mode = None
        elif self._pick_mode == "end":
            self.sp_cx_end.setValue(fx); self.sp_cy_end.setValue(fy)
            self.btn_pick_end.setChecked(False); self._pick_mode = None

    @property
    def picking(self) -> bool:
        return self._pick_mode in ("start", "end")

    def get_params(self) -> dict:
        return {
            "zoom_start": float(self.sp_zoom_start.value()),
            "zoom_end":   float(self.sp_zoom_end.value()),
            "cx_start":   float(self.sp_cx_start.value()),
            "cy_start":   float(self.sp_cy_start.value()),
            "cx_end":     float(self.sp_cx_end.value()),
            "cy_end":     float(self.sp_cy_end.value()),
            "ease":       str(self.cmb_ease.currentText()),
        }

    def get_fx(self) -> dict:
        fx: dict = {"animate_effects": self.chk_animate_fx.isChecked()}
        if self.chk_depth_warp.isChecked():
            fx["depth_warp"]       = self.sld_depth_warp.value() / 100.0
            fx["depth_blur_sigma"] = self.sld_depth_blur.value() / 10.0
            fx["depth_invert"]     = self.chk_depth_invert.isChecked()
        if self.chk_barrel.isChecked():
            fx["radial_stretch"]   = self.sld_barrel.value() / 1000.0
        if self.chk_zoom_blur.isChecked():
            fx["zoom_blur"]        = self.sld_zoom_blur.value() / 100.0
            fx["zoom_blur_samples"] = 12
        if self.chk_chroma.isChecked():
            fx["chroma"]           = self.sld_chroma.value() / 100.0
        return fx

    def markers(self) -> list:
        return [
            (self.sp_cx_start.value(), self.sp_cy_start.value(), self._accent, "S"),
            (self.sp_cx_end.value(),   self.sp_cy_end.value(),   self._accent, "E"),
        ]

    def save_settings(self, s: QSettings, key: str):
        p = self.get_params()
        for k, v in p.items():
            s.setValue(f"flythrough/{key}_{k}", v)
        s.setValue(f"flythrough/{key}_animate_fx", self.chk_animate_fx.isChecked())
        s.setValue(f"flythrough/{key}_dw_on",      self.chk_depth_warp.isChecked())
        s.setValue(f"flythrough/{key}_dw",         self.sld_depth_warp.value())
        s.setValue(f"flythrough/{key}_dblur",      self.sld_depth_blur.value())
        s.setValue(f"flythrough/{key}_dinvert",    self.chk_depth_invert.isChecked())
        s.setValue(f"flythrough/{key}_barrel_on",  self.chk_barrel.isChecked())
        s.setValue(f"flythrough/{key}_barrel_k",   self.sld_barrel.value())
        s.setValue(f"flythrough/{key}_zblur_on",   self.chk_zoom_blur.isChecked())
        s.setValue(f"flythrough/{key}_zblur",      self.sld_zoom_blur.value())
        s.setValue(f"flythrough/{key}_chroma_on",  self.chk_chroma.isChecked())
        s.setValue(f"flythrough/{key}_chroma",     self.sld_chroma.value())

    def load_settings(self, s: QSettings, key: str):
        self.sp_zoom_start.setValue(float(s.value(f"flythrough/{key}_zoom_start", 1.0)))
        self.sp_zoom_end.setValue(  float(s.value(f"flythrough/{key}_zoom_end",   6.0)))
        self.sp_cx_start.setValue(  float(s.value(f"flythrough/{key}_cx_start",   0.5)))
        self.sp_cy_start.setValue(  float(s.value(f"flythrough/{key}_cy_start",   0.5)))
        self.sp_cx_end.setValue(    float(s.value(f"flythrough/{key}_cx_end",     0.5)))
        self.sp_cy_end.setValue(    float(s.value(f"flythrough/{key}_cy_end",     0.5)))
        self.cmb_ease.setCurrentText(str(s.value(f"flythrough/{key}_ease", "Ease In-Out")))
        self.chk_animate_fx.setChecked(bool(s.value(f"flythrough/{key}_animate_fx", True,  type=bool)))
        self.chk_depth_warp.setChecked(bool(s.value(f"flythrough/{key}_dw_on",    False, type=bool)))
        self.sld_depth_warp.setValue(   int( s.value(f"flythrough/{key}_dw",      30)))
        self.sld_depth_blur.setValue(   int( s.value(f"flythrough/{key}_dblur",   80)))
        self.chk_depth_invert.setChecked(bool(s.value(f"flythrough/{key}_dinvert", False, type=bool)))
        self.chk_barrel.setChecked(    bool(s.value(f"flythrough/{key}_barrel_on", False, type=bool)))
        self.sld_barrel.setValue(      int( s.value(f"flythrough/{key}_barrel_k",  250)))
        self.chk_zoom_blur.setChecked( bool(s.value(f"flythrough/{key}_zblur_on",  False, type=bool)))
        self.sld_zoom_blur.setValue(   int( s.value(f"flythrough/{key}_zblur",     40)))
        self.chk_chroma.setChecked(    bool(s.value(f"flythrough/{key}_chroma_on", False, type=bool)))
        self.sld_chroma.setValue(      int( s.value(f"flythrough/{key}_chroma",    50)))
        self._dw_row.setEnabled(self.chk_depth_warp.isChecked())
        self._dblur_row.setEnabled(self.chk_depth_warp.isChecked())
        self.chk_depth_invert.setEnabled(self.chk_depth_warp.isChecked())
        self._barrel_row.setEnabled(self.chk_barrel.isChecked())
        self._zb_row.setEnabled(self.chk_zoom_blur.isChecked())
        self._ca_row.setEnabled(self.chk_chroma.isChecked())


# ---------------------------------------------------------------------------
# Main dialog
# ---------------------------------------------------------------------------

class FlythroughDialog(QDialog):
    def __init__(self, parent, list_open_docs_fn=None, doc_manager=None):
        super().__init__(parent)
        self.setWindowTitle("Nebula Flythrough")
        self.setWindowFlag(Qt.WindowType.Window, True)
        import platform
        if platform.system() == "Darwin":
            self.setWindowFlag(Qt.WindowType.Tool, True)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setModal(False)
        self.setMinimumWidth(860)

        self._list_open_docs  = list_open_docs_fn or (lambda: [])
        self._docman          = doc_manager
        self._worker: _FlythroughWorker | None = None

        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(150)
        self._preview_timer.timeout.connect(self._refresh_preview)
        self._play_timer = QTimer(self)
        self._play_timer.setInterval(200)   # ~20fps preview playback
        self._play_timer.timeout.connect(self._on_play_tick)
        self._playing = False
        self._starless_arr:  np.ndarray | None = None
        self._stars_arr:     np.ndarray | None = None
        self._starless_full: np.ndarray | None = None
        self._stars_full:    np.ndarray | None = None

        self._build_ui()
        self._populate_combos()
        self._load_settings()

    def _build_ui(self):
        root = QVBoxLayout(self)

        src_box  = QGroupBox("Source Images")
        src_form = QFormLayout(src_box)
        self.cmb_starless = QComboBox()
        self.cmb_stars    = QComboBox()
        for cmb in (self.cmb_starless, self.cmb_stars):
            cmb.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            cmb.setMinimumContentsLength(28)
            cmb.currentIndexChanged.connect(self._on_source_changed)
        src_form.addRow("Starless image:",   self.cmb_starless)
        src_form.addRow("Stars-only image:", self.cmb_stars)

        res_row = QHBoxLayout()
        self.cmb_working_res = QComboBox()
        self.cmb_working_res.addItems(list(WORKING_RES_PRESETS.keys()))
        self.cmb_working_res.setCurrentText("HD  (1920x1080)")
        self.cmb_working_res.currentIndexChanged.connect(self._on_source_changed)
        self.lbl_working_size = QLabel("")
        self.lbl_working_size.setStyleSheet("color:#888; font-size:11px;")
        res_row.addWidget(self.cmb_working_res, 1)
        res_row.addWidget(self.lbl_working_size)
        src_form.addRow("Working resolution:", res_row)
        root.addWidget(src_box)

        body = QHBoxLayout()
        root.addLayout(body, 1)

        left = QVBoxLayout()
        layers_row = QHBoxLayout()
        self.panel_sl = _LayerPanel("Starless Layer", "#88aaff")
        self.panel_st = _LayerPanel("Stars Layer",    "#ffcc66")
        layers_row.addWidget(self.panel_sl, 1)
        layers_row.addWidget(self.panel_st, 1)
        left.addLayout(layers_row)

        picker_box = QGroupBox("Click image to set zoom centre points")
        picker_v   = QVBoxLayout(picker_box)
        self.picker = _CentrePickerLabel()
        self.picker.setMinimumHeight(180)
        self.picker.pointPicked.connect(self._on_point_picked)
        picker_v.addWidget(self.picker)
        left.addWidget(picker_box, 1)
        body.addLayout(left, 3)

        right = QVBoxLayout()

        prev_box = QGroupBox("Frame Preview")
        prev_v   = QVBoxLayout(prev_box)
        self.preview_lbl = QLabel()
        self.preview_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_lbl.setMinimumSize(320, 220)
        self.preview_lbl.setStyleSheet("background:#111;")
        prev_v.addWidget(self.preview_lbl, 1)

        scrub_row = QHBoxLayout()
        scrub_row.addWidget(QLabel("t:"))
        self.sld_scrub = QSlider(Qt.Orientation.Horizontal)
        self.sld_scrub.setRange(0, 100); self.sld_scrub.setValue(0)
        self.sld_scrub.valueChanged.connect(self._schedule_preview)
        scrub_row.addWidget(self.sld_scrub, 1)
        self.lbl_scrub_t = QLabel("0.00")
        self.lbl_scrub_t.setFixedWidth(32)
        scrub_row.addWidget(self.lbl_scrub_t)
        prev_v.addLayout(scrub_row)

        self.btn_play = QPushButton("▶  Play Preview")
        self.btn_play.setCheckable(True)
        self.btn_play.clicked.connect(self._toggle_play)
        prev_v.addWidget(self.btn_play)
        right.addWidget(prev_box, 1)

        out_box  = QGroupBox("Output")
        out_form = QFormLayout(out_box)
        out_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)

        res_out = QHBoxLayout()
        self.sp_out_w = QSpinBox(); self.sp_out_w.setRange(64, 7680)
        self.sp_out_w.setValue(1920); self.sp_out_w.setSuffix(" px")
        self.sp_out_h = QSpinBox(); self.sp_out_h.setRange(64, 4320)
        self.sp_out_h.setValue(1080); self.sp_out_h.setSuffix(" px")
        res_out.addWidget(self.sp_out_w); res_out.addWidget(QLabel("x")); res_out.addWidget(self.sp_out_h)
        out_form.addRow("Resolution:", res_out)

        fps_row = QHBoxLayout()
        self.sp_fps = QSpinBox(); self.sp_fps.setRange(1, 120); self.sp_fps.setValue(30); self.sp_fps.setSuffix(" fps")
        self.sp_duration = QDoubleSpinBox(); self.sp_duration.setRange(0.5, 300.0)
        self.sp_duration.setValue(10.0); self.sp_duration.setSuffix(" s"); self.sp_duration.setSingleStep(0.5)
        fps_row.addWidget(self.sp_fps); fps_row.addWidget(QLabel("Dur:")); fps_row.addWidget(self.sp_duration)
        out_form.addRow("FPS:", fps_row)

        self.lbl_frame_count = QLabel("300 frames")
        self.lbl_frame_count.setStyleSheet("color:#888; font-size:11px;")
        out_form.addRow("", self.lbl_frame_count)
        self.sp_fps.valueChanged.connect(self._update_frame_count)
        self.sp_duration.valueChanged.connect(self._update_frame_count)
        right.addWidget(out_box)

        self.pbar = QProgressBar()
        self.pbar.setRange(0, 100); self.pbar.setValue(0); self.pbar.setVisible(False)
        right.addWidget(self.pbar)

        self.lbl_status = QLabel("Ready")
        self.lbl_status.setStyleSheet("color:#aaa; font-size:11px;")
        self.lbl_status.setWordWrap(True)
        right.addWidget(self.lbl_status)

        self.btn_export = QPushButton("Export Video...")
        self.btn_export.setStyleSheet("font-weight:bold; padding:6px 18px;")
        self.btn_export.clicked.connect(self._export)
        right.addWidget(self.btn_export)

        self.btn_cancel_export = QPushButton("Cancel Export")
        self.btn_cancel_export.setVisible(False)
        self.btn_cancel_export.clicked.connect(self._cancel_export)
        right.addWidget(self.btn_cancel_export)

        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.close)
        right.addWidget(btn_close)
        right.addStretch(1)
        body.addLayout(right, 2)

        foot = QLabel("Franklin Marek  |  www.setiastro.com")
        foot.setAlignment(Qt.AlignmentFlag.AlignCenter)
        foot.setStyleSheet("color:#444; font-size:10px;")
        root.addWidget(foot)

        for panel in (self.panel_sl, self.panel_st):
            for sp in (panel.sp_zoom_start, panel.sp_zoom_end,
                       panel.sp_cx_start, panel.sp_cy_start,
                       panel.sp_cx_end, panel.sp_cy_end):
                sp.valueChanged.connect(self._schedule_preview)
            panel.cmb_ease.currentIndexChanged.connect(self._schedule_preview)
            for btn in (panel.btn_pick_start, panel.btn_pick_end):
                btn.clicked.connect(self._update_markers)
            for sp in (panel.sp_cx_start, panel.sp_cy_start,
                       panel.sp_cx_end, panel.sp_cy_end):
                sp.valueChanged.connect(self._update_markers)
            for sld in (panel.sld_barrel, panel.sld_zoom_blur, panel.sld_chroma,
                        panel.sld_depth_warp, panel.sld_depth_blur):
                sld.valueChanged.connect(self._schedule_preview)
            for chk in (panel.chk_barrel, panel.chk_zoom_blur, panel.chk_chroma,
                        panel.chk_animate_fx, panel.chk_depth_warp,
                        panel.chk_depth_invert):
                chk.toggled.connect(self._schedule_preview)

    def _toggle_play(self, checked: bool):
        self._playing = checked
        if checked:
            self.btn_play.setText("⏸  Pause")
            # If scrubber is at end, rewind first
            if self.sld_scrub.value() >= self.sld_scrub.maximum():
                self.sld_scrub.setValue(0)
            self._play_timer.start()
        else:
            self.btn_play.setText("▶  Play Preview")
            self._play_timer.stop()

    def _on_play_tick(self):
        val = self.sld_scrub.value() + 1
        if val > self.sld_scrub.maximum():
            self._play_timer.stop()
            self._playing = False
            self.btn_play.setChecked(False)
            self.btn_play.setText("▶  Play Preview")
            self.sld_scrub.setValue(0)
        else:
            # Block signals so valueChanged doesn't trigger the debounce timer
            self.sld_scrub.blockSignals(True)
            self.sld_scrub.setValue(val)
            self.sld_scrub.blockSignals(False)
            self.lbl_scrub_t.setText(f"t={val/100.0:.2f}")
            # Render directly — bypass the debounce timer entirely
            self._refresh_preview()

    def _populate_combos(self):
        for cmb in (self.cmb_starless, self.cmb_stars):
            cmb.blockSignals(True); cmb.clear()
            cmb.addItem("-- select --", None)
            for title, doc in self._list_open_docs():
                cmb.addItem(title, doc)
            cmb.blockSignals(False)

    def _on_source_changed(self):
        doc_sl = self.cmb_starless.currentData()
        doc_st = self.cmb_stars.currentData()
        preset = WORKING_RES_PRESETS.get(self.cmb_working_res.currentText())

        sl_raw = (_ensure_rgb(doc_sl.image)
                  if doc_sl and getattr(doc_sl, "image", None) is not None else None)
        st_raw = (_ensure_rgb(doc_st.image)
                  if doc_st and getattr(doc_st, "image", None) is not None else None)

        self._starless_full = sl_raw
        self._stars_full    = st_raw

        if sl_raw is not None:
            self._starless_arr = (_downsize_to_fit(sl_raw, preset[0], preset[1])
                                  if preset else sl_raw)
            h, w = self._starless_arr.shape[:2]
            self.lbl_working_size.setText(
                f"{w}x{h} px  ({w*h/1e6:.1f} MP)  --  export uses full res")
            self.picker.set_image(self._starless_arr)
            self.sp_out_w.setValue(w); self.sp_out_h.setValue(h)
        else:
            self._starless_arr = None
            self.lbl_working_size.setText("")

        if st_raw is not None:
            if preset and self._starless_arr is not None:
                th, tw = self._starless_arr.shape[:2]
                self._stars_arr = _downsize_to_fit(st_raw, tw, th)
            elif preset:
                self._stars_arr = _downsize_to_fit(st_raw, preset[0], preset[1])
            else:
                self._stars_arr = st_raw
        else:
            self._stars_arr = None

        self._update_markers(); self._schedule_preview()

    def _update_frame_count(self):
        n = max(1, int(round(self.sp_fps.value() * self.sp_duration.value())))
        self.lbl_frame_count.setText(f"{n} frames")

    def _update_markers(self):
        self.picker.set_markers(self.panel_sl.markers() + self.panel_st.markers())

    def _on_point_picked(self, fx: float, fy: float):
        for panel in (self.panel_sl, self.panel_st):
            if panel.picking:
                panel.receive_picked_point(fx, fy)
                self._update_markers(); self._schedule_preview()
                return

    def _schedule_preview(self):
        self._preview_timer.start()

    def _refresh_preview(self):
        if self._starless_arr is None or self._stars_arr is None:
            self.preview_lbl.setText("Select both source images to preview.")
            return

        t_raw = self.sld_scrub.value() / 100.0
        self.lbl_scrub_t.setText(f"t={t_raw:.2f}")

        sl_p  = self.panel_sl.get_params()
        st_p  = self.panel_st.get_params()
        sl_fx = self.panel_sl.get_fx()
        st_fx = self.panel_st.get_fx()

        # Build depth maps from working-res arrays for preview
        sl_dm = None
        if sl_fx.get("depth_warp", 0.0) > 1e-4:
            sl_dm = _build_depth_map(self._starless_arr,
                                      blur_sigma=float(sl_fx.get("depth_blur_sigma", 8.0)),
                                      invert=bool(sl_fx.get("depth_invert", False)))
        st_dm = None
        if st_fx.get("depth_warp", 0.0) > 1e-4:
            st_dm = _build_depth_map(self._stars_arr,
                                      blur_sigma=float(st_fx.get("depth_blur_sigma", 8.0)),
                                      invert=bool(st_fx.get("depth_invert", False)))

        out_w = min(self.sp_out_w.value(), 640)
        out_h = min(self.sp_out_h.value(), 360)

        try:
            frame = render_frame(
                self._starless_arr, self._stars_arr, t_raw,
                sl_p["zoom_start"], sl_p["zoom_end"],
                sl_p["cx_start"],   sl_p["cy_start"],
                sl_p["cx_end"],     sl_p["cy_end"],
                EASE_FUNCTIONS[sl_p["ease"]], sl_fx,
                st_p["zoom_start"], st_p["zoom_end"],
                st_p["cx_start"],   st_p["cy_start"],
                st_p["cx_end"],     st_p["cy_end"],
                EASE_FUNCTIONS[st_p["ease"]], st_fx,
                out_h, out_w,
                sl_depth_map=sl_dm, st_depth_map=st_dm,
            )
            buf8 = np.ascontiguousarray((frame * 255.0).clip(0, 255).astype(np.uint8))
            h, w, _ = buf8.shape
            qimg = QImage(buf8.tobytes(), w, h, w * 3, QImage.Format.Format_RGB888)
            pm = QPixmap.fromImage(qimg).scaled(
                self.preview_lbl.width(), self.preview_lbl.height(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation)
            self.preview_lbl.setPixmap(pm)
        except Exception as e:
            self.preview_lbl.setText(f"Preview error: {e}")

    def _load_settings(self):
        s = QSettings()
        self.sp_fps.setValue(     int(  s.value("flythrough/fps",      30)))
        self.sp_duration.setValue(float(s.value("flythrough/duration", 10.0)))
        self.sp_out_w.setValue(   int(  s.value("flythrough/out_w",    1920)))
        self.sp_out_h.setValue(   int(  s.value("flythrough/out_h",    1080)))
        self.cmb_working_res.setCurrentText(
            str(s.value("flythrough/working_res", "HD  (1920x1080)")))
        self.panel_sl.load_settings(s, "sl")
        self.panel_st.load_settings(s, "st")
        self._update_frame_count()

    def _save_settings(self):
        s = QSettings()
        s.setValue("flythrough/fps",         self.sp_fps.value())
        s.setValue("flythrough/duration",    self.sp_duration.value())
        s.setValue("flythrough/out_w",       self.sp_out_w.value())
        s.setValue("flythrough/out_h",       self.sp_out_h.value())
        s.setValue("flythrough/working_res", self.cmb_working_res.currentText())
        self.panel_sl.save_settings(s, "sl")
        self.panel_st.save_settings(s, "st")

    def _export(self):
        if self._starless_full is None or self._stars_full is None:
            QMessageBox.warning(self, "Flythrough", "Please select both source images first.")
            return
        if not HAS_CV2:
            QMessageBox.critical(self, "Flythrough", "OpenCV (cv2) is required for video export.")
            return

        s        = QSettings()
        last_dir = str(s.value("flythrough/last_export_dir", ""))
        default  = (os.path.join(last_dir, "nebula_flythrough.mp4")
                    if last_dir else "nebula_flythrough.mp4")

        out_path, _ = QFileDialog.getSaveFileName(
            self, "Save Flythrough Video", default,
            "MP4 Video (*.mp4);;All Files (*)")
        if not out_path:
            return

        s.setValue("flythrough/last_export_dir",
                   os.path.dirname(os.path.abspath(out_path)))
        self._save_settings()

        sl_p  = self.panel_sl.get_params()
        st_p  = self.panel_st.get_params()
        sl_fx = self.panel_sl.get_fx()
        st_fx = self.panel_st.get_fx()

        params = {
            "starless":      self._starless_full,
            "stars":         self._stars_full,
            "fps":           self.sp_fps.value(),
            "duration":      self.sp_duration.value(),
            "out_w":         self.sp_out_w.value(),
            "out_h":         self.sp_out_h.value(),
            "out_path":      out_path,
            "sl_zoom_start": sl_p["zoom_start"], "sl_zoom_end": sl_p["zoom_end"],
            "sl_cx_start":   sl_p["cx_start"],   "sl_cy_start": sl_p["cy_start"],
            "sl_cx_end":     sl_p["cx_end"],      "sl_cy_end":   sl_p["cy_end"],
            "sl_ease":       sl_p["ease"],         "sl_fx":       sl_fx,
            "st_zoom_start": st_p["zoom_start"], "st_zoom_end": st_p["zoom_end"],
            "st_cx_start":   st_p["cx_start"],   "st_cy_start": st_p["cy_start"],
            "st_cx_end":     st_p["cx_end"],      "st_cy_end":   st_p["cy_end"],
            "st_ease":       st_p["ease"],         "st_fx":       st_fx,
        }

        n_frames = max(1, int(round(params["fps"] * params["duration"])))
        self.pbar.setRange(0, n_frames); self.pbar.setValue(0); self.pbar.setVisible(True)
        self.btn_export.setEnabled(False); self.btn_cancel_export.setVisible(True)

        full_h, full_w = self._starless_full.shape[:2]
        self.lbl_status.setText(
            f"Rendering {params['out_w']}x{params['out_h']} -- "
            f"cropping from {full_w}x{full_h} originals...")

        self._worker = _FlythroughWorker(params, parent=None)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    def _cancel_export(self):
        if self._worker: self._worker.cancel()

    def _on_progress(self, done: int, total: int, msg: str):
        self.pbar.setValue(done); self.lbl_status.setText(msg)

    def _on_finished(self, ok: bool, msg: str):
        if self._worker: self._worker.wait()
        self._worker = None
        self.pbar.setVisible(False)
        self.btn_export.setEnabled(True); self.btn_cancel_export.setVisible(False)
        if ok:
            self.lbl_status.setText(f"Done -> {os.path.basename(msg)}")
            QMessageBox.information(self, "Flythrough", f"Video exported successfully:\n{msg}")
        else:
            self.lbl_status.setText(f"Failed: {msg}")
            if msg != "Cancelled.":
                QMessageBox.critical(self, "Flythrough", f"Export failed:\n{msg}")

    def closeEvent(self, ev):
        self._play_timer.stop()
        if self._worker and self._worker.isRunning():
            self._worker.cancel(); self._worker.wait(2000)
        self._save_settings()
        super().closeEvent(ev)

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def open_flythrough_dialog(parent, list_open_docs_fn=None, doc_manager=None):
    dlg = FlythroughDialog(parent,
                            list_open_docs_fn=list_open_docs_fn,
                            doc_manager=doc_manager)
    dlg.show(); dlg.raise_()
    return dlg