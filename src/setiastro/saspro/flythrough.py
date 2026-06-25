# src/setiastro/saspro/flythrough.py — SASpro Nebula Flythrough Tool
# =============================================================================
#
#  Composites a stars-only image over a starless image using Screen blend,
#  with an optional mid layer blended between them.
#
#  Compositing order:
#      base      = composite(starless, mid, blend_mode, opacity)   ← mid over starless
#      final     = screen(stars, base)                              ← stars screened on top
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
# Mid-layer blend modes
# ---------------------------------------------------------------------------

# Names shown in the UI dropdown — shared by mid and stars layers
LAYER_BLEND_MODES = [
    "Screen",
    "Add",
    "Average",
    "Max",
    "Multiply",
    "Overlay",
    "Soft Light",
    "Hard Light",
    "Difference",
    "Color Dodge",
    "Lighten",
    "Darken",
    "Threshold Mask",   # pixels above threshold sit on top of base; below are transparent
]

# Back-compat alias
MID_BLEND_MODES = LAYER_BLEND_MODES


def _blend_mid_cpu(base: np.ndarray, mid: np.ndarray,
                   mode: str, opacity: float, **kwargs) -> np.ndarray:
    """
    Blend `mid` over `base` using `mode` at `opacity`.
    All arrays are float32 HxWx3 in [0,1].
    Returns float32 HxWx3.
    """
    b = np.clip(base.astype(np.float32, copy=False), 0.0, 1.0)
    m = np.clip(mid.astype(np.float32,  copy=False), 0.0, 1.0)

    if mode == "Screen":
        blended = 1.0 - (1.0 - b) * (1.0 - m)
    elif mode == "Add":
        blended = np.clip(b + m, 0.0, 1.0)
    elif mode == "Average":
        blended = (b + m) * 0.5
    elif mode == "Multiply":
        blended = b * m
    elif mode == "Overlay":
        blended = np.where(
            b < 0.5,
            2.0 * b * m,
            1.0 - 2.0 * (1.0 - b) * (1.0 - m),
        )
    elif mode == "Soft Light":
        # Pegtop formula
        blended = (1.0 - 2.0 * m) * b * b + 2.0 * m * b
    elif mode == "Hard Light":
        blended = np.where(
            m < 0.5,
            2.0 * b * m,
            1.0 - 2.0 * (1.0 - b) * (1.0 - m),
        )
    elif mode == "Difference":
        blended = np.abs(b - m)
    elif mode == "Color Dodge":
        blended = np.where(m >= 1.0, 1.0, np.clip(b / np.maximum(1.0 - m, 1e-7), 0.0, 1.0))
    elif mode == "Max":
        blended = np.maximum(b, m)
    elif mode == "Lighten":
        blended = np.maximum(b, m)
    elif mode == "Darken":
        blended = np.minimum(b, m)
    elif mode == "Threshold Mask":
        # Alpha compositing: pixels above threshold sit fully on top of base.
        # alpha=1 → show layer pixel; alpha=0 → show base pixel.
        thr     = float(kwargs.get("threshold", 0.2))
        feather = max(float(kwargs.get("feather", 0.1)), 1e-4)
        lum     = (0.2126 * m[..., 0] + 0.7152 * m[..., 1] + 0.0722 * m[..., 2])[..., None]
        alpha   = np.clip((lum - thr) / feather, 0.0, 1.0)
        blended = b * (1.0 - alpha) + m * alpha   # straight over composite
    else:
        blended = (b + m) * 0.5   # fallback: Average

    blended = np.clip(blended, 0.0, 1.0).astype(np.float32)
    # Apply opacity: lerp between base and blended result
    if opacity < 1.0:
        blended = b + opacity * (blended - b)
    return np.clip(blended, 0.0, 1.0).astype(np.float32)


def _blend_mid_gpu(base_t, mid_t, mode: str, opacity: float, **kwargs):
    """GPU equivalent of _blend_mid_cpu. Inputs are NCHW tensors."""
    import torch
    b = torch.clamp(base_t, 0.0, 1.0)
    m = torch.clamp(mid_t,  0.0, 1.0)

    if mode == "Screen":
        blended = 1.0 - (1.0 - b) * (1.0 - m)
    elif mode == "Add":
        blended = torch.clamp(b + m, 0.0, 1.0)
    elif mode == "Average":
        blended = (b + m) * 0.5
    elif mode == "Multiply":
        blended = b * m
    elif mode == "Overlay":
        blended = torch.where(b < 0.5,
                              2.0 * b * m,
                              1.0 - 2.0 * (1.0 - b) * (1.0 - m))
    elif mode == "Soft Light":
        blended = (1.0 - 2.0 * m) * b * b + 2.0 * m * b
    elif mode == "Hard Light":
        blended = torch.where(m < 0.5,
                              2.0 * b * m,
                              1.0 - 2.0 * (1.0 - b) * (1.0 - m))
    elif mode == "Difference":
        blended = torch.abs(b - m)
    elif mode == "Color Dodge":
        blended = torch.clamp(
            torch.where(m >= 1.0,
                        torch.ones_like(b),
                        b / torch.clamp(1.0 - m, min=1e-7)),
            0.0, 1.0)
    elif mode == "Max":
        blended = torch.maximum(b, m)
    elif mode == "Lighten":
        blended = torch.maximum(b, m)
    elif mode == "Darken":
        blended = torch.minimum(b, m)
    elif mode == "Threshold Mask":
        # Alpha compositing: pixels above threshold sit fully on top of base.
        thr     = float(kwargs.get("threshold", 0.2))
        feather = max(float(kwargs.get("feather", 0.1)), 1e-4)
        lum     = (0.2126 * m[:, 0:1] + 0.7152 * m[:, 1:2] + 0.0722 * m[:, 2:3])
        alpha   = torch.clamp((lum - thr) / feather, 0.0, 1.0)
        blended = b * (1.0 - alpha) + m * alpha   # straight over composite
    else:
        blended = (b + m) * 0.5

    blended = torch.clamp(blended, 0.0, 1.0)
    if opacity < 1.0:
        blended = b + opacity * (blended - b)
    return torch.clamp(blended, 0.0, 1.0)


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
# Depth map builder
# ---------------------------------------------------------------------------

def _build_depth_map(img: np.ndarray,
                     blur_sigma: float = 8.0,
                     depth_gamma: float = 1.0,
                     invert: bool = False) -> np.ndarray:
    img = _ensure_rgb(img)
    lum = (0.299 * img[:, :, 0] +
           0.587 * img[:, :, 1] +
           0.114 * img[:, :, 2]).astype(np.float32)

    if HAS_CV2 and blur_sigma > 0:
        lum = cv2.GaussianBlur(lum, (0, 0), float(blur_sigma))

    if invert:
        lum = 1.0 - lum

    lo, hi = float(lum.min()), float(lum.max())
    if hi > lo:
        lum = (lum - lo) / (hi - lo)

    return lum.astype(np.float32)


# ---------------------------------------------------------------------------
# CPU zoom-crop  (flat)
# ---------------------------------------------------------------------------

def _zoom_crop(img: np.ndarray,
               zoom: float,
               cx_frac: float, cy_frac: float,
               out_h: int, out_w: int) -> np.ndarray:
    """
    Zoom crop with torus (wrap-around) boundary — no edge streaks or reflections.
    When the crop window extends beyond the image edge it wraps to the opposite side,
    like a seamless tileable texture.
    """
    H, W = img.shape[:2]
    crop_w = max(1, int(round(W / zoom)))
    crop_h = max(1, int(round(H / zoom)))
    cx_px  = cx_frac * W
    cy_px  = cy_frac * H

    if HAS_CV2:
        # Build source coordinates in image pixel space, then wrap with modulo.
        out_ys = np.linspace(cy_px - crop_h / 2.0, cy_px + crop_h / 2.0,
                              out_h, dtype=np.float32)
        out_xs = np.linspace(cx_px - crop_w / 2.0, cx_px + crop_w / 2.0,
                              out_w, dtype=np.float32)
        # Torus wrap: modulo keeps coords in [0, W) and [0, H)
        out_xs = (out_xs % W).reshape(1, -1).repeat(out_h, axis=0)
        out_ys = (out_ys % H).reshape(-1, 1).repeat(out_w, axis=1)
        return cv2.remap(img,
                         out_xs.astype(np.float32),
                         out_ys.astype(np.float32),
                         interpolation=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_WRAP).astype(np.float32)

    # Fallback (no cv2): nearest-neighbour with integer wrap
    ys = (np.linspace(cy_px - crop_h / 2.0, cy_px + crop_h / 2.0,
                      out_h).astype(np.int32)) % H
    xs = (np.linspace(cx_px - crop_w / 2.0, cx_px + crop_w / 2.0,
                      out_w).astype(np.int32)) % W
    return (img[ys][:, xs] if img.ndim == 2
            else img[np.ix_(ys, xs)]).astype(np.float32)


# ---------------------------------------------------------------------------
# CPU depth-aware zoom crop
# ---------------------------------------------------------------------------

def _zoom_crop_depth(img, depth_map, zoom_base, depth_strength,
                     cx_frac, cy_frac, out_h, out_w):
    if not HAS_CV2 or depth_strength < 1e-4:
        return _zoom_crop(img, zoom_base, cx_frac, cy_frac, out_h, out_w)

    H, W = img.shape[:2]
    cx_px = cx_frac * W
    cy_px = cy_frac * H

    cols = (np.arange(out_w, dtype=np.float32) + 0.5) / out_w - 0.5
    rows = (np.arange(out_h, dtype=np.float32) + 0.5) / out_h - 0.5
    u, v = np.meshgrid(cols, rows)

    crop_w = W / zoom_base
    crop_h = H / zoom_base
    base_src_x = cx_px + u * crop_w
    base_src_y = cy_px + v * crop_h

    # Torus wrap for depth sampling
    sx = (base_src_x % W).astype(np.float32)
    sy = (base_src_y % H).astype(np.float32)
    dm_sampled = cv2.remap(depth_map, sx, sy,
                           interpolation=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_WRAP)

    dx = u * crop_w
    dy = v * crop_h
    dist = np.sqrt(dx * dx + dy * dy)
    safe_dist = np.maximum(dist, 1e-6)
    nx = dx / safe_dist
    ny = dy / safe_dist

    dm_centerd = dm_sampled - 0.5
    parallax_px = dm_centerd * depth_strength * (zoom_base - 1.0) * 2.0

    # Torus wrap for final sample
    final_src_x = ((base_src_x - nx * parallax_px) % W).astype(np.float32)
    final_src_y = ((base_src_y - ny * parallax_px) % H).astype(np.float32)

    return cv2.remap(img, final_src_x, final_src_y,
                     interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_WRAP).astype(np.float32)


# ---------------------------------------------------------------------------
# CPU optical effects
# ---------------------------------------------------------------------------

def apply_radial_stretch(img: np.ndarray,
                          strength: float,
                          cx_frac: float = 0.5,
                          cy_frac: float = 0.5) -> np.ndarray:
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
                     borderMode=cv2.BORDER_WRAP).astype(np.float32)


def apply_zoom_blur(img: np.ndarray,
                    strength: float,
                    cx_frac: float = 0.5,
                    cy_frac: float = 0.5,
                    samples: int = 12) -> np.ndarray:
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
    if not fx:
        return frame

    animate = bool(fx.get("animate_effects", True))
    if animate:
        zoom_vel = float(fx.get("zoom_velocity", 1.0))
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
    return t.permute(2, 0, 1).unsqueeze(0)


def _tensor_to_np(t) -> np.ndarray:
    return t.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float32)


def _np_depth_to_tensor(dm: np.ndarray, device):
    import torch
    return torch.from_numpy(np.ascontiguousarray(dm)).to(device).unsqueeze(0).unsqueeze(0)


# ---------------------------------------------------------------------------
# GPU flat zoom crop
# ---------------------------------------------------------------------------

def _wrap_norm_coords(coords):
    """
    Fold normalised grid_sample coords (range [-1,1]) into torus wrap.
    Maps any value outside [-1,1] back using modulo so the image tiles seamlessly.
    """
    # Shift to [0,2], modulo 2, shift back to [-1,1]
    return ((coords + 1.0) % 2.0) - 1.0


def _zoom_crop_gpu(t, zoom: float, cx_frac: float, cy_frac: float,
                   out_h: int, out_w: int):
    """Zoom crop with torus wrap — no edge streaks or reflections."""
    import torch
    import torch.nn.functional as F

    _, C, H, W = t.shape

    # Centre in normalised coords (no clamping — wrap handles out-of-bounds)
    cx_n = float(cx_frac * 2.0 - 1.0)
    cy_n = float(cy_frac * 2.0 - 1.0)
    hx = 1.0 / max(zoom, 1e-4)
    hy = 1.0 / max(zoom, 1e-4)

    gx = torch.linspace(cx_n - hx, cx_n + hx, out_w, device=t.device, dtype=torch.float32)
    gy = torch.linspace(cy_n - hy, cy_n + hy, out_h, device=t.device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(gy, gx, indexing="ij")

    # Torus wrap: fold any out-of-range normalised coords back into [-1, 1]
    grid_x = _wrap_norm_coords(grid_x)
    grid_y = _wrap_norm_coords(grid_y)

    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
    # zeros padding_mode is fine here because _wrap_norm_coords already keeps
    # all coords inside [-1, 1], so the padding zone is never reached.
    return F.grid_sample(t, grid, mode="bilinear", padding_mode="zeros", align_corners=True)


# ---------------------------------------------------------------------------
# GPU depth-aware zoom crop
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

    cols = (torch.arange(out_w, device=dev, dtype=torch.float32) + 0.5) / out_w - 0.5
    rows = (torch.arange(out_h, device=dev, dtype=torch.float32) + 0.5) / out_h - 0.5
    v_g, u_g = torch.meshgrid(rows, cols, indexing="ij")

    base_src_x = cx_px + u_g * crop_w
    base_src_y = cy_px + v_g * crop_h

    # Torus wrap for depth map sampling
    norm_bx = _wrap_norm_coords((base_src_x % W) / (W - 1) * 2.0 - 1.0)
    norm_by = _wrap_norm_coords((base_src_y % H) / (H - 1) * 2.0 - 1.0)
    depth_grid = torch.stack([norm_bx, norm_by], dim=-1).unsqueeze(0)
    dm_sampled = F.grid_sample(depth_t, depth_grid,
                                mode="bilinear", padding_mode="zeros",
                                align_corners=True).squeeze(0).squeeze(0)

    dx = u_g * crop_w
    dy = v_g * crop_h
    dist = torch.sqrt(dx * dx + dy * dy).clamp(min=1e-6)
    nx = dx / dist
    ny = dy / dist

    dm_centerd = dm_sampled - 0.5
    parallax_px = dm_centerd * depth_strength * (zoom_base - 1.0) * 2.0

    # Torus wrap for final image sample
    final_src_x = (base_src_x - nx * parallax_px) % W
    final_src_y = (base_src_y - ny * parallax_px) % H

    norm_fx = _wrap_norm_coords(final_src_x / (W - 1) * 2.0 - 1.0)
    norm_fy = _wrap_norm_coords(final_src_y / (H - 1) * 2.0 - 1.0)
    grid = torch.stack([norm_fx, norm_fy], dim=-1).unsqueeze(0)

    return F.grid_sample(t, grid, mode="bilinear",
                          padding_mode="zeros", align_corners=True)


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
    grid = torch.stack([_wrap_norm_coords(src_x), _wrap_norm_coords(src_y)], dim=-1).unsqueeze(0)
    return F.grid_sample(t, grid, mode="bilinear", padding_mode="zeros", align_corners=True)


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
        return torch.stack([_wrap_norm_coords(grid_x), _wrap_norm_coords(grid_y)], dim=-1).unsqueeze(0)

    out = t.clone()
    if r_zoom > 0.01:
        out[:, 0:1] = F.grid_sample(t[:, 0:1], _channel_grid(r_zoom),
                                     mode="bilinear", padding_mode="zeros", align_corners=True)
    out[:, 2:3] = F.grid_sample(t[:, 2:3], _channel_grid(b_zoom),
                                 mode="bilinear", padding_mode="zeros", align_corners=True)
    return torch.clamp(out, 0.0, 1.0)


def _screen_blend_gpu(a, b):
    return 1.0 - (1.0 - a) * (1.0 - b)


# ---------------------------------------------------------------------------
# CPU render_frame  (now with optional mid layer)
# ---------------------------------------------------------------------------

def render_frame(
    starless: np.ndarray,
    stars:    np.ndarray | None,
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
    # ── stars blend mode (default Screen for back-compat) ─────────────
    st_blend_mode: str = "Screen",
    st_opacity: float = 1.0,
    # ── mid layer (all optional) ──────────────────────────────────────
    mid: np.ndarray | None = None,
    mid_zoom_start: float = 1.0, mid_zoom_end: float = 1.0,
    mid_cx_start: float = 0.5,   mid_cy_start: float = 0.5,
    mid_cx_end: float = 0.5,     mid_cy_end: float = 0.5,
    mid_ease_fn = None,
    mid_fx: dict | None = None,
    mid_blend_mode: str = "Screen",
    mid_opacity: float = 1.0,
    mid_depth_map: np.ndarray | None = None,
) -> np.ndarray:
    """
    Render one composited frame at normalised time t ∈ [0,1].

    Compositing order:
        base  = composite(starless, mid, blend_mode, opacity)   [if mid enabled]
        frame = screen(stars, base)
    """
    te_sl = sl_ease_fn(t)
    te_st = st_ease_fn(t)

    sl_zoom = sl_zoom_start + (sl_zoom_end - sl_zoom_start) * te_sl
    sl_cx   = sl_cx_start   + (sl_cx_end   - sl_cx_start)   * te_sl
    sl_cy   = sl_cy_start   + (sl_cy_end   - sl_cy_start)   * te_sl
    st_zoom = st_zoom_start + (st_zoom_end - st_zoom_start) * te_st
    st_cx   = st_cx_start   + (st_cx_end   - st_cx_start)   * te_st
    st_cy   = st_cy_start   + (st_cy_end   - st_cy_start)   * te_st

    eps = 0.005
    t_lo = max(0.0, t - eps)
    t_hi = min(1.0, t + eps)

    sl_vel_raw = (sl_ease_fn(t_hi) - sl_ease_fn(t_lo)) / max(t_hi - t_lo, 1e-9)
    st_vel_raw = (st_ease_fn(t_hi) - st_ease_fn(t_lo)) / max(t_hi - t_lo, 1e-9)
    sl_zoom_vel = float(np.clip(sl_vel_raw / 3.0, 0.0, 1.0))
    st_zoom_vel = float(np.clip(st_vel_raw / 3.0, 0.0, 1.0))

    dw_sl = float(sl_fx.get("depth_warp", 0.0))
    dw_st = float(st_fx.get("depth_warp", 0.0))

    if sl_depth_map is not None and dw_sl > 1e-4:
        sl_frame = _zoom_crop_depth(starless, sl_depth_map, sl_zoom, dw_sl,
                                     sl_cx, sl_cy, out_h, out_w)
    else:
        sl_frame = _zoom_crop(starless, sl_zoom, sl_cx, sl_cy, out_h, out_w)

    sl_fx_post = {k: v for k, v in sl_fx.items() if k != "depth_warp"}
    sl_fx_post["zoom_velocity"] = sl_zoom_vel
    sl_rgb = _apply_layer_effects(_ensure_rgb(sl_frame), t, sl_cx, sl_cy, sl_fx_post)

    if stars is not None:
        if st_depth_map is not None and dw_st > 1e-4:
            st_frame = _zoom_crop_depth(stars, st_depth_map, st_zoom, dw_st,
                                         st_cx, st_cy, out_h, out_w)
        else:
            st_frame = _zoom_crop(stars, st_zoom, st_cx, st_cy, out_h, out_w)
        st_fx_post = {k: v for k, v in st_fx.items() if k != "depth_warp"}
        st_fx_post["zoom_velocity"] = st_zoom_vel
        st_rgb = _apply_layer_effects(_ensure_rgb(st_frame), t, st_cx, st_cy, st_fx_post)
    else:
        st_rgb = None

    # ── mid layer ────────────────────────────────────────────────────
    if mid is not None and mid_ease_fn is not None:
        te_mid = mid_ease_fn(t)
        m_zoom = mid_zoom_start + (mid_zoom_end - mid_zoom_start) * te_mid
        m_cx   = mid_cx_start   + (mid_cx_end   - mid_cx_start)   * te_mid
        m_cy   = mid_cy_start   + (mid_cy_end   - mid_cy_start)   * te_mid

        mid_vel_raw = (mid_ease_fn(t_hi) - mid_ease_fn(t_lo)) / max(t_hi - t_lo, 1e-9)
        mid_zoom_vel = float(np.clip(mid_vel_raw / 3.0, 0.0, 1.0))

        fx_m = dict(mid_fx or {})
        dw_m = float(fx_m.get("depth_warp", 0.0))
        if mid_depth_map is not None and dw_m > 1e-4:
            mid_frame = _zoom_crop_depth(mid, mid_depth_map, m_zoom, dw_m,
                                          m_cx, m_cy, out_h, out_w)
        else:
            mid_frame = _zoom_crop(mid, m_zoom, m_cx, m_cy, out_h, out_w)

        fx_m_post = {k: v for k, v in fx_m.items() if k != "depth_warp"}
        fx_m_post["zoom_velocity"] = mid_zoom_vel
        mid_rgb = _apply_layer_effects(_ensure_rgb(mid_frame), t, m_cx, m_cy, fx_m_post)

        # Composite: mid blended over starless
        base_rgb = _blend_mid_cpu(sl_rgb, mid_rgb, mid_blend_mode, mid_opacity)
    else:
        base_rgb = sl_rgb

    # Stars blended on top of the base using the selected blend mode (optional)
    if st_rgb is not None:
        st_fx_blend = {
            "threshold": (st_fx or {}).get("threshold", 0.2),
            "feather":   (st_fx or {}).get("feather",   0.1),
        }
        result = _blend_mid_cpu(base_rgb, st_rgb, st_blend_mode, st_opacity, **st_fx_blend)
    else:
        result = base_rgb
    return np.clip(result, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# GPU render_frame  (now with optional mid layer)
# ---------------------------------------------------------------------------

def render_frame_gpu(
    starless_t, stars_t,   # stars_t may be None
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
    # ── stars blend mode (default Screen for back-compat) ─────────────
    st_blend_mode: str = "Screen",
    st_opacity: float = 1.0,
    # ── mid layer ────────────────────────────────────────────────────
    mid_t=None,
    mid_zoom_start: float = 1.0, mid_zoom_end: float = 1.0,
    mid_cx_start: float = 0.5,   mid_cy_start: float = 0.5,
    mid_cx_end: float = 0.5,     mid_cy_end: float = 0.5,
    mid_ease_fn = None,
    mid_fx: dict | None = None,
    mid_blend_mode: str = "Screen",
    mid_opacity: float = 1.0,
    mid_depth_t=None,
) -> np.ndarray:
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

    eps = 0.005
    t_lo, t_hi = max(0.0, t - eps), min(1.0, t + eps)
    sl_vel_raw = (sl_ease_fn(t_hi) - sl_ease_fn(t_lo)) / max(t_hi - t_lo, 1e-9)

    animate_sl = bool(sl_fx.get("animate_effects", True))
    ramp_sl = float(np.clip(sl_vel_raw / 3.0, 0.0, 1.0)) if animate_sl else 1.0

    # Starless effects
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

    # Stars effects (optional)
    st_frame = None
    if stars_t is not None:
        if st_depth_t is not None and dw_st > 1e-4:
            st_frame = _zoom_crop_depth_gpu(stars_t, st_depth_t, st_zoom, dw_st,
                                             st_cx, st_cy, out_h, out_w)
        else:
            st_frame = _zoom_crop_gpu(stars_t, st_zoom, st_cx, st_cy, out_h, out_w)
        st_vel_raw = (st_ease_fn(t_hi) - st_ease_fn(t_lo)) / max(t_hi - t_lo, 1e-9)
        animate_st = bool(st_fx.get("animate_effects", True))
        ramp_st = float(np.clip(st_vel_raw / 3.0, 0.0, 1.0)) if animate_st else 1.0
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

    # ── mid layer ────────────────────────────────────────────────────
    if mid_t is not None and mid_ease_fn is not None:
        te_mid = mid_ease_fn(t)
        m_zoom = mid_zoom_start + (mid_zoom_end - mid_zoom_start) * te_mid
        m_cx   = mid_cx_start   + (mid_cx_end   - mid_cx_start)   * te_mid
        m_cy   = mid_cy_start   + (mid_cy_end   - mid_cy_start)   * te_mid

        mid_vel_raw = (mid_ease_fn(t_hi) - mid_ease_fn(t_lo)) / max(t_hi - t_lo, 1e-9)
        fx_m = dict(mid_fx or {})
        animate_m = bool(fx_m.get("animate_effects", True))
        ramp_m = float(np.clip(mid_vel_raw / 3.0, 0.0, 1.0)) if animate_m else 1.0

        dw_m = float(fx_m.get("depth_warp", 0.0))
        if mid_depth_t is not None and dw_m > 1e-4:
            mid_frame = _zoom_crop_depth_gpu(mid_t, mid_depth_t, m_zoom, dw_m,
                                              m_cx, m_cy, out_h, out_w)
        else:
            mid_frame = _zoom_crop_gpu(mid_t, m_zoom, m_cx, m_cy, out_h, out_w)

        rs = float(fx_m.get("radial_stretch", 0.0)) * ramp_m
        if abs(rs) > 1e-4:
            mid_frame = _radial_stretch_gpu(mid_frame, rs, m_cx, m_cy)
        zb = float(fx_m.get("zoom_blur", 0.0)) * ramp_m
        if zb > 1e-4:
            mid_frame = _zoom_blur_gpu(mid_frame, zb, m_cx, m_cy,
                                        int(fx_m.get("zoom_blur_samples", 12)))
        ca = float(fx_m.get("chroma", 0.0)) * ramp_m
        if abs(ca) > 1e-4:
            mid_frame = _chroma_gpu(mid_frame, ca, m_cx, m_cy)

        base_frame = _blend_mid_gpu(sl_frame, mid_frame, mid_blend_mode, mid_opacity)
    else:
        base_frame = sl_frame

    # Stars blended on top of base (optional)
    if st_frame is not None:
        composited = torch.clamp(
            _blend_mid_gpu(base_frame, st_frame, st_blend_mode, st_opacity,
                           threshold=float(st_fx.get("threshold", 0.2) if st_fx else 0.2),
                           feather=float(st_fx.get("feather", 0.1) if st_fx else 0.1)),
            0.0, 1.0
        )
    else:
        composited = torch.clamp(base_frame, 0.0, 1.0)
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
            stars_np    = _ensure_rgb(p["stars"]) if p.get("stars") is not None else None
            mid_np      = _ensure_rgb(p["mid"]) if p.get("mid") is not None else None

            fps      = int(p["fps"])
            duration = float(p["duration"])
            n_frames = max(1, int(round(fps * duration)))
            out_w    = int(p["out_w"])
            out_h    = int(p["out_h"])
            out_path = str(p["out_path"])

            sl_ease_fn  = EASE_FUNCTIONS.get(p.get("sl_ease",  "Ease In-Out"), _ease_in_out)
            st_ease_fn  = EASE_FUNCTIONS.get(p.get("st_ease",  "Ease In-Out"), _ease_in_out)
            mid_ease_fn = EASE_FUNCTIONS.get(p.get("mid_ease", "Ease In-Out"), _ease_in_out)
            sl_fx  = p.get("sl_fx",  {})
            st_fx  = p.get("st_fx",  {})
            mid_fx = p.get("mid_fx", {})

            mid_blend_mode = str(p.get("mid_blend_mode", "Screen"))
            mid_opacity    = float(p.get("mid_opacity", 1.0))
            st_blend_mode  = str(p.get("st_blend_mode", "Screen"))
            st_opacity     = float(p.get("st_opacity", 1.0))

            if not HAS_CV2:
                self.finished.emit(False, "OpenCV (cv2) is required for video export.")
                return

            # Pre-compute depth maps
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

            mid_depth_map = None
            if mid_np is not None and mid_fx.get("depth_warp", 0.0) > 1e-4:
                mid_depth_map = _build_depth_map(
                    mid_np,
                    blur_sigma=float(mid_fx.get("depth_blur_sigma", 8.0)),
                    invert=bool(mid_fx.get("depth_invert", False)))

            # GPU setup
            device  = _get_torch_device()
            use_gpu = device is not None and str(device) != "cpu"
            starless_t = stars_t = mid_t_gpu = None
            sl_depth_t = st_depth_t = mid_depth_t = None

            if use_gpu:
                try:
                    import torch
                    starless_t = _np_to_tensor(starless_np, device)
                    stars_t    = _np_to_tensor(stars_np, device) if stars_np is not None else None
                    if mid_np is not None:
                        mid_t_gpu = _np_to_tensor(mid_np, device)
                    if sl_depth_map is not None:
                        sl_depth_t = _np_depth_to_tensor(sl_depth_map, device)
                    if st_depth_map is not None:
                        st_depth_t = _np_depth_to_tensor(st_depth_map, device)
                    if mid_depth_map is not None:
                        mid_depth_t = _np_depth_to_tensor(mid_depth_map, device)
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
                            st_blend_mode=st_blend_mode,
                            st_opacity=st_opacity,
                            mid_t=mid_t_gpu,
                            mid_zoom_start=p.get("mid_zoom_start", 1.0),
                            mid_zoom_end=p.get("mid_zoom_end", 1.0),
                            mid_cx_start=p.get("mid_cx_start", 0.5),
                            mid_cy_start=p.get("mid_cy_start", 0.5),
                            mid_cx_end=p.get("mid_cx_end", 0.5),
                            mid_cy_end=p.get("mid_cy_end", 0.5),
                            mid_ease_fn=mid_ease_fn,
                            mid_fx=mid_fx,
                            mid_blend_mode=mid_blend_mode,
                            mid_opacity=mid_opacity,
                            mid_depth_t=mid_depth_t,
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
                        st_blend_mode=st_blend_mode,
                        st_opacity=st_opacity,
                        mid=mid_np,
                        mid_zoom_start=p.get("mid_zoom_start", 1.0),
                        mid_zoom_end=p.get("mid_zoom_end", 1.0),
                        mid_cx_start=p.get("mid_cx_start", 0.5),
                        mid_cy_start=p.get("mid_cy_start", 0.5),
                        mid_cx_end=p.get("mid_cx_end", 0.5),
                        mid_cy_end=p.get("mid_cy_end", 0.5),
                        mid_ease_fn=mid_ease_fn,
                        mid_fx=mid_fx,
                        mid_blend_mode=mid_blend_mode,
                        mid_opacity=mid_opacity,
                        mid_depth_map=mid_depth_map,
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
# Center-picker label
# ---------------------------------------------------------------------------

class _centerPickerLabel(QLabel):
    pointPicked = pyqtSignal(float, float)
    pickCancelled = pyqtSignal()

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
        if ev.button() == Qt.MouseButton.RightButton:
            self.pickCancelled.emit()
            return
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
# Per-layer panel  (unchanged except minor naming)
# ---------------------------------------------------------------------------

class _LayerPanel(QGroupBox):
    def __init__(self, title: str, accent_colour: str, parent=None, has_source: bool = False, has_blend: bool = True):
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
        self._on_pick_activated = None

        form = QFormLayout(self)

        # Source image combo (shown when has_source=True)
        self.cmb_source = QComboBox()
        self.cmb_source.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.cmb_source.setMinimumContentsLength(22)
        self._src_label = QLabel("Source image:")
        if has_source:
            form.addRow(self._src_label, self.cmb_source)
        else:
            self._src_label.setVisible(False)
            self.cmb_source.setVisible(False)

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
        form.addRow("center start:", cx_s)

        cx_e = QHBoxLayout()
        self.sp_cx_end = _dsb(); self.sp_cy_end = _dsb()
        cx_e.addWidget(QLabel("X:")); cx_e.addWidget(self.sp_cx_end)
        cx_e.addWidget(QLabel("Y:")); cx_e.addWidget(self.sp_cy_end)
        form.addRow("center end:", cx_e)

        self.cmb_ease = QComboBox()
        self.cmb_ease.addItems(list(EASE_FUNCTIONS.keys()))
        self.cmb_ease.setCurrentText("Ease In-Out")
        form.addRow("Easing:", self.cmb_ease)

        pick_row = QHBoxLayout()
        self.btn_pick_start = QPushButton("Pick Start center")
        self.btn_pick_end   = QPushButton("Pick End center")
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
        fx_v.addWidget(self.chk_animate_fx)

        self.chk_depth_warp = QCheckBox("Nebula depth warp  (bright = closer)")
        self.chk_depth_warp.setToolTip(
            "Treats luminance as a height field and moves the camera toward it.")
        fx_v.addWidget(self.chk_depth_warp)
        dw_row, self.sld_depth_warp, self.lbl_depth_warp = _slider_row(
            "  Strength:", 0.0, 2.0, 0.3, decimals=2, scale=100)
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

        self.chk_barrel = QCheckBox("Radial edge stretch")
        fx_v.addWidget(self.chk_barrel)
        barrel_row, self.sld_barrel, self.lbl_barrel = _slider_row(
            "  Strength:", -0.5, 0.5, 0.25, decimals=3, scale=1000)
        barrel_row.setEnabled(False)
        self.chk_barrel.toggled.connect(barrel_row.setEnabled)
        fx_v.addWidget(barrel_row)
        self._barrel_row = barrel_row

        self.chk_zoom_blur = QCheckBox("Zoom blur  (warp-speed streaks)")
        fx_v.addWidget(self.chk_zoom_blur)
        zb_row, self.sld_zoom_blur, self.lbl_zoom_blur = _slider_row(
            "  Strength:", 0.0, 1.0, 0.4, decimals=2, scale=100)
        zb_row.setEnabled(False)
        self.chk_zoom_blur.toggled.connect(zb_row.setEnabled)
        fx_v.addWidget(zb_row)
        self._zb_row = zb_row

        self.chk_chroma = QCheckBox("Chromatic aberration")
        fx_v.addWidget(self.chk_chroma)
        ca_row, self.sld_chroma, self.lbl_chroma = _slider_row(
            "  Strength:", 0.0, 1.0, 0.5, decimals=2, scale=100)
        ca_row.setEnabled(False)
        self.chk_chroma.toggled.connect(ca_row.setEnabled)
        fx_v.addWidget(ca_row)
        self._ca_row = ca_row

        form.addRow(fx_box)

        # ── Blend mode / opacity (applies this layer onto the composite below it) ──
        # Starless is the base layer — nothing is below it, so no blend controls needed.
        self._has_blend = bool(has_blend)

        # Always create the widgets (so blend_mode()/opacity()/get_fx() are always safe),
        # but only add them to the form and make them visible when has_blend=True.
        self.cmb_blend = QComboBox()
        self.cmb_blend.addItems(LAYER_BLEND_MODES)
        self.cmb_blend.setCurrentText("Screen")

        self.sld_opacity = QSlider(Qt.Orientation.Horizontal)
        self.sld_opacity.setRange(0, 100); self.sld_opacity.setValue(100)
        self.lbl_opacity = QLabel("1.00"); self.lbl_opacity.setFixedWidth(36)
        self.sld_opacity.valueChanged.connect(
            lambda v: self.lbl_opacity.setText(f"{v/100:.2f}"))

        self.lbl_threshold = QLabel("Threshold:")
        self.sld_threshold = QSlider(Qt.Orientation.Horizontal)
        self.sld_threshold.setRange(0, 100); self.sld_threshold.setValue(20)
        self.lbl_thr_val = QLabel("0.20"); self.lbl_thr_val.setFixedWidth(36)
        self.sld_threshold.valueChanged.connect(
            lambda v: self.lbl_thr_val.setText(f"{v/100:.2f}"))

        self.lbl_feather = QLabel("Feather:")
        self.sld_feather = QSlider(Qt.Orientation.Horizontal)
        self.sld_feather.setRange(1, 50); self.sld_feather.setValue(10)
        self.lbl_fth_val = QLabel("0.10"); self.lbl_fth_val.setFixedWidth(36)
        self.sld_feather.valueChanged.connect(
            lambda v: self.lbl_fth_val.setText(f"{v/100:.2f}"))

        if has_blend:
            blend_sep = QLabel("─── Layer Blend ───")
            blend_sep.setStyleSheet("color:#888; font-size:10px;")
            form.addRow(blend_sep)
            form.addRow("Blend mode:", self.cmb_blend)
            op_row = QHBoxLayout()
            op_row.addWidget(self.sld_opacity, 1); op_row.addWidget(self.lbl_opacity)
            form.addRow("Opacity:", op_row)
            thr_row = QHBoxLayout()
            thr_row.addWidget(self.sld_threshold, 1); thr_row.addWidget(self.lbl_thr_val)
            fth_row = QHBoxLayout()
            fth_row.addWidget(self.sld_feather, 1); fth_row.addWidget(self.lbl_fth_val)
            form.addRow(self.lbl_threshold, thr_row)
            form.addRow(self.lbl_feather,   fth_row)

            def _on_blend_changed(txt):
                is_mask = (txt == "Threshold Mask")
                self.lbl_threshold.setVisible(is_mask)
                self.sld_threshold.setVisible(is_mask)
                self.lbl_thr_val.setVisible(is_mask)
                self.lbl_feather.setVisible(is_mask)
                self.sld_feather.setVisible(is_mask)
                self.lbl_fth_val.setVisible(is_mask)
            self.cmb_blend.currentTextChanged.connect(_on_blend_changed)
            _on_blend_changed(self.cmb_blend.currentText())

    def _on_pick_start(self, checked):
        self._pick_mode = "start" if checked else None
        self.btn_pick_end.setChecked(False)
        if checked and self._on_pick_activated:
            self._on_pick_activated(self)

    def _on_pick_end(self, checked):
        self._pick_mode = "end" if checked else None
        self.btn_pick_start.setChecked(False)
        if checked and self._on_pick_activated:
            self._on_pick_activated(self)

    def receive_picked_point(self, fx: float, fy: float):
        if self._pick_mode == "start":
            self.sp_cx_start.setValue(fx)
            self.sp_cy_start.setValue(fy)
        elif self._pick_mode == "end":
            self.sp_cx_end.setValue(fx)
            self.sp_cy_end.setValue(fy)

    def deactivate_picking(self):
        self._pick_mode = None
        self.btn_pick_start.setChecked(False)
        self.btn_pick_end.setChecked(False)

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

    def blend_mode(self) -> str:
        return self.cmb_blend.currentText()

    def opacity(self) -> float:
        return self.sld_opacity.value() / 100.0

    def get_fx(self) -> dict:
        fx: dict = {"animate_effects": self.chk_animate_fx.isChecked()}
        if self.chk_depth_warp.isChecked():
            fx["depth_warp"]       = (self.sld_depth_warp.value() / 100.0) * 10.0
            fx["depth_blur_sigma"] = self.sld_depth_blur.value() / 10.0
            fx["depth_invert"]     = self.chk_depth_invert.isChecked()
        if self.chk_barrel.isChecked():
            fx["radial_stretch"]   = self.sld_barrel.value() / 1000.0
        if self.chk_zoom_blur.isChecked():
            fx["zoom_blur"]        = self.sld_zoom_blur.value() / 100.0
            fx["zoom_blur_samples"] = 12
        if self.chk_chroma.isChecked():
            fx["chroma"]           = self.sld_chroma.value() / 100.0
        # Threshold mask params — always included so render_frame can read them
        fx["threshold"] = self.sld_threshold.value() / 100.0
        fx["feather"]   = self.sld_feather.value()   / 100.0
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
        s.setValue(f"flythrough/{key}_blend_mode", self.cmb_blend.currentText())
        s.setValue(f"flythrough/{key}_opacity",    self.sld_opacity.value())
        s.setValue(f"flythrough/{key}_threshold",  self.sld_threshold.value())
        s.setValue(f"flythrough/{key}_feather",    self.sld_feather.value())

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
        self.cmb_blend.setCurrentText( str( s.value(f"flythrough/{key}_blend_mode", "Screen")))
        self.sld_opacity.setValue(     int( s.value(f"flythrough/{key}_opacity",    100)))
        self.sld_threshold.setValue(   int( s.value(f"flythrough/{key}_threshold",  20)))
        self.sld_feather.setValue(     int( s.value(f"flythrough/{key}_feather",    10)))
        self._dw_row.setEnabled(self.chk_depth_warp.isChecked())
        self._dblur_row.setEnabled(self.chk_depth_warp.isChecked())
        self.chk_depth_invert.setEnabled(self.chk_depth_warp.isChecked())
        self._barrel_row.setEnabled(self.chk_barrel.isChecked())
        self._zb_row.setEnabled(self.chk_zoom_blur.isChecked())
        self._ca_row.setEnabled(self.chk_chroma.isChecked())
        # Trigger visibility update for threshold controls
        self.cmb_blend.currentTextChanged.emit(self.cmb_blend.currentText())


# ---------------------------------------------------------------------------
# Mid-layer panel  (wraps _LayerPanel + adds source combo, blend mode, opacity)
# ---------------------------------------------------------------------------

class _MidLayerPanel(QGroupBox):
    """
    The middle layer panel.  Contains:
      • Enable/disable checkbox
      • Source image dropdown
      • Blend mode dropdown
      • Opacity slider
      • All the same zoom/center/ease/effects controls as the other layers
    """
    def __init__(self, accent_colour: str = "#88cc88", parent=None):
        super().__init__("Mid Layer", parent)
        self._accent = accent_colour
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

        root = QVBoxLayout(self)
        root.setSpacing(4)
        root.setContentsMargins(6, 6, 6, 6)

        # _LayerPanel with source combo embedded (has_source=True)
        self._layer = _LayerPanel("", accent_colour, has_source=True)
        self._layer.setStyleSheet("")
        self._layer.setTitle("")
        self._layer.setFlat(True)
        root.addWidget(self._layer)

        # Expose source combo at wrapper level for external wiring
        self.cmb_source = self._layer.cmb_source

    # ── delegate picking API to inner layer ───────────────────────────
    @property
    def _on_pick_activated(self):
        return self._layer._on_pick_activated

    @_on_pick_activated.setter
    def _on_pick_activated(self, fn):
        if fn is None:
            self._layer._on_pick_activated = None
            return
        # The inner _LayerPanel will call fn(self._layer) but the dialog's
        # deactivator compares by identity against self.panel_mid (the outer
        # _MidLayerPanel).  Wrap fn so we pass the outer wrapper instead.
        outer_self = self
        def _wrapped(inner_panel):
            fn(outer_self)
        self._layer._on_pick_activated = _wrapped

    @property
    def picking(self) -> bool:
        return self._layer.picking

    def deactivate_picking(self):
        self._layer.deactivate_picking()

    def receive_picked_point(self, fx: float, fy: float):
        self._layer.receive_picked_point(fx, fy)

    def markers(self) -> list:
        return self._layer.markers()

    # ── delegate button refs for signal wiring ────────────────────────
    @property
    def btn_pick_start(self):
        return self._layer.btn_pick_start

    @property
    def btn_pick_end(self):
        return self._layer.btn_pick_end

    # ── spinboxes / sliders for signal wiring ─────────────────────────
    @property
    def sp_cx_start(self): return self._layer.sp_cx_start
    @property
    def sp_cy_start(self): return self._layer.sp_cy_start
    @property
    def sp_cx_end(self):   return self._layer.sp_cx_end
    @property
    def sp_cy_end(self):   return self._layer.sp_cy_end

    # ── data accessors ────────────────────────────────────────────────
    def blend_mode(self) -> str:
        return self._layer.blend_mode()

    def opacity(self) -> float:
        return self._layer.opacity()

    def get_params(self) -> dict:
        return self._layer.get_params()

    def get_fx(self) -> dict:
        return self._layer.get_fx()

    # ── settings ──────────────────────────────────────────────────────
    def save_settings(self, s: QSettings, key: str = "mid"):
        self._layer.save_settings(s, key)  # _layer saves blend_mode, opacity, threshold, feather

    def load_settings(self, s: QSettings, key: str = "mid"):
        self._layer.load_settings(s, key)


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
        self.setMinimumWidth(1100)   # wider to accommodate 3 columns

        self._list_open_docs  = list_open_docs_fn or (lambda: [])
        self._docman          = doc_manager
        self._worker: _FlythroughWorker | None = None

        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(150)
        self._preview_timer.timeout.connect(self._refresh_preview)
        self._play_timer = QTimer(self)
        self._play_timer.setInterval(200)
        self._play_timer.timeout.connect(self._on_play_tick)
        self._playing = False

        self._starless_arr:  np.ndarray | None = None
        self._stars_arr:     np.ndarray | None = None
        self._mid_arr:       np.ndarray | None = None
        self._starless_full: np.ndarray | None = None
        self._stars_full:    np.ndarray | None = None
        self._mid_full:      np.ndarray | None = None

        self._build_ui()
        self._populate_combos()
        self._load_settings()

    def _build_ui(self):
        root = QVBoxLayout(self)

        # Working resolution label (updated when source changes; used in output box)
        self.lbl_working_size = QLabel("")
        self.lbl_working_size.setStyleSheet("color:#888; font-size:11px;")

        # ── body ───────────────────────────────────────────────────────
        body = QHBoxLayout()
        root.addLayout(body, 1)

        left = QVBoxLayout()

        # Three layer panels side by side — each has its own source combo
        layers_row = QHBoxLayout()
        self.panel_sl  = _LayerPanel("Starless Layer", "#88aaff", has_source=True, has_blend=False)
        self.panel_mid = _MidLayerPanel("#88cc88")   # mid already embeds has_source=True
        self.panel_st  = _LayerPanel("Stars Layer",    "#ffcc66", has_source=True, has_blend=True)
        # Expose top-level source combos for _populate_combos / _on_source_changed
        self.cmb_starless = self.panel_sl.cmb_source
        self.cmb_stars    = self.panel_st.cmb_source
        layers_row.addWidget(self.panel_sl,  1)
        layers_row.addWidget(self.panel_mid, 1)
        layers_row.addWidget(self.panel_st,  1)
        left.addLayout(layers_row)

        # Mutual pick deactivation across all three panels
        all_panels = [self.panel_sl, self.panel_mid, self.panel_st]

        def _on_any_pick_activated(active_panel):
            """Deactivate picking on every panel except the one that just activated."""
            for p in all_panels:
                if p is not active_panel:
                    p.deactivate_picking()

        for panel in all_panels:
            panel._on_pick_activated = _on_any_pick_activated

        # Picker
        picker_box = QGroupBox("Click image to set zoom center points")
        picker_v   = QVBoxLayout(picker_box)
        self.picker = _centerPickerLabel()
        self.picker.setMinimumHeight(180)
        self.picker.pointPicked.connect(self._on_point_picked)
        picker_v.addWidget(self.picker)
        left.addWidget(picker_box, 1)
        body.addLayout(left, 3)

        # ── right column ───────────────────────────────────────────────
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

        # Working resolution (moved here from the old source box)
        self.cmb_working_res = QComboBox()
        self.cmb_working_res.addItems(list(WORKING_RES_PRESETS.keys()))
        self.cmb_working_res.setCurrentText("HD  (1920x1080)")
        self.cmb_working_res.currentIndexChanged.connect(self._on_source_changed)
        out_form.addRow("Working res:", self.cmb_working_res)
        out_form.addRow("", self.lbl_working_size)

        res_out = QHBoxLayout()
        self.sp_out_w = QSpinBox(); self.sp_out_w.setRange(64, 7680)
        self.sp_out_w.setValue(1920); self.sp_out_w.setSuffix(" px")
        self.sp_out_h = QSpinBox(); self.sp_out_h.setRange(64, 4320)
        self.sp_out_h.setValue(1080); self.sp_out_h.setSuffix(" px")
        res_out.addWidget(self.sp_out_w); res_out.addWidget(QLabel("x")); res_out.addWidget(self.sp_out_h)
        out_form.addRow("Export res:", res_out)

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

        # ── signal wiring ──────────────────────────────────────────────
        self.picker.pickCancelled.connect(self._cancel_all_picking)

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
                        panel.sld_depth_warp, panel.sld_depth_blur,
                        panel.sld_opacity, panel.sld_threshold, panel.sld_feather):
                sld.valueChanged.connect(self._schedule_preview)
            for chk in (panel.chk_barrel, panel.chk_zoom_blur, panel.chk_chroma,
                        panel.chk_animate_fx, panel.chk_depth_warp,
                        panel.chk_depth_invert):
                chk.toggled.connect(self._schedule_preview)
            panel.cmb_blend.currentTextChanged.connect(self._schedule_preview)

        # Mid panel wiring — all controls now live on panel_mid._layer
        ml = self.panel_mid._layer   # shorthand
        for sp in (ml.sp_cx_start, ml.sp_cy_start, ml.sp_cx_end, ml.sp_cy_end,
                   ml.sp_zoom_start, ml.sp_zoom_end):
            sp.valueChanged.connect(self._schedule_preview)
            sp.valueChanged.connect(self._update_markers)
        ml.cmb_ease.currentIndexChanged.connect(self._schedule_preview)
        for btn in (ml.btn_pick_start, ml.btn_pick_end):
            btn.clicked.connect(self._update_markers)
        for sld in (ml.sld_barrel, ml.sld_zoom_blur, ml.sld_chroma,
                    ml.sld_depth_warp, ml.sld_depth_blur,
                    ml.sld_opacity, ml.sld_threshold, ml.sld_feather):
            sld.valueChanged.connect(self._schedule_preview)
        for chk in (ml.chk_barrel, ml.chk_zoom_blur, ml.chk_chroma,
                    ml.chk_animate_fx, ml.chk_depth_warp, ml.chk_depth_invert):
            chk.toggled.connect(self._schedule_preview)
        ml.cmb_blend.currentTextChanged.connect(self._schedule_preview)
        self.panel_mid.cmb_source.currentIndexChanged.connect(self._on_mid_source_changed)

    # ------------------------------------------------------------------

    def _on_mid_source_changed(self):
        doc = self.panel_mid.cmb_source.currentData()
        if doc is None or getattr(doc, "image", None) is None:
            self._mid_arr  = None
            self._mid_full = None
        else:
            raw = _ensure_rgb(doc.image)
            self._mid_full = raw
            preset = WORKING_RES_PRESETS.get(self.cmb_working_res.currentText())
            if preset and self._starless_arr is not None:
                th, tw = self._starless_arr.shape[:2]
                self._mid_arr = _downsize_to_fit(raw, tw, th)
            elif preset:
                self._mid_arr = _downsize_to_fit(raw, preset[0], preset[1])
            else:
                self._mid_arr = raw
        self._schedule_preview()

    def _cancel_all_picking(self):
        self.panel_sl.deactivate_picking()
        self.panel_mid.deactivate_picking()
        self.panel_st.deactivate_picking()

    def _toggle_play(self, checked: bool):
        self._playing = checked
        if checked:
            self.btn_play.setText("⏸  Pause")
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
            self.sld_scrub.blockSignals(True)
            self.sld_scrub.setValue(val)
            self.sld_scrub.blockSignals(False)
            self.lbl_scrub_t.setText(f"t={val/100.0:.2f}")
            self._refresh_preview()

    def _populate_combos(self):
        docs = self._list_open_docs()
        for cmb in (self.cmb_starless, self.cmb_stars):
            cmb.blockSignals(True); cmb.clear()
            cmb.addItem("-- select --", None)
            for title, doc in docs:
                cmb.addItem(title, doc)
            cmb.blockSignals(False)
            cmb.currentIndexChanged.connect(self._on_source_changed)

        # Mid source combo gets the same list plus a "-- none --" entry
        self.panel_mid.cmb_source.blockSignals(True)
        self.panel_mid.cmb_source.clear()
        self.panel_mid.cmb_source.addItem("-- none --", None)
        for title, doc in docs:
            self.panel_mid.cmb_source.addItem(title, doc)
        self.panel_mid.cmb_source.blockSignals(False)

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

        self._stars_arr = None
        if st_raw is not None:
            if preset and self._starless_arr is not None:
                th, tw = self._starless_arr.shape[:2]
                self._stars_arr = _downsize_to_fit(st_raw, tw, th)
            elif preset:
                self._stars_arr = _downsize_to_fit(st_raw, preset[0], preset[1])
            else:
                self._stars_arr = st_raw

        # Refresh mid if it had data
        self._on_mid_source_changed()
        self._update_markers(); self._schedule_preview()

    def _update_frame_count(self):
        n = max(1, int(round(self.sp_fps.value() * self.sp_duration.value())))
        self.lbl_frame_count.setText(f"{n} frames")

    def _update_markers(self):
        markers = self.panel_sl.markers() + self.panel_mid.markers() + self.panel_st.markers()
        self.picker.set_markers(markers)

    def _on_point_picked(self, fx: float, fy: float):
        for panel in (self.panel_sl, self.panel_mid, self.panel_st):
            if panel.picking:
                panel.receive_picked_point(fx, fy)
                self._update_markers()
                self._schedule_preview()
                return

    def _schedule_preview(self):
        self._preview_timer.start()

    def _refresh_preview(self):
        if self._starless_arr is None:
            self.preview_lbl.setText("Select a starless image to preview.")
            return

        t_raw = self.sld_scrub.value() / 100.0
        self.lbl_scrub_t.setText(f"t={t_raw:.2f}")

        sl_p  = self.panel_sl.get_params()
        st_p  = self.panel_st.get_params()
        mid_p = self.panel_mid.get_params()
        sl_fx  = self.panel_sl.get_fx()
        st_fx  = self.panel_st.get_fx()
        mid_fx = self.panel_mid.get_fx()

        # Build depth maps at working res
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
        mid_dm = None
        if self._mid_arr is not None and mid_fx.get("depth_warp", 0.0) > 1e-4:
            mid_dm = _build_depth_map(self._mid_arr,
                                       blur_sigma=float(mid_fx.get("depth_blur_sigma", 8.0)),
                                       invert=bool(mid_fx.get("depth_invert", False)))

        src_h, src_w = self._starless_arr.shape[:2]
        if src_w > 0 and src_h > 0:
            aspect = src_w / src_h
            if aspect >= 1.0:
                out_w = min(src_w, 640); out_h = max(1, int(round(out_w / aspect)))
            else:
                out_h = min(src_h, 360); out_w = max(1, int(round(out_h * aspect)))
        else:
            out_w, out_h = 640, 360

        try:
            frame = render_frame(
                self._starless_arr,
                self._stars_arr,   # may be None — stars layer is optional
                t_raw,
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
                st_blend_mode=self.panel_st.blend_mode(),
                st_opacity=self.panel_st.opacity(),
                mid=self._mid_arr,   # None when no mid image selected

                mid_zoom_start=mid_p["zoom_start"], mid_zoom_end=mid_p["zoom_end"],
                mid_cx_start=mid_p["cx_start"],     mid_cy_start=mid_p["cy_start"],
                mid_cx_end=mid_p["cx_end"],         mid_cy_end=mid_p["cy_end"],
                mid_ease_fn=EASE_FUNCTIONS[mid_p["ease"]],
                mid_fx=mid_fx,
                mid_blend_mode=self.panel_mid.blend_mode(),
                mid_opacity=self.panel_mid.opacity(),
                mid_depth_map=mid_dm,
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
        self.panel_mid.load_settings(s, "mid")
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
        self.panel_mid.save_settings(s, "mid")

    def _export(self):
        if self._starless_full is None:
            QMessageBox.warning(self, "Flythrough", "Please select a starless image first.")
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
        mid_p = self.panel_mid.get_params()
        sl_fx  = self.panel_sl.get_fx()
        st_fx  = self.panel_st.get_fx()
        mid_fx = self.panel_mid.get_fx()

        params = {
            "starless":      self._starless_full,
            "stars":         self._stars_full,   # may be None
            "mid":           self._mid_full,   # None when no mid image selected
            "mid_blend_mode": self.panel_mid.blend_mode(),
            "mid_opacity":   self.panel_mid.opacity(),
            "st_blend_mode": self.panel_st.blend_mode(),
            "st_opacity":    self.panel_st.opacity(),
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
            "mid_zoom_start": mid_p["zoom_start"], "mid_zoom_end": mid_p["zoom_end"],
            "mid_cx_start":  mid_p["cx_start"],   "mid_cy_start": mid_p["cy_start"],
            "mid_cx_end":    mid_p["cx_end"],      "mid_cy_end":   mid_p["cy_end"],
            "mid_ease":      mid_p["ease"],         "mid_fx":      mid_fx,
        }

        n_frames = max(1, int(round(params["fps"] * params["duration"])))
        self.pbar.setRange(0, n_frames); self.pbar.setValue(0); self.pbar.setVisible(True)
        self.btn_export.setEnabled(False); self.btn_cancel_export.setVisible(True)

        full_h, full_w = self._starless_full.shape[:2]
        mid_note = f" + mid ({self.panel_mid.blend_mode()})" if self._mid_full is not None else ""
        self.lbl_status.setText(
            f"Rendering {params['out_w']}x{params['out_h']}{mid_note} -- "
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