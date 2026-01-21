# src/setiastro/saspro/planetprojection.py
from __future__ import annotations

import numpy as np
import os
import tempfile, webbrowser
import plotly.graph_objects as go
from PyQt6.QtCore import Qt, QTimer, QPoint, QSize
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QGroupBox,
    QFormLayout, QDoubleSpinBox, QSpinBox, QCheckBox, QComboBox, QMessageBox,
    QSizePolicy, QFileDialog, QLineEdit, QSlider, QWidget
)
from PyQt6 import sip
from setiastro.saspro.widgets.themed_buttons import themed_toolbtn

import cv2


# -----------------------------
# Core math helpers
# -----------------------------

def _planet_centroid_and_area(ch: np.ndarray):
    """
    Estimate planet centroid (cx,cy) and blob area from a single channel.
    Uses percentile scaling + Otsu + largest component.
    Returns (cx, cy, area) or None.
    """
    if cv2 is None:
        return None

    img = ch.astype(np.float32, copy=False)

    p1 = float(np.percentile(img, 1.0))
    p99 = float(np.percentile(img, 99.5))
    if p99 <= p1:
        return None

    scaled = (img - p1) * (255.0 / (p99 - p1))
    scaled = np.clip(scaled, 0, 255).astype(np.uint8)
    scaled = cv2.GaussianBlur(scaled, (0, 0), 1.2)

    _, bw = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k, iterations=1)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k, iterations=2)

    num, labels, stats, cents = cv2.connectedComponentsWithStats(bw, connectivity=8)
    if num <= 1:
        return None

    areas = stats[1:, cv2.CC_STAT_AREA]
    j = int(np.argmax(areas)) + 1
    area = float(stats[j, cv2.CC_STAT_AREA])
    if area < 200:
        return None

    cx, cy = cents[j]
    return (float(cx), float(cy), float(area))


def _compute_roi_from_centroid(H: int, W: int, cx: float, cy: float, area: float,
                               pad_mul: float = 3.2,
                               min_size: int = 240,
                               max_size: int = 900):
    """
    Use area->radius estimate to make an ROI around the disk.
    """
    r = max(32.0, float(np.sqrt(area / np.pi)))
    s = int(np.clip(r * float(pad_mul), float(min_size), float(max_size)))
    cx_i, cy_i = int(round(cx)), int(round(cy))

    x0 = max(0, cx_i - s // 2)
    y0 = max(0, cy_i - s // 2)
    x1 = min(W, x0 + s)
    y1 = min(H, y0 + s)
    return (x0, y0, x1, y1)

def _ellipse_annulus_mask(H: int, W: int, cx: float, cy: float,
                          a_outer: float, b_outer: float,
                          a_inner: float, b_inner: float,
                          pa_deg: float) -> np.ndarray:
    """
    Elliptical annulus mask (outer ellipse minus inner ellipse).
    PA rotates ellipse in image plane.
    """
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    x = xx - float(cx)
    y = yy - float(cy)

    th = np.deg2rad(float(pa_deg))
    c, s = np.cos(th), np.sin(th)

    # rotate coords into ellipse frame
    xr = x * c + y * s
    yr = -x * s + y * c

    outer = (xr*xr)/(a_outer*a_outer + 1e-9) + (yr*yr)/(b_outer*b_outer + 1e-9) <= 1.0
    inner = (xr*xr)/(a_inner*a_inner + 1e-9) + (yr*yr)/(b_inner*b_inner + 1e-9) <= 1.0

    return outer & (~inner)


def _ring_front_back_masks(H: int, W: int, cx: float, cy: float, pa_deg: float,
                           ring_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Split ring pixels into 'front' vs 'back' halves for occlusion.
    Approximation: use ring minor-axis direction.
    """
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    x = xx - float(cx)
    y = yy - float(cy)

    # minor axis is PA + 90 degrees
    th = np.deg2rad(float(pa_deg) + 90.0)
    nx, ny = np.cos(th), np.sin(th)

    # signed distance along minor-axis normal
    s = x * nx + y * ny
    front = ring_mask & (s > 0)
    back  = ring_mask & (s <= 0)
    return front, back


def _yaw_warp_maps(H: int, W: int, theta_deg: float, cx: float, cy: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Simple planar 'yaw' warp around vertical axis:
      x' = cx + (x-cx)*cos(a)
      y' = y
    Used for rings to create stereo disparity without bending them onto the sphere.
    """
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    x = xx - float(cx)

    a = np.deg2rad(float(theta_deg))
    ca = np.cos(a)

    def make(sign: float):
        # left uses +theta, right uses -theta (sign flips)
        # we only need cos for this simple model, so sign doesn't matter here,
        # but keep signature consistent in case we extend to perspective later.
        mapx = (float(cx) + x * ca).astype(np.float32)
        mapy = yy.astype(np.float32)
        return mapx, mapy

    mapLx, mapLy = make(+1.0)
    mapRx, mapRy = make(-1.0)
    return mapLx, mapLy, mapRx, mapRy


def _to_u8_preview(rgb: np.ndarray) -> np.ndarray:
    """
    Robust per-channel percentile stretch to uint8 for display.
    """
    x = rgb.astype(np.float32, copy=False)
    out = np.empty_like(x, dtype=np.uint8)
    for c in range(3):
        ch = x[..., c]
        p1 = float(np.percentile(ch, 1.0))
        p99 = float(np.percentile(ch, 99.5))
        if p99 <= p1:
            out[..., c] = 0
        else:
            y = (ch - p1) * (255.0 / (p99 - p1))
            out[..., c] = np.clip(y, 0, 255).astype(np.uint8)
    return out


def _add_starfield(bg01_rgb: np.ndarray, density: float = 0.02, seed: int = 1,
                   star_sigma: float = 0.8, brightness: float = 0.9):
    """
    Add a visible synthetic starfield to a float32 RGB image in [0,1].
    Applied identically to left/right so it has ZERO parallax (screen-locked).
    """
    H, W = bg01_rgb.shape[:2]
    rng = np.random.default_rng(int(seed))

    # allow higher density (0..0.2)
    n = int(np.clip(density, 0.0, 0.5) * H * W)
    if n <= 0:
        return bg01_rgb

    stars = np.zeros((H, W), dtype=np.float32)

    ys = rng.integers(0, H, size=n, endpoint=False)
    xs = rng.integers(0, W, size=n, endpoint=False)

    # brighter distribution (more visible than **6)
    vals = rng.random(n).astype(np.float32)
    vals = (vals ** 2.0)  # more mid-range stars
    vals = (0.15 + 0.85 * vals) * float(brightness)  # visible floor

    # a few "bright" stars
    bright_n = max(1, n // 80)
    bi = rng.integers(0, n, size=bright_n, endpoint=False)
    vals[bi] = np.clip(vals[bi] * 2.5, 0.0, 1.0)

    stars[ys, xs] = np.maximum(stars[ys, xs], vals)

    if star_sigma > 0:
        stars = cv2.GaussianBlur(stars, (0, 0), float(star_sigma))

    out = bg01_rgb.astype(np.float32, copy=False).copy()
    out[..., 0] = np.clip(out[..., 0] + stars, 0.0, 1.0)
    out[..., 1] = np.clip(out[..., 1] + stars, 0.0, 1.0)
    out[..., 2] = np.clip(out[..., 2] + stars, 0.0, 1.0)
    return out

def _make_anaglyph(L_rgb8: np.ndarray, R_rgb8: np.ndarray, *, swap_eyes: bool = False) -> np.ndarray:
    """
    Build a red/cyan anaglyph from two uint8 RGB images.
    Output is uint8 RGB.

    Red channel comes from LEFT eye luminance.
    Cyan (G+B) comes from RIGHT eye luminance.

    swap_eyes=True flips which eye feeds red vs cyan (useful if glasses seem inverted).
    """
    if swap_eyes:
        L_rgb8, R_rgb8 = R_rgb8, L_rgb8

    L = L_rgb8.astype(np.float32)
    R = R_rgb8.astype(np.float32)

    # luminance (more robust than using only R/G/B)
    Llum = 0.299 * L[..., 0] + 0.587 * L[..., 1] + 0.114 * L[..., 2]
    Rlum = 0.299 * R[..., 0] + 0.587 * R[..., 1] + 0.114 * R[..., 2]

    out = np.zeros_like(L, dtype=np.float32)
    out[..., 0] = Llum           # red
    out[..., 1] = Rlum           # green (cyan)
    out[..., 2] = Rlum           # blue  (cyan)

    return np.clip(out, 0, 255).astype(np.uint8)



def _sphere_reproject_maps(H: int, W: int, theta_deg: float, radius_px: float | None = None):
    """
    Build cv2.remap maps for left/right views using sphere reprojection.
    Returns (mapLx, mapLy, mapRx, mapRy, mask_disk)
    """
    cx = (W - 1) * 0.5
    cy = (H - 1) * 0.5

    if radius_px is None:
        radius_px = 0.49 * min(W, H)  # conservative

    r = float(radius_px)

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    x = (xx - cx) / r
    y = (yy - cy) / r
    rr2 = x * x + y * y
    mask = rr2 <= 1.0

    z = np.zeros_like(x, dtype=np.float32)
    z[mask] = np.sqrt(np.maximum(0.0, 1.0 - rr2[mask])).astype(np.float32)

    a = np.deg2rad(float(theta_deg))
    ca, sa = np.cos(a), np.sin(a)

    def make(alpha_sign: float):
        ca2 = ca
        sa2 = sa * alpha_sign
        # rotate around Y
        x2 = x * ca2 + z * sa2
        y2 = y
        mapx = (cx + r * x2).astype(np.float32)
        mapy = (cy + r * y2).astype(np.float32)
        # outside disk: invalid → map to -1 so BORDER_CONSTANT applies
        mapx[~mask] = -1.0
        mapy[~mask] = -1.0
        return mapx, mapy

    mapLx, mapLy = make(+1.0)
    mapRx, mapRy = make(-1.0)
    return mapLx, mapLy, mapRx, mapRy, mask

def make_pseudo_surface_pair(
    rgb: np.ndarray,
    theta_deg: float = 10.0,
    *,
    depth_gamma: float = 1.0,
    blur_sigma: float = 1.2,
    invert: bool = False,
):
    """
    Pseudo surface stereo (astro-friendly):
      - height = luminance deviation around median (NO 0..1 normalization)
      - robust amplitude scaling using MAD so stars don't dominate
      - per-pixel horizontal disparity warp

    Returns (left, right, maskL, maskR) where masks are valid sampling regions.
    """
    if cv2 is None:
        m = np.ones(rgb.shape[:2], dtype=bool)
        return rgb, rgb, m, m

    x = np.asarray(rgb)
    orig_dtype = x.dtype

    # --- float01 for remap (image sampling) ---
    if x.dtype == np.uint8:
        xf = x.astype(np.float32) / 255.0
    elif x.dtype == np.uint16:
        xf = x.astype(np.float32) / 65535.0
    else:
        xf = x.astype(np.float32, copy=False)
        # for float inputs we assume "image-like" but keep it sane for remap
        xf = np.clip(xf, 0.0, 1.0)

    H, W = xf.shape[:2]

    # --- luminance (float32) ---
    lum = (0.299 * xf[..., 0] + 0.587 * xf[..., 1] + 0.114 * xf[..., 2]).astype(np.float32)

    # Optional smoothing to reduce noisy depth
    if blur_sigma and blur_sigma > 0:
        lum_s = cv2.GaussianBlur(lum, (0, 0), float(blur_sigma))
    else:
        lum_s = lum

    # Center around median so "background" ~ 0 height
    h = lum_s - float(np.median(lum_s))

    # Optional gamma shaping (on magnitude, preserving sign)
    g = float(max(1e-6, depth_gamma))
    if abs(g - 1.0) > 1e-6:
        h = np.sign(h) * (np.abs(h) ** g)

    # Invert AFTER centering/shaping (so it just flips relief)
    if invert:
        h = -h

    # Robust amplitude scaling so a few bright stars don't explode the height
    # MAD ~ median(|x - median(x)|)
    mad = float(np.median(np.abs(h)) + 1e-9)

    # "gain" controls how punchy the height is.
    # Larger gain_div => flatter relief (safer for stars).
    gain_div = 6.0
    h = (h / (gain_div * mad)).astype(np.float32)

    # Keep height in a sane range so disparity doesn't go nuts
    h = np.clip(h, -1.0, 1.0)

    # --- disparity scale ---
    # Empirical mapping: theta 6deg on ~600px gives ~15-20px max disparity.
    max_disp = (float(theta_deg) / 25.0) * (0.12 * float(min(H, W)))
    max_disp = float(np.clip(max_disp, 0.0, 0.35 * min(H, W)))

    disp = h * max_disp  # signed pixels: positive => "closer"

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)

    # left/right: split disparity
    mapLx = xx + 0.5 * disp
    mapRx = xx - 0.5 * disp
    mapy = yy

    # IMPORTANT: INTER_LINEAR avoids Lanczos ringing on stars
    left = cv2.remap(
        xf, mapLx, mapy,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    right = cv2.remap(
        xf, mapRx, mapy,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    # validity masks (where sampling stays in-bounds)
    maskL = (mapLx >= 0.0) & (mapLx <= (W - 1))
    maskR = (mapRx >= 0.0) & (mapRx <= (W - 1))

    # Back to original dtype (preserve your existing behavior)
    if orig_dtype == np.uint8:
        left = np.clip(left * 255.0, 0, 255).astype(np.uint8)
        right = np.clip(right * 255.0, 0, 255).astype(np.uint8)
    elif orig_dtype == np.uint16:
        left = np.clip(left * 65535.0, 0, 65535).astype(np.uint16)
        right = np.clip(right * 65535.0, 0, 65535).astype(np.uint16)
    else:
        # float inputs come back as float (same dtype), but remap ran on float32
        left = left.astype(orig_dtype, copy=False)
        right = right.astype(orig_dtype, copy=False)

    return left, right, maskL, maskR


def make_stereo_pair(
    roi_rgb: np.ndarray,
    theta_deg: float = 10.0,
    disk_mask: np.ndarray | None = None,
    *,
    interp: int = None,
):
    if cv2 is None:
        dummy_mask = np.ones(roi_rgb.shape[:2], dtype=bool)
        return roi_rgb, roi_rgb, dummy_mask, dummy_mask
    if interp is None:
        interp = cv2.INTER_LANCZOS4
    x = roi_rgb
    orig_dtype = x.dtype

    # Build a REAL disk mask from ROI (use green channel)
    disk = disk_mask if disk_mask is not None else _planet_disk_mask(roi_rgb[...,1])
    if disk is None:
        # fallback: simple circle
        H, W = roi_rgb.shape[:2]
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        cx = (W - 1) * 0.5
        cy = (H - 1) * 0.5
        r = 0.49 * min(W, H)
        disk = ((xx - cx) ** 2 + (yy - cy) ** 2) <= (r * r)

    # work in float32 for remap
    if x.dtype == np.uint8:
        xf = x.astype(np.float32) / 255.0
    elif x.dtype == np.uint16:
        xf = x.astype(np.float32) / 65535.0
    else:
        xf = x.astype(np.float32, copy=False)

    H, W = xf.shape[:2]
    # estimate radius_px from disk area (in pixels)
    area_px = float(disk.sum())
    radius = np.sqrt(area_px / np.pi)
    radius = np.clip(radius, 16.0, 0.49 * min(W, H))

    mapLx, mapLy, mapRx, mapRy, _ = _sphere_reproject_maps(H, W, theta_deg, radius_px=radius)

    left = cv2.remap(xf, mapLx, mapLy, interpolation=interp,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    right = cv2.remap(xf, mapRx, mapRy, interpolation=interp,
                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # Warp the disk mask through the SAME maps
    disk_u8 = (disk.astype(np.uint8) * 255)
    mL = cv2.remap(disk_u8, mapLx, mapLy, interpolation=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mR = cv2.remap(disk_u8, mapRx, mapRy, interpolation=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    maskL = mL > 127
    maskR = mR > 127

    # convert back to original dtype
    if orig_dtype == np.uint8:
        left = np.clip(left * 255.0, 0, 255).astype(np.uint8)
        right = np.clip(right * 255.0, 0, 255).astype(np.uint8)
    elif orig_dtype == np.uint16:
        left = np.clip(left * 65535.0, 0, 65535).astype(np.uint16)
        right = np.clip(right * 65535.0, 0, 65535).astype(np.uint16)
    else:
        left = left.astype(orig_dtype, copy=False)
        right = right.astype(orig_dtype, copy=False)

    return left, right, maskL, maskR


def _planet_disk_mask(ch: np.ndarray, grow: float = 1.015) -> np.ndarray | None:
    """
    Return a boolean mask of the planet disk.
    We still segment the planet to find the *component*, but then we create a
    circle mask using equivalent radius from area so the limb is not clipped.
    'grow' slightly expands the radius to avoid eating into the edge.
    """
    if cv2 is None:
        return None

    img = ch.astype(np.float32, copy=False)

    p1 = float(np.percentile(img, 1.0))
    p99 = float(np.percentile(img, 99.5))
    if p99 <= p1:
        return None

    scaled = (img - p1) * (255.0 / (p99 - p1))
    scaled = np.clip(scaled, 0, 255).astype(np.uint8)
    scaled = cv2.GaussianBlur(scaled, (0, 0), 1.2)

    _, bw = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k, iterations=1)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k, iterations=2)

    num, labels, stats, cents = cv2.connectedComponentsWithStats(bw, connectivity=8)
    if num <= 1:
        return None

    areas = stats[1:, cv2.CC_STAT_AREA]
    j = int(np.argmax(areas)) + 1
    area = float(stats[j, cv2.CC_STAT_AREA])
    if area < 200:
        return None

    cx, cy = cents[j]
    r = np.sqrt(area / np.pi) * float(grow)

    H, W = ch.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    mask = ((xx - float(cx)) ** 2 + (yy - float(cy)) ** 2) <= (float(r) ** 2)
    return mask

def _mask_bbox(mask: np.ndarray):
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _mask_centroid(mask: np.ndarray) -> tuple[float, float] | None:
    m = mask.astype(np.uint8)
    M = cv2.moments(m, binaryImage=True) if cv2 is not None else None
    if not M or M["m00"] <= 1e-6:
        return None
    cx = float(M["m10"] / M["m00"])
    cy = float(M["m01"] / M["m00"])
    return cx, cy


def _shift_image(img: np.ndarray, dx: float, dy: float, *, border_value=0):
    """
    Shift an image by (dx,dy) pixels using warpAffine.
    Works for 2D or 3D arrays.
    """
    H, W = img.shape[:2]
    M = np.array([[1.0, 0.0, dx],
                  [0.0, 1.0, dy]], dtype=np.float32)
    return cv2.warpAffine(
        img, M, (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value
    )


def _shift_mask(mask: np.ndarray, dx: float, dy: float):
    H, W = mask.shape[:2]
    M = np.array([[1.0, 0.0, dx],
                  [0.0, 1.0, dy]], dtype=np.float32)
    m = mask.astype(np.uint8) * 255
    mw = cv2.warpAffine(
        m, M, (W, H),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    return mw > 127

def _disk_to_equirect_texture(roi_rgb01: np.ndarray, disk_mask: np.ndarray,
                              tex_h: int = 256, tex_w: int = 512) -> np.ndarray:
    """
    Convert a planet disk image (ROI) into an equirectangular texture (lat/lon).
    roi_rgb01: float32 RGB in [0,1]
    disk_mask: bool mask where planet exists in ROI
    Returns tex (tex_h, tex_w, 3) float32 in [0,1]
    """
    H, W = roi_rgb01.shape[:2]
    cx = (W - 1) * 0.5
    cy = (H - 1) * 0.5

    # estimate radius from mask area
    area = float(disk_mask.sum())
    r = float(np.sqrt(max(area, 1.0) / np.pi))
    r = max(r, 8.0)

    # lon in [-pi, pi], lat in [-pi/2, pi/2]
    lons = np.linspace(-np.pi, np.pi, tex_w, endpoint=False).astype(np.float32)
    lats = np.linspace(+0.5*np.pi, -0.5*np.pi, tex_h, endpoint=True).astype(np.float32)  # top->bottom

    Lon, Lat = np.meshgrid(lons, lats)

    # sphere point (unit)
    X = np.cos(Lat) * np.sin(Lon)
    Y = np.sin(Lat)
    Z = np.cos(Lat) * np.cos(Lon)

    # orthographic projection to disk:
    # image-plane coords: x = X, y = Y, z = Z (visible hemisphere Z>=0)
    vis = Z >= 0.0

    u = (cx + r * X).astype(np.float32)
    v = (cy + r * Y).astype(np.float32)

    # sample ROI using cv2.remap
    mapx = u
    mapy = v
    tex = cv2.remap(roi_rgb01, mapx, mapy, interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # kill tex where back hemisphere or outside disk
    # (outside disk also lands in black due to borderConstant, but we enforce)
    tex[~vis] = 0.0
    return tex


def _build_sphere_mesh(n_lat: int = 120, n_lon: int = 240):
    """
    Build sphere vertices and triangle indices.
    Returns:
      verts: (N,3) float32
      lats:  (N,) float32
      lons:  (N,) float32
      i,j,k triangle index lists
    """
    # lat: +pi/2 (north) to -pi/2 (south)
    lats = np.linspace(+0.5*np.pi, -0.5*np.pi, n_lat, endpoint=True).astype(np.float32)
    lons = np.linspace(-np.pi, np.pi, n_lon, endpoint=False).astype(np.float32)

    Lon, Lat = np.meshgrid(lons, lats)  # (n_lat,n_lon)

    x = (np.cos(Lat) * np.sin(Lon)).astype(np.float32)
    y = (np.sin(Lat)).astype(np.float32)
    z = (np.cos(Lat) * np.cos(Lon)).astype(np.float32)

    verts = np.stack([x, y, z], axis=-1).reshape(-1, 3)

    # triangles on the grid
    def idx(a, b):
        return a * n_lon + b

    I = []
    J = []
    K = []
    for a in range(n_lat - 1):
        for b in range(n_lon):
            b2 = (b + 1) % n_lon
            p00 = idx(a, b)
            p01 = idx(a, b2)
            p10 = idx(a + 1, b)
            p11 = idx(a + 1, b2)
            # two triangles per quad
            I.extend([p00, p00])
            J.extend([p10, p11])
            K.extend([p11, p01])

    return verts, Lat.reshape(-1), Lon.reshape(-1), I, J, K


def _sample_tex_colors(tex: np.ndarray, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """
    tex: (H,W,3) float32 [0,1] equirectangular, top=+lat
    lats/lons are per-vertex in radians.
    Returns vertexcolor: (N,4) uint8 RGBA
    """
    H, W = tex.shape[:2]

    # map lon [-pi,pi) -> u [0,W)
    u = ((lons + np.pi) / (2*np.pi) * W).astype(np.float32)
    # map lat [+pi/2,-pi/2] -> v [0,H)
    v = ((0.5*np.pi - lats) / (np.pi) * (H - 1)).astype(np.float32)

    # cv2.remap needs maps shaped (H?,W?) but we can sample manually using bilinear
    # We'll do fast bilinear sampling in numpy.
    u0 = np.floor(u).astype(np.int32) % W
    v0 = np.clip(np.floor(v).astype(np.int32), 0, H - 1)
    u1 = (u0 + 1) % W
    v1 = np.clip(v0 + 1, 0, H - 1)

    fu = (u - np.floor(u)).astype(np.float32)
    fv = (v - np.floor(v)).astype(np.float32)

    c00 = tex[v0, u0]
    c10 = tex[v1, u0]
    c01 = tex[v0, u1]
    c11 = tex[v1, u1]

    c0 = c00 * (1 - fv)[:, None] + c10 * fv[:, None]
    c1 = c01 * (1 - fv)[:, None] + c11 * fv[:, None]
    c = c0 * (1 - fu)[:, None] + c1 * fu[:, None]

    rgba = np.clip(c * 255.0, 0, 255).astype(np.uint8)
    alpha = np.full((rgba.shape[0],), 255, dtype=np.uint8)
    vertexcolor = np.concatenate([rgba, alpha[:, None]], axis=1)
    return vertexcolor

def _bilinear_sample_rgb(img01: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """img01: float32 [0,1] (H,W,3). u,v float pixel coords. returns float32 (N,3)."""
    H, W = img01.shape[:2]
    u = np.asarray(u, np.float32)
    v = np.asarray(v, np.float32)

    u0 = np.floor(u).astype(np.int32)
    v0 = np.floor(v).astype(np.int32)
    u1 = u0 + 1
    v1 = v0 + 1

    u0 = np.clip(u0, 0, W - 1); u1 = np.clip(u1, 0, W - 1)
    v0 = np.clip(v0, 0, H - 1); v1 = np.clip(v1, 0, H - 1)

    fu = (u - np.floor(u)).astype(np.float32)
    fv = (v - np.floor(v)).astype(np.float32)

    c00 = img01[v0, u0]
    c10 = img01[v1, u0]
    c01 = img01[v0, u1]
    c11 = img01[v1, u1]

    c0 = c00 * (1 - fv)[:, None] + c10 * fv[:, None]
    c1 = c01 * (1 - fv)[:, None] + c11 * fv[:, None]
    return c0 * (1 - fu)[:, None] + c1 * fu[:, None]

def _build_ring_mesh(n_theta: int = 360):
    """Returns (x,y,z,I,J,K) for a unit annulus strip with two radii per theta."""
    th = np.linspace(0, 2*np.pi, n_theta, endpoint=False).astype(np.float32)

    # we build verts for inner and outer separately later (because radii vary), but indices are constant.
    # vertex order: [inner0, outer0, inner1, outer1, ...]
    I = []
    J = []
    K = []
    for t in range(n_theta):
        t2 = (t + 1) % n_theta
        i0 = 2*t
        o0 = 2*t + 1
        i1 = 2*t2
        o1 = 2*t2 + 1
        # two triangles per quad
        I.extend([i0, o0])
        J.extend([o0, o1])
        K.extend([i1, i1])
    return th, np.asarray(I, np.int32), np.asarray(J, np.int32), np.asarray(K, np.int32)

def _build_ring_grid(n_r: int = 160, n_theta: int = 720):
    th = np.linspace(0, 2*np.pi, n_theta, endpoint=False).astype(np.float32)
    rr = np.linspace(0.0, 1.0, n_r, endpoint=True).astype(np.float32)  # normalized radius
    TH, RR = np.meshgrid(th, rr)  # (n_r, n_theta)

    # indices
    def vid(r, t): return r*n_theta + t
    I = []; J = []; K = []
    for r in range(n_r - 1):
        for t in range(n_theta):
            t2 = (t + 1) % n_theta
            p00 = vid(r, t)
            p01 = vid(r, t2)
            p10 = vid(r+1, t)
            p11 = vid(r+1, t2)
            I += [p00, p00]
            J += [p10, p11]
            K += [p11, p01]
    return TH.reshape(-1), RR.reshape(-1), np.asarray(I,np.int32), np.asarray(J,np.int32), np.asarray(K,np.int32)

def _ring_to_polar_texture(roi01, cx0, cy0, rpx, pa_deg, tilt, k_in, k_out,
                           n_r=160, n_theta=720):
    H, W = roi01.shape[:2]
    th = np.linspace(0, 2*np.pi, n_theta, endpoint=False).astype(np.float32)
    rr = np.linspace(k_in, k_out, n_r, endpoint=True).astype(np.float32)

    TH, RR = np.meshgrid(th, rr)  # (n_r, n_theta)

    x_e = RR * np.cos(TH)
    y_e = RR * np.sin(TH) * tilt

    a = np.deg2rad(pa_deg)
    c, s = np.cos(a), np.sin(a)

    x_img = cx0 + rpx * (x_e*c - y_e*s)
    y_img = cy0 + rpx * (x_e*s + y_e*c)

    mapx = x_img.astype(np.float32)
    mapy = y_img.astype(np.float32)

    polar = cv2.remap(
        roi01, mapx, mapy,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    return polar  # (n_r, n_theta, 3) float01


def export_planet_sphere_html(
    roi_rgb: np.ndarray,
    disk_mask: np.ndarray,
    out_path: str | None = None,
    n_lat: int = 120,
    n_lon: int = 240,
    title: str = "Planet Sphere",
    rings: dict | None = None,
):
    """
    Build an interactive Plotly Mesh3d with the planet texture wrapped to a sphere.
    Optional: add Saturn rings as a second Mesh3d using ROI-sampled vertex colors.

    IMPORTANT COORDINATE NOTE:
    - Your refinement/UI uses IMAGE coords: +x right, +y down.
    - Plotly scene is effectively +y up.
    We FORCE Plotly to behave like image coords by flipping Y for ALL meshes
    (sphere + rings) and setting camera.up to (0,-1,0).
    This makes ring PA/tilt match what you see in the refinement overlay.
    """
    import plotly.graph_objects as go

    # -----------------------------
    # 0) Ensure float01 ROI
    # -----------------------------
    if roi_rgb.dtype == np.uint8:
        roi01 = roi_rgb.astype(np.float32) / 255.0
    elif roi_rgb.dtype == np.uint16:
        roi01 = roi_rgb.astype(np.float32) / 65535.0
    else:
        roi01 = roi_rgb.astype(np.float32, copy=False)
        roi01 = np.clip(roi01, 0.0, 1.0)

    # -----------------------------
    # 1) Build equirect texture from disk
    # -----------------------------
    tex = _disk_to_equirect_texture(roi01, disk_mask, tex_h=256, tex_w=512)

    # -----------------------------
    # 2) Build sphere mesh (world coords)
    # -----------------------------
    verts, lats, lons, I, J, K = _build_sphere_mesh(n_lat=n_lat, n_lon=n_lon)

    # -----------------------------
    # 3) Sample per-vertex colors (RGBA uint8)
    # -----------------------------
    vcol = _sample_tex_colors(tex, lats, lons)
    vcol = np.asarray(vcol)
    if vcol.dtype != np.uint8:
        vcol = np.clip(vcol, 0, 255).astype(np.uint8)

    if vcol.ndim != 2 or vcol.shape[0] != verts.shape[0]:
        raise ValueError(f"vertex colors shape mismatch: vcol={vcol.shape}, verts={verts.shape}")

    if vcol.shape[1] == 3:
        alpha = np.full((vcol.shape[0], 1), 255, dtype=np.uint8)
        vcol = np.concatenate([vcol, alpha], axis=1)
    elif vcol.shape[1] != 4:
        raise ValueError(f"vertex colors must be RGB or RGBA; got {vcol.shape[1]} channels")

    # -----------------------------
    # 4) Make back hemisphere black (camera from +Z; z<0 is "back")
    # -----------------------------
    back = verts[:, 2] < 0.0
    if np.any(back):
        vcol = vcol.copy()
        vcol[back, 0:3] = 0
        vcol[back, 3] = 255

    # -----------------------------
    # 5) Flip Y to match IMAGE coords (x right, y down)
    # -----------------------------
    verts_plot = verts.astype(np.float32, copy=True)
    verts_plot[:, 1] *= -1.0  # <--- key fix: Plotly now behaves like image coords

    # Flip winding to fix "inside out" sphere
    I = np.asarray(I, dtype=np.int32)
    J = np.asarray(J, dtype=np.int32)
    K = np.asarray(K, dtype=np.int32)

    sphere_mesh = go.Mesh3d(
        x=verts_plot[:, 0], y=verts_plot[:, 1], z=verts_plot[:, 2],
        i=I, j=K, k=J,  # flip winding
        vertexcolor=vcol,
        flatshading=False,
        lighting=dict(
            ambient=0.55, diffuse=0.85, specular=0.25,
            roughness=0.9, fresnel=0.15
        ),
        lightposition=dict(x=2, y=1, z=3),
        name="Planet",
        hoverinfo="skip",
        showscale=False,
    )

    data = [sphere_mesh]

    # -----------------------------
    # 6) Optional rings mesh (Saturn) — POLAR TEXTURE + GRID MESH
    # -----------------------------
    if rings is not None:
        try:
            cx0 = float(rings.get("cx"))
            cy0 = float(rings.get("cy"))
            rpx = float(rings.get("r"))
            pa_deg = float(rings.get("pa", 0.0))
            tilt = float(rings.get("tilt", 0.35))       # b/a
            k_out = float(rings.get("k_out", 2.2))
            k_in  = float(rings.get("k_in", 1.25))

            tilt = float(np.clip(tilt, 0.02, 1.0))
            if k_out <= k_in:
                k_out = k_in + 0.05

            # Pick resolution (you can expose these as args if you want)
            n_r = int(rings.get("n_r", 180))
            n_theta = int(rings.get("n_theta", 720))

            # --- Build POLAR texture (n_r x n_theta x 3) float01
            # IMPORTANT: sample directly from roi01 (no pre-masking) to avoid bilinear edge darkening
            ring_polar01 = _ring_to_polar_texture(
                roi01,
                cx0=cx0, cy0=cy0,
                rpx=rpx,
                pa_deg=pa_deg,
                tilt=tilt,
                k_in=k_in, k_out=k_out,
                n_r=n_r, n_theta=n_theta
            )

            # --- Build GRID mesh topology aligned with polar texture
            TH_flat, RRn_flat, I2, J2, K2 = _build_ring_grid(n_r=n_r, n_theta=n_theta)

            # Convert normalized radius -> actual radius in planet radii
            RR_flat = (k_in + RRn_flat * (k_out - k_in)).astype(np.float32)

            # Base ring plane coords (world-ish): x right, y up
            x = (RR_flat * np.cos(TH_flat)).astype(np.float32)
            y = (RR_flat * np.sin(TH_flat)).astype(np.float32)
            z = np.zeros_like(x, dtype=np.float32)

            # --- 3D tilt: tilt=b/a = cos(inc) -> inc=arccos(tilt)
            inc = float(np.arccos(np.clip(tilt, 0.0, 1.0)))
            ci, si = float(np.cos(inc)), float(np.sin(inc))

            # Tilt around X: foreshorten Y, push into Z
            y2 = y * ci
            z2 = y * si

            # Rotate around Z by PA (MATCH refinement/UI)
            a = np.deg2rad(pa_deg)
            ca, sa = float(np.cos(a)), float(np.sin(a))
            x3 = x * ca - y2 * sa
            y3 = x * sa + y2 * ca
            z3 = z2 + 0.002  # tiny lift to reduce z-fighting

            # --- Vertex colors from polar texture (aligned!)
            # ring_polar01 shape: (n_r, n_theta, 3)
            # _build_ring_grid flattens in meshgrid order (rr rows, th cols) -> matches reshape(-1)
            cols01 = ring_polar01.reshape(-1, 3)
            cols_u8 = np.clip(cols01 * 255.0, 0, 255).astype(np.uint8)

            # --- validity mask for alpha (do this BEFORE shadowing to avoid alpha=0 for black shadow)
            valid = np.any(cols_u8 > 2, axis=1)

            # --- Occlude ring where Saturn's disk covers it (true disk silhouette)
            # Planet is unit sphere centered at origin; camera from +Z.
            r2 = x3*x3 + y3*y3
            inside = r2 <= 1.0
            z_front = np.sqrt(np.clip(1.0 - r2, 0.0, 1.0))
            shadow = inside & (z3 < z_front)

            if np.any(shadow):
                cols_u8 = cols_u8.copy()
                cols_u8[shadow, :3] = 0  # paint ring behind planet black

            alpha = (valid.astype(np.uint8) * 255)[:, None]
            vcol_ring = np.concatenate([cols_u8, alpha], axis=1)

            # Double-sided: duplicate tris with reversed winding
            I2 = np.asarray(I2, dtype=np.int32)
            J2 = np.asarray(J2, dtype=np.int32)
            K2 = np.asarray(K2, dtype=np.int32)
            I_all = np.concatenate([I2, I2])
            J_all = np.concatenate([J2, K2])
            K_all = np.concatenate([K2, J2])

            # Flip Y for Plotly (match IMAGE coords y-down like sphere)
            y3_plot = -y3

            ring_mesh = go.Mesh3d(
                x=x3, y=y3_plot, z=z3,
                i=I_all, j=J_all, k=K_all,
                vertexcolor=vcol_ring,
                flatshading=False,
                lighting=dict(
                    ambient=0.75, diffuse=0.65, specular=0.10,
                    roughness=1.0, fresnel=0.05
                ),
                name="Rings",
                hoverinfo="skip",
                showscale=False,
            )

            data.append(ring_mesh)

        except Exception:
            # If rings fail, still return the sphere.
            pass


    # -----------------------------
    # 7) Figure + layout
    # -----------------------------
    fig = go.Figure(data=data)

    fig.update_layout(
        title=title,
        margin=dict(l=0, r=0, b=0, t=40),
        scene=dict(
            aspectmode="data",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor="black",
            camera=dict(
                eye=dict(x=0.0, y=0.0, z=2.2),
                center=dict(x=0.0, y=0.0, z=0.0),
                up=dict(x=0.0, y=-1.0, z=0.0),   # image y-down
            ),
        ),
        paper_bgcolor="black",
        plot_bgcolor="black",
        showlegend=False,
    )

    html = fig.to_html(include_plotlyjs="cdn", full_html=True)

    if out_path is None:
        out_path = os.path.expanduser("~/planet_sphere.html")

    return html, out_path


def export_pseudo_surface_html(
    rgb: np.ndarray,
    out_path: str | None = None,
    *,
    title: str = "Pseudo Surface (Point Cloud)",
    max_dim: int = 420,
    z_scale: float = 0.35,
    depth_gamma: float = 1.15,
    blur_sigma: float = 1.2,
    invert: bool = False,
    block: int = 10,
    block_blur_sigma: float = 0.6,
    max_vertices: int = 250_000,
    point_size: float = 1.6,

    # height source(s)
    height_from: str = "brightness",      # "brightness" | "color" | "dual"

    # how points are COLORED (from dropdown)
    color_mode: str = "brightness",       # "brightness" | "depth" | "dual"
    depth_colorscale: str = "Turbo",
    depth_opacity: float = 0.55,
    depth_point_size: float | None = None,

    # Dual height controls (saturation-height layer)
    sat_opacity: float = 0.45,
    sat_point_size: float | None = None,

    # Optional: reduce saturation-height noise in dark background
    sat_luma_gate: float = 0.02,
    sat_luma_soft: float = 0.18,
):
    """
    Interactive Plotly Scatter3d point cloud.

    Z (height_from):
      • "brightness": luminance-derived height
      • "color":      saturation-derived height
      • "dual":       overlays TWO clouds (brightness-height + saturation-height)

    Coloring (color_mode) is ALWAYS honored, even for dual heights:
      • "brightness": RGB image colors
      • "depth":      colormap by height
      • "dual":       RGB + depth overlay (per height cloud)
    """
    import os
    import numpy as np
    import cv2
    import plotly.graph_objects as go

    x = np.asarray(rgb)
    if x.ndim != 3 or x.shape[2] < 3:
        raise ValueError("export_pseudo_surface_html expects RGB image (H,W,3).")

    # normalize mode strings
    cmode = (color_mode or "brightness").strip().lower()
    if cmode not in ("brightness", "depth", "dual"):
        cmode = "brightness"

    hmode = (height_from or "brightness").strip().lower()
    if hmode not in ("brightness", "color", "dual"):
        hmode = "brightness"

    # ---- float01 ----
    if x.dtype == np.uint8:
        img01 = x[..., :3].astype(np.float32) / 255.0
    elif x.dtype == np.uint16:
        img01 = x[..., :3].astype(np.float32) / 65535.0
    else:
        img01 = np.clip(x[..., :3].astype(np.float32, copy=False), 0.0, 1.0)

    H, W = img01.shape[:2]

    # ---- downsample ----
    max_dim = int(np.clip(max_dim, 128, 2048))
    s = float(max_dim) / float(max(H, W))
    if s < 1.0:
        newW = max(64, int(round(W * s)))
        newH = max(64, int(round(H * s)))
        img01 = cv2.resize(img01, (newW, newH), interpolation=cv2.INTER_AREA)

    hH, hW = img01.shape[:2]

    # ---- vertex cap ----
    max_vertices = int(max(10_000, max_vertices))
    if hH * hW > max_vertices:
        scale = np.sqrt(float(max_vertices) / float(hH * hW))
        newW = max(64, int(round(hW * scale)))
        newH = max(64, int(round(hH * scale)))
        img01 = cv2.resize(img01, (newW, newH), interpolation=cv2.INTER_AREA)
        hH, hW = img01.shape[:2]

    # ---- helpers ----
    def _robust01(base: np.ndarray) -> np.ndarray:
        base = base.astype(np.float32, copy=False)
        p_lo = float(np.percentile(base, 1.0))
        p_hi = float(np.percentile(base, 99.5))
        h = np.clip((base - p_lo) / max(p_hi - p_lo, 1e-9), 0.0, 1.0)

        if invert:
            h = 1.0 - h

        # coherence smoothing
        b = max(1, int(block))
        if b > 1:
            h = cv2.blur(h, (b, b), borderType=cv2.BORDER_REFLECT101)

        if block_blur_sigma and block_blur_sigma > 0:
            h = cv2.GaussianBlur(h, (0, 0), float(block_blur_sigma))

        if blur_sigma and blur_sigma > 0:
            h = cv2.GaussianBlur(h, (0, 0), float(blur_sigma))

        h = np.clip(h, 0.0, 1.0) ** max(1e-3, float(depth_gamma))
        return h.astype(np.float32)

    # luminance (also used for saturation gating)
    lum = (0.299 * img01[..., 0] + 0.587 * img01[..., 1] + 0.114 * img01[..., 2]).astype(np.float32)
    h01_lum = _robust01(lum)

    # saturation
    hsv = cv2.cvtColor(img01.astype(np.float32), cv2.COLOR_RGB2HSV)
    sat = hsv[..., 1].astype(np.float32)
    h01_sat = _robust01(sat)

    # Optional: suppress sat-height in very dark background
    if sat_luma_soft and sat_luma_soft > 0:
        gate = float(sat_luma_gate)
        soft = float(sat_luma_soft)
        wgt = np.clip((lum - gate) / max(soft, 1e-6), 0.0, 1.0).astype(np.float32)
        h01_sat = h01_sat * wgt

    # ---- depth scaling ----
    zmax = 0.5 * float(min(hH, hW)) * float(z_scale)
    Z_lum = ((h01_lum * 2.0) - 1.0) * zmax
    Z_sat = ((h01_sat * 2.0) - 1.0) * zmax

    # ---- XY grid ----
    yy, xx = np.mgrid[0:hH, 0:hW]
    X = (xx - (hW - 1) * 0.5).reshape(-1)
    Y = (yy - (hH - 1) * 0.5).reshape(-1)

    Zlum = Z_lum.reshape(-1)
    Zsat = Z_sat.reshape(-1)

    # ---- RGB strings ----
    rgb_u8 = np.clip(img01.reshape(-1, 3) * 255.0, 0, 255).astype(np.uint8)
    rgb_strings = [f"rgb({r},{g},{b})" for r, g, b in rgb_u8]

    traces: list = []

    # sizes / opacities per cloud
    ps_lum = float(point_size)
    ps_sat = float(sat_point_size) if sat_point_size is not None else float(point_size) * 1.05
    op_lum = 0.95
    op_sat = float(np.clip(sat_opacity, 0.05, 1.0))

    dps = float(depth_point_size) if depth_point_size is not None else float(point_size)

    def _add_cloud(
        *,
        name_prefix: str,
        Z: np.ndarray,
        size_rgb: float,
        size_depth: float,
        op_rgb: float,
        op_depth: float,
    ):
        """
        Add traces for a single height cloud, honoring cmode.
        """
        if cmode in ("brightness", "dual"):
            traces.append(
                go.Scatter3d(
                    x=X, y=Y, z=Z,
                    mode="markers",
                    marker=dict(
                        size=float(size_rgb),
                        color=rgb_strings,
                        opacity=1.0 if cmode == "brightness" else float(op_rgb),
                    ),
                    hoverinfo="skip",
                    name=f"{name_prefix} (RGB)" if cmode != "brightness" else f"{name_prefix}",
                )
            )

        if cmode in ("depth", "dual"):
            traces.append(
                go.Scatter3d(
                    x=X, y=Y, z=Z,
                    mode="markers",
                    marker=dict(
                        size=float(size_depth),
                        color=Z,  # numeric -> colorscale
                        colorscale=depth_colorscale,
                        cmin=float(Z.min()),
                        cmax=float(Z.max()),
                        opacity=1.0 if cmode == "depth" else float(op_depth),
                        showscale=(cmode == "depth"),
                        colorbar=dict(
                            title="Depth",
                            thickness=14,
                            len=0.6,
                        ) if cmode == "depth" else None,
                    ),
                    hoverinfo="skip",
                    name=f"{name_prefix} (Depth)" if cmode != "depth" else f"{name_prefix}",
                )
            )

    if hmode == "brightness":
        _add_cloud(
            name_prefix="Brightness Height",
            Z=Zlum,
            size_rgb=ps_lum,
            size_depth=dps,
            op_rgb=0.95,
            op_depth=depth_opacity,
        )
        show_legend = (cmode == "dual")

    elif hmode == "color":
        _add_cloud(
            name_prefix="Color Height",
            Z=Zsat,
            size_rgb=ps_lum,     # reuse point_size for single-mode
            size_depth=dps,
            op_rgb=0.95,
            op_depth=depth_opacity,
        )
        show_legend = (cmode == "dual")

    else:
        # dual heights: add BOTH clouds, each honoring cmode
        _add_cloud(
            name_prefix="Brightness Height",
            Z=Zlum,
            size_rgb=ps_lum,
            size_depth=dps,
            op_rgb=op_lum,
            op_depth=depth_opacity,
        )
        _add_cloud(
            name_prefix="Color Height",
            Z=Zsat,
            size_rgb=ps_sat,
            size_depth=dps,
            op_rgb=op_sat,
            op_depth=min(1.0, depth_opacity * 0.85),  # slightly softer overlay feels nicer
        )
        show_legend = True  # dual heights should show what’s what

    # ---- figure ----
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title,
        margin=dict(l=0, r=0, b=0, t=40),
        scene=dict(
            aspectmode="data",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor="black",
            camera=dict(
                eye=dict(x=0.0, y=-1.6, z=1.0),
                up=dict(x=0.0, y=-1.0, z=0.0),
            ),
        ),
        paper_bgcolor="black",
        plot_bgcolor="black",
        showlegend=bool(show_legend),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
        ),
    )

    html = fig.to_html(include_plotlyjs="cdn", full_html=True)
    if out_path is None:
        out_path = os.path.expanduser("~/pseudo_surface_pointcloud.html")

    return html, out_path


def deproject_galaxy_topdown_u8(
    roi01: np.ndarray,  # float32 [0..1], (H,W,3)
    cx0: float, cy0: float,
    rpx: float,
    pa_deg: float,
    tilt: float,        # b/a
    out_size: int = 800
) -> np.ndarray:
    """
    Returns RGB uint8 top-down view, outside-disk black.
    """
    import numpy as np
    import cv2

    H, W = roi01.shape[:2]
    out = int(max(64, out_size))

    # grid in disk plane [-1,1]
    yy, xx = np.mgrid[0:out, 0:out].astype(np.float32)
    u = (xx - (out - 1) * 0.5) / ((out - 1) * 0.5)
    v = (yy - (out - 1) * 0.5) / ((out - 1) * 0.5)
    rho = np.sqrt(u*u + v*v)

    # ellipse squash (inclination): y compressed by tilt
    tilt = float(np.clip(tilt, 0.02, 1.0))
    xe = u
    ye = v * tilt

    # rotate by PA
    a = np.deg2rad(pa_deg)
    ca, sa = float(np.cos(a)), float(np.sin(a))
    xr = xe * ca - ye * sa
    yr = xe * sa + ye * ca

    # scale to pixels + translate to ROI coords
    mapx = (cx0 + xr * rpx).astype(np.float32)
    mapy = (cy0 + yr * rpx).astype(np.float32)

    # sample
    img = np.clip(roi01, 0.0, 1.0)
    top01 = cv2.remap(img, mapx, mapy, interpolation=cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # mask outside disk-plane circle
    top01[rho > 1.0] = 0.0

    return np.clip(top01 * 255.0, 0, 255).astype(np.uint8)


# -----------------------------
# UI dialog
# -----------------------------

class PlanetProjectionDialog(QDialog):
    def __init__(self, parent=None, document=None):
        super().__init__(parent)
        self.setMinimumSize(520, 520)

        self.resize(560, 640)        
        self.setWindowTitle("3D Projection")
        self.setModal(False)
        self.parent = parent
        self.doc = document
        self.image = getattr(self.doc, "image", None) if self.doc is not None else None
        self._bg_img01 = None   # float32 [0,1] RGB, resized per ROI
        self._bg_path = ""
        self._left = None
        self._right = None
        self._wiggle_timer = QTimer(self)
        self._wiggle_timer.timeout.connect(self._on_wiggle_tick)
        self._wiggle_state = False
        self._last_preview_u8 = None  # last frame we pushed to preview (H,W,3) uint8
        self._preview_zoom = 1.0  # kept for compatibility but preview window owns zoom now
        self._preview_win = None
        self._wiggle_frames = None   # list of RGB uint8 frames
        self._wiggle_idx = 0
        self._wiggle_steps = 36      # default smoothness (can make this a UI control later)

        # Persist disk refinement within this dialog session (per image)
        self._disk_key = None              # identifies the current image
        self._disk_last = None             # (cx, cy, r) in FULL IMAGE coords
        self._disk_last_was_user = False   # True once user clicks "Use This Disk"

        self._build_ui()
        QTimer.singleShot(0, self._apply_initial_layout_fix)        
        self._update_enable()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(8)

        self.lbl_top = QLabel(
            "Create a synthetic stereo pair from a planet ROI (sphere reprojection).\n"
            "• Stereo: side-by-side (Parallel or Cross-eye)\n"
            "• Wiggle: alternates L/R to create depth motion\n"
            "Optional: add a static starfield background (no parallax)."
        )
        self.lbl_top.setWordWrap(True)
        outer.addWidget(self.lbl_top)

        prev_row = QHBoxLayout()
        self.btn_preview = QPushButton("Preview")
        self.btn_save_still = QPushButton("Save Still…")
        self.btn_save_wiggle = QPushButton("Save Wiggle…")

        prev_row.addWidget(self.btn_preview)
        prev_row.addWidget(self.btn_save_still)
        prev_row.addWidget(self.btn_save_wiggle)
        prev_row.addStretch(1)
        outer.addLayout(prev_row)

        self.btn_preview.clicked.connect(self._show_preview_window)
        self.btn_save_still.clicked.connect(self._save_still)
        self.btn_save_wiggle.clicked.connect(self._save_wiggle)
        # Controls
        box = QGroupBox("Parameters")
        form = QFormLayout(box)

        self.cmb_mode = QComboBox()
        self.cmb_mode.addItems([
            "Stereo (Parallel)  L | R",
            "Stereo (Cross-eye)  R | L",
            "Wiggle stereo (toggle L/R)",
            "Anaglyph (Red/Cyan 3D Glasses)",
            "Interactive 3D (HTML)",
            "Galaxy Polar View (Top-Down)",
        ])
        form.addRow("Output:", self.cmb_mode)

        self.cmb_planet_type = QComboBox()
        self.cmb_planet_type.addItems([
            "Normal (Sphere only)",
            "Saturn (Sphere + Rings)",
            "Pseudo surface (Height from brightness)",
            "Galaxy (Disk deprojection)",
        ])
        form.addRow("Planet type:", self.cmb_planet_type)

        # Rings group
        rings_box = QGroupBox("Saturn rings")
        rings_form = QFormLayout(rings_box)

        self.chk_rings = QCheckBox("Enable rings")
        self.chk_rings.setChecked(True)
        rings_form.addRow("", self.chk_rings)

        self.spin_ring_pa = QDoubleSpinBox()
        self.spin_ring_pa.setRange(-180.0, 180.0)
        self.spin_ring_pa.setSingleStep(1.0)
        self.spin_ring_pa.setValue(0.0)
        self.spin_ring_pa.setToolTip("Ring position angle in the image (deg). Rotate ellipse.")
        rings_form.addRow("Ring PA (deg):", self.spin_ring_pa)

        self.spin_ring_tilt = QDoubleSpinBox()
        self.spin_ring_tilt.setRange(0.05, 1.0)
        self.spin_ring_tilt.setSingleStep(0.02)
        self.spin_ring_tilt.setValue(0.35)
        self.spin_ring_tilt.setToolTip("Ellipse minor/major ratio (0..1). Smaller = more edge-on.")
        rings_form.addRow("Ring tilt (b/a):", self.spin_ring_tilt)

        self.spin_ring_outer = QDoubleSpinBox()
        self.spin_ring_outer.setRange(1.0, 4.0)
        self.spin_ring_outer.setSingleStep(0.05)
        self.spin_ring_outer.setValue(2.20)
        self.spin_ring_outer.setToolTip("Outer ring radius factor relative to body radius.")
        rings_form.addRow("Outer factor:", self.spin_ring_outer)

        self.spin_ring_inner = QDoubleSpinBox()
        self.spin_ring_inner.setRange(0.2, 3.5)
        self.spin_ring_inner.setSingleStep(0.05)
        self.spin_ring_inner.setValue(1.25)
        self.spin_ring_inner.setToolTip("Inner ring radius factor relative to body radius.")
        rings_form.addRow("Inner factor:", self.spin_ring_inner)

        form.addRow(rings_box)

        self.spin_theta = QDoubleSpinBox()
        self.spin_theta.setRange(0.2, 25.0)
        self.spin_theta.setSingleStep(0.2)
        self.spin_theta.setValue(6.0)
        self.spin_theta.setToolTip("Stereo strength in degrees. 6° usually looks best.")
        form.addRow("Strength (deg):", self.spin_theta)

        # Pseudo surface group
        ps_box = QGroupBox("Pseudo surface depth")
        ps_form = QFormLayout(ps_box)

        self.spin_ps_gamma = QDoubleSpinBox()
        self.spin_ps_gamma.setRange(0.3, 4.0)
        self.spin_ps_gamma.setSingleStep(0.05)
        self.spin_ps_gamma.setValue(1.15)
        self.spin_ps_gamma.setToolTip("Depth gamma. >1 emphasizes bright peaks; <1 broadens depth.")
        ps_form.addRow("Depth gamma:", self.spin_ps_gamma)

        self.spin_ps_blur = QDoubleSpinBox()
        self.spin_ps_blur.setRange(0.0, 12.0)
        self.spin_ps_blur.setSingleStep(0.2)
        self.spin_ps_blur.setValue(1.2)
        self.spin_ps_blur.setToolTip("Smooth height map to avoid noisy depth.")
        ps_form.addRow("Depth blur (px):", self.spin_ps_blur)

        self.chk_ps_invert = QCheckBox("Normal depth (bright = closer), uncheck for inverted")
        self.chk_ps_invert.setChecked(True)
        ps_form.addRow("", self.chk_ps_invert)


        # Pseudo-surface 3D coloring mode (how points are COLORED)
        self.cmb_ps_3d_mode = QComboBox(self)
        self.cmb_ps_3d_mode.addItems([
            "Brightness (RGB)",
            "Depth (Height Colormap)",
            "Dual (RGB + Depth)",
        ])
        self.cmb_ps_3d_mode.setToolTip(
            "3D point cloud coloring:\n"
            "• Brightness: points colored from the image (RGB)\n"
            "• Depth: points colored by height (colormap)\n"
            "• Dual: overlays RGB + depth coloring"
        )
        ps_form.addRow("3D Color Mode:", self.cmb_ps_3d_mode)

        # Height-from (what drives HEIGHT / Z)
        self.cmb_ps_height_from = QComboBox(self)
        self.cmb_ps_height_from.addItems([
            "Brightness (Luminance)",
            "Color Intensity (Saturation)",
            "Dual (Brightness + Color)",
        ])
        self.cmb_ps_height_from.setToolTip(
            "What drives the HEIGHT (Z) of the 3D point cloud:\n"
            "• Brightness: luminance-derived height\n"
            "• Color Intensity: saturation/chroma-derived height\n"
            "• Dual: overlays TWO clouds (brightness height + saturation height)\n\n"
            "Tip: Dual gives nebulae \"bulk\" even where brightness is flatter."
        )
        ps_form.addRow("Height From:", self.cmb_ps_height_from)

        # Max points (vertex cap)
        self.spin_ps_max_points = QSpinBox(self)
        self.spin_ps_max_points.setRange(50_000, 900_000)   # tune if you want
        self.spin_ps_max_points.setSingleStep(50_000)
        self.spin_ps_max_points.setValue(250_000)           # good default
        self.spin_ps_max_points.setToolTip(
            "Maximum number of points used in the 3D plot.\n"
            "Higher = more detail but heavier in the browser."
        )
        ps_form.addRow("Max Points:", self.spin_ps_max_points)

        # Dual saturation cloud opacity (only used when Height From == Dual)
        self.spin_ps_sat_opacity = QDoubleSpinBox(self)
        self.spin_ps_sat_opacity.setRange(0.05, 1.0)
        self.spin_ps_sat_opacity.setSingleStep(0.05)
        self.spin_ps_sat_opacity.setValue(0.45)
        self.spin_ps_sat_opacity.setToolTip(
            "Opacity of the saturation-height cloud when Height From is Dual.\n"
            "Lower = subtle bulk; higher = more pronounced volume."
        )
        ps_form.addRow("Dual Sat Opacity:", self.spin_ps_sat_opacity)

        # (Keep your existing)
        form.addRow(ps_box)

        self.chk_auto_roi = QCheckBox("Auto ROI from planet centroid (green channel)")
        self.chk_auto_roi.setChecked(True)
        form.addRow("", self.chk_auto_roi)

        self.spin_pad = QDoubleSpinBox()
        self.spin_pad.setRange(1.5, 6.0)
        self.spin_pad.setSingleStep(0.1)
        self.spin_pad.setValue(3.2)
        self.spin_pad.setToolTip("ROI size ≈ pad × planet radius")
        form.addRow("ROI pad (×radius):", self.spin_pad)

        self.spin_min = QSpinBox()
        self.spin_min.setRange(128, 2000)
        self.spin_min.setValue(240)
        form.addRow("ROI min size:", self.spin_min)

        self.spin_max = QSpinBox()
        self.spin_max.setRange(128, 5000)
        self.spin_max.setValue(900)
        form.addRow("ROI max size:", self.spin_max)

        # Disk review + reset row
        disk_row_w = QWidget()
        disk_row = QHBoxLayout(disk_row_w)
        disk_row.setContentsMargins(0, 0, 0, 0)

        self.chk_adjust_disk = QCheckBox("Review / adjust detected disk before generating")
        self.chk_adjust_disk.setChecked(True)
        disk_row.addWidget(self.chk_adjust_disk)

        self.btn_reset_disk = themed_toolbtn("edit-undo", "Reset disk detection")
        ...
        disk_row.addStretch(1)
        disk_row.addWidget(self.btn_reset_disk)

        form.addRow("", disk_row_w)


        self.chk_starfield = QCheckBox("Add static starfield background (no parallax)")
        self.chk_starfield.setChecked(True)
        form.addRow("", self.chk_starfield)

        self.spin_density = QDoubleSpinBox()
        self.spin_density.setRange(0.0, 0.2)
        self.spin_density.setSingleStep(0.005)
        self.spin_density.setValue(0.03)
        self.spin_density.setToolTip("Star seed density. Try 0.01–0.06 for visible fields.")
        form.addRow("Star density:", self.spin_density)

        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(0, 999999)
        self.spin_seed.setValue(1)
        form.addRow("Star seed:", self.spin_seed)
        # Background image row
        bg_row = QHBoxLayout()
        self.chk_bg_image = QCheckBox("Use background image")
        self.chk_bg_image.setChecked(False)
        bg_row.addWidget(self.chk_bg_image)

        self.bg_path_edit = QLineEdit()
        self.bg_path_edit.setReadOnly(True)
        self.bg_path_edit.setPlaceholderText("No background image selected")
        bg_row.addWidget(self.bg_path_edit, 1)

        self.btn_bg_choose = QPushButton("Choose…")
        self.btn_bg_choose.clicked.connect(self._choose_bg)
        bg_row.addWidget(self.btn_bg_choose)

        form.addRow("Background:", bg_row)

        # Background depth (%): UI slider -2..10, internal -2000..10000 (x1000)
        bg_depth_row = QHBoxLayout()

        self.sld_bg_depth = QSlider(Qt.Orientation.Horizontal)
        self.sld_bg_depth.setRange(-1000, 1000)   # -2.00 .. 10.00 in steps of 0.01
        self.sld_bg_depth.setValue(300)
        self.sld_bg_depth.setSingleStep(5)       # 0.05
        self.sld_bg_depth.setPageStep(25)        # 0.25
        bg_depth_row.addWidget(self.sld_bg_depth, 1)

        self.lbl_bg_depth = QLabel("0.00")
        self.lbl_bg_depth.setMinimumWidth(55)
        self.lbl_bg_depth.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        bg_depth_row.addWidget(self.lbl_bg_depth)

        def _update_bg_depth_label(v: int):
            self.lbl_bg_depth.setText(f"{v/100.0:.2f}")

        self.sld_bg_depth.valueChanged.connect(_update_bg_depth_label)
        _update_bg_depth_label(self.sld_bg_depth.value())

        tip = (
            "Background parallax as a percent of the planet parallax.\n"
            "0% = no parallax (screen-locked)\n"
            "25% = far behind (recommended)\n"
            "100% = same depth as planet (not recommended)\n\n"
            "UI shows -2..10; internally this is multiplied by 1000."
        )
        self.sld_bg_depth.setToolTip(tip)
        self.lbl_bg_depth.setToolTip(tip)

        form.addRow("Background depth (xR):", bg_depth_row)


        self.spin_wiggle_ms = QSpinBox()
        self.spin_wiggle_ms.setRange(40, 800)
        self.spin_wiggle_ms.setValue(120)
        form.addRow("Wiggle period (ms):", self.spin_wiggle_ms)

        outer.addWidget(box)

        # Buttons
        btns = QHBoxLayout()
        self.btn_generate = QPushButton("Generate")
        self.btn_stop = QPushButton("Stop Wiggle")
        self.btn_stop.setEnabled(False)
        self.btn_close = QPushButton("Close")
        btns.addWidget(self.btn_generate)
        btns.addWidget(self.btn_stop)
        btns.addStretch(1)
        btns.addWidget(self.btn_close)
        outer.addLayout(btns)

        def _set_form_row_visible(form_layout: QFormLayout, field_widget: QWidget, visible: bool):
            """Hide/show the entire row in a QFormLayout that contains field_widget."""
            for r in range(form_layout.rowCount()):
                item = form_layout.itemAt(r, QFormLayout.ItemRole.FieldRole)
                if item and item.widget() is field_widget:
                    label_item = form_layout.itemAt(r, QFormLayout.ItemRole.LabelRole)
                    if label_item and label_item.widget():
                        label_item.widget().setVisible(visible)
                    field_widget.setVisible(visible)
                    return



        def _update_type_enable():
            t = self.cmb_planet_type.currentIndex()
            is_sat = (t == 1)
            is_gal = (t == 3)
            is_pseudo = (t == 2)

            rings_box.setVisible(is_sat or is_gal)
            ps_box.setVisible(is_pseudo)

            # These are “planet disk” concepts; don’t use for pseudo surface
            self.chk_auto_roi.setEnabled(not is_pseudo)
            self.spin_pad.setEnabled(not is_pseudo)
            self.spin_min.setEnabled(not is_pseudo)
            self.spin_max.setEnabled(not is_pseudo)
            if hasattr(self, "chk_adjust_disk"):
                self.chk_adjust_disk.setEnabled(not is_pseudo)

            # Background/starfield: pseudo surface fills the whole frame anyway
            self.chk_starfield.setEnabled(not is_pseudo)
            self.spin_density.setEnabled(not is_pseudo)
            self.spin_seed.setEnabled(not is_pseudo)
            self.chk_bg_image.setEnabled(not is_pseudo)
            self.bg_path_edit.setEnabled(not is_pseudo)
            self.btn_bg_choose.setEnabled(not is_pseudo)
            self.sld_bg_depth.setEnabled(not is_pseudo)
            self.lbl_bg_depth.setEnabled(not is_pseudo)

            # --- Galaxy vs Saturn UI tweaks inside rings_box ---
            if is_gal:
                rings_box.setTitle("Galaxy disk")
                self.chk_rings.setVisible(False)  # only meaningful for Saturn
                _set_form_row_visible(rings_form, self.spin_ring_outer, False)
                _set_form_row_visible(rings_form, self.spin_ring_inner, False)

                # force output to Galaxy Polar View (optional, but prevents confusion)
                if self.cmb_mode.currentIndex() != 5:
                    self.cmb_mode.setCurrentIndex(5)
            else:
                rings_box.setTitle("Saturn rings")
                self.chk_rings.setVisible(True)
                _set_form_row_visible(rings_form, self.spin_ring_outer, True)
                _set_form_row_visible(rings_form, self.spin_ring_inner, True)

            self.adjustSize()  # shrink/grow dialog to fit


        self.cmb_planet_type.currentIndexChanged.connect(_update_type_enable)
        _update_type_enable()

        self.btn_generate.clicked.connect(self._generate)
        self.btn_stop.clicked.connect(self._stop_wiggle)
        self.btn_close.clicked.connect(self.close)

    def _apply_initial_layout_fix(self):
        # Re-run the same logic you already use (rings_box/ps_box visibility + adjustSize)
        try:
            # call your existing closure logic by nudging without changing index
            # simplest: just call adjustSize + clamp to something sane
            self.adjustSize()

            # Optional: clamp width/height so it doesn't blow out
            sh = self.sizeHint()
            w = max(self.minimumWidth(), sh.width())
            h = max(self.minimumHeight(), sh.height())
            self.resize(w, h)
        except Exception:
            pass


    def _reset_disk_cache(self):
        # Forget disk refinement for the CURRENT image only.
        self._disk_last = None
        self._disk_last_was_user = False

        # Disable until we detect/accept again
        if hasattr(self, "btn_reset_disk") and self.btn_reset_disk is not None:
            self.btn_reset_disk.setEnabled(False)

        # Optional: small user feedback
        QMessageBox.information(self, "Planet Projection", "Disk refinement reset. Next Generate will re-detect.")


    def _current_image_key(self, img: np.ndarray):
        """
        Stable key for the underlying image buffer, even if we take views like img[..., :3].
        """
        a = np.asarray(img)

        # Walk to the base ndarray so views/slices map to the same identity
        base = a
        while isinstance(getattr(base, "base", None), np.ndarray):
            base = base.base

        # Use raw data pointer + base dtype/shape (stable across views)
        ptr = int(base.__array_interface__["data"][0])
        return (ptr, tuple(base.shape), str(base.dtype))



    def _set_preview_zoom(self, z: float):
        """
        z = 1.0 => Fit-to-window (KeepAspectRatio)
        z = 0.0 => True 1:1 (no scaling, centered)
        otherwise => scale relative to Fit (e.g., 0.7 = smaller than fit, 1.4 = bigger than fit)
        """
        if z < 0.05 and z != 0.0:
            z = 0.05
        if z > 6.0:
            z = 6.0
        self._preview_zoom = float(z)

        # re-show last content
        if self._left is not None and self._right is not None:
            mode = self.cmb_mode.currentIndex()
            if mode == 2:
                # wiggle uses _set_preview_u8 directly; force refresh of current wiggle frame
                frame = self._right if self._wiggle_state else self._left
                self._set_preview_u8(frame)
            else:
                cross_eye = (mode == 0)
                self._show_stereo_pair(cross_eye=cross_eye)

    def _fit_scaled_size(self, img_w: int, img_h: int) -> tuple[int, int]:
        """Compute the fit-to-preview size (KeepAspectRatio)."""
        pw = max(1, self.preview.width())
        ph = max(1, self.preview.height())
        s = min(pw / float(img_w), ph / float(img_h))
        return int(round(img_w * s)), int(round(img_h * s))

    def _show_preview_window(self):
        if self._preview_win is None:
            self._preview_win = PlanetProjectionPreviewDialog(self)
            try:
                self._preview_win.resize(980, 600)
            except Exception:
                pass
        self._preview_win.show()
        self._preview_win.raise_()
        self._preview_win.activateWindow()


    def _open_preview_window(self):
        if self._preview_win is None:
            self._preview_win = PlanetProjectionPreviewDialog(self)
            try:
                self._preview_win.resize(980, 600)
            except Exception:
                pass
        self._preview_win.show()
        self._preview_win.raise_()
        self._preview_win.activateWindow()

    def _raise_preview_window(self):
        if self._preview_win is None:
            self._open_preview_window()
            return
        self._preview_win.show()
        self._preview_win.raise_()
        self._preview_win.activateWindow()

    def _push_preview_u8(self, rgb8: np.ndarray):
        rgb8 = np.asarray(rgb8)
        if rgb8.dtype != np.uint8:
            rgb8 = np.clip(rgb8, 0, 255).astype(np.uint8)
        if rgb8.ndim == 2:
            rgb8 = np.stack([rgb8, rgb8, rgb8], axis=2)
        if rgb8.shape[2] > 3:
            rgb8 = rgb8[..., :3]

        self._last_preview_u8 = rgb8

        # ensure preview exists
        if self._preview_win is None or not self._preview_win.isVisible():
            self._open_preview_window()
        self._preview_win.set_frame_u8(rgb8)

    def _compose_side_by_side_u8(self, left8: np.ndarray, right8: np.ndarray, *, swap_eyes: bool, gap_px: int) -> np.ndarray:
        L = np.asarray(left8)
        R = np.asarray(right8)

        if L.dtype != np.uint8:
            L = np.clip(L, 0, 255).astype(np.uint8)
        if R.dtype != np.uint8:
            R = np.clip(R, 0, 255).astype(np.uint8)

        if L.ndim == 2:
            L = np.stack([L, L, L], axis=2)
        if R.ndim == 2:
            R = np.stack([R, R, R], axis=2)

        if L.shape[2] > 3:
            L = L[..., :3]
        if R.shape[2] > 3:
            R = R[..., :3]

        if swap_eyes:
            L, R = R, L

        gap = int(max(0, gap_px))
        H = max(L.shape[0], R.shape[0])
        W = L.shape[1] + gap + R.shape[1]

        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        canvas[:L.shape[0], :L.shape[1]] = L
        canvas[:R.shape[0], L.shape[1] + gap:L.shape[1] + gap + R.shape[1]] = R
        return canvas


    def _bg_depth_internal_signed(self) -> float:
        """
        Read background depth from the UI slider, apply Saturn sign flip.

        Slider shows -10.00 .. +10.00 (label uses v/100).
        We'll interpret slider units as "percent * 100":
            depth_pct = (slider_value / 100.0)
        so slider=25 -> 0.25 (25% of planet disparity).
        """
        v = float(self.sld_bg_depth.value())  # int
        # Saturn: invert background direction
        if self.cmb_planet_type.currentIndex() == 1:  # 1 = Saturn
            v = -v
        return v

    
    def _set_bg_depth_internal(self, v: float):
        # internal -2000..10000 -> slider -200..1000
        self.sld_bg_depth.setValue(int(round(float(v) / 10.0)))


    def _update_enable(self):
        ok = (
            self.image is not None and isinstance(self.image, np.ndarray)
            and self.image.ndim == 3 and self.image.shape[2] >= 3
        )
        self.btn_generate.setEnabled(bool(ok))

    def _compute_roi(self):
        img = np.asarray(self.image)
        H, W = img.shape[:2]

        if self.chk_auto_roi.isChecked():
            # use green channel for centroid detection
            c = _planet_centroid_and_area(img[..., 1])
            if c is not None:
                cx, cy, area = c
                return _compute_roi_from_centroid(
                    H, W, cx, cy, area,
                    pad_mul=float(self.spin_pad.value()),
                    min_size=int(self.spin_min.value()),
                    max_size=int(self.spin_max.value()),
                )
            # fallback to center if centroid fails
        # center ROI
        s = int(np.clip(min(H, W) * 0.45, float(self.spin_min.value()), float(self.spin_max.value())))
        cx_i, cy_i = W // 2, H // 2
        x0 = max(0, cx_i - s // 2)
        y0 = max(0, cy_i - s // 2)
        x1 = min(W, x0 + s)
        y1 = min(H, y0 + s)
        return (x0, y0, x1, y1)

    def _generate(self):
        self._stop_wiggle()
        mode = int(self.cmb_mode.currentIndex())

        if self.image is None:
            QMessageBox.information(self, "Planet Projection", "No image loaded.")
            return

        img = np.asarray(self.image)
        if img.ndim != 3 or img.shape[2] < 3:
            QMessageBox.information(self, "Planet Projection", "Image must be RGB (3 channels).")
            return

        img = img[..., :3]  # ensure exactly RGB
        Hfull, Wfull = img.shape[:2]

        ptype = int(self.cmb_planet_type.currentIndex())
        is_pseudo = (ptype == 2)

        # ---- 0) reset cached disk if image changed ----
        key = self._current_image_key(img)
        if self._disk_key != key:
            self._disk_key = key
            self._disk_last = None
            self._disk_last_was_user = False

        # ---- 1) initial disk estimate (FULL IMAGE coords) ----
        if self._disk_last is not None:
            cx, cy, r = self._disk_last
        else:
            c = _planet_centroid_and_area(img[..., 1])
            if c is not None:
                cx, cy, area = c
                r = max(32.0, float(np.sqrt(area / np.pi)))
            else:
                cx = 0.5 * (Wfull - 1)
                cy = 0.5 * (Hfull - 1)
                r = 0.25 * min(Wfull, Hfull)

        # ---- 2) optional user adjustment (preloads previous) ----
        if (
            (not is_pseudo)
            and self.chk_auto_roi.isChecked()
            and getattr(self, "chk_adjust_disk", None) is not None
            and self.chk_adjust_disk.isChecked()
        ):
            is_saturn = (ptype == 1)
            rings_on = bool(
                is_saturn
                and getattr(self, "chk_rings", None) is not None
                and self.chk_rings.isChecked()
            )
            is_galaxy = (ptype == 3)
            overlay_mode = "none"
            if is_galaxy:
                overlay_mode = "galaxy"
            elif is_saturn and rings_on:
                overlay_mode = "saturn"

            dlg = PlanetDiskAdjustDialog(
                self, img[..., :3], cx, cy, r,
                overlay_mode=overlay_mode,
                ring_pa=float(self.spin_ring_pa.value()),
                ring_tilt=float(self.spin_ring_tilt.value()),
                ring_outer=float(self.spin_ring_outer.value()),
                ring_inner=float(self.spin_ring_inner.value()),
            )

            if dlg.exec() != QDialog.DialogCode.Accepted:
                return
            cx, cy, r = dlg.get_result()
            self._disk_last = (float(cx), float(cy), float(r))
            self._disk_last_was_user = True

            if overlay_mode in ("galaxy", "saturn"):
                pa, tilt, kout, kin = dlg.get_ring_result()
                self.spin_ring_pa.setValue(pa)
                self.spin_ring_tilt.setValue(tilt)
                if overlay_mode == "saturn":
                    self.spin_ring_outer.setValue(kout)
                    self.spin_ring_inner.setValue(kin)


            self._disk_last = (float(cx), float(cy), float(r))
            self._disk_last_was_user = True
        else:
            if self._disk_last is None:
                self._disk_last = (float(cx), float(cy), float(r))
                self._disk_last_was_user = False

        if hasattr(self, "btn_reset_disk") and self.btn_reset_disk is not None:
            self.btn_reset_disk.setEnabled(self._disk_last is not None)

        # ---- 3) ROI size from adjusted disk (pad/min/max) ----
        pad_mul = float(self.spin_pad.value())
        s = int(np.clip(r * pad_mul, float(self.spin_min.value()), float(self.spin_max.value())))

        # ---- PSEUDO SURFACE MODE: early exit ----
        if is_pseudo:
            roi = img  # whole image
            theta = float(self.spin_theta.value())

            left_w, right_w, maskL, maskR = make_pseudo_surface_pair(
                roi,
                theta_deg=theta,
                depth_gamma=float(self.spin_ps_gamma.value()),
                blur_sigma=float(self.spin_ps_blur.value()),
                invert=bool(self.chk_ps_invert.isChecked()),
            )

            Lw01 = left_w.astype(np.float32) / 255.0 if left_w.dtype == np.uint8 else left_w.astype(np.float32, copy=False)
            Rw01 = right_w.astype(np.float32) / 255.0 if right_w.dtype == np.uint8 else right_w.astype(np.float32, copy=False)
            Lw01 = np.clip(Lw01, 0.0, 1.0)
            Rw01 = np.clip(Rw01, 0.0, 1.0)

            self._left = np.clip(Lw01 * 255.0, 0, 255).astype(np.uint8)
            self._right = np.clip(Rw01 * 255.0, 0, 255).astype(np.uint8)

            # smooth wiggle not implemented for pseudo surface (yet) — keep toggle behavior
            self._wiggle_frames = None
            self._wiggle_state = False

            if mode == 4:
                try:
                    # color mode (how points are colored)
                    idx = int(self.cmb_ps_3d_mode.currentIndex()) if hasattr(self, "cmb_ps_3d_mode") else 0
                    color_mode = ("brightness", "depth", "dual")[max(0, min(2, idx))]

                    # height source (what drives Z)
                    hidx = int(self.cmb_ps_height_from.currentIndex()) if hasattr(self, "cmb_ps_height_from") else 0
                    height_from = ("brightness", "color", "dual")[max(0, min(2, hidx))]

                    # cap
                    max_pts = int(self.spin_ps_max_points.value()) if hasattr(self, "spin_ps_max_points") else 250_000

                    # dual saturation opacity (only used in height_from="dual")
                    sat_opacity = float(self.spin_ps_sat_opacity.value()) if hasattr(self, "spin_ps_sat_opacity") else 0.45

                    # title
                    if height_from == "brightness":
                        ht = "Brightness Height"
                    elif height_from == "color":
                        ht = "Color Intensity Height"
                    else:
                        ht = "Dual Height (Brightness + Color)"

                    html, default_path = export_pseudo_surface_html(
                        roi,
                        out_path=None,
                        title=f"Pseudo Surface ({ht})",
                        max_dim=2048,
                        z_scale=0.35,
                        depth_gamma=float(self.spin_ps_gamma.value()),
                        blur_sigma=float(self.spin_ps_blur.value()),
                        invert=bool(self.chk_ps_invert.isChecked()),
                        block=10,
                        block_blur_sigma=0.6,
                        max_vertices=max_pts,
                        point_size=1.6,
                        height_from=height_from,          # "brightness" | "color" | "dual"
                        color_mode=color_mode,            # "brightness" | "depth" | "dual"
                        depth_colorscale="Turbo",
                        depth_opacity=0.55,
                        depth_point_size=1.9,
                        sat_opacity=sat_opacity,          # used in height_from="dual"
                        sat_point_size=1.75,              # slightly different size looks great
                        sat_luma_gate=0.02,               # suppress color-noise in very dark background
                        sat_luma_soft=0.18,               # soft knee range
                    )

                    fn, _ = QFileDialog.getSaveFileName(
                        self,
                        "Save Pseudo Surface As",
                        default_path,
                        "HTML Files (*.html)"
                    )
                    if fn:
                        if not fn.lower().endswith(".html"):
                            fn += ".html"
                        with open(fn, "w", encoding="utf-8") as f:
                            f.write(html)

                    import tempfile, webbrowser
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode="w", encoding="utf-8")
                    tmp.write(html)
                    tmp.close()
                    webbrowser.open("file://" + tmp.name)

                except Exception as e:
                    QMessageBox.warning(self, "Pseudo Surface", f"Failed to generate 3D pseudo surface:\n{e}")
                return

            if mode == 2:
                self._start_wiggle()
                return

            if mode == 3:
                try:
                    ana = _make_anaglyph(self._left, self._right, swap_eyes=False)
                    self._push_preview_u8(ana)
                except Exception as e:
                    QMessageBox.warning(self, "Anaglyph", f"Failed to build anaglyph:\n{e}")
                return

            cross_eye = (mode == 1)
            self._show_stereo_pair(cross_eye=cross_eye)
            return

        # ---- Saturn rings ROI expansion (only increases s) ----
        is_saturn = (self.cmb_planet_type.currentIndex() == 1)
        rings_on = bool(is_saturn and getattr(self, "chk_rings", None) is not None and self.chk_rings.isChecked())

        if rings_on:
            tilt = float(self.spin_ring_tilt.value())
            pa = float(self.spin_ring_pa.value())
            k_out = float(self.spin_ring_outer.value())

            outer_boost = 1.05
            a_out = k_out * float(r) * outer_boost
            b_out = max(1.0, a_out * tilt)

            th = np.deg2rad(pa)
            cth, sth = np.cos(th), np.sin(th)

            dx = np.sqrt((a_out * cth) ** 2 + (b_out * sth) ** 2)
            dy = np.sqrt((a_out * sth) ** 2 + (b_out * cth) ** 2)
            need_half = float(max(dx, dy))

            margin = 12.0
            s_need = int(np.ceil(2.0 * (need_half + margin)))
            s = max(s, s_need)

        s = int(np.clip(s, float(self.spin_min.value()), float(self.spin_max.value())))

        # ---- ROI crop ALWAYS (for normal/saturn) ----
        cx_i, cy_i = int(round(cx)), int(round(cy))
        x0 = max(0, cx_i - s // 2)
        y0 = max(0, cy_i - s // 2)
        x1 = min(Wfull, x0 + s)
        y1 = min(Hfull, y0 + s)

        roi = img[y0:y1, x0:x1, :3]

        # ---- disk mask (ROI coords) ----
        H0, W0 = roi.shape[:2]
        yy, xx = np.mgrid[0:H0, 0:W0].astype(np.float32)
        cx0 = float(cx - x0)
        cy0 = float(cy - y0)
        disk = ((xx - cx0) ** 2 + (yy - cy0) ** 2) <= (float(r) ** 2)

        def to01(x):
            if x.dtype == np.uint8:
                return x.astype(np.float32) / 255.0
            if x.dtype == np.uint16:
                return x.astype(np.float32) / 65535.0
            return x.astype(np.float32, copy=False)

        theta = float(self.spin_theta.value())

        # ---- GALAXY TOP-DOWN (early exit) ----
        is_galaxy = (ptype == 3) or (mode == 5)  # planet_type==Galaxy OR output==Galaxy Polar View

        if is_galaxy:
            # Galaxy wants the ROI disk params (cx0, cy0, r) + PA/tilt
            roi01 = to01(roi)

            pa = float(self.spin_ring_pa.value())      # reuse ring PA widget as galaxy PA
            tilt = float(self.spin_ring_tilt.value())  # reuse ring tilt widget as galaxy b/a

            # choose output size: use ROI size or clamp to something reasonable
            out_size = int(max(256, min(2000, max(roi.shape[0], roi.shape[1]))))

            try:
                top8 = deproject_galaxy_topdown_u8(
                    roi01,
                    cx0=float(cx0), cy0=float(cy0),
                    rpx=float(r),
                    pa_deg=pa,
                    tilt=tilt,
                    out_size=out_size,
                )
            except Exception as e:
                QMessageBox.warning(self, "Galaxy Polar View", f"Failed to deproject galaxy:\n{e}")
                return

            # push single-frame output
            self._left = None
            self._right = None
            self._wiggle_frames = None
            self._wiggle_state = False

            self._last_preview_u8 = top8
            self._push_preview_u8(top8)
            return

        # ---- BODY (sphere reprojection) ----
        interp = cv2.INTER_LANCZOS4
        left_w, right_w, maskL, maskR = make_stereo_pair(
            roi, theta_deg=theta, disk_mask=disk, interp=interp
        )
        Lw01 = to01(left_w)
        Rw01 = to01(right_w)

        # ---- SATURN RINGS (optional) ----
        ringL01 = ringR01 = None
        ringL_front = ringL_back = ringR_front = ringR_back = None

        if rings_on:
            tilt = float(self.spin_ring_tilt.value())
            pa = float(self.spin_ring_pa.value())
            k_out = float(self.spin_ring_outer.value())
            k_in = float(self.spin_ring_inner.value())

            outer_boost = 1.05
            a_out = k_out * float(r) * outer_boost
            b_out = max(1.0, a_out * tilt)

            a_in = k_in * float(r)
            b_in = max(1.0, a_in * tilt)

            ringMask = _ellipse_annulus_mask(H0, W0, cx0, cy0, a_out, b_out, a_in, b_in, pa)

            roi01 = to01(roi)
            ring_tex01 = roi01.copy()
            ring_tex01[~ringMask] = 0.0

            mapLx, mapLy, mapRx, mapRy = _yaw_warp_maps(H0, W0, theta, cx0, cy0)

            ringL01 = cv2.remap(ring_tex01, mapLx, mapLy, interpolation=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            ringR01 = cv2.remap(ring_tex01, mapRx, mapRy, interpolation=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)

            front0, back0 = _ring_front_back_masks(H0, W0, cx0, cy0, pa, ringMask)
            f_u8 = (front0.astype(np.uint8) * 255)
            b_u8 = (back0.astype(np.uint8) * 255)

            ringL_front = cv2.remap(f_u8, mapLx, mapLy, interpolation=cv2.INTER_NEAREST,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 127
            ringL_back = cv2.remap(b_u8, mapLx, mapLy, interpolation=cv2.INTER_NEAREST,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 127
            ringR_front = cv2.remap(f_u8, mapRx, mapRy, interpolation=cv2.INTER_NEAREST,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 127
            ringR_back = cv2.remap(b_u8, mapRx, mapRy, interpolation=cv2.INTER_NEAREST,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 127

        # ---- centroid lock (planet-only) ----
        cL = _mask_centroid(maskL)
        cR = _mask_centroid(maskR)
        if cL is not None and cR is not None:
            tx = 0.5 * (cL[0] + cR[0])
            ty = 0.5 * (cL[1] + cR[1])

            dxL, dyL = (tx - cL[0]), (ty - cL[1])
            dxR, dyR = (tx - cR[0]), (ty - cR[1])

            Lw01 = _shift_image(Lw01, dxL, dyL, border_value=0)
            Rw01 = _shift_image(Rw01, dxR, dyR, border_value=0)
            maskL = _shift_mask(maskL, dxL, dyL)
            maskR = _shift_mask(maskR, dxR, dyR)

            # IMPORTANT for smooth wiggle: ring masks/textures need to be shifted too
            if ringL01 is not None:
                ringL01 = _shift_image(ringL01, dxL, dyL, border_value=0)
                ringR01 = _shift_image(ringR01, dxR, dyR, border_value=0)
                ringL_front = _shift_mask(ringL_front, dxL, dyL) if ringL_front is not None else None
                ringL_back  = _shift_mask(ringL_back,  dxL, dyL) if ringL_back  is not None else None
                ringR_front = _shift_mask(ringR_front, dxR, dyR) if ringR_front is not None else None
                ringR_back  = _shift_mask(ringR_back,  dxR, dyR) if ringR_back  is not None else None

        # ---- build background (bg01) ----
        H, W = roi.shape[:2]
        if self.chk_bg_image.isChecked() and self._bg_img01 is not None:
            bg = cv2.resize(self._bg_img01, (W, H), interpolation=cv2.INTER_AREA)
            bg01 = np.clip(bg.astype(np.float32, copy=False), 0.0, 1.0)
        else:
            bg01 = np.zeros((H, W, 3), dtype=np.float32)

        if self.chk_starfield.isChecked():
            bg01 = _add_starfield(
                bg01,
                density=float(self.spin_density.value()),
                seed=int(self.spin_seed.value()),
                star_sigma=0.8,
                brightness=0.9,
            )

        # ---- background parallax (for the still L/R that you already show) ----
        cL2 = _mask_centroid(maskL)
        cR2 = _mask_centroid(maskR)
        planet_disp_px = float(cL2[0] - cR2[0]) if (cL2 is not None and cR2 is not None) else 0.0

        depth_pct = float(self._bg_depth_internal_signed()) / 100.0
        bg_disp_px = planet_disp_px * depth_pct
        bg_shift = 0.5 * bg_disp_px

        max_bg_shift = 10.0 * min(H, W)
        bg_shift = float(np.clip(bg_shift, -max_bg_shift, +max_bg_shift))

        bgL = _shift_image(bg01, +bg_shift, 0.0, border_value=0)
        bgR = _shift_image(bg01, -bg_shift, 0.0, border_value=0)

        # ---- composite L/R ----
        Ldisp01 = bgL.copy()
        Rdisp01 = bgR.copy()

        if ringL01 is not None:
            Ldisp01[ringL_back & (~maskL)] = ringL01[ringL_back & (~maskL)]
            Rdisp01[ringR_back & (~maskR)] = ringR01[ringR_back & (~maskR)]

        Ldisp01[maskL] = Lw01[maskL]
        Rdisp01[maskR] = Rw01[maskR]

        if ringL01 is not None:
            Ldisp01[ringL_front] = ringL01[ringL_front]
            Rdisp01[ringR_front] = ringR01[ringR_front]

        self._left = np.clip(Ldisp01 * 255.0, 0, 255).astype(np.uint8)
        self._right = np.clip(Rdisp01 * 255.0, 0, 255).astype(np.uint8)
        self._wiggle_state = False

        # ---- mode handling ----
        if mode == 4:
            try:
                rings_kwargs = None
                if rings_on:
                    rings_kwargs = dict(
                        cx=float(cx0),
                        cy=float(cy0),
                        r=float(r),
                        pa=float(self.spin_ring_pa.value()),
                        tilt=float(self.spin_ring_tilt.value()),
                        k_out=float(self.spin_ring_outer.value()),
                        k_in=float(self.spin_ring_inner.value()),
                    )

                html, default_path = export_planet_sphere_html(
                    roi_rgb=roi,
                    disk_mask=disk,
                    out_path=None,
                    n_lat=140,
                    n_lon=280,
                    title="Saturn" if rings_on else "Planet Sphere",
                    rings=rings_kwargs,
                )

                fn, _ = QFileDialog.getSaveFileName(
                    self,
                    "Save Planet Sphere As",
                    default_path,
                    "HTML Files (*.html)"
                )
                if fn:
                    if not fn.lower().endswith(".html"):
                        fn += ".html"
                    with open(fn, "w", encoding="utf-8") as f:
                        f.write(html)

                import tempfile, webbrowser
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode="w", encoding="utf-8")
                tmp.write(html)
                tmp.close()
                webbrowser.open("file://" + tmp.name)

            except Exception as e:
                QMessageBox.warning(self, "Planet Sphere", f"Failed to generate 3D sphere:\n{e}")
            return

        if mode == 2:
            self._start_wiggle()
            return

        if mode == 3:
            try:
                ana = _make_anaglyph(self._left, self._right, swap_eyes=False)
                self._push_preview_u8(ana)
            except Exception as e:
                QMessageBox.warning(self, "Anaglyph", f"Failed to build anaglyph:\n{e}")
            return

        cross_eye = (mode == 0)
        self._show_stereo_pair(cross_eye=cross_eye)
        return

    def _render_composited_view_u8(self, theta_deg: float) -> np.ndarray:
        """
        Render ONE view (not L/R pair) at a given theta using the real reprojection math.
        Returns RGB uint8 frame for preview/save.
        """
        if not hasattr(self, "_wiggle_ctx") or self._wiggle_ctx is None:
            return None

        ctx = self._wiggle_ctx
        roi = ctx["roi"]
        disk = ctx["disk"]
        bg01 = ctx["bg01"]
        H0, W0 = roi.shape[:2]
        cx0, cy0 = float(ctx["cx0"]), float(ctx["cy0"])

        def to01(x):
            x = np.asarray(x)
            if x.dtype == np.uint8:
                return x.astype(np.float32) / 255.0
            if x.dtype == np.uint16:
                return x.astype(np.float32) / 65535.0
            return x.astype(np.float32, copy=False)

        # --- BODY: we can reuse make_stereo_pair by asking for a symmetric pair
        # and then choosing the "left" for +theta and "right" for -theta.
        # Easiest: call make_stereo_pair with theta_deg and take left_w/maskL as view.
        interp = cv2.INTER_LANCZOS4
        left_w, right_w, maskL, maskR = make_stereo_pair(roi, theta_deg=float(theta_deg), disk_mask=disk, interp=interp)

        view01 = to01(left_w)
        mask = maskL

        # --- RINGS (optional): warp ring texture with same theta and composite back/front
        ring01 = None
        ring_front = ring_back = None
        if ctx["rings_on"]:
            tilt = float(ctx["ring_tilt"])
            pa = float(ctx["ring_pa"])
            k_out = float(ctx["ring_outer"])
            k_in = float(ctx["ring_inner"])
            r = float(ctx["r"])

            outer_boost = 1.05
            a_out = k_out * r * outer_boost
            b_out = max(1.0, a_out * tilt)
            a_in = k_in * r
            b_in = max(1.0, a_in * tilt)

            ringMask = _ellipse_annulus_mask(H0, W0, cx0, cy0, a_out, b_out, a_in, b_in, pa)

            roi01 = to01(roi)
            ring_tex01 = roi01.copy()
            ring_tex01[~ringMask] = 0.0

            mapLx, mapLy, mapRx, mapRy = _yaw_warp_maps(H0, W0, float(theta_deg), cx0, cy0)
            ring01 = cv2.remap(ring_tex01, mapLx, mapLy, interpolation=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)

            front0, back0 = _ring_front_back_masks(H0, W0, cx0, cy0, pa, ringMask)
            f_u8 = (front0.astype(np.uint8) * 255)
            b_u8 = (back0.astype(np.uint8) * 255)
            ring_front = (cv2.remap(f_u8, mapLx, mapLy, interpolation=cv2.INTER_NEAREST,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 127)
            ring_back  = (cv2.remap(b_u8, mapLx, mapLy, interpolation=cv2.INTER_NEAREST,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 127)

        # --- BACKGROUND PARALLAX: compute disparity for this theta using centroids in this view vs the opposite view
        # We already have maskL/maskR from make_stereo_pair.
        cL = _mask_centroid(maskL)
        cR = _mask_centroid(maskR)
        planet_disp_px = float(cL[0] - cR[0]) if (cL is not None and cR is not None) else 0.0

        depth_pct = float(self._bg_depth_internal_signed()) / 100.0
        bg_disp_px = planet_disp_px * depth_pct
        bg_shift = 0.5 * bg_disp_px
        max_bg_shift = 10.0 * min(H0, W0)
        bg_shift = float(np.clip(bg_shift, -max_bg_shift, +max_bg_shift))

        bg = _shift_image(bg01, +bg_shift, 0.0, border_value=0)  # single-view bg

        # --- COMPOSITE
        out01 = bg.copy()

        if ring01 is not None:
            out01[ring_back & (~mask)] = ring01[ring_back & (~mask)]

        out01[mask] = view01[mask]

        if ring01 is not None:
            out01[ring_front] = ring01[ring_front]

        out8 = np.clip(out01 * 255.0, 0, 255).astype(np.uint8)
        return out8

    def _build_smooth_wiggle_frames(self):
        if not hasattr(self, "_wiggle_ctx") or self._wiggle_ctx is None:
            self._wiggle_frames = None
            return

        theta_max = float(self.spin_theta.value())
        steps = int(getattr(self, "_wiggle_steps", 36))
        steps = max(8, min(240, steps))

        frames = []
        for i in range(steps):
            phase = (2.0 * np.pi * i) / float(steps)
            theta_i = theta_max * float(np.sin(phase))  # smooth motion
            f = self._render_composited_view_u8(theta_i)
            if f is not None:
                frames.append(f)

        self._wiggle_frames = frames if frames else None


    def _show_stereo_pair(self, cross_eye: bool = False):
        if self._left is None or self._right is None:
            return

        # Ensure preview exists (same logic as _push_preview_u8)
        if self._preview_win is None or not self._preview_win.isVisible():
            self._open_preview_window()

        # IMPORTANT: pass the RAW L and R (do NOT pre-compose into one canvas)
        # swap_eyes handles parallel vs cross-eye ordering inside the preview window
        self._preview_win.set_stereo_u8(
            self._left,
            self._right,
            swap_eyes=bool(cross_eye),
            gap_px=16
        )

        # keep "last still" meaningful for Save Still…
        # If you want Save Still to save the side-by-side, ask preview for its composed canvas,
        # but for now, we’ll store a simple composed copy here:
        self._last_preview_u8 = self._compose_side_by_side_u8(
            self._left, self._right, swap_eyes=bool(cross_eye), gap_px=16
        )

    def _choose_bg(self):
        fn, _ = QFileDialog.getOpenFileName(
            self,
            "Select background image",
            "",
            "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp);;All Files (*)",
        )
        if not fn:
            return
        self._bg_path = fn
        self.bg_path_edit.setText(fn)

        try:
            # load via cv2, convert to RGB float01
            im = cv2.imread(fn, cv2.IMREAD_UNCHANGED)
            if im is None:
                raise RuntimeError("Could not read file.")
            if im.ndim == 2:
                im = np.stack([im, im, im], axis=2)
            if im.shape[2] > 3:
                im = im[..., :3]

            # BGR->RGB
            im = im[..., ::-1]

            if im.dtype == np.uint8:
                im01 = im.astype(np.float32) / 255.0
            elif im.dtype == np.uint16:
                im01 = im.astype(np.float32) / 65535.0
            else:
                im01 = im.astype(np.float32, copy=False)
                # best effort clamp
                im01 = np.clip(im01, 0.0, 1.0)

            self._bg_img01 = im01
        except Exception as e:
            self._bg_img01 = None
            QMessageBox.warning(self, "Background Image", f"Failed to load background:\n{e}")

    def _start_wiggle(self):
        if self._left is None or self._right is None:
            QMessageBox.information(self, "Wiggle", "Nothing to wiggle yet. Click Generate first.")
            return

        self.btn_stop.setEnabled(True)
        self._wiggle_state = False

        interval = int(self.spin_wiggle_ms.value())  # old meaning: toggle period
        interval = max(10, interval)

        self._wiggle_timer.start(interval)
        self._on_wiggle_tick()


    def _stop_wiggle(self):
        if self._wiggle_timer.isActive():
            self._wiggle_timer.stop()
        self.btn_stop.setEnabled(False)


    def _on_wiggle_tick(self):
        if self._left is None or self._right is None:
            return

        frame = self._right if self._wiggle_state else self._left
        self._wiggle_state = not self._wiggle_state
        self._push_preview_u8(frame)

    def _save_still(self):
        if self._last_preview_u8 is None:
            QMessageBox.information(self, "Save Still", "Nothing to save yet. Click Generate first.")
            return

        fn, filt = QFileDialog.getSaveFileName(
            self,
            "Save Still Image",
            "",
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;TIFF (*.tif *.tiff)"
        )
        if not fn:
            return

        img = self._last_preview_u8  # RGB uint8

        # decide format from extension (default to png)
        ext = os.path.splitext(fn)[1].lower()
        if ext == "":
            fn += ".png"
            ext = ".png"

        try:
            # use PIL for consistent RGB save
            from PIL import Image
            im = Image.fromarray(img, mode="RGB")
            if ext in (".jpg", ".jpeg"):
                im.save(fn, quality=95, subsampling=0)
            else:
                im.save(fn)
        except Exception as e:
            QMessageBox.warning(self, "Save Still", f"Failed to save:\n{e}")

    def _save_wiggle(self):
        if self._left is None or self._right is None:
            QMessageBox.information(self, "Save Wiggle", "Nothing to save yet. Click Generate first.")
            return

        fn, filt = QFileDialog.getSaveFileName(
            self,
            "Save Wiggle Animation",
            "",
            "Animated GIF (*.gif);;MP4 Video (*.mp4)"
        )
        if not fn:
            return

        want_mp4 = ("*.mp4" in filt) or fn.lower().endswith(".mp4")
        want_gif = ("*.gif" in filt) or fn.lower().endswith(".gif")

        # add extension if missing
        if os.path.splitext(fn)[1] == "":
            fn += ".mp4" if want_mp4 else ".gif"
            want_mp4 = fn.lower().endswith(".mp4")
            want_gif = fn.lower().endswith(".gif")

        def _ensure_rgb_u8(x):
            x = np.asarray(x)
            if x.dtype != np.uint8:
                x = np.clip(x, 0, 255).astype(np.uint8)
            if x.ndim == 2:
                x = np.stack([x, x, x], axis=2)
            if x.shape[2] > 3:
                x = x[..., :3]
            return x

        L = _ensure_rgb_u8(self._left)
        R = _ensure_rgb_u8(self._right)

        toggle_ms = int(self.spin_wiggle_ms.value())
        toggle_ms = max(10, toggle_ms)

        # old behavior: ~2 seconds total, alternating every toggle_ms
        fps = 1000.0 / float(toggle_ms)
        n_frames = max(2, int(round(2.0 * fps)))
        if n_frames % 2 == 1:
            n_frames += 1

        frames = [R if (i % 2 == 1) else L for i in range(n_frames)]

        if want_gif:
            try:
                from PIL import Image
                pil_frames = [Image.fromarray(f, mode="RGB") for f in frames]
                pil_frames[0].save(
                    fn,
                    save_all=True,
                    append_images=pil_frames[1:],
                    duration=toggle_ms,
                    loop=0,
                    disposal=2,
                    optimize=False
                )
                return
            except Exception as e:
                QMessageBox.warning(self, "Save Wiggle", f"Failed to save GIF:\n{e}")
                return

        # MP4
        try:
            import cv2
            h, w = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            vw = cv2.VideoWriter(fn, fourcc, float(fps), (w, h))
            if not vw.isOpened():
                raise RuntimeError("Could not open MP4 encoder (mp4v). This system may lack an MP4 codec.")

            for f in frames:
                vw.write(f[..., ::-1])  # RGB->BGR
            vw.release()
            return

        except Exception as e:
            QMessageBox.warning(
                self,
                "Save Wiggle (MP4)",
                "Failed to save MP4.\n\n"
                f"{e}\n\n"
                "Tip: GIF export should always work. If you need MP4 reliably, we can bundle/use ffmpeg."
            )
            return

    def closeEvent(self, e):
        self._stop_wiggle()
        super().closeEvent(e)

class PlanetDiskAdjustDialog(QDialog):
    """
    Manual override for planet disk center/radius.
    - Ctrl+drag to move center.
    - +/- or slider/spin to change radius.
    - Arrow buttons (and arrow keys) to nudge.
    Returns cx, cy, r in FULL IMAGE pixel coords.
    """
    def __init__(self, parent, img_rgb: np.ndarray, cx: float, cy: float, r: float,
                *, show_rings: bool = False, overlay_mode: str = "none",
                ring_pa: float = 0.0, ring_tilt: float = 0.35,
                ring_outer: float = 2.2, ring_inner: float = 1.25):
        super().__init__(parent)

        self.setWindowTitle("Adjust Planet Disk")
        self.setModal(True)
        self._preview_zoom = 1.0   # 1.0 = Fit
        self.overlay_mode = str(overlay_mode)
        self.show_rings = (self.overlay_mode in ("saturn", "galaxy"))
        self.img = np.asarray(img_rgb)
        self.H, self.W = self.img.shape[:2]

        self.cx = float(cx)
        self.cy = float(cy)
        self.r = float(r)

        # --- rings (optional) ---

        self.ring_pa = float(ring_pa)
        self.ring_tilt = float(ring_tilt)
        self.ring_outer = float(ring_outer)
        self.ring_inner = float(ring_inner)

        self._dragging = False
        self._drag_offset = (0.0, 0.0)

        # preview state
        self._disp8 = _to_u8_preview(self.img[..., :3])
        self._scale = 1.0
        self._offx = 0.0
        self._offy = 0.0

        self._build_ui()
        self._redraw()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(8)

        help_txt = (
            "Ctrl+Click+Drag to move the circle.\n"
            "Use Radius controls and arrow nudges for precision."
        )
        if self.show_rings:
            help_txt += "\nAdjust ring PA / tilt / inner / outer to match Saturn's rings."

        self.lbl_help = QLabel(help_txt)
        self.lbl_help.setWordWrap(True)
        outer.addWidget(self.lbl_help)

        # Zoom controls
        zoom_row = QHBoxLayout()
        self.btn_zoom_out = themed_toolbtn("zoom-out", "Zoom Out")
        self.btn_zoom_in  = themed_toolbtn("zoom-in", "Zoom In")
        self.btn_zoom_100 = themed_toolbtn("zoom-original", "1:1")
        self.btn_zoom_fit = themed_toolbtn("zoom-fit-best", "Fit")

        zoom_row.addStretch(1)
        zoom_row.addWidget(self.btn_zoom_out)
        zoom_row.addWidget(self.btn_zoom_fit)
        zoom_row.addWidget(self.btn_zoom_100)
        zoom_row.addWidget(self.btn_zoom_in)
        outer.addLayout(zoom_row)

        self.btn_zoom_out.clicked.connect(lambda: self._set_preview_zoom(self._preview_zoom * 0.8))
        self.btn_zoom_in.clicked.connect(lambda: self._set_preview_zoom(self._preview_zoom * 1.25))
        self.btn_zoom_fit.clicked.connect(lambda: self._set_preview_zoom(1.0))
        self.btn_zoom_100.clicked.connect(lambda: self._set_preview_zoom(0.0))

        # preview label
        self.preview = QLabel(self)
        self.preview.setMinimumSize(780, 420)
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setStyleSheet("background:#111; border:1px solid #333;")
        self.preview.setMouseTracking(True)
        self.preview.installEventFilter(self)
        outer.addWidget(self.preview)

        # --- rings controls (optional) ---
        if self.overlay_mode in ("saturn", "galaxy"):
            title = "Galaxy disk alignment" if self.overlay_mode == "galaxy" else "Saturn ring alignment"
            rings_box = QGroupBox(title)
            rings_form = QFormLayout(rings_box)

            # PA
            row, self.sld_ring_pa, self.spin_ring_pa = self._make_slider_spin_row(
                min_v=-180.0, max_v=180.0, step_v=1.0,
                value=self.ring_pa, decimals=0,
                on_change=self._on_ring_widgets_changed
            )
            rings_form.addRow("Disk PA (deg):" if self.overlay_mode=="galaxy" else "Ring PA (deg):", row)

            # tilt
            row, self.sld_ring_tilt, self.spin_ring_tilt = self._make_slider_spin_row(
                min_v=0.05, max_v=1.0, step_v=0.01,
                value=self.ring_tilt, decimals=2,
                on_change=self._on_ring_widgets_changed
            )
            rings_form.addRow("Disk tilt (b/a):" if self.overlay_mode=="galaxy" else "Ring tilt (b/a):", row)

            # ONLY Saturn gets inner/outer
            if self.overlay_mode == "saturn":
                row, self.sld_ring_outer, self.spin_ring_outer = self._make_slider_spin_row(
                    min_v=1.0, max_v=4.0, step_v=0.05,
                    value=self.ring_outer, decimals=2,
                    on_change=self._on_ring_widgets_changed,
                )
                rings_form.addRow("Outer factor:", row)

                row, self.sld_ring_inner, self.spin_ring_inner = self._make_slider_spin_row(
                    min_v=0.2, max_v=3.5, step_v=0.05,
                    value=self.ring_inner, decimals=2,
                    on_change=self._on_ring_widgets_changed,
                )
                rings_form.addRow("Inner factor:", row)
            outer.addWidget(rings_box)

        if self.overlay_mode == "saturn":
            help_txt += "\nAdjust ring PA / tilt / inner / outer to match Saturn's rings."
        elif self.overlay_mode == "galaxy":
            help_txt += "\nAdjust disk PA / tilt to match the galaxy's projected ellipse."
        self.lbl_help.setText(help_txt)

        # radius row
        rad_row = QHBoxLayout()
        self.btn_r_minus = QPushButton("Radius -")
        self.btn_r_plus = QPushButton("Radius +")
        self.spin_r = QDoubleSpinBox()
        self.spin_r.setRange(5.0, float(max(self.W, self.H)))
        self.spin_r.setDecimals(2)
        self.spin_r.setSingleStep(1.0)
        self.spin_r.setValue(self.r)
        self.spin_r.valueChanged.connect(self._on_radius_spin)

        self.sld_r = QSlider(Qt.Orientation.Horizontal)
        self.sld_r.setRange(5, int(max(self.W, self.H)))
        self.sld_r.setValue(int(round(self.r)))
        self.sld_r.valueChanged.connect(self._on_radius_slider)

        self.btn_r_minus.clicked.connect(lambda: self._bump_radius(-2.0))
        self.btn_r_plus.clicked.connect(lambda: self._bump_radius(+2.0))

        rad_row.addWidget(self.btn_r_minus)
        rad_row.addWidget(self.btn_r_plus)
        rad_row.addWidget(QLabel("R:"))
        rad_row.addWidget(self.spin_r)
        rad_row.addWidget(self.sld_r, 1)
        outer.addLayout(rad_row)

        # nudge row
        nud_row = QHBoxLayout()
        self.spin_step = QSpinBox()
        self.spin_step.setRange(1, 200)
        self.spin_step.setValue(2)
        nud_row.addWidget(QLabel("Nudge (px):"))
        nud_row.addWidget(self.spin_step)

        self.btn_left = QPushButton("◀")
        self.btn_right = QPushButton("▶")
        self.btn_up = QPushButton("▲")
        self.btn_down = QPushButton("▼")

        self.btn_left.clicked.connect(lambda: self._nudge(-1, 0))
        self.btn_right.clicked.connect(lambda: self._nudge(+1, 0))
        self.btn_up.clicked.connect(lambda: self._nudge(0, -1))
        self.btn_down.clicked.connect(lambda: self._nudge(0, +1))

        nud_row.addStretch(1)
        nud_row.addWidget(self.btn_up)
        nud_row.addWidget(self.btn_left)
        nud_row.addWidget(self.btn_right)
        nud_row.addWidget(self.btn_down)
        outer.addLayout(nud_row)

        # status
        self.lbl_status = QLabel("")
        outer.addWidget(self.lbl_status)

        # ok/cancel
        btn_row = QHBoxLayout()
        self.btn_ok = QPushButton("Use This Disk")
        self.btn_cancel = QPushButton("Cancel")
        btn_row.addWidget(self.btn_ok)
        btn_row.addStretch(1)
        btn_row.addWidget(self.btn_cancel)
        outer.addLayout(btn_row)

        self.btn_ok.clicked.connect(self.accept)
        self.btn_cancel.clicked.connect(self.reject)

    # ---------- coordinate helpers ----------
    def _make_slider_spin_row(self, *,
                            min_v: float, max_v: float, step_v: float,
                            value: float, decimals: int,
                            on_change):
        """
        Returns (row_layout, slider, spin).
        Slider is int-based; spin is float. They stay in sync.
        """
        scale = int(round(1.0 / step_v))  # e.g. 0.05 -> 20, 0.02 -> 50
        if scale <= 0:
            scale = 1

        sld = QSlider(Qt.Orientation.Horizontal, self)
        sld.setRange(int(round(min_v * scale)), int(round(max_v * scale)))
        sld.setSingleStep(1)
        sld.setPageStep(max(1, int(round(10 * scale * step_v))))  # about 10 steps
        sld.setValue(int(round(value * scale)))

        spn = QDoubleSpinBox(self)
        spn.setRange(min_v, max_v)
        spn.setDecimals(decimals)
        spn.setSingleStep(step_v)
        spn.setValue(value)
        spn.setFixedWidth(100)

        def sld_to_spin(iv: int):
            fv = iv / float(scale)
            spn.blockSignals(True)
            spn.setValue(fv)
            spn.blockSignals(False)
            on_change()

        def spin_to_sld(fv: float):
            iv = int(round(fv * scale))
            sld.blockSignals(True)
            sld.setValue(iv)
            sld.blockSignals(False)
            on_change()

        sld.valueChanged.connect(sld_to_spin)
        spn.valueChanged.connect(spin_to_sld)

        row = QHBoxLayout()
        row.addWidget(sld, 1)
        row.addWidget(spn)

        return row, sld, spn


    def _on_ring_widgets_changed(self):
        # Always present in saturn+galaxy
        if hasattr(self, "spin_ring_pa"):
            self.ring_pa = float(self.spin_ring_pa.value())
        if hasattr(self, "spin_ring_tilt"):
            self.ring_tilt = float(self.spin_ring_tilt.value())

        # Only present for saturn
        if self.overlay_mode == "saturn":
            if hasattr(self, "spin_ring_outer"):
                self.ring_outer = float(self.spin_ring_outer.value())
            if hasattr(self, "spin_ring_inner"):
                self.ring_inner = float(self.spin_ring_inner.value())

        self._redraw()

    def get_ring_result(self) -> tuple[float, float, float, float]:
        pa = float(getattr(self, "ring_pa", 0.0))
        tilt = float(getattr(self, "ring_tilt", 0.35))

        if self.overlay_mode == "saturn":
            kout = float(getattr(self, "ring_outer", 2.2))
            kin  = float(getattr(self, "ring_inner", 1.25))
        else:
            kout = float(getattr(self, "ring_outer", 2.2))  # harmless
            kin  = float(getattr(self, "ring_inner", 1.25))

        return (pa, tilt, kout, kin)

    def _on_ring_changed(self, *_):
        self.ring_pa = float(self.spin_ring_pa.value())
        self.ring_tilt = float(self.spin_ring_tilt.value())
        self.ring_outer = float(self.spin_ring_outer.value())
        self.ring_inner = float(self.spin_ring_inner.value())
        self._redraw()


    def _compute_fit_transform(self):
        # label size
        pw = max(1, self.preview.width())
        ph = max(1, self.preview.height())

        # scale factor to fit image into label
        sw = pw / float(self.W)
        sh = ph / float(self.H)
        self._scale = float(min(sw, sh))

        # the fitted draw size (in LABEL coords)
        draw_w = self.W * self._scale
        draw_h = self.H * self._scale

        # offsets in LABEL coords (letterboxing)
        self._offx = 0.5 * (pw - draw_w)
        self._offy = 0.5 * (ph - draw_h)

        # ALSO store pixmap-space scale after scaling
        # (pixmap is the scaled-to-fit image)
        self._pix_w = int(round(draw_w))
        self._pix_h = int(round(draw_h))

        # pixmap space has NO offx/offy; it starts at (0,0)
        self._pix_scale = self._scale

    def _img_to_label(self, x: float, y: float) -> tuple[float, float]:
        return (self._offx + x * self._scale, self._offy + y * self._scale)

    def _label_to_img(self, x: float, y: float) -> tuple[float, float]:
        ix = (x - self._offx) / max(self._scale, 1e-9)
        iy = (y - self._offy) / max(self._scale, 1e-9)
        return (ix, iy)

    def _img_to_pix(self, x: float, y: float) -> tuple[float, float]:
        # pixmap coords (0..pix_w, 0..pix_h)
        return (x * self._pix_scale, y * self._pix_scale)
    
    def _set_preview_zoom(self, z: float):
        if z < 0.05 and z != 0.0:
            z = 0.05
        if z > 8.0:
            z = 8.0
        self._preview_zoom = float(z)
        # currently we always draw "fit"; keep behavior consistent by just redrawing
        self._redraw()


    # ---------- drawing ----------
    def _redraw(self):
        self._compute_fit_transform()

        # base pixmap (fit into preview)
        qimg = QImage(
            self._disp8.data,
            self.W,
            self.H,
            int(self._disp8.strides[0]),
            QImage.Format.Format_RGB888,
        )
        pix = QPixmap.fromImage(qimg).scaled(
            self.preview.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        painter = QPainter(pix)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        # Determine overlay mode:
        # - "saturn": inner+outer ellipses
        # - "galaxy": single disk ellipse
        # - otherwise: none
        overlay_mode = getattr(self, "overlay_mode", None)
        if overlay_mode is None:
            # backwards compatibility with old flag
            overlay_mode = "saturn" if getattr(self, "show_rings", False) else "none"
        overlay_mode = str(overlay_mode).lower()

        # Map center to pix coords
        cxp, cyp = self._img_to_pix(self.cx, self.cy)

        # -----------------------------
        # Main circle + crosshair
        # -----------------------------
        # In galaxy mode, the ellipse is the important overlay; circle is optional.
        DRAW_MAIN_CIRCLE_IN_GALAXY = True  # set False if you want ONLY ellipse for galaxy

        if overlay_mode != "galaxy" or DRAW_MAIN_CIRCLE_IN_GALAXY:
            pen = QPen(QColor(0, 255, 0))
            pen.setWidth(3)
            painter.setPen(pen)

            rv = float(self.r) * float(self._pix_scale)
            painter.drawEllipse(
                QPoint(int(round(cxp)), int(round(cyp))),
                int(round(rv)),
                int(round(rv)),
            )

        # center crosshair
        pen2 = QPen(QColor(0, 255, 0))
        pen2.setWidth(2)
        painter.setPen(pen2)
        painter.drawLine(int(round(cxp - 8)), int(round(cyp)),     int(round(cxp + 8)), int(round(cyp)))
        painter.drawLine(int(round(cxp)),     int(round(cyp - 8)), int(round(cxp)),     int(round(cyp + 8)))

        # -----------------------------
        # Ellipse overlays
        # -----------------------------
        if overlay_mode in ("saturn", "galaxy"):
            try:
                pa = float(getattr(self, "ring_pa", 0.0))
                tilt = float(getattr(self, "ring_tilt", 0.35))
                tilt = max(0.01, min(1.0, tilt))

                # ellipse semi-axes in SOURCE pixels
                if overlay_mode == "galaxy":
                    # ONE ellipse: major axis = r, minor = r * tilt
                    a = float(self.r)
                    b = max(1.0, a * tilt)

                    # convert to PIX coords
                    a_p = a * float(self._pix_scale)
                    b_p = b * float(self._pix_scale)

                    painter.save()
                    painter.translate(cxp, cyp)
                    painter.rotate(pa)

                    penr = QPen(QColor(0, 255, 0))
                    penr.setWidth(2)
                    painter.setPen(penr)

                    painter.drawEllipse(
                        int(round(-a_p)), int(round(-b_p)),
                        int(round(2 * a_p)), int(round(2 * b_p)),
                    )

                    # minor-axis guide
                    pena = QPen(QColor(0, 200, 0))
                    pena.setWidth(2)
                    painter.setPen(pena)
                    painter.drawLine(0, int(round(-b_p)), 0, int(round(b_p)))

                    painter.restore()

                else:
                    # SATURN: inner + outer ellipse annulus
                    k_out = float(getattr(self, "ring_outer", 2.2))
                    k_in  = float(getattr(self, "ring_inner", 1.25))

                    a_out = k_out * float(self.r)
                    b_out = max(1.0, a_out * tilt)
                    a_in  = k_in  * float(self.r)
                    b_in  = max(1.0, a_in  * tilt)

                    a_out_p = a_out * float(self._pix_scale)
                    b_out_p = b_out * float(self._pix_scale)
                    a_in_p  = a_in  * float(self._pix_scale)
                    b_in_p  = b_in  * float(self._pix_scale)

                    painter.save()
                    painter.translate(cxp, cyp)
                    painter.rotate(pa)

                    penr = QPen(QColor(0, 255, 0))
                    penr.setWidth(2)
                    painter.setPen(penr)

                    painter.drawEllipse(
                        int(round(-a_out_p)), int(round(-b_out_p)),
                        int(round(2 * a_out_p)), int(round(2 * b_out_p)),
                    )
                    painter.drawEllipse(
                        int(round(-a_in_p)), int(round(-b_in_p)),
                        int(round(2 * a_in_p)), int(round(2 * b_in_p)),
                    )

                    # minor-axis guide
                    pena = QPen(QColor(0, 200, 0))
                    pena.setWidth(2)
                    painter.setPen(pena)
                    painter.drawLine(0, int(round(-b_out_p)), 0, int(round(b_out_p)))

                    painter.restore()

            except Exception:
                # keep UI alive if something weird happens
                pass

        painter.end()

        self.preview.setPixmap(pix)

        # status label
        if overlay_mode == "galaxy":
            pa = float(getattr(self, "ring_pa", 0.0))
            tilt = float(getattr(self, "ring_tilt", 0.35))
            self.lbl_status.setText(
                f"Center: ({self.cx:.1f}, {self.cy:.1f})   Radius: {self.r:.1f}px   "
                f"PA: {pa:.1f}°   Tilt(b/a): {tilt:.2f}"
            )
        elif overlay_mode == "saturn":
            pa = float(getattr(self, "ring_pa", 0.0))
            tilt = float(getattr(self, "ring_tilt", 0.35))
            kout = float(getattr(self, "ring_outer", 2.2))
            kin  = float(getattr(self, "ring_inner", 1.25))
            self.lbl_status.setText(
                f"Center: ({self.cx:.1f}, {self.cy:.1f})   Radius: {self.r:.1f}px   "
                f"PA: {pa:.1f}°   Tilt(b/a): {tilt:.2f}   Outer: {kout:.2f}   Inner: {kin:.2f}"
            )
        else:
            self.lbl_status.setText(f"Center: ({self.cx:.1f}, {self.cy:.1f})   Radius: {self.r:.1f}px")


    # ---------- UI callbacks ----------
    def _clamp(self):
        self.cx = float(np.clip(self.cx, 0.0, self.W - 1.0))
        self.cy = float(np.clip(self.cy, 0.0, self.H - 1.0))
        # radius cannot exceed image bounds too much; keep sane
        self.r = float(np.clip(self.r, 5.0, 2.0 * max(self.W, self.H)))

    def _bump_radius(self, dr: float):
        self.r += float(dr)
        self._clamp()
        self.spin_r.blockSignals(True)
        self.sld_r.blockSignals(True)
        self.spin_r.setValue(self.r)
        self.sld_r.setValue(int(round(self.r)))
        self.spin_r.blockSignals(False)
        self.sld_r.blockSignals(False)
        self._redraw()

    def _on_radius_spin(self, v: float):
        self.r = float(v)
        self._clamp()
        self.sld_r.blockSignals(True)
        self.sld_r.setValue(int(round(self.r)))
        self.sld_r.blockSignals(False)
        self._redraw()

    def _on_radius_slider(self, v: int):
        self.r = float(v)
        self._clamp()
        self.spin_r.blockSignals(True)
        self.spin_r.setValue(self.r)
        self.spin_r.blockSignals(False)
        self._redraw()

    def _nudge(self, dx: int, dy: int):
        step = int(self.spin_step.value())
        self.cx += dx * step
        self.cy += dy * step
        self._clamp()
        self._redraw()

    # ---------- events ----------
    def keyPressEvent(self, e):
        key = e.key()
        if key == Qt.Key.Key_Left:
            self._nudge(-1, 0); return
        if key == Qt.Key.Key_Right:
            self._nudge(+1, 0); return
        if key == Qt.Key.Key_Up:
            self._nudge(0, -1); return
        if key == Qt.Key.Key_Down:
            self._nudge(0, +1); return
        super().keyPressEvent(e)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._redraw()

    def eventFilter(self, obj, ev):
        if obj is self.preview:
            if ev.type() == ev.Type.MouseButtonPress:
                if ev.button() == Qt.MouseButton.LeftButton and (ev.modifiers() & Qt.KeyboardModifier.ControlModifier):
                    self._compute_fit_transform()
                    mx, my = float(ev.position().x()), float(ev.position().y())
                    ix, iy = self._label_to_img(mx, my)
                    # store drag offset so the center doesn't jump
                    self._dragging = True
                    self._drag_offset = (self.cx - ix, self.cy - iy)
                    return True

            if ev.type() == ev.Type.MouseMove and self._dragging:
                self._compute_fit_transform()
                mx, my = float(ev.position().x()), float(ev.position().y())
                ix, iy = self._label_to_img(mx, my)
                ox, oy = self._drag_offset
                self.cx = ix + ox
                self.cy = iy + oy
                self._clamp()
                self._redraw()
                return True

            if ev.type() == ev.Type.MouseButtonRelease and self._dragging:
                self._dragging = False
                return True

        return super().eventFilter(obj, ev)

    def get_result(self) -> tuple[float, float, float]:
        return (float(self.cx), float(self.cy), float(self.r))

class PlanetProjectionPreviewDialog(QDialog):
    """
    Separate preview window:
    - Shows the latest output frame (stereo pair or wiggle frame)
    - Provides Zoom controls: Fit / 100% / +/-.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("3D Projection — Preview")
        self.setModal(False)
        self._img_zoom = 1.0      # content zoom (1.0 = full view)
        self._img_pan_x = 0.0     # in source pixels, relative to center
        self._img_pan_y = 0.0
        self._dragging = False
        self._last_pos = None
        self._last_left8 = None
        self._last_right8 = None
        self._last_swap = False
        self._gap_px = 16
        self._last_L8 = None
        self._last_R8 = None
        self._last_swap_eyes = False
        self._last_gap_px = 16
        self._last_frame_u8 = None
        # content zoom is RELATIVE TO FIT (1.0 = fit)
        self._content_zoom = 1.0
        self._pan_x = 0.0  # pan in SOURCE PIXELS
        self._pan_y = 0.0

        self._preview_zoom = 1.0  # 1.0 = Fit, 0.0 = 1:1, else relative to Fit

        self._build_ui()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(8)

        # Zoom controls (toolbtn icons like the rest of SASpro)
        zoom_row = QHBoxLayout()
        self.btn_zoom_out = themed_toolbtn("zoom-out", "Zoom Out")
        self.btn_zoom_in  = themed_toolbtn("zoom-in", "Zoom In")
        self.btn_zoom_100 = themed_toolbtn("zoom-original", "1:1")
        self.btn_zoom_fit = themed_toolbtn("zoom-fit-best", "Fit")
        self.btn_save_view = themed_toolbtn("document-save", "Save current preview view…")
        self.btn_push = QPushButton("Push to New Document")



        self.btn_save_view.clicked.connect(self._save_current_view)
        self.btn_push.clicked.connect(self._push_to_new_document)
        zoom_row.addStretch(1)
        zoom_row.addWidget(self.btn_zoom_out)
        zoom_row.addWidget(self.btn_zoom_fit)
        zoom_row.addWidget(self.btn_zoom_100)
        zoom_row.addWidget(self.btn_zoom_in)
        zoom_row.addWidget(self.btn_save_view)
        zoom_row.addWidget(self.btn_push)
        outer.addLayout(zoom_row)

        self.btn_zoom_out.clicked.connect(lambda: self.set_zoom(self._preview_zoom * 0.8 if self._preview_zoom not in (0.0, 1.0) else 0.8))
        self.btn_zoom_in.clicked.connect(lambda: self.set_zoom(self._preview_zoom * 1.25 if self._preview_zoom not in (0.0, 1.0) else 1.25))
        self.btn_zoom_fit.clicked.connect(lambda: self.set_zoom(1.0))
        self.btn_zoom_100.clicked.connect(lambda: self.set_zoom(0.0))

        imgzoom_row = QHBoxLayout()
        self.btn_img_reset = themed_toolbtn("edit-undo", "Reset Image Pan/Zoom")

        self.sld_img_zoom = QSlider(Qt.Orientation.Horizontal, self)
        self.sld_img_zoom.setRange(0, 200)   # 0 -> fit, +200 -> zoom in
        self.sld_img_zoom.setValue(0)
        self.sld_img_zoom.setToolTip("Zoom into the image content (pan with mouse drag)")

        imgzoom_row.addWidget(QLabel("Image zoom:"))
        imgzoom_row.addWidget(self.sld_img_zoom, 1)
        imgzoom_row.addWidget(self.btn_img_reset)
        outer.addLayout(imgzoom_row)

        self.sld_img_zoom.valueChanged.connect(self._on_img_zoom_changed)
        self.btn_img_reset.clicked.connect(self._reset_img_view)

        # Preview label
        self.preview = QLabel(self)
        self.preview.setMinimumSize(900, 520)
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setStyleSheet("background:#111; border:1px solid #333;")
        self.preview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        outer.addWidget(self.preview)
        self.preview.setMouseTracking(True)
        self.preview.installEventFilter(self)
        self._last_rgb8 = None

    def set_zoom(self, z: float):
        if z < 0.05 and z != 0.0:
            z = 0.05
        if z > 8.0:
            z = 8.0
        self._preview_zoom = float(z)

        # IMPORTANT: redraw whichever mode we’re in
        if self._last_left8 is not None and self._last_right8 is not None:
            self._redraw()  # stereo path uses _preview_zoom inside _render_stereo()
        elif self._last_rgb8 is not None:
            self.set_frame_u8(self._last_rgb8)

    def _fit_scaled_size(self, img_w: int, img_h: int) -> tuple[int, int]:
        pw = max(1, self.preview.width())
        ph = max(1, self.preview.height())
        s = min(pw / float(img_w), ph / float(img_h))
        return int(round(img_w * s)), int(round(img_h * s))

    def set_frame_u8(self, rgb8: np.ndarray):
        self._last_rgb8 = np.asarray(rgb8)

        # apply content zoom/pan by cropping to viewport size
        disp8 = self._apply_camera_crop(self._last_rgb8)

        # NEW: cache “what user is looking at” for Push-to-Doc
        self._last_frame_u8 = np.asarray(disp8)

        # --- sanitize for QImage ---
        disp8 = np.asarray(disp8, dtype=np.uint8)
        if disp8.ndim == 2:
            disp8 = np.stack([disp8, disp8, disp8], axis=2)
        if disp8.shape[2] > 3:
            disp8 = disp8[..., :3]
        if not disp8.flags["C_CONTIGUOUS"]:
            disp8 = np.ascontiguousarray(disp8)

        # IMPORTANT: keep buffer alive on self for as long as pixmap uses it
        self._qimg_buf = disp8

        h, w = disp8.shape[:2]
        bytes_per_line = int(disp8.strides[0])

        ptr = sip.voidptr(disp8.ctypes.data)
        qimg = QImage(ptr, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        base = QPixmap.fromImage(qimg)

        # NOTE: from this point onward, use disp8 dimensions (not rgb8)
        if self._preview_zoom == 1.0:
            self.preview.setPixmap(base)  # already fit-with-aspect (letterboxed by QLabel alignment)
            return

        if self._preview_zoom == 0.0:
            self.preview.setPixmap(base)
            return

        fit_w, fit_h = self._fit_scaled_size(w, h)
        target_w = max(1, int(round(fit_w * self._preview_zoom)))
        target_h = max(1, int(round(fit_h * self._preview_zoom)))

        pix = base.scaled(
            QSize(target_w, target_h),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.preview.setPixmap(pix)

    def set_stereo_u8(self, left8: np.ndarray, right8: np.ndarray, *, swap_eyes: bool, gap_px: int = 16):
        self._last_left8 = np.asarray(left8)
        self._last_right8 = np.asarray(right8)
        self._last_swap = bool(swap_eyes)
        self._gap_px = int(max(0, gap_px))

        # keep save-view path in sync
        self._last_L8 = self._last_left8
        self._last_R8 = self._last_right8
        self._last_swap_eyes = self._last_swap
        self._last_gap_px = self._gap_px

        self._clamp_pan()
        self._redraw()

    def _redraw(self):
        if self._last_left8 is None or self._last_right8 is None:
            # fallback: if you still use set_frame_u8 for non-stereo cases
            if getattr(self, "_last_rgb8", None) is not None:
                self.set_frame_u8(self._last_rgb8)
            return
        self._render_stereo()

    def _render_stereo(self):
        L = self._ensure_rgb8(self._last_left8)
        R = self._ensure_rgb8(self._last_right8)

        Hs, Ws = L.shape[:2]

        pw = max(8, self.preview.width())
        ph = max(8, self.preview.height())
        gap = int(self._gap_px)

        view_w = max(8, (pw - gap) // 2)
        view_h = max(8, ph)

        # --- NEW: fit-rect INSIDE each eye viewport (letterbox/pillarbox) ---
        rx, ry, rw, rh = self._fit_rect(view_w, view_h, Ws, Hs)
        rw = max(8, rw); rh = max(8, rh)

        # cache the actual displayed rect for pan math (drag + clamp)
        self._eye_fit_rect = (rx, ry, rw, rh, view_w, view_h)

        # render each eye ONLY into rw x rh (aspect-safe), then paste into a black canvas
        Limg = self._crop_and_scale(L, rw, rh)
        Rimg = self._crop_and_scale(R, rw, rh)

        if self._last_swap:
            Limg, Rimg = Rimg, Limg

        Lcan = np.zeros((view_h, view_w, 3), dtype=np.uint8)
        Rcan = np.zeros((view_h, view_w, 3), dtype=np.uint8)
        Lcan[ry:ry+rh, rx:rx+rw] = Limg
        Rcan[ry:ry+rh, rx:rx+rw] = Rimg

        canvas_w = view_w + gap + view_w
        canvas = np.zeros((view_h, canvas_w, 3), dtype=np.uint8)
        canvas[:, 0:view_w] = Lcan
        canvas[:, view_w:view_w + gap] = 0
        canvas[:, view_w + gap:view_w + gap + view_w] = Rcan
        self._last_frame_u8 = canvas

        self._qimg_buf = canvas
        h, w = canvas.shape[:2]
        ptr = sip.voidptr(canvas.ctypes.data)
        qimg = QImage(ptr, w, h, int(canvas.strides[0]), QImage.Format.Format_RGB888)
        base = QPixmap.fromImage(qimg)

        # preview zoom (same as before)
        if self._preview_zoom == 1.0:
            pix = base.scaled(self.preview.size(), Qt.AspectRatioMode.KeepAspectRatio,
                            Qt.TransformationMode.SmoothTransformation)
            self.preview.setPixmap(pix)
            return
        if self._preview_zoom == 0.0:
            self.preview.setPixmap(base)
            return

        fit_w, fit_h = self._fit_scaled_size(w, h)
        target_w = max(1, int(round(fit_w * self._preview_zoom)))
        target_h = max(1, int(round(fit_h * self._preview_zoom)))
        pix = base.scaled(QSize(target_w, target_h), Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation)
        self.preview.setPixmap(pix)


    def _ensure_rgb8(self, img: np.ndarray) -> np.ndarray:
        x = np.asarray(img)
        if x.ndim == 2:
            x = np.stack([x, x, x], axis=2)
        if x.shape[2] > 3:
            x = x[..., :3]
        if x.dtype != np.uint8:
            x = np.clip(x, 0, 255).astype(np.uint8)
        if not x.flags["C_CONTIGUOUS"]:
            x = np.ascontiguousarray(x)
        return x

    def _crop_and_scale(self, src: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
        # content zoom is RELATIVE TO FIT
        H, W = src.shape[:2]

        # fit scale into per-eye viewport
        s_fit = min(out_w / float(W), out_h / float(H))
        s_fit = max(1e-9, float(s_fit))

        z = float(max(1e-6, self._content_zoom))

        # visible window in SOURCE pixels:
        win_w = int(round(out_w / (s_fit * z)))
        win_h = int(round(out_h / (s_fit * z)))
        win_w = max(8, min(W, win_w))
        win_h = max(8, min(H, win_h))

        cx = (W - 1) * 0.5 + float(self._pan_x)
        cy = (H - 1) * 0.5 + float(self._pan_y)

        x0 = int(round(cx - 0.5 * win_w))
        y0 = int(round(cy - 0.5 * win_h))
        x0 = max(0, min(W - win_w, x0))
        y0 = max(0, min(H - win_h, y0))

        crop = src[y0:y0 + win_h, x0:x0 + win_w]

        if crop.shape[1] != out_w or crop.shape[0] != out_h:
            crop = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

        return crop

    def _clamp_pan(self):
        if self._last_left8 is None:
            return

        src = self._ensure_rgb8(self._last_left8)
        H, W = src.shape[:2]

        pw = max(8, self.preview.width())
        ph = max(8, self.preview.height())
        gap = int(self._gap_px)
        view_w = max(8, (pw - gap) // 2)
        view_h = max(8, ph)

        # use the SAME fit rect used for drawing
        rx, ry, rw, rh = self._fit_rect(view_w, view_h, W, H)
        rw = max(8, rw); rh = max(8, rh)

        s_fit = min(rw / float(W), rh / float(H))
        s_fit = max(1e-9, float(s_fit))
        z = float(max(1e-6, self._content_zoom))

        win_w = int(round(rw / (s_fit * z)))
        win_h = int(round(rh / (s_fit * z)))

        max_pan_x = max(0.0, (W - win_w) * 0.5)
        max_pan_y = max(0.0, (H - win_h) * 0.5)

        self._pan_x = float(np.clip(self._pan_x, -max_pan_x, +max_pan_x))
        self._pan_y = float(np.clip(self._pan_y, -max_pan_y, +max_pan_y))


    def resizeEvent(self, e):
        super().resizeEvent(e)

        # Clamp whichever pan system you’re using
        self._clamp_pan()
        self._clamp_img_view()

        # Redraw correct mode
        if self._last_left8 is not None and self._last_right8 is not None:
            self._redraw()
        elif self._last_rgb8 is not None:
            self.set_frame_u8(self._last_rgb8)


    def _on_img_zoom_changed(self, v: int):
        # 0 -> 1.0 (fit), +50 -> 2.0, +100 -> 4.0
        # negative values zoom OUT from fit: -50 -> 0.5, -100 -> 0.25
        self._content_zoom = float(2.0 ** (v / 50.0))
        self._clamp_pan()
        self._redraw()

    def _reset_img_view(self):
        self._content_zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self.sld_img_zoom.blockSignals(True)
        self.sld_img_zoom.setValue(0)
        self.sld_img_zoom.blockSignals(False)
        self._redraw()

    def _clamp_img_view(self):
        if self._last_rgb8 is None:
            return
        img = self._last_rgb8
        H, W = img.shape[:2]
        z = float(max(1e-6, self._img_zoom))

        # viewport size in *label* pixels (content crop target)
        vw = max(8, self.preview.width())
        vh = max(8, self.preview.height())

        # crop window size in source pixels
        win_w = int(round(vw / z))
        win_h = int(round(vh / z))
        win_w = max(8, min(W, win_w))
        win_h = max(8, min(H, win_h))

        max_pan_x = max(0.0, (W - win_w) * 0.5)
        max_pan_y = max(0.0, (H - win_h) * 0.5)

        self._img_pan_x = float(np.clip(self._img_pan_x, -max_pan_x, +max_pan_x))
        self._img_pan_y = float(np.clip(self._img_pan_y, -max_pan_y, +max_pan_y))

    def _fit_rect(self, view_w: int, view_h: int, img_w: int, img_h: int) -> tuple[int,int,int,int]:
        """Return (x,y,w,h) of the largest rect inside view that matches img aspect."""
        if img_w <= 0 or img_h <= 0:
            return (0, 0, view_w, view_h)
        s = min(view_w / float(img_w), view_h / float(img_h))
        w = max(1, int(round(img_w * s)))
        h = max(1, int(round(img_h * s)))
        x = (view_w - w) // 2
        y = (view_h - h) // 2
        return (x, y, w, h)

    def _crop_to_aspect(self, W: int, H: int, target_aspect: float) -> tuple[int,int]:
        """Return (win_w, win_h) clamped to image bounds with exact target aspect."""
        win_w = W
        win_h = int(round(win_w / target_aspect))
        if win_h > H:
            win_h = H
            win_w = int(round(win_h * target_aspect))
        win_w = max(8, min(W, win_w))
        win_h = max(8, min(H, win_h))
        return win_w, win_h


    def _apply_camera_crop(self, rgb8: np.ndarray) -> np.ndarray:
        img = np.asarray(rgb8)
        H, W = img.shape[:2]

        vw = max(8, self.preview.width())
        vh = max(8, self.preview.height())

        # IMPORTANT: we fit INSIDE the label, preserving image aspect (letterbox)
        _, _, out_w, out_h = self._fit_rect(vw, vh, W, H)
        out_w = max(8, out_w)
        out_h = max(8, out_h)

        a = out_w / float(out_h)  # target aspect matches IMAGE aspect, not label’s

        z = float(max(1e-6, self._img_zoom))

        # aspect-correct window size in source pixels, driven by zoom
        win_w = int(round(W / z))
        win_h = int(round(win_w / a))
        if win_h > H:
            win_h = int(round(H / z))
            win_w = int(round(win_h * a))

        win_w, win_h = self._crop_to_aspect(W, H, a)

        cx = (W - 1) * 0.5 + float(self._img_pan_x)
        cy = (H - 1) * 0.5 + float(self._img_pan_y)

        x0 = int(round(cx - win_w * 0.5))
        y0 = int(round(cy - win_h * 0.5))
        x0 = max(0, min(W - win_w, x0))
        y0 = max(0, min(H - win_h, y0))

        crop = img[y0:y0 + win_h, x0:x0 + win_w]

        # scale to out_w/out_h (aspect matched => no warp)
        if crop.shape[1] != out_w or crop.shape[0] != out_h:
            crop = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        return crop

    def eventFilter(self, obj, ev):
        if obj is self.preview:
            if ev.type() == ev.Type.MouseButtonPress and ev.button() == Qt.MouseButton.LeftButton:
                self._dragging = True
                self._last_pos = ev.position().toPoint()
                return True

            if ev.type() == ev.Type.MouseMove and self._dragging:
                p = ev.position().toPoint()
                d = p - self._last_pos
                self._last_pos = p

                # compute s_fit for current viewport
                if self._last_left8 is not None:
                    src = self._ensure_rgb8(self._last_left8)
                    H, W = src.shape[:2]

                    pw = max(8, self.preview.width())
                    ph = max(8, self.preview.height())
                    view_w = max(8, (pw - int(self._gap_px)) // 2)
                    view_h = ph

                    rx, ry, rw, rh = self._fit_rect(view_w, view_h, W, H)
                    rw = max(8, rw); rh = max(8, rh)
                    s_fit = min(rw / float(W), rh / float(H))
                    s_fit = max(1e-9, float(s_fit))

                    z = float(max(1e-6, self._content_zoom))
                    scale = s_fit * z  # view_px per source_px

                    # drag right should move content right (so we pan LEFT in source coords)
                    self._pan_x -= float(d.x()) / scale
                    self._pan_y -= float(d.y()) / scale

                    self._clamp_pan()
                    self._redraw()
                return True

            if ev.type() == ev.Type.MouseButtonRelease and ev.button() == Qt.MouseButton.LeftButton:
                self._dragging = False
                self._last_pos = None
                return True

            if ev.type() == ev.Type.Wheel:
                # wheel zoom: update slider (keeps everything synced)
                delta = ev.angleDelta().y()
                step = 6 if delta > 0 else -6
                v = int(self.sld_img_zoom.value()) + step
                v = max(self.sld_img_zoom.minimum(), min(self.sld_img_zoom.maximum(), v))
                self.sld_img_zoom.setValue(v)
                return True

        return super().eventFilter(obj, ev)
    
    def _get_current_view_canvas_u8(self) -> np.ndarray | None:
        """
        Return exactly what the preview is displaying as an RGB uint8 canvas:
        [left_view | gap | right_view], including linked pan/zoom and preview scaling.
        """
        if self._last_L8 is None or self._last_R8 is None:
            return None

        L = np.asarray(self._last_L8)
        R = np.asarray(self._last_R8)

        # sanitize
        def _to_rgb_u8(x):
            x = np.asarray(x)
            if x.dtype != np.uint8:
                x = np.clip(x, 0, 255).astype(np.uint8)
            if x.ndim == 2:
                x = np.stack([x, x, x], axis=2)
            if x.shape[2] > 3:
                x = x[..., :3]
            if not x.flags["C_CONTIGUOUS"]:
                x = np.ascontiguousarray(x)
            return x

        L = _to_rgb_u8(L)
        R = _to_rgb_u8(R)

        if self._last_swap_eyes:
            L, R = R, L

        gap = int(max(0, self._last_gap_px))

        # --- per-eye viewport size in label pixels ---
        pw = max(8, self.preview.width())
        ph = max(8, self.preview.height())

        # reserve gap inside the label width
        view_w = max(8, (pw - gap) // 2)
        view_h = max(8, ph)

        # --- crop+scale each eye independently to its viewport ---
        Lview = self._crop_and_scale(L, view_w, view_h)
        Rview = self._crop_and_scale(R, view_w, view_h)

        # compose L|gap|R at label resolution
        canvas = np.zeros((view_h, view_w + gap + view_w, 3), dtype=np.uint8)
        canvas[:, :view_w] = Lview
        canvas[:, view_w + gap:view_w + gap + view_w] = Rview

        # --- now apply the existing "preview zoom" (fit/100%/relative-to-fit)
        # Fit is already "canvas == label size", so:
        if self._preview_zoom == 1.0:
            return canvas

        if self._preview_zoom == 0.0:
            # 1:1 means: no scaling beyond current composed pixels
            return canvas

        # relative-to-fit scaling
        target_w = max(8, int(round(canvas.shape[1] * float(self._preview_zoom))))
        target_h = max(8, int(round(canvas.shape[0] * float(self._preview_zoom))))
        canvas = cv2.resize(canvas, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        return canvas

    def _apply_camera_crop_to_viewport(self, rgb8: np.ndarray, view_w: int, view_h: int) -> np.ndarray:
        """
        Apply linked pan/zoom (_img_zoom/_img_pan_x/_img_pan_y) to ONE eye image,
        producing exactly (view_h, view_w, 3) uint8.
        """
        img = np.asarray(rgb8)
        H, W = img.shape[:2]

        z = float(max(1e-6, self._img_zoom))

        win_w = int(round(view_w / z))
        win_h = int(round(view_h / z))
        win_w = max(8, min(W, win_w))
        win_h = max(8, min(H, win_h))

        cx = (W - 1) * 0.5 + float(self._img_pan_x)
        cy = (H - 1) * 0.5 + float(self._img_pan_y)

        x0 = int(round(cx - win_w * 0.5))
        y0 = int(round(cy - win_h * 0.5))

        x0 = max(0, min(W - win_w, x0))
        y0 = max(0, min(H - win_h, y0))

        crop = img[y0:y0 + win_h, x0:x0 + win_w]

        if crop.shape[1] != view_w or crop.shape[0] != view_h:
            crop = cv2.resize(crop, (view_w, view_h), interpolation=cv2.INTER_LINEAR)

        return crop

    def _save_current_view(self):
        canvas = self._get_current_view_canvas_u8()
        if canvas is None:
            QMessageBox.information(self, "Save View", "No preview image to save yet.")
            return

        fn, filt = QFileDialog.getSaveFileName(
            self,
            "Save Preview View",
            "",
            "PNG (*.png);;JPEG (*.jpg *.jpeg)"
        )
        if not fn:
            return

        ext = os.path.splitext(fn)[1].lower()
        if ext == "":
            fn += ".png"
            ext = ".png"

        try:
            from PIL import Image
            im = Image.fromarray(canvas, mode="RGB")
            if ext in (".jpg", ".jpeg"):
                im.save(fn, quality=95, subsampling=0)
            else:
                im.save(fn)
        except Exception as e:
            QMessageBox.warning(self, "Save View", f"Failed to save:\n{e}")

    def _find_main_window(self):
        w = self
        from PyQt6.QtWidgets import QMainWindow, QApplication
        while w is not None and not isinstance(w, QMainWindow):
            w = w.parentWidget()
        if w:
            return w
        for tlw in QApplication.topLevelWidgets():
            if isinstance(tlw, QMainWindow):
                return tlw
        return None


    def _push_to_new_document(self):
        img_u8 = None

        # Prefer exact displayed canvas for stereo
        if self._last_L8 is not None and self._last_R8 is not None:
            img_u8 = self._get_current_view_canvas_u8()
        else:
            img_u8 = getattr(self, "_last_frame_u8", None)

        if img_u8 is None:
            QMessageBox.warning(self, "Push to New Document", "Nothing to push yet.")
            return

        # Convert to float32 [0..1] for SASpro docs (matches your other tools)

        arr = np.asarray(img_u8)
        if arr.ndim == 2:
            arr01 = arr.astype(np.float32) / 255.0
            meta = {"is_mono": True, "bit_depth": "8-bit"}
        else:
            if arr.shape[2] > 3:
                arr = arr[..., :3]
            arr01 = arr.astype(np.float32) / 255.0
            meta = {"is_mono": False, "bit_depth": "8-bit"}

        mw = self._find_main_window()
        dm = getattr(mw, "docman", None) if mw else None
        if not mw or not dm:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Push to New Document", "Main window or DocManager not available.")
            return

        title = "Planet Projection Preview"
        try:
            if hasattr(dm, "open_array"):
                doc = dm.open_array(arr01, metadata=meta, title=title)
            elif hasattr(dm, "create_document"):
                doc = dm.create_document(image=arr01, metadata=meta, name=title)
            else:
                raise RuntimeError("DocManager lacks open_array/create_document")

            # Spawn a view (same pattern as NBtoRGBStars)
            if hasattr(mw, "_spawn_subwindow_for"):
                mw._spawn_subwindow_for(doc)
            else:
                from setiastro.saspro.subwindow import ImageSubWindow
                sw = ImageSubWindow(doc, parent=mw)
                sw.setWindowTitle(title)
                sw.show()

        except Exception as e:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Push to New Document", f"Failed to open new view:\n{e}")
