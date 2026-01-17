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
    QSizePolicy, QFileDialog, QLineEdit, QSlider
)
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


def make_stereo_pair(roi_rgb: np.ndarray, theta_deg: float = 10.0, disk_mask: np.ndarray | None = None):
    if cv2 is None:
        dummy_mask = np.ones(roi_rgb.shape[:2], dtype=bool)
        return roi_rgb, roi_rgb, dummy_mask, dummy_mask

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

    left = cv2.remap(xf, mapLx, mapLy, interpolation=cv2.INTER_LANCZOS4,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    right = cv2.remap(xf, mapRx, mapRy, interpolation=cv2.INTER_LANCZOS4,
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
    alpha = (np.any(rgba > 0, axis=1)).astype(np.uint8) * 255
    vertexcolor = np.concatenate([rgba, alpha[:, None]], axis=1)
    return vertexcolor


def export_planet_sphere_html(roi_rgb: np.ndarray, disk_mask: np.ndarray,
                             out_path: str | None = None,
                             n_lat: int = 120, n_lon: int = 240,
                             title: str = "Planet Sphere"):
    """
    Build an interactive Plotly Mesh3d with the planet texture wrapped to a sphere.
    Returns (html_string, out_path). Caller can save/open.
    """

    # Lazy import so this file can load even if plotly isn't installed everywhere.
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
    # 2) Build sphere mesh
    #    verts: (N,3)
    #    lats,lons: (N,) in radians (assumed)
    #    I,J,K: triangle index arrays (int)
    # -----------------------------
    verts, lats, lons, I, J, K = _build_sphere_mesh(n_lat=n_lat, n_lon=n_lon)

    # -----------------------------
    # 3) Sample per-vertex colors
    #    Expect uint8 RGBA (N,4) or uint8 RGB(A) compatible with Plotly Mesh3d.vertexcolor
    # -----------------------------
    vcol = _sample_tex_colors(tex, lats, lons)

    # Ensure vcol is uint8 and has 4 channels (RGBA), because we'll overwrite to black for back hemisphere.
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
    # 4) Make back hemisphere black
    #    With camera from +Z, front hemisphere is z >= 0.
    # -----------------------------
    back = verts[:, 2] < 0.0
    if np.any(back):
        vcol = vcol.copy()
        vcol[back, 0:3] = 0     # RGB -> black
        vcol[back, 3] = 255     # alpha opaque

    # -----------------------------
    # 5) Fix "inside out" sphere by flipping triangle winding
    #    Swap J and K (or any two indices) to reverse normals.
    # -----------------------------
    I = np.asarray(I)
    J = np.asarray(J)
    K = np.asarray(K)

    mesh = go.Mesh3d(
        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
        i=I, j=K, k=J,                 # <-- winding flip fixes inside-out lighting
        vertexcolor=vcol,
        flatshading=False,
        lighting=dict(
            ambient=0.55, diffuse=0.85, specular=0.25,
            roughness=0.9, fresnel=0.15
        ),
        lightposition=dict(x=2, y=1, z=3),
        name="Planet",
        hoverinfo="skip",
        showscale=False
    )

    fig = go.Figure(data=[mesh])

    # Default camera: outside the sphere on +Z looking toward origin.
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
                up=dict(x=0.0, y=1.0, z=0.0)
            )
        ),
        paper_bgcolor="black",
        plot_bgcolor="black",
        showlegend=False
    )

    html = fig.to_html(include_plotlyjs="cdn", full_html=True)

    if out_path is None:
        out_path = os.path.expanduser("~/planet_sphere.html")

    return html, out_path

# -----------------------------
# UI dialog
# -----------------------------

class PlanetProjectionDialog(QDialog):
    def __init__(self, parent=None, document=None):
        super().__init__(parent)
        self.setMinimumSize(520, 520)
        self.setMaximumSize(560, 640)
        self.resize(560, 640)        
        self.setWindowTitle("Planet Projection — Stereo / Wiggle")
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
        self._preview_zoom = 1.0  # kept for compatibility but preview window owns zoom now
        self._preview_win = None

        # Persist disk refinement within this dialog session (per image)
        self._disk_key = None              # identifies the current image
        self._disk_last = None             # (cx, cy, r) in FULL IMAGE coords
        self._disk_last_was_user = False   # True once user clicks "Use This Disk"

        self._build_ui()
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
        self.btn_open_preview = QPushButton("Open Preview")
        self.btn_raise_preview = QPushButton("Show Preview")
        prev_row.addWidget(self.btn_open_preview)
        prev_row.addWidget(self.btn_raise_preview)
        prev_row.addStretch(1)
        outer.addLayout(prev_row)

        self.btn_open_preview.clicked.connect(self._open_preview_window)
        self.btn_raise_preview.clicked.connect(self._raise_preview_window)

        # Controls
        box = QGroupBox("Parameters")
        form = QFormLayout(box)

        self.cmb_mode = QComboBox()
        self.cmb_mode.addItems([
            "Stereo (Parallel)  L | R",
            "Stereo (Cross-eye)  R | L",
            "Wiggle stereo (toggle L/R)",
            "Anaglyph (Red/Cyan 3D Glasses)",
            "Interactive 3D Sphere (HTML)",
        ])
        form.addRow("Output:", self.cmb_mode)

        self.spin_theta = QDoubleSpinBox()
        self.spin_theta.setRange(0.2, 25.0)
        self.spin_theta.setSingleStep(0.2)
        self.spin_theta.setValue(6.0)
        self.spin_theta.setToolTip("Stereo strength in degrees. 6° usually looks best.")
        form.addRow("Strength (deg):", self.spin_theta)

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
        disk_row = QHBoxLayout()
        self.chk_adjust_disk = QCheckBox("Review / adjust detected disk before generating")
        self.chk_adjust_disk.setChecked(True)
        disk_row.addWidget(self.chk_adjust_disk)

        self.btn_reset_disk = themed_toolbtn("edit-undo", "Reset disk detection")
        self.btn_reset_disk.setToolTip("Forget the last accepted disk (center/radius) for this image and re-detect.")
        self.btn_reset_disk.setEnabled(False)  # only enabled once we have a cached disk
        self.btn_reset_disk.clicked.connect(self._reset_disk_cache)
        disk_row.addStretch(1)
        disk_row.addWidget(self.btn_reset_disk)

        form.addRow("", disk_row)


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
        self.sld_bg_depth.setValue(-300)
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

        self.btn_generate.clicked.connect(self._generate)
        self.btn_stop.clicked.connect(self._stop_wiggle)
        self.btn_close.clicked.connect(self.close)

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
        # good enough: changes whenever a new numpy array is assigned / new size/dtype
        return (id(img), img.shape, str(img.dtype))


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
        # ensure preview exists
        if self._preview_win is None or not self._preview_win.isVisible():
            self._open_preview_window()
        self._preview_win.set_frame_u8(rgb8)


    def _bg_depth_internal(self) -> float:
        # slider is -200..1000 representing -2.00..10.00
        # internal is (-2..10) * 1000 => -2000..10000
        return float(self.sld_bg_depth.value()) * 10.0

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

        if self.image is None:
            QMessageBox.information(self, "Planet Projection", "No image loaded.")
            return

        img = np.asarray(self.image)
        if img.ndim != 3 or img.shape[2] < 3:
            QMessageBox.information(self, "Planet Projection", "Image must be RGB (3 channels).")
            return

        Hfull, Wfull = img.shape[:2]

        # ---- 0) reset cached disk if image changed ----
        key = self._current_image_key(img)
        if self._disk_key != key:
            self._disk_key = key
            self._disk_last = None
            self._disk_last_was_user = False

        # ---- 1) initial disk estimate (FULL IMAGE coords) ----
        if self._disk_last is not None:
            # Use last accepted disk for this image
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
            self.chk_auto_roi.isChecked()
            and getattr(self, "chk_adjust_disk", None) is not None
            and self.chk_adjust_disk.isChecked()
        ):
            dlg = PlanetDiskAdjustDialog(self, img[..., :3], cx, cy, r)
            if dlg.exec() != QDialog.DialogCode.Accepted:
                # IMPORTANT: if user cancels, keep previously accepted disk (if any)
                return

            cx, cy, r = dlg.get_result()

            # Persist for future generates (parallel/cross/wiggle/html) within this dialog session
            self._disk_last = (float(cx), float(cy), float(r))
            self._disk_last_was_user = True
        else:
            # Even if adjust dialog is disabled, keep auto-detected disk cached for this image
            # so switching modes doesn't re-detect differently each time.
            if self._disk_last is None:
                self._disk_last = (float(cx), float(cy), float(r))
                self._disk_last_was_user = False

        if hasattr(self, "btn_reset_disk") and self.btn_reset_disk is not None:
            self.btn_reset_disk.setEnabled(self._disk_last is not None)

        # ---- 3) ROI from adjusted disk (still uses your existing pad/min/max) ----
        pad_mul = float(self.spin_pad.value())
        s = int(np.clip(r * pad_mul, float(self.spin_min.value()), float(self.spin_max.value())))

        cx_i, cy_i = int(round(cx)), int(round(cy))
        x0 = max(0, cx_i - s // 2)
        y0 = max(0, cy_i - s // 2)
        x1 = min(Wfull, x0 + s)
        y1 = min(Hfull, y0 + s)

        roi = img[y0:y1, x0:x1, :3]

        # ---- 4) disk mask from adjusted circle (ROI coords) ----
        H0, W0 = roi.shape[:2]
        yy, xx = np.mgrid[0:H0, 0:W0].astype(np.float32)
        cx0 = float(cx - x0)
        cy0 = float(cy - y0)

        disk = ((xx - cx0) ** 2 + (yy - cy0) ** 2) <= (float(r) ** 2)

        # --------- helper: float01 ----------
        def to01(x):
            if x.dtype == np.uint8:
                return x.astype(np.float32) / 255.0
            if x.dtype == np.uint16:
                return x.astype(np.float32) / 65535.0
            return x.astype(np.float32, copy=False)

        # IMPORTANT: disk mask for Plotly sphere export (and safe fallbacks)

        if disk is None:
            # fallback: simple centered circle inside ROI
            H0, W0 = roi.shape[:2]
            yy, xx = np.mgrid[0:H0, 0:W0].astype(np.float32)
            cx0 = (W0 - 1) * 0.5
            cy0 = (H0 - 1) * 0.5
            r0 = 0.49 * min(W0, H0)
            disk = ((xx - cx0) ** 2 + (yy - cy0) ** 2) <= (r0 * r0)

        theta = float(self.spin_theta.value())
        left_w, right_w, maskL, maskR = make_stereo_pair(roi, theta_deg=theta, disk_mask=disk)

        Lw01 = to01(left_w)
        Rw01 = to01(right_w)

        # ---- centroid lock (planet-only) ----
        cL = _mask_centroid(maskL)
        cR = _mask_centroid(maskR)
        if cL is not None and cR is not None:
            tx = 0.5 * (cL[0] + cR[0])
            ty = 0.5 * (cL[1] + cR[1])

            dxL, dyL = (tx - cL[0]), (ty - cL[1])
            dxR, dyR = (tx - cR[0]), (ty - cR[1])

            # shift planet images + masks (planet-only alignment)
            Lw01 = _shift_image(Lw01, dxL, dyL, border_value=0)
            Rw01 = _shift_image(Rw01, dxR, dyR, border_value=0)
            maskL = _shift_mask(maskL, dxL, dyL)
            maskR = _shift_mask(maskR, dxR, dyR)

        # ---- build background (bg01) first (float32 RGB [0,1]) ----
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

        # ---- background parallax depth control ----
        cL2 = _mask_centroid(maskL)
        cR2 = _mask_centroid(maskR)
        if cL2 is not None and cR2 is not None:
            planet_disp_px = float(cL2[0] - cR2[0])  # L - R disparity in pixels (signed)
        else:
            planet_disp_px = 0.0

        depth_pct = float(self._bg_depth_internal()) / 100.0
        bg_disp_px = planet_disp_px * depth_pct
        bg_shift = 0.5 * bg_disp_px

        # allow BIG shifts (you asked for -1000 etc.)
        max_bg_shift = 10.0 * min(H, W)
        bg_shift = float(np.clip(bg_shift, -max_bg_shift, +max_bg_shift))

        bgL = _shift_image(bg01, +bg_shift, 0.0, border_value=0)
        bgR = _shift_image(bg01, -bg_shift, 0.0, border_value=0)

        # ---- composite using the WARPED DISK masks ----
        Ldisp01 = bgL.copy()
        Rdisp01 = bgR.copy()
        Ldisp01[maskL] = Lw01[maskL]
        Rdisp01[maskR] = Rw01[maskR]

        Ldisp8 = np.clip(Ldisp01 * 255.0, 0, 255).astype(np.uint8)
        Rdisp8 = np.clip(Rdisp01 * 255.0, 0, 255).astype(np.uint8)

        self._left = Ldisp8
        self._right = Rdisp8
        self._wiggle_state = False

        mode = self.cmb_mode.currentIndex()

        # NOTE:
        # 0 = Stereo (Parallel)  L|R
        # 1 = Stereo (Cross-eye) R|L
        # 2 = Wiggle
        # 3 = Interactive 3D Sphere (HTML)
        if mode == 4:
            # ---- Plotly interactive sphere export (HTML) ----
            try:
                # IMPORTANT: this function must exist (see helper code we added)
                # export_planet_sphere_html(roi_rgb, disk_mask, out_path=None, n_lat=..., n_lon=..., title=...)
                html, default_path = export_planet_sphere_html(
                    roi_rgb=roi,
                    disk_mask=disk,
                    out_path=None,
                    n_lat=140,
                    n_lon=280,
                    title="Planet Sphere",
                )

                # Save prompt like WIMI
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

                # Always open a temp preview
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

        if mode == 3:
            # Anaglyph preview (single image)
            try:
                L8 = self._left if self._left.dtype == np.uint8 else _to_u8_preview(self._left)
                R8 = self._right if self._right.dtype == np.uint8 else _to_u8_preview(self._right)

                # usually no swap needed; if depth feels inverted, set swap_eyes=True
                ana = _make_anaglyph(L8, R8, swap_eyes=False)
                self._push_preview_u8(ana)
            except Exception as e:
                QMessageBox.warning(self, "Anaglyph", f"Failed to build anaglyph:\n{e}")
            return

        else:
            # cross-eye view should swap L/R
            cross_eye = (mode == 0)
            self._show_stereo_pair(cross_eye=cross_eye)

    def _show_stereo_pair(self, cross_eye: bool = False):
        if self._left is None or self._right is None:
            return

        L = self._left
        R = self._right
        if cross_eye:
            L, R = R, L  # swap

        # compose side-by-side with small gap
        gap = 10
        H = max(L.shape[0], R.shape[0])
        W = L.shape[1] + gap + R.shape[1]

        # upcast for display composition
        if L.dtype != np.uint8:
            L8 = _to_u8_preview(L)
            R8 = _to_u8_preview(R)
        else:
            L8 = L
            R8 = R

        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        canvas[:L8.shape[0], :L8.shape[1]] = L8
        canvas[:R8.shape[0], L8.shape[1] + gap:L8.shape[1] + gap + R8.shape[1]] = R8


        self._push_preview_u8(canvas)

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
        self.btn_stop.setEnabled(True)
        self._wiggle_state = False
        self._wiggle_timer.start(int(self.spin_wiggle_ms.value()))
        self._on_wiggle_tick()

    def _stop_wiggle(self):
        if self._wiggle_timer.isActive():
            self._wiggle_timer.stop()
        self.btn_stop.setEnabled(False)

    def _on_wiggle_tick(self):
        if self._left is None or self._right is None:
            return

        self._wiggle_state = not self._wiggle_state
        frame = self._right if self._wiggle_state else self._left
        self._push_preview_u8(frame)

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
    def __init__(self, parent, img_rgb: np.ndarray, cx: float, cy: float, r: float):
        super().__init__(parent)


        self.setWindowTitle("Adjust Planet Disk")
        self.setModal(True)
        self._preview_zoom = 1.0   # 1.0 = Fit
        self.img = np.asarray(img_rgb)
        self.H, self.W = self.img.shape[:2]

        self.cx = float(cx)
        self.cy = float(cy)
        self.r = float(r)

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

        self.lbl_help = QLabel(
            "Ctrl+Click+Drag to move the circle.\n"
            "Use Radius controls and arrow nudges for precision."
        )
        self.lbl_help.setWordWrap(True)
        outer.addWidget(self.lbl_help)

        # Zoom controls (for relaxed parallel viewing)
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
        self.btn_zoom_100.clicked.connect(lambda: self._set_preview_zoom(0.0))  # 0.0 means "true 1:1 pixels"

        # preview label
        self.preview = QLabel(self)
        self.preview.setMinimumSize(780, 420)
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setStyleSheet("background:#111; border:1px solid #333;")
        self.preview.setMouseTracking(True)
        self.preview.installEventFilter(self)
        outer.addWidget(self.preview)

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
    # ---------- drawing ----------
    def _redraw(self):
        self._compute_fit_transform()

        # base pixmap (this pixmap is ALREADY the fitted size)
        qimg = QImage(self._disp8.data, self.W, self.H, self._disp8.strides[0], QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(
            self.preview.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        # ---- draw overlay in PIXMAP coordinates ----
        painter = QPainter(pix)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        pen = QPen(QColor(0, 255, 0))
        pen.setWidth(3)
        painter.setPen(pen)

        # map image center/radius -> pixmap coords
        cxp, cyp = self._img_to_pix(self.cx, self.cy)
        rv = self.r * self._pix_scale

        painter.drawEllipse(QPoint(int(round(cxp)), int(round(cyp))), int(round(rv)), int(round(rv)))

        # center crosshair
        pen2 = QPen(QColor(0, 255, 0))
        pen2.setWidth(2)
        painter.setPen(pen2)
        painter.drawLine(int(round(cxp - 8)), int(round(cyp)), int(round(cxp + 8)), int(round(cyp)))
        painter.drawLine(int(round(cxp)), int(round(cyp - 8)), int(round(cxp)), int(round(cyp + 8)))

        painter.end()

        # set pixmap on label (Qt handles centering within the label)
        self.preview.setPixmap(pix)

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
        self.setWindowTitle("Planet Projection — Preview")
        self.setModal(False)

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

        zoom_row.addStretch(1)
        zoom_row.addWidget(self.btn_zoom_out)
        zoom_row.addWidget(self.btn_zoom_fit)
        zoom_row.addWidget(self.btn_zoom_100)
        zoom_row.addWidget(self.btn_zoom_in)
        outer.addLayout(zoom_row)

        self.btn_zoom_out.clicked.connect(lambda: self.set_zoom(self._preview_zoom * 0.8 if self._preview_zoom not in (0.0, 1.0) else 0.8))
        self.btn_zoom_in.clicked.connect(lambda: self.set_zoom(self._preview_zoom * 1.25 if self._preview_zoom not in (0.0, 1.0) else 1.25))
        self.btn_zoom_fit.clicked.connect(lambda: self.set_zoom(1.0))
        self.btn_zoom_100.clicked.connect(lambda: self.set_zoom(0.0))

        # Preview label
        self.preview = QLabel(self)
        self.preview.setMinimumSize(900, 520)
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setStyleSheet("background:#111; border:1px solid #333;")
        self.preview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        outer.addWidget(self.preview)

        self._last_rgb8 = None

    def set_zoom(self, z: float):
        if z < 0.05 and z != 0.0:
            z = 0.05
        if z > 8.0:
            z = 8.0
        self._preview_zoom = float(z)
        if self._last_rgb8 is not None:
            self.set_frame_u8(self._last_rgb8)

    def _fit_scaled_size(self, img_w: int, img_h: int) -> tuple[int, int]:
        pw = max(1, self.preview.width())
        ph = max(1, self.preview.height())
        s = min(pw / float(img_w), ph / float(img_h))
        return int(round(img_w * s)), int(round(img_h * s))

    def set_frame_u8(self, rgb8: np.ndarray):
        """rgb8 must be uint8 RGB (H,W,3)."""
        self._last_rgb8 = rgb8

        qimg = QImage(rgb8.data, rgb8.shape[1], rgb8.shape[0], rgb8.strides[0], QImage.Format.Format_RGB888)
        base = QPixmap.fromImage(qimg)

        if self._preview_zoom == 1.0:
            pix = base.scaled(
                self.preview.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.preview.setPixmap(pix)
            return

        if self._preview_zoom == 0.0:
            self.preview.setPixmap(base)
            return

        fit_w, fit_h = self._fit_scaled_size(rgb8.shape[1], rgb8.shape[0])
        target_w = max(1, int(round(fit_w * self._preview_zoom)))
        target_h = max(1, int(round(fit_h * self._preview_zoom)))

        pix = base.scaled(
            QSize(target_w, target_h),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.preview.setPixmap(pix)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self._last_rgb8 is not None:
            self.set_frame_u8(self._last_rgb8)
