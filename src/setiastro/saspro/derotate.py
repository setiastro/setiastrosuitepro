#src/setiastro/saspro/derotate.py
from __future__ import annotations
import numpy as np
import cv2

def _build_lonlat_grids(
    h: int,
    w: int,
    cx: float,
    cy: float,
    r: float,
    pole_angle_rad: float,
    subobs_lat_rad: float = 0.0,
):
    """
    Precompute lon/lat and visibility mask for an orthographic projected sphere.

    The mapping supports:
      - pole_angle_rad: rotate image coords around (cx,cy) so the planet spin axis is "up"
      - subobs_lat_rad: sub-observer latitude (tilt of pole toward/away the viewer).
        subobs_lat_rad=0 -> equator-on. Positive -> north pole tilted toward observer.

    Notes:
      - Image y is down; we stay consistent with that convention.
      - Output lon/lat are in the planet/body frame.
    """
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = (xx - cx) / r
    dy = (yy - cy) / r

    # Rotate image coords into planet "axis-up" frame
    ca = float(np.cos(pole_angle_rad))
    sa = float(np.sin(pole_angle_rad))
    dxp = ca * dx - sa * dy
    dyp = sa * dx + ca * dy

    rr2 = dxp * dxp + dyp * dyp
    vis = rr2 <= 1.0

    # Camera-frame z (viewer-facing)
    z0 = np.zeros_like(dxp, dtype=np.float32)
    z0[vis] = np.sqrt(np.maximum(0.0, 1.0 - rr2[vis]))

    # Undo the "view tilt" so we can compute body-frame lon/lat.
    # If the body was rotated by -phi about X before projection, then unproject must rotate by +phi.
    phi = float(subobs_lat_rad)
    cphi = float(np.cos(phi))
    sphi = float(np.sin(phi))

    # camera frame: (x0,y0,z0) = (dxp,dyp,z0)
    # body frame: rotate around X by +phi
    x1 = dxp
    y1 = cphi * dyp + sphi * z0
    z1 = -sphi * dyp + cphi * z0

    lon = np.zeros_like(dxp, dtype=np.float32)
    lat = np.zeros_like(dxp, dtype=np.float32)
    lon[vis] = np.arctan2(x1[vis], z1[vis])
    lat[vis] = np.arcsin(np.clip(y1[vis], -1.0, 1.0))

    return lon, lat, vis


def derotate_stack_lonshift(
    img01: np.ndarray,
    *,
    cx: float,
    cy: float,
    r: float,
    dlon_rad: float,
    pole_angle_rad: float = 0.0,
    subobs_lat_rad: float = 0.0,
    border_value: float = 0.0,
    precomp=None,
) -> np.ndarray:

    """
    Derotate a single stack by shifting longitude on a sphere, accounting for pole tilt.

    pole_angle_rad rotates image coordinates into the planet frame (pole up).
    """
    img = np.asarray(img01, dtype=np.float32)
    h, w = img.shape[:2]

    if precomp is None:
        lon, lat, vis = _build_lonlat_grids(h, w, cx, cy, r, pole_angle_rad, subobs_lat_rad)
    else:
        lon, lat, vis = precomp

    lon2 = lon + float(dlon_rad)

    # ---- body frame point after lon shift ----
    clat = np.cos(lat)
    xb = clat * np.sin(lon2)
    yb = np.sin(lat)
    zb = clat * np.cos(lon2)

    # ---- apply view tilt (body -> camera) ----
    phi = float(subobs_lat_rad)
    cphi = float(np.cos(phi))
    sphi = float(np.sin(phi))

    # rotate around X by -phi
    xc = xb
    yc = cphi * yb - sphi * zb
    zc = sphi * yb + cphi * zb

    # ---- rotate back from planet "axis-up" frame to image frame ----
    ca = float(np.cos(pole_angle_rad))
    sa = float(np.sin(pole_angle_rad))

    x = ca * xc + sa * yc
    y = -sa * xc + ca * yc

    map_x = (x * r + cx).astype(np.float32)
    map_y = (y * r + cy).astype(np.float32)

    valid = vis & (zc >= 0.0)


    def _remap_plane(plane: np.ndarray) -> np.ndarray:
        out = cv2.remap(
            plane, map_x, map_y,
            interpolation=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=float(border_value),
        )
        out[~valid] = float(border_value)
        return out

    if img.ndim == 2:
        return _remap_plane(img)

    out = np.empty_like(img)
    for c in range(img.shape[2]):
        out[..., c] = _remap_plane(img[..., c])
    return out
