# src/setiastro/saspro/magnitude_regions.py
# SASpro — WCS-locked measurement regions + multi-background verification
# Copyright (c) Franklin Marek / www.setiastro.com
#
# Regions are stored in SKY coordinates (RA/Dec) so the same physical patch of
# sky can be re-measured on any future, independently plate-solved stack of the
# same object. Also finds the best background box per image quadrant for
# background-placement robustness checks.

from __future__ import annotations

import os
import json
import time
import math
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any

import numpy as np


# ---------------------------------------------------------------- geometry ----
def densify_polygon(verts, per_edge: int = 16) -> np.ndarray:
    """Sample straight segments between consecutive (closed) vertices.
    A box becomes ~64 verts; this preserves the shape faithfully after the
    pixel->sky->pixel round-trip even when the two images differ in rotation
    or carry SIP distortion (a straight pixel edge is not straight on the sky)."""
    v = np.asarray(verts, dtype=np.float64)
    if v.ndim != 2 or v.shape[0] < 2:
        return v
    per_edge = max(1, int(per_edge))
    n = len(v)
    out = []
    for i in range(n):
        p0 = v[i]; p1 = v[(i + 1) % n]
        ts = np.linspace(0.0, 1.0, per_edge, endpoint=False)
        out.append(p0[None, :] * (1.0 - ts)[:, None] + p1[None, :] * ts[:, None])
    return np.concatenate(out, axis=0)


def _celestial(wcs):
    return wcs.celestial if hasattr(wcs, "celestial") else wcs


def pixel_verts_to_sky(verts_px, wcs) -> np.ndarray:
    w = _celestial(wcs)
    v = np.asarray(verts_px, dtype=np.float64)
    return np.asarray(w.all_pix2world(v, 0), dtype=np.float64)  # Nx2 [ra,dec], origin 0


def sky_verts_to_pixel(sky_verts, wcs) -> np.ndarray:
    w = _celestial(wcs)
    s = np.asarray(sky_verts, dtype=np.float64)
    return np.asarray(w.all_world2pix(s, 0), dtype=np.float64)  # Nx2 [x,y], origin 0


def polygon_to_mask(verts_px, H: int, W: int) -> np.ndarray:
    """Rasterize a pixel-space polygon to an HxW bool mask (bbox-limited).
    Tests pixel centers at integer coords to match astropy origin-0."""
    from matplotlib.path import Path as _Path
    v = np.asarray(verts_px, dtype=np.float64)
    mask = np.zeros((H, W), dtype=bool)
    if v.shape[0] < 3:
        return mask
    xmin = int(max(0, math.floor(v[:, 0].min())))
    xmax = int(min(W - 1, math.ceil(v[:, 0].max())))
    ymin = int(max(0, math.floor(v[:, 1].min())))
    ymax = int(min(H - 1, math.ceil(v[:, 1].max())))
    if xmax < xmin or ymax < ymin:
        return mask
    xs = np.arange(xmin, xmax + 1)
    ys = np.arange(ymin, ymax + 1)
    gx, gy = np.meshgrid(xs, ys)
    inside = _Path(v).contains_points(np.column_stack([gx.ravel(), gy.ravel()]))
    mask[ymin:ymax + 1, xmin:xmax + 1] = inside.reshape(gy.shape)
    return mask


def mask_centroid_px(mask):
    ys, xs = np.nonzero(np.asarray(mask, dtype=bool))
    if xs.size == 0:
        return None
    return (float(xs.mean()), float(ys.mean()))


# ------------------------------------------------------------------ region ----
@dataclass
class SkyRegion:
    name: str
    kind: str = "freehand"                       # box | ellipse | freehand
    sky_verts: List[List[float]] = field(default_factory=list)  # [[ra,dec],...]
    ra_center: Optional[float] = None
    dec_center: Optional[float] = None
    object_name: str = ""
    pixscale_ref: Optional[float] = None
    created: str = ""
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SkyRegion":
        def _f(x):
            return None if x is None else float(x)
        return cls(
            name=str(d.get("name", "")),
            kind=str(d.get("kind", "freehand")),
            sky_verts=[[float(a), float(b)] for a, b in (d.get("sky_verts") or [])],
            ra_center=_f(d.get("ra_center")),
            dec_center=_f(d.get("dec_center")),
            object_name=str(d.get("object_name", "")),
            pixscale_ref=_f(d.get("pixscale_ref")),
            created=str(d.get("created", "")),
            notes=str(d.get("notes", "")),
        )


def build_sky_region(name, kind, verts_px, mask, wcs,
                     object_name="", pixscale=None, notes="") -> SkyRegion:
    sky = pixel_verts_to_sky(verts_px, wcs)
    ra_c = dec_c = None
    cen = mask_centroid_px(mask)
    if cen is not None:
        c = pixel_verts_to_sky(np.asarray([cen], dtype=np.float64), wcs)
        ra_c, dec_c = float(c[0, 0]), float(c[0, 1])
    return SkyRegion(
        name=name, kind=str(kind),
        sky_verts=[[float(a), float(b)] for a, b in sky],
        ra_center=ra_c, dec_center=dec_c,
        object_name=str(object_name or ""),
        pixscale_ref=(float(pixscale) if pixscale else None),
        created=time.strftime("%Y-%m-%dT%H:%M:%S"),
        notes=str(notes or ""),
    )


def region_to_mask(region: SkyRegion, H: int, W: int, wcs):
    px = sky_verts_to_pixel(region.sky_verts, wcs)
    return polygon_to_mask(px, H, W), px


# ----------------------------------------------------------------- library ----
def default_library_path() -> str:
    base = ""
    try:
        from PyQt6.QtCore import QStandardPaths
        base = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppConfigLocation) \
            or QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppDataLocation)
    except Exception:
        base = ""
    if not base:
        base = os.path.join(os.path.expanduser("~"), ".saspro")
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, "magnitude_regions.json")


class RegionLibrary:
    def __init__(self, path: Optional[str] = None):
        self.path = path or default_library_path()
        self._regions: Dict[str, SkyRegion] = {}
        self.load()

    def load(self) -> "RegionLibrary":
        self._regions = {}
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for d in (data.get("regions") or []):
                r = SkyRegion.from_dict(d)
                if r.name:
                    self._regions[r.name] = r
        except FileNotFoundError:
            pass
        except Exception:
            pass
        return self

    def save(self):
        tmp = self.path + ".tmp"
        payload = {"version": 1, "regions": [r.to_dict() for r in self._regions.values()]}
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, self.path)  # atomic

    def names(self) -> List[str]:
        return sorted(self._regions.keys())

    def get(self, name) -> Optional[SkyRegion]:
        return self._regions.get(name)

    def add(self, region: SkyRegion, overwrite: bool = True):
        if (region.name in self._regions) and not overwrite:
            raise KeyError(f"Region '{region.name}' already exists.")
        self._regions[region.name] = region
        self.save()

    def remove(self, name):
        if name in self._regions:
            del self._regions[name]
            self.save()


# ------------------------------------------------- quadrant backgrounds -------
def find_quadrant_backgrounds(img_f, box, margin, auto_rect_box):
    """Best background box in each of the 4 image quadrants.
    Returns [(label, x, y, w, h), ...] in full-image pixel coords."""
    a = np.asarray(img_f)
    H, W = a.shape[:2]
    if a.ndim == 2:
        rgb = np.dstack([a, a, a])
    elif a.ndim == 3 and a.shape[2] >= 3:
        rgb = a[..., :3]
    else:
        rgb = np.dstack([a[..., 0]] * 3)

    hh, ww = H // 2, W // 2
    quads = [
        ("TL", 0,  0,  ww,     hh),
        ("TR", ww, 0,  W - ww, hh),
        ("BL", 0,  hh, ww,     H - hh),
        ("BR", ww, hh, W - ww, H - hh),
    ]
    out = []
    for label, qx, qy, qw, qh in quads:
        crop = rgb[qy:qy + qh, qx:qx + qw, :]
        m = int(max(0, min(int(margin), min(qw, qh) // 2 - int(box) - 1)))
        try:
            bx, by, bw, bh = auto_rect_box(crop, box=int(box), margin=m)
            out.append((label, int(qx + bx), int(qy + by), int(bw), int(bh)))
        except Exception:
            continue
    return out


# ------------------------------------------------------- µ for one background -
def _mad_sigma(vals):
    v = np.asarray(vals, dtype=np.float64)
    if v.size == 0:
        return None
    med = np.median(v, axis=0)
    mad = np.median(np.abs(v - med), axis=0)
    return 1.4826 * mad


def measure_mu(img_f, obj_mask, bg_mask, pixscale, zp_state, sys_floor, mode):
    """µ (and total 3σ) for a given object/background pair.
    Mirrors MagnitudeToolDialog.measure_object_region() math exactly so the
    'Primary' entry reproduces the headline number.
      mono -> {"mu": float|None, "mu_3": float|None}
      rgb  -> {"mu": (R,G,B), "mu_3": (R,G,B)}"""
    a = np.asarray(img_f, dtype=np.float64)
    om = np.asarray(obj_mask, dtype=bool)
    bm = np.asarray(bg_mask, dtype=bool)
    obj_area = int(np.count_nonzero(om))
    bkg_area = int(np.count_nonzero(bm))
    if obj_area <= 0 or bkg_area <= 0 or not (pixscale and pixscale > 0):
        return None
    area_asec2 = float(obj_area) * float(pixscale) ** 2
    scale = float(obj_area) / max(1.0, float(bkg_area))
    LOGC = 2.5 / math.log(10.0)
    sf = float(sys_floor or 0.0)

    def _mu(net, flux_err, ZP, zp_sem):
        if ZP is None or not (net > 0):
            return None, None
        mu = -2.5 * math.log10(net / area_asec2) + float(ZP)
        stat = None
        if zp_sem is not None and flux_err is not None:
            stat = math.sqrt(float(zp_sem) ** 2 + (LOGC * (float(flux_err) / float(net))) ** 2)
        s = 0.0 if stat is None else stat
        tot = 3.0 * math.sqrt(s * s + sf * sf) if (stat is not None or sf > 0) else None
        return mu, tot

    if mode == "mono" or a.ndim == 2:
        net = float(np.sum(a[om])) - float(np.sum(a[bm])) * scale
        sig = _mad_sigma(a[bm])
        fe = (float(sig) * math.sqrt(obj_area)) if sig is not None else None
        mu, mu3 = _mu(net, fe, zp_state.get("ZP"), zp_state.get("zp_sem"))
        return {"mu": mu, "mu_3": mu3}

    a3 = a[..., :3]
    sig = _mad_sigma(a3[bm].reshape(-1, 3))
    muv, mu3v = [], []
    for c, (zk, sk) in enumerate([("ZP_R", "sem_R"), ("ZP_G", "sem_G"), ("ZP_B", "sem_B")]):
        net = float(np.sum(a3[..., c][om])) - float(np.sum(a3[..., c][bm])) * scale
        fe = (float(sig[c]) * math.sqrt(obj_area)) if sig is not None else None
        mu, mu3 = _mu(net, fe, zp_state.get(zk), zp_state.get(sk))
        muv.append(mu); mu3v.append(mu3)
    return {"mu": tuple(muv), "mu_3": tuple(mu3v)}