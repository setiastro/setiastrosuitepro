# src/setiastro/saspro/lunar_solver.py
# Copyright (c) Franklin Marek / SetiAstro (www.setiastro.com). GPLv3.
"""
Lunar Solver: find where an image sits on the Moon.

A star plate solve matches the image to a catalogue. This matches it to a
picture of what the Moon looked like at that moment: the ephemeris gives the
libration and the Sun's position, and the Moon is rendered from NASA's LOLA
laser-altimeter terrain (shading and cast shadows) and the Clementine 750 nm
albedo mosaic. The image is then placed on that render and a perspective model
of the Moon is fitted to hundreds of surface matches:

    pixel = c + R * Rot(theta) * diag(parity, 1) * (X, -Y)
    (X, Y, Z) = Rx(dlx) Ry(dly) B p,   then X, Y *= sqrt(d^2 - 1) / (d - Z)

B takes Moon body-frame (Mean Earth / polar axis) vectors to the sky (x west,
y north, z toward the observer); p is the surface point scaled by (1 + h/R)
from LOLA; d is the observer distance in lunar radii. Orthographic projection
is not good enough: it leaves up to 0.23 % of the radius as error.

Three ways in, picked automatically:
  * full disc / limb in frame, date known: lit-limb circle, then rotation and
    mirror by correlating the image with the render around the disc centre;
  * close-up, date known: SIFT against the render (both mirror states, three
    scales), then the same refinement;
  * no date: the Sun angle is scanned, libration is fitted from how the
    surface is foreshortened, then the same refinement.

Nothing here touches Qt, so solve_moon() is safe in a QThread worker. A solve
that fails validation raises LunarSolveError: a non-solution is never returned.

Geometry, the LOLA cache and the camera model come from lunar_relief.py.
"""
from __future__ import annotations

import contextlib
import datetime as _dt
import json
import math
import os
import re
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from dataclasses import replace as _dc_replace
from pathlib import Path
from typing import Callable, Optional

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

from numba import njit, prange

from setiastro.saspro.lunar_relief import (
    GnomonicCamera, LolaDEM, LunarGeometry, R_MOON, body_to_latlon, lunar_geometry,
)

__all__ = [
    "LunarSolveError", "LunarSolveCancelled", "LunarSolution",
    "solve_moon", "ensure_data", "data_ready", "capture_time", "load_features",
    "visible_features", "feature_outlines", "LANDING_SITES", "visible_sites",
    "detailed_maps_status", "detailed_maps_ready", "download_detailed_maps", "DETAILED_MAPS_MB",
    "solution_from_metadata", "store_solution_in_metadata", "lunar_solution_after_geometry",
]

# --------------------------------------------------------------------------
# errors
# --------------------------------------------------------------------------


class LunarSolveError(RuntimeError):
    """The image could not be placed on the Moon. The message is for the user."""


class LunarSolveCancelled(LunarSolveError):
    pass


def _check(cancel):
    if cancel is not None and cancel():
        raise LunarSolveCancelled("Cancelled.")


def _report(progress, frac, msg):
    if progress is not None:
        try:
            progress(float(frac), str(msg))
        except Exception:
            pass


# --------------------------------------------------------------------------
# reference data: LOLA (via lunar_relief.LolaDEM) + Clementine 750 nm albedo
# --------------------------------------------------------------------------

PPD = 16
_ALBEDO_WMS = ("https://planetarymaps.usgs.gov/cgi-bin/mapserv?map=/maps/earth/moon_simp_cyl.map"
               "&service=WMS&version=1.1.1&request=GetMap&styles=&srs=EPSG:4326"
               "&width=2880&height=2880&format=image/png&layers=uv_v2&bbox={bbox}")
# east half first: the stitched map then runs lon 0..360 east, like LOLA
_ALBEDO_TILES = (("clementine750_16ppd_e.png", "0,-90,180,90"),
                 ("clementine750_16ppd_w.png", "-180,-90,0,90"))
_UA = {"User-Agent": "SetiAstroSuitePro/1.0"}


def _data_dir(data_dir=None) -> Path:
    return Path(data_dir) if data_dir else LolaDEM.default_dir()


def data_ready(data_dir=None) -> bool:
    d = _data_dir(data_dir)
    return LolaDEM(d).has_global16() and all((d / n).is_file() for n, _ in _ALBEDO_TILES)


def _fetch_png(url, dest: Path, cancel=None, attempts=4, shape=(2880, 2880)):
    from setiastro.saspro.lunar_relief import is_network_error, network_reason
    req = urllib.request.Request(url, headers=_UA)
    tmp = dest.with_name(dest.name + ".part")
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as f:
                while True:
                    _check(cancel)
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            break
        except LunarSolveError:
            tmp.unlink(missing_ok=True)  # cancelled: no half file left behind
            raise
        except Exception as ex:          # DNS hiccups, resets, timeouts, 5xx: wait and retry
            if not is_network_error(ex):
                raise
            if attempt == attempts - 1:
                tmp.unlink(missing_ok=True)
                raise LunarSolveError("The albedo map could not be downloaded: %s. Check the internet "
                                      "connection and try again." % network_reason(ex))
            for _ in range(10 * (1, 3, 6)[min(attempt, 2)]):
                _check(cancel)
                time.sleep(0.1)
    img = cv2.imread(str(tmp), cv2.IMREAD_GRAYSCALE)
    if img is None or img.shape != tuple(shape):
        tmp.unlink(missing_ok=True)
        raise LunarSolveError("The albedo map download from the USGS map server was not a valid image. "
                              "Try again later.")
    tmp.replace(dest)


def ensure_data(progress=None, cancel=None, data_dir=None) -> Path:
    """Download the reference maps once (about 50 MB): LOLA LDEM_16 terrain
    (33 MB, PDS Geosciences) and the Clementine 750 nm albedo (USGS)."""
    d = _data_dir(data_dir)
    d.mkdir(parents=True, exist_ok=True)
    dem = LolaDEM(d)
    try:
        if not dem.has_global16():
            def _p(got, total):
                _report(progress, 0.8 * got / max(total, 1), "Downloading LOLA terrain (%d of %d MB)" % (
                    got >> 20, total >> 20))
            dem.global16(progress=_p, cancel=cancel)
        for i, (name, bbox) in enumerate(_ALBEDO_TILES):
            if not (d / name).is_file():
                _report(progress, 0.8 + 0.1 * i, "Downloading the albedo map (%d of 2)" % (i + 1))
                _fetch_png(_ALBEDO_WMS.format(bbox=bbox), d / name, cancel=cancel)
    except LunarSolveError:
        raise
    except RuntimeError as ex:
        if "cancel" in str(ex).lower():
            raise LunarSolveCancelled("Cancelled.")
        raise LunarSolveError("Reference data download failed: %s" % ex)
    except OSError as ex:
        from setiastro.saspro.lunar_relief import is_network_error, network_reason
        raise LunarSolveError("Reference data download failed: %s" % (
            network_reason(ex) + ". Check the internet connection and try again" if is_network_error(ex) else ex))
    return d


class _Maps:
    """Height (km over the 1737.4 km sphere), albedo and height gradients, all
    2880 x 5760, lon 0..360 east, lat +90 in row 0, pixel centres at (i+0.5)/16."""
    ppd, row0, col0 = float(PPD), 0.0, 0.0       # the whole globe (see _bil)

    @property
    def geo(self):
        return float(self.ppd), float(self.row0), float(self.col0)

    def __init__(self, d: Path):
        dem = LolaDEM(d).global16()
        self.h = (dem.data.astype(np.float32) * np.float32(0.5e-3))
        tiles = [cv2.imread(str(d / n), cv2.IMREAD_GRAYSCALE) for n, _ in _ALBEDO_TILES]
        if any(t is None for t in tiles):
            raise LunarSolveError("The albedo map in %s is unreadable. Delete it to download it again." % d)
        a8 = np.hstack(tiles)
        # Clementine has small no-data gaps (0): fill them from their neighbours
        hole = cv2.dilate((a8 < 3).astype(np.uint8), np.ones((3, 3), np.uint8))
        a8 = cv2.inpaint(a8, hole, 5, cv2.INPAINT_TELEA)
        self.alb = a8.astype(np.float32) / np.float32(255.0)
        lat = 90.0 - (np.arange(2880) + 0.5) / PPD
        dy = R_MOON * math.radians(1.0 / PPD)
        dx = dy * np.cos(np.radians(lat))[:, None]
        hp = np.pad(self.h, ((1, 1), (0, 0)), mode="edge")
        self.gn = ((hp[:-2] - hp[2:]) / (2 * dy)).astype(np.float32)
        self.ge = ((np.roll(self.h, -1, 1) - np.roll(self.h, 1, 1)) / (2 * np.maximum(dx, 1e-3))).astype(np.float32)


_MAPS: Optional[_Maps] = None
_MAPS_LOCK = threading.Lock()


def _maps(progress=None, cancel=None, data_dir=None) -> _Maps:
    global _MAPS
    with _MAPS_LOCK:
        if _MAPS is None:
            d = ensure_data(progress, cancel, data_dir)
            _report(progress, 0.9, "Loading the reference maps")
            _MAPS = _Maps(d)
        return _MAPS


# --------------------------------------------------------------------------
# detailed maps: LOLA LDEM_64 + Clementine 750 nm at 64 px/deg (0.47 km/px)
# --------------------------------------------------------------------------
# The 16 px/deg maps hold 1.9 km detail. A small, sharp field that fails on
# them is matched again on 64 px/deg maps of its own area, fetched the first
# time a solve needs them and kept (a 4 deg band: 12 MB of terrain, ~0.5 MB
# of albedo per 45 deg of longitude). download_detailed_maps() fetches the
# whole near side for use without an internet connection.

PPD_HI = 64
_HI_BAND = LolaDEM.CHUNK_DEG            # degrees of latitude per band (the LOLA chunks)
_HI_BLOCK = 45                          # degrees of longitude per albedo block
_HI_NEAR = (0, 1, 2, 5, 6, 7)           # blocks reaching lon -100..100 E: 135..225 E is never in view
_HI_BANDS = 180 // _HI_BAND
_HI_DEM_MB, _HI_ALB_MB = 11.8, 0.6      # one band of terrain, one block of albedo (0.4-0.8 MB)
DETAILED_MAPS_MB = int(round(_HI_BANDS * (_HI_DEM_MB + len(_HI_NEAR) * _HI_ALB_MB)))
_R_HI = 3700                            # reference radius at which 64 px/deg is 1:1
_ALBEDO_WMS_HI = ("https://planetarymaps.usgs.gov/cgi-bin/mapserv?map=/maps/earth/moon_simp_cyl.map"
                  "&service=WMS&version=1.1.1&request=GetMap&styles=&srs=EPSG:4326"
                  "&width=2880&height=256&format=image/png&layers=uv_v2&bbox={bbox}")


@contextlib.contextmanager
def _download_errors(what, after=""):
    """Downloads in plain words: cancel -> LunarSolveCancelled, network
    trouble -> LunarSolveError with the reason and what to do."""
    from setiastro.saspro.lunar_relief import is_network_error, network_reason
    try:
        yield
    except LunarSolveError:
        raise
    except RuntimeError as ex:
        if "cancel" in str(ex).lower():
            raise LunarSolveCancelled("Cancelled.")
        raise LunarSolveError(("%s could not be downloaded: %s. " % (what, ex)) + after)
    except OSError as ex:
        why = network_reason(ex) + ". Check the internet connection" if is_network_error(ex) else str(ex)
        raise LunarSolveError(("%s could not be downloaded: %s. " % (what, why)) + after)


def _hi_albedo_path(d, band, block) -> Path:
    return Path(d) / ("clementine750_64ppd_r%05d_c%05d.png" % (band * _HI_BAND * PPD_HI, block * _HI_BLOCK * PPD_HI))


def _hi_albedo_fetch(d, band, block, cancel=None):
    lat_hi = 90 - band * _HI_BAND
    w = block * _HI_BLOCK
    w = w - 360 if w >= 180 else w      # the map server's longitudes run -180..180
    _fetch_png(_ALBEDO_WMS_HI.format(bbox="%d,%d,%d,%d" % (w, lat_hi - _HI_BAND, w + _HI_BLOCK, lat_hi)),
               _hi_albedo_path(d, band, block), cancel=cancel, shape=(_HI_BAND * PPD_HI, _HI_BLOCK * PPD_HI))


def _hi_missing(d):
    """What download_detailed_maps() still has to fetch, band by band:
    [("dem", band, None) | ("alb", band, block)]."""
    miss = set(LolaDEM(d).missing64())
    out = []
    for b in range(_HI_BANDS):
        if b in miss:
            out.append(("dem", b, None))
        out.extend(("alb", b, k) for k in _HI_NEAR if not _hi_albedo_path(d, b, k).is_file())
    return out


def detailed_maps_status(data_dir=None):
    """(MB on this computer, MB in all) of the near side's detailed maps."""
    d = _data_dir(data_dir)
    if not d.is_dir():
        return 0, DETAILED_MAPS_MB
    left = sum(_HI_DEM_MB if kind == "dem" else _HI_ALB_MB for kind, _, _ in _hi_missing(d))
    return max(0, int(round(DETAILED_MAPS_MB - left))), DETAILED_MAPS_MB


def detailed_maps_ready(data_dir=None) -> bool:
    d = _data_dir(data_dir)
    return d.is_dir() and not _hi_missing(d)


def download_detailed_maps(progress=None, cancel=None, data_dir=None) -> Path:
    """Fetch the near side's detailed maps (about DETAILED_MAPS_MB MB) for
    solving without an internet connection. What is on disk already is kept,
    so an interrupted download picks up where it stopped."""
    d = _data_dir(data_dir)
    d.mkdir(parents=True, exist_ok=True)
    dem = LolaDEM(d)
    todo = _hi_missing(d)
    sizes = [_HI_DEM_MB if kind == "dem" else _HI_ALB_MB for kind, _, _ in todo]
    total = max(sum(sizes), 1e-6)
    done = 0.0
    with _download_errors("The detailed maps", "What was downloaded is kept: download again to continue."):
        for (kind, b, k), mb in zip(todo, sizes):
            _check(cancel)

            def _p(got, tot, _d=done, _mb=mb):
                x = _d + _mb * got / max(tot, 1)
                _report(progress, x / total, "Downloading the detailed maps (%d of %d MB)" % (int(x), int(round(total))))
            _p(0, 1)
            if kind == "dem":
                lat_hi = 90.0 - b * _HI_BAND
                dem.fetch64(lat_hi - _HI_BAND, lat_hi, progress=_p, cancel=cancel)
            else:
                _hi_albedo_fetch(d, b, k, cancel=cancel)
            done += mb
    _report(progress, 1.0, "The detailed maps are on this computer")
    return d


class _RegionMaps:
    """The _Maps fields at 64 px/deg for one window of the Moon (see _bil):
    rows from row0, columns from col0. A window over 180 deg of longitude
    (near a pole) takes the whole band. Missing pieces are downloaded."""
    ppd = float(PPD_HI)
    geo = _Maps.geo

    def __init__(self, d, lat_lo, lat_hi, lon_w, span, maps16, progress=None, cancel=None, frac=0.7):
        n = PPD_HI
        full = 360 * n
        rows_per = _HI_BAND * n
        lat_lo, lat_hi = max(-90.0, float(lat_lo)), min(90.0, float(lat_hi))
        dem = LolaDEM(d)
        bands = dem.chunks64(lat_lo, lat_hi)
        whole = span >= 180.0
        c0, W = (0, full) if whole else (int(math.floor((lon_w % 360.0) * n)), int(math.ceil(span * n)) + 2)
        cols = (c0 + np.arange(W)) % full
        blocks = sorted(set((cols // (_HI_BLOCK * n)).tolist()))
        need = dem.missing64(lat_lo, lat_hi)
        alb_todo = [(b, k) for b in bands for k in blocks if k in _HI_NEAR and not _hi_albedo_path(d, b, k).is_file()]
        mb = max(1, int(round(len(need) * _HI_DEM_MB + len(alb_todo) * _HI_ALB_MB)))
        with _download_errors("The detailed map of this area",
                              "Connect to the internet, or download the detailed maps for use offline in the "
                              "Lunar Solver window."):
            if need:
                dem.fetch64(lat_lo, lat_hi, cancel=cancel, progress=lambda g, t: _report(
                    progress, frac, "Downloading the detailed map of this area (%d of %d MB)" % (g >> 20, mb)))
            for i, (b, k) in enumerate(alb_todo):
                _report(progress, frac, "Downloading the detailed albedo of this area (%d of %d)" % (i + 1, len(alb_todo)))
                _hi_albedo_fetch(d, b, k, cancel=cancel)
        band = dem.band64(lat_lo, lat_hi, cancel=cancel)
        r0, rows = int(band.row0), band.data.shape[0]
        self.h = np.ascontiguousarray(band.data[:, cols]).astype(np.float32) * np.float32(0.5e-3)
        # albedo: only the blocks the window touches; 135..225 E is never in view,
        # so there the 16 px/deg map, enlarged, fills in
        strip = np.zeros((len(bands) * rows_per, full), np.uint8)
        for i, b in enumerate(bands):
            for k in blocks:
                if k in _HI_NEAR:
                    t = cv2.imread(str(_hi_albedo_path(d, b, k)), cv2.IMREAD_GRAYSCALE)
                    if t is None:
                        raise LunarSolveError("A detailed albedo map in %s is unreadable. Delete it to download "
                                              "it again." % d)
                else:
                    a16 = maps16.alb[b * _HI_BAND * PPD:(b + 1) * _HI_BAND * PPD,
                                     k * _HI_BLOCK * PPD:(k + 1) * _HI_BLOCK * PPD]
                    t = np.clip(cv2.resize(a16, (_HI_BLOCK * n, rows_per), interpolation=cv2.INTER_LINEAR) * 255.0,
                                0, 255).astype(np.uint8)
                strip[i * rows_per:(i + 1) * rows_per, k * _HI_BLOCK * n:(k + 1) * _HI_BLOCK * n] = t
        o = r0 - bands[0] * rows_per
        a8 = np.ascontiguousarray(strip[o:o + rows][:, cols])
        hole = cv2.dilate((a8 < 3).astype(np.uint8), np.ones((3, 3), np.uint8))
        if hole.any():
            a8 = cv2.inpaint(a8, hole, 5, cv2.INPAINT_TELEA)
        self.alb = a8.astype(np.float32) / np.float32(255.0)
        lat = 90.0 - (r0 + np.arange(rows) + 0.5) / n
        dy = R_MOON * math.radians(1.0 / n)
        dx = np.maximum(dy * np.cos(np.radians(lat))[:, None], 1e-3)
        hp = np.pad(self.h, ((1, 1), (0, 0)), mode="edge")
        self.gn = ((hp[:-2] - hp[2:]) / (2 * dy)).astype(np.float32)
        if whole:
            self.ge = ((np.roll(self.h, -1, 1) - np.roll(self.h, 1, 1)) / (2 * dx)).astype(np.float32)
        else:
            hx = np.pad(self.h, ((0, 0), (1, 1)), mode="edge")
            self.ge = ((hx[:, 2:] - hx[:, :-2]) / (2 * dx)).astype(np.float32)
        self.row0, self.col0 = float(r0), float(c0)
        self.box = (lat_lo, lat_hi, lon_w % 360.0, float(span), whole)

    def covers(self, lat_lo, lat_hi, lon_w, span):
        a_lo, a_hi, a_w, a_span, whole = self.box
        if lat_lo < a_lo - 1e-9 or lat_hi > a_hi + 1e-9:
            return False
        if whole:
            return True
        return span < 180.0 and ((lon_w - a_w) % 360.0) + span <= a_span + 1e-9


_REGION = None
_REGION_LOCK = threading.Lock()


def _region_maps(data_dir, box, maps16, progress=None, cancel=None, frac=0.7) -> _RegionMaps:
    """64 px/deg maps covering box = (lat_lo, lat_hi, lon_west, lon_span); the
    last area is kept, so the steps of one solve share it."""
    global _REGION
    d = _data_dir(data_dir)
    with _REGION_LOCK:
        if _REGION is not None and _REGION[0] == str(d) and _REGION[1].covers(*box):
            return _REGION[1]
        reg = _RegionMaps(d, *box, maps16=maps16, progress=progress, cancel=cancel, frac=frac)
        _REGION = (str(d), reg)
        return reg


# --------------------------------------------------------------------------
# feature names (IAU, shipped with SASpro)
# --------------------------------------------------------------------------

_FEATURES = None


def load_features():
    """IAU approved lunar names: dict of arrays name, type, lat, lon, diam_km.
    Types: AA crater, SF lettered satellite crater, ME mare, MO mons/montes,
    RI rima, VA vallis, ... (USGS Gazetteer of Planetary Nomenclature codes)."""
    global _FEATURES
    if _FEATURES is None:
        from setiastro.saspro.resources import get_data_path
        p = get_data_path(os.path.join("data", "lunar", "lunar_features.json"))
        with open(p, encoding="utf-8") as f:
            js = json.load(f)
        rows = js["features"]
        _FEATURES = dict(
            name=[r[0] for r in rows],
            type=np.array([r[1] for r in rows]),
            lat=np.array([r[2] for r in rows], np.float64),
            lon=np.array([r[3] for r in rows], np.float64),
            diam=np.array([r[4] for r in rows], np.float64),
        )
    return _FEATURES


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------

# body -> sky at zero libration with the pole up: lon 0 faces Earth and lon +90
# (Mare Crisium's side) lies toward sky west
_B0 = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
_MEAN_DIST_KM = 384400.0


@dataclass
class _View:
    B: np.ndarray          # body -> sky (rows: west, north, toward observer)
    S: np.ndarray          # unit Moon -> Sun, body frame
    D: float               # observer distance, km
    utc: str               # "" when the date is unknown
    topocentric: bool = False

    @property
    def d_lunar(self) -> float:
        return self.D / R_MOON


def _view_from_geometry(G: LunarGeometry) -> _View:
    return _View(B=np.stack([-G.e, G.n, -G.d]), S=np.asarray(G.S, np.float64), D=float(G.D),
                 utc=G.utc, topocentric=bool(getattr(G, "topocentric", False)))


def _rot_small(dx, dy):
    cx_, sx = math.cos(dx), math.sin(dx)
    cy_, sy = math.cos(dy), math.sin(dy)
    Rx = np.array([[1, 0, 0], [0, cx_, -sx], [0, sx, cx_]])
    Ry = np.array([[cy_, 0, sy], [0, 1, 0], [-sy, 0, cy_]])
    return Rx @ Ry


def _latlon_vec(lat_deg, lon_deg):
    la = np.radians(lat_deg); lo = np.radians(lon_deg)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=-1)


def _mean_view(colong, lib=(0.0, 0.0)) -> _View:
    """Geometry with no date: Sun at the given colongitude, libration lib (rad)."""
    return _View(B=_rot_small(lib[0], lib[1]) @ _B0, S=_latlon_vec(0.0, 90.0 - colong),
                 D=_MEAN_DIST_KM, utc="")


def _vec_latlon(v):
    v = np.asarray(v, np.float64)
    lat = np.degrees(np.arcsin(np.clip(v[..., 2] / np.linalg.norm(v, axis=-1), -1, 1)))
    lon = np.degrees(np.arctan2(v[..., 1], v[..., 0]))
    return lat, lon


# --------------------------------------------------------------------------
# numba: bilinear map sampling and the reference renderer
# --------------------------------------------------------------------------

@njit(cache=True, nogil=True)
def _bil(g, lat, lon, ppd, row0, col0):
    """Bilinear sample of a lat/lon map with ppd pixels per degree (lat +90 at
    global row 0, lon 0..360 east from global column 0, pixel centres at
    (i + 0.5) / ppd) whose array starts at global (row0, col0). The whole
    globe wraps in longitude; a regional window holds its edge values."""
    H, W = g.shape
    full = 360.0 * ppd
    x = (lon % 360.0) * ppd - 0.5 - col0
    y = (90.0 - lat) * ppd - 0.5 - row0
    if y < 0.0:
        y = 0.0
    if y > H - 1.0:
        y = H - 1.0
    y0 = int(math.floor(y))
    fy = y - y0
    y1 = y0 + 1 if y0 + 1 < H else H - 1
    if W < full:
        x = x % full
        if x > 0.5 * (full + W):        # west of the window
            x -= full
        if x < 0.0:
            x = 0.0
        if x > W - 1.0:
            x = W - 1.0
        x0i = int(math.floor(x))
        fx = x - x0i
        x1i = x0i + 1 if x0i + 1 < W else W - 1
    else:
        x0 = math.floor(x)
        fx = x - x0
        x0i = int(x0) % W
        x1i = (x0i + 1) % W
    return ((g[y0, x0i] * (1 - fx) + g[y0, x1i] * fx) * (1 - fy)
            + (g[y1, x0i] * (1 - fx) + g[y1, x1i] * fx) * fy)


@njit(parallel=True, cache=True, nogil=True)
def _sample_kernel(g, lat, lon, out, ppd, row0, col0):
    for i in prange(lat.shape[0]):
        out[i] = _bil(g, lat[i], lon[i], ppd, row0, col0)


@njit(parallel=True, cache=True, nogil=True)
def _render_kernel(out, B, s, v, R_px, cx, cy, alb, ge, gn, hh, L, shadows, max_km, steps, r_km, ppd, row0, col0):
    """Orthographic sky view, x = west (right), y = north (up). Lunar-Lambert
    shading on LOLA slopes, times albedo, with cast shadows found by marching
    toward the Sun along the great circle until a ridge blocks it."""
    h, w = out.shape
    for yy in prange(h):
        for xx in range(w):
            u = (xx - cx) / R_px
            vv = -(yy - cy) / R_px
            r2 = u * u + vv * vv
            if r2 >= 1.0:
                continue
            ww = math.sqrt(1.0 - r2)
            px = B[0, 0] * u + B[1, 0] * vv + B[2, 0] * ww
            py = B[0, 1] * u + B[1, 1] * vv + B[2, 1] * ww
            pz = B[0, 2] * u + B[1, 2] * vv + B[2, 2] * ww
            if pz > 1.0:
                pz = 1.0
            if pz < -1.0:
                pz = -1.0
            lat = math.degrees(math.asin(pz))
            lon = math.degrees(math.atan2(py, px))
            A = _bil(alb, lat, lon, ppd, row0, col0)
            gE = _bil(ge, lat, lon, ppd, row0, col0)
            gN = _bil(gn, lat, lon, ppd, row0, col0)
            la = math.radians(lat)
            lo = math.radians(lon)
            sla = math.sin(la)
            cla = math.cos(la)
            slo = math.sin(lo)
            clo = math.cos(lo)
            nx = px + gE * slo + gN * sla * clo
            ny = py - gE * clo + gN * sla * slo
            nz = pz - gN * cla
            nn = math.sqrt(nx * nx + ny * ny + nz * nz)
            nx /= nn
            ny /= nn
            nz /= nn
            ci = nx * s[0] + ny * s[1] + nz * s[2]
            ce = nx * v[0] + ny * v[1] + nz * v[2]
            if ci < 0.0:
                ci = 0.0
            if ce < 1e-3:
                ce = 1e-3
            refl = (1 - L) * ci + 2 * L * ci / (ci + ce + 1e-6)
            if shadows and refl > 0.0:
                sin_el = px * s[0] + py * s[1] + pz * s[2]
                if -0.02 < sin_el < 0.35:
                    tx = s[0] - sin_el * px
                    ty = s[1] - sin_el * py
                    tz = s[2] - sin_el * pz
                    tn = math.sqrt(tx * tx + ty * ty + tz * tz) + 1e-12
                    tx /= tn
                    ty /= tn
                    tz /= tn
                    el = math.asin(sin_el)
                    cel = math.cos(el)
                    sel = math.sin(el)
                    h0 = _bil(hh, lat, lon, ppd, row0, col0)
                    lr = math.log(0.6)
                    stp = (math.log(max_km) - lr) / (steps - 1)
                    for k in range(steps):
                        dd = math.exp(lr + stp * k)
                        th = dd / r_km
                        c = math.cos(th)
                        sn = math.sin(th)
                        qx = c * px + sn * tx
                        qy = c * py + sn * ty
                        qz = c * pz + sn * tz
                        if qz > 1.0:
                            qz = 1.0
                        if qz < -1.0:
                            qz = -1.0
                        hk = _bil(hh, math.degrees(math.asin(qz)), math.degrees(math.atan2(qy, qx)),
                                  ppd, row0, col0)
                        rk = r_km + hk
                        x = rk * sn
                        z = rk * c - (r_km + h0)
                        if z * cel - x * sel > 0.0:
                            refl = 0.0
                            break
            out[yy, xx] = A * refl


def _numba_lock():
    try:
        from setiastro.saspro.legacy.numba_utils import NUMBA_PARALLEL_LOCK
        return NUMBA_PARALLEL_LOCK
    except Exception:  # pragma: no cover
        import contextlib
        return contextlib.nullcontext()


def _render_centre(R_px) -> float:
    return (2 * int(R_px) + 40 - 1) / 2.0


def _render(view: _View, R_px: int, maps: _Maps, roi=None):
    """Reference render: the disc of radius R_px centred in a square of side
    2*R_px + 40. roi = (x0, y0, w, h) renders only that window of the square,
    which is all a close-up needs. Returns (img, mask, disc centre in img)."""
    size = 2 * int(R_px) + 40
    c = _render_centre(R_px)
    x0, y0, w, h = (0, 0, size, size) if roi is None else roi
    cx, cy = c - x0, c - y0
    v = view.B.T @ np.array([0.0, 0.0, 1.0])
    g = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(view.S, v))))))
    L = 1.0 - 0.019 * g + 0.000242 * g * g - 0.00000146 * g ** 3
    out = np.zeros((h, w), np.float32)
    with _numba_lock():
        _render_kernel(out, np.ascontiguousarray(view.B, np.float64), np.asarray(view.S, np.float64),
                       np.asarray(v, np.float64), float(R_px), cx, cy, maps.alb, maps.ge, maps.gn, maps.h,
                       float(L), True, 400.0, 72, float(R_MOON), *maps.geo)
    yy, xx = np.mgrid[0:h, 0:w]
    mask = ((xx - cx) ** 2 + (yy - cy) ** 2) < float(R_px) ** 2
    return out, mask, (cx, cy)


def _sample_h(maps: _Maps, lat, lon):
    lat = np.ascontiguousarray(np.asarray(lat, np.float64).ravel())
    lon = np.ascontiguousarray(np.asarray(lon, np.float64).ravel())
    out = np.empty(lat.shape[0], np.float32)
    with _numba_lock():
        _sample_kernel(maps.h, lat, lon, out, *maps.geo)
    return out


def _with_relief(maps: _Maps, p):
    lat, lon = _vec_latlon(p)
    return p * (1.0 + _sample_h(maps, lat, lon) / R_MOON)[:, None]


# --------------------------------------------------------------------------
# the model
# --------------------------------------------------------------------------

def _project(params, parity, B, P, d):
    """P: (N,3) body vectors in lunar radii (relief included). -> (N,2) px, Z."""
    cx, cy, Ri, th, dlx, dly = params
    q = P @ (_rot_small(dlx, dly) @ B).T
    f = math.sqrt(d * d - 1.0) / (d - q[:, 2])
    X = parity * q[:, 0] * f
    Y = -q[:, 1] * f
    c, s = math.cos(th), math.sin(th)
    return np.stack([cx + Ri * (c * X - s * Y), cy + Ri * (s * X + c * Y)], 1), q[:, 2]


def _body_from_sky(B, X, Y):
    Z = np.sqrt(np.clip(1 - X * X - Y * Y, 0, None))
    return np.stack([X, Y, Z], -1) @ B


def _fit_model(params0, parity, B, P, img_pts, d, fix_libration=False):
    from scipy.optimize import least_squares
    lib0 = np.array(params0[4:6], float)    # fix_libration holds the current correction

    def res(v):
        pr = np.asarray(v) if not fix_libration else np.concatenate([v, lib0])
        q, _ = _project(pr, parity, B, P, d)
        return (q - img_pts).ravel()
    x0 = np.array(params0[:4] if fix_libration else params0, float)
    r = least_squares(res, x0, loss="soft_l1", f_scale=1.5, x_scale="jac")
    v = r.x if not fix_libration else np.concatenate([r.x, lib0])
    q, _ = _project(v, parity, B, P, d)
    return v, np.hypot(*(q - img_pts).T)


# --------------------------------------------------------------------------
# image measurements
# --------------------------------------------------------------------------

def _to_mono01(image) -> np.ndarray:
    a = np.asarray(image)
    if a.ndim == 3:
        a = a[..., :3].astype(np.float32).mean(axis=2) if a.shape[2] >= 3 else a[..., 0].astype(np.float32)
    else:
        a = a.astype(np.float32)
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = np.percentile(a, 0.05), np.percentile(a, 99.95)
    if not hi > lo:
        raise LunarSolveError("The image is blank.")
    return np.clip((a - lo) / (hi - lo), 0, 1).astype(np.float32)


def _highpass(img, sigma):
    """Local-contrast representation: (img - blur) / local std. Brightness
    level and gradients drop out; craters, rilles, shadows and albedo edges stay."""
    b = cv2.GaussianBlur(img, (0, 0), sigma)
    dlt = img - b
    sd = np.sqrt(cv2.GaussianBlur(dlt * dlt, (0, 0), sigma * 2)) + 1e-4
    return dlt / sd


def _fit_limb(mono, iters=3000, tol=2.0):
    """RANSAC circle on the disc/sky boundary. Away from full Moon the boundary is
    limb + terminator; the terminator is an ellipse, so it loses."""
    u8 = (mono * 255).astype(np.uint8)
    lo, hi = np.percentile(u8, 0.5), np.percentile(u8, 99.5)
    if hi - lo < 20:
        return None
    disk = (u8 > lo + 0.25 * (hi - lo)).astype(np.uint8)
    if disk.mean() > 0.995 or disk.mean() < 0.01:
        return None
    k = np.ones((5, 5), np.uint8)
    disk = cv2.morphologyEx(cv2.morphologyEx(disk, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)
    n, lab, st, _ = cv2.connectedComponentsWithStats(disk)
    if n < 2:
        return None
    big = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    disk = (lab == big).astype(np.uint8)
    cnts, _ = cv2.findContours(disk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    c = max(cnts, key=len)[:, 0, :].astype(np.float64)
    H, W = mono.shape
    keep = (c[:, 0] > 3) & (c[:, 0] < W - 4) & (c[:, 1] > 3) & (c[:, 1] < H - 4)
    c = c[keep]
    if len(c) < 50:
        return None
    rng = np.random.RandomState(0)
    best_n, best = 0, None
    for _ in range(iters):
        i = rng.randint(0, len(c), 3)
        if i[0] == i[1] or i[1] == i[2] or i[0] == i[2]:
            continue
        (x1, y1), (x2, y2), (x3, y3) = c[i]
        A = np.array([[x2 - x1, y2 - y1], [x3 - x1, y3 - y1]]) * 2
        det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
        if abs(det) < 1e-6:
            continue
        b = np.array([x2 ** 2 - x1 ** 2 + y2 ** 2 - y1 ** 2, x3 ** 2 - x1 ** 2 + y3 ** 2 - y1 ** 2])
        cx, cy = np.linalg.solve(A, b)
        r = math.hypot(x1 - cx, y1 - cy)
        if r < 20 or r > 20 * max(H, W):
            continue
        inl = np.abs(np.hypot(c[:, 0] - cx, c[:, 1] - cy) - r) < tol
        ni = int(inl.sum())
        if ni > best_n:
            best_n, best = ni, inl
    if best is None or best_n < 40:
        return None
    p = c[best]
    A = np.column_stack([2 * p[:, 0], 2 * p[:, 1], np.ones(len(p))])
    sol = np.linalg.lstsq(A, (p ** 2).sum(1), rcond=None)[0]
    cx, cy = sol[0], sol[1]
    r = math.sqrt(max(sol[2] + cx * cx + cy * cy, 1e-9))
    rms = float(np.sqrt(np.mean((np.hypot(p[:, 0] - cx, p[:, 1] - cy) - r) ** 2)))
    return dict(cx=float(cx), cy=float(cy), R=float(r), rms=rms, n=len(p))


def _coarse_rotation(mono, limb, view, maps, Rr=300):
    """Rotation + mirror by circular correlation of polar unwraps (image vs render)."""
    s = Rr / limb["R"]
    small = cv2.resize(mono, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    cx, cy = limb["cx"] * s, limb["cy"] * s
    ref, _, rc = _render(view, Rr, maps)
    nang, nrad = 1440, 240
    Pr = cv2.warpPolar(_highpass(ref, 4.0), (nrad, nang), rc, Rr, cv2.WARP_POLAR_LINEAR + cv2.INTER_AREA)[:, 24:228]
    Fr = np.fft.fft(np.nan_to_num(Pr), axis=0)
    out = []
    for parity in (1, -1):
        im = small if parity == 1 else cv2.flip(small, 1)
        icx = cx if parity == 1 else (small.shape[1] - 1) - cx
        Pi = cv2.warpPolar(_highpass(im, 4.0), (nrad, nang), (icx, cy), Rr,
                           cv2.WARP_POLAR_LINEAR + cv2.INTER_AREA)[:, 24:228]
        Pi = np.nan_to_num(Pi)
        corr = np.real(np.fft.ifft(np.fft.fft(Pi, axis=0) * np.conj(Fr), axis=0)).sum(1)
        corr /= (np.linalg.norm(Pi) * np.linalg.norm(Pr) + 1e-9)
        k = int(np.argmax(corr))
        far = np.delete(corr, np.arange(k - 20, k + 21) % len(corr))
        out.append(dict(parity=parity, ang=k * 360.0 / nang, score=float(corr[k]), runner=float(far.max())))
    best = max(out, key=lambda o: o["score"])
    other = [o for o in out if o is not best][0]
    th = math.radians(best["ang"]) * (1 if best["parity"] == 1 else -1)
    return dict(theta=th, parity=best["parity"], score=best["score"],
                margin=best["score"] / max(best["runner"], other["score"], 1e-3))


def _ref_to_img_affine(params, parity, Rr, crr):
    cx, cy, Ri, th = params[:4]
    c, s = math.cos(th), math.sin(th)
    M = (Ri / Rr) * np.array([[c, -s], [s, c]]) @ np.diag([parity, 1.0])
    t = np.array([cx, cy]) - M @ np.array(crr)
    return np.hstack([M, t[:, None]])


def _footprint_roi(params, parity, Rr, img_shape, pad=0.15):
    """The window of an Rr reference render that the image covers, padded for
    what a similarity ignores (libration correction, perspective) and for the
    patch search. -> (x0, y0, w, h), or None for the whole render."""
    size = 2 * int(Rr) + 40
    c = _render_centre(Rr)
    Ai = cv2.invertAffineTransform(_ref_to_img_affine(params, parity, Rr, (c, c)))
    H, W = img_shape
    gx, gy = np.meshgrid(np.linspace(0, W - 1, 9), np.linspace(0, H - 1, 9))
    pts = np.column_stack([gx.ravel(), gy.ravel(), np.ones(gx.size)]) @ Ai.T
    lo, hi = pts.min(0), pts.max(0)
    m = pad * float((hi - lo).max()) + 96.0
    x0, y0 = max(0, int(math.floor(lo[0] - m))), max(0, int(math.floor(lo[1] - m)))
    x1, y1 = min(size, int(math.ceil(hi[0] + m))), min(size, int(math.ceil(hi[1] + m)))
    if x1 - x0 < 128 or y1 - y0 < 128:
        return None
    return x0, y0, x1 - x0, y1 - y0


def _model_warp(params, parity, B, d, Rr, crr, ref_shape, maps, ks, step=16):
    """Where each reference-render pixel lands in the (1/ks scaled) image under
    the full model: libration correction, perspective and relief. Computed at
    block centres and interpolated, which cv2.resize lines up exactly.
    -> (map_x, map_y) for cv2.remap."""
    H, W = ref_shape
    nx, ny = -(-W // step), -(-H // step)
    xx, yy = np.meshgrid((np.arange(nx) + 0.5) * step - 0.5, (np.arange(ny) + 0.5) * step - 0.5)
    X = (xx.ravel() - crr[0]) / Rr
    Y = -(yy.ravel() - crr[1]) / Rr
    r = np.hypot(X, Y)
    pin = np.where(r > 0.999, 0.999 / np.maximum(r, 1e-12), 1.0)     # off the disc: hold at the limb
    uv, _ = _project(params, parity, B, _with_relief(maps, _body_from_sky(B, X * pin, Y * pin)), d)
    out = []
    for k in (0, 1):
        g = (uv[:, k] / ks).reshape(ny, nx).astype(np.float32)
        out.append(np.ascontiguousarray(
            cv2.resize(g, (nx * step, ny * step), interpolation=cv2.INTER_LINEAR)[:H, :W]))
    return out[0], out[1]


def _grid_matches(mono, ref, ref_mask, A, patch=48, search=20, step=24, sigma=3.0, min_score=0.35):
    """Warp the image into the reference frame, then template-match a grid of
    bandpassed patches inside the disc. A is a 2x3 affine (ref -> img) or a
    (map_x, map_y) pair from _model_warp."""
    H, W = ref.shape
    if isinstance(A, tuple):
        mx, my = A
        warped = cv2.remap(mono, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        h_, w_ = mono.shape
        valid = (mx >= 0) & (mx <= w_ - 1) & (my >= 0) & (my <= h_ - 1)
    else:
        warped = cv2.warpAffine(mono, A, (W, H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        valid = cv2.warpAffine(np.ones_like(mono), A, (W, H), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0.5
    hw = _highpass(warped, sigma)
    hr = _highpass(ref, sigma)
    m = cv2.erode((ref_mask & valid).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    src, dst, sc = [], [], []
    h2 = patch // 2
    for y in range(h2 + search, H - h2 - search, step):
        for x in range(h2 + search, W - h2 - search, step):
            if not m[y - h2:y + h2, x - h2:x + h2].all():
                continue
            if not ref_mask[y - h2 - search:y + h2 + search, x - h2 - search:x + h2 + search].all():
                continue
            tpl = hw[y - h2:y + h2, x - h2:x + h2]
            if tpl.std() < 0.3:
                continue
            win = hr[y - h2 - search:y + h2 + search, x - h2 - search:x + h2 + search]
            res = cv2.matchTemplate(win, tpl, cv2.TM_CCOEFF_NORMED)
            _, mx, _, loc = cv2.minMaxLoc(res)
            if mx < min_score:
                continue
            px, py = loc
            ox = oy = 0.0
            if 0 < px < res.shape[1] - 1:
                a, b, c = res[py, px - 1], res[py, px], res[py, px + 1]
                den = a - 2 * b + c
                ox = 0.5 * (a - c) / den if den != 0 else 0.0
            if 0 < py < res.shape[0] - 1:
                a, b, c = res[py - 1, px], res[py, px], res[py + 1, px]
                den = a - 2 * b + c
                oy = 0.5 * (a - c) / den if den != 0 else 0.0
            src.append((x, y))
            dst.append((x + px + ox - search, y + py + oy - search))
            sc.append(mx)
    return np.array(src, float).reshape(-1, 2), np.array(dst, float).reshape(-1, 2), np.array(sc)


def _refine(mono, params, parity, view, ref, rmask, crr, Rr, maps, *, passes=3, search0=20,
            free_lib=None, min_score=0.35, cancel=None):
    """Patch matching against the render + perspective model fit. Returns
    (params, residuals) or (params, None) when too few patches match."""
    B = view.B
    d = view.d_lunar
    k = params[2] / Rr
    img_s = mono if k <= 1.2 else cv2.resize(mono, None, fx=1 / k, fy=1 / k, interpolation=cv2.INTER_AREA)
    ks = mono.shape[1] / img_s.shape[1]
    res = None
    for it in range(passes):
        _check(cancel)
        # the full model, not a similarity: once a pass has measured a libration
        # correction, a flat warp is off by R*sin(correction) (100 px for 1.4 deg
        # at R 3900) and the next pass's +-8 px search finds nothing
        mx, my = _model_warp(params, parity, B, d, Rr, crr, ref.shape, maps, ks)
        src, dst, _ = _grid_matches(img_s, ref, rmask, (mx, my), search=search0 if it == 0 else 8,
                                    min_score=min_score)
        if len(src) < 12:
            return params, None
        si = src.astype(np.intp)
        img_pts = np.column_stack([mx[si[:, 1], si[:, 0]], my[si[:, 1], si[:, 0]]]).astype(np.float64) * ks
        X = (dst[:, 0] - crr[0]) / Rr
        Y = -(dst[:, 1] - crr[1]) / Rr
        ok = X * X + Y * Y < 0.995
        if ok.sum() < 12:
            return params, None
        P = _with_relief(maps, _body_from_sky(B, X[ok], Y[ok]))
        if free_lib is None:
            fix = (len(src) < 60) or (np.ptp(X[ok]) < 0.3 and np.ptp(Y[ok]) < 0.3)
        else:
            fix = not free_lib
        params, res = _fit_model(params, parity, B, P, img_pts[ok], d, fix_libration=fix)
        inl = res < 3 * np.median(res) + 1.0
        params, res = _fit_model(params, parity, B, P[inl], img_pts[ok][inl], d, fix_libration=fix)
        if free_lib is None and fix and inl.sum() >= 60:
            # a small field holds the ephemeris libration, but without the
            # observing site that is up to 1 deg off, and near the limb
            # (strong foreshortening) it shows as a loose fit: free it then
            p2, r2 = _fit_model(params, parity, B, P[inl], img_pts[ok][inl], d, fix_libration=False)
            if (np.degrees(np.abs(p2[4:6])).max() < 2.0
                    and np.sqrt(np.mean(r2 ** 2)) < 0.7 * np.sqrt(np.mean(res ** 2))):
                params, res = p2, r2
    return params, res


def _ref_features(ref, rmask):
    """SIFT keypoints of a reference render: (points (N, 2), descriptors), or
    None when there is too little lit surface to match against."""
    sift = cv2.SIFT_create(nfeatures=20000, contrastThreshold=0.02)
    lo, hi = np.percentile(ref[rmask], 0.5), np.percentile(ref[rmask], 99.8)
    r8 = np.clip((ref - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    kr, dr = sift.detectAndCompute(r8, rmask.astype(np.uint8) * 255)
    if dr is None or len(kr) < 20:
        return None
    return np.float32([k.pt for k in kr]), dr


def _sift_locate(mono, feats, crr, Rr, max_side=(1600, 800, 400), img_cache=None, img_features=20000):
    """Place a close-up on a render from its _ref_features: SIFT + RANSAC
    similarity for both mirror states and a few image scales. -> dict or None."""
    if feats is None:
        return None                     # nothing lit to match against
    kr_pts, dr = feats
    sift = cv2.SIFT_create(nfeatures=img_features, contrastThreshold=0.02)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    H, W = mono.shape
    best = None
    best_by_parity = {1: 0, -1: 0}
    for ms in max_side:
        f = min(1.0, ms / max(H, W))
        small = cv2.resize(mono, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 1 else mono
        for parity in (1, -1):
            im = small if parity == 1 else cv2.flip(small, 1)
            key = (ms, parity)
            if img_cache is not None and key in img_cache:
                ki, di = img_cache[key]
            else:
                lo_i, hi_i = np.percentile(im, 0.5), np.percentile(im, 99.8)
                i8 = np.clip((im - lo_i) / max(hi_i - lo_i, 1e-6) * 255, 0, 255).astype(np.uint8)
                ki, di = sift.detectAndCompute(i8, None)
                if img_cache is not None:
                    img_cache[key] = (ki, di)
            if di is None or len(ki) < 10:
                continue
            pairs = bf.knnMatch(di, dr, k=2)
            good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.8 * n.distance]
            if len(good) < 8:
                continue
            pi = np.float32([ki[m.queryIdx].pt for m in good])
            pr = kr_pts[[m.trainIdx for m in good]]
            M, inl = cv2.estimateAffinePartial2D(pr, pi, method=cv2.RANSAC, ransacReprojThreshold=3.0,
                                                 maxIters=20000, confidence=0.999)
            if M is None:
                continue
            ni = int(inl.sum())
            best_by_parity[parity] = max(best_by_parity[parity], ni)
            if best is None or ni > best["inliers"]:
                Sc = np.diag([1 / f, 1 / f, 1.0])
                Fl = np.array([[-1, 0, im.shape[1] - 1], [0, 1, 0], [0, 0, 1.0]]) if parity == -1 else np.eye(3)
                Mi = Sc @ Fl @ np.vstack([M, [0, 0, 1]])           # ref -> full-res image
                Lm = Mi[:2, :2] @ np.diag([parity, 1.0])
                k = math.sqrt(abs(np.linalg.det(Lm)))
                th = math.atan2(Lm[1, 0], Lm[0, 0])
                c = Mi[:2, :2] @ np.array(crr) + Mi[:2, 2]
                sel = inl.ravel().astype(bool)
                pimg = np.column_stack([pi[sel], np.ones(int(sel.sum()))]) @ (Sc @ Fl)[:2].T
                best = dict(params=np.array([c[0], c[1], k * Rr, th, 0.0, 0.0]), parity=parity,
                            inliers=ni, matches=len(good), pimg=pimg, pref=pr[sel].astype(np.float64))
    if best is not None:
        best["other_parity"] = best_by_parity[-best["parity"]]
    return best


# --------------------------------------------------------------------------
# the solution
# --------------------------------------------------------------------------

@dataclass
class LunarSolution:
    """Where an image sits on the Moon. Pixels are image x right, y down."""
    mode: str                      # "disc", "closeup" or "blind"
    cx: float                      # disc centre (may be outside the frame)
    cy: float
    radius_px: float               # apparent lunar radius
    theta: float                   # model rotation, rad
    parity: int                    # +1 true sky view, -1 mirror image
    B: np.ndarray                  # body -> sky, libration correction included
    S: np.ndarray                  # unit Moon -> Sun, body frame
    D: float                       # observer distance, km
    utc: str                       # "" when the date was unknown
    rms_px: float
    n_matches: int
    libration_correction_deg: tuple
    shape: tuple                   # (H, W) of the solved image
    site: Optional[tuple] = None
    notes: list = field(default_factory=list)
    sun_approximate: bool = False  # blind solve whose field shows too little shading to pin the Sun
    confidence: Optional[float] = None   # fraction of the band-passed photo the Sun-free maps explain
    runner_up: Optional[float] = None    # the same for the best other placement (blind solves)

    # ---- derived ----
    @property
    def date_known(self) -> bool:
        return bool(self.utc)

    @property
    def mirrored(self) -> bool:
        return self.parity == -1

    @property
    def d_lunar(self) -> float:
        return self.D / R_MOON

    @property
    def rotation_deg(self) -> float:
        return math.degrees(self.theta) % 360.0

    @property
    def sub_earth(self):
        la, lo = _vec_latlon(self.B[2])
        return float(la), float(lo)

    @property
    def sub_solar(self):
        la, lo = _vec_latlon(self.S)
        return float(la), float(lo)

    @property
    def colongitude(self) -> float:
        return (90.0 - self.sub_solar[1]) % 360.0

    @property
    def phase_angle_deg(self) -> float:
        """Sun - Moon - observer angle: 0 at full Moon."""
        o = np.asarray(self.B[2], np.float64)
        s = np.asarray(self.S, np.float64)
        return math.degrees(math.acos(max(-1.0, min(1.0, float(o @ s) / (np.linalg.norm(o) * np.linalg.norm(s))))))

    @property
    def illuminated_fraction(self) -> float:
        return (1.0 + math.cos(math.radians(self.phase_angle_deg))) / 2.0

    @property
    def waxing(self) -> bool:
        """The morning terminator is on the visible side between new and full Moon."""
        c = self.colongitude
        return c >= 270.0 or c < 90.0

    @property
    def phase_name(self) -> str:
        k = self.illuminated_fraction
        if k < 0.03:
            return "New Moon"
        if k > 0.97:
            return "Full Moon"
        if 0.47 <= k <= 0.53:
            return "First quarter" if self.waxing else "Last quarter"
        return "%s %s" % ("Waxing" if self.waxing else "Waning", "crescent" if k < 0.5 else "gibbous")

    def frame_centre(self):
        """(lat, lon) of the image centre, or None when it is off the Moon."""
        H, W = self.shape
        la, lo = self.pixel_to_latlon(np.array([[W / 2.0, H / 2.0]]))
        if not (np.isfinite(la[0]) and np.isfinite(lo[0])):
            return None
        return float(la[0]), float(lo[0])

    @property
    def km_per_px(self) -> float:
        """At the disc centre."""
        return R_MOON / self.radius_px

    @property
    def arcsec_per_px(self) -> Optional[float]:
        if not self.date_known:
            return None
        return math.degrees(math.asin(R_MOON / self.D)) * 3600.0 / self.radius_px

    @property
    def north_up_deg(self) -> float:
        """Direction of lunar north in the image, degrees clockwise from up."""
        uv, _ = self._project_unit(np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]), relief=False)
        v = uv[0] - np.array([self.cx, self.cy])
        return math.degrees(math.atan2(v[0], -v[1])) % 360.0

    @property
    def geometry(self) -> LunarGeometry:
        """lunar_relief geometry with the fitted libration folded in."""
        e, n, d = -self.B[0], self.B[1], -self.B[2]
        se, ss = self.sub_earth, self.sub_solar
        return LunarGeometry(utc=self.utc, O=self.D * self.B[2], S=np.asarray(self.S, np.float64),
                             d=d, n=n, e=e, D=float(self.D), sub_observer=se, sub_solar=ss,
                             colongitude=self.colongitude, topocentric=self.site is not None,
                             notes=list(self.notes))

    @property
    def camera(self) -> GnomonicCamera:
        """Perspective camera (O, rays, project) for lunar_relief consumers."""
        dd = self.d_lunar
        return GnomonicCamera(self.geometry, self.cx, self.cy, self.radius_px * math.sqrt(dd * dd - 1.0),
                              -math.degrees(self.theta), self.parity == -1)

    # ---- mapping ----
    def _project_unit(self, P, relief=False):
        P = np.asarray(P, np.float64).reshape(-1, 3)
        if relief and _MAPS is not None:
            nz = np.linalg.norm(P, axis=1) > 0
            P = P.copy()
            P[nz] = _with_relief(_MAPS, P[nz])
        params = np.array([self.cx, self.cy, self.radius_px, self.theta, 0.0, 0.0])
        return _project(params, self.parity, self.B, P, self.d_lunar)

    def latlon_to_pixel(self, lat, lon, relief=True):
        """-> (uv (N,2), visible (N,)). relief uses LOLA heights when loaded."""
        P = _latlon_vec(np.atleast_1d(lat), np.atleast_1d(lon)).reshape(-1, 3)
        uv, Z = self._project_unit(P, relief=relief)
        return uv, Z > 1.0 / self.d_lunar

    def pixel_to_latlon(self, uv):
        """-> (lat, lon) degrees east, NaN off the disc (sphere, no relief)."""
        X = self.camera.unproject_sphere(np.asarray(uv, np.float64))
        return body_to_latlon(X)

    # ---- persistence ----
    def to_header_cards(self):
        cards = [
            ("LUNSOLVE", True, "Lunar Solver surface solution"),
            ("LUNMODE", self.mode, "disc, closeup or blind"),
            ("LUNUTC", self.utc or "unknown", "capture time used for the solve"),
            ("LUNCX", round(self.cx, 3), "[px] lunar disc centre x"),
            ("LUNCY", round(self.cy, 3), "[px] lunar disc centre y"),
            ("LUNRPX", round(self.radius_px, 3), "[px] apparent lunar radius"),
            ("LUNROT", round(math.degrees(self.theta), 5), "[deg] model rotation"),
            ("LUNPAR", int(self.parity), "+1 sky view, -1 mirror image"),
            ("LUNDIST", round(self.D, 1), "[km] observer distance"),
        ]
        for i in range(3):
            for j in range(3):
                cards.append(("LUNB%d%d" % (i + 1, j + 1), float(self.B[i, j]), "body->sky matrix"))
        for i in range(3):
            cards.append(("LUNSUN%d" % (i + 1), float(self.S[i]), "Moon->Sun unit vector, body frame"))
        cards += [
            ("LUNRMS", round(self.rms_px, 3), "[px] fit residual RMS"),
            ("LUNNMAT", int(self.n_matches), "surface matches in the fit"),
            ("LUNNAX1", int(self.shape[1]), "[px] width of the solved image"),
            ("LUNNAX2", int(self.shape[0]), "[px] height of the solved image"),
            ("LUNSUNAP", bool(self.sun_approximate), "Sun angle only approximate (blind solve)"),
        ]
        if self.confidence is not None:
            cards.append(("LUNCONF", round(float(self.confidence), 4), "fraction of the photo explained"))
        if self.runner_up is not None:
            cards.append(("LUNRUNUP", round(float(self.runner_up), 4), "fraction explained by the runner-up"))
        cards += [
        ]
        return cards

    @classmethod
    def from_header(cls, hdr) -> Optional["LunarSolution"]:
        try:
            g = _card_getter(hdr)               # fits.Header, plain dict or XISF properties
            if g is None or not _truthy(g("LUNSOLVE")):
                return None
            B = np.array([[float(g("LUNB%d%d" % (i + 1, j + 1))) for j in range(3)] for i in range(3)])
            S = np.array([float(g("LUNSUN%d" % (i + 1))) for i in range(3)])
            utc = str(g("LUNUTC"))
            return cls(mode=str(g("LUNMODE")), cx=float(g("LUNCX")), cy=float(g("LUNCY")),
                       radius_px=float(g("LUNRPX")), theta=math.radians(float(g("LUNROT"))),
                       parity=int(g("LUNPAR")), B=B, S=S, D=float(g("LUNDIST")),
                       utc="" if utc == "unknown" else utc, rms_px=float(g("LUNRMS")),
                       n_matches=int(g("LUNNMAT")), libration_correction_deg=(0.0, 0.0),
                       shape=(int(g("LUNNAX2")), int(g("LUNNAX1"))), sun_approximate=_opt_bool(g, "LUNSUNAP"),
                       confidence=_opt_float(g, "LUNCONF"), runner_up=_opt_float(g, "LUNRUNUP"))
        except Exception:
            return None

    def transformed(self, M_src_to_dst, out_shape) -> Optional["LunarSolution"]:
        """The same placement in an image moved by M (2x3 or 3x3, source pixel
        -> new pixel): a similarity, mirror included. None for anything else."""
        M = np.asarray(M_src_to_dst, np.float64)
        if M.shape == (3, 3):
            if abs(M[2, 2]) < 1e-12:
                return None
            M = M / M[2, 2]
            if abs(M[2, 0]) > 1e-9 or abs(M[2, 1]) > 1e-9:
                return None                     # a perspective warp
            M = M[:2]
        if M.shape != (2, 3):
            return None
        A, t = M[:, :2], M[:, 2]
        det = float(np.linalg.det(A))
        if abs(det) < 1e-12:
            return None
        f = 1.0 if det > 0 else -1.0
        Bm = A @ np.diag([f, 1.0])              # = s Rot(alpha) for a similarity
        alpha = math.atan2(Bm[1, 0] - Bm[0, 1], Bm[0, 0] + Bm[1, 1])
        s = 0.5 * math.hypot(Bm[0, 0] + Bm[1, 1], Bm[1, 0] - Bm[0, 1])
        A_sim = s * np.array([[math.cos(alpha), -math.sin(alpha)], [math.sin(alpha), math.cos(alpha)]]) @ np.diag([f, 1.0])
        # crops round their sides to whole pixels (0.3 % stretch on a 200 px crop):
        # fine while the best similarity stays within 0.75 px across the image
        h0, w0 = (float(v) for v in self.shape[:2])
        if float(np.linalg.norm(A - A_sim, 2)) * 0.5 * math.hypot(w0, h0) > 0.75:
            return None
        # anchored at the image centre, where the pixels are (the disc centre can be far off-frame)
        p0 = np.array([(w0 - 1.0) / 2.0, (h0 - 1.0) / 2.0])
        c = A @ p0 + t + A_sim @ (np.array([self.cx, self.cy]) - p0)
        # pixel = c + R Rot(theta) (parity X, -Y): A Rot(theta) diag(p, 1) = s Rot(alpha + f theta) diag(f p, 1)
        return _dc_replace(self, cx=float(c[0]), cy=float(c[1]), radius_px=float(self.radius_px * s),
                           theta=float(alpha + f * self.theta), parity=int(f * self.parity),
                           shape=(int(out_shape[0]), int(out_shape[1])), rms_px=float(self.rms_px * s))

    def summary(self) -> dict:
        return dict(mode=self.mode, utc=self.utc or None, rms_px=self.rms_px, n_matches=self.n_matches,
                    radius_px=self.radius_px, km_per_px=self.km_per_px, arcsec_per_px=self.arcsec_per_px,
                    rotation_deg=self.rotation_deg, north_up_deg=self.north_up_deg, mirrored=self.mirrored,
                    sub_earth=self.sub_earth, sub_solar=self.sub_solar, colongitude=self.colongitude,
                    libration_correction_deg=self.libration_correction_deg, sun_approximate=self.sun_approximate,
                    confidence=self.confidence, runner_up=self.runner_up, phase_name=self.phase_name,
                    illuminated_fraction=self.illuminated_fraction, phase_angle_deg=self.phase_angle_deg,
                    frame_centre=self.frame_centre())


def _opt_bool(g, key) -> bool:
    try:
        return _truthy(g(key))
    except Exception:
        return False


def _opt_float(g, key):
    try:
        return float(g(key))
    except Exception:
        return None


# --------------------------------------------------------------------------
# landing sites
# --------------------------------------------------------------------------
# Spacecraft on the surface. Positions are LRO/LROC measurements (Wagner et al.
# 2017, Icarus 283, 92-103, and LROC releases since), Mean Earth frame,
# east-positive longitude. Luna 9 and Luna 13 have never been found from orbit:
# theirs are the positions announced at the time, possibly tens of km off.
# Rover positions are where the rover ended; the Apollo rovers parked within
# 200 m of their landers and are left out.
# --------------------------------------------------------------------------
# the solution stored with a document
# --------------------------------------------------------------------------
# A SASpro document carries its header as original_header (the current header:
# what a save writes when it is a fits.Header), fits_header (as loaded from a
# FITS / TIFF / PNG / JPEG file) or an XISF property dict. The LUN* cards go
# where a save reads them, so FITS keeps them natively and TIFF / PNG / JPEG /
# XISF in their embedded header (WebP drops metadata); the reader looks in the
# same order.

def _card_text(v):
    if isinstance(v, str):
        return v.strip().strip("'").strip()
    return v


def _card_getter(h):
    """key -> value for a fits.Header, a plain dict or an XISF property dict
    (FITSKeywords: {key: [{'value': ...}]}); None when h is none of those."""
    if h is None:
        return None
    if isinstance(h, dict):
        fk = h.get("FITSKeywords")
        if isinstance(fk, dict):
            def g(k):
                v = fk[k]
                if isinstance(v, (list, tuple)):
                    v = v[0]
                if isinstance(v, dict):
                    v = v.get("value")
                return _card_text(v)
            return g

        def g(k):
            v = h[k]
            if isinstance(v, dict) and "value" in v:
                v = v["value"]
            return _card_text(v)
        return g
    if hasattr(h, "__getitem__"):
        return lambda k: _card_text(h[k])
    return None


def _truthy(v) -> bool:
    if isinstance(v, str):
        return v.strip().upper() in ("T", "TRUE", "1", "YES")
    return bool(v)


def _header_order(meta):
    """Metadata keys in the order a save picks its header (DocManager.save_document)."""
    try:
        from astropy.io.fits import Header
        if isinstance(meta.get("original_header"), Header):
            return ("original_header", "fits_header", "header")
    except Exception:
        pass
    return ("fits_header", "original_header", "header")


def solution_from_metadata(meta, shape=None) -> Optional["LunarSolution"]:
    """The Lunar Solver placement stored with a document: the live object, else
    the LUN* cards of whichever header it carries. None when there is none, or
    when shape (h, w) shows the image was resized or cropped outside SASpro's
    geometry tools since."""
    if not isinstance(meta, dict):
        return None
    sol = meta.get("lunar_solution")
    if not isinstance(sol, LunarSolution):
        sol = None
        for k in _header_order(meta):
            h = meta.get(k)
            if h is None:
                continue
            sol = LunarSolution.from_header(h)
            if sol is not None:
                break
    if sol is None:
        return None
    if shape is not None and tuple(int(v) for v in sol.shape) != tuple(int(v) for v in tuple(shape)[:2]):
        return None
    return sol


_LUN_KEYS = ("LUNSOLVE", "LUNMODE", "LUNUTC", "LUNCX", "LUNCY", "LUNRPX", "LUNROT", "LUNPAR", "LUNDIST",
             "LUNRMS", "LUNNMAT", "LUNNAX1", "LUNNAX2", "LUNSUNAP", "LUNCONF", "LUNRUNUP",
             "LUNSUN1", "LUNSUN2", "LUNSUN3") + tuple("LUNB%d%d" % (i, j) for i in (1, 2, 3) for j in (1, 2, 3))


def store_solution_in_metadata(meta, sol, copy_header=False):
    """Keep a solution with a document: the live object, plus LUN* cards in the
    header a save writes. sol None removes both. copy_header=True writes into a
    copy (an edit's new metadata must not change the header its undo step
    holds). Returns the header written, or None."""
    from astropy.io.fits import Header
    if not isinstance(meta, dict):
        return None
    if sol is None:
        meta.pop("lunar_solution", None)
    else:
        meta["lunar_solution"] = sol
    key, hdr = None, None
    for k in _header_order(meta):
        h = meta.get(k)
        if isinstance(h, Header):
            key, hdr = k, h
            break
    if hdr is None:
        if sol is None:
            return None
        # no fits.Header yet (an array, or an XISF property dict, which a save
        # converts only through fits_header): start one where a save reads it
        key = "original_header" if meta.get("original_header") is None else "fits_header"
        hdr = Header()
    elif copy_header:
        hdr = hdr.copy()
    for k in _LUN_KEYS:
        while k in hdr:
            del hdr[k]
    if sol is not None:
        for k, v, c in sol.to_header_cards():
            hdr[k] = (v, c)
    meta[key] = hdr
    meta.pop("__header_snapshot__", None)      # header viewers rebuild it
    return hdr


def lunar_solution_after_geometry(meta, M_src_to_dst, out_shape, in_shape=None):
    """Keep a stored placement right through a geometry edit (crop, flip,
    rotate, rescale, canvas resize): the image was moved by a similarity, so
    the solution moves with it exactly. A keystone (perspective) crop, or a
    placement that no longer matched the image, drops it. Changes meta in
    place (with a copied header) and returns it."""
    sol = solution_from_metadata(meta)
    if sol is None:
        return meta
    if in_shape is not None and tuple(int(v) for v in sol.shape) != tuple(int(v) for v in tuple(in_shape)[:2]):
        new = None
    else:
        new = sol.transformed(M_src_to_dst, out_shape)
    store_solution_in_metadata(meta, new, copy_header=True)
    return meta


LANDING_SITES = (
    # name, country, year, kind, lat, lon, position approximate
    ("Luna 9", "USSR", 1966, "lander", 7.08, -64.37, True),
    ("Surveyor 1", "USA", 1966, "lander", -2.4745, -43.3398, False),
    ("Luna 13", "USSR", 1966, "lander", 18.87, -62.05, True),
    ("Surveyor 3", "USA", 1967, "lander", -3.0162, -23.418, False),
    ("Surveyor 5", "USA", 1967, "lander", 1.4551, 23.1943, False),
    ("Surveyor 6", "USA", 1967, "lander", 0.4742, -1.4275, False),
    ("Surveyor 7", "USA", 1968, "lander", -40.9812, -11.5127, False),
    ("Apollo 11", "USA", 1969, "crewed", 0.67416, 23.47314, False),
    ("Apollo 12", "USA", 1969, "crewed", -3.0128, -23.4219, False),
    ("Luna 16", "USSR", 1970, "sample return", -0.5137, 56.3638, False),
    ("Luna 17", "USSR", 1970, "lander", 38.23764, -35.00163, False),
    ("Lunokhod 1", "USSR", 1970, "rover", 38.315, -35.0081, False),
    ("Apollo 14", "USA", 1971, "crewed", -3.64589, -17.47194, False),
    ("Apollo 15", "USA", 1971, "crewed", 26.13239, 3.6333, False),
    ("Apollo 16", "USA", 1972, "crewed", -8.9734, 15.5011, False),
    ("Luna 20", "USSR", 1972, "sample return", 3.7863, 56.6242, False),
    ("Apollo 17", "USA", 1972, "crewed", 20.1911, 30.7723, False),
    ("Luna 21", "USSR", 1973, "lander", 25.9994, 30.4076, False),
    ("Lunokhod 2", "USSR", 1973, "rover", 25.8323, 30.9222, False),
    ("Luna 23", "USSR", 1974, "lander", 12.6667, 62.1511, False),
    ("Luna 24", "USSR", 1976, "sample return", 12.7142, 62.2129, False),
    ("Chang'e 3", "China", 2013, "lander and rover", 44.1214, -19.5117, False),
    ("Chang'e 4", "China", 2019, "lander and rover", -45.4562, 177.5885, False),
    ("Chang'e 5", "China", 2020, "sample return", 43.0576, -51.9161, False),
    ("Chandrayaan-3", "India", 2023, "lander and rover", -69.3741, 32.32, False),
    ("SLIM", "Japan", 2024, "lander", -13.316, 25.251, False),
    ("Odysseus (IM-1)", "USA", 2024, "lander", -80.13, 1.44, False),
    ("Chang'e 6", "China", 2024, "sample return", -41.6385, -153.9852, False),
    ("Blue Ghost 1", "USA", 2025, "lander", 18.5623, 61.8103, False),
    ("Athena (IM-2)", "USA", 2025, "lander", -84.7906, 29.1957, False),
)


def visible_sites(sol: "LunarSolution", margin_px=0.0):
    """Landing sites on the visible side inside the frame: list of
    (name, country, year, kind, approximate, lat, lon, u, v)."""
    if not LANDING_SITES:
        return []
    lat = np.array([r[4] for r in LANDING_SITES], np.float64)
    lon = np.array([r[5] for r in LANDING_SITES], np.float64)
    uv, vis = sol.latlon_to_pixel(lat, lon, relief=True)
    H, W = sol.shape
    out = []
    for r, (u, v), ok in zip(LANDING_SITES, uv, vis):
        if ok and -margin_px <= u < W + margin_px and -margin_px <= v < H + margin_px:
            out.append((r[0], r[1], r[2], r[3], r[6], r[4], r[5], float(u), float(v)))
    return out


def visible_features(sol: LunarSolution, *, min_px=20.0, types=("AA", "ME", "OC", "SI", "LC", "PA", "MO", "VA", "RU"),
                     max_count=400, margin_px=0.0):
    """Named features large enough to label and on the visible side of the
    Moon inside the frame: list of (name, type, u, v, radius_px), largest first."""
    F = load_features()
    sel = np.isin(F["type"], list(types))
    r_px = F["diam"] / 2.0 / R_MOON * sol.radius_px
    sel &= 2.0 * r_px >= min_px
    idx = np.where(sel)[0]
    if not len(idx):
        return []
    uv, vis = sol.latlon_to_pixel(F["lat"][idx], F["lon"][idx], relief=False)
    H, W = sol.shape
    rr = r_px[idx]
    inside = ((uv[:, 0] > -rr - margin_px) & (uv[:, 0] < W + rr + margin_px)
              & (uv[:, 1] > -rr - margin_px) & (uv[:, 1] < H + rr + margin_px))
    keep = np.where(vis & inside)[0]
    keep = keep[np.argsort(-rr[keep])][:max_count]
    return [(F["name"][idx[i]], str(F["type"][idx[i]]), float(uv[i, 0]), float(uv[i, 1]), float(rr[i]))
            for i in keep]


def feature_outlines(sol: LunarSolution, lat, lon, diam_km, n=48):
    """Rims of circular features as seen in the image (foreshortened toward the
    limb): (N, n, 2) pixels and (N, n) visibility."""
    c = _latlon_vec(np.atleast_1d(lat), np.atleast_1d(lon)).reshape(-1, 3)
    a = (np.atleast_1d(np.asarray(diam_km, np.float64)) / 2.0 / R_MOON)[:, None]
    z = np.array([0.0, 0.0, 1.0])
    t1 = np.cross(z, c)
    bad = np.linalg.norm(t1, axis=1) < 1e-9
    t1[bad] = np.array([1.0, 0.0, 0.0])
    t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
    t2 = np.cross(c, t1)
    ph = np.linspace(0.0, 2 * np.pi, n, endpoint=False)[None, :, None]
    pts = (np.cos(a)[..., None] * c[:, None, :]
           + np.sin(a)[..., None] * (np.cos(ph) * t1[:, None, :] + np.sin(ph) * t2[:, None, :]))
    uv, Z = sol._project_unit(pts.reshape(-1, 3), relief=False)
    return uv.reshape(-1, n, 2), (Z > 1.0 / sol.d_lunar).reshape(-1, n)


# --------------------------------------------------------------------------
# capture time
# --------------------------------------------------------------------------

def _local_to_utc(naive: _dt.datetime) -> _dt.datetime:
    """Local wall-clock time on THIS computer -> naive UTC."""
    return naive.astimezone().astimezone(_dt.timezone.utc).replace(tzinfo=None)


def _parse_iso(s: str) -> Optional[_dt.datetime]:
    s = str(s).strip().replace(" ", "T")
    if not s:
        return None
    if s.endswith("Z"):
        s = s[:-1]
    try:
        t = _dt.datetime.fromisoformat(s)
    except ValueError:
        m = re.match(r"(\d{4})-(\d{2})-(\d{2})(?:T(\d{2}):(\d{2}):(\d{2})(\.\d+)?)?", s)
        if not m:
            return None
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        hh = int(m.group(4) or 0); mi = int(m.group(5) or 0); ss = int(m.group(6) or 0)
        frac = float(m.group(7) or 0.0)
        t = _dt.datetime(y, mo, d, hh, mi, ss) + _dt.timedelta(seconds=frac)
    if t.tzinfo is not None:
        t = t.astimezone(_dt.timezone.utc).replace(tzinfo=None)
    return t


def _header_get(h, key):
    try:
        if hasattr(h, "get"):
            v = h.get(key)
        else:
            v = h[key]
    except Exception:
        return None
    if isinstance(v, dict):                 # XISF-style {"value": ...}
        v = v.get("value")
    if isinstance(v, (list, tuple)) and v:
        v = v[0]
        if isinstance(v, dict):
            v = v.get("value")
    return v


def capture_time(meta_or_header, path=None):
    """Best guess at the capture time from an image's metadata.
    -> (naive UTC datetime or None, source text for the user)."""
    meta = meta_or_header if isinstance(meta_or_header, dict) else {}
    headers = []
    if isinstance(meta_or_header, dict):
        for k in ("original_header", "fits_header", "header"):
            if meta.get(k) is not None:
                headers.append(meta.get(k))
    elif meta_or_header is not None:
        headers.append(meta_or_header)
    for h in headers:
        for key in ("DATE-OBS", "DATE-BEG", "DATE_OBS", "DATE-AVG"):
            v = _header_get(h, key)
            if v is None:
                continue
            t = _parse_iso(v)
            if t is None:
                continue
            if t.hour == 0 and t.minute == 0 and t.second == 0 and "T" not in str(v):
                tv = _header_get(h, "TIME-OBS") or _header_get(h, "UT") or _header_get(h, "UTSTART")
                if tv:
                    m = re.match(r"(\d{1,2}):(\d{2}):?(\d{2}(?:\.\d+)?)?", str(tv).strip())
                    if m:
                        t = t.replace(hour=int(m.group(1)), minute=int(m.group(2))) + _dt.timedelta(
                            seconds=float(m.group(3) or 0.0))
            return t, "From the %s header keyword (UTC)." % key
    path = path or meta.get("file_path") or meta.get("path")
    if path:
        p = Path(str(path).split("::")[0])      # multi-HDU docs carry "path::HDU"
        # WinJUPOS / FireCapture style: 2023-05-01-0129_8 (UTC, tenths of a minute)
        m = re.search(r"(\d{4})-(\d{2})-(\d{2})-(\d{2})(\d{2})_(\d)", p.name)
        if m:
            y, mo, d, hh, mi, tenth = (int(g) for g in m.groups())
            t = _dt.datetime(y, mo, d, hh, mi) + _dt.timedelta(seconds=6 * tenth)
            return t, "From the WinJUPOS-style file name (UTC)."
        # SharpCap: <...>/YYYY-MM-DD/<target>/HH_MM_SS<...> in local time
        m = re.match(r"(\d{2})_(\d{2})_(\d{2})", p.name)
        if m:
            for parent in p.parents:
                dm = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", parent.name)
                if dm:
                    try:
                        local = _dt.datetime(int(dm.group(1)), int(dm.group(2)), int(dm.group(3)),
                                             int(m.group(1)), int(m.group(2)), int(m.group(3)))
                        return _local_to_utc(local), ("From the SharpCap folder and file name "
                                                      "(local time, converted with this computer's time zone).")
                    except ValueError:
                        break
        # EXIF DateTimeOriginal (camera clock, local time)
        if p.suffix.lower() in (".jpg", ".jpeg", ".tif", ".tiff", ".png", ".webp"):
            try:
                from PIL import Image
                with Image.open(str(p)) as im:
                    ex = im.getexif()
                    sub = ex.get_ifd(0x8769) if hasattr(ex, "get_ifd") else {}
                    # DateTimeOriginal only: plain DateTime (0x0132) is when the
                    # file was last processed, often a day after the capture
                    raw = sub.get(0x9003)
                    off = sub.get(0x9011)
                if raw:
                    t = _dt.datetime.strptime(str(raw).strip()[:19], "%Y:%m:%d %H:%M:%S")
                    if off:
                        sign = -1 if str(off).startswith("-") else 1
                        hh, mm = str(off).strip("+-").split(":")[:2]
                        t = t - sign * _dt.timedelta(hours=int(hh), minutes=int(mm))
                        return t, "From the EXIF capture time (UTC offset in the file)."
                    return _local_to_utc(t), ("From the EXIF capture time (local time, converted with this "
                                              "computer's time zone).")
            except Exception:
                pass
    # DATE-LOC: the capture PC's local clock with no UTC offset (N.I.N.A.;
    # Planetary Stacker results from an AVI's IDIT or a SER without UTC).
    for h in headers:
        v = _header_get(h, "DATE-LOC")
        t = _parse_iso(v) if v is not None else None
        if t is not None:
            return _local_to_utc(t), ("From the DATE-LOC header keyword (local time, converted with this "
                                      "computer's time zone).")
    return None, "No capture time found in the image."


def _as_utc(when) -> Optional[_dt.datetime]:
    if when is None:
        return None
    if isinstance(when, str):
        t = _parse_iso(when)
        if t is None:
            raise LunarSolveError("Could not read the capture time %r." % when)
        return t
    if isinstance(when, _dt.datetime):
        if when.tzinfo is not None:
            return when.astimezone(_dt.timezone.utc).replace(tzinfo=None)
        return when
    raise LunarSolveError("Unsupported capture time %r." % (when,))


# --------------------------------------------------------------------------
# solve
# --------------------------------------------------------------------------

def _rms_limit(R_img):
    return max(2.5, 0.001 * R_img)


def _accept(params, res, label):
    """Validation: a non-solution is never an output."""
    if res is None:
        raise LunarSolveError("%s: the surface matching found too few matches to trust." % label)
    rms = float(np.sqrt(np.mean(res ** 2)))
    if len(res) < 40:
        raise LunarSolveError("%s: only %d surface matches; at least 40 are needed." % (label, len(res)))
    if rms > _rms_limit(params[2]):
        raise LunarSolveError("%s: the fit is too loose (%.1f px RMS)." % (label, rms))
    return rms


def _acceptable(params, res) -> bool:
    """_accept's test without the exception."""
    return res is not None and len(res) >= 40 and float(np.sqrt(np.mean(res ** 2))) <= _rms_limit(params[2])


def _footprint_latlon(params, parity, B, shape, pad_frac=0.25, pad_deg=1.6):
    """The image's area on the Moon, padded: (lat_lo, lat_hi, lon_west, lon_span)
    in degrees, longitude east 0..360 (similarity model, off-disc pixels held
    at the limb)."""
    H, W = shape
    gx, gy = np.meshgrid(np.linspace(0, W - 1, 13), np.linspace(0, H - 1, 13))
    c, s = math.cos(params[3]), math.sin(params[3])
    dx = (gx.ravel() - params[0]) / params[2]
    dy = (gy.ravel() - params[1]) / params[2]
    X = parity * (c * dx + s * dy)
    Y = -(c * dy - s * dx)
    r = np.hypot(X, Y)
    pin = np.where(r > 0.999, 0.999 / np.maximum(r, 1e-12), 1.0)
    lat, lon = _vec_latlon(_body_from_sky(_rot_small(params[4], params[5]) @ B, X * pin, Y * pin))
    lon0 = float(lon[lon.size // 2])
    dl = (lon - lon0 + 180.0) % 360.0 - 180.0
    pl = pad_frac * float(np.ptp(lat)) + pad_deg
    lat_lo, lat_hi = float(lat.min()) - pl, float(lat.max()) + pl
    if lat_hi >= 89.0 or lat_lo <= -89.0:
        return lat_lo, lat_hi, 0.0, 360.0
    cl = max(math.cos(math.radians(max(abs(lat_lo), abs(lat_hi)))), 0.05)
    pw = pad_frac * float(np.ptp(dl)) + pad_deg / cl
    return lat_lo, lat_hi, (lon0 + float(dl.min()) - pw) % 360.0, float(np.ptp(dl)) + 2.0 * pw


def _detail_helps(params, shape) -> bool:
    """Sharper than ~1.2 km/px and under ~25 deg across: the 64 px/deg maps
    hold detail the patch matching can use that the 1.9 km maps do not."""
    km_px = R_MOON / max(float(params[2]), 1.0)
    return km_px < 1.2 and math.degrees(km_px * max(shape) / R_MOON) < 25.0


def _refine_on(mono, params, parity, view, mp, R_max, cancel=None, **kw):
    """Render the image's area from maps mp at its working scale, then _refine."""
    Rf = _work_radius(params, mono.shape, R_max, 3.2e6)
    ref, rmask, crr = _render(view, Rf, mp, roi=_footprint_roi(params, parity, Rf, mono.shape))
    return _refine(mono, params, parity, view, ref, rmask, crr, Rf, mp, cancel=cancel, **kw)


def _detail_fallback(mono, p0, parity, view, maps, current, data_dir=None, progress=None, cancel=None, frac=0.7,
                     **kw):
    """A small, sharp field the 1.9 km maps leave unverified (current =
    (params, res) from them): the same refinement from p0 on 64 px/deg maps
    of its area. -> (params, res, region maps or None); the detailed result
    when it is acceptable or has more matches."""
    if _acceptable(*current) or not _detail_helps(p0, mono.shape):
        return current[0], current[1], None
    _report(progress, frac, "Loading the detailed map of this area")
    reg = _region_maps(data_dir, _footprint_latlon(p0, parity, view.B, mono.shape), maps, progress, cancel, frac)
    _check(cancel)
    _report(progress, frac, "Matching the surface on the detailed map")
    p1, r1 = _refine_on(mono, p0, parity, view, reg, _R_HI, cancel=cancel, **kw)
    if _acceptable(p1, r1) or (r1 is not None and (current[1] is None or len(r1) > len(current[1]))):
        return p1, r1, reg
    return current[0], current[1], None


def _solve_disc(mono, limb, view, maps, progress, cancel):
    _report(progress, 0.3, "Finding the rotation and mirror")
    co = _coarse_rotation(mono, limb, view, maps)
    if co["score"] < 0.15 or co["margin"] < 3.0:
        return None
    params = np.array([limb["cx"], limb["cy"], limb["R"], co["theta"], 0.0, 0.0])
    Rf = int(min(1000, max(400, limb["R"])))
    _report(progress, 0.45, "Rendering the reference Moon")
    ref, rmask, crr = _render(view, Rf, maps, roi=_footprint_roi(params, co["parity"], Rf, mono.shape))
    _report(progress, 0.6, "Matching the surface")
    params, res = _refine(mono, params, co["parity"], view, ref, rmask, crr, Rf, maps, cancel=cancel)
    return params, co["parity"], res


def _solve_closeup(mono, view, maps, progress, cancel, R_hint=None, data_dir=None):
    Rr = 1500
    _report(progress, 0.3, "Rendering the reference Moon")
    ref, rmask, crr = _render(view, Rr, maps)
    _check(cancel)
    _report(progress, 0.45, "Locating the image on the Moon")
    loc = _sift_locate(mono, _ref_features(ref, rmask), crr, Rr)
    if loc is None or loc["inliers"] < 15 or loc["inliers"] < 3 * (loc["other_parity"] + 1):
        return None
    Rf = _work_radius(loc["params"], mono.shape, 1800, 3.2e6)
    _report(progress, 0.6, "Matching the surface")
    ref, rmask, crr = _render(view, Rf, maps, roi=_footprint_roi(loc["params"], loc["parity"], Rf, mono.shape))
    params, res = _refine(mono, loc["params"], loc["parity"], view, ref, rmask, crr, Rf, maps, cancel=cancel)
    params, res, _ = _detail_fallback(mono, loc["params"], loc["parity"], view, maps, (params, res), data_dir,
                                      progress, cancel, 0.7)
    return params, loc["parity"], res


def _evaluate(mono, params, parity, ref, rmask, crr, Rr, max_patches=400):
    k = params[2] / Rr
    img_s = mono if k <= 1.2 else cv2.resize(mono, None, fx=1 / k, fy=1 / k, interpolation=cv2.INTER_AREA)
    ks = mono.shape[1] / img_s.shape[1]
    ps = np.array(params, float)
    ps[:3] /= ks
    A = _ref_to_img_affine(ps, parity, Rr, crr)
    step = int(max(24, math.sqrt(float(rmask.sum()) / max_patches)))      # a full disc: ~400 patches, not 4000
    _, _, sc = _grid_matches(img_s, ref, rmask, A, search=6, step=step, min_score=0.0)
    return float(np.mean(sc)) if len(sc) else 0.0


# The blind search's reference renders (mean libration, one per Sun angle)
# never change: their SIFT features are built once and kept beside the maps.
_SCAN_COLONGS = tuple(range(0, 360, 15))
_SCAN_R = 700
_SCAN_FILE = "lunar_blind_scan_v1.npz"
_SCAN_FEATS: dict = {}
_SCAN_LOCK = threading.Lock()


def _scan_features(maps, data_dir=None, progress=None, cancel=None) -> dict:
    """{colongitude: _ref_features or None} for the blind search."""
    with _SCAN_LOCK:
        if len(_SCAN_FEATS) == len(_SCAN_COLONGS):
            return _SCAN_FEATS
        path = _data_dir(data_dir) / _SCAN_FILE
        feats = {}
        try:
            with np.load(str(path)) as z:
                for c in _SCAN_COLONGS:
                    pts, desc = z["p%03d" % c], z["d%03d" % c]
                    feats[c] = (pts.astype(np.float32), desc.astype(np.float32)) if len(pts) else None
        except Exception:
            feats = {}
        if len(feats) != len(_SCAN_COLONGS):
            feats = {}
            for i, c in enumerate(_SCAN_COLONGS):
                _check(cancel)
                _report(progress, 0.2 + 0.3 * i / len(_SCAN_COLONGS),
                        "Preparing the blind search, one time only (%d of %d)" % (i + 1, len(_SCAN_COLONGS)))
                ref, rmask, _ = _render(_mean_view(c), _SCAN_R, maps)
                feats[c] = _ref_features(ref, rmask)
            arrays = {}
            for c, f in feats.items():
                arrays["p%03d" % c] = f[0] if f is not None else np.zeros((0, 2), np.float32)
                # OpenCV's SIFT descriptors are whole numbers 0..255: uint8 is lossless
                arrays["d%03d" % c] = f[1].astype(np.uint8) if f is not None else np.zeros((0, 128), np.uint8)
            try:
                tmp = path.with_name(path.name + ".part.npz")
                np.savez_compressed(str(tmp), **arrays)
                tmp.replace(path)
            except OSError:
                pass                    # read-only data folder: rebuilt next session
        _SCAN_FEATS.update(feats)
        return _SCAN_FEATS


# --------------------------------------------------------------------------
# Sun-free matching (no date)
# --------------------------------------------------------------------------
# Under Lambert shading a photo of the Moon is I = s . (A n): the albedo times
# the surface normal, three fixed maps in the Moon's body frame, mixed by the
# Sun vector s. Band-passing keeps that linear. So a placement is scored by how
# much of the band-passed photo a least-squares mix of the three maps explains
# (R^2), with no Sun angle to search, and the fitted mix IS the Sun direction.
# Measured on real tiles: 0.5-0.9 at the true placement, under 0.01 a few
# pixels or degrees off.

_NB = 3
_PAIRS = tuple((k, l) for k in range(_NB) for l in range(k, _NB))
_SF_MIN_R2 = 0.20          # a placement must explain at least this much of the photo
_SF_MIN_RATIO = 2.5        # ... and this many times more than the best other placement
_POLAR_LIBS = tuple((float(a), float(b)) for a in np.arange(-7, 7.01, 2) for b in np.arange(-9, 9.01, 2))


@njit(parallel=True, cache=True, nogil=True)
def _basis_kernel(out, B, R_px, cx, cy, alb, ge, gn, ppd, row0, col0):
    """out (3, h, w): albedo x surface normal (body frame) on the orthographic
    sky grid _render uses (x right = sky west, y up = north). Zero off the disc."""
    _, h, w = out.shape
    for yy in prange(h):
        for xx in range(w):
            u = (xx - cx) / R_px
            vv = -(yy - cy) / R_px
            r2 = u * u + vv * vv
            if r2 >= 1.0:
                continue
            ww = math.sqrt(1.0 - r2)
            px = B[0, 0] * u + B[1, 0] * vv + B[2, 0] * ww
            py = B[0, 1] * u + B[1, 1] * vv + B[2, 1] * ww
            pz = B[0, 2] * u + B[1, 2] * vv + B[2, 2] * ww
            if pz > 1.0:
                pz = 1.0
            if pz < -1.0:
                pz = -1.0
            lat = math.degrees(math.asin(pz))
            lon = math.degrees(math.atan2(py, px))
            A = _bil(alb, lat, lon, ppd, row0, col0)
            gE = _bil(ge, lat, lon, ppd, row0, col0)
            gN = _bil(gn, lat, lon, ppd, row0, col0)
            la = math.radians(lat)
            lo = math.radians(lon)
            sla = math.sin(la)
            cla = math.cos(la)
            slo = math.sin(lo)
            clo = math.cos(lo)
            nx = px + gE * slo + gN * sla * clo
            ny = py - gE * clo + gN * sla * slo
            nz = pz - gN * cla
            nn = math.sqrt(nx * nx + ny * ny + nz * nz)
            out[0, yy, xx] = A * nx / nn
            out[1, yy, xx] = A * ny / nn
            out[2, yy, xx] = A * nz / nn


def _basis(B, R_px, cx, cy, h, w, maps):
    out = np.zeros((_NB, h, w), np.float32)
    with _numba_lock():
        _basis_kernel(out, np.ascontiguousarray(B, np.float64), float(R_px), float(cx), float(cy),
                      maps.alb, maps.ge, maps.gn, *maps.geo)
    return out


def _dog(a, s1, s2):
    return cv2.GaussianBlur(a, (0, 0), s1) - cv2.GaussianBlur(a, (0, 0), s2)


def _usable(small, s2):
    """Lit pixels whose band-pass footprint stays clear of the limb and the
    terminator: edges no mix of the maps can explain (left in, any place along
    the limb lines up the limb's edge and scores high)."""
    e = int(math.ceil(2 * s2)) + 2
    lit = cv2.GaussianBlur(small, (0, 0), 2.0) > 0.08 * np.percentile(small, 99.5)
    lit = cv2.erode(lit.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * e + 1, 2 * e + 1))) > 0
    h, w = small.shape
    frame = np.zeros((h, w), bool)
    frame[e:h - e, e:w - e] = True
    return (lit & frame).astype(np.float32)


def _lib_B(lib_deg):
    return _rot_small(math.radians(lib_deg[0]), math.radians(lib_deg[1])) @ _B0


def _lsq_r2(cv, G, EI):
    """Per-hypothesis least squares. cv (n, 3) correlations, G (n, 3, 3) Gram
    matrices, EI photo energy -> explained fraction (n,), mix (n, 3)."""
    G = G + np.eye(_NB)[None] * (1e-9 + 1e-6 * np.abs(G).max())
    beta = np.linalg.solve(G, cv[..., None])[..., 0]
    return (beta * cv).sum(1) / max(EI, 1e-12), beta


def _gram(g, n):
    G = np.empty((n, _NB, _NB))
    for i, (k, l) in enumerate(_PAIRS):
        G[:, k, l] = g[i].ravel()
        G[:, l, k] = g[i].ravel()
    return G


def _polar_grid(cx, cy, Rw, n_ang, j0, j1):
    """cv2.remap maps for radii j0..j1-1 (rows) x angles (cols), cv2.warpPolar's layout."""
    r = np.arange(j0, j1, dtype=np.float32)
    a = np.arange(n_ang, dtype=np.float32) * np.float32(2 * np.pi / n_ang)
    return ((cx + r[:, None] * np.cos(a)[None, :]).astype(np.float32),
            (cy + r[:, None] * np.sin(a)[None, :]).astype(np.float32))


class _PolarSearch:
    """Photo with a limb in frame: every rotation, both mirror states and a
    libration grid compared at once around the limb centre, rotation by FFT."""
    S1, S2, RHO_MAX = 1.2, 5.0, 0.97

    def __init__(self, mono, limbs, maps, Rw=300, n_ang=2048):
        import scipy.fft as sfft
        self.sfft = sfft
        self.maps = maps
        self.RW, self.NANG = Rw, n_ang = int(Rw), int(n_ang)
        prep = []
        lo_j, hi_j = Rw, 0
        rho = (np.arange(Rw) + 0.5) / Rw
        for limb in limbs:
            f = Rw / limb["R"]
            small = cv2.resize(mono, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
            I = _dog(small, self.S1, self.S2)
            W0 = _usable(small, self.S2)
            cx, cy = limb["cx"] * f, limb["cy"] * f
            mx, my = _polar_grid(cx, cy, Rw, n_ang, 0, Rw)
            Wfull = cv2.remap(W0, mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            rows = np.where((Wfull.sum(1) > 0) & (rho < self.RHO_MAX))[0]
            if len(rows):
                lo_j, hi_j = min(lo_j, int(rows[0])), max(hi_j, int(rows[-1]) + 1)
            prep.append((limb, f, I, W0, cx, cy))
        if hi_j <= lo_j:
            raise LunarSolveError("no usable surface around the limb")
        self.j0, self.j1 = lo_j, hi_j
        rw = rho[lo_j:hi_j].astype(np.float32)[:, None]
        self.variants = []
        for limb, f, I, W0, cx, cy in prep:
            h, w = I.shape
            F = {}
            for parity in (1, -1):
                Ii = I if parity == 1 else cv2.flip(I, 1)
                Wi = W0 if parity == 1 else cv2.flip(W0, 1)
                mx, my = _polar_grid(cx if parity == 1 else (w - 1) - cx, cy, Rw, n_ang, lo_j, hi_j)
                Ip = cv2.remap(Ii, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                Wp = cv2.remap(Wi, mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0) * rw
                # a ring's mean is the same at every rotation (the limb's falloff): drop it
                ws = Wp.sum(1, keepdims=True)
                Ip = Ip - np.where(ws > 0, (Wp * Ip).sum(1, keepdims=True) / np.maximum(ws, 1e-9), 0.0)
                J = (Wp * Ip).astype(np.float32)
                F[parity] = (sfft.rfft(J, axis=-1, workers=-1), sfft.rfft(Wp.astype(np.float32), axis=-1, workers=-1),
                             float((J * Ip).sum()))
            self.variants.append((limb, F))
        size = 2 * Rw + 40
        self.gc = (size - 1) / 2.0
        self.size = size
        self.mmx, self.mmy = _polar_grid(self.gc, self.gc, Rw, n_ang, lo_j, hi_j)

    def _lib_spectra(self, B):
        M = _basis(B, self.RW, self.gc, self.gc, self.size, self.size, self.maps)
        Mp = []
        for k in range(_NB):
            m = cv2.remap(_dog(M[k], self.S1, self.S2), self.mmx, self.mmy, cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            Mp.append(m - m.mean(1, keepdims=True))
        stack = np.stack(Mp + [Mp[k] * Mp[l] for k, l in _PAIRS])
        return np.conj(self.sfft.rfft(stack, axis=-1, workers=-1))

    def _scores(self, F, FMc):
        FJ, FW, EI = F
        n = self.NANG
        cJ = self.sfft.irfft((FJ[None] * FMc[:_NB]).sum(1), n=n, axis=-1, workers=-1)
        cW = self.sfft.irfft((FW[None] * FMc[_NB:]).sum(1), n=n, axis=-1, workers=-1)
        return _lsq_r2(cJ.T.astype(np.float64), _gram(cW, n), EI)

    def candidates(self, libs, peaks=2, cancel=None, progress=None, frac=(0.25, 0.4)):
        out = []
        for i, lib in enumerate(libs):
            _check(cancel)
            if progress is not None and i % 10 == 0:
                _report(progress, frac[0] + (frac[1] - frac[0]) * i / len(libs), "Searching around the limb")
            FMc = self._lib_spectra(_lib_B(lib))
            for limb, F in self.variants:
                for parity in (1, -1):
                    sc, beta = self._scores(F[parity], FMc)
                    s2 = sc.copy()
                    for _ in range(peaks):
                        k = int(np.argmax(s2))
                        th = math.radians(k * 360.0 / self.NANG) * (1 if parity == 1 else -1)
                        out.append(dict(cx=limb["cx"], cy=limb["cy"], R=limb["R"], theta=th, parity=parity,
                                        lib=lib, score=float(sc[k])))
                        d = max(1, self.NANG * 3 // 360)
                        s2[np.arange(k - d, k + d + 1) % self.NANG] = -np.inf
        out.sort(key=lambda c: -c["score"])
        return out


def _aff3(A):
    return np.vstack([A, [0.0, 0.0, 1.0]])


class _SunFreePolish:
    """Local Sun-free refinement of one candidate: translation by FFT with the
    exact per-shift fit, rotation / scale / libration by small grids. Map
    spectra are cached per libration on a fixed window, so rotation and scale
    steps only re-warp the photo."""

    def __init__(self, mono, maps, cand, km_px=4.0, s1=1.0, s2=4.0, margin=24, room=48):
        import scipy.fft as sfft
        self.sfft = sfft
        self.maps, self.s1, self.s2, self.m = maps, s1, s2, int(margin)
        self.Rr = R_MOON / km_px
        self.f = f = self.Rr / cand["R"]
        small = cv2.resize(mono, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        self.T = _dog(small, s1, s2).astype(np.float32)
        self.W = _usable(small, s2)
        h, w = self.T.shape
        self.gc = (2 * int(self.Rr) + 40 - 1) / 2.0
        Ai = cv2.invertAffineTransform(self._affine(cand))
        corners = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]], float) @ Ai.T
        pad = margin + room + 0.06 * max(w, h)
        self.x0 = int(math.floor(corners[:, 0].min() - pad))
        self.y0 = int(math.floor(corners[:, 1].min() - pad))
        self.wR = sfft.next_fast_len(int(math.ceil(corners[:, 0].max() + pad)) - self.x0)
        self.hR = sfft.next_fast_len(int(math.ceil(corners[:, 1].max() + pad)) - self.y0)
        self._spec = {}

    def _affine(self, c, dth=0.0, ds=0.0):
        f = self.f
        A = _ref_to_img_affine((c["cx"] * f, c["cy"] * f, c["R"] * f, c["theta"]), c["parity"], self.Rr,
                               (self.gc, self.gc))
        if dth == 0.0 and ds == 0.0:
            return A
        h, w = self.T.shape
        q0 = cv2.invertAffineTransform(A) @ np.array([w / 2.0, h / 2.0, 1.0])
        cs, sn = math.cos(dth), math.sin(dth)
        Tq = np.eye(3)
        Tq[:2, 2] = q0
        Sm = np.eye(3)
        Sm[:2, :2] = np.array([[cs, -sn], [sn, cs]]) / (1.0 + ds)
        Tm = np.eye(3)
        Tm[:2, 2] = -q0
        return (_aff3(A) @ Tq @ Sm @ Tm)[:2]

    def _spectra(self, c):
        B = c["B"] if c.get("B") is not None else _lib_B(c["lib"])
        key = tuple(np.round(B.ravel(), 7))
        if key not in self._spec:
            M = _basis(B, self.Rr, self.gc - self.x0, self.gc - self.y0, self.hR, self.wR, self.maps)
            Md = [_dog(M[k], self.s1, self.s2) for k in range(_NB)]
            self._spec[key] = self.sfft.rfft2(np.stack(Md + [Md[k] * Md[l] for k, l in _PAIRS]), workers=-1)
        return self._spec[key]

    def evaluate(self, c, dth=0.0, ds=0.0, margin=None):
        m = self.m if margin is None else int(margin)
        hR, wR, sfft = self.hR, self.wR, self.sfft
        A2 = self._affine(c, dth, ds)
        A2r = A2.copy()
        A2r[:, 2] += A2[:, :2] @ np.array([self.x0, self.y0], float)
        Tw = cv2.warpAffine(self.T, A2r, (wR, hR), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        Ww = cv2.warpAffine(self.W, A2r, (wR, hR), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        if Ww.sum() < 200:
            return None
        FM = self._spectra(c)
        WT = Ww * Tw
        cc = sfft.irfft2(np.conj(sfft.rfft2(WT, workers=-1))[None] * FM[:_NB], s=(hR, wR), workers=-1)
        gg = sfft.irfft2(np.conj(sfft.rfft2(Ww, workers=-1))[None] * FM[_NB:], s=(hR, wR), workers=-1)
        iy = np.r_[0:m + 1, hR - m:hR]
        ix = np.r_[0:m + 1, wR - m:wR]
        cc = cc[:, iy][:, :, ix]
        gg = gg[:, iy][:, :, ix]
        n = cc.shape[1] * cc.shape[2]
        r2, beta = _lsq_r2(cc.reshape(_NB, -1).T.astype(np.float64), _gram(gg, n), float((WT * Tw).sum()))
        j = int(np.argmax(r2))
        ty, tx = iy[j // len(ix)], ix[j % len(ix)]
        ty = ty if ty <= m else ty - hR
        tx = tx if tx <= m else tx - wR
        A3 = A2.copy()
        A3[:, 2] -= A2[:, :2] @ np.array([tx, ty], float)      # content at x sits at map x + t
        Lm = A3[:, :2] @ np.diag([c["parity"], 1.0])
        ctr = A3[:, :2] @ np.array([self.gc, self.gc]) + A3[:, 2]
        out = dict(c)
        out.update(cx=float(ctr[0]) / self.f, cy=float(ctr[1]) / self.f,
                   R=math.sqrt(abs(np.linalg.det(Lm))) * self.Rr / self.f,
                   theta=math.atan2(Lm[1, 0], Lm[0, 0]), r2=float(r2[j]), beta=beta[j])
        return out

    def run(self, c, steps, rounds=2, cancel=None):
        """steps: (libration deg, rotation deg, scale) per level, coordinate grids."""
        best = self.evaluate(c)
        if best is None:
            return None
        for lstep, tstep, sstep in steps:
            for _ in range(rounds):
                _check(cancel)
                improved = False
                for dth in (-tstep, 0.0, tstep):
                    for ds in (-sstep, 0.0, sstep):
                        if dth == 0.0 and ds == 0.0:
                            continue
                        r = self.evaluate(best, dth=math.radians(dth), ds=ds)
                        if r is not None and r["r2"] > best["r2"] + 1e-4:
                            best, improved = r, True
                if lstep and best.get("B") is None:
                    for da in (-lstep, 0.0, lstep):
                        for db in (-lstep, 0.0, lstep):
                            if da == 0.0 and db == 0.0:
                                continue
                            c2 = dict(best)
                            c2["lib"] = (best["lib"][0] + da, best["lib"][1] + db)
                            r = self.evaluate(c2)
                            if r is not None and r["r2"] > best["r2"] + 1e-4:
                                best, improved = r, True
                if not improved:
                    break
        return best


def _polish_km(c, shape, cap=480):
    """Working scale: 4 km/px, coarser when the photo would exceed `cap` px
    (a whole disc: 360 px, its features are big)."""
    if max(shape) >= 1.6 * c["R"]:
        cap = min(cap, 360)
    field_km = (R_MOON / c["R"]) * max(shape)
    if field_km < cap * 4.0:
        # small fields: down to the reference maps' own 1.9 km, at least ~120 px across
        return max(2.0, min(4.0, field_km / 120.0))
    return field_km / cap


def _cand_place(c, shape, ref_px):
    B = c["B"] if c.get("B") is not None else _lib_B(c["lib"])
    sol = LunarSolution("blind", c["cx"], c["cy"], c["R"], c["theta"], c["parity"], B, np.array([1.0, 0, 0]),
                        _MEAN_DIST_KM, "", 0.0, 0, (0.0, 0.0), shape)
    la, lo = sol.pixel_to_latlon(ref_px)
    return _latlon_vec(la, lo).reshape(3)


def _same_place(a, b, deg=3.0):
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return False
    return math.degrees(math.acos(max(-1.0, min(1.0, float(a @ b))))) < deg


def _distinct(cands, shape, ref_px, K, deg=3.0):
    out = []
    for c in cands:
        try:
            v = _cand_place(c, shape, ref_px)
        except Exception:
            continue
        if any(o["parity"] == c["parity"] and _same_place(v, o["_v"], deg) for o in out):
            continue
        c = dict(c)
        c["_v"] = v
        out.append(c)
        if len(out) >= K:
            break
    return out


def _sunfree_select(mono, maps, cands, ref_px, cancel=None):
    """Polish every candidate a little, merge those that converge on the same
    place, polish the leaders fully. -> (best, runner-up R^2)."""
    pol = []
    for c in cands:
        _check(cancel)
        try:
            lp = _SunFreePolish(mono, maps, c, km_px=_polish_km(c, mono.shape))
            r = lp.run(c, steps=((2.0, 2.0, 0.02),), rounds=1, cancel=cancel)
        except LunarSolveCancelled:
            raise
        except Exception:
            r = None                    # a candidate off the disc
        if r is not None:
            r["_lp"] = lp
            pol.append(r)
    if not pol:
        return None, 0.0
    pol.sort(key=lambda r: -r["r2"])
    uniq = _distinct(pol, mono.shape, ref_px, K=len(pol))
    lead = [uniq[0]] + [u for u in uniq[1:2] if u["r2"] * _SF_MIN_RATIO > uniq[0]["r2"]]
    top = [u["_lp"].run(u, steps=((1.0, 1.0, 0.01), (0.5, 0.5, 0.005)), rounds=1, cancel=cancel) for u in lead]
    top.sort(key=lambda r: -r["r2"])
    best = top[0]
    runner = max([r["r2"] for r in top[1:]] + [u["r2"] for u in uniq[len(lead):]] + [0.0])
    return best, runner


def _limb_is_real(mono, limb):
    """A lit limb is bright just inside its arc and sky-dark just outside. The
    circle _fit_limb finds in a close-up with no limb follows some brightness
    contour instead, with the same brightness on both sides (measured: real
    limbs 0.58-0.96 inside and 0.01-0.03 outside, as a fraction of the 99.5th
    percentile; false ones about 0.25 on both sides)."""
    u8 = (mono * 255).astype(np.uint8)
    lo, hi = np.percentile(u8, 0.5), np.percentile(u8, 99.5)
    disk = (u8 > lo + 0.25 * (hi - lo)).astype(np.uint8)
    cnts, _ = cv2.findContours(disk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return False
    c = np.vstack([k[:, 0, :] for k in cnts]).astype(np.float64)
    pts = c[np.abs(np.hypot(c[:, 0] - limb["cx"], c[:, 1] - limb["cy"]) - limb["R"]) < 2.0]
    if len(pts) < 20:
        return False
    H, W = mono.shape
    v = pts - np.array([limb["cx"], limb["cy"]])
    v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
    p99 = max(float(np.percentile(mono, 99.5)), 1e-6)

    def ring(off):
        q = pts + off * v
        ok = (q[:, 0] >= 0) & (q[:, 0] <= W - 1) & (q[:, 1] >= 0) & (q[:, 1] <= H - 1)
        if ok.sum() < 10:
            return None
        vals = cv2.remap(mono, q[ok, 0:1].astype(np.float32), q[ok, 1:2].astype(np.float32), cv2.INTER_LINEAR)
        return float(np.median(vals)) / p99

    inside = [ring(o) for o in (-12, -6)]
    outside = [ring(o) for o in (12, 24)]
    if any(x is None for x in inside + outside):
        return False
    return float(np.mean(inside)) >= 0.25 and float(np.mean(outside)) <= 0.08


def _limb_variants(mono, limb):
    """The limb's radius is poorly fixed by a short arc (the rest of the limb is
    unlit): keep the arc's middle and try +-10 % radius about it."""
    u8 = (mono * 255).astype(np.uint8)
    lo, hi = np.percentile(u8, 0.5), np.percentile(u8, 99.5)
    disk = (u8 > lo + 0.25 * (hi - lo)).astype(np.uint8)
    cnts, _ = cv2.findContours(disk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return [limb]
    c = np.vstack([k[:, 0, :] for k in cnts]).astype(np.float64)
    p = c[np.abs(np.hypot(c[:, 0] - limb["cx"], c[:, 1] - limb["cy"]) - limb["R"]) < 2.0]
    if len(p) < 10:
        return [limb]
    a = np.sort(np.arctan2(p[:, 1] - limb["cy"], p[:, 0] - limb["cx"]))
    gaps = np.diff(np.r_[a, a[0] + 2 * np.pi])
    span = 2 * np.pi - gaps.max()
    if span >= math.radians(90):
        return [limb]
    mid = a[(int(np.argmax(gaps)) + 1) % len(a)] + span / 2
    px = limb["cx"] + limb["R"] * math.cos(mid)
    py = limb["cy"] + limb["R"] * math.sin(mid)
    return [dict(limb, R=limb["R"] * k, cx=px - limb["R"] * k * math.cos(mid), cy=py - limb["R"] * k * math.sin(mid))
            for k in (1.0, 0.9, 1.1)]


def _sift_candidates(mono, maps, data_dir=None, progress=None, cancel=None, min_inliers=5):
    """Hypotheses from the cached Sun-angle SIFT scan, each with the libration
    its own inliers' foreshortening implies."""
    Rs = _SCAN_R
    feats = _scan_features(maps, data_dir, progress, cancel)
    crs = (_render_centre(Rs), _render_centre(Rs))
    cache = {}
    raw = []
    for i, c in enumerate(_SCAN_COLONGS):
        _check(cancel)
        _report(progress, 0.45 + 0.15 * i / len(_SCAN_COLONGS), "Matching surface features")
        # no limb in frame means the disc is at least as big as the photo, so
        # the scan render's radius (700) is reached by the 800 / 400 px scales;
        # 1600 px only added the slowest matching
        loc = _sift_locate(mono, feats[c], crs, Rs, img_cache=cache, img_features=5000, max_side=(800, 400, 200))
        if loc is None or loc["inliers"] < min_inliers:
            continue
        params = np.array(loc["params"], float)
        lib = (0.0, 0.0)
        X = (loc["pref"][:, 0] - crs[0]) / Rs
        Y = -(loc["pref"][:, 1] - crs[1]) / Rs
        ok = X * X + Y * Y < 0.99
        if ok.sum() >= 8:
            P = _with_relief(maps, _body_from_sky(_B0, X[ok], Y[ok]))
            p_fit, _ = _fit_model(params, loc["parity"], _B0, P, loc["pimg"][ok], _MEAN_DIST_KM / R_MOON)
            if np.degrees(np.abs(p_fit[4:6])).max() < 12.0:
                params, lib = p_fit, (math.degrees(p_fit[4]), math.degrees(p_fit[5]))
        raw.append(dict(cx=params[0], cy=params[1], R=params[2], theta=params[3], parity=loc["parity"], lib=lib,
                        score=float(loc["inliers"])))
    raw.sort(key=lambda r: -r["score"])
    return raw


def _usable_ref_px(mono):
    """A point on the usable surface (full-res px): candidates are compared by
    where they put it on the Moon."""
    f = min(1.0, 512.0 / max(mono.shape))
    small = cv2.resize(mono, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    yy, xx = np.nonzero(_usable(small, 2.0))
    if not len(xx):
        return None
    return np.array([[(xx.mean() + 0.5) / f - 0.5, (yy.mean() + 0.5) / f - 0.5]])


def _sunfree_confidence(mono, maps, params, parity, B):
    """Fraction of the photo the three maps explain at a given placement."""
    c = dict(cx=params[0], cy=params[1], R=params[2], theta=params[3], parity=parity, B=B, lib=(0.0, 0.0))
    try:
        r = _SunFreePolish(mono, maps, c, km_px=_polish_km(c, mono.shape), margin=2, room=8).evaluate(c)
    except Exception:
        return None
    return None if r is None else max(0.0, float(r["r2"]))


def _work_radius(params, shape, R_max, max_px):
    """Reference radius for patch matching: the photo's own scale up to R_max,
    smaller when the photo covers so much of the disc that the patch grid
    would span more than max_px reference pixels (a full disc at R 1500 is
    7 M px at full scale: 40 s of matching for no gain in accuracy)."""
    R_img = float(params[2])
    H, W = shape
    frac = min(1.0, (H * W) / (math.pi * R_img * R_img))
    R = min(float(R_max), max(400.0, R_img))
    area = frac * math.pi * R * R
    if area > max_px:
        R *= math.sqrt(max_px / area)
    # a small photo needs its own resolution, or too few patches fit in it
    # (a 400 px crop at R 1800 holds about 26)
    min_px = 2.0e5
    if H * W * (R / R_img) ** 2 < min_px:
        R = max(R, min(R_img, R_img * math.sqrt(min_px / (H * W)), 6000.0))
    return int(max(300.0, R))


def _placement_in_view(sol, view, maps):
    """Re-express a solution's placement in another view of the Moon (the
    date's true geometry): body points under a grid of pixels, a similarity
    to start, then the model fit with libration free."""
    H, W = sol.shape
    gx, gy = np.meshgrid(np.linspace(0.05 * W, 0.95 * W, 15), np.linspace(0.05 * H, 0.95 * H, 15))
    uv = np.column_stack([gx.ravel(), gy.ravel()])
    la, lo = sol.pixel_to_latlon(uv)
    ok = np.isfinite(la) & np.isfinite(lo)
    if ok.sum() < 12:
        return None
    P = _with_relief(maps, _latlon_vec(la[ok], lo[ok]).reshape(-1, 3))
    q = P @ view.B.T
    sky = np.column_stack([sol.parity * q[:, 0], -q[:, 1]]).astype(np.float32)
    M, _ = cv2.estimateAffinePartial2D(sky, uv[ok].astype(np.float32))
    if M is None:
        return None
    p0 = np.array([M[0, 2], M[1, 2], math.hypot(M[0, 0], M[1, 0]), math.atan2(M[1, 0], M[0, 0]), 0.0, 0.0])
    params, _ = _fit_model(p0, sol.parity, view.B, P, uv[ok], view.d_lunar)
    return params


@njit(parallel=True, cache=True, nogil=True)
def _r2_kernel(c, g, E, valid, out):
    """Explained fraction per shift: v' G^-1 v / E with the symmetric 3x3 Gram
    in closed form (g: a b c d e f of [[a b c][b d e][c e f]])."""
    h, w = out.shape
    for y in prange(h):
        for x in range(w):
            if not valid[y, x]:
                out[y, x] = 0.0
                continue
            a = g[0, y, x]
            b = g[1, y, x]
            cc = g[2, y, x]
            d = g[3, y, x]
            e = g[4, y, x]
            f = g[5, y, x]
            A00 = d * f - e * e
            A01 = cc * e - b * f
            A02 = b * e - cc * d
            A11 = a * f - cc * cc
            A12 = b * cc - a * e
            A22 = a * d - b * b
            det = a * A00 + b * A01 + cc * A02
            # near-singular (flat ground, or the half-resolution Gram's small
            # errors where the maps are faint): no reliable fit there
            if a <= 0.0 or d <= 0.0 or f <= 0.0 or det <= 1e-3 * a * d * f:
                out[y, x] = 0.0
                continue
            v0 = c[0, y, x]
            v1 = c[1, y, x]
            v2 = c[2, y, x]
            num = (A00 * v0 * v0 + A11 * v1 * v1 + A22 * v2 * v2
                   + 2.0 * (A01 * v0 * v1 + A02 * v0 * v2 + A12 * v1 * v2))
            r = num / det / E
            out[y, x] = r if r <= 1.0 else 0.0


def _dense_candidates(mono, maps, R_list, cancel=None, progress=None, frac=(0.3, 0.6), rot_step=6.0, K=4, B=None):
    """Scale known (a hint): every rotation and both mirror states, position
    anywhere on the near side by FFT, each position scored by the exact
    Sun-free fit (approximate Grams were tried: a disc-shaped one scored wrong
    places 0.78 that score 0.20 exactly; a half-resolution one blew up on flat
    ground). The usable mask is made symmetric, so one Gram serves both
    mirror states and theta + 180. -> candidate dicts for _sunfree_select."""
    import scipy.fft as sfft
    H, W = mono.shape
    R0 = float(np.median(R_list))
    field_km = R_MOON / R0 * max(H, W)
    km = float(np.clip(field_km / 48.0, 2.0, 8.0))       # the photo ~48 px across, never finer than LOLA's 1.9 km
    Rr = R_MOON / km
    size = sfft.next_fast_len(2 * int(Rr) + 40)
    gc = (size - 1) / 2.0
    S1, S2 = 1.0, 4.0
    yy, xx = np.mgrid[0:size, 0:size]
    centre_r = np.hypot(xx - gc, yy - gc)
    disc = (centre_r < 0.96 * Rr).astype(np.float32)   # the limb's band-pass edge is not surface
    M = _basis(_B0 if B is None else B, Rr, gc, gc, size, size, maps)     # B: the date's libration, when known
    Md = np.stack([_dog(M[k], S1, S2) * disc for k in range(_NB)]).astype(np.float32)
    FM = sfft.rfft2(Md, workers=-1)
    FP = sfft.rfft2(np.stack([Md[k] * Md[l] for k, l in _PAIRS]), workers=-1)
    peaks = []
    n_half = int(round(180.0 / rot_step))
    total = max(1, len(R_list) * n_half)
    done = 0
    out = np.empty((size, size), np.float32)
    for R in R_list:
        f = Rr / float(R)
        small = cv2.resize(mono, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        T = _dog(small, S1, S2).astype(np.float32)
        Wm = _usable(small, S2)
        Ws = Wm * cv2.flip(Wm, -1) * cv2.flip(Wm, 0) * cv2.flip(Wm, 1)
        h, w = T.shape
        if float(Ws.sum()) < 200:
            continue
        rad = 0.5 * math.hypot(h, w)
        # with the photo's centre at canvas index 0 the correlation index IS where
        # that centre sits on the map: keep places where the whole photo is on the disc
        valid = centre_r <= (0.96 * Rr - rad)
        if not valid.any():
            continue
        side = int(2 * math.ceil(rad) + 4)
        o = side // 2
        idx = (np.arange(side) - o) % size
        Tp = {1: Ws * T, -1: Ws * cv2.flip(T, 1)}
        E = float((Ws * T * T).sum())

        def _spectrum(img, A, flags):
            s_ = cv2.warpAffine(img, A, (side, side), flags=flags | cv2.WARP_INVERSE_MAP)
            canv = np.zeros((size, size), np.float32)
            canv[np.ix_(idx, idx)] = s_
            return np.conj(sfft.rfft2(canv, workers=-1))

        for i in range(n_half):
            _check(cancel)
            _report(progress, frac[0] + (frac[1] - frac[0]) * done / total, "Searching the whole near side")
            done += 1
            th0 = math.radians(i * rot_step)
            A0 = _ref_to_img_affine(((w - 1) / 2.0, (h - 1) / 2.0, Rr, th0), 1, Rr, (o, o))
            g = sfft.irfft2(_spectrum(Ws, A0, cv2.INTER_NEAREST)[None] * FP, s=(size, size), workers=-1)
            for th in (th0, th0 + math.pi):
                A = _ref_to_img_affine(((w - 1) / 2.0, (h - 1) / 2.0, Rr, th), 1, Rr, (o, o))
                for parity in (1, -1):
                    c = sfft.irfft2(_spectrum(Tp[parity], A, cv2.INTER_LINEAR)[None] * FM, s=(size, size),
                                    workers=-1)
                    _r2_kernel(c, g, E, valid, out)
                    q = out
                    for _ in range(2):                     # two best places per hypothesis
                        j = int(np.argmax(q))
                        ty, tx = divmod(j, size)
                        peaks.append((float(q.flat[j]), parity, th, float(R), tx - gc, ty - gc, f, w, h))
                        r_ = int(rad)
                        q[max(0, ty - r_):ty + r_ + 1, max(0, tx - r_):tx + r_ + 1] = 0.0
    peaks.sort(key=lambda p: -p[0])
    res = []
    for sc, parity, th, R, tx, ty, f, w, h in peaks[:6 * K]:
        # the photo's centre sits at map point (gc + tx, gc + ty); ref -> small image
        # maps that to the photo's centre pixel, so the disc centre (gc, gc) lands there - M t
        A = _ref_to_img_affine(((w - 1) / 2.0, (h - 1) / 2.0, Rr, th), 1, Rr, (gc, gc))
        cs = A[:, :2] @ np.array([tx, ty], float)
        sx, sy = (w - 1) / 2.0 - cs[0], (h - 1) / 2.0 - cs[1]
        if parity == -1:                                   # undo the flip: x -> (w - 1) - x
            sx = (w - 1) - sx
        res.append(dict(cx=(sx + 0.5) / f - 0.5, cy=(sy + 0.5) / f - 0.5, R=R, theta=th if parity == 1 else -th,
                        parity=parity, lib=(0.0, 0.0), B=B, score=sc))
    return res


# Below this field size the near-side search found nothing in the 2026-11-28 crop
# tests (0 of 46 at 100 km; 35 of 46 at 180 km) and cost 40-90 s per image.
_DENSE_MIN_KM = 140.0


def _hint_radii(scale_arcsec, view=None):
    """Apparent lunar radius in pixels from an image scale. Without a date the
    distance is unknown (356 400 - 406 700 km): two radii 7 % apart cover it,
    the polish takes up the rest."""
    if not scale_arcsec or scale_arcsec <= 0:
        return []
    D = view.D if view is not None else _MEAN_DIST_KM
    R = math.degrees(math.asin(R_MOON / D)) * 3600.0 / float(scale_arcsec)
    return [R] if view is not None else [R * 0.965, R * 1.035]


def _solve_blind(mono, maps, progress, cancel, data_dir=None, scale_arcsec=None, view=None):
    """No date. Candidates come from a Sun-free search around the limb when one
    is in frame (polar FFT over rotation, mirror and libration) and from the
    cached Sun-angle SIFT scan when that is not decisive. Each is polished and
    judged by how much of the photo the three Sun-free maps explain; the
    winner's fitted mix gives the Sun, then the rendered refinement and the
    Sun-angle check finish it. -> (params, parity, res, view, sun_sure,
    confidence, runner_up) or None."""
    ref_px = _usable_ref_px(mono)
    if ref_px is None:
        return None
    results = []
    limb = _fit_limb(mono)
    if limb is not None and limb["n"] >= 200 and limb["rms"] < 2.5 and _limb_is_real(mono, limb):
        _check(cancel)
        _report(progress, 0.22, "Searching around the limb")
        try:
            H, W = mono.shape
            whole = 0 <= limb["cx"] < W and 0 <= limb["cy"] < H
            # a disc with its centre in frame carries plenty of signal at a coarser grid
            ps = _PolarSearch(mono, _limb_variants(mono, limb), maps, *((200, 1440) if whole else (300, 2048)))
            cands = _distinct(ps.candidates(_POLAR_LIBS, cancel=cancel, progress=progress), mono.shape, ref_px, K=3)
        except LunarSolveCancelled:
            raise
        except LunarSolveError:
            cands = []
        if cands:
            _report(progress, 0.42, "Checking the best placements")
            b, r = _sunfree_select(mono, maps, cands, ref_px, cancel)
            if b is not None:
                results.append((b, r))
    # the limb search alone settles it when it clears the acceptance test with room
    decisive = bool(results) and results[0][0]["r2"] >= 1.5 * _SF_MIN_R2 and \
        results[0][0]["r2"] >= 1.2 * _SF_MIN_RATIO * results[0][1]
    if not decisive:
        cands = _distinct(_sift_candidates(mono, maps, data_dir, progress, cancel), mono.shape, ref_px, K=6)
        if cands:
            _report(progress, 0.6, "Checking the best placements")
            b, r = _sunfree_select(mono, maps, cands, ref_px, cancel)
            if b is not None:
                results.append((b, r))
    def _pick():
        if not results:
            return None, 0.0
        results.sort(key=lambda t: -t[0]["r2"])
        b0, run = results[0]
        for o, orun in results[1:]:
            run = max(run, orun)
            if not (o["parity"] == b0["parity"] and _same_place(_cand_place(o, mono.shape, ref_px),
                                                               _cand_place(b0, mono.shape, ref_px))):
                run = max(run, o["r2"])
        return b0, run

    best, runner = _pick()
    accepted = best is not None and best["r2"] >= _SF_MIN_R2 and best["r2"] >= _SF_MIN_RATIO * runner
    radii = _hint_radii(scale_arcsec, view) if scale_arcsec else []
    if radii and R_MOON / float(np.median(radii)) * max(mono.shape) < _DENSE_MIN_KM:
        radii = []                       # too small a field for it: fail in seconds, not minutes
    if not accepted and radii:
        # small fields give feature matching too little to work with; with the
        # image scale known, search every rotation and position instead
        cands = _distinct(_dense_candidates(mono, maps, radii, cancel, progress,
                                            B=None if view is None else view.B),
                          mono.shape, ref_px, K=4)
        if cands:
            _report(progress, 0.62, "Checking the best placements")
            b, r = _sunfree_select(mono, maps, cands, ref_px, cancel)
            if b is not None:
                results.append((b, r))
        best, runner = _pick()
    if best is None or best["r2"] < _SF_MIN_R2 or best["r2"] < _SF_MIN_RATIO * runner:
        return None
    parity = best["parity"]
    lib = np.radians(np.array(best["lib"], float))
    B_fix = best.get("B")               # a dense search run with the date's libration
    D_v = view.D if view is not None else _MEAN_DIST_KM
    beta = np.asarray(best["beta"], float)
    s = beta / max(float(np.linalg.norm(beta)), 1e-12)
    c_fit = (90.0 - float(_vec_latlon(s)[1])) % 360.0
    params = np.array([best["cx"], best["cy"], best["R"], best["theta"], 0.0, 0.0])
    Re = _work_radius(params, mono.shape, 1000, 1.0e6)

    def _view_at(colong):
        if B_fix is None:
            return _mean_view(colong, lib)
        return _View(B=B_fix, S=_latlon_vec(0.0, 90.0 - colong), D=D_v, utc="")

    _check(cancel)
    _report(progress, 0.7, "Matching the surface")
    view = _view_at(c_fit)
    p1, res = _refine_on(mono, params, parity, view, maps, 1800, cancel=cancel, free_lib=True, min_score=0.25)
    p1, res, reg = _detail_fallback(mono, params, parity, view, maps, (p1, res), data_dir, progress, cancel, 0.74,
                                    free_lib=True, min_score=0.25)
    if res is None:
        return None
    lib = lib + p1[4:6]
    if B_fix is not None:
        B_fix = _rot_small(p1[4], p1[5]) @ B_fix
    params = p1.copy()
    params[4:6] = 0.0
    # the Sun angle: the fit's estimate, checked against renders with shadows
    _report(progress, 0.8, "Checking the Sun angle")
    roi_e = _footprint_roi(params, parity, Re, mono.shape)
    scores = {}

    def _score(c):
        _check(cancel)
        refc, mc, crc = _render(_view_at(c), Re, maps, roi=roi_e)
        scores[c] = _evaluate(mono, params, parity, refc, mc, crc, Re)

    for c in np.arange(c_fit - 6.0, c_fit + 6.1, 3.0):
        _score(float(c))
    for _ in range(4):                   # best at the window's edge: look further that way
        ks = sorted(scores)
        kb = max(scores, key=scores.get)
        if kb == ks[0]:
            _score(kb - 3.0)
        elif kb == ks[-1]:
            _score(kb + 3.0)
        else:
            break
    ks = sorted(scores)
    kb = max(scores, key=scores.get)
    left = [scores[k] for k in ks if k < kb]
    right = [scores[k] for k in ks if k > kb]
    # a real peak falls off on both sides (0.11-0.24 near the terminator, under
    # 0.015 on fields 50 deg from it, where the shading barely changes)
    sun_sure = bool(left and right and min(left) <= scores[kb] - 0.015 and min(right) <= scores[kb] - 0.015)
    if sun_sure:
        for c in (kb - 1.0, kb + 1.0):
            _score(c)
        c_best = max(scores, key=scores.get)
    else:
        c_best = c_fit                   # flat: the fitted mix is the better guess
    view = _view_at(c_best)
    _report(progress, 0.88, "Final fit")
    fin = dict(passes=2, search0=8, free_lib=True)
    if reg is not None:
        p1, res = _refine_on(mono, params, parity, view, reg, _R_HI, cancel=cancel, **fin)
    else:
        p1, res = _refine_on(mono, params, parity, view, maps, 1800, cancel=cancel, **fin)
        p1, res, _ = _detail_fallback(mono, params, parity, view, maps, (p1, res), data_dir, progress, cancel, 0.92,
                                      **fin)
    return p1, parity, res, view, sun_sure, float(best["r2"]), float(runner)


def solve_moon(image, when_utc=None, site=None, *, progress: Optional[Callable] = None,
               cancel: Optional[Callable] = None, data_dir=None, scale_arcsec=None) -> LunarSolution:
    """Place a lunar image on the Moon.

    image     2-D or HxWxC array, any dtype.
    when_utc  capture time (datetime: naive = UTC; ISO string) or None when
              unknown, which runs the slower blind search.
    site      (lat_deg, lon_east_deg, height_m) or None (geocentre).
    scale_arcsec  image scale hint (arcsec/px) or None. Lets small fields be
              found by a dense search when feature matching finds nothing.
    progress  progress(fraction, message); cancel() -> True to stop.

    Raises LunarSolveError (LunarSolveCancelled when cancelled)."""
    if cv2 is None:
        raise LunarSolveError("OpenCV is not available.")
    t0 = time.perf_counter()
    maps = _maps(progress, cancel, data_dir)
    _check(cancel)
    mono = _to_mono01(image)
    H, W = mono.shape
    when = _as_utc(when_utc)
    notes = []

    def _build(mode, params, parity, res, view):
        rms = _accept(params, res, {"disc": "Full disc", "closeup": "Close-up", "blind": "Blind search"}[mode])
        B = _rot_small(params[4], params[5]) @ view.B
        n_ = list(notes)
        n_.append("solved in %.1f s" % (time.perf_counter() - t0))
        conf = _sunfree_confidence(mono, maps, params, parity, B)
        return LunarSolution(mode=mode, cx=float(params[0]), cy=float(params[1]), radius_px=float(params[2]),
                             theta=float(params[3]), parity=int(parity), B=B, S=np.asarray(view.S, np.float64),
                             D=float(view.D), utc=view.utc, rms_px=rms, n_matches=int(len(res)),
                             libration_correction_deg=(math.degrees(params[4]), math.degrees(params[5])),
                             shape=(H, W), site=tuple(site) if site else None, notes=n_, confidence=conf)

    if when is None:
        _report(progress, 0.2, "No capture time: searching the Sun angle")
        out = _solve_blind(mono, maps, progress, cancel, data_dir=data_dir, scale_arcsec=scale_arcsec)
        if out is None:
            raise LunarSolveError(
                "Could not place the image on the Moon without a capture time. Enter the date and time "
                "of the capture and solve again.")
        params, parity, res, view, sun_sure, _conf, runner = out
        sol = _build("blind", params, parity, res, view)
        sol.sun_approximate = not sun_sure
        sol.runner_up = runner
        return sol

    _report(progress, 0.2, "Computing the Moon's geometry")
    G = lunar_geometry(when, site=tuple(site) if site else None)
    notes.extend(getattr(G, "notes", []) or [])
    view = _view_from_geometry(G)
    _check(cancel)
    limb = _fit_limb(mono)
    errors = []
    if (limb is not None and limb["n"] >= 200 and limb["rms"] < 2.5
            and 0 <= limb["cx"] < W and 0 <= limb["cy"] < H):
        out = _solve_disc(mono, limb, view, maps, progress, cancel)
        if out is not None:
            try:
                return _build("disc", out[0], out[1], out[2], view)
            except LunarSolveError as ex:
                errors.append(str(ex))
    _check(cancel)
    out = _solve_closeup(mono, view, maps, progress, cancel, data_dir=data_dir)
    if out is not None:
        try:
            return _build("closeup", out[0], out[1], out[2], view)
        except LunarSolveError as ex:
            errors.append(str(ex))
    # the dated paths lean on feature matching; the Sun-free search finds
    # fields they miss. Run it, then finish with the date's exact geometry.
    _check(cancel)
    _report(progress, 0.5, "Searching with the Sun-free maps")
    ob = _solve_blind(mono, maps, progress, cancel, data_dir=data_dir, scale_arcsec=scale_arcsec, view=view)
    if ob is not None:
        try:
            p_b, par_b, res_b, view_b = ob[:4]
            p_t = _placement_in_view(_build("blind", p_b, par_b, res_b, view_b), view, maps)
            if p_t is not None:
                _report(progress, 0.9, "Final fit with the capture time")
                params, res = _refine_on(mono, p_t, par_b, view, maps, 1800, cancel=cancel)
                params, res, _ = _detail_fallback(mono, p_t, par_b, view, maps, (params, res), data_dir, progress,
                                                  cancel, 0.93)
                return _build("closeup", params, par_b, res, view)
        except LunarSolveCancelled:
            raise
        except LunarSolveError as ex:
            errors.append(str(ex))
    raise LunarSolveError(
        "Could not place the image on the Moon." + (" " + " ".join(errors) if errors else
        " No part of it matched the predicted Moon for that time. Check the capture date and time; "
        "if they are unknown, clear the date and solve again."))
