# src/setiastro/saspro/lunar_relief.py
# Copyright (c) Franklin Marek / SetiAstro (www.setiastro.com). GPLv3.
"""
Lunar Relief: real 3D terrain for a registered lunar image.

Two height sources, one geometry:

* LOLA    -- drape the user's image on NASA's LRO laser-altimeter terrain
             (LDEM_16 / LDEM_64 from the PDS Geosciences node).
* Shadows -- measure the terrain from the image itself: every shadow, together
             with the Sun's position for the capture time, gives the height of
             the crest that casts it.

Everything is in the Moon's Mean-Earth / polar-axis body frame (the frame LOLA
uses), in km, east-positive longitude.  Image registration is NOT done here: the
consumers below take any camera object with

    camera.O                  observer position, body frame, km
    camera.rays(uv)           (...,2) pixels -> (...,3) unit ray directions
    camera.project(X)         (...,3) body points -> ((...,2) pixels, visible)

plus a LunarGeometry for the Sun direction.  `GnomonicCamera` implements that for
a full-disk solution (centre, radius, rotation, mirror).

Geometry: astropy's built-in ephemeris (no download) with IERS kept offline, and
the IAU WGCCRE 2009 lunar rotation model, which matches NAIF's MOON_ME frame to
~0.003 deg -- far below what a telescope image can resolve.

Measured on Frank's 2023-05-01 full disk (1.15 km/px, seeing sigma 2.4 px):
shadow heights are 16 % low with ~12 % scatter against LOLA.  The low bias is the
seeing blur filling in the tapered shadow tips; a LOLA render blurred to the same
PSF and measured with the same code shows the same -15 %.
"""
from __future__ import annotations

import math
import socket
import time
import urllib.error
import urllib.request
import warnings
from contextlib import ExitStack, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

from numba import njit, prange

R_MOON = 1737.4            # km, LOLA reference sphere (LDEM OFFSET)
SUN_RADIUS_RAD = math.radians(0.2665)
C_KM_S = 299792.458

LOLA_BASE_URL = ("https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/"
                 "lrolol_1xxx/data/lola_gdr/cylindrical/img/")
_UA = {"User-Agent": "SetiAstroSuitePro/1.0"}


def _parallel_lock():
    try:
        from setiastro.saspro.legacy.numba_utils import NUMBA_PARALLEL_LOCK
        return NUMBA_PARALLEL_LOCK
    except Exception:
        return nullcontext()


def _unit(v):
    v = np.asarray(v, np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):     # NaN rows = off the disc
        return v / np.linalg.norm(v, axis=-1, keepdims=True)


def sph_to_body(lat_deg, lon_deg, r=R_MOON):
    la, lo = np.radians(lat_deg), np.radians(lon_deg)
    return np.stack([r * np.cos(la) * np.cos(lo), r * np.cos(la) * np.sin(lo),
                     r * np.sin(la) * np.ones_like(lo)], -1)


def body_to_latlon(X):
    X = np.asarray(X, np.float64)
    r = np.linalg.norm(X, axis=-1)
    return (np.degrees(np.arcsin(X[..., 2] / r)),
            np.degrees(np.arctan2(X[..., 1], X[..., 0])))


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def _iers_offline():
    """Keep astropy from fetching IERS tables (a GUI tool must not stall on the
    network) and let it use stale bundled predictions.  UT1-UTC only moves the
    observer by < 0.5 km here."""
    st = ExitStack()
    try:
        from astropy.utils import iers
    except Exception:
        return st
    for key, val in (("auto_download", False), ("auto_max_age", None),
                     ("iers_degraded_accuracy", "warn")):
        if hasattr(iers.conf, key):
            try:
                st.enter_context(iers.conf.set_temp(key, val))
            except Exception:
                pass
    return st


def iau_moon_rotation(jd_tdb: float) -> np.ndarray:
    """ICRF -> Moon body (IAU WGCCRE 2009 model, ~ Mean-Earth frame)."""
    d = float(jd_tdb) - 2451545.0
    T = d / 36525.0
    E = np.radians(np.array([
        125.045 - 0.0529921 * d, 250.089 - 0.1059842 * d, 260.008 + 13.0120009 * d,
        176.625 + 13.3407154 * d, 357.529 + 0.9856003 * d, 311.589 + 26.4057084 * d,
        134.963 + 13.0649930 * d, 276.617 + 0.3287146 * d, 34.226 + 1.7484877 * d,
        15.134 - 0.1589763 * d, 119.743 + 0.0036096 * d, 239.961 + 0.1643573 * d,
        25.053 + 12.9590088 * d]))
    s, c = np.sin(E), np.cos(E)
    a0 = (269.9949 + 0.0031 * T - 3.8787 * s[0] - 0.1204 * s[1] + 0.0700 * s[2]
          - 0.0172 * s[3] + 0.0072 * s[5] - 0.0052 * s[9] + 0.0043 * s[12])
    d0 = (66.5392 + 0.0130 * T + 1.5419 * c[0] + 0.0239 * c[1] - 0.0278 * c[2]
          + 0.0068 * c[3] - 0.0029 * c[5] + 0.0009 * c[6] + 0.0008 * c[9]
          - 0.0009 * c[12])
    W = (38.3213 + 13.17635815 * d - 1.4e-12 * d * d + 3.5610 * s[0] + 0.1208 * s[1]
         - 0.0642 * s[2] + 0.0158 * s[3] + 0.0252 * s[4] - 0.0066 * s[5]
         - 0.0047 * s[6] - 0.0046 * s[7] + 0.0028 * s[8] + 0.0052 * s[9]
         + 0.0040 * s[10] + 0.0019 * s[11] - 0.0044 * s[12])

    def rz(a):
        a = math.radians(a)
        return np.array([[math.cos(a), math.sin(a), 0.0],
                         [-math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])

    def rx(a):
        a = math.radians(a)
        return np.array([[1.0, 0.0, 0.0], [0.0, math.cos(a), math.sin(a)],
                         [0.0, -math.sin(a), math.cos(a)]])
    return rz(W) @ rx(90.0 - d0) @ rz(90.0 + a0)


@dataclass
class LunarGeometry:
    """Viewing / illumination geometry at one instant, Moon body frame (km)."""
    utc: str
    O: np.ndarray            # observer position
    S: np.ndarray            # unit vector Moon -> Sun
    d: np.ndarray            # unit observer -> Moon centre
    n: np.ndarray            # celestial north on the sky
    e: np.ndarray            # celestial east on the sky
    D: float                 # observer - Moon centre distance
    sub_observer: tuple      # (lat, lon) deg
    sub_solar: tuple         # (lat, lon) deg
    colongitude: float       # deg
    topocentric: bool = True
    notes: list = field(default_factory=list)

    @property
    def angular_radius(self) -> float:
        """Apparent lunar radius, radians."""
        return math.asin(R_MOON / self.D)

    def sun_elevation_deg(self, X) -> np.ndarray:
        """Sun elevation above the mean-sphere horizon at body points X."""
        return np.degrees(np.arcsin(np.clip(_unit(X) @ self.S, -1, 1)))


def _naive_utc(when):
    """datetime / ISO string (with or without a UTC offset) -> naive UTC datetime.
    astropy refuses ISO strings carrying an offset such as '+00:00'."""
    import datetime as _dt
    if isinstance(when, str):
        s = when.strip().replace(" ", "T")
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        when = _dt.datetime.fromisoformat(s)
    if isinstance(when, _dt.datetime) and when.tzinfo is not None:
        when = when.astimezone(_dt.timezone.utc).replace(tzinfo=None)
    return when


def lunar_geometry(when_utc, site: Optional[tuple] = None) -> LunarGeometry:
    """Geometry for a UTC instant.  `site` = (lat_deg, lon_east_deg, height_m)
    or None for the geocentre (then the sub-observer point can be off by up to
    ~1 deg of topocentric libration; registration should absorb it)."""
    from astropy.time import Time, TimeDelta
    from astropy.coordinates import get_body_barycentric, EarthLocation
    import astropy.units as u

    t = Time(_naive_utc(when_utc), scale="utc")
    notes = []
    topo = site is not None
    with _iers_offline(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        earth = get_body_barycentric("earth", t, ephemeris="builtin").xyz.to_value(u.km)
        obs = np.array(earth, np.float64)
        if site is not None:
            try:
                loc = EarthLocation.from_geodetic(float(site[1]) * u.deg, float(site[0]) * u.deg,
                                                  float(site[2] if len(site) > 2 else 0.0) * u.m)
                p, _v = loc.get_gcrs_posvel(t)
                obs = obs + p.xyz.to_value(u.km)
            except Exception as ex:   # pragma: no cover - IERS trouble
                topo = False
                notes.append(f"topocentric correction failed ({ex}); using the geocentre")
        moon = get_body_barycentric("moon", t, ephemeris="builtin").xyz.to_value(u.km)
        tau = float(np.linalg.norm(moon - obs)) / C_KM_S
        t_emit = t - TimeDelta(tau, format="sec")
        moon = get_body_barycentric("moon", t_emit, ephemeris="builtin").xyz.to_value(u.km)
        sun = get_body_barycentric("sun", t_emit, ephemeris="builtin").xyz.to_value(u.km)
        jd_tdb = float(t_emit.tdb.jd1) + float(t_emit.tdb.jd2)
    Rm = iau_moon_rotation(jd_tdb)
    O = Rm @ (obs - moon)
    S = _unit(Rm @ (sun - moon))
    d = _unit(-O)
    ncp = Rm @ np.array([0.0, 0.0, 1.0])
    n = _unit(ncp - np.dot(ncp, d) * d)
    e = np.cross(n, d)
    sub_obs = tuple(float(v) for v in body_to_latlon(O))
    sub_sun = tuple(float(v) for v in body_to_latlon(S))
    return LunarGeometry(utc=t.isot, O=O, S=S, d=d, n=n, e=e, D=float(np.linalg.norm(O)),
                         sub_observer=sub_obs, sub_solar=sub_sun,
                         colongitude=(90.0 - sub_sun[1]) % 360.0, topocentric=topo, notes=notes)


class GnomonicCamera:
    """Perspective camera for a registered lunar frame.

    pixel = c + k * Rot(phi) @ diag(m, 1) @ (xi, eta), image x right / y down,
    where (xi, eta) are gnomonic offsets from the Moon centre along sky east and
    north and k is px per radian.  mirror=False is the true sky (phi=0: north up,
    east LEFT, m = -1); mirror=True is its mirror image (diagonal etc., m = +1).
    """

    def __init__(self, geom: LunarGeometry, cx: float, cy: float, k: float,
                 phi_deg: float, mirror: bool):
        self.geom = geom
        self.cx, self.cy, self.k = float(cx), float(cy), float(k)
        self.phi_deg = float(phi_deg)
        self.mirror = bool(mirror)
        self._phi = math.radians(self.phi_deg)
        self._m = 1.0 if self.mirror else -1.0

    @classmethod
    def from_disk(cls, geom, cx, cy, radius_px, phi_deg, mirror):
        k = float(radius_px) / math.tan(geom.angular_radius)
        return cls(geom, cx, cy, k, phi_deg, mirror)

    @property
    def O(self):
        return self.geom.O

    @property
    def disk_radius_px(self) -> float:
        return self.k * math.tan(self.geom.angular_radius)

    def km_per_px(self) -> float:
        return (self.geom.D - R_MOON) / self.k

    def to_dict(self) -> dict:
        return dict(utc=self.geom.utc, cx=self.cx, cy=self.cy, k=self.k,
                    phi_deg=self.phi_deg, mirror=self.mirror)

    def project(self, X):
        g = self.geom
        X = np.asarray(X, np.float64)
        v = X - g.O
        z = v @ g.d
        xi = (v @ g.e) / z
        eta = (v @ g.n) / z
        c, s = math.cos(self._phi), math.sin(self._phi)
        a0, a1 = self._m * xi, eta
        u = self.cx + self.k * (c * a0 - s * a1)
        w = self.cy - self.k * (s * a0 + c * a1)
        vis = np.einsum("...i,...i->...", _unit(X), -_unit(v)) > 0
        return np.stack([u, w], -1), vis

    def rays(self, uv):
        g = self.geom
        uv = np.asarray(uv, np.float64)
        a0 = (uv[..., 0] - self.cx) / self.k
        a1 = (self.cy - uv[..., 1]) / self.k
        c, s = math.cos(self._phi), math.sin(self._phi)
        b0, b1 = c * a0 + s * a1, -s * a0 + c * a1
        xi, eta = self._m * b0, b1
        return _unit(g.d + xi[..., None] * g.e + eta[..., None] * g.n)

    def unproject_sphere(self, uv, r=R_MOON):
        """Pixels -> body points on a sphere of radius r (NaN off the disk)."""
        O = self.geom.O
        dr = self.rays(uv)
        b = dr @ O
        disc = b * b - (O @ O - r * r)
        t = -b - np.sqrt(np.where(disc >= 0, disc, np.nan))
        return O + t[..., None] * dr


# ---------------------------------------------------------------------------
# LOLA terrain
# ---------------------------------------------------------------------------

@dataclass
class DemBand:
    """Rows [row0, row0+data.shape[0]) of an equirectangular LDEM (int16, 0.5 m DN
    over the 1737.4 km sphere; row 0 = north pole, east longitude 0..360)."""
    data: np.ndarray
    ppd: float
    row0: int

    @property
    def lat_range(self):
        return (90.0 - (self.row0 + self.data.shape[0]) / self.ppd, 90.0 - self.row0 / self.ppd)


def is_network_error(ex) -> bool:
    """A download failure worth retrying / reporting as 'no connection':
    DNS, refused/reset connections, timeouts, server-side HTTP errors, short reads."""
    if isinstance(ex, urllib.error.HTTPError):
        return ex.code >= 500 or ex.code in (408, 429)
    if isinstance(ex, (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError)):
        return True
    s = str(ex).lower()
    return any(k in s for k in ("urlopen error", "getaddrinfo", "timed out", "short download",
                                "connection reset", "connection aborted", "temporary failure"))


def network_reason(ex) -> str:
    """Plain words for a network failure."""
    s = str(ex).lower()
    if "getaddrinfo" in s or "name or service" in s or "nodename" in s:
        return "the server name could not be looked up (DNS); the internet connection may be down"
    if "timed out" in s or isinstance(ex, (socket.timeout, TimeoutError)):
        return "the server did not answer in time"
    if isinstance(ex, urllib.error.HTTPError):
        return "the server answered with an error (HTTP %d)" % ex.code
    if "refused" in s or "reset" in s or "aborted" in s:
        return "the connection was refused or dropped"
    return str(ex)


class LolaDEM:
    """Download-on-demand LOLA cache.

    LDEM_16 (33 MB, whole globe, ~1.9 km/px) is fetched in one piece.  LDEM_64
    (~0.47 km/px, 531 MB) is fetched in 4-degree latitude chunks with HTTP Range
    requests, so a lunar image only pulls the latitudes it covers.  A complete
    LDEM_64.IMG dropped into the cache folder is used directly.
    """
    CHUNK_DEG = 4

    def __init__(self, cache_dir):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._g16 = None

    @staticmethod
    def default_dir() -> Path:
        from PyQt6.QtCore import QStandardPaths
        base = Path(QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppDataLocation))
        return base / "lola"

    # -- download helpers ----------------------------------------------------
    @staticmethod
    def _fetch(url, dest: Path, byte_range=None, progress=None, cancel=None, expected=None, attempts=4):
        """Download to `dest` (via a .part file). Transient network failures
        (DNS hiccups, timeouts, resets, 5xx) are retried after 1, 3 and 6 s."""
        for i in range(int(attempts)):
            try:
                return LolaDEM._fetch_once(url, dest, byte_range, progress, cancel, expected)
            except Exception as ex:
                if i == attempts - 1 or not is_network_error(ex) or (cancel is not None and cancel()):
                    raise
            t_end = time.monotonic() + (1.0, 3.0, 6.0)[min(i, 2)]
            while time.monotonic() < t_end:
                if cancel is not None and cancel():
                    raise RuntimeError("cancelled")
                time.sleep(0.1)

    @staticmethod
    def _fetch_once(url, dest: Path, byte_range, progress, cancel, expected):
        hdr = dict(_UA)
        if byte_range is not None:
            hdr["Range"] = "bytes=%d-%d" % byte_range
        req = urllib.request.Request(url, headers=hdr)
        tmp = dest.with_name(dest.name + ".part")
        got = 0
        try:
            with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as f:
                total = expected or int(resp.headers.get("Content-Length") or 0)
                while True:
                    if cancel is not None and cancel():
                        raise RuntimeError("cancelled")
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    got += len(chunk)
                    if progress is not None and total:
                        progress(got, total)
        except BaseException:
            # cancelled or dropped: no half file left behind (closed by now, so
            # the delete works on Windows too)
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            raise
        if expected and got != expected:
            tmp.unlink(missing_ok=True)
            raise IOError(f"short download from {url}: {got} of {expected} bytes")
        tmp.replace(dest)

    def has_global16(self) -> bool:
        return (self.cache_dir / "LDEM_16.IMG").is_file()

    def global16(self, progress=None, cancel=None) -> DemBand:
        if self._g16 is None:
            p = self.cache_dir / "LDEM_16.IMG"
            if not p.is_file():
                self._fetch(LOLA_BASE_URL + "LDEM_16.IMG", p, progress=progress, cancel=cancel,
                            expected=2880 * 5760 * 2)
            self._g16 = DemBand(np.fromfile(p, dtype="<i2").reshape(2880, 5760), 16.0, 0)
        return self._g16

    # -- LDEM_64 in 4-degree latitude chunks (kept once fetched) --------------
    def _has_full64(self) -> bool:
        full = self.cache_dir / "LDEM_64.IMG"
        return full.is_file() and full.stat().st_size == 11520 * 23040 * 2

    def chunk64_path(self, c) -> Path:
        """Cache file of LDEM_64 chunk c (rows c*256 .. c*256+255, lat 90-4c down)."""
        return self.cache_dir / f"LDEM_64_r{c * self.CHUNK_DEG * 64:05d}.i16"

    def chunks64(self, lat_lo=-90.0, lat_hi=90.0):
        """Chunk indices covering a latitude range."""
        lat_lo, lat_hi = max(-90.0, float(lat_lo)), min(90.0, float(lat_hi))
        r_a = int(math.floor((90.0 - lat_hi) * 64))
        r_b = int(math.ceil((90.0 - lat_lo) * 64))
        rows_per = self.CHUNK_DEG * 64
        return list(range(r_a // rows_per, (r_b - 1) // rows_per + 1))

    def missing64(self, lat_lo=-90.0, lat_hi=90.0):
        """Chunks of a latitude range not on disk yet ([] with a full LDEM_64.IMG)."""
        if self._has_full64():
            return []
        return [c for c in self.chunks64(lat_lo, lat_hi) if not self.chunk64_path(c).is_file()]

    def fetch64(self, lat_lo=-90.0, lat_hi=90.0, progress=None, cancel=None) -> int:
        """Download the missing chunks of a latitude range without loading them.
        progress(bytes done, bytes to fetch). -> number of chunks fetched."""
        ns = 23040
        rows_per = self.CHUNK_DEG * 64
        todo = self.missing64(lat_lo, lat_hi)
        done_bytes, total_bytes = 0, len(todo) * rows_per * ns * 2
        for c in todo:
            b0 = c * rows_per * ns * 2
            nb = rows_per * ns * 2

            def _p(g, t, _base=done_bytes):
                if progress is not None:
                    progress(_base + g, total_bytes)
            self._fetch(LOLA_BASE_URL + "LDEM_64.IMG", self.chunk64_path(c), byte_range=(b0, b0 + nb - 1),
                        progress=_p, cancel=cancel, expected=nb)
            done_bytes += nb
        return len(todo)

    def band64(self, lat_lo, lat_hi, progress=None, cancel=None) -> DemBand:
        ppd, ns, nl = 64, 23040, 11520
        lat_lo, lat_hi = max(-90.0, float(lat_lo)), min(90.0, float(lat_hi))
        full = self.cache_dir / "LDEM_64.IMG"
        r_a = int(math.floor((90.0 - lat_hi) * ppd))
        r_b = int(math.ceil((90.0 - lat_lo) * ppd))
        if self._has_full64():
            mm = np.memmap(full, dtype="<i2", mode="r", shape=(nl, ns))
            return DemBand(np.ascontiguousarray(mm[r_a:r_b]), float(ppd), r_a)
        rows_per = self.CHUNK_DEG * ppd
        c_a, c_b = r_a // rows_per, (r_b - 1) // rows_per
        self.fetch64(lat_lo, lat_hi, progress=progress, cancel=cancel)
        parts = [np.fromfile(self.chunk64_path(c), dtype="<i2")
                 .reshape(rows_per, ns) for c in range(c_a, c_b + 1)]
        data = np.concatenate(parts, 0)
        off = r_a - c_a * rows_per
        return DemBand(np.ascontiguousarray(data[off:off + (r_b - r_a)]), float(ppd), r_a)


# ---------------------------------------------------------------------------
# numba kernels (terrain sampling, ray casting, shadows, rendering)
# ---------------------------------------------------------------------------

@njit(fastmath=True, cache=True, nogil=True)
def _terrain_r(dem, ppd, row0, x, y, z):
    """(terrain radius under (x,y,z), radius of (x,y,z)), km."""
    r = math.sqrt(x * x + y * y + z * z)
    lat = math.degrees(math.asin(z / r))
    lon = math.degrees(math.atan2(y, x)) % 360.0
    fr = (90.0 - lat) * ppd - 0.5 - row0
    fc = lon * ppd - 0.5
    r0 = int(math.floor(fr))
    c0 = int(math.floor(fc))
    ar = fr - r0
    ac = fc - c0
    nr, nc = dem.shape
    ra = min(max(r0, 0), nr - 1)
    rb = min(max(r0 + 1, 0), nr - 1)
    ca = c0 % nc
    cb = (c0 + 1) % nc
    h = ((1.0 - ar) * ((1.0 - ac) * dem[ra, ca] + ac * dem[ra, cb])
         + ar * ((1.0 - ac) * dem[rb, ca] + ac * dem[rb, cb]))
    return R_MOON + h * 0.0005, r


@njit(fastmath=True, cache=True, nogil=True)
def _hit_one(dem, ppd, row0, O, dx, dy, dz):
    b = dx * O[0] + dy * O[1] + dz * O[2]
    oo = O[0] * O[0] + O[1] * O[1] + O[2] * O[2]
    rs = R_MOON
    x = y = z = np.nan
    for _ in range(6):
        disc = b * b - (oo - rs * rs)
        if disc < 0:
            return np.nan, np.nan, np.nan
        t = -b - math.sqrt(disc)
        x = O[0] + t * dx
        y = O[1] + t * dy
        z = O[2] + t * dz
        rs, _r = _terrain_r(dem, ppd, row0, x, y, z)
    return x, y, z


@njit(parallel=True, fastmath=True, cache=True, nogil=True)
def _hit_terrain(dem, ppd, row0, O, D):
    n = D.shape[0]
    out = np.empty((n, 3))
    for i in prange(n):
        x, y, z = _hit_one(dem, ppd, row0, O, D[i, 0], D[i, 1], D[i, 2])
        out[i, 0] = x
        out[i, 1] = y
        out[i, 2] = z
    return out


@njit(fastmath=True, cache=True, nogil=True)
def _horizon_beta(dem, ppd, row0, x, y, z, S, cell_km):
    """Highest terrain angle above the Sun-centre ray from (x,y,z), radians
    (small-angle).  > +sun radius: umbra, < -sun radius: fully lit."""
    r = math.sqrt(x * x + y * y + z * z)
    sin_e = (x * S[0] + y * S[1] + z * S[2]) / r
    smax = min(500.0, 9.0 / max(sin_e + 0.02, 0.012))
    beta = -1.0
    s = 0.4 * cell_km
    while s < smax:
        rt, rr = _terrain_r(dem, ppd, row0, x + s * S[0], y + s * S[1], z + s * S[2])
        a = (rt - rr) / s
        if a > beta:
            beta = a
        if s < 40.0:
            s += 0.4 * cell_km
        elif s < 150.0:
            s += cell_km
        else:
            s += 2.0 * cell_km
    return beta


@njit(fastmath=True, cache=True, nogil=True)
def _disk_fraction(q):
    """Visible fraction of the solar disk with the horizon q sun-radii above centre."""
    if q >= 1.0:
        return 0.0
    if q <= -1.0:
        return 1.0
    return (math.acos(q) - q * math.sqrt(1.0 - q * q)) / math.pi


@njit(parallel=True, fastmath=True, cache=True, nogil=True)
def _render_terrain(dem, ppd, row0, O, D, S, sun_rad, cell_km, max_sin_e):
    """Lommel-Seeliger render of the terrain with cast shadows and penumbra.
    Returns (reflectance, height_km) per ray; NaN height where the ray misses."""
    n = D.shape[0]
    refl = np.zeros(n, np.float32)
    hgt = np.full(n, np.nan, np.float32)
    for i in prange(n):
        x, y, z = _hit_one(dem, ppd, row0, O, D[i, 0], D[i, 1], D[i, 2])
        if x != x:
            continue
        r = math.sqrt(x * x + y * y + z * z)
        hgt[i] = r - R_MOON
        ux, uy, uz = x / r, y / r, z / r
        en = math.sqrt(ux * ux + uy * uy)
        if en < 1e-9:
            continue
        ex, ey = -uy / en, ux / en
        nx, ny, nz = -uz * ey, uz * ex, ux * ey - uy * ex
        dd = 0.6 * cell_km
        hE1, _a = _terrain_r(dem, ppd, row0, x + dd * ex, y + dd * ey, z)
        hE0, _a = _terrain_r(dem, ppd, row0, x - dd * ex, y - dd * ey, z)
        hN1, _a = _terrain_r(dem, ppd, row0, x + dd * nx, y + dd * ny, z + dd * nz)
        hN0, _a = _terrain_r(dem, ppd, row0, x - dd * nx, y - dd * ny, z - dd * nz)
        gE = (hE1 - hE0) / (2 * dd)
        gN = (hN1 - hN0) / (2 * dd)
        Nx = ux - gE * ex - gN * nx
        Ny = uy - gE * ey - gN * ny
        Nz = uz - gN * nz
        nn = math.sqrt(Nx * Nx + Ny * Ny + Nz * Nz)
        Nx /= nn
        Ny /= nn
        Nz /= nn
        cosi = Nx * S[0] + Ny * S[1] + Nz * S[2]
        vx, vy, vz = O[0] - x, O[1] - y, O[2] - z
        vn = math.sqrt(vx * vx + vy * vy + vz * vz)
        cose = (Nx * vx + Ny * vy + Nz * vz) / vn
        if cose <= 0.0:
            continue
        f = _disk_fraction(-cosi / sun_rad)
        if ux * S[0] + uy * S[1] + uz * S[2] < max_sin_e and f > 0.0:
            beta = _horizon_beta(dem, ppd, row0, x, y, z, S, cell_km)
            f = min(f, _disk_fraction(beta / sun_rad))
        ci = max(cosi, 0.0)
        if ci > 0.0:
            refl[i] = f * ci / (ci + cose)
    return refl, hgt


def _cell_km(dem: DemBand) -> float:
    return 2 * math.pi * R_MOON / 360.0 / dem.ppd


def hit_terrain(camera, uv, dem: DemBand) -> np.ndarray:
    """Pixels -> body points on the LOLA surface (NaN where the ray misses)."""
    uv = np.asarray(uv, np.float64)
    D = np.ascontiguousarray(camera.rays(uv).reshape(-1, 3))
    with _parallel_lock():
        X = _hit_terrain(dem.data, float(dem.ppd), int(dem.row0), np.asarray(camera.O, np.float64), D)
    return X.reshape(uv.shape[:-1] + (3,))


def render_terrain(camera, geom: LunarGeometry, dem: DemBand, shape, box=None,
                   max_sun_elev_deg: float = 27.0):
    """Render LOLA through `camera` (image-shaped float32 reflectance + height).
    `box` = (x0, y0, x1, y1) limits the work to part of the frame."""
    H, W = shape
    x0, y0, x1, y1 = box if box is not None else (0, 0, W, H)
    uu, vv = np.meshgrid(np.arange(x0, x1, dtype=np.float64), np.arange(y0, y1, dtype=np.float64))
    D = np.ascontiguousarray(camera.rays(np.stack([uu, vv], -1)).reshape(-1, 3))
    with _parallel_lock():
        refl, hgt = _render_terrain(dem.data, float(dem.ppd), int(dem.row0),
                                    np.asarray(camera.O, np.float64), D,
                                    np.asarray(geom.S, np.float64), SUN_RADIUS_RAD,
                                    _cell_km(dem), math.sin(math.radians(max_sun_elev_deg)))
    out_r = np.zeros((H, W), np.float32)
    out_h = np.full((H, W), np.nan, np.float32)
    out_r[y0:y1, x0:x1] = refl.reshape(y1 - y0, x1 - x0)
    out_h[y0:y1, x0:x1] = hgt.reshape(y1 - y0, x1 - x0)
    return out_r, out_h


def footprint_latitudes(camera, shape, margin_deg: float = 1.5, step: int = 16):
    """Latitude span of the visible lunar surface in a frame (for LDEM band fetch)."""
    H, W = shape
    yy, xx = np.mgrid[0:H:step, 0:W:step]
    X = camera.unproject_sphere(np.stack([xx, yy], -1).astype(np.float64))
    ok = np.isfinite(X[..., 0])
    if not ok.any():
        return None
    lat, _lon = body_to_latlon(X[ok])
    return max(-90.0, float(lat.min()) - margin_deg), min(90.0, float(lat.max()) + margin_deg)


# ---------------------------------------------------------------------------
# Drape: the user's image on a terrain mesh
# ---------------------------------------------------------------------------

@dataclass
class ReliefMesh:
    verts: np.ndarray        # (N,3) float32, local frame km: x east, y north, z up
    faces: np.ndarray        # (M,3) int32
    colors: np.ndarray       # (N,3) float32 in [0,1]
    uv: np.ndarray           # (N,2) float32, source-image pixel coordinates
    height_km: np.ndarray    # (N,) true (unexaggerated) height over the 1737.4 km sphere
    exaggeration: float
    centre_latlon: tuple
    view_dir: np.ndarray     # unit vector toward the observer, local frame
    view_up: np.ndarray      # image "up" direction, local frame
    source: str = "LOLA"


def _local_basis(C):
    u = _unit(C)
    e = _unit(np.cross([0.0, 0.0, 1.0], u)) if abs(u[2]) < 0.999 else np.array([0.0, 1.0, 0.0])
    n = np.cross(u, e)
    return e, n, u


def _bilinear(a, x, y):
    """Bilinear sample of a 2-D array at float pixel coords (edge-clamped).
    (cv2.remap asserts on maps longer than 32767, so vertex lists can't use it.)"""
    a = np.asarray(a, np.float32)
    H, W = a.shape
    x = np.clip(np.asarray(x, np.float64), 0, W - 1)
    y = np.clip(np.asarray(y, np.float64), 0, H - 1)
    x0 = np.minimum(np.floor(x).astype(np.int64), W - 2)
    y0 = np.minimum(np.floor(y).astype(np.int64), H - 2)
    fx, fy = x - x0, y - y0
    return ((1 - fy) * ((1 - fx) * a[y0, x0] + fx * a[y0, x0 + 1])
            + fy * ((1 - fx) * a[y0 + 1, x0] + fx * a[y0 + 1, x0 + 1])).astype(np.float32)


def _grid_mesh(valid):
    gh, gw = valid.shape
    idx = np.arange(gh * gw).reshape(gh, gw)
    q = valid[:-1, :-1] & valid[:-1, 1:] & valid[1:, :-1] & valid[1:, 1:]
    a, b = idx[:-1, :-1][q], idx[:-1, 1:][q]
    c, d = idx[1:, :-1][q], idx[1:, 1:][q]
    faces = np.concatenate([np.stack([a, c, b], 1), np.stack([b, c, d], 1)], 0)
    # compact to valid vertices
    remap = np.full(gh * gw, -1, np.int64)
    vflat = valid.ravel()
    remap[vflat] = np.arange(int(vflat.sum()))
    return remap[faces].astype(np.int32), vflat


def build_mesh_from_points(image, uv_grid, X_grid, valid, camera, exaggeration=1.0,
                           source="LOLA", flatten=False) -> ReliefMesh:
    """Shared mesh assembly for any height source.  X_grid = (gh,gw,3) body points
    on the surface for each grid pixel (NaN = no surface).

    flatten=False keeps the globe's curvature (right for a full disk).
    flatten=True unrolls the area onto an azimuthal-equidistant map around its
    centre and keeps only the (exaggerated) relief -- far more legible for a
    regional crop, where curvature would otherwise dominate the view."""
    img = np.asarray(image, np.float32)
    if img.ndim == 2:
        img = img[..., None]
    faces, vflat = _grid_mesh(valid)
    X = X_grid.reshape(-1, 3)[vflat]
    uvv = uv_grid.reshape(-1, 2)[vflat].astype(np.float32)
    r = np.linalg.norm(X, axis=1)
    h = (r - R_MOON).astype(np.float32)
    Xe = X / r[:, None] * (R_MOON + float(exaggeration) * h)[:, None]
    C = _unit(X.mean(0)) * R_MOON
    e, n, u = _local_basis(C)
    if flatten:
        uh = X / r[:, None]
        rho = R_MOON * np.arccos(np.clip(uh @ u, -1.0, 1.0))
        tang = uh - (uh @ u)[:, None] * u[None]
        tn = np.linalg.norm(tang, axis=1)
        tang = tang / np.where(tn > 0, tn, 1.0)[:, None]
        verts = np.stack([rho * (tang @ e), rho * (tang @ n), float(exaggeration) * h], 1)
        verts = verts.astype(np.float32)
    else:
        rel = Xe - C
        verts = np.stack([rel @ e, rel @ n, rel @ u], 1).astype(np.float32)
    cols = np.stack([_bilinear(img[..., ch], uvv[:, 0], uvv[:, 1]) for ch in range(img.shape[2])], 1)
    if cols.shape[1] == 1:
        cols = np.repeat(cols, 3, 1)
    cols = np.clip(cols[:, :3], 0.0, 1.0).astype(np.float32)
    v = _unit(camera.O - C)
    view_dir = np.array([v @ e, v @ n, v @ u])
    # image-up in 3D: displacement that moves a pixel up in the frame
    cen_uv = uvv.mean(0)
    p0 = camera.rays(cen_uv[None].astype(np.float64))[0]
    p1 = camera.rays((cen_uv - [0, 10.0])[None].astype(np.float64))[0]
    up3 = _unit(p1 - p0)
    view_up = _unit(np.array([up3 @ e, up3 @ n, up3 @ u]))
    lat, lon = body_to_latlon(C)
    return ReliefMesh(verts=verts, faces=faces, colors=cols, uv=uvv, height_km=h,
                      exaggeration=float(exaggeration), centre_latlon=(float(lat), float(lon)),
                      view_dir=view_dir, view_up=view_up, source=source)


@dataclass
class ReliefGrid:
    """Vertices on the image's own pixel grid (every `step` px from x0, y0):
    the surface point each pixel sees, the pixel itself, and which exist."""
    uv: np.ndarray           # (gh,gw,2) image pixel of each vertex
    X: np.ndarray            # (gh,gw,3) body points on the surface, km (NaN = none)
    valid: np.ndarray        # (gh,gw) bool
    x0: int
    y0: int
    step: int
    source: str = "LOLA"


def relief_grid(camera, shape, *, dem: Optional[DemBand] = None, surf=None, roi=None,
                max_vertices=400_000) -> ReliefGrid:
    """Vertex grid for a frame: each vertex is the point its pixel sees on LOLA
    terrain (dem), on a derived surface (surf; the plain globe outside it), or on
    the bare sphere.  roi = (x0, y0, x1, y1) in pixels, default the whole frame."""
    H, W = shape[:2]
    x0, y0, x1, y1 = roi if roi is not None else (0, 0, W, H)
    # pick a grid step that keeps the vertex count under the cap
    yy, xx = np.mgrid[y0:y1:8, x0:x1:8]
    ok = np.isfinite(camera.unproject_sphere(np.stack([xx, yy], -1).astype(np.float64))[..., 0])
    n_valid = ok.sum() * 64.0
    step = max(1, int(math.ceil(math.sqrt(max(n_valid, 1.0) / float(max_vertices)))))
    gx = np.arange(x0, x1, step, dtype=np.float64)
    gy = np.arange(y0, y1, step, dtype=np.float64)
    uu, vv = np.meshgrid(gx, gy)
    uv = np.stack([uu, vv], -1)
    if dem is not None:
        X = hit_terrain(camera, uv, dem)
        source = "LOLA"
    else:
        X = camera.unproject_sphere(uv)
        source = "sphere"
        if surf is not None:
            Xs = surf.hit(camera, uv)
            X = np.where(np.isfinite(Xs[..., :1]), Xs, X)
            source = "this image (shadows + shading)"
    valid = np.isfinite(X[..., 0])
    if valid.any():
        _p, vis = camera.project(np.where(valid[..., None], X, 1.0))
        valid &= vis
    if valid.sum() < 4:
        raise ValueError("no lunar surface inside the selected area")
    return ReliefGrid(uv=uv, X=X, valid=valid, x0=int(x0), y0=int(y0), step=int(step), source=source)


def drape_on_lola(image, camera, dem: DemBand, *, roi=None, max_vertices=400_000,
                  exaggeration=1.0, flatten=False) -> ReliefMesh:
    """Drape an image (float [0,1], mono or RGB, display-ready) on LOLA terrain.
    roi = (x0, y0, x1, y1) in pixels, default the whole frame."""
    g = relief_grid(camera, np.asarray(image).shape, dem=dem, roi=roi, max_vertices=max_vertices)
    return build_mesh_from_points(image, g.uv, g.X, g.valid, camera, exaggeration, source=g.source,
                                  flatten=flatten)


def export_ply(path, mesh: ReliefMesh):
    """Binary PLY with per-vertex colour (opens in Blender, MeshLab, CloudCompare)."""
    n, m = len(mesh.verts), len(mesh.faces)
    hdr = ("ply\nformat binary_little_endian 1.0\n"
           f"comment SetiAstro Suite Pro lunar relief, source {mesh.source}, "
           f"exaggeration {mesh.exaggeration:g}, units km\n"
           f"element vertex {n}\nproperty float x\nproperty float y\nproperty float z\n"
           "property uchar red\nproperty uchar green\nproperty uchar blue\n"
           f"element face {m}\nproperty list uchar int vertex_indices\nend_header\n")
    vt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    va = np.empty(n, vt)
    va["x"], va["y"], va["z"] = mesh.verts[:, 0], mesh.verts[:, 1], mesh.verts[:, 2]
    c8 = np.clip(np.round(mesh.colors * 255.0), 0, 255).astype(np.uint8)
    va["r"], va["g"], va["b"] = c8[:, 0], c8[:, 1], c8[:, 2]
    ft = np.dtype([("n", "u1"), ("a", "<i4"), ("b", "<i4"), ("c", "<i4")])
    fa = np.empty(m, ft)
    fa["n"] = 3
    fa["a"], fa["b"], fa["c"] = mesh.faces[:, 0], mesh.faces[:, 1], mesh.faces[:, 2]
    with open(path, "wb") as f:
        f.write(hdr.encode("ascii"))
        f.write(va.tobytes())
        f.write(fa.tobytes())


def export_obj(path, mesh: ReliefMesh, image, texture_name=None):
    """Wavefront OBJ + MTL + 16-bit PNG texture (the source image region)."""
    path = Path(path)
    img = np.clip(np.asarray(image, np.float32), 0, 1)
    if img.ndim == 3 and img.shape[2] == 1:
        img = img[..., 0]
    x0, y0 = np.floor(mesh.uv.min(0)).astype(int)
    x1, y1 = np.ceil(mesh.uv.max(0)).astype(int) + 1
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, img.shape[1]), min(y1, img.shape[0])
    tex = img[y0:y1, x0:x1]
    tex16 = np.round(tex * 65535.0).astype(np.uint16)
    if tex16.ndim == 3:
        tex16 = cv2.cvtColor(tex16[..., :3], cv2.COLOR_RGB2BGR)
    texture_name = texture_name or (path.stem + "_texture.png")
    cv2.imwrite(str(path.with_name(texture_name)), tex16)
    mtl = path.with_suffix(".mtl")
    mtl.write_text(f"newmtl lunar\nKa 1 1 1\nKd 1 1 1\nKs 0 0 0\nillum 1\nmap_Kd {texture_name}\n",
                   encoding="ascii")
    tw, th = float(x1 - x0), float(y1 - y0)
    vt = np.stack([(mesh.uv[:, 0] - x0 + 0.5) / tw, 1.0 - (mesh.uv[:, 1] - y0 + 0.5) / th], 1)
    with open(path, "w", encoding="ascii", newline="\n") as f:
        f.write(f"# SetiAstro Suite Pro lunar relief ({mesh.source}), exaggeration "
                f"{mesh.exaggeration:g}, units km\nmtllib {mtl.name}\nusemtl lunar\n")
        np.savetxt(f, mesh.verts, fmt="v %.4f %.4f %.4f")
        np.savetxt(f, vt, fmt="vt %.6f %.6f")
        fi = mesh.faces + 1
        np.savetxt(f, np.stack([fi[:, 0], fi[:, 0], fi[:, 1], fi[:, 1], fi[:, 2], fi[:, 2]], 1),
                   fmt="f %d/%d %d/%d %d/%d")


# ---------------------------------------------------------------------------
# Shadow measurement
# ---------------------------------------------------------------------------

def estimate_psf_sigma(mono, camera, *, min_contrast=0.3) -> Optional[float]:
    """Gaussian PSF sigma (px) from the lit limb's edge spread; None if no lit limb."""
    img = np.asarray(mono, np.float32)
    H, W = img.shape
    cx, cy = camera.cx, camera.cy
    R = camera.disk_radius_px
    widths = []
    for th in np.radians(np.arange(0.0, 360.0, 2.0)):
        rr = np.arange(R - 25.0, R + 25.0, 0.1)
        x = (cx + rr * math.cos(th)).astype(np.float32)
        y = (cy + rr * math.sin(th)).astype(np.float32)
        if x.min() < 0 or y.min() < 0 or x.max() > W - 1 or y.max() > H - 1:
            continue
        p = cv2.remap(img, x[:, None], y[:, None], cv2.INTER_LINEAR)[:, 0]
        hi, lo = np.median(p[:60]), np.median(p[-60:])
        if hi - lo < min_contrast * float(np.percentile(img, 99.5)):
            continue
        q = (p - lo) / (hi - lo)
        a = np.nonzero(q < 0.9)[0]
        b = np.nonzero(q < 0.1)[0]
        if len(a) and len(b) and b[0] > a[0]:
            widths.append((b[0] - a[0]) * 0.1)
    if len(widths) < 6:
        return None
    return float(np.median(widths) / 2.563)


@dataclass
class ShadowRuns:
    tip_uv: np.ndarray       # (N,2) shadow tip pixel (anti-sunward end)
    caster_uv: np.ndarray    # (N,2) casting crest pixel
    T: np.ndarray            # (N,3) tip on the mean sphere, body km
    P: np.ndarray            # (N,3) crest, body km
    height_km: np.ndarray    # crest height above the tip
    length_km: np.ndarray    # distance tip -> crest along the Sun ray
    sun_elev_deg: np.ndarray
    emission_deg: np.ndarray
    length_px: np.ndarray


def measure_shadows(mono, camera, geom: LunarGeometry, *, roi=None, max_sun_elev=22.0,
                    tile=256, dark_frac=0.20, lit_frac=0.40, min_len_px=2.5,
                    progress: Optional[Callable] = None) -> ShadowRuns:
    """Find shadows along the projected Sun direction and turn each into a height.

    tip    = anti-sunward end, 50 % crossing to the lit floor beyond it
    crest  = sunward end, 50 % crossing to the lit slope before it
    3D: T = tip ray on the mean sphere; crest P = T + s*S with s such that P
    projects onto the crest pixel (handles curvature, foreshortening, perspective
    and the parallax of the elevated crest).  height = |P| - |T|.
    """
    img = np.asarray(mono, np.float32)
    H, W = img.shape
    x0r, y0r, x1r, y1r = roi if roi is not None else (0, 0, W, H)
    sky = float(np.percentile(img, 1.0))
    env = cv2.GaussianBlur(cv2.dilate(img, np.ones((21, 21), np.uint8)), (0, 0), 30)
    STEP, half, marg = 0.25, tile // 2, 90
    S = np.asarray(geom.S, np.float64)

    def crossing(prof, i_from, i_to, level):
        stp = 1 if i_to > i_from else -1
        for i in range(i_from, i_to, stp):
            a, b = prof[i], prof[i + stp]
            if (a - level) * (b - level) <= 0 and a != b:
                return i + stp * (level - a) / (b - a)
        return None

    centres = []
    for cy in range(y0r + half, y1r - half + 1, tile):
        for cx in range(x0r + half, x1r - half + 1, tile):
            X = camera.unproject_sphere(np.array([cx, cy], np.float64))
            if not np.isfinite(X[0]):
                continue
            se = float(_unit(X) @ S)
            if -0.02 < se < math.sin(math.radians(max_sun_elev)):
                centres.append((cx, cy, X))
    rows = []
    xs = np.arange(-half * 1.45 - marg, half * 1.45 + marg, STEP)
    for ti, (cx, cy, X) in enumerate(centres):
        if progress is not None:
            progress(ti, len(centres))
        p0, _ = camera.project(X)
        p1, _ = camera.project(X + S)
        ju = (p1 - p0) / np.linalg.norm(p1 - p0)
        jp = np.array([-ju[1], ju[0]])
        for r in np.arange(-half * 1.45, half * 1.45, 1.0):
            P = np.array([cx, cy]) + xs[:, None] * ju + r * jp
            inside = (P[:, 0] > 2) & (P[:, 0] < W - 3) & (P[:, 1] > 2) & (P[:, 1] < H - 3)
            if inside.sum() < 50:
                continue
            mx = P[:, 0].astype(np.float32)[:, None]
            my = P[:, 1].astype(np.float32)[:, None]
            prof = cv2.remap(img, mx, my, cv2.INTER_LINEAR)[:, 0]
            envp = cv2.remap(env, mx, my, cv2.INTER_LINEAR)[:, 0]
            thr_dark = sky + dark_frac * (envp - sky)
            thr_lit = sky + lit_frac * (envp - sky)
            dark = (prof < thr_dark) & inside
            dd = np.diff(np.r_[0, dark.astype(np.int8), 0])
            for i0, i1 in zip(np.nonzero(dd == 1)[0], np.nonzero(dd == -1)[0] - 1):
                if i0 < 30 or i1 > len(prof) - 30:
                    continue
                a0, a1 = i0 - int(5 / STEP), i0 - int(1 / STEP)
                plate = float(np.median(prof[a0:a1]))
                if not np.all(prof[a0:a1] > thr_dark[a0:a1]) or plate < thr_lit[i0]:
                    continue
                b1 = i1 + int(4 / STEP)
                peak = float(prof[i1:b1].max())
                if peak < thr_lit[i1]:
                    continue
                floor = float(np.percentile(prof[i0:i1 + 1], 10))
                t_i = crossing(prof, i0 + (i1 - i0) // 2, a0, 0.5 * (plate + floor))
                c_i = crossing(prof, i1 - (i1 - i0) // 2, b1, 0.5 * (peak + floor))
                if t_i is None or c_i is None:
                    continue
                lpx = (c_i - t_i) * STEP
                if lpx < min_len_px:
                    continue
                tip = np.array([cx, cy]) + (xs[0] + t_i * STEP) * ju + r * jp
                cas = np.array([cx, cy]) + (xs[0] + c_i * STEP) * ju + r * jp
                if np.abs(tip - [cx, cy]).max() > half:
                    continue          # each tip belongs to exactly one tile
                if not (x0r <= tip[0] < x1r and y0r <= tip[1] < y1r):
                    continue
                rows.append((tip[0], tip[1], cas[0], cas[1], lpx))
    if not rows:
        z = np.zeros((0,))
        return ShadowRuns(np.zeros((0, 2)), np.zeros((0, 2)), np.zeros((0, 3)), np.zeros((0, 3)),
                          z, z, z, z, z)
    rows = np.asarray(rows, np.float64)
    tip_uv, cas_uv = rows[:, 0:2], rows[:, 2:4]
    T = camera.unproject_sphere(tip_uv)
    s = np.zeros(len(T))
    for _ in range(4):
        Pk = T + s[:, None] * S
        q0, _ = camera.project(Pk)
        q1, _ = camera.project(Pk + S)
        J = q1 - q0
        s = s + np.einsum("ij,ij->i", cas_uv - q0, J) / np.einsum("ij,ij->i", J, J)
    P = T + s[:, None] * S
    qP, _ = camera.project(P)
    lateral = np.linalg.norm(qP - cas_uv, axis=1)
    h = np.linalg.norm(P, axis=1) - np.linalg.norm(T, axis=1)
    e_deg = np.degrees(np.arcsin(np.clip(_unit(T) @ S, -1, 1)))
    emis = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", _unit(T), _unit(camera.O - T)), -1, 1)))
    ok = (e_deg > 0.3) & (e_deg < max_sun_elev) & (emis < 75) & (h > 0) & (lateral < 1.5)
    return ShadowRuns(tip_uv[ok], cas_uv[ok], T[ok], P[ok], h[ok], s[ok], e_deg[ok], emis[ok],
                      rows[ok, 4])


@njit(parallel=True, fastmath=True, cache=True, nogil=True)
def _lola_shadow_truth(dem, ppd, row0, T_hat, A_hat, S, x_start, x_stop, dx, cell_km):
    """For each run: walk anti-sunward from inside the observed shadow until LOLA says
    the surface is lit; report (tip offset km, crest height over that tip km)."""
    n = T_hat.shape[0]
    out = np.full((n, 2), np.nan)
    for i in prange(n):
        x = x_start[i]
        first = True
        while x < x_stop[i]:
            th = x / R_MOON
            ux = T_hat[i, 0] * math.cos(th) + A_hat[i, 0] * math.sin(th)
            uy = T_hat[i, 1] * math.cos(th) + A_hat[i, 1] * math.sin(th)
            uz = T_hat[i, 2] * math.cos(th) + A_hat[i, 2] * math.sin(th)
            rs = R_MOON
            for _ in range(3):
                rs, _r = _terrain_r(dem, ppd, row0, ux * rs, uy * rs, uz * rs)
            qx, qy, qz = ux * rs, uy * rs, uz * rs
            # max terrain excess over the sun ray and where it occurs
            best, sb = -1e9, 0.0
            s = 0.4 * cell_km
            smax = min(500.0, 12.0 / max(ux * S[0] + uy * S[1] + uz * S[2], 0.01))
            while s < smax:
                rt, rr = _terrain_r(dem, ppd, row0, qx + s * S[0], qy + s * S[1], qz + s * S[2])
                if rt - rr > best:
                    best = rt - rr
                    sb = s
                s += 0.4 * cell_km if s < 60.0 else cell_km
            if first:
                first = False
                if best <= 0.0:
                    break              # LOLA: the middle of this shadow is lit
            elif best <= 0.0:
                rt, rr = _terrain_r(dem, ppd, row0, qx + sb * S[0], qy + sb * S[1], qz + sb * S[2])
                out[i, 0] = x
                out[i, 1] = rt - rs
                break
            x += dx
    return out


def compare_shadows_to_lola(runs: ShadowRuns, geom: LunarGeometry, dem: DemBand) -> dict:
    """LOLA truth for each measured shadow: where LOLA's shadow ends on the same Sun
    line, and the true crest height over that tip.  Returns per-run arrays + stats."""
    S = np.asarray(geom.S, np.float64)
    T_hat = _unit(runs.T)
    A_hat = _unit(-(S[None] - (T_hat @ S)[:, None] * T_hat))
    L_obs = runs.length_km * np.cos(np.radians(runs.sun_elev_deg))
    with _parallel_lock():
        out = _lola_shadow_truth(dem.data, float(dem.ppd), int(dem.row0), np.ascontiguousarray(T_hat),
                                 np.ascontiguousarray(A_hat), S, -0.5 * L_obs,
                                 np.full(len(T_hat), 25.0), 0.05, _cell_km(dem))
    found = np.isfinite(out[:, 1])
    res = runs.height_km[found] - out[found, 1]
    rel = res / out[found, 1]

    def rsig(a):
        return float(1.4826 * np.median(np.abs(a - np.median(a)))) if len(a) else float("nan")
    return dict(found=found, tip_offset_km=out[:, 0], true_height_km=out[:, 1],
                n=int(len(found)), n_confirmed=int(found.sum()),
                bias_km=float(np.median(res)) if len(res) else float("nan"),
                sigma_km=rsig(res),
                bias_rel=float(np.median(rel)) if len(rel) else float("nan"),
                sigma_rel=rsig(rel))


# ---------------------------------------------------------------------------
# Terrain derived from the image itself (shadow-anchored photoclinometry)
# ---------------------------------------------------------------------------

@dataclass
class DerivedSurface:
    """Height field on an azimuthal-equidistant map grid (row 0 = north)."""
    centre_latlon: tuple
    basis: tuple             # (e, n, u) at the centre
    cell_km: float
    half_km: float
    h: np.ndarray            # (GH,GW) km over the 1737.4 km sphere (offset arbitrary)
    usable: np.ndarray       # (GH,GW) cells the photometry could see
    n_shadows: int
    photometric_rms: float
    shadow_rms_km: float

    def grid_dirs(self):
        e0, n0, u0 = self.basis
        ax = np.arange(-self.half_km, self.half_km + 1e-6, self.cell_km)
        gx, gy = np.meshgrid(ax, -ax)
        rho = np.hypot(gx, gy)
        th = rho / R_MOON
        td = (gx[..., None] * e0 + gy[..., None] * n0) / np.where(rho > 0, rho, 1.0)[..., None]
        return np.cos(th)[..., None] * u0 + np.sin(th)[..., None] * td

    def to_map(self, U):
        e0, n0, u0 = self.basis
        U = _unit(U)
        rr = R_MOON * np.arccos(np.clip(U @ u0, -1.0, 1.0))
        tg = U - (U @ u0)[..., None] * u0
        tg = tg / np.maximum(np.linalg.norm(tg, axis=-1), 1e-12)[..., None]
        return rr * (tg @ e0), rr * (tg @ n0)

    def height_at(self, U):
        """Height (km) at body directions U (...,3); NaN outside the usable area."""
        mx, my = self.to_map(U)
        c = (mx + self.half_km) / self.cell_km
        r = (self.half_km - my) / self.cell_km
        GH, GW = self.h.shape
        inside = (c >= 0) & (c <= GW - 1) & (r >= 0) & (r <= GH - 1)
        hv = _bilinear(np.where(self.usable, self.h, np.nan), np.where(inside, c, 0), np.where(inside, r, 0))
        return np.where(inside, hv, np.nan)

    def hit(self, camera, uv, iters=3):
        """Pixels -> body points on the derived surface (NaN outside it)."""
        O = np.asarray(camera.O, np.float64)
        dr = camera.rays(np.asarray(uv, np.float64))
        b = dr @ O
        h = np.zeros(dr.shape[:-1])
        X = None
        for _ in range(iters):
            rs = R_MOON + np.nan_to_num(h)
            disc = b * b - (O @ O - rs * rs)
            t = -b - np.sqrt(np.where(disc >= 0, disc, np.nan))
            X = O + t[..., None] * dr
            h = self.height_at(X)
        return np.where(np.isfinite(h)[..., None], X, np.nan)


def _pick_shadow_cluster(runs: ShadowRuns, box_deg=12.0):
    lat, lon = body_to_latlon(runs.T)
    if len(lat) == 0:
        return None
    best = None
    for la in np.arange(-80.0, 80.1, 2.0):
        cl = max(math.cos(math.radians(la)), 0.1)
        for lo in np.arange(-180.0, 180.0, 2.0):
            dlon = (lon - lo + 180.0) % 360.0 - 180.0
            n = int(((np.abs(lat - la) < box_deg / 2) & (np.abs(dlon * cl) < box_deg / 2)).sum())
            if best is None or n > best[0]:
                best = (n, la, lo)
    return best[1], best[2]


def derive_surface(mono, camera, geom: LunarGeometry, *, runs: Optional[ShadowRuns] = None,
                   centre_latlon=None, half_km=None, cell_km=None, max_cells=300,
                   lam_smooth=3.0, lam_shadow=3.0, anisotropy=4.0, lam_prior=0.02,
                   iterations=10, progress: Optional[Callable] = None,
                   cancel: Optional[Callable] = None) -> DerivedSurface:
    """Derive terrain from one lunar image.

    Shading near the terminator gives the slope toward the Sun (Lommel-Seeliger
    with the ephemeris Sun and the registered view); every shadow pins a crest-
    over-floor height difference, which fixes what shading cannot: the drift along
    each Sun line and the absolute scale.  Albedo x gain is estimated from the data
    as a smooth (18 km) ratio of observed to modelled brightness.  Cross-sun slopes
    are invisible to one image, so smoothing is stronger across the Sun direction.

    Real-data check (Frank's 2023-05-01 disc, 1.15 km/px, seeing 2.8 km): against
    LOLA, along-sun slopes corr 0.73, local relief (<30 km) corr 0.51 at the right
    amplitude; absolute heights over long baselines are not reliable.  Use a LINEAR
    image (photometry), near the terminator.
    """
    import scipy.sparse as sps
    from scipy.sparse.linalg import lsmr

    img = np.asarray(mono, np.float32)
    if img.ndim == 3:
        img = img.mean(axis=2)
    H, W = img.shape
    lin = np.clip(img - float(np.percentile(img, 1.0)), 0, None)
    S = np.asarray(geom.S, np.float64)
    O = np.asarray(camera.O, np.float64)

    def _prog(f, msg):
        if progress is not None:
            progress(float(f), msg)
        if cancel is not None and cancel():
            raise RuntimeError("cancelled")

    if runs is None:
        _prog(0.02, "measuring shadows")
        runs = measure_shadows(img, camera, geom)
    if centre_latlon is None:
        centre_latlon = _pick_shadow_cluster(runs)
        if centre_latlon is None:
            raise ValueError("no measurable shadows: the terminator is not in this frame")
    km_px = float(camera.km_per_px()) if hasattr(camera, "km_per_px") else 1.0
    cell = float(cell_km) if cell_km else max(0.4, 1.3 * km_px)
    half = float(half_km) if half_km else min(180.0, 0.5 * max_cells * cell)
    half = min(half, 0.5 * max_cells * cell)

    C = sph_to_body(*centre_latlon)
    e0, n0, u0 = _local_basis(C)
    surf = DerivedSurface(tuple(float(v) for v in centre_latlon), (e0, n0, u0), cell, half,
                          np.zeros((1, 1)), np.zeros((1, 1), bool), 0, float("nan"), float("nan"))
    DIR = surf.grid_dirs()
    GH, GW = DIR.shape[:2]
    N = GH * GW
    U = DIR

    def _mapdir(x, y):
        rr = np.hypot(x, y)
        t = rr / R_MOON
        td = (x[..., None] * e0 + y[..., None] * n0) / np.where(rr > 0, rr, 1.0)[..., None]
        return np.cos(t)[..., None] * u0 + np.sin(t)[..., None] * td
    ax = np.arange(-half, half + 1e-6, cell)
    gx, gy = np.meshgrid(ax, -ax)
    mx = _unit(_mapdir(gx + 0.01, gy) - _mapdir(gx - 0.01, gy))
    mx = _unit(mx - np.sum(mx * U, -1, keepdims=True) * U)
    my = np.cross(U, mx)
    sx, sy, sz = ((S * mx).sum(-1).ravel(), (S * my).sum(-1).ravel(), (S * U).sum(-1).ravel())

    # shadows inside the grid
    cx_, cy_ = surf.to_map(runs.P)
    tx_, ty_ = surf.to_map(runs.T)
    lim = half - 3.0 * cell
    inr = (np.abs(cx_) < lim) & (np.abs(cy_) < lim) & (np.abs(tx_) < lim) & (np.abs(ty_) < lim)
    Hs = runs.height_km[inr]

    def _bil_rows(xm, ym):
        c = (xm + half) / cell
        r = (half - ym) / cell
        c0, r0 = np.floor(c).astype(np.int64), np.floor(r).astype(np.int64)
        fc, fr = c - c0, r - r0
        rows, cols, vals = [], [], []
        for (rr_, cc_, w_) in ((r0, c0, (1 - fr) * (1 - fc)), (r0, c0 + 1, (1 - fr) * fc),
                               (r0 + 1, c0, fr * (1 - fc)), (r0 + 1, c0 + 1, fr * fc)):
            rows.append(np.arange(len(xm)))
            cols.append(rr_ * GW + cc_)
            vals.append(w_)
        return sps.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                              shape=(len(xm), N))
    Bs = (_bil_rows(cx_[inr], cy_[inr]) - _bil_rows(tx_[inr], ty_[inr])).tocsr()

    idx = np.arange(N).reshape(GH, GW)

    def _d(axis):
        if axis == "x":
            a, b, c = idx[:, 2:].ravel(), idx[:, :-2].ravel(), idx[:, 1:-1].ravel()
        else:
            a, b, c = idx[:-2, :].ravel(), idx[2:, :].ravel(), idx[1:-1, :].ravel()
        return sps.csr_matrix((np.r_[np.full(len(c), 0.5 / cell), np.full(len(c), -0.5 / cell)],
                               (np.r_[c, c], np.r_[a, b])), shape=(N, N))
    Dx, Dy = _d("x"), _d("y")
    ci = idx[1:-1, 1:-1].ravel()
    lap5 = sps.csr_matrix((np.r_[np.full(len(ci), -4.0), np.ones(4 * len(ci))],
                           (np.r_[ci, np.tile(ci, 4)],
                            np.r_[ci, idx[:-2, 1:-1].ravel(), idx[2:, 1:-1].ravel(),
                                  idx[1:-1, :-2].ravel(), idx[1:-1, 2:].ravel()])),
                          shape=(N, N)) / cell ** 2
    azc = np.arctan2(sy, sx)
    Du = (sps.diags(np.cos(azc)) @ Dx + sps.diags(np.sin(azc)) @ Dy).tocsr()
    Dv = (sps.diags(-np.sin(azc)) @ Dx + sps.diags(np.cos(azc)) @ Dy).tocsr()
    Reg = sps.vstack([(Du @ Du) / anisotropy, (Dv @ Dv) * anisotropy, Du @ Dv, 0.5 * lap5]).tocsr()

    # image samples at the grid (mean-sphere positions)
    X0 = (U * R_MOON).reshape(-1, 3)
    uv, vis = camera.project(X0)
    inframe = (uv[:, 0] >= 1) & (uv[:, 0] <= W - 2) & (uv[:, 1] >= 1) & (uv[:, 1] <= H - 2)
    uvf = uv.reshape(GH, GW, 2).astype(np.float32)
    I = cv2.remap(lin, uvf[..., 0], uvf[..., 1], cv2.INTER_LINEAR, borderValue=0).ravel().astype(np.float64)
    V = _unit(O - X0)
    Vx, Vy, Vz = (V * mx.reshape(-1, 3)).sum(-1), (V * my.reshape(-1, 3)).sum(-1), (V * U.reshape(-1, 3)).sum(-1)
    env = cv2.GaussianBlur(cv2.dilate(I.reshape(GH, GW).astype(np.float32), np.ones((15, 15), np.uint8)),
                           (0, 0), 20).ravel()
    usable = (vis & inframe & (sz > math.sin(math.radians(0.3))) & (sz < math.sin(math.radians(25.0)))
              & (Vz > 0.25))
    if usable.sum() < 100:
        raise ValueError("too little sunlit terrain near the terminator in this area")
    sig_A = 18.0 / cell
    out_w = np.where(usable, lam_prior, 1.0)
    h = np.zeros(N)
    rp = np.zeros(1)
    for it in range(int(iterations)):
        _prog(0.1 + 0.85 * it / iterations, f"solving terrain ({it + 1}/{iterations})")
        p, q = Dx @ h, Dy @ h
        a = sz - p * sx - q * sy
        b = Vz - p * Vx - q * Vy
        ab = np.maximum(a + b, 1e-6)
        R = np.where(a > 0, a / ab, 0.0)
        lit = usable & (a > 0.01) & (I > 0.25 * env)
        num = cv2.GaussianBlur((I * lit).reshape(GH, GW).astype(np.float32), (0, 0), sig_A)
        den = cv2.GaussianBlur((R * lit).reshape(GH, GW).astype(np.float32), (0, 0), sig_A)
        A = (num / np.maximum(den, 1e-9)).ravel().astype(np.float64)
        Rp = (-sx * b + a * Vx) / ab ** 2
        Rq = (-sy * b + a * Vy) / ab ** 2
        w = lit / max(float(np.median(I[lit])), 1e-9)
        Jp = (sps.diags(w * A * Rp) @ Dx + sps.diags(w * A * Rq) @ Dy).tocsr()[np.nonzero(lit)[0]]
        rp = (w * (I - A * R))[lit]
        J = sps.vstack([Jp, lam_shadow * Bs, lam_smooth * Reg, sps.diags(out_w), 0.05 * sps.eye(N)]).tocsr()
        rhs = np.r_[rp, lam_shadow * (Hs - Bs @ h), -lam_smooth * (Reg @ h), -out_w * h, np.zeros(N)]
        dh = np.clip(lsmr(J, rhs, maxiter=400, atol=1e-6, btol=1e-6)[0], -1.0, 1.0)
        h = h + 0.8 * dh
        if np.abs(dh).max() < 0.01:
            break
    _prog(1.0, "done")
    hh = h.reshape(GH, GW)
    us = usable.reshape(GH, GW)
    hh = hh - float(np.median(hh[us]))
    surf.h, surf.usable = hh.astype(np.float32), us
    surf.n_shadows = int(inr.sum())
    surf.photometric_rms = float(np.sqrt(np.mean(rp ** 2))) if len(rp) else float("nan")
    surf.shadow_rms_km = float(np.sqrt(np.mean((Hs - Bs @ h) ** 2))) if len(Hs) else float("nan")
    return surf


def score_against_lola(surf: DerivedSurface, geom: LunarGeometry, dem: DemBand, band_km=30.0) -> dict:
    """How well a derived surface matches LOLA (never used to derive it)."""
    U = surf.grid_dirs()
    GH, GW = surf.h.shape
    X = (U * R_MOON).reshape(-1, 3)
    hl = np.array([_terrain_r(dem.data, float(dem.ppd), int(dem.row0), *x)[0] for x in X]) - R_MOON
    hl = hl.reshape(GH, GW)
    m = surf.usable.copy()
    e = max(2, int(10.0 / surf.cell_km))
    m[:e], m[-e:], m[:, :e], m[:, -e:] = False, False, False, False
    if m.sum() < 50:
        return {}
    ax = np.arange(-surf.half_km, surf.half_km + 1e-6, surf.cell_km)
    gx, gy = np.meshgrid(ax, -ax)

    def detrend(z):
        G = np.c_[np.ones(m.sum()), gx[m], gy[m]]
        c = np.linalg.lstsq(G, z[m], rcond=None)[0]
        return z - (c[0] + c[1] * gx + c[2] * gy)

    def lp(z, s):
        return cv2.GaussianBlur(z.astype(np.float32), (0, 0), max(s, 0.3))

    hd, ld = detrend(surf.h.astype(np.float64)), detrend(hl)
    a_ = lp(hd, 1.0) - lp(hd, band_km / surf.cell_km)
    b_ = lp(ld, 1.0) - lp(ld, band_km / surf.cell_km)
    S = np.asarray(geom.S)
    az = np.arctan2(*[(S @ v) for v in (surf.basis[1], surf.basis[0])])  # atan2(n, e)
    sa = (np.gradient(lp(hd, 3.0 / surf.cell_km), surf.cell_km)[1] * math.cos(az)
          - np.gradient(lp(hd, 3.0 / surf.cell_km), surf.cell_km)[0] * math.sin(az))
    sb = (np.gradient(lp(ld, 3.0 / surf.cell_km), surf.cell_km)[1] * math.cos(az)
          - np.gradient(lp(ld, 3.0 / surf.cell_km), surf.cell_km)[0] * math.sin(az))
    return dict(relief_corr=float(np.corrcoef(a_[m], b_[m])[0, 1]),
                relief_amplitude=float(a_[m].std() / max(b_[m].std(), 1e-9)),
                relief_rms_lola_km=float(b_[m].std()),
                relief_rms_diff_km=float((a_ - b_)[m].std()),
                sun_slope_corr=float(np.corrcoef(sa[m], sb[m])[0, 1]),
                lola_heights=hl.astype(np.float32))
