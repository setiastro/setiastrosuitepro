# src/setiastro/saspro/photometry_aavso.py
"""
Shared, UI-free engine for star identification and AAVSO Extended-format
photometry submissions.

Both the Magnitude/Surface-Brightness tool and the Exoplanet Detector use the
same underlying operations:
  - turn a pixel centroid into an object identity (SIMBAD cone search),
  - resolve an AAVSO/VSX variable-star designation,
  - resolve a typed name back to RA/Dec (for "find star by name"),
  - pull observation time (JD) and airmass out of a FITS header,
  - format an AAVSO Extended upload file.

Everything here takes and returns plain Python values (floats, dicts, strings)
and raises on hard failure. All Qt — prompts, message boxes, file dialogs —
stays in the calling tool. That keeps this module importable from a worker,
unit-testable, and shareable.

Network calls (astroquery) have small bounded retries. Callers may pass a
`sleep_fn` (e.g. a non-blocking Qt sleep) to keep the GUI responsive.
"""
from __future__ import annotations

import re
import time
import math
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable, Tuple

import numpy as np

from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
import astropy.units as u


# ─────────────────────────────────────────────────────────────────────────
# AAVSO Extended format
# ─────────────────────────────────────────────────────────────────────────
@dataclass
class AavsoRow:
    """One AAVSO Extended data row.

    Column order (fixed by AAVSO):
      NAME,DATE,MAG,MERR,FILT,TRANS,MTYPE,CNAME,CMAG,KNAME,KMAG,AMASS,GROUP,CHART,NOTES
    """
    name: str                       # STARID — target designation (VSX name preferred)
    date_jd: float                  # DATE — Julian Date of mid-exposure
    mag: float                      # MAG
    filt: str                       # FILT — V, B, R, TG, TB, TR, CV, …
    merr: Optional[float] = None    # MERR — 1σ; None → "na"
    cname: str = "ENSEMBLE"         # comparison (ensemble ZP → ENSEMBLE)
    cmag: Any = "na"                # comparison mag ("na" for ensemble)
    kname: str = "na"               # check-star designation
    kmag: Any = "na"                # check-star catalog mag
    amass: Optional[float] = None   # airmass; None → "na"
    trans: str = "NO"               # transformed? NO until color terms exist
    mtype: str = "STD"              # magnitude type
    group: str = "na"
    chart: str = "na"
    notes: str = "MAG via ensemble ZP: m = m_inst + ZP"

    def to_fields(self, delim: str = ",") -> str:
        def num(x, fmt):
            if x is None:
                return "na"
            try:
                xf = float(x)
            except (TypeError, ValueError):
                return str(x)
            return "na" if not np.isfinite(xf) else format(xf, fmt)

        # NOTES must never contain the delimiter; swap to ';' if someone used ','
        note = str(self.notes).replace(delim, ";")
        fields = [
            str(self.name),
            num(self.date_jd, ".5f"),
            num(self.mag, ".3f"),
            num(self.merr, ".3f"),
            str(self.filt),
            str(self.trans),
            str(self.mtype),
            str(self.cname),
            (num(self.cmag, ".3f") if _is_number(self.cmag) else str(self.cmag)),
            str(self.kname),
            (num(self.kmag, ".3f") if _is_number(self.kmag) else str(self.kmag)),
            num(self.amass, ".3f"),
            str(self.group),
            str(self.chart),
            note,
        ]
        return delim.join(fields)


def _is_number(x) -> bool:
    try:
        float(x)
        return True
    except (TypeError, ValueError):
        return False


def format_aavso_extended(
    rows: List[AavsoRow],
    obscode: str,
    ra_deg: float,
    dec_deg: float,
    *,
    software: str = "Seti Astro Suite Pro",
    delim: str = ",",
    obstype: str = "CCD",
) -> str:
    """Return the full AAVSO Extended file body (header block + data rows).

    ra_deg / dec_deg are the TARGET coordinates for the #RA/#DEC header lines.
    """
    if not rows:
        raise ValueError("No rows to write.")

    c = SkyCoord(ra=float(ra_deg) * u.deg, dec=float(dec_deg) * u.deg, frame="icrs")
    ra_str = c.ra.to_string(unit=u.hour, sep=":", pad=True, precision=2)
    dec_str = c.dec.to_string(unit=u.degree, sep=":", pad=True, alwayssign=True, precision=1)

    header = [
        "#TYPE=EXTENDED",
        f"#OBSCODE={obscode}",
        f"#SOFTWARE={software}",
        f"#DELIM={delim}",
        "#DATE=JD",
        f"#OBSTYPE={obstype}",
        f"#RA={ra_str}",
        f"#DEC={dec_str}",
        "#NAME,DATE,MAG,MERR,FILT,TRANS,MTYPE,CNAME,CMAG,KNAME,KMAG,AMASS,GROUP,CHART,NOTES",
    ]
    body = [r.to_fields(delim=delim) for r in rows]
    return "\n".join(header + body) + "\n"


# Channel → (AAVSO tri-color filter code, catalog band the ZP is tied to)
RGB_FILTER_MAP = {
    "R": ("TR", "R"),
    "G": ("TG", "V"),   # green channel is calibrated against catalog V
    "B": ("TB", "B"),
}


def build_rgb_rows(
    *,
    name: str,
    date_jd: float,
    mags: Dict[str, float],           # {"R":.., "G":.., "B":..}
    merrs: Dict[str, Optional[float]],
    amass: Optional[float] = None,
    kname: str = "na",
    kmag: Any = "na",
    notes: Optional[str] = None,
) -> List[AavsoRow]:
    """Turn one RGB measurement into up to three AAVSO rows (TR/TG/TB).

    Channels with a non-finite magnitude are skipped.
    """
    out: List[AavsoRow] = []
    for ch in ("B", "G", "R"):        # conventional order B,G,R in reports
        m = mags.get(ch)
        if m is None or not np.isfinite(m):
            continue
        filt, _band = RGB_FILTER_MAP[ch]
        row = AavsoRow(
            name=name, date_jd=date_jd, mag=float(m), filt=filt,
            merr=merrs.get(ch), amass=amass, kname=kname, kmag=kmag,
        )
        if notes:
            row.notes = notes
        out.append(row)
    return out


def build_mono_row(
    *,
    name: str,
    date_jd: float,
    mag: float,
    merr: Optional[float],
    filt: str,
    amass: Optional[float] = None,
    kname: str = "na",
    kmag: Any = "na",
    notes: Optional[str] = None,
) -> List[AavsoRow]:
    row = AavsoRow(name=name, date_jd=date_jd, mag=float(mag), filt=filt,
                   merr=merr, amass=amass, kname=kname, kmag=kmag)
    if notes:
        row.notes = notes
    return [row]


# ─────────────────────────────────────────────────────────────────────────
# FITS FILTER card → AAVSO filter code
# ─────────────────────────────────────────────────────────────────────────
# Vendor/filler words stripped before matching.
_FILTER_JUNK = (
    "astrodon", "baader", "chroma", "optolong", "antlia", "svbony", "zwo",
    "playerone", "player one", "bessell", "bessel", "johnson", "cousins",
    "sloan", "photometric", "filter", "generic", "nm",
)

# Valid AAVSO photometric filter codes we are willing to emit.
_AAVSO_CODES = {
    "U", "B", "V", "R", "I", "SU", "SG", "SR", "SI", "SZ",
    "CV", "CR", "TG", "TB", "TR", "J", "H", "K",
}
# Of those, the ones we treat as unambiguous when they appear bare.
_AAVSO_CODES_HIGH = {"U", "B", "V", "CV", "CR", "TG", "TB", "TR"}

# Exact, high-confidence normalized token → code.
_FILTER_ALIASES_HIGH = {
    "u": "U", "b": "B", "v": "V",
    "rc": "R", "ic": "I",
    "u'": "SU", "g'": "SG", "r'": "SR", "i'": "SI", "z'": "SZ",
    "up": "SU", "gp": "SG", "rp": "SR", "ip": "SI", "zp": "SZ",
    "su": "SU", "sg": "SG", "sr": "SR", "si": "SI", "sz": "SZ",
    "cv": "CV", "cr": "CR",
    "tg": "TG", "tb": "TB", "tr": "TR",
    "j": "J", "k": "K",
}
# Ambiguous guesses — caller should confirm.
_FILTER_ALIASES_LOW = {
    "r": "R", "i": "I", "g": "SG", "z": "SZ",      # Cousins vs Sloan / generic
    "l": "CV", "lum": "CV", "luminance": "CV", "clear": "CV", "c": "CV",
    "cls": "CV", "uvir": "CV", "uvircut": "CV", "h": "H",
    "red": "TR", "green": "TG", "grn": "TG", "blue": "TB", "blu": "TB",
}
_NARROWBAND_TOKENS = {"o3", "oiii", "s2", "sii", "n2", "nii", "hb", "hbeta",
                      "hei", "heii", "siii", "ha", "halpha"}


def _is_narrowband_token(norm: str) -> bool:
    if norm in _NARROWBAND_TOKENS:
        return True
    for p in ("halpha", "ha", "hbeta", "oiii", "sii", "nii"):
        if norm.startswith(p):
            return True
    return False


def filter_to_aavso(raw) -> Tuple[Optional[str], str, str]:
    """Map a FITS FILTER value to an AAVSO photometric filter code.

    Returns (code, confidence, raw_str):
      code       : an AAVSO filter code, or None if unmappable/narrowband
      confidence : 'high' (auto-fill), 'low' (best guess — confirm), or 'none'
      raw_str    : the original header value (for display in a prompt)
    """
    if raw is None:
        return None, "none", ""
    s = str(raw).strip()
    if not s:
        return None, "none", ""

    norm = s.lower()
    for j in _FILTER_JUNK:
        norm = norm.replace(j, "")
    norm = re.sub(r"[\s_\-/]+", "", norm)
    if not norm:
        return None, "none", s

    if _is_narrowband_token(norm):
        return None, "none", s
    if norm in _FILTER_ALIASES_HIGH:
        return _FILTER_ALIASES_HIGH[norm], "high", s
    if norm in _FILTER_ALIASES_LOW:
        return _FILTER_ALIASES_LOW[norm], "low", s

    up = s.strip().upper()
    if up in _AAVSO_CODES:
        return up, ("high" if up in _AAVSO_CODES_HIGH else "low"), s

    return None, "none", s


# ─────────────────────────────────────────────────────────────────────────
# Header extraction: observation time (JD) and airmass
# ─────────────────────────────────────────────────────────────────────────
_TZ_RE = re.compile(r'([+-])(\d{2})(\d{2})$')   # -0700 -> -07:00


def _fix_iso_tz(s: str) -> str:
    s = str(s).strip()
    m = _TZ_RE.search(s)
    if m:
        s = s[:m.start()] + f"{m.group(1)}{m.group(2)}:{m.group(3)}"
    return s


def _hget(hdr, key):
    try:
        return hdr.get(key)
    except Exception:
        try:
            return hdr[key]
        except Exception:
            return None


def obs_time_from_header(hdr) -> Optional[Time]:
    """Best-effort mid-exposure (or start) time as an astropy Time (UTC)."""
    if hdr is None:
        return None
    for key in ("UT-OBS", "DATE-OBS", "DATE-END"):
        v = _hget(hdr, key)
        if isinstance(v, str) and v.strip():
            try:
                return Time(_fix_iso_tz(v), format="isot", scale="utc")
            except Exception:
                pass
    v = _hget(hdr, "MJD-OBS")
    if v is not None:
        try:
            return Time(float(v), format="mjd", scale="utc")
        except Exception:
            pass
    v = _hget(hdr, "JD") or _hget(hdr, "JD-OBS")
    if v is not None:
        try:
            return Time(float(v), format="jd", scale="utc")
        except Exception:
            pass
    return None


def obs_jd_from_header(hdr, exptime_s: Optional[float] = None,
                       to_midpoint: bool = True) -> Optional[float]:
    """JD of the observation. If `exptime_s` is given and to_midpoint is True,
    shifts a start-time by half the exposure (AAVSO wants mid-exposure)."""
    t = obs_time_from_header(hdr)
    if t is None:
        return None
    jd = float(t.utc.jd)
    if to_midpoint and exptime_s:
        try:
            jd += (float(exptime_s) / 2.0) / 86400.0
        except Exception:
            pass
    return jd


def exptime_from_header(hdr) -> Optional[float]:
    for k in ("EXPTIME", "EXPOSURE", "EXP_TIME", "EXPOSED"):
        v = _hget(hdr, k)
        if v is not None:
            try:
                f = float(v)
                if f > 0:
                    return f
            except Exception:
                pass
    return None


def _site_from_header(hdr) -> Optional[EarthLocation]:
    def first(*keys):
        for k in keys:
            v = _hget(hdr, k)
            if v is not None:
                return v
        return None

    lat = first("SITELAT", "LAT-OBS", "GEOLAT", "OBSGEO-B", "LATITUDE")
    lon = first("SITELONG", "LONG-OBS", "GEOLONG", "OBSGEO-L", "LONGITUD")
    elev = first("SITEELEV", "ALT-OBS", "HEIGHT", "ELEVATIO", "OBSGEO-H")
    if lat is None or lon is None:
        return None

    def _deg(x):
        # Accept float degrees or sexagesimal strings ("-30 12 34", "-30:12:34")
        try:
            return float(x)
        except (TypeError, ValueError):
            pass
        s = str(x).strip().replace(":", " ")
        parts = s.split()
        try:
            vals = [float(p) for p in parts]
        except ValueError:
            return None
        sign = -1.0 if (vals and vals[0] < 0) else 1.0
        acc = 0.0
        for i, val in enumerate(vals):
            acc += abs(val) / (60.0 ** i)
        return sign * acc

    latd = _deg(lat)
    lond = _deg(lon)
    if latd is None or lond is None:
        return None
    try:
        elevm = float(elev) if elev is not None else 0.0
    except Exception:
        elevm = 0.0
    try:
        return EarthLocation(lat=latd * u.deg, lon=lond * u.deg, height=elevm * u.m)
    except Exception:
        return None


def airmass_from_altitude(alt_deg: float) -> float:
    """Plane-parallel airmass = sec(z). Clamped away from the horizon."""
    alt_rad = np.deg2rad(np.clip(float(alt_deg), 0.1, 90.0))
    return float(1.0 / np.sin(alt_rad))


def airmass_from_header(hdr, ra_deg: Optional[float] = None,
                        dec_deg: Optional[float] = None) -> Optional[float]:
    """Airmass, in priority order:
      1) an AIRMASS/SECZ header card,
      2) an altitude card (ALT-OBS/OBJCTALT/CENTALT) → sec(z),
      3) site location + obs time + target RA/Dec via AltAz,
      4) None (caller should prompt / default to 1.0).
    """
    if hdr is not None:
        for k in ("AIRMASS", "SECZ"):
            v = _hget(hdr, k)
            if v is not None:
                try:
                    f = float(v)
                    if f >= 1.0:
                        return f
                except Exception:
                    pass
        for k in ("ALT-OBS", "OBJCTALT", "CENTALT", "ALTITUDE", "EL"):
            v = _hget(hdr, k)
            if v is not None:
                try:
                    return airmass_from_altitude(float(v))
                except Exception:
                    pass

    if hdr is not None and ra_deg is not None and dec_deg is not None:
        site = _site_from_header(hdr)
        t = obs_time_from_header(hdr)
        if site is not None and t is not None:
            try:
                target = SkyCoord(ra=float(ra_deg) * u.deg, dec=float(dec_deg) * u.deg, frame="icrs")
                altaz = target.transform_to(AltAz(obstime=t, location=site))
                alt = float(altaz.alt.deg)
                if alt > 0:
                    return airmass_from_altitude(alt)
            except Exception:
                pass
    return None


# ─────────────────────────────────────────────────────────────────────────
# Identification (SIMBAD / VSX / name resolution)
# ─────────────────────────────────────────────────────────────────────────
def _retry(fn: Callable, tries: int = 5, sleep_fn: Callable[[float], None] = time.sleep):
    last = None
    for attempt in range(1, tries + 1):
        try:
            return fn()
        except Exception as e:      # noqa: BLE001 — bubble the last error up
            last = e
            if attempt < tries:
                sleep_fn(1.0)
    if last is not None:
        raise last
    return None


def identify_object(ra_deg: float, dec_deg: float, *,
                    radius_arcsec: float = 5.0, tries: int = 5,
                    sleep_fn: Callable[[float], None] = time.sleep) -> Optional[Dict[str, Any]]:
    """SIMBAD cone search around a position.

    Returns {'main_id','otype','vmag','ra_deg','dec_deg','offset_arcsec'} for the
    nearest object, or None if nothing is within the radius. Raises on a hard
    network failure after `tries` attempts.
    """
    from astroquery.simbad import Simbad

    coord = SkyCoord(ra=float(ra_deg) * u.deg, dec=float(dec_deg) * u.deg, frame="icrs")

    def _do():
        s = Simbad()
        s.reset_votable_fields()
        s.add_votable_fields("otype")
        s.add_votable_fields("flux(V)")
        return s.query_region(coord, radius=radius_arcsec * u.arcsec)

    table = _retry(_do, tries=tries, sleep_fn=sleep_fn)
    if table is None or len(table) == 0:
        return None

    cols = {c.lower(): c for c in table.colnames}
    id_col = cols.get("main_id")
    ra_col = cols.get("ra")
    dec_col = cols.get("dec")
    otype_col = cols.get("otype")
    v_col = next((table.colnames[i] for i, c in enumerate(table.colnames)
                  if c.upper() in ("FLUX_V", "V")), None)
    if id_col is None:
        return None

    row = table[0]

    def _s(val):
        if isinstance(val, bytes):
            return val.decode("utf-8", "replace")
        return None if val is None else str(val)

    main_id = _s(row[id_col])
    otype = _s(row[otype_col]) if otype_col else None

    ra_val = dec_val = None
    if ra_col and dec_col:
        try:
            ra_val = float(row[ra_col]); dec_val = float(row[dec_col])
        except Exception:
            ra_val = dec_val = None

    offset = None
    if ra_val is not None and dec_val is not None:
        try:
            offset = float(coord.separation(
                SkyCoord(ra=ra_val * u.deg, dec=dec_val * u.deg, frame="icrs")).arcsec)
        except Exception:
            offset = None

    vmag = None
    if v_col:
        try:
            vv = float(row[v_col])
            if np.isfinite(vv):
                vmag = vv
        except Exception:
            vmag = None

    return {
        "main_id": (main_id.strip() if main_id else None),
        "otype": (otype.strip() if otype else None),
        "vmag": vmag,
        "ra_deg": ra_val,
        "dec_deg": dec_val,
        "offset_arcsec": offset,
    }


def resolve_vsx_name(*, main_id: Optional[str] = None,
                     ra_deg: Optional[float] = None, dec_deg: Optional[float] = None,
                     radius_arcsec: float = 5.0, tries: int = 3,
                     sleep_fn: Callable[[float], None] = time.sleep) -> Optional[str]:
    """Resolve an AAVSO/VSX variable-star designation.

    Tries a positional VSX cone search first (most reliable for photometry),
    then falls back to a name lookup against the SIMBAD main_id. Returns the VSX
    'Name' or None.
    """
    from astroquery.vizier import Vizier

    # 1) Positional cone search on B/vsx
    if ra_deg is not None and dec_deg is not None:
        coord = SkyCoord(ra=float(ra_deg) * u.deg, dec=float(dec_deg) * u.deg, frame="icrs")

        def _cone():
            v = Vizier(columns=["Name", "RAJ2000", "DEJ2000"], catalog="B/vsx")
            v.ROW_LIMIT = 5
            return v.query_region(coord, radius=radius_arcsec * u.arcsec)

        try:
            res = _retry(_cone, tries=tries, sleep_fn=sleep_fn)
            if res and len(res) > 0 and len(res[0]) > 0:
                t = res[0]
                name = t["Name"][0]
                if isinstance(name, bytes):
                    name = name.decode("utf-8", "replace")
                name = str(name).strip()
                if name:
                    return name
        except Exception:
            pass

    # 2) Name lookup against SIMBAD id
    if main_id:
        def _byname():
            Vizier.ROW_LIMIT = 1
            v = Vizier(columns=["Name"], catalog="B/vsx")
            return v.query_object(main_id)

        try:
            res = _retry(_byname, tries=tries, sleep_fn=sleep_fn)
            if res and len(res) > 0 and len(res[0]) > 0:
                name = res[0]["Name"][0]
                if isinstance(name, bytes):
                    name = name.decode("utf-8", "replace")
                name = str(name).strip()
                if name:
                    return name
        except Exception:
            pass

    return None


def resolve_name_to_radec(name: str, *, tries: int = 3,
                          sleep_fn: Callable[[float], None] = time.sleep
                          ) -> Optional[Tuple[float, float]]:
    """Resolve a typed object name (e.g. 'IRAS 17382-5308') to (ra_deg, dec_deg).

    Uses the SIMBAD/CDS name resolver via SkyCoord.from_name, then a direct
    SIMBAD query as a fallback. Returns None if unresolved.
    """
    nm = (name or "").strip()
    if not nm:
        return None

    def _from_name():
        c = SkyCoord.from_name(nm)
        return float(c.ra.deg), float(c.dec.deg)

    try:
        return _retry(_from_name, tries=tries, sleep_fn=sleep_fn)
    except Exception:
        pass

    # Fallback: SIMBAD query_object with explicit coordinate fields
    try:
        from astroquery.simbad import Simbad

        def _q():
            s = Simbad()
            return s.query_object(nm)

        table = _retry(_q, tries=tries, sleep_fn=sleep_fn)
        if table is None or len(table) == 0:
            return None
        cols = {c.lower(): c for c in table.colnames}
        ra_col = cols.get("ra")
        dec_col = cols.get("dec")
        if not ra_col or not dec_col:
            return None
        ra_raw = table[0][ra_col]
        dec_raw = table[0][dec_col]
        # Newer SIMBAD returns degrees; older returns sexagesimal strings.
        try:
            return float(ra_raw), float(dec_raw)
        except (TypeError, ValueError):
            c = SkyCoord(str(ra_raw), str(dec_raw), unit=(u.hourangle, u.deg))
            return float(c.ra.deg), float(c.dec.deg)
    except Exception:
        return None


def identify_name_and_vmag(ra_deg: float, dec_deg: float, *,
                           radius_arcsec: float = 5.0,
                           sleep_fn: Callable[[float], None] = time.sleep
                           ) -> Tuple[Optional[str], Optional[float]]:
    """Convenience for check-star lookup: (main_id, Vmag) at a position."""
    info = identify_object(ra_deg, dec_deg, radius_arcsec=radius_arcsec, sleep_fn=sleep_fn)
    if not info:
        return None, None
    return info.get("main_id"), info.get("vmag")


# ─────────────────────────────────────────────────────────────────────────
# Geometry helper (shared): flux-weighted centroid of a boolean mask
# ─────────────────────────────────────────────────────────────────────────
def flux_weighted_centroid(image: np.ndarray, mask: np.ndarray) -> Optional[Tuple[float, float]]:
    """Return (x, y) flux-weighted centroid (pixels) of `mask` over `image`.

    `image` may be mono (H,W) or RGB (H,W,3); RGB uses the mean plane. Falls
    back to the geometric centroid if the weighted sum is non-positive.
    """
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return None
    img = np.asarray(image)
    if img.ndim == 3:
        plane = img[..., :3].mean(axis=2)
    else:
        plane = img
    ys, xs = np.nonzero(m)
    w = plane[ys, xs].astype(np.float64)
    w = w - np.min(w)                       # relative to local floor
    wsum = float(w.sum())
    if not np.isfinite(wsum) or wsum <= 0:
        return float(xs.mean()), float(ys.mean())
    cx = float((xs * w).sum() / wsum)
    cy = float((ys * w).sum() / wsum)
    return cx, cy


if __name__ == "__main__":
    # Smoke test: reproduce David's row shape (TR/TG/TB, one target, one check star).
    rows = build_rgb_rows(
        name="000-BQP-573",
        date_jd=2461205.88939,
        mags={"R": 10.184, "G": 10.059, "B": 11.202},
        merrs={"R": 0.302, "G": 0.302, "B": 0.302},
        amass=1.7,
        kname="TYC 8729-1664-1",
        kmag=11.600,
        notes="MAG calc via ensemble: m=-2.5 log10(F/Fe)+K",
    )
    text = format_aavso_extended(rows, obscode="XYZ",
                                 ra_deg=265.659250, dec_deg=-53.963667)
    print(text)
    assert "#TYPE=EXTENDED" in text
    assert text.count("\n") >= 12  # header (9) + 3 rows + trailing
    print("OK")