# src/setiastro/saspro/exoplanet_transit.py
# Copyright (c) Franklin Marek / SetiAstro (www.setiastro.com). GPLv3.
"""
Transit timing, modelling and fitting for the exoplanet tool.

Pure numpy / scipy / astropy -- no Qt -- so every piece can be exercised
headless.  What lives here:

  frame times     mid-exposure UTC from whatever keys a header carries, then
                  BJD_TDB for the target: the time standard every transit
                  timing database (AAVSO Exoplanet DB, ETD, ExoClock) expects.
  airmass         from the header, or computed offline when it is absent.
  transit model   a quadratic limb-darkened star occulted by an opaque disc,
                  circular or eccentric orbit, optional exposure supersampling.
  fitting         grid + bounded least squares for the starting point, then an
                  affine-invariant ensemble MCMC for posteriors.  Errors are
                  rescaled by reduced chi^2 and by the time-averaging red-noise
                  factor beta, so a Tc uncertainty means what it says.
  archive lookup  NASA Exoplanet Archive priors by name or by position, and
                  Claret (2011) quadratic limb darkening from VizieR.
  report writer   AAVSO Exoplanet Database format (EXOTIC-compatible layout).

The method follows EXOTIC (NASA JPL Exoplanet Watch): transit x a1*exp(a2*X)
airmass systematics, Tc / Rp/R* / inc free, the rest fixed from the archive,
best aperture and comparison set chosen by residual scatter.  It is an
independent implementation, not a port -- no EXOTIC code is reused.  Where it
departs from EXOTIC on purpose it says so at the point of departure.
"""
from __future__ import annotations

import math
import re
import warnings
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np


# ═════════════════════════════ header access ══════════════════════════════

def header_value(hdr, key):
    """Raw value of `key` from a fits.Header or an XISF-style metadata dict."""
    if hdr is None:
        return None
    try:
        from astropy.io import fits
        if isinstance(hdr, fits.Header):
            try:
                return hdr.get(key)
            except Exception:
                return None
    except Exception:
        pass
    if isinstance(hdr, dict):
        kw = None
        im = hdr.get("image_meta")
        if isinstance(im, dict):
            kw = im.get("FITSKeywords")
        if kw is None:
            kw = hdr.get("FITSKeywords")
        if isinstance(kw, dict) and key in kw:
            ent = kw[key]
            if isinstance(ent, list):
                if not ent:
                    return None
                e0 = ent[0]
                return e0.get("value") if isinstance(e0, dict) else e0
            return ent
        v = hdr.get(key)
        if v is not None and not isinstance(v, (dict, list)):
            return v
        for sub in ("fits_header", "original_header", "header"):
            h = hdr.get(sub)
            if h is not None and h is not hdr:
                r = header_value(h, key)
                if r is not None:
                    return r
    return None


def header_float(hdr, keys) -> Optional[float]:
    """First of `keys` that parses as a finite float."""
    for k in keys:
        v = header_value(hdr, k)
        if v is None or isinstance(v, bool):
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            return f
    return None


def header_str(hdr, keys) -> str:
    for k in keys:
        v = header_value(hdr, k)
        if v is not None and str(v).strip():
            return str(v).strip()
    return ""


# ═══════════════════════════════ frame times ══════════════════════════════

_EXPTIME_KEYS = ("EXPTIME", "EXPOSURE", "EXPOS", "EXP")
_TZ_TAIL = re.compile(r"([+-])(\d{2}):?(\d{2})$")


def _iso_to_jd_utc(s) -> Optional[float]:
    """JD (UTC) from an ISO-ish date string, tolerating 'Z' and +HH:MM tails."""
    if not isinstance(s, str):
        return None
    s = s.strip()
    if not s:
        return None
    from astropy.time import Time
    offset_h = 0.0
    if s.endswith("Z"):
        s = s[:-1]
    m = _TZ_TAIL.search(s)
    # Only treat a +HH:MM tail as a zone offset when there is a time part --
    # a bare date "2024-01-02" ends in "-02" too.
    if m and "T" in s or (m and " " in s.strip()):
        sign = -1.0 if m.group(1) == "-" else 1.0
        offset_h = sign * (int(m.group(2)) + int(m.group(3)) / 60.0)
        s = s[:m.start()]
    try:
        t = Time(s, scale="utc")
    except Exception:
        try:
            t = Time(s.replace(" ", "T"), format="isot", scale="utc")
        except Exception:
            return None
    return float(t.jd) - offset_h / 24.0


def frame_mid_time_utc(hdr):
    """
    Mid-exposure time of one frame.

    Returns (jd_utc or None, exposure_s or None, how).

    FITS defines DATE-OBS as the START of the observation, so the exposure
    midpoint is DATE-OBS + EXPTIME/2.  EXOTIC adds the half exposure only
    when the keyword's comment contains the word 'start'; on files whose
    comment doesn't say so (MaxIm, N.I.N.A., many others) that leaves every
    timestamp early by half an exposure -- a full minute on 120 s subs, which
    is larger than the Tc error bars this tool reports.  Here the half
    exposure is always added to a start time, and explicit mid-exposure keys
    (DATE-AVG, MJD-MID, ...) are used as-is when present.
    """
    exp = header_float(hdr, _EXPTIME_KEYS)
    if exp is not None and exp < 0:
        exp = None

    for k in ("DATE-AVG", "DATE-MID"):
        jd = _iso_to_jd_utc(header_value(hdr, k))
        if jd is not None:
            return jd, exp, k
    for k in ("MJD-AVG", "MJD-MID"):
        v = header_float(hdr, (k,))
        if v is not None and v > 1.0e4:
            return v + 2400000.5, exp, k
    for k in ("JD-MID", "JD_MID"):
        v = header_float(hdr, (k,))
        if v is not None and v > 2.0e6:
            return v, exp, k

    start, how = None, ""
    ds = header_value(hdr, "DATE-OBS")
    if isinstance(ds, str) and ds.strip():
        s = ds.strip()
        if "T" not in s and ":" not in s:
            tobs = header_str(hdr, ("TIME-OBS", "UTSTART", "UT", "UT-OBS"))
            if tobs and ":" in tobs:
                s = f"{s}T{tobs}"
        start = _iso_to_jd_utc(s)
        how = "DATE-OBS"
    if start is None:
        start = _iso_to_jd_utc(header_value(hdr, "UT-OBS"))
        how = "UT-OBS"
    if start is None:
        v = header_float(hdr, ("MJD-OBS",))
        if v is not None and v > 1.0e4:
            start, how = v + 2400000.5, "MJD-OBS"
    if start is None:
        v = header_float(hdr, ("JD-OBS", "JD", "JULIAN"))
        if v is not None and v > 2.0e6:
            start, how = v, "JD"
    if start is None:
        end = _iso_to_jd_utc(header_value(hdr, "DATE-END"))
        if end is not None and exp is not None:
            return end - 0.5 * exp / 86400.0, exp, "DATE-END-EXPTIME/2"
        if end is not None:
            return end, exp, "DATE-END (no EXPTIME)"
        return None, exp, "no time keyword"

    if exp is not None:
        return start + 0.5 * exp / 86400.0, exp, f"{how}+EXPTIME/2"
    return start, exp, f"{how} (no EXPTIME - start time used)"


# ═════════════════════════════ site / location ════════════════════════════

_SITE_KEY_SETS = (
    ("SITELAT", "SITELONG", "SITEELEV"),
    ("LATITUDE", "LONGITUD", "HEIGHT"),          # LCO / BANZAI (deg N, deg E, m)
    ("LAT-OBS", "LONG-OBS", "ALT-OBS"),
    ("OBS-LAT", "OBS-LONG", "OBS-ELEV"),
    ("OBSLAT", "OBSLONG", "OBSALT"),
    ("GEOLAT", "GEOLON", "GEOELEV"),
    ("LAT", "LONG", "ELEVATION"),
)


def _angle_deg(v) -> Optional[float]:
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        pass
    try:
        from astropy.coordinates import Angle
        import astropy.units as u
        s = str(v).strip().replace("d", " ").replace("m", " ").replace("s", " ")
        return float(Angle(s, unit=u.deg).degree)
    except Exception:
        return None


def observatory_site(hdr):
    """
    (lat_deg, lon_deg_east, height_m, source) from the header, or None.
    Longitudes are taken as east-positive (the FITS/ASCOM/LCO convention).
    """
    for la, lo, el in _SITE_KEY_SETS:
        lat = _angle_deg(header_value(hdr, la))
        lon = _angle_deg(header_value(hdr, lo))
        if lat is None or lon is None or not (-90.0 <= lat <= 90.0):
            continue
        lon = ((lon + 180.0) % 360.0) - 180.0
        h = header_float(hdr, (el, "SITEELEV", "HEIGHT", "ELEVATION",
                               "ALT-OBS", "OBSGEO-H"))
        if h is None or not (-500.0 <= h <= 9000.0):
            h = 0.0
        return float(lat), float(lon), float(h), f"{la}/{lo}"
    x = header_float(hdr, ("OBSGEO-X",))
    y = header_float(hdr, ("OBSGEO-Y",))
    z = header_float(hdr, ("OBSGEO-Z",))
    if None not in (x, y, z):
        r = math.sqrt(x * x + y * y + z * z)
        if 6.2e6 < r < 6.5e6:
            try:
                from astropy.coordinates import EarthLocation
                import astropy.units as u
                loc = EarthLocation.from_geocentric(x, y, z, unit=u.m)
                return (float(loc.lat.deg), float(loc.lon.deg),
                        float(loc.height.to_value(u.m)), "OBSGEO-X/Y/Z")
            except Exception:
                pass
    return None


def _iers_offline():
    """Context that keeps astropy from downloading IERS tables mid-fit."""
    st = ExitStack()
    try:
        from astropy.utils import iers
        st.enter_context(iers.conf.set_temp("auto_download", False))
        if hasattr(iers.conf, "iers_degraded_accuracy"):
            st.enter_context(iers.conf.set_temp("iers_degraded_accuracy", "warn"))
    except Exception:
        pass
    return st


def bjd_tdb(jd_utc, ra_deg: float, dec_deg: float, site=None):
    """
    BJD_TDB for each JD_UTC, for a target at (ra, dec).

    `site` is (lat, lon_east, height_m[, ...]) or None for the geocentre.
    Ignoring the site costs at most R_earth/c = 21 ms, so a missing site
    degrades gracefully.  Runs with IERS auto-download off: the only IERS
    quantity involved here (UT1-UTC, for the site's position) moves the
    answer by microseconds, and a GUI tool must not stall on a network fetch.
    """
    from astropy.time import Time
    from astropy.coordinates import SkyCoord, EarthLocation
    import astropy.units as u

    jd = np.asarray(jd_utc, dtype=np.float64)
    target = SkyCoord(ra=float(ra_deg) * u.deg, dec=float(dec_deg) * u.deg,
                      frame="icrs")
    if site is not None:
        loc = EarthLocation.from_geodetic(float(site[1]) * u.deg,
                                          float(site[0]) * u.deg,
                                          float(site[2]) * u.m)
    else:
        loc = EarthLocation.from_geocentric(0.0, 0.0, 0.0, unit=u.m)
    with _iers_offline(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t = Time(jd, format="jd", scale="utc", location=loc)
        ltt = t.light_travel_time(target, kind="barycentric",
                                  ephemeris="builtin")
        out = t.tdb + ltt
        return np.asarray(out.jd1, np.float64) + np.asarray(out.jd2, np.float64)


def kasten_young_airmass(alt_deg):
    """Kasten & Young (1989) airmass; NaN below the horizon."""
    alt = np.asarray(alt_deg, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        x = 1.0 / (np.sin(np.radians(alt))
                   + 0.50572 * np.power(np.clip(alt, 0.0, 90.0) + 6.07995, -1.6364))
    return np.where(alt > 0.0, x, np.nan)


def airmass_from_time(jd_utc, ra_deg, dec_deg, lat_deg, lon_deg_east):
    """
    Offline airmass from time and site: GMST from the IAU 1982 expression
    plus Kasten-Young.  Precession is ignored -- a few tenths of a degree --
    which moves X by well under 1% at X=2; airmass is only a regressor here.
    """
    jd = np.asarray(jd_utc, dtype=np.float64)
    d = jd - 2451545.0
    gmst = (280.46061837 + 360.98564736629 * d) % 360.0
    ha = np.radians(gmst + float(lon_deg_east) - float(ra_deg))
    lat = math.radians(float(lat_deg))
    dec = math.radians(float(dec_deg))
    s = math.sin(lat) * math.sin(dec) + math.cos(lat) * math.cos(dec) * np.cos(ha)
    alt = np.degrees(np.arcsin(np.clip(s, -1.0, 1.0)))
    return kasten_young_airmass(alt)


# ═══════════════════════════ observing sessions ═══════════════════════════

def _instrument_key(hdr) -> str:
    parts = (header_str(hdr, ("SITEID", "SITE", "OBSERVAT")),
             header_str(hdr, ("TELESCOP", "TELID")),
             header_str(hdr, ("INSTRUME",)))
    return " / ".join(p for p in parts if p)


def assign_sessions(jd_utc, headers=None, gap_hours: float = 6.0):
    """
    Group frames into observing sessions: one instrument, one local night.

    A night runs local noon to local noon, from the site longitude in each
    frame's header: floor(JD + lon/360) steps exactly at local mean noon,
    because JD itself starts at noon UT.  Without a longitude on every frame
    the frames are split wherever the time gap exceeds `gap_hours`.

    The instrument is part of the key because LCO hands a sequence from one
    site to the next with no daytime gap, and each telescope has its own
    zero point and flat-field errors.

    Returns (session_index[n_frames], sessions) where sessions is a list,
    ordered by start time, of dicts: index, label, date, instrument,
    start_jd, end_jd, n.
    """
    jd = np.asarray(jd_utc, dtype=np.float64)
    n = jd.size
    hdrs = list(headers) if headers is not None else [None] * n
    if len(hdrs) != n:
        hdrs = (hdrs + [None] * n)[:n]
    if n == 0 or not np.all(np.isfinite(jd)):
        return np.zeros(n, dtype=int), [{
            "index": 0, "label": f"All frames — {n} frames", "date": "",
            "instrument": "", "start_jd": float("nan"), "end_jd": float("nan"), "n": n}]

    inst = [_instrument_key(h) for h in hdrs]
    lons = []
    for h in hdrs:
        site = observatory_site(h) if h is not None else None
        lons.append(site[1] if site else None)
    if all(lon is not None for lon in lons):
        night = np.floor(jd + np.asarray(lons, dtype=np.float64) / 360.0).astype(np.int64)
        by_noon = True
    else:
        order = np.argsort(jd)
        night = np.zeros(n, dtype=np.int64)
        k = 0
        for a, b in zip(order[:-1], order[1:]):
            if (jd[b] - jd[a]) * 24.0 > gap_hours:
                k += 1
            night[b] = k
        by_noon = False

    keys = {}
    for i in range(n):
        keys.setdefault((int(night[i]), inst[i]), []).append(i)
    groups = sorted(keys.items(), key=lambda kv: float(np.min(jd[kv[1]])))
    multi_inst = len({k[1] for k, _ in groups}) > 1

    from astropy.time import Time
    idx = np.zeros(n, dtype=int)
    sessions = []
    for s, ((nid, ins), members) in enumerate(groups):
        members = np.asarray(members, dtype=int)
        idx[members] = s
        t0, t1 = float(jd[members].min()), float(jd[members].max())
        try:
            # JD N.0 is noon UT on the evening's date; for gap-split sessions
            # use the UT date half a day before the first frame.
            ref = float(nid) if by_noon else t0 - 0.5
            date = Time(ref, format="jd").to_datetime().strftime("%Y-%m-%d")
        except Exception:
            date = ""
        label = f"Night {s + 1}: {date} — {members.size} frames"
        if multi_inst and ins:
            label += f" — {ins}"
        sessions.append({"index": s, "label": label, "date": date, "instrument": ins,
                         "start_jd": t0, "end_jd": t1, "n": int(members.size)})
    return idx, sessions


# ════════════════════════════ transit geometry ════════════════════════════

def _solve_kepler(M, e, n_iter: int = 40):
    E = M + e * np.sin(M)
    for _ in range(n_iter):
        f = E - e * np.sin(E) - M
        fp = 1.0 - e * np.cos(E)
        dE = f / fp
        E = E - dE
        if np.all(np.abs(dE) < 1e-12):
            break
    return E


def projected_separation(t, tc, per, ars, inc_deg, ecc=0.0, w_deg=90.0):
    """
    Sky-projected star-planet separation z (stellar radii) and a mask of
    which points have the planet IN FRONT of the star.  All parameters
    broadcast against t, so a (W,1) column of walkers against an (N,) time
    array gives (W,N) outputs.
    """
    t = np.asarray(t, dtype=np.float64)
    tc = np.asarray(tc, dtype=np.float64)
    per = np.asarray(per, dtype=np.float64)
    ars = np.asarray(ars, dtype=np.float64)
    inc = np.radians(np.asarray(inc_deg, dtype=np.float64))
    ecc = np.asarray(ecc, dtype=np.float64)

    if np.all(ecc == 0.0):
        ph = 2.0 * np.pi * (t - tc) / per
        x = ars * np.sin(ph)
        y = ars * np.cos(inc) * np.cos(ph)
        return np.hypot(x, y), np.cos(ph) > 0.0

    e = np.clip(ecc, 0.0, 0.95)
    w = np.radians(np.asarray(w_deg, dtype=np.float64))
    f_c = 0.5 * np.pi - w                      # true anomaly at mid-transit
    E_c = 2.0 * np.arctan2(np.sqrt(1.0 - e) * np.sin(0.5 * f_c),
                           np.sqrt(1.0 + e) * np.cos(0.5 * f_c))
    M_c = E_c - e * np.sin(E_c)
    M = 2.0 * np.pi * (t - tc) / per + M_c
    E = _solve_kepler(M, e)
    f = 2.0 * np.arctan2(np.sqrt(1.0 + e) * np.sin(0.5 * E),
                         np.sqrt(1.0 - e) * np.cos(0.5 * E))
    r = ars * (1.0 - e * np.cos(E))
    X = -r * np.cos(w + f)
    Y = -r * np.sin(w + f) * np.cos(inc)
    Z = r * np.sin(w + f) * np.sin(inc)
    return np.hypot(X, Y), Z > 0.0


_NODES: dict = {}


def _end_clustered_nodes(n: int):
    """
    Quadrature nodes on [0,1] clustered at both ends.

    The occulted-arc integrand has square-root behaviour at both limits: the
    arc closes at |z-k| and at z+k, and the limb profile mu = sqrt(1-r^2) has
    an infinite slope at the limb.  Substituting s = (1-cos th)/2 puts a
    sin(th) factor in the Jacobian that cancels those singularities, so plain
    Gauss-Legendre in th converges fast.
    """
    hit = _NODES.get(n)
    if hit is None:
        x, w = np.polynomial.legendre.leggauss(int(n))
        th = 0.5 * np.pi * (x + 1.0)
        s = 0.5 * (1.0 - np.cos(th))
        ws = w * (0.5 * np.pi) * (0.5 * np.sin(th))
        hit = (s, ws)
        _NODES[n] = hit
    return hit


def _ld_antideriv(mu, u1, u2):
    """H(mu) with dH/dmu = I(mu)*mu for the quadratic law."""
    m2 = mu * mu
    m3 = m2 * mu
    return (0.5 * m2
            - u1 * (0.5 * m2 - m3 / 3.0)
            - u2 * (0.5 * m2 - 2.0 * m3 / 3.0 + 0.25 * m2 * m2))


def occulted_flux(z, k, u1, u2, nq: int = 24):
    """
    Relative flux (1 = unocculted) of a quadratic limb-darkened star,
    I(mu) = 1 - u1(1-mu) - u2(1-mu)^2, behind an opaque disc of radius k at
    projected separation z (both in stellar radii).

    Blocked flux = integral over stellar radius r of I(r) times the length of
    the circle of radius r that lies inside the planet's disc.  Where the
    planet covers the whole circle (r < k - z) that part is done in closed
    form; the partial-arc band |z-k| < r < min(1, z+k) is integrated with the
    end-clustered rule above.  Accuracy is checked against brute-force 2-D
    integration in the tests -- better than 1e-6 of the stellar flux at the
    default node count.
    """
    z = np.asarray(z, dtype=np.float64)
    k, u1, u2 = (np.broadcast_to(np.asarray(a, dtype=np.float64), z.shape)
                 for a in (k, u1, u2))
    out = np.ones(z.shape, dtype=np.float64)
    m = (z < 1.0 + k) & (k > 0.0)
    if not np.any(m):
        return out

    zz = np.maximum(z[m], 1e-9)
    kk = k[m]
    a = u1[m]
    b = u2[m]

    H1 = _ld_antideriv(1.0, a, b)
    F0 = 2.0 * np.pi * H1

    r_full = np.clip(kk - zz, 0.0, 1.0)
    mu_full = np.sqrt(np.clip(1.0 - r_full * r_full, 0.0, 1.0))
    full = 2.0 * np.pi * (H1 - _ld_antideriv(mu_full, a, b))

    lo = np.abs(zz - kk)
    hi = np.minimum(1.0, zz + kk)
    width = np.clip(hi - lo, 0.0, None)
    s, ws = _end_clustered_nodes(nq)
    R = lo[:, None] + width[:, None] * s[None, :]
    R = np.maximum(R, 1e-12)
    cos_al = (R * R + (zz * zz - kk * kk)[:, None]) / (2.0 * R * zz[:, None])
    al = np.arccos(np.clip(cos_al, -1.0, 1.0))
    mu = np.sqrt(np.clip(1.0 - R * R, 0.0, 1.0))
    om = 1.0 - mu
    I = 1.0 - a[:, None] * om - b[:, None] * om * om
    part = width * np.sum(ws[None, :] * I * 2.0 * al * R, axis=1)

    out[m] = 1.0 - (full + part) / F0
    return out


def transit_model(t, tc, per, rp, ars, inc_deg, u1, u2, ecc=0.0, w_deg=90.0,
                  exptime_d: float = 0.0, n_sub: int = 1, nq: int = 24):
    """
    Relative flux of a transiting planet.  Scalar parameters give an (N,)
    array; (W,) parameter arrays give (W,N) -- one row per walker -- so an
    MCMC step is a single vectorised call.

    exptime_d / n_sub: average the model over the exposure.  A finite
    exposure smears ingress and egress; on 2-5 minute subs and a 15-minute
    ingress that bias in Tc and Rp/R* is real, so the fitter turns this on
    automatically for long exposures.
    """
    t = np.asarray(t, dtype=np.float64)
    pars = [np.asarray(p, dtype=np.float64) for p in
            (tc, per, rp, ars, inc_deg, u1, u2, ecc, w_deg)]
    batched = any(p.ndim > 0 and p.size > 1 for p in pars)
    W = max((p.size for p in pars), default=1) if batched else 1
    cols = [np.broadcast_to(p.reshape(-1), (W,))[:, None] if p.size > 1
            else np.full((W, 1), float(p.reshape(-1)[0])) for p in pars]
    tc_, per_, rp_, ars_, inc_, u1_, u2_, e_, w_ = cols

    n_sub = max(1, int(n_sub))
    if n_sub > 1 and exptime_d > 0.0:
        off = ((np.arange(n_sub) + 0.5) / n_sub - 0.5) * float(exptime_d)
        tt = (t[:, None] + off[None, :]).reshape(-1)
    else:
        n_sub = 1
        tt = t

    z, front = projected_separation(tt[None, :], tc_, per_, ars_, inc_, e_, w_)
    flux = np.ones(z.shape, dtype=np.float64)
    m = front & (z < 1.0 + rp_)
    if np.any(m):
        kk = np.broadcast_to(rp_, z.shape)[m]
        aa = np.broadcast_to(u1_, z.shape)[m]
        bb = np.broadcast_to(u2_, z.shape)[m]
        flux[m] = occulted_flux(z[m], kk, aa, bb, nq=nq)
    if n_sub > 1:
        flux = flux.reshape(W, t.size, n_sub).mean(axis=2)
    return flux if batched else flux[0]


def impact_parameter(ars, inc_deg, ecc=0.0, w_deg=90.0):
    e = np.asarray(ecc, dtype=np.float64)
    w = np.radians(np.asarray(w_deg, dtype=np.float64))
    return (np.asarray(ars, dtype=np.float64)
            * np.cos(np.radians(np.asarray(inc_deg, dtype=np.float64)))
            * (1.0 - e * e) / (1.0 + e * np.sin(w)))


def transit_duration(per, rp, ars, inc_deg, ecc=0.0, w_deg=90.0):
    """Total (first to fourth contact) duration T14 in the units of `per`."""
    per = np.asarray(per, dtype=np.float64)
    rp = np.asarray(rp, dtype=np.float64)
    ars = np.asarray(ars, dtype=np.float64)
    inc = np.radians(np.asarray(inc_deg, dtype=np.float64))
    e = np.asarray(ecc, dtype=np.float64)
    w = np.radians(np.asarray(w_deg, dtype=np.float64))
    b = impact_parameter(ars, np.degrees(inc), e, np.degrees(w))
    with np.errstate(invalid="ignore", divide="ignore"):
        arg = np.sqrt(np.clip((1.0 + rp) ** 2 - b * b, 0.0, None)) / (ars * np.sin(inc))
        T = per / np.pi * np.arcsin(np.clip(arg, 0.0, 1.0))
        T = T * np.sqrt(1.0 - e * e) / (1.0 + e * np.sin(w))
    return T


def ars_from_duration(per, t14, rp=0.1, b=0.0):
    """a/R* that gives total duration t14 (same units as per) at impact b."""
    per = float(per)
    t14 = float(t14)
    if not (per > 0 and t14 > 0):
        return float("nan")
    s = math.sin(math.pi * t14 / per)
    if s <= 0:
        return float("nan")
    chord2 = max((1.0 + rp) ** 2 - b * b, 1e-6)
    # sin(pi T/P) = sqrt(chord2)/(a sin i), with a cos i = b
    # => a^2 = (chord2 / s^2) + b^2
    return math.sqrt(chord2 / (s * s) + b * b)


# ════════════════════════════ noise diagnostics ═══════════════════════════

def robust_std(x) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return float("nan")
    med = np.median(x)
    return float(1.4826 * np.median(np.abs(x - med)))


def binned_rms(t, resid, width_d: float):
    """RMS of residuals averaged into time bins of width_d (days)."""
    t = np.asarray(t, dtype=np.float64)
    r = np.asarray(resid, dtype=np.float64)
    ok = np.isfinite(t) & np.isfinite(r)
    t, r = t[ok], r[ok]
    if t.size < 4 or width_d <= 0:
        return float("nan"), 0.0
    idx = np.floor((t - t.min()) / width_d).astype(np.int64)
    uniq, inv, cnt = np.unique(idx, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=r)
    means = sums / cnt
    good = cnt >= max(2, int(0.5 * np.median(cnt)))
    if np.count_nonzero(good) < 3:
        return float("nan"), float(np.mean(cnt)) if cnt.size else 0.0
    return float(np.std(means[good])), float(np.mean(cnt[good]))


def red_noise_beta(t, resid, bin_minutes: Sequence[float] = (10, 15, 20, 25, 30)):
    """
    Time-averaging red-noise factor (Pont, Zucker & Queloz 2006; Winn et al.
    2008).  For white noise the RMS of N-point bin means falls as 1/sqrt(N);
    beta is how much slower it actually falls, measured over bins near the
    ingress timescale.  Multiplying the per-point errors by beta makes the
    posterior widths honest in the presence of correlated noise.  EXOTIC does
    not apply this; it is the standard correction in the transit-timing
    literature, and ground-based light curves always need it.
    """
    t = np.asarray(t, dtype=np.float64)
    r = np.asarray(resid, dtype=np.float64)
    ok = np.isfinite(t) & np.isfinite(r)
    t, r = t[ok], r[ok]
    if t.size < 12:
        return 1.0, []
    s1 = float(np.std(r))
    if not (s1 > 0):
        return 1.0, []
    out = []
    for m in bin_minutes:
        width = float(m) / 1440.0
        idx = np.floor((t - t.min()) / width).astype(np.int64)
        uniq, inv, cnt = np.unique(idx, return_inverse=True, return_counts=True)
        means = np.bincount(inv, weights=r) / cnt
        good = cnt >= max(2, int(0.5 * np.median(cnt)))
        M = int(np.count_nonzero(good))
        if M < 4:
            continue
        N = float(np.mean(cnt[good]))
        if N < 2:
            continue
        sN = float(np.std(means[good]))
        expected = s1 / math.sqrt(N) * math.sqrt(M / (M - 1.0))
        if expected > 0:
            out.append((float(m), sN / expected))
    if not out:
        return 1.0, []
    beta = float(np.median([b for _, b in out]))
    return max(1.0, beta), out


# ═══════════════════════════════ fitting ══════════════════════════════════

SYSTEMATICS = {
    "airmass":        ("c0", "am"),
    "airmass_linear": ("c0", "am", "c1"),
    "linear":         ("c0", "c1"),
    "quadratic":      ("c0", "c1", "c2"),
    "none":           ("c0",),
}

SYSTEMATICS_LABELS = {
    "airmass":        "Airmass  a1·exp(a2·X)   (EXOTIC)",
    "airmass_linear": "Airmass + linear time",
    "linear":         "Linear in time",
    "quadratic":      "Quadratic in time",
    "none":           "None (normalisation only)",
}

PARAM_LABELS = {
    "tc": "Tc", "rp": "Rp/R*", "ars": "a/R*", "inc": "inc",
    "u1": "u1", "u2": "u2", "c0": "norm", "am": "airmass coeff",
    "c1": "slope", "c2": "curvature",
}


@dataclass
class TransitPriors:
    """Planet parameters and their 1-sigma uncertainties (0 = unknown)."""
    name: str = ""
    host: str = ""
    per: float = float("nan")
    per_err: float = 0.0
    tc_ref: float = float("nan")      # reference mid-transit, BJD_TDB
    tc_ref_err: float = 0.0
    rp: float = 0.1                   # Rp/R*
    rp_err: float = 0.0
    ars: float = float("nan")         # a/R*
    ars_err: float = 0.0
    inc: float = 90.0                 # degrees
    inc_err: float = 0.0
    ecc: float = 0.0
    omega: float = 90.0               # degrees
    u1: float = 0.465
    u2: float = 0.257
    teff: float = float("nan")
    logg: float = float("nan")
    feh: float = float("nan")
    vmag: float = float("nan")
    source: str = ""
    ld_source: str = ""

    def predicted_tc(self, t_mid: float):
        """(Tc nearest t_mid, its 1-sigma, epoch) or (nan, nan, 0)."""
        if not (np.isfinite(self.tc_ref) and np.isfinite(self.per) and self.per > 0):
            return float("nan"), float("nan"), 0
        n = int(round((t_mid - self.tc_ref) / self.per))
        pred = self.tc_ref + n * self.per
        sig = math.sqrt(max(self.tc_ref_err, 0.0) ** 2 + (n * max(self.per_err, 0.0)) ** 2)
        return float(pred), float(sig), n


@dataclass
class FitOptions:
    systematics: str = "airmass"
    free_inc: bool = True        # EXOTIC frees inc within +/-5 deg
    free_ars: bool = False
    free_ld: bool = False
    ld_sigma: float = 0.1
    clip_sigma: float = 4.0      # 0 disables the residual clip
    mcmc_steps: int = 2000
    mcmc_walkers: int = 0        # 0 = automatic
    exptime_s: float = 0.0
    use_beta: bool = True
    tc_window: Optional[tuple] = None   # (lo, hi) override
    seed: int = 20261119


@dataclass
class FitResult:
    ok: bool = False
    message: str = ""
    names: list = field(default_factory=list)
    best: dict = field(default_factory=dict)
    median: dict = field(default_factory=dict)
    err_lo: dict = field(default_factory=dict)
    err_hi: dict = field(default_factory=dict)
    fixed: dict = field(default_factory=dict)
    derived: dict = field(default_factory=dict)
    t: np.ndarray = None
    y: np.ndarray = None
    yerr: np.ndarray = None            # errors used by the sampler
    airmass: np.ndarray = None
    used: np.ndarray = None            # bool, False = clipped
    model: np.ndarray = None           # transit * systematics
    transit: np.ndarray = None
    systematics: np.ndarray = None
    stats: dict = field(default_factory=dict)
    samples: np.ndarray = None         # thinned posterior (n, ndim)
    priors: TransitPriors = None
    options: FitOptions = None
    notes: list = field(default_factory=list)


class TransitFitter:
    """
    Likelihood, priors and model for one light curve.

    Parameters are a flat vector in `self.names` order.  Fixed parameters
    live in `self.fixed`.  The systematics multiply the transit:

        F = T(t) * c0 * exp(am * (X - X0)) * (1 + c1*tau + c2*tau^2) [* (1 + fs*side)]

    with tau = t - t_mid (days) and X0 the median airmass.  Centring X and t
    decorrelates the normalisation from the slopes; the EXOTIC-convention
    a1 = c0*exp(-am*X0), a2 = am are reported alongside.
    """

    def __init__(self, t, y, yerr, airmass, priors: TransitPriors,
                 options: FitOptions, nq: int = 24, side=None,
                 fit_flip: Optional[bool] = None):
        self.t = np.asarray(t, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64)
        self.yerr = np.asarray(yerr, dtype=np.float64)
        X = np.asarray(airmass, dtype=np.float64) if airmass is not None else None
        if X is None or X.shape != self.t.shape or not np.all(np.isfinite(X)):
            X = np.ones_like(self.t)
        self.X = X
        self.X0 = float(np.median(X))
        self.t_mid = float(0.5 * (self.t.min() + self.t.max()))
        self.tau = self.t - self.t_mid
        self.pr = priors
        self.opt = options
        self.nq = int(nq)
        # Meridian flip inside the light curve: one multiplicative step for
        # the frames on the far pier side (AstroImageJ's flip-offset term).
        # Every star sits on a different part of the sensor after a flip, so
        # flat-field error steps the target/comparison ratio; measured 8-20
        # ppt on flat-fielded NGC660 frames.
        sd = np.asarray(side, dtype=np.float64) if side is not None else None
        if sd is None or sd.shape != self.t.shape:
            sd = None
        use = (bool(fit_flip) if fit_flip is not None
               else sd is not None and np.any(sd != 0) and np.any(sd == 0))
        # fit_flip=True keeps the parameter even if clipping emptied one side
        # (refits must share the first fit's parameter vector).
        self.side = (sd if sd is not None else np.zeros_like(self.t)) if use else None

        sys_names = SYSTEMATICS.get(options.systematics, SYSTEMATICS["airmass"])
        if "am" in sys_names and np.ptp(self.X) < 1e-4:
            # No airmass information: the coefficient would be unconstrained.
            sys_names = tuple(n for n in sys_names if n != "am")
        names = ["tc", "rp"]
        if options.free_ars:
            names.append("ars")
        if options.free_inc:
            names.append("inc")
        if options.free_ld:
            names += ["u1", "u2"]
        names += list(sys_names)
        if self.side is not None:
            names.append("fs")
        self.names = names
        self.ix = {n: i for i, n in enumerate(names)}

        self.fixed = {
            "per": float(priors.per), "ars": float(priors.ars),
            "inc": float(priors.inc), "u1": float(priors.u1),
            "u2": float(priors.u2), "ecc": float(priors.ecc),
            "w": float(priors.omega), "am": 0.0, "c1": 0.0, "c2": 0.0, "fs": 0.0,
        }

        exp_d = max(0.0, float(options.exptime_s)) / 86400.0
        # Supersample when one exposure is a meaningful slice of ingress.
        # Ingress ~ T14 * k/(1+k); 1/10 of that is a safe threshold.
        t14 = float(transit_duration(priors.per, max(priors.rp, 0.01),
                                     priors.ars, priors.inc,
                                     priors.ecc, priors.omega))
        ingress = t14 * max(priors.rp, 0.01) / (1.0 + max(priors.rp, 0.01)) \
            if np.isfinite(t14) and t14 > 0 else 0.01
        if exp_d > 0.1 * ingress:
            self.n_sub = int(min(15, max(3, math.ceil(exp_d / (0.05 * ingress)))))
            if self.n_sub % 2 == 0:
                self.n_sub += 1
        else:
            self.n_sub = 1
        self.exp_d = exp_d
        self.t14_prior = t14 if np.isfinite(t14) else float("nan")

        self._build_bounds()

    # ---- parameter space ------------------------------------------------

    def _build_bounds(self):
        pr, opt = self.pr, self.opt
        lo, hi = {}, {}
        span_lo, span_hi = float(self.t.min()), float(self.t.max())
        t14 = self.t14_prior if np.isfinite(self.t14_prior) and self.t14_prior > 0 else 0.1
        if opt.tc_window is not None:
            lo["tc"], hi["tc"] = float(opt.tc_window[0]), float(opt.tc_window[1])
        else:
            pred, sig, _ = pr.predicted_tc(self.t_mid)
            if np.isfinite(pred):
                half = max(10.0 * (sig if np.isfinite(sig) else 0.0), 0.5 * t14, 1.0 / 24.0)
                if np.isfinite(pr.per):
                    half = min(half, 0.25 * pr.per)
                lo["tc"], hi["tc"] = pred - half, pred + half
            else:
                lo["tc"], hi["tc"] = span_lo - 0.5 * t14, span_hi + 0.5 * t14
            lo["tc"] = max(lo["tc"], span_lo - 0.5 * t14)
            hi["tc"] = min(hi["tc"], span_hi + 0.5 * t14)
            if hi["tc"] <= lo["tc"]:
                lo["tc"], hi["tc"] = span_lo - 0.5 * t14, span_hi + 0.5 * t14
        rp0 = pr.rp if np.isfinite(pr.rp) and pr.rp > 0 else 0.1
        lo["rp"], hi["rp"] = 0.0, float(min(0.7, max(2.0 * rp0, 0.05)))
        lo["ars"], hi["ars"] = 1.5, 200.0
        if np.isfinite(pr.inc):
            lo["inc"], hi["inc"] = max(0.0, pr.inc - 5.0), min(90.0, pr.inc + 5.0)
        else:
            lo["inc"], hi["inc"] = 70.0, 90.0
        lo["u1"], hi["u1"] = 0.0, 1.0
        lo["u2"], hi["u2"] = -0.5, 1.0
        lo["c0"], hi["c0"] = 0.0, np.inf
        lo["am"], hi["am"] = -3.0, 3.0
        span = max(span_hi - span_lo, 1e-3)
        lo["c1"], hi["c1"] = -5.0 / span, 5.0 / span
        lo["c2"], hi["c2"] = -20.0 / span ** 2, 20.0 / span ** 2
        lo["fs"], hi["fs"] = -0.3, 0.3
        self.lo = np.array([lo[n] for n in self.names], dtype=np.float64)
        self.hi = np.array([hi[n] for n in self.names], dtype=np.float64)

    def initial_theta(self, tc0: Optional[float] = None) -> np.ndarray:
        pr = self.pr
        v = {
            "tc": tc0 if tc0 is not None else 0.5 * (self.lo[0] + self.hi[0]),
            "rp": pr.rp if np.isfinite(pr.rp) and pr.rp > 0 else 0.1,
            "ars": pr.ars, "inc": pr.inc, "u1": pr.u1, "u2": pr.u2,
            "c0": float(np.median(self.y)), "am": 0.0, "c1": 0.0, "c2": 0.0, "fs": 0.0,
        }
        th = np.array([v[n] for n in self.names], dtype=np.float64)
        span = self.hi - self.lo
        eps = np.where(np.isfinite(span), 1e-6 * span, 1e-9)
        return np.clip(th, self.lo + eps, np.where(np.isfinite(self.hi), self.hi - eps, th))

    def _get(self, theta, name):
        i = self.ix.get(name)
        if i is None:
            return np.asarray(self.fixed[name])
        return theta[..., i]

    # ---- model ------------------------------------------------------------

    def components(self, theta):
        """(transit, systematics) for theta of shape (D,) or (W,D)."""
        th = np.asarray(theta, dtype=np.float64)
        g = lambda n: self._get(th, n)
        tr = transit_model(self.t, g("tc"), self.fixed["per"], g("rp"),
                           g("ars"), g("inc"), g("u1"), g("u2"),
                           self.fixed["ecc"], self.fixed["w"],
                           exptime_d=self.exp_d, n_sub=self.n_sub, nq=self.nq)
        c0 = g("c0")
        am = g("am")
        c1 = g("c1")
        c2 = g("c2")
        fs = g("fs")
        if th.ndim == 2:
            c0, am, c1, c2, fs = (np.asarray(a).reshape(-1, 1) if np.ndim(a) else a
                                  for a in (c0, am, c1, c2, fs))
        sysm = (c0 * np.exp(am * (self.X - self.X0))
                * (1.0 + c1 * self.tau + c2 * self.tau ** 2))
        if self.side is not None:
            sysm = sysm * (1.0 + fs * self.side)
        return tr, sysm

    def model(self, theta):
        tr, sysm = self.components(theta)
        return tr * sysm

    # ---- priors ------------------------------------------------------------

    def log_prior(self, theta):
        th = np.atleast_2d(np.asarray(theta, dtype=np.float64))
        lp = np.zeros(th.shape[0])
        bad = np.any(th < self.lo, axis=1) | np.any(th > self.hi, axis=1)
        bad |= ~np.all(np.isfinite(th), axis=1)
        pr = self.pr
        if "ars" in self.ix:
            a = th[:, self.ix["ars"]]
            rp = th[:, self.ix["rp"]]
            bad |= a <= 1.0 + rp
            if pr.ars_err > 0 and np.isfinite(pr.ars):
                lp += -0.5 * ((a - pr.ars) / pr.ars_err) ** 2
        if "inc" in self.ix:
            i = th[:, self.ix["inc"]]
            if pr.inc_err > 0 and np.isfinite(pr.inc):
                lp += -0.5 * ((i - pr.inc) / pr.inc_err) ** 2
        # Grazing allowed, a miss is not: b < 1 + k.
        a = self._get(th, "ars")
        i = self._get(th, "inc")
        rp = self._get(th, "rp")
        b = impact_parameter(a, i, self.fixed["ecc"], self.fixed["w"])
        bad |= np.broadcast_to(b >= 1.0 + rp, bad.shape)
        if "u1" in self.ix:
            u1 = th[:, self.ix["u1"]]
            u2 = th[:, self.ix["u2"]]
            # Kipping (2013) physical region: positive, monotone intensity.
            bad |= (u1 < 0) | (u1 + u2 > 1.0) | (u1 + 2.0 * u2 < 0)
            s = max(self.opt.ld_sigma, 1e-3)
            lp += -0.5 * (((u1 - pr.u1) / s) ** 2 + ((u2 - pr.u2) / s) ** 2)
        lp[bad] = -np.inf
        return lp

    def log_prob(self, theta, yerr=None):
        th = np.atleast_2d(np.asarray(theta, dtype=np.float64))
        lp = self.log_prior(th)
        ok = np.isfinite(lp)
        out = np.full(th.shape[0], -np.inf)
        if not np.any(ok):
            return out
        s = self.yerr if yerr is None else yerr
        m = self.model(th[ok])
        chi2 = np.sum(((self.y[None, :] - m) / s[None, :]) ** 2, axis=1)
        out[ok] = lp[ok] - 0.5 * chi2
        out[~np.isfinite(out)] = -np.inf
        return out

    # ---- least squares ------------------------------------------------------

    def _lsq_resid(self, theta, mask, yerr):
        m = self.model(theta)
        r = (self.y - m) / yerr
        r = r[mask]
        extra = []
        pr = self.pr
        if "ars" in self.ix and pr.ars_err > 0 and np.isfinite(pr.ars):
            extra.append((theta[self.ix["ars"]] - pr.ars) / pr.ars_err)
        if "inc" in self.ix and pr.inc_err > 0 and np.isfinite(pr.inc):
            extra.append((theta[self.ix["inc"]] - pr.inc) / pr.inc_err)
        if "u1" in self.ix:
            s = max(self.opt.ld_sigma, 1e-3)
            extra.append((theta[self.ix["u1"]] - pr.u1) / s)
            extra.append((theta[self.ix["u2"]] - pr.u2) / s)
        if extra:
            r = np.concatenate([r, np.asarray(extra, dtype=np.float64)])
        return np.where(np.isfinite(r), r, 1e6)

    def tc_grid_start(self, n: int = 160, mask=None, yerr=None) -> float:
        """
        Coarse scan of Tc across its prior window with everything else at its
        starting value and the normalisation profiled out.  Least squares on
        Tc alone has a flat chi^2 wherever the transit model misses the data
        entirely, so starting it blind lands in that plateau.
        """
        mask = np.ones(self.t.size, bool) if mask is None else mask
        s = self.yerr if yerr is None else yerr
        lo, hi = self.lo[0], self.hi[0]
        grid = np.linspace(lo, hi, int(max(16, n)))
        base = self.initial_theta()
        th = np.repeat(base[None, :], grid.size, axis=0)
        th[:, 0] = grid
        if "c0" in self.ix:
            th[:, self.ix["c0"]] = 1.0
        tr, sysm = self.components(th)
        m = (tr * sysm)[:, mask]
        y = self.y[mask]
        w = 1.0 / s[mask] ** 2
        c0 = np.sum(w * y * m, axis=1) / np.maximum(np.sum(w * m * m, axis=1), 1e-30)
        chi2 = np.sum(w * (y - c0[:, None] * m) ** 2, axis=1)
        return float(grid[int(np.nanargmin(chi2))])

    def fit_lsq(self, theta0, mask=None, yerr=None):
        from scipy.optimize import least_squares
        mask = np.ones(self.t.size, bool) if mask is None else mask
        s = self.yerr if yerr is None else yerr
        lo = self.lo.copy()
        hi = self.hi.copy()
        x0 = np.clip(np.asarray(theta0, dtype=np.float64), lo, hi)
        # least_squares wants a strictly-interior start.
        span = np.where(np.isfinite(hi - lo), hi - lo, np.abs(x0) + 1.0)
        x0 = np.minimum(np.maximum(x0, lo + 1e-9 * span), hi - 1e-9 * span)
        try:
            res = least_squares(self._lsq_resid, x0, args=(mask, s),
                                bounds=(lo, hi), method="trf", x_scale="jac",
                                max_nfev=400 * len(x0))
            th = res.x
            try:
                J = res.jac
                cov = np.linalg.pinv(J.T @ J)
            except Exception:
                cov = None
        except Exception:
            th, cov = x0, None
        return th, cov


# ═════════════════════════════ ensemble sampler ═══════════════════════════

def ensemble_sampler(log_prob: Callable, p0: np.ndarray, nsteps: int,
                     rng: np.random.Generator, a: float = 2.0,
                     progress: Optional[Callable] = None,
                     cancelled: Optional[Callable] = None):
    """
    Affine-invariant ensemble MCMC (Goodman & Weare 2010 stretch move, the
    algorithm behind emcee), with the parallel red/black split so a whole
    half-ensemble is proposed -- and its likelihood evaluated -- in one
    vectorised call.  Returns (chain[nsteps,W,D], logp[nsteps,W], acceptance).
    """
    p = np.array(p0, dtype=np.float64, copy=True)
    W, D = p.shape
    if W % 2:
        raise ValueError("walker count must be even")
    lp = log_prob(p)
    chain = np.empty((nsteps, W, D))
    lps = np.empty((nsteps, W))
    halves = (np.arange(0, W, 2), np.arange(1, W, 2))
    n_acc = 0
    for s in range(nsteps):
        for g in (0, 1):
            act, oth = halves[g], halves[1 - g]
            n = act.size
            z = ((a - 1.0) * rng.random(n) + 1.0) ** 2 / a
            partners = p[oth[rng.integers(0, oth.size, n)]]
            prop = partners + z[:, None] * (p[act] - partners)
            lp_new = log_prob(prop)
            with np.errstate(invalid="ignore"):
                log_r = (D - 1.0) * np.log(z) + lp_new - lp[act]
            acc = np.log(rng.random(n)) < log_r
            if np.any(acc):
                p[act[acc]] = prop[acc]
                lp[act[acc]] = lp_new[acc]
                n_acc += int(acc.sum())
        chain[s] = p
        lps[s] = lp
        if progress is not None and (s % 25 == 0 or s == nsteps - 1):
            progress(s + 1, nsteps)
        if cancelled is not None and cancelled():
            return chain[: s + 1], lps[: s + 1], n_acc / max(1, (s + 1) * W)
    return chain, lps, n_acc / max(1, nsteps * W)


# ═══════════════════════════════ full fit ═════════════════════════════════

def _clip_mask(t, resid, sigma: float):
    """
    Residual outlier mask: a point is kept unless it sits more than `sigma`
    robust standard deviations from the median of its own neighbours.  Using
    neighbours (not a global median) keeps a slow systematic from turning a
    whole stretch of the curve into "outliers".
    """
    r = np.asarray(resid, dtype=np.float64)
    n = r.size
    keep = np.isfinite(r)
    if sigma <= 0 or n < 8:
        return keep
    order = np.argsort(t)
    rs = r[order]
    loc = np.empty(n)
    for i in range(n):
        a, b = max(0, i - 3), min(n, i + 4)
        nb = np.concatenate([rs[a:i], rs[i + 1:b]])
        nb = nb[np.isfinite(nb)]
        loc[i] = np.median(nb) if nb.size else 0.0
    d = rs - loc
    s = robust_std(d)
    if not (np.isfinite(s) and s > 0):
        return keep
    k_sorted = np.abs(d) <= sigma * s
    out = np.empty(n, bool)
    out[order] = k_sorted
    return keep & out


def fit_quick(t, y, yerr, airmass, priors: TransitPriors, options: FitOptions, side=None):
    """
    Least-squares fit only (no MCMC) -- the selection-stage fit.  Returns
    (score, theta, fitter) where score is EXOTIC's selection metric,
    std(residuals)/median(flux), computed after a 4-sigma neighbour clip.
    """
    f = TransitFitter(t, y, yerr, airmass, priors, options, nq=24, side=side)
    tc0 = f.tc_grid_start(n=96)
    th, _ = f.fit_lsq(f.initial_theta(tc0))
    resid = f.y - f.model(th)
    keep = _clip_mask(f.t, resid, 4.0)
    if np.count_nonzero(keep) < 0.8 * keep.size:
        keep = np.isfinite(resid)
    if np.count_nonzero(~keep):
        th, _ = f.fit_lsq(th, mask=keep)
        resid = f.y - f.model(th)
    med = float(np.median(f.y[keep])) if np.any(keep) else 1.0
    score = float(np.std(resid[keep]) / med) if med > 0 else float("inf")
    return score, th, f


def fit_transit(t, y, yerr, airmass, priors: TransitPriors,
                options: FitOptions,
                progress: Optional[Callable[[int, str], None]] = None,
                cancelled: Optional[Callable[[], bool]] = None,
                side=None) -> FitResult:
    """
    Full fit of one light curve.

      1. Tc grid scan, then bounded least squares.
      2. Neighbour-median residual clip (options.clip_sigma), refit.  EXOTIC
         clips once at 3 sigma around a Savitzky-Golay smooth; the
         neighbour median does the same job without flagging the steep
         parts of ingress/egress on deep, well-sampled eclipses.
      3. Error model: photometric errors x sqrt(chi2_red) (only ever
         inflated) x beta.  EXOTIC instead replaces the errors with the
         out-of-transit scatter; scaling the per-point errors keeps their
         relative weights (bad frames stay down-weighted) and adds the
         red-noise term EXOTIC lacks.
      4. Ensemble MCMC; medians and 16/84 percentiles reported.
    """
    res = FitResult(priors=priors, options=options)
    say = (lambda p, m: progress(p, m)) if progress else (lambda p, m: None)
    stop = cancelled or (lambda: False)

    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    e = np.asarray(yerr, dtype=np.float64) if yerr is not None else np.full_like(y, np.nan)
    X = np.asarray(airmass, dtype=np.float64) if airmass is not None else np.ones_like(t)
    ok = np.isfinite(t) & np.isfinite(y) & (y > 0)
    if X.shape == t.shape:
        ok &= np.isfinite(X)
    else:
        X = np.ones_like(t)
    if np.count_nonzero(ok) < 15:
        res.message = "Fewer than 15 usable points -- nothing to fit."
        return res
    Ps = (np.asarray(side, dtype=np.float64) if side is not None
          and np.shape(side) == np.shape(ok) else None)
    t, y, e, X = t[ok], y[ok], e[ok], X[ok]
    order = np.argsort(t)
    t, y, e, X = t[order], y[order], e[order], X[order]
    if Ps is not None:
        Ps = Ps[ok][order]

    # Normalise to the median so every coefficient is order-unity.
    med = float(np.median(y))
    y = y / med
    e = e / med
    bad_e = ~(np.isfinite(e) & (e > 0))
    if np.all(bad_e):
        e = np.full_like(y, max(robust_std(np.diff(y)) / math.sqrt(2.0), 1e-5))
        res.notes.append("No photometric errors: using point-to-point scatter.")
    elif np.any(bad_e):
        e[bad_e] = float(np.median(e[~bad_e]))

    if not (np.isfinite(priors.per) and priors.per > 0):
        res.message = "A period is required (use any plausible value for an unknown planet)."
        return res
    if not (np.isfinite(priors.ars) and priors.ars > 1.0):
        res.message = "a/R* is required."
        return res

    say(2, "Scanning Tc…")
    f = TransitFitter(t, y, e, X, priors, options, nq=24, side=Ps)
    if f.side is not None:
        res.notes.append("Meridian flip inside the night: a flux step for the far pier side "
                         "is fitted (fs).")
    if f.n_sub > 1:
        res.notes.append(f"Model averaged over the {options.exptime_s:.0f} s exposure "
                         f"({f.n_sub} sub-samples).")
    tc0 = f.tc_grid_start(n=200)
    say(5, "Least-squares fit…")
    th, cov = f.fit_lsq(f.initial_theta(tc0))
    if stop():
        res.message = "Cancelled."
        return res

    used = np.ones(t.size, bool)
    if options.clip_sigma and options.clip_sigma > 0:
        for _ in range(2):
            resid = f.y - f.model(th)
            keep = _clip_mask(t, resid, float(options.clip_sigma))
            if np.count_nonzero(keep) < 0.7 * t.size:
                break
            if np.array_equal(keep, used):
                break
            used = keep
            th, cov = f.fit_lsq(th, mask=used)
    n_used = int(np.count_nonzero(used))
    k = len(f.names)

    resid = (f.y - f.model(th))[used]
    chi2 = float(np.sum((resid / e[used]) ** 2))
    dof = max(1, n_used - k)
    chi2_red = chi2 / dof
    scale = math.sqrt(max(1.0, chi2_red))
    beta, beta_table = red_noise_beta(t[used], resid)
    if not options.use_beta:
        beta_eff = 1.0
    else:
        beta_eff = beta
    s_final = e * scale * beta_eff

    say(10, "Refitting with the final error model…")
    fe = TransitFitter(t[used], y[used], s_final[used], X[used], priors, options, nq=24,
                       side=(Ps[used] if Ps is not None else None),
                       fit_flip=f.side is not None)
    fe.lo, fe.hi = f.lo, f.hi        # same prior window as the scan
    th, cov = fe.fit_lsq(th)

    # ---- MCMC ----------------------------------------------------------------
    D = len(fe.names)
    W = int(options.mcmc_walkers) if options.mcmc_walkers else max(24, 4 * D)
    W += W % 2
    rng = np.random.default_rng(int(options.seed))
    scale_vec = None
    if cov is not None:
        d = np.sqrt(np.clip(np.diag(cov), 0, None))
        if np.all(np.isfinite(d)) and np.all(d > 0):
            scale_vec = d
    if scale_vec is None:
        guess = {"tc": 1e-3, "rp": 1e-3, "ars": 0.1, "inc": 0.1, "u1": 0.01,
                 "u2": 0.01, "c0": 1e-4, "am": 1e-4, "c1": 1e-4, "c2": 1e-4}
        scale_vec = np.array([guess[n] for n in fe.names])
    p0 = th[None, :] + 0.1 * scale_vec[None, :] * rng.standard_normal((W, D))
    lp0 = fe.log_prob(p0)
    for _ in range(50):
        badw = ~np.isfinite(lp0)
        if not np.any(badw):
            break
        p0[badw] = th[None, :] + 0.01 * scale_vec[None, :] * rng.standard_normal((int(badw.sum()), D))
        lp0[badw] = fe.log_prob(p0[badw])
    if not np.all(np.isfinite(lp0)):
        p0[~np.isfinite(lp0)] = th
    nsteps = max(200, int(options.mcmc_steps))

    def _prog(s, n):
        say(10 + int(88 * s / n), f"MCMC step {s}/{n}")

    chain, lps, acc = ensemble_sampler(fe.log_prob, p0, nsteps, rng,
                                       progress=_prog, cancelled=stop)
    if stop():
        res.message = "Cancelled."
        return res
    burn = chain.shape[0] // 2
    flat = chain[burn:].reshape(-1, D)
    flat_lp = lps[burn:].reshape(-1)
    fin = np.isfinite(flat_lp)
    flat, flat_lp = flat[fin], flat_lp[fin]
    if flat.shape[0] < 10:
        res.message = "MCMC produced no valid samples."
        return res
    best = flat[int(np.argmax(flat_lp))]
    q16, q50, q84 = np.percentile(flat, [15.865, 50.0, 84.135], axis=0)

    res.names = list(fe.names)
    for i, n in enumerate(fe.names):
        res.best[n] = float(best[i])
        res.median[n] = float(q50[i])
        res.err_lo[n] = float(q50[i] - q16[i])
        res.err_hi[n] = float(q84[i] - q50[i])
    res.fixed = {n: v for n, v in fe.fixed.items() if n not in fe.ix}

    # ---- derived quantities ------------------------------------------------------
    get = lambda n: flat[:, fe.ix[n]] if n in fe.ix else np.full(flat.shape[0], fe.fixed[n])
    per = fe.fixed["per"]
    rp_s, ars_s, inc_s = get("rp"), get("ars"), get("inc")
    t14_s = transit_duration(per, rp_s, ars_s, inc_s, fe.fixed["ecc"], fe.fixed["w"])
    b_s = impact_parameter(ars_s, inc_s, fe.fixed["ecc"], fe.fixed["w"])

    def _summ(a):
        a = np.asarray(a, dtype=np.float64)
        a = a[np.isfinite(a)]
        if a.size == 0:
            return (float("nan"),) * 3
        p = np.percentile(a, [15.865, 50.0, 84.135])
        return float(p[1]), float(p[1] - p[0]), float(p[2] - p[1])

    der = {}
    der["depth_rp2"] = _summ(rp_s ** 2)
    der["t14_h"] = _summ(t14_s * 24.0)
    der["b"] = _summ(b_s)
    # model depth at mid-transit, including limb darkening
    tr_best, sys_best = fe.components(best)
    der["depth_model"] = float(1.0 - np.min(transit_model(
        np.array([res.best["tc"]]), res.best["tc"], per, res.best["rp"],
        res.best.get("ars", fe.fixed["ars"]), res.best.get("inc", fe.fixed["inc"]),
        res.best.get("u1", fe.fixed["u1"]), res.best.get("u2", fe.fixed["u2"]),
        fe.fixed["ecc"], fe.fixed["w"])))
    if "am" in fe.ix:
        a1 = get("c0") * np.exp(-get("am") * fe.X0)
        der["a1"] = _summ(a1)
        der["a2"] = _summ(get("am"))
    else:
        der["a1"] = _summ(get("c0"))
        der["a2"] = (0.0, 0.0, 0.0)
    pred, sig_pred, epoch = priors.predicted_tc(res.median["tc"])
    if np.isfinite(pred):
        oc = (flat[:, 0] - pred) * 1440.0
        m_, l_, h_ = _summ(oc)
        der["oc_min"] = (m_, l_, h_)
        der["oc_pred_sigma_min"] = float(sig_pred * 1440.0) if np.isfinite(sig_pred) else float("nan")
        der["epoch"] = int(epoch)
        der["tc_pred"] = float(pred)
    res.derived = der

    # ---- curves at the best fit, on ALL points (clipped ones shown) -----------------
    f_all = TransitFitter(t, y, s_final, X, priors, options, nq=24, side=Ps,
                          fit_flip=f.side is not None)
    trb, sysb = f_all.components(np.asarray([res.best[n] for n in fe.names]))
    res.t, res.y, res.yerr, res.airmass = t, y, s_final, X
    res.used = used
    res.transit, res.systematics = trb, sysb
    res.model = trb * sysb
    r_all = (y - res.model)
    rms = float(np.std(r_all[used]))
    rms10, n10 = binned_rms(t[used], r_all[used], 10.0 / 1440.0)
    cad = float(np.median(np.diff(t))) * 1440.0 if t.size > 1 else float("nan")
    res.stats = {
        "n_points": int(t.size), "n_used": n_used, "n_clipped": int(t.size - n_used),
        "chi2_red_raw": float(chi2_red), "err_scale": float(scale),
        "beta": float(beta), "beta_used": float(beta_eff), "beta_table": beta_table,
        "rms_ppt": rms * 1e3, "rms10_ppt": rms10 * 1e3, "pts_per_10min": n10,
        "cadence_min": cad, "acceptance": float(acc), "walkers": W, "steps": nsteps,
        "norm_median": med, "X0": fe.X0, "t_mid": fe.t_mid, "n_sub": fe.n_sub,
    }
    if acc < 0.1:
        res.notes.append(f"Low MCMC acceptance ({acc:.2f}); posteriors may be under-sampled.")
    thin = max(1, flat.shape[0] // 4000)
    res.samples = flat[::thin]
    res.ok = True
    res.message = "OK"
    say(100, "Done")
    return res


# ═══════════════════════════ NASA Exoplanet Archive ═══════════════════════

NEA_TAP = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
_NEA_COLS = ("pl_name,hostname,ra,dec,pl_orbper,pl_orbpererr1,pl_orbpererr2,"
             "pl_tranmid,pl_tranmiderr1,pl_tranmiderr2,pl_ratror,pl_ratrorerr1,"
             "pl_ratrorerr2,pl_trandep,pl_ratdor,pl_ratdorerr1,pl_ratdorerr2,"
             "pl_orbincl,pl_orbinclerr1,pl_orbinclerr2,pl_orbeccen,pl_orblper,"
             "pl_imppar,pl_orbsmax,pl_radj,st_rad,st_teff,st_logg,st_met,"
             "sy_vmag,pl_trandur")


def _nea_query(where: str, timeout: float = 20.0):
    import requests
    q = f"select {_NEA_COLS} from pscomppars where {where}"
    # POST, not GET: the archive's web firewall answers 403 to a GET whose
    # query string contains ADQL geometry (contains(point(...),circle(...))),
    # while the identical query POSTed is served normally.
    r = requests.post(NEA_TAP, data={"query": q, "format": "json"},
                      timeout=timeout,
                      headers={"User-Agent": "SetiAstroSuitePro exoplanet tool"})
    r.raise_for_status()
    return r.json()


def _num(row, key):
    v = row.get(key)
    try:
        f = float(v)
        return f if math.isfinite(f) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _sym_err(row, key):
    a, b = _num(row, key + "err1"), _num(row, key + "err2")
    if np.isfinite(a) and np.isfinite(b):
        return float(math.sqrt(abs(a * b)))
    if np.isfinite(a):
        return abs(a)
    if np.isfinite(b):
        return abs(b)
    return 0.0


def priors_from_nea_row(row: dict) -> TransitPriors:
    """
    TransitPriors from one pscomppars row, filling gaps the way EXOTIC does:
    Rp/R* from the ratio, else sqrt(depth), else Rp/R*; a/R* from the ratio,
    else a/R*; inc from the table, else from the impact parameter, else 90.
    Uncertainties are the geometric mean of the two-sided errors.
    """
    p = TransitPriors()
    p.name = str(row.get("pl_name") or "")
    p.host = str(row.get("hostname") or "")
    p.per = _num(row, "pl_orbper")
    p.per_err = _sym_err(row, "pl_orbper")
    p.tc_ref = _num(row, "pl_tranmid")
    p.tc_ref_err = _sym_err(row, "pl_tranmid")
    rp = _num(row, "pl_ratror")
    p.rp_err = _sym_err(row, "pl_ratror")
    if not np.isfinite(rp):
        dep = _num(row, "pl_trandep")
        if np.isfinite(dep) and dep > 0:
            rp = math.sqrt(dep / 100.0)
    if not np.isfinite(rp):
        rj, rs = _num(row, "pl_radj"), _num(row, "st_rad")
        if np.isfinite(rj) and np.isfinite(rs) and rs > 0:
            rp = rj * 0.10045 / rs
    p.rp = rp if np.isfinite(rp) and rp > 0 else 0.1
    ars = _num(row, "pl_ratdor")
    p.ars_err = _sym_err(row, "pl_ratdor")
    if not np.isfinite(ars):
        a_au, rs = _num(row, "pl_orbsmax"), _num(row, "st_rad")
        if np.isfinite(a_au) and np.isfinite(rs) and rs > 0:
            ars = a_au * 215.032 / rs
    p.ars = ars
    inc = _num(row, "pl_orbincl")
    p.inc_err = _sym_err(row, "pl_orbincl")
    if not np.isfinite(inc):
        b = _num(row, "pl_imppar")
        if np.isfinite(b) and np.isfinite(ars) and ars > 0 and abs(b) < ars:
            inc = math.degrees(math.acos(abs(b) / ars))
    p.inc = inc if np.isfinite(inc) else 90.0
    e = _num(row, "pl_orbeccen")
    p.ecc = e if np.isfinite(e) and 0 <= e < 0.95 else 0.0
    w = _num(row, "pl_orblper")
    p.omega = w if np.isfinite(w) else 90.0
    p.teff = _num(row, "st_teff")
    p.logg = _num(row, "st_logg")
    p.feh = _num(row, "st_met")
    p.vmag = _num(row, "sy_vmag")
    if not np.isfinite(p.ars) and np.isfinite(p.per):
        dur_h = _num(row, "pl_trandur")
        if np.isfinite(dur_h) and dur_h > 0:
            p.ars = ars_from_duration(p.per, dur_h / 24.0, p.rp, 0.0)
    p.source = "NASA Exoplanet Archive (pscomppars)"
    return p


def _adql_str(s: str) -> str:
    return str(s).replace("'", "''")


def nea_lookup_name(name: str, timeout: float = 20.0):
    """List of pscomppars rows matching a planet or host name."""
    nm = _adql_str(name.strip())
    if not nm:
        return []
    rows = _nea_query(f"upper(pl_name) = upper('{nm}')", timeout)
    if rows:
        return rows
    rows = _nea_query(f"upper(hostname) = upper('{nm}')", timeout)
    if rows:
        return rows
    # "HAT-P-32b" -> "HAT-P-32 b"
    m = re.match(r"^(.*\S)([b-z])$", name.strip())
    if m:
        alt = _adql_str(f"{m.group(1)} {m.group(2)}")
        rows = _nea_query(f"upper(pl_name) = upper('{alt}')", timeout)
    return rows or []


def nea_lookup_position(ra_deg: float, dec_deg: float, radius_arcsec: float = 20.0,
                        timeout: float = 20.0):
    r = float(radius_arcsec) / 3600.0
    return _nea_query(
        f"contains(point('icrs',ra,dec),circle('icrs',{float(ra_deg):.7f},"
        f"{float(dec_deg):.7f},{r:.6f}))=1", timeout)


# ═══════════════════════════════ filters & LD ═════════════════════════════

# AAVSO filter code -> (description, FWHM blue edge nm, red edge nm, Claret
# 2011 passband).  Passband edges are AAVSO's published FWHM ranges.  Clear
# and broad exoplanet filters have no Claret passband; they map to the
# nearest standard band (marked approximate in the UI).
FILTERS = {
    "U":   ("Johnson U", 333.8, 398.8, "U"),
    "B":   ("Johnson B", 391.6, 480.6, "B"),
    "V":   ("Johnson V", 502.8, 586.8, "V"),
    "R":   ("Cousins R", 561.7, 719.7, "R"),
    "I":   ("Cousins I", 721.0, 875.0, "I"),
    "SU":  ("Sloan u", 321.8, 386.8, "u*"),
    "SG":  ("Sloan g", 402.5, 551.5, "g*"),
    "SR":  ("Sloan r", 553.1, 693.1, "r*"),
    "SI":  ("Sloan i", 697.5, 827.5, "i*"),
    "SZ":  ("Sloan z", 841.2, 978.2, "z*"),
    "ZS":  ("PanSTARRS z-short", 826.0, 920.0, "z*"),
    "CV":  ("Clear (unfiltered) reduced to V", 350.0, 850.0, "R"),
    "CBB": ("Astrodon ExoPlanet-BB", 500.0, 1000.0, "R"),
}
APPROXIMATE_LD_BANDS = {"CV", "CBB", "ZS"}


def guess_filter_code(filter_keyword: str) -> Optional[str]:
    """Best AAVSO filter code for a FITS FILTER value, or None if unknown."""
    raw = (filter_keyword or "").strip()
    s = raw.lower()
    sloan = any(tag in s for tag in ("sdss", "sloan", "'", "prime")) or \
        re.fullmatch(r"[ugriz]p", s) is not None
    core = re.sub(r"(johnson|bessell|cousins|sdss|sloan|lco|filter|prime|[\s_\-'\"*])",
                  "", s)
    if core in ("", "clear", "c", "l", "lum", "luminance", "cv", "w", "air",
                "open", "none", "nofilter", "unfiltered", "uvir", "uvircut", "lpr"):
        return "CV"
    if core in ("cbb", "exo", "exoplanet", "exoplanetbb", "bb"):
        return "CBB"
    if core in ("zs", "zshort"):
        return "ZS"
    if core in ("up", "su") or (sloan and core == "u"):
        return "SU"
    if core in ("gp", "sg") or (sloan and core == "g"):
        return "SG"
    if core in ("rp", "sr") or (sloan and core == "r"):
        return "SR"
    if core in ("ip", "si") or (sloan and core == "i"):
        return "SI"
    if core in ("zp", "sz", "z"):
        return "SZ"
    if core in ("v", "johnsonv", "green", "g"):
        return "V"
    if core in ("b", "blue"):
        return "B"
    if core in ("u",):
        return "U"
    if core in ("r", "rc", "red"):
        return "R"
    if core in ("i", "ic"):
        return "I"
    return None


def claret_quadratic_ld(teff: float, logg: float, feh: float, band: str,
                        status_cb: Optional[Callable[[str], None]] = None):
    """
    Quadratic limb darkening (u1, u2) from Claret & Bloemen (2011),
    VizieR J/A+A/529/A75/table-af (ATLAS models, least-squares fit, xi = 2
    km/s).  Picks the nearest metallicity and log g on the grid and
    interpolates linearly in Teff.

    Runs on whichever VizieR mirror cds_health finds serving, failing over
    to another if that one stops answering (Strasbourg's votable endpoint
    has been answering 404 for weeks; CfA and IDIA have each been the only
    live mirror on different days).  A mirror that answers with no rows is
    passed over once for the next one -- mirrors are partial.  Raises on
    failure; callers fall back to the values already in the form.
    """
    from setiastro.saspro import cds_health

    if not (np.isfinite(teff) and teff > 0):
        raise ValueError("stellar Teff unknown")
    logg = 4.5 if not np.isfinite(logg) else float(logg)
    feh = 0.0 if not np.isfinite(feh) else float(feh)

    def _run(host):
        v = cds_health.vizier_client(host, columns=["logg", "Teff", "Z", "xi", "a", "b",
                                                    "Filt", "Met", "Mod"],
                                     row_limit=-1, timeout=20)
        tabs = v.query_constraints(
            catalog="J/A+A/529/A75/table-af",
            Teff=f"{teff - 260:.0f}..{teff + 260:.0f}",
            logg=f"{logg - 0.51:.2f}..{logg + 0.51:.2f}",
            Z=f"{feh - 0.6:.2f}..{feh + 0.6:.2f}",
            xi="2", Met="L", Mod="A")
        return tabs

    # require_rows: a real grid cell is never empty for a Teff inside the
    # table's range, so an empty answer means the mirror -- try another.
    tabs, used = cds_health.query_vizier(_run, context="exoplanet/claret-ld",
                                         attempts=3, timeout=20.0, require_rows=True,
                                         status_cb=status_cb)
    if not tabs or not len(tabs[0]):
        raise LookupError(f"no Claret 2011 rows from any VizieR mirror (last: {used})")
    tb = tabs[0]
    filt = np.array([str(x).strip() for x in tb["Filt"]])
    sel = (filt == band)
    if "Met" in tb.colnames and "Mod" in tb.colnames:
        sel &= (np.array([str(x).strip() for x in tb["Met"]]) == "L")
        sel &= (np.array([str(x).strip() for x in tb["Mod"]]) == "A")
    if not np.any(sel):
        raise LookupError(f"passband {band!r} not in the table")
    T = np.asarray(tb["Teff"], float)[sel]
    G = np.asarray(tb["logg"], float)[sel]
    Z = np.asarray(tb["Z"], float)[sel]
    A = np.asarray(tb["a"], float)[sel]
    B = np.asarray(tb["b"], float)[sel]
    z0 = Z[np.argmin(np.abs(Z - feh))]
    g0 = G[np.argmin(np.abs(G - logg) + 10.0 * np.abs(Z - z0))]
    m = (Z == z0) & (G == g0)
    T, A, B = T[m], A[m], B[m]
    o = np.argsort(T)
    T, A, B = T[o], A[o], B[o]
    if T.size == 1:
        return (float(A[0]), float(B[0]),
                f"Claret 2011 {band} (Teff {T[0]:.0f}, logg {g0:g}, [M/H] {z0:g}) via {used}")
    u1 = float(np.interp(teff, T, A))
    u2 = float(np.interp(teff, T, B))
    return u1, u2, f"Claret 2011 {band} (Teff {teff:.0f}, logg {g0:g}, [M/H] {z0:g}) via {used}"


# ═══════════════════════════ AAVSO exoplanet report ═══════════════════════

def round_to_unc(x: float, unc: Optional[float] = None) -> float:
    """Round x to the second significant figure of its uncertainty."""
    y = x if unc is None else unc
    try:
        if not (np.isfinite(x) and np.isfinite(y)):
            return x
        if math.floor(abs(y)) >= 1.0:
            nd = 2
        elif y == 0:
            nd = 6
        else:
            nd = -int(math.floor(math.log10(abs(y)))) + 1
        return round(float(x), nd)
    except Exception:
        return x


def aavso_exoplanet_report(res: FitResult, *, obscode: str, secondary: str = "",
                           star_name: str, planet_name: str, filter_code: str,
                           exposure_s: float, binning: str = "1x1",
                           camera: str = "CCD", comp_star: Optional[dict] = None,
                           notes: str = "", software: str = "Seti Astro Suite Pro",
                           date_type: str = "BJD_TDB") -> str:
    """
    AAVSO Exoplanet Database submission text, in the layout EXOTIC writes
    (field order, JSON '-XC' blocks, column set).  DIFF is the
    out-of-transit-normalised flux, NOT detrended -- the database stores the
    raw differential curve and the detrend model separately (DETREND_2).
    """
    import json

    pr = res.priors
    d = res.derived
    fc = FILTERS.get(filter_code, (filter_code, None, None, None))
    filt_xc = {"name": filter_code, "desc": fc[0],
               "fwhm": [{"value": f"{fc[1]}" if fc[1] else "", "units": "nm"},
                        {"value": f"{fc[2]}" if fc[2] else "", "units": "nm"}]}

    def pm(name):
        if name in res.median:
            m, lo, hi = res.median[name], res.err_lo[name], res.err_hi[name]
            return m, 0.5 * (lo + hi)
        return res.fixed.get(name, float("nan")), 0.0

    tc, tc_e = pm("tc")
    rp, rp_e = pm("rp")
    inc, inc_e = pm("inc") if "inc" in res.median else (pr.inc, 0.0)
    a1m, a1l, a1h = d.get("a1", (float("nan"), 0, 0))
    a2m, a2l, a2h = d.get("a2", (float("nan"), 0, 0))
    a1e, a2e = 0.5 * (a1l + a1h), 0.5 * (a2l + a2h)
    dur, dl, dh = d.get("t14_h", (float("nan"), 0, 0))

    priors_line = (
        f"Period={round_to_unc(pr.per, pr.per_err or None)} +/- {round_to_unc(pr.per_err)},"
        f"Rp/R*={round_to_unc(pr.rp, pr.rp_err or None)} +/- {round_to_unc(pr.rp_err)},"
        f"a/R*={round_to_unc(pr.ars, pr.ars_err or None)} +/- {round_to_unc(pr.ars_err)},"
        f"inc={round_to_unc(pr.inc, pr.inc_err or None)} +/- {round_to_unc(pr.inc_err)},"
        f"ecc={round_to_unc(pr.ecc)},"
        f"u1={round_to_unc(pr.u1, 0.01)} +/- 0,u2={round_to_unc(pr.u2, 0.01)} +/- 0")
    priors_xc = {
        "Period": {"value": str(pr.per), "uncertainty": str(pr.per_err), "units": "days"},
        "Rp/R*": {"value": str(pr.rp), "uncertainty": str(pr.rp_err)},
        "a/R*": {"value": str(pr.ars), "uncertainty": str(pr.ars_err)},
        "inc": {"value": str(pr.inc), "uncertainty": str(pr.inc_err), "units": "degrees"},
        "ecc": {"value": str(pr.ecc), "uncertainty": None},
        "u1": {"value": str(pr.u1), "uncertainty": None},
        "u2": {"value": str(pr.u2), "uncertainty": None},
        "limb_darkening_law": {"value": "quadratic", "uncertainty": None},
    }
    results_line = (
        f"Tc={round_to_unc(tc, tc_e)} +/- {round_to_unc(tc_e)},"
        f"Rp/R*={round_to_unc(rp, rp_e)} +/- {round_to_unc(rp_e)},"
        f"inc={round_to_unc(inc, inc_e or None)} +/- {round_to_unc(inc_e)},"
        f"Am1={round_to_unc(a1m, a1e)} +/- {round_to_unc(a1e)},"
        f"Am2={round_to_unc(a2m, a2e)} +/- {round_to_unc(a2e)}")
    results_xc = {
        "Tc": {"value": str(tc), "uncertainty": str(tc_e), "units": date_type},
        "Rp/R*": {"value": str(rp), "uncertainty": str(rp_e)},
        "inc": {"value": str(inc), "uncertainty": str(inc_e), "units": "degrees"},
        "Am1": {"value": str(a1m), "uncertainty": str(a1e)},
        "Am2": {"value": str(a2m), "uncertainty": str(a2e)},
        "Duration": {"value": str(dur / 24.0), "uncertainty": str(0.5 * (dl + dh) / 24.0),
                     "units": "days"},
    }

    # DIFF: divide out the normalisation only (median of the out-of-transit
    # points), keep the airmass signature in the data.
    oot = res.transit >= 1.0 - 1e-9
    use = res.used & oot if np.count_nonzero(res.used & oot) >= 5 else res.used
    norm = float(np.median(res.y[use])) if np.any(use) else 1.0
    diff = res.y / norm
    derr = res.yerr / norm
    if "am" in res.best:
        det2 = res.systematics / norm
    else:
        det2 = res.systematics / norm

    lines = [
        "#TYPE=EXOPLANET",
        f"#OBSCODE={obscode}",
        f"#SECONDARY_OBSCODES={secondary}",
        f"#SOFTWARE={software}",
        "#DELIM=,",
        f"#DATE_TYPE={date_type}",
        f"#OBSTYPE={camera}",
        f"#STAR_NAME={star_name}",
        f"#EXOPLANET_NAME={planet_name}",
        f"#BINNING={binning}",
        f"#EXPOSURE_TIME={exposure_s:g}" if exposure_s and exposure_s > 0 else "#EXPOSURE_TIME=-1",
        f"#COMP_STAR-XC={json.dumps(comp_star) if comp_star else 'null'}",
        f"#NOTES={notes}",
        "#DETREND_PARAMETERS=AIRMASS, AIRMASS CORRECTION FUNCTION",
        "#MEASUREMENT_TYPE=Rnflux",
        f"#FILTER={filter_code}",
        f"#FILTER-XC={json.dumps(filt_xc)}",
        f"#PRIORS={priors_line}",
        f"#PRIORS-XC={json.dumps(priors_xc)}",
        f"#RESULTS={results_line}",
        f"#RESULTS-XC={json.dumps(results_xc)}",
        "# Use of this data is governed by the AAVSO Data Usage Guidelines: "
        "aavso.org/data-usage-guidelines",
        "#DATE,DIFF,ERR,DETREND_1,DETREND_2",
    ]
    for i in range(res.t.size):
        if not res.used[i]:
            continue
        lines.append(f"{round(float(res.t[i]), 8)},{round(float(diff[i]), 7)},"
                     f"{round(float(derr[i]), 7)},{round(float(res.airmass[i]), 7)},"
                     f"{round(float(det2[i]), 7)}")
    return "\n".join(lines) + "\n"
