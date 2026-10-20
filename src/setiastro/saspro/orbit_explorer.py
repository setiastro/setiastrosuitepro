#src/setiastro/saspro/orbit_explorer.py
#!/usr/bin/env python3
# ======================================================
#   _____      __  _ ___         __
#  / ___/___  / /_(_)   |  _____/ /__________
#  \__ \/ _ \/ __/ / /| | / ___/ __/ ___/ __ \
# ___/ /  __/ /_/ / ___ |(__  ) /_/ /  / /_/ /
#/____/\___/\__/_/_/  |_/____/\__/_/   \____/
#  SASpro Orbit Explorer
#
#  orbit_explorer.py  —  interactive 3D heliocentric orbit viewer
#
#  Draws the Sun, the inner planets (+ outer planets as needed), and a
#  selected minor body with its full orbit ellipse, all placed at one
#  exact instant (the frame's observation JD). Same pyqtgraph.opengl
#  stack the Gaia Stellar Neighborhood Explorer uses, so it inherits the
#  same lazy-GL-build discipline and degrades to a message when
#  pyqtgraph.opengl / PyOpenGL aren't present.
#
#  Frame: heliocentric ECLIPTIC J2000, units of AU. Minor-body osculating
#  elements and the Standish planetary elements are both referenced to
#  this frame, so nothing needs an equatorial→ecliptic rotation — every
#  body lives in the same coordinate system and the scene is just points
#  and lines in AU.
#
#  Written by Franklin Marek
#  www.setiastro.com
# ======================================================

from __future__ import annotations

import math
from typing import Optional

import numpy as np

# Gaussian gravitational constant (rad/day). n = GAUSS_K / a**1.5 is the
# heliocentric mean motion for a in AU — i.e. two-body, Sun-dominated.
# Good to well under a pixel over the span of one night's subs, which is
# all this viewer propagates across.
GAUSS_K = 0.01720209895
# Heliocentric gravitational parameter, AU^3/day^2 (μ = k²). Matches the
# value minorbodycatalog.py propagates comets with, so conic positions
# computed here agree with the hunter's own cross-match.
MU_SUN = GAUSS_K * GAUSS_K
_TWO_PI = 2.0 * math.pi
J2000_JD = 2451545.0
_PARABOLIC_TOL = 1e-8


# ─────────────────────────────────────────────────────────────────────
# Kepler two-body geometry (pure numpy — importable with no Qt present)
# ─────────────────────────────────────────────────────────────────────

def _solve_kepler(M: float, e: float, tol: float = 1e-12,
                  itmax: int = 80) -> float:
    """
    Solve Kepler's equation  M = E − e·sin E  for the eccentric anomaly E
    (elliptical orbits, e < 1). Newton–Raphson from a good initial guess;
    for the eccentricities of real asteroids and short-period comets this
    converges in a handful of iterations.
    """
    M = (M + math.pi) % _TWO_PI - math.pi          # wrap to [−π, π]
    e = float(e)
    # Danby's starter — robust across the whole e < 1 range.
    E = M + 0.85 * e * math.copysign(1.0, math.sin(M)) if e > 0.8 else M
    for _ in range(itmax):
        f = E - e * math.sin(E) - M
        fp = 1.0 - e * math.cos(E)
        dE = f / fp
        E -= dE
        if abs(dE) < tol:
            break
    return E


def _rotation_matrix(i_rad: float, node_rad: float, argp_rad: float) -> np.ndarray:
    """
    Perifocal → heliocentric-ecliptic rotation  R = Rz(Ω)·Rx(i)·Rz(ω).

    Columns are the perifocal basis vectors expressed in the ecliptic
    frame, so R @ (x_p, y_p, 0) places a point of the orbit directly in
    ecliptic AU.
    """
    cO, sO = math.cos(node_rad), math.sin(node_rad)
    ci, si = math.cos(i_rad), math.sin(i_rad)
    cw, sw = math.cos(argp_rad), math.sin(argp_rad)
    Rz_O = np.array([[cO, -sO, 0.0], [sO, cO, 0.0], [0.0, 0.0, 1.0]])
    Rx_i = np.array([[1.0, 0.0, 0.0], [0.0, ci, -si], [0.0, si, ci]])
    Rz_w = np.array([[cw, -sw, 0.0], [sw, cw, 0.0], [0.0, 0.0, 1.0]])
    return Rz_O @ Rx_i @ Rz_w


def orbit_ellipse(a: float, e: float, i_deg: float, node_deg: float,
                  argp_deg: float, n: int = 512) -> np.ndarray:
    """
    The full orbit as an (n, 3) array of heliocentric-ecliptic points, in
    AU. Parameterised uniformly in eccentric anomaly E (not true anomaly)
    so the ellipse is traced exactly, and the perifocal form
    x = a(cos E − e), y = b·sin E avoids ever forming the true anomaly.
    The last point repeats the first so a line-strip closes the loop.
    """
    i_rad = math.radians(i_deg)
    node_rad = math.radians(node_deg)
    argp_rad = math.radians(argp_deg)
    R = _rotation_matrix(i_rad, node_rad, argp_rad)
    b = a * math.sqrt(max(0.0, 1.0 - e * e))
    E = np.linspace(0.0, _TWO_PI, int(max(16, n)))
    peri = np.empty((E.size, 3), dtype=np.float64)
    peri[:, 0] = a * (np.cos(E) - e)
    peri[:, 1] = b * np.sin(E)
    peri[:, 2] = 0.0
    return (peri @ R.T).astype(np.float64)


def position_at(a: float, e: float, i_deg: float, node_deg: float,
                argp_deg: float, M0_deg: float, epoch_jd: float,
                jd: float, n_deg: float | None = None) -> tuple[np.ndarray, float]:
    """
    Heliocentric-ecliptic position (AU) and heliocentric distance (AU) of
    an ELLIPTICAL body at `jd`, from its mean-anomaly elements and the
    epoch M0 is stated at. Two-body: advance the mean anomaly by n·Δt,
    solve Kepler, place the point.

    `n_deg` is the mean daily motion in deg/day. When the catalog supplies
    it (MPCORB publishes it per object), it is used directly — MPC's n
    differs from k/a^1.5 in the fourth decimal, which over a few-hundred-
    day epoch→observation span is ~0.1° of mean anomaly, enough to visibly
    offset the dot. Falls back to k/a^1.5 when absent. epoch_jd=None takes
    the elements as already current at `jd`.
    """
    i_rad = math.radians(i_deg)
    node_rad = math.radians(node_deg)
    argp_rad = math.radians(argp_deg)
    R = _rotation_matrix(i_rad, node_rad, argp_rad)

    dt = 0.0 if epoch_jd is None else (float(jd) - float(epoch_jd))
    if n_deg is not None and math.isfinite(n_deg) and n_deg != 0.0:
        M = math.radians(M0_deg + n_deg * dt)
    else:
        M = math.radians(M0_deg) + (GAUSS_K / (a ** 1.5)) * dt
    E = _solve_kepler(M, e)
    b = a * math.sqrt(max(0.0, 1.0 - e * e))
    peri = np.array([a * (math.cos(E) - e), b * math.sin(E), 0.0])
    pos = R @ peri
    r = a * (1.0 - e * math.cos(E))
    return pos.astype(np.float64), float(r)


# ── Conic propagation (elliptical / parabolic / hyperbolic) ──────────
# Mirrors minorbodycatalog._conic_r_nu so comet positions agree with the
# hunter's own cross-match. Used for perihelion-form elements (q, Tp),
# the only set that can express e ≥ 1.

def _solve_kepler_hyperbolic(M: float, e: float,
                             tol: float = 1e-11, itmax: int = 100) -> float:
    if M == 0.0:
        return 0.0
    sign = 1.0 if M > 0 else -1.0
    Ma = abs(M)
    H = math.asinh(Ma / e) if Ma > 6.0 else Ma / max(e - 1.0, 1e-6)
    if not math.isfinite(H):
        H = math.asinh(Ma / e)
    for _ in range(itmax):
        fp = e * math.cosh(H) - 1.0
        if fp == 0.0 or not math.isfinite(fp):
            break
        dH = (e * math.sinh(H) - H - Ma) / fp
        H -= dH
        if abs(dH) < tol:
            break
    return sign * H


def _conic_r_nu(dt_days: float, q: float, e: float) -> tuple[float, float]:
    """Heliocentric distance and true anomaly `dt_days` after perihelion."""
    mu = MU_SUN
    if abs(e - 1.0) <= _PARABOLIC_TOL:              # parabolic (Barker)
        M_bar = math.sqrt(mu / (2.0 * q ** 3)) * dt_days
        A = 1.5 * M_bar
        B = np.cbrt(A + math.sqrt(A * A + 1.0))
        D = B - 1.0 / B
        return float(q * (1.0 + D * D)), float(2.0 * math.atan(D))
    if e > 1.0:                                     # hyperbolic
        a_abs = q / (e - 1.0)
        M = math.sqrt(mu / a_abs ** 3) * dt_days
        H = _solve_kepler_hyperbolic(float(M), float(e))
        r = a_abs * (e * math.cosh(H) - 1.0)
        nu = 2.0 * math.atan2(math.sqrt(e + 1.0) * math.tanh(H / 2.0),
                              math.sqrt(e - 1.0))
        return float(r), float(nu)
    a = q / (1.0 - e)                               # elliptical
    M = (math.sqrt(mu / a ** 3) * dt_days) % _TWO_PI
    E = _solve_kepler(M, float(e))
    r = a * (1.0 - e * math.cos(E))
    nu = math.atan2(math.sqrt(1.0 - e * e) * math.sin(E), math.cos(E) - e)
    return float(r), float(nu)


# ─────────────────────────────────────────────────────────────────────
# Planets — Standish (JPL) mean Keplerian elements + centennial rates,
# J2000. Enough for context rings and a current-position dot at the few
# arcminute level, with no ephemeris file to ship or download.
#
# Row: a(AU), e, i(deg), L(deg), long.peri ϖ(deg), node Ω(deg)
# then the same six as per-century rates.
# ─────────────────────────────────────────────────────────────────────

_PLANETS = {
    "Mercury": (dict(a=0.38709927, e=0.20563593, i=7.00497902, L=252.25032350,
                     peri=77.45779628, node=48.33076593),
                dict(a=0.00000037, e=0.00001906, i=-0.00594749, L=149472.67411175,
                     peri=0.16047689, node=-0.12534081),
                (0.62, 0.60, 0.58, 1.0)),
    "Venus":   (dict(a=0.72333566, e=0.00677672, i=3.39467605, L=181.97909950,
                     peri=131.60246718, node=76.67984255),
                dict(a=0.00000390, e=-0.00004107, i=-0.00078890, L=58517.81538729,
                     peri=0.00268329, node=-0.27769418),
                (0.85, 0.72, 0.42, 1.0)),
    "Earth":   (dict(a=1.00000261, e=0.01671123, i=-0.00001531, L=100.46457166,
                     peri=102.93768193, node=0.0),
                dict(a=0.00000562, e=-0.00004392, i=-0.01294668, L=35999.37244981,
                     peri=0.32327364, node=0.0),
                (0.35, 0.60, 1.00, 1.0)),
    "Mars":    (dict(a=1.52371034, e=0.09339410, i=1.84969142, L=-4.55343205,
                     peri=-23.94362959, node=49.55953891),
                dict(a=0.00001847, e=0.00007882, i=-0.00813131, L=19140.30268499,
                     peri=0.44441088, node=-0.29257343),
                (0.90, 0.45, 0.32, 1.0)),
    "Jupiter": (dict(a=5.20288700, e=0.04838624, i=1.30439695, L=34.39644051,
                     peri=14.72847983, node=100.47390909),
                dict(a=-0.00011607, e=-0.00013253, i=-0.00183714, L=3034.74612775,
                     peri=0.21252668, node=0.20469106),
                (0.82, 0.70, 0.52, 1.0)),
    "Saturn":  (dict(a=9.53667594, e=0.05386179, i=2.48599187, L=49.95424423,
                     peri=92.59887831, node=113.66242448),
                dict(a=-0.00125060, e=-0.00050991, i=0.00193609, L=1222.49362201,
                     peri=-0.41897216, node=-0.28867794),
                (0.86, 0.80, 0.58, 1.0)),
    "Uranus":  (dict(a=19.18916464, e=0.04725744, i=0.77263783, L=313.23810451,
                     peri=170.95427630, node=74.01692503),
                dict(a=-0.00196176, e=-0.00004397, i=-0.00242939, L=428.48202785,
                     peri=0.40805281, node=0.04240589),
                (0.55, 0.80, 0.85, 1.0)),
    "Neptune": (dict(a=30.06992276, e=0.00859048, i=1.77004347, L=-55.12002969,
                     peri=44.96476227, node=131.78422574),
                dict(a=0.00026291, e=0.00005105, i=0.00035372, L=218.45945325,
                     peri=-0.32241464, node=-0.00508664),
                (0.35, 0.50, 0.95, 1.0)),
}

_PLANET_ORDER = ["Mercury", "Venus", "Earth", "Mars",
                 "Jupiter", "Saturn", "Uranus", "Neptune"]


def _planet_elements_at(name: str, jd: float) -> dict:
    """Standish element set advanced to `jd`, returned as a, e, i, node, argp, M (deg)."""
    base, rate, _col = _PLANETS[name]
    T = (float(jd) - J2000_JD) / 36525.0
    a = base["a"] + rate["a"] * T
    e = base["e"] + rate["e"] * T
    i = base["i"] + rate["i"] * T
    L = base["L"] + rate["L"] * T
    peri = base["peri"] + rate["peri"] * T          # longitude of perihelion ϖ
    node = base["node"] + rate["node"] * T
    argp = peri - node                              # ω = ϖ − Ω
    M = (L - peri + 180.0) % 360.0 - 180.0          # M = L − ϖ, wrapped
    return dict(a=a, e=e, i=i, node=node, argp=argp, M=M)


def planet_scene(jd: float, max_r: float) -> list:
    """
    Build the planets worth showing for an orbit whose aphelion is
    `max_r` AU. Always the terrestrials through Jupiter (they frame the
    main belt); outer planets join only when the body reaches their
    neighbourhood, so a near-Earth asteroid isn't drawn as a speck next
    to Neptune's ring.

    Returns a list of dicts: name, color, orbit (N,3), pos (3,), r_helio.
    """
    want = ["Mercury", "Venus", "Earth", "Mars", "Jupiter"]
    for name, cut in (("Saturn", 6.5), ("Uranus", 11.0), ("Neptune", 20.0)):
        if max_r > cut:
            want.append(name)
    out = []
    for name in want:
        el = _planet_elements_at(name, jd)
        orbit = orbit_ellipse(el["a"], el["e"], el["i"], el["node"], el["argp"])
        # _planet_elements_at already advanced the mean anomaly to `jd`
        # (via L's fast rate term), so the position is evaluated AT that M
        # with no further propagation — epoch_jd=None makes Δt = 0. Passing
        # a real epoch here would double-count the planet's motion.
        pos, r = position_at(el["a"], el["e"], el["i"], el["node"],
                             el["argp"], el["M"], None, jd)
        out.append(dict(name=name, color=_PLANETS[name][2],
                        orbit=orbit, pos=pos, r_helio=r))
    return out


# ─────────────────────────────────────────────────────────────────────
# Element extraction from a catalog row
#
# ADJUST-FOR-SCHEMA: these alias lists are the one place that has to
# match minorbodycatalog.py's actual column names. They already cover
# the common MPC / JPL / astropy spellings; if a column in the catalog
# goes by another name, add it here and everything downstream works. On
# a miss the function returns None and stashes the row's available keys
# on `._last_available_keys` so the caller can log exactly what WAS
# there — no guessing from a silent failure.
# ─────────────────────────────────────────────────────────────────────

# Alias lists. The FIRST entries in each are minorbodycatalog.py's actual
# column names; the rest are common MPC/JPL/astropy spellings kept so the
# extractor still works against another schema or a hand-built row. If the
# catalog ever renames a column, add the new name at the front here.
_ELEMENT_ALIASES = {
    "e":     ("eccentricity", "e", "ecc"),
    "i":     ("inclination_degrees", "i", "incl", "inclination", "i_deg"),
    "node":  ("longitude_of_ascending_node_degrees", "node", "Omega", "om",
              "OM", "ascending_node", "raan", "lan", "node_deg"),
    "argp":  ("argument_of_perihelion_degrees", "argp", "w", "W", "arg_peri",
              "argument_of_perihelion", "peri", "arg_perihelion", "w_deg"),
    "a":     ("semimajor_axis_au", "a", "a_au", "semimajor_axis",
              "semi_major_axis", "sma"),
    "M":     ("mean_anomaly_degrees", "M", "ma", "mean_anomaly", "m0", "M0"),
    "n":     ("mean_daily_motion_degrees", "n", "mean_daily_motion", "n_deg"),
    "epoch": ("epoch_jd", "epoch", "epoch_tdb", "epoch_mjd"),
    "q":     ("perihelion_distance_au", "q", "perihelion_distance", "q_au"),
    "tp":    ("perihelion_jd", "tp", "Tp", "t_peri", "time_of_perihelion"),
}

_last_available_keys: list = []


def _row_get(row, names):
    """First present, non-null, non-NaN alias from a dict / Series / mapping."""
    for k in names:
        try:
            if k not in row:
                continue
            v = row[k]
        except Exception:
            continue
        if v is None:
            continue
        try:
            if isinstance(v, float) and math.isnan(v):
                continue
        except Exception:
            pass
        return v
    return None


def _num(x):
    try:
        f = float(x)
        return f if math.isfinite(f) else None
    except Exception:
        return None


def extract_elements_from_row(row) -> Optional[dict]:
    """
    Normalise one catalog row into the element set the viewer uses:

        common:  e, i, node, argp              (degrees)
        form "mean":  a, M, n_deg (opt), epoch (AU / deg / deg·day⁻¹ / JD)
        form "peri":  q, tp                    (AU / JD)   — handles e ≥ 1

    Returns None (and stashes the row's columns on ``_last_available_keys``)
    only when even the orientation set is missing. Mirrors how
    minorbodycatalog.compute_positions_astropy picks a propagation form:
    mean-anomaly when a+M are present and e<1, otherwise perihelion.
    """
    global _last_available_keys
    if row is None:
        _last_available_keys = []
        return None
    try:
        _last_available_keys = list(row.keys())
    except Exception:
        _last_available_keys = []

    e = _num(_row_get(row, _ELEMENT_ALIASES["e"]))
    i = _num(_row_get(row, _ELEMENT_ALIASES["i"]))
    node = _num(_row_get(row, _ELEMENT_ALIASES["node"]))
    argp = _num(_row_get(row, _ELEMENT_ALIASES["argp"]))
    if None in (e, i, node, argp):
        return None

    a = _num(_row_get(row, _ELEMENT_ALIASES["a"]))
    M = _num(_row_get(row, _ELEMENT_ALIASES["M"]))
    n_deg = _num(_row_get(row, _ELEMENT_ALIASES["n"]))
    epoch = _num(_row_get(row, _ELEMENT_ALIASES["epoch"]))
    if epoch is not None and epoch < 100000.0:      # MJD given as epoch
        epoch += 2400000.5
    q = _num(_row_get(row, _ELEMENT_ALIASES["q"]))
    tp = _num(_row_get(row, _ELEMENT_ALIASES["tp"]))
    if tp is not None and tp < 100000.0:
        tp += 2400000.5

    out = dict(e=float(e), i=float(i), node=float(node), argp=float(argp))

    # Mean-anomaly form, exactly as the catalog prefers it: a + M, e < 1.
    if a is not None and a > 0 and M is not None and e < 1.0:
        out.update(form="mean", a=float(a), M=float(M),
                   n_deg=(None if n_deg is None else float(n_deg)),
                   epoch=(None if epoch is None else float(epoch)))
        if q is not None:
            out["q"] = float(q)
        return out

    # Perihelion form — the only one that expresses e ≥ 1, and the comet
    # path. Needs q; Tp gives the phase (without it the body still draws,
    # positioned at perihelion).
    if q is not None and q > 0:
        out.update(form="peri", q=float(q),
                   tp=(None if tp is None else float(tp)))
        return out

    # Elliptical but only q/a known without M (rare) — still drawable.
    if a is not None and a > 0 and e < 1.0:
        out.update(form="mean", a=float(a), M=0.0, n_deg=None,
                   epoch=None)
        return out
    return None


def body_summary(el: dict) -> dict:
    """q, Q (bound orbits only), period (yr) and a, for the info panel."""
    e = el["e"]
    a = el.get("a")
    if a is None and el.get("q") is not None and e < 1.0:
        a = el["q"] / (1.0 - e)
    q = el.get("q")
    if q is None and a is not None:
        q = a * (1.0 - e)
    Q = (a * (1.0 + e)) if (a is not None and e < 1.0) else None
    period = (a ** 1.5) if (a is not None and a > 0 and e < 1.0) else None
    return dict(a=a, e=e, q=q, Q=Q, period_yr=period)


def body_orbit_points(el: dict, n: int = 512,
                      r_max: float | None = None) -> np.ndarray:
    """
    The body's orbit as (N,3) heliocentric-ecliptic points. A closed
    ellipse for e<1; for e≥1 an open arc swept in true anomaly out to the
    asymptote (hyperbola) or a wide cap (parabola), optionally clipped at
    r_max so an interstellar track doesn't shoot off to infinity.
    """
    e = el["e"]
    if e < 1.0:
        a = el.get("a")
        if a is None:
            a = el["q"] / (1.0 - e)
        return orbit_ellipse(a, e, el["i"], el["node"], el["argp"], n)

    # Open conic: r = p / (1 + e·cos ν), p = q(1+e).
    q = el.get("q")
    if q is None:                                   # nothing to anchor on
        return np.zeros((0, 3), dtype=np.float64)
    p = q * (1.0 + e)
    if abs(e - 1.0) <= _PARABOLIC_TOL:
        nu_lim = math.radians(170.0)
    else:
        nu_lim = math.acos(-1.0 / e) - 1e-3         # just inside the asymptote
    nu = np.linspace(-nu_lim, nu_lim, int(max(16, n)))
    r = p / (1.0 + e * np.cos(nu))
    if r_max is not None:
        r = np.minimum(r, float(r_max))
    R = _rotation_matrix(math.radians(el["i"]), math.radians(el["node"]),
                         math.radians(el["argp"]))
    peri = np.empty((nu.size, 3), dtype=np.float64)
    peri[:, 0] = r * np.cos(nu)
    peri[:, 1] = r * np.sin(nu)
    peri[:, 2] = 0.0
    return (peri @ R.T).astype(np.float64)


def body_position(el: dict, jd: float) -> tuple[np.ndarray, float]:
    """Heliocentric-ecliptic position (AU) and distance (AU) at `jd`."""
    if el.get("form") == "peri":
        q, e = el["q"], el["e"]
        tp = el.get("tp")
        dt = 0.0 if tp is None else (float(jd) - float(tp))
        r, nu = _conic_r_nu(dt, q, e)
        R = _rotation_matrix(math.radians(el["i"]), math.radians(el["node"]),
                             math.radians(el["argp"]))
        pos = R @ np.array([r * math.cos(nu), r * math.sin(nu), 0.0])
        return pos.astype(np.float64), float(r)
    return position_at(el["a"], el["e"], el["i"], el["node"], el["argp"],
                       el["M"], el.get("epoch"), jd, n_deg=el.get("n_deg"))


# ─────────────────────────────────────────────────────────────────────
# Qt / OpenGL dialog — guarded exactly like the Gaia explorer so the
# math above imports fine on a headless box or in tests.
# ─────────────────────────────────────────────────────────────────────

try:
    from PyQt6.QtCore import Qt, QTimer
    from PyQt6.QtGui import QIcon
    from PyQt6.QtWidgets import (
        QApplication, QCheckBox, QDialog, QHBoxLayout, QLabel, QPushButton,
        QSplitter, QVBoxLayout, QWidget, QComboBox, QFrame,
    )
    _HAS_QT = True
except Exception as _qt_err:                        # pragma: no cover
    _HAS_QT = False
    _QT_IMPORT_ERROR = str(_qt_err)

try:
    import pyqtgraph.opengl as gl
    from pyqtgraph import Vector
    HAS_GL = True
    _GL_IMPORT_ERROR = ""
except Exception as _gl_err:                        # ImportError / OpenGL platform
    HAS_GL = False
    _GL_IMPORT_ERROR = str(_gl_err)


if _HAS_QT:

    class OrbitExplorerDialog(QDialog):
        """
        Interactive heliocentric orbit view for one minor body at one
        instant. Mirrors the Gaia Stellar Neighborhood Explorer: the GL
        widget is built lazily on first show (constructing a QOpenGLWidget
        can make Qt recreate the window's native handle — better that
        happen to this dialog than to the SASpro main window), and the
        tab degrades to a message when pyqtgraph.opengl / PyOpenGL are
        missing.

        `body` is one entry from the hunter's catalogued-bodies list:
        it needs `designation`, `elements` (a, e, i, node, argp, M,
        epoch) and optionally `kind`, `H`. `jd` is the frame's
        observation Julian Date — the exact moment everything is placed
        at.
        """

        AU_LABEL = "AU"

        def __init__(self, body: dict, jd: float, settings=None,
                     icon: "QIcon | None" = None, parent=None):
            super().__init__(parent)
            self._body = dict(body or {})
            self._jd = float(jd)
            self._settings = settings
            self._el = self._body.get("elements")
            self._name = str(self._body.get("designation", "minor body"))
            self.setWindowTitle(f"Orbit Explorer — {self._name}")
            if icon is not None:
                self.setWindowIcon(icon)
            self.setModal(False)

            # Scene state, filled by _populate() once GL is up.
            self._view = None
            self._grid = None
            self._sun = None
            self._planet_orbits = []            # list[GLLinePlotItem]
            self._planet_markers = None         # GLScatterPlotItem
            self._planet_labels = []            # list[GLTextItem]
            self._ast_orbit = None
            self._ast_marker = None
            self._ast_drop = None               # vertical line to ecliptic plane
            self._los_line = None               # Earth → asteroid line of sight
            self._body_labels = []              # Sun / Earth / asteroid GLTextItem
            self._gl_ready = False
            self._view_slot = None
            self._scene = None                  # cached computed arrays
            self._max_r = 3.0

            self._build_ui()

        # ── scene maths ────────────────────────────────────────────────

        def _compute_scene(self) -> dict:
            el = self._el
            ast_pos, r_helio = body_position(el, self._jd)
            summ = body_summary(el)
            # Framing radius: aphelion for a bound orbit; for an open orbit
            # (e≥1, no Q) fall back to where the body actually is, so the
            # arc is drawn and clipped to something sane.
            if summ["Q"] is not None:
                max_r = max(summ["Q"], 1.6)
            else:
                max_r = max(r_helio * 1.2, 3.0)
            ast_orbit = body_orbit_points(el, r_max=max_r * 1.3)
            planets = planet_scene(self._jd, max_r)
            earth = next((p for p in planets if p["name"] == "Earth"), None)
            earth_pos = earth["pos"] if earth else np.array([1.0, 0.0, 0.0])
            delta = float(np.linalg.norm(ast_pos - earth_pos))   # Earth distance, AU
            scene_r = max(max_r, max((p["r_helio"] for p in planets), default=max_r))
            return dict(ast_orbit=ast_orbit, ast_pos=ast_pos, r_helio=r_helio,
                        planets=planets, earth_pos=earth_pos, earth_delta=delta,
                        summary=summ, scene_r=scene_r)

        # ── UI ─────────────────────────────────────────────────────────

        def _build_ui(self):
            root = QVBoxLayout(self)
            root.setContentsMargins(12, 12, 12, 12)
            root.setSpacing(8)

            if self._el is None:
                root.addWidget(self._message(
                    "<b>No orbital elements for this object.</b><br><br>"
                    "The catalog row for "
                    f"<code>{self._name}</code> didn't carry the Keplerian "
                    "elements (a, e, i, Ω, ω, M), so its orbit can't be "
                    "drawn.<br><br>"
                    "Everything else about the detection still works — this "
                    "only affects the 3D view."))
                self.resize(560, 300)
                return

            if not HAS_GL:
                root.addWidget(self._message(
                    "<b>3D view unavailable.</b><br><br>"
                    "The Orbit Explorer needs <code>pyqtgraph.opengl</code> "
                    "and <code>PyOpenGL</code>.<br><br>"
                    f"<span style='color:#a66;'>Import error: {_GL_IMPORT_ERROR}"
                    "</span><br><br>Install with:<br><code>pip install PyOpenGL"
                    "</code>"))
                self.resize(560, 320)
                return

            self.setStyleSheet(
                "QDialog{background:#0a0a16;color:#ddd;}"
                "QLabel{color:#cdd;}"
                "QCheckBox{color:#cdd;}"
                "QComboBox{background:#12122a;color:#ddd;border:1px solid #334;"
                "border-radius:4px;padding:2px 6px;}"
                "QPushButton{background:#15152e;color:#cdd;border:1px solid #334;"
                "border-radius:5px;padding:5px 12px;}"
                "QPushButton:hover{background:#20204a;}")

            body = QSplitter(Qt.Orientation.Horizontal)
            body.setChildrenCollapsible(False)
            body.setHandleWidth(6)
            body.setStyleSheet("QSplitter::handle{background:#1a1a2e;}"
                               "QSplitter::handle:hover{background:#2a2a4e;}")
            root.addWidget(body, stretch=1)

            # View container — GL built lazily on first show.
            self._view_container = QWidget()
            self._view_container.setMinimumSize(360, 360)
            self._view_slot = QVBoxLayout(self._view_container)
            self._view_slot.setContentsMargins(0, 0, 0, 0)
            ph = QLabel("Initialising 3D view…")
            ph.setAlignment(Qt.AlignmentFlag.AlignCenter)
            ph.setStyleSheet("color:#667;")
            self._view_slot.addWidget(ph)
            body.addWidget(self._view_container)

            # Side panel — info + toggles.
            side = QWidget()
            side.setMaximumWidth(320)
            sv = QVBoxLayout(side)
            sv.setContentsMargins(6, 2, 2, 2)
            sv.setSpacing(10)

            self._info = QLabel()
            self._info.setWordWrap(True)
            self._info.setTextFormat(Qt.TextFormat.RichText)
            self._info.setStyleSheet("font-size:12px;")
            sv.addWidget(self._info)

            line = QFrame()
            line.setFrameShape(QFrame.Shape.HLine)
            line.setStyleSheet("color:#223;")
            sv.addWidget(line)

            self._chk_planet_orbits = QCheckBox("Planet orbits")
            self._chk_planet_orbits.setChecked(True)
            self._chk_planet_orbits.toggled.connect(self._apply_visibility)
            sv.addWidget(self._chk_planet_orbits)

            self._chk_labels = QCheckBox("Labels")
            self._chk_labels.setChecked(True)
            self._chk_labels.toggled.connect(self._apply_visibility)
            sv.addWidget(self._chk_labels)

            self._chk_grid = QCheckBox("Ecliptic grid")
            self._chk_grid.setChecked(True)
            self._chk_grid.toggled.connect(self._apply_visibility)
            sv.addWidget(self._chk_grid)

            self._chk_los = QCheckBox("Earth → asteroid sight line")
            self._chk_los.setChecked(True)
            self._chk_los.toggled.connect(self._apply_visibility)
            sv.addWidget(self._chk_los)

            self._chk_drop = QCheckBox("Height drop to ecliptic")
            self._chk_drop.setChecked(True)
            self._chk_drop.setToolTip(
                "Vertical line from the asteroid down to the ecliptic plane —\n"
                "shows how far its orbit tilts out of the plane (inclination).")
            self._chk_drop.toggled.connect(self._apply_visibility)
            sv.addWidget(self._chk_drop)

            btn_row = QHBoxLayout()
            self._btn_reset = QPushButton("Reset view")
            self._btn_reset.clicked.connect(self._reset_view)
            self._btn_top = QPushButton("Top-down")
            self._btn_top.setToolTip("Look straight down the ecliptic north pole.")
            self._btn_top.clicked.connect(self._top_view)
            btn_row.addWidget(self._btn_reset)
            btn_row.addWidget(self._btn_top)
            sv.addLayout(btn_row)

            sv.addStretch(1)
            hint = QLabel("drag to rotate · wheel to zoom · ⌃-drag to pan")
            hint.setStyleSheet("color:#667;font-size:10px;")
            sv.addWidget(hint)

            body.addWidget(side)
            body.setStretchFactor(0, 1)
            body.setStretchFactor(1, 0)

            try:
                avail = self.screen().availableGeometry()
                self.resize(min(1100, int(avail.width() * 0.8)),
                            min(760, int(avail.height() * 0.8)))
            except Exception:
                self.resize(980, 680)

        def _message(self, html: str) -> QLabel:
            m = QLabel(html)
            m.setWordWrap(True)
            m.setAlignment(Qt.AlignmentFlag.AlignCenter)
            m.setStyleSheet("color:#aab;font-size:12px;")
            return m

        # ── lazy GL build (mirrors the Gaia explorer) ───────────────────

        def showEvent(self, ev):
            super().showEvent(ev)
            if self._el is None or not HAS_GL:
                return
            self._ensure_gl()

        def _ensure_gl(self):
            if self._gl_ready or not HAS_GL or self._view_slot is None:
                return
            self._gl_ready = True                   # set first — see Gaia note

            while self._view_slot.count():
                it = self._view_slot.takeAt(0)
                w = it.widget()
                if w is not None:
                    w.setParent(None)
                    w.deleteLater()

            self._view = gl.GLViewWidget()
            self._view.setMinimumSize(360, 360)
            self._view.setBackgroundColor((6, 6, 18))

            scene = self._compute_scene()
            self._scene = scene
            self._max_r = scene["scene_r"]

            # Ecliptic-plane grid (XY, z=0 is the ecliptic by construction).
            self._grid = gl.GLGridItem()
            step = max(round(self._max_r / 6.0, 1), 0.2)
            self._grid.setSize(x=2.2 * self._max_r, y=2.2 * self._max_r, z=1)
            self._grid.setSpacing(x=step, y=step, z=step)
            self._view.addItem(self._grid)

            # Planet orbit rings — one line item each, faint, planet-tinted.
            self._planet_orbits = []
            for p in scene["planets"]:
                c = p["color"]
                col = (c[0], c[1], c[2], 0.30)
                ln = gl.GLLinePlotItem(pos=p["orbit"].astype(np.float32),
                                       color=col, width=1.4, antialias=True)
                self._view.addItem(ln)
                self._planet_orbits.append(ln)

            # Planet position dots (one scatter, per-point colour/size).
            pp = np.array([p["pos"] for p in scene["planets"]], dtype=np.float32)
            pc = np.array([[c for c in p["color"]] for p in scene["planets"]],
                          dtype=np.float32)
            ps = np.array([12.0 if p["name"] == "Earth" else 8.0
                           for p in scene["planets"]], dtype=np.float32)
            self._planet_markers = gl.GLScatterPlotItem(
                pos=pp, color=pc, size=ps, pxMode=True)
            self._view.addItem(self._planet_markers)

            # Asteroid orbit — bright, the thing you came to see.
            self._ast_orbit = gl.GLLinePlotItem(
                pos=scene["ast_orbit"].astype(np.float32),
                color=(1.0, 0.62, 0.22, 0.95), width=2.2, antialias=True)
            self._view.addItem(self._ast_orbit)

            # Vertical drop from the asteroid to the ecliptic plane.
            ap = scene["ast_pos"]
            drop = np.array([[ap[0], ap[1], ap[2]],
                             [ap[0], ap[1], 0.0]], dtype=np.float32)
            self._ast_drop = gl.GLLinePlotItem(pos=drop, mode="lines",
                                               color=(1.0, 0.62, 0.22, 0.30),
                                               width=1.2, antialias=True)
            self._view.addItem(self._ast_drop)

            # Earth → asteroid sight line (the direction you imaged along).
            los = np.array([scene["earth_pos"], ap], dtype=np.float32)
            self._los_line = gl.GLLinePlotItem(pos=los, mode="lines",
                                               color=(0.55, 0.85, 1.0, 0.45),
                                               width=1.2, antialias=True)
            self._view.addItem(self._los_line)

            # Sun at the origin.
            self._sun = gl.GLScatterPlotItem(
                pos=np.zeros((1, 3), dtype=np.float32),
                color=np.array([[1.0, 0.92, 0.35, 1.0]], dtype=np.float32),
                size=np.array([18.0], dtype=np.float32), pxMode=True)
            self._view.addItem(self._sun)

            # Asteroid current position — glowing cyan.
            self._ast_marker = gl.GLScatterPlotItem(
                pos=ap.reshape(1, 3).astype(np.float32),
                color=np.array([[0.40, 1.0, 1.0, 1.0]], dtype=np.float32),
                size=np.array([15.0], dtype=np.float32), pxMode=True)
            self._view.addItem(self._ast_marker)

            # Text labels: Sun, each planet, the asteroid.
            self._planet_labels = []
            for p in scene["planets"]:
                c = p["color"]
                t = gl.GLTextItem(
                    pos=p["pos"].astype(np.float32), text=p["name"],
                    color=(int(c[0] * 255), int(c[1] * 255), int(c[2] * 255), 255))
                self._view.addItem(t)
                self._planet_labels.append(t)

            self._body_labels = []
            t_sun = gl.GLTextItem(pos=np.array([0, 0, 0], dtype=np.float32),
                                  text="Sun", color=(255, 235, 110, 255))
            self._view.addItem(t_sun)
            self._body_labels.append(t_sun)
            t_ast = gl.GLTextItem(pos=ap.astype(np.float32),
                                  text=self._name, color=(130, 255, 255, 255))
            self._view.addItem(t_ast)
            self._body_labels.append(t_ast)

            self._view_slot.addWidget(self._view)
            self._update_info()
            self._apply_visibility()
            self._reset_view()

        # ── visibility toggles / framing ────────────────────────────────

        def _apply_visibility(self):
            if not self._gl_ready:
                return
            po = self._chk_planet_orbits.isChecked()
            for ln in self._planet_orbits:
                ln.setVisible(po)
            if self._grid is not None:
                self._grid.setVisible(self._chk_grid.isChecked())
            if self._los_line is not None:
                self._los_line.setVisible(self._chk_los.isChecked())
            if self._ast_drop is not None:
                self._ast_drop.setVisible(self._chk_drop.isChecked())
            labels_on = self._chk_labels.isChecked()
            for t in self._planet_labels:
                t.setVisible(labels_on and po)
            for t in self._body_labels:
                t.setVisible(labels_on)

        def _reset_view(self):
            if not self._gl_ready or self._view is None:
                return
            r = float(self._max_r)
            try:
                self._view.opts["center"] = Vector(0, 0, 0)
            except Exception:
                pass
            self._view.setCameraPosition(pos=Vector(0, 0, 0),
                                         distance=max(r * 3.0, 3.0),
                                         azimuth=45.0, elevation=28.0)
            self._view.update()

        def _top_view(self):
            if not self._gl_ready or self._view is None:
                return
            r = float(self._max_r)
            try:
                self._view.opts["center"] = Vector(0, 0, 0)
            except Exception:
                pass
            self._view.setCameraPosition(pos=Vector(0, 0, 0),
                                         distance=max(r * 3.0, 3.0),
                                         azimuth=-90.0, elevation=89.9)
            self._view.update()

        def _update_info(self):
            el = self._el
            s = self._scene["summary"]
            kind = str(self._body.get("kind", "minor body"))
            H = self._body.get("H")
            try:
                hstr = f"{float(H):.1f}" if H is not None and np.isfinite(float(H)) else "—"
            except Exception:
                hstr = "—"
            # JD → calendar UTC for the header, kept dependency-free.
            try:
                from astropy.time import Time
                when = Time(self._jd, format="jd", scale="utc").utc.isot
            except Exception:
                when = f"JD {self._jd:.5f}"
            def _f(v, suffix="", nd=3):
                return f"{v:.{nd}f}{suffix}" if v is not None else "—"

            a_row = (f"<tr><td style='color:#89a;'>a</td>"
                     f"<td>{_f(s['a'], ' AU', 4)}</td></tr>"
                     if s.get("a") is not None else "")
            qQ = (f"{_f(s['q'])} / {_f(s['Q'])} AU"
                  if s.get("Q") is not None else f"{_f(s['q'])} AU (q)")
            period = (f"<tr><td style='color:#89a;'>period</td>"
                      f"<td>{_f(s['period_yr'], ' yr', 2)}</td></tr>"
                      if s.get("period_yr") is not None else
                      "<tr><td style='color:#89a;'>orbit</td>"
                      "<td>open (e ≥ 1)</td></tr>")
            self._info.setText(
                f"<div style='font-size:14px;font-weight:bold;color:#9ef;'>"
                f"{self._name}</div>"
                f"<div style='color:#89a;'>{kind} · H={hstr}</div>"
                f"<div style='color:#89a;font-size:11px;margin-top:2px;'>"
                f"{when}<br>JD {self._jd:.5f}</div>"
                "<br>"
                "<table style='font-size:12px;color:#cdd;' cellspacing='3'>"
                f"{a_row}"
                f"<tr><td style='color:#89a;'>e</td><td>{el['e']:.4f}</td></tr>"
                f"<tr><td style='color:#89a;'>i</td><td>{el['i']:.3f}°</td></tr>"
                f"<tr><td style='color:#89a;'>Ω</td><td>{el['node']:.3f}°</td></tr>"
                f"<tr><td style='color:#89a;'>ω</td><td>{el['argp']:.3f}°</td></tr>"
                f"<tr><td style='color:#89a;'>q / Q</td><td>{qQ}</td></tr>"
                f"{period}"
                "<tr><td colspan='2' style='border-top:1px solid #223;'></td></tr>"
                f"<tr><td style='color:#9ef;'>r (Sun)</td>"
                f"<td style='color:#9ef;'>{self._scene['r_helio']:.3f} AU</td></tr>"
                f"<tr><td style='color:#9ef;'>Δ (Earth)</td>"
                f"<td style='color:#9ef;'>{self._scene['earth_delta']:.3f} AU</td></tr>"
                "</table>")


def open_orbit_explorer(parent, body: dict, jd: float, settings=None,
                        icon=None) -> "OrbitExplorerDialog | None":
    """
    Convenience entry point used by the hunter. Returns the (shown)
    dialog so the caller can keep a reference, or None if Qt is somehow
    absent. The parent is responsible for keeping the reference alive;
    the dialog is non-modal so the results window stays usable.
    """
    if not _HAS_QT:
        return None
    dlg = OrbitExplorerDialog(body, jd, settings=settings, icon=icon, parent=parent)
    dlg.show()
    dlg.raise_()
    return dlg