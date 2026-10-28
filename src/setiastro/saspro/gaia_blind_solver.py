#src/setiastro/saspro/gaia_blind_solver.py
#!/usr/bin/env python3
# ======================================================
#  gaia_blind_solver.py  —  blind plate solving from quad-hash indexes
#
#  No RA/Dec, no field of view, no hints. The star pattern itself says where
#  the camera was pointed.
#
#    detect (brightness-ranked)
#      -> form quads over an |AB| window
#      -> code + mirror code                       (quad_core.py)
#      -> searchsorted into each layer             (gaia_quad_index.py)
#      -> similarity from the matched A/B pair
#      -> Bayesian log-odds verification against the real catalogue
#      -> a seed RA/Dec/scale/roll
#
#  What comes out is a SEED, not a final WCS. plate_solver's existing in-house
#  Gaia path then does the catalogue re-query, the iterative nearest-neighbour
#  refinement, the SIP fit and the quality gate exactly as it does for a seeded
#  solve. Blind is a new front end on machinery that is already trusted.
#
#  Why log-odds and not an RMS gate: a whole-sky search manufactures thousands
#  of six-pair coincidences, and six cherry-picked pairs always satisfy an RMS
#  gate. On synthetic fields the correct candidate scores +380 to +530 nats
#  while the best wrong candidate reaches -55. That gap is the safety margin.
#
#  Written by Franklin Marek
#  www.setiastro.com
# ======================================================

from __future__ import annotations

import math
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from setiastro.saspro.quad_core import (
    make_quads, quad_codes, mirror_codes, log_odds,
)
from setiastro.saspro.gaia_quad_index import BIN_TOL

ARCSEC = 206264.806247
LOG_ODDS_ACCEPT = math.log(1e9)        # astrometry.net's default odds-to-solve

# fetch_box(ra_lo, ra_hi, dec_lo, dec_hi, max_mag) -> (ra[], dec[], mag[])
FetchBox = Callable[[float, float, float, float, float],
                    Tuple[np.ndarray, np.ndarray, np.ndarray]]


# ─────────────────────────────────────────────────────────────────────────────
#  Tangent plane
# ─────────────────────────────────────────────────────────────────────────────

def _project(ra, dec, ra0, dec0):
    """Gnomonic projection to tangent-plane radians about (ra0, dec0)."""
    ra = np.radians(np.asarray(ra, np.float64))
    dec = np.radians(np.asarray(dec, np.float64))
    r0, d0 = math.radians(ra0), math.radians(dec0)
    cd, sd = np.cos(dec), np.sin(dec)
    dra = ra - r0
    cdr = np.cos(dra)
    w = np.sin(d0) * sd + np.cos(d0) * cd * cdr
    # Floor w at a small POSITIVE value, never just guard its magnitude: a
    # point more than 90 deg away has w < 0 and would otherwise fold through
    # the origin and land back inside the frame, superimposing the antipodal
    # sky on the real one. Clamping sends it off to infinity instead, where
    # the in-bounds filter drops it.
    w = np.where(w < 1e-9, 1e-9, w)
    x = cd * np.sin(dra) / w
    y = (np.cos(d0) * sd - np.sin(d0) * cd * cdr) / w
    return x, y


def _deproject(x, y, ra0, dec0):
    """Tangent-plane radians back to RA/Dec degrees."""
    r0, d0 = math.radians(ra0), math.radians(dec0)
    rho = math.hypot(x, y)
    if rho < 1e-14:
        return ra0 % 360.0, dec0
    c = math.atan(rho)
    sc, cc = math.sin(c), math.cos(c)
    dec = math.asin(cc * math.sin(d0) + y * sc * math.cos(d0) / rho)
    ra = r0 + math.atan2(x * sc,
                         rho * math.cos(d0) * cc - y * math.sin(d0) * sc)
    return math.degrees(ra) % 360.0, math.degrees(dec)


def _midpoint(ra_a, dec_a, ra_b, dec_b):
    """Direction midway between two sky positions, as RA/Dec degrees."""
    def xyz(ra, dec):
        ra, dec = np.radians(ra), np.radians(dec)
        cd = np.cos(dec)
        return np.stack([cd * np.cos(ra), cd * np.sin(ra), np.sin(dec)], -1)
    m = xyz(ra_a, dec_a) + xyz(ra_b, dec_b)
    m /= np.linalg.norm(m, axis=-1, keepdims=True)
    return (np.degrees(np.arctan2(m[..., 1], m[..., 0])) % 360.0,
            np.degrees(np.arcsin(np.clip(m[..., 2], -1.0, 1.0))))


# ─────────────────────────────────────────────────────────────────────────────
#  Detection
# ─────────────────────────────────────────────────────────────────────────────

#: Extraction ceiling per sigma level. Only `max_stars` survive this function
#: and solve_blind_core uses 80 of those, so the old 4000 extracted roughly
#: thirty times what any quad would ever see. The headroom that remains is
#: deliberate: the ladder breaks at 2x max_stars, so there has to be room to
#: reach that threshold comfortably before the brightest are taken.
_DETECT_CEILING = 400


def detect_stars_by_brightness(gray: np.ndarray, max_stars: int = 120,
                               det_sigma_levels=(50, 25, 15, 10, 5, 3)) -> np.ndarray:
    """
    Detections ranked by flux, brightest first. Returns (N,3) of x, y, flux.

    This deliberately does NOT grid-balance. _detect_stars_uniform spreads
    detections evenly across the frame, which is right for the seeded matcher
    and wrong here: the index keeps the brightest stars per sky cell, so the
    solver has to rank globally by brightness or the two sets never line up.
    That part is load-bearing and stays.

    The ladder starts at sigma=50, matching the seeded path. Starting at 25
    meant the very first and most expensive extraction was already two steps
    down the ladder, on a frame where sigma=50 would have answered.
    """
    from setiastro.saspro.star_alignment import _detect_stars_uniform

    # The seeded path raises sep's pixel stack before it extracts; this path
    # never did, so whichever ran first decided the buffer for both. On a
    # dense frame the default is small enough to change the character of the
    # extraction entirely -- and under force-blind THIS runs first.
    try:
        import sep
        sep.set_extract_pixstack(20000000)
    except Exception:
        pass

    best = None
    for s in det_sigma_levels:
        try:
            c = _detect_stars_uniform(gray, det_sigma=float(s), minarea=5,
                                      grid=(1, 1),
                                      max_per_cell=_DETECT_CEILING,
                                      max_total=_DETECT_CEILING)
        except Exception:
            continue
        if c is None or len(c) == 0:
            continue
        best = c
        if len(c) >= max_stars * 2:
            break
    if best is None or len(best) == 0:
        return np.zeros((0, 3), np.float64)

    a = np.asarray(best, np.float64)
    if a.shape[1] < 3:                       # no flux column — keep as given
        a = np.column_stack([a[:, 0], a[:, 1],
                             np.arange(len(a), 0, -1, dtype=np.float64)])
    a = a[np.argsort(-a[:, 2])][:max_stars]
    return a[:, :3]


# ─────────────────────────────────────────────────────────────────────────────
#  Core solve
# ─────────────────────────────────────────────────────────────────────────────

def _candidate_tolerance(ab_px, sigma_px, lo=0.0025, hi=BIN_TOL):
    """
    Per-quad code tolerance.

    Code error propagates as roughly sigma/|AB|, and false matches scale as
    tol^4 — so a 400 px baseline is worth about 25x fewer false candidates
    than a 120 px one. The bins were sized once at BIN_TOL; this is the exact
    filter applied afterwards, which is what lets a long baseline collect the
    full benefit of its own precision.
    """
    return np.clip(3.0 * float(sigma_px) / np.maximum(ab_px, 1.0), lo, hi)


def _fmt_ang(arcmin: float) -> str:
    return f"{arcmin:.0f}′" if arcmin < 60.0 else f"{arcmin / 60.0:.1f}°"


def _layer_label(layer) -> str:
    """
    'layer 4 (1.3°-4.3° fields)'. A layer index means nothing to someone
    reading a status bar; the field range it covers means everything.
    """
    q = layer.q_lo_arcmin / 0.70                      # nominal quad size
    return (f"layer {layer.layer_index} "
            f"({_fmt_ang(2.5 * q)}–{_fmt_ang(8.0 * q)} fields)")


def solve_blind_core(det_xy: np.ndarray,
                     img_w: int,
                     img_h: int,
                     layers: List,
                     fetch_box: FetchBox,
                     scale_hint_arcsec: Optional[float] = None,
                     scale_tol: float = 0.30,
                     sigma_px: float = 1.0,
                     max_stars: int = 80,
                     max_b_per_a: int = 4,
                     max_verify: int = 48,
                     eps: float = 0.35,
                     min_match: int = 12,
                     status_cb: Optional[Callable[[str], None]] = None) -> Dict:
    """
    Blind solve. `det_xy` is (N,2) pixel positions, brightest first.
    `layers` are QuadLayer objects, already ordered best-first.

    Returns {'ok', 'ra', 'dec', 'scale_arcsec', 'roll_deg', 'parity',
             'log_odds', 'n_match', 'layer', 'error'}.
    """
    def say(m):
        if status_cb:
            status_cb(m)

    det_xy = np.asarray(det_xy, np.float64)[:, :2]
    n = min(max_stars, len(det_xy))
    if n < 8:
        return dict(ok=False, error=f"only {len(det_xy)} stars detected")

    pts = det_xy[:n]
    diag = float(math.hypot(img_w, img_h))
    fail = "no layer matched"
    best_overall = -np.inf

    say(f"Blind: {n} brightest of {len(det_xy)} detections, searching "
        f"{len(layers)} index layer{'s' if len(layers) != 1 else ''}"
        + (f", pixel scale hinted at {scale_hint_arcsec:.3f}\"/px"
           if scale_hint_arcsec else ", pixel scale unknown"))

    for li, layer in enumerate(layers, 1):
        # |AB| window in pixels. With a usable header pixel scale this pins the
        # quads to the layer's own angular band; blind it has to be swept wide,
        # because a window that is too narrow silently discards every quad the
        # index could have matched.
        if scale_hint_arcsec:
            lo_px = layer.q_lo_arcmin * 60.0 / scale_hint_arcsec / (1.0 + scale_tol)
            hi_px = layer.q_hi_arcmin * 60.0 / scale_hint_arcsec * (1.0 + scale_tol)
            lo_px = max(lo_px, 0.03 * diag)
            hi_px = min(hi_px, 0.70 * diag)
            if hi_px <= lo_px:
                say(f"Blind: {_layer_label(layer)} [{li}/{len(layers)}] — the "
                    f"hinted scale puts this layer's quads outside the frame, "
                    f"skipping")
                continue
        else:
            lo_px, hi_px = 0.04 * diag, 0.60 * diag

        fq = make_quads(pts, np.arange(n), lo_px, hi_px, max_b_per_a=max_b_per_a)
        if len(fq) == 0:
            say(f"Blind: {_layer_label(layer)} [{li}/{len(layers)}] — no quads "
                f"fit a {lo_px:.0f}–{hi_px:.0f} px baseline, skipping")
            continue
        code, idx, good, ab = quad_codes(pts, fq)
        code, idx, ab = code[good], idx[good], ab[good]
        if len(code) == 0:
            say(f"Blind: {_layer_label(layer)} [{li}/{len(layers)}] — "
                f"{len(fq)} quads formed but none gave a valid code, skipping")
            continue
        tol = _candidate_tolerance(ab, sigma_px)
        mcode, midx = mirror_codes(code, idx)

        say(f"Blind: {_layer_label(layer)} [{li}/{len(layers)}] — hashing "
            f"{len(code)} quads × 2 parities against "
            f"{int(layer.n_quads):,} index quads "
            f"(baseline {lo_px:.0f}–{hi_px:.0f} px, "
            f"tolerance {float(tol.min()):.4f}–{float(tol.max()):.4f})…")

        cands = []
        for flipped, qc, qi_idx in ((False, code, idx), (True, mcode, midx)):
            qi, ci = layer.query(qc, tol)
            if len(qi) == 0:
                continue
            ra_a, dec_a, ra_b, dec_b = layer.quad_ab(ci)
            ra0, dec0 = _midpoint(ra_a, dec_a, ra_b, dec_b)
            for t, q in enumerate(qi):
                p0, p1 = ra0[t], dec0[t]
                cax, cay = _project(ra_a[t], dec_a[t], p0, p1)
                cbx, cby = _project(ra_b[t], dec_b[t], p0, p1)
                ia, ib = qi_idx[q, 0], qi_idx[q, 1]
                zi_a = complex(*pts[ia]); zi_b = complex(*pts[ib])
                zc_a = complex(float(cax), float(cay))
                zc_b = complex(float(cbx), float(cby))
                dz = (np.conj(zi_b - zi_a) if flipped else (zi_b - zi_a))
                if abs(dz) < 1e-9:
                    continue
                m = (zc_b - zc_a) / dz
                t_off = zc_a - m * (np.conj(zi_a) if flipped else zi_a)
                sc = abs(m) * ARCSEC                           # arcsec / pixel
                if sc <= 0 or not np.isfinite(sc):
                    continue
                if scale_hint_arcsec and not (
                        (1 - scale_tol) * scale_hint_arcsec <= sc
                        <= (1 + scale_tol) * scale_hint_arcsec):
                    continue
                zc = complex(img_w / 2.0, img_h / 2.0)
                zs = m * (np.conj(zc) if flipped else zc) + t_off
                cx, cy = _deproject(zs.real, zs.imag, p0, p1)
                cands.append(dict(m=m, t=t_off, flipped=flipped, scale=sc,
                                  ra=cx, dec=cy, p0=p0, p1=p1,
                                  ab=float(ab[q])))

        if not cands:
            say(f"Blind: {_layer_label(layer)} — no code matches")
            continue

        # Rank before spending a catalogue query on anything. True candidates
        # all agree on centre and scale, so agreement is free evidence; after
        # that, prefer the longest baseline, which has the most precise code.
        ra_c = np.array([c["ra"] for c in cands])
        dec_c = np.array([c["dec"] for c in cands])
        sc_c = np.array([c["scale"] for c in cands])
        cosd = np.cos(np.radians(dec_c))
        agree = np.zeros(len(cands), np.int32)
        for i in range(len(cands)):
            d = np.hypot((ra_c - ra_c[i]) * cosd * 60.0, (dec_c - dec_c[i]) * 60.0)
            agree[i] = int(((d < 15.0) &
                            (np.abs(sc_c / sc_c[i] - 1.0) < 0.03)).sum())
        order = np.lexsort((-np.array([c["ab"] for c in cands]), -agree))
        n_ver = min(len(order), max_verify)
        say(f"Blind: {_layer_label(layer)} — {len(cands)} code match"
            f"{'es' if len(cands) != 1 else ''}, {agree.max()} agreeing on the "
            f"same field; verifying the best {n_ver} against the catalogue…")

        best_lo = -np.inf
        t_layer = time.perf_counter()
        for vi, k in enumerate(order[:max_verify], 1):
            c = cands[k]
            t_cand = time.perf_counter()
            half = 0.5 * diag * c["scale"] / 3600.0 * 1.15       # degrees
            cd = max(1e-6, math.cos(math.radians(c["dec"])))
            try:
                cra, cdec, cmag = fetch_box(
                    c["ra"] - half / cd, c["ra"] + half / cd,
                    c["dec"] - half, c["dec"] + half,
                    float(layer.gmag_limit))
            except Exception as e:
                say(f"Blind: catalogue probe failed: {e}")
                continue
            t_fetch = time.perf_counter() - t_cand
            if cra is None or len(cra) < 10:
                say(f"Blind: {_layer_label(layer)} — candidate {vi}/{n_ver} at "
                    f"RA {c['ra']:.3f}° Dec {c['dec']:+.3f}° — catalogue "
                    f"returned only {0 if cra is None else len(cra)} stars "
                    f"({t_fetch * 1000:.0f} ms), skipping")
                continue

            cx, cy = _project(cra, cdec, c["p0"], c["p1"])
            z = (cx + 1j * cy - c["t"]) / c["m"]
            if c["flipped"]:
                z = np.conj(z)
            proj = np.column_stack([z.real, z.imag])
            inb = ((proj[:, 0] > -20) & (proj[:, 0] < img_w + 20) &
                   (proj[:, 1] > -20) & (proj[:, 1] < img_h + 20))
            if inb.sum() < 10:
                say(f"Blind: {_layer_label(layer)} — candidate {vi}/{n_ver} at "
                    f"RA {c['ra']:.3f}° Dec {c['dec']:+.3f}° — only "
                    f"{int(inb.sum())} of {len(cra):,} catalogue stars land in "
                    f"frame, skipping")
                continue

            lo_, nm = log_odds(pts, proj[inb], max(sigma_px * 1.6, 1.2),
                               float(img_w * img_h), eps=eps)
            # The log-odds gain goes as 1/N_cat, so a catalogue probe that
            # came back with only a sliver of the field can clear the
            # threshold on a handful of coincidences. Require real matches too.
            best_lo = max(best_lo, lo_)
            if lo_ > LOG_ODDS_ACCEPT and nm >= min_match:
                roll = math.degrees(math.atan2(c["m"].imag, c["m"].real)) % 360.0
                say(f"Blind: SOLVED on {_layer_label(layer)} — "
                    f"RA {c['ra']:.4f}° Dec {c['dec']:+.4f}°, "
                    f"{c['scale']:.3f}\"/px, roll {roll:.1f}°, "
                    f"{'mirrored' if c['flipped'] else 'normal'} parity, "
                    f"{nm} of {n} stars matched, log-odds {lo_:,.0f} "
                    f"(candidate {vi}/{n_ver}, "
                    f"{time.perf_counter() - t_layer:.1f}s in this layer)")
                return dict(ok=True, ra=c["ra"], dec=c["dec"],
                            scale_arcsec=c["scale"], roll_deg=roll,
                            parity=("flipped" if c["flipped"] else "normal"),
                            log_odds=float(lo_), n_match=int(nm),
                            layer=int(layer.layer_index), error="")

            # Report EVERY candidate. This is the slow stage, and when it
            # stalls the only useful question is "on what, and where".
            say(f"Blind: {_layer_label(layer)} — candidate {vi}/{n_ver} at "
                f"RA {c['ra']:.3f}° Dec {c['dec']:+.3f}° "
                f"({c['scale']:.2f}\"/px, "
                f"{'mirrored' if c['flipped'] else 'normal'}) — "
                f"{int(inb.sum()):,} of {len(cra):,} catalogue stars in frame, "
                f"{nm} matched, log-odds {lo_:,.0f} "
                f"(best here {best_lo:,.0f}, need {LOG_ODDS_ACCEPT:.0f}) "
                f"[probe {t_fetch * 1000:.0f} ms]")

            # Report the BEST score reached anywhere, not whichever candidate
            # happened to be checked last, so the progress lines and the final
            # error agree about how close it got.
            if lo_ >= best_overall:
                best_overall = lo_
                fail = (f"best log-odds {lo_:,.0f} on {_layer_label(layer)} "
                        f"(need {LOG_ODDS_ACCEPT:.0f})")

        say(f"Blind: {_layer_label(layer)} — no match after {n_ver} "
            f"candidate{'s' if n_ver != 1 else ''}, best log-odds "
            f"{best_lo:,.0f} ({time.perf_counter() - t_layer:.1f}s)")

    return dict(ok=False, error=f"blind solve found no match — {fail}")


# ─────────────────────────────────────────────────────────────────────────────
#  Catalogue probes
# ─────────────────────────────────────────────────────────────────────────────

def make_fetch_box(prefer_astrometric: bool = True) -> Optional[FetchBox]:
    """A fetch_box over whichever Gaia library the user has installed."""
    from setiastro.saspro.gaia_database import get_astro_library, get_library

    astro = None
    if prefer_astrometric:
        try:
            a = get_astro_library()
            if a.installed_bands():
                astro = a
        except Exception:
            astro = None

    if astro is not None:
        def fetch(ra_lo, ra_hi, dec_lo, dec_hi, gmax):
            rows = astro.find_stars_in_box(ra_lo % 360.0, ra_hi % 360.0,
                                           dec_lo, dec_hi, max_mag=gmax)
            if not rows:
                return None, None, None
            ra = np.fromiter((s.ra for s in rows), np.float64, len(rows))
            dec = np.fromiter((s.dec for s in rows), np.float64, len(rows))
            mag = np.fromiter((99.0 if s.gmag is None else s.gmag
                               for s in rows), np.float64, len(rows))
            return ra, dec, mag
        return fetch

    try:
        xp = get_library()
        if not xp.installed_bands():
            return None
    except Exception:
        return None

    def fetch_xp(ra_lo, ra_hi, dec_lo, dec_hi, gmax):
        ra_l, dec_l, mag_l = [], [], []
        ranges = ([(ra_lo % 360.0, 360.0), (0.0, ra_hi % 360.0)]
                  if (ra_lo % 360.0) > (ra_hi % 360.0)
                  else [(ra_lo % 360.0, ra_hi % 360.0)])
        for conn in xp._connections.values():
            for r0, r1 in ranges:
                try:
                    cur = conn.execute(
                        "SELECT ra, dec, phot_g_mean_mag FROM sources "
                        "WHERE dec BETWEEN ? AND ? AND ra BETWEEN ? AND ? "
                        "  AND phot_g_mean_mag IS NOT NULL "
                        "  AND phot_g_mean_mag <= ?",
                        (dec_lo, dec_hi, r0, r1, float(gmax)))
                    for a, b, g in cur.fetchall():
                        ra_l.append(a); dec_l.append(b); mag_l.append(g)
                except Exception:
                    continue
        if not ra_l:
            return None, None, None
        return (np.array(ra_l), np.array(dec_l), np.array(mag_l))
    return fetch_xp


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point used by plate_solver
# ─────────────────────────────────────────────────────────────────────────────

def blind_seed(image: np.ndarray,
               scale_hint_arcsec: Optional[float] = None,
               status_cb: Optional[Callable[[str], None]] = None) -> Tuple[bool, object]:
    """
    Blind-solve a numpy image and return (True, seed_header) or (False, reason).

    The returned header carries only CRVAL1/2, a CD matrix and CROTA2 — enough
    for plate_solver's in-house Gaia path to take over and produce the real
    solution with SIP, refinement and the quality gate.
    """
    from astropy.io.fits import Header
    from setiastro.saspro.gaia_quad_index import get_quad_library

    def say(m):
        if status_cb:
            status_cb(m)

    lib = get_quad_library()
    if not lib.any_installed():
        return False, ("No blind-solve index built. Build one on the Gaia DR3 "
                       "Library → Blind Solve Index tab (it is built locally "
                       "from the Gaia data you already have).")

    from setiastro.saspro.plate_solver import _normalize_for_astap, _to_gray2d_unit
    gray = _to_gray2d_unit(_normalize_for_astap(image))
    mn, mx = float(gray.min()), float(gray.max())
    if mx > mn:
        gray = ((gray - mn) / (mx - mn)).astype(np.float32)

    h, w = gray.shape[:2]
    say(f"Blind: detecting stars in the {w}×{h} frame…")
    det = detect_stars_by_brightness(gray, max_stars=120)
    if len(det) < 8:
        say(f"Blind: only {len(det)} stars detected — too few to form quads")
        return False, f"blind solve: only {len(det)} stars detected"

    field_arcmin = None
    if scale_hint_arcsec:
        field_arcmin = max(w, h) * scale_hint_arcsec / 60.0

    fetch = make_fetch_box()
    if fetch is None:
        return False, ("blind solve: no Gaia catalogue installed to verify "
                       "against (install an Astrometry or XP tier)")

    chosen = lib.layers_for_field(field_arcmin)
    if field_arcmin:
        say(f"Blind: {len(det)} stars detected; the header pixel scale implies "
            f"a {_fmt_ang(field_arcmin)} field, so "
            + (f"{_layer_label(chosen[0])} is tried first"
               if chosen else "no built layer is a good fit"))
    else:
        say(f"Blind: {len(det)} stars detected; no usable pixel scale, so all "
            f"{len(chosen)} built layer{'s' if len(chosen) != 1 else ''} will "
            f"be tried, coarsest first")

    res = solve_blind_core(det[:, :2], w, h, chosen, fetch,
                           scale_hint_arcsec=scale_hint_arcsec,
                           status_cb=say)
    if not res.get("ok"):
        say(f"Blind: FAILED — {res.get('error', 'no match')}")
        return False, res.get("error", "blind solve failed")

    sc_deg = res["scale_arcsec"] / 3600.0
    th = math.radians(res["roll_deg"])
    par = -1.0 if res["parity"] == "flipped" else 1.0

    hdr = Header()
    hdr["CTYPE1"] = "RA---TAN"
    hdr["CTYPE2"] = "DEC--TAN"
    hdr["CRPIX1"] = w / 2.0
    hdr["CRPIX2"] = h / 2.0
    hdr["CRVAL1"] = float(res["ra"])
    hdr["CRVAL2"] = float(res["dec"])
    # A mirror negates a whole axis. Scaling CD1_1 and CD2_1 by the parity
    # sign while leaving CD1_2 and CD2_2 alone takes half of each instead,
    # which shears the matrix rather than reflecting it: det becomes
    # -par * sc^2 * cos(2*th), singular at a 45 degree roll and parity-
    # inverted past it. Kept in step with _cd_matrix() in plate_solver.py;
    # duplicated rather than imported because plate_solver imports this
    # module, and a module-level import back would be a cycle.
    _c, _s = math.cos(th), math.sin(th)
    if par < 0:                      # mirrored: det = -sc^2
        hdr["CD1_1"], hdr["CD1_2"] = -sc_deg * _c, -sc_deg * _s
        hdr["CD2_1"], hdr["CD2_2"] = -sc_deg * _s,  sc_deg * _c
    else:                            # det = +sc^2
        hdr["CD1_1"], hdr["CD1_2"] = -sc_deg * _c,  sc_deg * _s
        hdr["CD2_1"], hdr["CD2_2"] = -sc_deg * _s, -sc_deg * _c
    hdr["CROTA2"] = float(res["roll_deg"])
    hdr["RADESYS"] = "ICRS"
    hdr["WCSAXES"] = 2
    hdr["BLNDODDS"] = (round(res["log_odds"], 1),
                       "Blind solve log-odds vs background")
    hdr["BLNDLAYR"] = (res["layer"], "Quad index layer that matched")
    hdr["BLNDPAR"] = (res["parity"], "Quad match parity (normal|flipped)")
    return True, hdr